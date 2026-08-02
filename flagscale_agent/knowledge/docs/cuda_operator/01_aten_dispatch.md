# Chapter 01: ATen CUDA算子调度机制 深度源码分析

## 1. 设计动机

**WHY**: PyTorch的`torch.mm(A, B)`一行Python代码，背后需要经过多层dispatch才能到达GPU kernel。
理解这个链路是手写算子替换/集成的基础。

**核心问题**：
- Python层如何路由到C++ CUDA实现？
- 同一个`mm`操作，何时走cuBLAS、何时走cublasLt、何时走CUTLASS？
- FP8/INT8等新精度如何接入调度？

## 2. 调度全链路

```
torch.mm(A, B)           # Python层
    │
    ▼
torch._C._TensorBase.mm()  # C++绑定
    │
    ▼
at::native::mm()          # ATen native function
    │ (dispatcher根据device/dtype dispatch)
    ▼
at::native::mm_out_cuda()   # aten/src/ATen/native/cuda/Blas.cpp L645
    │
    ▼
addmm_out_cuda_impl()      # 统一入口 L339
    │
    ├─── cublasLt path (优先)     → launchGemmAndBiasCublasLt<T>()
    │         │
    │         └── at::cuda::blas::gemm_and_bias()
    │                    │
    │                    └── cublasLtMatmul()  ← cuBLAS Lt API
    │
    └─── cublas fallback path     → at::cuda::blas::gemm<T>()
                │
                └── cublas[S|D|H]gemm() / cublasGemmEx()
```

> **源码**: `aten/src/ATen/native/cuda/Blas.cpp` L339-460

## 3. Dispatcher机制

### 3.1 注册与分发

PyTorch使用**多dispatch key**机制，CUDA算子通过structured kernel注册：

```cpp
// aten/src/ATen/native/native_functions.yaml
- func: mm(Tensor self, Tensor mat2) -> Tensor
  structured_delegate: mm.out
  dispatch:
    CUDA: mm_out_cuda      // ← 注册CUDA实现
    CPU: mm_out_cpu
```

**L645**: `TORCH_IMPL_FUNC(mm_out_cuda)` 是实际注册点：
```cpp
// Blas.cpp L645-647
TORCH_IMPL_FUNC(mm_out_cuda)(const Tensor& self, const Tensor& mat2, const Tensor& result) {
  addmm_out_cuda_impl(const_cast<Tensor&>(result), result, self, mat2, 0, 1);
}
```

**WHY统一到addmm**: `mm(A,B) = addmm(0, _, A, B, 0, 1)`，所有GEMM变体归一化处理。

### 3.2 Shape检查与内存布局

```cpp
// L339-355: Shape validation
TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tensors must be 2-D");
TORCH_CHECK(mat1.dtype() == mat2.dtype(), ...);
```

```cpp
// L422: cublasCommonArgs 处理内存布局
cublasCommonArgs args(mat1, mat2, result);
```

`cublasCommonArgs` 的作用（定义在 cuBlasCommonArgs.h）：
- 判断 row-major / col-major
- 计算 leading dimension (ld)
- 处理 transpose flags
- 解析 conjugate tensors

## 4. cublasLt优先策略

### 4.1 选择逻辑

```cpp
// L362-371: 决定是否使用cublasLt
static bool persistent_disable = isGloballyDisabledAddmmCudaLt(self.device());
bool disable_addmm_cuda_lt = persistent_disable || disable_override;
disable_addmm_cuda_lt = disable_addmm_cuda_lt ||
    !isInputCompliesAddmmCudaLt(result, self, mat1, mat2, beta, alpha, activation);
```

**WHY优先cublasLt**：
- cublasLt支持epilogue fusion (bias + activation 融合)
- 支持更灵活的内存布局（不要求col-major）
- 支持FP8/INT8等新数据类型
- 支持workspace-based算法搜索

### 4.2 cublasLt启用条件

`isInputCompliesAddmmCudaLt()` 检查：
| 条件 | 原因 |
|------|------|
| dtype ∈ {FP16, BF16, FP32} | Lt支持的类型 |
| 无complex类型 | 不支持复数 |
| result连续或转置连续 | 内存布局约束 |
| CUDA_VERSION >= 11020 | API可用性 |

### 4.3 Fallback机制

```cpp
// L456-458: Lt失败时递归回退
if (!lt_success) {
  return addmm_out_cuda_impl(result, self, mat1, mat2, beta, alpha,
                             activation, /*disable_lt=*/true);
}
```

**WHY递归回退而非直接走cublas**: 保持接口统一，disable_lt=true确保不会无限递归。

## 5. Bias + Activation Fusion (Epilogue)

### 5.1 融合设计

```cpp
// L380-382
bool use_bias_ptr_lt = (self.dim() == 1) && !disable_addmm_cuda_lt;
use_bias_ptr_lt &= !is_float_output_with_half_input;
```

**WHY 1D bias特判**: `addmm(bias, A, B)` 当bias是1D向量时，可以用cublasLt的epilogue直接加bias，
省去单独的broadcast加法kernel：

```
传统路径:  GEMM kernel → bias add kernel → activation kernel (3次global memory读写)
融合路径:  cublasLt(epilogue=BIAS+GELU) (1次global memory写)
```

节省带宽 = 2×M×N×sizeof(dtype)，对于大矩阵可达GB级。

### 5.2 Activation类型

```cpp
// L98-102
enum class Activation {
  None,
  RELU,
  GELU,
};
```

```cpp
// 传递到cublasLt epilogue
// CUDABlas.cpp 中:
//   CUBLASLT_EPILOGUE_RELU_BIAS  → GEMM + Bias + ReLU
//   CUBLASLT_EPILOGUE_GELU_BIAS  → GEMM + Bias + GELU
```

## 6. _scaled_mm: FP8 GEMM入口

### 6.1 调度路径

```
torch._scaled_mm(A_fp8, B_fp8, scale_a, scale_b)
    │
    ▼
_scaled_mm_out_cuda()      // Blas.cpp (另一个注册点)
    │
    ▼
at::cuda::blas::scaled_gemm()   // CUDABlas.cpp
    │
    ▼
cublasLtMatmul() with FP8 desc   // 必须走cublasLt
```

**WHY必须cublasLt**: 经典cublas API不支持FP8，只有Lt的Matmul Descriptor可以设置FP8 input type。

### 6.2 Scale处理

FP8 GEMM的scale有两种模式：
| 模式 | Scale位置 | 适用场景 |
|------|-----------|----------|
| Tensor-wise | device pointer | 延迟scaling（当前主流） |
| Per-channel | 每列一个scale | MXFP8 block scaling |

```
D = alpha * (scale_a * A_fp8) @ (scale_b * B_fp8) + beta * C
  = alpha * scale_a * scale_b * (A_fp8 @ B_fp8) + beta * C
```

## 7. Tunable GEMM (自动调参)

### 7.1 机制

```cpp
// L432: 检查tunable是否启用
if (at::cuda::tunable::getTuningContext()->IsTunableOpEnabled()) {
  ...
}
```

PyTorch 2.x 引入 TunableOp 机制：
- 首次调用时枚举所有可能的GEMM算法
- Benchmark每个算法的实际耗时
- 缓存最优算法到文件
- 后续调用直接使用最优算法

**WHY tunable优于默认heuristic**: cuBLAS的默认选择基于shape/dtype静态规则，
实际性能受L2 cache状态、并发kernel等影响，实测选择可提升5-30%性能。

### 7.2 使用方式

```python
# 环境变量启用
os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
os.environ["PYTORCH_TUNABLEOP_TUNING"] = "1"  # 开启tuning阶段
os.environ["PYTORCH_TUNABLEOP_FILENAME"] = "gemm_tuning.csv"
```

## 8. Batch GEMM (bmm/baddbmm)

### 8.1 路径

```cpp
// L650-660
TORCH_IMPL_FUNC(bmm_out_cuda)(...) {
  baddbmm_out_cuda_impl(result, result, batch1, batch2, beta, alpha);
}
```

Batch GEMM 使用 `cublasGemmStridedBatchedEx()` 或 `cublasLtMatmul` with batch_count。

**WHY Strided vs Grouped**:
| 类型 | 条件 | API |
|------|------|-----|
| Strided Batch | 所有矩阵等大、等stride | cublasGemmStridedBatchedEx |
| Grouped | 矩阵大小不同 | cublasGemmGroupedBatchedEx (CUDA 12.5+) |

## 9. 自定义算子替换策略

### 9.1 替换点选择

要替换PyTorch默认GEMM实现，有三个插入点：

```
插入点1: Python层 torch.mm = custom_mm           # 最简单但有overhead
插入点2: Dispatcher注册 TORCH_LIBRARY_IMPL(...)  # 推荐：零overhead
插入点3: cuBLAS替换 LD_PRELOAD                   # 激进：全局替换
```

### 9.2 Dispatcher注册示例

```cpp
// custom_ops.cpp
#include <torch/library.h>

Tensor my_fast_mm(const Tensor& a, const Tensor& b) {
  // 自定义CUTLASS实现
  return cutlass_gemm(a, b);
}

TORCH_LIBRARY_IMPL(aten, CUDA, m) {
  m.impl("mm.out", my_fast_mm_out);  // 替换默认mm
}
```

**WHY Dispatcher替换**: 零Python overhead，自动被autograd追踪，兼容torch.compile。

## 10. 总结

| 阶段 | 关键文件 | 核心决策 |
|------|----------|----------|
| Python dispatch | torch/_C/ | device routing |
| ATen native | native_functions.yaml | structured kernel注册 |
| CUDA impl | native/cuda/Blas.cpp | cublasLt vs cublas选择 |
| Library call | cuda/CUDABlas.cpp | 算法参数设置 |
| GPU执行 | cublasLtMatmul() | 硬件执行 |

**核心洞察**:
1. 所有GEMM变体统一到`addmm_out_cuda_impl`
2. cublasLt优先，支持fusion；fallback到经典cublas
3. FP8必须走cublasLt（`_scaled_mm`入口）
4. TunableOp可自动选择最优算法
5. 自定义算子推荐用Dispatcher TORCH_LIBRARY_IMPL 注册
