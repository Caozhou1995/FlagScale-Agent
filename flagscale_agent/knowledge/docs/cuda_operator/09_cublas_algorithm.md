# Chapter 09: cuBLAS GEMM算法选择与调优 深度分析

## 1. 设计动机

**WHY理解cuBLAS算法选择**: cuBLAS是闭源的，但它暴露了算法选择接口(cublasLtMatmulAlgoGetHeuristic)。
理解其行为模式，可以：
- 知道何时默认选择已足够好
- 何时需要手动搜索或用CUTLASS替换
- 如何利用autotuning获得最优性能

## 2. cuBLAS API层次

```
┌─────────────────────────────────────────────┐
│  Level 3 (Legacy): cublasSgemm, cublasHgemm │
│  简单接口，内部自动选算法                      │
├─────────────────────────────────────────────┤
│  Level 3 (Ex): cublasGemmEx                  │
│  支持mixed precision，仍自动选算法            │
├─────────────────────────────────────────────┤
│  cublasLt: cublasLtMatmul                    │
│  完全可控：算法ID、workspace、epilogue        │
│  推荐用于生产环境                             │
└─────────────────────────────────────────────┘
```

**WHY cublasLt取代cublasGemmEx**:
- 支持epilogue fusion (GEMM + bias + activation)
- 支持FP8/INT8等新类型
- 可指定算法ID进行确定性计算
- 支持workspace-based算法搜索

## 3. cublasLt算法搜索

### 3.1 Heuristic API

```cpp
// 获取推荐算法列表
cublasLtMatmulAlgoGetHeuristic(
    ltHandle,
    operationDesc,    // 定义GEMM操作 (transA, transB, compute_type)
    Adesc, Bdesc,     // 输入矩阵描述 (type, layout, leading_dim)
    Cdesc, Ddesc,     // 输出矩阵描述
    preference,       // 偏好设置 (workspace大小, reduction scheme)
    maxAlgoCount,     // 最多返回几个算法
    heuristicResults, // 输出: 算法列表 + 预估时间
    &returnedCount    // 实际返回数
);
```

### 3.2 算法参数空间

每个算法由以下参数定义:
| 参数 | 含义 | 典型值 |
|------|------|--------|
| algoId | 算法族ID | 0-23 (H100) |
| tile | CTA tile大小 | 128×128, 256×64, etc. |
| stages | Pipeline stages | 3-7 |
| splitK | K维度split数 | 1 (无split), 2-64 |
| reductionScheme | splitK归约方式 | NONE, INPLACE, COMPUTE |
| swizzle | CTA调度模式 | 0-3 |
| customOption | 自定义位 | 0-255 |

### 3.3 SplitK策略

```
标准GEMM (splitK=1):
CTA处理完整K维度，直接写结果

SplitK GEMM (splitK=4):
┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
│ K/4 │ │ K/4 │ │ K/4 │ │ K/4 │  ← 4个CTA各算部分
└──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
   │       │       │       │
   └───────┴───────┴───────┘
            │ Reduce (原子加或二次kernel)
            ▼
         D[M,N]

WHY SplitK: 当M,N很小但K很大时（如LLM的FFN第二层），
单个CTA的wave不够覆盖所有SM，splitK增加并行度。
```

## 4. Workspace管理

```cpp
// Workspace用于:
// 1. SplitK的中间结果存储
// 2. 算法内部的临时buffer
// 3. Epilogue的辅助计算

size_t workspaceSize = 0;
cublasLtMatmulAlgoGetHeuristic(..., &heuristic);
workspaceSize = heuristic[0].workspaceSize;  // 获取需要的大小

// 分配workspace
void* workspace;
cudaMalloc(&workspace, workspaceSize);

// 传给matmul
cublasLtMatmul(ltHandle, ..., workspace, workspaceSize, stream);
```

**WHY需要Workspace**: 某些高性能算法(带splitK)需要额外内存做归约。
不提供workspace → 这些算法不可用 → 可能退化到慢算法。

PyTorch中workspace大小: 默认32MB (at::cuda::getNewWorkspace())

## 5. 性能特征分析

### 5.1 Shape对性能的影响

```
H100上 FP16 GEMM 性能vs shape:

M=N=K=4096:  ~85% peak (compute-bound, 大tile)
M=N=K=1024:  ~60% peak (wave效率下降)
M=N=K=256:   ~25% peak (SM利用不足)

M=1, N=K=4096 (向量×矩阵):
  → 完全memory-bound, 看HBM带宽
  → 实测 ~2.8 TB/s (接近3.35 peak)

M=4096, N=4096, K=1:
  → 退化为element-wise add
  → cuBLAS无法优化
```

### 5.2 dtype对算法选择的影响

| dtype | 推荐tile | TC指令 | 峰值 |
|-------|----------|--------|------|
| FP16 | 128×256×64 | WGMMA m64n128k16 | 989 TFLOPS |
| BF16 | 128×256×64 | WGMMA m64n128k16 | 989 TFLOPS |
| FP8 (E4M3) | 128×256×128 | WGMMA m64n128k32 | 1979 TFLOPS |
| TF32 | 128×128×32 | WGMMA m64n128k8 | 495 TFLOPS |
| FP32 | 无TC | FFMA | 67 TFLOPS |

### 5.3 Profiling方法

```bash
# 用NSight Compute分析cuBLAS kernel
ncu --set full -o gemm_profile \
    python -c "import torch; a=torch.randn(4096,4096,device='cuda',dtype=torch.float16); torch.mm(a,a)"

# 关注指标:
# sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed  ← TC利用率
# dram__bytes_read.sum / dram__bytes_write.sum  ← HBM流量
# l2__throughput.avg.pct_of_peak_sustained_elapsed  ← L2命中
```

## 6. Autotuning实战

### 6.1 PyTorch TunableOp

```python
import torch
import os

# 启用tunable op
os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
os.environ["PYTORCH_TUNABLEOP_TUNING"] = "1"
os.environ["PYTORCH_TUNABLEOP_FILENAME"] = "/workspace/gemm_tune.csv"
os.environ["PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS"] = "30"  # 每shape最多搜30ms

# Warmup + tuning
a = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
b = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
for _ in range(100):
    torch.mm(a, b)  # 前几次自动tuning，后续用缓存结果
```

### 6.2 手动cublasLt Autotuning

```cpp
// 枚举所有算法并benchmark
int numAlgos;
cublasLtMatmulAlgoGetHeuristic(handle, desc, ..., 32, results, &numAlgos);

float best_time = 1e9;
int best_idx = 0;
for (int i = 0; i < numAlgos; i++) {
    // Warmup
    cublasLtMatmul(..., &results[i].algo, ...);
    
    // Benchmark
    cudaEventRecord(start);
    for (int r = 0; r < 100; r++)
        cublasLtMatmul(..., &results[i].algo, ...);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    if (ms < best_time) { best_time = ms; best_idx = i; }
}
// 保存 results[best_idx].algo 用于后续调用
```

## 7. cuBLAS vs CUTLASS选择指南

| 场景 | 推荐 | 理由 |
|------|------|------|
| 标准GEMM (大矩阵) | cuBLAS | 已高度优化,无需自定义 |
| GEMM+Bias+Activation | cublasLt | epilogue fusion内置 |
| GEMM+自定义epilogue | CUTLASS | 灵活的epilogue模板 |
| FP8 GEMM | cublasLt (>= CUDA 11.8) | 成熟支持 |
| Grouped/Batched | cublasLtMatmulBatched | 高效batch |
| 极小矩阵 (M<64) | CUTLASS custom tile | cuBLAS tile太大 |
| 非标准layout | CUTLASS | cuBLAS要求连续 |

## 8. 总结

| 要点 | 建议 |
|------|------|
| 默认使用 | cublasLt + workspace 32MB |
| 性能不够时 | 先试tunable op自动搜索 |
| 仍不够时 | ncu profile确认瓶颈 |
| 确认cuBLAS次优 | 用CUTLASS自定义 |
| FP8场景 | 必须cublasLt或CUTLASS |
| 确定性需求 | 固定algoId + splitK=1 |
