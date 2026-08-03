# Chapter 02: cublasLt选择策略与Epilogue Fusion 深度分析

## 1. 设计动机

**WHY深入cublasLt**: cublasLt是PyTorch/Megatron中GEMM的实际执行者。
理解其MatmulDescriptor、LayoutDescriptor和AlgorithmSelection，
才能正确配置FP8 GEMM、利用epilogue fusion、和做autotuning。

## 2. cublasLt完整调用链

```
cublasLtMatmul()
    │
    ├── operationDesc  ← 操作类型(compute_type, scale_type, epilogue)
    │     │
    │     ├── CUBLAS_COMPUTE_32F        (FP32累加)
    │     ├── CUBLASLT_EPILOGUE_GELU_BIAS (融合GELU+Bias)
    │     └── CUBLASLT_POINTER_MODE_DEVICE (scale在GPU上)
    │
    ├── Adesc / Bdesc / Cdesc / Ddesc  ← 矩阵描述
    │     │
    │     ├── dataType: CUDA_R_16F / CUDA_R_8F_E4M3
    │     ├── layout: row/col-major
    │     ├── leading_dimension
    │     └── batch_count / stride (for batched)
    │
    ├── algo  ← 算法选择
    │     │
    │     ├── algoId: 0-23
    │     ├── tileId: CUBLASLT_MATMUL_TILE_128x128 等
    │     └── splitKFactor: 1-64
    │
    └── workspace  ← 工作空间 (splitK归约/临时buffer)
```

## 3. Epilogue Fusion类型

### 3.1 支持的Epilogue

```
D = epilogue(alpha * A @ B + beta * C)

┌──────────────────────────────────────────────────────┐
│  Epilogue枚举                    │ 计算公式            │
├──────────────────────────────────┼────────────────────┤
│ CUBLASLT_EPILOGUE_DEFAULT        │ D = αAB + βC       │
│ CUBLASLT_EPILOGUE_BIAS           │ D = αAB + βC + bias│
│ CUBLASLT_EPILOGUE_RELU           │ D = ReLU(αAB + βC) │
│ CUBLASLT_EPILOGUE_RELU_BIAS      │ D = ReLU(αAB+bias) │
│ CUBLASLT_EPILOGUE_GELU           │ D = GELU(αAB + βC) │
│ CUBLASLT_EPILOGUE_GELU_BIAS      │ D = GELU(αAB+bias) │
│ CUBLASLT_EPILOGUE_GELU_AUX       │ +保存pre-GELU给BWD │
│ CUBLASLT_EPILOGUE_GELU_AUX_BIAS  │ 融合bias+GELU+save │
│ CUBLASLT_EPILOGUE_DGELU          │ BWD: dGELU          │
│ CUBLASLT_EPILOGUE_DGELU_BGRAD    │ BWD: dGELU+dbias    │
│ CUBLASLT_EPILOGUE_BGRADB         │ 计算bias梯度        │
└──────────────────────────────────┴────────────────────┘
```

**WHY Epilogue如此重要**: 对于Transformer的Linear+GELU层：
```
无fusion:  GEMM(写D) → Read D → Add Bias(写D') → Read D' → GELU(写D'')
           3次写 + 2次读 = 5×M×N×sizeof(dtype) 额外带宽

有fusion:  GEMM+Bias+GELU(写D)
           0次额外带宽!

节省量(M=N=4096, FP16): 5×4096×4096×2 = 160MB
对应时间: 160MB / 3.35TB/s ≈ 0.048ms (每层!)
```

### 3.2 AUX输出 (训练必需)

```
GELU_AUX epilogue:
D = GELU(αAB + bias)
AUX = αAB + bias    ← 保存pre-activation值，BWD需要

WHY: GELU的反向传播需要前向的输入值:
dX = dY * GELU'(X)  其中X = αAB + bias
不保存AUX则BWD需要重算forward，浪费计算
```

### 3.3 TransformerEngine中的使用

```python
# transformer_engine/pytorch/module/linear.py
# TE通过cublasLt epilogue实现:
# Forward:  Linear = GEMM + Bias + GELU (一个kernel)
# Backward: dLinear = dGELU(AUX) + GEMM(dW) + dbias (fusion)

# 在FP8模式下更关键:
# GEMM输出FP32 → scale+cast to FP8在epilogue中完成
# 避免额外的scale kernel
```

## 4. FP8 GEMM配置

### 4.1 Matrix Descriptor设置

```cpp
// FP8 E4M3 input
cublasLtMatrixLayoutSetAttr(Adesc, 
    CUBLASLT_MATRIX_LAYOUT_TYPE, CUDA_R_8F_E4M3);

// 输出通常FP16/BF16 (FP8精度不够累积)
cublasLtMatrixLayoutSetAttr(Ddesc,
    CUBLASLT_MATRIX_LAYOUT_TYPE, CUDA_R_16BF);

// Operation必须FP32累加
cublasLtMatmulDescSetAttr(operationDesc,
    CUBLASLT_MATMUL_DESC_COMPUTE_TYPE, CUBLAS_COMPUTE_32F);
```

### 4.2 Scale模式

```
Per-tensor scaling (当前主流):
D = scale_d * (scale_a * A_fp8) @ (scale_b * B_fp8)

设置:
cublasLtMatmulDescSetAttr(desc,
    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_a);  // device pointer
cublasLtMatmulDescSetAttr(desc,
    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_b);
cublasLtMatmulDescSetAttr(desc,
    CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_scale_d);  // 输出scale
```

### 4.3 AMAX输出 (延迟scaling)

```
cublasLt可以在epilogue中计算amax:
D_fp8 = cast_to_fp8(D_fp32 * scale_d)
amax_d = max(abs(D_fp32))  ← 用于下一次迭代的scale计算

设置:
cublasLtMatmulDescSetAttr(desc,
    CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &d_amax);

WHY内置AMAX: 避免单独的reduce kernel遍历整个输出矩阵
```

## 5. Layout约束与性能

### 5.1 支持的Layout组合

```
cublasLt支持的GEMM layout (D = A @ B):
┌────────┬────────┬────────┬────────┐
│  Case  │   A    │   B    │   D    │
├────────┼────────┼────────┼────────┤
│  NN    │ Col    │ Col    │ Col    │ ← 经典BLAS
│  NT    │ Col    │ Row    │ Col    │ ← 常用
│  TN    │ Row    │ Col    │ Col    │ ← 常用
│  TT    │ Row    │ Row    │ Col    │
│  *     │ Col/Row│ Col/Row│ Row    │ ← CUDA 11.4+支持
└────────┴────────┴────────┴────────┘

性能差异: NT通常最优(A连续读、B连续读)
PyTorch默认: mm(A,B) → A=Row, B=Col → 内部转为TN调用
```

### 5.2 Alignment要求

```
FP16:  矩阵地址需16B对齐 (8个FP16)
FP8:   矩阵地址需16B对齐 (16个FP8)
LD要求: leading_dim必须为alignment的倍数

不满足时: cublasLt可能fallback到慢路径或返回错误
PyTorch中: 通过pad到对齐大小处理 (at::native::pad_to_alignment)
```

## 6. Batched GEMM与Grouped GEMM

### 6.1 Strided Batch

```cpp
// 所有batch相同大小、等stride
cublasLtMatrixLayoutSetAttr(Adesc,
    CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_size);
cublasLtMatrixLayoutSetAttr(Adesc,
    CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_a);

// 适用: Multi-head attention中的QK^T, Attention×V
```

### 6.2 Grouped GEMM (CUDA 12.5+)

```cpp
// 不同batch可以有不同M,N,K
// cublasLtMatmulGrouped()
// 适用: MoE中不同expert的FFN(不同token数)

// 替代方案(旧版本): 
// 1. Pad到最大size → 浪费计算
// 2. 循环调用单GEMM → kernel launch overhead
// 3. CUTLASS Grouped GEMM → 自定义实现
```

## 7. 算法选择策略总结

```
决策树:
┌── 大矩阵 (M,N > 1024)?
│   ├── Yes → 默认heuristic通常够好
│   │         可选: tunable op搜索 (+5~15%)
│   │
│   └── No → 小矩阵需要特殊处理
│       ├── K很大? → splitK (增加SM利用)
│       ├── Batch多? → strided batch
│       └── shape不规则? → CUTLASS custom tile
│
├── 需要epilogue fusion?
│   ├── Bias+Act → cublasLt epilogue (零开销)
│   ├── 自定义epilogue → CUTLASS
│   └── FP8+scale+amax → cublasLt (内置支持)
│
└── 性能瓶颈在哪?
    ├── Compute → 增大tile, 用FP8
    ├── Memory → 减少数据量, L2 residency
    └── Launch → persistent kernel, CUDA Graph
```

## 8. 总结

| 维度 | 最佳实践 |
|------|----------|
| API选择 | 优先cublasLt (最灵活) |
| Epilogue | 尽可能fusion (bias+act+scale) |
| FP8 | 用AMAX epilogue避免额外kernel |
| 算法搜索 | 生产用autotuning,固定结果 |
| Workspace | 给够32-256MB |
| Alignment | Pad到16B对齐 |
| 小矩阵 | splitK + stream parallelism |
