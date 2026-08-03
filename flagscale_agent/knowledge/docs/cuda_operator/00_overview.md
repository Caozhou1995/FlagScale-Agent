# CUDA算子深度分析 — 总纲

## 1. 目标定位

本系列文档面向**手写高性能CUDA算子**的场景，目标：
- 理解从Python调用到GPU执行的完整链路
- 掌握CUTLASS/CuTe编程模型，能独立实现GEMM/Attention等算子
- 理解cuBLAS/cuDNN的内部算法选择逻辑
- 掌握H100微架构特性，能做到指令级调优

## 2. 分级体系

```
┌─────────────────────────────────────────────────────┐
│  Level 4: 算子调度与选择                              │
│  PyTorch ATen dispatch → cuBLAS/cuDNN/CUTLASS 选择  │
├─────────────────────────────────────────────────────┤
│  Level 3: CUDA编程模型与优化                          │
│  CUTLASS GEMM / CuTe / Shared Memory / TMA / WGMMA │
├─────────────────────────────────────────────────────┤
│  Level 2: Library内部行为                             │
│  cuBLAS算法选择 / cuDNN图模式 / Autotuning           │
├─────────────────────────────────────────────────────┤
│  Level 1: 硬件微架构                                  │
│  H100 SM / Tensor Core / Memory Hierarchy / Pipeline │
└─────────────────────────────────────────────────────┘
```

## 3. 章节索引

### Level 4: 算子调度与选择
| 章节 | 主题 | 关键源码 |
|------|------|----------|
| 01 | ATen CUDA Dispatch机制 | pytorch/aten/src/ATen/native/cuda/Blas.cpp |
| 02 | cublasLt vs cublas选择策略 | pytorch/aten/src/ATen/cuda/CUDABlas.cpp |
| 03 | TransformerEngine Fused Ops调度 | TE-FL/pytorch/ops/*.py |

### Level 3: CUDA编程模型与优化
| 章节 | 主题 | 关键源码 |
|------|------|----------|
| 04 | CUTLASS 3.x GEMM架构 | cutlass/include/cutlass/gemm/ |
| 05 | CuTe Layout与Tensor抽象 | cutlass/include/cute/ |
| 06 | TMA与异步流水线 | cutlass/include/cute/arch/copy_sm90_tma.hpp |
| 07 | WGMMA (Warpgroup MMA) | cutlass/include/cute/atom/mma_sm90*.hpp |
| 08 | Shared Memory与Bank Conflict | CUDA Programming Guide |

### Level 2: Library内部行为
| 章节 | 主题 | 关键源码 |
|------|------|----------|
| 09 | cuBLAS GEMM算法枚举与选择 | CUDA Toolkit headers |
| 10 | cuDNN Graph API与Fusion | cudnn_frontend |

### Level 1: 硬件微架构
| 章节 | 主题 | 关键源码 |
|------|------|----------|
| 11 | H100 SM微架构 | NVIDIA Whitepaper + PTX ISA |
| 12 | Tensor Core指令详解 | PTX ISA: mma/wgmma/stmatrix |
| 13 | 内存层级与带宽模型 | Roofline分析 |
| 14 | 指令流水线与Occupancy | NSight Compute metrics |

## 4. 前置知识
- C/C++ 和 CUDA C++ 基础
- 矩阵乘法数学（M×K @ K×N = M×N）
- GPU 基本概念（Grid/Block/Thread/Warp）
