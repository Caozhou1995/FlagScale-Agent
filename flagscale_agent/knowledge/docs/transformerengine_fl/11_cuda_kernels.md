# TE-FL 第11章：CUDA Kernel 层深度源码分析

## 1. 概述与设计动机

### 1.1 核心问题

Transformer 训练中，以下操作是性能瓶颈：
- GEMM (矩阵乘): 计算密集，需 FP8 加速
- Attention: 内存密集，需 FlashAttention 融合
- Activation: 带宽受限，需融合到邻近操作中
- Cast/Transpose: 数据搬运，需与量化融合
- Softmax: 多次内存访问，需 fused kernel

TE-FL 通过 C++/CUDA 层提供高性能 kernel，由 PyTorch 扩展暴露给 Python。

### 1.2 WHY: 为什么不直接用 PyTorch native 操作？

| 操作 | PyTorch native | TE 融合 kernel | 加速比 |
|------|---------------|----------------|--------|
| LayerNorm + Cast_FP8 | 2 kernel launch | 1 kernel launch | ~1.5× |
| GEMM (BF16) | cuBLAS | cuBLAS (相同) | 1× |
| GEMM (FP8) | 不支持 | cuBLAS FP8 API | 2× (H100) |
| SwiGLU + Cast_FP8 | 3 ops | 1 fused kernel | ~2× |
| Attention (seq 8K) | O(N²) 内存 | FlashAttention O(N) | 10×+ |

核心收益：**减少 kernel launch 开销 + 减少中间 tensor 内存分配 + 利用硬件特性**。

## 2. 源码目录结构

```
transformer_engine/common/
├── activation/                # 激活融合 kernel
│   ├── gelu.cu               # GELU + FP8 cast
│   ├── swiglu.cu             # SwiGLU + FP8 cast
│   ├── relu.cu               # ReLU 变体
│   ├── glu.cu                # GLU 变体
│   └── activation_template.h # 通用模板
│
├── cast/                      # FP8 cast kernel
│   └── (cast + amax 计算)
│
├── fused_attn/                # 融合注意力
│   ├── fused_attn.cpp         # 路由入口
│   ├── flash_attn.cu          # Flash Attention 封装
│   ├── fused_attn_fp8.cu      # FP8 注意力
│   ├── fused_attn_f16_max512_seqlen.cu   # 短序列优化
│   ├── fused_attn_f16_arbitrary_seqlen.cu # 长序列
│   ├── context_parallel.cu    # CP 通信 kernel
│   └── kv_cache.cu            # KV 缓存管理
│
├── fused_rope/                # RoPE 融合
│   └── fused_rope.cu          # RoPE + QKV projection 融合
│
├── fused_softmax/             # Softmax 融合
│   ├── scaled_masked_softmax.cu
│   ├── scaled_upper_triang_masked_softmax.cu
│   └── scaled_aligned_causal_masked_softmax.cu
│
├── fused_router/              # MoE Router 融合
│   └── (Top-K 选择 + capacity 计算)
│
├── gemm/                      # GEMM 封装
│   └── (cuBLAS FP8/BF16 GEMM 接口)
│
├── normalization/             # 归一化 kernel
│   └── (LayerNorm / RMSNorm + FP8 cast 融合)
│
├── transpose/                 # 转置 kernel
│   └── (FP8 cast + transpose 融合)
│
├── comm_gemm_overlap/         # 通信-计算重叠
│   └── (AllGather + GEMM overlap)
│
├── permutation/               # Token permutation (MoE)
│   └── (专家分派/合并)
│
└── multi_tensor/              # Multi-tensor 操作
    └── (批量 scale/cast)
```

## 3. Fused Attention Kernel 详解

### 3.1 路由策略 (fused_attn.cpp)

```cpp
// fused_attn.cpp: 根据序列长度/精度/硬件选择 backend
NVTE_Fused_Attn_Backend get_fused_attn_backend(
    NVTEDType q_dtype, NVTEDType kv_dtype,
    NVTE_QKV_Layout qkv_layout,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
    float dropout, size_t num_attn_heads, size_t num_gqa_groups,
    size_t max_seqlen_q, size_t max_seqlen_kv, size_t head_dim)
{
    // 决策树:
    // 1. FP8 + 支持的 head_dim → FP8 FlashAttention
    // 2. seq_len <= 512 + 无 dropout → F16 短序列 kernel
    // 3. 任意长度 → F16 FlashAttention (cuDNN backend)
}
```

### 3.2 Flash Attention 封装 (flash_attn.cu)

```cpp
// 调用 cuDNN Flash Attention API
void fused_attn_fwd_qkvpacked(
    const Tensor *QKV, const Tensor *Bias,
    Tensor *S, Tensor *O,          // S=softmax stats, O=output
    NVTEDType dtype, float scale_factor,
    bool causal, float dropout_probability,
    cudnnHandle_t handle, cudaStream_t stream)
{
    // 配置 cuDNN descriptor
    auto plan = cudnn_frontend::get_plan(params);
    // 执行
    cudnn_frontend::execute(plan, variant_pack, workspace, stream);
}
```

### 3.3 Context Parallel 通信 (context_parallel.cu)

```cpp
// Ring attention 的 P2P 通信 kernel
void context_parallel_fwd(
    Tensor *Q, Tensor *K, Tensor *V, Tensor *O,
    int cp_size, int cp_rank,
    ncclComm_t comm, cudaStream_t stream)
{
    for (int step = 0; step < cp_size - 1; step++) {
        // 异步发送 K/V 到下一个 rank
        nccl_send(K_chunk, next_rank, comm, stream);
        nccl_recv(K_remote, prev_rank, comm, stream);
        
        // 计算当前 chunk 的 attention
        flash_attn_fwd(Q, K_remote, V_remote, O_partial);
        
        // Online softmax 合并
        merge_attention_outputs(O, O_partial, lse, lse_partial);
    }
}
```

**WHY ring attention 实现在 CUDA 层而非 Python？**
- P2P 通信与计算必须在同一 CUDA stream 精确重叠
- Python 层的 GIL 和调度延迟使得精确 overlap 不可能
- Kernel 级别可使用 NCCL CUDA kernel 直接调度

## 4. Activation 融合模板

### 4.1 模板设计 (activation_template.h)

```cpp
// activation_template.h: 统一的激活 + 量化 融合模板
template <typename ComputeType, typename ParamType,
          typename InputType, typename OutputType,
          ComputeType (*ActivationOp)(ComputeType, const ParamType &)>
__global__ void activation_fp8_kernel(
    const InputType *input,
    OutputType *output,             // FP8 输出
    const float *scale,             // 量化 scale
    float *amax,                    // 输出 amax
    const int rows, const int cols)
{
    // 1. 加载 input (BF16/FP32)
    ComputeType val = static_cast<ComputeType>(input[idx]);
    
    // 2. 应用激活函数
    val = ActivationOp(val, param);
    
    // 3. 更新 amax (block-level reduce)
    atomicMax(amax, abs(val));
    
    // 4. 量化到 FP8
    output[idx] = cast_to_fp8(val * scale[0]);
}
```

### 4.2 具体实例化

```cpp
// gelu.cu:
using GeLU_FP8_Kernel = activation_fp8_kernel<float, EmptyParam, 
                                               __nv_bfloat16, __nv_fp8_e4m3, gelu_op>;

// swiglu.cu:  
// SwiGLU(x, gate) = x * silu(gate)
// 融合: 输入 [x|gate] → SwiGLU → FP8 输出
using SwiGLU_FP8_Kernel = gated_activation_fp8_kernel<float, EmptyParam,
                                                       __nv_bfloat16, __nv_fp8_e4m3, silu_op>;
```

**WHY 模板而非独立实现？**
- 激活函数种类多 (GELU, SiLU, ReLU, GEGLU, SwiGLU...)
- 量化逻辑完全相同（只有激活函数不同）
- 模板实例化在编译期完成，零运行时开销

## 5. Cast + Transpose 融合

### 5.1 WHY 融合 cast 和 transpose？

```
分离执行:
  step 1: BF16 → FP8 (memory read + write)       O(N) bandwidth
  step 2: FP8 transpose (memory read + write)     O(N) bandwidth
  总计: 4N 次内存访问

融合执行:
  single kernel: BF16 → FP8 + 同时写入 rowwise & columnwise
  总计: 3N 次内存访问 (read once, write twice)
  
节省: 25% 带宽
```

### 5.2 实现策略

```cpp
// transpose/cast_transpose_fused.cu
__global__ void cast_transpose_kernel(
    const __nv_bfloat16 *input,    // [M, N] BF16
    __nv_fp8_e4m3 *output_row,     // [M, N] FP8 (rowwise)
    __nv_fp8_e4m3 *output_col,     // [N, M] FP8 (columnwise)
    const float *scale, float *amax,
    int M, int N)
{
    // 使用 shared memory 做 tile transpose
    __shared__ __nv_fp8_e4m3 tile[TILE_DIM][TILE_DIM + 1]; // +1 避免 bank conflict
    
    // 1. 从 global memory 读取 BF16 tile
    // 2. 计算 FP8 值 + 更新 amax
    // 3. 写入 output_row (正常写)
    // 4. tile 写入 shared memory
    __syncthreads();
    // 5. 从 shared memory 转置读取 → 写入 output_col
}
```

## 6. GEMM 封装 (gemm/)

### 6.1 FP8 GEMM 接口

```cpp
// gemm.h 关键接口
void nvte_cublas_gemm(
    const NVTETensor A, const NVTETensor B, NVTETensor D,
    const NVTETensor bias, NVTETensor pre_gelu_out,
    bool transa, bool transb,
    bool grad,           // 是否为梯度计算
    NVTETensor workspace,
    bool accumulate, bool use_split_accumulator,
    int math_sm_count,   // 使用的 SM 数量（GEMM overlap 时限制）
    cudaStream_t stream);
```

### 6.2 Comm-GEMM Overlap (comm_gemm_overlap/)

```
AllGather + GEMM 重叠策略:
─────────────────────────────────────────
时间 →
Stream 1 (comm): |AG chunk0|AG chunk1|AG chunk2|...
Stream 2 (comp):           |GEMM c0 |GEMM c1 |GEMM c2|...

将 AllGather 分为 N 个 chunk:
- chunk 0 到达后立即开始 GEMM
- 通信和计算在不同 stream 上 overlap
- 总时间 ≈ max(comm_time, compute_time) 而非 sum
─────────────────────────────────────────
```

**WHY 在 kernel 层实现 overlap？**
Python 层的 `torch.cuda.Stream` 调度有 ~10μs 延迟，
多 chunk 策略需要 μs 级精确控制，必须在 C++ 层实现。

## 7. Normalization 融合

### 7.1 LayerNorm + FP8 Cast

```cpp
// normalization/ 目录
// 融合: LayerNorm(x) → FP8 cast → output
// 节省: 1 次中间 tensor 分配 + 1 次 memory write

__global__ void layernorm_cast_fp8_kernel(
    const InputType *input,
    const float *gamma, const float *beta,
    OutputType *output_fp8,
    float *amax, const float *scale,
    float eps, int hidden_size)
{
    // Warp-level reduction 计算 mean/variance
    float mean = warp_reduce_sum(local_sum) / hidden_size;
    float var = warp_reduce_sum(local_sq_sum) / hidden_size - mean*mean;
    float rstd = rsqrtf(var + eps);
    
    // 归一化 + 量化 (单次 write)
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normed = (input[row*hidden_size + i] - mean) * rstd;
        float result = normed * gamma[i] + beta[i];
        atomicMax(amax, fabsf(result));
        output_fp8[row*hidden_size + i] = cast_fp8(result * scale[0]);
    }
}
```

## 8. MoE 专用 Kernel

### 8.1 Fused Router (fused_router/)

```cpp
// Top-K 选择 + capacity 计算 + token dispatch 索引
// 融合避免多次 kernel launch 和中间 tensor
void fused_topk_router(
    const float *router_logits,  // [num_tokens, num_experts]
    int *expert_indices,          // [num_tokens, top_k]
    float *expert_weights,        // [num_tokens, top_k]
    int *token_permutation,       // dispatch 排列索引
    int num_tokens, int num_experts, int top_k, int capacity);
```

### 8.2 Token Permutation (permutation/)

```cpp
// 按专家分组重排 token → 执行 expert GEMM → 恢复原序
// 融合 permute + unpermute 减少中间 tensor
void permute_tokens(input, permutation_indices, output);
void unpermute_tokens(input, restore_indices, output, combine_weights);
```

## 9. 性能量化对比

| Kernel 类别 | 独立执行 | 融合执行 | 带宽节省 | 适用层 |
|-------------|----------|----------|----------|--------|
| LN + Cast_FP8 | 2 launch, 4N IO | 1 launch, 3N IO | 25% | 每层 |
| Activation + Cast_FP8 | 2 launch, 4N IO | 1 launch, 3N IO | 25% | 每层 |
| Cast + Transpose | 2 launch, 4N IO | 1 launch, 3N IO | 25% | 每次 quantize |
| Fused Attention | O(N²) mem | O(N) mem | >>10× | Attention |
| AG + GEMM overlap | sum(T_comm + T_comp) | max(T_comm, T_comp) | ~40% 时间 | Linear |

## 10. 与其他章节的关联

- **→ TE-FL 第1章**: FP8 训练整体流程调用这些 kernel
- **→ TE-FL 第2章 Fused Linear**: Linear 层组合 GEMM + activation + cast kernel
- **→ TE-FL 第4章 Attention**: Python 层调用 fused_attn kernel
- **→ TE-FL 第5章 Context Parallel**: 使用 context_parallel.cu 通信 kernel
- **→ TE-FL 第6章 Userbuffers**: comm_gemm_overlap 底层通信机制
- **→ TE-FL 第10章 Float8Tensor**: cast kernel 是 Float8Tensor.quantize 的底层

## 11. 源码版本信息

- 目录: `transformer_engine/common/` (~50+ .cu/.cpp 文件)
- 核心子目录: fused_attn (9 文件), activation (5 文件), gemm, normalization, cast, transpose
- 编译: CMakeLists.txt, 支持 SM80+ (A100/H100)
- FlagScale 扩展: 平台适配层 (NPU kernel 替换), 自定义 router kernel
