# FlashAttention 源码深度分析 — 第1章：架构总览与核心设计

## 1. 设计动机与核心问题

### 1.1 为什么需要 FlashAttention？

**问题本质**：标准 Attention 的内存复杂度为 O(N²)，对于长序列（N=128K+）完全无法放入 GPU HBM。

```
标准Attention内存瓶颈分析:
┌─────────────────────────────────────────────────────┐
│  Q × K^T → S (N×N)   →   softmax(S)  →   P × V    │
│                                                     │
│  seq_len=8192, heads=32, fp16:                      │
│  S矩阵: 8192×8192×2bytes = 128MB (每head)          │
│  32 heads = 4GB  ← 仅attention score就爆显存       │
└─────────────────────────────────────────────────────┘
```

**WHY tiling？** GPU 的内存层次: HBM(80GB, 3.35TB/s) → SRAM(~20MB, ~50TB/s)。
FlashAttention 的核心思想：将 O(N²) 的中间矩阵保持在 SRAM 中，通过 tiling 分块计算，
避免 HBM ↔ SRAM 之间的 N² 级别数据搬运。

### 1.2 IO 复杂度对比

| 方法 | HBM 读写量 | SRAM 使用 | 数值等价 |
|------|-----------|----------|---------|
| 标准 Attention | O(N²d + N²) | O(N²) | 精确 |
| FlashAttention | O(N²d²/M) | O(M) | 精确 |
| 稀疏 Attention | O(N√N) | O(N) | 近似 |

其中 M = SRAM 大小，d = head_dim。当 M = O(Nd) 时，IO 降为 O(N²d/N) = O(Nd)。

## 2. 代码仓库结构 (commit: main branch)

```
flash-attention/
├── flash_attn/                     # Python 接口层
│   ├── flash_attn_interface.py     # 1627行 - 核心API（6个Function类 + 7个用户函数）
│   ├── bert_padding.py             # 218行 - padding/unpadding工具
│   └── flash_attn_triton.py        # 1160行 - Triton 后端（AMD ROCm）
├── hopper/                         # SM90 (H100) CUDA Kernel
│   ├── flash.h                     # 224行 - Flash_fwd_params/Flash_bwd_params 结构体
│   ├── flash_api.cpp               # PyTorch C++ 绑定入口
│   ├── mainloop_fwd_sm90_tma_gmma_ws.hpp  # 前向主循环（TMA + WGMMA）
│   ├── mainloop_bwd_sm90_tma_gmma_ws.hpp  # 反向主循环
│   ├── flash_fwd_kernel_sm90.h     # 前向kernel启动模板
│   ├── flash_bwd_kernel_sm90.h     # 反向kernel启动模板
│   ├── softmax.h                   # Online Softmax 实现
│   ├── mask.h                      # Causal/Local mask
│   ├── rotary.h                    # Fused RoPE
│   └── tile_scheduler.hpp          # Tile 调度（persistent kernel）
├── csrc/                           # SM80 (A100) CUDA Kernel
│   ├── flash_attn/                 # SM80版本源码
│   └── flash_fwd_kernel_sm80.h     # SM80前向kernel
└── tests/                          # 测试套件
```

## 3. 分层架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                   Python 用户接口层                           │
│  flash_attn_func() / flash_attn_varlen_func()               │
│  flash_attn_with_kvcache()                                   │
├─────────────────────────────────────────────────────────────┤
│              torch.autograd.Function 层                       │
│  FlashAttnFunc.forward() / .backward()                       │
│  FlashAttnVarlenFunc / FlashAttnQKVPackedFunc               │
├─────────────────────────────────────────────────────────────┤
│           C++ Binding 层 (flash_api.cpp)                     │
│  mha_fwd() / mha_bwd() / mha_fwd_kvcache()                 │
│  参数校验 → Flash_fwd_params 填充 → kernel dispatch         │
├─────────────────────────────────────────────────────────────┤
│              CUDA Kernel 层                                   │
│  ┌──────────────┐  ┌──────────────────────────────────┐     │
│  │ SM80 (A100)  │  │ SM90 (H100/Hopper)              │     │
│  │ flash_fwd_   │  │ CollectiveMainloopFwdSm90       │     │
│  │ kernel_sm80  │  │ TMA + WGMMA + Persistent Thread │     │
│  └──────────────┘  └──────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## 4. 核心参数结构体 (hopper/flash.h)

### 4.1 Qkv_params 基类 (L12-33)

```cpp
// hopper/flash.h L12-33
struct Qkv_params {
    using index_t = int64_t;
    void *__restrict__ q_ptr;       // Q 矩阵指针
    void *__restrict__ k_ptr;       // K 矩阵指针
    void *__restrict__ v_ptr;       // V 矩阵指针
    // 步长系统：支持任意 strided tensor
    index_t q_batch_stride, k_batch_stride, v_batch_stride;
    index_t q_row_stride, k_row_stride, v_row_stride;
    index_t q_head_stride, k_head_stride, v_head_stride;
    index_t v_dim_stride;
    int h, h_k;  // Q heads 数, KV heads 数（支持 GQA/MQA）
};
```

**WHY 继承设计？** Forward 和 Backward 共享 QKV 相关参数，Backward 只需额外添加 dO/dQ/dK/dV。

### 4.2 Flash_fwd_params 前向参数 (L37-168)

关键字段分组：

| 字段组 | 源码行 | 用途 |
|--------|-------|------|
| 输出相关 | L40-48 | o_ptr, oaccum_ptr + strides |
| FP8 缩放 | L53-62 | q/k/v_descale_ptr（Per-head FP8 量化） |
| 维度信息 | L65-68 | b, seqlen_q/k, d, dv, total_q/k |
| Varlen 支持 | L75-82 | cu_seqlens_q/k, seqused_q/k |
| KV Cache | L95-105 | knew_ptr/vnew_ptr（增量KV） |
| RoPE | L112-115 | rotary_cos/sin_ptr（Fused旋转） |
| Paged KV | L121-125 | page_table, page_size, num_pages |
| Split-KV | L151 | num_splits（长序列分片） |
| Scheduling | L154-164 | tile_count_semaphore, persistent kernel 调度 |
| 硬件适配 | L166-167 | arch, num_sm |

### 4.3 Flash_bwd_params 反向参数 (L172-214)

```cpp
// hopper/flash.h L172-214
struct Flash_bwd_params : public Flash_fwd_params {
    void *__restrict__ do_ptr;          // dO 梯度输入
    void *__restrict__ dq_ptr, *dk_ptr, *dv_ptr;  // dQ/dK/dV 输出
    void *__restrict__ dq_accum_ptr;    // dQ 原子累加缓冲
    void *__restrict__ dk_accum_ptr, *dv_accum_ptr;
    // 信号量（用于多 split 间的原子同步）
    int *__restrict__ dq_semaphore;
    int *__restrict__ dk_semaphore;
    int *__restrict__ dv_semaphore;
    bool deterministic;                 // 确定性模式（影响累加顺序）
};
```

**WHY dq_accum_ptr？** 反向传播中多个 K-block 需要向同一个 dQ 位置累加梯度，使用 float32
累加缓冲避免 fp16 精度损失。

## 5. Python API 层设计 (flash_attn_interface.py)

### 5.1 torch.autograd.Function 封装模式

```python
# flash_attn/flash_attn_interface.py L461-540
class FlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, dropout_p, softmax_scale, causal,
                window_size, softcap, alibi_slopes, deterministic,
                return_softmax, is_grad_enabled):
        # 1. Head dimension 对齐到 8 的倍数（硬件要求）
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
        # 2. 调用 C++ kernel
        out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(...)
        # 3. 保存反向所需张量
        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
        return out[..., :head_size_og]  # 截断回原始维度
```

**WHY pad to 8？** CUDA kernel 的向量化访存（128-bit load/store）要求 head_dim 是 8 的倍数。

### 5.2 6个 Function 类对应不同输入模式

| 类名 | 输入格式 | 适用场景 |
|------|---------|---------|
| FlashAttnQKVPackedFunc | qkv: [B,S,3,H,D] | Q/K/V 打包存储 |
| FlashAttnVarlenQKVPackedFunc | qkv + cu_seqlens | 变长序列打包 |
| FlashAttnKVPackedFunc | q + kv: [B,S,2,H,D] | Q 单独, KV 打包 |
| FlashAttnVarlenKVPackedFunc | q + kv + cu_seqlens | 变长 + KV打包 |
| FlashAttnFunc | q, k, v 分离 | 最通用接口 |
| FlashAttnVarlenFunc | q,k,v + cu_seqlens | 变长 + 分离输入 |

### 5.3 用户级 API 函数

```python
# flash_attn/flash_attn_interface.py L1156
def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None,
                    causal=False, window_size=(-1,-1), softcap=0.0,
                    alibi_slopes=None, deterministic=False,
                    return_attn_probs=False):
    """
    参数说明:
    - q: (batch_size, seqlen, nheads, headdim)
    - k/v: (batch_size, seqlen_k, nheads_k, headdim)  ← 支持GQA
    - window_size: 滑动窗口 [left, right]
    - softcap: tanh soft-capping (Gemma2 用)
    - alibi_slopes: ALiBi 位置编码斜率
    """
```

### 5.4 KV Cache API (L1485-1627)

```python
# flash_attn/flash_attn_interface.py L1485
def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None,
                            rotary_cos=None, rotary_sin=None,
                            cache_seqlens=None, block_table=None, ...):
    """
    Fused 操作合一:
    1. 将新 k/v append 到 cache（inplace更新）
    2. 应用 RoPE 到 k 和 q
    3. 执行 attention（支持 paged KV cache）
    
    WHY fused? 避免3次独立kernel launch的开销:
    - 传统: rotary_kernel → kv_cache_update → attention_kernel
    - Fused: 单kernel完成全部操作，减少2次kernel launch + HBM读写
    """
```

## 6. SM90 Hopper 架构 Kernel 设计

### 6.1 核心模板参数 (mainloop_fwd_sm90_tma_gmma_ws.hpp L31-36)

```cpp
// hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp L31-36
template <int Stages, class ClusterShape_, class TileShape_MNK_,
          int kHeadDimV, class Element_, class ElementAccum_, class ArchTag_,
          bool Is_causal_, bool Is_local_, bool Has_softcap_, bool Varlen_,
          bool PagedKVNonTMA_, bool AppendKV_, bool HasQv_,
          bool MmaPV_is_RS, bool IntraWGOverlap, bool PackGQA_,
          bool Split_, bool V_colmajor_>
struct CollectiveMainloopFwdSm90 { ... };
```

**WHY 如此多 bool 模板参数？** 编译期特化，避免运行时分支。每种组合生成独立 kernel binary，
确保零开销抽象。代价是编译时间长（instantiations/ 目录有大量特化）。

### 6.2 硬件特性利用

| Hopper 特性 | 在 FlashAttention 中的应用 | 源码位置 |
|------------|---------------------------|---------|
| TMA (Tensor Memory Accelerator) | 异步 Q/K/V 数据搬运 HBM→SRAM | mainloop L:Use_TMA_Q/KV |
| WGMMA (Warpgroup MMA) | QK^T 和 PV 矩阵乘 | TiledMmaQK/TiledMmaPV |
| Persistent Thread | 减少 kernel launch overhead | tile_scheduler.hpp |
| Cluster | 跨 SM 协作（DSMEM） | ClusterShape_ 模板参数 |
| PDL (Programmatic Dependent Launch) | varlen 预处理 pipeline | prepare_varlen_pdl |

### 6.3 Tile 大小选择策略

```
SM90 Tile 配置 (hopper/tile_size.h):
┌─────────────┬──────────────┬─────────────┬───────────────────┐
│ HeadDim (d) │ kBlockM (Br) │ kBlockN (Bc)│ Stages (Pipeline) │
├─────────────┼──────────────┼─────────────┼───────────────────┤
│ 64          │ 192          │ 128         │ 2                 │
│ 128         │ 128          │ 128         │ 2                 │
│ 192         │ 128          │ 96          │ 2                 │
│ 256         │ 128          │ 64          │ 2                 │
└─────────────┴──────────────┴─────────────┴───────────────────┘

WHY 这些选择?
- kBlockM × d × sizeof(fp16) ≤ SRAM_per_SM / num_stages
- kBlockN 影响 softmax 重计算频率
- Pipeline stages=2: 双缓冲，隐藏 TMA latency
```

### 6.4 GQA Pack 优化 (pack_gqa.h)

```
GQA Pack 示意 (nheads_q=8, nheads_kv=2, ratio=4):
                                                    
未Pack:  每个 Q head 独立处理 → 4个head重复读取同一KV          
                                                    
Pack后:  将 4 个 Q heads 打包为一组:                          
┌────┬────┬────┬────┐                                
│ Q0 │ Q1 │ Q2 │ Q3 │  → 共享同一 K/V head              
└────┴────┴────┴────┘                                
效果: kBlockM *= ratio, 单次 tile 处理更多 Q tokens          
```

**WHY Pack GQA？** 减少 KV 重复加载次数，当 GQA ratio=8 时减少 ~8× KV HBM 读取。

## 7. Online Softmax 算法 (核心数学)

### 7.1 三趟变两趟

```
标准 Softmax（3趟）:
  Pass 1: m = max(S_i)           — 需要全部 N 个值
  Pass 2: l = Σ exp(S_i - m)     — 需要全部 N 个值  
  Pass 3: P_i = exp(S_i - m) / l — 需要全部 N 个值

Online Softmax（单趟 + 递推更新）:
  对每个 block j:
    m_new = max(m_old, max(S_block_j))
    l_new = l_old * exp(m_old - m_new) + Σ exp(S_block_j - m_new)
    O_new = O_old * (l_old * exp(m_old - m_new) / l_new)
          + exp(S_block_j - m_new) / l_new × V_block_j
```

### 7.2 Kernel 中的 LSE (Log-Sum-Exp) 保存

```python
# 保存 softmax_lse = log(l) + m，用于:
# 1. 反向传播重计算 softmax
# 2. Split-KV 多 split 结果合并
# 形状: (batch, nheads, seqlen_q) — 仅 O(N) 空间
```

## 8. 反向传播策略

### 8.1 重计算 vs 存储

| 存储项 | 大小 | 用途 |
|-------|------|------|
| O (输出) | O(BNd) | 反向时不重新计算 |
| softmax_lse | O(BN) | 重计算 P = softmax(QK^T) |
| rng_state | 8 bytes | 重现 dropout mask |
| S (attention score) | ❌ 不存 | 反向时重计算 ← 节省 O(N²) |
| P (attention prob) | ❌ 不存 | 反向时重计算 ← 节省 O(N²) |

**WHY 重计算而非存储？** O(N²) → O(N) 内存，计算多出一倍 QK^T GEMM 但减少 HBM IO。

### 8.2 dQ 累加问题 (Flash_bwd_params L177-183)

```
反向数据流:
dO × V^T → dS（对 K-block 循环）
dS × K  → dQ_partial（每个 K-block 产生部分 dQ）

问题: 多个 K-block 写入同一 dQ 位置
解决方案:
  1. dq_accum_ptr: float32 缓冲，原子累加
  2. dq_semaphore: 信号量同步，确保所有 split 完成后再 downcast
  3. deterministic=True: 强制固定累加顺序（牺牲性能换精度复现）
```

## 9. 与 Megatron/TE-FL 集成方式

### 9.1 调用路径

```
Megatron-LM-FL 调用链:
  TransformerLayer
  └─ SelfAttention
     └─ core_attention (attention.py)
        └─ flash_attn_func() / flash_attn_varlen_func()
           └─ flash_attn_2_cuda.fwd()
              └─ run_mha_fwd_<SM90>()

TransformerEngine-FL 调用链:
  DotProductAttention
  └─ FusedAttention (backend="FlashAttention")
     └─ _fused_attn_fwd()  ← TE 自己封装了一层
```

### 9.2 Context Parallel 与 FlashAttention

在 Context Parallel (Ring Attention) 中，FlashAttention 的 varlen 接口是关键：
- 每个 CP rank 持有部分 Q 和完整（或分段）K/V
- 使用 `flash_attn_varlen_func` + `cu_seqlens` 处理不等长分片
- LSE 需要跨 rank 合并（allreduce max + sum_exp）

## 10. 性能特征总结

| 指标 | 值 (H100, d=128, seq=8192, bf16) |
|------|-------------------------------|
| 前向 TFLOPS | ~400-500 TFLOPS |
| 反向 TFLOPS | ~350-400 TFLOPS |
| HBM 带宽利用 | ~2.8 TB/s (接近峰值3.35) |
| 相比标准 Attention 加速 | 2-4× (取决于 seq_len) |
| 内存节省 | O(N²) → O(N) |

## 11. 总结：FlashAttention 设计哲学

1. **IO-Aware**: 以 HBM 带宽为首要优化目标，非 FLOPs
2. **Exact**: 数值等价于标准 attention（非近似方法）
3. **Fused**: 尽量将多操作合并到单 kernel（RoPE + KV update + Attention）
4. **Hardware-Specific**: SM80/SM90 分别优化，充分利用各代硬件特性
5. **Compile-time Specialization**: 模板参数消除运行时分支
