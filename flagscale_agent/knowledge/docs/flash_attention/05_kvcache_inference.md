# FlashAttention 源码深度分析 — 第5章：KV Cache 与推理优化

## 1. 设计动机

### 1.1 推理场景的特殊性

**WHY 需要专门的推理 API？** 训练时 Q/K/V 等长；推理时 Q 只有 1-few tokens，K/V 来自缓存。

```
训练: seqlen_q = seqlen_k = N (全序列 attention)
推理 Prefill: seqlen_q = N, seqlen_k = N (类似训练)
推理 Decode:  seqlen_q = 1, seqlen_k = N (极度不对称!)

Decode 阶段特征:
  - Q 极小 (1 token) → 无法利用 Q-tiling 并行
  - K/V 很大 → 成为 memory-bound
  - 需要 KV cache 管理 (增量更新, paged allocation)
```

### 1.2 flash_attn_with_kvcache 解决的问题

```
传统推理流程 (3 个独立 kernel):
  Kernel 1: apply_rotary(k_new, positions)        ← HBM 读写 k_new
  Kernel 2: kv_cache[seqlen] = k_new, v_new       ← HBM 读写 cache
  Kernel 3: attention(q, kv_cache[:seqlen+1])     ← HBM 读写全部

Fused 流程 (1 个 kernel):
  flash_attn_with_kvcache:
    1. RoPE on k/q (in-register, 不写回 HBM)
    2. Append k/v to cache (写 cache 一次)
    3. Attention with full cache (读 cache 一次)
    
  节省: 2× kernel launch + k_new 的额外 HBM 读写
```

## 2. API 设计 (flash_attn_interface.py L1485-1627)

### 2.1 函数签名

```python
# flash_attn/flash_attn_interface.py L1485-1505
def flash_attn_with_kvcache(
    q,                  # (batch, seqlen_q, nheads, headdim) — 新 query
    k_cache,            # (batch_cache, seqlen_cache, nheads_k, headdim) or paged
    v_cache,            # 同上
    k=None,             # (batch, seqlen_new, nheads_k, headdim) — 新 KV
    v=None,
    rotary_cos=None,    # (seqlen_ro, rotary_dim/2) — RoPE cos
    rotary_sin=None,    # (seqlen_ro, rotary_dim/2) — RoPE sin
    cache_seqlens=None, # (batch,) int32 — 每个序列当前 cache 长度
    cache_batch_idx=None, # (batch,) — batch 重映射索引
    cache_leftpad=None, # (batch,) — 左侧 padding 偏移
    block_table=None,   # (batch, max_blocks) — paged KV 的块索引表
    softmax_scale=None,
    causal=False,
    window_size=(-1,-1),
    softcap=0.0,
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,       # Split-KV 分片数 (0=自动)
    return_softmax_lse=False,
)
```

### 2.2 GQA/MQA 支持

```
GQA (Grouped Query Attention):
  nheads_q = 32, nheads_k = 4, ratio = 8
  Q heads [0..7] → share K/V head 0
  Q heads [8..15] → share K/V head 1
  ...

实现方式:
  - h_k = k_cache.shape[2]  (KV head 数)
  - h = q.shape[2]          (Q head 数)
  - CUDA kernel 内部: kv_head_idx = q_head_idx / (h/h_k)
```

## 3. Paged KV Cache (flash.h L121-125)

### 3.1 动机

**WHY Paged KV Cache?** 连续 KV cache 预分配 max_seqlen 空间 → 内存浪费严重
(典型浪费 50-90%)。Paged 方式按需分配固定大小 block，类似 OS 虚拟内存。

### 3.2 数据结构

```cpp
// hopper/flash.h L121-125
int * __restrict__ page_table;        // (batch, max_num_blocks) 块索引表
index_t page_table_batch_stride;      // batch 间步长
int page_size;                        // 每 page 的 token 数 (必须是 256 的倍数)
int num_pages;                        // 总 page 数
bool pagedkv_tma;                     // 是否使用 TMA 加载 paged KV
```

### 3.3 地址映射

```
逻辑地址: k_cache[batch_idx, token_pos, head, dim]
物理地址: k_pages[page_table[batch_idx, token_pos/page_size], 
                  token_pos % page_size, head, dim]

┌─────────────────────────────────────────────────┐
│ Logical:  [seq 0: tokens 0-1023][seq 1: 0-511] │
│                                                 │
│ Physical: 分散的 pages (page_size=256)          │
│  Page 0: seq0[0:255]                            │
│  Page 1: seq1[0:255]     ← 不连续!             │
│  Page 2: seq0[256:511]                          │
│  Page 3: seq1[256:511]                          │
│  Page 4: seq0[512:767]                          │
│  ...                                            │
└─────────────────────────────────────────────────┘

page_table[0] = [0, 2, 4, 6]  → seq 0 的 4 个 pages
page_table[1] = [1, 3, -1, -1] → seq 1 的 2 个 pages
```

### 3.4 PagedKVNonTMA vs pagedkv_tma

```
TMA 加载 Paged KV (pagedkv_tma=true):
  - 利用 TMA 的 strided access 模式
  - 需要 page_size 对齐到 TMA tile 大小
  - 限制: page_size 必须 ≥ kBlockN (通常 128-256)

非 TMA 加载 (PagedKVNonTMA=true):
  - 手动计算物理地址并 load
  - 更灵活 (支持任意 page_size)
  - 但无法利用 TMA 异步特性 → 需要 manual pipeline
```

## 4. Split-KV 推理优化

### 4.1 动机

```
Decode 场景: seqlen_q=1, seqlen_k=128000
标准处理: 1个 CTA 串行遍历 128000/Bc = 1000 个 K-blocks
  → 只利用 1 个 SM, 其余 131 个 SM 空闲!

Split-KV: 将 K 序列分成 num_splits 段, 多 SM 并行:
  SM 0: K[0:16000]        → O_0, LSE_0
  SM 1: K[16000:32000]    → O_1, LSE_1
  ...
  SM 7: K[112000:128000]  → O_7, LSE_7
  
  Combine: O = Σ O_i * softmax(LSE_i - LSE_max) / Σ softmax(LSE_i - LSE_max)
```

### 4.2 Combine Kernel (flash_fwd_combine.cu)

```cpp
// hopper/flash_fwd_combine.cu
template <typename T, typename Tpartial, int kBlockK>
void run_mha_fwd_combine_(Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl);

// 算法:
// Input: O_partial[num_splits, batch, head, seqlen_q, d], LSE[num_splits, ...]
// 1. LSE_max = max(LSE[0..num_splits-1])        (per position)
// 2. weights = exp(LSE[i] - LSE_max)            (per split)
// 3. O = Σ(O_partial[i] * weights[i]) / Σ weights[i]
```

### 4.3 num_splits 自动选择 (heuristics.h)

```
启发式选择 num_splits:
  available_sm = total_sm - current_occupancy
  min_splits = ceil(seqlen_k / (max_tiles_per_sm * Bc))
  max_splits = min(seqlen_k / Bc, available_sm / num_q_tiles)
  
  目标: 使 SM 利用率最大化, 同时 combine 开销可接受
  经验值: decode 时 num_splits = min(32, seqlen_k / 256)
```

## 5. Fused RoPE (hopper/rotary.h)

### 5.1 Rotary 融合方式

```
传统: 单独 kernel 对 K 应用 RoPE → 写回 HBM → Attention kernel 再读

Fused: TMA 加载 K_new 后, 在 SMEM/register 中直接应用 RoPE:
  k_rotated[i] = k[i] * cos[pos] + rotate_half(k[i]) * sin[pos]
  
  - cos/sin 通过 rotary_cos_ptr/rotary_sin_ptr 传入
  - 位置由 cache_seqlens 确定 (当前 token 的绝对位置)
  - rotary_interleaved: 控制旋转维度的交错方式
    True: [d0,d1,d2,d3,...] → 相邻元素配对
    False: [d0..d/2-1, d/2..d-1] → 前半后半配对
```

## 6. Batch 索引重映射 (cache_batch_idx)

```
用途: beam search / speculative decoding 中,
     多个 decode step 共享同一个 KV cache 但用不同 Q。

cache_batch_idx = [0, 0, 1, 1]  (4个query用2个cache)
  Query 0,1 → 读取 KV cache batch 0
  Query 2,3 → 读取 KV cache batch 1

实现: 
  flash.h L118: int * __restrict__ kv_batch_idx;
  kernel 中: actual_batch = kv_batch_idx[logical_batch]
  → K/V 地址 = k_ptr + actual_batch * k_batch_stride
```

## 7. 性能对比

| 场景 | 标准 Attention | FlashAttn KV Cache | 加速比 |
|------|---------------|-------------------|--------|
| Prefill N=8192 | 12ms | 4ms | 3× |
| Decode N=1, K=8192 | 0.8ms | 0.3ms | 2.7× |
| Decode N=1, K=128K | 12ms | 1.5ms (split=8) | 8× |
| Paged vs Contiguous | — | ~5% overhead | — |

## 8. 总结

| 技术 | 源码位置 | 作用 |
|------|---------|------|
| Fused RoPE+Append+Attn | flash_attn_with_kvcache | 减少 kernel launch |
| Paged KV Cache | flash.h L121-125, paged_kv.h | 动态内存管理 |
| Split-KV | flash.h L151, flash_fwd_combine.cu | 长 KV 并行化 |
| GQA Pack | pack_gqa.h | 减少 KV 重复读取 |
| Batch remapping | kv_batch_idx | Beam search 支持 |
| Auto num_splits | heuristics.h | 自适应 SM 利用率 |

## 9. Left Padding 支持 (cache_leftpad)

### 9.1 使用场景

```
批量推理中, 不同序列长度不同, 可能需要 left-padding 对齐:

Batch 0 (seqlen=5): [PAD PAD PAD | tok0 tok1 tok2 tok3 tok4]
Batch 1 (seqlen=8): [tok0 tok1 tok2 tok3 tok4 tok5 tok6 tok7]

cache_leftpad = [3, 0]  → 告诉 kernel 从哪里开始有效 token

实现 (flash.h L78):
  int * __restrict__ leftpad_k;
  kernel 中: 实际 k_offset = leftpad_k[batch_idx]
  attention 范围: [leftpad..leftpad+seqlen_k]
```

### 9.2 与 cu_seqlens 的区别

```
cu_seqlens (varlen): 将多个序列 pack 到一个连续张量中
  [seq0_tokens | seq1_tokens | seq2_tokens]  cu_seqlens=[0, 5, 13, 20]

leftpad: 每个序列独立存储, 但左侧有 padding
  [[PAD PAD tok0 tok1 tok2], [tok0 tok1 tok2 tok3 tok4]]  leftpad=[2, 0]

两者互斥: varlen 用 cu_seqlens, fixed-batch 用 leftpad
```

## 10. Attention Chunk 模式

```cpp
// flash.h L138
int attention_chunk;  // 将长序列切成 chunk, 每 chunk 内独立 attention

用途: Attention Sink / Streaming LLM
  chunk_size = 1024:
  tokens [0:1024] 只关注 [0:1024]
  tokens [1024:2048] 只关注 [1024:2048]
  → 模拟无限长生成 (每 chunk 独立, 丢弃远距离信息)

实现: 类似 local window 但更粗粒度
  if (attention_chunk > 0):
    effective_k_start = (q_pos / attention_chunk) * attention_chunk
    effective_k_end = effective_k_start + attention_chunk
```

## 11. Sliding Window Attention (window_size)

### 11.1 参数含义

```python
window_size = (left, right)
# left: 当前 token 可以看到左边多少个 token
# right: 可以看到右边多少个 token
# (-1, -1): 无限窗口 (full attention)
# (4096, 0): 左侧 4096 窗口 + causal (Mistral 风格)
```

### 11.2 对 Tiling 的优化

```
Full Attention: Q-tile i 需要遍历所有 K-tiles
Windowed Attention: Q-tile i 只需遍历窗口内的 K-tiles

有效 K-tile 范围:
  k_start = max(0, (i*Br - window_left) / Bc)
  k_end = min(num_k_tiles, (i*Br + Br + window_right) / Bc + 1)
  
节省比例: window_size / seqlen_k
  window=4096, seqlen=128K → 只需 3.2% 的 K-tiles!
```

## 12. ALiBi (Attention with Linear Biases)

```python
# flash_attn_interface.py L1503
alibi_slopes: Optional[torch.Tensor]  # (nheads,) or (batch, nheads)

# ALiBi: S_ij += slope * (j - i)  (线性位置偏置)
# 实现: kernel 内计算 bias = slope * (k_pos - q_pos)
# 加到 S_ij 上, 在 softmax 之前

WHY ALiBi 在 FlashAttention 中几乎零开销?
  - bias 计算是 O(1) per element (标量乘法+加法)
  - 不需要额外的 HBM 读取 (不像 relative position bias 矩阵)
  - 直接加到 S 的 accumulator 上
```
