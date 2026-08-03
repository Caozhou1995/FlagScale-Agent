# 第17章：RoPE 旋转位置编码系统 深度源码分析

## 1. 概述与设计动机

### 1.1 核心问题

Transformer 的自注意力机制对 token 顺序不敏感（permutation invariant）。
位置编码为每个 token 注入位置信息。RoPE (Rotary Position Embedding) 的核心优势：
- 相对位置自然编码（Q·K^T 只依赖相对距离）
- 无需额外参数（通过旋转矩阵实现）
- 良好的长度外推性（通过 scaling/YaRN 等方案）

### 1.2 WHY: 为什么 Megatron 需要独立的 RoPE 模块？

RoPE 在分布式训练中有特殊约束：
1. **Context Parallel**: 序列被切分到多个 rank，每个 rank 只需对应位置的 embedding
2. **Sequence Parallel**: TP 切分时序列长度变化，需乘以 `tp_world_size` 恢复
3. **Inference**: 需要增量位置偏移（KV cache 场景）
4. **Multimodal**: Qwen2-VL 等模型需要 3D 位置编码（T/H/W）

### 1.3 数学基础

```
RoPE 核心公式:
  θ_i = base^(-2i/d), i ∈ [0, d/2)          频率向量
  R(pos) = [cos(pos·θ), sin(pos·θ)]         旋转矩阵

对 query/key 向量 x 应用:
  RoPE(x, pos) = [x_even·cos(pos·θ) - x_odd·sin(pos·θ),
                  x_even·sin(pos·θ) + x_odd·cos(pos·θ)]

Q·K^T 的结果只依赖 (pos_q - pos_k)，实现相对位置编码。
```

## 2. 源码定位

| 文件 | 路径 | 行数 | 职责 |
|------|------|------|------|
| rotary_pos_embedding.py | `megatron/core/models/common/embeddings/rotary_pos_embedding.py` | 451 | RoPE 主类 |
| rope_utils.py | `megatron/core/models/common/embeddings/rope_utils.py` | ~200 | apply 函数 |
| TE rope.py | `transformer_engine/pytorch/attention/rope.py` | 542 | TE 融合 kernel |

## 3. RotaryEmbedding 类详解 (L41-310)

### 3.1 构造函数 (L63-95)

```python
# rotary_pos_embedding.py L63-95
class RotaryEmbedding(nn.Module):
    def __init__(self,
        kv_channels: int,            # 注意力头维度 (d_head)
        rotary_percent: float,       # 使用 RoPE 的维度比例 (0.0-1.0)
        rotary_interleaved: bool,    # True: [r0,r0,r1,r1,...] vs False: [r0,...,r_{d/2-1},r0,...]
        seq_len_interpolation_factor: float,  # 长度外推因子
        rotary_base: int = 10000,    # θ base (高 base = 更好的长序列建模)
        rope_scaling: bool = False,  # LLaMA 3.x 风格 scaling
        rope_scaling_factor: float = 8.0,
        use_cpu_initialization: bool = False,
        cp_group = None,             # Context Parallel 进程组
    ):
        dim = int(kv_channels * rotary_percent)  # 实际使用的 RoPE 维度
        
        # 核心: 计算频率倒数
        # inv_freq[i] = 1 / (base^(2i/dim)), i ∈ [0, dim/2)
        self.inv_freq = 1.0 / (
            rotary_base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        
        if rope_scaling:
            self.inv_freq = self._apply_scaling(self.inv_freq, factor=rope_scaling_factor)
```

**WHY `rotary_percent`？**
某些模型（如 GPT-NeoX）只对部分维度应用 RoPE，其余维度保持绝对位置自由。

### 3.2 LLaMA 3.x Rope Scaling (L97-130)

```python
# L97-130: 三段式频率调整
def _apply_scaling(self, freqs, factor=8, low_freq_factor=1, 
                   high_freq_factor=4, original_max_position_embeddings=8192):
    old_context_len = original_max_position_embeddings
    low_freq_wavelen = old_context_len / low_freq_factor   # 8192
    high_freq_wavelen = old_context_len / high_freq_factor # 2048
    
    wavelen = 2 * π / freqs
    
    # 三段处理:
    # wavelen < 2048: 高频，不缩放（短距离位置信息保留）
    # wavelen > 8192: 低频，除以 factor（长距离外推）
    # 2048-8192:     中频，平滑插值
    inv_freq_llama = where(wavelen > low_freq_wavelen, freqs / factor, freqs)
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    inv_freq_llama = where(is_medium_freq, smoothed, inv_freq_llama)
```

**WHY 三段式？**
- 高频信息（局部位置关系）对模型性能关键，不能损失
- 低频信息（远距离关系）需要缩放来支持更长序列
- 中频平滑过渡避免分布突变

### 3.3 频率矩阵生成 (L132-180)

```python
# L132: 非重复频率矩阵
def get_freqs_non_repeated(self, max_seq_len, offset=0):
    seq = torch.arange(max_seq_len) + offset    # [0, 1, 2, ..., L-1]
    if self.seq_len_interpolation_factor:
        seq *= 1 / self.seq_len_interpolation_factor  # 线性插值外推
    freqs = torch.outer(seq, self.inv_freq)     # [L, dim/2]
    return freqs

# L155: 完整 embedding
def get_emb(self, max_seq_len, offset=0):
    freqs = self.get_freqs_non_repeated(max_seq_len, offset)
    if not self.rotary_interleaved:
        emb = torch.cat((freqs, freqs), dim=-1)  # [L, dim] (复制拼接)
    else:
        emb = torch.stack((freqs, freqs), dim=-1).view(...)  # 交错排列
    emb = emb[:, None, None, :]  # [L, 1, 1, dim] 方便 broadcast
    return emb
```

### 3.4 forward 与 Context Parallel 切分 (L182-211)

```python
# L182-211
@lru_cache(maxsize=32)  # 缓存常见长度，避免重复计算
def forward(self, max_seq_len, offset=0, packed_seq=False, cp_group=None):
    emb = self.get_emb(max_seq_len, offset)
    
    # 关键: CP 切分
    if cp_group is not None and cp_group.size() > 1 and not packed_seq:
        # 将 [0, L) 的 embedding 切分为 cp_size 份
        # 当前 CP rank 只取自己负责的那段位置
        emb = get_pos_emb_on_this_cp_rank(emb, seq_dim=0, cp_group=cp_group)
    
    return emb
```

**WHY `lru_cache`？**
训练中同一 micro-batch 的所有样本长度相同（固定 seq_len），
同一步内所有 micro-batch 也共享相同 embedding → 缓存命中率极高。

### 3.5 Cos/Sin 缓存机制 (L213-252)

```python
# L213-252: 推理优化 — 预计算并缓存 cos/sin
def _set_cos_sin_cache(self, seq_len, offset, dtype, packed_seq=False, cp_group=None):
    self.max_seq_len_cached = seq_len
    emb = self.forward(seq_len, offset, packed_seq, cp_group)
    self.register_buffer("cos_cached", emb.cos().to(dtype).contiguous())
    self.register_buffer("sin_cached", emb.sin().to(dtype).contiguous())

def get_cached_cos_sin(self, seq_len, offset=0, dtype=..., ...):
    """按需重建缓存"""
    if seq_len > self.max_seq_len_cached or offset != self.offset_cached:
        self._set_cos_sin_cache(seq_len, offset, dtype, packed_seq, cp_group)
    return (self.cos_cached[:seq_len], self.sin_cached[:seq_len])
```

**WHY register_buffer？**
- 缓存随模型保存/加载（但不参与梯度计算）
- persistent=False: 不序列化到 state_dict（因为可重建）

## 4. 序列长度计算 (L258-309)

### 4.1 get_rotary_seq_len

```python
# L258-309: 确定 RoPE 需要覆盖的最大长度
def get_rotary_seq_len(self, inference_context, transformer, transformer_input, ...):
    
    if packed_seq_params is not None:
        # 打包序列: 用最大子序列长度
        return max(packed_seq_params.max_seqlen_q, packed_seq_params.max_seqlen_kv)
    
    elif inference_context is not None:
        # 推理模式: max(context限制, 实际输入长度)
        rotary_seq_len = max(context_max_seq_len, input_seq_len)
    
    else:
        # 训练模式: 输入 tensor 的序列维度
        rotary_seq_len = transformer_input.size(0)
    
    # Sequence Parallel: TP 切分后需要恢复原始长度
    if transformer_config.sequence_parallel:
        rotary_seq_len *= tp_world_size  # FlagScale Add (L305)
    
    # Context Parallel: 需要覆盖全局序列长度
    rotary_seq_len *= transformer_config.context_parallel_size  # L307
    
    return rotary_seq_len
```

**WHY 乘以 `context_parallel_size`？**
CP 将序列切分到多个 rank，但 RoPE 生成时需要全局位置信息，
然后在 forward() 中用 `get_pos_emb_on_this_cp_rank` 截取本地部分。

## 5. MultimodalRotaryEmbedding (L312-451)

### 5.1 设计动机

Qwen2-VL / Qwen3.5-VL 需要 3D 位置编码：
- 时间维度 T（视频帧）
- 高度维度 H
- 宽度维度 W

### 5.2 两种 MRoPE 布局

```python
# L343: interleaved_mrope 参数控制布局

# Qwen2-VL 风格 (section-based):
# dim 分为 3 段: [T_dims | H_dims | W_dims]
# pos_ids shape: [3, seq_len]

# Qwen3.5-VL 风格 (interleaved):
# dim 交错: [T_0, H_0, W_0, T_1, H_1, W_1, ...]
# pos_ids shape: [3, seq_len]
```

### 5.3 forward (L380-451)

```python
def forward(self, max_seq_len, position_ids, packed_seq=False, ...):
    """
    position_ids: [3, seq_len] — T/H/W 三维位置
    """
    inv_freq_expanded = self.inv_freq[None, :, None]  # [1, dim/2, 1]
    
    if not self.interleaved_mrope:
        # Section-based: 每段独立计算
        section_size = dim // 3
        freqs_t = position_ids[0] * inv_freq[:section_size]
        freqs_h = position_ids[1] * inv_freq[section_size:2*section_size]
        freqs_w = position_ids[2] * inv_freq[2*section_size:]
        freqs = cat([freqs_t, freqs_h, freqs_w], dim=-1)
    else:
        # Interleaved: T/H/W 交错
        freqs = sum of position_ids[k] * inv_freq[k::3]
```

## 6. RoPE 应用函数 (rope_utils.py)

### 6.1 两种 layout

```python
# _apply_rotary_pos_emb_bshd: 标准 layout [batch, seq, head, dim]
def _apply_rotary_pos_emb_bshd(t, freqs, rotary_interleaved):
    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]  # 部分维度不旋转
    
    cos_ = torch.cos(freqs)
    sin_ = torch.sin(freqs)
    t = (t * cos_) + (_rotate_half(t) * sin_)  # 核心旋转操作
    return torch.cat((t, t_pass), dim=-1)

# _apply_rotary_pos_emb_thd: packed sequence layout [total_tokens, head, dim]
def _apply_rotary_pos_emb_thd(t, cu_seqlens, freqs):
    # 按 cu_seqlens 索引位置，支持变长序列
```

### 6.2 WHY 两种 layout？

| Layout | 适用场景 | 特点 |
|--------|----------|------|
| BSHD | 固定长度训练 | 简单高效，tensor 连续 |
| THD | 打包变长序列 | FlashAttention varlen 接口 |

### 6.3 CP rank 切分 (rope_utils.py)

```python
def get_pos_emb_on_this_cp_rank(pos_emb, seq_dim, cp_group):
    """切分 position embedding 到 CP ranks"""
    cp_size = cp_group.size()
    cp_rank = cp_group.rank()
    cp_idx = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], ...)
    # 使用 striped 切分确保因果一致性
    pos_emb = pos_emb.view(*shape[:seq_dim], 2*cp_size, -1, *shape[seq_dim+1:])
    pos_emb = pos_emb.index_select(seq_dim, cp_idx)
    return pos_emb.view(original_shape_with_local_seq_len)
```

**WHY striped 切分而非 block 切分？**
Block 切分 [0..L/P) → rank0, [L/P..2L/P) → rank1, ...
- 因果 attention 时 rank0 无法看到 rank1+ 的 token → 信息不完整

Striped 切分 [0, 2P-1, 2P, 4P-1, ...] → 每个 rank 覆盖均匀分布的位置：
- 每个 rank 可以看到全序列的代表性 token
- Ring attention 通信后获得完整 KV

## 7. TransformerEngine 融合 RoPE Kernel

### 7.1 fused_rope (TE common/fused_rope/)

TE 提供 CUDA kernel 直接将 RoPE 与 QKV projection 融合：
- 避免单独的 RoPE 计算 pass
- 减少内存读写（无需中间 tensor 存储 cos/sin）
- 支持 FP8 精度下的 RoPE 应用

### 7.2 性能对比

```
标准 RoPE (PyTorch):
  t = t * cos + rotate_half(t) * sin     # 3 次内存读 + 2 次写
  
融合 RoPE (TE kernel):
  fused_rope_forward(t, cos, sin)         # 1 次内存读 + 1 次写
  
带宽节省: ~60% (对于大 head_dim 如 128)
```

## 8. FlagScale 特有扩展

### 8.1 平台抽象 (L29-33)

```python
# FlagScale Begin
from megatron.plugin.platform import get_platform
cur_platform = get_platform()
# FlagScale End

# 使用场景:
device = cur_platform.current_device()  # 替代 torch.cuda.current_device()
# 支持非 NVIDIA 硬件（如华为 NPU、天数智芯 GPU）
```

### 8.2 Sequence Parallel 修正 (L305)

```python
# 原始 Megatron 无此行:
if transformer_config.sequence_parallel:
    rotary_seq_len *= parallel_state.get_tensor_model_parallel_world_size()
# FlagScale Add: SP 切分后输入长度是 seq_len / tp_size
# 需要恢复到全局长度才能生成正确的频率矩阵
```

## 9. 设计决策对比

| 维度 | RoPE | 绝对位置编码 | ALiBi | 选择理由 |
|------|------|-------------|-------|----------|
| 参数量 | 0 | O(L·d) | 0 | RoPE 无需学习 |
| 外推性 | 好 (scaling) | 差 | 好 | 长序列关键 |
| 相对位置 | 自然支持 | 不支持 | 支持 | attention pattern |
| 推理缓存 | cos/sin 可预计算 | 查表 | 无需 | 效率高 |
| 多模态 | MRoPE 扩展 | 需重新设计 | 难扩展 | 灵活性 |

| 维度 | `lru_cache` 频率 | 每步重算 | 选择理由 |
|------|-----------------|----------|----------|
| 内存 | O(L·d) 缓存 | 0 | 可接受 |
| 计算 | 首次 O(L·d/2) | 每步 O(L·d/2) | 避免重复 |
| 缓存命中 | ~100% (固定 L) | N/A | 训练场景理想 |

## 10. 与其他章节的关联

- **→ 第4章 Context Parallel**: `get_pos_emb_on_this_cp_rank` 实现 CP 感知切分
- **→ 第2章 Tensor Parallel**: SP 模式下 rotary_seq_len 需乘以 tp_size
- **→ 第14章 TransformerLayer**: `_forward_attention` 传入 `rotary_pos_emb`
- **→ 第16章 MLA**: MLA 的 compressed KV 需要特殊 RoPE 处理
- **→ TE-FL 第4章**: 融合 RoPE kernel 实现

## 11. 源码版本信息

- `rotary_pos_embedding.py`: 451 行 (含 RotaryEmbedding + MultimodalRotaryEmbedding)
- `rope_utils.py`: ~200 行 (apply_rotary_pos_emb BSHD/THD)
- TE `rope.py`: 542 行 (融合 CUDA kernel Python 接口)
- FlagScale 扩展: 平台抽象, SP 修正, NPU 适配
