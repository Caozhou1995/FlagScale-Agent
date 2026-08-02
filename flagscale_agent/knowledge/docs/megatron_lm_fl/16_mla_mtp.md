# 第16章：MLA & MTP (Multi-Latent Attention / Multi-Token Prediction) 深度源码分析

## 1. 概述与设计动机

### 1.1 解决什么问题

MLA 和 MTP 是 DeepSeek-V3 的两大核心创新：
- **MLA (Multi-Latent Attention)**：通过低秩压缩 KV 缓存，将推理 KV 内存降至 MHA 的 1/10
- **MTP (Multi-Token Prediction)**：训练时预测多个未来 token，提升数据效率和表征质量

### 1.2 核心设计思想

**MLA 动机**：标准 MHA 的 KV cache 为 `2 * n_heads * head_dim * seq_len`。
GQA 通过减少 KV heads 来压缩，但牺牲了表达能力。
MLA 另辟蹊径：将 KV 投影到低维空间存储，推理时再上投影恢复。

**MTP 动机**：标准训练只预测下一个 token，信号稀疏。
MTP 让每层预测未来 D 个 token，相当于将训练信号密度提升 D 倍，
且各层共享嵌入/输出头，参数开销极小。

### 1.3 与其他技术的关系

| 技术 | KV Cache 大小 | 表达能力 | 推理速度 |
|------|-------------|----------|----------|
| MHA | 2 * n * d * s | 最强 | 慢 |
| GQA (g=8) | 2 * 8 * d * s | 中等 | 快 |
| MQA (g=1) | 2 * d * s | 弱 | 最快 |
| MLA | kv_lora_rank * s + rope_dim * s | 强（接近 MHA） | 快 |

## 2. 源码定位

| 组件 | 文件路径 | 行数 | 职责 |
|------|----------|------|------|
| MLA | `core/transformer/multi_latent_attention.py` | 1343 | MLA 全部实现 |
| MTP | `core/transformer/multi_token_prediction.py` | 1613 | MTP 全部实现 |
| MLA Config | `core/transformer/transformer_config.py` | - | MLATransformerConfig |

## 3. MLA 架构总览

### 3.1 类继承关系

```
Attention (基类, attention.py)
  └── MultiLatentAttention (MLA 基类, L98)
        ├── MLASelfAttention (标准 MLA, L387)
        │     └── FusedMLASelfAttention (融合优化版, L1130)
        └── [future: MLACrossAttention]
```

### 3.2 核心数据流

```
hidden_states [s, b, h]
    │
    ├─── Q path ────────────────────────────────────────────┐
    │    │                                                  │
    │    ├─ linear_q_down_proj: [s,b,h] → [s,b,q_lora]    │ (可选,DeepSeek用)
    │    ├─ q_layernorm                                     │
    │    └─ linear_q_up_proj: [s,b,q_lora] → [s,b,n*d_q]  │
    │         └─ reshape → [s,b,n,d_q]                     │
    │              ├─ q_no_pe: [s,b,n,qk_head_dim]         │
    │              └─ q_pos_emb: [s,b,n,qk_pos_dim]        │
    │                    └─ apply_RoPE                      │
    │                    └─ cat → query [s,b,n,d_q+pos_dim] │
    │                                                      │
    ├─── KV path ──────────────────────────────────────────┤
    │    │                                                  │
    │    ├─ linear_kv_down_proj: [s,b,h] → [s,b,kv_lora+pos_dim]
    │    │    └─ split → kv_compressed [s,b,kv_lora]        │
    │    │              + k_pos_emb [s,b,pos_dim]           │
    │    ├─ kv_layernorm(kv_compressed)                     │
    │    └─ linear_kv_up_proj: [s,b,kv_lora] → [s,b,n*(qk+v)]
    │         └─ reshape + split:                           │
    │              ├─ k_no_pe: [s,b,n,qk_head_dim]         │
    │              └─ value: [s,b,n,v_head_dim]            │
    │         k_pos_emb → apply_RoPE → expand to n heads   │
    │         key = cat(k_no_pe, k_pos_emb) [s,b,n,d_q+pos_dim]
    │                                                      │
    └─── Attention ─────────────────────────────────────────┘
         core_attention(query, key, value) → output [s,b,n,v]
         linear_proj: [s,b,n*v] → [s,b,h]
```

## 4. MLASelfAttention 详解

### 4.1 投影矩阵构建 (L420-524)

```python
# Q path: h → q_lora → n*(qk_head_dim + qk_pos_emb_head_dim)
# DeepSeek-V3 参数: h=7168, q_lora=1536, qk_head_dim=128, qk_pos_emb_head_dim=64, n=128
self.linear_q_down_proj: [7168] → [1536]      # 压缩
self.linear_q_up_proj:   [1536] → [128*(128+64)] = [128*192] = [24576]  # 恢复

# KV path: h → kv_lora + pos_dim → n*(qk_head_dim + v_head_dim)
# DeepSeek-V3: kv_lora=512, v_head_dim=128
self.linear_kv_down_proj: [7168] → [512+64] = [576]    # KV 压缩
self.linear_kv_up_proj:   [512] → [128*(128+128)] = [128*256] = [32768]  # KV 恢复
```

**KV Cache 压缩比** (DeepSeek-V3):
- MHA: 2 * 128 * 128 = 32768 per token
- MLA: 512 + 64 = 576 per token
- 压缩比: 32768 / 576 ≈ **57x**

### 4.2 TP 并行策略 (L435-524)

```python
# Q down proj: duplicated (非 TP 切分)，因为 q_lora_rank 远小于 h
linear_q_down_proj: parallel_mode='duplicated' 或 gather_output=False

# Q up proj: ColumnParallelLinear，输出按 head 切分
linear_q_up_proj: gather_output=False, tp_group=pg_collection.tp

# KV down proj: duplicated（kv_lora_rank 很小，不值得切分）
linear_kv_down_proj: parallel_mode='duplicated'

# KV up proj: ColumnParallelLinear，输出按 head 切分
linear_kv_up_proj: gather_output=False, tp_group=pg_collection.tp
```

**WHY duplicated for down_proj？**
- kv_lora_rank=512 << hidden_size=7168
- TP 切分后每个 rank 只有 512/8=64 维，GEMM 效率极低
- 宁可复制计算（小矩阵），也不要低效 TP

### 4.3 get_query_key_value_tensors 核心流程 (L567-891)

```python
def get_query_key_value_tensors(self, hidden_states, ...):
    # Phase 1: Down projection + Norm (L636-672)
    q_compressed, kv_combined = self._qkv_down_projection(hidden_states)
    # 分离 kv_compressed 和 k_pos_emb
    kv_compressed, k_pos_emb = split(kv_combined, [kv_lora, pos_dim])
    q_compressed = q_layernorm(q_compressed)
    kv_compressed = kv_layernorm(kv_compressed)
    
    # Phase 2: Up projection + RoPE (L746-891)
    q = linear_q_up_proj(q_compressed)        # [s,b,n*d_q]
    q = reshape(q, [s,b,n,d_q])
    kv = linear_kv_up_proj(kv_compressed)     # [s,b,n*(qk+v)]
    kv = reshape(kv, [s,b,n,qk+v])
    
    # 分离位置相关和位置无关部分
    q_no_pe, q_pos_emb = split(q, [qk_head_dim, pos_dim])
    k_no_pe, value = split(kv, [qk_head_dim, v_head_dim])
    
    # 只对位置相关部分应用 RoPE
    q_pos_emb = apply_rotary_pos_emb(q_pos_emb, freqs)
    k_pos_emb = apply_rotary_pos_emb(k_pos_emb, freqs)
    
    # 拼接为最终 query/key
    query = cat([q_no_pe, q_pos_emb])     # [s,b,n, qk+pos]
    key = cat([k_no_pe, k_pos_emb])       # [s,b,n, qk+pos]
    
    return query, key, value
```

### 4.4 Selective Recomputation (L872-877)

```python
if self.recompute_up_proj:
    # 训练时释放 up_proj 中间结果，backward 重算
    # 节省显存: n*(qk+v)*s*b 的激活
    self.qkv_up_checkpoint = CheckpointWithoutOutput(fp8=quantization)
    query, key, value = self.qkv_up_checkpoint.checkpoint(
        qkv_up_proj_and_rope_apply, q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb)
```

**WHY recompute up_proj？**
up_proj 输出 shape 远大于 down_proj 输出（24576 vs 1536），
重算代价（一次小 GEMM）远小于存储代价。

## 5. FusedMLASelfAttention 优化 (L1130-1343)

### 5.1 设计动机

标准 MLASelfAttention 中 Q down_proj 和 KV down_proj 是两个独立 GEMM。
FusedMLASelfAttention 将两者融合为一个 GEMM：

```python
class FusedMLASelfAttention(MLASelfAttention):
    """融合 Q/KV down projection 为单个 GEMM"""
    
    def __init__(self, ...):
        # 替代独立的 q_down_proj 和 kv_down_proj
        # 融合输出维度: q_lora_rank + kv_lora_rank + qk_pos_emb_head_dim
        self.linear_qkv_down_proj = build_module(
            submodules.linear_qkv_down_proj,
            config.hidden_size,
            config.q_lora_rank + config.kv_lora_rank + config.qk_pos_emb_head_dim,
            ...)
```

### 5.2 性能收益

| 方案 | GEMM 次数 (down) | 矩阵大小 |
|------|------------------|-----------|
| 非融合 | 2 | [h,q_lora] + [h,kv_lora+pos] |
| 融合 | 1 | [h, q_lora+kv_lora+pos] = [7168, 2112] |

单个大 GEMM 比两个小 GEMM 效率更高（GPU SM 利用率更好）。

## 6. MTP (Multi-Token Prediction) 详解

### 6.1 架构设计 (multi_token_prediction.py:L741-860)

```python
class MultiTokenPredictionLayer(MegatronModule):
    """第 k 层 MTP 模块结构:
    
    input: hidden_states[i] (第 i 个 token 的表征, 来自前一层 MTP 或主干)
           + embedding[i+k] (第 i+k 个 token 的词嵌入)
    
    流程:
    1. enorm(embedding[i+k])       # 嵌入归一化
    2. hnorm(hidden_states[i])     # 隐藏态归一化
    3. eh_proj(cat(enorm, hnorm))  # 合并投影 → [h]
    4. transformer_layer(eh_proj)  # 一层 Transformer
    5. output_head(result)         # 共享词表投影 → logits
    """
```

### 6.2 核心组件 (L811-870)

```python
# 归一化层
self.enorm = RMSNorm(hidden_size, eps=layernorm_epsilon)  # (L811)
self.hnorm = RMSNorm(hidden_size, eps=layernorm_epsilon)  # (L817)

# 投影（标准模式）: cat([h, h]) → [h]
self.eh_proj = ColumnParallelLinear(
    2 * hidden_size, hidden_size, ...)                     # (L862)

# mHC 模式（Hyper-Connections）: 独立 e_proj 和 h_proj
self.e_proj = ColumnParallelLinear(hidden_size, hidden_size)  # (L827)
self.h_proj = ColumnParallelLinear(hidden_size, hidden_size)  # (L840)
```

### 6.3 MTP Loss 计算 (L641-738)

```python
def process_mtp_loss(hidden_states, mtp_labels, loss_mask, ...):
    """MTP loss 计算流程 (L641-738)"""
    
    # 1. 生成 logits
    mtp_logits = output_head(hidden_states)  # 共享主干输出头
    
    # 2. 标签 roll（将预测目标右移 1 位）
    mtp_labels = roll_tensor(mtp_labels, shifts=-1, cp_group=cp_group)
    loss_mask = roll_tensor(loss_mask, shifts=-1, cp_group=cp_group)
    
    # 3. 计算交叉熵
    mtp_loss = compute_language_model_loss(mtp_labels, mtp_logits)
    mtp_loss = loss_mask * mtp_loss
    
    # 4. 缩放并通过 autograd 附加到主 loss
    mtp_loss_scale = mtp_loss_scaling_factor / mtp_num_layers  # (L719)
    hidden_states = MTPLossAutoScaler.apply(hidden_states, scaled_loss)
    
    return hidden_states  # 梯度通过 hidden_states 传播
```

### 6.4 MTPLossAutoScaler (梯度注入)

```python
class MTPLossAutoScaler(torch.autograd.Function):
    """将 MTP loss 梯度注入到 hidden_states 的梯度流中
    
    forward: 直接返回 hidden_states（不修改）
    backward: grad_hidden += grad(mtp_loss)
    
    WHY 这样设计？
    - MTP loss 独立于主 loss，但需要通过主干反传梯度
    - 不能用 loss.backward() 因为会破坏 PP 调度
    - 用 autograd.Function 将额外梯度"附着"到 hidden_states
    """
```

### 6.5 roll_tensor 与 CP 协调 (L701-706)

```python
# MTP 需要将标签右移 1 位
# 在 Context Parallel 下，序列分布在多个 rank
# roll_tensor 跨 CP 组进行 P2P 通信完成 shift
mtp_labels = roll_tensor(mtp_labels, shifts=-1, dims=-1, 
                         cp_group=cp_group, packed_seq_params=packed_seq_params)
```

## 7. MTP 与 PP/VP 的交互

### 7.1 MTP 层放置策略

```
MTP 层只放在最后一个 PP stage（因为需要 output head）:
PP stage 0: Layers 0-15
PP stage 1: Layers 16-31 + MTP_1 + MTP_2 + ... + MTP_D

Virtual Pipeline 模式:
VP stage 0: Layers 0-7  | VP stage 2: Layers 16-23
VP stage 1: Layers 8-15 | VP stage 3: Layers 24-31 + MTP layers
```

### 7.2 get_mtp_layer_offset (L776)

```python
self.layer_number = layer_number + get_mtp_layer_offset(config, vp_stage)
# 确保 MTP 层的 layer_number 与主干层不冲突
# 用于正确的 activation checkpointing 和 FP8 state 管理
```

## 8. 性能量化分析

### 8.1 MLA 计算量 (DeepSeek-V3 配置)

```
h=7168, n=128, qk=128, v=128, pos=64, q_lora=1536, kv_lora=512

Q path FLOPs per token:
  q_down: 2 * 7168 * 1536 = 22.0M
  q_up:   2 * 1536 * 128*192 = 75.5M
  Total Q: 97.5M

KV path FLOPs per token:
  kv_down: 2 * 7168 * 576 = 8.3M
  kv_up:   2 * 512 * 128*256 = 33.6M
  Total KV: 41.9M

对比标准 MHA:
  QKV proj: 2 * 7168 * 3*128*128 = 703M
  MLA total: 97.5 + 41.9 = 139.4M (节省 80% 投影计算！)
```

### 8.2 MTP 额外开销

```
每层 MTP 额外计算:
  eh_proj: 2 * 2*7168 * 7168 = 205M FLOPs
  1 Transformer layer: ~2 * 12 * h^2 = 1234M FLOPs
  output head: 2 * 7168 * vocab = ~1880M FLOPs (vocab=131072)
  
D=2 层 MTP 相对主干 61 层的开销:
  2 * (205 + 1234 + 1880) / (61 * 12 * h^2) ≈ 0.9%
  （参数共享使得 MTP 开销极小）
```

## 9. 设计决策对比表

| 维度 | MLA | GQA-8 | MQA | 选择理由 |
|------|-----|-------|-----|----------|
| KV Cache / token | 576 | 2048 | 256 | MLA 在 cache 和表达力间平衡 |
| 投影参数量 | 较多 (up/down) | 少 | 最少 | MLA 用参数换 cache |
| 推理延迟 | 中 (需 up proj) | 低 | 最低 | 可用 absorption 优化 |
| 训练收敛 | 快 | 中 | 慢 | MLA 表达力接近 MHA |
| TP 切分 | down 复制, up 切分 | 按 group 切 | 不切 | MLA 需要注意小矩阵 |

| 维度 | MTP | 标准 NTP | 选择理由 |
|------|-----|---------|----------|
| 训练信号密度 | D*token | 1*token | MTP 更高效利用数据 |
| 参数开销 | ~1% (共享头) | 0 | 极小额外开销 |
| 推理时 | 可移除 | - | 不影响推理 |
| 与 CP 兼容 | 需要 roll_tensor | 天然 | 额外通信 |

## 10. 边界条件与约束

### 10.1 MLA 约束

- Hybrid CP 不支持 MLA (`packed_seq_params.local_cp_size` 必须为 None, L586-589)
- 推理 + RoPE fusion 互斥 (L613)
- `cache_mla_latents` 仅支持动态批处理推理 (L881)
- EP 和 CP 互斥（RankGenerator 层面约束）

### 10.2 MTP 约束

- MTP 层必须在最后一个 PP stage
- 注意力 mask 类型限制 (SUPPORTED_ATTN_MASK, L803)
- `calculate_per_token_loss` 需要特殊归一化处理 (L720-731)
- MTP loss 通过 `MTPLossAutoScaler` 注入，不能直接加到主 loss

### 10.3 配置参数

```python
# MLATransformerConfig 关键参数:
q_lora_rank: int = 1536        # Q 低秩维度
kv_lora_rank: int = 512        # KV 低秩维度
qk_head_dim: int = 128         # 非位置相关的 head 维度
qk_pos_emb_head_dim: int = 64  # RoPE 作用的 head 维度
v_head_dim: int = 128          # Value head 维度
num_attention_heads: int = 128
rope_type: str = "yarn"        # yarn or rope
rotary_scaling_factor: float   # Yarn 缩放因子
cache_mla_latents: bool = True # 推理时缓存低维表示

# MTP 配置:
mtp_num_layers: int = 2          # MTP 深度
mtp_loss_scaling_factor: float   # MTP loss 权重
enable_hyper_connections: bool   # mHC 模式
recompute_modules: List[str]     # 包含 "mla_up_proj" 则重算 up_proj
```

## 11. 配置建议

### 11.1 MLA 调优

- `recompute_modules=["mla_up_proj"]`：强烈推荐，节省大量激活内存
- kv_down_proj 使用 `duplicated` 模式（小矩阵不值得 TP）
- FP8 训练时 up_proj 精度敏感，建议保持 BF16 或用 current scaling

### 11.2 MTP 调优

- `mtp_num_layers=2` 是论文推荐值（收益递减）
- `mtp_loss_scaling_factor=0.3` 是 DeepSeek-V3 默认值
- 推理时可移除 MTP 层（不影响生成质量）

## 12. 推理优化：KV Absorption

### 12.1 Absorption 原理 (L722-736)

在纯 decode 阶段，MLA 可以将 up_proj 权重"吸收"到 attention 计算中：

```python
# 标准路径 (训练/prefill):
k_no_pe = kv_up_proj(kv_compressed)[:, :, :qk_head_dim]  # [s,b,n,qk]
score = einsum("sbnd,sbnd->sbn", q_no_pe, k_no_pe)

# Absorption 路径 (decode):
# 预计算 W_absorbed = W_q_no_pe @ W_kv_up[:qk_head_dim]
# score = q_no_pe @ W_absorbed @ kv_compressed
# 等价于: q_content = einsum("sbhd,hdk->sbhk", q_no_pe, up_k_weight)  (L731)
# 然后 score = einsum("sbhk,sbk->sbh", q_content, kv_compressed)
```

### 12.2 为什么只在 decode 用 absorption？

| 阶段 | batch*seq | kv_compressed 大小 | 收益 |
|------|-----------|-------------------|------|
| Prefill | 1 * 4096 | [4096, 512] | 负收益（up_proj 一次完成） |
| Decode | 128 * 1 | [1, 512] per token | 大收益（避免对每个新 token 做 up_proj） |

Decode 阶段每步只产生 1 个新 token，对其做 up_proj 浪费；
直接用 absorption 避免显式恢复 full KV。

## 13. FlagScale 定制扩展

### 13.1 Platform 适配

MLA 中通过 `get_platform()` 抽象化设备操作：
```python
# rotary_pos_embedding.py:L83
device = 'cpu' if use_cpu_initialization else cur_platform.current_device()
```

### 13.2 Hetero Parallel 支持

`pg_collection` 参数允许传入自定义进程组集合：
```python
# 异构并行场景下，不同层可以有不同的 TP/CP 组
MLASelfAttention(config, submodules, pg_collection=custom_pg)
```

这使得 MLA 层可以在异构 GPU 集群上运行不同的并行策略。
