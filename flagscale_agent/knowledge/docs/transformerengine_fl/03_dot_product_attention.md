# 第三章：DotProductAttention & Flash Attention 系统

## 1. 概述与设计动机

TransformerEngine-FL 的注意力子系统将三种后端（FlashAttention、cuDNN FusedAttention、Unfused）统一在同一个 `DotProductAttention` 类中，根据硬件能力和输入特征自动选择最优后端。同时深度集成 Context Parallelism (CP)，支持 P2P Ring、AllGather 和 A2A 三种通信模式。

**核心设计目标**:
- 用户无感知的后端自动选择
- 长序列通过 CP 实现线性 memory scaling
- FP8 attention (Hopper cuDNN) 进一步提升吞吐

---

## 2. 源文件定位

| 文件路径 | 行数 | 核心职责 |
|---------|------|---------|
| `attention/dot_product_attention/dot_product_attention.py` | 1628 | DotProductAttention 统一入口类 |
| `attention/dot_product_attention/backends.py` | 2029 | FlashAttention / FusedAttention / Unfused 后端实现 |
| `attention/dot_product_attention/context_parallel.py` | 4365 | CP 通信 + Ring Attention 全部实现 |
| `attention/dot_product_attention/utils.py` | 2304 | get_attention_backend / AttentionParams / layout工具 |
| `attention/dot_product_attention/softmax.py` | 288 | Softmax 变体 (vanilla/off-by-one/learnable) |
| `attention/rope.py` | ~400 | Rotary Position Embedding |

---

## 3. DotProductAttention 类架构

### 3.1 类定义与初始化 (L167-495)

```python
class DotProductAttention(TransformerEngineBaseModule):  # L167
    def __init__(self,
        num_attention_heads: int,
        kv_channels: Union[int, Tuple[int, int]],  # 支持 Q/K 和 V 不同 head_dim
        num_gqa_groups: Optional[int] = None,       # GQA 支持
        qkv_format: str = "sbhd",                   # sbhd/bshd/thd
        attn_mask_type: str = "causal",
        window_size: Optional[Tuple[int, int]] = None,  # 滑动窗口
        cp_group: Optional[Union[dist_group_type, List[dist_group_type]]] = None,
        cp_comm_type: str = "p2p",                  # p2p/all_gather/a2a
        softmax_scale: Optional[float] = None,
        softmax_type: str = "vanilla",              # vanilla/off-by-one/learnable
        ...
    ):
```

**关键设计**: `__init__` 中同时实例化三种后端（L466-493）:
```python
self.flash_attention = FlashAttention(softmax_scale, ...)      # L466
self.fused_attention = FusedAttention(softmax_scale, ...)      # L476
self.unfused_attention = UnfusedDotProductAttention(...)        # L486
```

预创建所有后端实例，避免运行时动态创建的开销。

### 3.2 GQA 支持 (L395-400)

```python
self.num_gqa_groups = num_attention_heads if num_gqa_groups is None else num_gqa_groups
self.num_gqa_groups_per_partition = int(self.num_gqa_groups // self.tp_size)
assert num_attention_heads % self.num_gqa_groups == 0
```

GQA 的 head 数量必须被 TP size 整除，每个 TP rank 持有 `num_gqa_groups // tp_size` 个 KV head。

### 3.3 Softmax 变体 (L446-459)

| softmax_type | 实现 | 用途 |
|-------------|------|------|
| `"vanilla"` | 标准 softmax | 默认 |
| `"off-by-one"` | softmax + learnable offset | 防止注意力退化 |
| `"learnable"` | 可学习 offset 参数 (nn.Parameter) | 实验性 |

---

## 4. Forward 完整流程 (L821-1628)

### 4.1 输入验证阶段 (L1050-1180)

```python
# FP8 检查 (L1052-1071):
if self.fp8 and self.fp8_meta["recipe"].fp8_dpa:
    forward_dtype = get_fp8_te_dtype(...)  # E4M3 for forward
    backward_dtype = get_fp8_te_dtype(...)  # E5M2 for backward

# QKV shape 检查 (L1073-1098):
# - Q/K 必须同 head_dim, V 可以不同 (head_dim_qk vs head_dim_v)
# - KV head 数 = num_gqa_groups_per_partition

# qkv_format 处理 (L1129-1180):
# sbhd: [seq, batch, heads, dim]
# bshd: [batch, seq, heads, dim]  
# thd:  [total_tokens, heads, dim] — 需要 cu_seqlens
```

### 4.2 KV Cache 推理路径 (L1182-1220)

```python
if inference_params is not None:
    # 转换 top-left causal → bottom-right causal (KV cache 需要)
    attn_mask_type = "padding_causal_bottom_right"
    # 从 cache 获取完整 KV
    key_layer, value_layer, cu_seqlens_q, cu_seqlens_kv, max_seqlen_kv, qkv_format = \
        inference_params.step(self.layer_number, key_layer, value_layer, qkv_format)
```

### 4.3 Context Parallel 序列长度调整 (L1254-1295)

```python
cp_size = get_distributed_world_size(self.cp_group)  # L1257
if q_format in ["sbhd", "bshd"]:
    max_seqlen_q *= cp_size   # L1263: 恢复全局序列长度
    max_seqlen_kv *= cp_size  # L1280
```

**重要**: 每个 CP rank 只持有 `seq_len/cp_size` 的 local tokens，但 `max_seqlen` 需要报告全局长度给后端。

### 4.4 后端选择决策 (L1360-1443)

```python
# 收集所有注意力参数到 AttentionParams 结构体
attention_params = dpa_utils.AttentionParams(
    qkv_type, qkv_dtype, qkv_layout, batch_size,
    num_heads, num_gqa_groups, max_seqlen_q, max_seqlen_kv,
    head_dim_qk, head_dim_v, attn_mask_type, window_size,
    context_parallel, cp_comm_type, fp8, ...
)

# 调用 get_attention_backend() 决定使用哪个后端
(use_flash, flash_backend, use_fused, fused_backend, use_unfused, _) = \
    dpa_utils.get_attention_backend(attention_params)
```

### 4.5 后端选择优先级

```
优先级排序:
1. FusedAttention (cuDNN) — 当 FP8 启用 + SM90+ (Hopper)
2. FlashAttention — 当 flash-attn 包可用 + 支持的 head_dim
3. UnfusedDotProductAttention — fallback

选择缓存机制 (L1403-1425):
- _attention_backends 全局字典缓存上次选择结果
- 仅当 attention_params 变化时重新计算 (避免每次 forward 重复选择)
```

### 4.6 后端 Dispatch (L1450-1628)

```python
if use_flash_attention:
    # CP 路径: AttnFuncWithCPAndKVP2P / AttnFuncWithCPAndKVAllGather
    if context_parallel:
        return attn_forward_func_with_cp(...)  # context_parallel.py L3934
    # 标准路径
    return self.flash_attention(query_layer, key_layer, value_layer, ...)

elif use_fused_attention:
    return self.fused_attention(query_layer, key_layer, value_layer, ...)

elif use_unfused_attention:
    return self.unfused_attention(query_layer, key_layer, value_layer, ...)
```

---

## 5. FlashAttention 后端详解 (backends.py L684-1148)

### 5.1 类结构

```python
class FlashAttention(torch.nn.Module):  # L684
    def __init__(self, softmax_scale, attention_dropout, ...):
        # 版本检查: fa_utils.version >= version_required
        # 支持 flash-attn v2.3+ 到 v2.7+
        ...
    
    def forward(self, query_layer, key_layer, value_layer, ...):  # L721
        # 支持: FP16/BF16/Float8Tensor 输入
        # 支持: sbhd/bshd/thd 格式
        # 支持: CP 集成
```

### 5.2 QKV 格式转换 (L773-810)

```python
# sbhd → bshd 转换 (FlashAttention 要求 bshd)
if qkv_format == "sbhd":
    if head_dim == 128 and seq*batch >= 512 and layout == "sbh3d":
        # 优化路径: 使用自定义 autograd function 避免额外内存
        query_layer, key_layer, value_layer = _PrepareQKVForFA.apply(...)
    else:
        # 标准路径: transpose(0,1) + contiguous
        query_layer = query_layer.transpose(0, 1).contiguous()

# FP8 tensor 特殊处理: 只转换内部 _data，保持 Float8Tensor 包装
if isinstance(query_layer, Float8Tensor):
    query_layer._data = query_layer._data.transpose(0, 1).contiguous()
    query_layer = Float8Tensor.make_like(query_layer, data=query_layer._data, ...)
```

### 5.3 核心调用链

```
FlashAttention.forward()
  ├── CP 路径: attn_forward_func_with_cp()
  │     ├── p2p → AttnFuncWithCPAndKVP2P.forward()
  │     ├── all_gather → AttnFuncWithCPAndKVAllGather.forward()
  │     └── a2a → AttnFuncWithCPAndQKVOA2A.forward()
  └── 标准路径: 
        └── flash_attn_varlen_func() / flash_attn_func()
              └── tex.fused_attn_fwd() [C++ extension]
```

### 5.4 FP8 Attention (cuDNN FusedAttention, L1149-2029)

```python
class FusedAttnFunc(torch.autograd.Function):  # L1149
    @staticmethod
    def forward(ctx, ...):  # L1153
        # cuDNN graph-based attention
        # 支持 FP8 Q/K/V: E4M3 forward, E5M2 backward
        # 内部调用 tex.fused_attn_fwd_qkvpacked / tex.fused_attn_fwd_kvpacked
        
    @staticmethod  
    def backward(ctx, d_out):  # L1458
        # 保存 softmax_lse 用于反向
        # 调用 tex.fused_attn_bwd_*
```

**FP8 Attention 限制**:
- 仅 SM90+ (Hopper)
- cuDNN ≥ 9.0
- softmax 内部仍为 FP32（精度敏感）
- head_dim 必须为 64/128

---

## 6. Context Parallel 注意力实现 (context_parallel.py, 4365行)

### 6.1 三种 CP 通信模式

| 模式 | 类 | 通信原语 | 内存 | 适用 |
|------|-----|---------|------|------|
| P2P Ring | `AttnFuncWithCPAndKVP2P` (L1249) | isend/irecv | O(S/CP) | 大 CP, 长序列 |
| KV AllGather | `AttnFuncWithCPAndKVAllGather` (L2797) | AllGather KV | O(S) | 小 CP, 带宽充足 |
| QKV-O A2A | `AttnFuncWithCPAndQKVOA2A` (L3307) | All-to-All | O(S/CP) | 平衡计算+通信 |

### 6.2 P2P Ring Attention 核心循环 (L1262-2054)

```python
class AttnFuncWithCPAndKVP2P(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ...):  # L1262
        # 核心思想: Ring 传递 KV，每步与本地 attention 重叠
        
        for i in range(cp_size):
            # Step 1: 异步 P2P 通信 (发送当前KV, 接收下一个KV)
            if i < cp_size - 1:
                send_recv_reqs = flash_attn_p2p_communicate(
                    rank, kv_send, send_dst, kv_recv, recv_src, cp_group
                )
            
            # Step 2: 本地 Flash Attention 计算
            out_per_step, lse_per_step, _ = cp_p2p_fwd_flash_attn(
                ..., q_part, k_part, v_part, ..., section=section
            )
            
            # Step 3: Online Softmax 合并
            flash_attn_fwd_out_correction(output, out_per_step, lse, lse_per_step)
            flash_attn_fwd_softmax_lse_correction(lse, lse_per_step)
            
            # Step 4: 等待通信完成, 交换 buffer
            for req in send_recv_reqs:
                req.wait()
            kv_send, kv_recv = kv_recv, kv_send  # 双 buffer 交替
```

### 6.3 P2P 通信的奇偶交错 (L62-101)

```python
def flash_attn_p2p_communicate(rank, send_tensor, send_dst, recv_tensor, recv_src, ...):
    # 关键: 奇偶 rank 交替发送/接收顺序，避免死锁
    if rank % 2 == 0:
        send_op → recv_op  # 偶数 rank 先发后收
    else:
        recv_op → send_op  # 奇数 rank 先收后发
    
    # 支持 batch P2P (单次 NCCL call) 或独立 isend/irecv
    if batch_p2p_comm:
        return torch.distributed.batch_isend_irecv(ops)
```

### 6.4 Online Softmax 合并 (L105-175)

```python
@jit_fuser
def flash_attn_fwd_out_correction(out, out_per_step, lse, lse_per_step):
    """合并两个 partial attention 的输出"""
    # 核心公式:
    # new_lse = log(exp(lse) + exp(lse_per_step))
    # correction = exp(lse_per_step - new_lse)
    # out = out * exp(lse - new_lse) + out_per_step * correction
    
    # 等价于: softmax(concat(scores_1, scores_2)) 的分布式计算

@jit_fuser  
def flash_attn_fwd_softmax_lse_correction(lse, lse_per_step):
    """更新全局 log-sum-exp"""
    # lse = log(exp(lse) + exp(lse_per_step))
    # 数值稳定版: lse = max(lse, lse_per_step) + log(1 + exp(-|lse - lse_per_step|))
```

### 6.5 Causal Mask 在 CP 中的 Section 处理 (L913-975)

```python
def cp_p2p_fwd_flash_attn(..., section):
    """每个 ring step 对应不同的 causal section"""
    # section 类型:
    # "diagonal" — causal boundary 所在的块 (需要 causal=True)
    # "lower-triangle" — 全部有效 (causal boundary 以下)
    # "upper-triangle" — 全部 mask (causal boundary 以上, 可跳过)
    # "all" — 非 causal mask, 全部计算
    
    if section == "diagonal":
        causal_ = True       # 只有对角块需要 causal mask
    elif section == "lower-triangle":
        max_seqlen_kv_ = max_seqlen_kv // 2  # 只用 KV 的前半部分
    elif section == "upper-triangle":
        max_seqlen_q_ = max_seqlen_q // 2    # 只用 Q 的前半部分
```

**时序图 (CP=4, Causal)**:
```
Ring Step:     0          1          2          3
GPU 0:    [Q0,K0] D   [Q0,K1] L   [Q0,K2] L   [Q0,K3] L
GPU 1:    [Q1,K1] D   [Q1,K2] L   [Q1,K3] L   [Q1,K0] U(skip)
GPU 2:    [Q2,K2] D   [Q2,K3] L   [Q2,K0] U   [Q2,K1] U(skip)  
GPU 3:    [Q3,K3] D   [Q3,K0] U   [Q3,K1] U   [Q3,K2] U(skip)

D=diagonal(causal), L=lower-triangle(full), U=upper-triangle(可跳过)
```

### 6.6 A2A 通信模式 (L3307-3933)

```python
class AttnFuncWithCPAndQKVOA2A(torch.autograd.Function):
    # 与 P2P Ring 不同: 一次性 All-to-All 交换所有 QKV
    # 每个 rank 获得所有序列位置的一个 head subset
    # 优势: 一次通信 vs ring 的 CP-1 次通信
    # 劣势: 需要 head_num 能被 CP 整除
```

---

## 7. 后端选择决策引擎 (utils.py)

### 7.1 get_attention_backend 逻辑

```python
def get_attention_backend(params: AttentionParams):
    """返回 (use_flash, flash_ver, use_fused, fused_backend, use_unfused, reason)"""
    
    # 决策树 (简化):
    # 1. FP8 + SM90 → FusedAttention (cuDNN)
    # 2. FlashAttention 可用 + 支持的配置 → FlashAttention
    #    - head_dim ∈ {64, 96, 128, 192, 256}
    #    - dtype ∈ {fp16, bf16}
    #    - 非 arbitrary mask (或 flash v2.4+)
    # 3. FusedAttention (非FP8) → cuDNN BF16/FP16 path
    # 4. Unfused → 最终 fallback
```

### 7.2 AttentionParams 结构体 (L1360-1394)

包含 23 个字段，完整描述一次注意力操作的所有特征：
- 数据: qkv_type, qkv_dtype, qkv_layout, batch_size, num_heads, head_dim
- Mask: attn_mask_type, window_size, bottom_right_diagonal
- 并行: context_parallel, cp_comm_type
- 精度: fp8, fp8_meta
- 模式: is_training, deterministic, cuda_graph

---

## 8. 性能分析

### 8.1 后端性能对比 (seq=4096, heads=32, dim=128)

| 后端 | 内存复杂度 | 计算复杂度 | H100 实测 (ms) |
|------|-----------|-----------|---------------|
| Unfused | O(N²) | O(N²d) | ~12.5 |
| FlashAttention v2 | O(N) | O(N²d) | ~2.8 |
| cuDNN FP8 | O(N) | O(N²d) | ~1.9 |

### 8.2 CP 通信量分析

设 `s`=序列长度, `h`=heads, `d`=head_dim, `b`=batch, `c`=cp_size:

| CP 模式 | 每步通信量 | 总通信量 | 通信次数 |
|---------|-----------|---------|---------|
| P2P Ring | `2 × b × (s/c) × h_kv × d × 2B` | `× (c-1)` | c-1 |
| AllGather | `b × s × h_kv × d × 2B` | 一次性 | 1 |
| A2A | `b × s × (h_q/c) × d × 2B` | 一次性 | 1 |

### 8.3 CP 模式选择指南

| 条件 | 推荐模式 | 原因 |
|------|---------|------|
| CP ≤ 4, NVLink | AllGather | 简单，一次通信 |
| CP > 4, 长序列 | P2P Ring | 内存 O(S/CP)，通信可重叠 |
| head_num 大且可整除 CP | A2A | 最平衡，但需要 head_num % CP == 0 |
| 跨节点 CP | P2P Ring | 网络延迟大，pipeline 更好 |

---

## 9. FP8 Attention 详解

### 9.1 数据流

```
Q (BF16) → FP8 E4M3 量化 → cuDNN fused_attn_fwd
K (BF16) → FP8 E4M3 量化 →     ↓
V (BF16) → FP8 E4M3 量化 →     ↓
                                 ↓
         softmax (FP32 内部) → attention_probs
                                 ↓
         Output (BF16/FP8) ←── probs × V
```

### 9.2 精度分析

```
FP8 attention 精度风险点:
1. Q×K^T: 动态范围大，FP8 E4M3 只有 ±448 → 需要 softmax_scale 控制
2. Softmax: 内部 FP32 (不做 FP8)，保证数值稳定
3. Attn × V: 精度相对安全 (attention weights ∈ [0,1])

实际影响:
- head_dim=64: 几乎无损
- head_dim=128: 微小精度差异 (|Δloss| < 0.1%)
- head_dim=256: 需要谨慎验证
```

---

## 10. 设计决策总结

| 设计选择 | 方案 | 替代方案 | 权衡理由 |
|---------|------|---------|---------|
| 三后端并存 | 全部预创建 | 按需创建 | 避免运行时判断开销，代价仅为少量内存 |
| 后端选择缓存 | 全局 _attention_backends | 每次重新计算 | params 不变时跳过决策，减少 overhead |
| sbhd→bshd 转换 | transpose+contiguous | 修改 FlashAttention | FA 原生要求 bshd，改动成本低 |
| P2P Ring 奇偶交替 | even先发/odd先收 | 统一顺序 | 避免环形死锁 |
| Online Softmax | @jit_fuser | 手写 CUDA | Triton JIT 足够快，维护成本低 |
| Causal + CP section | diagonal/lower/upper | 全 causal 每步 | 跳过 upper-triangle 节省 ~50% 计算 |
| CP 模式多选 | p2p/allgather/a2a | 只支持一种 | 不同场景最优不同 |

---

## 11. 调优指南

### 11.1 环境变量

```bash
NVTE_ALLOW_NONDETERMINISTIC_ALGO=1    # 允许非确定性 (性能优先)
NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT=1 # cuDNN workspace 优化
NVTE_FLASH_ATTN=1                      # 强制使用 FlashAttention
NVTE_FUSED_ATTN=1                      # 强制使用 FusedAttention
```

### 11.2 配置建议

| 场景 | 推荐配置 |
|------|---------|
| seq ≤ 4K | FlashAttention, 无需 CP |
| seq 4K-32K | FlashAttention + CP=2-4 (allgather) |
| seq 32K-128K | FlashAttention + CP=8 (p2p_ring) |
| seq 128K+ | FlashAttention + CP=16+ (p2p_ring) + 滑动窗口 |
| Hopper + FP8 | cuDNN FusedAttention (自动选择) |
| 调试/对齐 | Unfused (设置 NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=0) |
