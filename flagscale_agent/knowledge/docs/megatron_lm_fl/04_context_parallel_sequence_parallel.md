# 04 - Context Parallelism (CP) & Sequence Parallelism (SP) 源码深度分析

## 源码位置

| 文件 | 行数 | 核心功能 |
|------|------|----------|
| `megatron/core/tensor_parallel/mappings.py` | 602 | SP 通信原语: `_ScatterToSequenceParallelRegion`(L282), `_GatherFromSequenceParallelRegion`(L302), `_ReduceScatterToSequenceParallelRegion`(L357) |
| `megatron/core/tensor_parallel/layers.py` | 1369 | ColumnParallelLinear/RowParallelLinear + SP 集成 (gather_input/reduce_scatter_output) |
| `megatron/core/extensions/transformer_engine.py` | ~1570 | `TEDotProductAttention` — CP 集成入口 (L1447-1471), cp_stream 管理 (L1368,1452) |
| `megatron/core/pipeline_parallel/hybrid_cp_schedule.py` | 665 | `BalancedCPScheduler` 变长序列调度 + `hybrid_context_parallel_forward_backward` 执行入口 |
| `megatron/core/parallel_state.py` | ~2800 | CP group 创建 (L583), `create_hierarchical_groups`(L389), hierarchical CP (L1046-1052) |
| `megatron/core/transformer/transformer_config.py` | ~2567 | CP 配置字段: `cp_comm_type`(L974), per-layer list 校验 (L2506-2514), KV head 约束 (L1283-1299) |

---

## 1. Sequence Parallelism (SP) — 激活内存的序列维切分

### 1.1 设计动机与核心思想

**问题**: TP 将参数沿 hidden 维切分, 但 LayerNorm、Dropout 等 element-wise 操作在 TP rank 间是 **冗余计算** — 每个 rank 持有完整 `[S, B, H]` 激活。

**解决**: SP 将这些操作沿 sequence 维切分, 每个 rank 只处理 `[S/TP, B, H]`:

```
┌─────────────────────────────────────────────────────────────────────┐
│              Without SP (TP only)                                     │
│  LayerNorm: [S, B, H] on EACH rank (完整复制 → 浪费 (TP-1)/TP 内存) │
│  Linear:    all-reduce gradient → 计算                               │
├─────────────────────────────────────────────────────────────────────┤
│              With SP (TP + SP)                                        │
│  LayerNorm: [S/TP, B, H] per rank (切分 → 内存 ÷ TP)                │
│  Linear:    all-gather input / reduce-scatter output (替代 all-reduce)│
└─────────────────────────────────────────────────────────────────────┘
```

**关键约束**: SP 必须与 TP 配合使用 (`sequence_parallel=True` 要求 `TP > 1`)。

### 1.2 通信原语实现 (mappings.py)

SP 的核心是将 all-reduce 拆分为 reduce-scatter + all-gather, 各自可与计算 overlap:

```python
# mappings.py L282-300: _ScatterToSequenceParallelRegion
class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, group):
        # Forward: scatter along seq dim → [S/TP, B, H]
        # 将完整序列切分给各 TP rank
        world_size = torch.distributed.get_world_size(group)
        dim_size = input_.size()[0] // world_size
        return input_.narrow(0, rank * dim_size, dim_size).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        # Backward: all-gather gradients → [S, B, H]
        return _gather_along_first_dim(grad_output, group)

# mappings.py L302-355: _GatherFromSequenceParallelRegion
class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, group):
        # Forward: all-gather [S/TP,B,H] → [S,B,H] (为 Linear 准备完整输入)
        return _gather_along_first_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward: reduce-scatter grad → [S/TP,B,H]
        return _reduce_scatter_along_first_dim(grad_output, group)

# mappings.py L357-404: _ReduceScatterToSequenceParallelRegion
class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, group):
        # Forward: reduce-scatter [S,B,H] → [S/TP,B,H] (替代 all-reduce!)
        return _reduce_scatter_along_first_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward: all-gather grad → [S,B,H]
        return _gather_along_first_dim(grad_output, group)
```

**核心洞察**: all-reduce = reduce-scatter + all-gather, 通信总量不变, 但拆分后:
1. 中间状态从 `[S,B,H]` 变为 `[S/TP,B,H]` → 激活内存 ÷ TP
2. 可分别与前后计算 overlap

### 1.3 SP 在 Linear 层的集成 (layers.py)


```python
# === ColumnParallelLinear.forward (layers.py ~L965-1010) ===
def forward(self, input_):
    if self.config.sequence_parallel:
        # SP 模式: input 是 [S/TP, B, H], 需要 all-gather 恢复 [S, B, H]
        input_parallel = gather_from_sequence_parallel_region(input_, ...)
        # → 调用 _GatherFromSequenceParallelRegion.forward
        # → all-gather: [S/TP,B,H] × TP ranks → [S,B,H]
    else:
        input_parallel = input_  # 已是完整 [S,B,H]

    # 矩阵乘: [S,B,H] × [H, H/TP] → [S,B,H/TP]
    output_parallel = linear_with_grad_accumulation(input_parallel, weight, bias)
    # 输出已按 hidden 维切分 → 直接返回
    return output_parallel

# === RowParallelLinear.forward (layers.py ~L1300-1369) ===
def forward(self, input_):
    # input 已经是 [S,B,H/TP] (来自上游 ColumnParallel 输出)
    # 矩阵乘: [S,B,H/TP] × [H/TP, H] → [S,B,H] (partial sum)
    output_parallel = linear_with_grad_accumulation(input_, weight, bias)

    if self.config.sequence_parallel:
        # SP 模式: reduce-scatter → [S/TP,B,H] (替代 all-reduce!)
        output = reduce_scatter_to_sequence_parallel_region(output_parallel)
        # → 调用 _ReduceScatterToSequenceParallelRegion.forward
        # → reduce partial sums + scatter along seq dim
    else:
        output = all_reduce(output_parallel)  # 传统: all-reduce → [S,B,H]
    return output
```

**完整 Forward 数据流** (一个 Transformer Layer, SP=True, TP=2):
```
Input: [S/TP, B, H]  ← LayerNorm 输出 (SP 切分态)
  │
  ├─ all-gather ──→ [S, B, H]      ← ColumnParallel input (Q/K/V projection)
  │                    │
  │              matmul (per-rank)
  │                    ↓
  │              [S, B, H/TP]       ← Q/K/V 已切分 (attention heads 维)
  │                    │
  │              Self-Attention (local heads)
  │                    ↓
  │              [S, B, H/TP]       ← attention output (切分态)
  │                    │
  │              RowParallel (output projection)
  │                    ↓
  │              [S, B, H] partial  ← 需要跨 rank reduce
  │                    │
  └─ reduce-scatter → [S/TP, B, H] ← 回到 SP 切分态 (下一层输入)
```

### 1.4 SP 的内存收益量化

| 配置 | 每层激活 (per microbatch) | 40层总计 |
|------|--------------------------|----------|
| 无 SP (TP=2, S=4096, H=5120) | S×B×H×2B = 40MB | 1.6GB |
| 有 SP (TP=2) | S/TP×B×H×2B = 20MB | 800MB |
| 有 SP (TP=4) | S/TP×B×H×2B = 10MB | 400MB |

**节省**: activation 内存 ÷ TP, 对于 TP=8 节省 87.5%。

### 1.5 SP 通信开销分析

```
传统 TP (无 SP): all-reduce per layer = 2 × S×B×H×dtype (ring)
SP:             all-gather + reduce-scatter = S×B×H×dtype + S×B×H×dtype = 2×S×B×H×dtype
                通信总量完全相同!

差异在于 overlap 能力:
  - all-gather 可在 forward 计算开始前异步发起 (prefetch next layer)
  - reduce-scatter 可在 backward 计算完成后异步执行

Qwen3-10B (TP=2, S=4096, H=5120, BF16):
  Per layer: AG = 4096×1×5120×2B = 40MB, RS = 40MB
  40 layers fwd: 40×40 = 1.6GB
  NVLink 450GB/s (单向): 1.6GB / 450 ≈ 3.6ms
  可与计算完全 overlap → 实际暴露延迟 ≈ 0
```

---

## 2. Context Parallelism (CP) — Attention 层的长序列切分

### 2.1 设计动机

SP 解决了 MLP/LayerNorm 的序列内存问题, 但 **Self-Attention** 的计算复杂度是 O(S²), 且需要完整 KV 序列:

```
问题: seq_length = 128K, hidden = 5120, num_heads = 40
  Q×K^T: [S, head_dim] × [head_dim, S] → [S, S] = 128K × 128K × 2B = 32GB per head!
  单卡 80GB 内存根本放不下

解决: CP 将序列切分到多个 rank, 通过通信获取远程 KV:
  CP=4: 每 rank 持有 S/4 = 32K tokens 的 Q
  通过 ring/all-gather 逐步获取其他 rank 的 KV
  Attention(Q_local, KV_全部) → output_local
```

### 2.2 四种通信类型 (transformer_config.py L974-988)

```python
# transformer_config.py L974
cp_comm_type: Optional[Union[str, List[str]]] = None
# 可以是全局字符串 (所有层相同), 也可以是 per-layer list:
# transformer_config.py L2506-2514
if isinstance(self.cp_comm_type, list):
    assert len(self.cp_comm_type) == self.num_layers
```

| cp_comm_type | 通信模式 | 通信量 | 计算-通信 Overlap | 适用场景 |
|:-------------|:---------|:-------|:-----------------|:---------|
| `"p2p"` | Ring Attention: 逐步 P2P 异步交换 KV | (CP-1)×KV_chunk | ✅ 完美 overlap | **默认**, 通用 |
| `"all_gather"` | 先 AG 获取完整 KV 再计算 | KV_full 一次性 | ❌ 无 overlap | 小 CP, 极低延迟需求 |
| `"a2a"` | Ulysses: all-to-all 按 head 重分布 | all-to-all | 部分 | heads >> CP |
| `"a2a+p2p"` | 层级: 节点内 a2a + 节点间 p2p | 混合 | ✅ | 多节点大 CP |

### 2.3 Ring Attention 实现 (TEDotProductAttention)

Ring Attention 在 TransformerEngine 中实现, Megatron 通过 `TEDotProductAttention` 传参集成:

```python
# transformer_engine.py L1447-1471
if self.config.context_parallel_size > 1:
    extra_kwargs["cp_group"] = pg_collection.cp
    extra_kwargs["cp_global_ranks"] = torch.distributed.get_process_group_ranks(
        pg_collection.cp
    )
    # FlagScale 扩展: 使用平台抽象的 Stream
    if getattr(TEDotProductAttention, "cp_stream") is None:
        TEDotProductAttention.cp_stream = cur_platform.Stream()  # L1452
    extra_kwargs["cp_stream"] = TEDotProductAttention.cp_stream

    # 选择通信类型
    if cp_comm_type is None:
        extra_kwargs["cp_comm_type"] = "p2p"       # 默认 ring
    elif cp_comm_type == "a2a+p2p":
        extra_kwargs["cp_comm_type"] = "a2a+p2p"
        extra_kwargs["cp_group"] = get_hierarchical_context_parallel_groups(...)
        # 传入两级 group (节点内 + 节点间)
    else:
        extra_kwargs["cp_comm_type"] = cp_comm_type  # "all_gather" / "a2a"
```


**Ring Attention 时序图** (CP=4, 4 ranks):
```
Step 0: 各 rank 计算 local attention
  rank_0: Attn(Q_0, KV_0) → partial_0    [同时] send KV_0→rank_1, recv KV_3←rank_3
  rank_1: Attn(Q_1, KV_1) → partial_1    [同时] send KV_1→rank_2, recv KV_0←rank_0
  rank_2: Attn(Q_2, KV_2) → partial_2    [同时] send KV_2→rank_3, recv KV_1←rank_1
  rank_3: Attn(Q_3, KV_3) → partial_3    [同时] send KV_3→rank_0, recv KV_2←rank_2

Step 1: 计算接收到的 KV chunk
  rank_0: Attn(Q_0, KV_3) → accumulate   [同时] send KV_3→rank_1, recv KV_2←rank_3
  rank_1: Attn(Q_1, KV_0) → accumulate   [同时] send KV_0→rank_2, recv KV_3←rank_0
  ...

Step 2: 继续 ring...
Step CP-1 (Step 3): 最终累加 → 完整 attention 输出
```

**关键实现细节**:
1. P2P 通信在独立 CUDA stream (`cp_stream`) 上执行, 与 attention 计算并行
2. 每步的 attention 输出使用 online softmax (log-sum-exp) 累加, 无需存储完整 [S,S] 矩阵
3. Causal mask 优化: 只有 `Q_i` 与 `KV_j (j≤i)` 的 chunk 才有有效计算, 跳过无效步

### 2.4 Hierarchical CP — 多级通信拓扑 (parallel_state.py L389-448, L1046-1052)

**问题**: 多节点 CP 时, 跨节点 IB 带宽远低于节点内 NVLink, 统一 ring 效率低。

**解决**: 两级层次 — 节点内 all-to-all (高带宽) + 节点间 P2P ring (低延迟):

```python
# parallel_state.py L1046-1052
if hierarchical_context_parallel_sizes:
    assert np.prod(hierarchical_context_parallel_sizes) == context_parallel_size
    # e.g., [8, 2] → 节点内 CP=8 (NVLink a2a), 节点间 CP=2 (IB p2p)
    hierarchical_groups, _ = create_hierarchical_groups(
        cp_group_ranks, hierarchical_context_parallel_sizes, ...
    )

# parallel_state.py L389-448: create_hierarchical_groups
def create_hierarchical_groups(ranks, hierarchical_group_sizes, pg_options=None):
    """
    创建多级通信 group。
    hierarchical_group_sizes = [2, 2, 4] 意味着:
      Level 0: 2 GPU 一组 (最近邻)
      Level 1: 2 组合并 (4 GPU)
      Level 2: 4 组合并 (16 GPU)
    返回每级的 ProcessGroup list。
    """
    hierarchical_groups = []
    for level in range(len(hierarchical_group_sizes)):
        # 按 stride 公式切分: u×s×l 模式
        u = int(np.prod(hierarchical_group_sizes[:level]))
        s = hierarchical_group_sizes[level]
        l = int(np.prod(hierarchical_group_sizes[level + 1:]))
        # 创建该级别的所有 sub-group
        ...
    return hierarchical_groups, hierarchical_groups_gloo
```

**通信拓扑示例**:
```
配置: 4 nodes × 8 GPUs, CP=16, hierarchical_context_parallel_sizes=[8, 2]

Level 1 (节点内, NVLink 900GB/s):
  Group A: GPU 0-7 (node 0) — all-to-all 重分布 attention heads
  Group B: GPU 8-15 (node 1) — all-to-all

Level 2 (节点间, IB 400Gb/s):
  Ring: node_0 ↔ node_1 — P2P 交换 KV chunks

优势:
  - 节点内 a2a: S/8 × num_heads 的数据, 利用 NVLink 全带宽
  - 节点间 p2p: 仅交换 KV (远小于 Q+K+V), 隐藏于计算中
```

### 2.5 CP 约束条件 (transformer_config.py L1283-1299)

```python
# GQA 场景: num_kv_heads 必须能被 TP×CP 整除
tp_cp_size = tensor_model_parallel_size * context_parallel_size
assert num_query_groups % tp_cp_size == 0, (
    f"num_query_groups ({num_query_groups}) must be divisible by "
    f"TP×CP ({tensor_model_parallel_size} × {context_parallel_size})."
)

# 非 GQA (MHA): num_attention_heads % TP×CP == 0
tp_cp_size = tensor_model_parallel_size * context_parallel_size
assert num_attention_heads % tp_cp_size == 0
```

**实际影响**: Qwen3-10B (num_kv_heads=8, TP=2) 时 CP 最大 = 8/2 = 4。

---

## 3. Hybrid CP Schedule — 变长序列的动态调度 (hybrid_cp_schedule.py)

### 3.1 设计动机

**问题**: 实际训练中 packed samples 的序列长度差异巨大 (4K ~ 128K), 固定 CP 导致:
- 长序列 rank 成为瓶颈 (计算量 ∝ S²)
- 短序列 rank 空闲等待
- 全局 barrier 效率极低

**解决**: `BalancedCPScheduler` 动态分配不同 CP size 给不同子序列:

```
Global batch 包含: [128K, 64K, 32K, 8K, 8K, 4K, 4K, 4K] 8个 sub-samples
DPxCP = 16 GPUs

调度结果:
  GPU 0-7:  处理 128K sample (CP=8, 工作量: 128K²/8)
  GPU 8-11: 处理 64K sample  (CP=4, 工作量: 64K²/4)
  GPU 12-13: 处理 32K sample (CP=2, 工作量: 32K²/2)
  GPU 14:   处理 8K+8K       (CP=1, 工作量: 2×8K²)
  GPU 15:   处理 4K+4K+4K    (CP=1, 工作量: 3×4K²)

→ 所有 GPU 工作量 ≈ 均衡!
```

### 3.2 BalancedCPScheduler 核心算法 (L19-479)

```python
class BalancedCPScheduler:
    def __init__(self, max_seq_len_per_rank, dp_cp_group):
        self.max_seq_len_per_rank = max_seq_len_per_rank  # 单 rank 最大处理长度
        self.total_hdp_gpus = dp_cp_group.size()          # DPxCP 总 GPU 数

    @lru_cache(maxsize=128)
    def gpus_needed(self, seq_len: int) -> int:
        """计算 sub-sample 需要多少 CP rank (向上取 2 的幂)"""
        return max(1, 2 ** ceil(log2(seq_len / self.max_seq_len_per_rank)))
        # 例: max_seq_len_per_rank=16K
        #   seq=128K → 2^ceil(log2(8)) = 8 GPUs
        #   seq=32K  → 2^ceil(log2(2)) = 2 GPUs
        #   seq=8K   → max(1, 2^ceil(log2(0.5))) = 1 GPU

    @lru_cache(maxsize=128)
    def get_total_workload(self, seq_length, cp_size=None):
        """估算工作量: attention 复杂度 ∝ S²/CP"""
        if cp_size is None:
            cp_size = self.gpus_needed(seq_length)
        return (seq_length * seq_length) / cp_size
```

### 3.3 调度五步法 — next_hdp_group (L109-459)

`next_hdp_group()` 是核心调度函数, 为一个 microbatch 分配所有 GPU 的工作:

```
┌──────────────────────────────────────────────────────────────────────┐
│  Step 1: make_buckets_equal()                                         │
│    按 CP size 将 sub-samples 分桶, 桶间工作量均衡                      │
│    长序列桶 (CP=8): [128K]                                            │
│    中序列桶 (CP=2): [32K, 32K]                                        │
│    短序列桶 (CP=1): [8K, 8K, 4K, 4K]                                  │
├──────────────────────────────────────────────────────────────────────┤
│  Step 2: 从桶中取序列, 分配给 GPU group                               │
│    (a) 若有已存在的同 size group → 分配给负载最低的 group              │
│    (b) 若有足够空闲 GPU → 创建新 group                                │
│    选择 (a) vs (b): 比较现有 group 最大负载 vs 新 group 负载           │
├──────────────────────────────────────────────────────────────────────┤
│  Step 3: 分配序列到 group 内所有 member GPU                           │
│    per_gpu_cost = get_total_workload(seq_len)                         │
│    for r in chosen_members: exec_times[r] += per_gpu_cost             │
├──────────────────────────────────────────────────────────────────────┤
│  Step 4: trim_overload() — 裁剪过载                                   │
│    若 max(exec_times) - min(exec_times) > delta × max(exec_times):    │
│      从最重负载 group 移除最后一个序列 → 放回 leftovers                │
│    反复执行直到 slack ≤ 5%                                            │
├──────────────────────────────────────────────────────────────────────┤
│  Step 5: fill_empty_gpus() — 填充空闲 GPU                            │
│    若存在无工作的 GPU:                                                │
│      找到最小 group, 将其 CP size 翻倍 (扩展 member 到相邻空闲 GPU)    │
│      递归执行直到无空闲 GPU                                           │
│    确保: total_work_after >= total_work_before (不丢序列)             │
└──────────────────────────────────────────────────────────────────────┘
```


### 3.4 执行入口 — hybrid_context_parallel_forward_backward (L482-665)

```python
def hybrid_context_parallel_forward_backward(
    forward_step_func, data_iterator, model, num_microbatches,
    input_tensor, output_tensor_grad, forward_data_store, config, ...
):
    """
    Hybrid CP 的完整执行流程:
    1. DP rank 0 获取数据, 通过 BalancedCPScheduler 调度
    2. Broadcast 每组 sub-sample 数量给所有 TP rank
    3. 按 group 执行 forward/backward, group 间插入 barrier
    """
    # --- Phase 1: 数据获取与调度 (仅 TP rank 0) ---
    hdp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=True)
    is_first_tp_rank = parallel_state.get_tensor_model_parallel_rank() == 0

    if is_first_tp_rank:
        data = next(data_iterator)  # 获取 pre-scheduled batch
        sample_id_groups = data[1]  # 调度结果: 每组每 GPU 的 sample IDs
        batch = data[0]             # 原始 samples

    # --- Phase 2: Broadcast 调度信息 ---
    # 将 [每组的 sub-sample 数量] broadcast 给所有 TP rank
    num_samples_this_group = _broadcast_num_samples_this_group(...)
    num_total_groups = num_samples_this_group.shape[0]

    # --- Phase 3: 按 group 执行 (group 间有 barrier) ---
    with no_sync_func():  # 禁止梯度同步 (除最后一步)
        for j in range(num_total_groups - 1):
            for i in range(num_samples_this_group[j]):
                # 每个 sub-sample 独立 forward + backward
                new_data_iterator = _get_new_data_iterator(i, j)
                output_tensor, num_tokens = forward_step(...)
                if not forward_only:
                    backward_step(...)
            # Group 间 barrier: 确保所有 rank 准备好切换 CP group size
            torch.distributed.barrier(
                parallel_state.get_data_parallel_group(with_context_parallel=True)
            )

    # --- Phase 4: 最后一个 group 的最后一个 sub-sample (梯度同步点) ---
    # 在 no_sync 外执行 → 触发梯度 all-reduce
    output_tensor, num_tokens = forward_step(...)
    if not forward_only:
        backward_step(...)
```

**Barrier 的必要性**:
```
Group 0: GPU 0-7 (CP=8, 128K), GPU 8-15 (CP=8, 另一个128K)
Group 1: GPU 0-3 (CP=4, 64K), GPU 4-7 (CP=4, 64K), GPU 8-11 (CP=4), GPU 12-15 (CP=4)

若无 barrier: GPU 8-15 可能在 Group 0 未完成时进入 Group 1
             → Group 1 需要 GPU 4-7 参与 CP 通信, 但它们还在 Group 0 → DEADLOCK!
```

### 3.5 Sub-sample CP Size 传播

```python
# L544-555: _get_new_data_iterator
def _get_new_data_iterator(sample_id_in_group, group_id):
    if is_first_tp_rank:
        sub_sample_id = sample_ids_this_group[sample_id_in_group]
        sample = batch[sub_sample_id]
        # 计算该 sub-sample 实际使用的 CP size
        partner_cp_size = len(
            [True for sample_ids in sample_id_groups[group_id]
             if sub_sample_id in sample_ids]
        )
        sample["local_cp_size"] = torch.tensor(partner_cp_size, dtype=torch.int32)
        # TE 使用 local_cp_size 决定 ring 步数和 group 范围
        return RerunDataIterator(iter([sample]))
```

---

## 4. SP vs CP 深度对比

| 维度 | Sequence Parallelism (SP) | Context Parallelism (CP) |
|:-----|:--------------------------|:-------------------------|
| **切分位置** | MLP, LayerNorm, Dropout (element-wise ops) | Attention (Q/KV 交互) |
| **通信域** | TP group (同一组 GPU) | CP group (可独立于 TP) |
| **通信原语** | all-gather + reduce-scatter | ring P2P / all-gather / all-to-all |
| **内存节省** | 激活 ÷ TP (非 attention 部分) | KV cache + attention buffer ÷ CP |
| **序列扩展** | ❌ 不增加可训练序列长度 | ✅ max_seq = base × CP |
| **约束** | 必须 TP > 1 | num_kv_heads % (TP×CP) == 0 |
| **Overlap** | AG 可 prefetch, RS 可 defer | Ring P2P 与 compute 完美 overlap |
| **计算正确性** | 数学等价 (切分+通信=完整计算) | 需要 online softmax 保证数值一致 |
| **开销来源** | 通信 (但可完全 overlap) | 通信 + barrier (group 切换) |

### 4.1 组合使用数据流

```
完整长序列配置: TP=4, CP=4, SP=True, seq=128K

数据形状变换 (单个 rank 视角):
  Input tokens: [128K/(4×4), B, H] = [8K, B, H]     ← SP+CP 双重切分
    │
    ├─ LayerNorm: [8K, B, H]                          ← SP 域, element-wise
    │
    ├─ all-gather (TP group): [32K, B, H]             ← 恢复 S/CP 长度
    │
    ├─ QKV projection: [32K, B, H] → [32K, B, H/4]   ← TP 切分 heads
    │
    ├─ Ring Attention (CP group):
    │     Q_local=[32K, B, H_head/4], KV 通过 ring 获取
    │     输出: [32K, B, H_head/4]                     ← 本地 Q 的完整 attention
    │
    ├─ Output projection (RowParallel):
    │     [32K, B, H/4] × [H/4, H] → [32K, B, H] partial
    │
    └─ reduce-scatter (TP group): [8K, B, H]          ← 回到 SP+CP 切分态
```

---

## 5. FlagScale 平台抽象扩展

### 5.1 cp_stream 平台适配 (transformer_engine.py L1368, L1452)

```python
# 原始 Megatron: torch.cuda.Stream() — 仅支持 NVIDIA GPU
# FlagScale 扩展:
class TEDotProductAttention:
    cp_stream: cur_platform.Stream = None  # FlagScale Add

# 初始化时:
if getattr(TEDotProductAttention, "cp_stream") is None:
    TEDotProductAttention.cp_stream = cur_platform.Stream()
    # cur_platform 可以是 CUDA/ROCm/Ascend 等

# 作用: CP ring 通信在专用 stream 上执行
#       与 attention 计算 stream 并行 → overlap
```

### 5.2 Hybrid CP 设备抽象 (hybrid_cp_schedule.py L11-15)

```python
# FlagScale Begin
from megatron.plugin.platform import get_platform
cur_platform = get_platform()
# FlagScale End

# 使用场景: broadcast 时获取当前设备
def _broadcast_num_samples_this_group(num_samples_this_group):
    dev = cur_platform.current_device()  # 替代 torch.cuda.current_device()
    n = torch.tensor([n], dtype=torch.int64, device=dev)
    ...
```

### 5.3 设计意义

FlagScale 的 CP 平台扩展遵循 **最小侵入原则**:
- 仅修改设备获取和 stream 创建 (2 处)
- 不改变调度算法、通信拓扑、数值计算
- 使 CP 可运行在 华为 Ascend / AMD ROCm 等非 NVIDIA 硬件上

---

## 6. 配置参数速查

| 参数 | 默认值 | 说明 | 来源 |
|:-----|:-------|:-----|:-----|
| `sequence_parallel` | False | 启用 SP (需 TP>1) | transformer_config |
| `context_parallel_size` | 1 | CP 并行度 | parallel_state init |
| `cp_comm_type` | None→"p2p" | CP 通信: p2p/all_gather/a2a/a2a+p2p | transformer_config L974 |
| `cp_comm_type` (list) | — | 每层独立设置, len==num_layers | transformer_config L2506 |
| `hierarchical_context_parallel_sizes` | None | 多级 CP 分组 (e.g., [8,2]) | parallel_state L1046 |
| `hybrid_context_parallel` | False | 启用变长序列动态调度 | parallel_state L585 |
| `max_seq_len_per_rank` | — | 单 rank 处理的最大序列长度 | BalancedCPScheduler init |

### 6.1 约束条件汇总

```python
# 1. KV head 约束 (transformer_config.py L1283-1299)
if using_GQA:
    assert num_query_groups % (TP × CP) == 0
else:
    assert num_attention_heads % (TP × CP) == 0

# 2. Per-layer cp_comm_type 长度校验 (L2506-2514)
if isinstance(cp_comm_type, list):
    assert len(cp_comm_type) == num_layers

# 3. Hierarchical CP 乘积校验 (parallel_state.py L1047)
assert np.prod(hierarchical_context_parallel_sizes) == context_parallel_size

# 4. Hybrid CP 与 PP 互斥 (当前实现)
# hybrid_cp_schedule 不支持 PP > 1 (TODO in source)
```

---

## 7. 通信量量化分析

### 7.1 SP 通信 (TP=2, S=4096, H=5120, BF16)

```
单层 Forward:
  ColumnParallel AG: S × B × H × 2B = 4096 × mbs × 5120 × 2 = 40MB × mbs
  RowParallel RS:    S × B × H × 2B = 40MB × mbs
  
  MLP 有两组 (gate+up / down): 2 × (AG + RS) = 160MB × mbs
  Attention (QKV + output): 2 × (AG + RS) = 160MB × mbs
  
  单层总计: 320MB × mbs (fwd only)
  40层 fwd+bwd: 320 × 2 × 40 = 25.6GB × mbs
  NVLink 900GB/s 双向: 25.6GB / 900 ≈ 28ms × mbs
  
  实际暴露: 与计算高度 overlap, 通常 < 5ms 暴露
```

### 7.2 CP Ring 通信 (CP=4, S=128K, H=5120, n_kv_heads=8, TP=2, BF16)

```
每步交换 KV:
  KV_per_rank = S/CP × 2 × (n_kv_heads/TP) × head_dim × 2B
             = 32K × 2 × 4 × 128 × 2 = 64MB

Ring 步数: CP-1 = 3
单层总 CP 通信: 3 × 64MB = 192MB

40层: 192 × 40 = 7.5GB
NVLink 450GB/s (单向 P2P): 7.5 / 450 ≈ 17ms

但 ring 通信与 attention 计算完美 overlap:
  Attention 计算时间 (per step): 32K×32K×128×4heads / 312 TFLOPS ≈ 3.4ms
  通信时间 (per step): 64MB / 450GB/s ≈ 0.14ms
  → 通信完全隐藏在计算中!
```

### 7.3 Hierarchical CP 通信 (CP=16, [8,2])

```
Level 1 (节点内 a2a, NVLink 900GB/s):
  All-to-all: 重分布 heads, 数据量 = KV_total / 8 × 8 = KV_total
  8K tokens × 2 × 8heads × 128 × 2B = 32MB
  时间: 32MB / 900GB/s ≈ 0.04ms

Level 2 (节点间 p2p, IB 50GB/s):
  Ring KV 交换: 8K × 2 × 4heads × 128 × 2B / 步 = 16MB/步
  1 步: 16MB / 50GB/s ≈ 0.32ms
  
  vs 统一 ring (纯 IB): 16步 × 0.32ms = 5.1ms
  层级优化后: 1步 IB + 7步 NVLink ≈ 0.32 + 7×0.04 = 0.6ms
  → 节省 88%!
```

---

## 8. 设计决策与权衡

| 决策 | 选择 | 替代方案 | 理由 |
|:-----|:-----|:---------|:-----|
| SP 通信原语 | RS+AG 替代 AR | 保持 AR | 内存 ÷ TP, 总通信量不变但可 overlap |
| CP 默认通信 | Ring P2P | All-gather | P2P 可与 compute overlap, AG 不可 |
| Ring vs Ulysses | 均支持 | 二选一 | Ring 对 heads 数无约束; Ulysses 需 heads≥CP |
| Hybrid CP 调度 | 贪心 + 裁剪 | 整数规划 | 低延迟在线调度, O(n) 复杂度 |
| Group 间同步 | Barrier | 无同步 | 防止 CP size 切换时 deadlock |
| fill_empty_gpus | 扩展 CP size | 复制数据 | 保持正确性, 不引入冗余计算 |
| Hierarchical | a2a+p2p | 统一 ring | 匹配硬件拓扑 (NVLink >> IB) |
| Per-layer cp_comm | List[str] | 全局统一 | 不同层可能有不同 head 数 (MoE) |
| Platform 抽象 | cur_platform.Stream | torch.cuda.Stream | 支持非 NVIDIA 硬件 |

---

## 9. 性能调优建议

### 9.1 短序列 (S ≤ 8K)
- **SP=True**: 必须启用, 内存收益大, 通信可完全 overlap
- **CP=1**: 不需要 CP, 通信开销 > 收益

### 9.2 中等序列 (8K < S ≤ 64K)
- **SP=True, CP=2~4**: 平衡通信与内存
- **cp_comm_type="p2p"**: Ring attention overlap 效果最佳
- 确保 `num_kv_heads % (TP × CP) == 0`

### 9.3 超长序列 (S > 64K)
- **SP=True, CP=8~16**: 必须使用 CP 才能放入内存
- **多节点**: 使用 `hierarchical_context_parallel_sizes=[8, CP/8]`
- **变长 packed samples**: 启用 `hybrid_context_parallel=True`

### 9.4 变长序列训练
- 启用 `hybrid_context_parallel`, 设置合理的 `max_seq_len_per_rank`
- Group 数量影响 barrier 频率 — 过多 group 增加同步开销
- 监控 GPU 利用率: 若某些 rank 经常空闲 → 调小 `max_seq_len_per_rank`

---

## 10. 扩展阅读

- Ring Attention: "Ring Attention with Blockwise Transformers for Near-Infinite Context" (Liu et al., 2023)
- Sequence Parallelism: "Reducing Activation Recomputation in Large Transformer Models" (Korthikanti et al., 2022, NVIDIA)
- DeepSpeed Ulysses: "DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models" (Jacobs et al., 2023)
- Hierarchical CP: Megatron-LM internal extension for multi-node long-sequence training
- Online Softmax: "Online normalizer calculation for softmax" (Milakov & Gimelshein, 2018) — Ring Attention 的数学基础
