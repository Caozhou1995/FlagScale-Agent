# 03 - Data Parallelism (DP) & Distributed Optimizer 完整分析

## 源码位置

| 文件 | 行数 | 功能 |
|------|------|------|
| `megatron/core/distributed/distributed_data_parallel.py` | 722 | DDP wrapper, backward hook, forward pre-hook, no_sync |
| `megatron/core/distributed/distributed_data_parallel_config.py` | 232 | DDPConfig dataclass (所有 DP 配置选项) |
| `megatron/core/distributed/param_and_grad_buffer.py` | 1496 | 连续 buffer + bucket 分组 + 通信调度 (核心) |
| `megatron/core/optimizer/distrib_optimizer.py` | 2741 | 分布式优化器 (ZeRO-1+2): shard 映射 + reduce-scatter/all-gather |
| `megatron/core/optimizer/optimizer.py` | ~800 | MegatronOptimizer / MixedPrecisionOptimizer 基类 |
| `megatron/core/distributed/finalize_model_grads.py` | 606 | 梯度最终处理: embedding/position 跨 PP 同步, per-token loss 归一化 |
| `megatron/core/distributed/data_parallel_base.py` | — | _BaseDataParallel 基类 |

---

## 1. DDP 架构概述

### 1.1 设计哲学

Megatron DDP **完全自研**，不使用 PyTorch 的 `torch.nn.parallel.DistributedDataParallel`。关键区别:

| 特性 | PyTorch DDP | Megatron DDP |
|------|------------|--------------|
| 梯度存储 | 参数各自的 `.grad` | 连续 buffer (`main_grad`) |
| 通信触发 | autograd hook + bucket ready | 手动 hook + golden count 匹配 |
| Bucket 策略 | 按参数逆序自动分配 | 显式 bucket_size + 可关闭 |
| Overlap 粒度 | bucket ready 即通信 | 同 + param gather overlap |
| 优化器集成 | 独立 | 深度集成 (DistributedOptimizer) |
| ZeRO 支持 | 需 DeepSpeed/FSDP | 原生 ZeRO-1+2 |

**设计目标**: 连续梯度 buffer → 零碎片 + 一次性 reduce-scatter + 直接映射到 optimizer shard。

### 1.2 核心数据结构 (param_and_grad_buffer.py)

```
_ParamAndGradBuffer (line 768)
├── param_data: torch.Tensor  [连续 buffer, 所有参数的数据]
├── grad_data:  torch.Tensor  [连续 buffer, 所有参数的梯度]
├── buckets: List[_ParamAndGradBucket]  [按 bucket_size 切分]
│   └── _ParamAndGradBucket
│       ├── param_data: view of buffer segment
│       ├── grad_data:  view of buffer segment
│       ├── params_list: List[Parameter]  [属于此 bucket 的参数]
│       └── numel: int  [元素总数]
└── bucket_groups: List[_ParamAndGradBucketGroup]  [bucket 的通信调度单元]

_ParamAndGradBucketGroup (line 165)
├── buckets: List[_ParamAndGradBucket]  [一个或多个 bucket]
├── param_to_bucket: Dict[Param → Bucket]
├── per_param_grad_ready_counts: Dict[Param → int]  [当前 batch 的 grad ready 计数]
├── golden_per_param_grad_ready_counts: Dict[Param → int]  [首 batch 学到的目标计数]
├── is_last_microbatch: bool  [no_sync 控制: False 时跳过通信]
├── grad_reduce_handle: Optional[async handle]  [RS/AR 异步句柄]
├── param_gather_handle: Optional[async handle]  [AG 异步句柄]
└── next_param_gather_bucket_group: Optional[Self]  [prefetch 链指针]
```

### 1.3 参数三路分类 (ddp.py:118-144)

```python
for name, param in module.named_parameters():
    if getattr(param, 'allreduce', True):
        dense_params.append(param)                  # → dp_group (标准 DP 参数)
    elif getattr(param, "is_engram_embedding", False):
        engram_embedding_params.append(param)       # → engram_dp_group (FlagScale)
    else:
        expert_parallel_params.append(param)        # → expt_dp_group (MoE expert 参数)
```

**三组独立 buffer**: 每类参数有自己的 `_ParamAndGradBuffer` 和 `bucket_group`，互不干扰:
```python
self.buffers, self.bucket_groups = _allocate_buffers_for_parameters(
    dense_params, self.intra_dp_cp_group, gradient_scaling_factor)
self.expert_parallel_buffers, self.expert_parallel_bucket_groups = _allocate_buffers_for_parameters(
    expert_parallel_params, self.intra_expt_dp_group, expert_gradient_scaling_factor)
# FlagScale: engram embedding 独立 buffer
self.engram_embedding_buffers, self.engram_embedding_bucket_groups = _allocate_buffers_for_parameters(
    engram_embedding_params, self.engram_dp_group, engram_embedding_gradient_scaling_factor)
```

### 1.4 梯度缩放策略 (ddp.py:276-322)

两种模式确保最终梯度被 `1/dp_size` 缩放:

| 模式 | gradient_scaling_factor | Reduce Op | 最终效果 |
|------|------------------------|-----------|---------|
| `average_in_collective=False` (默认) | 1/dp_size | SUM | grad × (1/dp) → sum = grad |
| `average_in_collective=True` | 1.0 | AVG | grad × 1.0 → avg = grad/dp |

Expert 参数在 `average_in_collective=True` 时: factor = `edp_size/dp_size`, reduce op = AVG over edp_group → 最终 = factor × (1/edp) = 1/dp。

---

## 2. Overlap Grad Reduce 机制

### 2.1 原理 (overlap_grad_reduce=True)

Backward 过程中, 每个参数计算完梯度后立即尝试触发 bucket group 的通信。由于参数按逆序（最后一层先）完成 backward，bucket group 也按逆序就绪:

```
Timeline (4 layers, 2 bucket groups):
Backward:    [grad_L4] [grad_L3]  [grad_L2] [grad_L1]
                              ↓                     ↓
Bucket ready:    BG1 (L4+L3 grads done)    BG0 (L2+L1 grads done)
Comm:                 |── RS/AR BG1 ──|         |── RS/AR BG0 ──|
                      ←── overlap ────→         ←── overlap ────→
```

### 2.2 Backward Hook 注册 (ddp.py:366-498)

**注册方式** (ddp.py:388-394):
```python
for param in self.module.parameters():
    if param.requires_grad:
        # 获取 gradient accumulator function (autograd 内部节点)
        param_tmp = param.expand_as(param)
        grad_acc = param_tmp.grad_fn.next_functions[0][0]
        grad_acc.register_hook(self._make_backward_post_hook(param))
        self.grad_accs.append(grad_acc)  # 持有引用防止 GC
```

**Hook 逻辑** (ddp.py:470-498):
```python
def _make_backward_post_hook(self, param):
    def hook(*unused):
        if param in self.param_to_bucket_group:
            # Step 1: 累加 param.grad 到连续 buffer 的 main_grad
            if param.grad is not None and not param.grad_added_to_main_grad:
                param.main_grad.add_(param.grad.data)
            param.grad = None  # 释放 autograd 分配的 grad

            # Step 2: 如果 overlap 启用, 通知 bucket group
            if self.ddp_config.overlap_grad_reduce:
                self.param_to_bucket_group[param].register_grad_ready(param)
    return hook
```

### 2.3 Golden Count 机制 (param_and_grad_buffer.py)

**问题**: 多个 microbatch 时，同一个参数的 hook 会被调用多次。Bucket group 需要知道"所有 microbatch 都完成了"才能发起通信。

**解决**: 第一个 batch (is_first_batch=True) 记录每个参数被调用的次数作为 "golden count"，后续 batch 通过比较判断就绪:

```python
def register_grad_ready(self, param, force_all_reduce=False):
    self.per_param_grad_ready_counts[param] += 1

    if self.is_first_batch:
        # 首 batch: 只记录, 不发起通信
        self.golden_per_param_grad_ready_counts[param] += 1
    else:
        # 后续 batch: 检查是否匹配 golden count
        if self.per_param_grad_ready_counts == self.golden_per_param_grad_ready_counts:
            self.start_grad_sync()  # ← 所有参数所有 microbatch 的 grad 都 ready
```

**为什么是首 batch 学习**: 不同的并行策略 (PP interleaved, MoE EP) 导致参数的 backward 调用次数不一定是 `num_microbatches`。Golden count 自适应学习正确的值。

### 2.4 start_grad_sync 通信调度 (param_and_grad_buffer.py:527-676)

```python
def start_grad_sync(self):
    if not self.is_last_microbatch:
        return  # no_sync 模式: 跳过通信 (梯度累积)

    async_op = self.ddp_config.overlap_grad_reduce

    # 可选: 使用独立 CUDA stream 执行通信
    if self.communication_stream is not None:
        self.communication_stream.wait_stream(torch.cuda.current_stream())
        stream_context = torch.cuda.stream(self.communication_stream)
    else:
        stream_context = nullcontext()

    with stream_context:
        with _coalescing_manager(group, async_ops=async_op) as cm:
            for bucket in self.buckets:
                if self.ddp_config.use_distributed_optimizer:
                    # reduce-scatter: 每 rank 保留 1/DP shard
                    local_shard = shard_buffer(bucket.grad_data, dp_size)[dp_rank]
                    dist_reduce_scatter_func(local_shard, bucket.grad_data, group=dp_group,
                                            async_op=async_op)
                else:
                    # all-reduce: 所有 rank 保持完整梯度
                    torch.distributed.all_reduce(bucket.grad_data, group=dp_group,
                                                 async_op=async_op)
        self.grad_reduce_handle = cm  # 保存句柄, 后续 wait
```

### 2.5 Bucket 大小策略 (ddp.py:60-76, 106-113)

```python
# 自动计算 bucket size
if ddp_config.bucket_size is None:
    ddp_config.bucket_size = max(40000000, 1000000 * dp_group.size())
    # dp_size=8: bucket = 40M params → ~80MB per bucket (bf16)
    # dp_size=64: bucket = 64M params → ~128MB per bucket

# 非 overlap 模式: 单 bucket (全部参数一次通信)
if not ddp_config.overlap_grad_reduce:
    ddp_config.bucket_size = None  # None = 不分 bucket

# PP rank > 0: 关闭 bucket (不在 critical path)
if disable_bucketing or pp_rank > 0:
    self.bucket_size = None
```

**PP rank > 0 不 bucket 的原因**: 标准 1F1B schedule 中，DP 通信只发生在 backward 最后阶段。PP rank > 0 的 backward 通信被 PP rank 0 的 forward 遮盖，不在 critical path 上。

### 2.6 no_sync 上下文 (ddp.py:500-523)

```python
@contextmanager
def no_sync(self):
    """梯度累积模式: 跳过中间 microbatch 的通信"""
    for bucket_group in (self.bucket_groups + self.expert_parallel_bucket_groups
                         + self.engram_embedding_bucket_groups):
        bucket_group.is_last_microbatch = False  # start_grad_sync 检查此标志
    try:
        yield
    finally:
        for bucket_group in (...):
            bucket_group.is_last_microbatch = True  # 恢复, 下次 ready 时触发通信
```

**PP schedule 中的使用**: 前 n-1 个 microbatch 在 `no_sync()` 内执行，最后一个退出后触发通信。

---

## 3. Distributed Optimizer (ZeRO Stage 1+2)

### 3.1 原理

将 optimizer state 按 DP rank 分片 (ZeRO-1) + gradient 分片 (ZeRO-2):

```
┌─────────────────────────────────────────────────────────┐
│ 传统 DP (all-reduce):                                    │
│   每 rank: 完整 params + 完整 grads + 完整 opt state     │
│   通信: all-reduce(grads) = 2×(T-1)/T × Params×dtype    │
│   内存: Params×(2 + 2 + 12)B = 16×Params (bf16+Adam)   │
├─────────────────────────────────────────────────────────┤
│ Distributed Optimizer (reduce-scatter + all-gather):     │
│   每 rank: 完整 params + 1/DP grads + 1/DP opt state    │
│   通信: RS(grads) + AG(params) = 2×(T-1)/T × Params    │
│   内存: Params×(2 + 2/DP + 12/DP)B                      │
│   DP=8: 内存 = Params×(2 + 0.25 + 1.5)B = 3.75×Params │
│   节省: (16-3.75)/16 = 76.6%!                           │
└─────────────────────────────────────────────────────────┘
```

### 3.2 数据布局: Shard 映射 (distrib_optimizer.py:124-186)

连续 grad buffer 按 DP rank 等分:

```
Buffer: [───────────────── grad_data ─────────────────]
        [  rank0 shard  |  rank1 shard  |  rank2 shard  |  rank3 shard  ]
                                           ↑ 本 rank (rank2) 负责:
                                           - reduce-scatter 后保留此 shard
                                           - 在此 shard 上执行 Adam step
                                           - all-gather 回全量
```

**参数到 shard 的映射** (distrib_optimizer.py):
```python
param_range_map[param] = {
    "gbuf_world": Range(start, end),          # 在全局 buffer 中的绝对位置
    "gbuf_world_in_bucket": Range(...),       # 在所属 bucket 中的偏移
    "gbuf_local": Range(local_start, local_end),  # 在本 rank shard 中的位置
    "param": Range(sub_start, sub_end),       # 参数自身哪一段落在本 rank
}
```

**参数可能跨 shard 边界**: 一个参数可能被切分到两个 rank 的 shard 中。`param_range_map` 精确记录每个 rank 负责参数的哪一段。

### 3.3 完整训练步骤时序

```
Timeline (overlap_grad_reduce=True, overlap_param_gather=True):
─────────────────────────────────────────────────────────────────────
Step N Forward:
  comp: [Layer0 fwd] [Layer1 fwd] [Layer2 fwd] ... [LayerN fwd]
  comm: [AG bucket_K-1]...[AG bucket_0]  ← param gather from step N-1
        ↑ forward pre-hook: finish AG → dispatch next AG

Step N Backward:
  comp: [LayerN bwd] [LayerN-1 bwd] ... [Layer0 bwd]
  comm:        [RS bucket_K] [RS bucket_K-1] ... [RS bucket_0]
               ↑ backward hook: register_grad_ready → start_grad_sync

Step N Optimizer:
  comp: [wait RS] [Adam on shard] [dispatch AG for step N+1]
        ↑ finish_grad_sync: wait for all RS handles

Step N+1 Forward:
  comp: [Layer0 fwd] ...
  comm: [AG bucket_K-1] ... ← dispatched at end of optimizer step
─────────────────────────────────────────────────────────────────────
```

### 3.4 Optimizer Step 实现 (distrib_optimizer.py:2714-2741)

```python
def step_with_ready_grads(self):
    # 1. 等待所有 grad reduce 完成
    for model_chunk in self.model_chunks:
        model_chunk.finish_grad_sync()

    # 2. 在本地 shard 上执行 Adam
    update_successful = super().step_with_ready_grads()  # fp32 master weight update

    # 3. 将更新后的参数 all-gather 回所有 rank
    if not self.ddp_config.overlap_param_gather:
        for model_chunk in self.model_chunks:
            model_chunk.start_param_sync(force_sync=True)  # 同步 all-gather
    else:
        # overlap 模式: AG 在 next forward 的 pre-hook 中 lazy dispatch
        pass

    return update_successful
```

---

## 4. Overlap Param Gather 机制

### 4.1 原理 (overlap_param_gather=True)

Optimizer step 后, 每个 rank 只持有 1/DP 的参数。Forward 前需要 all-gather 恢复完整参数。此机制将 all-gather 与 forward 计算重叠:

```
Timeline:
Step N Optimizer:  [Adam on shard_0] [dispatch AG bucket_K-1]
Step N+1 Forward:  [Layer0 compute] [Layer1 compute] [Layer2 compute]...
All-gather:        [AG BG_K-1] [AG BG_K-2] [AG BG_K-3]...
                    ↑ dispatch  ↑ pre-hook: finish + dispatch next
```

### 4.2 Forward Pre-Hook (ddp.py:434-468)

```python
def _make_forward_pre_hook(self):
    def hook(module, *unused):
        # 遍历本 module 的参数, 确保其 bucket group 的 AG 已完成
        for param in module.parameters(recurse=False):
            if param not in self.param_to_bucket_group:
                continue
            # skip_next_bucket_dispatch: 如果 align_param_gather 或
            # overlap_param_gather_with_optimizer_step, 则不在此处 dispatch next
            skip_next_bucket_dispatch = (
                self.ddp_config.align_param_gather
                or self.overlap_param_gather_with_optimizer_step
            )
            self.param_to_bucket_group[param].finish_param_sync(
                skip_next_bucket_dispatch=skip_next_bucket_dispatch
            )
    return hook
```

**Bucket Group 链式 Prefetch** (ddp.py:256-266):
```python
# 设置 prefetch 链: 逆序遍历 (因为 AG 按逆序执行)
if self.ddp_config.overlap_param_gather:
    num_bucket_groups = len(bucket_groups)
    for i in range(1, num_bucket_groups):
        bucket_groups[num_bucket_groups - i].next_param_gather_bucket_group = (
            bucket_groups[num_bucket_groups - i - 1]
        )
# 效果: BG_K-1.next → BG_K-2.next → BG_K-3.next → ... → BG_0.next → None
```

### 4.3 finish_param_sync 内部流程

```python
def finish_param_sync(self, skip_next_bucket_dispatch=False):
    # 1. 等待本 bucket group 的 AG handle 完成
    if self.param_gather_handle is not None:
        self.param_gather_handle.wait()
        self.param_gather_handle = None

    # 2. FP8 参数后处理 (transpose, etc.)
    if self.has_fp8_params:
        post_all_gather_processing(...)

    # 3. Dispatch next bucket group 的 AG (prefetch)
    if not skip_next_bucket_dispatch and self.next_param_gather_bucket_group is not None:
        self.next_param_gather_bucket_group.start_param_sync()
```

### 4.4 内存代价

Overlap param gather 需要在 all-gather 进行中保持一份完整参数 buffer:
```
额外内存 ≈ params_per_rank × sizeof(param_dtype)
Qwen3-10B (TP=2, PP=2): 2.5B × 2B = 5 GB 额外
```

---

## 5. finalize_model_grads: 梯度后处理 (finalize_model_grads.py)

### 5.1 调用时机

在所有 microbatch 的 backward 完成后、optimizer step 之前调用。处理以下特殊情况:

### 5.2 Shared Word Embedding 梯度同步

**问题**: 当 `share_embeddings_and_output_weights=True` 时，input embedding (PP stage 0) 和 output layer (PP last stage) 共享权重。两个 stage 各自计算了梯度，需要合并。

**实现** (finalize_model_grads.py:186-284):
```python
def _allreduce_word_embedding_grads(model, config, embd_group, pp_group):
    # embd_group: 包含 PP first stage + PP last stage 的 process group
    if is_pp_first_stage(pp_group) or is_pp_last_stage(pp_group):
        weight = model_module.shared_embedding_or_output_weight()
        grad = weight.main_grad
        torch.distributed.all_reduce(grad, group=embd_group)
```

**FlagScale 扩展: Partial Reduce** (finalize_model_grads.py:322-356):
当 `use_partial_reduce_for_shared_embedding=True` 时，只 reduce 本 DP rank 负责的参数 shard（配合 DistributedOptimizer）:
```python
if ddp_config.use_partial_reduce_for_shared_embedding:
    per_partition_size = grad.shape[0] // dp_world_size
    offset = per_partition_size * dp_rank
    # 只 reduce 本 rank 负责的 shard
    torch.distributed.all_reduce(grad[offset:offset+per_partition_size, :], group=embd_group)
```

### 5.3 Sequence Parallel 梯度求和

**问题**: SP 模式下，LayerNorm 的 weight/bias 的梯度在每个 TP rank 上只来自 S/T 个 token。需要跨 TP group all-reduce。

```python
# finalize_model_grads.py:463-467
if config.sequence_parallel and getattr(param, "sequence_parallel", False):
    # param 标记了 sequence_parallel=True (如 LayerNorm weight)
    grads_sum.append(grad)  # 收集, 后续 coalesced all-reduce over tp_group
```

### 5.4 Per-Token Loss 归一化

```python
# finalize_model_grads.py:593-606
if num_tokens is not None:
    # num_tokens 只在 PP last stage 有值 → broadcast to all PP stages
    last_rank = get_pp_last_rank(pp_group)
    torch.distributed.broadcast(num_tokens, src=last_rank, group=pp_group)
    # all-reduce across DP ranks (total tokens in global batch)
    torch.distributed.all_reduce(num_tokens, group=dp_cp_group)
    # 缩放所有梯度
    for model_chunk in model:
        model_chunk.scale_gradients(1.0 / num_tokens)
```

### 5.5 完整调用序列

```python
def finalize_model_grads(model, num_tokens=None):
    # 1. SP 参数梯度 all-reduce (跨 TP group)
    _allreduce_layernorm_grads(model, tp_group)

    # 2. Conditional Embedding 梯度 all-reduce (跨 PP group, 如 diffusion)
    _allreduce_conditional_embedding_grads(model, config, pp_group)

    # 3. Shared Word Embedding 梯度 all-reduce (PP first ↔ PP last)
    _allreduce_word_embedding_grads(model, config, embd_group, pp_group)

    # 4. Position Embedding 梯度 all-reduce (encoder ↔ decoder)
    _allreduce_position_embedding_grads(model, config, pos_emb_group, pp_group)

    # 5. MoE Router Expert Bias 更新
    if config.moe_router_enable_expert_bias:
        _update_router_expert_bias(model, config)

    # 6. 重置临时张量 (global aux loss tracker 等)
    reset_model_temporary_tensors(config, model)

    # 7. Per-token loss 归一化
    if num_tokens is not None:
        broadcast(num_tokens) → all_reduce(num_tokens) → scale_gradients(1/num_tokens)
```

---

## 6. 内存分析

### 6.1 各模式内存对比 (Qwen3-10B, bf16, Adam)

假设每 rank 持有 `P` 个参数 (已经过 TP/PP 切分):

| 组件 | 传统 DP | Distributed Optimizer (DP=D) |
|------|---------|------------------------------|
| 参数 (bf16) | P × 2B | P × 2B |
| 梯度 (bf16) | P × 2B | P × 2B / D (reduce-scatter 后) |
| FP32 Master Weight | P × 4B | P × 4B / D |
| Adam Momentum | P × 4B | P × 4B / D |
| Adam Variance | P × 4B | P × 4B / D |
| **总计** | P × 16B | P × (2 + 14/D) B |

### 6.2 实际配置计算

**Qwen3-10B (TP=2, PP=2, DP=2)**:
```
P = 10B / (TP × PP) = 2.5B params per rank

传统 DP:   2.5B × 16B = 40 GB
DistOpt:   2.5B × (2 + 14/2)B = 2.5B × 9B = 22.5 GB
节省:      17.5 GB per rank (43.8%)

DP=4 (TP=2, PP=1):
P = 5B
DistOpt:   5B × (2 + 14/4)B = 5B × 5.5B = 27.5 GB
```

### 6.3 额外内存开销

| 额外项 | 大小 | 触发条件 |
|--------|------|---------|
| overlap_param_gather buffer | P × 2B | overlap_param_gather=True |
| grad_reduce_in_fp32 | P × 2B (额外 FP32 grad) | grad_reduce_in_fp32=True |
| fp8_param_gather | ~0 (节省) | fp8_param_gather=True |
| Bucket group metadata | ~negligible | 始终 |

---

## 7. 通信量分析

### 7.1 每步通信总量 (Ring Algorithm)

| 模式 | Backward 通信 | Optimizer→Forward 通信 | 总计 |
|------|-------------|----------------------|------|
| 传统 DP (all-reduce) | 2(D-1)/D × P×dtype | 0 | ~2P×dtype |
| DistOpt (RS+AG) | (D-1)/D × P×dtype (RS) | (D-1)/D × P×dtype (AG) | ~2P×dtype |

**总通信量相同!** DistOpt 的优势纯粹在内存，不在通信。

### 7.2 Qwen3-10B 实例 (TP=2, PP=2, DP=2, bf16)

```
P = 2.5B, dtype = 2B
RS: (2-1)/2 × 2.5B × 2B = 2.5 GB
AG: (2-1)/2 × 2.5B × 2B = 2.5 GB
Total: 5 GB per step

NVLink 900 GB/s: 5 GB / 900 = 5.6 ms (理论)
实际: 大部分被 overlap 隐藏
  - RS 被 backward compute 隐藏
  - AG 被 forward compute 隐藏
暴露时间: 最后一个 bucket 的 RS + 第一层 forward 前的 AG wait
```

### 7.3 跨节点场景 (DP 跨 IB)

```
IB HDR 200 Gb/s = 25 GB/s
DP=8 (跨 8 节点), P = 2.5B:
RS: 7/8 × 2.5B × 2B = 4.375 GB → 4.375/25 = 175 ms (无 overlap)
有 overlap: 只暴露最后 bucket → ~175ms × (bucket_size/total_params)
bucket=40M / total=2.5B → 暴露 ~2.8 ms
```

---

## 8. Num Distributed Optimizer Instances

### 8.1 原理 (ddp_config:num_distributed_optimizer_instances > 1)

将 DP group 拆分为多个子组，每个子组独立执行 ZeRO:

```
DP=8, num_instances=2:
  Instance 0: rank [0,1,2,3] → ZeRO 在 4 个 rank 间分片
  Instance 1: rank [4,5,6,7] → ZeRO 在 4 个 rank 间分片
  跨 instance: inter_dist_opt_group [rank0, rank4] 负责 extra all-reduce
```

**用途**: 当 DP group 很大 (如 64) 时，reduce-scatter 的 ring 延迟过高。分成多个小 instance 可以:
1. 减少 ring 大小 → 降低延迟
2. 代价: 内存只节省 1/instance_size (而非 1/full_dp_size)

### 8.2 通信模式

```
每个 instance 内: reduce-scatter (gradient) + all-gather (params)
跨 instance:     all-reduce (确保 gradient 一致)
```

---

## 9. 关键配置参数

| 参数 | 默认 | 说明 | 性能影响 |
|------|------|------|----------|
| `use_distributed_optimizer` | False | 启用 ZeRO-1+2 | 内存 ÷ DP, 无速度损失 |
| `overlap_grad_reduce` | False | Backward RS/AR overlap | 隐藏 ~90% grad 通信 |
| `overlap_param_gather` | False | Forward AG overlap | 隐藏 ~90% param 通信 |
| `align_param_gather` | False | 所有 PP stage 同步 AG | 减少 straggler |
| `bucket_size` | auto | Bucket 参数数 | 太小→latency, 太大→overlap 差 |
| `grad_reduce_in_fp32` | False | FP32 通信 | 2× 通信量, 更高精度 |
| `average_in_collective` | False | Reduce op = AVG | 避免 pre-scaling |
| `num_distributed_optimizer_instances` | 1 | ZeRO 分片域 | >1 时部分 ZeRO |
| `fp8_param_gather` | False | FP8 all-gather | 通信量 ÷ 2 |
| `use_partial_reduce_for_shared_embedding` | False | 部分 embedding reduce | FlagScale: 异构训练 |
| `delay_wgrad_compute` | False | 延迟 wgrad | 与 overlap 配合 |

---

## 10. FlagScale 扩展

### 10.1 Engram Embedding (ddp.py:94-96, 121-144, 338-351)

FlagScale 新增第三类参数 `engram_embedding_params`:
- 标记: `param.is_engram_embedding = True`
- 独立 DP group: `engram_dp_group` (可能与标准 dp_group 不同)
- 独立 buffer 和 bucket group
- 用途: Engram 模型的 embedding 参数需要在特定 rank 间同步

### 10.2 Platform Abstraction (ddp.py:20-24, 698-722)

```python
from megatron.plugin.platform import get_platform
cur_platform = get_platform()

# 设备相关操作全部通过 platform 抽象
cur_platform.current_device()     # 替代 torch.cuda.current_device()
cur_platform.synchronize()        # 替代 torch.cuda.synchronize()
cur_platform.empty_cache()        # 替代 torch.cuda.empty_cache()
cur_platform.Stream()             # 替代 torch.cuda.Stream()
```

### 10.3 Partial Reduce for Shared Embedding (finalize_model_grads.py:322-356)

当 embd_group 为 list 时 (FlagScale 多 PP group 场景)，支持:
- 非 DistOpt: 在每个 embd_group 内独立 all-reduce
- DistOpt: 只 reduce 本 DP rank 负责的 shard (减少通信量)

---

## 11. 组合约束矩阵

| 组合 | 支持 | 约束/说明 |
|------|:----:|---------|
| DistOpt + overlap_grad_reduce | ✅ | 最佳组合: RS overlap backward |
| DistOpt + overlap_param_gather | ✅ | AG overlap forward, +内存开销 |
| DistOpt + PP | ✅ | PP rank > 0 不分 bucket |
| DistOpt + TP+SP | ✅ | TP 的 SP reduce-scatter 与 DP 独立 |
| DistOpt + CP | ✅ | 使用 dp_cp_group 联合通信 |
| overlap_grad_reduce + PP rank>0 | ⚠️ | 自动禁用 bucketing (非 critical path) |
| bucket_size 过小 + 大 DP | ❌ | ring message 太小 → latency-bound |
| grad_reduce_in_fp32 + fp8_param_gather | ⚠️ | 混合精度: 确保数值稳定 |
| num_dist_opt_instances > 1 + !use_distributed_optimizer | ❌ | 前者依赖后者 |
| overlap_param_gather + CUDA Graph | ⚠️ | pre-hook 在 graph capture 时跳过 |

---

## 12. 设计决策与 Trade-off

| 决策点 | 选项 A | 选项 B | Megatron 选择 |
|--------|--------|--------|--------------|
| DDP 实现 | PyTorch DDP | 自研 | 自研 (连续 buffer + ZeRO 集成) |
| 梯度存储 | param.grad (分散) | 连续 buffer (main_grad) | 连续 buffer (零碎片) |
| ZeRO 方式 | DeepSpeed / FSDP | 原生集成 | 原生 (DistributedOptimizer) |
| Overlap 触发 | timer-based | golden count | golden count (自适应) |
| Bucket 策略 | 固定大小 | auto + 动态 | auto (dp_size 相关) |
| Embedding sync | 每步 all-reduce | partial reduce | 两种均支持 |
| 通信精度 | 与 param dtype 同 | FP32 / FP8 可选 | 可配置 |

---

## 13. 性能建议

1. **必须启用 `use_distributed_optimizer: True`**: 内存节省 50-76%，无速度损失
2. **必须启用 `overlap_grad_reduce: True`**: 隐藏 backward 通信，几乎零暴露
3. **推荐启用 `overlap_param_gather: True`**: 隐藏 forward 参数恢复 (代价: ~P×2B 额外内存)
4. **Bucket size**: 使用默认 auto，不要手动减小。大 DP 时默认值自动增大
5. **DP degree 选择**: 在满足内存约束下尽量大。TP×PP×DP = total_gpus
6. **跨节点 DP**: DP 通信走 IB/RoCE，确保 overlap 充分 (bucket 不能太大)
7. **grad_reduce_in_fp32**: 仅在观察到精度问题时启用 (2× 通信代价)
8. **fp8_param_gather**: H100/H200 上可尝试，通信量减半
