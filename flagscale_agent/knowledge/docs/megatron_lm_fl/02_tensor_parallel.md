# 02 - Tensor Parallelism (TP) & Sequence Parallelism (SP) 完整分析

## 源码位置

| 文件 | 行数 | 功能 |
|------|------|------|
| `megatron/core/tensor_parallel/layers.py` | 1369 | ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding |
| `megatron/core/tensor_parallel/mappings.py` | 602 | 通信原语 (autograd Function 封装的 all-reduce/gather/scatter) |
| `megatron/core/tensor_parallel/cross_entropy.py` | 232 | 分布式 Vocab Parallel Cross Entropy |
| `megatron/core/tensor_parallel/random.py` | — | TP-aware CUDA RNG 状态管理 |
| `megatron/core/tensor_parallel/data.py` | — | 数据分发工具 (broadcast to TP group) |
| `megatron/core/parallel_state.py` | — | Process group 创建与管理 |

---

## 1. TP 切分原理

Tensor Parallelism 将每一层的权重矩阵切分到多个 GPU，通过在 forward/backward 中插入集合通信来保证数学等价性。

### 1.1 Column Parallel Linear (layers.py:776-1116)

**数学定义**: `Y = XA + b`, 其中 A 按列切分: `A = [A_0 | A_1 | ... | A_{T-1}]`

**权重形状** (每 rank):
```python
self.weight = Parameter(shape=[output_size_per_partition, input_size])
# output_size_per_partition = output_size / tp_size
# weight 存储为转置形式 (PyTorch convention: F.linear does XW^T)
```

**Forward 数据流** (layers.py:980-1077):
```
所有 rank 持有:  input [S, B, H]
                    │
            ┌───────┴───────┐  (如果非 SP/allreduce 模式: copy_to_tp_region)
            │   identity    │  (如果 SP/allreduce 模式: input 已在各 rank)
            └───────┬───────┘
                    │
     ┌──────────────┼──────────────┐
     │ Rank 0       │ Rank 1       │  ... Rank T-1
     │ W₀[H/T, H]  │ W₁[H/T, H]  │  W_{T-1}[H/T, H]
     │ Y₀ = X·W₀ᵀ  │ Y₁ = X·W₁ᵀ  │  Y_{T-1} = X·W_{T-1}ᵀ
     │ [S, B, H/T]  │ [S, B, H/T]  │  [S, B, H/T]
     └──────────────┴──────────────┘
                    │
            ┌───────┴───────┐
            │ gather_output │  if True: all-gather → [S, B, H]
            │   (optional)  │  if False: keep [S, B, H/T] 传给 RowParallel
            └───────────────┘
```

**Backward 数据流** (layers.py:497-650):
```
grad_output: [S, B, H/T]  (如果 gather_output=False, 即 RowParallel 的输入)
     │
     ├── grad_input = grad_output · W  → [S, B, H]
     │       │
     │       ├── if allreduce_dgrad: async all-reduce(grad_input)
     │       └── if sequence_parallel: async reduce-scatter(grad_input) → [S/T, B, H]
     │
     └── grad_weight = grad_outputᵀ · total_input  (与通信 overlap)
```

**关键设计** (layers.py:945-964):
```python
# allreduce_dgrad 和 sequence_parallel 互斥
self.allreduce_dgrad = (world_size > 1 and not self.sequence_parallel and not self.disable_grad_reduce)
if self.allreduce_dgrad and self.sequence_parallel:
    raise RuntimeError("`allreduce_dgrad` and `sequence_parallel` cannot be enabled at the same time.")
```

**gather_output 使用场景**:
- `True`: 最终输出层（logits 需要完整 vocab 维度）
- `False`: 中间层（输出直接作为 RowParallelLinear 的输入，保持分片）

---

### 1.2 Row Parallel Linear (layers.py:1118-1369)

**数学定义**: `Y = XA + b`, 其中 X 按列切分、A 按行切分:
```
X = [X_0 | X_1 | ... | X_{T-1}]
A = [A_0; A_1; ...; A_{T-1}]  (行切分)
Y = Σ X_i · A_i  (各 rank 的 partial sum 需要 reduce)
```

**权重形状** (每 rank):
```python
self.weight = Parameter(shape=[output_size, input_size_per_partition])
# input_size_per_partition = input_size / tp_size
# partition_dim = 1 (沿第 1 维切分)
```

**Forward 数据流** (layers.py:1275-1333):
```
Rank i 持有: input_parallel [S, B, H/T] (来自 ColumnParallel 的输出)
                    │
     ┌──────────────┼──────────────┐
     │ Rank 0       │ Rank 1       │
     │ Y₀ = X₀·A₀  │ Y₁ = X₁·A₁  │  partial results [S, B, H]
     └──────────────┴──────────────┘
                    │
            ┌───────┴───────┐
            │   if SP:      │  reduce-scatter → [S/T, B, H]
            │   else:       │  all-reduce → [S, B, H]
            └───────────────┘
```

**input_is_parallel 参数** (line 1133):
- `True` (默认): 假设输入已按 TP 切分（来自 ColumnParallel 的 H/T 输出）
- `False`: 输入是完整的，需要先 scatter 到各 rank

---

### 1.3 Vocab Parallel Embedding (layers.py:207-430)

**切分方式**: 词表按行切分, 每个 rank 负责 `vocab[rank*V/T : (rank+1)*V/T]`

**Forward 流程**:
```python
# 1. 判断 token 是否属于本 rank
mask = (input_ids >= vocab_start) & (input_ids < vocab_end)
masked_input = input_ids - vocab_start  # 本地索引
masked_input[~mask] = 0  # 不属于本 rank 的置 0

# 2. Embedding lookup (本地)
output = F.embedding(masked_input, self.weight)
output[~mask] = 0.0  # 不属于本 rank 的输出置 0

# 3. 跨 rank 合并
if reduce_scatter_embeddings:  # SP 模式
    output = reduce_scatter(output)  # → [S/T, B, H]
else:
    output = all_reduce(output)      # → [S, B, H]
```

**设计动机**: 大词表 (如 Qwen3 的 151936) 时，embedding 权重矩阵 `[V, H]` 可能占数 GB，TP 切分减少每 GPU 内存。

---

## 2. Sequence Parallelism (SP)

### 2.1 原理 (arXiv:2205.05198, Reducing Activation Recomputation in Large Transformer Models)

**问题**: 标准 TP 中，LayerNorm 和 Dropout 等非 TP 操作在所有 rank 上持有完整的 `[S, B, H]` activation，浪费内存。

**解决**: SP 在非 TP 区域沿 sequence 维度切分，每 rank 只持有 `[S/T, B, H]`:
- 进入 TP 区域: all-gather (恢复完整 S)
- 离开 TP 区域: reduce-scatter (切回 S/T, 同时完成 reduce)

**内存节省**: 非 TP 区域 activation 减少为 1/T

### 2.2 完整 Transformer Layer 数据流 (TP + SP)

```
┌────────────────────────────────────────────────────────────────────────┐
│ 每 rank 持有 [S/T, B, H] (SP 切分后)                                   │
│                                                                        │
│ ┌─────────────┐                                                       │
│ │ LayerNorm   │ [S/T, B, H]  ← 本地计算，无通信                        │
│ └──────┬──────┘                                                       │
│        │ all-gather along S dim → [S, B, H]                           │
│ ┌──────┴──────┐                                                       │
│ │ QKV Linear  │ ColumnParallel: [S, B, H] → [S, B, 3H/T]            │
│ │ (Column TP) │ 无通信 (已有完整 input)                                │
│ └──────┬──────┘                                                       │
│        │                                                              │
│ ┌──────┴──────┐                                                       │
│ │ Attention   │ [S, B, H/T] (每 rank 独立计算自己的 heads)             │
│ └──────┬──────┘                                                       │
│        │                                                              │
│ ┌──────┴──────┐                                                       │
│ │  O Linear   │ RowParallel: [S, B, H/T] → reduce-scatter → [S/T,B,H]│
│ │  (Row TP)   │                                                       │
│ └──────┬──────┘                                                       │
│        │ + residual ([S/T, B, H])                                     │
│ ┌──────┴──────┐                                                       │
│ │ LayerNorm   │ [S/T, B, H]  ← 本地计算                               │
│ └──────┬──────┘                                                       │
│        │ all-gather along S dim → [S, B, H]                           │
│ ┌──────┴──────┐                                                       │
│ │ Gate+Up     │ ColumnParallel: [S, B, H] → [S, B, FFN/T]            │
│ │ (Column TP) │                                                       │
│ └──────┬──────┘                                                       │
│        │                                                              │
│ ┌──────┴──────┐                                                       │
│ │ Down Linear │ RowParallel: [S, B, FFN/T] → reduce-scatter → [S/T,B,H]│
│ │ (Row TP)    │                                                       │
│ └──────┬──────┘                                                       │
│        │ + residual ([S/T, B, H])                                     │
│        ↓                                                              │
│ 输出: [S/T, B, H] (继续 SP 模式)                                      │
└────────────────────────────────────────────────────────────────────────┘
```

**每层通信**: 4 次 collective
- 2 × all-gather (进入 Attention TP, 进入 MLP TP)
- 2 × reduce-scatter (退出 Attention TP, 退出 MLP TP)

### 2.3 通信原语实现 (mappings.py)

每个通信原语封装为 `torch.autograd.Function`，在 forward 和 backward 中插入对偶通信:

| 类名 | Forward 操作 | Backward 操作 | 典型用途 |
|------|-------------|--------------|---------|
| `_CopyToModelParallelRegion` (line 203) | identity | all-reduce | 进入 TP: 复制 input, backward 时 reduce grad |
| `_ReduceFromModelParallelRegion` (line 223) | all-reduce | identity | 退出 TP (无 SP): reduce output |
| `_ScatterToSequenceParallelRegion` (line 282) | split along dim0 | gather along dim0 | 进入 SP: 切分 sequence |
| `_GatherFromSequenceParallelRegion` (line 302) | gather along dim0 | reduce-scatter / split | 离开 SP: 恢复完整 sequence |
| `_ReduceScatterToSequenceParallelRegion` (line 357) | reduce-scatter dim0 | gather along dim0 | 退出 TP+SP: reduce 且切分 |
| `_AllGatherFromTensorParallelRegion` (line 386) | gather along last-dim | reduce-scatter last-dim | TP 维度 gather |

**对偶性设计**: forward 做 gather 的原语，backward 自动做 scatter (反之亦然)。这保证了 autograd 链正确。

```python
# 例: _GatherFromSequenceParallelRegion (line 302-354)
class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, group, tensor_parallel_output_grad=True, ...):
        return _gather_along_first_dim(input_, group)  # S/T → S

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.tensor_parallel_output_grad:
            # 后续是 TP 计算 → 需要 reduce-scatter (而非 split)
            return _reduce_scatter_along_first_dim(grad_output, ctx.group)
        else:
            # 后续是复制计算 → 简单 split 即可
            return _split_along_first_dim(grad_output, ctx.group)
```

**`tensor_parallel_output_grad` 的意义** (line 342): 决定 backward 时用 reduce-scatter 还是 split。当 gather 的输出进入 TP 区域时，backward 的 grad 来自多个 rank 的部分计算，需要 reduce-scatter；当输出进入非 TP 区域时，grad 已经是完整的，只需 split。

---

## 3. 异步通信与计算重叠 (layers.py:443-650)

### 3.1 核心类: LinearWithGradAccumulationAndAsyncCommunication

这是 TP 性能优化的核心。它在 backward 中实现:
1. **dgrad 通信** (reduce-scatter / all-reduce) 与 **wgrad 计算** 的 overlap
2. **SP 的 all-gather** (backward 中需要恢复完整 input 来计算 wgrad) 与 **dgrad 计算** 的 overlap

### 3.2 Forward 路径 (layers.py:480-493)

```python
def forward(ctx, input, weight, bias, gradient_accumulation_fusion,
            allreduce_dgrad, sequence_parallel, grad_output_buffer,
            wgrad_deferral_limit, tp_group, te_fl_prefer):
    # 保存用于 backward
    ctx.save_for_backward(input, weight)

    if sequence_parallel:
        # SP 模式: all-gather input 恢复完整 sequence
        dim_size[0] = dim_size[0] * tp_group.size()  # S/T → S
        all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, dtype, "mpu")
        dist_all_gather_func(all_gather_buffer, input, group=tp_group)
        total_input = all_gather_buffer
    else:
        total_input = input

    output = torch.matmul(total_input, weight.t())  # [S, B, H] × [H, H/T]ᵀ → [S, B, H/T]
    return output
```

### 3.3 Backward 路径 — 三阶段 overlap (layers.py:497-650)

**时序图** (sequence_parallel=True):

```
Timeline →
──────────────────────────────────────────────────────────────────
Phase 1: 计算 dgrad + 发起 all-gather (为 wgrad 准备)
  comp_stream: │ dgrad = grad_output · W  │
  comm_stream: │ all-gather(input, S/T→S) │  ← async, 与 dgrad 并行
               ├─────── overlap ───────────┤

Phase 2: 发起 dgrad reduce-scatter (异步)
  comm_stream: │ reduce-scatter(dgrad, S→S/T) │  ← async
               └──────────────────────────────┘

Phase 3: 计算 wgrad (与 reduce-scatter overlap)
  comp_stream: │ wgrad = grad_outputᵀ · total_input │
               ├────────── overlap ──────────────────┤
  comm_stream: │        reduce-scatter 继续...       │

Phase 4: wait (确保 reduce-scatter 完成)
  return sub_grad_input  ← [S/T, B, H]
──────────────────────────────────────────────────────────────────
```

**关键代码** (layers.py:517-566, 简化):
```python
def backward(ctx, grad_output):
    input, weight = ctx.saved_tensors

    # Phase 1: async all-gather for wgrad computation
    if ctx.sequence_parallel and wgrad_compute:
        handle = dist_all_gather_func(all_gather_buffer, input, group=tp_group, async_op=True)
        total_input = all_gather_buffer

    # Phase 1: dgrad computation (与 all-gather 并行)
    grad_input = grad_output.matmul(weight)  # [S, B, H/T] × [H/T, H] → [S, B, H]

    # Wait for all-gather (wgrad 需要完整 input)
    if ctx.sequence_parallel and wgrad_compute:
        handle.wait()

    # Phase 2: async reduce-scatter on dgrad
    if ctx.sequence_parallel:
        sub_grad_input = torch.empty([S/T, B, H])
        handle = dist_reduce_scatter_func(sub_grad_input, grad_input, group=tp_group, async_op=True)
        # ↑ "Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
        #    reduce scatter is scheduled before the weight gradient computation"

    # Phase 3: wgrad computation (与 reduce-scatter 并行!)
    if ctx.gradient_accumulation_fusion:
        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(total_input, grad_output, weight.main_grad)
    else:
        grad_weight = grad_output.t().matmul(total_input)

    # Phase 4: wait for reduce-scatter
    if ctx.sequence_parallel:
        handle.wait()
    return sub_grad_input, grad_weight, grad_bias, ...
```

### 3.4 CUDA_DEVICE_MAX_CONNECTIONS=1 的必要性

**为什么需要**: PyTorch 的 async collective 和 matmul 默认可能在同一个 CUDA stream 上执行（如果有多个 connection）。设置 `MAX_CONNECTIONS=1` 强制 NCCL 使用独立 stream，确保:
1. async collective 被调度到 NCCL stream
2. matmul 被调度到 compute stream
3. 两者自然并行（不同 stream 间无依赖）

**没有此设置的后果**: async_op=True 可能退化为同步，通信与计算串行。

### 3.5 Gradient Accumulation Fusion (layers.py:568-599)

**作用**: 使用 CUDA fused kernel 将 wgrad GEMM 结果直接累加到 `weight.main_grad` (FP32)，避免:
1. 分配 intermediate grad_weight tensor
2. 额外的 cast + add 操作

**FlagScale 扩展** (layers.py:576-599):
```python
if not (ctx.te_fl_prefer == 'flagos' or ctx.te_fl_prefer == 'reference'):
    # 使用 fused CUDA kernel (NVIDIA APEX)
    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(total_input, grad_output, weight.main_grad)
else:
    # FlagScale fallback: 纯 PyTorch (兼容非 NVIDIA 硬件)
    grad_weight = torch.matmul(grad_output.t(), total_input)
    weight.main_grad += grad_weight.view_as(weight.main_grad)
```

### 3.6 Wgrad Deferral (layers.py:511-514)

**作用**: 延迟 wgrad 计算到更晚的时间点（用于 embedding wgrad 与其他操作重叠）:
```python
if grad_output_buffer is not None:
    if wgrad_deferral_limit == 0 or len(grad_output_buffer) < wgrad_deferral_limit:
        grad_output_buffer.append(grad_output)  # 存储 grad，稍后统一计算
        wgrad_compute = False  # 跳过本次 wgrad
```

---

## 4. 分布式 Cross Entropy (cross_entropy.py)

### 4.1 问题

当 vocab 按 TP 切分时，logits 为 `[S, B, V/T]`（每 rank 只有部分 vocab 的 logits）。标准 softmax 需要全 vocab 的 max 和 exp-sum，无法直接本地计算。

### 4.2 算法 (VocabParallelCrossEntropy)

**Forward** (_VocabParallelCrossEntropy.forward, line 122-189):

```python
# Step 1: 数值稳定 — 全局 max (跨所有 rank)
logits_max = max(local_logits, dim=-1)
all_reduce(logits_max, op=MAX)  # ← 通信 1: 获取全局最大值
logits -= logits_max  # in-place subtract 防止 overflow

# Step 2: 获取 target 对应的 logit (本地索引)
target_mask = (target < vocab_start) | (target >= vocab_end)
masked_target = target - vocab_start
predicted_logits = logits[arange, masked_target]
predicted_logits[target_mask] = 0.0  # 不属于本 rank 的 target 置 0
all_reduce(predicted_logits, op=SUM)  # ← 通信 2: 合并 target logit

# Step 3: 全局 exp-sum
exp_logits = exp(logits)
sum_exp = exp_logits.sum(dim=-1)
all_reduce(sum_exp, op=SUM)  # ← 通信 3: 合并 exp-sum

# Step 4: 计算 loss
loss = log(sum_exp) - predicted_logits
```

**通信量**: 3 次 all-reduce (logits_max, predicted_logits, sum_exp)，每次 `[S, B]` 大小 — 远小于 logits 本身。

**Backward** (line 191-216): 各 rank 独立计算本地 softmax gradient，无额外通信:
```python
# grad = softmax - one_hot(target)
grad_2d[arange, masked_target] -= 1.0  # 仅本 rank 负责的 target
grad_input *= grad_output  # chain rule
```

### 4.3 Label Smoothing 支持 (line 165-182)

```python
smoothing = label_smoothing * vocab_size / (vocab_size - 1)
log_probs = torch.log(exp_logits)  # 本地 log-probs (V/T 维度)
mean_log_probs = log_probs.mean(dim=-1)  # 本地均值
loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs
```
注意: `mean_log_probs` 是本地的 (V/T 维度的 mean)，需要后续 all-reduce 才是全局均值。但代码中 mean 已乘以 smoothing 系数，数学上等价于对全局 mean 的近似。

---

## 5. 关键配置参数

| 参数 | 默认值 | 说明 | 影响 |
|------|--------|------|------|
| `tensor_model_parallel_size` | 1 | TP degree | 权重切分粒度 |
| `sequence_parallel` | False | 启用 SP | 非 TP 区域 activation 减少 1/T |
| `async_tensor_model_parallel_allreduce` | True | 异步 all-reduce | dgrad-wgrad overlap |
| `gradient_accumulation_fusion` | True | Fused wgrad GEMM | 省内存+加速 |
| `te_fl_prefer` | None | FlagScale: wgrad 路径选择 | 'flagos'/'reference' 用 PyTorch |
| `overlap_grad_reduce` | False | DP grad reduce 与 compute overlap | 独立于 TP |
| `overlap_param_gather` | False | ZeRO param gather overlap | 独立于 TP |

---

## 6. TP 通信量分析

### 6.1 每层通信 (单个 Transformer Layer)

| 场景 | Forward 通信 | Backward 通信 | 总通信量 |
|------|-------------|--------------|---------|
| TP (无 SP) | 2 × all-reduce [S,B,H] | 2 × all-reduce [S,B,H] | 4 × 2(T-1)/T × S×B×H×dtype |
| TP + SP | 2 × all-gather [S/T→S] + 2 × reduce-scatter [S→S/T] | 同 forward (对偶) | 4 × 2(T-1)/T × S×B×H×dtype |

注: all-reduce 和 all-gather+reduce-scatter 的总通信量相同 (ring algorithm 下都是 `2(T-1)/T × data_size`)。SP 的优势在于**内存**而非通信量。

### 6.2 Qwen3-10B 实例 (H=4096, S=4096, B=1, TP=2, bf16)

```
单次 collective 数据量: S × B × H × 2 bytes = 4096 × 1 × 4096 × 2 = 32 MB
Ring algorithm 通信量: 2 × (T-1)/T × 32MB = 2 × 0.5 × 32MB = 32 MB per collective
每层 4 次 collective: 4 × 32 = 128 MB
48 层总计: 48 × 128 = 6.1 GB per iteration

NVLink 900 GB/s (双向): 6.1 GB / 900 GB/s ≈ 6.8 ms (理论下界)
实际 (含 overlap): 大部分被 wgrad 计算隐藏
```

### 6.3 内存节省对比 (SP vs 无 SP)

| 区域 | 无 SP | 有 SP | 节省 |
|------|-------|-------|------|
| LayerNorm activation | [S, B, H] per rank | [S/T, B, H] per rank | (T-1)/T |
| Dropout mask | [S, B, H] | [S/T, B, H] | (T-1)/T |
| Residual connection | [S, B, H] | [S/T, B, H] | (T-1)/T |
| TP 区域 (QKV, MLP) | [S, B, H/T] | [S, B, H/T] | 0 (相同) |

TP=2 时: 非 TP 区域 activation 减少 50%

---

## 7. FlagScale 扩展

### 7.1 Platform Abstraction (layers.py:68-72)

```python
from megatron.plugin.platform import get_platform
cur_platform = get_platform()
```

所有 device allocation (`torch.empty(..., device=...)`) 改为 `cur_platform.current_device()`，使 TP 实现可在非 NVIDIA 硬件 (如华为 Ascend NPU) 上运行。

### 7.2 te_fl_prefer wgrad 路径 (layers.py:576-599)

FlagScale 在 wgrad 计算处添加了分支:
- **NVIDIA (默认)**: 使用 `fused_weight_gradient_mlp_cuda` APEX kernel — 高性能
- **'flagos' / 'reference'**: 使用纯 PyTorch `torch.matmul` — 兼容所有硬件

### 7.3 AMP custom_fwd/custom_bwd 适配 (layers.py:74-87)

```python
try:
    if is_torch_min_version("2.4.0a0"):
        custom_fwd = partial(torch.amp.custom_fwd, device_type="cuda")
    else:
        custom_fwd = cur_platform.amp.custom_fwd  # FlagScale: 平台适配
except:
    custom_fwd = cur_platform.amp.custom_fwd
```

---

## 8. 组合约束矩阵

| 组合 | 支持 | 约束/说明 |
|------|:----:|---------|
| TP + SP | ✅ | 推荐组合: SP 减少 activation 内存 |
| TP + PP | ✅ | TP group 在 node 内, PP 跨 node |
| TP + DP | ✅ | world = TP × PP × DP |
| TP=1 + SP | ❌ | SP 需要 TP>1 (layers.py:938-943 自动禁用) |
| allreduce_dgrad + SP | ❌ | 互斥 (layers.py:961-964 报错) |
| TP + CUDA_DEVICE_MAX_CONNECTIONS≠1 | ⚠️ | 失去 async overlap, 性能退化 |
| TP + overlap_grad_reduce | ✅ | DP grad overlap 与 TP 正交 |
| TP + EP | ✅ | Expert TP 独立于 model TP (expert_tp 属性) |

---

## 9. 设计决策与 Trade-off

| 决策点 | 选项 A | 选项 B | Megatron 选择 |
|--------|--------|--------|--------------|
| TP 通信方式 | all-reduce (简单) | all-gather + reduce-scatter (SP) | 提供两种, 推荐 SP |
| wgrad 与通信 overlap | 显式 stream 管理 | CUDA_DEVICE_MAX_CONNECTIONS=1 | 后者 (隐式, 依赖 CUDA 调度) |
| wgrad 精度 | FP16/BF16 | FP32 accumulation | FP32 (gradient_accumulation_fusion) |
| Cross Entropy | 先 all-gather logits 再本地 CE | 分布式 CE (仅 reduce scalar) | 分布式 (通信量 O(S×B) vs O(S×B×V)) |
| Embedding 后的 reduce | all-reduce | reduce-scatter (SP) | 可选, SP 时 reduce-scatter |

---

## 10. 性能建议

1. **TP 应限制在 NVLink 域内** (通常 ≤8 GPU): 跨节点 TP 受 IB/RoCE 带宽限制
2. **TP>1 时必须启用 SP** (`--sequence-parallel`): 减少 (T-1)/T activation 内存, 零额外通信开销
3. **必须设置 `CUDA_DEVICE_MAX_CONNECTIONS=1`**: 否则 dgrad-wgrad overlap 失效
4. **gradient_accumulation_fusion=True**: fused wgrad kernel 省内存且更快
5. **TP degree 选择**: hidden_size % TP == 0 且 num_heads % TP == 0
6. **gather_output 最小化**: 中间层保持 `gather_output=False`，仅最终输出层 gather
7. **TP=2 通常是最优起点**: NVLink 带宽充裕, overhead 低, 可与 PP/DP 灵活组合
