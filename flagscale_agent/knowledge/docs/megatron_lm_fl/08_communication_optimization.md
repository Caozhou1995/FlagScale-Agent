# 第八章：通信优化与 Overlap

## 1. 概述

分布式训练中通信占总时间 20-40%。Megatron-LM-FL 的核心优化思路：**将通信隐藏在计算之下**（overlap），使通信接近零开销。

**源码定位**：

| 组件 | 源码路径 | 职责 |
|------|----------|------|
| DDP Config | `distributed/distributed_data_parallel_config.py` | overlap_grad_reduce/param_gather 开关 |
| Bucket Group | `distributed/param_and_grad_buffer.py` L167-766 | Bucket 通信调度 |
| ParamAndGradBuffer | `distributed/param_and_grad_buffer.py` L768+ | 连续内存 + bucket 划分 |
| P2P 通信 | `pipeline_parallel/p2p_communication.py` | PP stage 间通信 |
| TP Overlap | `tensor_parallel/layers.py` (AsyncComm Linear) | TP all-reduce/scatter overlap |
| MoE Overlap | `transformer/moe/` | Expert A2A 与计算 overlap |
| TE Overlap | `extensions/transformer_engine.py` | Fused AG+GEMM / GEMM+RS |

**通信类型全景**：

```
┌──────────────────────────────────────────────────────────────────────┐
│                 Distributed Training Communication Map                 │
├──────────────┬──────────────────┬───────────────────┬────────────────┤
│  并行维度    │  通信原语        │  触发时机         │  Overlap 方式  │
├──────────────┼──────────────────┼───────────────────┼────────────────┤
│  DP          │  reduce-scatter  │  Backward (grad)  │  逐bucket异步  │
│  DP          │  all-gather      │  Forward (param)  │  预取下层参数  │
│  TP          │  all-gather      │  Forward (SP)     │  与GEMM fuse   │
│  TP          │  reduce-scatter  │  Forward (SP)     │  与GEMM fuse   │
│  TP          │  all-reduce      │  Backward (grad)  │  延迟聚合      │
│  PP          │  P2P send/recv   │  Stage 切换       │  与计算overlap │
│  CP          │  ring send/recv  │  Attention 每步   │  与QKV计算重叠 │
│  EP          │  all-to-all      │  Expert dispatch  │  与expert计算  │
└──────────────┴──────────────────┴───────────────────┴────────────────┘
```

---

## 2. Gradient Reduce Overlap

### 2.1 配置

```python
# distributed_data_parallel_config.py L16
overlap_grad_reduce: bool = False
# True: backward 计算与 grad reduce 异步重叠
# False: backward 完成后同步 reduce

bucket_size: Optional[int] = None
# 默认: max(40_000_000, 1_000_000 × dp_size)
# 每个 bucket 包含的参数数量
```

### 2.2 Bucket 机制

```
源码: param_and_grad_buffer.py L768+

_ParamAndGradBuffer:
  所有参数的梯度存放在连续 buffer 中
  buffer 按 bucket_size 切分为多个 bucket
  bucket 进一步组合为 BucketGroup

┌─────── ParamAndGradBuffer (连续内存) ──────────────────────┐
│  ┌─ BucketGroup 0 ──┐  ┌─ BucketGroup 1 ──┐               │
│  │ Bucket 0 │ Buck 1│  │ Bucket 2 │ Buck 3│  ...          │
│  │ params.. │ par.. │  │ params.. │ par.. │               │
│  └──────────────────┘  └──────────────────┘               │
└────────────────────────────────────────────────────────────┘

设计动机:
  1. 连续内存 → 单次 NCCL collective 覆盖整个 bucket (高带宽利用)
  2. Bucket 分组 → 部分 grad ready 即可启动通信 (overlap 机会)
  3. BucketGroup → 跨 bucket 合并通信 (用 _coalescing_manager)
```

### 2.3 Grad Ready 注册与异步触发

```
源码: param_and_grad_buffer.py L743-766

register_grad_ready(param):
  │
  ├── 条件: overlap_grad_reduce=True AND is_last_microbatch
  │
  ├── 记录: per_param_grad_ready_counts[param] += 1
  │
  └── 检查: if counts == golden_counts:
              # BucketGroup 内所有参数的 grad 都 ready
              start_grad_sync()  ← 立即启动异步通信

golden_per_param_grad_ready_counts:
  首轮训练 (is_first_batch) 记录每个 param 的 register 次数
  → 作为后续轮次的"完成标准"
  → 支持控制流中同一 param 被多次访问的情况
```

### 2.4 start_grad_sync 实现

```
源码: param_and_grad_buffer.py L527-640

start_grad_sync():
  │
  ├── Step 1: 拷贝 extra main_grad → grad_buffer (如高精度累积)
  │
  ├── Step 2: 检查 NaN/Inf/Large grad (if check_for_nan_in_grad)
  │
  ├── Step 3: gradient_scaling_factor 缩放
  │     grad_data *= gradient_scaling_factor
  │
  ├── Step 4: 选择 reduce_op (SUM or AVG)
  │
  ├── Step 5: Stream 同步
  │     # Compute Stream: ---Gradient compute--->
  │     # Comm. Stream:   --(wait NCCL)---
  │     # NCCL Stream:    ----RS/AR----
  │     if multi_distopt_instances:
  │         communication_stream.wait_stream(current_stream())
  │
  └── Step 6: 执行通信 (用 _coalescing_manager 合并 bucket)
        if use_distributed_optimizer:
            reduce_scatter(local_shard, grad_data, group=dp_group, async_op=True)
        else:
            all_reduce(grad_data, group=dp_group, async_op=True)
```

### 2.5 时序图

```
┌─ Backward Timeline (overlap_grad_reduce=True) ──────────────────────┐
│                                                                       │
│  Compute Stream:                                                      │
│    [Layer N grad]──[Layer N-1 grad]──[Layer N-2 grad]──[...]──done   │
│                                                                       │
│  NCCL Stream:                                                         │
│              ┌─RS bucket2─┐                                          │
│                       ┌─RS bucket1─┐                                 │
│                                ┌─RS bucket0─┐                        │
│                                                                       │
│  同步点: 仅在 optimizer.step() 前 wait 所有 handle                   │
│                                                                       │
│  vs. 无 overlap:                                                      │
│    [Layer N grad]...[Layer 0 grad]──[RS bucket2]─[RS bucket1]─[RS 0] │
│                                     ↑ 这段时间 GPU 空闲              │
└───────────────────────────────────────────────────────────────────────┘
```

### 2.6 Bucket Size 调优

```
bucket_size 的 trade-off:

大 bucket (100M params):
  + 高 NCCL 带宽利用率 (大消息 → 接近峰值带宽)
  - 需要更多 grad ready 才能启动 → overlap 窗口小
  - 极端: 1个bucket = 所有param → 退化为同步 reduce

小 bucket (1M params):
  + 细粒度触发 → overlap 窗口大
  - NCCL kernel launch 多 → 延迟累积
  - 小消息 → 带宽利用率低

经验法则:
  bucket_size = max(40M, 1M × dp_size)
  确保 message_size = bucket_size / dp_size ≥ 几MB
  
NCCL 对齐优化:
  pad_buckets_for_high_nccl_busbw: bool = False
  # 确保 bucket_size 被 2^16 整除
  # NCCL ring 算法对 2 的幂次消息有最佳性能
```

---

## 3. Parameter Gather Overlap

### 3.1 配置

```python
# distributed_data_parallel_config.py L19-22
overlap_param_gather: bool = False
# True: forward 计算与 param all-gather 异步重叠

align_param_gather: bool = False  
# True: 所有 PP stage 同时启动 param gather
# False: 各 stage 按需独立启动
```

### 3.2 前提条件

```
仅在 Distributed Optimizer (ZeRO) 模式有效:
  参数被分片到 DP ranks → forward 前需要 all-gather 恢复

标准路径 (无 overlap):
  all-gather(全部参数) → forward(所有层) → backward → reduce-scatter

Overlap 路径:
  gather(layer 0) → forward(layer 0) ←overlap→ gather(layer 1)
                  → forward(layer 1) ←overlap→ gather(layer 2)
                  → ...
```

### 3.3 实现: start_param_sync

```
源码: param_and_grad_buffer.py L304-437

start_param_sync(force_sync=False):
  │
  ├── Distributed Optimizer 路径 (标准):
  │     with _coalescing_manager(dp_group, async_ops=True):
  │         for bucket in buckets:
  │             local_shard = cached_param_buffer_shard[rank]
  │             all_gather_into_tensor(
  │                 bucket.param_data,    # output: 完整参数
  │                 local_shard,          # input: 本地分片
  │                 group=dp_group,
  │                 async_op=True
  │             )
  │     param_gather_handle = cm  # 异步 handle
  │
  └── Layer-wise Optimizer 路径 (特殊):
        # 各 rank 持有不同数量的参数 → 使用 all_gather (list版本)
        for bucket in buckets:
            gather_list = [empty(size_i) for i in range(dp_size)]
            all_gather(gather_list, flat_local_params, group, async_op=True)
```

### 3.4 Forward 中的调度

```
时序:

Forward Layer 0:
  finish_param_sync(bucket_group_0)   ← 等待 layer 0 参数就绪
  compute(layer_0)                     ← 前向计算
  start_param_sync(bucket_group_1)    ← 异步预取 layer 1 参数

Forward Layer 1:
  finish_param_sync(bucket_group_1)   ← 等 layer 1 gather 完成
  compute(layer_1)
  start_param_sync(bucket_group_2)    ← 预取 layer 2

...以此类推

Backward 时类似: 反序预取参数
```

### 3.5 align_param_gather 设计

```
问题: PP 各 stage 的 layer 数不同, gather 时机不对齐
  → NVLink/IB 带宽争用不均匀

align_param_gather=True:
  所有 PP stage 在相同时间点发起 param gather
  → 网络流量更均匀 → NCCL 性能更可预测
  代价: 某些 stage 可能提前 gather 了暂时不需要的参数

适用: 跨节点 DP (IB带宽有限时，对齐减少拥塞)
```

---

## 4. PP P2P 通信优化

### 4.1 P2P 通信类

```
源码: pipeline_parallel/p2p_communication.py L157

class P2PCommunicator:
  # 管理 Pipeline stage 间的 send/recv

通信原语:
  _batched_p2p_ops(L34): 批量 P2P (减少 kernel launch)
  _p2p_ops(L72): 逐个 P2P
```

### 4.2 batch_p2p_comm

```python
# 配置: config.batch_p2p_comm = True (推荐)

# 机制: 将多个 send/recv 合并为一次 batch 调用
# 例如 1F1B 中同时有 send(activation to next) + recv(grad from next)
# → 合并为单次 batch op → 减少 NCCL kernel launch overhead

# batch_p2p_sync: 是否在 batch P2P 后立即同步
# True: 保证通信完成 (安全但可能阻塞)
# False: 延迟同步到需要数据时 (更好的 overlap)
```

### 4.3 overlap_p2p_comm (FlagScale)

```
源码: p2p_communication.py L675-717

recv_forward(overlap_p2p_comm=True):
    reqs = _batched_p2p_ops(..., wait_on_reqs=False)  # 不等待
    return tensor, reqs  # 返回 handle

send_backward(overlap_p2p_comm=True):
    reqs = _batched_p2p_ops(..., wait_on_reqs=False)
    return reqs

使用场景 (1F1B schedule):
  # 同时进行 P2P 通信和下一个 micro-batch 的计算
  tensor, reqs = recv_forward(overlap=True)
  output = forward(prev_tensor)   # ← 与 recv overlap
  for req in reqs: req.wait()     # 需要数据时才等待
```

### 4.4 P2P 时序 (1F1B, PP=4)

```
Stage 0: F₀ F₁ F₂ F₃ │ B₃ B₂ B₁ B₀
         ↓  ↓  ↓  ↓  │ ↑  ↑  ↑  ↑
Stage 1:    F₀ F₁ F₂ F₃ │ B₃ B₂ B₁ B₀
             ↓  ↓  ↓  ↓  │ ↑  ↑  ↑  ↑
Stage 2:       F₀ F₁ F₂ F₃ │ B₃ B₂ B₁ B₀
                ↓  ↓  ↓  ↓  │ ↑  ↑  ↑  ↑
Stage 3:          F₀ F₁ F₂ F₃ B₃ B₂ B₁ B₀

↓ = send activation (forward P2P)
↑ = send gradient (backward P2P)

overlap_p2p_comm: send 与下一个 F/B 的计算重叠
  Stage 0: F₀[send₀ ←async] F₁[wait₀ if needed]...
```

---

## 5. TP 通信 Overlap

### 5.1 Sequence Parallel 通信模式

```
标准 TP (无SP): all-reduce
  ColumnParallelLinear.forward: 输入完整, 输出分片, reduce=None
  RowParallelLinear.forward: 输入分片, 输出完整, 需要 all-reduce

SP 模式: reduce-scatter + all-gather
  ColumnParallelLinear.forward:
    输入 [s/TP, b, h] → all-gather → [s, b, h] → GEMM → 输出 [s, b, h/TP]
  RowParallelLinear.forward:
    输入 [s, b, h/TP] → GEMM → reduce-scatter → 输出 [s/TP, b, h]

通信量相同 (all-reduce = reduce-scatter + all-gather)
但 SP 的 scatter/gather 可以与 LayerNorm 等非 TP 操作 overlap
```

### 5.2 Async All-Reduce (gradient_accumulation_fusion)

```
源码: tensor_parallel/layers.py (LinearWithGradAccumulationAndAsyncCommunication)

机制:
  Forward: 正常计算 (无通信)
  Backward:
    - dgrad: 立即计算 (需要传给前一层)
    - wgrad: 累积到 param.main_grad (延迟 reduce)
      → wgrad 不急 (optimizer 才需要) → 可以延迟 all-reduce
      → 与 dgrad 计算 overlap

gradient_accumulation_fusion=True:
  wgrad 计算融合为 GEMM + 原地累加
  → 减少一次中间 buffer → 省内存 + 省带宽
```

### 5.3 TE Fused Communication (TransformerEngine-FL)

```
All-Gather + GEMM Fusion:
  传统: all-gather(完整input) → GEMM(input, weight)
  Fused: 分块 all-gather → 每块到达后立即计算该块的 GEMM
  
  ┌─ AG+GEMM Fused Pipeline ─────────────────────────────┐
  │  Comm:  [AG chunk0]─[AG chunk1]─[AG chunk2]─[AG chunk3]  │
  │  Comp:       [GEMM0]─────[GEMM1]─────[GEMM2]─────[GEMM3] │
  │  → 通信完全隐藏在计算中                                   │
  └────────────────────────────────────────────────────────────┘

GEMM + Reduce-Scatter Fusion:
  传统: GEMM(完整output) → reduce-scatter(output)
  Fused: GEMM 分块输出 → 每块完成后立即 reduce-scatter
  
  → 效果: TP 通信几乎零开销 (需 TE≥1.7 + 特定 API)
```

---

## 6. MoE Expert Parallel 通信优化

### 6.1 配置

```python
# transformer_config.py L2430+
overlap_moe_expert_parallel_comm: bool = False
# Overlap expert parallel all-to-all with expert compute
```

### 6.2 All-to-All 与 Expert 计算 Overlap

```
标准路径:
  All-to-All dispatch → Expert GEMM → All-to-All combine

Overlap 路径 (Combined 1F1B):
  将 Expert 计算按层拆分:
  [A2A dispatch chunk0] → [Expert0 compute] ← overlap → [A2A dispatch chunk1]
                       → [Expert1 compute] ← overlap → [A2A combine chunk0]

约束条件:
  1. 仅支持 expert parallelism (EP > 1)
  2. 仅支持 alltoall / flex token dispatcher
  3. 不能与 full recompute 同时使用
  4. 仅 bf16/fp16 模型
  5. 不能与 MoE shared expert overlap 同时使用
```

### 6.3 设计动机

```
MoE A2A 通信量:
  tokens × hidden × 2 (dispatch + combine) × 2 (双向)
  
  示例: 4096 tokens × 4096 hidden × 2B × 4 = 256 MB / layer
  
  NVLink: 256MB / 900GB/s ≈ 0.3 ms
  IB:     256MB / 400GB/s ≈ 0.6 ms (跨节点 EP)
  
  Expert 计算时间: ~1-5 ms (取决于 expert 大小)
  
  → A2A 可以被 Expert 计算隐藏 (当 compute > comm)
```

---

## 7. FP32 Reduce-Scatter 累积

### 7.1 问题

```
标准 reduce-scatter: 以 BF16 跨 GPU 累加
  → 当 dp_size 大时，多次 BF16 加法 → 精度损失累积
  
  误差: O(dp_size × 2^{-7}) (BF16 尾数 7 位)
  dp_size=64: 相对误差可达 ~0.5%
```

### 7.2 解决方案

```python
# distributed_data_parallel_config.py
reduce_scatter_with_fp32_accumulation: bool = False

实现:
  不使用标准 reduce_scatter (NCCL 内部 BF16 累加)
  而是: all-to-all 分发 → 本地 FP32 reduce
  
  步骤:
    1. All-to-all: 每个 rank 将 grad 分发到负责的 rank
    2. 接收端: 以 FP32 精度累加收到的所有 chunk
    3. 结果: 等价 reduce-scatter 但精度更高

  代价: 额外 all-to-all kernel + FP32 buffer
  适用: dp_size >= 32 且精度敏感的训练
```

---

## 8. 通信-计算 Overlap 全景时序

### 8.1 单步训练 (TP=2, PP=2, DP=4, Dist-Opt)

```
┌─── Forward ──────────────────────────────────────────────────────┐
│                                                                    │
│  param_gather(L0) ───────────────────┐                            │
│                                       ▼                            │
│  [LayerNorm] → [AG input] → [QKV GEMM] → [RS output] → [Attn]   │
│                 ↑ TP AG     ↑ overlap    ↑ TP RS                  │
│                              param_gather(L1) async                │
│                                                                    │
│  ...(repeat for each layer)                                        │
└────────────────────────────────────────────────────────────────────┘

┌─── Backward ─────────────────────────────────────────────────────┐
│                                                                    │
│  [Layer N dgrad] ──────────────────────────┐                      │
│                                             │ grad ready for       │
│  [Layer N-1 dgrad] ───────────────┐        │ bucket containing    │
│                                    │        │ Layer N params        │
│  NCCL Stream:                      │        ▼                      │
│    ──────────────────────[RS BucketN]─────[RS Bucket N-1]─────    │
│                           ↑ async overlap                          │
│                                                                    │
│  同时: param_gather(反序) for backward weight reuse               │
└────────────────────────────────────────────────────────────────────┘

┌─── Optimizer Step ───────────────────────────────────────────────┐
│  wait(所有 grad_reduce_handles)                                    │
│  optimizer.step() on local FP32 shard                              │
│  param_gather(L0) ← 为下一步 forward 预取                        │
└────────────────────────────────────────────────────────────────────┘
```

---

## 9. 关键设计决策

| 决策 | 选择 | 替代方案 | 为什么 |
|------|------|----------|--------|
| Grad reduce 触发 | bucket 粒度 | 单参数/全模型 | 平衡通信效率与overlap窗口 |
| Bucket 分组 | BucketGroup 合并通信 | 每bucket独立 | _coalescing_manager 减少launch |
| Param gather 预取 | Layer N+1 预取 | 全量预取 | 避免内存峰值 (全量=2× params) |
| P2P 模式 | batch + async | 逐个同步 | 减少 kernel launch + overlap |
| TP overlap | AG+GEMM fuse | 全量 AG 后 GEMM | Fuse 隐藏全部 TP 通信 |
| Grad ready 检测 | Hook counter | 显式标记 | 自动检测 (无需用户干预) |
| FP32 accumulation | 可选 | 始终 BF16 | 大 DP 时精度保护 |

---

## 10. 实践建议

### 10.1 配置推荐

```
优先级从高到低:

1. overlap_grad_reduce=True
   → 几乎无代价，隐藏 DP grad 通信 (5-15% 收益)

2. use_distributed_optimizer=True + overlap_param_gather=True
   → 前提: 已开启 dist-opt (隐藏 param gather, 3-8% 收益)

3. batch_p2p_comm=True (PP场景)
   → 合并 P2P，减少 launch 开销 (1-3%)

4. TE fused AG+GEMM / GEMM+RS (TE≥1.7)
   → 隐藏 TP 通信 (5-15%, 取决于 TP 大小)

5. bucket_size 调优
   → 默认值通常足够；跨节点 DP 可适当增大
```

### 10.2 诊断通信瓶颈

```
1. NCCL env: NCCL_DEBUG=INFO 观察实际带宽
2. nsys profile: 观察 NCCL kernel 是否与 compute overlap
3. 检查: 如果 NCCL stream 在 compute stream 之后才启动 → overlap 未生效
4. bucket_size: 如果看到很多小 NCCL kernel → bucket 太小
5. align_param_gather: 如果跨节点带宽不均 → 尝试开启
```

### 10.3 通信量估算

```
模型参数 P, DP=D, TP=T, PP=S

DP 通信 (per step):
  grad reduce-scatter: P/(T×S) × 2B × (D-1)/D ≈ P/(T×S) × 2B
  param all-gather:    同上

TP 通信 (per layer, per micro-batch):
  SP模式: 2 × (all-gather + reduce-scatter) = 4 × s×b×h×2B × (T-1)/T

PP 通信 (per micro-batch):
  P2P: s × b × h × 2B × 2 (fwd activation + bwd gradient)

总通信占比 = Σ(comm_volume) / bandwidth / total_step_time
  目标: < 5% (通信完全被overlap隐藏)
```
