# 第6章：分布式加载与断点恢复 深度源码分析

## 1. 概述与设计动机

大规模训练（数天/数周）中，故障恢复是硬需求。Energon 提供了一个**全链路可保存的数据加载系统**：从底层迭代器状态到顶层 wrapper 链，每一层都能保存和恢复精确位置。

核心挑战：
- 多 worker 进程并行读取，每个 worker 有独立状态
- Wrapper 链中间可能有 buffer（shuffle/packing），含未 yield 的数据
- 断点时机不可控（每 N 步/N 秒触发），需要异步快照
- 恢复时需要 skip 已产出但未消费的样本

## 2. 源码定位

| 组件 | 文件 | 行数 | 职责 |
|------|------|------|------|
| 核心加载器 | savable_loader.py | 1411 | SavableDataLoader + SavableDatasetWrapper |
| Worker 配置 | worker.py | 288 | WorkerConfig — 分布式 worker ID |
| RNG 管理 | rng.py | ~150 | WorkerRng — 确定性随机 |
| 状态基类 | savable.py | ~100 | Savable + FlexState |
| 数据集基类 | flavors/base_dataset.py | ~200 | SavableDataset — 状态接口 |

## 3. 与 Megatron 并行训练的集成

### 3.1 并行模式下的数据加载与恢复

在 TP/PP/DP 并行训练中，数据加载和断点恢复必须正确处理并行语义：

```python
def get_batch(data_iterator):
    """完整的并行感知 get_batch 实现 — 含断点恢复语义"""
    # PP guard: 只有 pipeline first stage (pre_process=True) 加载数据
    if mpu.is_pipeline_first_stage():
        # TP guard: 只有 TP rank 0 执行数据加载（含恢复后的 skip）
        if mpu.get_tensor_model_parallel_rank() == 0:
            # SavableDataLoader 在恢复时自动 skip 已消费样本
            batch = next(data_iterator)
        else:
            batch = None
        # TP broadcast: rank 0 广播到所有 TP ranks
        batch = broadcast_data(
            keys=["tokens", "labels", "loss_mask", "position_ids"],
            data=batch, datatype=torch.int64,
            src=mpu.get_tensor_model_parallel_src_rank()
        )
    else:
        # PP 非 first stage (post_process=True): 数据通过 PP 通信获取
        batch = None
    return batch
```

### 3.2 DP ranks 间的状态一致性

每个 DP rank 独立运行 SavableDataLoader，各自保存/恢复自己的状态。关键保证：
- checkpoint 时所有 DP ranks 同步（由 Megatron checkpoint barrier 保证）
- 恢复时每个 DP rank 加载自己的 data state（按 DP rank 索引）
- WorkerConfig(rank=dp_rank) 确保恢复后 shard 分配与 checkpoint 前一致

### 3.3 PP stages 间的数据恢复

只有 PP first stage 持有 SavableDataLoader 状态。Checkpoint 保存时：
- PP first stage: 保存完整 data loader state
- PP other stages: 不保存数据状态（它们不直接读数据）

## 4. 状态保存架构

### 4.1 三层状态模型

```
┌─────────────────────────────────────────┐
│ SavableDataLoader (顶层)                 │
│  ├─ worker_states: List[WorkerState]    │ ← 每个 worker 的状态
│  ├─ sample_queue residuals              │ ← 已产出未消费的样本
│  └─ global_sample_index                 │
├─────────────────────────────────────────┤
│ SavableDatasetWrapper (每 worker)        │
│  ├─ SavableDatasetState                 │
│  │   ├─ rng: SystemRngState             │ ← 全系统 RNG 快照
│  │   ├─ dataset_state: FlexState        │ ← wrapper链递归状态
│  │   └─ sample_index: int               │ ← 已产出样本数
│  └─ checkpoints: deque[SavableCheckpoint]│ ← 滑动窗口快照
├─────────────────────────────────────────┤
│ Wrapper Chain (递归 FlexState)           │
│  BlendDataset.save_state()              │
│    ├─ exhausted: List[bool]             │
│    ├─ _worker_rng: state                │
│    └─ datasets: [inner.save_state()]    │ ← 递归到叶子
└─────────────────────────────────────────┘
```

### 4.2 SavableCheckpoint — 滑动窗口快照

```python
@edataclass
class SavableCheckpoint:
    state: Optional[SavableDatasetState]
    checkpoint_time: float
    sample_index: int
```

每隔 `checkpoint_every_sec` 秒且至少产出 `checkpoint_every_min_n_samples` 样本后创建新快照，保留最近 `n_checkpoints` 个。

## 5. 断点触发与异步快照

### 5.1 Command Thread + 双队列

```
Main Process (训练循环)           Worker Process (数据加载)
       │                                │
       │ ──── cmd_queue ──────►          │
       │      "save_state"               │ → 返回最近 checkpoint
       │                                │
       │ ◄──── result_queue ────         │
       │      SavableCheckpoint          │
```

### 5.2 恢复流程：Skip 语义

恢复时，worker 从最近 checkpoint 的 state 恢复，然后 skip `(current_index - checkpoint_index)` 个样本，追上断点时的实际位置。

## 6. 设计决策对比

| 维度 | Energon SavableLoader | PyTorch DataLoader | DeepSpeed DataLoader |
|------|----------------------|-------------------|---------------------|
| 状态保存 | ✅ 全链路递归 | ❌ | 部分(seed only) |
| 异步快照 | ✅ command thread | ❌ | ❌ |
| Buffer 状态 | ✅ SavableSampleBuffer | ❌ | ❌ |
| Skip 恢复 | ✅ 精确到样本 | ❌ | epoch 级别 |
| 多 worker | ✅ 每 worker 独立状态 | ✅ | ✅ |
| 并行感知 | ✅ DP rank 分片 | ❌ | ✅ DP only |
