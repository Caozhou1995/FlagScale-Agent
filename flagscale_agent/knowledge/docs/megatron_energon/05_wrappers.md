# 第5章：Wrappers 数据管道组合 深度源码分析

## 1. 概述与设计动机

Wrappers 是 Energon 数据管道的**组合层** — 每个 Wrapper 接收一个或多个 SavableDataset，施加一种变换后产出新的 SavableDataset。通过链式组合，构建出完整的数据处理管道。

核心设计思想：**单一职责 + 可组合** — 每个 wrapper 只做一件事（shuffle/batch/blend/filter/...），通过嵌套组合实现任意复杂管道，且每一层都支持状态保存/恢复。

## 2. 源码定位

| 组件 | 文件 | 行数 | 职责 |
|------|------|------|------|
| 基类 | base.py | 192 | BaseWrapperDataset — 统一接口 |
| 混合 | blend_dataset.py | 122 | 按权重混合多个数据集 |
| 批处理 | batch_dataset.py | 240 | 组 batch + collate |
| 分组批 | group_batch_dataset.py | 264 | 按 key 分组后 batch |
| 混合批 | mix_batch_dataset.py | 133 | 多数据集混合后 batch |
| Packing | packing_dataset.py | 439 | 序列 packing（最复杂） |
| Shuffle | shuffle_buffer_dataset.py | 71 | buffer shuffle |
| Map | map_dataset.py | 215 | 函数映射变换 |
| IterMap | iter_map_dataset.py | 207 | 迭代器级映射 |
| Filter | filter_dataset.py | 79 | 条件过滤 |
| Concat | concat_dataset.py | 51 | 顺序拼接 |
| Repeat | repeat_dataset.py | 112 | 重复迭代 |
| Limit | limit_dataset.py | 125 | 限制样本数 |
| Epochize | epochize_dataset.py | 122 | epoch 边界标记 |
| GC | gc_dataset.py | 155 | 垃圾回收触发 |
| Buffer | buffer.py | 154 | 可保存 buffer |
| Log | log_sample_dataset.py | 73 | 采样日志 |
| Skip | skip.py | 17 | 跳过 N 个样本 |
| Watchdog | watchdog_dataset.py | 76 | 超时看门狗 |

总计：~2900 行，19 个 wrapper 实现。

## 3. BaseWrapperDataset — 基础框架（base.py）

### 3.1 设计动机（WHY）

所有 wrapper 需要统一管理：
- 内部 dataset 引用（单个或多个）
- worker_config 一致性检查
- 状态保存/恢复的递归调用
- restore_sample 的路由

### 3.2 核心实现

```python
class BaseWrapperDataset(SavableDataset[T_out], Generic[T_in, T_out], ABC):
    datasets: Tuple[SavableDataset[T_in], ...]
    
    def __init__(self, datasets, *, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        if isinstance(datasets, SavableDataset):
            self.datasets = (datasets,)
        else:
            self.datasets = tuple(datasets)
        # 一致性检查：所有内部 dataset 的 worker_config 必须相同
        for d in self.datasets:
            assert d.worker_config == self.worker_config

    def save_state(self) -> FlexState:
        own_state = super().save_state()
        return FlexState(datasets=[ds.save_state() for ds in self.datasets], **own_state)
    
    def restore_state(self, state: FlexState) -> None:
        for dataset, dstate in zip(self.datasets, state["datasets"]):
            dataset.restore_state(dstate)
        super().restore_state(state)
```

### 3.3 restore_sample 路由

```python
def restore_sample(self, restore_key):
    if len(self.datasets) == 1:
        return self.datasets[0].restore_sample(restore_key)
    else:
        id, ds_idx = restore_key[:2]  # 解码来源 dataset 索引
        return self.datasets[ds_idx].restore_sample(restore_key[2:])
```

**设计精妙之处**：每层 wrapper 在 yield 时通过 `add_sample_restore_key(sample, idx, src=self)` 向 restore_key 添加自己的路由信息，形成从外到内的路径链。


## 4. BlendDataset — 加权混合（blend_dataset.py）

### 4.1 设计动机（WHY）

多数据集按权重采样是大模型预训练的核心需求（如 code:0.3 + text:0.7）。BlendDataset 提供无限流式混合，支持动态权重调整和耗尽检测。

### 4.2 核心算法

```python
class BlendDataset(BaseWrapperDataset[T, T]):
    datasets: List[SavableDataset[T]]
    weights: Tuple[float, ...]
    exhausted: List[bool]
    _worker_rng: WorkerRng
    
    _savable_fields = ("exhausted", "_worker_rng")  # 状态保存字段
    
    def __iter__(self):
        dataset_iters = []
        weights = torch.tensor([...], dtype=torch.float32)
        
        # 无限循环采样
        while True:
            ds_idx = self._worker_rng.choice_idx(probs=weights)  # 按权重随机选
            try:
                sample = next(dataset_iters[ds_idx])
            except StopIteration:
                weights[ds_idx] = 0          # 该数据集耗尽
                self.exhausted[ds_idx] = True
                if all(exhausted): break
            else:
                yield add_sample_restore_key(sample, ds_idx, src=self)
```

### 4.3 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| RNG | WorkerRng (确定性) | 每个 worker 独立可复现 |
| 耗尽处理 | 权重置 0 继续 | 不中断其他数据集 |
| 恢复策略 | 保存 exhausted + rng state | 恢复后从相同位置继续 |

## 5. PackingDataset — 序列拼接（packing_dataset.py）

### 5.1 设计动机（WHY）

LLM 训练中短序列浪费计算（padding to max_len）。Packing 将多个短序列拼接到一个 max_len 窗口内，提升 GPU 利用率。这是 Energon 最复杂的 wrapper（439 行）。

### 5.2 三阶段流水线

```
Input samples → [Reading Buffer] → pre_packer → [Pre-packing Buffer]
                                                        │
                                                        ▼
                                              sample_encoder (可选)
                                                        │
                                                        ▼
                                              final_packer → Packed Batch
```

### 5.3 状态保存的复杂性

PackingDataset 需要保存 3 个 buffer + 3 个 sample_index：
```python
_savable_fields = (
    "_reading_buffer",              # 正在收集的样本
    "_pre_packing_buffer",          # 已分组待编码的样本
    "_pre_packing_lengths",         # 分组长度列表
    "_pre_packing_sample_index",
    "_sample_encoder_sample_index",
    "_final_packing_sample_index",
)
```

## 6. ShuffleBufferDataset — 流式 Shuffle

Buffer shuffle 在内存中维护固定大小 buffer，随机替换产出。71 行极简实现。

## 7. 与 Megatron 并行训练的集成

### 7.1 Wrapper 链在并行模式下的位置

Wrapper 链运行在 **DP rank 的数据加载进程**中。每个 DP rank 独立运行自己的 wrapper 链。

### 7.2 并行感知的 get_batch 实现

```python
def get_batch(data_iterator):
    """完整的并行感知 get_batch — wrapper链产出的batch如何进入模型"""
    # PP guard: 只有 pipeline first stage (pre_process=True) 加载数据
    if mpu.is_pipeline_first_stage():
        # TP guard: 只有 TP rank 0 执行完整 wrapper 链
        if mpu.get_tensor_model_parallel_rank() == 0:
            # data_iterator 背后是: WebDataset→Shuffle→Map→Blend→Batch→Pack
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
        # PP 非 first stage (post_process=True): 不需要数据输入
        batch = None
    return batch
```

### 7.3 DP 分片保证

BlendDataset 的 WorkerRng 使用 `worker_config.rank`(=DP rank) 作为 seed，保证各 DP rank 数据不重叠且可复现。

## 8. 设计决策对比

| 维度 | Energon Wrappers | PyTorch DataPipe | HF datasets.map |
|------|-----------------|------------------|-----------------|
| 组合方式 | 嵌套对象 | 链式方法 | 函数式 |
| 状态保存 | ✅ 每层独立 | ❌ | ❌ |
| 无限流 | ✅ 原生 | ✅ | ❌ (有限) |
| 序列 Packing | ✅ 内置 | ❌ | ❌ |
| 确定性 RNG | ✅ WorkerRng | ❌ | ❌ |
| 分布式感知 | ✅ WorkerConfig | 部分 | ❌ |
