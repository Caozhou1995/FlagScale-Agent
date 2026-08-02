# 第3章：TaskEncoder 与数据编码 深度源码分析

## 1. 概述与设计动机

TaskEncoder 是 Megatron-Energon 数据管道的核心抽象层，解决"原始数据→模型输入"的转换问题。它将数据处理分解为可组合的阶段：

- **Cooking**：CrudeSample (dict) → 结构化 Sample（由 Cooker 完成）
- **Encoding**：Sample → encoded tensor（模型前的最后变换）
- **Batching**：多个 encoded sample → batch（含 padding/stacking 逻辑）
- **Packing**：可选的 sample 合并（如 LLM 的 sequence packing）

核心设计思想：**用户只需实现少量 hook 方法，框架自动组装完整的数据管道**。TaskEncoder 通过 `_is_overridden()` 检测用户实现了哪些方法，动态构建相应的 wrapper 链。

## 2. 源码定位

| 组件 | 文件路径 | 行数 | 职责 |
|------|----------|------|------|
| TaskEncoder 基类 | task_encoder/base.py | 1128 | 核心抽象、管道构建、batch/encode hooks |
| Cooker 装饰器 | task_encoder/cooking.py | 121 | @cooker 装饰器、Cooker dataclass |
| Dataset 加载器 | task_encoder/loader.py | 295 | get_train_dataset/get_val_dataset 高层API |
| 导出 | task_encoder/__init__.py | 38 | 公开接口 |

## 3. 架构总览

### 3.1 数据处理流水线

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────────┐
│ WebDataset   │──▶│  Cooker      │──▶│ encode_sample│──▶│ Shuffle     │
│ (CrudeSample)│   │ (dict→Sample)│   │ (Sample→T)   │   │ Buffer      │
└──────────────┘   └──────────────┘   └──────────────┘   └──────┬──────┘
                                                                  │
     ┌────────────────────────────────────────────────────────────┘
     ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────────┐
│ [Packing]    │──▶│ batch()      │──▶│ encode_batch │──▶│ Epochize    │
│ (可选)       │   │ (List→Batch) │   │ (Batch→Final)│   │ (虚拟epoch) │
└──────────────┘   └──────────────┘   └──────────────┘   └─────────────┘
```

### 3.2 类型泛型体系

```python
class TaskEncoder(Generic[T_sample, T_encoded_sample, T_raw_batch, T_batch]):
    # T_sample:         cook 后的结构化 sample
    # T_encoded_sample: encode_sample 后的 tensor 化 sample
    # T_raw_batch:      batch() 输出的原始 batch
    # T_batch:          encode_batch() 输出的最终 batch
```

### 3.3 方法调用关系

```
get_train_dataset()
  └─▶ task_encoder.build_train_datasets()
       ├─▶ build_cook_crude_sample()  [如有 cookers]
       │    └─ MapDataset(cook_crude_sample)
       ├─▶ build_encode_sample()
       │    └─ MapDataset(encode_sample)
       ├─▶ ShuffleBufferDataset
       ├─▶ build_batch()
       │    ├─ [PackingDataset] if packing_buffer_size
       │    ├─ [GroupBatchDataset] if batch_group_criterion
       │    ├─ BatchDataset(batch_size, batcher=self.batch)
       │    └─ MapDataset(encode_batch)
       └─▶ EpochizeDataset
```

## 4. Cooker — 原始数据解码

### 4.1 设计动机（WHY）

WebDataset 读出的 sample 是一个 dict（CrudeSample），key 是文件名（如 "jpg", "txt"），value 是 raw bytes。不同任务需要不同的解码逻辑（图像 → PIL/tensor，文本 → token ids），但解码逻辑不应硬编码在数据集类中。

Cooker 提供了一个 **声明式匹配 + 转换** 机制：用户声明 cooker 函数，框架根据 sample 的 subflavors 自动选择匹配的 cooker。

### 4.2 @cooker 装饰器（cooking.py:L31-55）

```python
def cooker(fn=None, *, need_cache=False, need_primary=False):
    """标记函数为 cooker，可选启用 cache 和 primary dataset 参数。"""
    @functools.wraps(fn)
    def fn_wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    
    # 通过属性标记能力
    setattr(fn_wrapper, "__cooker_need_cache__", need_cache)
    setattr(fn_wrapper, "__cooker_need_primary__", need_primary)
    return fn_wrapper
```

`need_cache=True` 时，cooker 函数签名可接收 `cache` 参数（用于缓存解码结果）；
`need_primary=True` 时，可接收 `primary` 参数（用于随机访问关联 sample）。

### 4.3 Cooker dataclass — 匹配逻辑（cooking.py:L68-104）

```python
@dataclass
class Cooker(Generic[T_sample]):
    cook: Callable[..., T_sample]       # 实际转换函数
    has_subflavors: Optional[dict]      # 匹配条件
    
    def is_match(self, subflavors: dict) -> bool:
        """检查 sample 的 subflavors 是否满足此 cooker 的过滤条件。
        规则：cooker 要求的 key-value 必须全部在 sample 中存在且相等。"""
        if self.has_subflavors is None:
            return True  # 无条件匹配
        for k, v in self.has_subflavors.items():
            if k not in subflavors or subflavors[k] != v:
                return False
        return True
```

**设计选择**：subflavors 匹配是"子集包含"关系——cooker 只要求部分 key 匹配，sample 可以有更多 key。这允许一个通用 cooker 处理多种 sample 变体。

### 4.4 basic_sample_keys 辅助函数（cooking.py:L107-121）

```python
def basic_sample_keys(crude_sample: dict, additional_source_info=()):
    """从 crude_sample 提取 Sample 基类的必需字段（__key__, __sources__ 等）。
    cooker 函数应调用此函数获取元数据字段：
    
    @cooker
    def my_cook(raw: dict) -> MySample:
        return MySample(**basic_sample_keys(raw), image=decode(raw['jpg']))
    """
    res = {field.name: crude_sample[field.name] 
           for field in dataclasses.fields(Sample) if field.name in crude_sample}
    if additional_source_info:
        res["__sources__"] = (*crude_sample["__sources__"], *additional_source_info)
    return res
```

### 4.5 cook_crude_sample 执行流程（base.py:L403-435）

```python
@stateless
def cook_crude_sample(self, sample, cooker, aux):
    """执行 cooking：根据 cooker 的能力需求注入参数。"""
    kwargs = {}
    if cooker.need_cache:
        kwargs["cache"] = self._cache_pool
    if cooker.need_primary:
        kwargs["primary"] = aux.get("primary")
    kwargs.update({k: v for k, v in aux.items() if k != "primary"})
    return cooker.cook(sample, **kwargs)
```

注意 `@stateless` 装饰器确保 cooking 过程不影响外部 RNG 状态。

## 5. @stateless 装饰器 — RNG 隔离

### 5.1 设计动机（WHY）

数据增强（如随机裁剪、颜色抖动）需要随机数。但在断点恢复时，必须保证：
1. 恢复后产出与中断前完全相同的数据序列
2. 数据增强的随机数不影响外部（如 shuffle buffer 的 RNG）

`@stateless` 解决这个问题：在函数调用前后保存/恢复 RNG 状态，使函数内部的随机操作对外部"不可见"。

### 5.2 实现机制（base.py:L124-218）

```python
def stateless(*, restore_seeds=False, failure_tolerance=None):
    """对普通函数：保存外部 RNG → 执行函数 → 恢复外部 RNG
    对生成器函数：额外管理 inner/outer 两套 RNG 状态。"""
    
    # 生成器版本的核心逻辑：
    outer_rng_state = SystemRng.save_state()
    it = fn(*args, **kwargs)
    while True:
        if inner_rand_state is not None:
            SystemRng.restore_state(inner_rand_state)   # 恢复内部状态
        sample = next(it)                               # 执行生成器
        inner_rand_state = SystemRng.save_state()       # 保存内部状态
        SystemRng.restore_state(outer_rng_state)        # 恢复外部状态
        yield sample                                    # 返回给调用者
        outer_rng_state = SystemRng.save_state()        # 调用者可能改了 RNG
```

### 5.3 时序图：RNG 状态切换

```
调用者 RNG:    A ──────────── A' ──────────── A'' ─────
                  ↓ save         ↑ restore       ↓ save
@stateless fn:    │    B ──── B' │               │    B' ──── B''
                  │    ↑ restore │               │    ↑ restore
                  └────┘         └───────────────┘
              yield sample1                  yield sample2
```

## 6. Batch 构建系统

### 6.1 generic_batch — 自动 batching（base.py:L72-98）

**设计动机**：不同字段需要不同的 batching 策略（tensor 需 pad+stack，string 只能放 list），但用户不想为每个字段手写。

```python
def generic_batch(batch: List[Any]) -> Any:
    """根据类型自动选择 batching 策略："""
    if isinstance(batch[0], torch.Tensor):
        return batch_pad_stack(batch)      # pad 到最大 shape 后 stack
    elif isinstance(batch[0], dict):
        return {k: generic_batch([s[k] for s in batch]) for k in batch[0]}
    elif is_dataclass(batch[0]):
        if hasattr(batch[0], "from_samples"):
            return batch[0].from_samples(batch)  # 自定义 batching
        return type(batch[0])(**{
            f.name: generic_batch([getattr(s, f.name) for s in batch])
            for f in dataclasses.fields(batch[0])
        })
    else:
        return batch_list(batch)           # 其他类型直接放 list
```

### 6.2 batch_pad_stack — 变长 tensor padding（base.py:L106-113）

```python
def batch_pad_stack(batch: List[torch.Tensor]) -> torch.Tensor:
    """将不同 shape 的 tensor pad 0 到统一 shape 后 stack。"""
    max_size = [max(b.shape[dim] for b in batch) for dim in range(batch[0].ndim)]
    batch_tensor = batch[0].new_zeros((len(batch), *max_size))
    for i, b in enumerate(batch):
        batch_tensor[(i, *(slice(0, s) for s in b.shape))] = b
    return batch_tensor
```

**约束**：所有 tensor 必须同 dtype 和 ndim，否则 new_zeros 会报错。

### 6.3 Batch dataclass（base.py:L256-305）

```python
@edataclass
class Batch:
    __key__: Optional[List[str]] = None
    __restore_key__: Optional[List[Tuple]] = None
    __subflavors__: Optional[list] = None
    __sources__: Optional[tuple[SourceInfo, ...]] = None
    
    @classmethod
    def derive_from(cls, base_batch, **kwargs):
        """从已有 batch 保留元数据，替换数据字段。"""
        base_kwargs = {f.name: getattr(base_batch, f.name) 
                       for f in dataclasses.fields(Batch)}
        return cls(**base_kwargs, **kwargs)
    
    @classmethod
    def from_samples(cls, samples, **kwargs):
        """默认 batching：tensor pad+stack，其他 list。"""
        ...
```

### 6.4 build_batch 管道构建（base.py:L565-673）

TaskEncoder.build_batch() 根据用户实现的方法动态组装：

```
用户实现了什么？                    框架构建什么？
─────────────────────────────────────────────────────
packing_buffer_size 设置了         → PackingDataset
  + select_samples_to_pack
  + pack_selected_samples

batch_group_criterion 实现了       → GroupBatchDataset
  (按 criterion 分组后再 batch)

都没实现                           → BatchDataset (标准固定大小 batch)

encode_batch 实现了                → + MapDataset(encode_batch)
```

## 7. Packing 子系统

### 7.1 设计动机（WHY）

LLM 训练中，不同 sequence 长度差异大。固定 batch_size × max_seq_len 浪费计算（短序列全是 padding）。Packing 将多个短序列拼接到一个长序列中，提高 GPU 利用率。

### 7.2 用户接口

```python
class MyTaskEncoder(TaskEncoder):
    def select_samples_to_pack(self, samples: List[T]) -> List[List[T]]:
        """从 buffer 中选择哪些 sample 打包在一起。
        输入：packing_buffer_size 个 sample
        输出：分组列表，每组将被 pack 在一起。"""
        # 例：贪心 bin-packing by sequence length
        ...
    
    def pack_selected_samples(self, samples: List[T]) -> T:
        """将一组 sample 实际合并为一个。
        例：拼接 input_ids，构造 attention_mask。"""
        ...
```

### 7.3 PackingDataset 内部流程

```
samples from upstream
    │
    ▼ 累积到 buffer（大小 = packing_buffer_size）
    │
    ▼ select_samples_to_pack(buffer) → [[s1,s2], [s3,s4,s5], ...]
    │
    ▼ pack_selected_samples([s1,s2]) → packed_sample_1
    │
    ▼ [postencode_sample(packed_sample)] if overridden
    │
    ▼ emit packed samples one by one
```

## 8. get_train_dataset — 完整管道组装（loader.py:L107-179）

### 8.1 函数签名与关键参数

```python
def get_train_dataset(
    path,                              # 数据集路径（指向 metadataset yaml）
    split_part="train",                # train/val/test
    worker_config=...,                 # WorkerConfig (rank, world_size, num_workers)
    batch_size=...,                    # batch 大小（None = 不 batch）
    shuffle_buffer_size=...,           # shuffle buffer（sample 级随机化）
    max_samples_per_sequence=...,      # 控制顺序读取粒度
    virtual_epoch_length=0,            # 虚拟 epoch 长度（0 = 无限循环）
    task_encoder=DefaultTaskEncoder(), # 用户自定义 TaskEncoder
    repeat=True,                       # 是否无限循环
) -> SavableDataset[T]:
```

### 8.2 完整执行流程

```
1. load_dataset(path)              → MetadatasetLoader
2. loader.get_datasets(...)        → LoadedDataset (含多个子数据集)
3. task_encoder.build_train_datasets(
     datasets=...,
     worker_config=...,
     batch_size=...,
     shuffle_buffer_size=...,
     blend_mode=...,                → blend/concat 多数据集
   )
   内部：
   a. 对每个 dataset:
      - build_cook_crude_sample()   [如有 cooker]
      - build_encode_sample()       [MapDataset(encode_sample)]
   b. blend/concat 多个 dataset
   c. ShuffleBufferDataset
   d. build_batch()                 [含 packing/group/encode_batch]
   e. EpochizeDataset               [如设了 virtual_epoch_length]
   f. LogSampleDataset              [日志记录]
```

### 8.3 train vs val 的区别

| 维度 | get_train_dataset | get_val_dataset |
|------|-------------------|-----------------|
| shuffle | ShuffleBufferDataset | 无 shuffle |
| repeat | 默认无限循环 | 单 epoch |
| epochize | virtual_epoch_length | 无 |
| limit | 无 | 可设 limit 参数 |
| blend_mode | 支持 blend/concat | 支持 |

## 9. _is_overridden — 动态管道检测（base.py:L385-401）

### 9.1 设计动机（WHY）

TaskEncoder 有很多可选 hook（encode_sample, encode_batch, batch_group_criterion 等）。如果用户没 override，就不应该创建对应的 wrapper（节省开销）。框架需要在运行时判断哪些方法被子类覆盖了。

### 9.2 实现

```python
def _is_overridden(self, bound_method, bases=None):
    """检查 bound_method 是否在子类中被覆盖（而非使用基类默认实现）。"""
    if not isinstance(bound_method, MethodType):
        return True  # 非 bound method 一定是覆盖的
    func = bound_method.__func__
    if bases is None:
        bases = (TaskEncoder,)
    # 如果 func 和任一基类的同名方法是同一个对象，说明没覆盖
    return not any(getattr(base, func.__name__) is func for base in bases)
```

**效果**：用户只实现需要的方法，框架自动跳过未实现的阶段。

## 10. 与 Megatron 训练循环的并行集成

### 10.1 TaskEncoder 与并行策略的关系

TaskEncoder 工作在 **DP 维度的数据加载侧**，它本身不感知 TP/PP/CP 等并行。但在 Megatron 训练循环中，数据从 TaskEncoder 产出后必须正确分发到所有并行 rank：

```
TaskEncoder (在 DP rank 上运行)
    │
    ▼ 产出 batch
    │
    ▼ get_batch() 中的并行处理：
    │
    ├─ PP 维度：只有 first pipeline stage (pre_process=True) 加载数据
    │   if not mpu.is_pipeline_first_stage():
    │       return None  # 其他 PP stage 不加载
    │
    ├─ TP 维度：TP rank 0 加载数据，然后 broadcast 给同组其他 rank
    │   if mpu.get_tensor_model_parallel_rank() == 0:
    │       batch = next(data_iterator)
    │   batch = broadcast_data(keys, batch, torch.int64)  # TP 组内广播
    │
    └─ DP 维度：每个 DP rank 独立加载不同的 sample
        （WorkerConfig.rank = dp_rank, world_size = dp_world_size）
```

### 10.2 get_batch 标准模式

```python
def get_batch(data_iterator):
    """Megatron 训练循环中的 get_batch 实现。"""
    # PP guard: 只有 pipeline 第一级需要数据
    if not mpu.is_pipeline_first_stage():
        return None
    
    # TP rank 0 从 Energon TaskEncoder 获取 batch
    if mpu.get_tensor_model_parallel_rank() == 0:
        batch = next(data_iterator)  # TaskEncoder 产出的 batch
        tokens = batch['tokens']
        labels = batch['labels']
    else:
        tokens = None
        labels = None
    
    # TP 组内广播
    tokens = broadcast_data(['tokens'], {'tokens': tokens}, torch.int64)['tokens']
    labels = broadcast_data(['labels'], {'labels': labels}, torch.int64)['labels']
    
    return tokens, labels
```

### 10.3 WorkerConfig 中的 DP 维度映射

TaskEncoder 通过 WorkerConfig 获取 DP 维度信息：
- `worker_config.rank` = 当前进程在 **DP 组**中的 rank（不是全局 rank）
- `worker_config.world_size` = DP 组的大小（不是总 GPU 数）
- Sharder 用这两个值计算当前 rank 应加载哪些 sample

例如 8 GPU, TP=2, PP=2, DP=2 时：
- 总 rank: 0-7
- DP world_size = 2
- 只有 DP rank 0 和 1 各自运行 TaskEncoder
- TP/PP 其他 rank 通过 broadcast_data / PP schedule 获取数据

## 11. 设计决策对比表

| 维度 | Energon TaskEncoder | PyTorch DataLoader collate_fn | HuggingFace Trainer |
|------|--------------------|-----------------------------|---------------------|
| 管道组合 | 声明式（override hook → 自动组装） | 手动组装 | 配置式 |
| 类型安全 | Generic 4-type 泛型 | 无 | 部分 |
| RNG 隔离 | @stateless 自动管理 | 用户自己管理 | 无 |
| Packing | 内置 PackingDataset | 手动实现 | 通过 DataCollator |
| 断点恢复 | 精确到 sample | 不支持 | 按 step 计数 |
| 多数据集 | blend/concat 内置 | 手动 ConcatDataset | 不支持 |
| 并行感知 | WorkerConfig 映射 DP 维度 | 无（需手动 DistributedSampler）| 内置 |
| batch 策略 | 自动 pad+stack / GroupBatch | 用户定义 collate_fn | DataCollator |

## 12. 配置建议与调优指南

### 12.1 自定义 TaskEncoder 最佳实践

| 场景 | 推荐实现的方法 | 原因 |
|------|---------------|------|
| 简单分类/检测 | encode_sample + batch | 标准流程 |
| LLM 预训练 | encode_sample + packing 两方法 | 提高 GPU 利用率 |
| 多模态 | cooker + encode_sample + batch | 不同模态解码逻辑不同 |
| 变长序列分桶 | batch_group_criterion + batch | 同长度一起 batch |
| 复杂后处理 | encode_batch | 在 batch 级别做变换 |

### 12.2 常见陷阱

1. **忘记 @stateless**：数据增强中的随机操作会破坏断点恢复的确定性
2. **cooker 匹配顺序**：多个 cooker 时，框架取第一个 `is_match=True` 的
3. **batch_pad_stack OOM**：变长序列差异太大时，padding 浪费显存 → 考虑 GroupBatch 或 Packing
4. **get_batch 缺少 broadcast_data**：TP>1 时非 rank0 拿到 None → 必须广播
5. **get_batch 缺少 PP guard**：非 first stage 不应调用 data_iterator

### 12.3 性能考量

| 配置 | 推荐值 | 影响 |
|------|--------|------|
| shuffle_buffer_size | 1000-10000 | 太小→随机性不足；太大→内存占用 |
| packing_buffer_size | 100-1000 | 太小→packing 效率低；太大→首 batch 延迟 |
| virtual_epoch_length | total_samples/dp_world_size | 控制 checkpoint 频率 |
| batch_size | 依模型显存 / DP 维度 | Energon batch_size = micro_batch_size |
