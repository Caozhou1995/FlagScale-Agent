# 第4章：Metadataset 多数据集混合 深度源码分析

## 1. 概述与设计动机

大规模训练通常需要混合多个数据源（如：代码+自然语言+数学），且每个数据源的采样权重不同。Metadataset 提供了一个**层级化配置系统**，解决：

- **多数据集混合**：按权重 blend 或按比例 concat 多个数据集
- **层级嵌套**：metadataset 可引用其他 metadataset，递归组合
- **子集选择**：对每个数据集取特定范围（绝对/百分比）
- **辅助数据集关联**：通过 aux 引用为主数据集关联辅助信息

核心设计思想：**配置即管道** — 用户编写 YAML 配置文件描述数据混合策略，框架自动解析并构建对应的 Dataset wrapper 链。

## 2. 源码定位

| 组件 | 文件路径 | 行数 | 职责 |
|------|----------|------|------|
| 入口加载 | metadataset/loader.py | 140 | load_dataset() — 根据类型分发 |
| 接口定义 | metadataset/loader_interface.py | 140 | DatasetLoaderInterface/LoadedDataset/BlendMode |
| V1 Metadataset | metadataset/metadataset.py | 310 | Metadataset/MetadatasetSplit/DatasetReference |
| V2 Metadataset | metadataset/metadataset_v2.py | 821 | MetadatasetV2 — 完整特性(aux/subset/join) |
| 单数据集加载 | metadataset/dataset_loader.py | 111 | DatasetLoader — 叶子节点 |
| Join 加载 | metadataset/join_dataset_loader.py | 562 | JoinDatasetLoader — 多数据集关联 |

## 3. 架构总览

### 3.1 类继承关系

```
DatasetLoaderInterface (ABC)
├── Metadataset         ← V1: train/val/test splits, DatasetReference[]
│     └── MetadatasetSplit → 含 datasets[] 列表
│           └── DatasetReference → 递归引用其他 metadataset/dataset
├── MetadatasetV2       ← V2: 新增 aux, subset, join, blend_epochized
├── DatasetLoader       ← 叶子节点：直接加载单个 WebDataset/JSONL
└── JoinDatasetLoader   ← 关联多个数据集（按 key join）
```

### 3.2 配置解析流程

```
metadataset.yaml
    │
    ▼ load_dataset(path)              [loader.py:L19-50]
    │
    ├─ 检测类型: get_dataset_type(path)
    │   ├─ METADATASET → load_config → Metadataset/V2
    │   ├─ WEBDATASET  → DatasetLoader(path)
    │   └─ JSONL       → DatasetLoader(path)
    │
    ▼ mds.post_initialize()            [递归解析所有引用]
    │
    ▼ mds.get_datasets(training, split_part, worker_config)
    │
    └─▶ LoadedDatasetList(datasets=[...], blend_mode=...)
```

### 3.3 Blend 模式

```python
class DatasetBlendMode(Enum):
    NONE = "none"                      # 单数据集，无混合
    DATASET_WEIGHT = "dataset_weight"  # 按权重采样（code:0.3, text:0.7）
    SAMPLE_REPETITIONS = "sample_repetitions"  # 按重复倍率（code×2, text×1）
```

## 4. load_dataset — 入口分发（loader.py）

### 4.1 设计动机（WHY）

用户可能传入不同类型的路径：metadataset YAML、WebDataset 目录、dict 配置。需要统一入口自动识别类型。

### 4.2 实现

```python
def load_dataset(path) -> DatasetLoaderInterface:
    if isinstance(path, dict):
        mds = load_config(path, default_type=Metadataset)
        mds.post_initialize()
        return mds
    
    path = EPath(path)
    ds_type = get_dataset_type(path)
    
    if ds_type == EnergonDatasetType.METADATASET:
        mds = load_config(path, default_type=Metadataset)
        mds.post_initialize()
        return mds
    elif ds_type in (WEBDATASET, JSONL):
        ds = DatasetLoader(path=path)
        ds.post_initialize()
        return ds
```

**关键**：`get_dataset_type` 通过检查路径下是否有 `.nv-meta/` 目录或特定文件来判断类型。

## 5. DatasetLoaderInterface — 统一接口（loader_interface.py）

### 5.1 设计动机（WHY）

所有数据集加载器需要统一接口，支持递归组合和 SavableDataLoader 统一调度。

### 5.2 核心抽象

```python
class DatasetLoaderInterface(ABC):
    @abstractmethod
    def get_datasets(
        self, *, training: bool, split_part: str,
        worker_config: WorkerConfig, subflavors: Dict, 
        shuffle_over_epochs_multiplier: int,
        dataset_config: Optional[DatasetConfig] = None,
    ) -> LoadedDatasetList: ...
    
    @abstractmethod
    def post_initialize(self) -> None: ...
```

### 5.3 LoadedDataset 数据结构

```python
@dataclass
class LoadedDataset:
    dataset: SavableDataset        # 底层可保存迭代器
    num_samples: int               # 样本总数
    weight: float                  # blend 权重
    aux_datasets: Dict[str, LoadedDataset]  # 辅助数据集

@dataclass
class LoadedDatasetList:
    datasets: List[LoadedDataset]
    blend_mode: DatasetBlendMode   # 混合方式
    blend_per_worker: bool = False # 是否每 worker 独立 blend
```

## 6. 与 Megatron 并行训练的集成

### 6.1 设计动机（WHY）

Metadataset 加载出的数据最终需要进入 Megatron 的分布式训练循环。在 TP/PP/DP 并行模式下，数据分发必须满足：
- **DP**：每个 DP rank 读取不同 shard（由 WorkerConfig 的 rank/world_size 控制）
- **TP**：TP rank 0 执行实际数据加载，其余 rank 通过 `broadcast_data()` 接收
- **PP**：只有 first pipeline stage (`pre_process=True`) 执行数据加载

### 6.2 get_batch 中的并行处理实现

```python
# Megatron get_batch 集成 Energon 的标准实现
def get_batch(data_iterator):
    """完整的并行感知数据获取实现"""
    # PP guard：仅 first stage 加载数据
    if mpu.is_pipeline_first_stage():  # pre_process guard
        # TP guard：仅 TP rank 0 实际从 iterator 取数据
        if mpu.get_tensor_model_parallel_rank() == 0:
            batch = next(data_iterator)  # Energon LoadedDataset 提供
        else:
            batch = None
        # TP broadcast：rank 0 广播到其余 TP ranks
        batch = broadcast_data(
            keys=["tokens", "labels", "loss_mask", "position_ids"],
            data=batch,
            datatype=torch.int64,
            src=mpu.get_tensor_model_parallel_src_rank()
        )
    else:
        # PP 非 first stage (post_process stage) 不需要输入数据
        batch = None
    return batch
```

### 6.3 WorkerConfig 与 DP 分片

Metadataset 通过 `worker_config` 感知 DP 并行度：

```python
worker_config = WorkerConfig(
    rank=mpu.get_data_parallel_rank(),
    world_size=mpu.get_data_parallel_world_size(),
    num_workers=num_workers,
)
# Sharder 根据 rank/world_size 自动分配不重叠的 shard 子集
# 确保每个 DP rank 只读取自己负责的数据分片
```

### 6.4 并行模式下的数据流

```
┌─────────────────────────────────────────────────┐
│ DP rank 0                                        │
│  ┌─────────────────────────────────────────┐    │
│  │ PP stage 0 (pre_process=True)           │    │
│  │  TP rank 0: Energon iter → batch        │    │
│  │  TP rank 1: broadcast_data() ← batch    │    │
│  │  TP rank 2: broadcast_data() ← batch    │    │
│  └─────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────┐    │
│  │ PP stage 1 (post_process=True)          │    │
│  │  所有 TP ranks: batch = None            │    │
│  │  数据通过 PP send/recv 从 stage 0 获得   │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────┐
│ DP rank 1  (独立 shard 子集)                     │
│  ... 相同结构，不同数据 ...                      │
└─────────────────────────────────────────────────┘
```


## 7. Metadataset V1 — 基础混合（metadataset.py）

### 7.1 配置格式示例

```yaml
# metadataset.yaml
__module__: megatron.energon
__class__: Metadataset
splits:
  train:
    datasets:
      - weight: 0.7
        path: ./text_dataset
      - weight: 0.3
        path: ./code_dataset
        subflavors:
          source: "code"
  val:
    datasets:
      - path: ./val_dataset
```

### 7.2 DatasetReference — 数据集引用

```python
@edataclass
class DatasetReference:
    path: Union[str, EPath]           # 数据集路径
    split_part: Optional[str] = None  # 覆盖 split
    subflavors: Optional[Dict] = None # Cooker 匹配条件
    shuffle_over_epochs_multiplier: Optional[int] = 1
    
    def get_datasets(self, *, training, split_part, worker_config, ...):
        # 递归调用：加载引用的 dataset/metadataset
        return self._dataset.get_datasets(
            training=training,
            split_part=self.split_part or split_part,
            worker_config=worker_config,
            subflavors=subflavors,
            shuffle_over_epochs_multiplier=merged_multiplier,
        )
```

### 7.3 shuffle_over_epochs_multiplier 乘法语义

嵌套引用时 multiplier 逐级相乘：
- 外层 metadataset: multiplier=3
- 内层 DatasetReference: multiplier=2
- 最终效果：shuffle_over_epochs = 3 × 2 = 6

特殊值：
- `None` → 不 shuffle（最高优先级）
- `-1` → 无限 shuffle（有放回采样）
- 正整数 N → N epoch 内每个 sample 恰好出现 N 次

## 8. MetadatasetV2 — 完整特性（metadataset_v2.py）

### 8.1 设计动机（WHY）

V1 仅支持简单 weight blend。V2 新增：
- **subset**：取数据集的部分范围（如 0%-50%）
- **aux datasets**：主数据集关联辅助数据（如 text+caption pair）
- **join**：按 key 关联多个数据集
- **blend_epochized**：epoch 边界感知的混合

### 8.2 配置格式

```yaml
__module__: megatron.energon
__class__: MetadatasetV2
datasets:
  - path: ./large_corpus
    weight: 0.6
    subset:
      start: 0%
      end: 80%
  - path: ./high_quality
    weight: 0.4
    subflavors:
      quality: "high"
    aux:
      captions:
        path: ./caption_dataset
blend_epochized: true
```

### 8.3 核心差异对比

| 特性 | V1 (Metadataset) | V2 (MetadatasetV2) |
|------|-------------------|---------------------|
| splits 结构 | splits.train.datasets[] | 顶层 datasets[] |
| subset 选择 | ❌ | ✅ start/end (绝对/百分比) |
| aux datasets | ❌ | ✅ 按名称关联 |
| join | ❌ | ✅ JoinDatasetLoader |
| blend_epochized | ❌ | ✅ epoch 对齐混合 |
| 嵌套层级 | DatasetReference 递归 | 直接 datasets 列表 |

## 9. JoinDatasetLoader — 多数据集关联

### 9.1 设计动机（WHY）

某些任务需要将不同数据集的样本按共享 key 对齐。例如：
- 图像数据集 + 标注数据集（按 image_id 关联）
- 多语言平行语料（按 sentence_id 关联）

### 9.2 Join 策略

JoinDatasetLoader 支持 inner/outer join 语义：
- **inner join**：只产出在所有数据集中都有匹配的样本
- **outer join**：缺失字段填充默认值

## 10. 设计决策对比

| 维度 | Energon Metadataset | HF datasets.interleave | 手动 ConcatDataset |
|------|---------------------|------------------------|---------------------|
| 配置方式 | YAML 声明式 | Python 代码 | Python 代码 |
| 权重混合 | ✅ 原生 | ✅ probabilities | ❌ 需自己实现 |
| 递归嵌套 | ✅ DatasetReference | ❌ | ❌ |
| 分布式感知 | ✅ WorkerConfig | ❌ | ❌ |
| 状态可保存 | ✅ SavableDataset | ❌ | ❌ |
| 子集选择 | ✅ subset start/end | ✅ split | ✅ Subset |
| 辅助数据集 | ✅ aux | ❌ | ❌ |
