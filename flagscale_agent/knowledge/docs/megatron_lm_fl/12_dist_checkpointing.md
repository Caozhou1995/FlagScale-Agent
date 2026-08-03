# 第12章：分布式 Checkpoint 系统深度源码分析

## 1. 概述与设计动机

### 1.1 解决什么问题

大模型训练中 checkpoint 面临三大挑战：

1. **并行度耦合**：传统 checkpoint 将 TP/PP/DP 分布信息硬编码进文件结构，改变并行策略需要重新转换
2. **写入阻塞**：同步 checkpoint 在万亿参数模型下需数分钟 I/O，直接停顿训练
3. **读写扩展性**：所有 rank 同时读写同一目录导致文件系统瓶颈

### 1.2 核心设计思想

Megatron-LM-FL 的 dist_checkpointing 系统基于一个核心抽象：**ShardedTensor — 将本地张量与全局张量的映射关系显式编码**。每个 rank 声明"我持有的是全局张量 X 的哪个切片"，存储层根据此映射自动完成 resharding、去重、分布式读写。

### 1.3 与其他系统的关系

```
训练代码 (model.state_dict())
    │
    ▼ sharded_state_dict() — 添加 ShardedTensor 注解
dist_checkpointing.save/load  — 协调层
    │
    ▼ Strategy（可替换后端）
PyTorch DCP (torch.distributed.checkpoint) — 实际 I/O
    │
    ▼
FileSystem / NVRx AsyncWriter / MSC (Multi-Storage Client)
```

## 2. 源码定位

| 组件 | 文件路径 | 行数 | 核心职责 |
|------|----------|------|----------|
| 元数据管理 | `core/dist_checkpointing/core.py` | 93 | CheckpointingConfig, metadata.json |
| 分片映射 | `core/dist_checkpointing/mapping.py` | 559 | ShardedTensor, ShardedObject, ShardedTensorFactory |
| 序列化入口 | `core/dist_checkpointing/serialization.py` | 420 | save(), load() 顶层 API |
| PyTorch DCP 策略 | `core/dist_checkpointing/strategies/torch.py` | 1097 | TorchDistSave/LoadShardedStrategy |
| 全并行策略 | `core/dist_checkpointing/strategies/fully_parallel.py` | 529 | FullyParallelSave/LoadStrategyWrapper |
| 异步写入 | `core/dist_checkpointing/strategies/async_utils.py` | 729 | AsyncRequest, AsyncCallsQueue |
| 文件系统异步写 | `core/dist_checkpointing/strategies/filesystem_async.py` | 678 | FileSystemWriterAsync |
| 数据交换 | `core/dist_checkpointing/exchange_utils.py` | 590 | ShardDistribution, 贪心分配算法 |
| Optimizer 映射 | `core/dist_checkpointing/optimizer.py` | 150 | optimizer state → ShardedTensor |
| 训练层集成 | `training/checkpointing.py` | 2093 | save/load_checkpoint 训练级封装 |


## 3. 架构总览

### 3.1 类继承与协作关系

```
ShardedBase (ABC)
├── ShardedTensor        — 分片张量映射（核心）
├── ShardedObject        — 分片非张量对象（如 optimizer param_groups）
└── ShardedTensorFactory — 延迟构建/合并（用于 optimizer state）

Strategy Layer:
TorchDistSaveShardedStrategy ← FullyParallelSaveStrategyWrapper (装饰器模式)
TorchDistLoadShardedStrategy ← FullyParallelLoadStrategyWrapper (装饰器模式)

Async Layer:
AsyncRequest (NamedTuple: async_fn, args, finalize_fns, preload_fn)
├── TemporalAsyncCaller  — 每次 fork 新进程
└── PersistentAsyncCaller — 持久后台进程 (NVRx)
```

### 3.2 Save 数据流

```
model.sharded_state_dict()          # Step 0: 模型生成 ShardedStateDict
        │
        ▼
serialization.save()                 # Step 1: 入口
        │
        ├─ save_preprocess()         # Step 2: 分离 sharded vs common
        │   ├─ apply_factories       #   展开 ShardedTensorFactory
        │   ├─ extract ShardedBase   #   提取 ShardedTensor/Object
        │   └─ validate integrity    #   验证每个 shard 恰好有一个 main replica
        │
        ├─ save_common()             # Step 3: Rank 0 保存 common.pt (args, iteration等)
        │
        ├─ strategy.async_save()     # Step 4: 分布式保存 sharded tensors
        │   ├─ _replace_state_dict_keys_with_sharded_keys  # 按 ShardedTensor.key 分组
        │   ├─ mcore_to_pyt_state_dict()                   # MCore → PyT ShardedTensor 转换
        │   ├─ MCoreSavePlanner.create_local_plan()        # 生成本地写入计划
        │   └─ FileSystemWriterAsync.write_data()          # 数据写入（可异步）
        │
        └─ metadata_finalize_fn()    # Step 5: 写 metadata.json（仅异步完成后）
```

### 3.3 Load 数据流

```
serialization.load(sharded_state_dict, checkpoint_dir)
        │
        ├─ force_all_tensors_to_non_fp8()   # FP8 → 高精度（避免量化问题）
        ├─ load_common()                     # 加载 common.pt
        ├─ load_preprocess()                 # 处理 factories, nonpersistent
        ├─ validate_integrity_and_strict_load()  # 验证 + strict 模式处理
        │
        └─ strategy.load()                   # 分布式加载
            ├─ _replace_state_dict_keys_with_sharded_keys
            ├─ mcore_to_pyt_state_dict(is_loading=True)  # 初始化空 tensor
            ├─ MCoreLoadPlanner.create_local_plan()       # 验证 global shape
            └─ FileSystemReader.read_data()              # 读取数据到本地 tensor
```

## 4. 核心模块分析

### 4.1 ShardedTensor — 分片映射核心

#### 4.1.1 设计动机（WHY）

传统 checkpoint 按"文件=rank"的方式存储，加载时必须匹配并行度。ShardedTensor 将这种隐式映射**显式化**：

- `key`: 全局张量唯一标识（如 `"model.layers.0.attention.qkv.weight"`）
- `global_shape`: 全局完整张量形状
- `global_offset`: 本地 shard 在全局张量中的偏移
- `local_shape`: 本地持有的形状
- `axis_fragmentations`: 每轴的切分数
- `replica_id`: 标识是否为主副本（用于去重）

#### 4.1.2 实现分析

关键构造方法 `from_rank_offsets` (mapping.py:189-245):

```python
# 使用示例：TP=4, 当前 rank 持有第 2 个切片
ShardedTensor.from_rank_offsets(
    key="layers.0.qkv.weight",
    data=local_weight,           # shape: [hidden_size, hidden_size/4]
    (0, tp_rank, tp_size),       # axis=0, offset=tp_rank, fragmentation=tp_size
    replica_id=dp_rank           # DP 副本标识（非0的不写入）
)
# → global_shape = [hidden_size, hidden_size]
# → global_offset = [tp_rank * hidden_size/4, 0]
# → axis_fragmentations = [4, 1]
```

#### 4.1.3 Resharding 机制

加载时，ShardedTensor 的 `global_offset` 和 `local_shape` 由**当前模型**决定，而非 checkpoint 文件。PyTorch DCP 会自动根据当前 ShardedTensor 的元数据从 checkpoint 文件中读取对应区域：

```
保存时: TP=2, global_shape=[8192, 4096], local_shape=[4096, 4096], offset=[0, 0]
加载时: TP=4, global_shape=[8192, 4096], local_shape=[2048, 4096], offset=[0, 0]
→ DCP 自动只读取 [0:2048, 0:4096] 区域
```

#### 4.1.4 边界条件

- `flattened_range`: 已废弃（L134: raises exception），历史上用于 DistributedOptimizer 的 1D flatten
- `allow_shape_mismatch`: padded tensor 支持（如 vocab embedding padding）
- `prepend_axis_num`: 用于 DP 副本的额外维度标记

### 4.2 ShardedTensorFactory — Optimizer State 映射

#### 4.2.1 设计动机（WHY）

Optimizer state (如 Adam 的 exp_avg, exp_avg_sq) 与 model parameter 形状相同，分片方式也相同。但 optimizer 并不直接持有 ShardedTensor。ShardedTensorFactory 解决这一问题：**用一个工厂函数将 model parameter 的分片信息"复制"给 optimizer state**。

#### 4.2.2 实现分析 (mapping.py:437-499)

```python
@dataclass
class ShardedTensorFactory(ShardedBase):
    key: str
    data: torch.Tensor          # 原始模型参数
    build_fn: FactoryBuildFn    # 保存时: param → sharded sub-state-dict
    merge_fn: FactoryMergeFn    # 加载时: loaded sub-dict → merged tensor
```

配合 optimizer.py:83-108 的 `make_sharded_optimizer_tensor`:

```python
def make_sharded_optimizer_tensor(model_param, optim_param, prefix):
    # 直接复用 model_param 的分片元数据，仅替换 key 和 data
    return replace(model_param, 
                   key=f'{prefix}.{model_param.key}', 
                   data=optim_param, 
                   dtype=optim_param.dtype)
```

#### 4.2.3 完整流程

```
保存: optim_state_to_sharding_state() 
  → 对每个 param_id 的 exp_avg/exp_avg_sq
  → make_sharded_optimizer_tensor(model_sh_ten, optim_tensor, "optimizer.state.exp_avg")
  → 生成 ShardedTensor(key="optimizer.state.exp_avg.layers.0.qkv.weight", ...)

加载: apply_factory_merges()
  → 读取各 optimizer sub-tensor
  → merge_fn 还原完整 state
```

### 4.3 TorchDistSaveShardedStrategy — 保存策略

#### 4.3.1 设计动机（WHY）

MCore 的 ShardedTensor 需要转换为 PyTorch DCP 理解的格式。策略层封装了这个转换，同时支持同步/异步两种模式。

#### 4.3.2 核心流程 (torch.py:593-792)

```
async_save(sharded_state_dict, checkpoint_dir):
  1. _replace_state_dict_keys_with_sharded_keys()
     → 按 ShardedTensor.key 分组，处理 replica_id 去重
  
  2. mcore_to_pyt_state_dict(sharded_state_dict, is_loading=False)
     → ShardedTensor → CheckpointableShardedTensor (PyT 2.6+)
     → 或 → TorchShardedTensor (legacy)
  
  3. get_async_strategy() → 选择 "nvrx" 或 "mcore" 异步实现
  
  4. FileSystemWriterAsync(checkpoint_dir, thread_count, separation_hint)
  
  5. save_state_dict_async_plan(pyt_state_dict, writer, planner=MCoreSavePlanner)
     → 返回 AsyncRequest
  
  6. _get_save_and_finalize_callbacks() → 包装为最终 AsyncRequest
```

#### 4.3.3 关键优化

- **cached_metadata**: 首次 save 后缓存 central_plan/local_plan，后续 save 跳过全局通信 (L618-646)
- **separation_hint**: 将特定前缀的 tensor 写入独立文件（如分离 model 和 optimizer）
- **keep_only_main_replica**: replica_id != 0 的不写入磁盘，节省空间和 I/O

### 4.4 异步 Checkpoint 机制

#### 4.4.1 设计动机（WHY）

同步 checkpoint 流程：

```
训练 → [停顿] GPU→CPU拷贝 + 磁盘写入 [/停顿] → 训练
```

异步流程：

```
训练 → [短暂停顿] GPU→CPU 拷贝 (preload) [/停顿] → 训练继续
                          ↓
         后台进程: CPU→磁盘写入 (async_fn)
                          ↓
         下次 save 前: finalize (确认完成, 写 metadata.json)
```

#### 4.4.2 AsyncRequest 结构 (async_utils.py:104-128)

```python
class AsyncRequest(NamedTuple):
    async_fn: Optional[Callable]      # 后台执行的写入函数
    async_fn_args: Tuple              # 写入参数
    finalize_fns: List[Callable]      # 完成后的回调（写 metadata.json 等）
    preload_fn: Optional[Callable]    # GPU→CPU 预加载函数
    is_frozen: bool                   # 冻结标志
    call_idx: int                     # 排序索引
```

#### 4.4.3 FileSystemWriterAsync 流程 (filesystem_async.py:76-120)

```
1. write_data() — 在主进程中
   → 生成 write_buckets: List[WriteBucket]
   → 每个 bucket = (file_path, storage_key, (write_items, tensor_data))
   
2. get_save_function_and_args() → (save_fn, preload_fn, args)
   → preload_fn: GPU tensor → pinned CPU tensor (D2H copy)
   → save_fn: writer_proxy_func (多线程写入磁盘)

3. 外部调度:
   → preload_fn()    # 主进程短暂停顿，完成 D2H
   → fork process   # 后台执行 save_fn
   → 训练继续

4. finalize:
   → 确认写入完成
   → 调用 finalize_fns (写 metadata.json)
```

#### 4.4.4 QoS 控制 (async_utils.py:35-88)

后台写入进程自动降低优先级，避免干扰训练：
- CPU: `nice` 值设为 10（中度降低优先级）
- I/O: `ionice` 设为 idle 类（仅在无其他 I/O 时写入）

#### 4.4.5 两种异步实现对比

| 维度 | MCore 实现 | NVRx (nvidia_resiliency_ext) |
|------|-----------|------------------------------|
| 进程模型 | TemporalAsyncCaller (每次 fork) | PersistentAsyncCaller (持久进程) |
| 元数据缓存 | 手动管理 cached_central_plan | CheckpointMetadataCache 封装 |
| 全局计划 | 需要 coordinator 汇聚 | 支持 decentralized_global_plan |
| 状态 | 默认，即将废弃 | 推荐方案 (async_strategy="nvrx") |
| 进程开销 | 每次 fork 有启动开销 | 一次启动，后续复用 |

### 4.5 FullyParallelSaveStrategyWrapper — 分布式写入优化

#### 4.5.1 设计动机（WHY）

默认情况下，DP ranks 持有相同参数副本（replica），但只有 replica_id=0 的 rank 执行写入。这意味着 N 个 DP rank 中只有 1 个在写，其余空闲等待。FullyParallel 策略将写入**均匀分配**给所有 replica rank，使 I/O 并行度提升 N 倍。

#### 4.5.2 核心算法 (fully_parallel.py:53-146, exchange_utils.py:123-150)

**Save 并行化**:
```
1. determine_main_replica_uniform_distribution(state_dict, group)
   → 收集所有 rank 的 shard 元数据
   → 调用 distribute_shards_to_ranks() 贪心分配

2. distribute_shards_to_ranks():  (exchange_utils.py:123)
   排序优先级:
   a) 跨并行组依赖（延后处理，确保 save/load 分布相似）
   b) 覆盖率（shard 在越少 rank 上有副本 → 越先分配）
   c) 大小（越大的 shard → 越先分配到负载最轻的 rank）
   d) shard_id（确定性打破平局）
   
   → 贪心: 每次将最大未分配 shard 分配给当前负载最小的 rank

3. distribute_main_replicas_with_precomputed_distribution():
   → 被分配到当前 rank 的 shard: replica_id = 0 (会写入)
   → 其余: replica_id = 1 (跳过写入)
```

#### 4.5.3 Load 并行化 (fully_parallel.py:149-305)

```
FullyParallelLoadStrategyWrapper.load():
  Step 1: apply_loading_parallelization() — 计算谁加载哪些 shard
  Step 2: _defer_loading_sharded_tensors() — 分为 to_load 和 unloaded
  Step 3: base_strategy.load(to_load_shards) — 每个 rank 只读自己分配的部分
  Step 4: exchange_by_distribution() — rank 间交换数据
          支持算法: broadcast / gather_object / gather_rounds
```

#### 4.5.4 性能分析

假设 DP=8, 每个 rank 持有完整模型副本:
- **无 FullyParallel**: 1 个 rank 写全部 shards → I/O 时间 = T
- **有 FullyParallel**: 8 个 rank 各写 1/8 → I/O 时间 ≈ T/8 + 通信开销

通信开销仅在 Load 时存在（需要 exchange），Save 时只交换 metadata（KB 级别）。

### 4.6 MCore ↔ PyTorch DCP 转换层

#### 4.6.1 设计动机（WHY）

MCore 的 ShardedTensor 与 PyTorch 的 `torch.distributed._shard.ShardedTensor` 是不同的类。转换层将 MCore 语义映射到 PyT DCP 的存储格式，使得底层可以复用 PyT 的 FileSystemReader/Writer。

#### 4.6.2 转换流程 (torch.py:141-337)

```python
def mcore_to_pyt_state_dict(state_dict, is_loading):
    for k, sh_tens in state_dict.items():
        if isinstance(sh_tens[0], ShardedTensor):
            # PyTorch >= 2.6: 直接使用 CheckpointableShardedTensor
            # PyTorch < 2.6: 转为 TorchShardedTensor (legacy路径)
            if not is_pre_mcore_014 and is_torch_min_version("2.6a0"):
                pyt_state_dict[k] = CheckpointableShardedTensor.from_sh_ten(sh_ten)
            else:
                pyt_state_dict[k] = sharded_tensor_to_torch_sharded_tensor(sh_tens, rank)
```

`sharded_tensor_to_torch_sharded_tensor` (torch.py:141-245):
- 假设 regular grid sharding
- 构建全部 shard 的 metadata（本地 + 远端 rank 的 placeholder）
- 使用 `_init_from_local_shards_and_global_metadata` 避免通信

#### 4.6.3 FP8 支持

MCoreLoadPlanner 特殊处理 Float8Tensor (torch.py:558-590):
- `resolve_tensor`: 如果 FP8 tensor 不连续 → 创建连续副本
- `commit_tensor`: 写回原始 tensor
- 原因：Float8Tensor 的 narrow 操作可能产生不连续内存，DCP 的 copy_ kernel 不支持

### 4.7 训练层集成 (training/checkpointing.py)

#### 4.7.1 save_checkpoint 关键路径

```python
def save_checkpoint(iteration, model, optimizer, opt_param_scheduler, ...):
    # 1. 确定路径
    ckpt_dir = get_checkpoint_name(..., return_base_dir=True)
    
    # 2. 构建 state_dict
    state_dict = generate_state_dict(...)  # model + optimizer + rng
    
    # 3. 选择策略
    if args.ckpt_fully_parallel_save:
        strategy = FullyParallelSaveStrategyWrapper(
            TorchDistSaveShardedStrategy(cached_metadata=True),
            parallelization_group=mpu.get_data_parallel_group()
        )
    
    # 4. 调用 dist_checkpointing.save
    if args.async_save:
        async_request = dist_checkpointing.save(
            state_dict, ckpt_dir, strategy, async_sharded_save=True)
        schedule_async_save(async_request)  # 提交到后台队列
    else:
        dist_checkpointing.save(state_dict, ckpt_dir, strategy)
    
    # 5. 管理历史 checkpoint (保留最近 N 个)
    cleanup_old_checkpoints(...)
```

#### 4.7.2 load_checkpoint 关键路径

```python
def load_checkpoint(model, optimizer, ...):
    # 1. 确定加载路径和迭代
    iteration, release = read_metadata(tracker_filename)
    
    # 2. 构建 sharded_state_dict (空壳，仅有元数据)
    state_dict = generate_state_dict(..., is_loading=True)
    
    # 3. 选择加载策略
    if args.ckpt_fully_parallel_load:
        strategy = FullyParallelLoadStrategyWrapper(
            TorchDistLoadShardedStrategy(),
            parallelization_group=mpu.get_data_parallel_group()
        )
    
    # 4. 加载
    loaded = dist_checkpointing.load(state_dict, ckpt_dir, strategy)
    
    # 5. 恢复状态
    model.load_state_dict(loaded['model'])
    optimizer.load_state_dict(loaded['optimizer'])
    # ... rng states, iteration, etc.
```

## 5. 性能与通信量化分析

### 5.1 Checkpoint 大小估算

对于典型 LLM (参数量 P, BF16 训练, Adam 优化器):
- Model params: P × 2 bytes (BF16)
- Optimizer state: P × 4 bytes (exp_avg, FP32) + P × 4 bytes (exp_avg_sq, FP32) = P × 8 bytes
- 总计: P × 10 bytes

| 模型规模 | 参数量 | Checkpoint 大小 |
|----------|--------|-----------------|
| 7B | 7×10⁹ | ~70 GB |
| 70B | 70×10⁹ | ~700 GB |
| 405B | 405×10⁹ | ~4 TB |

### 5.2 I/O 时间估算

| 存储类型 | 带宽 | 70B 模型写入时间 | 使用 FullyParallel (DP=8) |
|----------|------|------------------|---------------------------|
| NVMe SSD | 3 GB/s | 233s | 29s |
| Lustre PFS | 10 GB/s | 70s | 9s |
| NVMe RAID (4x) | 12 GB/s | 58s | 7s |

### 5.3 异步 Checkpoint 的训练停顿

```
同步:  停顿 = D2H 拷贝 + 磁盘写入  (数十秒到数分钟)
异步:  停顿 = D2H 拷贝 only ≈ P × 10 / PCIe_BW

PCIe 5.0 x16: ~64 GB/s
70B 模型: 700 GB / 64 GB/s ≈ 11s 停顿 (vs 70s+ 同步)
```

实际中 preload_fn 使用 pinned memory + CUDA stream，进一步减少停顿。

### 5.4 FullyParallel 通信开销

**Save**: 仅交换 metadata (shard_id + size)，通常 < 1 MB per rank
**Load**: 需要 exchange_by_distribution
- broadcast 模式: 每个 rank 广播自己加载的 tensors → O(N × shard_size)
- gather_rounds 模式: 按轮次 all_gather → 内存峰值可控

## 6. 设计决策对比表

### 6.1 存储策略对比

| 维度 | 朴素 per-rank 存储 | MCore dist_checkpointing | PyTorch FSDP 存储 |
|------|-------------------|--------------------------|-------------------|
| Resharding | 需要离线转换脚本 | 自动（加载时重映射） | 自动（DTensor） |
| 存储格式 | 单一 .pt 文件/rank | PyT DCP (.distcp) | PyT DCP |
| 异步写入 | 不支持 | 支持（NVRx/MCore） | 有限支持 |
| 分布式并行写 | 天然并行但无负载均衡 | FullyParallel 均匀分配 | 天然并行 |
| Optimizer state | 随 model 一起存储 | 独立映射，支持 reshape | DTensor 管理 |
| FP8 支持 | 需手动处理 | 内置 dequantize 逻辑 | 不支持 |

### 6.2 异步方案对比

| 维度 | 无异步 | MCore Temporal | NVRx Persistent |
|------|--------|----------------|-----------------|
| 训练停顿 | 全量 I/O 时间 | D2H 时间 | D2H 时间 |
| 进程管理 | 无 | 每次 fork | 持久后台进程 |
| 内存开销 | 无额外 | 临时双倍 CPU 内存 | 固定 CPU buffer |
| 元数据缓存 | 不适用 | 手动 plan 缓存 | CheckpointMetadataCache |
| 推荐场景 | 小模型/低频 ckpt | 兼容性需求 | 万卡训练（推荐） |

## 7. FlagScale 扩展

FlagScale 对 dist_checkpointing 的修改集中在硬件抽象层:

1. **平台抽象** (torch.py:54-58, fully_parallel.py:43-47):
   ```python
   from megatron.plugin.platform import get_platform
   cur_platform = get_platform()
   ```
   所有 `torch.cuda.synchronize()` → `cur_platform.synchronize()`
   所有 `"cuda"` 设备引用 → `cur_platform.device_name()`

2. **TE 可用性判断** (torch.py:74-80):
   ```python
   if not cur_platform.is_available():
       raise ImportError  # 非 NVIDIA 平台跳过 TE
   ```

3. **exchange_utils.py 设备适配** (L103-115):
   空 tensor 分配和 D2H 使用 `cur_platform.device_name()` 替代硬编码 "cuda"

这些修改使 checkpoint 系统可在非 NVIDIA 硬件（如昇腾）上运行。

## 8. 配置建议与调优指南

### 8.1 关键配置参数

| 参数 | 默认值 | 建议 | 说明 |
|------|--------|------|------|
| `--use-dist-ckpt` | False | True | 启用分布式 checkpoint |
| `--async-save` | False | True | 异步保存减少停顿 |
| `--ckpt-fully-parallel-save` | False | True | 多 rank 并行写入 |
| `--ckpt-fully-parallel-load` | False | True | 多 rank 并行读取 |
| `--save-interval` | - | 根据故障率设定 | 过频影响训练 |

### 8.2 大模型训练推荐配置

```yaml
# 70B+ 模型推荐
use_dist_ckpt: true
async_save: true
ckpt_fully_parallel_save: true
ckpt_fully_parallel_load: true
# FullyParallel 的 parallelization_group 自动使用 DP group
```

### 8.3 注意事项

1. **异步 save 的内存压力**: preload 需要额外 CPU 内存存放整个 checkpoint 副本
2. **metadata.json 写入时机**: 异步模式下仅在 finalize 时写入，crash 前未 finalize 的 checkpoint 不完整
3. **FP8 checkpoint**: 加载时自动 dequantize，确保 main params 精度正确
4. **strict 模式**: 默认 `ASSUME_OK_UNEXPECTED`（不额外通信），生产环境可用 `RETURN_ALL` 排查 mismatch
5. **dist_checkpointing 与 non-dist 不兼容**: 一旦启用，不能回退到 per-rank .pt 格式

## 9. 总结

dist_checkpointing 系统的核心设计理念是**将分布式拓扑信息从存储格式中解耦**：

1. **ShardedTensor** 显式编码 local↔global 映射，使 resharding 变为自动操作
2. **Strategy 模式** 将 I/O 后端可插拔化，支持同步/异步、单点/全并行
3. **ShardedTensorFactory** 使 optimizer state 能复用 model 的分片信息
4. **FullyParallel** wrapper 利用 DP 冗余实现 I/O 负载均衡
5. **AsyncRequest** 抽象将 GPU→CPU→Disk 的流水线显式化

这套系统使万卡训练中的 checkpoint 从"训练瓶颈"变为"几乎透明的后台操作"。
