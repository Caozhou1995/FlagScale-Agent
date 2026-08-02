# PyTorch Distributed 源码深度分析 — 第3章：FSDP (Fully Sharded Data Parallel)

## 1. 设计动机

### 1.1 WHY FSDP？DDP 的内存瓶颈

```
DDP 内存开销分析 (以 7B 参数模型为例):
─────────────────────────────────────────────
参数 (FP32):        7B × 4B = 28 GB
梯度 (FP32):        7B × 4B = 28 GB
优化器状态 (Adam):  7B × 8B = 56 GB (momentum + variance)
───────────────────────────────────────
总计每 GPU:         112 GB  > H100 80GB!

DDP: 每个 rank 持有完整的参数 + 梯度 + 优化器状态
FSDP (ZeRO-3): 每个 rank 只持有 1/N 的参数 + 梯度 + 优化器状态

8 GPU FSDP 内存:
参数 shard:         28/8 = 3.5 GB
梯度 shard:         28/8 = 3.5 GB
优化器 shard:       56/8 = 7 GB
临时 AllGather:     28 GB (一层的参数, 用完释放)
───────────────────────────────────────
总计每 GPU:         ~42 GB ✓ 可以训练!
```

### 1.2 ZeRO 论文映射

| ZeRO Stage | 分片内容 | FSDP 对应 | 内存节省 |
|------------|---------|-----------|---------|
| Stage 1 | 优化器状态 | - | Ndata × (OS) |
| Stage 2 | 优化器 + 梯度 | SHARD_GRAD_OP | Ndata × (OS+G) |
| Stage 3 | 优化器 + 梯度 + 参数 | FULL_SHARD | Ndata × (OS+G+P) |

## 2. ShardingStrategy 分片策略 (api.py L1-100)

```python
# torch/distributed/fsdp/api.py
class ShardingStrategy(Enum):
    FULL_SHARD = auto()       # ZeRO-3: 参数+梯度+优化器全分片
    SHARD_GRAD_OP = auto()    # ZeRO-2: 只分片梯度+优化器
    NO_SHARD = auto()         # 等价于 DDP (不分片)
    HYBRID_SHARD = auto()     # 节点内 FULL_SHARD + 节点间 replicate
    _HYBRID_SHARD_ZERO2 = auto()  # 节点内 SHARD_GRAD_OP + 节点间 replicate
```

### 2.1 HYBRID_SHARD 设计 (L184-235)

```
WHY HYBRID_SHARD?
──────────────────
FULL_SHARD 32 GPU: AllGather 跨所有 32 GPU (含跨节点 IB)
HYBRID_SHARD 4节点×8GPU: 
  - 节点内 AllGather (NVLink 900GB/s) → 参数恢复
  - 节点间 AllReduce (IB 400Gb/s) → 梯度同步
  
优势: 利用节点内高带宽 NVLink, 减少跨节点通信
适用: 模型能放入单节点 (8×80GB=640GB) 但不能放入单卡

┌─ Node 0 ─────────────────────────────┐
│ GPU0  GPU1  GPU2  GPU3  GPU4...GPU7  │
│ ←── FULL_SHARD (NVLink AllGather) ──→│
└──────────────────────────────────────┘
         ↕ AllReduce (IB, 梯度)
┌─ Node 1 ─────────────────────────────┐
│ GPU0  GPU1  GPU2  GPU3  GPU4...GPU7  │
│ ←── FULL_SHARD (NVLink AllGather) ──→│
└──────────────────────────────────────┘
```

## 3. FSDP 生命周期

### 3.1 Forward 阶段

```
FSDP Forward 执行流 (FULL_SHARD):
═══════════════════════════════════════════════
                  ┌──────────────┐
                  │ 参数 (sharded)│  ← 平时只存 1/N
                  └──────┬───────┘
                         │ pre_forward hook
                         ▼
                  AllGather (收集完整参数)
                         │
                         ▼
                  ┌──────────────┐
                  │ 参数 (full)   │  ← 临时: 完整参数
                  └──────┬───────┘
                         │ forward computation
                         ▼
                  ┌──────────────┐
                  │ 输出 tensor   │
                  └──────┬───────┘
                         │ post_forward hook
                         ▼
                  释放完整参数 (reshard)
                  ┌──────────────┐
                  │ 参数 (sharded)│  ← 恢复为 1/N
                  └──────────────┘
```

### 3.2 Backward 阶段

```
FSDP Backward 执行流:
═══════════════════════════════════════════════
Layer N (最后层):
  1. pre_backward hook → AllGather (恢复完整参数)
  2. 计算梯度
  3. post_backward hook:
     - ReduceScatter (梯度分片, 每 rank 只保留 1/N 梯度)
     - 释放完整参数 (reshard)
     
Layer N-1:
  1. [Prefetch] AllGather (与 Layer N 的 ReduceScatter overlap)
  2. 计算梯度
  3. ReduceScatter + reshard
  ...

关键优化: backward_prefetch
  BACKWARD_PRE:  Layer N 计算时预取 Layer N-1 的参数
  BACKWARD_POST: Layer N 完成后才取 Layer N-1 (节省内存但无overlap)
```

## 4. 通信模式

### 4.1 通信量对比

| 策略 | Forward | Backward | 总通信量 |
|------|---------|----------|---------|
| DDP | 0 | AllReduce(2P) | 2P |
| FSDP FULL_SHARD | AllGather(P) | AllGather(P) + ReduceScatter(P) | 3P |
| FSDP SHARD_GRAD_OP | AllGather(P) | ReduceScatter(P) | 2P |

```
WHY FSDP 通信量更大但仍然值得?
─────────────────────────────────
1. 内存节省 → 可以训练更大模型 (or 更大 batch)
2. 通信可以与计算 overlap (prefetch)
3. NVLink 带宽充裕 (900GB/s), 通信不是瓶颈
4. 节点内 FULL_SHARD + 节点间 DDP (HYBRID) 兼顾两者
```

### 4.2 AllGather vs ReduceScatter

```
AllGather (Forward, 恢复参数):
  Rank 0: [S0] → [S0|S1|S2|S3]    收集所有 shard → 完整参数
  Rank 1: [S1] → [S0|S1|S2|S3]
  Rank 2: [S2] → [S0|S1|S2|S3]
  Rank 3: [S3] → [S0|S1|S2|S3]

ReduceScatter (Backward, 分片梯度):
  Rank 0: [G0|G1|G2|G3] → [sum(G0)]  归约后只保留自己的 shard
  Rank 1: [G0|G1|G2|G3] → [sum(G1)]
  Rank 2: [G0|G1|G2|G3] → [sum(G2)]
  Rank 3: [G0|G1|G2|G3] → [sum(G3)]
```

## 5. auto_wrap_policy (wrap.py)

### 5.1 WHY 需要 auto_wrap?

```
问题: 如果整个模型只用一个 FSDP 包装:
  - Forward: AllGather 所有参数 → 峰值内存 = 完整模型
  - 无法 overlap: 必须等所有参数就绪才能开始计算

解决: 将模型分层 FSDP 包装 (每层独立 AllGather/reshard):
  - 峰值内存 = 一层的完整参数
  - overlap: Layer N compute ↔ Layer N+1 AllGather

auto_wrap_policy: 自动按 module 类型或参数量决定包装粒度
```

### 5.2 常用策略

```python
# 按模块类型包装 (推荐):
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
policy = ModuleWrapPolicy({TransformerBlock})
# 每个 TransformerBlock 独立 FSDP 单元

# 按参数量包装:
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1e6)
```

## 6. CPU Offload

```python
# torch/distributed/fsdp/api.py
@dataclass
class CPUOffload:
    offload_params: bool = False  # 参数 reshard 后移到 CPU

# 效果:
# GPU 内存: 几乎只需一层的参数 + 激活
# 代价: PCIe 带宽成为瓶颈 (PCIe 5.0: ~64 GB/s vs NVLink 900GB/s)
```

## 7. Mixed Precision (L283-287)

```python
@dataclass  
class MixedPrecision:
    param_dtype: torch.dtype | None = None     # AllGather 后 cast 到此精度
    reduce_dtype: torch.dtype | None = None    # ReduceScatter 通信精度
    buffer_dtype: torch.dtype | None = None    # Buffer 精度

# 常用配置:
mp = MixedPrecision(
    param_dtype=torch.bfloat16,     # forward 用 BF16 (省激活内存)
    reduce_dtype=torch.float32,     # 梯度归约用 FP32 (保精度)
)
```

## 8. FSDP2 (torch.distributed._composable.fsdp)

```python
# 新版 composable API (fully_shard.py):
from torch.distributed._composable.fsdp import fully_shard

# 不再需要包装模型 (composable style):
for layer in model.layers:
    fully_shard(layer)
fully_shard(model)

# WHY FSDP2?
# 1. 非侵入式: 不改变 module 结构
# 2. 与其他 composable API 组合 (如 checkpoint, TP)
# 3. 更好的 DeviceMesh 集成
# 4. 支持 per-parameter sharding
```

## 9. 与 Megatron 对比

| 维度 | FSDP | Megatron DistributedOptimizer |
|------|------|------------------------------|
| 抽象层次 | 通用 wrapper | 训练框架内置 |
| 通信模式 | AllGather + ReduceScatter | ReduceScatter (overlap bucket) |
| 参数管理 | 运行时 unshard/reshard | 静态 shard mapping |
| 激活内存 | 由 FSDP 管理 | 由 PP/CP/recompute 管理 |
| 适用规模 | 中等 (数百 GPU) | 超大 (数千 GPU, 5D 并行) |
| 复杂度 | 低 (auto_wrap) | 高 (需手动配置并行策略) |

## 10. 总结

```
FSDP 核心价值:
┌────────────────────────────────────────────────────────┐
│ 1. 内存民主化: ZeRO-3 让单卡 80GB 可训练 70B+ 模型     │
│ 2. 自动化: auto_wrap + HYBRID_SHARD 降低使用门槛        │
│ 3. Overlap: AllGather prefetch 隐藏通信延迟             │
│ 4. 灵活性: ShardingStrategy 适配不同集群/模型规模       │
│ 5. Composable: FSDP2 支持与 TP/PP 自由组合              │
└────────────────────────────────────────────────────────┘
```

## 11. 限流机制 limit_all_gathers

```
WHY limit_all_gathers?
─────────────────────────
问题: prefetch 激进 → 同时多个 AllGather 进行中
      每个 AllGather 临时占用 P/N_layer 的完整参数内存
      多个叠加 → OOM

解决: limit_all_gathers=True
  - CPU 线程同步 → 确保上一个 AllGather 完成并 reshard 后才发下一个
  - 牺牲少量 overlap → 保证内存不超限

现象: CUDA timeline 中 pre_forward 会出现 CPU gap
      这是限流器在等待, 并非性能 bug
```

## 12. summon_full_params 上下文管理器

```python
# fully_sharded_data_parallel.py L1500+
@contextmanager
def summon_full_params(self, recurse=True, writeback=True, with_grads=False):
    """临时恢复完整参数用于保存/检查"""
    # 1. AllGather 恢复所有 shard → 完整参数
    # 2. yield (用户操作完整参数)
    # 3. writeback=True: 修改写回 shard
    # 4. reshard

# 用途:
# - checkpoint 保存 (state_dict)
# - 参数检查/打印
# - 评估 (eval)

with FSDP.summon_full_params(model, writeback=False):
    torch.save(model.state_dict(), "checkpoint.pt")
```

## 13. state_dict 策略

```python
# torch/distributed/fsdp/api.py
class StateDictType(Enum):
    FULL_STATE_DICT = auto()    # 收集完整参数 (兼容非 FSDP 加载)
    LOCAL_STATE_DICT = auto()   # 保存本地 shard (快速 checkpoint)
    SHARDED_STATE_DICT = auto() # 分布式保存 (DTensor format)

# FULL_STATE_DICT:
#   - 内部调用 summon_full_params
#   - 只在 rank 0 汇聚 → 写入单文件
#   - 优势: 兼容性好, 非 FSDP 可直接 load
#   - 劣势: rank 0 需要足够内存放完整模型
#
# SHARDED_STATE_DICT:
#   - 每个 rank 保存自己的 shard
#   - 使用 DTensor + dist_checkpointing
#   - 优势: 无内存峰值, 可变 world_size 加载
#   - 适用: 大模型生产训练
```

## 14. no_sync 梯度累积

```python
# 梯度累积模式:
# WHY no_sync? 
# 每个 micro step 都 ReduceScatter 太浪费
# 累积 N 步后再同步 → 通信量减少 N 倍

for i, batch in enumerate(dataloader):
    if i % accumulation_steps != 0:
        with model.no_sync():  # 跳过 ReduceScatter
            loss = model(batch).sum()
            loss.backward()    # 梯度本地累积
    else:
        loss = model(batch).sum()
        loss.backward()        # 触发 ReduceScatter
        optimizer.step()
        optimizer.zero_grad()
```

## 15. 关键源码文件索引

| 文件 | 行数 | 核心职责 |
|------|------|---------|
| fsdp/fully_sharded_data_parallel.py | 2167 | FSDP 主类 (v1) |
| fsdp/api.py | 416 | 配置数据类 (策略/精度/offload) |
| fsdp/wrap.py | 608 | auto_wrap 策略 |
| fsdp/sharded_grad_scaler.py | 377 | 分片 GradScaler |
| _composable/fsdp/fully_shard.py | ~500 | FSDP2 composable API |
| fsdp/_runtime_utils.py | ~800 | AllGather/ReduceScatter 调度 |
| fsdp/_flat_param.py | ~2000 | FlatParameter 参数扁平化 |

## 16. 性能调优建议

```python
# 最佳配置 (8×H100 单节点, 70B 模型):
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=ModuleWrapPolicy({TransformerBlock}),
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    ),
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    limit_all_gathers=True,
    use_orig_params=True,          # 支持 optimizer 按原始参数分组
    device_id=torch.cuda.current_device(),
)

# 多节点 (4×8 H100, IB 带宽有限):
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,  # 节点内 shard, 节点间 replicate
    # ... 其余同上
)
```
