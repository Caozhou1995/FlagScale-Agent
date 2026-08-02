# 第9章：优化器系统 (Optimizer System) 源码深度解析

## 1. 概述与源码定位

Megatron-LM-FL 优化器系统的核心职责是：管理混合精度梯度流、分布式状态分片、梯度裁剪/缩放、以及与通信 overlap 的协调。

### 1.1 核心源码文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `megatron/core/optimizer/optimizer.py` | ~1614 | 优化器类层次：MegatronOptimizer → MixedPrecisionOptimizer → Float16OptimizerWithFloat16Params |
| `megatron/core/optimizer/distrib_optimizer.py` | ~2741 | DistributedOptimizer：ZeRO-1/2 风格状态分片 |
| `megatron/core/optimizer/clip_grads.py` | ~309 | 梯度范数计算与裁剪 |
| `megatron/core/optimizer/grad_scaler.py` | ~130 | Loss scaling（ConstantGradScaler / DynamicGradScaler）|
| `megatron/core/optimizer/optimizer_config.py` | ~200 | OptimizerConfig dataclass |
| `megatron/core/optimizer/__init__.py` | ~450 | 工厂函数 get_megatron_optimizer，µP 支持 |
| `megatron/core/optimizer/cpu_offloading.py` | ~300 | HybridDeviceOptimizer：CPU offload Adam 状态 |

### 1.2 类继承体系

```
MegatronOptimizer (optimizer.py:115)
 ├── MixedPrecisionOptimizer (optimizer.py:480)
 │    ├── Float16OptimizerWithFloat16Params (optimizer.py:693)
 │    └── DistributedOptimizer (distrib_optimizer.py:103)
 └── ChainedOptimizer (chained_optimizer.py) — 多优化器组合
```

---

## 2. MegatronOptimizer 基类 (optimizer.py:115-478)

### 2.1 核心职责

基类定义了 Megatron 所有优化器的统一接口和公共逻辑：

```python
class MegatronOptimizer(ABC):
    def __init__(self, optimizer, config, init_state_fn):
        self.optimizer = optimizer          # 底层 torch.optim.Optimizer
        self.config = config                # OptimizerConfig
        self.init_state_fn = init_state_fn  # 延迟初始化 state
```

### 2.2 关键方法

| 方法 | 行号 | 功能 |
|------|------|------|
| `get_parameters()` | L203 | 聚合所有 param_group 中的参数 |
| `get_main_grads_for_grad_norm()` | L218 | 收集需要计算 grad norm 的梯度（排除 shared/duplicate）|
| `get_model_parallel_group()` | L241 | 获取 model-parallel 通信组 |
| `clip_grad_norm()` | L260 | 调用 clip_grads.py 的梯度裁剪 |
| `count_zeros()` | L285 | 统计梯度中零值数量（诊断用）|
| `prepare_grads()` | abstract | 预处理梯度（复制、unscale、检查 inf）|
| `step_with_ready_grads()` | abstract | 执行优化器 step |
| `step()` | L345 | 模板方法 = prepare_grads + step_with_ready_grads |

### 2.3 sharded_state_dict 接口

```python
@abstractmethod
def sharded_state_dict(self, model_sharded_state_dict, is_loading=False, metadata=None):
    """构建 optimizer 的 ShardedStateDict，支持跨 DP/TP/PP 重分片加载"""
```

这是 dist_checkpointing 模块与优化器集成的关键接口，允许改变并行度后重新加载。

---

## 3. MixedPrecisionOptimizer (optimizer.py:480-691)

### 3.1 设计动机

BF16/FP16 训练中，模型参数用低精度存储节省显存，但优化器必须在 FP32 上计算以保持数值稳定。MixedPrecisionOptimizer 管理这一转换。

### 3.2 初始化

```python
class MixedPrecisionOptimizer(MegatronOptimizer):
    def __init__(self, optimizer, config, grad_scaler, init_state_fn):
        super().__init__(optimizer, config, init_state_fn)
        self.grad_scaler = grad_scaler  # None when bf16=True (no scaling needed)
```

### 3.3 prepare_grads 流程 (L591-655)

```
┌─────────────────────────────────────────────────────────────┐
│                    prepare_grads() 流程                       │
├─────────────────────────────────────────────────────────────┤
│ 1. _copy_model_grads_to_main_grads()                        │
│    将 model params (bf16) 的 .grad → main params (fp32)       │
│                                                             │
│ 2. [if grad_scaler]:                                        │
│    a. _unscale_main_grads_and_check_for_nan()               │
│       - 除以 loss_scale → 还原真实梯度                         │
│       - 检测 inf/nan → found_inf_flag                        │
│    b. grad_scaler.update(found_inf_flag)                    │
│       - inf → 减半 scale, 跳过 step                          │
│       - 正常 → 累加 growth_tracker, 满则倍增 scale             │
│                                                             │
│ 3. return found_inf_flag                                    │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 step_with_ready_grads 流程 (L657-690)

```python
def step_with_ready_grads(self):
    # 1. Grad clipping: ||g||₂ → clip_coeff = max_norm / (||g||₂ + ε)
    grad_norm = self.clip_grad_norm(self.config.clip_grad)
    
    # 2. Count zeros (optional diagnostic)
    num_zeros = self.count_zeros() if config.log_num_zeros_in_grad else 0
    
    # 3. Actual optimizer.step()
    success = self.step_with_ready_grads()
    
    return success, grad_norm, num_zeros
```

### 3.5 _unscale_main_grads_and_check_for_nan (L509-588)

```python
def _unscale_main_grads_and_check_for_nan(self):
    # 对所有 main_grads 执行: grad *= inv_scale
    # 使用 multi_tensor_applier 批量操作（TE/Apex 加速）
    # 同时检测 inf/nan → self.found_inf
    
    # FlagScale 扩展: 跨设备同步 found_inf
    if self.found_inf.device != torch.device(cur_platform.device_name()):
        self.found_inf = self.found_inf.to(cur_platform.device())
```

---

## 4. 梯度裁剪系统 (clip_grads.py)

### 4.1 get_grad_norm_fp32 (L74-163)

计算 FP32 精度下所有梯度的 p-norm，支持分布式聚合。

**执行流程：**

```python
def get_grad_norm_fp32(grads_for_norm, norm_type=2, grad_stats_parallel_group=None):
    # DTensor 处理：获取 data_parallel_group（FSDP 场景）
    grads_for_norm = [to_local_if_dtensor(grad) for grad in grads_for_norm]
    
    if norm_type == inf:
        # L∞ norm: max(|g|) → all_reduce MAX 跨 DP 和 MP 组
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        torch.distributed.all_reduce(total_norm_cuda, ReduceOp.MAX, group=dp_group)
        torch.distributed.all_reduce(total_norm_cuda, ReduceOp.MAX, group=mp_group)
    
    elif norm_type == 2.0:
        # L2 norm: 使用 multi_tensor_l2norm（TE > Apex > local fallback）
        grad_norm, _ = multi_tensor_applier(l2_norm_impl, ...)
        total_norm = grad_norm ** 2  # sum of squares
        torch.distributed.all_reduce(total_norm, ReduceOp.SUM, group=dp_group)
        torch.distributed.all_reduce(total_norm, ReduceOp.SUM, group=mp_group)
        total_norm = total_norm ** 0.5
```

**关键设计决策：**
- 先 sum-of-squares 再 all_reduce，最后开根号 → 数学正确的分布式 L2 norm
- 使用 TE/Apex 的 multi_tensor_applier → 单 kernel 处理所有梯度张量，避免 per-tensor launch overhead
- 三级 fallback：TE → Apex → local Python 实现

### 4.2 clip_grad_by_total_norm_fp32 (L166-216)

```python
def clip_grad_by_total_norm_fp32(parameters, max_norm, total_norm):
    clip_coeff = max_norm / (total_norm + 1e-6)
    if isinstance(clip_coeff, torch.Tensor):
        # TE path: tensor-based scaling (保持 GPU，避免 H2D sync)
        clip_coeff.clamp_max_(1.0)
        multi_tensor_applier(multi_tensor_scale_tensor_impl, ...)
    elif clip_coeff < 1.0:
        # Apex/local path: scalar scaling
        multi_tensor_applier(multi_tensor_scale_impl, ..., clip_coeff)
```

**设计要点：**
- `clip_coeff ≥ 1.0` 时不做任何操作（梯度已在范围内）
- TE 版本保持 clip_coeff 为 tensor → 避免 `.item()` 的 GPU→CPU 同步
- 支持 `decoupled_grad` 属性（µP 等场景的独立梯度）

### 4.3 count_zeros_fp32 (L218-309)

统计梯度中零值数量，用于诊断训练健康度（过多零值 = 梯度消失）。

---

## 5. Loss Scaling (grad_scaler.py)

### 5.1 类层次

```python
class MegatronGradScaler(ABC):      # L17
    scale: torch.Tensor              # 当前 loss scale 值
    inv_scale: torch.Tensor          # 1 / scale (用于 unscale)
    
class ConstantGradScaler(MegatronGradScaler):    # L53 — bf16 常量 scale
class DynamicGradScaler(MegatronGradScaler):     # L71 — fp16 动态 scale
```

### 5.2 DynamicGradScaler 策略

```
┌────────────────────────────────────────────────────────────┐
│            Dynamic Loss Scaling 状态机                       │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  正常 step (no inf/nan):                                    │
│    growth_tracker += 1                                     │
│    if growth_tracker >= growth_interval:                    │
│       scale *= growth_factor (通常 2x)                      │
│       growth_tracker = 0                                   │
│                                                            │
│  异常 step (inf/nan detected):                              │
│    scale *= backoff_factor (通常 0.5x)                      │
│    growth_tracker = 0                                      │
│    skip optimizer step!                                    │
│                                                            │
│  约束: scale ∈ [min_scale, max_scale]                       │
│  典型配置: initial=2^16, growth_interval=2000               │
└────────────────────────────────────────────────────────────┘
```

**BF16 vs FP16 选择：**
- BF16：指数范围与 FP32 相同 → 几乎不会 overflow → 使用 ConstantGradScaler 或 None
- FP16：指数范围小 → 容易 overflow → 必须 DynamicGradScaler

---

## 6. Float16OptimizerWithFloat16Params (optimizer.py:693-1100+)

### 6.1 核心设计

该类管理 "FP16/BF16 model params ↔ FP32 main params" 的映射：

```python
class Float16OptimizerWithFloat16Params(MixedPrecisionOptimizer):
    def __init__(self, optimizer, config, grad_scaler, ...):
        # 1. 构建 fp32_from_float16_groups: 与 model params 对应的 fp32 副本
        # 2. 构建 fp32_from_fp32_groups: 已经是 fp32 的参数（如 LayerNorm）
        # 3. optimizer.param_groups 指向 fp32 main params
```

### 6.2 数据流

```
Forward/Backward:
  model_params (bf16) → grad (bf16) 
                                    ↓
_copy_model_grads_to_main_grads:
  main_params (fp32).grad = model_params.grad.float()
                                    ↓
optimizer.step():
  main_params (fp32) -= lr * grad   (in fp32 precision)
                                    ↓
_copy_main_params_to_model_params:
  model_params (bf16) = main_params.to(bf16)
```

### 6.3 显存开销

| 组件 | 精度 | 大小 (per param) |
|------|------|-----------------|
| model_params | BF16 | 2 bytes |
| main_params (fp32 copy) | FP32 | 4 bytes |
| momentum (Adam) | FP32 | 4 bytes |
| variance (Adam) | FP32 | 4 bytes |
| grad (临时) | FP32 | 4 bytes |
| **总计** | | **18 bytes/param** |

对比纯 FP32 训练的 16 bytes/param（param + grad + momentum + variance），mixed precision 反而多 2 bytes，但 activation 显存大幅节省。

---

## 7. DistributedOptimizer (distrib_optimizer.py:103-2741)

### 7.1 设计原理

DistributedOptimizer 实现 ZeRO Stage-1/2 风格的优化器状态分片：

```
传统 DP：每个 rank 持有完整 optimizer state → N 倍冗余
ZeRO-1：optimizer state 分片到 DP ranks → 1/N 显存
ZeRO-2：+ gradient 分片 → reduce-scatter 替代 all-reduce
```

### 7.2 核心数据结构

```python
class Range:  # L68
    """索引范围 [start, end)，表示 grad buffer 中的一段区域"""
    start: int
    end: int
    size: int  # = end - start

class DistributedOptimizer(MixedPrecisionOptimizer):  # L103
    # 关键属性：
    model_chunks: List[MegatronModule]
    per_model_buffers: Dict[int, List[_ParamAndGradBuffer]]
    param_to_all_gather_handle_index_map: Dict  # param → handle 索引
    all_gather_handle_index_to_bucket_index_map: Dict  # handle → bucket
```

### 7.3 分片映射构建 (_build_model_gbuf_param_range_map, L124-200)

每个 grad buffer 被均匀划分为 DP-world-size 份，每份属于一个 DP rank：

```
Grad Buffer (padded to dp_world_size 整除):
┌──────────┬──────────┬──────────┬──────────┐
│  rank 0  │  rank 1  │  rank 2  │  rank 3  │
│  owns    │  owns    │  owns    │  owns    │
└──────────┴──────────┴──────────┴──────────┘
     ↑ 注意：分片边界不尊重 parameter 边界
       → 单个 param 可能跨越两个 rank
```

为每个 param 构建 4 种 range：
- `gbuf_world`: param 在整个 grad buffer 中的位置
- `gbuf_world_in_bucket`: param 在其 bucket 内的位置
- `gbuf_local`: param 在当前 rank 拥有的那段中的位置
- `param`: param 内部的偏移（因为只拥有 param 的一部分）

### 7.4 优化器 Step 流程

```
┌─────────────────────────────────────────────────────────────────┐
│            DistributedOptimizer.step() 流程                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 1. Backward 完成 → 各 rank 持有完整梯度（通过 grad hook 触发）      │
│                                                                 │
│ 2. reduce_scatter (async)                                       │
│    全量 grad → 每个 rank 只保留自己拥有的 shard                     │
│    节省通信量：all-reduce = 2N, reduce-scatter = N               │
│                                                                 │
│ 3. prepare_grads()                                              │
│    a. 将 reduced grad shard 复制到 fp32 main_grad                │
│    b. unscale (if fp16)                                         │
│    c. check inf/nan                                             │
│                                                                 │
│ 4. clip_grad_norm()                                             │
│    计算 local shard 的 partial norm → all-reduce SUM → sqrt       │
│                                                                 │
│ 5. optimizer.step() — 只更新自己拥有的 param shard                 │
│    显存节省：state = momentum + variance = 8 bytes/param / DP     │
│                                                                 │
│ 6. all-gather                                                   │
│    将更新后的 param shard 广播给所有 DP ranks                       │
│    → 所有 rank 重新获得完整 model params                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.5 与 Overlap 机制的集成

DistributedOptimizer 深度集成了两种 overlap：

**Overlap Grad Reduce (backward 期间)：**
```python
# param_and_grad_buffer.py: register_grad_ready()
# 当一个 bucket 内所有 grad 就绪 → 立即发起 reduce-scatter
# → backward 计算与 reduce-scatter 通信重叠
```

**Overlap Param Gather (forward 期间)：**
```python
# 在 forward 的第 i 层计算时，预取第 i+1 层的 all-gather
# → forward 计算与 all-gather 通信重叠
# 需要额外 1 个 bucket 大小的显存作为预取 buffer
```

### 7.6 底层 Adam 实现选择

```python
# distrib_optimizer.py L20-35：三级 fallback
try:
    from transformer_engine.pytorch.optimizers import FusedAdam  # 最优：TE fused kernel
except ImportError:
    try:
        from apex.optimizers import FusedAdam  # 次优：Apex fused kernel  
    except ImportError:
        from torch.optim import Adam  # 最慢：PyTorch 原生
```

FusedAdam 优势：
- 单个 CUDA kernel 完成 param_update + momentum + variance 计算
- 避免多次 kernel launch 和中间临时张量
- 支持 multi-tensor 批量处理

### 7.7 CPU Offload (cpu_offloading.py)

```python
class HybridDeviceOptimizer:
    """将 optimizer state 部分卸载到 CPU"""
    # Adam state (momentum, variance) 存储在 CPU pinned memory
    # optimizer.step() 在 CPU 执行
    # 更新后的 params 通过 H2D copy 同步回 GPU
    # 适用于 GPU 显存极度紧张的大模型训练
```

---

## 8. 完整训练 Step 中的优化器交互时序

```
Time →
─────────────────────────────────────────────────────────────────────
[Forward]
  Layer 1 compute  ←── overlap: all-gather layer 2 params (if enabled)
  Layer 2 compute  ←── overlap: all-gather layer 3 params
  ...
  Loss computation
─────────────────────────────────────────────────────────────────────
[Backward]  
  Layer N grad     ──→ bucket N filled → reduce-scatter (async)
  Layer N-1 grad   ──→ bucket N-1 filled → reduce-scatter (async)
  ...                   ↑ overlap: backward compute + grad reduce
  Layer 1 grad     ──→ bucket 1 filled → reduce-scatter (async)
─────────────────────────────────────────────────────────────────────
[Optimizer Step]
  finish_grad_sync()       ← 等待所有 reduce-scatter 完成
  prepare_grads()          ← copy to fp32 + unscale + inf check
  clip_grad_norm()         ← L2 norm + clip
  optimizer.step()         ← Adam update (只更新 local shard)
  all-gather params        ← 广播更新后的参数
─────────────────────────────────────────────────────────────────────
```

---

## 9. OptimizerConfig 关键配置项

### 9.1 精度与缩放

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `fp16` | False | 启用 FP16 训练 + DynamicGradScaler |
| `bf16` | False | 启用 BF16 训练（通常 None scaler）|
| `loss_scale` | None | 固定 loss scale（None = dynamic）|
| `initial_loss_scale` | 2^32 | 动态 loss scale 初始值 |
| `min_loss_scale` | 1.0 | loss scale 下界 |
| `loss_scale_window` | 1000 | growth_interval: 连续成功步数后倍增 |

### 9.2 梯度处理

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `clip_grad` | 1.0 | 梯度裁剪阈值（L2 norm）|
| `log_num_zeros_in_grad` | False | 统计梯度零值数量 |
| `check_for_nan_in_loss_and_grad` | True | 检测 nan/inf |

### 9.3 分布式优化

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `use_distributed_optimizer` | False | 启用 ZeRO 分片 |
| `overlap_grad_reduce` | False | backward 期间 overlap reduce-scatter |
| `overlap_param_gather` | False | forward 期间 overlap all-gather |
| `bucket_size` | 40M | 通信 bucket 大小（元素数）|

### 9.4 学习率

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `lr` | 1e-4 | 峰值学习率 |
| `lr_decay_style` | 'cosine' | 衰减策略：linear/cosine/WSD |
| `lr_warmup_iters` | 0 | warmup 步数 |
| `lr_decay_iters` | None | 衰减总步数 |
| `min_lr` | 0.0 | 最小学习率 |

---

## 10. µP (Maximal Update Parameterization) 支持

### 10.1 源码位置

```python
# __init__.py L214, L221
def should_scale_lr_with_mup(param, param_name) -> bool:
def should_scale_vector_like_lr_with_mup(param, param_name) -> bool:
```

### 10.2 原理

µP 理论要求不同宽度的层使用不同的学习率缩放：
- 嵌入层和输出层：lr * (base_width / actual_width)
- 隐藏层权重：lr * (base_width / actual_width)
- 偏置和 LayerNorm：lr 不变

这允许在小模型上调好超参后直接迁移到大模型，无需重新搜索。

---

## 11. Checkpoint 与 Resharding

### 11.1 Sharded State Dict

DistributedOptimizer 支持多种 checkpoint 格式：

```python
checkpoint_fully_reshardable_formats = {
    'fully_reshardable',          # 按 param 维度分片
    'fully_sharded_model_space',  # model 空间分片
    'fsdp_dtensor',               # FSDP DTensor 格式
}
```

### 11.2 跨并行度加载

当改变 TP/PP/DP 配置后恢复训练：
1. 通过 `sharded_state_dict()` 构建目标 layout
2. dist_checkpointing 自动 reshard optimizer state
3. momentum/variance 按新的 param 映射重新对齐

这使得用户可以在不同并行策略间无缝切换（如从 TP=2 迁移到 TP=4）。

---

## 12. 设计决策与权衡分析

| 设计决策 | 选择 | 原因 |
|----------|------|------|
| FP32 main params 副本 | 保留 | 保证优化器数值精度，防止 bf16 精度丢失累积 |
| Dynamic Loss Scaling | 默认 fp16 | 自适应处理 overflow，无需手动调参 |
| multi_tensor_applier | TE→Apex→local | 单 kernel 批处理所有梯度，减少 launch overhead |
| clip_coeff 保持 tensor | TE path | 避免 GPU→CPU sync 的 `.item()` 调用 |
| reduce-scatter 替代 all-reduce | ZeRO | 通信量减半 + 显存分片 |
| bucket 化通信 | 默认 40M | 平衡 overlap 粒度与 kernel launch 效率 |
| CPU offload Adam | 可选 | 极端显存场景，牺牲速度换显存 |
| Reshardable checkpoint | 默认 | 允许并行度变化，提升训练灵活性 |

---

## 13. 调优建议

### 13.1 显存优化

```yaml
# 开启 distributed optimizer (ZeRO)
optimizer:
  use_distributed_optimizer: true
  
# 配合 overlap 隐藏通信
training:
  overlap_grad_reduce: true     # backward overlap
  overlap_param_gather: true    # forward overlap (需额外 1 bucket 显存)
```

### 13.2 训练稳定性

```yaml
# 梯度裁剪（防止 loss spike）
optimizer:
  clip_grad: 1.0               # 大模型推荐 1.0

# Loss scale 配置
training:
  initial_loss_scale: 65536    # 2^16, 比默认 2^32 更保守
  loss_scale_window: 1000      # 1000 步后尝试倍增
  min_loss_scale: 1.0          # 下界保护
```

### 13.3 大规模训练

```yaml
# 7B+ 模型推荐配置
optimizer:
  type: adam
  lr: 3e-4
  min_lr: 3e-5
  lr_decay_style: cosine
  lr_warmup_iters: 2000
  weight_decay: 0.1
  adam_beta1: 0.9
  adam_beta2: 0.95
  clip_grad: 1.0
  use_distributed_optimizer: true
```

---

## 14. 总结

Megatron-LM-FL 优化器系统的核心创新：

1. **层次化类设计**：MegatronOptimizer → MixedPrecision → Distributed，每层解耦一个关注点
2. **深度通信集成**：grad reduce-scatter 与 backward overlap，param all-gather 与 forward overlap
3. **数值安全保证**：Dynamic Loss Scaling + FP32 main params + distributed grad norm
4. **灵活状态管理**：Reshardable checkpoint 支持任意并行度变换
5. **FlagScale 扩展**：跨平台设备适配、multi_tensor 加速 fallback 链
