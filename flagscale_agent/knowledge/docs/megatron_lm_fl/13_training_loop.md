# 第13章：训练主循环与初始化 深度源码分析

## 1. 概述与设计动机

### 1.1 解决什么问题

训练主循环是 Megatron-LM-FL 的核心调度器，协调以下子系统：
- Pipeline Parallel 调度（forward_backward_func）
- 梯度累积与优化器步进
- 学习率调度
- Checkpoint 保存/加载
- 日志与监控
- RL/GRPO 训练扩展

### 1.2 核心设计思想

**分层抽象**：
- `pretrain()`: 顶层入口，初始化一切 → 调用 `train()`
- `train()`: 主循环，迭代调用 `train_step()`
- `train_step()`: 单步训练（forward + backward + optimizer）
- `forward_backward_func`: PP 调度（1F1B / interleaved）

### 1.3 为什么需要定制训练循环？

PyTorch 原生 `loss.backward()` + `optimizer.step()` 无法处理：
- PP 跨 rank 的 micro-batch 调度
- 多 model chunk (virtual PP) 的梯度同步
- FP8 动态 scaling 状态更新
- Rerun State Machine (容错)

## 2. 源码定位

| 组件 | 文件路径 | 行数 | 职责 |
|------|----------|------|------|
| 主训练模块 | `megatron/training/training.py` | 3864 | pretrain/train/train_step |
| 初始化 | `megatron/training/initialize.py` | ~800 | 环境/进程组/日志初始化 |
| Checkpointing | `megatron/training/checkpointing.py` | ~800 | 保存/加载 |
| PP schedules | `megatron/core/pipeline_parallel/schedules.py` | - | forward_backward_func |

## 3. pretrain() 入口 (L932-1390)

### 3.1 函数签名与参数

```python
def pretrain(
    train_valid_test_dataset_provider,  # 数据集构建函数
    model_provider,                     # 模型构建函数（返回 vanilla model）
    model_type,                         # ModelType.encoder_or_decoder
    forward_step_func,                  # 前向步（返回 loss + metrics）
    process_non_loss_data_func=None,    # 后处理（如 dump images）
    extra_args_provider=None,           # 额外命令行参数
    args_defaults={},                   # 参数默认值覆盖
    store=None,                         # torch.distributed.Store
    inprocess_call_wrapper=None,        # 进程内重启包装
):
```

### 3.2 执行流程

```
pretrain() 时序图:
─────────────────────────────────────────────────
1. ft_integration.setup()              容错系统初始化 (L1000)
2. initialize_megatron()               全局初始化 (L1004-1010)
   ├── args 解析
   ├── distributed 初始化 (init_process_group)
   ├── parallel_state 初始化 (initialize_model_parallel)
   ├── 日志/Tensorboard 初始化
   └── 随机种子设置 (各并行维度独立种子)
3. set_jit_fusion_options()            JIT 融合优化 (L1028)
4. setup_model_and_optimizer()         模型+优化器 (L1713+)
   ├── get_model(model_provider)
   ├── get_optimizer(model)
   ├── get_optimizer_param_scheduler(optimizer)
   └── load_checkpoint() (如果有)
5. build_train_valid_test_data_iterators()  数据加载器 (L3784)
6. train()                             进入主循环 (L2726)
─────────────────────────────────────────────────
```

### 3.3 WHY: 为什么将 model_provider 作为回调？

用户只需定义"干净"的模型（无 DDP、无 FP16），框架负责：
- DDP 包装（根据并行配置）
- FP16/BF16 包装
- FP8 状态注册
- Activation checkpointing 注入

这使得模型定义与训练基础设施解耦。

## 4. train_step() 详解 (L1897-2064)

### 4.1 完整流程

```python
def train_step(forward_step_func, data_iterator, model, optimizer,
               opt_param_scheduler, config, forward_backward_func, iteration):
    """单步训练"""
    
    # Phase 1: 梯度清零 (L1909-1913)
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()
    
    # Phase 2: Forward + Backward (L1946-1957)
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=get_num_microbatches(),
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        forward_only=False,
    )
    # forward_backward_func = get_forward_backward_func()
    # 返回 forward_backward_pipelining_with_interleaving 或
    # forward_backward_pipelining_without_interleaving 或
    # forward_backward_no_pipelining
    
    # Phase 3: Optimizer step (L1996-2005)
    update_successful, grad_norm, num_zeros = optimizer.step()
    
    # Phase 4: LR scheduler step (L2022-2027)
    if update_successful:
        increment = num_microbatches * micro_batch_size * dp_size
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1  # grad overflow, 跳过更新
    
    # Phase 5: Loss 汇总 (L2033-2053)
    if is_pipeline_last_stage():
        loss_reduced = average_losses_across_microbatches(losses_reduced)
```

### 4.2 Rerun State Machine (L1902, 1907)

```python
rerun_state_machine = get_rerun_state_machine()
while rerun_state_machine.should_run_forward_backward(data_iterator):
    # 可能重跑 forward-backward（检测到数值异常时）
    ...
should_checkpoint, should_exit, exit_code = rerun_state_machine.should_checkpoint_and_exit()
```

**WHY Rerun State Machine？**
- 检测到 loss spike / NaN → 回退并重跑
- 容错训练的核心机制
- 与 fault tolerance 系统联动

### 4.3 梯度同步与 allreduce (L2009-2014)

```python
# 跨 Model Parallel 组同步 update_successful
update_successful = logical_and_across_model_parallel_group(update_successful)
# 取所有 MP rank 的 grad_norm 最大值（用于日志）
grad_norm = reduce_max_stat_across_model_parallel_group(grad_norm)
```

**WHY logical_and？**
不同 PP stage 可能有不同的 grad norm（因为模型参数不同）。
只有所有 stage 都成功（无 overflow），才真正更新参数。

## 5. train() 主循环 (L2726-3390)

### 5.1 循环结构

```python
def train(forward_step_func, model, optimizer, ...):
    iteration = args.iteration  # 从 checkpoint 恢复的迭代数
    
    for model_module in model:
        model_module.train()  # 开启 dropout 等训练行为
    
    while iteration < args.train_iters:
        # 5.1 更新动态 batch size (warmup)
        update_num_microbatches(...)
        
        # 5.2 Profiling 控制
        if should_profile(iteration):
            profiler.start()
        
        # 5.3 核心训练步
        loss_dict, skipped_iter, ... = train_step(
            forward_step_func, data_iterator, model, optimizer, ...)
        
        # 5.4 日志记录
        training_log(loss_dict, ...)
        
        # 5.5 评估
        if iteration % args.eval_interval == 0:
            evaluate(forward_step_func, model, valid_data_iterator, ...)
        
        # 5.6 Checkpoint
        if should_save_checkpoint(iteration):
            save_checkpoint_and_time(iteration, model, optimizer, ...)
        
        iteration += 1
```

### 5.2 RL/GRPO 扩展 (L2743-2807)

```python
if args.perform_rl_step:
    # 加载预训练权重作为 reference model
    ref_state_dict = load_pretrained_for_reference()
    
    # 重新初始化 microbatch calculator（RL batch 不同于 SFT）
    destroy_num_microbatches_calculator()
    init_num_microbatches_calculator(
        rank, rampup_batch_size, global_batch_size, micro_batch_size, dp_size)
```

**WHY 重新初始化 microbatch calculator？**
GRPO 训练中，一个 prompt 对应多个 response (N 个样本)，
实际 batch 大小 = global_batch_size × N，需要调整 micro-batch 数量。

## 6. setup_model_and_optimizer (L1713-1885)

### 6.1 模型创建流程

```
get_model(model_provider) (L1427-1627):
│
├─ model = model_provider()          # 用户定义的 vanilla model
├─ model.cuda()                      # 移到 GPU
├─ 设置 FP16/BF16 参数类型
├─ DDP 包装:
│   ├─ 无 PP: LocalDDP 或 DistributedDataParallel
│   └─ 有 PP: 每个 model chunk 独立 DDP
└─ 返回 [model_chunk_0, model_chunk_1, ...]  (VP 多个 chunk)
```

### 6.2 优化器创建

```python
# setup_model_and_optimizer (L1713+):
optimizer = get_megatron_optimizer(model, ...)
# → MixedPrecisionOptimizer
#   → 内部包装 Adam/Muon
#   → 如果 use_distributed_optimizer: DistributedOptimizer (ZeRO-1)

opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
# → OptimizerParamScheduler
#   → LR warmup + decay (linear/cosine/WSD)
```

## 7. training_log 日志系统 (L2067-2408)

### 7.1 记录的指标

```python
def training_log(loss_dict, total_loss_dict, learning_rate,
                 decoupled_learning_rate, iteration, loss_scale,
                 report_memory_flag, skipped_iter, grad_norm, ...):
    """每 N 步打印训练指标"""
    
    # 核心指标:
    # - loss (per key: lm loss, mtp loss, etc.)
    # - learning_rate
    # - grad_norm
    # - skipped_iters (overflow 计数)
    # - tokens_per_sec, TFLOPS
    # - memory usage (allocated, max_allocated, reserved)
```

### 7.2 Throughput 计算 (L2409-2440)

```python
def compute_throughputs_and_append_to_progress_log(iteration, flops_so_far):
    """计算并写入进度日志"""
    elapsed_time = time.time() - start_time
    samples_per_sec = consumed_samples / elapsed_time
    tokens_per_sec = samples_per_sec * seq_length
    tflops = flops_so_far / elapsed_time / (10**12) / world_size
```

## 8. 性能量化分析

### 8.1 训练步耗时分解（典型 DeepSeek-V3 配置）

| 阶段 | 占比 | 关键因素 |
|------|------|----------|
| Forward | 30-35% | 计算密集 |
| Backward | 50-55% | 梯度计算 + 通信 |
| Optimizer | 5-10% | Adam/Muon 更新 |
| Data loading | 1-3% | IO/预处理 |
| Checkpoint | 0-5% | 异步时接近 0% |

### 8.2 micro-batch 数量对 PP bubble 的影响

```
PP bubble ratio = (P-1) / (P-1 + M)
  P = PP stages, M = num_microbatches

例: PP=8, micro_batch=32:  bubble = 7/(7+32) = 18%
    PP=8, micro_batch=64:  bubble = 7/(7+64) = 10%
    PP=8, micro_batch=128: bubble = 7/(7+128) = 5%
```

## 9. 设计决策对比表

| 维度 | Megatron train_step | PyTorch 原生 | 选择理由 |
|------|-------------------|-------------|----------|
| PP 调度 | forward_backward_func 抽象 | 不支持 | 必须定制 |
| 梯度累积 | micro-batch 循环 | 手动实现 | 与 PP 融合 |
| Loss scaling | 动态 loss scale + skip | 固定 | FP16 必需 |
| 容错 | Rerun State Machine | 无 | 大规模训练必需 |
| 日志 | 内置 TFLOPs/memory | 手动 | 运维友好 |

| 维度 | 同步 optimizer.step() | 异步 param gather | 选择理由 |
|------|---------------------|-------------------|----------|
| 实现复杂度 | 低 | 高 | — |
| 性能 | 有 bubble | 隐藏通信 | 大模型推荐异步 |
| 适用场景 | 小模型/调试 | 生产训练 | — |

## 10. 边界条件与约束

### 10.1 关键约束

- `num_microbatches` 必须 ≥ PP stages（否则有 stage 空闲）
- `global_batch_size` = `micro_batch_size` × `num_microbatches` × `dp_size`
- CUDA_DEVICE_MAX_CONNECTIONS=1 时 PP P2P 串行（性能下降）
- `rampup_batch_size` warmup 期间动态调整 micro-batch 数量

### 10.2 train_step 返回值约定

```python
# 只有 PP last stage 有 loss_reduced（其他 stage 返回 {}）
if mpu.is_pipeline_last_stage(ignore_virtual=True):
    return (loss_reduced, skipped_iter, ...)
return ({}, skipped_iter, ...)
```

### 10.3 FlagScale 扩展点

- `perform_rl_step`: GRPO 强化学习训练模式 (L2743)
- `hybrid_context_parallel`: HybridCPDataLoaderWrapper (L2812)
- `run_workload_inspector_server`: 运行时诊断 (L2815)
- `fine_grained_activation_offloading`: CPU 亲和性设置 (L1017)

## 11. 配置建议

### 11.1 性能调优

```yaml
# 推荐配置（大模型训练）:
training:
  micro_batch_size: 1-2         # 大模型内存受限
  global_batch_size: 2048-4096  # 足够多 microbatch 减少 PP bubble
  rampup_batch_size: [32, 32, 1000]  # 前 1000 步逐步增大 batch
  empty_unused_memory_level: 1   # 每步清空未用内存
  overlap_param_gather: true     # 异步参数 all-gather
  overlap_grad_reduce: true      # 异步梯度 reduce-scatter
```

### 11.2 常见陷阱

1. **loss 不下降**：检查 `skipped_iter` 是否频繁（loss scale 问题）
2. **OOM**：减小 `micro_batch_size` 或开启 activation recomputation
3. **低 TFLOPS**：增大 `global_batch_size` 减少 PP bubble
4. **Checkpoint 慢**：启用 async checkpoint (`async_save=True`)

## 12. initialize_megatron() 详解

### 12.1 初始化序列 (initialize.py)

```
initialize_megatron():
├── 1. parse_args_and_setup()           命令行/YAML参数解析
│     ├── Hydra/argparse 合并
│     └── 环境变量注入 (MASTER_ADDR, MASTER_PORT)
│
├── 2. _set_random_seed()               随机种子设置
│     ├── torch.manual_seed(seed)
│     ├── numpy.random.seed(seed)
│     ├── random.seed(seed)
│     └── cuda_rng_tracker (各并行维度独立种子)
│         ├── TP 维度: seed + tp_rank (确保权重初始化不同)
│         └── DP 维度: seed (相同，确保数据划分一致)
│
├── 3. torch.distributed.init_process_group()  通信初始化
│     ├── backend: nccl (GPU) / gloo (CPU)
│     ├── world_size: total GPUs
│     └── rank: 当前 GPU 编号
│
├── 4. initialize_model_parallel()      并行组创建 (→ 第15章)
│
├── 5. _initialize_memory_buffer()      内存预分配
│     └── 预分配 communication buffer 避免运行时 malloc
│
└── 6. set_jit_fusion_options()         JIT 优化
      ├── torch._C._jit_set_profiling_executor(True)
      ├── torch._C._jit_set_profiling_mode(True)
      └── torch._C._jit_override_can_fuse_on_cpu(False)
```

### 12.2 WHY: 为什么需要独立随机种子管理？

TP 并行中，同一层的权重被切分到不同 GPU。如果所有 GPU 用同一种子初始化，
切分后拼回去就是重复数据。每个 TP rank 用 `seed + tp_rank` 保证权重多样性。

但 Dropout 需要在 TP 组内一致（否则 AllReduce 结果不正确），
所以 `model_parallel_cuda_manual_seed` 确保 TP 组共享 dropout 种子。

## 13. 源码版本信息

- 文件: `megatron/training/training.py` (3864 行)
- 文件: `megatron/training/initialize.py` (~800 行)
- FlagScale 特有扩展: GRPO/RL, HybridCP, WorkloadInspector, RerunStateMachine
