# DeepSpeed 源码深度分析 — 第1章：整体架构与 ZeRO 优化器

## 1. 设计动机

### 1.1 WHY DeepSpeed？

```
DeepSpeed 解决的核心问题:
─────────────────────────────────
1. 内存墙: 大模型参数 + 优化器状态 + 梯度 + 激活 > GPU 内存
2. 通信墙: 数据并行 AllReduce 在模型变大时成为瓶颈
3. 易用性: 用户只需改几行代码, 不需要重构模型

DeepSpeed vs PyTorch DDP/FSDP vs Megatron:
┌──────────────┬────────────┬────────────┬──────────────┐
│ 维度         │ DeepSpeed  │ FSDP       │ Megatron     │
├──────────────┼────────────┼────────────┼──────────────┤
│ 侵入性       │ 低 (wrap)  │ 低 (wrap)  │ 高 (重写模型)│
│ ZeRO Stage  │ 1/2/3/++   │ 2/3        │ 1 (dist opt) │
│ Offload     │ CPU+NVMe   │ CPU only   │ 无           │
│ 推理优化     │ ✓          │ ✗          │ ✗            │
│ 模型并行     │ PP         │ TP(DTensor)│ TP+PP+EP     │
│ MoE         │ ✓          │ 部分       │ ✓            │
│ 适用规模     │ 中大       │ 中等       │ 超大         │
└──────────────┴────────────┴────────────┴──────────────┘
```

### 1.2 DeepSpeed 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                  用户训练脚本                             │
├─────────────────────────────────────────────────────────┤
│  deepspeed.initialize(model, optimizer, config)          │
├─────────────────────────────────────────────────────────┤
│  DeepSpeedEngine (runtime/engine.py: 5680行)             │
│  ├── ZeRO Optimizer (Stage 1/2/3)                       │
│  ├── Pipeline Parallel                                   │
│  ├── Mixed Precision (FP16/BF16)                        │
│  ├── Gradient Accumulation                               │
│  ├── Activation Checkpointing                            │
│  ├── Offload (CPU/NVMe)                                 │
│  └── Communication Backend                               │
├─────────────────────────────────────────────────────────┤
│  torch.distributed (ProcessGroup / NCCL)                 │
├─────────────────────────────────────────────────────────┤
│  Hardware (GPU / CPU / NVMe SSD)                         │
└─────────────────────────────────────────────────────────┘
```

## 2. DeepSpeedEngine 核心 (runtime/engine.py L235-500)

### 2.1 初始化

```python
# deepspeed/runtime/engine.py L235-238
class DeepSpeedEngine(Module):
    def __init__(self,
                 args,
                 model,
                 optimizer=None,
                 model_parameters=None,
                 training_data=None,
                 lr_scheduler=None,
                 mpu=None,           # model parallel unit (Megatron 兼容)
                 dist_init_required=None,
                 collate_fn=None,
                 config=None,        # ds_config dict 或 json 路径
                 dont_change_device=False):
```

### 2.2 deepspeed.initialize API

```python
# deepspeed/__init__.py
def initialize(args=None, model=None, optimizer=None, 
               model_parameters=None, training_data=None,
               lr_scheduler=None, mpu=None, config=None, ...):
    """DeepSpeed 一键初始化"""
    # 1. 解析 ds_config (JSON/dict)
    # 2. 创建 DeepSpeedEngine
    # 3. 根据 config 选择 ZeRO Stage
    # 4. 包装 optimizer
    # 5. 返回 (engine, optimizer, dataloader, lr_scheduler)
    
    engine = DeepSpeedEngine(args, model, optimizer, ...)
    return engine, engine.optimizer, engine.training_data, engine.lr_scheduler
```

## 3. ZeRO (Zero Redundancy Optimizer)

### 3.1 ZeRO Stage 对比

```
ZeRO 三阶段内存优化 (N = DP world_size):
══════════════════════════════════════════════
              Parameters  Gradients  Optimizer States
Baseline:     每 rank P   每 rank G  每 rank OS (2P for Adam)
ZeRO-1:       每 rank P   每 rank G  OS/N per rank
ZeRO-2:       每 rank P   G/N        OS/N per rank
ZeRO-3:       P/N          G/N        OS/N per rank

内存节省 (7B model, N=8, Adam FP32):
┌────────┬─────────┬─────────────┬──────────────────────┐
│ Stage  │ Params  │ Gradients   │ Optimizer            │ Total/GPU│
├────────┼─────────┼─────────────┼──────────────────────┤
│ None   │ 28 GB   │ 28 GB       │ 56 GB                │ 112 GB  │
│ ZeRO-1 │ 28 GB   │ 28 GB       │ 56/8 = 7 GB          │ 63 GB   │
│ ZeRO-2 │ 28 GB   │ 28/8=3.5 GB │ 7 GB                 │ 38.5 GB │
│ ZeRO-3 │ 3.5 GB  │ 3.5 GB      │ 7 GB (+临时AllGather)│ ~42 GB  │
└────────┴─────────┴─────────────┴──────────────────────┘
```

### 3.2 ZeRO Stage 1 & 2 (stage_1_and_2.py)

```python
# deepspeed/runtime/zero/stage_1_and_2.py
class DeepSpeedZeroOptimizer(ZeROOptimizer):
    """ZeRO Stage 1 and 2 实现"""
    
    # Stage 1: Partition optimizer states
    #   每个 rank 只为 P/N 的参数维护优化器状态
    #   AllReduce gradients (全部 rank 持有完整梯度)
    #   每个 rank 只 update 自己负责的参数 shard
    #   AllGather 更新后的参数 (恢复完整参数)
    
    # Stage 2: + Partition gradients  
    #   ReduceScatter gradients (每 rank 只保留自己 shard 的梯度)
    #   每个 rank update 自己的参数 shard
    #   AllGather 更新后的参数
```

### 3.3 ZeRO Stage 3 (stage3.py: 3814行)

```
ZeRO-3 Forward/Backward 流程:
═══════════════════════════════════════
Forward:
  for each layer:
    1. AllGather 完整参数 (从各 rank 收集)
    2. 计算 forward
    3. 释放完整参数 (只保留 shard)
    ← 与 FSDP 类似!
    
Backward:
  for each layer (逆序):
    1. AllGather 完整参数 (再次收集)
    2. 计算梯度
    3. ReduceScatter 梯度 (只保留梯度 shard)
    4. 释放完整参数
    
Optimizer Step:
  每个 rank 只更新自己的参数 shard (1/N)
  使用本地梯度 shard + 本地优化器状态
  无需通信!
```

### 3.4 WHY ZeRO-3 通信量 = 1.5× DDP?

```
通信量分析:
──────────────────
DDP:         AllReduce(gradients) = 2P (ring: reduce-scatter + all-gather)
ZeRO-3:     Forward AllGather(P) + Backward AllGather(P) + ReduceScatter(G)
             = P + P + P = 3P
             
但 ZeRO-3 通常使用 gradient_accumulation:
  K 个 micro-step 只有最后一步做通信
  等效: 3P/K ≈ 0 (K large enough)
  
  WHY 不直接用 DDP?
  因为 DDP 放不下模型! ZeRO-3 的价值是内存, 不是通信效率
```

## 4. ZeRO-Offload (CPU & NVMe)

### 4.1 CPU Offload

```
ZeRO-Offload 数据流:
═══════════════════════
                    GPU                          CPU
              ┌──────────────┐           ┌──────────────┐
   Forward:   │ AllGather P  │ ←─────── │ P shard      │
              │ Compute FWD  │           │              │
              └──────────────┘           └──────────────┘
              ┌──────────────┐           ┌──────────────┐
   Backward:  │ AllGather P  │ ←─────── │ P shard      │
              │ Compute BWD  │           │              │
              │ ReduceScatter│ ──────→  │ G shard      │
              └──────────────┘           └──────────────┘
                                         ┌──────────────┐
   Update:                               │ Adam step    │
              ┌──────────────┐ ←─────── │ (CPU compute)│
              │ P shard (new)│           │ OS (CPU RAM) │
              └──────────────┘           └──────────────┘

  WHY offload 到 CPU?
  - CPU RAM: 256-2048 GB vs GPU: 80 GB
  - 优化器状态(Adam): 2× 参数 → 大量内存
  - 代价: PCIe 带宽 (~64 GB/s) << NVLink (900 GB/s)
  - 适用: 内存严重不足时的 fallback
```

### 4.2 NVMe Offload (ZeRO-Infinity)

```
ZeRO-Infinity: 进一步 offload 到 NVMe SSD
──────────────────────────────────────────
层级:     GPU (80GB) → CPU (512GB) → NVMe (数 TB)
带宽:     NVLink      PCIe          NVMe (~6 GB/s)

适用: 超大模型 (1T+ parameters) 在有限 GPU 上训练
代价: 显著降低训练吞吐 (NVMe 带宽成为瓶颈)
```

## 5. ds_config 配置

```json
{
  "train_batch_size": 256,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 8,
  
  "zero_optimization": {
    "stage": 3,
    "offload_param": {"device": "cpu", "pin_memory": true},
    "offload_optimizer": {"device": "cpu", "pin_memory": true},
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  },
  
  "fp16": {"enabled": true, "loss_scale": 0, "initial_scale_power": 16},
  "gradient_clipping": 1.0
}
```

## 6. 关键源码文件

| 文件 | 行数 | 核心职责 |
|------|------|---------|
| runtime/engine.py | 5680 | DeepSpeedEngine 主类 |
| runtime/zero/stage3.py | 3814 | ZeRO-3 实现 |
| runtime/zero/stage_1_and_2.py | ~2800 | ZeRO-1/2 实现 |
| runtime/zero/partition_parameters.py | ~1500 | 参数分片管理 |
| runtime/zero/offload_config.py | ~200 | Offload 配置 |
| runtime/activation_checkpointing/ | ~800 | 激活重计算 |
| pipe/ | ~3000 | Pipeline Parallel |
| moe/ | ~2000 | MoE 支持 |

## 7. 总结

```
DeepSpeed 核心价值:
┌──────────────────────────────────────────────────┐
│ 1. ZeRO: 系统化解决内存冗余问题 (Stage 1→3)     │
│ 2. Offload: 利用 CPU/NVMe 扩展有效内存           │
│ 3. 低侵入: deepspeed.initialize() 一行替换       │
│ 4. JSON 配置: 无需改代码即可调优                  │
│ 5. 生态集成: HuggingFace Trainer 原生支持         │
└──────────────────────────────────────────────────┘
```

## 8. DeepSpeedEngine 训练循环

### 8.1 核心方法

```python
# runtime/engine.py — 用户接口
engine.forward(batch)          # = model.forward (透传)
engine.backward(loss)          # 梯度计算 + ZeRO 通信
engine.step()                  # 优化器更新 + AllGather 参数
engine.save_checkpoint(path)   # ZeRO-aware checkpoint
engine.load_checkpoint(path)   # 自动处理 shard 恢复

# 典型训练循环:
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model, config=ds_config)

for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

### 8.2 Gradient Accumulation

```
DeepSpeed gradient_accumulation_steps:
──────────────────────────────────────
train_batch_size = micro_batch × gradient_accum × dp_size

例: train_batch_size=256, micro_batch=4, dp=8
    gradient_accum = 256 / (4×8) = 8

流程:
  for i in range(gradient_accum):
    micro_loss = model(micro_batch[i])
    engine.backward(micro_loss)     # 累积梯度, 不通信
  engine.step()                     # 统一 AllReduce + update
  
WHY 在 engine 层管理?
  - ZeRO 需要知道何时触发 ReduceScatter
  - 累积期间跳过通信 → 节省带宽
  - 自动 loss scaling (除以 gradient_accum)
```

## 9. 与 FSDP 的技术对比

```
ZeRO-3 vs FSDP FULL_SHARD 实现差异:
┌─────────────────┬──────────────────┬───────────────────┐
│ 维度            │ DeepSpeed ZeRO-3 │ PyTorch FSDP      │
├─────────────────┼──────────────────┼───────────────────┤
│ 参数管理        │ partition_params  │ FlatParameter     │
│                 │ (独立分片)        │ (扁平化连续内存)  │
├─────────────────┼──────────────────┼───────────────────┤
│ 通信触发        │ pre/post hook    │ pre/post forward  │
│                 │ + 手动 prefetch   │ + backward_prefetch│
├─────────────────┼──────────────────┼───────────────────┤
│ NVMe offload   │ ✓ ZeRO-Infinity  │ ✗ (仅 CPU)        │
├─────────────────┼──────────────────┼───────────────────┤
│ torch.compile  │ 部分支持          │ 完全支持           │
├─────────────────┼──────────────────┼───────────────────┤
│ Checkpoint     │ 自定义 shard      │ DTensor/distributed│
│ 格式           │ format            │ checkpoint         │
├─────────────────┼──────────────────┼───────────────────┤
│ 配置方式       │ JSON config       │ Python API         │
└─────────────────┴──────────────────┴───────────────────┘
```

## 10. MoE (Mixture of Experts) 支持

```python
# deepspeed/moe/
# DeepSpeed MoE 支持:
#   - Expert Parallel: 专家分布到不同 GPU
#   - All-to-All routing: token → expert 通信
#   - Capacity factor: 限制每个专家处理的 token 数
#   - Top-K routing: 选择 K 个专家

# 与 Megatron EP 的区别:
# DeepSpeed: 独立 MoE layer 实现, 通过 config 启用
# Megatron: 深度集成在 TransformerLayer, 支持更多并行策略组合
```

## 11. Pipeline Parallel (deepspeed/pipe/)

```
DeepSpeed Pipeline 并行:
──────────────────────────
特点:
1. 1F1B schedule (与 Megatron 类似)
2. 通过 PipelineModule 自动切分 model
3. Activation checkpointing 集成
4. 与 ZeRO-1 兼容 (ZeRO-2/3 与 PP 部分兼容)

vs Megatron PP:
  DeepSpeed: 用户传 layer list, 自动按 num_stages 切分
  Megatron: 在 TransformerBlock 层精确控制 PP 边界
  Megatron 更灵活 (interleaved, virtual PP), DeepSpeed 更易用
```
