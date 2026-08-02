# 第七章：内存优化

## 1. 概述

大模型训练的 GPU 内存被四大组件占据。Megatron-LM-FL 提供多层次内存优化策略，通过 recompute、offload、分布式分片组合使用来突破内存瓶颈。

**源码定位**：

| 组件 | 源码路径 | 职责 |
|------|----------|------|
| Recompute 配置 | `transformer_config.py` L574-597 | granularity/method/num_layers |
| Recompute 执行 | `transformer_block.py` L529-650 | custom/checkpoint_handler |
| Selective Recompute | `transformer_config.py` L1612-1640 | recompute_modules 列表 |
| Activation Offload | `pipeline_parallel/fine_grained_activation_offload.py` (1284行) | 细粒度 GPU→CPU offload |
| Distribute Activations | `transformer_config.py` L610 | activation scatter 到 TP group |
| FP8 Activation Store | `transformer_config.py` | activation 低精度存储 |
| Distributed Optimizer | `optimizer/distrib_optimizer.py` | ZeRO 参数/梯度分片 |
| Param/Grad Buffer | `distributed/param_and_grad_buffer.py` | 连续内存 bucket |

---

## 2. 内存组成分析

### 2.1 各组件占比

```
以 10B 参数 LLM (BF16) 为例:

┌─────────────────────────────────────────────────────┐
│           GPU 内存分布 (无优化, 单卡)                │
├──────────────────┬──────────┬────────────────────────┤
│ 组件             │ 大小     │ 公式                    │
├──────────────────┼──────────┼────────────────────────┤
│ Parameters (BF16)│  20 GB   │ P × 2B                 │
│ Master Weights   │  40 GB   │ P × 4B (FP32)          │
│ Optimizer States │  80 GB   │ P × 8B (Adam m + v)    │
│ Gradients (BF16) │  20 GB   │ P × 2B                 │
│ Activations      │  变量    │ f(L, s, h, mbs)        │
│ Temp Buffers     │  ~2 GB   │ comm buffer + workspace│
├──────────────────┼──────────┼────────────────────────┤
│ 总计 (无优化)    │ ~162+ GB │ 远超单卡 80GB           │
└──────────────────┴──────────┴────────────────────────┘

其中:
  P = 参数量 (10B)
  L = 层数, s = seq_len, h = hidden_size, mbs = micro_batch_size
```

### 2.2 Activation 内存详细计算

```
单层 Transformer 的 activation 存储 (BF16, 保存用于backward):
  
  Attention 部分:
    - QKV 输入:       s × h × 2B
    - Q, K, V:        3 × s × h × 2B
    - Attention Score: s × s × n_heads × 2B  ← 最大项 (seq²)
    - Softmax output:  s × s × n_heads × 2B
    - Context:         s × h × 2B
    小计 ≈ 2×s²×n_heads×2B + 6×s×h×2B

  MLP 部分:
    - Input:           s × h × 2B
    - Gate/Up output:  2 × s × h_ffn × 2B  (h_ffn ≈ 4h)
    - Activation func: s × h_ffn × 2B
    小计 ≈ 12×s×h×2B

  每层总计 ≈ 2×s²×n_heads×2B + 18×s×h×2B

示例 (Qwen3-10B: L=48, h=4096, s=4096, n_heads=32, mbs=1):
  Attention Score: 2 × 4096² × 32 × 2 = 2 GB / 层
  其余: 18 × 4096 × 4096 × 2 ≈ 0.6 GB / 层
  单层: ~2.6 GB
  48层: ~125 GB  ← 这就是为什么必须优化 activation
```

---

## 3. Activation Recompute (梯度检查点)

### 3.1 核心思想

```
标准 backward:
  Forward 保存所有中间 activation → Backward 直接使用
  内存: O(L) 层 activation

Recompute:
  Forward 只保存 checkpoint 点的 activation
  Backward 时从 checkpoint 重新前向计算获取中间值
  内存: O(1) 或 O(√L)  代价: 额外一次前向计算
```

### 3.2 配置参数

```python
# transformer_config.py L574-597

recompute_granularity: Literal['full', 'selective'] = None
# 'full': 整层 recompute (整个 transformer layer 前向重算)
# 'selective': 仅 recompute 指定子模块 (推荐)

recompute_method: Literal['uniform', 'block'] = None
# 'uniform': 每 N 层为一个 chunk，只保存 chunk 入口的 activation
# 'block': 前 N 层 recompute，后面层正常保存

recompute_num_layers: int = None
# uniform: chunk 大小 (每 chunk 包含几层)
# block: 需要 recompute 的层数
```

### 3.3 Full Recompute 实现

```
源码: transformer_block.py L529-650

┌─ custom(start, end) 闭包 ─────────────────────────────────┐
│  def custom_forward(hidden_states, attention_mask, ...):    │
│      for index in range(start, end):                        │
│          layer = self._get_layer(index)                     │
│          with inner_quantization_context:                   │
│              hidden_states, context = layer(hidden_states)  │
│      return hidden_states, context                          │
└─────────────────────────────────────────────────────────────┘

┌─ checkpoint_handler(forward_func) ──────────────────────────┐
│  if config.fp8:                                              │
│      return te_checkpoint(forward_func,                      │
│          distribute_saved_activations,                        │
│          get_cuda_rng_tracker,     # RNG状态保存/恢复        │
│          tp_group,                                           │
│          hidden_states, ...)                                  │
│  else:                                                       │
│      return tensor_parallel.checkpoint(forward_func,         │
│          distribute_saved_activations,                        │
│          hidden_states, ...)                                  │
└──────────────────────────────────────────────────────────────┘
```

### 3.4 Uniform 方法 (L602-622)

```
┌─ Uniform Recompute (recompute_num_layers=4, 总12层) ──────────┐
│                                                                 │
│  Chunk 0: Layer 0-3   → checkpoint(custom(0, 4))               │
│  Chunk 1: Layer 4-7   → checkpoint(custom(4, 8))               │
│  Chunk 2: Layer 8-11  → checkpoint(custom(8, 12))              │
│                                                                 │
│  内存: 仅保存 3 个 checkpoint (chunk 入口的 hidden_states)     │
│  重计算: backward 时每个 chunk 重算 4 层前向                   │
│                                                                 │
│  时序图 (Backward of Chunk 1):                                  │
│    保存的 h₄ ──→ [重算 Layer4→5→6→7] ──→ 得到 h₅,h₆,h₇       │
│                                        ──→ 正常 backward        │
└─────────────────────────────────────────────────────────────────┘

伪代码 (L602-622):
  layer_idx = 0
  while layer_idx < num_layers_per_pipeline_rank:
      chunk_end = min(layer_idx + recompute_num_layers, num_layers)
      hidden_states, context = checkpoint_handler(custom(layer_idx, chunk_end))
      layer_idx += recompute_num_layers
```

### 3.5 Block 方法 (L624-650)

```
┌─ Block Recompute (recompute_num_layers=8, 总12层) ────────────┐
│                                                                 │
│  Layer  0: checkpoint(custom(0, 1))  ← recompute              │
│  Layer  1: checkpoint(custom(1, 2))  ← recompute              │
│  ...                                                           │
│  Layer  7: checkpoint(custom(7, 8))  ← recompute              │
│  Layer  8: custom(8, 9)(...)         ← 正常 (保存 activation)  │
│  ...                                                           │
│  Layer 11: custom(11,12)(...)        ← 正常                    │
│                                                                 │
│  设计动机: 前层 activation 存活时间长 → recompute 节省更多      │
│           后层 activation 很快被 backward 消费 → 保存更划算    │
└─────────────────────────────────────────────────────────────────┘

特殊处理 (FP8, L634):
  if (config.fp8) and not hidden_states.requires_grad:
      recompute_skip_num_layers += 1
  # FP8 模式下若 input 无梯度则跳过 checkpoint
  # (re-entrant autograd 需要至少一个需要梯度的输入)
```

### 3.6 Selective Recompute

```
源码: transformer_config.py L1612-1640

recompute_modules: list[str] = ["core_attn"]  # 默认仅重算 attention

可选模块:
  "core_attn"       → Attention Score (QK^T, Softmax, ×V) — seq² 内存最大
  "mlp"             → MLP 全部中间激活
  "moe"             → MoE 层 (含 dispatch/combine)
  "moe_act"         → MoE 激活函数 (需 grouped_gemm)
  "shared_experts"  → 共享专家
  "layernorm"       → LayerNorm 输入
  "mla_up_proj"     → Multi-Latent Attention 上投影
  "mhc"             → Hyper-Connections

实现原理:
  被选中的子模块的 forward 用 torch.utils.checkpoint 包裹
  backward 时仅重算该子模块 (而非整层)
  → 更细粒度的 memory/compute trade-off
```

### 3.7 内存节省与计算开销对比

| 策略 | 内存节省 | 额外计算 | 适用场景 |
|------|---------|---------|----------|
| selective (core_attn) | ~40% activation | ~33% | **推荐默认** |
| selective (core_attn+mlp) | ~70% | ~60% | 内存紧张 |
| full uniform (chunk=4) | ~75% | ~100% | 长序列/大模型 |
| full uniform (chunk=1) | ~95% | ~100% | 极端内存受限 |
| full block (前N层) | ~N/L × 95% | ~N/L × 100% | 平衡策略 |

```
设计决策: 为什么 selective 优于 full?
  
  core_attn 占激活内存的 ~60% (seq² 项)
  但只占计算的 ~25%
  → selective(core_attn) 以 25% 额外计算换 60% 内存节省
  → full 以 100% 额外计算换 95% 内存节省
  
  ROI: selective = 60%/25% = 2.4   full = 95%/100% = 0.95
  selective 的投入产出比远优于 full
```

---

## 4. Fine-Grained Activation Offloading

### 4.1 架构

```
源码: pipeline_parallel/fine_grained_activation_offload.py (1284行)

类层次:
  PipelineOffloadManager (singleton, L391)
    ├── _d2h_stream: CUDA Stream (GPU→CPU 专用)
    ├── _h2d_stream: CUDA Stream (CPU→GPU 专用)
    ├── _cpu_tensor_pool: GPUTensorPool (CPU pinned memory 池)
    ├── _cached_chunks_forward: List[ChunkOffloadHandler]
    └── _cached_chunks_backward: List[ChunkOffloadHandler]
  
  ChunkOffloadHandler (L738, 每个 microbatch 一个)
    ├── offload_groups: List[OffloadTensorGroup]
    ├── d2h_stream / h2d_stream (共享 Manager 的 stream)
    └── cpu_tensor_pool (共享 Manager 的 pool)
  
  OffloadTensorGroup (L338, 按模块分组)
    ├── _name: str ("attn_norm", "qkv_linear", "core_attn", ...)
    ├── _tensors: Dict[tag, Tensor]
    ├── _offload_event: CUDA Event (D2H完成信号)
    └── _reload_event: CUDA Event (H2D完成信号)

  GPUTensorPool (L102, 内存池)
    ├── _pools: Dict[(shape,dtype), {free: deque, all: list}]
    └── 支持 O(1) allocate/free, 避免反复 malloc
```

### 4.2 Offload/Reload 流程

```
Forward (Layer N):
  ┌─ Compute Stream ──────────────────────────────────────┐
  │  hidden = layer_N(hidden)                              │
  │  record_event(compute_done)  ←── 标记计算完成          │
  └────────────────────────────────────────────────────────┘
  
  ┌─ D2H Stream (异步) ──────────────────────────────────┐
  │  wait_event(compute_done)    ←── 等计算完成            │
  │  cpu_buf = pool.allocate(shape, dtype)                 │
  │  cpu_buf.copy_(gpu_tensor, non_blocking=True)          │
  │  record_event(offload_done)  ←── 标记D2H完成          │
  │  # GPU tensor 可被释放                                 │
  └────────────────────────────────────────────────────────┘

Backward (Layer N):
  ┌─ H2D Stream (异步, 提前触发) ────────────────────────┐
  │  wait_event(offload_done)    ←── 确保D2H已完成        │
  │  gpu_tensor = torch.empty(..., device='cuda')          │
  │  gpu_tensor.copy_(cpu_buf, non_blocking=True)          │
  │  record_event(reload_done)                             │
  │  pool.free(cpu_buf)           ←── CPU buffer归还池    │
  └────────────────────────────────────────────────────────┘
  
  ┌─ Compute Stream ──────────────────────────────────────┐
  │  wait_event(reload_done)     ←── 等H2D完成            │
  │  grad = backward(gpu_tensor)  ←── 正常反向传播        │
  └────────────────────────────────────────────────────────┘
```

### 4.3 Offload 模块选择

```python
# transformer_config.py L1155
fine_grained_activation_offloading: bool = False
offload_modules: list[str] = []

# 可选模块 (粒度从小到大):
#   "attn_norm"    → Attention 前的 LayerNorm 输入
#   "qkv_linear"   → QKV 线性层输入
#   "core_attn"    → Attention Score/Softmax 输出
#   "attn_proj"    → Attention O 投影输入
#   "mlp_norm"     → MLP 前的 LayerNorm 输入
#   "expert_fc1"   → Expert 第一层输入 (MoE)
#   "moe_act"      → MoE 激活函数输入
```

### 4.4 性能模型

```
PCIe Gen5 带宽: 64 GB/s (单向)
H100 NVLink: 450 GB/s (GPU-GPU, 不走PCIe)

单层 offload 量 (core_attn, s=4096, h=4096, BF16):
  attn_score: 4096×4096×32×2B = 1 GB
  offload 时间: 1 GB / 64 GB/s = 15.6 ms

单层计算时间 (10B/48层, H100):
  ~20-30 ms/layer (取决于batch size)

关键条件: offload_time < compute_time → 可完全隐藏
  15.6 ms < 20-30 ms → ✓ core_attn offload 可隐藏

约束:
  1. 不能与 cpu_offloading 同时使用 (二者都争 PCIe 带宽)
  2. offload_margin: 保留最后 X 个 group 不 offload (确保 reload 不阻塞计算)
  3. Warmup 阶段: 首轮收集 tensor shape → 后续复用 CPU pinned buffer
```

### 4.5 与 Recompute 的互补

```
场景: s=8192, 内存极度紧张

方案 A: full recompute
  节省: 95% activation
  代价: +100% FLOPs

方案 B: selective recompute + offload
  selective(core_attn): 节省 60% activation, +25% FLOPs
  offload(mlp_norm + attn_proj): 额外节省 20%, 0% FLOPs (hidden by compute)
  总计: 节省 80%, 代价仅 25% FLOPs

→ 组合方案 B 在同等内存节省下性能损失更小
```

---

## 5. Distribute Saved Activations

### 5.1 原理

```
源码: transformer_config.py L610
  distribute_saved_activations: bool = False

机制:
  当 sequence_parallel=True 时:
    checkpoint 保存的 activation 沿 sequence 维度 scatter 到 TP group
    每个 GPU 只存 1/TP_size 的 activation
    Backward 时做 all-gather 恢复完整 tensor

数据流:
  Forward (checkpoint 入口):
    hidden_states [s, b, h]
      → reduce_scatter → hidden_shard [s/TP, b, h]  ← 存储 (小 TP 倍)
  
  Backward (checkpoint 重算前):
    hidden_shard [s/TP, b, h]
      → all-gather → hidden_states [s, b, h]  ← 重算用

内存节省: checkpoint activation / TP_size
通信代价: 1次 all-gather (与 SP 的 all-gather 合并，近乎零额外开销)

约束: 必须 sequence_parallel=True (已有 scatter/gather 通信基础设施)
```

---

## 6. FP8 Activation Storage

### 6.1 原理

```
源码: transformer_config.py — activation_func_fp8_input_store

标准: activation 以 BF16 存储 (2 bytes/element)
FP8:  activation 以 E4M3 存储 (1 byte/element)

节省: 50% activation 内存
代价: backward 时 activation 精度降低 (E4M3 精度 ~1/16)

适用条件:
  - activation 值分布集中 (无大 outlier)
  - 配合 scaling factor 可减轻精度损失
  - 与 selective recompute 互补:
    recompute 的模块不存储 activation → 无精度损失
    非 recompute 模块用 FP8 存储 → 省内存
```

---

## 7. Distributed Optimizer (ZeRO) 内存节省

### 7.1 分片效果

```
(详见第三章)

ZeRO-1/2 将 optimizer states + gradients 分片到 DP ranks:

Without ZeRO (DP=8, 10B params):
  每GPU: params(20GB) + optim(80GB) + grads(20GB) = 120GB

With ZeRO (DP=8):
  每GPU: params(20GB) + optim(10GB) + grads(2.5GB) = 32.5GB
  节省: 87.5 GB/GPU (73%)

关键: ZeRO 优化的是 optimizer states 和 grads
      Activations 不受 ZeRO 影响 → 需要 recompute/offload
```

### 7.2 Param/Grad Buffer 连续内存

```
源码: distributed/param_and_grad_buffer.py

设计:
  所有参数的梯度存储在连续 buffer 中
  → 避免碎片化
  → 一次 all-reduce/reduce-scatter 覆盖多个参数
  → bucket-based 通信与计算 overlap

内存占用 = max(param_buffer, grad_buffer) ≈ 2 × model_size (BF16)
```

---

## 8. Gradient Accumulation 内存效应

### 8.1 机制

```
grad_accum_steps = global_batch / (micro_batch × dp_size)

Forward/Backward 逐 micro-batch 执行:
  for step in range(grad_accum_steps):
      output = forward(micro_batch[step])
      backward(output)  # grad 累加到 param.grad
      # activation 在 backward 后立即释放
  optimizer.step()

内存效应:
  - Activation 内存 = 1 个 micro_batch 的量 (而非 global_batch)
  - Gradient 内存 = 不变 (FP32 累加)
  - 允许 global_batch=很大 而 micro_batch=很小
```

### 8.2 与 Recompute 的关系

```
grad_accum_steps 增大时:
  activation 内存不变 (始终 1 个 micro_batch)
  但 recompute 的额外计算 × grad_accum_steps
  → 总 recompute 开销 = (1 + recompute_ratio) × grad_accum_steps

所以: 大 grad_accum + full recompute = 计算开销翻倍
     大 grad_accum + selective recompute = 计算开销增加可控
```

---

## 9. 内存优化策略交互矩阵

| 策略A \ 策略B | Selective Recompute | Full Recompute | Offload | Dist-Opt | FP8 Act | Dist-Act |
|---------------|--------------------:|---------------:|--------:|---------:|--------:|---------:|
| Selective Recompute | - | 互斥 | ✓互补 | ✓独立 | ✓互补 | ✓ |
| Full Recompute | 互斥 | - | 部分冗余 | ✓独立 | N/A | ✓ |
| Offload | ✓互补 | 部分冗余 | - | ✓独立 | ✓ | ✓ |
| Dist-Opt | ✓独立 | ✓独立 | ✓独立 | - | ✓ | ✓ |
| FP8 Act Store | ✓互补 | N/A | ✓ | ✓ | - | ✓ |
| Dist Saved Act | ✓ | ✓ | ✓ | ✓ | ✓ | - |

说明：
- 互补: 两者优化不同部分，叠加效果好
- 互斥: 不能同时使用
- 独立: 不影响彼此
- 部分冗余: full recompute 后 activation 已被释放，offload 意义减小

---

## 10. 关键设计决策

| 决策 | 选择 | 替代方案 | 为什么 |
|------|------|----------|--------|
| 默认 recompute | selective (core_attn) | full | ROI最高 (60%节省/25%开销) |
| Offload 粒度 | 模块级 (fine-grained) | 层级 | 更灵活的 overlap 控制 |
| CPU buffer | Pinned memory pool | 动态分配 | 避免反复 cudaMallocHost |
| Offload stream | 独立 D2H/H2D stream | 共享计算 stream | 异步不阻塞计算 |
| Distribute act | scatter 到 TP group | scatter 到 DP group | TP已有scatter基础设施 |
| Grad 精度 | FP32 累加 | BF16 累加 | 长 accum_steps 需要精度 |

---

## 11. 实践建议

### 11.1 配置推荐路径

```
Step 1 (基础): selective_recompute + distributed_optimizer
  → 通常足够 (60% activation节省 + 73% optimizer节省)

Step 2 (进阶): + distribute_saved_activations (需SP)
  → 额外 1/TP activation 节省

Step 3 (长序列): + fine_grained_offloading (core_attn)
  → seq² 项 offload 到 CPU

Step 4 (极端): full recompute (uniform, chunk=2-4)
  → 最后手段，性能损失大
```

### 11.2 内存估算公式

```
总GPU内存 ≈ 
  params_per_gpu × 2B                          # BF16 params
  + params_per_gpu × (8B + 4B) / DP            # Optimizer (ZeRO)
  + params_per_gpu × 2B / DP                   # Gradients (ZeRO)
  + activation_per_layer × layers_per_gpu      # Activations
    × (1 - recompute_savings)
    × (1 - offload_savings)
    / (TP if distribute_saved_act)
  + comm_buffers                               # ~1-2 GB

其中 params_per_gpu = total_params / (TP × PP)
```
