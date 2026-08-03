# 第11章：TransformerEngine-FL 深度源码解析

**源码位置**: `/workspace/deps/TransformerEngine-FL/`

## 1. 架构概览

TransformerEngine-FL (TE-FL) 是 NVIDIA TransformerEngine 的 FlagScale 定制分支，提供 **FP8 量化训练**、**高性能融合算子**和**通信-计算重叠 (Userbuffers)**三大核心能力。

### 1.1 目录结构与行数

```
transformer_engine/
├── common/              # 跨框架公共组件 (C++/CUDA)
│   ├── recipe/          # FP8 recipe (DelayedScaling, CurrentScaling, MXFP8)
│   └── fused_attn/      # cuDNN 融合注意力后端
├── pytorch/             # PyTorch 前端 (核心)
│   ├── module/          # 高层融合模块
│   │   ├── linear.py            (1670行) — FP8 Linear + UB overlap
│   │   ├── layernorm_linear.py  (~1200行) — LayerNorm + Linear 融合
│   │   └── layernorm_mlp.py     (~1500行) — LayerNorm + MLP 融合
│   ├── attention/       # Attention 实现
│   │   └── dot_product_attention/
│   │       ├── __init__.py           — DotProductAttention 主入口
│   │       └── context_parallel.py   (4365行) — CP 三种模式
│   ├── quantization.py  (1411行) — FP8GlobalStateManager + amax 管理
│   ├── distributed.py   (2125行) — 激活 recompute + 分布式工具
│   ├── graph.py         (1400行) — CUDA Graph 封装
│   └── cpu_offload.py   (943行)  — 激活 CPU offload
└── plugin/              # 硬件后端插件系统
```

### 1.2 核心设计理念

| 理念 | 实现方式 |
|------|---------|
| 透明 FP8 | 模块接口与 PyTorch Linear 兼容，内部自动量化/反量化 |
| 融合算子 | LayerNorm+Linear+Bias → 单 CUDA kernel，减少内存带宽瓶颈 |
| 通信隐藏 | Userbuffers 将 GEMM 与 all-gather/reduce-scatter 重叠 |
| 多 Recipe | DelayedScaling / CurrentScaling / MXFP8 / BlockScaling / NVFP4 |

---

## 2. FP8GlobalStateManager (quantization.py:237-600+)

### 2.1 设计定位

FP8 训练的核心挑战是 **scaling factor 管理** — 每个 FP8 张量需要一个合适的 scale 来映射动态范围。FP8GlobalStateManager 是管理所有 FP8 模块 scale 状态的全局单例。

### 2.2 全局状态字段 (L242-264)

```python
class FP8GlobalStateManager:
    # 开关状态
    FP8_ENABLED = False            # 是否启用 FP8
    FP8_CALIBRATION = False        # 是否在校准模式（收集 amax 不实际量化）
    FP8_RECIPE = None              # 当前 Recipe 实例
    FP8_DISTRIBUTED_GROUP = None   # amax all-reduce 通信组
    FP8_PARAMETERS = False         # 参数是否以 FP8 存储
    IS_FIRST_FP8_MODULE = False    # 首模块标记（触发全局更新）
    FP8_GRAPH_CAPTURING = False    # CUDA Graph capture 中
    AUTOCAST_DEPTH = 0             # 嵌套 autocast 深度
    
    # 全局 buffer（DelayedScaling 专用）
    global_amax_buffer = {}        # {key: [amax_tensor, ...]}
    global_amax_history_buffer = {}# {key: [amax_history_tensor, ...]}
    global_scale_buffer = {}       # {key: [scale_tensor, ...]}
    fp8_tensors_recompute_buffer = []  # recompute 场景的 FP8 张量暂存
```

### 2.3 Delayed Scaling 全局 Amax 管理

**核心流程：**

```
┌─────────────────────────────────────────────────────────────────────┐
│         Delayed Scaling — Amax 收集与 Scale 更新                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ Forward pass:                                                       │
│   每个 FP8 module 记录本步 amax(|tensor|) → global_amax_buffer       │
│                                                                     │
│ reduce_and_update_fp8_tensors(forward=True):  [L502-553]            │
│   1. torch.cat(amax_buffer) → contiguous_amax                      │
│   2. if recipe.reduce_amax:                                         │
│        all_reduce(contiguous_amax, MAX, group=fp8_group)            │
│      → 跨 TP/DP ranks 取全局最大 amax                                │
│   3. fused_amax_and_scale_update_after_reduction():                 │
│        a. 更新 amax_history (rolling window)                        │
│        b. amax_compute_algo: "max" → max(history) / "most_recent"   │
│        c. scale = fp8_max / (computed_amax * 2^margin)              │
│      → 计算下一步使用的 scale                                         │
│                                                                     │
│ 关键: scale 基于历史 amax 计算 → "delayed" by 1 step                 │
│       这确保了 scale 与 CUDA Graph capture 兼容                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.4 add_fp8_tensors_to_global_buffer (L364-416)

```python
@classmethod
def add_fp8_tensors_to_global_buffer(cls, fp8_meta):
    """每个 FP8 module 在 forward 时调用一次，注册到全局 buffer"""
    
    # 只有 delayed scaling 需要全局 buffer
    if not fp8_meta["recipe"].delayed():
        return
    
    # 为 forward 和 backward 分别注册
    for forward in (True, False):
        key = cls.get_key_in_buffer(forward, recipe, fp8_group)
        # 追加: amax_history[0] (当前步), amax_history (完整历史), scale
        cls.global_amax_buffer[key].append(fp8_meta[tensor_key].amax_history[0])
        cls.global_amax_history_buffer[key].append(fp8_meta[tensor_key].amax_history)
        cls.global_scale_buffer[key].append(fp8_meta[tensor_key].scale)
```

### 2.5 autocast_enter/exit (L567-620+)

```python
@classmethod
def autocast_enter(cls, enabled, calibrating, fp8_recipe, fp8_group, _graph):
    """进入 FP8 autocast 区域"""
    cls.FP8_ENABLED = enabled
    cls.FP8_RECIPE = fp8_recipe
    cls.FP8_DISTRIBUTED_GROUP = fp8_group
    
    if cls.AUTOCAST_DEPTH == 0:
        cls.IS_FIRST_FP8_MODULE = True  # 首模块触发全局更新
    cls.AUTOCAST_DEPTH += 1
    
    # 验证硬件支持
    if enabled:
        fp8_available, reason = cls.is_fp8_available()
        assert fp8_available, reason
```

### 2.6 支持的量化格式

| Recipe 类 | 量化方式 | Scale 粒度 | 硬件要求 |
|-----------|---------|-----------|---------|
| `DelayedScaling` | per-tensor, 基于历史 amax | Tensor 级 | Hopper+ |
| `Float8CurrentScaling` | per-tensor, 即时计算 | Tensor 级 | Hopper+ |
| `MXFP8BlockScaling` | 按 block (32 元素) | Block 级 | Blackwell |
| `Float8BlockScaling` | 按 block (自定义大小) | Block 级 | Blackwell |
| `NVFP4` | 4-bit | Block 级 | Blackwell |

---

## 3. _Linear 自定义 Autograd (module/linear.py:82-985)

### 3.1 Forward 签名 (L88-131)

`_Linear.forward` 接收 30+ 个参数通过 `non_tensor_args` 元组传递：

```python
@staticmethod
def forward(ctx, weight, inp, bias, non_tensor_args):
    (
        is_first_microbatch,
        fp8, fp8_calibration,           # FP8 状态
        wgrad_store,                     # wgrad 延迟计算
        input_quantizer, weight_quantizer,  # 量化器
        output_quantizer, grad_input_quantizer,
        grad_weight_quantizer, grad_output_quantizer,
        fuse_wgrad_accumulation,         # wgrad 累加优化
        cpu_offloading,                  # 激活卸载
        tp_group, tp_size,              # TP 配置
        sequence_parallel, tensor_parallel,
        activation_dtype, parallel_mode,  # "column" / "row"
        is_grad_enabled,
        ub_overlap_rs_fprop,            # UB: reduce-scatter in forward
        ub_overlap_ag_dgrad,            # UB: all-gather in dgrad
        ub_overlap_ag_fprop,            # UB: all-gather in forward
        ub_overlap_rs_dgrad,            # UB: reduce-scatter in dgrad
        ub_bulk_dgrad, ub_bulk_wgrad,   # UB: bulk overlap
        ub_name,                         # UB 通道名
        ...
    ) = non_tensor_args
```

### 3.2 Forward 数据流

```
┌──────────────────────────────────────────────────────────────────┐
│              _Linear.forward() 数据流                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│ [Input Preparation]                                              │
│  if column_parallel + sequence_parallel:                         │
│    inputmat = all_gather(inp, tp_group)  ← 或 UB ag_fprop        │
│  if fp8:                                                         │
│    inputmat = input_quantizer.quantize(inputmat)  → FP8 tensor   │
│                                                                  │
│ [Weight Quantization]                                            │
│  if fp8:                                                         │
│    weight_fp8 = weight_quantizer.quantize(weight)                │
│    → 使用 delayed/current scale 映射到 E4M3 格式                  │
│                                                                  │
│ [GEMM Execution]                                                 │
│  output = tex.gemm(inputmat_fp8, weight_fp8)                     │
│    → cuBLAS FP8 GEMM, 结果为 BF16/FP32                           │
│    → 或 UB overlap: GEMM + reduce-scatter 同时执行                │
│                                                                  │
│ [Output Processing]                                              │
│  if row_parallel:                                                │
│    output = reduce_scatter(output, tp_group)  ← 或 UB rs_fprop   │
│  output += bias (if any)                                         │
│                                                                  │
│ [Save for Backward]                                              │
│  ctx.save_for_backward(inputmat, weight, ...)                    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.3 Backward 中的 Userbuffers Overlap (L496-985)

```python
@staticmethod
def backward(ctx, grad_output):
    # Phase 1: dgrad (input gradient)
    # if ub_overlap_ag_dgrad:
    #   all-gather grad_output 与 dgrad GEMM 重叠
    dgrad = tex.gemm(grad_output, weight.T)  # + UB ag 通信
    
    # Phase 2: wgrad (weight gradient)  
    # if ub_bulk_wgrad:
    #   wgrad GEMM 与 dgrad 的 reduce-scatter 重叠
    wgrad = tex.gemm(input.T, grad_output)   # + UB rs 通信
    
    # Wgrad accumulation 优化:
    if fuse_wgrad_accumulation:
        # 直接累加到 weight.main_grad，避免分配临时 wgrad tensor
        weight.main_grad += wgrad  # in-place
```

---

## 4. Attention 模块

### 4.1 DotProductAttention 概览

TE-FL 的 Attention 支持多种后端和并行模式：

| 后端 | 触发条件 | 适用场景 |
|------|---------|---------|
| FlashAttention | `NVTE_FLASH_ATTN=1` | 通用，长序列 |
| FusedAttention (cuDNN) | `NVTE_FUSED_ATTN=1` | FP8 attention |
| UnfusedAttention | fallback | 调试/不支持的配置 |

### 4.2 FP8 Attention

```
条件: cuDNN ≥ 9.5 + Hopper/Blackwell + NVTE_FP8_DPA_BWD=1

Forward:
  Q, K, V (BF16) → FP8 quantize → cuDNN fused SDPA → Output (BF16)
  
Backward:
  如果 NVTE_FP8_DPA_BWD=1:
    dO, Q, K, V → FP8 → cuDNN backward → dQ, dK, dV (BF16)
  否则:
    BF16 backward (精度更高但更慢)
```

### 4.3 Context Parallelism (context_parallel.py, 4365行)

**三种 CP 通信模式：**

| 模式 | 实现类 | 行号 | 通信方式 | 显存/通信权衡 |
|------|--------|------|---------|-------------|
| P2P Ring | `AttnFuncWithCPAndKVP2P` | L1249 | Ring send/recv KV | 低显存，CP_size 步通信 |
| KV All-Gather | `AttnFuncWithCPAndKVAllGather` | L2797 | 一次性 all-gather KV | 高显存，1 步通信 |
| QKV-O A2A | `AttnFuncWithCPAndQKVOA2A` | L3307 | All-to-All Q/K/V/O | 中显存，适合 DeepSeek |

**P2P Ring 详细时序（CP=4 示例）：**

```
Step 0: rank_i 计算 attn(Q_i, K_i, V_i)
        同时发送 K_i,V_i → rank_{i+1}, 接收 K_{i-1},V_{i-1}

Step 1: rank_i 计算 attn(Q_i, K_{i-1}, V_{i-1})  — 累加到 output
        同时发送 K_{i-1},V_{i-1} → rank_{i+1}

Step 2: rank_i 计算 attn(Q_i, K_{i-2}, V_{i-2})
...

Step CP-1: 最后一轮，完成所有 KV 的 attention

关键优化: 通信与计算完全重叠 (double buffering)
```

**Causal Mask 处理：**
- 对称分布法: 每个 rank 持有 2 个 KV chunk (前半 + 后半)
- 避免三角 mask 导致的负载不均

---

## 5. Userbuffers 通信-计算重叠系统

### 5.1 核心思想

传统方式：GEMM → all-gather/reduce-scatter（串行）
Userbuffers：GEMM 与通信在不同 SM 上并行执行

```
┌─────────────────────────────────────────────────────┐
│           Userbuffers (UB) 重叠原理                   │
├─────────────────────────────────────────────────────┤
│                                                     │
│ 传统 (no overlap):                                   │
│   ├── GEMM ──────┤├── All-Gather ──┤               │
│   Time: T_gemm + T_comm                             │
│                                                     │
│ UB overlap (AG):                                    │
│   ├── GEMM (分块) ─────────────────┤               │
│   ├── AG chunk1 ──┤                                 │
│        ├── AG chunk2 ──┤                            │
│             ├── AG chunk3 ──┤                       │
│   Time: max(T_gemm, T_comm) ≈ T_gemm               │
│                                                     │
│ UB overlap (RS):                                    │
│   ├── GEMM (分块) ─────────────────┤               │
│        ├── RS chunk1 ──┤                            │
│             ├── RS chunk2 ──┤                       │
│   Time: max(T_gemm, T_comm) ≈ T_gemm               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 5.2 UB 类型定义 (tex.CommOverlapType)

| 类型 | Forward 使用 | Backward 使用 | 通信操作 |
|------|-------------|-------------|---------|
| AG (All-Gather) | ub_overlap_ag_fprop | ub_overlap_ag_dgrad | 收集分片输入 |
| RS (Reduce-Scatter) | ub_overlap_rs_fprop | ub_overlap_rs_dgrad | 分发分片输出 |
| BULK | — | ub_bulk_dgrad/wgrad | 批量通信 |

### 5.3 Linear 中的 UB 集成 (linear.py L149-163)

```python
# Forward 中选择 UB 通道
ub_obj = None
ub_type = None
if ub_overlap_rs_fprop:
    ub_obj = get_ub(ub_name + "_fprop", fp8)  # 获取预注册的 UB 对象
    ub_type = tex.CommOverlapType.RS
elif ub_overlap_ag_fprop:
    ub_obj = get_ub(ub_name + "_fprop", fp8)
    ub_type = tex.CommOverlapType.AG

# 传给 tex.gemm → CUDA kernel 内部流水线化 GEMM + 通信
output = tex.gemm(input, weight, ub_obj=ub_obj, ub_type=ub_type)
```

### 5.4 典型 TP 配置下的 UB 分配

```
Column Parallel Linear (QKV):
  Forward:  all-gather input → GEMM      → ub_overlap_ag_fprop
  Backward: GEMM → reduce-scatter dgrad  → ub_overlap_rs_dgrad

Row Parallel Linear (O/Down):  
  Forward:  GEMM → reduce-scatter output  → ub_overlap_rs_fprop
  Backward: all-gather grad → GEMM        → ub_overlap_ag_dgrad
```

---

## 6. 激活 Recompute 与 CPU Offload (distributed.py)

### 6.1 activation_recompute_forward 上下文 (L243-290)

```python
class activation_recompute_forward(AbstractContextManager, ContextDecorator):
    """标记当前处于 recompute forward 阶段"""
    def __init__(self, activation_recompute=False, recompute_phase=False):
        self.activation_recompute = activation_recompute
        self.recompute_phase = recompute_phase
    
    def __enter__(self):
        # 设置全局标记：当前是否在 recompute phase
        # FP8 module 据此决定是否更新 amax (recompute 时不更新)
```

### 6.2 _CheckpointFunction (L332-478)

TE 自定义的 gradient checkpoint 实现，与 FP8 状态深度集成：

```python
class _CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, *args):
        # 1. 保存 RNG 状态（CUDA + CPU + TE tracker）
        # 2. 执行 forward (不保存中间激活)
        # 3. 保存 FP8 autocast 状态
        ctx.fwd_fp8_state = FP8GlobalStateManager.get_autocast_state()
    
    @staticmethod
    def backward(ctx, *grads):
        # 1. 恢复 RNG 状态
        # 2. 恢复 FP8 状态
        FP8GlobalStateManager.set_autocast_state(ctx.fwd_fp8_state)
        # 3. 在 activation_recompute_forward 上下文中重新 forward
        with activation_recompute_forward(recompute_phase=True):
            outputs = ctx.run_function(*ctx.inputs)
        # 4. 执行 backward
```

**关键：FP8 recompute 时不能更新 amax/scale（否则干扰 delayed scaling 的统计）**

### 6.3 CPU Offload (cpu_offload.py, 943行)

将激活张量在 forward 时移到 CPU，backward 时移回 GPU：

```python
# 工作流:
# Forward: activation → pin_memory → D2H copy (async) → CPU storage
# Backward: CPU storage → H2D copy (async) → GPU → grad computation
# 
# 优化: 使用 CUDA stream 异步 copy，与计算重叠
```

---

## 7. CUDA Graph 支持 (graph.py, 1400行)

### 7.1 挑战

FP8 的 delayed scaling 需要动态更新 scale，但 CUDA Graph 要求固定操作。

### 7.2 解决方案

```python
# graph.py 关键设计:
# 1. Scale/amax 使用固定地址的 tensor（graph capture 时分配）
# 2. Scale 更新作为 graph 的一部分被 capture
# 3. FP8GlobalStateManager.FP8_GRAPH_CAPTURING = True
#    → 通知所有 module 使用 graph-safe 路径
```

### 7.3 FP8 + CUDA Graph 兼容

```
Graph Capture:
  1. 预分配 amax_buffer、scale_buffer（固定地址）
  2. 将 fused_amax_and_scale_update 操作纳入 graph
  3. 每次 replay 时 buffer 内容更新但地址不变

关键约束:
  - Scale 更新必须在 graph 内（不能在 host-side 动态决策）
  - add_fp8_tensors_to_global_buffer() 在 capture 前执行
  - IS_FIRST_FP8_MODULE 标记确保只更新一次
```

---

## 8. 与 Megatron-LM-FL 的集成

### 8.1 模块替换链

```
Megatron model 初始化:
  if use_te:
    ColumnParallelLinear → te.Linear(parallel_mode="column", ...)
    RowParallelLinear    → te.Linear(parallel_mode="row", ...)
    LayerNorm + MLP      → te.LayerNormMLP(...)
    Attention            → te.DotProductAttention(...)
```

### 8.2 FP8 训练启用流程

```python
# Megatron training loop:
with te.fp8_autocast(
    enabled=True,
    fp8_recipe=DelayedScaling(
        margin=0,
        fp8_format=Format.HYBRID,  # fwd=E4M3, bwd=E5M2
        amax_history_len=1024,
        amax_compute_algo="max",
    ),
    fp8_group=data_parallel_group,
):
    output = model(input)
    loss = loss_fn(output)
    loss.backward()
    
# 每步结束后:
FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=True)
FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)
```

### 8.3 Userbuffers 初始化

```python
# Megatron 启动时调用 te.module.base.initialize_ub()
# 预注册所有 UB 通道:
#   "qkv_fprop", "qkv_dgrad", "proj_fprop", "proj_dgrad"
#   "fc1_fprop", "fc1_dgrad", "fc2_fprop", "fc2_dgrad"
# 每个通道分配固定 GPU 内存作为通信 buffer
```

---

## 9. Recipe 详解

### 9.1 DelayedScaling（默认推荐）

```python
DelayedScaling(
    margin=0,                    # scale = fp8_max / (amax * 2^margin)
    fp8_format=Format.HYBRID,    # fwd: E4M3 (精度优先), bwd: E5M2 (范围优先)
    amax_history_len=1024,       # 保留最近 1024 步的 amax 历史
    amax_compute_algo="max",     # 从历史中取 max（保守但安全）
    reduce_amax=True,            # 跨 DP ranks all-reduce amax
)
```

**优势:** CUDA Graph 兼容，稳定性好
**劣势:** Scale 滞后 1 步，极端分布变化时可能 overflow

### 9.2 Float8CurrentScaling（低延迟）

```python
Float8CurrentScaling(
    # 即时计算: scale = fp8_max / max(|current_tensor|)
    # 无历史、无滞后
)
```

**优势:** Scale 精确匹配当前数据
**劣势:** 不兼容 CUDA Graph（需要动态计算 scale）

### 9.3 MXFP8BlockScaling（Blackwell 专用）

```python
MXFP8BlockScaling(
    block_size=32,  # 每 32 元素共享一个 scale (E8M0 格式)
)
```

**优势:** 细粒度 scale → 几乎无精度损失
**劣势:** 仅 Blackwell (SM100+) 支持

---

## 10. 性能特征

### 10.1 H100 实测加速比

| 操作 | BF16 TFLOPS | FP8 TFLOPS | 加速比 |
|------|------------|-----------|--------|
| GEMM (4096×4096×4096) | ~400 | ~800 | 2.0× |
| Attention (seq=2K, head=32) | ~350 | ~600 | 1.7× |
| LayerNormMLP 融合 vs 分离 | — | — | 1.15× |
| UB overlap vs no overlap | — | — | 1.2-1.4× |
| 端到端训练 (7B) | — | — | 1.3-1.8× |

### 10.2 显存影响

| FP8 组件 | 额外显存开销 |
|----------|------------|
| amax_history (per tensor) | 1024 × 4B = 4KB |
| scale + scale_inv (per tensor) | 8B |
| FP8 weight 缓存 | weight_size / 2 (额外 E4M3 副本) |
| Userbuffers | 配置的 buffer_size（通常 64-256MB）|

---

## 11. 设计决策与权衡

| 设计决策 | 选择 | 原因 |
|----------|------|------|
| Global amax buffer | 集中式管理 | 避免每个 module 独立 all-reduce → 合并为一次 |
| DelayedScaling 为默认 | 1步滞后 | CUDA Graph 兼容 + 训练稳定 |
| Hybrid format (E4M3/E5M2) | Forward/Backward 区分 | Forward 精度优先，Backward 范围优先 |
| Userbuffers 预注册 | 固定 buffer | 避免 runtime 分配，CUDA Graph 兼容 |
| Autograd Function | 手写 forward/backward | 细粒度控制量化/通信/GEMM 融合 |
| 30+ non_tensor_args | 元组传递 | autograd.Function 对 tensor 参数有特殊处理 |

---

## 12. 调优建议

### 12.1 FP8 训练配置

```yaml
# Megatron args:
--fp8-format hybrid              # E4M3 forward + E5M2 backward
--fp8-amax-history-len 1024      # 历史窗口
--fp8-amax-compute-algo max      # 保守取最大值
--fp8-wgrad                      # wgrad 也用 FP8（进一步加速）

# 环境变量:
NVTE_FP8_DPA_BWD=1              # 启用 FP8 attention backward
NVTE_FLASH_ATTN=1               # FlashAttention (非 FP8 fallback)
NVTE_FUSED_ATTN=1               # cuDNN fused attention
```

### 12.2 Userbuffers 配置

```yaml
--tp-comm-overlap                # 启用 UB
--tp-comm-overlap-cfg /path.yaml # 详细配置

# UB config YAML 示例:
qkv_fprop:
  method: ring_exchange          # 或 pipeline, bulk
  num_splits: 4                  # GEMM 分块数
  cga_size: 2                    # Cooperative Group Array
  set_sm_margin: true            # 为通信预留 SM
  num_sm: 16                     # 通信使用的 SM 数
```

### 12.3 精度问题排查

```python
# 1. 开启 calibration 模式（只收集 amax，不实际量化）
with te.fp8_autocast(enabled=True, calibrating=True, ...):
    model(input)
# 检查 amax 分布是否合理

# 2. 增大 margin 应对 overflow
DelayedScaling(margin=2)  # scale = fp8_max / (amax * 4)

# 3. 缩短 history 应对分布突变
DelayedScaling(amax_history_len=16)

# 4. 切换到 CurrentScaling 排除 delayed 问题
Float8CurrentScaling()
```

---

## 13. 总结

TransformerEngine-FL 的核心价值：

1. **FP8 全生命周期管理**：FP8GlobalStateManager 集中管理 scale/amax，支持 delayed/current/block 三大 scaling 策略
2. **GEMM+通信融合**：Userbuffers 在 CUDA kernel 级别实现 GEMM 与 TP 通信重叠，接近理论带宽利用
3. **Autograd 深度定制**：_Linear 手写 forward/backward 精确控制量化时机、通信时机、内存分配
4. **多 CP 模式**：P2P Ring / All-Gather / A2A 三种上下文并行，覆盖不同序列长度和集群规模
5. **与 CUDA Graph 兼容**：fixed-address buffer + graph-safe scale 更新，支持低开销 replay
