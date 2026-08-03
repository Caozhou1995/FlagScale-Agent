# 01 - Pipeline Parallelism (PP) 完整分析

## 源码位置

| 文件 | 行数 | 功能 |
|------|------|------|
| `megatron/core/pipeline_parallel/schedules.py` | 2432 | 主调度器: 1F1B (有/无 interleaving) |
| `megatron/core/pipeline_parallel/combined_1f1b.py` | 449 | Combined 1F1B (EP A2A 通信隐藏) |
| `megatron/core/pipeline_parallel/hybrid_cp_schedule.py` | 665 | Hybrid CP+PP 调度 |
| `megatron/core/pipeline_parallel/p2p_communication.py` | — | P2P send/recv 通信原语 |
| `megatron/core/pipeline_parallel/utils.py` | 385 | ScheduleNode / AbstractSchedulePlan 基础设施 |
| `megatron/core/models/common/model_chunk_schedule_plan.py` | 639 | TransformerLayerSchedulePlan / ModelChunkSchedulePlan |
| `megatron/plugin/dualpipev/dualpipev_schedules.py` | 2007 | **FlagScale 扩展**: DualPipeV 双向流水线调度 |
| `megatron/plugin/dualpipev/fb_overlap/` | — | **FlagScale 扩展**: Forward-Backward 层级重叠 |

---

## 1. Schedule 调度入口

**入口函数**: `get_forward_backward_func()` (schedules.py:54)

```python
def get_forward_backward_func(pp_size, vp_size):
    if pp_size > 1:
        if get_dualpipev_pipeline_model_parallel_world_size() is not None:
            return forward_backward_pipelining_with_dualpipev  # FlagScale 扩展
        elif vp_size is not None:
            return forward_backward_pipelining_with_interleaving  # Virtual PP
        else:
            return forward_backward_pipelining_without_interleaving  # 标准 1F1B
    else:
        return forward_backward_no_pipelining  # PP=1, 纯 DP/TP
```

选择逻辑:
1. **DualPipeV** (FlagScale 独有) → 优先级最高
2. **Virtual PP (Interleaved)** → `virtual_pipeline_model_parallel_size > 1`
3. **Standard 1F1B** → 默认 PP>1 行为
4. **No Pipelining** → PP=1

**二级分派** (schedules.py:686, 1397): 在上述函数内部，若 `config.overlap_moe_expert_parallel_comm=True`，则进入 Combined 1F1B 路径:
```python
# no_pipelining 内 (schedules.py:686)
if config.overlap_moe_expert_parallel_comm and not forward_only:
    return combined_1f1b_schedule_for_no_pipelining(...)

# interleaved 内 (schedules.py:1397)
if config.overlap_moe_expert_parallel_comm and not forward_only:
    return combined_1f1b_schedule_for_interleaved_pipelining(...)
```

---

## 2. Schedule 变体详解

### 2.1 forward_backward_no_pipelining (schedules.py:621)

**适用条件**: `pipeline_parallel_size == 1`

**行为**: 顺序执行所有 microbatch 的 forward + backward, 无流水线重叠。

**数据流**:
```
GPU: [F0][F1]...[Fn-1][B0][B1]...[Bn-1]
     ←── all forward ──→←── all backward ──→
```

**Bubble**: 0% (无流水线，但也无 overlap)

**关键实现** (schedules.py:621-730):
- 前 n-1 个 microbatch 在 `no_sync()` 上下文中执行（跳过 gradient allreduce）
- 最后一个 microbatch 退出 `no_sync()` 触发 gradient 同步
- 支持 `forward_only` 模式（推理/评估）

---

### 2.2 forward_backward_pipelining_without_interleaving (schedules.py:2059)

**适用条件**: `pp_size > 1, virtual_pipeline_model_parallel_size == None`

**原理**: 经典 1F1B (One Forward One Backward) 三阶段调度:
- **Warmup 阶段**: 前面的 stage 先做 forward, 逐步填满 pipeline
- **1F1B 稳态阶段**: 每做一个 forward 紧跟一个 backward, 内存恒定
- **Cooldown 阶段**: 清空剩余 backward

**Warmup 数量计算** (schedules.py:852):
```python
num_warmup_microbatches = pipeline_parallel_size - pipeline_parallel_rank - 1
# Stage 0 (第一个): warmup = pp_size - 1 (最多)
# Stage pp-1 (最后): warmup = 0 (最少, 立即进入 steady state)
```

**Bubble 公式**:
```
bubble_fraction = (pp_size - 1) / num_microbatches
```

**时序图** (PP=4, num_microbatches=16, bubble=18.75%):
```
时间 →
Stage 0: [F0 ][F1 ][F2 ][F3 ][F4 B0][F5 B1][F6 B2]...[F12 B8 ][B9  ][B10 ][B11 ]
Stage 1:      [F0 ][F1 ][F2 ][F3 B0][F4 B1][F5 B2]...[F11 B8 ][B9  ][B10 ]
Stage 2:           [F0 ][F1 ][F2 B0][F3 B1][F4 B2]...[F10 B8 ][B9  ]
Stage 3:                [F0 ][F1 B0][F2 B1][F3 B2]...[F9  B8 ]
          ←─ warmup ─→ ←──────── 1F1B steady ────────→←cooldown→
```

**内存行为**:
- Warmup 阶段: 每个 forward 产生一份 activation, 内存线性增长
- Steady 阶段: 每做一个 backward 释放一份 activation, 净内存恒定
- 峰值内存 = `num_warmup_microbatches` 份 activation

**关键优化选项**:
| 选项 | 行号 | 作用 | 约束 |
|------|------|------|------|
| `overlap_p2p_comm` | 1604 | P2P send/recv 与计算重叠 | 需要 `CUDA_DEVICE_MAX_CONNECTIONS=1` |
| `overlap_p2p_comm_warmup_flush` | 1488 | Warmup/Cooldown 阶段也做 P2P overlap | 依赖 `overlap_p2p_comm` |
| `batch_p2p_comm` | 990 | 多个 send/recv 合并为一次 batch 调用 | 与 `overlap_p2p_comm` 互斥 |
| `deallocate_pipeline_outputs` | 171 | 发送后立即释放 output tensor 内存 | — |

---

### 2.3 forward_backward_pipelining_with_interleaving (schedules.py:914)

**适用条件**: `pp_size > 1, virtual_pipeline_model_parallel_size >= 2`

**原理**: 每个 GPU 持有多个 "virtual stage" (模型 chunk), 交替执行不同 chunk 的 microbatch。
- 等价于增加 pipeline "频率": 更细的 stage 意味着更短的单步时间
- 代价: P2P 通信量翻倍 (每个 virtual stage 都需要跨 GPU 通信)

**Warmup 计算** (schedules.py:859):
```python
num_warmup = (pp_size - pp_rank - 1) * 2 + (num_model_chunks - 1) * microbatch_group_size
```

**Bubble 公式** (arXiv:2104.04473 Section 4.2):
```
bubble_fraction ≈ (pp_size - 1) / (num_microbatches × num_model_chunks)
```

**示例** (PP=4, vpp=2, num_microbatches=16):
```
非交错 bubble: (4-1)/16 = 18.75%
交错 bubble:   (4-1)/(16×2) = 9.4%  → 减少约 50%
```

**Virtual Stage 到 GPU 的映射**:
```
PP=4, vpp=2 → 8 个 virtual stages
GPU 0: chunk[0] = layers 0-3,  chunk[1] = layers 28-31
GPU 1: chunk[0] = layers 4-7,  chunk[1] = layers 24-27
GPU 2: chunk[0] = layers 8-11, chunk[1] = layers 20-23
GPU 3: chunk[0] = layers 12-15, chunk[1] = layers 16-19
```
注意: chunk[1] 的 forward 方向是反向的 (从高层到低层), 形成 V 形路径。

**约束**:
- `num_layers` 必须被 `pp_size × vpp_size` 整除
- `overlap_p2p_comm` 在 interleaved schedule 中被强制关闭 (arguments.py:849)

---

### 2.4 Combined 1F1B (combined_1f1b.py) — EP A2A 通信隐藏调度

**设计动机**: MoE 模型的 Expert Parallelism 需要 All-to-All (A2A) 通信来 dispatch/combine tokens。A2A 延迟显著（尤其跨节点 EP）。Combined 1F1B 通过在**层级粒度**交错 forward/backward 来用计算隐藏 A2A 通信。

**核心思想**: 不是简单地把 forward+backward 合并为一个 CUDA Graph，而是将 forward microbatch i+1 的通信（dispatch A2A）与 backward microbatch i 的计算（MLP backward）在不同 CUDA stream 上并行执行。

**触发条件**: `config.overlap_moe_expert_parallel_comm = True` 且 `not forward_only`

#### 2.4.1 三层架构

```
┌─────────────────────────────────────────────────────────────────┐
│ Level 1: Microbatch 调度                                         │
│ combined_1f1b_schedule_for_no_pipelining (line 23)               │
│ combined_1f1b_schedule_for_interleaved_pipelining (line 116)     │
│ → 决定哪个 microbatch 的 F/B 配对                                  │
├─────────────────────────────────────────────────────────────────┤
│ Level 2: Model Chunk 调度                                        │
│ TransformerModelChunkSchedulePlan.run() (model_chunk_schedule_plan.py:484) │
│ → 逐层交错: forward_layer[i] + backward_layer[N-1-i]             │
├─────────────────────────────────────────────────────────────────┤
│ Level 3: Layer 内调度                                            │
│ TransformerLayerSchedulePlan.run() (model_chunk_schedule_plan.py:204)      │
│ → 双 stream 交错: comp_stream (attn/mlp) ∥ comm_stream (dispatch/combine) │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.4.2 Level 1: Microbatch 配对 (combined_1f1b.py:23-113)

**No-pipelining 场景** (PP=1): 将相邻 microbatch 的 F/B 配对:

```python
# 伪代码 (combined_1f1b_schedule_for_no_pipelining)
Phase 0: F[0] alone          # 第一个 forward 无配对
Phase 1: F[1] + B[0]         # forward mb1 与 backward mb0 交错
Phase 2: F[2] + B[1]         # forward mb2 与 backward mb1 交错
...
Phase n-1: F[n-1] + B[n-2]
Phase n:   B[n-1] alone      # 最后一个 backward 无配对
```

**时序对比**:
```
Standard (无 overlap):
[F0][F1][F2][F3][B0][B1][B2][B3]

Combined (有 overlap):
[F0][F1+B0][F2+B1][F3+B2][B3]
     ↑ A2A 通信被计算隐藏
```

**Interleaved 场景** (PP>1, vpp≥2): `combined_1f1b_schedule_for_interleaved_pipelining` (line 116) 将 interleaved schedule 中的 `forward_step_helper` + `backward_step_helper` 合并:

```python
# 原始 (schedules.py):
forward_step_helper(f_virtual_microbatch_id)   # preprocess → forward → postprocess
backward_step_helper(b_virtual_microbatch_id)  # preprocess → backward → postprocess

# Combined (combined_1f1b.py:116):
def combined_1f1b_schedule_for_interleaved_pipelining():
    forward_step_helper_preprocess(f_id)   # P2P recv, set model chunk
    backward_step_helper_preprocess(b_id)  # get saved tensors
    combined_forward_backward_step(...)    # F+B 层级交错执行
    forward_step_helper_postprocess(f_id)  # P2P send
    backward_step_helper_postprocess(b_id) # gradient accumulation
```

#### 2.4.3 Level 2: Model Chunk 内层间交错 (model_chunk_schedule_plan.py:484-639)

`TransformerModelChunkSchedulePlan.run()` 是核心调度器。给定 forward plan (N 层) 和 backward plan (M 层):

```python
@staticmethod
def run(f_schedule_plan, b_schedule_plan, b_grad, pre_forward, pre_backward, ...):
    # 1. Forward pre_process (embedding / input layernorm)
    f_input = f_schedule_plan.pre_process.forward()

    # 2. Backward post_process (output layernorm grad)
    b_grad = b_schedule_plan.post_process.backward(b_grad)

    # 3. 逐层交错: forward layer[i] 配对 backward layer[N-1-i]
    overlapped_layers = min(f_num_layers, b_num_layers)
    for i in range(overlapped_layers):
        f_layer = f_schedule_plan.get_layer(i)       # 正序
        b_layer = b_schedule_plan.pop_layer()        # 逆序 (FILO)
        f_input, b_grad = TransformerLayerSchedulePlan.run(
            f_layer, b_layer, f_input, b_grad
        )

    # 4. 剩余 backward-only 层
    for i in range(overlapped_layers, b_num_layers):
        _, b_grad = TransformerLayerSchedulePlan.run(None, b_layer, b_grad=b_grad)

    # 5. 剩余 forward-only 层
    for i in range(overlapped_layers, f_num_layers):
        f_input, _ = TransformerLayerSchedulePlan.run(f_layer, None, f_input=f_input)

    # 6. P2P overlap: post_forward (send_forward) 在 comm_stream 上
    #    与 backward attn_dw 在 comp_stream 上重叠
    with stream(comm_stream):
        post_forward(f_input, vp_stage)
    post_backward(b_grad, vp_stage)

    # 7. 延迟的 attn backward_dw (第一层的, 与 P2P 通信重叠)
    b_layer.attn.backward_dw()

    # 8. Forward post_process + Backward pre_process
    f_input = f_schedule_plan.post_process.forward(f_input)
    b_schedule_plan.pre_process.backward(b_grad)

    return f_input
```

**关键设计**: backward 层以 **FILO** (后进先出) 顺序处理，即 backward 从最后一层开始，与 forward 从第一层开始形成交叉配对:
```
Forward:  layer[0] → layer[1] → layer[2] → layer[3]
Backward: layer[3] → layer[2] → layer[1] → layer[0]
配对:     (F[0],B[3]) → (F[1],B[2]) → (F[2],B[1]) → (F[3],B[0])
```

#### 2.4.4 Level 3: Layer 内双 Stream 交错 (model_chunk_schedule_plan.py:204-269)

`TransformerLayerSchedulePlan.run()` 在单层内实现 **计算-通信重叠**:

```
层内子模块:
├── attn: Attention + LayerNorm + Router + Dispatch preprocess (comp_stream)
├── moe_dispatch: Dispatch All2All 通信 (comm_stream)
├── mlp: Expert MLP 计算 (comp_stream)
├── moe_combine: Combine All2All 通信 (comm_stream)
└── mtp_post_process: MTP 后处理 (comp_stream)
```

**双 Stream 时序** (forward layer[i] + backward layer[j]):
```
comm_stream: |combine_B[j]|  dispatch_F[i] → dispatch_B[j]  |combine_F[i]|
comp_stream: |  attn_F[i] |  mlp_B[j] → mlp_dw_B[j] → mlp_F[i]  |attn_B[j]|
             ←── phase1 ──→ ←────────── phase2 ─────────────→ ←── phase3 ──→
```

**伪代码** (model_chunk_schedule_plan.py:229-268):
```python
@staticmethod
def run(f_layer, b_layer, f_input, b_grad, is_last_layer_in_bwd):
    # Phase 1: backward combine + forward attn (并行)
    b_grad = b_layer.moe_combine.backward(b_grad)       # comm_stream
    f_input = f_layer.attn.forward(f_input)              # comp_stream

    # Phase 2: forward dispatch + backward mlp (并行)
    f_input = f_layer.moe_dispatch.forward(f_input)      # comm_stream
    b_layer.mlp.backward_dw()                            # comp_stream (weight grad)
    b_grad = b_layer.moe_dispatch.backward(b_grad)       # comm_stream

    # Phase 3: forward mlp+combine + backward attn (并行)
    f_input = f_layer.mlp.forward(f_input)               # comp_stream
    f_input = f_layer.moe_combine.forward(f_input)       # comm_stream
    b_grad = b_layer.attn.backward(b_grad)               # comp_stream

    # 延迟 attn backward_dw 到下一轮 (与 P2P comm 重叠)
    if not is_last_layer_in_bwd:
        b_layer.attn.backward_dw()

    return f_input, b_grad
```

#### 2.4.5 ScheduleNode 基础设施 (utils.py:163-337)

**ScheduleNode** 是计算图的最小单元:
```python
class ScheduleNode:
    def __init__(self, forward_func, stream, event, backward_func=None, name=""):
        self.stream = stream          # 执行在哪个 CUDA stream
        self.event = event            # 用于 stream 间同步
        self.forward_func = forward_func
        self.backward_func = backward_func

    def forward(self, inputs):
        with stream_acquire_context():  # event.wait → compute → event.record
            self.inputs = [detach(e) for e in inputs]
            self.output = self.forward_func(*self.inputs)
        return self.output

    def backward(self, output_grad):
        with stream_acquire_context():
            run_backward(self.output, output_grad)  # autograd backward
        grads = [e.grad for e in self.inputs]
        self._release_state()  # 立即释放引用, 避免内存泄漏
        return grads
```

**Event 同步机制**: 同一个 microbatch 内的所有 node 共享一个 `cuda.Event`。每个 node 执行前 `event.wait(stream)` 确保前序依赖完成，执行后 `event.record(stream)` 通知后续 node。

#### 2.4.6 SchedulePlan 构建流程

```
GPTModel.forward(input_ids, ...)
    ↓ return_schedule_plan=True
GPTModel.build_schedule_plan(input_ids, position_ids, attention_mask, ...)
    ↓
TransformerModelChunkSchedulePlan.__init__()
    ├── PreProcessNode(embedding, input_layernorm)
    ├── for each layer:
    │   └── TransformerLayerSchedulePlan(layer, event, chunk_state, comp_stream, comm_stream)
    │       ├── attn node (comp_stream)
    │       ├── moe_dispatch node (comm_stream)  ← A2A 通信
    │       ├── mlp node (comp_stream)
    │       ├── moe_combine node (comm_stream)   ← A2A 通信
    │       └── mtp_post_process node (comp_stream)
    └── PostProcessNode(output_layernorm, final_linear)
```

#### 2.4.7 性能收益分析

**隐藏的通信**: EP A2A dispatch + combine (双向)
- 每层 2 次 A2A: dispatch (forward) + combine (forward)
- backward 同样 2 次 A2A
- Combined 将 forward A2A 与 backward compute 重叠，反之亦然

**适用场景**:
- MoE 模型 + Expert Parallelism (`--num-experts > 1, --expert-model-parallel-size > 1`)
- A2A 延迟占比显著时收益大（跨节点 EP）

**不适用场景**:
- Dense 模型（无 A2A 通信，Combined 无意义）
- EP=1（All-to-All 退化为本地操作）
- `forward_only=True`（评估模式无 backward）

**约束**:
- 仅支持 GPTModel（`assert isinstance(unwrapped_model, GPTModel)`，line 343）
- 模型必须实现 `build_schedule_plan()` 方法
- 与 CUDA Graph 不兼容（动态 schedule plan 无法 capture）

---

### 2.5 Hybrid CP Schedule (hybrid_cp_schedule.py:482)

**入口**: `hybrid_context_parallel_forward_backward()`

**设计动机**: Context Parallelism (CP) 将长序列切分到多个 rank，需要 all-gather/reduce-scatter 通信。当 CP 与 PP 同时使用时，两者的通信模式可能冲突（CP 的 collective 操作 vs PP 的 P2P send/recv），需要协同调度。

**原理**:
- CP 通信 (all-gather context) 安排在 PP warmup/cooldown 的 idle 时间
- PP P2P 通信与 CP 通信使用不同 NCCL stream, 避免互相阻塞
- 调度器统一管理两种通信的时序依赖

**适用条件**: `context_parallel_size > 1` 且 `pipeline_parallel_size > 1`

**约束**:
- CP group 和 PP group 不能重叠（不同的 GPU 分组维度）
- 增加内存压力: 需要缓存更多 KV 用于跨 CP rank 的 attention

---

### 2.6 DualPipeV (FlagScale 扩展)

**源码**: `megatron/plugin/dualpipev/` (dualpipev_schedules.py: 2007 行)

**设计动机**: 标准 1F1B 的 bubble 为 `(pp-1)/m`，对于大 PP 或小 batch 场景浪费严重。DualPipeV 通过双向流水线 + Forward-Backward 细粒度重叠将 bubble 降至接近 0。

**核心思想**:
1. **双向流水线**: 模型分为 2 个 chunk，分别从两个方向执行 pipeline
   - Chunk 0: 正向流动 (GPU 0 → GPU n-1)
   - Chunk 1: 反向流动 (GPU n-1 → GPU 0)
2. **F/B Overlap**: 当一个 chunk 在某 GPU 上做 forward 时，另一个 chunk 可以同时做 backward
3. **层级细粒度重叠**: 不是整个 stage 级别的 overlap，而是 Transformer layer 内部子模块级别

**时序图** (PP=4, DualPipeV):
```
GPU 0: [F0_c0][F1_c0+B0_c1][F2_c0+B1_c1]...[Bn_c1]
GPU 1: [F0_c0][F1_c0+B0_c1][F2_c0+B1_c1]...[Bn_c1]
GPU 2: [F0_c1][F1_c1+B0_c0][F2_c1+B1_c0]...[Bn_c0]
GPU 3: [F0_c1][F1_c1+B0_c0][F2_c1+B1_c0]...[Bn_c0]
        c0 = chunk0 (正向), c1 = chunk1 (反向)
```

**文件结构与职责**:
```
dualpipev/
├── dualpipev_schedules.py          # 主调度: microbatch 分配, P2P 通信, chunk 管理
└── fb_overlap/                     # Forward-Backward 层级重叠实现
    ├── gpt_model.py                # GPTModel 适配: gpt_model_forward_backward_overlapping()
    ├── transformer_block.py        # TransformerBlock 级别 F/B overlap 入口
    ├── transformer_layer.py        # 层级 overlap: transformer_layer_forward_backward_overlapping()
    ├── modules/
    │   ├── attention.py            # Attention 子模块 overlap (QKV proj + FlashAttn + Out proj)
    │   └── token_dispatcher.py     # MoE token dispatch overlap (A2A 与计算重叠)
    └── overlap_funcs/
        ├── fwd.py                  # Forward overlap 辅助函数
        ├── bwd.py                  # Backward overlap 辅助函数
        └── fwdbwd.py               # Forward+Backward 联合 overlap
```

**关键实现细节**:
1. `set_dualpipe_chunk(chunkid)` / `get_dualpipe_chunk()`: 全局状态追踪当前执行的 chunk
2. `disable_dw_detach(model)`: 禁用 weight gradient 延迟计算，确保 F/B overlap 时 wgrad 正确
3. P2P 通信使用 `P2PCommunicator` 类，支持双向 send/recv
4. 采用 MindSpeed (华为 Ascend) 的调度算法移植到 NVIDIA GPU

**Bubble 分析**:
```
Standard 1F1B: bubble = (pp-1) / m
DualPipeV:     bubble ≈ 0 (理论值, 实际受限于通信带宽)
代价:          2× P2P 通信量 (双向各一次), 更高的实现复杂度
```

**参数**:
- `use_dualpipev`: 启用 DualPipeV
- `dualpipev_pipeline_model_parallel_world_size`: DualPipeV PP size

**约束**:
- 与标准 PP schedule 互斥（DualPipeV 完全替代 standard/interleaved）
- 模型层数必须被 `2 × pp_size` 整除（双 chunk）
- 仅支持 GPTModel

---

## 3. P2P 通信机制

**源码**: `p2p_communication.py`

### 3.1 通信模式

| 模式 | 原理 | 适用场景 |
|------|------|---------|
| Blocking send/recv | 同步: 发送后等待对端接收完成 | 默认, 最安全 |
| `batch_p2p_comm` | 多个 send/recv 合并为一次 batch 调用 | 减少 launch 次数 |
| `overlap_p2p_comm` | send/recv 与 forward/backward 计算重叠 | 追求 overlap |
| `overlap_p2p_comm_warmup_flush` | warmup/cooldown 阶段也做 overlap | 减少 warmup 时间 |

### 3.2 通信数据量

```
activation_shape = [seq_length, micro_batch_size, hidden_size]
bytes_per_p2p = seq_length × mbs × hidden_size × sizeof(dtype)
```

**示例** (Qwen3-10B: S=4096, mbs=1, H=4096, bf16):
```
单次 P2P: 4096 × 1 × 4096 × 2 bytes = 32 MB
每步总 P2P: 2 × pp × m × 32MB (forward send + backward send)
```

### 3.3 overlap_p2p_comm 实现原理

```python
# schedules.py:1604 (简化)
# 1. 异步发送上一步的 output
send_forward_async(prev_output)
# 2. 执行当前步的 forward (计算与 send 并行)
output = forward_step(input_tensor)
# 3. 等待接收下一步的 input
input_tensor = recv_forward_wait()
```

**前置条件**: `CUDA_DEVICE_MAX_CONNECTIONS=1` — 确保 send/recv 与 compute 使用不同的 CUDA stream, 且同一 stream 内串行。

---

## 4. 关键配置参数

| 参数 | 默认值 | 说明 | 影响 |
|------|--------|------|------|
| `pipeline_model_parallel_size` | 1 | PP degree | bubble ∝ pp-1 |
| `virtual_pipeline_model_parallel_size` | None | Virtual PP chunks | bubble ÷ vpp |
| `num_microbatches` | global_batch / (dp × mbs) | microbatch 数量 | bubble ∝ 1/m |
| `overlap_p2p_comm` | False | P2P overlap with compute | 减少通信暴露时间 |
| `overlap_p2p_comm_warmup_flush` | False | Warmup P2P overlap | 依赖 overlap_p2p_comm |
| `batch_p2p_comm` | True | Batch P2P ops | 减少 kernel launch |
| `overlap_moe_expert_parallel_comm` | False | EP A2A overlap (Combined 1F1B) | MoE 加速 |
| `use_dualpipev` | False | FlagScale DualPipeV | bubble → ~0 |
| `deallocate_pipeline_outputs` | False | 释放已发送 tensors | 节省内存 |

---

## 5. 组合约束矩阵

| 组合 | 支持 | 约束/说明 |
|------|:----:|---------|
| PP + TP | ✅ | 标准组合, TP 在 PP stage 内部 |
| PP + DP | ✅ | 标准组合, DP 跨 PP group |
| PP + CP | ✅ | 使用 hybrid_cp_schedule 协同调度 |
| PP + SP | ✅ | SP 在 TP group 内, 与 PP 正交 |
| PP + EP | ✅ | MoE 层在各 PP stage 内独立 EP |
| PP + Combined 1F1B | ✅ | Interleaved PP + EP overlap |
| PP interleaved + overlap_p2p_comm | ❌ | arguments.py:849 强制关闭 |
| overlap_p2p_comm + batch_p2p_comm | ❌ | schedules.py:990 互斥检查 |
| DualPipeV + standard PP | ❌ | 互斥, DualPipeV 完全替代 |
| Combined 1F1B + CUDA Graph | ❌ | 动态 schedule plan 不可 capture |
| Combined 1F1B + forward_only | ❌ | 无 backward 则无交错意义 |
| Combined 1F1B + Dense model | ⚠️ | 可运行但无收益 (无 A2A) |

---

## 6. 性能影响总结

| Schedule | Bubble | P2P 通信量 | 额外开销 | 适用场景 |
|----------|--------|-----------|---------|---------|
| No pipelining | 0% | 0 | 无 | PP=1 |
| Standard 1F1B | (pp-1)/m | 2×pp×m | — | 通用 PP |
| Interleaved 1F1B | (pp-1)/(m×v) | 2×pp×m×v | 2× P2P 次数 | 大 bubble, 需更小 bubble |
| Combined 1F1B | 同 base schedule | 同 base schedule | 双 stream 调度 | MoE + EP (A2A 隐藏) |
| DualPipeV | ≈0% | 2× standard | 双向通信+实现复杂度 | 追求极致 GPU 利用率 |

其中 m = num_microbatches, v = virtual_pipeline_parallel_size, pp = pipeline_parallel_size

---

## 7. 设计决策对比

| 决策点 | Standard 1F1B | Combined 1F1B | DualPipeV |
|--------|--------------|---------------|-----------|
| 调度粒度 | microbatch 级 | layer 级 (子模块级) | layer 级 |
| 隐藏的通信 | 无 (仅 P2P overlap 可选) | EP A2A dispatch/combine | PP P2P + EP A2A |
| CUDA Stream 数 | 1 (+ optional comm stream) | 2 (comp + comm) | 2+ |
| 内存压力 | num_warmup × activation | 同 base + schedule plan 元数据 | 2× chunk activation |
| 实现复杂度 | 低 | 中 (需模型适配 build_schedule_plan) | 高 (需重写 forward/backward) |
| 模型兼容性 | 所有模型 | 仅 GPTModel + MoE | 仅 GPTModel |
