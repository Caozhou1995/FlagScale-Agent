# Chapter 07: 分布式通信与张量并行集成 — 源码深度分析

## 源码位置

| 文件 | 行数 | 核心职责 |
|------|------|---------|
| `transformer_engine/pytorch/distributed.py` | 2125 | 通信原语、激活检查点、RNG管理、FP8通信 |
| `transformer_engine/pytorch/ops/comm/all_gather.py` | ~200 | AllGather FusibleOperation封装 |
| `transformer_engine/pytorch/ops/comm/reduce_scatter.py` | ~200 | ReduceScatter FusibleOperation封装 |
| `transformer_engine/pytorch/module/linear.py` | 1670 | TP线性层高层封装 |

## 1. 架构总览

TE-FL的分布式通信层**不是简单的torch.distributed包装**，而是一个支持多种量化格式的智能通信系统：

```
应用层 (Linear/Attention)
    ↓
通信原语层 (gather_along_first_dim / reduce_scatter_along_first_dim)
    ↓ 根据tensor类型自动路由
    ├── BF16/FP32:  torch.distributed.all_gather_into_tensor
    ├── FP8 (per-tensor):  _all_gather_fp8 (只通信uint8 data)
    ├── FP8 (blockwise):   _start_all_gather_fp8_blockwise (data+scales)
    ├── MXFP8:             _all_gather_mxfp8 (data+shared exponents)
    ├── NVFP4:             _all_gather_nvfp4 (packed 4-bit data)
    └── symmetric_all_reduce: NVLink P2P直接内存访问
```

**核心设计思想**：通信时保持低精度格式，避免dequantize→communicate→requantize的精度和带宽浪费。

## 2. CudaRNGStatesTracker：TP-RNG一致性管理 (L812-923)

### 2.1 问题背景

TP中，每个rank执行相同的dropout但需要**不同的mask**（否则等价于没有dropout）。同时，激活检查点的重计算需要**完全复现**首次前向的dropout结果。

### 2.2 类结构

```python
class CudaRNGStatesTracker:  # L812
    def __init__(self):
        self.states_ = {}   # name → cuda rng state (torch.Tensor)
        self.seeds_ = set() # 防止重复seed
    
    def add(self, name: str, seed: int):  # L860
        """注册一个新的RNG状态（通常在模型初始化时调用）"""
        # 保存当前RNG → 设置新seed → 保存新state → 恢复原RNG
        orig_rng_state = _get_cuda_rng_state()
        torch.cuda.manual_seed(seed)
        self.states_[name] = _get_cuda_rng_state(clone=True)
        _set_cuda_rng_state(orig_rng_state)
    
    @contextmanager
    def fork(self, name: str = "model-parallel-rng"):  # L897
        """临时切换到指定RNG状态执行代码块"""
        orig_cuda_rng_state = _get_cuda_rng_state()
        _set_cuda_rng_state(self.states_[name])
        try:
            yield
        finally:
            # graph_safe模式下不需要回存（Generator对象自动追踪）
            if not graph_safe_rng_available():
                self.states_[name] = _get_cuda_rng_state()
            _set_cuda_rng_state(orig_cuda_rng_state)
```

### 2.3 使用模式

```python
# 模型初始化时 (每个TP rank不同seed)
rng_tracker.add("model-parallel-rng", seed=base_seed + tp_rank)

# Dropout执行时
with rng_tracker.fork("model-parallel-rng"):
    # 此处的dropout使用TP-specific RNG
    output = F.dropout(hidden, p=0.1)
# 退出后恢复全局RNG（保证非TP操作的一致性）
```

### 2.4 Graph-safe RNG

PyTorch 2.0+支持`torch.Generator`对象作为RNG状态（而非全局CUDA RNG）：
- `graph_safe_rng_available()`: 检测是否支持
- 优势：CUDA Graph兼容、无需全局state切换、更高效
- L879-884: 若支持，使用Generator.manual_seed()初始化

## 3. _CheckpointFunction：TE增强型激活检查点 (L332-477)

### 3.1 与PyTorch原生torch.utils.checkpoint的区别

| 特性 | torch原生 | TE-FL |
|------|----------|-------|
| RNG恢复 | 仅全局CUDA RNG | 全局 + model-parallel tracker |
| 激活分布 | 全量保存 | 可scatter到TP组 (÷tp_size) |
| FP8支持 | 无 | 保存FP8 recipe/enablement状态 |
| AMP上下文 | 部分恢复 | 完整保存torch.amp.autocast上下文 |
| CUDA Graph | 不兼容 | graph_safe RNG支持 |

### 3.2 Forward核心逻辑 (L341-405)

```python
@staticmethod
def forward(ctx, run_function, distribute_saved_activations, 
            get_rng_state_tracker, tp_group, context_fn, kwargs, *args):
    
    # [阶段1] 保存完整RNG状态（用于backward重计算）
    ctx.fwd_cpu_rng_state = torch.get_rng_state()          # CPU RNG
    ctx.fwd_cuda_rng_state = _get_cuda_rng_state(...)      # CUDA全局RNG
    ctx.fwd_cuda_rng_state_tracker = tracker.get_states()  # TP各通道RNG
    
    # [阶段2] 保存AMP/FP8上下文
    torch_gpu_amp_ctx, torch_cpu_amp_ctx = _get_active_autocast_contexts()
    ctx.fp8 = FP8GlobalStateManager.is_fp8_enabled()
    ctx.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe()
    
    # [阶段3] 执行前向（torch.no_grad下，不构建autograd图）
    with torch.no_grad(), forward_ctx:
        with activation_recompute_forward(activation_recompute=True, recompute_phase=False):
            outputs = run_function(*args, **kwargs)
    
    # [阶段4] 分布式激活保存（关键优化）
    if distribute_saved_activations:
        # 将输入tensor scatter到TP组：每rank只保存 1/tp_size
        ctx.input_0_shape = args[0].data.shape  # 记录原始shape
        safely_set_viewless_tensor_data(
            args[0],
            split_tensor_into_1d_equal_chunks(args[0].data, tp_group, new_buffer=True)
        )
    
    # 保存tensor输入（用于backward恢复）
    ctx.save_for_backward(*tensor_inputs)
```

### 3.3 Backward重计算逻辑 (L408-477)

```python
@staticmethod
def backward(ctx, *args):
    inputs = tuple(t if t is not None else arg 
                   for (t, arg) in zip(ctx.saved_tensors, ctx.inputs))
    
    # [阶段1] 恢复分布式激活
    if ctx.distribute_saved_activations:
        # all-gather从TP组恢复完整激活
        full_data = gather_split_1d_tensor(inputs[0].data, ctx.tp_group)
        safely_set_viewless_tensor_data(inputs[0], full_data.view(ctx.input_0_shape))
    
    # [阶段2] 保存当前backward的RNG（稍后恢复）
    bwd_cpu_rng_state = torch.get_rng_state()
    bwd_cuda_rng_state = _get_cuda_rng_state(...)
    bwd_cuda_rng_state_tracker = tracker.get_states()
    
    # [阶段3] 切换到forward时的RNG（确保dropout一致）
    torch.set_rng_state(ctx.fwd_cpu_rng_state)
    _set_cuda_rng_state(ctx.fwd_cuda_rng_state)
    tracker.set_states(ctx.fwd_cuda_rng_state_tracker)
    
    # [阶段4] 重计算前向（带grad，构建局部autograd图）
    detached_inputs = detach_variable(inputs)
    with torch.enable_grad(), ctx.recompute_ctx, \
         ctx.torch_gpu_amp_ctx, ctx.torch_cpu_amp_ctx, \
         activation_recompute_forward(activation_recompute=True, recompute_phase=True), \
         autocast(enabled=ctx.fp8, recipe=ctx.fp8_recipe):  # 恢复FP8配置
        outputs = ctx.run_function(*detached_inputs, **ctx.kwargs)
    
    # [阶段5] 恢复backward的RNG
    torch.set_rng_state(bwd_cpu_rng_state)
    _set_cuda_rng_state(bwd_cuda_rng_state)
    tracker.set_states(bwd_cuda_rng_state_tracker)
    
    # [阶段6] 反向传播（在重计算的局部图上）
    torch.autograd.backward(outputs_with_grad, args_with_grad)
    grads = tuple(inp.grad for inp in detached_inputs)
    return (None, None, None, None, None, None) + grads
```

### 3.4 内存节省分析

```
场景: hidden=8192, seq_len=4096, tp_size=4

标准检查点:  保存完整输入 = 4096 × 8192 × 2B = 64MB / layer
TP分布检查点: 保存1/4输入 = 4096 × 8192 × 2B ÷ 4 = 16MB / layer

代价: backward重计算前需要1次all-gather (通信量 = 48MB)
```

## 4. gather_along_first_dim：多格式智能AllGather (L1637-1787)

### 4.1 统一入口，多格式路由

```python
def gather_along_first_dim(
    inp: torch.Tensor,
    process_group: dist_group_type,
    async_op: bool = False,
    quantizer: Optional[Quantizer] = None,  # 指定通信前的量化方式
) -> tuple[torch.Tensor, Optional[torch.distributed.Work]]:
```

**路由决策树** (L1696-1787)：

```python
if isinstance(inp, Float8TensorStorage) or isinstance(quantizer, Float8Quantizer):
    return _all_gather_fp8(...)          # Per-tensor FP8: 只通信uint8 data
elif isinstance(inp, Float8BlockwiseQTensorStorage):
    return _start_all_gather_fp8_blockwise(...)  # Block FP8: data + block_scales
elif isinstance(inp, MXFP8TensorStorage):
    return _all_gather_mxfp8(...)        # MXFP8: data + shared_exponents
elif isinstance(inp, NVFP4TensorStorage):
    return _all_gather_nvfp4(...)        # FP4: packed 4-bit data + scales
else:
    # 标准BF16/FP32 all-gather
    torch.distributed.all_gather_into_tensor(out, inp, group=process_group)
```

### 4.2 FP8 AllGather详解 (_all_gather_fp8, L979-1070)

**关键优化**：FP8 per-tensor scaling时，scale在各rank间假定相同，因此只需通信uint8 data：

```python
def _all_gather_fp8(inp, process_group, *, quantizer=None, out_shape=None):
    # 1. 若输入非FP8，先量化
    if not isinstance(inp, Float8TensorStorage):
        quantizer.set_usage(rowwise=True, columnwise=False)  # 禁用转置
        inp = quantizer(inp)  # BF16 → FP8
    
    # 2. 创建输出tensor (FP8格式)
    out = quantizer.make_empty(out_shape, ...)
    
    # 3. 复用输入的scale（假定各rank scale相同）
    out._scale_inv = inp._scale_inv  # ← 零拷贝共享
    
    # 4. 只通信底层uint8数据
    handle = torch.distributed.all_gather_into_tensor(
        out._data,         # [world_size * local_tokens, ...] uint8
        inp._data,         # [local_tokens, ...] uint8
        group=process_group
    )
    
    # 5. 按需创建列式转置（用于后续GEMM）
    if quantizer.columnwise_usage and not is_non_tn_fp8_gemm_supported():
        handle.wait()
        out._create_transpose()  # 计算转置版本
    
    return out, handle
```

**通信量对比**：
| 格式 | 通信量/元素 | 相对BF16 |
|------|-----------|---------|
| BF16 | 2 bytes | 1.0× |
| FP8 (per-tensor) | 1 byte | 0.5× |
| FP8 (blockwise) | 1 byte + scale/128 | ~0.51× |
| MXFP8 | 1 byte + exp/32 | ~0.53× |
| NVFP4 | 0.5 byte + scale/32 | ~0.27× |

### 4.3 scale一致性假设

`out._scale_inv = inp._scale_inv`成立的条件：
- DelayedScaling: 各rank用相同的全局amax（通过all-reduce amax history同步）
- CurrentScaling: 各rank对相同输入分片计算local scale → **不严格相等**
  - 实践中误差可忽略（各rank处理不同token，scale接近）
  - 若严格要求：需额外all-reduce scale（增加延迟）

## 5. reduce_scatter_along_first_dim (L925-948)

```python
def reduce_scatter_along_first_dim(inp, tp_group, async_op=False):
    """ReduceScatter: 先reduce再scatter到第一维"""
    world_size = get_distributed_world_size(tp_group)
    if world_size == 1:
        return inp, None
    
    # 验证维度整除
    assert inp.size(0) % world_size == 0
    
    dim_size = list(inp.size())
    dim_size[0] //= world_size
    
    output = torch.empty(dim_size, dtype=inp.dtype, device=torch.cuda.current_device())
    handle = torch.distributed.reduce_scatter_tensor(
        output, inp.contiguous(), group=tp_group, async_op=async_op
    )
    return output, handle
```

**注意**：当前reduce_scatter**不支持FP8格式**直接通信——因为reduce操作需要加法，而FP8加法精度不够。实际流程：
```
FP8 input → dequantize → reduce_scatter (BF16) → requantize → FP8 output
```

## 6. symmetric_all_reduce：NVLink对称内存加速 (L1842-1943)

### 6.1 设计动机

标准NCCL AllReduce需要经过NCCL的ring/tree算法，有kernel launch和协议开销。NVLink对称内存允许GPU直接读写其他GPU的内存，实现**单kernel AllReduce**。

### 6.2 实现

```python
def symmetric_all_reduce(inp, tp_group=None, async_op=False, 
                         all_reduce_type="multimem_all_reduce"):
    """
    支持的allreduce类型：
    - "nccl": 标准torch.distributed.all_reduce
    - "multimem_all_reduce": 多内存访问对称allreduce
    - "two_shot": 两阶段对称allreduce (reduce-scatter + all-gather)
    - "one_shot": 单阶段直接allreduce (适合小tensor)
    """
    if all_reduce_type == "nccl":
        handle = torch.distributed.all_reduce(inp, group=tp_group)
        return inp, handle
    
    # 获取/创建对称内存tensor
    symm_tensor = get_symmetric_memory_tensor(numel, dtype, device, tp_group)
    
    # 拷贝到对称内存区域 → 执行allreduce → 拷贝回
    symm_tensor.copy_(inp)
    all_reduce_impl(symm_tensor)  # P2P直接内存访问
    inp.copy_(symm_tensor)
    return inp, None
```

### 6.3 使用场景

- TP≤8（同节点NVLink互联）：symmetric_all_reduce延迟更低
- 小tensor（LayerNorm参数同步）：one_shot模式避免NCCL协议开销
- 大tensor（GEMM输出）：multimem或two_shot利用NVLink带宽

## 7. SP通信数据流：完整链路分析

### 7.1 Column Linear + SP Forward

```
初始状态: input shape = [s/tp, h] (SP分片)

Step 1: AllGather (gather_along_first_dim)
  input [s/tp, h] × tp_size ranks → output [s, h]
  通信量 = s × h × (tp-1)/tp × dtype_bytes

Step 2: GEMM
  input [s, h] × weight [h, 4h/tp] → output [s, 4h/tp]
  (weight已按列切分)

结果: output [s, 4h/tp] — 全序列，局部列
```

### 7.2 Row Linear + SP Forward

```
初始状态: input shape = [s, h/tp] (TP分片)

Step 1: GEMM
  input [s, h/tp] × weight [h/tp, h] → output [s, h]
  (部分结果，需要跨rank求和)

Step 2: ReduceScatter (reduce_scatter_along_first_dim)
  input [s, h] → reduce(sum) → scatter → output [s/tp, h]
  通信量 = s × h × (tp-1)/tp × dtype_bytes
  等价于: AllReduce + Split，但通信量 = AllReduce / 2

结果: output [s/tp, h] — SP分片，全hidden
```

### 7.3 SP vs 非SP通信量对比

```
非SP ColumnParallel:
  Forward:  无通信
  Backward: AllReduce(dx) = 2 × s × h × dtype  (ring算法)

SP ColumnParallel:
  Forward:  AllGather = s × h × (tp-1)/tp × dtype
  Backward: ReduceScatter(dx) = s × h × (tp-1)/tp × dtype

总通信量相同，但SP的优势：
1. AG/RS可以pipeline (分chunk overlap with compute)
2. LayerNorm/Dropout激活只需存 s/tp (内存节省)
```

### 7.4 与Userbuffers的集成

当启用`ub_overlap=True`时：

```
标准SP Column Forward:
  [AllGather完成] → [GEMM开始] = 串行

UB overlap SP Column Forward:
  chunk 0: [AG_0 start] → [AG_0 wait] → [GEMM_0 start]
  chunk 1:                 [AG_1 start] → [AG_1 wait] → [GEMM_1 start]
  chunk 2:                                [AG_2 start] → [AG_2 wait] → [GEMM_2 start]
  chunk 3:                                              [AG_3 start] → [AG_3 wait] → [GEMM_3 start]
  
  时间: T_AG/chunks + T_GEMM (接近只有GEMM的时间)
  条件: NVLink带宽 > GEMM计算时间 / chunks
```

## 8. FusibleOperation通信算子封装

### 8.1 AllGather Op (ops/comm/all_gather.py)

```python
class AllGather(FusibleOperation):
    """可被OperationFuser融合的AllGather算子"""
    
    def __init__(self, process_group: dist_group_type):
        super().__init__()
        self._process_group = process_group
    
    def fuser_forward(self, basic_op_ctxs, input_, ...):
        """标准前向：直接调用gather_along_first_dim"""
        output, _ = gather_along_first_dim(input_, self._process_group)
        return output, ()
    
    def fuser_backward(self, basic_op_ctxs, grad_output, ...):
        """反向：对应ReduceScatter"""
        grad_input, _ = reduce_scatter_along_first_dim(grad_output, self._process_group)
        return grad_input, [], []
```

### 8.2 融合触发条件

OperationFuser检测到以下模式时自动替换为UB版本：

```
可融合模式:
  [AllGather] → [BasicLinear]     → UserbuffersForwardLinear (AG+GEMM)
  [BasicLinear] → [ReduceScatter] → UserbuffersForwardLinear (GEMM+RS)

不可融合条件:
  - tp_size == 1 (无需通信)
  - linear._userbuffers_options is None (未配置UB)
  - 中间有非线性op (Activation等)
```

## 9. TP线性层通信模式矩阵（完整版）

| 层类型 | SP | FP8 AG | Forward通信 | Backward dx通信 | Backward dw通信 |
|--------|----|---------|-----------|-----------------| --------------|
| Column | off | off | None | AllReduce | None |
| Column | on | off | AllGather(BF16) | ReduceScatter(BF16) | None |
| Column | on | on | AllGather(FP8) | ReduceScatter(BF16) | None |
| Row | off | off | AllReduce | None | None |
| Row | on | off | ReduceScatter(BF16) | AllGather(BF16) | None |
| Row | on | on | ReduceScatter(BF16) | AllGather(FP8) | None |

**注意**：
- Forward的AllGather可以FP8通信（节省50%带宽）
- ReduceScatter需要BF16（reduce需要加法精度）
- dw通信为None因为weight已经是TP分片的local tensor

## 10. activation_recompute_forward上下文管理 (L243-289)

```python
class activation_recompute_forward(AbstractContextManager, ContextDecorator):
    """追踪当前是否处于激活重计算模式"""
    
    def __init__(self, activation_recompute=False, recompute_phase=False):
        self.activation_recompute = activation_recompute
        self.recompute_phase = recompute_phase
    
    def __enter__(self):
        global _FP8_ACTIVATION_RECOMPUTE_ENABLED, _FP8_ACTIVATION_RECOMPUTE_PHASE
        self.prev_enabled = _FP8_ACTIVATION_RECOMPUTE_ENABLED
        self.prev_phase = _FP8_ACTIVATION_RECOMPUTE_PHASE
        
        _FP8_ACTIVATION_RECOMPUTE_ENABLED = self.activation_recompute
        _FP8_ACTIVATION_RECOMPUTE_PHASE = self.recompute_phase
```

**用途**：FP8层在重计算阶段可以跳过某些操作（如amax更新），节省重计算开销。

```python
# FP8 linear的forward中:
if is_fp8_activation_recompute_enabled() and in_fp8_activation_recompute_phase():
    # 重计算阶段：不更新amax history，复用首次forward的scale
    skip_amax_update = True
```

## 11. 通信量化精度影响分析

### 11.1 FP8 AllGather的精度保证

```
标准流程: BF16 local → AllGather(BF16) → BF16 global
FP8流程:  BF16 local → quantize(FP8) → AllGather(uint8) → FP8 global → [GEMM直接使用]

精度差异发生在quantize步骤：
- E4M3 (forward): max相对误差 ≈ 2^(-3) = 12.5% (最坏情况)
- 实际训练中：loss收敛不受显著影响（已有大量实验验证）

关键洞察：
  FP8 AllGather省下的带宽 → 允许更大batch/seq → 训练吞吐提升 > 精度微小损失
```

### 11.2 NVFP4 AllGather带宽优势

```
NVFP4: 4-bit data + block scales (block_size=32)
通信量 = N × 0.5B + N/32 × 2B = 0.5625N bytes
vs BF16 = 2N bytes

带宽节省 = 71.9%

代价：推理时dequantize成本、训练收敛可能需要warm-up阶段
```

## 12. 端到端通信流水线时序

### 12.1 单层Transformer Block (SP + UB overlap)

```
时间 →
┌──────────────────────────────────────────────────────────────┐
│ RMSNorm (local compute, s/tp tokens)                         │
├──────────────────────────────────────────────────────────────┤
│ QKV Linear (Column, UB AG+GEMM overlap)                      │
│   ├── AG chunk[0] ──┤ GEMM[0] ────┤                         │
│   ├── AG chunk[1] ──┤ GEMM[1] ────┤                         │
│   ├── AG chunk[2] ──┤ GEMM[2] ────┤                         │
│   └── AG chunk[3] ──┤ GEMM[3] ────┤                         │
├──────────────────────────────────────────────────────────────┤
│ Attention (local compute, no TP comm)                        │
├──────────────────────────────────────────────────────────────┤
│ Output Linear (Row, UB GEMM+RS overlap)                      │
│   ├── GEMM[0] ────┤ RS chunk[0] ──┤                         │
│   ├── GEMM[1] ────┤ RS chunk[1] ──┤                         │
│   ├── GEMM[2] ────┤ RS chunk[2] ──┤                         │
│   └── GEMM[3] ────┤ RS chunk[3] ──┤                         │
├──────────────────────────────────────────────────────────────┤
│ Residual + RMSNorm (local, s/tp tokens)                      │
├──────────────────────────────────────────────────────────────┤
│ MLP Gate+Up (Column, UB AG+GEMM overlap)                     │
├──────────────────────────────────────────────────────────────┤
│ SiLU + element-wise mul (local)                              │
├──────────────────────────────────────────────────────────────┤
│ MLP Down (Row, UB GEMM+RS overlap)                           │
├──────────────────────────────────────────────────────────────┤
│ Residual (local)                                             │
└──────────────────────────────────────────────────────────────┘

通信总量/layer: 4 × s × h × (tp-1)/tp × dtype_bytes
  = 2×AG(Column) + 2×RS(Row)
  全部被UB重叠隐藏

实际通信暴露时间 ≈ max(0, T_comm - T_compute) per chunk
```

## 13. 设计决策总结

| 设计选择 | 方案 | 对比方案 | 选择理由 |
|---------|------|---------|---------|
| RNG管理 | 显式tracker + fork() | 全局seed offset | 支持检查点重计算 |
| 激活分布保存 | scatter到TP组 | 全量保存 | 内存节省tp_size倍 |
| FP8通信 | 只传uint8,共享scale | 先DQ再通信 | 50%带宽节省 |
| symmetric allreduce | NVLink P2P内存 | NCCL | 低延迟(同节点) |
| 通信op融合 | FusibleOperation + Fuser | 手动编排 | 自动化,可扩展 |
| RS不支持FP8 | BF16通信+后量化 | FP8 reduce | reduce需加法精度 |
| graph_safe RNG | Generator对象 | 全局state切换 | CUDA Graph兼容 |

## 14. 配置与调优建议

```yaml
# Megatron配置
sequence_parallel: true               # 启用SP (TP>1时推荐)
distribute_saved_activations: true    # 检查点激活分散 (搭配recompute)

# TransformerEngine配置
ub_overlap: true                      # 启用Userbuffers通信重叠
tp_comm_overlap: true                 # 同上（Megatron接口）
tp_comm_overlap_cfg:
  num_comm_splits: 4                  # 通信分块数 (NVLink 4-8, IB 2-4)
  
# FP8通信
fp8_comm: true                        # AllGather使用FP8格式
# 注意：仅影响forward的AG通信，RS仍为BF16

# Symmetric Memory (实验性)
use_symmetric_all_reduce: false       # NVLink对称内存allreduce
symmetric_all_reduce_type: "multimem_all_reduce"  # multimem/two_shot/one_shot
```

### 14.1 调优决策树

```
TP > 1?
  └─ Yes → 启用SP (减少激活内存)
       └─ NVLink节点内?
            ├─ Yes → ub_overlap=True, num_splits=4-8
            │        fp8_comm=True (带宽优先)
            └─ No  → ub_overlap=True, num_splits=2-4
                     fp8_comm=True (IB带宽有限,更需要压缩)
  └─ No → 无TP通信,跳过本节优化
```
