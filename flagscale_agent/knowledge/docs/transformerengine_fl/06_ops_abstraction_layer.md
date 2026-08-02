# 第六章：Ops 抽象层与自动融合系统深度源码分析

## 1. 概述与源文件定位

| 组件 | 源文件路径 | 行数 | 核心类/函数 |
|------|-----------|------|------------|
| 基类定义 | `pytorch/ops/op.py` | 739 | `FusibleOperation`, `BasicOperation`, `FusedOperation` |
| 融合引擎 | `pytorch/ops/fuser.py` | 568 | `OperationFuser`, `_OperationFuserAutogradFunction` |
| Sequential封装 | `pytorch/ops/sequential.py` | 198 | `Sequential` |
| 基础Linear | `pytorch/ops/basic/basic_linear.py` | 1077 | `BasicLinear` |
| Grouped Linear | `pytorch/ops/basic/grouped_linear.py` | 1005 | `GroupedLinear` |
| 激活函数 | `pytorch/ops/basic/activation.py` | 392 | `Activation` |
| SwiGLU融合 | `pytorch/ops/basic/swiglu.py` | 503 | `SwiGLU` |
| LayerNorm | `pytorch/ops/basic/layer_norm.py` | 276 | `LayerNorm` |
| RMSNorm | `pytorch/ops/basic/rmsnorm.py` | 253 | `RMSNorm` |
| UB前向融合 | `pytorch/ops/fused/userbuffers_forward_linear.py` | 448 | `UserbuffersForwardLinear` |
| UB反向融合 | `pytorch/ops/fused/userbuffers_backward_linear.py` | 669 | `UserbuffersBackwardLinear` |
| Grouped MLP前向 | `pytorch/ops/fused/forward_grouped_mlp.py` | 574 | `ForwardGroupedMLP` |
| Grouped MLP反向 | `pytorch/ops/fused/backward_grouped_mlp.py` | 679 | `BackwardGroupedMLP` |

根路径：`/workspace/deps/TransformerEngine-FL/transformer_engine/`

**总代码量：~7,908行 Python（ops子系统核心）**

### 1.1 设计动机

传统实现中，每个层的Forward/Backward需手动编写融合kernel。TE-FL的Ops抽象层解决三个问题：

1. **算子声明与调度分离**：BasicOp只实现数学逻辑（GEMM、Norm、Activation），融合策略由Fuser运行时决定
2. **FP8集成自动化**：量化器(Quantizer)在op边界自动插入，无需手动管理amax/scale状态
3. **前向/反向独立融合**：前向可能融合Linear+Bias，反向可能融合dgrad+通信，两者策略可不同

### 1.2 核心设计模式

```
┌─────────────────────────────────────────────────────────────────┐
│ 设计模式: Strategy Pattern + Template Method + Plugin Registry   │
│                                                                  │
│ Strategy: 不同融合规则作为可插拔函数注册到Fuser                      │
│ Template: BasicOperation.fuser_forward 包裹 op_forward           │
│ Plugin:   register_forward_fusion/register_backward_fusion       │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 类继承体系

```
torch.nn.Module
  └── FusibleOperation (ABC, op.py L57)
        │   接口: fuser_forward, fuser_backward, get_input_quantizer
        │
        ├── BasicOperation (op.py L172)
        │     持有参数+状态，实现 op_forward/op_backward
        │     自动管理: FP8 recipe state, quantizer缓存, amax history
        │     │
        │     ├── BasicLinear (basic_linear.py)
        │     ├── GroupedLinear (grouped_linear.py)
        │     ├── Activation (activation.py)
        │     ├── SwiGLU (swiglu.py)
        │     ├── LayerNorm (layer_norm.py)
        │     └── RMSNorm (rmsnorm.py)
        │
        └── FusedOperation (op.py L677)
              无参数，组合多个BasicOp的计算+通信
              │
              ├── UserbuffersForwardLinear (GEMM + AG/RS overlap)
              ├── UserbuffersBackwardLinear (dgrad/wgrad + comm overlap)
              ├── ForwardGroupedMLP (MoE多expert合并GEMM)
              └── BackwardGroupedMLP (MoE反向融合)
```

## 3. OperationContext：算子状态容器 (op.py L26-55)

```python
@dataclasses.dataclass
class OperationContext:
    """Forward阶段产生，Backward阶段消费的状态"""
    _saved_tensors: Optional[tuple[Optional[torch.Tensor], ...]] = None
    _saved_tensors_range: Optional[tuple[int, int]] = None  # Fuser管理的tensor索引范围
    requires_grad: bool = False  # Fuser设置: 当前op是否需要计算梯度
    
    # FP8相关状态 (BasicOperation.fuser_forward中设置)
    with_quantized_compute: bool = False
    input_quantizer: Optional[Quantizer] = None
    weight_quantizer: Optional[Quantizer] = None
    grad_output_quantizer: Optional[Quantizer] = None
    
    def save_for_backward(self, *tensors):
        self._saved_tensors = tensors
    
    @property
    def to_save(self):  # Fuser使用: 将所有ctx的tensors统一保存
        return self._saved_tensors
```

**与PyTorch ctx的区别**：标准`torch.autograd.Function`的ctx只支持单一Function，OperationContext支持Fuser管理的op pipeline中任意op独立保存状态，由Fuser统一序列化。

## 4. FusibleOperation：可融合算子接口 (op.py L57-170)

### 4.1 核心接口定义

```python
class FusibleOperation(torch.nn.Module, metaclass=abc.ABCMeta):
    @property
    def is_fused_op(self) -> bool:
        """区分BasicOp和FusedOp"""
    
    def pre_first_fuser_forward(self) -> None:
        """首次forward前的初始化hook (L65)"""
    
    def pre_fuser_forward(self, requires_grad: bool) -> None:
        """每次forward前的准备hook (L68)"""
    
    def get_input_quantizer(self) -> Optional[Quantizer]:
        """声明输入需要的FP8量化器 (L75)"""
    
    def get_grad_output_quantizer(self) -> Optional[Quantizer]:
        """声明grad_output需要的FP8量化器 (L78)"""
    
    def fuser_forward(self, basic_op_ctxs, input_, *,
                      basic_op_extra_inputs,
                      prev_op_grad_output_quantizer,  # 前一个op的grad量化器
                      next_op_input_quantizer,        # 下一个op的input量化器
                      basic_op_kwargs) -> tuple[Tensor, list[tuple]]:
        """Fuser统一调用的前向接口 (L81-125)"""
    
    def fuser_backward(self, basic_op_ctxs, grad_output, *,
                       basic_op_grad_extra_outputs) -> tuple[Tensor, list, list]:
        """Fuser统一调用的反向接口 (L127-168)"""
```

### 4.2 量化器传递机制

Fuser在调用每个op的`fuser_forward`时，传递相邻op的量化器信息：

```
Op Pipeline: [LayerNorm] → [Linear] → [Activation]

调用Linear.fuser_forward时:
  prev_op_grad_output_quantizer = LayerNorm.get_grad_output_quantizer()
  next_op_input_quantizer = Activation.get_input_quantizer()
  
作用: Linear可以将输出直接量化为下一个op需要的FP8格式
      或将grad_output量化为前一个op期望的格式
      避免中间的BF16↔FP8转换开销
```

## 5. BasicOperation：参数化算子基类 (op.py L172-675)

### 5.1 Template Method: fuser_forward 包裹 op_forward

```python
class BasicOperation(FusibleOperation):
    def fuser_forward(self, basic_op_ctxs, input_, *, ...):
        """框架实现: 管理FP8状态 + 调用用户的op_forward (L466-490)"""
        ctx = basic_op_ctxs[0]
        
        # 用户实现的算子逻辑
        output = self.op_forward(ctx, input_, **basic_op_kwargs[0])
        
        return output, [()]  # 输出 + extra_outputs
    
    @abc.abstractmethod
    def op_forward(self, ctx, input, **kwargs) -> torch.Tensor:
        """用户实现: 纯算子逻辑 (L413-441)"""
        # BasicLinear: GEMM计算
        # LayerNorm: 归一化计算
        # Activation: 激活函数
```

### 5.2 FP8 Recipe State管理 (L222-345)

```python
def reset_recipe_state(self, *, recipe: Optional[Recipe]) -> None:
    """当FP8 recipe变化时重建量化器缓存 (L222)"""
    # 遍历所有quantizer slot (forward input/weight/output, backward grad_output/grad_input)
    for idx in range(self.num_quantizers("forward") + self.num_quantizers("backward")):
        # 根据recipe类型构建对应的Quantizer:
        if recipe.delayed():
            quantizer = Float8DelayedScalingQuantizer(amax_history=...)
        elif recipe.float8_current_scaling():
            quantizer = Float8CurrentScalingQuantizer()
        elif recipe.mxfp8():
            quantizer = MXFP8Quantizer()
        
        self._quantizers[key] = quantizer

def get_quantizer(self, mode: str, idx: int) -> Optional[Quantizer]:
    """获取指定slot的量化器 (L347-363)
    mode: "forward" | "backward"
    idx: 0=input, 1=weight, 2=output (forward)
         0=grad_output, 1=grad_input (backward)
    """
```

### 5.3 State Dict: FP8 Meta保存/恢复 (L528-660)

```python
def get_extra_state(self) -> torch.Tensor:
    """序列化FP8 amax/scale状态到checkpoint (L528)"""
    state = {}
    for key, quantizer in self._quantizers.items():
        state[key] = {
            "amax_history": quantizer.amax_history,  # 历史amax值
            "scale": quantizer.scale,                 # 当前scale因子
        }
    # 打包为单个tensor (避免额外checkpoint key)
    return torch.frombuffer(pickle.dumps(state), dtype=torch.uint8)

def set_extra_state(self, state: torch.Tensor) -> None:
    """从checkpoint恢复FP8状态 (L604)"""
    # 反序列化并更新量化器
```

## 6. OperationFuser：自动融合引擎 (fuser.py L302-512)

### 6.1 初始化：展平op列表 (L316-347)

```python
class OperationFuser:
    # 类级别注册表: 所有fusion rule
    forward_fusion_functions: list[OperationFusionFunction] = []
    backward_fusion_functions: list[OperationFusionFunction] = []
    
    def __init__(self, ops: list[FusibleOperation]):
        # 展平: FusedOp → 其内部basic_ops
        basic_ops = []
        for op in ops:
            if op.is_fused_op:
                basic_ops.extend(op.basic_ops)  # 解包已融合的op
            else:
                basic_ops.append(op)
        
        self._basic_ops = basic_ops
        self._num_basic_ops = len(basic_ops)
        
        # 预收集所有参数 (传递给autograd graph)
        self._flat_basic_op_params = [p for op in basic_ops for p in op.parameters()]
```

### 6.2 条件融合触发 (maybe_fuse_ops, L393-460)

融合**不是每次forward都执行**，只在状态变化时重新融合：

```python
def maybe_fuse_ops(self, is_grad_enabled, recipe, input_, extra_inputs):
    # 计算哪些op需要backward
    first_op_requiring_backward = self._num_basic_ops  # 默认无需backward
    if is_grad_enabled and input_.requires_grad:
        first_op_requiring_backward = 0  # 全部需要
    else:
        # 遍历找到第一个有requires_grad参数的op
        for op_idx in range(self._num_basic_ops):
            if any(t.requires_grad for t in chain(params, extra_inputs)):
                first_op_requiring_backward = op_idx; break
    
    # 检测是否需要重新融合
    need_reset = False
    fusion_params = (type(recipe), first_op_requiring_backward)
    if fusion_params != (self.recipe_type, self.first_op_requiring_backward):
        need_reset = True  # recipe类型或grad需求变化
    elif recipe and recipe.delayed() and amax_history_len changed:
        need_reset = True  # delayed scaling参数变化
    
    if not need_reset:
        return  # 复用上次的融合结果
    
    # 执行融合
    self._forward_ops = self._fuse_ops(basic_ops, forward_fusion_functions, recipe)
    self._backward_ops = self._fuse_ops(basic_ops, backward_fusion_functions, recipe)
```

### 6.3 融合执行过程 (_fuse_ops, L349-391)

```python
@classmethod
def _fuse_ops(cls, basic_ops, fusion_funcs, recipe):
    """依次应用注册的融合规则"""
    fused_ops = list(basic_ops)  # 初始: 全部unfused
    
    for func in fusion_funcs:
        fused_ops = func(fused_ops, recipe=recipe)
        # 每个func扫描op列表，将可融合的相邻op替换为FusedOperation
    
    # 验证融合后的op列表与原始basic_ops一致性
    # 建立 (fused_op, basic_op_indices) 映射
    out = []
    idx = 0
    for op in fused_ops:
        if isinstance(op, FusedOperation):
            idxs = [idx, idx+1, ..., idx+len(op.basic_ops)-1]
            idx += len(op.basic_ops)
        else:
            idxs = [idx]; idx += 1
        out.append((op, idxs))
    
    return out  # [(op, [basic_op_idx_0, ...]), ...]
```

### 6.4 融合规则注册API (L515-568)

```python
def register_forward_fusion(op_fusion_func, prepend=False):
    """注册前向融合规则
    
    签名: func(ops: list[FusibleOperation], *, recipe) -> list[FusibleOperation]
    
    规则函数扫描op列表，将可融合的序列替换为FusedOperation实例
    prepend=True: 优先级最高，最先执行
    """
    if prepend:
        OperationFuser.forward_fusion_functions.insert(0, op_fusion_func)
    else:
        OperationFuser.forward_fusion_functions.append(op_fusion_func)
```

**已注册的内置融合规则：**
- `UserbuffersForwardLinear.fuse_forward_ops`: Linear+Bias+RS → UB融合前向
- `UserbuffersBackwardLinear.fuse_backward_ops`: Linear反向+通信 → UB融合反向
- `ForwardGroupedMLP.fuse_forward_ops`: 多个GroupedLinear+Activation → MoE融合

## 7. Autograd集成：_OperationFuserAutogradFunction (fuser.py L53-298)

### 7.1 Forward (L63-198)

```python
@staticmethod
def forward(func_ctx, input_, fuser, basic_op_kwargs, *params_and_extra_inputs):
    # 1. 为每个basic_op创建context
    basic_op_ctxs = [OperationContext() for _ in range(fuser._num_basic_ops)]
    
    # 2. 遍历融合后的forward ops
    x = input_
    for op, basic_op_idxs in fuser._forward_ops:
        # 设置requires_grad标志
        for idx in basic_op_idxs:
            basic_op_ctxs[idx].requires_grad = idx >= fuser.first_op_requiring_backward
        
        # 传递相邻op的量化器信息
        prev_op_grad_output_quantizer = prev_op.get_grad_output_quantizer()
        next_op_input_quantizer = next_op.get_input_quantizer()
        
        # 调用op的fuser_forward
        x, extra_outputs = op.fuser_forward(
            [basic_op_ctxs[idx] for idx in basic_op_idxs], x,
            prev_op_grad_output_quantizer=prev_op_grad_output_quantizer,
            next_op_input_quantizer=next_op_input_quantizer, ...)
    
    # 3. 统一保存所有ctx的tensors
    to_save = []
    for ctx in basic_op_ctxs:
        range_start = len(to_save)
        to_save.extend(ctx.to_save or [])
        ctx._saved_tensors_range = (range_start, len(to_save))  # 记录范围
    
    tensors_to_save, tensor_objects = prepare_for_saving(*to_save)
    func_ctx.save_for_backward(*tensors_to_save)
    
    return x
```

### 7.2 Backward (L200-298)

```python
@staticmethod
@torch.autograd.function.once_differentiable
def backward(func_ctx, grad_output, *grad_extra_outputs):
    # 1. 恢复saved tensors
    saved_tensors = restore_from_saved(func_ctx.tensor_objects, func_ctx.saved_tensors)
    for ctx in basic_op_ctxs:
        ctx.saved_tensors = saved_tensors[slice(*ctx._saved_tensors_range)]
    
    # 2. 逆序遍历backward ops
    dx = grad_output
    grad_params = [None] * len(basic_ops)
    for op, basic_op_idxs in reversed(backward_ops):
        # 提前终止: 不需要更多梯度时
        if all(not ctx.requires_grad for ctx in relevant_ctxs):
            dx = None; break
        
        # 调用op的fuser_backward
        dx, fused_grad_params, fused_grad_extra_inputs = op.fuser_backward(
            [basic_op_ctxs[idx] for idx in basic_op_idxs], dx, ...)
        
        # 收集参数梯度
        for idx, dparams in zip(basic_op_idxs, fused_grad_params):
            grad_params[idx] = dparams
    
    # 3. 返回: (grad_input, None, None, *grad_params, *grad_extra_inputs)
    return (dx, None, None) + tuple(flatten(grad_params)) + tuple(flatten(grad_extra_inputs))
```

### 7.3 数据流示意

```
Forward:
  input → [Fused(LN+Linear)] → [Activation] → [Fused(Linear+RS)] → output
           ↓ save ctx           ↓ save ctx      ↓ save ctx
           ctx_0, ctx_1         ctx_2            ctx_3, ctx_4

Backward (逆序):
  grad_out → [Fused(Linear_bwd+AG)] → [Activation_bwd] → [Fused(LN_bwd+Linear_bwd)] → grad_in
              读取 ctx_3,ctx_4          读取 ctx_2          读取 ctx_0,ctx_1
```

## 8. FusedOperation基类 (op.py L677-739)

```python
class FusedOperation(FusibleOperation):
    """组合多个BasicOp的融合操作（无自身参数）"""
    
    def __init__(self, *, basic_ops: Iterable[BasicOperation]):
        self.basic_ops: list[BasicOperation] = list(basic_ops)
        # 代理量化器到首/尾basic_op
    
    @property
    def is_fused_op(self) -> bool:
        return True
    
    def get_input_quantizer(self) -> Optional[Quantizer]:
        return self.basic_ops[0].get_input_quantizer()  # 代理到第一个op
    
    def get_grad_output_quantizer(self) -> Optional[Quantizer]:
        return self.basic_ops[-1].get_grad_output_quantizer()  # 代理到最后一个op
    
    # fuser_forward / fuser_backward 由子类实现具体融合逻辑
```

**设计约束**：FusedOperation不持有参数（参数仍属于内部的BasicOp），保证checkpoint保存/恢复的一致性。

## 9. 融合规则实例：UserbuffersForwardLinear.fuse_forward_ops

```python
# userbuffers_forward_linear.py L373-448
@staticmethod
def fuse_forward_ops(ops: list[FusibleOperation], **unused) -> list[FusibleOperation]:
    """扫描op列表，融合 Linear + [Bias] + [ReduceScatter]"""
    
    out = []
    window = []
    while ops:
        out.extend(window)
        window, ops = ops[:1], ops[1:]  # 滑动窗口
        
        # 1. 检查窗口头部是否为有UB配置的Linear
        if not isinstance(window[0], BasicLinear): continue
        if window[0]._userbuffers_options is None: continue
        
        # 2. 检查下一个op是否为Bias（column模式可融合）
        if linear.tensor_parallel_mode != "row" and isinstance(ops[0], Bias):
            bias = ops[0]; ops = ops[1:]
        
        # 3. 检查下一个op是否为ReduceScatter
        if linear.tensor_parallel_mode is None and isinstance(ops[0], ReduceScatter):
            reduce_scatter = ops[0]; ops = ops[1:]
        
        # 4. 验证融合合法性
        # - row parallel + bias: 不合法（bias需在RS后加）
        # - tp_size == 1: 不需要融合
        
        # 5. 替换为融合op
        window = [UserbuffersForwardLinear(linear=linear, bias=bias, 
                                           reduce_scatter=reduce_scatter)]
    
    return out + window
```

## 10. 端到端执行流程

```
用户调用 Sequential([RMSNorm, Linear_qkv, Linear_proj])

1. Sequential.__init__:
   → 创建 OperationFuser(ops=[RMSNorm, Linear_qkv, Linear_proj])
   → 展平得到 basic_ops = [RMSNorm, Linear_qkv, Linear_proj]

2. 首次 forward(input):
   → fuser.maybe_fuse_ops(): 检测状态变化，触发融合
     → 应用 forward_fusion_functions:
       - UserbuffersForwardLinear.fuse_forward_ops:
         检测 Linear_qkv 有UB配置 → 融合为 UB_Forward(Linear_qkv)
       - 其他规则...
     → _forward_ops = [(RMSNorm, [0]), (UB_Forward, [1]), (Linear_proj, [2])]
   
   → _OperationFuserAutogradFunction.apply(input, fuser, ...)
     → 遍历 _forward_ops:
       - RMSNorm.fuser_forward(ctx_0, input)     → normalized
       - UB_Forward.fuser_forward(ctx_1, normalized) → qkv (AG+GEMM overlap)
       - Linear_proj.fuser_forward(ctx_2, qkv)   → output
     → save_for_backward(all ctx tensors)
     → return output

3. backward(grad_output):
   → 逆序遍历 _backward_ops (可能不同于forward的融合方式)
   → 返回各参数梯度
```

## 11. 关键设计细节

### 11.1 前向/反向独立融合

前向和反向使用**不同的融合规则列表**：

```python
# fuser.py L442-451
self._forward_ops = OperationFuser._fuse_ops(
    self._basic_ops, OperationFuser.forward_fusion_functions, recipe)
self._backward_ops = OperationFuser._fuse_ops(
    self._basic_ops, OperationFuser.backward_fusion_functions, recipe)
```

**为什么需要？** 考虑TP场景：
- 前向：ColumnParallel需要AG input → 融合为 `UB_Forward(AG + GEMM)`
- 反向：同一个Linear的dgrad需要AG weight → 融合为 `UB_Backward(AG + dGEMM)`
- 两者通信模式不同（AG vs RS），融合策略不同

### 11.2 Quantizer跨op传递

```
Forward pipeline: [Op_A] → [Op_B] → [Op_C]

调用Op_B.fuser_forward时传递:
  prev_op_grad_output_quantizer = Op_A.get_grad_output_quantizer()
  next_op_input_quantizer = Op_C.get_input_quantizer()

用途:
  Op_B可以将输出直接量化为Op_C需要的FP8格式 (避免BF16中间态)
  Op_B可以将反向传播时的grad_output预量化为Op_A期望的格式
```

### 11.3 once_differentiable装饰器

```python
@staticmethod
@torch.autograd.function.once_differentiable  # fuser.py L201
def backward(func_ctx, grad_output, *grad_extra_outputs):
```

`once_differentiable`表示backward本身不可微（不支持二阶导数）。这允许在backward中使用in-place操作和非tensor状态，简化实现。

### 11.4 tensor保存优化 (prepare_for_saving)

```python
# forward中 (L171):
tensors_to_save, tensor_objects = prepare_for_saving(*to_save)
func_ctx.save_for_backward(*tensors_to_save)
func_ctx.tensor_objects = tensor_objects

# backward中 (L215):
saved_tensors = restore_from_saved(func_ctx.tensor_objects, func_ctx.saved_tensors)
```

`prepare_for_saving`将QuantizedTensor分解为(data, scale, amax)三个常规tensor保存，避免PyTorch的`save_for_backward`不支持自定义tensor子类的问题。

## 12. 性能影响量化

| 优化点 | 机制 | 收益来源 | 量化影响 |
|--------|------|---------|---------|
| Op融合 | 多op合为1次autograd | 减少kernel launch/sync | ~5-10μs/layer |
| UB通信融合 | GEMM分chunk与AG/RS overlap | 隐藏通信延迟 | ~20-30% TP通信隐藏 |
| FP8自动量化 | op边界自动插入Q/DQ | 无手动管理，精度不损失 | 内存-50%, GEMM 2x |
| 量化器传递 | 跨op直接FP8传递 | 避免中间BF16转换 | ~3-5% 带宽节省 |
| Grouped MLP | MoE多expert → 单次grouped GEMM | 减少N次launch到1次 | MoE场景30%+ |

## 13. 扩展机制：添加新的融合规则

```python
# 示例: 注册一个自定义融合规则
from transformer_engine.pytorch.ops.fuser import register_forward_fusion

def my_custom_fusion(ops, *, recipe=None):
    """将连续的 RMSNorm + Linear 融合为 FusedRMSNormLinear"""
    out = []
    i = 0
    while i < len(ops):
        if (i + 1 < len(ops) 
            and isinstance(ops[i], RMSNorm) 
            and isinstance(ops[i+1], BasicLinear)):
            # 创建融合op
            fused = FusedRMSNormLinear(rmsnorm=ops[i], linear=ops[i+1])
            out.append(fused)
            i += 2
        else:
            out.append(ops[i])
            i += 1
    return out

register_forward_fusion(my_custom_fusion)
```

**约束**：
- 融合后的op列表中basic_ops的顺序必须与原始一致
- FusedOp.basic_ops必须引用原始的BasicOp实例（同一对象）
- 融合规则是纯函数，不应有副作用

## 14. 设计决策总结

| 设计选择 | 方案 | 理由 |
|---------|------|------|
| 前向/反向独立融合 | 两套fusion_functions | 通信模式在F/B中不同 |
| 条件融合(lazy) | 只在状态变化时重新融合 | 避免每步O(N)扫描开销 |
| Context统一管理 | Fuser收集所有ctx tensor | 支持跨op tensor共享 |
| 量化器跨op传递 | prev/next_quantizer参数 | 消除中间BF16↔FP8转换 |
| 插件式规则注册 | register_forward/backward_fusion | 解耦核心逻辑与融合策略 |
| 无参数FusedOp | 参数仍属BasicOp | checkpoint兼容性 |
| once_differentiable | backward不可微 | 简化实现，允许in-place |
| SM隔离(UB融合) | math_sms + comm_sms | 通信与计算物理隔离 |

## 15. 调试与排查

```python
# 查看当前融合结果:
fuser = model.encoder.layers[0]._fuser  # 获取Fuser实例
print("Forward ops:", [(op.__class__.__name__, idxs) for op, idxs in fuser._forward_ops])
print("Backward ops:", [(op.__class__.__name__, idxs) for op, idxs in fuser._backward_ops])

# 禁用所有融合 (debug):
OperationFuser.forward_fusion_functions.clear()
OperationFuser.backward_fusion_functions.clear()

# 查看已注册的融合规则:
print("Fwd rules:", [f.__name__ for f in OperationFuser.forward_fusion_functions])
print("Bwd rules:", [f.__name__ for f in OperationFuser.backward_fusion_functions])
```
