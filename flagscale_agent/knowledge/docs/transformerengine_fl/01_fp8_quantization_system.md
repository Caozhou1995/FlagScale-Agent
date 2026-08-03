# 第一章：FP8 量化系统源码深度分析

## 1. 概述与源文件定位

TransformerEngine-FL 的 FP8 量化系统实现了从高精度张量到 FP8 格式的完整量化/反量化流水线，支持多种 scaling 策略（DelayedScaling、CurrentScaling、MXFP8 BlockScaling 等）。

### 1.1 核心源文件映射

| 文件路径 | 行数 | 核心职责 |
|---------|------|---------|
| `pytorch/tensor/float8_tensor.py` | 1178 | Float8Quantizer / Float8CurrentScalingQuantizer / Float8Tensor 类定义 |
| `pytorch/quantization.py` | 1411 | FP8GlobalStateManager / RecipeState 层次 / reduce_and_update_fp8_tensors |
| `pytorch/tensor/quantized_tensor.py` | ~300 | Quantizer / QuantizedTensor 基类抽象 |
| `pytorch/tensor/mxfp8_tensor.py` | ~600 | MXFP8Quantizer / MXFP8Tensor (Microscaling 分块量化) |
| `pytorch/fp8.py` (deprecated path) | - | 旧版 fp8_autocast 入口 (已迁移到 quantization.py) |
| `common/recipe/__init__.py` | ~400 | Recipe dataclass 定义 (DelayedScaling, Float8CurrentScaling 等) |

### 1.2 架构层次图

```
┌─────────────────────────────────────────────────────────────┐
│                    用户层 (Training Loop)                      │
│   fp8_model_init(recipe=DelayedScaling(...))                │
│   fp8_autocast(enabled=True)                                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              FP8GlobalStateManager (quantization.py L237+)   │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ RecipeState  │  │ amax_history │  │ scale 管理       │  │
│  │ (per module) │  │ [H, N] tensor│  │ scale = max/448  │  │
│  └──────┬──────┘  └──────┬───────┘  └────────┬──────────┘  │
│         │                 │                    │              │
└─────────┼─────────────────┼────────────────────┼─────────────┘
          │                 │                    │
┌─────────▼─────────────────▼────────────────────▼─────────────┐
│                   Quantizer 层                                │
│  ┌───────────────────┐  ┌────────────────────────────────┐   │
│  │  Float8Quantizer  │  │ Float8CurrentScalingQuantizer  │   │
│  │  (delayed scale)  │  │ (just-in-time scale compute)   │   │
│  │  L42-225          │  │ L228-466                       │   │
│  └────────┬──────────┘  └────────────┬───────────────────┘   │
│           │                          │                        │
└───────────┼──────────────────────────┼────────────────────────┘
            │                          │
┌───────────▼──────────────────────────▼────────────────────────┐
│                  Float8Tensor (L470-1178)                      │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ _data: uint8 (rowwise FP8 storage)                       │ │
│  │ _transpose: uint8 (columnwise FP8 for backward GEMM)    │ │
│  │ _scale_inv: float32 (dequantize factor = 1/scale)        │ │
│  │ _fp8_dtype: TE_DType (E4M3/E5M2)                        │ │
│  └──────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
```

---

## 2. Quantizer 抽象与实现

### 2.1 Quantizer 基类协议

`Quantizer` 是所有量化器的抽象基类，定义核心接口：

```python
# quantized_tensor.py (抽象基类)
class Quantizer(ABC):
    rowwise_usage: bool    # 是否生成行方向 FP8 数据
    columnwise_usage: bool # 是否生成列方向 FP8 数据 (用于反向 GEMM)
    
    @abstractmethod
    def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor: ...
    @abstractmethod
    def make_empty(self, shape, ...) -> QuantizedTensor: ...
    @abstractmethod
    def calibrate(self, tensor: torch.Tensor) -> None: ...
    @abstractmethod
    def update_quantized(self, src, dst, *, noop_flag=None) -> QuantizedTensor: ...
```

**设计动机**：将"如何确定 scale"的策略与"如何执行量化"的机制解耦。不同 Recipe 产出不同 Quantizer 实例，但 Linear 层只需调用统一接口。

### 2.2 Float8Quantizer — Delayed Scaling 实现

**源码位置**: `float8_tensor.py` L42-225

```python
class Float8Quantizer(Quantizer):
    scale: torch.Tensor   # 由外部 (FP8GlobalStateManager) 预计算的 scaling factor
    amax: torch.Tensor    # 本次 cast 记录的 max-abs 值 (写入 amax_history[0])
    dtype: TE_DType       # E4M3 (前向) 或 E5M2 (反向)
```

**核心量化流程** (L112-114):
```python
def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor:
    return tex.quantize(tensor, self)  # C++ CUDA kernel
```

实际执行链：
1. `tex.quantize(src, quantizer)` → 调用 CUDA kernel
2. Kernel 内部: `fp8_value = cast_to_fp8(src * scale)`，同时写入 `amax = max(|src|)`
3. 返回 `Float8Tensor` 实例，包含 uint8 数据 + scale_inv

**make_empty** (L116-157) — 预分配 FP8 存储：
```python
def make_empty(self, shape, ...):
    data = torch.empty(shape, dtype=torch.uint8, ...)           # 行存储
    data_transpose = torch.empty([shape[-1]]+shape[:-1], ...)   # 列存储 (转置)
    return Float8Tensor(shape, data=data, data_transpose=data_transpose, ...)
```

**关键设计**: rowwise + columnwise 双份存储——前向 GEMM 需要 A 的行格式，反向 GEMM (dgrad) 需要 A^T 的行格式（即原始 A 的列格式）。这是用 2x 显存换取避免运行时转置的权衡。

### 2.3 Float8CurrentScalingQuantizer — 即时 Scaling

**源码位置**: `float8_tensor.py` L228-466

与 DelayedScaling 的核心区别：**不需要外部预计算 scale**，量化 kernel 内部同时计算 amax 并推导 scale。

```python
class Float8CurrentScalingQuantizer(Quantizer):
    scale: torch.Tensor     # 工作缓冲区 (kernel 写入)
    amax: torch.Tensor      # 工作缓冲区 (kernel 写入)
    dtype: TE_DType
    force_pow_2_scales: bool     # 强制 scale 为 2 的幂 (硬件友好)
    amax_epsilon: float          # 避免除零的 epsilon
    with_amax_reduction: bool    # 多 GPU 时是否 all-reduce amax
    amax_reduction_group: Optional[dist_group_type]
```

**量化内核行为**:
```
// 伪代码 (CUDA kernel 内部)
amax = max(|tensor|)
if with_amax_reduction: amax = all_reduce_max(amax, group)
scale = (FP8_MAX / amax) * (1 - epsilon)
if force_pow_2_scales: scale = 2^floor(log2(scale))
fp8_data = cast_to_fp8(tensor * scale)
```

**优势对比**:
| 属性 | DelayedScaling | CurrentScaling |
|------|---------------|----------------|
| Scale 来源 | 上一步的 amax_history 推导 | 本次数据直接计算 |
| 溢出风险 | 有 (scale 滞后于数据分布变化) | 无 (总是精确适配当前数据) |
| 额外开销 | 无 (scale 预计算) | 需额外 pass 计算 amax |
| amax 存储 | history tensor [H, N] | 单个工作缓冲区 |
| 适用场景 | 稳定训练阶段 | 训练初期/loss spike |

---

## 3. Float8Tensor — FP8 数据容器

### 3.1 类继承与存储布局

**源码位置**: `float8_tensor.py` L470-1178

```python
class Float8Tensor(Float8TensorStorage, QuantizedTensor):
    # Float8TensorStorage 提供底层存储管理
    # QuantizedTensor 提供 dequantize/quantize_ 接口
```

**核心成员变量**:
```python
_data: torch.Tensor          # uint8, shape=[...], 行优先 FP8 数据
_transpose: torch.Tensor     # uint8, shape=[last_dim, ...], 列优先 FP8 数据
_scale_inv: torch.Tensor     # float32, 反量化因子 = 1/scale
_fp8_dtype: TE_DType         # E4M3FN / E5M2 / E4M3FNUZ / E5M2FNUZ
dtype: torch.dtype           # 名义精度 (float32/bfloat16, 用于兼容 PyTorch)
```

### 3.2 dequantize 流程 (L510-523)

```python
def dequantize(self, *, dtype=None) -> torch.Tensor:
    if dtype is None:
        dtype = self.dtype  # 名义精度
    tensor = self.contiguous()
    if torch.is_grad_enabled():
        return _FromFloat8Func.apply(tensor, dtype)  # 带梯度的反量化
    return _FromFloat8Func.forward(None, tensor, dtype)  # 无梯度
```

**_FromFloat8Func** 是 autograd Function：
- **forward**: `output = tex.cast_from_fp8(data, scale_inv, fp8_dtype, out_dtype)` — CUDA kernel 执行 `bf16_value = fp8_data * scale_inv`
- **backward**: 直接透传梯度 (STE, Straight-Through Estimator)，因为量化是不可微的

### 3.3 quantize_ 原地量化 (L525-543)

```python
def quantize_(self, tensor, *, noop_flag=None) -> Float8Tensor:
    if isinstance(tensor, QuantizedTensor):
        return self.quantize_(tensor.dequantize(), noop_flag=noop_flag)
    return super().quantize_(tensor, noop_flag=noop_flag)
```

`noop_flag` 设计用于 **gradient scaling** 场景——当 loss scaler 检测到 inf/nan 时，设置 `noop_flag=1` 跳过本次参数更新，避免污染 FP8 缓存。

### 3.4 双存储 (rowwise + columnwise) 机制

Float8Tensor 同时维护两份 FP8 数据：
- `_data` (rowwise): 形状 `[M, K]`，用于前向 GEMM `Y = X @ W^T` 中的 X
- `_transpose` (columnwise): 形状 `[K, M]`，用于反向 GEMM `dX = dY @ W` 中的 W

**为何不运行时转置？**
1. FP8 转置需要额外 kernel launch，增加延迟
2. 转置后的内存布局不是连续的，GEMM 效率降低
3. 用 2x 存储换取零额外开销的双向 GEMM 支持

这个决策在 `Float8Quantizer.make_empty()` (L116-157) 中实现：
```python
if self.rowwise_usage:
    data = torch.empty(shape, dtype=torch.uint8, ...)
if self.columnwise_usage:
    transpose_shape = [shape[-1]] + list(shape[:-1])
    data_transpose = torch.empty(transpose_shape, dtype=torch.uint8, ...)
```

---

## 4. RecipeState — 量化策略状态管理

### 4.1 RecipeState 抽象层次

**源码位置**: `quantization.py` L982-1400

```
Recipe (dataclass, 用户配置)
    │
    ▼
RecipeState (运行时状态, per-module)
    │
    ▼
Quantizer (per-tensor 量化执行器)
```

每个 `RecipeState` 子类对应一种量化策略：

| RecipeState 类 | 对应 Recipe | 状态内容 |
|----------------|------------|---------|
| `DelayedScalingRecipeState` (L1054) | `DelayedScaling` | scale[N], amax_history[H,N] |
| `Float8CurrentScalingRecipeState` (L1104) | `Float8CurrentScaling` | 无有状态 buffer |
| `MXFP8BlockScalingRecipeState` (L1145) | `MXFP8BlockScaling` | 无有状态 buffer |
| `Float8BlockScalingRecipeState` (L1180) | `Float8BlockScaling` | 无有状态 buffer |
| `NVFP4BlockScalingRecipeState` (L1285) | `NVFP4BlockScaling` | 无有状态 buffer |
| `CustomRecipeState` (L1361) | 用户自定义 | 自定义 |

### 4.2 DelayedScalingRecipeState 详解 (L1054-1101)

```python
class DelayedScalingRecipeState(RecipeState):
    recipe: DelayedScaling
    mode: str              # "forward" 或 "backward"
    dtype: tex.DType       # E4M3 (forward) 或 E5M2 (backward)
    scale: torch.Tensor    # shape=[N], N=num_quantizers (通常=1)
    amax_history: torch.Tensor  # shape=[H, N], H=amax_history_len (默认1024)
```

**初始化** (L1070-1091):
```python
def __init__(self, recipe, *, mode, num_quantizers=1, device=None):
    self.scale = torch.ones(num_quantizers, dtype=torch.float32, device=device)
    self.amax_history = torch.zeros(
        recipe.amax_history_len, num_quantizers, dtype=torch.float32, device=device
    )
```

**make_quantizers** (L1094-1101):
```python
def make_quantizers(self) -> list:
    return [
        Float8Quantizer(
            self.scale[i],             # 共享 scale tensor 的第 i 个元素
            self.amax_history[0][i].reshape((1,)),  # 当前 step 的 amax 槽位
            self.dtype
        )
        for i in range(self.num_quantizers)
    ]
```

**关键机制**: Quantizer 的 `amax` 字段指向 `amax_history[0][i]`——每次量化时 CUDA kernel 将当前 batch 的 max-abs 写入此位置。在 step 结束时，FP8GlobalStateManager 会滚动 history 并更新 scale。

### 4.3 Scale 更新流程 (reduce_and_update_fp8_tensors)

**触发时机**: 每个 training step 结束时调用

```python
# 用户代码 (training loop)
with fp8_autocast(recipe=...):
    output = model(input)
loss.backward()
# === Scale 更新发生在此处 ===
# FP8GlobalStateManager.reduce_and_update_fp8_tensors() 被自动调用
```

**更新算法** (DelayedScaling 默认策略):
```
1. 滚动 amax_history: history[1:] = history[:-1], history[0] = 当前 amax
2. 计算 amax_compute: amax = max(history[:])  // 取历史最大值
3. 计算新 scale: scale = FP8_MAX / amax  // FP8_MAX = 448 (E4M3) 或 57344 (E5M2)
4. 应用 margin: scale = scale / (2^margin)  // margin 默认=0
```

---

## 5. FP8GlobalStateManager — 全局状态协调

### 5.1 核心职责

**源码位置**: `quantization.py` L237-980

FP8GlobalStateManager 是一个全局单例，管理所有 FP8 模块的量化状态：

1. **FP8 使能控制**: `is_fp8_enabled()` — 判断当前是否在 `fp8_autocast` 上下文中
2. **Recipe 分发**: 为每个注册模块创建对应的 RecipeState
3. **Scale 同步**: 在分布式训练中 all-reduce amax (跨 TP/DP groups)
4. **History 管理**: 自动滚动 amax_history，更新 scale

### 5.2 模块注册与 RecipeState 绑定

```python
# quantization.py 中的注册逻辑 (简化)
class FP8GlobalStateManager:
    _fp8_modules: List[TransformerEngineBaseModule] = []
    
    @classmethod
    def register_module(cls, module):
        # 为模块创建 forward/backward RecipeState
        module._recipe_state_forward = create_recipe_state(recipe, mode="forward")
        module._recipe_state_backward = create_recipe_state(recipe, mode="backward")
        cls._fp8_modules.append(module)
```

### 5.3 分布式 Amax 同步

在 TP 模式下，同一个 tensor 被切分到多个 GPU，各 GPU 只看到局部 amax。必须跨 TP group all-reduce 取全局 max：

```python
# reduce_and_update_fp8_tensors 内部逻辑 (简化)
for module in _fp8_modules:
    amax = module._recipe_state_forward.amax_history[0]
    # 跨 TP group 取最大值
    torch.distributed.all_reduce(amax, op=ReduceOp.MAX, group=tp_group)
```

**性能优化**: 将所有模块的 amax 合并为一个大 tensor，执行单次 all-reduce，避免多次小通信。

---

## 6. 量化精度与数值分析

### 6.1 FP8 格式对比

| 格式 | 符号 | 指数 | 尾数 | 最大值 | 最小非零 | 精度 (ULP) |
|------|------|------|------|--------|----------|-----------|
| E4M3 | 1 | 4 | 3 | 448 | 2^-9 | ~6.25% |
| E5M2 | 1 | 5 | 2 | 57344 | 2^-16 | ~12.5% |

**选择策略** (源码中硬编码):
- 前向 (weights, activations): **E4M3** — 精度更高，表示范围够用
- 反向 (gradients): **E5M2** — 梯度动态范围大，需要更宽的指数位

### 6.2 Scaling Factor 计算公式

```
scale = FP8_MAX / amax_computed
     = 448 / max(|tensor|)     [E4M3 前向]
     = 57344 / max(|tensor|)   [E5M2 反向]
```

**force_pow_2_scales** 选项 (CurrentScaling, L258):
```python
if force_pow_2_scales:
    scale = 2 ** floor(log2(scale))
```
强制 scale 为 2 的幂，使得乘法退化为移位操作，对硬件更友好且不引入额外舍入误差。

### 6.3 溢出保护机制

1. **noop_flag** (Float8Tensor.quantize_ L529): 当 GradScaler 检测到 inf/nan 时跳过量化
2. **amax_epsilon** (CurrentScaling L259): 防止 amax=0 导致除零 (`scale = MAX / (amax + eps)`)
3. **amax_history 滚动** (DelayedScaling): 使用历史最大值而非当前值，提供安全裕度

---

## 7. GEMM 集成与数据流

### 7.1 前向 GEMM 数据流

```
BF16 Input (X)                  FP8 Weight (W, 预量化)
     │                                │
     ▼                                │
Float8Quantizer.quantize_impl()       │
     │                                │
     ▼                                ▼
Float8Tensor (X_fp8)            Float8Tensor (W_fp8)
  ._data [M, K]                   ._data [N, K] (rowwise)
  ._transpose [K, M]             ._transpose [K, N] (columnwise)
     │                                │
     └──────────┬─────────────────────┘
                │
                ▼
        cuBLAS FP8 GEMM
        Y = X_fp8._data @ W_fp8._transpose
        (M×K) × (K×N) → (M×N) in BF16
                │
                ▼
          BF16 Output (Y)
```

### 7.2 反向 GEMM 数据流

```
dY (BF16, from upstream)
     │
     ├─── dX = dY @ W._data        (需要 W 的 rowwise 数据)
     │         (M×N) × (N×K) → (M×K)
     │
     └─── dW = dY^T @ X._data      (需要 X 的 rowwise 数据, 但 dY^T 实际使用 X._transpose)
              (N×M) × (M×K) → (N×K)
```

**关键洞察**: 这就是为什么 Float8Tensor 需要同时存储 rowwise 和 columnwise 数据——前向和反向的 GEMM 需要同一个 tensor 的不同内存布局。

---

## 8. 与 Megatron-LM-FL 的集成点

### 8.1 集成入口

Megatron 通过 `TransformerEngine` 的 `Linear` 类自动使用 FP8：

```python
# megatron/core/transformer/transformer_config.py
class TransformerConfig:
    fp8: str = None           # "e4m3" / "hybrid" / None
    fp8_margin: int = 0
    fp8_amax_history_len: int = 1024
    fp8_amax_compute_algo: str = "max"  # "max" / "most_recent"
```

### 8.2 fp8_autocast 上下文管理

```python
# megatron training loop 中
with te.fp8_autocast(
    enabled=config.fp8 is not None,
    fp8_recipe=DelayedScaling(
        margin=config.fp8_margin,
        amax_history_len=config.fp8_amax_history_len,
        amax_compute_algo=config.fp8_amax_compute_algo,
    )
):
    output = transformer_layer(input)
```

---

## 9. 设计决策总结

| 设计选择 | 方案 | 替代方案 | 权衡理由 |
|---------|------|---------|---------|
| 双份存储 (row+col) | 2x 显存 | 运行时转置 | 避免转置 kernel 开销，GEMM 直接使用连续内存 |
| Delayed 默认 history=1024 | 大窗口 | 小窗口/无窗口 | 覆盖训练中的 loss spike，避免激进 scale 导致溢出 |
| 前向 E4M3 / 反向 E5M2 | 非对称格式 | 统一格式 | 梯度范围大需宽指数，权重/激活需高精度 |
| Quantizer 与 RecipeState 分离 | 策略-执行解耦 | 单一类 | 同一 RecipeState 可产出多个 Quantizer (权重/激活各一) |
| CUDA kernel 内联 amax 计算 | 单 pass | 先算 amax 再量化 | 节省一次全局内存遍历 (bandwidth-bound) |
| scale_inv 而非 scale | 反量化时乘法 | 反量化时除法 | 乘法比除法快，且精度更好 |

---

## 10. 调优与问题排查

### 10.1 常见配置参数

```yaml
# FlagScale 配置示例
model:
  fp8: "hybrid"              # E4M3 forward + E5M2 backward
  fp8_margin: 0              # scale 安全裕度 (增大减少溢出风险，但降低精度利用)
  fp8_amax_history_len: 1024 # history 窗口 (增大更保守)
  fp8_amax_compute_algo: "max"  # "max" 最保守, "most_recent" 最激进
```

### 10.2 问题排查清单

| 症状 | 可能原因 | 排查方法 |
|------|---------|---------|
| Loss 突然 NaN | FP8 溢出 (scale 太小) | 检查 amax_history 是否有突变，增大 margin |
| 收敛变慢 | Scale 太保守 (信息损失) | 减小 history_len，切换到 current scaling |
| 显存增加 ~50% | 双份 FP8 存储 | 检查 columnwise_usage 是否必要 |
| 分布式训练 scale 不一致 | Amax 未跨 TP group 同步 | 验证 reduce_and_update 的 group 配置 |
