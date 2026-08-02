# TE-FL 第10章：Float8Tensor 底层机制 深度源码分析

## 1. 概述与设计动机

### 1.1 核心问题

FP8 训练需要将高精度 tensor (BF16/FP32) 量化为 8-bit 浮点格式。
关键挑战：
- 量化/反量化对 PyTorch 自动微分透明
- 支持两种 scaling 策略（Delayed / Current）
- 同时维护 rowwise + columnwise 数据（GEMM 优化需要）
- 与 PyTorch 的 view/reshape/clone/detach 等操作兼容

### 1.2 WHY: 为什么需要自定义 Tensor 类？

```python
# 如果用标准 PyTorch tensor:
x_fp8 = x.to(torch.float8_e4m3fn)  # 丢失 scale 信息！
# 无法正确反量化：需要 x_fp8 * scale_inv

# Float8Tensor 解决方案:
x_fp8 = Float8Tensor(data=raw_uint8, fp8_scale_inv=1/scale, fp8_dtype=E4M3)
x_high = x_fp8.dequantize()  # 自动还原: raw_uint8 → FP8 → * scale_inv → BF16
```

Float8Tensor 是 **PyTorch Tensor 子类**，在 `__torch_dispatch__` 层拦截操作，
使上层代码无需感知底层 FP8 存储。

## 2. 源码定位

| 文件 | 路径 | 行数 | 核心内容 |
|------|------|------|----------|
| float8_tensor.py | `pytorch/tensor/float8_tensor.py` | 1178 | Float8Tensor + Quantizer |
| quantized_tensor.py | `pytorch/tensor/quantized_tensor.py` | ~400 | QuantizedTensor 基类 |

## 3. Quantizer 体系 — 量化策略抽象

### 3.1 Float8Quantizer (Delayed Scaling, L42-226)

```python
# float8_tensor.py L42-226
class Float8Quantizer(Quantizer):
    """Delayed scaling: scale 由历史 amax 推导，非实时计算"""
    
    scale: torch.Tensor   # 量化缩放因子 (由外部 FP8GlobalStateManager 更新)
    amax: torch.Tensor    # 上一步的 max-abs 值 (用于推导下一步 scale)
    dtype: TE_DType       # E4M3 (forward) 或 E5M2 (backward)
    
    def quantize_impl(self, tensor):
        """核心量化: 调用 C++ kernel"""
        return tex.quantize(tensor, self)  # L114
    
    def update_quantized(self, src, dst, noop_flag=None):
        """原地更新已有 Float8Tensor 的数据"""
        tex.quantize(src, self, dst, noop_flag)  # L105
        dst._fp8_dtype = self.dtype
    
    def calibrate(self, tensor):
        """记录 amax 供下一步 scale 计算"""
        amin, amax = tensor.aminmax()  # L160
        self.amax.copy_(torch.max(-amin, amax))
```

**Delayed Scaling 流程**:
```
Step N: 用 scale_N 量化 → 记录 amax_N
Step N+1: scale_{N+1} = FP8_MAX / amax_N   (由 amax 历史推导)
```

### 3.2 Float8CurrentScalingQuantizer (L228-468)

```python
# float8_tensor.py L228-468
class Float8CurrentScalingQuantizer(Quantizer):
    """Current scaling: 实时计算当前 tensor 的 scale"""
    
    dtype: TE_DType
    rowwise: bool = True     # 按行量化 (for forward GEMM)
    columnwise: bool = True  # 按列量化 (for backward GEMM)
    amax_epsilon: float      # 避免除零
    
    def quantize_impl(self, tensor):
        """实时计算 scale 并量化"""
        # scale = FP8_MAX / (amax + epsilon)
        # 无需外部 state manager
        return tex.quantize(tensor, self)  # L316
```

**WHY 两种策略？**

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| Delayed | 无额外 sync | scale 可能不准（数值突变时 overflow） | 稳定训练阶段 |
| Current | scale 精确 | 需实时 amax 计算（额外 kernel） | 训练初期/不稳定时 |

### 3.3 make_empty — 内存预分配 (L116-157)

```python
def make_empty(self, shape, dtype=torch.float32, device=None, ...):
    """预分配 FP8 tensor 空间（未量化数据）"""
    
    # Rowwise data: 原始形状
    data = torch.empty(shape, dtype=torch.uint8, device=device)
    
    # Columnwise data: 转置形状（用于 backward GEMM）
    if self.columnwise_usage:
        transpose_shape = [shape[-1]] + list(shape[:-1])
        data_transpose = torch.empty(transpose_shape, dtype=torch.uint8, device=device)
    
    return Float8Tensor(shape=shape, data=data, data_transpose=data_transpose, ...)
```

**WHY 同时存 rowwise + columnwise？**
GEMM: C = A × B^T
- Forward: A=activation (rowwise), B=weight^T → 需要 weight columnwise
- Backward: dX = dY × W (rowwise W), dW = X^T × dY → 需要 X columnwise

同时存储避免运行时转置的开销（O(N²) 内存访问）。

## 4. Float8Tensor 类详解 (L470-1178)

### 4.1 核心属性

```python
class Float8Tensor(Float8TensorStorage, QuantizedTensor):
    """FP8 数据 + scale 信息的封装"""
    
    _data: torch.Tensor           # uint8 原始数据 (rowwise)
    _data_transpose: torch.Tensor # uint8 转置数据 (columnwise)
    _fp8_dtype: TE_DType          # E4M3 或 E5M2
    _scale_inv: torch.Tensor      # 1/scale (反量化因子)
    _quantizer: Quantizer         # 创建此 tensor 的量化器
    dtype: torch.dtype            # 名义 dtype (BF16/FP32, 对外展示)
```

### 4.2 dequantize — 反量化 (L510-523)

```python
# L510-523
def dequantize(self, dtype=None):
    """FP8 → 高精度: data * scale_inv"""
    if dtype is None:
        dtype = self.dtype
    tensor = self.contiguous()
    if torch.is_grad_enabled():
        return _FromFloat8Func.apply(tensor, dtype)  # 走 autograd
    return _FromFloat8Func.forward(None, tensor, dtype)  # 直接计算
```

### 4.3 quantize_ — 原地量化 (L525-543)

```python
# L525-543
def quantize_(self, tensor, noop_flag=None):
    """高精度 → FP8 (原地更新)"""
    if isinstance(tensor, QuantizedTensor):
        return self.quantize_(tensor.dequantize(), noop_flag=noop_flag)
    return super().quantize_(tensor, noop_flag=noop_flag)
```

### 4.4 view / reshape 支持 (L1087-1178)

```python
# L1087: 自定义 autograd Function 保持 Float8Tensor 类型
class _ViewFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, shape):
        ctx.shape = tensor.shape
        return Float8Tensor.make_like(tensor, data=tensor._data.view(shape))
    
    @staticmethod
    def backward(ctx, grad):
        return grad.view(ctx.shape), None

class _ReshapeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, shape):
        ctx.shape = tensor.shape
        return Float8Tensor.make_like(tensor, data=tensor._data.reshape(shape))
```

**WHY 自定义 view/reshape？**
标准 `tensor.view()` 会将 Float8Tensor 降级为普通 Tensor（丢失 scale 等元数据）。
自定义实现确保 view 后仍然是 Float8Tensor。

## 5. 与 PyTorch 的交互模式

### 5.1 __torch_dispatch__ 拦截

Float8Tensor 继承 QuantizedTensor，后者通过 `__torch_dispatch__` 拦截 PyTorch 操作：

```
PyTorch 操作调用链:
  torch.matmul(fp8_a, b)
    │
    ├── __torch_dispatch__ 检查参数类型
    │     ├── 如果是 GEMM: 调用 TE FP8 GEMM kernel (不反量化)
    │     └── 其他操作: 先 dequantize，执行操作，再 quantize 回去
    │
    └── 返回结果 (可能是 Float8Tensor 或标准 Tensor)
```

### 5.2 ONNX 导出 (L203-216)

```python
def onnx_quantize(self, tensor):
    """ONNX 兼容量化（纯 FP32 输入）"""
    data = torch.ops.tex.fp8_quantize(tensor, self.scale.item())
    return self.create_tensor_from_data(data)

def onnx_dequantize(self, tensor):
    """ONNX 兼容反量化"""
    out = torch.ops.tex.fp8_dequantize(tensor._data, tensor._scale_inv)
    return out.to(tensor.dtype)
```

## 6. 数据流示意

```
训练 Forward 完整流程:
─────────────────────────────────────────────
Input (BF16) [s, b, h]
    │
    ├── Quantizer.quantize_impl(input)
    │     ├── tex.quantize(input, quantizer)    // C++ kernel
    │     │     ├── 计算 amax (Delayed: 记录; Current: 实时用)
    │     │     ├── scale = FP8_MAX / amax
    │     │     ├── data_uint8 = cast_to_fp8(input * scale)
    │     │     └── data_transpose_uint8 = transpose(data_uint8)  // columnwise
    │     └── 返回 Float8Tensor(data, data_transpose, 1/scale)
    │
    ├── FP8 GEMM (cuBLAS fp8_gemm or custom kernel)
    │     ├── A._data (rowwise) × B._data_transpose (columnwise)
    │     ├── output_scale = A._scale_inv * B._scale_inv
    │     └── 输出: BF16 或 Float8Tensor
    │
    └── 下一层...
─────────────────────────────────────────────
```

## 7. 性能量化分析

### 7.1 内存布局

| 存储 | 大小 (以 [4096, 4096] weight 为例) | 说明 |
|------|------|------|
| BF16 原始 | 32 MB | 不存储 (节省) |
| data (rowwise) | 16 MB | uint8 |
| data_transpose | 16 MB | uint8 (可选) |
| scale_inv | 4 B | float32 scalar |
| **总计** | **32 MB** (双存) 或 **16 MB** (单存) | vs BF16: 32 MB |

### 7.2 量化 kernel 开销

```
量化 kernel (tex.quantize):
  - 计算 amax: O(N) reduce (memory-bound)
  - 缩放+转换: O(N) elementwise (memory-bound)
  - 转置: O(N) 内存搬运

典型开销 (H100, 4096×4096 tensor):
  - Delayed scaling: ~10 μs (无需 amax reduce)
  - Current scaling: ~15 μs (含 amax reduce)
  - vs GEMM 计算: ~100 μs
  
量化占 GEMM 总时间: ~10-15%
```

## 8. Delayed vs Current Scaling 决策流

```
FP8GlobalStateManager 决策树:
─────────────────────────────────────────
recipe = DelayedScaling:
  │
  ├── 维护 amax_history[tensor_id] (长度=H 的环形缓冲)
  ├── scale = FP8_MAX / max(amax_history)  // 取历史最大值
  ├── 构建 Float8Quantizer(scale, amax_buffer)
  └── 每步更新: amax_history.append(current_amax)

recipe = CurrentScaling:
  │
  ├── 无需维护历史
  ├── 每次 quantize 时实时计算 scale
  └── 构建 Float8CurrentScalingQuantizer(dtype)
─────────────────────────────────────────
```

## 9. 设计决策对比

| 维度 | Float8Tensor (子类) | 手动 scale 管理 | 选择理由 |
|------|--------------------|--------------------|----------|
| 透明性 | 上层无需修改 | 每处 GEMM 手动传 scale | 易用 |
| 类型安全 | 编译时类型检查 | 运行时 bug | 正确性 |
| 内存管理 | 统一生命周期 | 散落的 scale tensor | 可维护 |
| Autograd | 自动处理反量化梯度 | 手动实现 | 正确性 |

| 维度 | 同时存 row+col | 按需转置 | 选择理由 |
|------|---------------|----------|----------|
| 内存 | 2× FP8 | 1× FP8 | 多 16MB/tensor |
| 计算 | 0 额外开销 | 每次 GEMM 前转置 | 性能优先 |
| 适用 | 权重 (重复使用) | 激活 (用一次) | 权重双存，激活单存 |

## 10. 与其他章节的关联

- **→ TE-FL 第1章 FP8 Quantization**: 整体 FP8 训练流程
- **→ TE-FL 第2章 Fused Linear**: 消费 Float8Tensor 的 GEMM 层
- **→ Megatron 第6章 Mixed Precision**: Megatron 侧的 FP8 配置
- **→ Megatron 第12章 Checkpoint**: FP8 state 的保存/恢复

## 11. 源码版本信息

- `float8_tensor.py`: 1178 行
- 核心类: Float8Quantizer (L42), Float8CurrentScalingQuantizer (L228), Float8Tensor (L470)
- 辅助: _ViewFunc (L1087), _ReshapeFunc (L1134)
- FlagScale 扩展: 平台无关 device 选择, NPU FP8 格式适配
