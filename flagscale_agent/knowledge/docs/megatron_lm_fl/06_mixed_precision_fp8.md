# 第六章：混合精度与 FP8 训练

## 1. 概述

Megatron-LM-FL + TransformerEngine-FL 提供多层次精度策略，覆盖从 BF16 基础混合精度到 FP8/FP4 的极致低精度训练。

**源码定位**：

| 组件 | 源码路径 | 职责 |
|------|----------|------|
| FP8 核心工具 | `megatron/core/fp8_utils.py` (791行) | Recipe构造、Context管理、Param量化 |
| TransformerConfig | `megatron/core/transformer/transformer_config.py` L630-710 | FP8参数定义 |
| TransformerBlock | `megatron/core/transformer/transformer_block.py` L873-883 | FP8 Context 调用点 |
| TE Recipe | `transformer_engine/common/recipe/__init__.py` | Scaling策略定义 |
| TE FP8 Tensor | `transformer_engine/pytorch/tensor/` | QuantizedTensor类层次 |
| Param Buffer | `megatron/core/distributed/param_and_grad_buffer.py` | Grad dtype转换 |
| Optimizer | `megatron/core/optimizer/optimizer.py` | FP32 master weights |

---

## 2. BF16 混合精度架构

### 2.1 数据流

```
┌─────────────── Training Step ─────────────────┐
│                                                │
│  Forward Pass:                                 │
│    BF16 params ──→ BF16 activations ──→ FP32 loss  │
│                                                │
│  Backward Pass:                                │
│    FP32 loss ──→ BF16 grads                    │
│    (loss_scale=1.0 for BF16, dynamic for FP16) │
│                                                │
│  Optimizer Step:                               │
│    BF16 grads ──→ FP32 master weights (Adam)   │
│              ──→ BF16 params (copy back)       │
└────────────────────────────────────────────────┘
```

### 2.2 BF16 vs FP16 设计选择

| 维度 | BF16 | FP16 | 设计动机 |
|------|------|------|----------|
| 指数位 | 8-bit (同FP32) | 5-bit | BF16动态范围=FP32，无overflow |
| 尾数位 | 7-bit | 10-bit | FP16精度更高但范围小 |
| Loss Scaling | 不需要 | 必须 (dynamic) | BF16不会grad overflow |
| Grad Overflow | 极少发生 | 常见需处理 | BF16训练稳定性更好 |
| H100 TFLOPS | 989 | 989 | 硬件性能相同 |
| 推荐场景 | 大模型训练 | 兼容旧硬件 | BF16是现代训练默认 |

**为什么BF16不需要Loss Scaling**: BF16的指数范围与FP32相同(±3.4×10³⁸)，梯度值几乎不会溢出。FP16指数仅5位(范围±65504)，小梯度下溢、大梯度上溢频繁发生，必须通过动态loss scaling补偿。

### 2.3 Master Weights 机制

```
源码: megatron/core/optimizer/optimizer.py

设计动机:
  BF16 尾数仅 7 位 → 当 weight >> grad_update 时:
    weight + small_update ≈ weight (精度丢失)
  
  解决: 维护 FP32 master copy
    FP32_weight += learning_rate * FP32_grad
    BF16_weight = cast(FP32_weight)  # 仅用于前向

内存开销: 额外 2 bytes/param (FP32=4B vs BF16=2B，增50%)
```

---

## 3. FP8 训练架构

### 3.1 FP8 数据格式

```
E4M3 (Forward 使用):
  ┌─sign(1)─┬──exp(4)──┬──mantissa(3)──┐
  │    s    │  e₃e₂e₁e₀  │    m₂m₁m₀    │
  └─────────┴───────────┴──────────────┘
  范围: ±448,  精度: ~1/16 (相对误差)
  适用: 激活值/权重 (需要精度)

E5M2 (Backward 使用):
  ┌─sign(1)─┬──exp(5)──┬─mantissa(2)─┐
  │    s    │ e₄e₃e₂e₁e₀ │    m₁m₀    │
  └─────────┴───────────┴─────────────┘
  范围: ±57344,  精度: ~1/4 (相对误差)
  适用: 梯度 (需要动态范围)

设计动机 (HYBRID format):
  Forward: 激活值分布集中 → E4M3 精度足够
  Backward: 梯度分布宽 (outlier多) → E5M2 范围更安全
```

### 3.2 Recipe 体系 (TE2.x)

```
源码: transformer_engine/common/recipe/__init__.py

Recipe (base)                         # L86
  ├── DelayedScaling                  # L128: per-tensor, 延迟1步
  ├── Float8CurrentScaling            # L231: per-tensor, 即时
  ├── MXFP8BlockScaling              # L272: block-of-32, 微缩块
  ├── Float8BlockScaling             # blockwise (TE≥2.3)
  └── [Custom via fp8_quantizer_factory]
```

### 3.3 Recipe 选择与构造

```
源码: megatron/core/fp8_utils.py L542-600

get_fp8_recipe(config: TransformerConfig):
  │
  ├── config.fp8 == "hybrid" → Format.HYBRID (fwd=E4M3, bwd=E5M2)
  ├── config.fp8 == "e4m3"  → Format.E4M3 (统一)
  │
  └── 按 config.fp8_recipe 枚举:
        ├── Fp8Recipe.delayed   → TEDelayedScaling(config, fp8_format,
        │                          override_linear_precision=(F,F,not fp8_wgrad))
        │                          # TE≥2.1.0
        │
        ├── Fp8Recipe.tensorwise → Float8CurrentScaling(fp8_format, fp8_dpa)
        │                          # TE≥2.2.0
        │
        ├── Fp8Recipe.blockwise  → Float8BlockScaling(fp8_format)
        │                          # TE≥2.3.0
        │
        ├── Fp8Recipe.mxfp8     → MXFP8BlockScaling(fp8_format)
        │                          # TE≥2.1.0
        │
        └── Fp8Recipe.custom    → _get_custom_recipe(config.fp8_quantizer_factory)
                                   # 用户自定义 quantizer 路径
```

### 3.4 FP8 Context 管理

```
源码: fp8_utils.py L602-650

get_fp8_context(config, layer_no=-1, is_init=False):
  │
  ├── 判断是否需要 FP8:
  │     need_fp8 = config.fp8 if not is_init else config.fp8_param
  │     if not need_fp8 or is_first_last_bf16_layer(config, layer_no):
  │         return nullcontext()  # BF16 fallback
  │
  ├── 获取 amax reduction group:
  │     fp8_group = parallel_state.get_amax_reduction_group(
  │         with_context_parallel=True,
  │         tp_only_amax_red=config.tp_only_amax_red
  │     )
  │     # 跨 TP/CP group 做 amax all-reduce
  │
  └── 返回 TE fp8_autocast context:
        if not is_init:
            return te.pytorch.fp8_autocast(
                enabled=True, fp8_recipe=recipe, fp8_group=fp8_group)
        else:
            return te.pytorch.fp8_model_init(
                enabled=True, recipe=recipe)
```

**调用位置 (transformer_block.py L873-883)**：
```python
# DelayedScaling: 外层 wrap 整个 forward pass (一次 context 多层共享 amax)
# CurrentScaling/MXFP8: 每层独立 wrap (fine-grained scaling)
quantization_context = get_fp8_context(self.config, layer_no)
with quantization_context:
    hidden_states = layer(hidden_states, ...)
```

**设计动机**: DelayedScaling 使用历史 amax 做 scaling，多层共享一个 context 即可。CurrentScaling 每次前向实时计算 amax，需要更细粒度的 context 控制。

---

## 4. DelayedScaling 详解

### 4.1 算法原理

```
核心思想: 用前一步的 amax 推算当前步的 scaling factor
  优点: 无需当前步额外 kernel launch 计算 amax
  缺点: 延迟1步 → 如果 tensor 分布突变，可能 overflow

时序:
  Step t:   计算 GEMM (使用 scale_t)，同时记录 amax_t
  Step t+1: scale_{t+1} = FP8_MAX / amax_t  (使用上一步的 amax)
```

### 4.2 参数配置

```python
# recipe/__init__.py L128
class DelayedScaling(Recipe):
    margin: int = 0                    # scale = FP8_MAX / (amax × 2^margin)
    fp8_format: Format = Format.HYBRID # fwd=E4M3, bwd=E5M2
    amax_history_len: int = 1024       # 历史 buffer 长度
    amax_compute_algo: str = "max"     # "max" (取历史最大) 或 "most_recent"
    reduce_amax: bool = True           # 跨 GPU amax all-reduce
    fp8_dpa: bool = False              # FP8 dot-product attention
    fp8_mha: bool = False              # FP8 multi-head attention
```

### 4.3 Scaling Factor 计算

```
FP8_MAX(E4M3) = 448
FP8_MAX(E5M2) = 57344

公式: new_scale = FP8_MAX / (amax_history_value × 2^margin)

amax_history_value 计算:
  if amax_compute_algo == "max":
      value = max(amax_history[0:amax_history_len])  # 历史窗口最大值
  elif amax_compute_algo == "most_recent":
      value = amax_history[0]                         # 仅最近一步

margin 的作用:
  margin > 0 → scale 更保守 (留 headroom 防 overflow)
  margin = 0 → scale 最大化精度利用
  典型: margin=0 即可 (E4M3 范围够用)
```

### 4.4 Amax Reduction 通信

```
源码: fp8_utils.py L628-631

fp8_group = parallel_state.get_amax_reduction_group(
    with_context_parallel=True,
    tp_only_amax_red=config.tp_only_amax_red
)

通信范围:
  tp_only_amax_red=False (默认): TP × CP × DP group 全局 all-reduce
  tp_only_amax_red=True:         仅 TP group 内 all-reduce (通信量少)

设计动机:
  - 不同 TP rank 持有同一权重的不同 shard → amax 不同 → 需要对齐
  - DP rank 处理不同 data → amax 可能差异大 → 全局取 max 更安全
  - tp_only_amax_red=True 是 trade-off: 牺牲少量精度换通信节省
```

---

## 5. MXFP8 (Microscaling FP8) 详解

### 5.1 原理

```
传统 per-tensor scaling:
  整个 tensor 共享一个 scale → outlier 决定 scale → 其他值精度浪费

MXFP8 block-of-32 scaling:
  每 32 个连续值共享一个 E8M0 scale (纯指数, power-of-2)
  ┌────────────────────────────┐
  │ data[0:32]  → scale_0 (E8M0) │  ← 8-bit exponent only
  │ data[32:64] → scale_1 (E8M0) │
  │ ...                            │
  └────────────────────────────┘

优势: 局部 scaling → outlier 只影响局部 32 个值 → 整体精度更高
代价: 额外存储 scale (每32值1字节) + 计算开销
```

### 5.2 配置

```python
# recipe/__init__.py L272
class MXFP8BlockScaling(Recipe):
    fp8_format: Format = Format.E4M3   # 前后向统一 E4M3 (MXFP8 精度足够)
```

### 5.3 与 DelayedScaling 对比

| 维度 | DelayedScaling | MXFP8 |
|------|---------------|-------|
| Scaling 粒度 | per-tensor (1个scale/整个tensor) | per-block (1个scale/32元素) |
| Scale 格式 | FP32 | E8M0 (8-bit exponent only) |
| FP8 格式 | HYBRID (E4M3 fwd + E5M2 bwd) | 统一 E4M3 |
| 延迟 | 1步 (用上步amax) | 无延迟 (当步计算) |
| 通信 | 需要 amax all-reduce | 不需要 |
| Outlier 鲁棒性 | 差 (全tensor被一个outlier影响) | 好 (只影响32个值) |
| 计算开销 | 低 (仅维护history) | 中 (逐块scale计算) |
| 内存开销 | 低 (1个scale/tensor) | 中 (+3.1% 额外scale存储) |
| 硬件要求 | sm_90+ (H100) | sm_90+ (H100) |
| TE版本要求 | ≥1.0 | ≥2.1 |

---

## 6. FP8 Parameter Gather

### 6.1 原理

```
源码: fp8_utils.py L490-496 (quantize_param_shard)

标准路径 (无 FP8 param gather):
  Optimizer step: FP32 master → BF16 model params
  All-gather: BF16 params (2 bytes/param) → 全量参数
  Forward: BF16 → cast to FP8 (在 TE Linear 内部)

FP8 Param Gather 路径:
  Optimizer step: FP32 master → 直接 cast to FP8 model params (1 byte/param)
  All-gather: FP8 params (1 byte/param) → 全量参数
  Forward: FP8 params 直接用于 GEMM (无需再 cast)

通信节省: all-gather 通信量减半 (2B → 1B per param)
```

### 6.2 实现流程

```
源码: fp8_utils.py L239-363 (TE≥2.2 路径)

_quantize_param_shard_impl(model_params, main_params, start_offsets, dp_group):
  │
  ├── Step 1: FP32 → FP8 量化
  │   for model_param, main_param in zip(model_params, main_params):
  │       main_param = main_param.to(model_param.dtype)  # FP32→BF16 (一致性)
  │       # TE2.x: 使用 quantizer.update_quantized()
  │       quantizer = model_param._quantizer
  │       out = Float8Tensor(shape, dtype, data=shard, quantizer=quantizer)
  │       quantizer.update_quantized(main_param, out)
  │
  ├── Step 2: 更新 Scale
  │   packed_scales = torch.empty(len(scales), dtype=float32)
  │   torch.reciprocal(packed_scales, out=packed_scales)  # scale → scale_inv
  │   copy scale_inv back to each model_param._scale_inv
  │
  └── Step 3: Amax All-Reduce (跨 DP group)
      packed_amaxes = torch.empty(len(amaxes), dtype=float32)
      torch.distributed.all_reduce(packed_amaxes, op=MAX, group=dp_group)
      # 确保所有 DP rank 使用相同 scale (避免参数不一致)
```

### 6.3 约束条件

```
1. 仅在 use_distributed_optimizer=True 时有效
   (需要参数 shard 化才有 all-gather)

2. 需要 config.fp8_param = True 触发 fp8_model_init context

3. Scale 跨 DP group all-reduce → 额外小通信
   (但远小于节省的 all-gather 通信量)

4. 数值一致性: 即使启用 FP8 param gather，
   main_param 仍先 cast to BF16 再量化为 FP8
   (保持与非 FP8 路径的数值对齐)
```

---

## 7. First/Last Layer BF16 策略

### 7.1 设计动机

```
源码: fp8_utils.py L519-535

def is_first_last_bf16_layer(config, layer_no):
    if config.first_last_layers_bf16 and (is_first or is_last):
        return True  # 该层使用 BF16，跳过 FP8

原因:
  - 第一层: embedding → 首个 transformer 的输入分布特殊
  - 最后一层: 输出 → loss 计算，精度敏感
  - FP8 对这两层的精度损失最大 (分布未稳定/直接影响loss)

配置: config.first_last_layers_bf16 = True
      config.num_bf16_layers_at_start = N  (前N层BF16)
      config.num_bf16_layers_at_end = M    (后M层BF16)
```

---

## 8. FP8 Tensor 类层次 (TE2.x)

### 8.1 类层次

```
源码: transformer_engine/pytorch/tensor/

QuantizedTensor (base)              # TE2.x 统一基类
  ├── Float8Tensor                   # delayed/current scaling
  │     ├── _data: torch.Tensor (FP8 raw bytes)
  │     ├── _scale_inv: torch.Tensor (FP32)
  │     ├── _fp8_dtype: tex.DType (E4M3/E5M2)
  │     └── _quantizer: Quantizer
  │
  ├── MXFP8Tensor                    # microscaling block-of-32
  │     ├── _data: torch.Tensor (FP8 raw bytes)
  │     └── _scales: torch.Tensor (E8M0, 每32元素一个)
  │
  └── BlockwiseFP8Tensor             # generic blockwise

TE1.x 兼容:
  Float8Tensor (独立类)
    ├── _data
    ├── _fp8_meta: dict ("scaling_fwd"/"scaling_bwd")
    ├── _fp8_meta_index: int
    └── _scale_inv
```

### 8.2 版本兼容处理

```
源码: fp8_utils.py L46-59

if is_te_min_version("2.0"):
    from transformer_engine.pytorch.tensor import QuantizedTensor as FP8_TENSOR_CLASS
else:
    from transformer_engine.pytorch.float8_tensor import Float8Tensor as FP8_TENSOR_CLASS

# is_float8tensor() 使用 isinstance(tensor, FP8_TENSOR_CLASS)
# → TE2.x: 检查是否是 QuantizedTensor (包含所有FP8变体)
# → TE1.x: 只检查 Float8Tensor
```

---

## 9. TransformerConfig FP8 参数全景

```
源码: transformer_config.py L630-710

# ─── 基础开关 ───
fp8: str = None                      # None/"hybrid"/"e4m3" → 启用FP8
fp8_margin: int = 0                  # scaling margin (DelayedScaling)
fp8_amax_history_len: int = 1024     # amax history 长度
fp8_amax_compute_algo: str = "max"   # "max" / "most_recent"
fp8_recipe: Fp8Recipe = Fp8Recipe.delayed  # recipe 类型
fp8_wgrad: bool = True               # weight grad 是否用 FP8

# ─── 高级选项 ───
fp8_param_gather: bool = False       # all-gather 时用 FP8
fp8_param: str = None                # 参数初始化时 FP8 (fp8_model_init)
fp8_dot_product_attention: bool = False  # attention score FP8
first_last_layers_bf16: bool = False     # 首尾层强制BF16
tp_only_amax_red: bool = False           # amax 仅在 TP group reduce

# ─── Recipe 类型 (Fp8Recipe enum) ───
#   delayed    → TEDelayedScaling     (TE≥1.0)
#   tensorwise → Float8CurrentScaling (TE≥2.2)
#   mxfp8      → MXFP8BlockScaling   (TE≥2.1)
#   blockwise  → Float8BlockScaling   (TE≥2.3)
#   custom     → 用户自定义 quantizer
```

---

## 10. FP8 训练数据流时序图

```
┌─── Step t ─────────────────────────────────────────────────────────┐
│                                                                     │
│  ┌─ Forward ────────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │  Input (BF16) ──┐                                             │  │
│  │                  ▼                                             │  │
│  │  [Quantize: BF16→E4M3]  scale_fwd = FP8_MAX/amax_{t-1}       │  │
│  │         │                                                     │  │
│  │         ▼                                                     │  │
│  │  ┌─────────────────┐                                          │  │
│  │  │  FP8 GEMM (E4M3) │  ← Weight也是FP8 (或从BF16 cast)       │  │
│  │  │  H100: 1979 TFLOPS│                                        │  │
│  │  └─────────────────┘                                          │  │
│  │         │                                                     │  │
│  │         ▼                                                     │  │
│  │  [Dequantize: E4M3→BF16]                                      │  │
│  │         │                                                     │  │
│  │         ▼                                                     │  │
│  │  Output (BF16) → LayerNorm/Activation (BF16) → next layer    │  │
│  │                                                               │  │
│  │  同时记录: amax_t = max(|input|)  → amax_history[0]           │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─ Backward ───────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │  Grad (BF16) ──┐                                              │  │
│  │                 ▼                                              │  │
│  │  [Quantize: BF16→E5M2]  scale_bwd = FP8_MAX_E5M2/amax_{t-1}  │  │
│  │         │                                                     │  │
│  │         ▼                                                     │  │
│  │  ┌──────────────────┐                                         │  │
│  │  │  FP8 GEMM (E5M2)  │  ← dgrad/wgrad                        │  │
│  │  └──────────────────┘                                         │  │
│  │         │                                                     │  │
│  │         ▼                                                     │  │
│  │  [Dequantize: E5M2→BF16]                                      │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─ Scale Update ───────────────────────────────────────────────┐  │
│  │  all-reduce(amax_t, group=fp8_group, op=MAX)                  │  │
│  │  scale_{t+1} = FP8_MAX / (max(amax_history) × 2^margin)      │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 11. 性能模型与加速分析

### 11.1 理论峰值

```
H100 SXM Tensor Core TFLOPS:
  BF16:  989 TFLOPS
  FP8:   1,979 TFLOPS (2×)
  FP4:   3,958 TFLOPS (4×, 推理only)

理论加速上限 = GEMM占比 × 2 + Non-GEMM占比
```

### 11.2 实际加速估算

```
典型 LLM (如 Qwen3-10B) 计算分布:
  GEMM 占比: ~85% (QKV proj + O proj + FFN up/gate/down)
  Non-GEMM:  ~15% (LayerNorm, Softmax, Residual Add, Activation)

BF16 基准时间 = 1.0×

FP8 理论最优:
  T = 0.85/2 + 0.15 = 0.575× → 1.74× 加速

FP8 实际 (含 overhead):
  Scaling overhead: ~3% (amax计算 + scale更新 + amax all-reduce)
  Cast overhead: ~2% (BF16↔FP8 quantize/dequantize kernels)
  T = 0.85/1.6 + 0.15 + 0.05 = 0.73× → 1.37× 加速

MXFP8 实际:
  Block-level scaling 额外开销: ~5%
  但精度更高 → 可用更激进的 batch size
  T ≈ 0.78× → 1.28× 加速 (但可能 batch 更大补偿)
```

### 11.3 Scaling Overhead 量化

| Overhead 来源 | DelayedScaling | CurrentScaling | MXFP8 |
|---------------|---------------|----------------|-------|
| Amax 计算 | 极低 (记录max) | 中 (每次full-scan) | 无 (block内) |
| Scale 更新 | 低 (1次/step) | 中 (每次forward) | 无 (即时) |
| Amax all-reduce | 有 (跨GPU) | 有 | 无 |
| Cast kernel | BF16→FP8 + FP8→BF16 | 同左 | 同左+block scale |
| 额外内存 | amax_history (4KB/tensor) | amax (4B/tensor) | scales (3.1%) |

---

## 12. 与其他并行策略的交互

### 12.1 兼容性矩阵

| 并行策略 | BF16 | FP8 Delayed | FP8 Current | MXFP8 | 注意事项 |
|----------|------|-------------|-------------|-------|----------|
| TP | ✓ | ✓ | ✓ | ✓ | amax需跨TP reduce |
| PP | ✓ | ✓ | ✓ | ✓ | 无特殊约束 |
| DP | ✓ | ✓ | ✓ | ✓ | amax跨DP reduce (可选) |
| CP | ✓ | ✓ | ✓ | ✓ | amax跨CP reduce |
| SP | ✓ | ✓ | ✓ | ✓ | scatter/gather在BF16域 |
| EP | ✓ | ✓ | ✓ | ? | Expert内部独立scaling |
| Dist-Opt | ✓ | ✓ | ✓ | ✓ | fp8_param_gather需要 |

### 12.2 TP + FP8 交互

```
ColumnParallelLinear + FP8:
  Input (BF16) → [All-Gather if SP] → Quantize(E4M3) → FP8 GEMM → BF16 output
  
  amax 语义:
    每个 TP rank 持有权重的不同列分片
    → 各 rank 看到相同的 input → input amax 相同 (无需reduce)
    → 各 rank 持有不同的 weight shard → weight amax 不同 → 需要 reduce?
    
  实际处理:
    tp_only_amax_red=True: 仅 TP 内 reduce weight amax
    tp_only_amax_red=False: TP×DP×CP 全局 reduce (更保守)
```

### 12.3 Distributed Optimizer + FP8 Param Gather

```
交互流程:
  1. Optimizer step: 每个 DP rank 更新自己的 FP32 shard
  2. quantize_param_shard(): FP32 shard → FP8 (per-rank独立量化)
  3. All-Gather: FP8 params (1B/param, 通信量减半)
  4. 接收端: 直接使用 FP8 参数做 GEMM

  关键: Step 2 中 amax 需要跨 DP group all-reduce
        确保所有 rank 使用相同 scale → 参数数值一致
```

---

## 13. FlagScale 扩展点

### 13.1 平台适配

```
源码: fp8_utils.py L23-27 (FlagScale Begin)

from megatron.plugin.platform import get_platform
cur_platform = get_platform()

# FP8 相关的设备操作通过 platform 抽象:
#   cur_platform.device_name() → "cuda" / "npu" / ...
# 使 FP8 逻辑可适配非 NVIDIA 硬件 (如华为 Ascend)
```

### 13.2 Overridable Recipe

```
源码: fp8_utils.py L541

@overridable
def get_fp8_recipe(config):
    ...

# FlagScale 的 @overridable 装饰器允许下游项目替换 recipe 逻辑
# 无需修改源码即可自定义 FP8 策略
```

---

## 14. 关键设计决策总结

| 决策 | 选择 | 替代方案 | 为什么选当前方案 |
|------|------|----------|----------------|
| Forward 精度 | E4M3 | E5M2 | E4M3 精度高，激活值分布集中 |
| Backward 精度 | E5M2 | E4M3 | 梯度需要大范围，E5M2 动态范围=57344 |
| 默认 recipe | DelayedScaling | CurrentScaling | 开销最低，兼容性最好 |
| Amax reduce scope | TP×CP×DP | 仅TP | 全局reduce更安全，可选tp_only优化 |
| Scale 存储 | FP32 | FP16 | scale 精度直接影响量化误差 |
| 首尾层处理 | 可选BF16 | 全部FP8 | 经验: 首尾层FP8精度损失最大 |
| Param gather | 可选FP8 | 始终BF16 | 通信受限场景收益大 |
| History长度 | 1024 | 更短/更长 | 1024步足够捕获训练动态 |

---

## 15. 实践建议

### 15.1 推荐配置路径

```
阶段1 (稳定性优先):
  fp8="hybrid", fp8_recipe="delayed", first_last_layers_bf16=True
  → 最保守，几乎无精度损失

阶段2 (性能优化):
  + fp8_param_gather=True, tp_only_amax_red=True
  → 减少通信开销

阶段3 (极致精度):
  fp8_recipe="mxfp8" (TE≥2.1)
  → Block-level scaling，outlier 鲁棒

阶段4 (极致性能):
  fp8_recipe="tensorwise" (TE≥2.2), fp8_wgrad=True
  → 当前步 scaling + weight grad FP8
```

### 15.2 调试技巧

```
1. 精度对比: 先跑BF16 baseline 100步，再开FP8对比 loss 曲线
2. Overflow 检测: 监控 amax_history 是否持续增长
3. Scale 异常: scale 接近 0 或 inf 说明 amax 计算有问题
4. 首尾层: 如果loss不收敛，尝试 first_last_layers_bf16=True
5. Warmup: 前200步BF16，之后切FP8 (amax history 需要填充)
```
