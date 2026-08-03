# Chapter 03: TransformerEngine Fused Ops调度 深度分析

## 1. 设计动机

**WHY TE Fused Ops**: 标准PyTorch的Linear→LayerNorm→GELU链路产生多个kernel launch和
中间tensor的global memory读写。TransformerEngine将这些操作融合为单个kernel，
同时集成FP8量化逻辑。

**核心问题**:
- TE如何决定使用哪个fused kernel?
- FP8 cast何时发生(在GEMM前还是后)?
- LayerNorm+Linear的fusion边界在哪?

## 2. TE Module层次

```
用户API层:
┌─────────────────────────────────────────────────────┐
│ te.Linear(in, out, bias=True)                        │
│ te.LayerNorm(hidden)                                 │
│ te.TransformerLayer(...)                             │
└────────────────┬────────────────────────────────────┘
                 │
Python调度层:     │
┌────────────────▼────────────────────────────────────┐
│ linear.py: _Linear.forward()                         │
│   → 判断FP8/BF16模式                                │
│   → 调用 _te_ops.gemm() 或 torch.nn.functional      │
├─────────────────────────────────────────────────────┤
│ layernorm_linear.py: LayerNormLinear.forward()        │
│   → Fused: LayerNorm + Cast_FP8 + GEMM 一个path    │
└────────────────┬────────────────────────────────────┘
                 │
C++/CUDA层:      │
┌────────────────▼────────────────────────────────────┐
│ transformer_engine/common/fused_attn/               │
│ transformer_engine/common/layer_norm/               │
│ transformer_engine/common/gemm/                     │
│   → cuBLASLt GEMM with FP8 descriptors             │
│   → Custom CUDA kernels for cast+transpose          │
└─────────────────────────────────────────────────────┘
```

> **源码**: `TransformerEngine-FL/transformer_engine/pytorch/module/linear.py`

## 3. FP8训练中的Op Fusion

### 3.1 Forward Fusion链路

```
标准BF16 Forward:
Input(BF16) → Linear(BF16 GEMM) → LayerNorm → GELU → Linear(BF16 GEMM) → ...
              ↑ 5个独立kernel

FP8 Forward (TE fusion):
Input(BF16) → [Cast_FP8 + GEMM_FP8 + Scale_Output](fused) 
            → [LayerNorm + Cast_FP8](fused)
            → [GEMM_FP8 + GELU_AUX](cublasLt epilogue)
              ↑ 大幅减少kernel数量
```

### 3.2 Cast+Transpose Fusion

```
FP8 GEMM要求B矩阵为column-major (转置)
标准做法: Transpose(B) → Cast_to_FP8(B_T) → GEMM
TE做法:   CastTranspose(B) → GEMM  (一个kernel同时做cast和transpose)

// 源码: transformer_engine/common/transpose/cast_transpose.cu
// 一个kernel同时完成:
// 1. 从BF16读取
// 2. 计算amax (for next iteration scale)
// 3. 乘scale转FP8
// 4. 写出为转置layout

节省: 一次global memory读 + 一次global memory写 = 2×M×K×sizeof(BF16)
```

**WHY不能直接在GEMM里做transpose**: cublasLt虽然支持transA/transB flag，
但FP8要求输入已经是正确格式。且我们需要同时获得非转置(给下一层forward)和转置(给backward)两个版本。

### 3.3 LayerNorm + FP8 Cast Fusion

```python
# layernorm_linear.py 中的fusion:
# 当 FP8 enabled 时:
#   1. LayerNorm输出直接cast到FP8 (不需额外BF16中间结果)
#   2. 减少一次完整tensor的GMEM读写

# 数据流:
# X(BF16) → LayerNorm_Fwd(输出FP8, 同时计算mean/var用于BWD)
#         → FP8 GEMM
```

## 4. 调度决策逻辑

### 4.1 FP8 Recipe选择

```python
# transformer_engine/common/recipe.py
class DelayedScaling:
    margin: int = 0          # scale计算的margin
    fp8_format: str = "HYBRID"  # E4M3(fwd) + E5M2(bwd)
    amax_history_len: int = 1024
    amax_compute_algo: str = "max"  # max of history
    
# 选择逻辑:
# if fp8_recipe.fp8_format == "HYBRID":
#     forward: E4M3 (更高精度, 适合activation)
#     backward: E5M2 (更大范围, 适合gradient)
```

### 4.2 是否使用FP8的判断

```python
# linear.py forward中:
def forward(self, inp):
    use_fp8 = FP8GlobalStateManager.is_fp8_enabled()
    use_fp8 = use_fp8 and self.fp8_meta is not None
    
    # 额外条件:
    # - tensor shape满足alignment (16B for FP8)
    # - 非inference mode (某些path不支持)
    # - GPU capability >= 9.0 (H100+)
    
    if use_fp8:
        return self._forward_fp8(inp)
    else:
        return self._forward_bf16(inp)
```

### 4.3 GEMM Backend选择

```
TE内部GEMM调度:
┌── FP8 enabled?
│   ├── Yes → cublasLt FP8 GEMM
│   │         (CUDA_R_8F_E4M3 input, COMPUTE_32F)
│   │
│   └── No → BF16 GEMM
│       ├── cublasLt (默认, with epilogue)
│       └── cuBLAS legacy (fallback)
│
├── Epilogue需求?
│   ├── Bias → CUBLASLT_EPILOGUE_BIAS
│   ├── GELU+Bias → CUBLASLT_EPILOGUE_GELU_AUX_BIAS
│   ├── dGELU → CUBLASLT_EPILOGUE_DGELU
│   └── None → CUBLASLT_EPILOGUE_DEFAULT
│
└── 特殊路径?
    ├── GroupedGEMM (MoE) → cutlass grouped gemm
    ├── FP8 block scaling → cublasLt block-scaled API
    └── Multi-stream overlap → separate streams for comm
```

## 5. Autograd Integration

### 5.1 Custom Autograd Function

```python
# TE使用自定义autograd function实现BWD fusion:
class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, ...):
        # FP8 forward: cast + gemm + epilogue (fused)
        out = fused_gemm_with_epilogue(input_fp8, weight_fp8, bias, ...)
        
        # 保存给backward:
        ctx.save_for_backward(input_fp8, weight_fp8, ...)
        ctx.fp8_meta = fp8_meta  # scale/amax history
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # BWD也融合:
        # 1. dGELU (如果有GELU) → 融合在grad_output处理中
        # 2. dW = grad_output^T @ input  (FP8 GEMM)
        # 3. dX = grad_output @ weight    (FP8 GEMM)
        # 4. dbias = sum(grad_output, dim=0) (epilogue中完成)
        
        # 关键: BWD的两个GEMM也用FP8!
        # grad_output cast to E5M2, input/weight reuse FP8 from FWD
        ...
```

### 5.2 为什么不用torch.compile替代

**WHY手写fusion而非compiler**: 
- torch.compile不理解FP8 scaling语义
- 跨GEMM的fusion (LayerNorm→Cast→GEMM) 超出compiler范围
- FP8 scale/amax管理需要跨iteration状态

## 6. Attention Fusion (FlashAttention集成)

```
TE的fused_attn支持:
┌─────────────────────────────────────────┐
│ FusedAttention                           │
│                                          │
│ QKV Linear → Split → FlashAttn → Proj   │
│     ↑ FP8 GEMM     ↑ FP8/BF16    ↑ FP8 │
│                                          │
│ 选择backend:                             │
│ - cuDNN Flash Attention (FP8支持)        │
│ - Dao FlashAttention (BF16/FP16)         │
│ - TE自带fused kernel                     │
└─────────────────────────────────────────┘

Backend选择:
if fp8 and cudnn_available: → cuDNN fused_attn (FP8原生)
elif flash_attn_available:  → FlashAttention v2/v3
else:                       → TE fallback (unfused)
```

## 7. 性能对比

```
Transformer Layer (hidden=4096, seq=2048, heads=32, BF16):

标准PyTorch:         TE without FP8:       TE with FP8:
15 kernel launches   8 kernel launches     6 kernel launches
12 intermediate      5 intermediate        3 intermediate tensors
tensors              tensors
Latency: 1.0×       Latency: 0.82×        Latency: 0.58×

WHY FP8快近一倍:
1. GEMM吞吐翻倍 (FP8=2× FP16 Tensor Core)
2. 中间tensor减半 (8bit vs 16bit)
3. 更多fusion (scale/cast在epilogue中)
```

## 8. 总结

| 优化手段 | 机制 | 收益 |
|----------|------|------|
| GEMM+Bias+Act fusion | cublasLt epilogue | 省2-3个kernel |
| Cast+Transpose fusion | 自定义CUDA kernel | 省1次读写 |
| LayerNorm+Cast fusion | 自定义CUDA kernel | 省1次读写 |
| FP8 GEMM | Tensor Core 2× | 计算翻倍 |
| AMAX in epilogue | cublasLt内置 | 省reduce kernel |
| BWD fusion | dGELU+dbias+GEMM | 省2-3个kernel |

**核心设计原则**: 在FP8训练中，每次数据"触碰"global memory都是昂贵的。
TE的目标是最小化这些memory touch——能fusion就fusion，能在epilogue做就不单独launch。
