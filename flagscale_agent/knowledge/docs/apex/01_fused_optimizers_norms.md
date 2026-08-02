# NVIDIA Apex 源码深度分析 — 第1章：Fused Optimizers 与 Fused Norms

## 1. 设计动机

### 1.1 WHY Apex？

```
Apex 解决的核心问题: Kernel Launch Overhead
──────────────────────────────────────────────
标准 PyTorch Adam 优化器:
  for param in params:
    # 每个参数执行 ~8 个独立 CUDA kernel:
    exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)     # kernel 1,2
    exp_avg_sq.mul_(beta2).addcmul_(grad,grad,...)    # kernel 3,4
    step_size = lr / (1 - beta1**step)                # CPU
    denom = exp_avg_sq.sqrt().add_(eps)               # kernel 5,6
    param.addcdiv_(exp_avg, denom, value=-step_size)  # kernel 7,8

问题: 1000 个参数 → 8000 次 kernel launch
      每次 launch overhead: ~5-10μs
      总 overhead: 40-80ms (显著!)
      
Apex FusedAdam:
  一次 kernel 处理所有参数的所有操作
  multi_tensor_apply: 将多个 tensor 打包为一次 kernel
  overhead: 1 次 launch = ~10μs (降低 1000×)
```

### 1.2 Apex 在训练栈中的位置

```
┌────────────────────────────────────────────┐
│  Training Framework (Megatron / DeepSpeed) │
├────────────────────────────────────────────┤
│  Apex (fused ops)                          │
│  ├── FusedAdam / FusedSGD / FusedLAMB     │
│  ├── FusedLayerNorm / FusedRMSNorm        │
│  ├── FusedMLP (deprecated → TE)           │
│  └── multi_tensor_apply 基础设施           │
├────────────────────────────────────────────┤
│  PyTorch + CUDA                            │
└────────────────────────────────────────────┘

注: Apex 的很多功能已被 PyTorch native 或 TransformerEngine 替代
    但 FusedAdam 仍是 Megatron 默认优化器
```

## 2. multi_tensor_apply 基础设施

### 2.1 核心原理

```
multi_tensor_apply: 将多个 tensor 打包为单次 kernel
─────────────────────────────────────────────────────
标准方式 (N 个 tensor):
  for i in range(N):
    kernel_launch(tensor[i])     # N 次 launch
    
multi_tensor_apply:
  tensor_lists = [[p1,p2,...,pN],  # 参数
                  [g1,g2,...,gN],  # 梯度
                  [m1,m2,...,mN],  # momentum
                  [v1,v2,...,vN]]  # variance
  single_kernel(tensor_lists)     # 1 次 launch, 内部循环

实现: CUDA kernel 通过 chunk_size 分片处理
  每个 thread block 处理一段连续内存
  通过 tl_mem (tensor list metadata) 定位各 tensor
```

### 2.2 源码结构

```python
# apex/multi_tensor_apply/multi_tensor_apply.py
class MultiTensorApply:
    """批量 tensor 操作封装"""
    def __init__(self, chunk_size):
        self.chunk_size = chunk_size  # 每个 thread block 处理的元素数
    
    def __call__(self, op, noop_flag, tensor_lists, *args):
        """
        op: CUDA kernel function (C++ extension)
        noop_flag: 跳过标志 (用于 grad scaler overflow)
        tensor_lists: [[t1,t2,...], [t1,t2,...], ...]
        *args: 额外标量参数 (lr, beta1, beta2, etc.)
        """
        return op(self.chunk_size, noop_flag, tensor_lists, *args)
```

## 3. FusedAdam (optimizers/fused_adam.py: 355行)

### 3.1 API

```python
# apex/optimizers/fused_adam.py
class FusedAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, bias_correction=True,
                 betas=(0.9, 0.999), eps=1e-8, 
                 adam_w_mode=True,      # AdamW (decoupled weight decay)
                 weight_decay=0.,
                 amsgrad=False,
                 set_grad_none=True,    # 用 None 代替 zero_grad
                 capturable=False,      # CUDA Graph 兼容
                 master_weight=False):  # FP32 master weight
        
    # WHY adam_w_mode=True 默认?
    # AdamW 将 weight_decay 与 Adam 解耦
    # 原始 Adam 的 L2 正则在自适应学习率下不等价于 weight decay
    # AdamW 是大模型训练的标准选择
```

### 3.2 混合精度支持

```
FusedAdam 混合精度模式:
─────────────────────────
模式 1: FP16 params + FP32 optimizer states
  - 参数存储为 FP16 (省内存)
  - Adam states (m, v) 保持 FP32 (保精度)
  - 更新: FP32 计算 → cast 回 FP16
  
模式 2: BF16 params + FP32 master weights
  - master_weight=True
  - 额外维护 FP32 参数副本
  - 更新在 FP32 master 上进行 → cast 回 BF16
  - WHY master weight? BF16 精度有限, 小学习率更新可能被舍入为 0
  
模式 3: FP32 全精度
  - 传统模式, 无精度损失
```

### 3.3 CUDA Kernel 实现

```cpp
// csrc/multi_tensor_adam.cu (简化)
template <typename T, typename GRAD_T>
__global__ void adam_cuda_kernel(
    int chunk_size,
    T** tensor_lists,   // [params, grads, exp_avg, exp_avg_sq]
    float lr, float beta1, float beta2, float eps,
    float weight_decay, int step, int adam_w_mode) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 一次 kernel 处理所有参数的 Adam 更新
    // 内部循环遍历 tensor_list 中每个 tensor 的每个元素
    
    float grad = grads[tid];
    float param = params[tid];
    float m = exp_avg[tid];
    float v = exp_avg_sq[tid];
    
    m = beta1 * m + (1-beta1) * grad;
    v = beta2 * v + (1-beta2) * grad * grad;
    
    if (adam_w_mode)
        param -= weight_decay * lr * param;  // decoupled
    
    float denom = sqrtf(v / (1-pow(beta2,step))) + eps;
    param -= lr / (1-pow(beta1,step)) * m / denom;
    
    params[tid] = param;
    exp_avg[tid] = m;
    exp_avg_sq[tid] = v;
}
```

## 4. FusedLayerNorm (normalization/fused_layer_norm.py)

### 4.1 WHY Fused LayerNorm？

```
标准 LayerNorm 操作分解:
  1. mean = x.mean(dim=-1)           # reduction kernel
  2. var = ((x - mean)**2).mean(-1)  # element-wise + reduction
  3. x_norm = (x - mean) / sqrt(var + eps)  # element-wise
  4. output = gamma * x_norm + beta  # element-wise
  
  → 4+ 个 kernel, 多次读写 global memory
  
Fused LayerNorm:
  单个 kernel 完成全部操作
  - 只读一次输入 x
  - Warp-level reduction 计算 mean/var
  - 原地归一化 + scale/bias
  - 只写一次输出
  
  性能提升: ~2-3× (memory-bound op, 减少 HBM 访问次数)
```

### 4.2 RMSNorm (Llama/Qwen 标准)

```python
# FusedRMSNorm — 无 mean 计算, 更简单更快:
# RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
# 
# vs LayerNorm: 无 centering (不减 mean), 无 beta bias
# 
# WHY RMSNorm 在大模型中流行?
# 1. 计算更简单 (少一次 mean reduction)
# 2. 效果与 LayerNorm 相当 (empirical)
# 3. 训练更稳定 (参考 Llama/Qwen 架构)
```

## 5. FusedSGD / FusedLAMB

### 5.1 FusedSGD

```python
# apex/optimizers/fused_sgd.py
# 与 FusedAdam 类似原理:
# multi_tensor_apply 一次性更新所有参数
# 支持 momentum + weight_decay + nesterov
```

### 5.2 FusedLAMB (大 batch 训练)

```
LAMB (Layer-wise Adaptive Moments for Batch training):
──────────────────────────────────────────────────────
用途: BERT-like 超大 batch 训练 (batch_size > 64K)
  
与 Adam 的区别:
  Adam:  param -= lr * m / (sqrt(v) + eps)
  LAMB:  trust_ratio = ||param|| / ||adam_update||
         param -= lr * trust_ratio * adam_update

WHY trust_ratio?
  大 batch 时 gradient 方差大
  trust_ratio 约束每层更新幅度 ∝ 参数范数
  防止某层更新过大导致训练不稳定
```

## 6. 与 PyTorch Native 的对比

```
Apex vs PyTorch Native (2024+):
┌────────────────┬──────────────────┬────────────────────┐
│ 功能           │ Apex             │ PyTorch Native     │
├────────────────┼──────────────────┼────────────────────┤
│ Fused Adam     │ apex.FusedAdam   │ torch.optim.Adam   │
│                │                  │ (fused=True, 2.0+) │
├────────────────┼──────────────────┼────────────────────┤
│ LayerNorm      │ FusedLayerNorm   │ torch.nn.LayerNorm │
│                │                  │ (已内置 fused)     │
├────────────────┼──────────────────┼────────────────────┤
│ RMSNorm        │ FusedRMSNorm     │ torch.nn.RMSNorm   │
│                │                  │ (PyTorch 2.4+)     │
├────────────────┼──────────────────┼────────────────────┤
│ AMP            │ apex.amp (废弃)  │ torch.cuda.amp     │
├────────────────┼──────────────────┼────────────────────┤
│ GradScaler     │ apex (废弃)      │ torch.GradScaler   │
└────────────────┴──────────────────┴────────────────────┘

WHY Megatron 仍用 Apex FusedAdam?
  1. 历史原因 + 深度集成
  2. master_weight 支持更完善
  3. 与 distributed_optimizer 的 overlap 优化
  4. 性能经过大规模验证
```

## 7. 总结

```
Apex 核心价值:
┌──────────────────────────────────────────────────┐
│ 1. multi_tensor_apply: 消除 kernel launch 开销   │
│ 2. Fused 优化器: 单 kernel 完成 Adam/SGD/LAMB   │
│ 3. Fused Norm: 减少内存访问 (memory-bound 友好)  │
│ 4. 混合精度: master weight + FP16/BF16 参数      │
│ 5. CUDA Graph 兼容: capturable mode              │
└──────────────────────────────────────────────────┘

发展趋势: Apex 功能逐步被 PyTorch native 和 TE 吸收
  - AMP → torch.cuda.amp (已完成)
  - FusedNorm → torch.nn (已完成)  
  - FusedAdam → torch.optim.Adam(fused=True) (进行中)
  - FusedMLP → TransformerEngine (已迁移)
```

## 8. CUDA Kernel 性能分析

### 8.1 FusedAdam 性能建模

```
Roofline 分析:
──────────────────
FusedAdam 是 memory-bound 操作:
  每个参数需读/写: param(4B) + grad(4B) + m(4B) + v(4B) = 16B 读 + 12B 写
  计算量: ~10 FLOPs
  Arithmetic Intensity: 10/28 ≈ 0.36 FLOP/Byte → 远低于 GPU 计算峰值
  
  瓶颈: HBM 带宽 (H100: 3.35 TB/s)
  理论最大吞吐: 3.35TB/s ÷ 28B × 参数数 
  7B 参数: 7e9 × 28B = 196GB → 196/3350 = 58ms (理论下界)
  
  实际 FusedAdam: ~65ms (接近理论)
  标准 PyTorch Adam: ~150ms (kernel launch + 多次 HBM 访问)
  加速比: ~2.3×
```

### 8.2 FusedLayerNorm 性能建模

```
LayerNorm 性能 (seq_len=8192, hidden=4096, BF16):
  输入: 8192 × 4096 × 2B = 64MB
  
  标准 (4 kernels): 4× 读写 = 512MB HBM 流量
  Fused (1 kernel): 2× 读写 = 128MB HBM 流量 (读入+写出)
  
  H100 带宽 3.35TB/s:
    标准: 512MB / 3.35TB/s = 153μs
    Fused: 128MB / 3.35TB/s = 38μs
    加速比: ~4×
```

## 9. capturable 模式 (CUDA Graph 兼容)

```python
# FusedAdam(capturable=True):
# 
# CUDA Graph 要求: 所有操作在 capture 期间确定地址
# 标准 Adam: step counter 在 CPU, 无法 capture
# capturable=True: step counter 放 GPU tensor
#
# 使用场景:
# Megatron + CUDA Graph (训练加速):
#   graph.capture() 时优化器也被 capture
#   replay 时零 CPU overhead

optimizer = FusedAdam(params, lr=1e-4, capturable=True)
# 内部: self.state['step'] = torch.zeros(1, device='cuda')
# 每次 step: GPU tensor += 1 (非 CPU int)
```

## 10. grad_scaler 集成

```
FusedAdam + GradScaler 交互:
────────────────────────────
混合精度训练中:
  1. Forward: FP16/BF16 计算
  2. Loss scaling: loss × scale → 防止 FP16 underflow
  3. Backward: scaled gradients
  4. Optimizer: 检查 grad overflow → 跳过 or 更新

FusedAdam 的 noop_flag:
  multi_tensor_apply(op, noop_flag, ...)
  
  if GradScaler 检测到 inf/nan:
    noop_flag = 1 → kernel 不执行更新
    → 跳过本次 step (避免参数损坏)
    → GradScaler 降低 scale factor
  else:
    noop_flag = 0 → 正常更新
    
  WHY 通过 flag 而非 Python if?
  - CUDA Graph 模式下 Python if 无法工作
  - flag 在 GPU 上, 由前置检查 kernel 设置
  - 整个 train_step 在 graph 中一路执行
```

## 11. Megatron 中的使用模式

```python
# Megatron-LM/megatron/core/optimizer/__init__.py
# Megatron 创建优化器时:
from apex.optimizers import FusedAdam as Adam

optimizer = Adam(
    params,
    lr=args.lr,
    weight_decay=args.weight_decay,
    betas=(args.adam_beta1, args.adam_beta2),
    eps=args.adam_eps,
    adam_w_mode=True,     # Always AdamW
)

# 配合 DistributedOptimizer:
# 每个 rank 只 step 自己 shard 的参数
# FusedAdam 处理的 param list 是 shard 后的子集
# → 更小的 tensor_list → 更少的 kernel 工作
```

## 12. 关键源码文件索引

| 文件 | 行数 | 核心职责 |
|------|------|---------|
| optimizers/fused_adam.py | 355 | FusedAdam Python 包装 |
| optimizers/fused_sgd.py | ~200 | FusedSGD |
| optimizers/fused_lamb.py | ~300 | FusedLAMB (大batch) |
| normalization/fused_layer_norm.py | ~150 | FusedLayerNorm/RMSNorm |
| multi_tensor_apply/ | ~100 | multi_tensor 基础设施 |
| csrc/multi_tensor_adam.cu | ~300 | Adam CUDA kernel |
| csrc/layer_norm_cuda.cu | ~400 | LayerNorm CUDA kernel |
