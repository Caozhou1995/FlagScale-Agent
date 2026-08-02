# PyTorch Distributed 源码深度分析 — 第2章：DistributedDataParallel (DDP)

## 1. 设计动机

### 1.1 WHY DDP 而非 DataParallel？

| 对比项 | DataParallel (DP) | DistributedDataParallel (DDP) |
|--------|-------------------|-------------------------------|
| 并行模型 | 单进程多线程 | 多进程 (每 GPU 一进程) |
| GIL 影响 | 受 GIL 限制 | 无 GIL 限制 |
| 通信方式 | 主 GPU 汇聚梯度 | Ring AllReduce |
| 扩展性 | 仅单机 | 支持多机 |
| 主 GPU 瓶颈 | 严重 (内存+通信) | 无主 GPU 概念 |
| 吞吐比 | ~60-70% 线性扩展 | ~95%+ 线性扩展 |

### 1.2 DDP 核心设计目标

```
DDP 要解决的核心问题:
┌─────────────────────────────────────────────────────────┐
│ 1. 梯度同步: 确保所有 rank 的模型参数在 step 后一致      │
│ 2. 通信隐藏: 梯度 AllReduce 与 backward 计算 overlap    │
│ 3. 内存效率: 避免梯度拷贝开销 (gradient_as_bucket_view)  │
│ 4. 灵活性: 支持 unused params、static graph、混合精度    │
└─────────────────────────────────────────────────────────┘
```

## 2. DDP __init__ 核心参数 (L816-838)

```python
# torch/nn/parallel/distributed.py L816-838
class DistributedDataParallel(Module, Joinable):
    def __init__(self,
        module,                              # 被包装的模型
        device_ids=None,                     # 放置的 GPU device
        process_group=None,                  # DP 通信组
        bucket_cap_mb=None,                  # 梯度桶大小 (默认 25MB)
        find_unused_parameters=False,        # 是否检测未使用参数
        gradient_as_bucket_view=False,       # 梯度作为桶视图 (省内存)
        static_graph=False,                  # 静态图优化
        mixed_precision=None,                # 混合精度通信
        device_mesh=None,                    # DeviceMesh (2D: DP+TP)
        batched_grad_copy=False,             # 批量梯度拷贝
    ):
```

### 2.1 关键参数设计动机

**bucket_cap_mb (默认 25MB)**:
```
WHY 需要梯度桶 (Bucket)?
─────────────────────────────────────────────────
问题: 每个参数单独 AllReduce → 通信启动开销巨大
      典型模型有 1000+ 参数, 单独通信 latency 太高
      
解决: 将参数打包为桶, 桶满时统一 AllReduce
      25MB 是通信 bandwidth 与 latency 的经验平衡点:
      - 太小: 频繁通信, latency 占主导
      - 太大: overlap 窗口小, 通信无法隐藏

桶填充顺序: 与 backward 梯度就绪顺序相反 (从最后一层开始)
      parameters 按 model.parameters() 逆序分桶
```

**gradient_as_bucket_view**:
```
WHY gradient_as_bucket_view?
─────────────────────────────────────────────────
Default (False):
  param.grad (各自分散) → copy 到 bucket → AllReduce → copy 回
  
gradient_as_bucket_view=True:
  param.grad 直接指向 bucket 内存 → AllReduce (原地)
  省去 2 次 copy, 节省等量梯度内存
  
注意: 必须配合 optimizer.zero_grad(set_to_none=True)
```

## 3. Reducer 机制 — 梯度桶与 Hook (C++ 实现)

### 3.1 架构

```
DDP Reducer 工作流:
                                        backward 计算
                                            │
                                            ▼
┌──────────────────────────────────────────────────────┐
│  Autograd Hook (每个参数注册)                         │
│  当 param.grad 就绪 → mark_variable_ready()          │
├──────────────────────────────────────────────────────┤
│  Bucket 管理                                         │
│  参数按逆序分桶, 桶满 → 触发 AllReduce               │
│                                                      │
│  Bucket[N-1] ← 先就绪 (最后层梯度先算完)             │
│  Bucket[N-2]                                         │
│  ...                                                 │
│  Bucket[0] ← 最后就绪 (第一层)                       │
├──────────────────────────────────────────────────────┤
│  AllReduce (异步, 与下一个 bucket 的 backward overlap)│
│  process_group.allreduce(bucket_tensor)              │
├──────────────────────────────────────────────────────┤
│  Wait & ÷ world_size (梯度平均)                      │
└──────────────────────────────────────────────────────┘
```

### 3.2 Autograd Hook 注册

```python
# 简化逻辑 (实际在 C++ Reducer 中):
for i, param in enumerate(all_params):
    # 注册 post-accumulate grad hook
    param.register_post_accumulate_grad_hook(
        lambda p, idx=i: reducer.mark_variable_ready(idx)
    )

# WHY post_accumulate_grad_hook 而非 register_hook?
# register_hook: 在 grad_fn 输出时触发, grad 可能还没累积完
# post_accumulate_grad: 在 .grad 赋值/累积完成后触发
# 对于 gradient_accumulation (多 micro-batch) 更正确
```

### 3.3 Bucket 就绪与 AllReduce 触发

```
时序图 (3 个 Bucket):

Timeline ────────────────────────────────────────────►
              
Backward: ═══[Layer N]═══[Layer N-1]══[...]═══[Layer 1]═══
              │              │                    │
              ▼              ▼                    ▼
Bucket 2:   ready ──►  AllReduce(async)          
Bucket 1:              ready ──────► AllReduce(async)
Bucket 0:                             ready ──► AllReduce
              
Overlap: AllReduce(Bucket 2) 与 Backward(Layer N-2...) 同时进行
```

## 4. Comm Hook 自定义通信 (L1600+)

### 4.1 默认 Hook: AllReduce + Average

```python
# 默认行为等价于:
def default_hook(state, bucket):
    tensor = bucket.buffer()
    fut = dist.all_reduce(tensor, op=ReduceOp.AVG, 
                          group=state.process_group, async_op=True)
    return fut.get_future()
```

### 4.2 常用自定义 Hook

| Hook | 用途 | 通信量 | 源码位置 |
|------|------|--------|---------|
| fp16_compress_hook | 梯度压缩为 FP16 | 50% 减少 | ddp_comm_hooks/default_hooks.py |
| powerSGD_hook | 低秩近似梯度 | >90% 减少 | ddp_comm_hooks/powerSGD_hook.py |
| quantization_hook | INT8 量化梯度 | 75% 减少 | ddp_comm_hooks/quantization_hooks.py |
| batched_comm_hook | 多桶合并通信 | 减少 latency | ddp_comm_hooks/default_hooks.py |

```python
# 使用 FP16 压缩:
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
ddp_model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)

# WHY Comm Hook?
# 默认 AllReduce(FP32) 通信量大, 尤其跨节点 (IB 带宽有限)
# 通过 hook 可以:
# 1. 压缩通信 (FP16/INT8)
# 2. 使用稀疏通信 (TopK)
# 3. 实现自定义 aggregation (如联邦学习)
```

## 5. Static Graph 优化 (L743-771)

```
Static Graph 模式 (static_graph=True):
──────────────────────────────────────
前提: 模型的前向/后向图在训练过程中不变

优势:
1. 首次迭代记录 unused params → 后续跳过 graph 搜索
2. 支持多次 activation checkpointing
3. 支持 forward 外的参数 (如 class variable)
4. 跳过每次迭代的 unused param 检测 → 减少开销

判断条件: ddp_logging_data.get("can_set_static_graph") == True
```

## 6. find_unused_parameters 机制 (L719-729)

```
WHY 需要 find_unused_parameters?
───────────────────────────────
问题: 如果某 param 在本次 forward 中未使用 (如条件分支)
      → 其 grad hook 永远不触发
      → 包含该 param 的 bucket 永远不会 ready
      → 所有 rank 的 DDP 永久 hang

解决: find_unused_parameters=True
      forward 后遍历 autograd graph
      标记不在 graph 中的参数为 "ready"
      
代价: 每次 forward 后需遍历 graph → O(参数数) 开销
```

## 7. DeviceMesh 集成 (DP + TP 混合)

```python
# torch/nn/parallel/distributed.py L854-883
# DDP + TP 混合并行 (2D 并行):
#
# 8 GPUs, TP=4, DP=2:
# DeviceMesh([[0,1,2,3], [4,5,6,7]])
#   dim=0: DP 维度 (跨行 AllReduce)
#   dim=1: TP 维度 (tensor 切分)
#
# DDP 接收 1D device_mesh (DP 维度):
if device_mesh.ndim != 1:
    raise RuntimeError("Only 1D device mesh is supported")
self.process_group = device_mesh.get_group(mesh_dim=0)

# 如果 mesh 是从 root_mesh 切出的子 mesh:
# 需要 _pre_dp_module_transform → 处理 DTensor 参数
```

## 8. 与 Megatron DistributedOptimizer 对比

| 维度 | PyTorch DDP | Megatron Distributed Optimizer |
|------|-------------|-------------------------------|
| 梯度同步 | AllReduce (全部 rank 持有完整梯度) | ReduceScatter (每 rank 只持有 shard) |
| 优化器状态 | 每 rank 完整副本 | 每 rank 1/N (ZeRO-1) |
| 内存效率 | 较差 (3x 参数内存) | 优 (1+1/N 参数内存) |
| 适用场景 | 模型 < 单卡内存 | 大模型 + 多卡 |
| 实现位置 | torch.nn.parallel | megatron/core/distributed/ |

## 9. 性能优化配置建议

```python
# 最佳实践配置:
model = DDP(
    model,
    device_ids=[local_rank],
    gradient_as_bucket_view=True,    # 省内存, 避免 copy
    static_graph=True,               # 如果图不变
    bucket_cap_mb=25,                # 默认, 可调
    find_unused_parameters=False,    # 关闭如果没有 unused params
    batched_grad_copy=True,          # 减少 kernel launch
)

# 环境变量:
# TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # 异步错误处理
# NCCL_IB_HCA=mlx5                   # 指定 IB 网卡
# NCCL_SOCKET_IFNAME=eth0            # 指定 TCP 网卡
```

## 10. 总结

```
DDP 核心设计决策:
┌────────────────┬───────────────────────────────────────┐
│ 决策           │ 原因                                   │
├────────────────┼───────────────────────────────────────┤
│ 多进程模型      │ 避免 GIL, 每 GPU 独立内存空间         │
│ 梯度桶打包      │ 平衡通信 latency vs bandwidth          │
│ Autograd Hook  │ 精确检测梯度就绪时机                    │
│ Overlap 通信   │ AllReduce 与 backward 并行              │
│ Comm Hook 扩展 │ 允许自定义压缩/聚合策略                 │
│ Static Graph   │ 避免重复 graph 遍历                     │
└────────────────┴───────────────────────────────────────┘
```

## 11. Reducer C++ 内部实现

### 11.1 Bucket 数据结构

```cpp
// torch/csrc/distributed/c10d/reducer.hpp
struct Bucket {
  std::vector<size_t> variable_indices;  // 桶内参数索引
  at::Tensor flat_tensor;                // 扁平化梯度 (连续内存)
  std::vector<at::Tensor> gradients;     // 梯度视图
  size_t size;                           // 桶大小 (字节)
  bool pending;                          // 是否等待 AllReduce 完成
  c10::intrusive_ptr<Work> work;         // AllReduce Work handle
};
```

### 11.2 mark_variable_ready 流程

```
mark_variable_ready(variable_index) 执行流:
─────────────────────────────────────────────
1. 找到参数所在 bucket_index
2. bucket.pending_count -= 1
3. if bucket.pending_count == 0:
   │  ┌── 桶就绪 ──┐
   │  │ 拷贝 grad 到 flat_tensor (或已是 view) │
   │  │ 调用 comm_hook 或默认 AllReduce         │
   │  │ 记录 Work handle                        │
   │  └──────────────┘
4. if all buckets ready:
   │  finalize_backward()
   │  等待所有 Work 完成
   │  除以 world_size (梯度平均)
```

### 11.3 参数分桶算法 (L1001-1050 参考)

```python
# 分桶策略 (简化 Python 伪代码):
def compute_buckets(params, bucket_cap_bytes):
    buckets = []
    current_bucket = []
    current_size = 0
    
    # 参数按 model.parameters() 逆序排列
    # WHY 逆序? backward 从最后一层开始计算梯度
    # 逆序使得最先就绪的梯度在同一个桶中
    for param in reversed(params):
        param_size = param.numel() * param.element_size()
        if current_size + param_size > bucket_cap_bytes and current_bucket:
            buckets.append(current_bucket)
            current_bucket = []
            current_size = 0
        current_bucket.append(param)
        current_size += param_size
    
    if current_bucket:
        buckets.append(current_bucket)
    return buckets
```

## 12. Mixed Precision 通信 (L193-233)

```python
# torch/nn/parallel/distributed.py L193
@dataclass
class _MixedPrecision:
    param_dtype: torch.dtype | None = None      # forward 参数精度
    reduce_dtype: torch.dtype | None = None     # AllReduce 通信精度
    buffer_dtype: torch.dtype | None = None     # buffer 精度

# 使用示例:
ddp_model = DDP(model, mixed_precision=_MixedPrecision(
    reduce_dtype=torch.float16  # AllReduce 用 FP16, 减少通信量 50%
))

# WHY reduce_dtype 单独控制?
# 参数和梯度计算可以保持 FP32 精度
# 仅在跨节点通信时降精度 → 平衡精度 vs 带宽
```

## 13. Join 机制 (不均匀输入处理)

```python
# torch/nn/parallel/distributed.py L412-464
# _DDPJoinHook: 处理不同 rank 数据量不一致的情况
#
# 问题: 如果 rank 0 有 100 batch, rank 1 有 98 batch
#        rank 1 先结束 → rank 0 的 AllReduce 永远等待
#
# 解决: Join context manager
with Join([ddp_model]):
    for batch in dataloader:  # 不同 rank batch 数可能不同
        loss = ddp_model(batch).sum()
        loss.backward()
        optimizer.step()
# 先结束的 rank 会发送 "shadow" AllReduce 参与通信
# 确保其他 rank 不会 hang
```

## 14. _DDPSink (Autograd Function) (L378-410)

```python
# torch/nn/parallel/distributed.py L378
class _DDPSink(Function):
    """DDP 在 forward 输出上挂的 autograd 节点"""
    
    @staticmethod
    def forward(ctx, reducer, *inputs):
        # 记录 reducer 引用, forward 时不做额外操作
        ctx.reducer = reducer
        return inputs
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        # backward 开始时: 通知 reducer 准备接收梯度
        # 用于实现 "所有 bucket 就绪后再 finalize" 的语义
        ctx.reducer.prepare_for_backward()
        return (None, *grad_outputs)
```

## 15. 关键源码文件索引

| 文件 | 行数 | 核心职责 |
|------|------|---------|
| torch/nn/parallel/distributed.py | 2667 | DDP Python 层 |
| torch/csrc/distributed/c10d/reducer.hpp | ~400 | Reducer C++ 头 |
| torch/csrc/distributed/c10d/reducer.cpp | ~2500 | Reducer C++ 实现 |
| torch/distributed/algorithms/ddp_comm_hooks/ | ~1000 | 通信 Hook 集合 |
| torch/distributed/_composable/replicate.py | ~300 | 新版 composable DDP |
