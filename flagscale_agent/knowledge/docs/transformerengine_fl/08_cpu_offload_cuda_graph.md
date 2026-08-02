# Chapter 08: CPU Offload & CUDA Graph — 源码深度分析

## 源码位置

| 文件 | 行数 | 核心职责 |
|------|------|---------|
| `transformer_engine/pytorch/cpu_offload.py` | 943 | 激活CPU Offload框架 (V2) |
| `transformer_engine/pytorch/graph.py` | 1400 | CUDA Graph capture/replay + PP集成 |

---

## Part I: CPU Offload 系统

## 1. 架构总览

```
用户代码层:
  get_cpu_offload_context(enabled, num_layers, model_layers)
    → (_CpuOffloadContext, sync_function, [manual_controller])

调度层:
  OffloadSynchronizer (base)
  ├── DefaultOffloadSynchronizer   # 自动按layer编号调度D2H/H2D
  └── ManualOffloadSynchronizer    # PP场景手动控制

执行层:
  OffloadableLayerState            # 单层tensor管理、stream操作
  TensorGroupProcessor             # tensor去重/view优化
  TensorGroup                      # tensor集合 + CUDA events

硬件层:
  offload_stream (D2H/H2D)         # 专用CUDA stream
  pinned memory                    # 页锁定CPU内存
```

## 2. TensorGroup与TensorGroupProcessor (L90-225)

### 2.1 TensorGroup数据结构

```python
@dataclass
class TensorGroup:
    tensor_list: list[torch.Tensor] = field(default_factory=list)
    events: list[torch.cuda.Event] = field(default_factory=list)  # 每个tensor的就绪event
    aux: Any = None  # 辅助信息(view/dedup元数据)
```

### 2.2 Offload前优化 (tensor_group_process_before_offload)

offload前执行两步优化，减少实际传输量：

```python
@staticmethod
def tensor_group_process_before_offload(tensor_group):
    aux = {}
    # 优化1: view→base tensor（多个view共享同一base只传一次）
    tensor_group = TensorGroupProcessor._switch_to_base_tensors(aux, tensor_group)
    # 优化2: 去重（同一tensor对象只传一份）
    tensor_group = TensorGroupProcessor._deduplicate_tensors(aux, tensor_group)
    return tensor_group, aux
```

**_switch_to_base_tensors** (L138-171):
```python
def _check_if_offload_base_tensor(tensor):
    if getattr(tensor, "offload_base_tensor", False):
        return True  # 显式标记
    if tensor._base is not None:
        # 相同元素数的view → 安全offload base
        return tensor._base.numel() == tensor.numel()
    return False

# 记录view元信息(shape, stride, offset)用于reload后恢复
aux["views"].append((tensor.shape, tensor.stride(), tensor.storage_offset()))
tensor_group.tensor_list[id] = tensor._base  # 替换为base
```

**应用场景**：MultiHeadAttention中interleaved QKV tensor是同一buffer的三个view，只需offload一次base。

**_deduplicate_tensors** (L174-197):
```python
# 用id(tensor)做key，跳过重复引用
tensor_to_index: dict[int, int] = {}
for tensor_id, tensor in enumerate(tensor_group.tensor_list):
    if id(tensor) in tensor_to_index:
        aux["original_tensor_ids"].append(tensor_to_index[id(tensor)])
    else:
        tensor_to_index[id(tensor)] = len(dedup_tensors)
        dedup_tensors.append(tensor)
```

### 2.3 Reload后恢复 (tensor_group_process_after_reload)

```python
@staticmethod
def tensor_group_process_after_reload(tensor_group):
    # 恢复去重: 复制引用到原始位置
    tensor_group = TensorGroupProcessor._restore_tensor_duplicates(tensor_group)
    # 恢复view: 用as_strided重建原始view
    tensor_group = TensorGroupProcessor._switch_to_views(tensor_group)
    return tensor_group
```

## 3. OffloadableLayerState：单层Offload执行引擎 (L228-492)

### 3.1 状态机

```
                start_offload()           release_gpu_memory()
not_offloaded ─────────────────→ offload_started ──────────────→ offload_finished
                                                                       │
                                        start_reload()                 │
                                  reload_started ←─────────────────────┘
```

### 3.2 三阶段TensorGroup

```python
def __init__(self, offload_stream, retain_pinned_cpu_buffers=False):
    self.fwd_gpu_tensor_group = TensorGroup()   # forward时GPU上的tensor
    self.cpu_tensor_group = TensorGroup()       # offload后的CPU pinned tensor
    self.bwd_gpu_tensor_group = TensorGroup()   # reload后的GPU tensor
    self.state = "not_offloaded"
```

### 3.3 start_offload() (L259-312)

```python
def start_offload(self):
    self._validate_state("start_offload", ["not_offloaded"])
    self.state = "offload_started"
    
    # 优化: view合并 + 去重
    tensor_group, aux = TensorGroupProcessor.tensor_group_process_before_offload(
        self.fwd_gpu_tensor_group)
    self.aux = aux
    
    # 异步D2H传输
    for tensor in tensor_group.tensor_list:
        # 分配pinned CPU buffer
        cpu_tensor = torch.empty_like(tensor, 
            device="cpu", pin_memory=True)  # 页锁定内存
        
        # offload_stream等待compute stream上tensor就绪
        self.offload_stream.wait_stream(torch.cuda.current_stream())
        
        with torch.cuda.stream(self.offload_stream):
            cpu_tensor.copy_(tensor, non_blocking=True)  # async D2H
        
        # 记录event标记传输完成
        event = torch.cuda.Event()
        event.record(self.offload_stream)
        
        self.cpu_tensor_group.tensor_list.append(cpu_tensor)
        self.cpu_tensor_group.events.append(event)
```

### 3.4 release_activation_forward_gpu_memory() (L313-328)

```python
def release_activation_forward_gpu_memory(self):
    """等待D2H完成，释放GPU内存"""
    self._validate_state("release_gpu", ["offload_started"])
    self.state = "offload_finished"
    
    # 等待最后一个offload event
    for event in self.cpu_tensor_group.events:
        torch.cuda.current_stream().wait_event(event)
    
    # 释放forward GPU tensors
    self.fwd_gpu_tensor_group = TensorGroup()  # GC回收
```

### 3.5 start_reload() (L330-363)

```python
def start_reload(self):
    """异步H2D预取，reload到新的GPU buffer"""
    self._validate_state("start_reload", ["offload_finished"])
    self.state = "reload_started"
    
    self.bwd_gpu_tensor_group = TensorGroup()
    for tensor in self.cpu_tensor_group.tensor_list:
        # 注意: 在main stream上分配（避免跨stream内存池问题）
        reloaded_tensor = torch.empty_like(tensor, device="cuda")
        
        # offload_stream等待main stream
        self.offload_stream.wait_stream(torch.cuda.current_stream())
        
        with torch.cuda.stream(self.offload_stream):
            reloaded_tensor.copy_(tensor, non_blocking=True)  # async H2D
        
        # 记录reload完成event
        reload_event = torch.cuda.Event()
        reload_event.record(self.offload_stream)
        
        self.bwd_gpu_tensor_group.events.append(reload_event)
        self.bwd_gpu_tensor_group.tensor_list.append(reloaded_tensor)
    
    # 恢复view + dedup
    self.bwd_gpu_tensor_group.aux = self.aux
    self.bwd_gpu_tensor_group = TensorGroupProcessor.tensor_group_process_after_reload(
        self.bwd_gpu_tensor_group)
```

**关键设计**：reloaded tensor在main stream分配而非offload_stream上，因为PyTorch内存分配器不支持跨stream移动tensor（否则需要cudaFree+cudaMalloc）。

### 3.6 Offload过滤条件 (_check_if_offload, L453-492)

```python
def _check_if_offload(self, t: torch.Tensor) -> bool:
    # 条件1: 至少256K个元素(~512KB for BF16)，小tensor不值得offload
    if t.numel() < 256 * 1024:
        return False
    # 条件2: 非Parameter（权重不offload）
    if isinstance(t, torch.nn.Parameter):
        return False
    # 条件3: 未被标记为_TE_do_not_offload
    if getattr(t, "_TE_do_not_offload", False):
        return False
    # 条件4: 必须是GPU tensor
    if t.device.type != te_device_type():
        return False
    # 条件5: 必须连续（非连续tensor不支持）
    if not t.is_contiguous() and not getattr(t, "offload_base_tensor", False):
        return False
    return True
```

## 4. DefaultOffloadSynchronizer：自动调度策略 (L575-653)

### 4.1 调度算法 (_init_offload_synchronization_dicts, L601-625)

```python
def _init_offload_synchronization_dicts(self, num_offloaded_layers):
    """
    策略: offload前num_offloaded_layers层，在forward后半段释放GPU内存
    
    peak_memory = (num_layers - num_offloaded_layers) × T_per_layer
    
    对于layer i (i < num_offloaded_layers):
      finish_offload时机 = num_layers - num_offloaded_layers + i
      start_reload时机 = num_layers - 1 - num_offloaded_layers + i
    """
    for layer_id in range(self.num_layers):
        if layer_id < num_offloaded_layers:
            self.offload_layer_map[layer_id] = True
            # 延迟释放GPU: 给D2H足够时间完成
            self.finish_offload_map[
                self.num_layers - num_offloaded_layers + layer_id
            ].append(layer_id)
            # 提前reload: backward到达前预取
            self.start_reload_map[
                self.num_layers - 1 - num_offloaded_layers + layer_id
            ].append(layer_id)
        else:
            self.offload_layer_map[layer_id] = False
```

### 4.2 时序示例 (8层模型，offload 4层)

```
Forward:
  Layer 0: compute → start_offload(0)
  Layer 1: compute → start_offload(1)
  Layer 2: compute → start_offload(2)
  Layer 3: compute → start_offload(3)
  Layer 4: compute, release_gpu(0)     ← D2H(0)已完成
  Layer 5: compute, release_gpu(1)
  Layer 6: compute, release_gpu(2)
  Layer 7: compute, release_gpu(3)

GPU激活内存峰值 = 4层 × T (而非8层 × T)

Backward (逆序):
  Layer 7: bwd, start_reload(3)        ← 预取layer 3
  Layer 6: bwd, start_reload(2)
  Layer 5: bwd, start_reload(1)
  Layer 4: bwd, start_reload(0)
  Layer 3: bwd (使用reloaded激活)
  Layer 2: bwd
  Layer 1: bwd
  Layer 0: bwd
```

## 5. _CpuOffloadContext：用户接口 (L868-943)

### 5.1 saved_tensors_hooks机制

核心创新：利用PyTorch的`saved_tensors_hooks`拦截autograd的tensor保存/恢复：

```python
class _CpuOffloadContext(contextlib.ContextDecorator):
    def __enter__(self):
        # 注册全局hooks: autograd保存tensor时调用push，恢复时调用pop
        self._hooks_ctx = saved_tensors_hooks(
            offload_synchronizer.push_tensor,   # pack_hook: 保存时 → offload
            offload_synchronizer.pop_tensor     # unpack_hook: 恢复时 → reload
        )
        self._hooks_ctx.__enter__()
        
        # 设置全局OFFLOAD_SYNCHRONIZER供内部模块查询
        OFFLOAD_SYNCHRONIZER = offload_synchronizer
        self.current_layer = offload_synchronizer.fwd_step()
```

**原理**：PyTorch的`saved_tensors_hooks(pack, unpack)`允许自定义autograd中间tensor的序列化方式。TE-FL利用此API将tensor透明地offload到CPU，用户代码无需修改。

### 5.2 synchronization_function (L898-930)

```python
def synchronization_function(self, tensor):
    """注册backward hook，在backward到达时触发bwd_step"""
    cur_layer = self.current_layer
    
    def hook(_):
        # 在backward结束后清理内存
        torch.autograd.variable.Variable._execution_engine.queue_callback(
            offload_synchronizer.finish_part_of_bwd
        )
        # 触发当前层的bwd_step（启动reload等）
        offload_synchronizer.bwd_step(cur_layer)
    
    tensor.grad_fn.register_prehook(hook)
    return tensor
```

### 5.3 使用模式

```python
# 自动模式
ctx, sync_fn = get_cpu_offload_context(enabled=True, num_layers=4, model_layers=8)

for i in range(8):
    with ctx:
        x = layers[i](x)
    x = sync_fn(x)  # 注册backward hook + 同步点

# 手动模式 (PP场景)
ctx, sync_fn, controller = get_cpu_offload_context(
    enabled=True, model_layers=8, manual_synchronization=True)

for i in range(8):
    with ctx:
        x = layers[i](x)
    x = sync_fn(x)
    controller.start_offload_layer(i)

# 延迟释放 (overlap with后续compute)
for i in range(8):
    controller.release_activation_forward_gpu_memory(i)

# backward前预取
for i in reversed(range(8)):
    controller.start_reload_layer(i)
```

---

## Part II: CUDA Graph 系统

## 6. 架构总览 (graph.py)

```
用户入口: make_graphed_callables(modules, sample_args, num_warmup_iters=3)
    ↓
内部引擎: _make_graphed_callables()
    │
    ├── Phase 1: Warmup (num_warmup_iters次)
    │   → 稳定FP8 scaling factors
    │   → 检测TE modules和参数
    │   → 识别需要wgrad分离的层
    │
    ├── Phase 2: Capture Forward Graphs
    │   → 按PP _order顺序capture每层forward
    │   → 使用共享mempool避免内存碎片
    │
    ├── Phase 3: Capture Backward Graphs
    │   → 逆序capture每层backward
    │   → 可选: 分离dw graph (delay_wgrad_compute)
    │
    └── Phase 4: Return Wrapped Callables
        → 替换原始forward为graph.replay()
        → 处理输入/输出tensor的static buffer映射
```

## 7. Warmup阶段详解 (L452-556)

### 7.1 为什么需要Warmup？

1. **cuDNN Benchmarking**: 首次运行时cuDNN会benchmark不同算法，结果非确定性
2. **FP8 Scaling**: DelayedScaling需要几步积累amax history才能得到稳定scale
3. **Lazy Init**: 部分CUDA操作(cublas workspace等)首次调用时分配

### 7.2 Warmup执行

```python
with torch.cuda.stream(torch.cuda.Stream()):  # 隔离stream
    for func_idx, func in zip(warmup_func_idx, warmup_func):
        # Forward hook: 收集访问的TE modules
        hook = module.register_forward_hook(hook_fn)
        outputs = func(*args, **kwargs)
        hook.remove()
        
        if is_training:
            # Backward: 检测哪些参数有梯度
            torch.autograd.backward(outputs_requiring_grad, grad_tensors)
            
            # 检测delay_wgrad_compute需求
            for module in visited_te_modules[func_idx]:
                if hasattr(module, "need_backward_dw") and module.need_backward_dw():
                    need_bwd_dw_graph[func_idx] = True
                    module.backward_dw()  # 执行wgrad
```

### 7.3 TE Module发现机制

```python
def hook_fn(module, inputs, outputs, func_idx):
    modules = set()
    if isinstance(module, TransformerEngineBaseModule):
        modules.add(module)
    elif isinstance(module, Sequential):
        # 遍历OperationFuser内的basic_ops
        for module_group in module._module_groups:
            if isinstance(module_group, OperationFuser):
                for basic_op in module_group._basic_ops:
                    modules.add(basic_op)
    visited_te_modules[func_idx] = modules
```

## 8. Graph Capture阶段 (L558-700)

### 8.1 Forward Capture

```python
fwd_graphs = [torch.cuda.CUDAGraph() for _ in range(num_callables)]
mempool = graph_pool_handle()  # 共享内存池

# RNG状态注册（确保graph replay时RNG正确推进）
if graph_safe_rng_available():
    for _, state in get_all_rng_states().items():
        fwd_graph.register_generator_state(state)

# Capture
with _graph_context_wrapper(fwd_graph, pool=mempool):
    outputs = func(*args, **kwargs)  # 录制所有kernel
```

### 8.2 Pipeline Parallel支持 (_order参数)

```python
# _order示例 (2 model chunks, 2 microbatches):
# [1, 2, 1, 2, -1, -2, -1, -2]
#  ↑fwd chunk0  ↑fwd chunk1  ↑bwd chunk0  ↑bwd chunk1

# delay_wgrad_compute: 小数值表示wgrad
# [1, 2, 1, 2, -1.5, -2.5, -1, -2, -1.5, -2.5]
#                ↑wgrad only     ↑dx only     ↑wgrad
```

支持interleaved 1F1B schedule的graph capture，每个(chunk, microbatch, direction)都有独立graph。

### 8.3 三个Graph类型

```python
fwd_graphs = [...]      # Forward graphs: 每层一个
bwd_graphs = [...]      # Backward dx graphs: 输入梯度
bwd_dw_graphs = [...]   # Backward dw graphs: 权重梯度 (可选分离)
```

分离dw graph的优势：允许wgrad与下一层的forward overlap。

## 9. FP8与CUDA Graph的交互 (L1075-1122)

### 9.1 问题

FP8 DelayedScaling每步更新amax history和scale，这是动态操作，不能在static graph中执行。

### 9.2 解决方案：save/restore FP8 tensors

```python
def save_fp8_tensors(modules, amax_buffer):
    """Graph capture前保存FP8状态"""
    for module in modules:
        # 保存当前的scale/amax到buffer
        # Graph内使用固定的scale值
        saved_fp8_meta = module.get_fp8_meta()
        amax_buffer[module] = saved_fp8_meta["scaling_fwd"].amax_history.clone()

def restore_fp8_tensors(modules, amax_buffer):
    """Graph replay后恢复并更新FP8状态"""
    for module in modules:
        # 从graph执行结果中提取新的amax
        new_amax = module.get_fp8_meta()["scaling_fwd"].amax_history[0]
        # 更新到buffer中的history
        amax_buffer[module] = torch.roll(amax_buffer[module], -1, dims=0)
        amax_buffer[module][0] = new_amax
        # 基于更新后的history重新计算scale (在graph外)
        module.get_fp8_meta()["scaling_fwd"].scale = compute_scale(amax_buffer[module])
```

### 9.3 执行流程

```
每个训练step:
  1. [Graph外] 将当前scale写入graph的static buffer
  2. [Graph replay] forward+backward (使用固定scale)
  3. [Graph外] 读取graph产生的新amax
  4. [Graph外] 更新amax_history → 重新计算scale
  5. 下一步使用新scale
```

### 9.4 cache_quantized_params优化

```python
# make_graphed_callables参数
cache_quantized_params: bool = False
```

当`cache_quantized_params=True`时：
- 首次capture时量化权重为FP8并缓存
- 后续replay直接使用缓存的FP8权重
- 避免每步重新量化（节省~5%时间）
- **约束**：权重不能改变（推理/frozen层适用）

## 10. Static Buffer与输入/输出映射

### 10.1 问题

CUDA Graph要求tensor地址固定。但用户每步传入不同的input tensor。

### 10.2 解决方案

```python
# Capture时记录static input surface
per_callable_static_input_surfaces = [
    flatten_sample_args[i] + per_callable_module_params[i]
    for i in range(len(callables))
]

# Replay时: 将新input拷贝到static buffer
def graphed_forward(*user_args):
    for static, user in zip(static_inputs, user_args):
        static.copy_(user)  # 拷贝到graph认识的地址
    graph.replay()           # replay使用static buffer
    return tuple(out.clone() for out in static_outputs)  # 返回副本
```

## 11. 限制条件与规避

| 限制 | 原因 | 规避方案 |
|------|------|---------|
| 固定shape | Graph录制固定tensor尺寸 | Padding到固定长度 |
| 无条件分支 | Graph是静态执行图 | 将分支移到graph外 |
| 无动态allocation | Graph不能包含malloc | 预分配所有buffer |
| 与checkpoint冲突 | Recompute引入动态性 | 不用checkpoint或graph每个micro-op |
| FP8 scale更新 | 动态值 | save/restore机制(§9) |
| PP dynamic schedule | 微批次数可变 | _order参数指定静态schedule |

## 12. 端到端性能模型

### 12.1 CPU Offload收益分析

```
模型: 32层, hidden=4096, seq=4096, batch=4, BF16
激活/层 ≈ seq × batch × hidden × 10 × 2B = 4096×4×4096×10×2 = 1.28GB

全部保留: 32 × 1.28GB = 40.96GB
Offload 16层: 峰值 = 16 × 1.28GB = 20.48GB (节省50%)

D2H时间/层 = 1.28GB / 25GB/s(PCIe4) = 51ms
Layer compute = ~80ms (H100)
→ 可以完全重叠 (compute > transfer)
```

### 12.2 CUDA Graph收益分析

```
无Graph:
  每层kernel数 ≈ 50-100 (LayerNorm, Linear, Softmax, etc.)
  Launch overhead = 5μs × 75 = 375μs/层
  32层总overhead = 12ms

有Graph:
  Launch overhead = 1次replay ≈ 10μs/层
  32层总overhead = 0.32ms

节省 = 11.7ms/step
若step time = 200ms → 5.8%加速
```

### 12.3 两者组合

```
CPU Offload + CUDA Graph:
  - retain_pinned_cpu_buffers=True (Graph要求固定地址)
  - Graph capture时offload stream也被录制
  - 注意: Graph内不能有条件offload逻辑
```

## 13. 设计决策总结

| 设计选择 | 方案 | 对比方案 | 理由 |
|---------|------|---------|------|
| saved_tensors_hooks | 全局hook拦截 | 手动在每层插入 | 透明、无需改模型代码 |
| View→Base优化 | offload base tensor | offload每个view | QKV等场景节省2/3传输量 |
| Tensor去重 | id()检测 | 值比较 | O(1)判断，无GPU操作 |
| Reload在main stream分配 | main stream alloc | offload stream alloc | 避免跨stream内存池问题 |
| 调度策略 | 前N层offload | LRU/动态 | 确定性、无运行时决策开销 |
| Graph三分(fwd/dx/dw) | 独立capture | 单一graph | 支持wgrad与fwd overlap |
| Graph RNG | register_generator_state | 外部管理 | Graph内RNG正确推进 |
| FP8 scale更新 | graph外执行 | graph内条件更新 | 保持graph静态性 |

## 14. 调试与排查

```python
# 检查offload统计
synchronizer = OFFLOAD_SYNCHRONIZER
print(f"Total offloaded: {synchronizer.get_offloaded_total_size_mb():.1f} MB")

# 检查单层offload状态
for i, state in synchronizer.layer_states.items():
    print(f"Layer {i}: state={state.state}, "
          f"fwd_tensors={len(state.fwd_gpu_tensor_group.tensor_list)}, "
          f"cpu_tensors={len(state.cpu_tensor_group.tensor_list)}")

# 标记不需要offload的tensor
tensor._TE_do_not_offload = True

# 强制offload base tensor (view场景)
tensor.offload_base_tensor = True

# CUDA Graph debug: 禁用graph查看kernel list
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 不兼容graph，仅用于debug
```
