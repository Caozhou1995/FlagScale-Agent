# Chapter 03: PyTorch Profiler深度使用 深度分析

## 1. 设计动机

**WHY PyTorch Profiler**: 它理解Op语义——不只是kernel名字，还能关联到
Python层的nn.Module、autograd Function。对于定位"哪个层最慢"最直观。
且输出可在TensorBoard中可视化，适合日常迭代。

**与nsys的互补**: nsys看timeline全貌，PyTorch Profiler看Op级聚合统计。
nsys发现"某段时间慢了"，PyTorch Profiler告诉你"是哪个Op导致的"。

## 2. API详解

### 2.1 核心API

```python
import torch.profiler
from torch.profiler import profile, ProfilerActivity, schedule

# 完整配置:
with profile(
    # 采集什么:
    activities=[
        ProfilerActivity.CPU,       # CPU端操作
        ProfilerActivity.CUDA,      # GPU kernel
    ],
    
    # 何时采集 (step-based schedule):
    schedule=schedule(
        wait=5,      # 前5步不采集
        warmup=2,    # 第6-7步预热(采集但不记录)
        active=3,    # 第8-10步正式记录
        repeat=1,    # 只做一轮
    ),
    
    # 输出:
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        './profiler_output'
    ),
    
    # 额外信息:
    record_shapes=True,        # 记录tensor shape
    profile_memory=True,       # 记录内存分配/释放
    with_stack=True,           # 记录Python调用栈
    with_flops=True,           # 估算FLOPS
    with_modules=True,         # 关联nn.Module名
) as prof:
    for step in range(15):
        train_step(model, data)
        prof.step()  # 通知profiler一步结束
```

### 2.2 Schedule工作原理

```
Step:  0  1  2  3  4 | 5  6 | 7  8  9 |
Phase: [--- wait ---] [warm] [active  ]

wait: 完全不采集, 零开销
warmup: 开始采集但不输出 (让CUDA cache热起来)
active: 正式记录, 结束时触发on_trace_ready

WHY需要warmup:
- 第一次kernel launch有JIT编译开销
- 内存分配器冷启动
- warmup让后续active步的数据更有代表性
```

### 2.3 输出方式

```python
# 方式1: TensorBoard (最常用)
on_trace_ready=torch.profiler.tensorboard_trace_handler('./tb_logs')
# → tensorboard --logdir=./tb_logs 查看

# 方式2: Chrome Trace (轻量)
prof.export_chrome_trace("trace.json")
# → chrome://tracing 打开

# 方式3: 表格统计
print(prof.key_averages().table(
    sort_by="cuda_time_total",   # 按GPU总耗时排序
    row_limit=20,                # 只显示前20
))

# 方式4: 按Module聚合
print(prof.key_averages(group_by_input_shape=True).table(
    sort_by="cpu_time_total"
))

# 方式5: Stacks (火焰图)
prof.export_stacks("profiler_stacks.txt", "self_cuda_time_total")
# → 用 FlameGraph工具可视化
```

## 3. Megatron中的集成

### 3.1 源码分析

```python
# training.py L2957-2981 (源码已验证):
# Megatron的PyTorch Profiler集成:

if args.profile and args.use_pytorch_profiler:
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=max(args.profile_step_start - 1, 0),
            warmup=1 if args.profile_step_start > 0 else 0,
            active=args.profile_step_end - args.profile_step_start,
            repeat=1,
        ),
        on_trace_ready=trace_handler,  # 输出到tensorboard_dir/../torch_profile
        record_shapes=args.pytorch_profiler_collect_shapes,
        with_stack=args.pytorch_profiler_collect_callstack,
    )
    prof.start()

# 每步调用 prof.step() (L3019)
# profile_step_end时调用 prof.stop() (L2597)
```

### 3.2 配置参数

```
Megatron-LM-FL支持的profiler参数:
# 源码: config/common_config.py L28-67

--profile                         # 启用profiling (总开关)
--profile-step-start 10           # 开始step
--profile-step-end 12             # 结束step  
--profile-ranks 0 7               # 指定rank
--use-pytorch-profiler            # 用PyTorch Profiler (否则nsys模式)
--pytorch-profiler-collect-shapes # 记录tensor shape
--pytorch-profiler-collect-callstack  # 记录调用栈
--pytorch-profiler-collect-chakra     # 输出Chakra execution trace
```

## 4. Table输出解读

### 4.1 关键列

```
Name                    Self CPU   CPU total  Self CUDA  CUDA total  # Calls
-----------------------  --------  ---------  ---------  ----------  -------
aten::mm                  0.5ms     0.5ms     45.2ms     45.2ms       48
ProfilerStep#8            1.2ms   245.0ms      0.0ms    180.5ms        1
aten::linear              0.1ms    46.0ms      0.0ms     45.5ms       24
ncclDevKernel_AllReduce   0.1ms     0.1ms     12.3ms     12.3ms        6
aten::layer_norm          0.2ms     0.3ms      3.1ms      3.1ms       24
aten::gelu                0.1ms     0.1ms      1.2ms      1.2ms       24
autograd::engine::run     0.0ms   180.0ms      0.0ms    130.0ms        1

关键理解:
- Self CPU: 该Op自身在CPU的时间(不含子调用)
- CPU total: 该Op+所有子调用的CPU时间
- Self CUDA: 该Op直接发起的CUDA kernel时间
- CUDA total: 该Op+子调用的所有CUDA时间
- Self CPU很小但CPU total很大 → 该Op是容器(如Linear包含mm)
```

### 4.2 常见分析模式

```python
# 模式1: 找最耗时的CUDA操作
table = prof.key_averages().table(sort_by="self_cuda_time_total")

# 模式2: 找CPU开销大的操作 (可能是调度瓶颈)
table = prof.key_averages().table(sort_by="self_cpu_time_total")

# 模式3: 按input shape聚合 (找shape导致的性能差异)
table = prof.key_averages(group_by_input_shape=True).table(
    sort_by="cuda_time_total")

# 模式4: 按Module名聚合
for event in prof.key_averages(group_by_stack_n=5):
    if event.self_cuda_time_total > 1000:  # >1ms
        print(f"{event.key}: {event.self_cuda_time_total/1000:.1f}ms")
        print(f"  Stack: {event.stack[:3]}")
```

## 5. Memory Profiling

### 5.1 内存快照

```python
# 方式1: 在profiler中 (profile_memory=True)
# Table会多出 Self Memory 列

# 方式2: 独立内存snapshot (更详细)
torch.cuda.memory._record_memory_history(max_entries=100000)

# 运行几步训练...
train_step()
train_step()

# 导出:
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
torch.cuda.memory._record_memory_history(enabled=None)  # 停止

# 可视化: https://pytorch.org/memory_viz
# 上传pickle文件即可看到:
# - 内存分配timeline
# - 每个tensor的大小和生命周期
# - 碎片化情况
```

### 5.2 OOM诊断

```python
# 快速查看当前内存状态:
print(torch.cuda.memory_summary())

# 输出示例:
# |         CUDA OOMs: 0            |
# |  Allocated memory : 45.2 GB     |  ← 实际使用
# |  Reserved memory  : 52.8 GB     |  ← 预留(含碎片)
# |  Active memory    : 42.1 GB     |  ← 活跃tensor
# |  Inactive memory  : 3.1 GB      |  ← 已释放但未归还
# |  # Segments       : 1247        |
# |  # Allocations    : 8923        |

# 如果 Reserved >> Allocated → 碎片严重
# 解决: torch.cuda.memory.set_per_process_memory_fraction(0.95)
#       或使用 expandable_segments=True
```

## 6. FLOPS估算

```python
# with_flops=True 时, profiler估算每个Op的FLOPS:
events = prof.key_averages()
for e in events:
    if e.flops > 0:
        # TFLOPS = flops / time_in_seconds / 1e12
        tflops = e.flops / (e.cuda_time_total * 1e-6) / 1e12
        print(f"{e.key}: {tflops:.1f} TFLOPS "
              f"({tflops/989*100:.1f}% of H100 peak)")

# 注意:
# - 只对标准Op有效(mm, conv, etc)
# - 自定义kernel无法自动估算
# - 用于快速判断kernel是否接近峰值
```

## 7. Execution Trace (Chakra)

```python
# Chakra trace用于分布式训练模拟和重放:
# 源码: training.py L2959-2962

if args.pytorch_profiler_collect_chakra:
    et_dir = Path(f"{args.tensorboard_dir}/../chakra")
    et = torch.profiler.ExecutionTraceObserver().register_callback(
        f"{et_dir}/rank-{rank}.json.gz"
    )

# Chakra trace记录:
# - 每个Op的依赖关系 (DAG)
# - tensor大小和device
# - 通信collective的参与者

# 用途:
# 1. 训练模拟器 (预测不同并行策略性能)
# 2. 通信pattern分析
# 3. 调度优化
```

## 8. 实战技巧

### 8.1 减少Profiler开销

```python
# 开销来源和控制:
# record_shapes: +5-10% → 不需要时关闭
# with_stack: +10-15% → 只在debug时开
# profile_memory: +5% → 内存问题时开
# with_flops: +2% → 性能分析时开

# 最低开销配置:
prof = profile(
    activities=[ProfilerActivity.CUDA],  # 不要CPU
    schedule=schedule(wait=9, warmup=1, active=1),
    record_shapes=False,
    with_stack=False,
)
```

### 8.2 与nsys互补使用

```
工作流:
1. nsys跑2步 → 发现BWD阶段有30ms unexplained gap
2. PyTorch Profiler (with_stack=True) → 发现是aten::copy_ 占30ms
3. 查看stack → 来自gradient accumulation中的tensor拷贝
4. 优化: 使用inplace操作避免拷贝
5. 重新nsys验证gap消失
```

## 9. 总结

```
PyTorch Profiler使用场景:
┌─────────────────────────────────────────────────────────┐
│ □ 找最耗时Op → sort_by="self_cuda_time_total"          │
│ □ 找CPU瓶颈 → sort_by="self_cpu_time_total"           │
│ □ 看module性能 → group_by_stack_n, with_modules        │
│ □ 内存分析 → profile_memory=True + memory_viz          │
│ □ 通信分析 → 看ncclKernel在table中的占比              │
│ □ Shape敏感分析 → group_by_input_shape=True            │
│ □ 火焰图 → export_stacks + FlameGraph                  │
│ □ TensorBoard → tensorboard_trace_handler               │
└─────────────────────────────────────────────────────────┘
```
