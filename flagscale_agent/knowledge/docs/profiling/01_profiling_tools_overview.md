# Chapter 01: Profiling工具体系总览

## 1. 设计动机

**WHY需要多层Profiling**: 分布式训练的性能问题可能出现在任何层级——
Python层的数据加载、框架层的调度开销、通信层的collective延迟、
算子层的kernel效率。不同工具覆盖不同层级，必须组合使用。

**核心问题**:
- 训练慢了，是计算/通信/IO哪个环节?
- 通信慢了，是带宽问题还是overlap不足?
- Kernel慢了，是TC利用率低还是内存瓶颈?

## 2. 工具矩阵

```
┌────────────────────────────────────────────────────────────────────┐
│ 层级         │ 工具              │ 粒度        │ 开销    │ 场景    │
├──────────────┼───────────────────┼─────────────┼─────────┼─────────┤
│ 系统级       │ nsys (NSight Sys) │ μs timeline │ <5%     │ 全局瓶颈│
│ 框架级       │ PyTorch Profiler  │ Op/Kernel   │ 5-15%   │ Op分析  │
│ 训练框架级   │ Megatron Profiler │ Step/Phase  │ <2%     │ 训练循环│
│ Kernel级     │ ncu (NSight Comp) │ 指令级      │ 100×+   │ 单kernel│
│ 通信级       │ NCCL_DEBUG/nsys   │ collective  │ <5%     │ 通信分析│
│ 内存级       │ torch.cuda.memory │ Allocation  │ <1%     │ OOM诊断│
└────────────────────────────────────────────────────────────────────┘

使用顺序(Top-Down):
1. Megatron内置timer → 定位哪个phase慢
2. nsys → 看compute/comm timeline overlap
3. PyTorch Profiler → 看具体哪些Op/Kernel耗时
4. ncu → 对单个kernel做深度profiling
```

## 3. 工具选择决策树

```
训练性能不达标?
├── Step time波动大?
│   ├── Yes → Megatron timer + 检查数据加载
│   └── No → 稳定但慢，继续↓
│
├── 通信占比高?
│   ├── 如何判断? → nsys timeline看comm kernel占比
│   ├── Yes → NCCL调优 (algo/proto/overlap)
│   └── No → 计算侧问题，继续↓
│
├── GPU利用率低?
│   ├── 如何判断? → nsys看GPU idle gap
│   ├── Yes → Pipeline bubble / 数据饥饿 / sync开销
│   └── No → kernel本身效率问题，继续↓
│
└── Kernel效率低?
    └── ncu分析 → TC利用率/带宽/bank conflict
```

## 4. 各工具Quick Start

### 4.1 NSight Systems (nsys)

```bash
# 基本用法 — profile整个训练:
nsys profile \
    -s none \                    # 不采样CPU (减小文件)
    -t cuda,nvtx,osrt \         # trace: CUDA + NVTX标记 + OS Runtime
    -o output_profile \          # 输出文件名
    --force-overwrite true \
    --capture-range=cudaProfilerApi \  # 只在代码标记范围内采集
    --capture-range-end=stop \
    python train.py --profile --profile-step-start 10 --profile-step-end 12

# 多进程 (torchrun):
nsys profile \
    -t cuda,nvtx,nccl \
    --mpi-impl openmpi \         # 自动跟踪所有rank
    -o profile_%q{RANK} \        # 每rank一个文件
    torchrun --nproc_per_node=8 train.py

# 输出格式:
# .nsys-rep → NSight Systems GUI打开
# .sqlite → 可用SQL查询
```

### 4.2 PyTorch Profiler

```python
# 独立使用:
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(
        wait=1,      # 跳过前1步
        warmup=1,    # 预热1步(不记录)
        active=3,    # 记录3步
        repeat=1,    # 重复1次
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step in range(10):
        train_step()
        prof.step()

# 输出查看:
# 1. TensorBoard: tensorboard --logdir=./log/profiler
# 2. Chrome trace: prof.export_chrome_trace("trace.json")
# 3. Table: print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### 4.3 Megatron-LM-FL内置Profiler

```yaml
# FlagScale config中启用:
system:
  profile: true                    # 启用profiling
  profile_step_start: 10           # 从第10步开始
  profile_step_end: 12             # 到第12步结束
  profile_ranks: [0, 7]            # 只profile这些rank
  use_pytorch_profiler: true       # 用PyTorch Profiler (可选)
  # 不设use_pytorch_profiler则用nsys模式
```

### 4.4 NSight Compute (ncu)

```bash
# 在CUDA Operator Analysis Ch13已详细覆盖
# Quick reference:
ncu --set full -o kernel_profile \
    --kernel-name "cutlass" --launch-count 1 \
    python inference.py

# 不适合profile整个训练(开销太大)!
# 只用于分析单个kernel
```

## 5. 关键指标与解读

### 5.1 nsys Timeline关键观察

```
一个典型的training step timeline:
                                                            
Rank 0 (PP stage 0):
│ FWD      │ SEND │ IDLE(bubble)  │ BWD       │ AllReduce  │
├──────────┼──────┼───────────────┼───────────┼────────────┤

Rank 7 (PP stage last):
│ IDLE     │ RECV │ FWD           │ BWD       │ AllReduce  │
├──────────┼──────┼───────────────┼───────────┼────────────┤

观察要点:
1. Bubble占比 = IDLE时间 / Step时间 → PP效率
2. AllReduce是否与BWD overlap? → 通信隐藏效率
3. FWD/BWD内部有无gap? → 数据加载/sync开销
4. NCCL kernel是否被拆分? → 小message overhead
```

### 5.2 PyTorch Profiler Table解读

```
----  -------  -------  -------  -------  -------
Name  Self CPU  CPU total  Self CUDA  CUDA total  # Calls
----  -------  -------  -------  -------  -------
aten::mm         0.5ms    0.5ms    45.2ms    45.2ms    48
aten::linear     0.1ms   46.0ms     0.0ms    45.5ms    24
ncclAllReduce    0.1ms    0.1ms    12.3ms    12.3ms     6
aten::layer_norm 0.2ms    0.3ms     3.1ms     3.1ms    24
aten::gelu       0.1ms    0.1ms     1.2ms     1.2ms    24

解读:
- Self CUDA最大的是mm (GEMM) → 正常，这是主要计算
- ncclAllReduce 12.3ms → 检查是否合理(对比理论通信时间)
- 如果layer_norm占比异常高 → 考虑fused kernel
- CPU total >> Self CPU → 看子调用链
```

### 5.3 Megatron Timer输出

```
 iteration     100 |  ...  | elapsed time per iteration (ms): 245.3
  forward-compute: 85.2ms (34.7%)
  backward-compute: 120.1ms (48.9%)
  backward-params-all-reduce: 15.3ms (6.2%)    ← DP gradient sync
  backward-embedding-all-reduce: 0.8ms (0.3%)
  optimizer: 12.5ms (5.1%)
  batch-generator: 2.1ms (0.9%)
  layernorm-grads-all-reduce: 0.0ms (0.0%)     ← SP grad sync
  send-forward: 5.2ms (2.1%)                   ← PP communication
  recv-forward: 4.8ms (2.0%)
  
分析:
- forward + backward compute = 83.6% → 正常
- params-all-reduce = 6.2% → 检查是否与compute overlap
- send/recv = 4.1% → PP通信开销, 看是否可overlap
```

## 6. 实战场景

### 6.1 场景: 训练MFU低于预期

```
预期MFU: 45% (H100 8卡)
实际MFU: 32%

诊断步骤:
1. Megatron timer → backward-compute占比正常
2. nsys timeline → 发现AllReduce未与BWD overlap!
   原因: DP=1 (只有TP/PP), 但gradient accumulation后有一次大AllReduce
3. 修复: 开启gradient bucket overlap (--overlap-grad-reduce)
4. 验证: MFU提升到 41%

继续:
5. nsys → 发现PP bubble 15%
6. 增加micro-batch数 (从4→8) 减少bubble
7. MFU → 44%
```

### 6.2 场景: 通信瓶颈定位

```
nsys中观察到AllReduce耗时异常:
- 数据量: 200MB (gradient)
- 耗时: 15ms
- 理论: 200MB / (400Gbps/8×2) = 2ms (ring, 8GPU)
- 实际/理论 = 7.5× → 严重异常

进一步诊断:
1. 检查NCCL_ALGO: 默认可能选了Tree (小消息好, 大消息差)
2. NCCL_DEBUG=INFO看实际选择的algorithm/protocol
3. 设置NCCL_ALGO=Ring, NCCL_PROTO=Simple
4. 重测: 3.2ms → 正常范围

根因: NCCL auto-tuning在该shape下选择了次优算法
```

### 6.3 场景: Memory Profiling

```python
# OOM诊断:
torch.cuda.memory._record_memory_history(max_entries=100000)

# 训练几步后:
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

# 分析:
# 用 https://pytorch.org/memory_viz 可视化
# 或:
snapshot = torch.cuda.memory._snapshot()
for seg in snapshot['segments']:
    print(f"Size: {seg['total_size']/1e9:.2f}GB, "
          f"Allocated: {seg['allocated_size']/1e9:.2f}GB")
```

## 7. 最佳实践

```
Daily工作流:
┌─────────────────────────────────────────────────────────┐
│ 1. 每次训练默认启用 Megatron timer (零额外成本)          │
│ 2. 性能异常时, 先用nsys跑2-3步 (低开销, 全局视角)       │
│ 3. 定位到具体阶段后, PyTorch Profiler看Op级别            │
│ 4. 必要时ncu分析单个kernel (离线, 不影响训练)           │
│ 5. 通信问题: NCCL_DEBUG=INFO + nsys的NCCL trace         │
│ 6. 内存问题: torch.cuda.memory snapshot                  │
└─────────────────────────────────────────────────────────┘

避免的坑:
- 不要在生产训练中开ncu (100×+ slowdown)
- nsys文件可能很大 (>1GB), 只profile 2-3步
- PyTorch Profiler的record_shapes增加~10%开销
- profile_ranks不要全开, 选代表性rank (first PP + last PP)
```
