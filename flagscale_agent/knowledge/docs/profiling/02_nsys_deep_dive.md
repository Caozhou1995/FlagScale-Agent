# Chapter 02: NSight Systems深度使用 深度分析

## 1. 设计动机

**WHY nsys是首选**: nsys提供系统级timeline视图，是唯一能同时展示
compute/communication/data pipeline重叠关系的工具。其他工具只能看局部，
nsys看全局。

**WHY不直接用ncu**: ncu是kernel级(微观)，nsys是system级(宏观)。
优化分布式训练必须先从宏观定位瓶颈区域，再用ncu深入微观。

## 2. 采集策略

### 2.1 分布式训练场景

```bash
# 方法1: Megatron内置profile flag
# 启动训练时添加:
#   --profile --profile-step-start 10 --profile-step-end 12
# 这会在step 10调用cudaProfilerStart(), step 12调用cudaProfilerStop()

# 用nsys包裹训练进程:
nsys profile \
    -s none \
    -t cuda,nvtx,osrt,cudnn,cublas \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    -o /workspace/profiles/train_rank0 \
    python train.py --profile --profile-step-start 10 --profile-step-end 12

# 方法2: 多进程profile (MPI模式)
nsys profile \
    -t cuda,nvtx,nccl \
    --mpi-impl openmpi \
    -o profile_%q{RANK} \
    mpirun -np 8 python train.py
```

### 2.2 关键Trace类型

```
-t 参数选择:
├── cuda      → CUDA API + Kernel launch (必选)
├── nvtx      → 用户标记区间 (Megatron用NVTX标记forward/backward)
├── osrt      → OS runtime (pthread, mmap, IO)
├── cudnn     → cuDNN API调用
├── cublas    → cuBLAS API调用
├── nccl      → NCCL通信 (nsys 2024+ 直接支持)
├── mpi       → MPI调用
└── opengl    → 不需要(训练场景)

推荐组合:
训练全局分析: -t cuda,nvtx,osrt
通信分析: -t cuda,nvtx,nccl
IO分析: -t cuda,nvtx,osrt (关注IO syscall)
```

### 2.3 减小Profile文件

```bash
# 问题: 完整训练profile可能>10GB
# 解决方案:

# 1. 只采集指定step范围 (配合cudaProfilerApi)
--capture-range=cudaProfilerApi

# 2. 不采样CPU stack (减50%+文件大小)
-s none

# 3. 限制CUDA event buffer
--cuda-event-buffer-size=8192

# 4. 过滤kernel (只看特定名字)
--cuda-launch-filter="ncclKernel|gemm|flash"

# 5. 用stats导出而非完整trace
nsys stats output.nsys-rep --report gputrace > gpu_trace.csv
```

## 3. Timeline分析方法

### 3.1 识别Pipeline Bubble

```
理想的PP=4 pipeline (1F1B schedule):
Step boundary:
Rank0: |F0|F1|F2|F3|     |B3|B2|B1|B0| Opt |
Rank1: |  |F0|F1|F2|F3|  |B3|B2|B1|B0| Opt |
Rank2: |  |  |F0|F1|F2|F3|B3|B2|B1|B0| Opt |
Rank3: |  |  |  |F0|F1|F2|F3|B3|B2|B1|B0|Opt|

nsys中看到的bubble:
- Rank0前面等待recv → idle gap
- Rank3后面等待send → idle gap
- Bubble总时间 = (PP-1) × micro_batch_time

如果bubble > (PP-1)/PP × 30%:
→ micro batch太少, 或pipeline schedule不优
```

### 3.2 识别通信与计算Overlap

```
Good (overlap):
GPU Compute: |████GEMM████|████GEMM████|████GEMM████|
NCCL Stream: |▓▓AllReduce▓▓|            |▓▓AllReduce▓▓|
→ 通信被计算隐藏

Bad (no overlap):
GPU Compute: |████GEMM████|          |████GEMM████|
NCCL Stream: |            |▓▓AllRed▓▓|            |
→ GPU idle等待通信!

nsys中如何看:
1. 展开CUDA Streams行
2. Compute stream和NCCL stream应该有重叠区域
3. 如果NCCL stream活跃时compute stream为空 → overlap失败

常见原因:
- --overlap-grad-reduce 未启用
- gradient accumulation steps后的sync AllReduce
- bucket太大导致必须等所有gradient ready
```

### 3.3 识别Data Loading瓶颈

```
Data loading瓶颈在nsys中的表现:
GPU Stream: |████compute████|         gap        |████compute████|
CPU Thread: |               |████dataloader████|               |

如果gap > 5% step time → data pipeline是瓶颈

解决:
1. 增加num_workers
2. 使用prefetch_factor > 2
3. 数据预处理放到offline
4. 使用DALI (GPU加速数据加载)
```

## 4. NVTX标记利用

### 4.1 Megatron内置NVTX标记

```python
# Megatron在关键阶段自动添加NVTX标记:
# (当 --profile 启用时, emit_nvtx=True)
# 源码: training.py L3022
# nsys_nvtx_context = torch.autograd.profiler.emit_nvtx(record_shapes=True)

# 在nsys timeline中可见的标记:
# "forward-compute"
# "backward-compute"  
# "optimizer-step"
# "batch-generator"
# "send-forward" / "recv-forward"
# "send-backward" / "recv-backward"

# 对应training.py中的timers:
# timers('forward-compute', log_level=1).start()
# output_tensor = forward_step(...)
# timers('forward-compute').stop()
```

### 4.2 自定义NVTX标记

```python
# 在自己的代码中添加标记:
import torch.cuda.nvtx as nvtx

# 简单标记:
nvtx.range_push("my_attention")
output = self.attention(q, k, v)
nvtx.range_pop()

# Context manager方式:
with torch.autograd.profiler.record_function("custom_mla"):
    out = mla_forward(q, kv_compressed)

# 在nsys timeline中会显示为带颜色的区间
# 方便快速识别不同阶段
```

## 5. nsys命令行分析(无GUI)

### 5.1 Stats报告

```bash
# GPU kernel统计:
nsys stats output.nsys-rep --report gputrace \
    --format csv > kernels.csv

# 查看Top-10耗时kernel:
nsys stats output.nsys-rep --report gpukernsum \
    | sort -t',' -k3 -rn | head -10

# NVTX区间统计:
nsys stats output.nsys-rep --report nvtxsum

# CUDA API统计:
nsys stats output.nsys-rep --report cudaapisum

# 生成summary报告:
nsys stats output.nsys-rep --report summary
```

### 5.2 自动化分析脚本

```python
# 用sqlite分析nsys输出:
import sqlite3

conn = sqlite3.connect("output.sqlite")
cur = conn.cursor()

# 查询所有NCCL kernel耗时:
cur.execute("""
    SELECT shortName, 
           SUM(end - start) / 1e6 as total_ms,
           COUNT(*) as count,
           AVG(end - start) / 1e6 as avg_ms
    FROM CUPTI_ACTIVITY_KIND_KERNEL
    WHERE shortName LIKE '%nccl%'
    GROUP BY shortName
    ORDER BY total_ms DESC
""")
for row in cur.fetchall():
    print(f"{row[0]}: total={row[1]:.1f}ms, count={row[2]}, avg={row[3]:.3f}ms")

# 查询compute/comm overlap ratio:
cur.execute("""
    SELECT 
        SUM(CASE WHEN shortName LIKE '%nccl%' THEN end-start ELSE 0 END) as comm_ns,
        MAX(end) - MIN(start) as total_ns
    FROM CUPTI_ACTIVITY_KIND_KERNEL
""")
comm, total = cur.fetchone()
print(f"Communication fraction: {comm/total*100:.1f}%")
```

## 6. 多GPU Profile对比

```bash
# Profile所有8个GPU (每个单独文件):
# 训练脚本自动为每个rank生成不同文件名

# 对比不同rank的kernel分布:
for i in $(seq 0 7); do
    echo "=== Rank $i ==="
    nsys stats profile_rank${i}.nsys-rep --report gpukernsum \
        | grep -E "total|nccl" | head -5
done

# 找straggler rank:
# 如果某个rank的step time远高于其他 → 该rank有问题
# 常见原因: 
# - NUMA affinity错误 (GPU和NIC跨NUMA node)
# - NIC绑定错误 (IB流量走了错误的port)
# - 数据不均匀 (某rank的micro-batch更长)
# - 热节流 (某GPU温度过高降频)
```

## 7. WHY nsys优于其他Timeline工具

| 特性 | nsys | PyTorch Profiler | nvprof(deprecated) |
|------|------|------------------|---------------------|
| 多GPU支持 | 原生 | 需手动 | 有限 |
| NCCL trace | ✅ | 间接(通过kernel名) | ❌ |
| CPU+GPU联合 | ✅ | ✅ | GPU为主 |
| 文件大小 | 可控 | 较大 | 较大 |
| 远程分析 | CLI stats | TensorBoard | ❌ |
| 低开销 | <5% | 5-15% | 10-20% |
| OS级别trace | ✅ | ❌ | ❌ |

## 8. 总结

```
nsys使用checklist:
□ 只profile 2-3步 (capture-range=cudaProfilerApi)
□ 选代表性rank (PP first + last, 或DP的不同node)
□ 用-s none减小文件
□ 先用stats做CLI分析, 需要时再开GUI
□ 关注: bubble / overlap / data gap / straggler
□ 用NVTX标记定位自定义阶段
□ 保存baseline profile, 优化后对比
```
