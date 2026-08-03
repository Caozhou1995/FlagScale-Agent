# Chapter 13: NSight Compute Profiling实战 深度分析

## 1. 设计动机

**WHY Profiling是优化的基础**: 不量化就不能优化。CUDA kernel性能问题往往隐藏在
非直觉之处——看似简单的访存pattern可能有32×的bank conflict，看似高效的kernel可能
TC利用率不到40%。NCU(NSight Compute)提供指令级精度的性能剖析。

## 2. NCU基础使用

### 2.1 命令行模式

```bash
# 基础profile (所有metrics):
ncu --set full -o profile_output ./my_program

# 只profile特定kernel:
ncu --kernel-name "gemm_kernel" ./my_program

# 限制kernel实例:
ncu --launch-skip 5 --launch-count 3 ./my_program
# 跳过前5次launch, profile接下来3次

# 指定metric子集 (更快):
ncu --metrics \
    sm__throughput.avg_pct_of_peak_sustained_elapsed,\
    dram__throughput.avg_pct_of_peak_sustained_elapsed,\
    l1tex__throughput.avg_pct_of_peak_sustained_elapsed \
    ./my_program
```

### 2.2 关键Metric分类

```
计算类:
├── sm__pipe_tensor_op_hmma_cycles_active.avg_pct_of_peak
│   → Tensor Core利用率 (GEMM核心指标)
├── sm__inst_executed_pipe_fma.avg           
│   → FMA指令数
├── sm__sass_thread_inst_executed_op_dfma_pred_on.sum
│   → FP64 FMA (double precision)
└── sm__pipe_tensor_op_hmma_instructions.sum
    → Tensor Core指令总数

访存类:
├── dram__bytes.sum → HBM总传输字节
├── dram__throughput.avg_pct_of_peak_sustained_elapsed
│   → HBM带宽利用率 (memory-bound核心指标)
├── l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum
│   → Global load字节
├── l2__throughput.avg_pct_of_peak → L2带宽利用率
└── l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum
    → SMEM bank conflict次数

效率类:
├── smsp__warps_active.avg_pct_of_peak_sustained_active
│   → Theoretical occupancy
├── sm__warps_active.avg → 实际active warps
├── sm__inst_executed.avg_per_cycle_elapsed
│   → IPC (instructions per cycle)
└── sm__warps_launched.sum → 总launched warps
```

## 3. 性能瓶颈诊断

### 3.1 Speed of Light分析

```
NCU Speed of Light (SOL) Dashboard:
┌─────────────────────────────────────────┐
│ Compute (SM) Throughput: 45%            │ 
│ Memory (DRAM) Throughput: 72%           │
│ L1/TEX Throughput: 35%                  │
│ L2 Throughput: 28%                      │
└─────────────────────────────────────────┘

解读:
- Memory >> Compute: Memory-bound → 优化访存
- Compute >> Memory: Compute-bound → 优化算术
- 两者都低: Latency-bound → 优化occupancy/pipeline
- 两者都高: 接近极限 → 算法层面优化

本例: Memory-bound (72% HBM, 仅45% compute)
→ 应该: fusion减少GMEM访问, 或用更低精度
```

### 3.2 Stall原因分析

```
ncu --metrics smsp__warp_issue_stalled_* ./kernel

常见stall原因:
├── stalled_long_scoreboard: 等待GMEM load (最常见)
│   → 解决: 更多prefetch, 提高occupancy
├── stalled_math_pipe_throttle: 数学管线满
│   → 通常是好事(compute-bound)!
├── stalled_mio_throttle: MIO(memory I/O)管线满
│   → SMEM带宽受限, 检查bank conflict
├── stalled_barrier: 等待__syncthreads
│   → 减少同步频率, 检查负载均衡
├── stalled_short_scoreboard: 等待SMEM/L1 result
│   → SMEM延迟, 检查bank conflict
└── stalled_not_selected: warp就绪但未被选择
    → scheduler争抢, 通常影响小
```

## 4. Tensor Core Profiling

### 4.1 TC利用率计算

```bash
ncu --metrics \
    sm__pipe_tensor_op_hmma_cycles_active.avg_pct_of_peak_sustained_active,\
    sm__pipe_tensor_op_hmma_instructions.sum,\
    sm__cycles_elapsed.avg \
    ./gemm_kernel

# 目标值:
# - 简单GEMM: > 80% (CUTLASS benchmark)
# - FlashAttention: 60-70% (有softmax overhead)
# - TE Linear (FP8): > 75%
# - MoE grouped GEMM: 50-65% (小矩阵)

# 如果TC < 50%:
# 检查1: tile size是否太小 (TC under-utilized per instruction)
# 检查2: K-loop是否有unnecessary sync
# 检查3: epilogue是否太重 (compute在non-TC pipe上)
```

### 4.2 Pipeline效率

```
理想pipeline (TC持续满载):
Cycle: |TMA|TMA|TMA|TMA|TMA|TMA|...
       |   |MMA|MMA|MMA|MMA|MMA|...
       |   |   |EPI|EPI|EPI|EPI|...

实际(有bubble):
Cycle: |TMA|   |TMA|   |TMA|   |...   ← TMA启动慢
       |   |   |   |MMA|MMA|   |...   ← TC有间隔
       
NCU中的表现:
- sm__pipe_tensor_op_hmma_cycles_active < cycles_elapsed
- 有大量 stalled_long_scoreboard (等TMA)
- pipeline depth不够: 增加stages

诊断方法:
ncu --metrics sm__pipe_tensor_op_hmma_cycles_active.sum,\
    sm__cycles_active.sum ./kernel
ratio = hmma_cycles / active_cycles  → 越接近1越好
```

## 5. Memory Profiling

### 5.1 带宽效率

```bash
# 实际达到的HBM带宽:
ncu --metrics dram__bytes.sum,gpu__time_duration.sum ./kernel

effective_bw = dram__bytes.sum / gpu__time_duration.sum
# 与理论峰值(3.35 TB/s)比较

# 理想情况:
# Bandwidth-limited kernel: > 80% peak
# 如果 < 60%:
# 1. Non-coalesced access → 检查access pattern
# 2. Small transfers → 检查vectorization
# 3. Page faults → 检查unified memory usage
```

### 5.2 Cache效率

```bash
# L2 hit rate:
ncu --metrics \
    l2__t_sectors_op_read_lookup_hit.sum,\
    l2__t_sectors_op_read_lookup_miss.sum ./kernel

hit_rate = hit / (hit + miss)
# GEMM: 通常 30-50% (tile reuse)
# Elementwise: 接近0% (streaming, no reuse)
# 小tensor操作: > 80% (fit in L2)

# L1 (SMEM/Texture) 效率:
ncu --metrics \
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum \
    ./kernel
```

## 6. 自定义Kernel Profiling流程

```
Step 1: Baseline (full set)
  ncu --set full -o baseline ./kernel
  → 获取所有metrics, 确定瓶颈类型

Step 2: Focus profiling  
  根据Step 1确定:
  - Memory-bound → focus on access pattern, bank conflicts
  - Compute-bound → focus on TC utilization, pipeline
  - Latency-bound → focus on stall reasons, occupancy

Step 3: Source correlation
  ncu --set source -o source_profile ./kernel
  → 每行源码/PTX/SASS对应的metric
  → 找到热点指令

Step 4: Comparison
  ncu --set full -o optimized ./kernel_v2
  ncu compare baseline.ncu-rep optimized.ncu-rep
  → 对比优化前后
```

## 7. 常见问题诊断表

| 症状 | 可能原因 | 诊断metric | 解决方案 |
|------|---------|------------|----------|
| TC利用率低 | Tile太小 | hmma_inst.sum | 增大tile |
| TC利用率低 | Pipeline bubble | hmma_cycles vs total | 增加stages |
| HBM带宽低 | Non-coalesced | sectors/request | 修复access |
| HBM带宽低 | 小request | bytes/request | Vectorize |
| 大量bank conflict | SMEM layout | bank_conflicts.sum | Swizzle |
| 高stall_barrier | 负载不均 | inst_per_warp variance | 重新分配work |
| 高stall_lg_sb | GMEM延迟 | sectors_pending | Prefetch/occupancy |

## 8. NCCL通信Profiling

```bash
# 通信kernel profiling:
ncu --target-processes all \
    --kernel-name "ncclKernel" \
    -o nccl_profile \
    python train.py

# 关键指标:
# - gpu__time_duration: 通信耗时
# - dram__bytes: 数据搬运量
# - 计算 effective bandwidth:
#   BW = data_size / time (应接近IB/NVLink带宽)

# nsys更适合系统级通信分析:
nsys profile --trace=cuda,nvtx,nccl \
    -o system_profile \
    python train.py
# → Timeline view showing compute/comm overlap
```

## 9. 总结

```
Profiling黄金法则:
1. 先确定瓶颈类型 (SOL dashboard)
2. 深入对应子系统 (compute/memory/latency)
3. Source-level定位热点
4. 修改 → 重新profile → 对比
5. 达到理论上限或满足需求后停止

关键指标速查:
┌────────────────────────────────────────────┐
│ GEMM:  TC util > 80%, pipeline ratio > 0.9 │
│ Fused: HBM BW > 80% peak, 0 bank conflict │
│ Attn:  TC util > 60%, no long stalls       │
│ Comm:  effective BW > 85% link bandwidth   │
└────────────────────────────────────────────┘
```
