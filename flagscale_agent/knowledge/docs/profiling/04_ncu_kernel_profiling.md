# Chapter 04: Nsight Compute (NCU) Kernel级Profiling 深度分析

## 1. 设计定位

**WHY需要NCU**: nsys看kernel何时执行(timeline)，PyTorch Profiler看哪个Op慢，
而NCU回答**kernel为什么慢** — 它给出每个kernel的hardware counter：
SM利用率、内存带宽、指令吞吐、warp stall原因等。

**与nsys/PyTorch Profiler的分工**:
```
nsys       → 宏观timeline, 找到可疑的慢kernel名
PyTorch    → Op级统计, 关联到Python层  
NCU        → 单kernel显微镜, 给出硬件瓶颈
```

## 2. 命令行使用

### 2.1 基本用法

```bash
# 采集单个kernel的全部metrics:
ncu --target-processes all \
    --set full \                  # 采集所有section
    -o profile_output \           # 输出.ncu-rep文件
    python train.py --args...

# 只采集特定kernel:
ncu --kernel-name "ampere_fp16_s1688gemm" \
    --kernel-id ::1:3 \           # 第1次到第3次调用
    --set full \
    -o gemm_profile \
    python train.py

# 只采集特定范围 (配合NVTX):
ncu --nvtx --nvtx-include "forward/" \
    --set full \
    python train.py
```

### 2.2 常用选项

```bash
# Section选择 (减少采集时间):
--section SpeedOfLight_RooflineChart   # 只看roofline
--section MemoryWorkloadAnalysis       # 只看内存
--section ComputeWorkloadAnalysis      # 只看计算
--section LaunchStatistics             # 只看launch配置
--section WarpStateStatistics          # 只看warp状态
--section Occupancy                    # 只看占用率

# 采样控制:
--launch-count 5        # 只采集前5次kernel launch
--launch-skip 100       # 跳过前100次launch
--replay-mode kernel    # kernel replay (最准确但最慢)
--replay-mode range     # range replay (快但less accurate)

# 多GPU:
--target-processes all  # 所有进程
```

## 3. 核心Metrics解读

### 3.1 Speed of Light (SOL)

```
SOL表示达到硬件理论峰值的百分比:

┌──────────────────────────────────────────────────────────────┐
│ Metric                           │  Value  │  Peak   │  SOL │
├──────────────────────────────────┼─────────┼─────────┼──────┤
│ SM Throughput (Compute)          │  624 op │  989 TF │  63% │
│ Memory Throughput                │  2.1 TB │  3.35TB │  63% │
│ L1 Cache Throughput              │ 45.2 TB │  96 TB  │  47% │
│ L2 Cache Hit Rate                │  68%    │         │      │
└──────────────────────────────────────────────────────────────┘

解读:
- Compute SOL 63% → 计算密集型, 还有37%优化空间
- Memory SOL 63% → 也受内存限制 → 混合瓶颈
- 如果 Compute SOL >> Memory SOL → 计算瓶颈
- 如果 Memory SOL >> Compute SOL → 带宽瓶颈
- 两者都低 → latency瓶颈(可能occupancy不够)
```

### 3.2 Roofline分析

```
Roofline图核心概念:
  Performance = min(峰值算力, 算术强度 × 峰值带宽)

  TFLOPS
    │     ╱ 理论峰值 (989 TF H100)
    │    ╱─────────────────────────
    │   ╱          ← actual
    │  ╱    ●
    │ ╱        
    │╱            
    └────────────────────────── Arithmetic Intensity (FLOP/Byte)
    
    拐点 = 峰值算力 / 峰值带宽 = 989 / 3350 ≈ 295 FLOP/Byte

WHY Roofline重要:
- 告诉你kernel是compute-bound还是memory-bound
- 如果在斜线上(memory-bound): 优化内存访问/减少数据搬运
- 如果在天花板下(compute-bound): 优化指令/增加并行度
```

### 3.3 Warp State Statistics

```
Warp Stall原因分析:

┌────────────────────────┬────────┬──────────────────────────────┐
│ Stall Reason           │   %    │ 含义                          │
├────────────────────────┼────────┼──────────────────────────────┤
│ stall_barrier          │  35%   │ 等__syncthreads/同步          │
│ stall_long_scoreboard  │  28%   │ 等GMEM load (L2 miss)        │
│ stall_short_scoreboard │  15%   │ 等SMEM/L1 数据                │
│ stall_math_pipe        │  12%   │ 等数学pipeline空出            │
│ stall_not_selected     │   8%   │ 调度器选了别的warp            │
│ stall_mio_throttle     │   2%   │ 内存IO限流                    │
└────────────────────────┴────────┴──────────────────────────────┘

对策:
- stall_barrier高 → 减少sync点, 或pipeline化
- stall_long_scoreboard高 → prefetch, 增大tile减少GMEM访问
- stall_math_pipe高 → 数学计算饱和(好事!或用async)
- stall_not_selected高 → occupancy不够(增加block数)
```

## 4. 典型Kernel分析流程

### 4.1 GEMM Kernel分析

```bash
# Step 1: 找到GEMM kernel名
ncu --list-kernels python train.py  # 列出所有kernel

# Step 2: 采集GEMM详细metrics
ncu --kernel-name "sm90_xmma_gemm" \
    --section SpeedOfLight_RooflineChart \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    --launch-count 3 \
    -o gemm_analysis \
    python forward_only.py
```

```
典型好GEMM的指标 (H100, M=N=K=4096, FP16):
- Compute SOL: 80-90%  (接近989 TFLOPS)
- Memory SOL: 30-40%   (计算密集)
- Achieved TFLOPS: 750-850
- Occupancy: 50-75%

如果你的GEMM SOL只有40%:
1. 检查shape → 小GEMM天然SOL低
2. 检查对齐 → M,N,K需要16/32对齐
3. 检查tile选择 → ncu显示used tile size
4. 检查epilogue → 融合bias/activation是否生效
```

### 4.2 Softmax Kernel分析

```
Softmax是典型memory-bound kernel:

理论分析:
  - 读Q矩阵: seq_len × head_dim × 2B (FP16)
  - 写output: seq_len × head_dim × 2B  
  - 算术强度 ≈ 5 FLOP/Byte (exp, sub, div)
  - 远低于roofline拐点(295) → 纯memory-bound

NCU应该显示:
  Memory SOL: 70-90%  (接近带宽峰值)
  Compute SOL: 5-15%  (大量idle)
  
优化方向:
  → 与前后Op融合(减少GMEM round-trip)
  → Flash Attention的online算法(避免materialization)
```

## 5. 分布式训练Profiling技巧

### 5.1 单卡分析 (最常用)

```bash
# 只profile rank 0, 跳过前20步(让模型warm up):
CUDA_VISIBLE_DEVICES=0 \
ncu --launch-skip 200 --launch-count 50 \
    --set full \
    -o single_rank_profile \
    torchrun --nproc_per_node=1 train.py --no-save
```

### 5.2 通信Kernel分析

```bash
# NCCL kernel也能被ncu采集:
ncu --kernel-name "ncclDevKernel" \
    --section MemoryWorkloadAnalysis \
    --launch-count 5 \
    torchrun --nproc_per_node=8 train.py

# 看NCCL kernel的带宽利用率:
# 理论: NVLink 900 GB/s (all-to-all)
# 实际显示Memory SOL可判断NVLink是否打满
```

## 6. 实战案例: 定位性能问题

### 6.1 案例: Linear层TFLOPS只有300 (期望>700)

```bash
# Step 1: ncu确认kernel名
ncu --list-kernels | grep gemm
# → sm90_xmma_gemm_f16f16_f16f32_f32...

# Step 2: 详细profile
ncu --kernel-name "sm90_xmma_gemm_f16f16" \
    --section full --launch-count 1 -o debug

# Step 3: 检查输出
# 发现: Occupancy只有25% (期望>50%)
#        Launch配置: blocks=32, threads=128
#        正常应该: blocks=132, threads=256

# Step 4: 根因
# shape是 [1, 4096] × [4096, 11008] → M=1!
# batch_size=1导致GEMM退化为GEMV
# GEMV的SOL天然很低

# Step 5: 解决
# 增大micro_batch_size, 或使用专用GEMV kernel
```

## 7. 与FlagScale集成

```bash
# 方法1: 环境变量控制nsys/ncu
# 在FlagScale config中:
experiment:
  runner:
    envs:
      NCU_ENABLED: "1"  # 自定义控制变量

# 方法2: 配合Megatron的profile机制
# --profile --profile-step-start 10 --profile-step-end 12
# 在这些步骤中手动添加NVTX标记:
# ncu --nvtx --nvtx-include "train_step/" 精准采集

# 方法3: 独立运行 (推荐)
# 用单卡跑forward-only脚本, ncu包裹
```

## 8. 总结: NCU分析检查清单

```
性能诊断步骤:
┌─────────────────────────────────────────────────────────────┐
│ 1. Roofline定位 → compute-bound or memory-bound?           │
│ 2. SOL百分比  → 距峰值多远? 有多少优化空间?               │
│ 3. Occupancy  → block配置是否合理? register压力?           │
│ 4. Warp Stall → 等什么? barrier/scoreboard/MIO?           │
│ 5. Memory效率 → L2 hit rate? 合并访问比例?                │
│ 6. Instruction → 无用指令? predicated off?                 │
│ 7. 比较基线   → 同shape的cuBLAS/CUTLASS达到多少?          │
└─────────────────────────────────────────────────────────────┘
```
