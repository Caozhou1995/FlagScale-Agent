# Chapter 12: Occupancy与Launch配置 深度分析

## 1. 设计动机

**WHY Launch配置关键**: grid/block尺寸直接决定硬件资源利用率。
错误配置可能导致SM空闲50%+。但"最大化occupancy"并非万能——
GEMM反而需要低occupancy以获得最多SMEM和寄存器。

## 2. 硬件资源约束 (H100)

```
每SM资源上限:
├── Threads: 2048 (= 64 warps)
├── Blocks (CTAs): 32
├── Registers: 65536 (32-bit)
├── Shared Memory: 228 KB (最大配置)
└── Warps in-flight: 64

每Block约束:
├── Threads: 1024
├── Registers: 65536 (与SM共享)
├── Shared Memory: 228 KB (与SM共享)
└── Warps: 32

SM资源分配 = min(
    2048 / threads_per_block,          // thread限制
    32,                                 // block数限制  
    65536 / (regs_per_thread × threads_per_block),  // 寄存器限制
    228KB / smem_per_block             // SMEM限制
) × threads_per_block / 2048
```

## 3. Occupancy分析

### 3.1 计算方法

```python
# 示例: GEMM kernel
threads_per_block = 128  # 1 warpgroup
regs_per_thread = 160
smem_per_block = 180 * 1024  # 180 KB (multi-stage)

# 约束1: thread限制
max_blocks_thread = 2048 / 128 = 16

# 约束2: register限制
max_blocks_reg = 65536 / (160 × 128) = 3.2 → 3 blocks

# 约束3: SMEM限制  
max_blocks_smem = 228 / 180 = 1.26 → 1 block

# 实际blocks per SM = min(16, 3, 1, 32) = 1
# Occupancy = 1 × 128 / 2048 = 6.25%

# 看似很低，但这个GEMM kernel能达到85%+ peak!
```

### 3.2 WHY低Occupancy不等于低性能

```
延迟隐藏的两种方式:

方式1: TLP (Thread-Level Parallelism)
- 多个warp切换隐藏延迟
- 需要高occupancy
- 适合: memory-bound, 简单kernel

方式2: ILP/MLP (Instruction/Memory-Level Parallelism)
- 单warp内多条独立指令重叠
- 需要: pipeline + async
- 适合: compute-bound GEMM (TMA + WGMMA pipeline)

GEMM pipeline:
  TMA load stage[i+3]     (async, non-blocking)
  WGMMA compute stage[i]  (async, 多条连续提交)
  → 即使只有1个block, 延迟被pipeline完全隐藏
  → occupancy不重要!
```

### 3.3 什么时候需要高Occupancy

```
Memory-bound kernel (LayerNorm, Softmax):
- 几乎没有计算可做 pipeline
- 延迟只能靠warp切换隐藏
- 需要 occupancy > 50% (32+ warps)

配置:
threads_per_block = 256 或 512
regs_per_thread < 64 (avoid spill)
smem_per_block < 16 KB

→ 多block并发, 高occupancy, 最大化HBM带宽利用
```

## 4. Grid配置策略

### 4.1 Wave Quantization

```
GPU有132 SMs。如果kernel需要133 blocks:
Wave 1: 132 blocks → 全部SM满载
Wave 2: 1 block → 只有1个SM工作 (131个空闲!)

利用率: (132 + 1) / (132 × 2) = 50.4%

解决: 确保block数是SM数的整数倍(或接近)
grid_size = ceil(work / block_work) 
→ 如果接近 N×132+1, 调整block_work让总数更均匀
```

### 4.2 Persistent Grid

```cpp
// 传统: 每tile一个block
dim3 grid(M_tiles * N_tiles);  // 可能很大

// Persistent: grid固定为SM数×concurrent_blocks
int num_sms;
cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
dim3 grid(num_sms * blocks_per_sm);  // 132×1 = 132

// 内部循环处理所有tile:
while (tile_idx < total_tiles) { ... tile_idx += gridDim.x; }

// 优势:
// 1. 消除wave quantization问题
// 2. 更好的L2 cache利用
// 3. 可实现tile-level load balancing
```

### 4.3 Thread Block Cluster (H100)

```
SM90新特性: Thread Block Cluster
将相邻SM上的block组合为cluster:
- Cluster内block可以直接访问彼此SMEM (distributed SMEM)
- Cluster同步 (cluster.sync)
- 硬件保证cluster内block调度到相邻SM

配置:
cudaLaunchConfig_t config = {};
config.gridDim = {num_blocks, 1, 1};
config.blockDim = {128, 1, 1};

cudaLaunchAttribute attrs[1];
attrs[0].id = cudaLaunchAttributeClusterDimension;
attrs[0].val.clusterDim = {2, 1, 1};  // 2个block为一个cluster
config.attrs = attrs;
config.numAttrs = 1;

cudaLaunchKernelEx(&config, kernel, args...);

// WHY Cluster: 跨SM通信无需经过L2
// 用途: split-K GEMM, Attention的跨tile通信
```

## 5. Block Size选择

### 5.1 常见选择

| Block Size | 适用场景 | 原因 |
|-----------|----------|------|
| 32 | 小规模reduce | 刚好1 warp, 无__syncthreads |
| 128 | GEMM (SM90) | 1 warpgroup = WGMMA单位 |
| 256 | Memory-bound | 高occupancy + vectorize |
| 512 | Reduction | 大规模并行reduce |
| 1024 | 罕见 | 寄存器压力大, 特殊需求 |

### 5.2 经验法则

```
GEMM类 (compute-bound):
  block = 128 (1 warpgroup for WGMMA)
  grid = M_tiles × N_tiles (或persistent)
  occupancy: 1-2 blocks/SM

Elementwise类 (memory-bound):
  block = 256
  grid = min(total_elements / (256 × 4), num_sms × 4)
  occupancy: 4-8 blocks/SM

Reduction类:
  block = 256 或 512
  grid = batch_size (或 elements / block_work)
  occupancy: 2-4 blocks/SM
```

## 6. Occupancy Calculator使用

```python
# Python API:
import torch
from torch.cuda import max_shared_memory_per_block_optin

# 自动选择最优配置:
def auto_launch_config(kernel, smem_size):
    max_threads = 1024
    for block_size in [256, 128, 512, 64, 1024]:
        occupancy = torch.cuda.get_device_properties(0).multi_processor_count
        # 实际使用cudaOccupancyMaxActiveBlocksPerMultiprocessor
        ...
    return best_block_size

# CUDA API:
int min_grid, block_size;
cudaOccupancyMaxPotentialBlockSize(&min_grid, &block_size, kernel, 0, 0);
// 自动搜索最优block_size

# 指定SMEM:
cudaOccupancyMaxPotentialBlockSizeVariableSMem(
    &min_grid, &block_size, kernel, smem_calculator, 0);
```

## 7. 调优工作流

```
1. Profile baseline:
   ncu --target-processes all ./my_kernel
   
2. 检查关键metrics:
   - sm__warps_active.avg_pct_of_peak_sustained_active  (occupancy)
   - sm__pipe_tensor_op_hmma_cycles_active.avg_pct      (TC利用率)
   - dram__throughput.avg_pct_of_peak_sustained_elapsed  (HBM利用率)
   
3. 判断bottleneck:
   if TC < 60% and HBM < 60%: → Launch config或pipeline问题
   if TC > 80%: → Compute-bound, 正常
   if HBM > 80%: → Memory-bound, 考虑fusion/quantize
   
4. 迭代调整:
   - Compute-bound: 增大tile (更多ILP)
   - Memory-bound: 增大occupancy (更多TLP)
   - Latency-bound: 增加pipeline stages
```

## 8. 总结

```
Launch配置决策树:
┌── Compute-bound (GEMM)?
│   ├── Yes → block=128, low occupancy, max SMEM
│   │         pipeline stages=3-7, persistent grid
│   └── No → Memory-bound?
│       ├── Yes → block=256, high occupancy, min SMEM
│       │         vectorized access, max blocks/SM
│       └── Mixed → profile后决定

关键认知:
- 高occupancy ≠ 高性能 (GEMM的反例)
- Wave quantization是隐藏杀手 (grid设计)
- H100的Cluster是新维度 (跨SM协作)
- 先profile再优化，不要猜
```
