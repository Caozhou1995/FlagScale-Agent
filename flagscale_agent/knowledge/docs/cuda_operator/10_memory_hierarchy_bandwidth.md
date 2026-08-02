# Chapter 10: Memory Hierarchy与带宽优化 深度分析

## 1. 设计动机

**WHY内存层次是性能瓶颈**: 现代GPU的Tensor Core算力增长远超内存带宽增长。
H100的FP16算力989 TFLOPS，但HBM带宽仅3.35 TB/s。
计算/访存比 = 989×10^12 / (3.35×10^12 / 2) = 591:1。
意味着每从HBM读1个FP16(2B)，TC必须做591次FP16 FMA才不浪费。

## 2. H100内存层次

```
┌─────────────────────────────────────────────────────────┐
│                   Registers (RF)                          │
│ 256 KB/SM × 132 SM = 33.8 MB total                       │
│ 带宽: ~200 TB/s (估算, per SM ~1.5 TB/s)                 │
│ 延迟: 1 cycle                                            │
├─────────────────────────────────────────────────────────┤
│                Shared Memory (SMEM)                        │
│ 228 KB/SM × 132 SM = 29.4 MB total                       │
│ 带宽: ~30 TB/s (all SMs, 无conflict)                     │
│ 延迟: ~20 cycles                                          │
├─────────────────────────────────────────────────────────┤
│                    L2 Cache                                │
│ 50 MB (centralized)                                       │
│ 带宽: ~12 TB/s                                            │
│ 延迟: ~200 cycles                                         │
├─────────────────────────────────────────────────────────┤
│              HBM3 (Global Memory)                         │
│ 80 GB, 5120-bit interface                                 │
│ 带宽: 3.35 TB/s                                           │
│ 延迟: ~400-600 cycles                                     │
└─────────────────────────────────────────────────────────┘
```

### 2.1 各层带宽利用率

```
以GEMM为例 (M=N=K=4096, FP16):
- 计算量: 2×M×N×K = 2×4096^3 = 137.4 GFLOP
- 数据量: (M×K + K×N + M×N) × 2B = 96 MB
- 计算密度: 137.4G / 96M = 1431 FLOP/Byte

需要的HBM带宽: 137.4T / 1431 = 96 GB/s (仅用HBM 3%!)
→ GEMM是compute-bound

以Softmax为例 (M=4096, N=4096):
- 计算量: ~5×M×N = 84 MFLOP (exp, sum, div)
- 数据量: 2×M×N×2B = 64 MB (读+写)
- 计算密度: 84M / 64M = 1.3 FLOP/Byte

需要的HBM带宽: 无法满足TC
→ Softmax是memory-bound
```

## 3. Roofline Model

### 3.1 H100 Roofline

```
            989 TFLOPS _______________
           /          |    FP16 peak  
          /           |
TFLOPS   /            |
        /             |
       /              |
      / 3.35 TB/s     |
     /    slope       |
    /                 |
   +------------------+---------→ Arithmetic Intensity (FLOP/B)
   0          295            ∞
        (ridge point)

Ridge Point = 989 TFLOPS / 3.35 TB/s = 295 FLOP/Byte

< 295 FLOP/B: Memory-bound (带宽受限)
> 295 FLOP/B: Compute-bound (算力受限)
```

### 3.2 常见Op分类

| 算子 | 计算密度 | 分类 | 优化重点 |
|------|---------|------|----------|
| GEMM (大) | >1000 | Compute-bound | TC利用率 |
| GEMM (小,batch) | 50-200 | Mixed | Launch开销 |
| LayerNorm | 2-5 | Memory-bound | 融合 |
| Softmax | 1-3 | Memory-bound | 在线算法 |
| Activation (GELU) | 1-2 | Memory-bound | 融合到GEMM |
| Embedding lookup | <1 | Memory-bound | 减少读取 |
| AllReduce | N/A | Communication-bound | 重叠 |

## 4. Memory-Bound Kernel优化策略

### 4.1 核心原则: 减少GMEM Touch

```
Kernel Fusion: 多个memory-bound op合并
优化前 (3个kernel, 3次读写):
  read X → LayerNorm → write Y₁
  read Y₁ → GELU → write Y₂  
  read Y₂ → Dropout → write Y₃
  总GMEM流量: 6 × N × sizeof(dtype)

优化后 (1个fused kernel):
  read X → LayerNorm → GELU → Dropout → write Y₃
  总GMEM流量: 2 × N × sizeof(dtype)
  节省: 66% GMEM带宽!
```

### 4.2 Vectorized Load/Store

```cpp
// 标准: 每thread加载4B
float val = input[tid];

// 向量化: 每thread加载16B (128-bit)
float4 val = reinterpret_cast<float4*>(input)[tid];

// WHY有效:
// 1. 减少memory transactions数量 (1条LDG.128 vs 4条LDG.32)
// 2. 更好利用cache line (128B)
// 3. 减少instruction issue压力

// FP16: 一次读8个元素
half8 val = *reinterpret_cast<half8*>(&input[tid * 8]);
```

### 4.3 Coalesced Access

```
全局内存以128B(32×4B) cache line为单位传输。

Coalesced (高效): warp中32个thread访问连续128B
Thread 0: addr[0]
Thread 1: addr[1]
...
Thread 31: addr[31]
→ 1次128B transaction

Non-coalesced (低效): warp中thread访问分散地址
Thread 0: addr[0]
Thread 1: addr[1000]
Thread 2: addr[2000]
→ 32次128B transaction (每次只用4B, 利用率3.1%!)
```

## 5. Compute-Bound Kernel优化策略

### 5.1 Occupancy vs Instruction-Level Parallelism

```
传统认知: 高occupancy = 好性能 (错误!)

GEMM的反例:
- Occupancy: 1-2 blocks/SM (很低!)
- 但性能接近峰值

WHY低occupancy GEMM仍高效:
1. 每thread有大量独立MMA指令 → ILP隐藏延迟
2. TMA pipeline提供多stage → 数据预取
3. WGMMA异步执行 → 计算和访存overlap
4. 不需要多block切换，减少context switch

关键insight: Latency hiding不只靠并发warps(TLP)，
还可以靠每个warp内的pipeline depth(ILP/MLP)
```

### 5.2 Register Pressure管理

```
每SM: 65536 registers
每thread最多: 255 registers

GEMM典型:
Accumulator 128×128 FP32: 128 regs/thread
Index/Counter: ~10 regs
Pipeline state: ~10 regs
Total: ~148 regs/thread

影响:
< 128 regs: 多block并发可能
128-192 regs: 通常1-2 blocks, 性能最优区间
> 192 regs: 可能spill到LMEM (灾难!)

WHY不能让compiler随意spill:
LMEM spill = GMEM读写 (通过L1 cache)
一次spill: ~100 cycles vs register: 1 cycle
```

## 6. L2 Cache优化

### 6.1 Persistent Kernel

```
// 非persistent: 每CTA处理一个tile然后退出
// 问题: CTA启动开销 + L2 cache thrash

// Persistent: 少量CTA常驻, 循环处理多个tile
__global__ void persistent_gemm(Args args) {
    int tile_id = blockIdx.x;
    int total_tiles = args.M_tiles * args.N_tiles;
    
    while (tile_id < total_tiles) {
        // 计算当前tile
        compute_tile(tile_id, args);
        
        // 获取下一个tile
        tile_id += gridDim.x;  // grid-stride loop
    }
}

// WHY更好:
// 1. L2中的数据可被复用(相邻tile共享A或B列)
// 2. 减少CTA launch开销
// 3. 更可预测的调度
```

### 6.2 Tile Traversal Order

```
标准行优先遍历:
Tile(0,0) Tile(0,1) Tile(0,2) ...
Tile(1,0) Tile(1,1) Tile(1,2) ...
→ 同行tile共享A的同一行块, 但B的列块每次变化

L形遍历(Swizzle tile ID):
Tile(0,0) Tile(1,0) Tile(0,1) Tile(1,1) ...
→ 2×2的块内，A和B的局部性都更好

CUTLASS的tile scheduler策略:
- Linear: 简单, L2利用率低
- Swizzle: 2D空间填充, L2利用率高
- Persistent: 最优, 但需要更多SMEM state
```

## 7. HBM带宽优化

### 7.1 Page与Channel

```
HBM3组织:
- 8 stacks × 16 channels/stack = 128 channels
- 每channel: 2 pseudo channels
- 每pseudo channel: 16B burst

理论峰值:
128 channels × 2 pseudo × 16B × 1.64 GHz = 3.35 TB/s

实际利用率取决于:
1. 地址均匀分布到所有channel (避免channel conflict)
2. 连续地址利用burst模式
3. 避免同时读写同一channel
```

### 7.2 为什么实际带宽<理论

```
理论: 3.35 TB/s
bandwidthTest 实测: ~3.0 TB/s (90%)
GEMM中实测: ~2.5 TB/s (75%)

损失原因:
- Page miss (~10%): 地址不连续导致page切换
- Read-write turnaround (~5%): 读写方向切换延迟
- Channel imbalance (~5%): 非均匀channel分布
- Refresh overhead (~3%): DRAM刷新周期
- Controller overhead (~2%): 排队/仲裁延迟
```

## 8. 量化: 减少数据搬运

### 8.1 精度与带宽

```
相同算子，不同精度的有效带宽提升:

FP32 (4B/elem): baseline
FP16/BF16 (2B/elem): 2× 有效带宽
FP8 (1B/elem): 4× 有效带宽
INT4 (0.5B/elem): 8× 有效带宽

对memory-bound kernels:
FP8 vs FP16 = 2× speedup (带宽翻倍)

对compute-bound GEMM:
FP8 vs FP16 = 2× speedup (TC吞吐翻倍)
→ FP8两头都赢!
```

## 9. Profiling带宽利用率

```bash
# DRAM带宽利用率:
ncu --metrics dram__bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum \
    --metrics dram__throughput.avg_pct_of_peak_sustained_elapsed \
    ./kernel

# L2 hit rate:
ncu --metrics lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum \
    --metrics lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum \
    ./kernel

# SMEM利用率:
ncu --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared.sum \
    --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
    ./kernel

# 带宽指标解读:
# < 60% peak: 优化空间大 (访问模式/vectorize/coalesce)
# 60-80% peak: 正常 (多数优化后kernel)
# > 80% peak: 接近极限 (考虑算法改进)
```

## 10. 总结

```
内存优化核心原则:
┌───────────────────────────────────────────────────┐
│ 1. 最小化数据搬运量 (Fusion, 量化)                 │
│ 2. 最大化带宽利用率 (Coalesce, Vectorize)         │
│ 3. 利用数据局部性 (Tiling, Cache-friendly)         │
│ 4. 用高层存储替代低层 (SMEM替代GMEM)              │
│ 5. 重叠计算与访存 (Pipeline, Async)               │
└───────────────────────────────────────────────────┘

不同层的优化收益:
GMEM→SMEM: 减少main loop GMEM load → 数量级提升
SMEM→RF: 消除bank conflict → 2-10× 
L2 reuse: Tile traversal → 10-30% 
Vectorize: LDG.128 → 10-20%
```
