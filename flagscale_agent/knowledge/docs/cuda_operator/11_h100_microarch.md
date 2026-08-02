# Chapter 11: H100 SM微架构 深度分析

## 1. 设计动机

**WHY理解微架构**: 写高性能CUDA算子必须理解硬件约束——
寄存器数量决定tile大小、SMEM容量决定pipeline深度、Tensor Core形状决定MMA指令选择。
不理解这些，写出的kernel必然次优。

## 2. H100 SXM5 全局规格

```
┌─────────────────────────────────────────────────────┐
│  NVIDIA H100 SXM5 (GH100 GPU)                       │
├─────────────────────────────────────────────────────┤
│  SM Count:        132                                │
│  FP16 Tensor:     989.4 TFLOPS                      │
│  FP8 Tensor:      1978.9 TFLOPS                     │
│  FP32 (non-TC):   66.9 TFLOPS                       │
│  HBM3 Bandwidth:  3.35 TB/s                         │
│  HBM3 Capacity:   80 GB                             │
│  L2 Cache:        50 MB                             │
│  SMEM/SM:         228 KB (configurable)              │
│  Registers/SM:    65536 × 32-bit                     │
│  Clock:           1830 MHz (boost)                   │
│  TDP:             700W                               │
└─────────────────────────────────────────────────────┘
```

## 3. SM内部结构

### 3.1 SM分区

```
┌─ H100 Streaming Multiprocessor (SM) ────────────────┐
│                                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │  Sub-    │ │  Sub-    │ │  Sub-    │ │  Sub-  │ │
│  │  Core 0  │ │  Core 1  │ │  Core 2  │ │  Core 3│ │
│  │          │ │          │ │          │ │        │ │
│  │ Warp Sch │ │ Warp Sch │ │ Warp Sch │ │Warp Sch│ │
│  │ 16384 RF │ │ 16384 RF │ │ 16384 RF │ │16384 RF│ │
│  │ FP32×32  │ │ FP32×32  │ │ FP32×32  │ │FP32×32 │ │
│  │ INT32×16 │ │ INT32×16 │ │ INT32×16 │ │INT32×16│ │
│  │ FP64×16  │ │ FP64×16  │ │ FP64×16  │ │FP64×16 │ │
│  │ TC×1     │ │ TC×1     │ │ TC×1     │ │ TC×1   │ │
│  │ LSU×16   │ │ LSU×16   │ │ LSU×16   │ │LSU×16  │ │
│  │ SFU×4    │ │ SFU×4    │ │ SFU×4    │ │ SFU×4  │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────┘ │
│                                                      │
│  ┌─── Shared Memory / L1 Cache ──────────────────┐  │
│  │  228 KB (configurable split)                   │  │
│  └───────────────────────────────────────────────┘  │
│                                                      │
│  ┌─── TMA Engine ────────────────────────────────┐  │
│  │  Asynchronous Tensor Memory Access unit        │  │
│  └───────────────────────────────────────────────┘  │
│                                                      │
│  ┌─── Tex/L1 ────────────────────────────────────┐  │
│  │  Texture/Surface path + L1 cache               │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### 3.2 4th Gen Tensor Core

```
每个SM有4个Tensor Core (每个sub-core 1个)

单个Tensor Core per cycle:
┌────────────────────────────────────────┐
│  FP16:  256 FMA ops = 512 FP16 ops    │
│  BF16:  256 FMA ops = 512 BF16 ops    │
│  TF32:  128 FMA ops = 256 TF32 ops    │
│  FP8:   512 FMA ops = 1024 FP8 ops    │
│  INT8:  512 ops = 1024 INT8 ops       │
└────────────────────────────────────────┘

4 TC × 512 FP16 ops/cycle × 132 SM × 1830 MHz
= 4 × 512 × 132 × 1.83 GHz = 989 TFLOPS ✓
```

**WHY FP8翻倍**: FP8每个元素仅8bit(vs FP16的16bit)，同样的数据通路
可以放双倍操作数，吞吐翻倍。

### 3.3 Warpgroup MMA (WGMMA)

```
WGMMA (SM90新增): 4个warp(128 threads)协作发射一条mma指令

传统HMMA (SM80):      WGMMA (SM90):
1 warp → 16×8×16      4 warps → 64×256×16 (最大)
每cycle 512 FP16 ops  每cycle 2048 FP16 ops (单SM)

但WGMMA从SMEM直接读取A/B操作数！
不需要先load到register再做mma。
```

## 4. 内存层级

### 4.1 层级图

```
                    容量          带宽           延迟
┌─────────┐
│ Register │  256KB/SM      ∞ (same cycle)    0 cycle
├─────────┤    (65536×32b)
│  SMEM   │  228KB/SM      ~19.4 TB/s/SM     ~20-30 cycles
├─────────┤               (32 banks × 4B × 1830MHz)
│ L1/Tex  │  256KB/SM      ~12.7 TB/s/SM     ~30-40 cycles
├─────────┤    (unified with SMEM)
│   L2    │  50MB (全局)   ~12 TB/s (全局)    ~200 cycles  
├─────────┤
│  HBM3   │  80GB          3.35 TB/s          ~400-600 cycles
└─────────┘
```

### 4.2 Bandwidth计算

```
Per-SM SMEM bandwidth:
32 banks × 4 bytes × 1830 MHz = 234.2 GB/s per SM
× 132 SM = 30.9 TB/s (全GPU SMEM聚合带宽)

但实际受限于SMEM容量和bank conflict。

HBM → L2: 3.35 TB/s
L2 → SMEM: 受NoC带宽限制 (~12 TB/s estimate)
SMEM → RF: 受bank数×port数限制
```

### 4.3 SMEM配置

```
H100 SMEM/L1 总pool = 256KB per SM (228KB可配置为SMEM)

配置选项:
┌─────────────────────────────────────┐
│ SMEM     │ L1 Data Cache            │
├──────────┼──────────────────────────┤
│ 228KB    │ 28KB                     │ ← GEMM推荐
│ 192KB    │ 64KB                     │
│ 128KB    │ 128KB                    │ ← 访存密集推荐
│ 64KB     │ 192KB                    │
│ 32KB     │ 224KB                    │
└──────────┴──────────────────────────┘

cudaFuncSetAttribute(kernel, 
    cudaFuncAttributeMaxDynamicSharedMemorySize, 228*1024);
```

**WHY GEMM用最大SMEM**: 更多SMEM = 更多pipeline stages = 更好隐藏HBM延迟。

## 5. TMA Engine (Tensor Memory Accelerator)

### 5.1 硬件能力

```
TMA是SM内的专用DMA引擎:
- 支持1D/2D/3D/4D/5D tensor descriptor
- 硬件计算地址（不需thread参与）
- 自动处理out-of-bound (填zero或clamp)
- 支持multicast到cluster内多个SM的SMEM
- 与mbarrier集成实现异步同步

单次TMA传输: 128B对齐，最大传输16×256B = 4KB
```

### 5.2 TMA vs 传统LDG

| 指标 | LDG.128 | TMA |
|------|---------|-----|
| 每次传输 | 16B/thread | 整个tile (4KB+) |
| 需要线程 | 全部(128 threads) | 1 thread发起 |
| 地址计算 | 每thread算自己 | 硬件descriptor |
| OOB处理 | 需要if判断 | 硬件自动 |
| 带宽利用 | 可能有waste | 最优 |

## 6. 指令流水线

### 6.1 Warp Scheduler

```
每个sub-core有1个warp scheduler:
- 管理最多16个warp (每sub-core)
- 每个cycle选择一个eligible warp发射指令
- SM共4个scheduler → 同时4条指令

指令延迟与吞吐:
┌──────────┬────────┬───────────┐
│ 指令类型  │ 延迟    │ 吞吐      │
├──────────┼────────┼───────────┤
│ FP32 ADD │ 4 cyc  │ 32/SM/cyc │
│ FP16 MUL │ 4 cyc  │ 64/SM/cyc │
│ SMEM LDS │ 20 cyc │ 128B/cyc  │
│ WGMMA    │ varies │ 异步       │
│ TMA      │ 异步    │ 硬件DMA   │
└──────────┴────────┴───────────┘
```

### 6.2 Occupancy与寄存器压力

```
每SM总register: 65536 × 32-bit
每thread最大: 255 registers

Occupancy = active_warps / max_warps_per_SM

示例:
kernel用128 registers/thread:
- 一个warp: 32 threads × 128 regs = 4096 regs
- SM max warps: 65536 / 4096 = 16 warps = 50% occupancy

kernel用64 registers/thread:
- 一个warp: 32 × 64 = 2048 regs
- SM max warps: 65536 / 2048 = 32 warps → capped at 64 = 100%

**但GEMM不需要高occupancy!**
Warp specialization用pipeline隐藏延迟，而非高occupancy。
```

## 7. NVLink与多GPU

```
H100 NVLink 4.0:
- 18条NVLink链路/GPU
- 每链路: 25 GB/s × 2(双向) = 50 GB/s
- 总带宽: 900 GB/s (全双工)
- NVSwitch: 全连接拓扑(8 GPU all-to-all)

对算子优化的影响:
- AllReduce用NVLink而非PCIe
- 大模型TP切分后GEMM变小，需考虑通信overlap
- Ring AllReduce: 理论带宽利用率 (N-1)/N × 900 GB/s
```

## 8. Async Architecture总结

```
H100的核心设计哲学: 异步一切

┌──────────────────────────────────────────┐
│           Async Data Movement            │
│  TMA (GMEM→SMEM)  ──→  异步，不阻塞thread │
│  cp.async (GMEM→SMEM)  ──→  异步流水线      │
│  Bulk Copy  ──→  cluster内SMEM互拷         │
├──────────────────────────────────────────┤
│           Async Compute                   │
│  WGMMA  ──→  异步提交，fence后再读结果      │
│  Warp Specialization  ──→ 搬运和计算并行    │
├──────────────────────────────────────────┤
│           Async Synchronization           │
│  mbarrier  ──→  硬件计数barrier           │
│  Cluster barrier  ──→  cross-CTA同步      │
└──────────────────────────────────────────┘
```

## 9. 对算子开发的指导意义

| 硬件约束 | 对kernel的影响 | 优化方向 |
|----------|---------------|----------|
| SMEM 228KB | Tile size × Stages ≤ 228KB | 平衡tile和pipeline |
| RF 255/thread | Accumulator大小受限 | 选合适MNK tile |
| TC 4个/SM | WGMMA占满4个TC | 用满warpgroup |
| HBM 3.35TB/s | 大矩阵compute-bound | 增大arithmetic intensity |
| L2 50MB | 小矩阵可能cache | L2 residency control |
| TMA 4KB/op | 选合适tile对齐 | 128B aligned tiles |

## 10. 关键PTX指令参考

```
// WGMMA (Tensor Core)
wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
wgmma.mma_async.sync.aligned.m64n256k32.f32.e4m3.e4m3  // FP8

// TMA
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes

// Barrier
mbarrier.init.shared.b64
mbarrier.arrive.shared.b64
mbarrier.try_wait.shared.b64

// Cluster
barrier.cluster.arrive
barrier.cluster.wait

// Shared Memory
stmatrix.sync.aligned.m8n8.x4.shared.b16  // RF→SMEM
ldmatrix.sync.aligned.m8n8.x4.shared.b16  // SMEM→RF
```
