# Chapter 04: CUTLASS 3.x GEMM架构 深度源码分析

## 1. 设计动机

**WHY CUTLASS**: cuBLAS是闭源的黑盒；当需要自定义epilogue、非标准数据类型、
或特殊tile形状时，CUTLASS提供了**可编程的GEMM模板库**。

**核心价值**:
- 开源：可以学习NVIDIA如何实现高性能GEMM
- 可扩展：通过模板参数定制每个阶段
- 接近cuBLAS性能：在H100上可达理论峰值95%+

## 2. CUTLASS 3.x vs 2.x架构对比

```
CUTLASS 2.x                          CUTLASS 3.x
┌──────────────────┐                 ┌──────────────────┐
│ threadblock_swizzle │               │ KernelSchedule     │
│ ThreadblockMma     │               │ CollectiveMma      │  ← CuTe取代
│ EpilogueOp         │               │ CollectiveEpilogue │
│ GemmKernel          │               │ GemmUniversal      │
└──────────────────┘                 └──────────────────┘

关键区别:
- 2.x: 手动计算index，ThreadMap管理内存映射
- 3.x: CuTe Layout统一描述，TMA硬件加载，WGMMA取代wmma
```

**WHY 3.x重写**: SM90(H100)引入TMA和WGMMA，手动管理已不可行；
CuTe的代数Layout自然表达了硬件能力。

## 3. 整体架构分层

```
┌────────────────────────────────────────────────────────┐
│                  GemmUniversal                           │
│  (gemm/kernel/sm90_gemm_tma_warpspecialized.hpp)        │
├────────────────────────────────────────────────────────┤
│  CollectiveMma          │  CollectiveEpilogue           │
│  (Mainloop: A×B→Accum)  │  (Accum→D with bias/act)     │
│  gemm/collective/       │  epilogue/collective/          │
│  sm90_mma_tma_gmma_     │  sm90_epilogue_tma_           │
│  ss_warpspecialized.hpp  │  warpspecialized.hpp          │
├────────────────────────────────────────────────────────┤
│  CuTe Primitives                                        │
│  TiledMma (WGMMA atom)   │  TiledCopy (TMA atom)       │
│  cute/atom/mma_sm90.hpp   │  cute/arch/copy_sm90_tma.hpp│
├────────────────────────────────────────────────────────┤
│  Pipeline                                               │
│  pipeline/sm90_pipeline_tma_async.hpp                    │
│  Producer-Consumer协调, AsyncBarrier                     │
└────────────────────────────────────────────────────────┘
```

> **源码**: `cutlass/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp` (584行)

## 4. Warp Specialization模型

### 4.1 设计原理

**WHY Warp Specialization**: H100的TMA引擎与Tensor Core可以完全并行工作。
将warp分为Producer(负责数据搬运)和Consumer(负责计算)，实现真正的双缓冲流水线。

```
Thread Block (128 threads = 4 warps)
┌─────────────────────────────────────────┐
│  Warp 0 (Producer)                       │
│  ─ 发起TMA copy: GMEM → SMEM            │
│  ─ 管理pipeline barrier                  │
├─────────────────────────────────────────┤
│  Warp 1,2,3 (Consumer = 1 Warpgroup)    │
│  ─ 等待SMEM数据就绪                      │
│  ─ 执行WGMMA: SMEM × SMEM → Registers   │
│  ─ 累加到fragment                        │
└─────────────────────────────────────────┘
```

### 4.2 源码中的角色分配

```cpp
// sm90_mma_tma_gmma_ss_warpspecialized.hpp 中:
// CollectiveMma::load() → Producer warp执行
// CollectiveMma::mma()  → Consumer warpgroup执行

// Pipeline模板参数
using MainloopPipeline = typename cutlass::PipelineTmaAsync<Stages>;
//                                          ↑ 多stage双缓冲
```

### 4.3 Pipeline阶段

```
时间 →
Producer:  [Load S0] [Load S1] [Load S2] [Load S0] ...
                ↓         ↓         ↓
SMEM:      [S0 ready] [S1 ready] [S2 ready] [S0 ready]
                ↓         ↓         ↓
Consumer:       [MMA S0]  [MMA S1]  [MMA S2]  [MMA S0]

Stages=3: 三级流水线，隐藏GMEM延迟(~400 cycles)
```

## 5. CollectiveMma模板剖析

### 5.1 模板参数

```cpp
// L55-78: 完整模板参数列表
template <
  int Stages,                // 流水线级数 (通常3-7)
  class ClusterShape,        // CTA cluster形状 (e.g., Shape<_2,_1,_1>)
  class KernelSchedule,      // 调度策略标签
  class TileShape_,          // CTA级tile (e.g., Shape<_128,_128,_64>)
  class ElementA_,           // A矩阵数据类型
  class StrideA_,            // A矩阵stride (CuTe Layout)
  class ElementB_,           // B矩阵数据类型
  class StrideB_,            // B矩阵stride
  class TiledMma_,           // WGMMA指令配置
  class GmemTiledCopyA_,     // TMA加载A的配置
  class SmemLayoutAtomA_,    // A在SMEM的layout atom
  class SmemCopyAtomA_,      // SMEM→RF的copy atom
  class TransformA_,         // A的变换 (如conjugate)
  class GmemTiledCopyB_,     // TMA加载B的配置
  class SmemLayoutAtomB_,    // B在SMEM的layout atom
  class SmemCopyAtomB_,      // SMEM→RF的copy atom
  class TransformB_>         // B的变换
```

**WHY如此多模板参数**: 每个参数控制一个正交维度，组合产生数千种kernel变体，
编译器在编译期生成最优代码路径。

### 5.2 TileShape语义

```
TileShape = Shape<_128, _128, _64>
             │      │      │
             │      │      └── K-tile: 每次MMA处理的K维度
             │      └────────── N-tile: 每个CTA负责的N维度
             └───────────────── M-tile: 每个CTA负责的M维度

一个CTA计算 C[128×128] 的子块，每次从K方向取64列做MMA
总K-loop迭代次数 = K / 64
```

## 6. 数据加载：TMA + Pipeline

### 6.1 TMA (Tensor Memory Accelerator)

```
传统加载:                    TMA加载:
Thread 0: load A[0][0]       TMA Engine: 
Thread 1: load A[0][1]       copy_desc → 一条指令搬运整个tile
Thread 2: load A[1][0]       ┌────────────┐
...                          │ GMEM tile  │ ──TMA──→ SMEM tile
(128个thread各发LDG)          └────────────┘    (硬件DMA)
```

**WHY TMA取代LDG**: 
- 1条TMA指令 = 128个LDG.128指令的效果
- 释放所有thread做其他工作（计算）
- 硬件处理bank conflict避免和地址计算

### 6.2 Pipeline Barrier机制

```cpp
// pipeline/sm90_pipeline_tma_async.hpp
// Producer端:
pipeline.producer_acquire(stage);    // 获取空buffer
copy(tma_load_a, ...);               // 发起TMA
copy(tma_load_b, ...);               // 发起TMA
pipeline.producer_commit(stage);     // 通知consumer

// Consumer端:
pipeline.consumer_wait(stage);       // 等待数据就绪
gemm(tiled_mma, ...);               // WGMMA执行
pipeline.consumer_release(stage);    // 释放buffer给producer
```

### 6.3 SMEM布局优化

```
SmemLayoutAtomA_ 控制A在共享内存中的排布:
- 使用swizzle避免bank conflict
- 对齐到128B (TMA最小传输粒度)
- 示例: SMEM_A layout = (128, 64) with Swizzle<3,4,3>

Bank conflict分析:
SMEM有32个bank，每bank 4字节
128×64×sizeof(half) = 16KB per stage
Swizzle使相邻thread访问不同bank
```

## 7. 计算：WGMMA指令

### 7.1 WGMMA vs HMMA

| 特性 | HMMA (SM80) | WGMMA (SM90) |
|------|-------------|--------------|
| 操作单位 | 1 warp (32 threads) | 1 warpgroup (128 threads) |
| 指令粒度 | 16×8×16 | 64×256×16 (最大) |
| 输入来源 | Registers | **Shared Memory直接** |
| Accumulator | Registers | Registers |
| 吞吐 | 256 FP16 ops/cycle | 1024 FP16 ops/cycle |

**WHY WGMMA更快**: 直接从SMEM读取操作数，省去SMEM→Register的搬运，
且操作粒度扩大4×，amortize指令发射开销。

### 7.2 TiledMma配置

```cpp
// CuTe中定义WGMMA atom:
using MmaAtom = SM90_64x128x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>;
//               │   │   │   │  │  │   │                       │
//               M   N   K   Acc A  B   Source=SharedMemory     Layout

using TiledMma = decltype(make_tiled_mma(MmaAtom{},
    Layout<Shape<_2,_1,_1>>{}));  // 2个atom在M方向tile
// → 总MMA shape: 128×128×16
```

### 7.3 K-loop主循环

```
for k_tile in range(K / K_TILE):
    consumer_wait(stage[k_tile % Stages])
    
    wgmma(accum, smem_A[stage], smem_B[stage])
    // 128×128×16 matmul累加到accum寄存器
    
    consumer_release(stage[k_tile % Stages])
    
// K-loop结束后, accum包含完整的C[128×128]
```

## 8. Epilogue: Accumulator → Output

### 8.1 CollectiveEpilogue职责

```
Accum (FP32, in registers)
    │
    ▼ scale + bias
    │
    ▼ activation (ReLU/GELU/SiLU)
    │
    ▼ type convert (FP32 → FP16/BF16/FP8)
    │
    ▼ store to GMEM (via TMA or STG)
    
D[m,n] = activation(alpha * Accum[m,n] + beta * C[m,n] + bias[n])
```

### 8.2 为什么Epilogue需要独立模块

**WHY分离**: 
- GEMM mainloop和epilogue的寄存器压力不同
- Epilogue可以复用mainloop释放的SMEM
- 不同应用需要不同epilogue（普通、fusion、reduction）

## 9. Cluster与多CTA协作

### 9.1 CTA Cluster (SM90新特性)

```
ClusterShape = Shape<_2, _1, _1>

┌─────────┐ ┌─────────┐
│  CTA 0  │ │  CTA 1  │   ← 2个CTA组成一个Cluster
│  SM  0  │ │  SM  1  │
│  SMEM 0 │ │  SMEM 1 │   ← 可以互相直接访问SMEM!
└─────────┘ └─────────┘
    ↕ Distributed SMEM ↕
```

**WHY Cluster**: CTA间SMEM直接访问(无需经过L2)，用于A/B矩阵的multicast：
- 同一行的cluster共享A tile
- 同一列的cluster共享B tile
- 减少重复GMEM读取

### 9.2 Multicast TMA

```
不使用Cluster:              使用Cluster (2×1):
CTA0 加载 A[0:128, :]      CTA0 加载 A[0:128, :]
CTA1 加载 A[0:128, :]      CTA1 通过distributed SMEM读取CTA0的A
↑ 重复读取!                 ↑ 带宽节省50%!
```

## 10. 性能模型

### 10.1 Roofline分析

```
H100 SXM5:
- 计算峰值: 989 TFLOPS (FP16 Tensor Core)
- HBM带宽: 3.35 TB/s
- 算术强度阈值: 989/3.35 = 295 ops/byte

GEMM算术强度 = 2MNK / (2(MK+KN+MN))  (FP16)
M=N=K=4096: AI = 2×4096³ / (2×3×4096²×2) = 1365 ops/byte
→ 远超阈值，计算bound

M=N=4096, K=64: AI = 2×4096²×64 / (2×(4096×64+64×4096+4096²)×2) ≈ 21
→ 低于阈值，带宽bound!
```

### 10.2 Tile Size选择指南

| M,N,K范围 | 推荐TileShape | 理由 |
|-----------|---------------|------|
| 大矩阵 (>2048) | 128×256×64 | 最大化Tensor Core利用 |
| 中矩阵 (512-2048) | 128×128×64 | 平衡occupancy |
| 小矩阵 (<512) | 64×64×64 | 增加wave数覆盖SM |
| Skinny (M small) | 64×256×64 | N方向展开 |

## 11. 实战：构建自定义GEMM

```cpp
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/collective/collective_builder.hpp>

using namespace cute;

// Step 1: 定义问题参数
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccum = float;

// Step 2: 用CollectiveBuilder自动选择最优配置
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,           // 目标架构
    cutlass::arch::OpClassTensorOp, // Tensor Core
    ElementA, cutlass::layout::RowMajor, 16,  // A: row-major, align 16B
    ElementB, cutlass::layout::ColumnMajor, 16,// B: col-major, align 16B
    ElementAccum,
    Shape<_128,_128,_64>,          // TileShape MNK
    Shape<_1,_1,_1>,              // ClusterShape
    cutlass::gemm::collective::StageCountAutoCarveout<
      sizeof(typename EpilogueOp::SharedStorage)>,  // 自动stage计算
    cutlass::gemm::KernelTmaWarpSpecialized  // 调度策略
>::CollectiveOp;

// Step 3: 组装完整Kernel
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,  // Problem shape MNK Batch
    CollectiveMainloop,
    CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

## 12. 总结

| 层次 | 组件 | 源码位置 |
|------|------|----------|
| Device | GemmUniversalAdapter | gemm/device/ |
| Kernel | GemmUniversal | gemm/kernel/ |
| Mainloop | CollectiveMma | gemm/collective/sm90_*.hpp |
| Epilogue | CollectiveEpilogue | epilogue/collective/ |
| Primitives | TiledMma/TiledCopy | cute/atom/ |
| Pipeline | PipelineTmaAsync | pipeline/ |

**核心设计思想**:
1. **Warp Specialization**: Producer搬数据，Consumer做计算，完全overlap
2. **TMA**: 硬件DMA取代软件Load，释放线程给计算
3. **WGMMA**: Warpgroup级MMA，直接从SMEM操作
4. **CuTe Layout**: 统一的数学抽象描述内存布局
5. **Pipeline**: 多stage异步流水线隐藏延迟
