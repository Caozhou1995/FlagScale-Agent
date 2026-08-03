# Chapter 05: CuTe Layout代数系统 深度分析

## 1. 设计动机

**WHY CuTe**: CUDA kernel中最容易出错的部分是**索引计算**——thread到数据的映射、
shared memory bank conflict避免、TMA地址计算等。CuTe用**代数Layout**统一描述所有映射，
让编译器在编译期完成正确性验证。

**核心洞察**: 任何多维数组访问都可以表示为 `offset = f(coordinate)`，
CuTe将这个函数`f`表示为 `Layout = (Shape, Stride)` 的代数对象。

## 2. Layout基础

### 2.1 定义

```cpp
// cute/layout.hpp L49-57
template <class... Shapes>  using Shape  = cute::tuple<Shapes...>;
template <class... Strides> using Stride = cute::tuple<Strides...>;

// Layout = (Shape, Stride) pair
Layout layout = make_layout(Shape<_4, _8>{}, Stride<_8, _1>{});
//              shape=(4,8)  stride=(8,1)  → 4×8 col-major矩阵
```

### 2.2 索引计算

```
Layout(shape=(4,8), stride=(8,1))

逻辑坐标 (i, j) → 物理偏移 = i*8 + j*1

示例:
(0,0)→0  (0,1)→1  (0,2)→2  ...  (0,7)→7
(1,0)→8  (1,1)→9  (1,2)→10 ...  (1,7)→15
(2,0)→16 (2,1)→17 ...
(3,0)→24 (3,1)→25 ...

这就是一个4行8列的col-major矩阵!
```

### 2.3 Row-major vs Col-major

```
Col-major: Layout(Shape<M,N>{}, Stride<_1, M>{})  → offset = i + j*M
Row-major: Layout(Shape<M,N>{}, Stride<N, _1>{})  → offset = i*N + j

区别仅在Stride，Shape不变! 这就是CuTe的优雅之处。
```

## 3. 层次化Layout (Hierarchical)

### 3.1 嵌套Shape

**WHY嵌套**: GPU的线程层次(Grid→Block→Warp→Thread)天然是层次化的，
嵌套Layout直接映射这种结构。

```cpp
// Thread层次化layout
// 8个thread，每个thread处理4个元素，总32元素
Layout thread_layout = make_layout(
    Shape<Shape<_4>, Shape<_8>>{},    // (elements_per_thread, num_threads)
    Stride<Stride<_1>, Stride<_4>>{}  // thread内连续，thread间间隔4
);

// 物理排布:
// Thread 0: elem 0,1,2,3
// Thread 1: elem 4,5,6,7
// Thread 2: elem 8,9,10,11
// ...
```

### 3.2 Composition (组合)

```cpp
// 两个Layout组合
auto composed = composition(layout_A, layout_B);
// 等价于 composed(x) = layout_A(layout_B(x))
// 先用B映射坐标，再用A映射到物理地址
```

## 4. Swizzle: Bank Conflict消除

### 4.1 问题

```
Shared Memory: 32 banks, 4 bytes/bank
如果32个thread同时访问同一bank的不同地址 → Bank Conflict!

标准col-major layout:
Thread 0 → addr 0  (bank 0)
Thread 1 → addr 4  (bank 1)  
...
Thread 31 → addr 124 (bank 31) ✓ 无冲突

但如果stride=32*4=128:
Thread 0 → addr 0   (bank 0)
Thread 1 → addr 128 (bank 0)  ← 冲突!
```

### 4.2 Swizzle解决方案

```cpp
// cute/swizzle.hpp
// Swizzle<B, M, S> 对地址做bit-level变换
// addr_new = addr XOR ((addr >> S) & mask)

// 典型配置: Swizzle<3,4,3>
// 取addr的bit[6:4] (3位) XOR到 bit[9:7]
// 效果: 打乱bank映射，消除规律性冲突
```

```
不使用Swizzle (32-way conflict):         使用Swizzle<3,4,3>:
Bank: 0 0 0 0 0 0 0 0 ...               Bank: 0 1 2 3 4 5 6 7 ...
      ↑ 所有thread命中同一bank                  ↑ 均匀分布到不同bank
```

**WHY XOR而不是其他变换**: XOR是自逆的(XOR两次恢复原值)，
且只需简单位运算，零性能开销。

## 5. TiledCopy: 数据搬运抽象

### 5.1 Copy Atom

```cpp
// Copy Atom定义最小搬运单位
// TMA: 一次搬运一个tile
using CopyAtom = SM90_TMA_LOAD;  // TMA硬件加载

// 传统LDG: 一个thread加载128bit
using CopyAtom = SM90_U32x4_LDSM_N;  // SMEM→RF, 4×32bit

// Tiled: 多个thread协作
using TiledCopy = make_tiled_copy(
    CopyAtom{},
    Layout<Shape<_32, _4>>{},   // thread layout
    Layout<Shape<_4, _1>>{}     // value layout per thread
);
// 32×4=128 threads, 每thread 4×1个值, 总搬运 128×4 = 512个元素
```

### 5.2 TMA Copy

```cpp
// cute/arch/copy_sm90_tma.hpp
// TMA_LOAD: 全自动，只需descriptor和坐标
// 1条PTX指令: cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes

// 使用方式:
auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, tensor_A, smem_layout);
copy(tma_load, gA_partition, sA_partition);  // 一行代码完成2D tile加载
```

## 6. TiledMma: 计算抽象

### 6.1 MMA Atom

```cpp
// cute/atom/mma_sm90.hpp
// 定义WGMMA最小计算单位
using MmaAtom = SM90_64x128x16_F32F16F16_SS;
//               M   N    K  Acc  A   B  Source(SS=SMEM×SMEM)

// 含义: 64×128×16的matmul，输入FP16从SMEM，输出FP32到RF
// 由1个warpgroup(128 threads)协作完成
```

### 6.2 MMA指令映射

```
PTX指令: wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16

一条指令完成:
C[64×128] += A[64×16] × B[16×128]
= 64×128×16×2 = 262,144 FP16 ops
每个SM每个cycle发射一条 → 262,144 ops/cycle

H100有132个SM → 132 × 262,144 × 1.83GHz ≈ 63 TFLOPS (per wgmma)
```

### 6.3 Tiling多个Atom

```cpp
// 用多个atom tile覆盖更大区域
using TiledMma = make_tiled_mma(
    SM90_64x128x16_F32F16F16_SS{},
    Layout<Shape<_2, _1, _1>>{}   // M方向2个atom
);
// 总计算: 128×128×16 per iteration
```

## 7. 完整GEMM数据流

```
┌─ Global Memory ──────────────────────────────────────┐
│  A[M,K] (HBM)               B[K,N] (HBM)            │
└──────────────────┬───────────────────────┬───────────┘
                   │ TMA (Producer warp)   │
                   ▼                       ▼
┌─ Shared Memory (per CTA, 228KB on H100) ────────────┐
│  sA[128,64] (stage 0)   sB[64,128] (stage 0)        │
│  sA[128,64] (stage 1)   sB[64,128] (stage 1)        │
│  sA[128,64] (stage 2)   sB[64,128] (stage 2)        │
│         ↑ Swizzled layout避免bank conflict           │
└──────────────────┬───────────────────────┬───────────┘
                   │ WGMMA (Consumer warpgroup)
                   ▼
┌─ Register File (per warpgroup) ─────────────────────┐
│  accum[128,128] in FP32 (256 registers/thread)      │
│  K-loop累加: accum += sA × sB                       │
└──────────────────────────────────┬───────────────────┘
                                   │ Epilogue
                                   ▼
┌─ Global Memory ──────────────────────────────────────┐
│  D[M,N] = f(alpha*accum + beta*C + bias)             │
└──────────────────────────────────────────────────────┘
```

## 8. 性能关键参数

| 参数 | 影响 | H100推荐值 |
|------|------|------------|
| Stages | 隐藏延迟能力 | 3-5 (受SMEM限制) |
| TileShape M | Wave效率 | 128 or 256 |
| TileShape N | 输出带宽 | 128 or 256 |
| TileShape K | MMA利用率 | 64 (FP16) or 32 (FP8) |
| ClusterShape | 多播效率 | (2,1,1) or (1,2,1) |

## 9. 总结

CuTe的核心思想是**用代数描述硬件行为**:
1. **Layout = (Shape, Stride)**: 统一描述任何内存映射
2. **Swizzle**: 编译期消除bank conflict，零运行时开销
3. **Atom**: 最小硬件操作单位（TMA copy / WGMMA mma）
4. **Tiled**: 多个atom协作覆盖CTA级tile
5. **Hierarchical**: 自然映射GPU的线程/内存层次
