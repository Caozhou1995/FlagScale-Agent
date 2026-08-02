# Chapter 08: Shared Memory与Bank Conflict 深度分析

## 1. 设计动机

**WHY理解Bank Conflict**: Shared Memory是CUDA kernel中最关键的on-chip buffer。
Bank Conflict是SMEM性能的头号杀手——一个32-way conflict让带宽退化32×。
高性能GEMM/Attention必须保证SMEM访问无冲突。

## 2. SMEM硬件结构

### 2.1 Bank组织

```
H100 Shared Memory:
- 228 KB per SM (最大配置)
- 32 banks
- 每bank宽度: 4 bytes (32 bits)
- 每bank深度: 228KB / 32 = 7.125 KB
- 访问延迟: ~20 cycles (无conflict)

地址到bank映射:
bank_id = (byte_address / 4) % 32

addr  0-3   → bank 0
addr  4-7   → bank 1
addr  8-11  → bank 2
...
addr 124-127 → bank 31
addr 128-131 → bank 0  (wrap around)
```

### 2.2 带宽模型

```
无conflict: 32 banks × 4B × 1 read/cycle = 128 B/cycle
= 128 × 1.83 GHz = 234 GB/s per SM

有N-way conflict: 带宽 / N
32-way conflict: 234 / 32 = 7.3 GB/s  ← 灾难性!

总GPU SMEM带宽 (无conflict):
234 GB/s × 132 SM = 30.9 TB/s
(是HBM 3.35 TB/s的9.2×!)
```

## 3. Bank Conflict分析

### 3.1 常见场景

```
场景1: 32 threads读同一列(col-major矩阵)
float A[32][33];  // 32×33, col-major
// Thread i reads A[i][0]
// addr(A[i][0]) = i × 33 × 4
// bank(A[i][0]) = (i × 33) % 32
// i=0: bank 0, i=1: bank 1, ... → NO conflict (33互素于32!)

float A[32][32];  // 32×32
// bank(A[i][0]) = (i × 32) % 32 = 0 for all i
// → 32-way conflict! 所有thread访问bank 0!
```

**WHY padding(+1)有效**: stride从32变33，33和32互素，保证映射均匀分布。

### 3.2 GEMM中的典型conflict

```
SMEM tile: float A_smem[TILE_M][TILE_K];  // 128×64

加载时(GMEM→SMEM): 通常无conflict (连续地址)
使用时(SMEM→RF for MMA): 看access pattern

WMMA/WGMMA的access pattern:
一个warp的32个thread同时读取A的一个"片段":
- 行方向: 连续的16个元素(16×2B=32B)
- 列方向: 2行

如果 TILE_K=64, stride=64×2=128 bytes:
行间跳跃128B = 32 banks正好对齐 → conflict!

解决: TILE_K=64但stride=66 (padding) 或 Swizzle
```

### 3.3 Conflict类型

| 类型 | 说明 | 解决 |
|------|------|------|
| N-way conflict | N个thread访问同bank不同地址 | Swizzle/Padding |
| Broadcast | 多个thread访问同bank同地址 | 免费!(硬件广播) |
| No conflict | 每thread访问不同bank | 最优 |

## 4. Swizzle详解

### 4.1 原理

```
Swizzle通过XOR变换打乱bank映射:

原始地址: addr
Swizzle后: addr_new = addr XOR ((addr >> shift) & mask)

CuTe Swizzle<B,M,S>:
- B: mask位数 (2^B个bank被swizzle)
- M: 起始bit位置
- S: 偏移bit位置

Swizzle<3,4,3> 含义:
mask = 0b111 (3位)
取 addr bits[6:4] (位置M=4,M+B-1=6)
XOR到 addr bits[3+3-1:3] = bits[5:3]  (位置S=3)

效果: 将行号的低3位XOR到列号的低3位
→ 相邻行映射到不同bank组
```

### 4.2 可视化

```
不使用Swizzle (32×32 FP16, stride=32):
Row 0: bank 0,1,2,3,...,31
Row 1: bank 0,1,2,3,...,31  ← 同列=同bank!
Row 2: bank 0,1,2,3,...,31
...
列方向读取 → 32-way conflict

使用Swizzle<3,4,3> (128B粒度):
Row 0: bank 0,1,2,3,4,5,6,7, 8,9,...
Row 1: bank 4,5,6,7,0,1,2,3, 12,13,... (XOR了row[2:0]=001)
Row 2: bank 0,1,2,3,4,5,6,7, ...       (XOR了row[2:0]=010)
Row 3: bank 4,5,6,7,0,1,2,3, ...
...
列方向读取 → 最多4-way conflict (降低8×)
```

### 4.3 为什么Swizzle优于Padding

| 方法 | SMEM浪费 | 代码复杂度 | 适用范围 |
|------|----------|-----------|----------|
| Padding (+1) | 3-6% | 简单 | 小tile |
| Swizzle | 0% | 复杂(CuTe自动) | 大tile/WGMMA |
| Permuted layout | 0% | 中等 | 特定pattern |

**WHY WGMMA必须Swizzle**: WGMMA的descriptor要求特定swizzle模式(32B/64B/128B)，
padding不被支持。

## 5. SMEM大小与配置

### 5.1 动态SMEM分配

```cpp
// 超过48KB需要显式设置:
__global__ void kernel() {
    extern __shared__ char smem[];
    // 使用smem[0..size-1]
}

// Launch时设置:
int smem_size = 228 * 1024;  // 228KB
cudaFuncSetAttribute(kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

kernel<<<grid, block, smem_size, stream>>>();
```

### 5.2 SMEM vs L1 Trade-off

```
总pool: 256KB per SM (H100)
分配: SMEM + L1 = 256KB

GEMM (计算密集): 最大化SMEM → 更多pipeline stages
Reduction (访存密集): 最大化L1 → 更好cache

cudaFuncSetAttribute(kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxL1);  // 或 MaxShared
```

## 6. Async Copy: GMEM→SMEM

### 6.1 cp.async vs 传统LDG+STS

```
传统方式:
LDG.128 R0, [gmem_addr]  // GMEM → Register (400+ cycles)
STS.128 [smem_addr], R0  // Register → SMEM

cp.async方式:
cp.async.ca.shared.global [smem_addr], [gmem_addr], 16
// GMEM → SMEM 直接搬运，不经过RF!
// 释放寄存器给计算使用
// 延迟与LDG相同，但不阻塞线程

// 同步:
cp.async.commit_group;     // 打包为一组
cp.async.wait_group N;     // 等待直到≤N组未完成
```

### 6.2 TMA vs cp.async

| 特性 | cp.async | TMA |
|------|----------|-----|
| 发起者 | 每thread独立 | 单thread(或硬件) |
| 粒度 | 4-16 bytes/thread | 整个tile (4KB+) |
| 地址计算 | 每thread计算 | descriptor一次性 |
| OOB处理 | 需要mask | 硬件自动 |
| 同步 | wait_group | mbarrier |
| 适用 | SM80+(A100) | SM90+(H100) |

## 7. SMEM Atomic操作

```
H100 SMEM支持原子操作:
atomicAdd.shared   // 原子加
atomicCAS.shared   // 比较交换

用途:
- Reduction in shared memory
- Histogram accumulation  
- Lock-free data structures

性能: ~10 cycles/op (vs global atomic ~100-1000 cycles)
限制: 仍有bank conflict问题(同bank原子操作串行)
```

## 8. 实战: 最优SMEM Layout设计

### 8.1 GEMM SMEM设计原则

```
目标: 128×64 FP16 tile in SMEM

需求分析:
1. TMA加载: 需要128B对齐, 2D tile
2. WGMMA消费: 需要Swizzle<3,4,3>(128B swizzle)
3. 多stage: N个完整tile (N × 128×64×2B = N × 16KB)

Layout:
SmemLayoutA = composition(
    Swizzle<3,4,3>{},
    make_layout(
        make_shape(Int<128>{}, Int<64>{}),
        make_stride(Int<64>{}, Int<1>{})
    )
);
// 物理大小: 128×64×2 = 16384 bytes per stage
// 起始地址: 128B对齐 (TMA要求)
```

### 8.2 Double Buffer vs Multi-stage

```
Double Buffer (2 stages):
┌──────────┐ ┌──────────┐
│ Stage 0  │ │ Stage 1  │
│ 16KB A   │ │ 16KB A   │
│ 16KB B   │ │ 16KB B   │
└──────────┘ └──────────┘
SMEM used: 64KB
隐藏延迟: 有限

Multi-stage (5 stages):
┌────┐┌────┐┌────┐┌────┐┌────┐
│ S0 ││ S1 ││ S2 ││ S3 ││ S4 │
│32KB││32KB││32KB││32KB││32KB│
└────┘└────┘└────┘└────┘└────┘
SMEM used: 160KB
隐藏延迟: 充分 (5× pipeline depth)
```

## 9. Bank Conflict检测工具

```bash
# NSight Compute检测bank conflict:
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared \
    ./my_kernel

# 关键metric:
# l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum  → Load冲突
# l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum  → Store冲突

# 理想值: 0
# 可接受: < 10% of total transactions
# 需优化: > 20%

# 也可用CUDA-MEMCHECK:
compute-sanitizer --tool memcheck --report-api-errors all ./kernel
```

## 10. 总结

```
SMEM性能优化清单:
□ 使用Swizzle消除bank conflict (WGMMA必须)
□ 128B对齐 (TMA/WGMMA要求)
□ 最大化SMEM配置 (GEMM场景)
□ 多stage pipeline (3-7 stages)
□ 用cp.async/TMA而非LDG+STS
□ ncu验证conflict = 0
□ 计算SMEM budget: stages × (A_tile + B_tile) ≤ 228KB
```
