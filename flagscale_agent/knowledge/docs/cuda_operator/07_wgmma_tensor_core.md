# Chapter 07: WGMMA与Tensor Core指令 深度分析

## 1. 设计动机

**WHY理解WGMMA**: WGMMA(Warpgroup Matrix Multiply-Accumulate)是H100 Tensor Core
的核心指令。写高性能GEMM/Attention kernel，必须理解其操作语义、寄存器映射、
和流水线行为。

**WHY不直接用cuBLAS**: 当需要自定义计算pattern(如FlashAttention的online softmax)，
cuBLAS无法满足，必须手动使用WGMMA构建kernel。

## 2. Tensor Core演进

```
Volta (SM70):  HMMA 8×8×4    → 第一代Tensor Core
Turing (SM75): HMMA 16×8×8   → 加INT8/INT4
Ampere (SM80): HMMA 16×8×16  → 加TF32/BF16
Hopper (SM90): WGMMA 64×N×K  → Warpgroup级, SMEM直接输入
Blackwell(SM100): WGMMA extended → FP4, 更大tile
```

### 2.1 关键跃迁: SM80→SM90

| 特性 | HMMA (SM80) | WGMMA (SM90) |
|------|-------------|--------------|
| 参与线程 | 32 (1 warp) | 128 (4 warps = 1 warpgroup) |
| 输入来源 | Registers only | **SMEM** (A and/or B) or RF |
| 最大shape | 16×8×16 (FP16) | 64×256×16 (FP16) |
| 累加器 | RF (same warp) | RF (distributed across warpgroup) |
| 异步性 | 同步执行 | **异步提交** (commit/wait分离) |
| 吞吐/SM/cycle | 512 FP16 ops | 2048 FP16 ops |

**WHY 4× throughput提升**:
- 4个warp→1个warpgroup: amortize指令fetch/decode开销
- SMEM直接读: 省去128条LDS指令 (RF load)
- 更大tile: 更好数据复用

## 3. WGMMA指令格式

### 3.1 PTX指令语法

```
wgmma.mma_async.sync.aligned
    .shape          // m64n128k16, m64n256k16, etc.
    .dtype_d        // f32, f16
    .dtype_a        // f16, bf16, e4m3, e5m2, tf32
    .dtype_b        // f16, bf16, e4m3, e5m2, tf32
    .input_a        // SS (shared), RS (register+shared)
    D, A_desc, B_desc, scale_D, imm_A, imm_B, ...;

// 示例:
wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16.ss
    {d0,...,d63},     // 64个accumulator寄存器
    desc_a,           // A的SMEM descriptor (64bit)
    desc_b,           // B的SMEM descriptor (64bit)
    1,                // scale_D=1 (累加)
    1, 1,             // imm flags
    0, 0;             // negate flags
```

### 3.2 支持的Shape

```
M固定=64:
┌────────────┬─────┬─────────────────────────┐
│ dtype A×B  │  K  │ 可选N值                  │
├────────────┼─────┼─────────────────────────┤
│ FP16×FP16  │ 16  │ 8,16,24,...,256 (步长8)  │
│ BF16×BF16  │ 16  │ 8,16,24,...,256          │
│ TF32×TF32  │  8  │ 8,16,24,...,256          │
│ E4M3×E4M3  │ 32  │ 8,16,24,...,256          │
│ E5M2×E5M2  │ 32  │ 8,16,24,...,256          │
│ E4M3×E5M2  │ 32  │ 8,16,24,...,256 (混合)   │
└────────────┴─────┴─────────────────────────┘

最大单条指令计算量:
FP16: 64×256×16×2 = 524,288 FP16 ops
FP8:  64×256×32×2 = 1,048,576 FP8 ops
```

### 3.3 Source模式: SS vs RS

```
SS模式: A从SMEM, B从SMEM (Shared×Shared)
- 优势: 寄存器压力最低
- 适用: 标准GEMM mainloop (A和B都预加载到SMEM)
- 约束: SMEM layout必须满足WGMMA对齐要求

RS模式: A从Register, B从SMEM (Register×Shared)
- 优势: A可以做运行时变换(如apply RoPE)
- 适用: Attention中Q×K^T (Q在RF中保持以做rescale)
- 约束: A占用寄存器，增加register pressure
```

## 4. SMEM Descriptor

### 4.1 Descriptor格式

```
WGMMA的SMEM输入通过64-bit descriptor指定:

bits[63:62]: leading_byte_offset[17:16]
bits[61:59]: stride_byte_offset[2:0] (×16B)
bits[58:56]: leading_byte_offset[15:14] + mode
bits[55:49]: start_addr[20:14] (128B aligned)
bits[48:32]: start_addr[13:4]
bits[31:16]: leading_byte_offset[13:0]
bits[15:14]: stride_dimension  
bits[13:4]:  base_offset
bits[3:0]:   swizzle_mode (0=none, 1=32B, 2=64B, 3=128B)
```

### 4.2 对SMEM Layout的要求

```
WGMMA要求SMEM数据满足:
1. 128B对齐的起始地址
2. 特定的swizzle模式(匹配descriptor中的swizzle_mode)
3. Leading dimension为swizzle_size的倍数

CuTe中自动满足:
SmemLayoutAtom = composition(
    Swizzle<3,4,3>{},          // 128B swizzle
    Layout<Shape<_8,_64>, Stride<_64,_1>>{}  // 8×64 atom
);
// → 自动生成满足WGMMA要求的SMEM layout
```

**WHY Swizzle对WGMMA关键**: WGMMA从SMEM批量读取128B行(一个warpgroup128thread各4B)，
若无swizzle则32个bank全被同一row访问 → 32-way conflict。

## 5. 异步执行模型

### 5.1 Commit-Wait分离

```
// WGMMA是异步提交的:
wgmma.mma_async ...;  // 只是提交，不等待完成
wgmma.mma_async ...;  // 可以连续提交多条
wgmma.mma_async ...;  // pipeline中的多次MMA

// 需要读取结果时:
wgmma.wait_group.sync.aligned N;
// 等待直到pipeline中还剩≤N条未完成的wgmma
// N=0: 等待全部完成

// 典型pattern:
for (k = 0; k < K_TILES; k++) {
    wgmma.mma_async ...;  // 提交第k次
}
wgmma.wait_group.sync.aligned 0;  // 等待全部完成
// 现在可以安全读取accumulator
```

### 5.2 Commit Group

```
// wgmma.commit_group 将之前的wgmma打包为一个group
wgmma.mma_async ...;
wgmma.mma_async ...;
wgmma.commit_group.sync.aligned;  // 这2条打包为group 0

wgmma.mma_async ...;
wgmma.commit_group.sync.aligned;  // 这1条打包为group 1

wgmma.wait_group.sync.aligned 1;  // 等待group 0完成(还剩≤1个group)

// WHY commit_group: 允许软件控制MMA pipeline深度
// 可以overlap不同stage的MMA计算
```

## 6. Accumulator寄存器映射

### 6.1 分布规则

```
WGMMA m64n128k16 输出 D[64×128] 的FP32累加器:
总元素: 64×128 = 8192 个 FP32
分布到: 128 threads (1 warpgroup)
每thread: 8192/128 = 64 个 FP32 寄存器

Thread t 持有的元素:
行: t % 64 对应的一部分
列: t / 64 对应的若干列

具体映射(FP16 m64n128):
Thread[t] holds D[row_set(t)][col_set(t)]
row_set: 由warp_id和lane_id共同决定
col_set: 由warp_id和MMA的N维度决定
```

### 6.2 寄存器压力计算

```
典型GEMM配置 (TileShape 128×128×64, FP16):
- Accumulator: 128×128 FP32 = 16384 values
- 分配到128 threads: 128 regs/thread
- 附加overhead (indices, counters): ~10 regs

总: ~138 regs/thread
Occupancy: 65536 / (138 × 32) = 14.8 warps → 14 warps (实际)

若TileShape 128×256×64:
- Accumulator: 128×256 FP32 = 32768 values
- Per thread: 256 regs (接近255上限!)
- → 需要register spilling或更小的tile

WHY GEMM不需要高occupancy: Pipeline隐藏延迟，不是靠多warp切换
```

## 7. CuTe中的MMA Atom

### 7.1 定义

```cpp
// cute/atom/mma_sm90.hpp
// 每个atom对应一条PTX wgmma指令:
struct SM90_64x128x16_F32F16F16_SS {
    // M=64, N=128, K=16, Acc=F32, A=F16, B=F16, Source=SS
    using DRegisters = float[64];  // per-thread accumulator
    using ARegisters = uint64_t[1]; // A descriptor (from SMEM)
    using BRegisters = uint64_t[1]; // B descriptor (from SMEM)
    
    CUTE_HOST_DEVICE static void
    fma(DRegisters& d, ARegisters const& a, BRegisters const& b, DRegisters const& c) {
        // 内部调用PTX asm:
        // wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16.ss
        //   {d[0],...,d[63]}, a[0], b[0], ...;
    }
};

// FP8版本:
struct SM90_64x128x32_F32E4M3E4M3_SS {
    // K=32 (FP8一次处理更多K)
    ...
};
```

### 7.2 TiledMma构建

```cpp
// 将多个atom tile起来覆盖更大区域:
using TiledMma = decltype(make_tiled_mma(
    SM90_64x128x16_F32F16F16_SS{},     // 基础atom
    Layout<Shape<_2,_1,_1>>{}           // M方向tile 2个
));
// 最终: 128×128×16 (每K-iteration)

// 使用:
auto tiled_mma = TiledMma{};
auto accum = partition_fragment_C(tiled_mma, Shape<_128,_128>{});
// accum: 每thread 128个FP32 regs

// 在K-loop中:
gemm(tiled_mma, sA(_, _, k), sB(_, _, k), accum);
// → 内部展开为2条wgmma.mma_async指令
```

## 8. FlashAttention中的WGMMA使用

```
FlashAttention用WGMMA做两次matmul:

1. S = Q × K^T  (attention scores)
   - Q在RF中(需要rescale), K^T在SMEM → RS模式
   - Shape per tile: (Br, d) × (d, Bc) = (Br, Bc)
   
2. O = P × V    (attention output)  
   - P在RF中(softmax output), V在SMEM → RS模式
   - Shape per tile: (Br, Bc) × (Bc, d) = (Br, d)

WHY RS模式: Q和P需要在RF中做online softmax rescale:
after S = Q×K^T:  S *= scale; m_new = max(m_old, rowmax(S))
after P = softmax(S): O = rescale(O_old) + P×V
这些操作必须在RF中完成，不能写回SMEM再重读
```

## 9. 性能调优要点

| 要点 | 说明 | 影响 |
|------|------|------|
| 选择N维度 | N=128或256，匹配TC硬件 | 直接影响TC利用率 |
| K-unroll | 每个stage内多次wgmma | 摊分pipeline overhead |
| Accumulator type | FP32 (精度) vs FP16 (省regs) | 精度vs occupancy |
| Fence placement | wgmma.wait位置 | 过早wait浪费overlap |
| Mixed precision | E4M3×E5M2 | training BWD常用 |

## 10. 总结

```
WGMMA是H100 GEMM性能的根基:
- 单指令计算64×256×16=524K FP16 ops
- 直接从SMEM读取，消除RF load开销
- 异步执行，与TMA pipeline完美配合
- CuTe通过Atom/TiledMma抽象，使用简洁

要写高性能kernel:
1. 选合适的Atom shape (匹配问题规模)
2. 保证SMEM layout满足descriptor要求 (Swizzle)
3. 利用异步性overlap多个K-iteration
4. 管理好register pressure (不超255)
5. 在正确位置放fence (wait_group)
```
