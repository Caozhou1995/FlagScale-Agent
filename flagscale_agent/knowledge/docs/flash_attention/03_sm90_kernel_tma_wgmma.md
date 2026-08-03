# FlashAttention 源码深度分析 — 第3章：SM90 Kernel 与 TMA/WGMMA

## 1. 设计动机

### 1.1 为什么 Hopper 架构需要专门的 Kernel？

**WHY 不复用 SM80 (A100) 代码？** Hopper (H100) 引入了三大硬件特性，若不利用则浪费 2-3× 性能：

| 硬件特性 | SM80 (A100) | SM90 (H100) | 性能影响 |
|---------|-------------|-------------|---------|
| 数据搬运 | 手动 cp.async | TMA (硬件引擎) | 释放寄存器+减少指令 |
| 矩阵乘 | HMMA (Warp级) | WGMMA (Warpgroup级) | 128线程协同, 2× MMA吞吐 |
| 跨SM协作 | 无 | Cluster + DSMEM | 减少 HBM 读取 |
| Kernel 调度 | 一次性 | Persistent Thread | 消除launch开销 |
| 预取 | 软件 pipeline | TMA 硬件 pipeline | 硬件自动双缓冲 |

### 1.2 SM90 FlashAttention Kernel 的核心思想

```
将 Attention 计算分解为两类协作 WarpGroup:
┌─────────────────────────────────────────────────────────────┐
│  Producer WG (1 warpgroup = 128 threads)                    │
│  职责: TMA 发起 → K/V 从 HBM 搬入 SMEM (异步)             │
├─────────────────────────────────────────────────────────────┤
│  Consumer WG (1-3 warpgroups = 128-384 threads)             │
│  职责: WGMMA 计算 QK^T → Softmax → PV                     │
└─────────────────────────────────────────────────────────────┘
Producer 和 Consumer 通过 Pipeline barrier 同步,
实现计算与搬运的完全重叠 (双缓冲/多级缓冲)
```

## 2. FlashAttnFwdSm90 Kernel 类 (flash_fwd_kernel_sm90.h)

### 2.1 类结构 (L27-78)

```cpp
// hopper/flash_fwd_kernel_sm90.h L27-78
template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class FlashAttnFwdSm90 {
    // 从 Mainloop 获取配置
    static constexpr bool Is_causal = CollectiveMainloop::Is_causal;
    static constexpr bool Is_FP8 = CollectiveMainloop::Is_FP8;
    static constexpr bool Use_TMA_Q = CollectiveMainloop::Use_TMA_Q;
    static constexpr bool Use_TMA_KV = CollectiveMainloop::Use_TMA_KV;
    static constexpr bool PackGQA = CollectiveMainloop::PackGQA;
    
    // 线程组织
    static constexpr uint32_t NumLoadWarpGroups = 1;      // 1个 Producer WG
    static constexpr uint32_t NumMmaWarpGroups = ...;     // 1-3个 Consumer WG
    static constexpr uint32_t MaxThreadsPerBlock = 
        size(TiledMmaPV) + NumLoadWarpGroups * 128;       // 总线程数
};
```

**WHY 分离 Load/MMA WarpGroup?** WGMMA 执行时不需要通用寄存器，将数据搬运工作交给
独立 WG 可以让 MMA WG 的寄存器全部用于计算，最大化 MMA 吞吐。

### 2.2 SharedStorage 布局 (L88-115)

```cpp
// flash_fwd_kernel_sm90.h L88-115
struct SharedStorage {
    struct TensorStorage {
        union {  // Q 和 K/V 共享 SMEM（不同时使用）
            typename CollectiveMainloop::TensorStorage mainloop;
            typename CollectiveEpilogue::TensorStorage epilogue;
        };
    } tensors;
    struct PipelineStorage {
        BarrierQ barrier_Q;
        typename MainloopPipelineK::SharedStorage pipeline_k;   // K pipeline 状态
        typename MainloopPipelineV::SharedStorage pipeline_v;   // V pipeline 状态
        typename MainloopPipelineVt::SharedStorage pipeline_vt; // V^T pipeline
        typename TileScheduler::SharedStorage smem_scheduler;   // Tile 调度
    } pipelines;
};
```

**WHY Union?** Q 只需加载一次（外层循环），之后 SMEM 空间可以复用给 K/V 的多级缓冲，
最大化可用 pipeline stages。

### 2.3 Thread 角色分配 (L166-200)

```
Block 线程布局 (以 kBlockM=128, d=128 为例):
┌────────────────────────────────────────────┐
│ WarpGroup 0 (threads 0-127):  Producer     │
│   - TMA 发起 K/V prefetch                  │
│   - Pipeline barrier 管理                   │
├────────────────────────────────────────────┤
│ WarpGroup 1 (threads 128-255): Consumer    │
│   - WGMMA: S = Q × K^T                    │
│   - Online Softmax                          │
│   - WGMMA: O += P × V                     │
├────────────────────────────────────────────┤
│ WarpGroup 2 (threads 256-383): Consumer    │
│   (仅 kBlockM≥192 时启用)                   │
│   - 处理另一半 Q-rows 的 attention          │
└────────────────────────────────────────────┘
```

## 3. TMA (Tensor Memory Accelerator) 使用

### 3.1 TMA Descriptor 创建 (flash_api.cpp)

```cpp
// 在 host 端创建 TMA descriptor (一次性)
// TMA descriptor 编码: 基地址, 维度, stride, 分块模式, swizzle
auto tma_desc_Q = cute::make_tma_copy(
    SM90_TMA_LOAD{}, tensor_Q, smem_layout_Q, tileQ, ...);
auto tma_desc_K = cute::make_tma_copy(
    SM90_TMA_LOAD{}, tensor_K, smem_layout_K, tileK, ...);
```

### 3.2 TMA 异步加载模式

```
TMA Pipeline 时序 (2-stage 双缓冲):
                                                
Time ─────────────────────────────────────────→
                                                
Producer:  [TMA K0] [TMA V0] [TMA K1] [TMA V1] [TMA K2] ...
                     │              │              │
Barrier:         arrive_0       arrive_1       arrive_0 ...
                     │              │              │
Consumer:        wait_0 ─[WGMMA0]─ wait_1 ─[WGMMA1]─ ...
                                                
Key: Consumer 处理 stage0 时, Producer 已在加载 stage1
```

### 3.3 Multicast TMA (Cluster 模式)

```
当 ClusterShape > 1 (如 2×1):
TMA 一次加载可以广播到 Cluster 内多个 SM 的 SMEM:

           HBM                    SM0 SMEM    SM1 SMEM
    ┌──────────────┐          ┌──────────┐ ┌──────────┐
    │    K block   │──TMA────→│  K copy  │=│  K copy  │
    └──────────────┘          └──────────┘ └──────────┘
                              (一次 DMA, 两份 SMEM)

WHY? 减少 HBM 带宽需求。当 GQA ratio 大时，多个 Q heads
共享同一 K/V，multicast 避免重复加载。
```

## 4. WGMMA (Warpgroup Matrix Multiply-Accumulate)

### 4.1 QK^T 计算 (TiledMmaQK)

```cpp
// mainloop_fwd_sm90_tma_gmma_ws.hpp
// SS mode: Q from SMEM, K^T from SMEM
using TiledMmaQK = cute::make_tiled_mma(
    GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>(),
    AtomLayoutQK{});
// 一条 WGMMA 指令: 128 threads 协同计算 128×128 tile
// 吞吐: 1024 FP16 FLOPs/clock (H100)
```

### 4.2 PV 计算 (TiledMmaPV)

```cpp
// RS mode 或 SS mode (取决于 MmaPV_is_RS)
// RS: P 在 Register, V 从 SMEM
// SS: P 写回 SMEM 再计算 (寄存器压力大时)
using TiledMmaPV = cute::make_tiled_mma(
    std::conditional_t<
        MmaPV_is_RS,
        GMMA::rs_op_selector<...>(),  // P in registers
        GMMA::ss_op_selector<...>()   // P in SMEM
    >{}, AtomLayoutPV{});
```

**WHY RS vs SS 选择?** (mainloop L:MmaPV_is_RS)
- RS (P in register): 最快，但 P 占用大量寄存器 → 适合 small d (64, 128)
- SS (P in SMEM): P 先写 SMEM 再读回 → 适合 large d (192, 256) 寄存器不够时

### 4.3 IntraWG Overlap 优化

```
传统序列: QK^T → Softmax → PV (串行)
IntraWG 重叠: 
  WGMMA QK^T[block j+1] 发起后, 
  在等待 WGMMA 完成期间执行 Softmax[block j]
  → 计算与计算重叠, 减少 pipeline bubble
```

## 5. Persistent Kernel 与 Tile Scheduling

### 5.1 Persistent Thread 模式 (tile_scheduler.hpp)

```cpp
// hopper/tile_scheduler.hpp
// 传统模式: 每个 tile 启动一个 block → N_tiles 次 kernel launch
// Persistent: 启动 num_sm 个 block, 每个 block 循环处理多个 tile

while (tile_idx < total_tiles) {
    auto [m_block, head_idx, batch_idx, split_idx] = scheduler.get_tile(tile_idx);
    // 处理当前 tile
    mainloop.run(...);
    epilogue.store(...);
    tile_idx = scheduler.advance();  // 原子获取下一个 tile
}
```

**WHY Persistent?**
1. 消除 kernel launch overhead (~5μs/launch × N_tiles)
2. 利用 tile_count_semaphore 实现 work stealing
3. 配合 PDL (Programmatic Dependent Launch) 实现 pipeline 跨 kernel

### 5.2 Tile 调度策略

```
Tile 索引映射:
total_tiles = num_m_blocks × num_heads × batch_size × num_splits

调度顺序优化 (head_swizzle):
  标准: (batch, head, m_block) — 相邻 block 处理相邻 head
  Swizzle: 交错分配 head 到不同 SM cluster
  → 减少 L2 cache 冲突 (不同 head 的 K/V 分散在不同 cache line)
```

## 6. 完整前向 Kernel 执行流

```
FlashAttnFwdSm90::operator()() 执行流:

1. 初始化 SharedStorage, Pipeline barriers
2. TMA prefetch descriptors (单线程)
3. if (Producer WG):
     for each tile from scheduler:
       a. TMA_load Q (if Use_TMA_Q)
       b. for each K-block:
            TMA_load K[j] → smem stage[j%2]
            TMA_load V[j] → smem stage[j%2]
            arrive(pipeline_barrier)

4. if (Consumer WG):
     for each tile from scheduler:
       a. Load Q from SMEM to registers (if !Use_TMA_Q, manual load for PackGQA)
       b. Initialize Softmax state (row_max=-inf, row_sum=0)
       c. for each K-block j:
            wait(pipeline_barrier)  // 等待 TMA 完成
            S = WGMMA(Q_reg, K_smem)  // QK^T
            apply_mask(S)              // causal/local mask
            scores_scale = softmax.max_get_scale(S)
            softmax.rescale_o(acc_o, scores_scale)
            softmax.online_softmax(S)  // S → P in-place
            acc_o += WGMMA(P, V_smem)  // PV
            release(pipeline_barrier)  // 通知 Producer 可复用 stage
       d. final_scale = softmax.finalize()
       e. acc_o *= final_scale
       f. Epilogue: store O to HBM (via TMA_store or direct write)
       g. store LSE to HBM
```

## 7. 性能关键路径分析

### 7.1 Roofline 分析

```
H100 SXM 理论峰值:
  - FP16 Tensor Core: 989 TFLOPS  
  - HBM 带宽: 3.35 TB/s
  - 计算强度阈值: 989/3.35 = 295 FLOPs/byte

FlashAttention 计算强度:
  FLOPs per tile: 2 × Br × Bc × d (QK^T) + 2 × Br × d × Bc (PV)
                = 4 × Br × Bc × d
  Bytes per tile: (Bc × d + Bc × d) × 2 = 4 × Bc × d (K+V, fp16)
  
  Intensity = 4×Br×Bc×d / (4×Bc×d) = Br FLOPs/byte
  With Br=128: Intensity = 128 > 295? NO → 仍是 memory-bound!
  
  但实际在 SMEM 中重用: Q 加载一次用于所有 K-blocks
  有效 Intensity = Br × (N/Bc) = Br × seqlen/Bc
  With seqlen=8192, Bc=128: Intensity = 128×64 = 8192 → compute-bound ✓
```

## 8. 总结

| 技术 | 实现位置 | 作用 |
|------|---------|------|
| TMA async load | mainloop (Producer WG) | 零开销数据搬运 |
| WGMMA ss/rs | mainloop (Consumer WG) | 最大化 MMA 吞吐 |
| Multi-stage pipeline | PipelineStorage | 隐藏 TMA latency |
| Persistent thread | tile_scheduler.hpp | 消除 launch 开销 |
| Cluster multicast | ClusterShape_ param | 减少 HBM 读取 |
| IntraWG overlap | WGMMA + softmax 交错 | 计算-计算重叠 |
| head_swizzle | TileScheduler | 优化 L2 局部性 |

## 9. FP8 支持 (Is_FP8 路径)

### 9.1 FP8 在 FlashAttention 中的挑战

```
FP8 E4M3 动态范围: [-448, 448], 精度: ~0.1%
标准 Softmax 输出 P ∈ [0, 1] → 大部分值接近 0, FP8 严重 underflow

解决方案 (Max_offset=8):
  不计算 P = exp(S-max)/sum
  而是   P' = exp(S-max) * 256 / sum  (扩展到 [0, 256])
  最终:  O = (P' × V) / 256
  → P' 利用了 FP8 的完整正数范围
```

### 9.2 FP8 Per-Head Descale (flash.h L53-62)

```cpp
// hopper/flash.h L53-62
float * __restrict__ q_descale_ptr;   // Q 的 per-head FP8 缩放因子
float * __restrict__ k_descale_ptr;   // K 的 per-head FP8 缩放因子
float * __restrict__ v_descale_ptr;   // V 的 per-head FP8 缩放因子
// 步长支持 per-batch × per-head 的独立缩放
index_t q_descale_batch_stride, q_descale_head_stride;
```

**WHY Per-Head Descale?** 不同 attention head 的数值范围差异可达 10×，
per-tensor 量化会导致某些 head 精度损失严重。Per-head 是精度与开销的平衡点。

### 9.3 V Transpose 优化 (V_colmajor/Transpose_V)

```
FP8 WGMMA 对 V 的布局要求:
  - WGMMA RS mode: V 需要 row-major (标准)
  - WGMMA SS mode: V 需要特定 swizzle 布局
  
  当 Is_FP8=true 且 V 存储为 row-major:
    Transpose_V=true → 在 SMEM 中转置 V
    pipeline_vt: 专门的 V-transpose pipeline stage
    
  WHY 不直接存 col-major V?
  → 推理场景 KV-cache 通常按 row-major 存储（方便 append）
  → 训练场景 V 来自线性层输出（row-major）
  → 为 kernel 内转置付出少量 SMEM 带宽代价
```

## 10. Varlen (变长序列) 处理

### 10.1 SeqlenInfo 结构

```cpp
// hopper/seqlen.h
template <bool Varlen, bool AppendKV>
struct SeqlenInfoQKNewK {
    int seqlen_q, seqlen_k, seqlen_knew;
    int leftpad_k;
    
    // Varlen 模式: 从 cu_seqlens 数组获取每个序列的起止位置
    // Fixed 模式: 所有序列等长，直接用 params.seqlen_q/k
};
```

### 10.2 Varlen 对 Tile 调度的影响

```
Fixed-length:  total_m_blocks = seqlen_q / kBlockM × batch × heads
Varlen:        每个序列 m_blocks 不同 → 需要预计算 per-batch 的 block 数量

varlen_num_blocks 预处理 (prepare_varlen_num_blocks):
  - 提前扫描 cu_seqlens 确定每个 batch 的 tile 数量
  - 使用 num_m_blocks_ptr 存储累计 tile 偏移
  - Scheduler 根据 flat tile_idx 反查 (batch, m_block)
```

## 11. 与 SM80 Kernel 的关键差异

| 特性 | SM80 (flash_fwd_kernel_sm80.h) | SM90 (flash_fwd_kernel_sm90.h) |
|------|-------------------------------|-------------------------------|
| 数据搬运 | cp.async + SMEM barrier | TMA hardware engine |
| MMA 指令 | HMMA (32 threads/warp) | WGMMA (128 threads/warpgroup) |
| Pipeline | 手动 2-stage buffer | Pipeline primitive (多 stage) |
| Thread 角色 | 所有 thread 都做 load+compute | Producer/Consumer 分离 |
| Persistent | 否 (标准 grid launch) | 是 (persistent loop) |
| Cluster | 不支持 | 支持 (multicast TMA) |
| FP8 | 不支持 | 原生支持 E4M3/E5M2 |
| GQA pack | 不支持 | 支持 (Q heads 打包) |

## 12. 编译特化与 Instantiation

### 12.1 模板实例化策略 (hopper/instantiations/)

```
每种 (HeadDim, FP8, Causal, Split, PackGQA, ...) 组合
生成一个独立的 .cu 文件:

instantiations/
├── flash_fwd_hdim64_fp16.cu
├── flash_fwd_hdim64_fp16_causal.cu
├── flash_fwd_hdim128_bf16.cu
├── flash_fwd_hdim128_bf16_causal_split.cu
├── flash_fwd_hdim128_e4m3.cu
├── flash_fwd_hdim256_bf16_packgqa.cu
└── ... (数百个特化文件)

WHY 不用 if/else 分支?
- 分支预测失败代价: ~20 cycles
- Kernel 内循环可能执行百万次
- 模板特化 = 编译期消除所有分支 → 零额外开销
- 代价: 编译时间长 (~30min), binary 大 (~500MB)
```
