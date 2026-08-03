# Chapter 06: TMA与异步流水线 深度源码分析

## 1. 设计动机

**WHY异步流水线**: GPU的HBM延迟约400-600 cycles，而一次WGMMA只需几十cycles。
如果同步等待数据，Tensor Core利用率不到10%。异步流水线通过多stage buffer，
让数据搬运和计算完全重叠(overlap)。

**核心思想**: 让Producer(TMA)持续填充SMEM buffer，Consumer(WGMMA)持续消费，
两者通过Pipeline Barrier解耦。

## 2. TMA硬件原理

### 2.1 Tensor Memory Accelerator

```
传统数据加载 (LDG):
┌──────────────┐     LDG.128 × 128       ┌───────────┐
│  HBM (GMEM)  │ ◄──────────────────────── │ 128 Threads│
│  A[M,K]      │  每thread计算自己的地址     │ (1 Block)  │
└──────────────┘  每thread发128bit请求      └───────────┘
问题: 128条独立请求，SM的L1/LSU成为瓶颈

TMA数据加载:
┌──────────────┐     TMA descriptor       ┌───────────┐
│  HBM (GMEM)  │ ◄──────────────────────── │ 1 Thread   │
│  A[M,K]      │  1条指令搬运整个tile       │ (Producer) │
└──────────────┘  硬件处理地址/对齐/OOB     └───────────┘
优势: 释放127个thread给计算，零地址开销
```

### 2.2 TMA Descriptor

```cpp
// cute/arch/copy_sm90_tma.hpp
// TMA需要预先创建descriptor:
struct TmaDescriptor {
  uint64_t tensor_base_addr;    // GMEM基地址
  uint64_t tensor_dims[5];      // 各维度大小
  uint64_t tensor_strides[4];   // 各维度stride
  uint32_t box_dims[5];         // 单次传输的tile大小
  uint32_t element_strides[5];  // 元素间隔
  uint32_t interleave;          // 交错模式
  uint32_t swizzle;             // swizzle模式
  uint32_t fill_value;          // OOB填充值
};
// 总大小: 128 bytes, 需要128B对齐
```

**WHY Descriptor**: 将地址计算前移到kernel launch前，运行时只需传坐标，
硬件DMA引擎直接根据descriptor计算所有地址。

### 2.3 支持的操作

| 操作 | PTX | 功能 |
|------|-----|------|
| TMA_LOAD | cp.async.bulk.tensor.Xd...global | GMEM→SMEM |
| TMA_STORE | cp.async.bulk.tensor.Xd...shared | SMEM→GMEM |
| TMA_LOAD_MULTICAST | +multicast::cluster | 多播到cluster |
| TMA_REDUCE | cp.reduce.async.bulk | SMEM→GMEM atomic reduce |

### 2.4 CuTe中的TMA

```cpp
// cutlass/include/cute/arch/copy_sm90_tma.hpp

// 创建TMA copy atom:
auto tma_a = make_tma_copy(
    SM90_TMA_LOAD{},           // TMA Load操作
    gA,                         // 全局tensor (含layout信息)
    sA_layout,                  // SMEM目标layout
    tile_shape,                 // 每次加载的tile形状
    cluster_shape               // cluster multicast配置
);

// 执行:
// Producer warp中:
copy(tma_a, gA(_,_,k_tile), sA(_,_,stage));
// ↑ 一行代码: 自动生成PTX cp.async.bulk.tensor
```

## 3. 异步Pipeline机制

### 3.1 Pipeline抽象

```cpp
// cutlass/include/cutlass/pipeline/sm90_pipeline_tma_async.hpp
template <int Stages>
struct PipelineTmaAsync {
    // 内部使用mbarrier数组
    uint64_t mbarrier_[Stages];  // 每stage一个barrier
    
    // Producer接口
    void producer_acquire(PipelineState state);  // 等待buffer空闲
    void producer_commit(PipelineState state);   // 通知consumer数据就绪
    
    // Consumer接口  
    void consumer_wait(PipelineState state);     // 等待数据就绪
    void consumer_release(PipelineState state);  // 释放buffer给producer
};
```

### 3.2 时序图

```
Stage:    0     1     2     0     1     2     0  ...
         ┌─┐   ┌─┐   ┌─┐   ┌─┐   ┌─┐   ┌─┐
Producer:│L│   │L│   │L│   │L│   │L│   │L│      L=TMA Load
         └┬┘   └┬┘   └┬┘   └┬┘   └┬┘   └┬┘
          │     │     │     │     │     │
          ▼     ▼     ▼     ▼     ▼     ▼
mbarrier: [✓]   [✓]   [✓]   [✓]   [✓]   [✓]    ✓=arrive
          │     │     │     │     │     │
          ▼     ▼     ▼     ▼     ▼     ▼
               ┌─┐   ┌─┐   ┌─┐   ┌─┐   ┌─┐
Consumer:      │M│   │M│   │M│   │M│   │M│      M=WGMMA
               └─┘   └─┘   └─┘   └─┘   └─┘

时间重叠: Load[k+2]与MMA[k]同时执行
延迟隐藏: 只要Stages×K_TILE_SIZE覆盖HBM延迟
```

### 3.3 mbarrier硬件

```
mbarrier是H100的硬件barrier:
- 位于Shared Memory中 (8 bytes each)
- 支持异步completion计数
- TMA完成时自动arrive (无需thread参与!)

流程:
1. Producer: mbarrier.init(expected_tx_count=tile_bytes)
2. Producer: cp.async.bulk...mbarrier::complete_tx::bytes  ← TMA带mbarrier
3. 硬件: TMA完成后自动 mbarrier.arrive(tx_bytes)
4. Consumer: mbarrier.try_wait() → 轮询直到count=0
5. Consumer: 使用SMEM数据做WGMMA

**WHY mbarrier优于__syncthreads**: 
- __syncthreads阻塞所有thread
- mbarrier只阻塞consumer，producer继续工作
- 硬件自动arrive，零软件overhead
```

## 4. Stages数量选择

### 4.1 延迟隐藏分析

```
H100 HBM延迟 ≈ 400-600 cycles
WGMMA 128×128×16 延迟 ≈ 32-64 cycles (含pipeline)

需要的stages:
Stages ≥ ceil(HBM_latency / MMA_latency)
       = ceil(500 / 48) ≈ 11

但受SMEM容量限制:
每stage SMEM使用 = sizeof(A_tile) + sizeof(B_tile)
= (128×64 + 64×128) × 2 bytes (FP16)
= 32 KB per stage

可用SMEM = 228 KB
最大stages = 228 / 32 = 7 stages

实际: 3-5 stages已足够（因为L2 cache缩短了平均延迟）
```

### 4.2 SMEM Budget公式

```
SMEM_per_stage = M_tile × K_tile × sizeof(A) + K_tile × N_tile × sizeof(B)

示例 (FP16, Tile=128×128×64):
= 128×64×2 + 64×128×2 = 16384 + 16384 = 32KB

示例 (FP8, Tile=128×128×128):
= 128×128×1 + 128×128×1 = 16384 + 16384 = 32KB

剩余给Epilogue: SMEM_epilogue = 228KB - Stages×SMEM_per_stage
```

## 5. Cluster Multicast TMA

### 5.1 原理

```
不使用multicast:                使用multicast (ClusterShape=2×1):
CTA0 TMA load A[0:128,:]       CTA0 TMA load A[0:128,:] → 广播到CTA0+CTA1的SMEM
CTA1 TMA load A[0:128,:]       CTA1 无需加载A!
↑ 两次HBM读取                   ↑ 一次HBM读取，带宽节省50%

条件: cluster内的CTA处理同一行A但不同列B
```

### 5.2 Distributed Shared Memory

```
Cluster内CTA可互相访问对方SMEM:
PTX: ld.shared::cluster [addr]
延迟: 比本地SMEM稍高(~30-50 cycles vs ~20 cycles)
但远低于走L2/HBM
```

## 6. 完整Pipeline代码流

```cpp
// Simplified from CUTLASS sm90_mma_tma_gmma_ss_warpspecialized.hpp

// === Producer Warp (Warp 0) ===
for (int k = 0; k < num_k_tiles; ++k) {
    int stage = k % Stages;
    
    // 1. 等待buffer空闲(consumer已释放)
    pipeline.producer_acquire(stage);
    
    // 2. 发起TMA加载
    copy(tma_load_a, gA(_,_,k), sA(_,_,stage));  // A tile
    copy(tma_load_b, gB(_,_,k), sB(_,_,stage));  // B tile
    
    // 3. 提交(设置expected bytes给mbarrier)
    pipeline.producer_commit(stage);
}

// === Consumer Warpgroup (Warp 1,2,3) ===
Tensor accum = partition_fragment_C(tiled_mma, tile_shape);
clear(accum);

for (int k = 0; k < num_k_tiles; ++k) {
    int stage = k % Stages;
    
    // 1. 等待数据就绪
    pipeline.consumer_wait(stage);
    
    // 2. 执行WGMMA
    gemm(tiled_mma, sA(_,_,stage), sB(_,_,stage), accum);
    
    // 3. 释放buffer
    pipeline.consumer_release(stage);
}

// accum now contains final C tile
// → pass to epilogue
```

## 7. 性能优化要点

| 要点 | 说明 | 影响 |
|------|------|------|
| 128B对齐 | TMA要求tile维度×element_size为128B倍数 | 否则fallback慢路径 |
| Swizzle | SMEM layout必须swizzle避免wgmma bank conflict | 10-30%性能差异 |
| Prefetch | consumer_prefetch_wait() 提前触发 | 减少等待 |
| Warp Scheduling | Producer只需1个warp，剩余给compute | 最大化TC利用 |
| K-tile对齐 | K_TILE应为指令K维度整数倍(16/32) | 避免残余处理 |

## 8. 总结

```
TMA + Async Pipeline = H100 GEMM高性能的基石

性能公式:
Achieved_TFLOPS = Peak_TFLOPS × TC_Utilization × Pipeline_Efficiency

TC_Utilization = MMA_cycles / total_cycles
Pipeline_Efficiency = overlap_ratio (理想=1.0)

优化目标: 通过足够的stages使pipeline_efficiency→1.0
限制因素: SMEM容量 → stages数 → pipeline_efficiency上界
```
