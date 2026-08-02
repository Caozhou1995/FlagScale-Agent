# N03: 集合通信算法实现 — 深度源码分析

> 核心文件: src/device/all_reduce.h (788行), src/enqueue.cc (3246行), src/collectives.cc (398行)
> 设备代码: src/device/primitives.h, src/device/prims_simple.h, src/device/prims_ll.h

## 1. 本章解决什么问题？

AllReduce / AllGather / ReduceScatter 等集合操作有多种算法实现:
- **Ring**: 适合大消息, 带宽利用率高
- **Tree**: 适合小消息, 延迟低
- **NVLS**: 利用 NVSwitch 硬件 Reduce, H100+ 专属
- **CollNet Direct/Chain**: 利用 SHARP (交换机内 Reduce)

本章分析:
1. 每种算法的 GPU kernel 实现
2. 算法-协议的组合策略
3. 运行时选择逻辑

## 2. 算法与协议的组合 (RunWorkColl 模板)

### 2.1 模板调度结构 (all_reduce.h L228-245)

```cpp
// 每种 (集合操作, 算法, 协议) 组合 → 一个 GPU kernel
template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
    runRing<T, RedOp, Proto>(tid, nthreads, work);
  }
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE> {
  __device__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runTreeSplit<T, RedOp, ProtoSimple<1, 1>>(tid, nthreads, work);
  }
};
```

**WHY 模板而非 if-else？**
编译期确定路径 → 消除分支 → GPU warp 无 divergence。
代价: 编译后 kernel 数量 = 操作类型 × 数据类型 × 算法 × 协议 ≈ 数千个。

### 2.2 三种协议 (Proto)

| 协议 | 文件 | 传输方式 | 适用场景 |
|------|------|----------|----------|
| Simple | prims_simple.h | 直接 RDMA/NVLink 传输 | 大消息 (>256KB) |
| LL (Low Latency) | prims_ll.h | 带 flag 的 64B 小包 | 小消息 (<8KB) |
| LL128 | prims_ll128.h | 128B 包 + NVLink LDST | 中等消息 (8-256KB) |

```
数据格式对比:
  Simple:  [data chunk 全量传输, 需要同步 flag]
  LL:      [4B data | 4B flag] × N  (50% 带宽利用率)
  LL128:   [120B data | 8B flag]    (93.75% 带宽利用率, NVLink only)
```

**WHY LL 协议牺牲 50% 带宽?**
- 小消息场景下延迟 >> 带宽
- flag 嵌入数据 → 无需额外同步操作
- 单次 load 同时获取数据 + 完成通知 → 省一次 memory fence

## 3. Ring AllReduce 算法 (all_reduce.h L14-70)

### 3.1 算法原理

```
N 个 GPU, 数据分 N 份, Ring 拓扑上执行:
  Phase 1: ReduceScatter (N-1 步)
    每步每个 GPU: 发送一份给下一个, 从上一个接收并 reduce
    
  Phase 2: AllGather (N-1 步)  
    每步每个 GPU: 发送已完成的份给下一个, 从上一个接收

通信量: 每个 GPU 发送 2×(N-1)/N × DataSize
带宽利用率: (N-1)/N → N 大时趋近 100%
```

### 3.2 源码实现 (all_reduce.h runRing L14-70)

```cpp
template <typename T, typename RedOp, typename Proto>
__device__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
  const int nranks = ncclShmem.comm.nRanks;    // Ring 上 GPU 总数
  const int rank = cycleRank(ncclShmem.comm.rank);
  ncclRing* ring = &ncclShmem.channel.ring;
  
  // 创建 Primitives: 从 ring->prev 接收, 向 ring->next 发送
  Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0> prims(
    tid, nthreads, &ring->prev, &ring->next, 
    work->sendbuff, work->recvbuff, work->redOpArg, ...);

  // Phase 1: ReduceScatter
  for (int step = 0; step < nranks - 1; step++) {
    int chunk = cycleChunk(ring, nranks, step, rank);  // 当前处理的分块
    offset = chunkOffset(chunk, ...);
    nelem = min(chunkCount, channelCount - offset);
    prims.directRecvReduceDirectSend(offset, offset, nelem);
    // 关键: 从 prev 接收 + 与本地 reduce + 发送给 next, 流水线化
  }
  
  // Phase 2: AllGather 
  for (int step = 0; step < nranks - 1; step++) {
    int chunk = cycleChunk(ring, nranks, step + nranks - 1, rank);
    offset = chunkOffset(chunk, ...);
    nelem = min(chunkCount, channelCount - offset);
    prims.directRecvCopyDirectSend(offset, offset, nelem);
    // 从 prev 接收 + 写本地 + 发送给 next
  }
}
```

### 3.3 Multi-Channel Ring

```
单 channel Ring: 所有数据走一条环路
Multi-channel Ring: 数据分 nChannels 份, 每份走不同环路

你的 8 GPU NVSwitch:
  Channel 0: GPU0→GPU1→GPU2→...→GPU7→GPU0  (NVLink ring #1)
  Channel 1: GPU0→GPU3→GPU6→...→GPU5→GPU0  (NVLink ring #2)
  ...共 8 channels (由 search.cc 搜索得到)
  
  总带宽 = nChannels × 单 channel 带宽
  H100 8-GPU: ~450 GB/s 实测 AllReduce 带宽
```

## 4. Tree AllReduce 算法 (all_reduce.h L79-225)

### 4.1 算法原理

```
二叉树结构:
  Phase 1: Reduce (叶→根)
    叶节点发送数据给父节点
    内部节点: 接收子节点数据 + reduce + 发送给父节点
    根节点: 接收所有数据的 reduce 结果
    
  Phase 2: Broadcast (根→叶)
    根节点广播最终结果给子节点
    逐层向下传播

延迟: 2×log2(N) 步 (vs Ring 的 2×(N-1) 步)
带宽: 每个节点只处理全部数据 → 带宽利用率 = 1/N (远低于 Ring)
```

### 4.2 TreeSplit 优化 (all_reduce.h L145-225)

```cpp
template <typename T, typename RedOp, typename Proto>
__device__ void runTreeSplit(int tid, int nthreads, struct ncclDevWorkColl* work) {
  ncclTree* tree = &ncclShmem.channel.tree;
  
  // 关键优化: 将 block 的线程分成两组同时执行 reduce + broadcast
  int nthreadsSplit;
  if (Proto::Id == NCCL_PROTO_SIMPLE) {
    nthreadsSplit = nthreads / 2;     // Simple: 50/50 分割
    if (nthreadsSplit >= 256) nthreadsSplit += 64;  // 偏向 reduce 端
  } else {
    nthreadsSplit = (nthreads * 7 / 10);  // LL: 70% reduce, 30% bcast
  }
  
  if (tree->up == -1) {  // 根节点
    // 同时 reduce 和 broadcast
    prims.directRecvReduceCopyDirectSend(...);
  } else if (tid < nthreadsSplit) {  // 非根: 前半线程做 Reduce
    prims.directRecvReduceDirectSend(...);   // recv from children → reduce → send to parent
  } else {  // 后半线程做 Broadcast
    prims.directRecvCopyDirectSend(...);     // recv from parent → send to children
  }
}
```

**WHY TreeSplit?**
```
普通 Tree (UpDown): Reduce 全部完成 → 再 Broadcast
  延迟: T_reduce + T_broadcast = 2 × depth × (α + β×M)

TreeSplit: Reduce 和 Broadcast 重叠
  一旦第一个 chunk reduce 完成 → 立即开始 broadcast
  第 i 个 chunk 的 broadcast 与第 i+1 个的 reduce 重叠
  有效延迟降低约 40%
```

### 4.3 线程分配的不对称性 (L156-163)

```cpp
// LL/LL128 协议: 70% 线程给 reduce, 30% 给 broadcast
nthreadsSplit = (nthreads * 7 / (10 * WARP_SIZE)) * WARP_SIZE;

// WHY 不对称?
// Reduce 需要: recv + compute(add) + send → 3 次内存操作 + ALU
// Broadcast 只需: recv + send → 2 次内存操作, 无 compute
// Reduce 更需要线程来隐藏内存延迟
```

## 5. NVLS (NVLink SHARP) 算法

### 5.1 硬件原理

```
传统 AllReduce (Ring):
  GPU0 ──data──> GPU1 ──data──> GPU2 ──...──> GPU7
  每个 GPU 执行 local reduce → O(N) 步

NVLS (NVSwitch Multicast + InPlace Reduce):
  所有 GPU 同时写入 NVSwitch 上的 multicast 地址
  NVSwitch 硬件在数据流过时完成 reduce
  结果自动广播到所有 GPU

  延迟: 1 步 (所有 GPU 并行)
  带宽: 受限于单 GPU 到 NVSwitch 的链路 (370 GB/s)
```

### 5.2 NVLS AllReduce 实现

```
NVLS AllReduce 分两阶段 (跨节点时):
  Phase 1: 节点内 NVLS Reduce
    - 8 GPU 通过 NVSwitch multicast 完成节点内 reduce
    - 结果存在每个 GPU 的 1/8 份数据中
    
  Phase 2: 跨节点 Ring/Tree
    - 各节点的部分结果通过 IB 做跨节点 AllReduce
    
  Phase 3: 节点内 NVLS Broadcast
    - 通过 NVSwitch multicast 广播最终结果

纯节点内 (你的 8-GPU 测试):
  直接 NVLS → 一步完成 → 接近 NVLink 峰值带宽
```

### 5.3 NVLS 内存映射 (CUDA Multicast)

```c
// NVLS 需要特殊的 multicast 地址映射:
cuMulticastCreate(&mcHandle, ...);                // 创建 multicast 组
cuMulticastAddDevice(mcHandle, dev);              // 每个 GPU 加入
cuMemMap(mcAddr, size, 0, mcHandle, 0);          // 映射到虚拟地址
cuMemSetAccess(mcAddr, size, accessDescs, ...);  // 设置访问权限

// GPU 写入 mcAddr → NVSwitch 硬件 reduce
// GPU 读取 mcAddr → 获取 reduce 后的结果
```

## 6. CollNet (SHARP) 算法 (all_reduce.h L248-500)

### 6.1 SHARP 硬件 Reduce

```
SHARP (Scalable Hierarchical Aggregation and Reduction Protocol):
  IB 交换机内置 reduction 引擎
  数据流过交换机时自动 reduce → 无需到达对端 GPU

层次:
  Level 1: 叶交换机 reduce 同一 leaf 下的节点
  Level 2: 脊交换机 reduce 不同 leaf 的部分结果

你的集群 (2 层 fat-tree):
  Leaf switch: 各连 ~33 节点, 硬件 reduce 这 33 节点
  Spine switch: reduce 64 个 leaf 的结果
```

### 6.2 CollNet Direct 实现 (all_reduce.h L248-380)

```cpp
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE> {
  __device__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    // 线程分为 4 组:
    // 1. Gather: 收集节点内其他 GPU 的数据
    // 2. Reduce: 本地 reduce
    // 3. Scatter: 发送 reduce 结果到 SHARP 网络
    // 4. Bcast: 从 SHARP 接收最终结果并广播
    
    const int nThreadsScatter = WARP_SIZE + COLLNET_COPY_THREADS;
    const int nThreadsGather = COLLNET_COPY_THREADS;
    const int nThreadsBcast = WARP_SIZE + COLLNET_COPY_THREADS;
    const int nThreadsReduce = nWarps*WARP_SIZE - others;
    
    // Scatter: GPU→NIC→SHARP
    prims.scatter(offset, nelem, chunkSize, peerOffset, headRank, shift);
    // Gather: SHARP→NIC→GPU
    prims.gather(offset, nelem, chunkSize, peerOffset, headRank, shift);
  }
};
```

**WHY CollNet 不总是最优?**
- 需要 SHARP 硬件支持 (Mellanox/NVIDIA IB 交换机)
- 交换机 reduce 有容量限制 (QP 数量, buffer 大小)
- 小消息延迟可能高于 Tree (交换机处理开销)
- 大消息时 Ring 的带宽利用率更高

## 7. 算法选择逻辑 (graph/tuning.cc)

### 7.1 Tuning Model

```c
ncclResult_t ncclTopoGetAlgoTime(struct ncclComm* comm, int coll, int algorithm, 
                                  int protocol, size_t nBytes, int numPipeOps, float* time) {
  // 估算每种 (算法, 协议) 组合的执行时间
  // time = latency + nBytes / bandwidth
  
  // latency 与 协议和网络跳数相关
  // bandwidth 由 topo 搜索结果 (bwIntra/bwInter) 决定
}
```

### 7.2 选择策略

```
消息大小 vs 最优算法:
  ┌─────────────────────────────────────────────────────────┐
  │  < 8KB        → Tree + LL    (最低延迟)                 │
  │  8KB ~ 256KB  → Tree + LL128 (平衡延迟/带宽)            │
  │  256KB ~ 1MB  → Ring + Simple (带宽开始主导)             │
  │  > 1MB        → Ring + Simple (带宽最优)                │
  │  节点内 > 1MB → NVLS (如果 NVSwitch 可用)               │
  │  跨节点 +SHARP→ CollNet (如果 SHARP 可用)               │
  └─────────────────────────────────────────────────────────┘

环境变量覆盖:
  NCCL_ALGO=Ring/Tree/NVLS/CollNet
  NCCL_PROTO=Simple/LL/LL128
```

### 7.3 Channel 数量与数据分片

```
数据分片:
  channelCount = totalCount / nChannels         // 每 channel 处理的元素数
  chunkCount = channelCount 内进一步分 chunk    // 流水线粒度
  
  totalCount = 1GB, nChannels = 8:
    channelCount = 128MB per channel
    chunkCount = 512KB (Simple) / 8KB (LL) per chunk
    
  Pipeline: 多个 chunk 在 send/recv/reduce 间重叠执行
```

## 8. Primitives 通信原语 (prims_simple.h)

### 8.1 核心接口

```cpp
template <typename T, typename RedOp, typename Fan, int Direct, typename Proto, int P2p>
class Primitives {
  // 基本操作:
  void send(intptr_t offset, int nelem);           // 发送
  void recv(intptr_t offset, int nelem);           // 接收
  void recvReduceSend(intptr_t offset, int n);     // 接收→reduce→发送 (Ring核心)
  void recvCopySend(intptr_t offset, int n);       // 接收→复制→发送 (Broadcast核心)
  void recvReduceCopyDirectSend(...);              // 接收→reduce→写本地→直接发送
  void scatter(offset, nelem, ...);                // CollNet scatter
  void gather(offset, nelem, ...);                 // CollNet gather
};
```

### 8.2 Direct 操作 (零拷贝)

```
普通模式:
  GPU0: data → local_buf → [NVLink/PCIe] → remote_buf → GPU1
  需要 2 次 GPU 内存访问 + 1 次传输

Direct 模式 (P2P 映射):
  GPU0: data → [NVLink] → GPU1 的 user_buffer (直接写入最终位置)
  消除中间 buffer → 带宽翻倍
  
条件: 目标 buffer 提前 IPC 映射, PATH_NVL 或 PATH_PIX
```

### 8.3 同步机制 (Flag + Counter)

```
Simple 协议的生产者-消费者:
  发送端: 
    1. 写 chunk 数据到共享 buffer
    2. tail++ (memory_order_release)
    
  接收端:
    1. 等待 tail > expected (spin on flag)
    2. 读取数据 (memory_order_acquire)
    3. head++ (通知发送端 buffer 可复用)

LL 协议:
  数据自带 flag → 无需单独同步
  [data_31:0 | flag_31:0] → atomicCAS 检查 flag
```

## 9. ReduceScatter / AllGather 分解

### 9.1 ReduceScatter (Ring)

```
与 Ring AllReduce Phase 1 相同:
  N GPU, 数据分 N 份
  N-1 步后: GPU_i 持有第 i 份的完整 reduce 结果
  
实现: 复用 runRing 的前半部分逻辑
通信量: (N-1)/N × DataSize per GPU (带宽最优)
```

### 9.2 AllGather (Ring)

```
与 Ring AllReduce Phase 2 相同:
  初始: GPU_i 持有 1/N 的数据
  N-1 步后: 所有 GPU 持有全部数据
  
实现: 复用 runRing 的后半部分逻辑
```

### 9.3 组合优化

```
Megatron-LM 中的 TP AllReduce 实际分解为:
  ReduceScatter → AllGather (两次调用)
  
WHY 分解而非直接 AllReduce?
  ReduceScatter 后可以插入 GEMM 计算 (只需 1/N 数据)
  重叠通信和计算 → 有效隐藏通信延迟
  这是 Megatron "Async-TP" 的核心思想
```

## 10. 性能模型

### 10.1 带宽公式

```
Ring AllReduce:
  总通信量 per GPU = 2 × (N-1)/N × D
  时间 = 2(N-1)×α + 2×(N-1)/N × D/B
  
  其中: α=延迟, D=数据量, B=链路带宽, N=GPU数

Tree AllReduce:
  总通信量 per GPU = 2 × D
  时间 = 2×log2(N)×α + 2×D/B
  
交叉点 (Ring 优于 Tree):
  当 D > α×B×N×log2(N) / (1 - log2(N)/N)
  H100 8-GPU (NVLink): D > ~200KB 时 Ring 更优
```

### 10.2 你的集群预期性能

```
节点内 AllReduce (8 GPU NVSwitch):
  NVLS: ~420 GB/s (接近 NVLink 双向峰值 900 GB/s 的理论 bus bandwidth)
  Ring: ~370 GB/s (8-channel × 46 GB/s/channel)
  
跨节点 AllReduce (4×8=32 GPU):
  Ring: ~45 GB/s per GPU (受 IB 带宽限制: 50 GB/s × 7/8)
  Tree: 低延迟但带宽 ~25 GB/s
  
  实际最优: Hierarchical (节点内 NVLS + 跨节点 Ring)
    step1: 节点内 ReduceScatter via NVLS → 每 GPU 持有 1/8
    step2: 跨节点 AllReduce via Ring (数据量 1/8) → 45 GB/s  
    step3: 节点内 AllGather via NVLS → 全量恢复
    有效带宽: ~42 GB/s per GPU
```

## 11. 设计洞察总结

| 设计决策 | 原因 | 影响 |
|----------|------|------|
| 模板化 kernel 分发 | 编译期消除分支 | GPU 效率最大化, 编译慢 |
| TreeSplit 重叠 | Reduce 和 Bcast 并行 | 延迟降低 40% |
| 70/30 线程分配 | Reduce 比 Bcast 计算密集 | ALU/MEM 利用率平衡 |
| LL flag-in-data | 消除同步开销 | 小消息延迟极低 |
| Direct 零拷贝 | 避免中间 buffer | 有效带宽翻倍 |
| Hierarchical NVLS+Ring | 利用各层最优算法 | 跨节点大消息最优 |
| 通信-计算重叠 | RS/AG 分解 | Megatron async-TP |

