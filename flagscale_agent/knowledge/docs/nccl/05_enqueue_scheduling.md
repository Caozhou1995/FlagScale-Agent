# N05: 任务排队与 Kernel 调度 — 深度源码分析

> 核心文件: src/enqueue.cc (3246行), src/group.cc (913行), src/channel.cc (185行)
> 关键: 从用户调 ncclAllReduce() 到 GPU kernel launch 的完整路径

## 1. 本章解决什么问题？

用户调用 `ncclAllReduce(sendbuff, recvbuff, count, ...)` 后:
1. 请求如何排队？(多个集合操作可以 batch)
2. 算法/协议如何选择？
3. 工作如何分配到多个 channel？
4. GPU kernel 如何 launch？

这是用户 API → GPU 执行的**关键路径**, 决定了启动延迟。

## 2. 用户 API 入口 (collectives.cc → enqueue.cc)

### 2.1 ncclAllReduce 的调用链

```
用户代码:
  ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);

内部展开:
  ncclAllReduce()
    └─ ncclEnqueueCheck(&info)         // enqueue.cc L3124
         ├─ 参数验证 (count, datatype, op)
         ├─ 创建 ncclTaskColl 对象
         ├─ 插入 comm->planner.collSorter  // 按大小排序
         └─ 如果不在 ncclGroupStart/End 内 → 立即 launch
```

### 2.2 ncclGroupStart / ncclGroupEnd 机制 (group.cc)

```c
// 用户代码:
ncclGroupStart();
  ncclAllReduce(..., comm1, ...);     // task 1
  ncclAllReduce(..., comm2, ...);     // task 2
  ncclSend(..., comm1, ...);          // task 3
ncclGroupEnd();                       // 统一 launch

// WHY 需要 Group?
// 1. 多个 comm 的操作需要协调 (避免死锁)
// 2. Batch 多个操作 → 一次 kernel launch (减少启动开销)
// 3. 支持 P2P send/recv 配对 (必须在同一 group 内)
```

### 2.3 Group 内的排序 (enqueue.cc L392-400)

```c
ncclResult_t ncclPrepareTasks(struct ncclComm* comm, ...) {
  // 从 collSorter 取出所有 task (按 traffic 大小降序)
  struct ncclTaskColl* task = ncclTaskCollSorterDequeueAll(&planner->collSorter);
  
  // 按 (function, op, datatype) 分组
  struct ncclTaskColl* tasksByFnOpTy[ncclNumFuncs * ncclNumDevRedOps * ncclNumTypes];
  
  // WHY 按大小降序?
  // 大操作优先 → 大操作独占 channel → 带宽利用最大化
  // 小操作可以合并到同一 kernel (kernel fusion)
}
```

## 3. 算法与协议选择 (enqueue.cc L2123)

### 3.1 ncclGetAlgoInfo — 核心选择逻辑

```c
ncclResult_t ncclGetAlgoInfo(struct ncclComm* comm, struct ncclTaskColl* info,
                              int collNetSupport, int nvlsSupport, ...) {
  float minTime = 3600000000.0;  // 初始: 极大值
  
  // 遍历所有 (算法, 协议) 组合
  for (int algo = 0; algo < NCCL_NUM_ALGORITHMS; algo++) {
    for (int proto = 0; proto < NCCL_NUM_PROTOCOLS; proto++) {
      // 跳过不可用的组合
      if (algo == NCCL_ALGO_NVLS && !nvlsSupport) continue;
      if (algo == NCCL_ALGO_COLLNET_* && !collNetSupport) continue;
      
      // 计算预期时间
      float time;
      ncclTopoGetAlgoTime(comm, info->func, algo, proto, nBytes, numPipeOps, &time);
      
      // 选择最优
      if (time < minTime) {
        minTime = time;
        info->algorithm = algo;
        info->protocol = proto;
      }
    }
  }
  
  // 设置 nChannels, nWarps, chunkSteps, sliceSteps
  info->nChannels = computeNChannels(comm, algo, proto, nBytes);
  info->nWarps = computeNWarps(proto);  // Simple:16, LL:4, LL128:4
}
```

### 3.2 时间模型参数

```
时间 = latency + nBytes / bandwidth

latency 组成:
  - kernel launch: ~5μs
  - 网络延迟: IB ~1.5μs per hop (你的: 2层交换, ~3μs)
  - 协议开销: Simple ~2μs, LL ~0.5μs, LL128 ~1μs
  - 算法步数: Ring=2(N-1), Tree=2log2(N)

bandwidth 由 topo 图搜索得到:
  Ring bwIntra: 370 GB/s (NVSwitch)
  Ring bwInter: 50 GB/s (IB)
  Tree bwIntra: 370 GB/s
  Tree bwInter: 50 GB/s
  NVLS bwIntra: 420 GB/s
```

## 4. Channel 分配与工作分片

### 4.1 Channel 概念

```
Channel = 一条独立的通信管道
  - 每个 channel 有自己的 Ring/Tree 连接
  - 每个 channel 的 buffer 独立
  - 多个 channel 并行执行 → 聚合带宽

你的 8-GPU NVSwitch:
  Ring: 8 channels (每个走不同的 NVLink 环路)
  Tree: 4 channels (2层二叉树, 4个独立 tree)
  NVLS: 8 channels (每个 GPU 的不同 buffer region)
```

### 4.2 数据分片 (enqueue.cc)

```
总数据 D, nChannels=8:
  每 channel 处理: D/8
  
每 channel 内, 按 chunkSize 切分:
  Simple: chunkSize = channelSize / NCCL_STEPS
  LL:     chunkSize = 固定小值 (8KB)
  
pipeline: chunk[i] 在传输时, chunk[i+1] 在计算
  → NCCL_STEPS=8 → 8 个 chunk 同时 in-flight
```

### 4.3 Channel 与 threadblock 的映射

```
一次 kernel launch:
  gridDim.x = nChannels × nBlocks_per_channel
  blockDim.x = nWarps × 32
  
  每个 threadblock 处理一个 channel 的一个 slice
  block_id → channel_id = block_id % nChannels
  
H100 配置:
  nWarps = 16 (Simple协议, 512 threads)
  nChannels = 8
  → 8 个 block, 每个 512 threads
  → 总共 4096 threads = 32 SMs (H100 有 132 SMs)
  → 占 GPU 25% 资源 → 剩余可做计算-通信重叠
```

## 5. Kernel Launch 路径 (enqueue.cc L1568-1876)

### 5.1 ncclLaunchPrepare

```c
ncclResult_t ncclLaunchPrepare(struct ncclComm* comm) {
  // 1. 将 planner 中的 tasks 组装成 KernelPlan
  struct ncclKernelPlan* plan = ...;
  
  // 2. 上传工作描述到 GPU (uploadWork)
  //    通过 cudaMemcpyAsync → 将 ncclDevWorkColl 结构传到 device
  
  // 3. 设置 proxy 操作 (如果有 NET transport)
  for (每个需要 proxy 的 channel) {
    ncclAddProxyOpIfNeeded(comm, plan, &proxyOp);
  }
  
  // 4. 通知 proxy thread: 有新任务
  return ncclSuccess;
}
```

### 5.2 ncclLaunchKernel (enqueue.cc L1753)

```c
ncclResult_t ncclLaunchKernel(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  // 选择正确的 kernel 函数指针
  void* fn = ncclDevKernelForPlan(plan);
  
  // CUDA kernel launch
  void* args[] = {&plan->devWorkList};
  CUDACHECK(cudaLaunchKernel(fn, 
    dim3(plan->nBlocks),         // grid: nChannels
    dim3(plan->nWarps * 32),     // block: 512 threads
    args, 
    plan->sharedMemSize,         // shared memory
    plan->stream));              // user stream
    
  return ncclSuccess;
}
```

### 5.3 Kernel 内部调度 (device/common.cu)

```cpp
__global__ void ncclDevKernelGeneric(struct ncclDevWorkList* workList) {
  int bid = blockIdx.x;          // channel ID
  int tid = threadIdx.x;
  
  // 加载工作描述到 shared memory
  ncclShmem.channelId = bid;
  loadWorkToCacheFromDeviceWorkList(workList);
  
  // 根据 (function, algo, proto) 调用对应实现
  // 编译期已确定: 通过模板 RunWorkColl<func, T, RedOp, algo, proto>
  RunWorkColl<...>::run(tid, nthreads, &work);
}
```

## 6. Kernel Fusion (操作合并)

### 6.1 多操作合并

```
Group 内多个小操作:
  ncclGroupStart();
    ncclAllReduce(buf1, 1KB, ...);
    ncclAllReduce(buf2, 2KB, ...);
    ncclAllReduce(buf3, 500B, ...);
  ncclGroupEnd();
  
NCCL 可以将这 3 个操作合并到一次 kernel launch:
  - 同一 kernel 内依次处理多个 work item
  - 减少 kernel launch 开销 (每次 ~5μs)
  - 特别适合: Megatron 中大量小 AllReduce (embedding sync)
```

### 6.2 numPipeOps 参数

```
ncclPrepareTasks 中按 (fn, op, type) 分组:
  同组的操作 → numPipeOps = 组内操作数
  
对 tuning 的影响:
  numPipeOps 大 → 总传输量大 → 倾向 Ring (带宽优先)
  numPipeOps=1 + 小消息 → 倾向 Tree (延迟优先)
```

## 7. CUDA Graph 支持 (enqueue.cc)

### 7.1 为什么需要 Graph?

```
普通模式:
  每次 ncclAllReduce() → cudaLaunchKernel() → 5μs kernel launch
  1000 次迭代 × 10 次 allreduce = 50ms pure launch overhead

Graph 模式:
  首次: capture 所有操作到 graph
  后续: cudaGraphLaunch() → ~2μs 重放全部操作
  
  节省: (5μs - 0.2μs) × 10 × 1000 = 48ms per 1000 iters
```

### 7.2 实现要点

```c
ncclResult_t ncclLaunchPrepare(struct ncclComm* comm) {
  struct ncclKernelPlanner* planner = &comm->planner;
  
  if (planner->persistent) {
    // CUDA Graph Capture 模式:
    // plan 不能释放 → 保持到 graph 销毁
    // proxy 操作预先配置好 → graph replay 时不重新配置
    plan->persistent = 1;
  }
}
```

## 8. 延迟分解与优化

### 8.1 单次 AllReduce 启动延迟

```
ncclAllReduce() 被调用到数据开始传输:
  
  ├─ 用户空间处理 ─────────── ~0.5μs
  │    ncclEnqueueCheck (参数校验, task 创建)
  │
  ├─ 算法选择 ──────────────── ~1μs  
  │    ncclGetAlgoInfo (遍历 algo×proto, 时间模型计算)
  │
  ├─ Plan 构建 ─────────────── ~1μs
  │    ncclLaunchPrepare (work 结构组装)
  │
  ├─ Work 上传 (H2D) ─────── ~1μs
  │    cudaMemcpyAsync(devWork, hostWork)
  │
  ├─ Kernel Launch ────────── ~5μs
  │    cudaLaunchKernel(ncclDevKernel, ...)
  │
  ├─ Kernel 启动延迟 ──────── ~3μs
  │    GPU scheduler dispatch warp to SM
  │
  └─ 首个数据包发出 ─────── ~2μs
       Primitive::send() → 写数据 + set flag
  
  总延迟: ~13μs (节点内)
  跨节点额外: proxy 轮询响应 ~1μs + RDMA post ~1μs + 网络 ~3μs
```

### 8.2 优化技术

```
1. Persistent Kernel (减少 launch):
   NCCL_KERNEL_PERSISTENT=1
   → kernel 长驻 GPU, 通过 flag 接收新任务
   → 消除每次 launch 的 5μs

2. CUDA Graph (减少 CPU 开销):
   → 固定通信 pattern 时一次 capture 多次 replay

3. Group 合并 (减少 kernel 数量):
   → 多个操作 → 1 个 kernel → 内部依次处理
```

## 9. 设计洞察总结

| 设计点 | 动机 | 性能影响 |
|--------|------|----------|
| 任务排序器 (collSorter) | 大操作优先获得资源 | 带宽利用率提升 |
| (algo,proto) 暴力搜索 | 组合空间不大 (~20) | 总能找到最优 |
| nChannels 动态计算 | 小消息少 channel (省 SM) | 计算-通信重叠 |
| Multi-work kernel | 合并小操作 | 减少 launch 开销 |
| Upload work via memcpy | GPU 不能直接读 host struct | 必要开销, ~1μs |
| Persistent kernel | 避免反复 launch | 延迟降低 5μs |
| CUDA Graph | 重复 pattern 零 CPU | 训练主循环最优 |


## 10. 初始化连接建立 (group.cc + init.cc)

### 10.1 ncclCommInitRank 全流程

```
ncclCommInitRank(comm, nranks, commId, rank)
  │
  ├─ bootstrapInit()              // TCP bootstrap: 建立 rank 间控制面
  │    └─ 每个 rank connect 到 rank 0 → 获取所有 rank 的地址
  │
  ├─ ncclTopoGetSystem()          // [N02] 检测本地拓扑
  ├─ ncclTopoComputePaths()       // [N02] BFS 计算路径
  ├─ ncclTopoCompute(graphs)      // [N02] 搜索最优 Ring/Tree/NVLS
  │
  ├─ ncclTopoPreset()             // 为每个 channel 预设 ring/tree 拓扑
  │    └─ topoRanks.ring{Send,Recv,Prev,Next}[channel]
  │
  ├─ allGather(topoRanks)         // 广播各 rank 的拓扑排名信息
  │
  ├─ ncclTopoPostset()            // 连接 channel: ring 首尾连接, tree 父子连接
  │    ├─ channel.ring.prev = ...
  │    ├─ channel.ring.next = ...
  │    ├─ channel.tree.up = ...
  │    └─ channel.tree.down[0,1,2] = ...
  │
  └─ ncclTransportP2pSetup()      // 建立物理传输连接
       ├─ selectTransport()       // 对每对 (send_rank, recv_rank) 选 transport
       │    // 优先级: P2P > SHM > NET
       ├─ transport->send.setup() // 初始化发送端资源 (buffer, IPC handle)
       ├─ exchangeConnectInfo()   // bootstrap 交换连接信息
       └─ transport->recv.connect() // 接收端建立连接 (映射 IPC, 创建 QP)
```

### 10.2 连接建立的延迟

```
32 GPU (4 节点) 集群初始化时间分解:
  Bootstrap TCP:     ~50ms (建立控制面, 交换地址)
  Topo detection:    ~200ms (NVML 查询, PCI 遍历)
  Graph search:      ~100ms (Ring/Tree/NVLS 搜索)
  AllGather topo:    ~20ms (广播拓扑信息)
  Transport setup:   ~500ms (IB QP 创建, IPC 映射)
  ─────────────────────────
  总计: ~900ms
  
优化: NCCL_COMM_SPLIT (从已有 comm 创建子 comm → 复用拓扑信息)
```

## 11. ncclGroupEnd 的执行流程

```c
ncclResult_t ncclGroupEnd() {
  // 1. 收集所有在 group 期间提交的 tasks
  for (每个 comm) {
    ncclPrepareTasks(comm, &algoNeedConnect, &needConnect, &simInfo);
  }
  
  // 2. 如果有新的 peer → 建立连接 (lazy connection)
  if (needConnect) {
    ncclTransportP2pSetup(comm, ...);
  }
  
  // 3. 注册 buffer 并生成 device work
  ncclTasksRegAndEnqueue(comm);
  
  // 4. 准备 launch (上传 work, 配置 proxy)
  ncclLaunchPrepare(comm);
  
  // 5. Launch kernel
  ncclLaunchKernel(comm, plan);
  
  // 6. 异步完成 (kernel 执行, proxy 驱动网络)
  return ncclSuccess;
}
```

### 11.2 Lazy Connection 机制

```
NCCL 不在 init 时建立所有连接, 而是第一次通信时才连接:

初始化: 只建立 bootstrap (TCP 控制面)
首次 AllReduce: 发现需要 Ring channel → 建立 Ring transport 连接
首次 P2P send: 发现需要 P2P → 建立 P2P IPC 连接

WHY Lazy?
  - 32 GPU Ring 只需 2 个邻居连接 (不需要连所有 peer)
  - P2P send/recv 可能只涉及部分 rank
  - 减少初始化时间: 只连必要的 peer
  
代价: 首次通信有额外延迟 (~几百ms)
mitigations: warmup 通信 (Megatron 在训练前做 dummy allreduce)
```

## 12. 环境变量汇总

| 环境变量 | 作用 | 默认值 |
|---------|------|--------|
| NCCL_LAUNCH_MODE | GROUP/PARALLEL | GROUP |
| NCCL_NTHREADS | kernel 线程数 | 512 |
| NCCL_ALGO | 强制算法 | 自动 |
| NCCL_PROTO | 强制协议 | 自动 |
| NCCL_MAX_NCHANNELS | 最大 channel 数 | 32 |
| NCCL_MIN_NCHANNELS | 最小 channel 数 | 1 |
| NCCL_COLLNET_ENABLE | 启用 CollNet | 0 |
| NCCL_NVLS_ENABLE | 启用 NVLS | 1 |
| NCCL_KERNEL_PERSISTENT | 持久 kernel | 0 |
| NCCL_GROUP_CUDA_STREAM | Group 共享 stream | 0 |
| NCCL_CHUNK_SIZE | 强制 chunk 大小 | 自动 |

