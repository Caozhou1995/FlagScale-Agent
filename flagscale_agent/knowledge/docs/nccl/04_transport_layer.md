# N04: Transport 传输层 — 深度源码分析

> 核心文件: src/transport/p2p.cc (1489行), src/transport/net.cc (2092行), src/transport/shm.cc (472行)
> 头文件: src/include/transport.h, src/transport/nvls.cc (1209行), src/proxy.cc (2151行)

## 1. 本章解决什么问题？

算法层 (Ring/Tree) 描述了 "谁和谁通信", Transport 层解决 "如何物理传输":
- GPU 在同一节点 → P2P (NVLink/PCIe) 或 SHM (共享内存)
- GPU 在不同节点 → NET (IB/RoCE) 或 CollNet (SHARP)
- NVSwitch 硬件 reduce → NVLS

每种 Transport 实现相同的接口 → 上层算法与传输解耦。

## 2. Transport 抽象接口 (transport.h)

### 2.1 四种 Transport 类型

```c
#define NTRANSPORTS 4
#define TRANSPORT_P2P 0       // NVLink / PCIe P2P
#define TRANSPORT_SHM 1       // POSIX 共享内存 (同节点, 无 P2P 时 fallback)
#define TRANSPORT_NET 2       // IB/RoCE 网络
#define TRANSPORT_COLLNET 3   // SHARP 集合网络
#define TRANSPORT_PROFILER 4  // Profiling (性能追踪)

extern struct ncclTransport p2pTransport;
extern struct ncclTransport shmTransport;
extern struct ncclTransport netTransport;
extern struct ncclTransport collNetTransport;
```

### 2.2 统一接口 (ncclTransport 结构体)

```c
struct ncclTransport {
  const char name[8];                    // "P2P", "SHM", "NET", "COLL"
  ncclResult_t (*canConnect)(int* ret, struct ncclComm*, struct ncclTopoGraph*,
                             struct ncclPeerInfo*, struct ncclPeerInfo*);
  struct ncclTransportComm {
    ncclResult_t (*setup)(struct ncclComm*, struct ncclTopoGraph*, ...);
    ncclResult_t (*connect)(struct ncclComm*, struct ncclConnectInfo*, ...);
    ncclResult_t (*free)(struct ncclConnector*);
    ncclResult_t (*proxySharedInit)(struct ncclProxyConnection*, ...);
    ncclResult_t (*proxySetup)(struct ncclProxyConnection*, ...);
    ncclResult_t (*proxyConnect)(struct ncclProxyConnection*, ...);
    ncclResult_t (*proxyFree)(struct ncclProxyConnection*, ...);
    ncclResult_t (*proxyProgress)(struct ncclProxyState*, ...);
  } send, recv;
};
```

**WHY 统一接口设计？**
- 上层代码 (channel.cc, connect.cc) 不关心底层是 NVLink 还是 IB
- 新增 transport (如 NVLS) 只需实现接口, 不改核心逻辑
- 运行时根据拓扑自动选择最优 transport

## 3. P2P Transport — NVLink/PCIe 直连 (p2p.cc)

### 3.1 连接判定 (p2p.cc L130-150)

```c
ncclResult_t p2pCanConnect(int* ret, struct ncclComm* comm, 
                           struct ncclTopoGraph* graph,
                           struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  int intermediateRank;
  // 调用拓扑系统判断 P2P 可达性
  ncclTopoCheckP2p(comm, comm->topo, info1->rank, info2->rank, 
                   ret, NULL, &intermediateRank, NULL);
  if (*ret == 0) return ncclSuccess;
  
  // 如果需要中间节点 (PXN) 且使用 CE memcpy, 则不用 P2P
  if (intermediateRank != -1 && useMemcpy) { *ret = 0; return; }
  
  // 检查是否 NET 更优 (跨 NUMA 且无 NVLink 时)
  int useNet = 0;
  ncclTopoCheckNet(comm->topo, info1->rank, info2->rank, &useNet);
  if (useNet) { *ret = 0; return; }
}
```

### 3.2 四种 P2P 模式 (p2p.cc L20-25)

```c
enum p2pType {
  P2P_DIRECT,        // cudaDeviceCanAccessPeer=true → 直接指针访问
  P2P_INTERMEDIATE,  // PXN: 通过中间 GPU 转发
  P2P_IPC,           // CUDA IPC: 跨进程共享 GPU 内存
  P2P_CUMEM          // cuMem API: 新一代内存管理
};
```

### 3.3 Direct P2P (NVLink)

```
条件: 两个 GPU 在同一进程或支持 P2P Access
流程:
  1. cudaDeviceEnablePeerAccess(remoteDev)  → 启用 peer 访问
  2. 获取远端 buffer 的指针 (IPC 映射)
  3. GPU kernel 直接 load/store 远端 GPU 内存

数据流:
  GPU0 kernel: *(remote_ptr + offset) = local_data;  // NVLink 直接写入 GPU1
  
性能: NVLink 全带宽 (H100: 370.8 GB/s per GPU pair via NVSwitch)
```

### 3.4 IPC P2P (跨进程)

```
条件: 同一节点, 不同进程 (常见: torchrun 每 GPU 一个进程)
流程:
  1. 发送端: cudaIpcGetMemHandle(&handle, devPtr) → 导出 IPC 句柄
  2. 句柄通过 bootstrap (TCP/socket) 传给接收端
  3. 接收端: cudaIpcOpenMemHandle(&ptr, handle) → 映射远端内存
  4. 之后与 Direct 相同 → GPU kernel 直接访问

WHY IPC 而非 Direct?
  CUDA 默认只有同进程 GPU 间可以 EnablePeerAccess
  跨进程需要 IPC 机制导出/导入内存句柄
  性能相同 (都走 NVLink), 只是初始化多一步
```

### 3.5 PXN 中间节点转发 (P2P_INTERMEDIATE)

```
场景: GPU0 (NUMA0) → GPU4 (NUMA1), 无 NVLink 直连, 但 GPU2 连接两端
路径: GPU0 → NVLink → GPU2 (proxy) → NVLink → GPU4

实现 (p2p.cc L380+):
  1. 选择 intermediateRank (由 topo 系统在 ncclTopoCheckP2p 中计算)
  2. GPU0 写数据到 GPU2 的 bounce buffer
  3. GPU2 上的 proxy thread 将数据从 bounce buffer 复制到 GPU4
  
通信开销: 2 × NVLink hop (vs 1 × UPI + 2 × PCIe)
实际带宽: min(NVLink_0_2, NVLink_2_4) ≈ 370 GB/s >> UPI 22 GB/s
```

## 4. SHM Transport — 共享内存 (shm.cc)

### 4.1 使用场景

```
同一节点, P2P 不可用时的 fallback:
  - GPU 不支持 P2P (不同 IOMMU group)
  - NCCL_P2P_DISABLE=1 (用户手动禁用)
  - MIG 模式下不同 instance

路径: GPU0 → PCIe → CPU内存(mmap共享区) → PCIe → GPU1
```

### 4.2 实现 (shm.cc 核心逻辑)

```c
// 共享内存结构
struct shmConnectInfo {
  ncclShmIpcDesc_t desc;    // POSIX shm 描述符
  size_t size;              // buffer 大小
};

// 数据传输:
// 发送端: cudaMemcpyAsync(shmBuf, gpuBuf, D2H) → 写入共享内存
// 接收端: cudaMemcpyAsync(gpuBuf, shmBuf, H2D) → 读取共享内存

// WHY 性能远低于 P2P?
//   P2P: GPU→NVLink→GPU  (370 GB/s)
//   SHM: GPU→PCIe→CPU→PCIe→GPU  (2×PCIe = min(32,32)=32 GB/s, 且经过CPU缓存)
```

## 5. NET Transport — IB/RoCE 网络 (net.cc)

### 5.1 架构

```
发送路径:
  GPU(src) → [GDR: NIC直读GPU] 或 [bounce: GPU→CPU→NIC] → IB fabric → 
  → NIC(dst) → [GDR: NIC直写GPU] 或 [bounce: NIC→CPU→GPU] → GPU(dst)

两种数据搬运方式:
  GDR (GPUDirect RDMA):  NIC 通过 PCIe P2P 直接访问 GPU 显存
  Bounce Buffer:         GPU↔CPU 显式拷贝, CPU↔NIC 走 DMA
```

### 5.2 GDR vs Bounce Buffer 选择

```c
// net.cc 中的判定逻辑:
if (topoPathType <= PATH_PXB) {
  // GPU 和 NIC 在同一 PCIe root complex → GDR 可行
  useGdr = 1;
} else {
  // 跨 Host Bridge → GDR 性能差, 用 bounce buffer
  useGdr = 0;
}

// 环境变量覆盖:
NCCL_NET_GDR_LEVEL=LOC   → 只有同设备才 GDR (几乎不用)
NCCL_NET_GDR_LEVEL=PIX   → 同 PCIe switch 才 GDR
NCCL_NET_GDR_LEVEL=PXB   → 同 root complex 才 GDR (默认)
NCCL_NET_GDR_LEVEL=PHB   → 跨 host bridge 也 GDR
NCCL_NET_GDR_LEVEL=SYS   → 总是 GDR
```

### 5.3 Proxy Thread 架构 (proxy.cc)

```
NET transport 的数据传输由 CPU proxy thread 驱动:

┌──GPU kernel──┐    ┌──Proxy Thread──┐    ┌──NIC/RDMA──┐
│  produce data│    │ poll GPU flag   │    │            │
│  set flag    │───>│ ibv_post_send() │───>│ RDMA Write │
│  wait ack    │<───│ poll CQ         │<───│ completion │
└──────────────┘    └────────────────┘    └────────────┘

WHY 需要 proxy thread?
  - GPU kernel 不能直接调用 IB verbs API
  - NIC 需要 CPU 来配置 QP, 发起 RDMA 操作
  - Proxy 在后台轮询, 实现 GPU kernel 和网络传输的重叠

性能关键:
  - Proxy 独占一个 CPU core → 无上下文切换
  - 使用 busy-polling → 最低延迟响应
  - 每个 NIC 一个 proxy thread (你的系统: 8 个 proxy)
```

### 5.4 多 NIC 并行 (Rail-Optimized)

```
你的集群: 8 NIC / 节点, 每个 GPU 绑定一个本地 NIC

通信模式 (8-channel Ring AllReduce 跨节点):
  Channel 0: GPU0 → NIC0 → IB → NIC0(remote) → GPU0(remote)
  Channel 1: GPU1 → NIC1 → IB → NIC1(remote) → GPU1(remote)
  ...
  Channel 7: GPU7 → NIC7 → IB → NIC7(remote) → GPU7(remote)

总带宽: 8 × 50 GB/s (NDR400) = 400 GB/s 双向
每个 GPU 有效带宽: 50 GB/s (单 NIC)
```

## 6. NVLS Transport — NVSwitch Multicast (nvls.cc)

### 6.1 CUDA Multicast API

```c
// nvls.cc 核心初始化:
cuMulticastCreate(&mcHandle, &prop);          // L200: 创建 multicast 组
for (int i = 0; i < nGpus; i++) {
  cuMulticastAddDevice(mcHandle, devices[i]); // L220: 注册 GPU
}
// 每个 GPU 映射 multicast 地址:
cuMemMap(mcVa, size, 0, mcHandle, 0);         // L250: 映射虚拟地址
cuMemSetAccess(mcVa, size, descs, nGpus);     // L260: 设置读写权限
```

### 6.2 NVLS 数据流

```
AllReduce via NVLS:
  Step 1: 每个 GPU 写入 multicast 地址
    GPU_i: atomicAdd(mcAddr + offset, local_data[offset])
    NVSwitch 在硬件上累加所有 GPU 的写入
    
  Step 2: 每个 GPU 读取 multicast 地址
    GPU_i: result = *(mcAddr + offset)
    获取 reduce 后的结果

WHY NVLS 比 Ring 快?
  Ring 8-GPU: 需要 14 步 (7 reduce + 7 gather), 串行依赖
  NVLS: 1 次写入 + 1 次读取, 所有 GPU 并行
  实测: NVLS ~420 GB/s vs Ring ~370 GB/s (8-GPU H100)
```

## 7. Transport 选择决策树

```
给定 rank_i 和 rank_j, NCCL 如何选择 transport?

                    同一节点?
                   /         \
                 是            否
                /               \
        P2P 可达?            ──→ NET Transport
       /         \                (proxy + RDMA)
     是           否
     /             \
  P2P Direct    SHM Transport
  (NVLink/PCIe)  (bounce buffer)
     
特殊路径:
  - NVSwitch 可用 + 节点内 AllReduce → NVLS
  - SHARP 可用 + 跨节点 → CollNet
  - PXN 场景 → P2P_INTERMEDIATE

选择优先级: NVLS > P2P > NET > SHM
```

## 8. Proxy 线程模型 (proxy.cc)

### 8.1 生命周期

```
ncclCommInitRank()
  └─ ncclProxyCreate()              // 创建 proxy 线程池
       ├─ pthread_create(proxyThread, ...)
       └─ affinity: 绑定到 NIC 对应的 NUMA core

proxyThread 主循环:
  while (!stop) {
    // 1. 轮询 GPU 产生的新请求 (send/recv)
    ncclProxyProgress(state);
    
    // 2. 对每个活跃连接:
    //    - 检查 GPU flag → 有新数据?
    //    - ibv_post_send/recv → 提交到 NIC
    //    - ibv_poll_cq → 检查完成
    //    - 通知 GPU → 更新 ack flag
  }
```

### 8.2 请求流水线

```
GPU kernel 和 Proxy 的交互 (Simple 协议):

GPU side:                     Proxy side:
  produce chunk_0             
  tail = 1 ─────────────────> see tail=1
                                ibv_post_send(chunk_0)
  produce chunk_1               
  tail = 2 ─────────────────> see tail=2
                                ibv_post_send(chunk_1)
                              ibv_poll_cq → chunk_0 done
                              head = 1 ───────────────> GPU sees head=1
  produce chunk_2               (buffer 0 可复用)
  ...

NCCL_STEPS=8: 最多 8 个 chunk 在飞行中
  → 隐藏网络延迟 (8 × 512KB = 4MB in flight)
```

## 9. 性能分析与调优

### 9.1 关键参数

| 参数 | 含义 | 默认值 | 你的集群推荐 |
|------|------|--------|--------------|
| NCCL_BUFFSIZE | 每 channel buffer 大小 | 4MB | 8MB (大模型) |
| NCCL_NTHREADS | kernel 线程数 | 512 | 512 |
| NCCL_MAX_NCHANNELS | 最大 channel 数 | 32 | 8 (= NIC数) |
| NCCL_MIN_NCHANNELS | 最小 channel 数 | 1 | 8 |
| NCCL_P2P_NET_CHUNKSIZE | P2P/NET chunk 大小 | 512KB | 1MB |
| NCCL_STEPS | 流水线深度 | 8 | 8 |
| NCCL_NET_GDR_LEVEL | GDR 级别 | PXB | PXB |
| NCCL_SOCKET_NTHREADS | Socket 线程数 | 1 | N/A (用 IB) |
| NCCL_IB_HCA | IB 网卡选择 | 全部 | mlx5_0:1,...,mlx5_7:1 |

### 9.2 你的集群性能预期

```
节点内 (8 GPU NVSwitch):
  P2P bandwidth: 370 GB/s per pair (all-to-all via NVSwitch)
  NVLS AllReduce: ~420 GB/s algorithm bandwidth
  
跨节点 (4 × 8 = 32 GPU):
  单 NIC: 50 GB/s (NDR400 单向)
  8-NIC parallel: 400 GB/s 单向总带宽
  Ring AllReduce 有效带宽: 8 × 50 × (31/32) ≈ 387 GB/s
  
瓶颈分析:
  TP 通信 (节点内): ~420 GB/s → 极少成为瓶颈
  DP 通信 (跨节点): 50 GB/s/GPU → 大模型 gradient sync 的主要瓶颈
```

## 10. 设计洞察总结

| 设计决策 | 动机 | 影响 |
|----------|------|------|
| Transport 接口抽象 | 解耦算法与物理传输 | 易于扩展新硬件 |
| Proxy 线程模型 | GPU 不能调 IB verbs | 延迟开销 ~1μs |
| GDR 路径感知 | 跨 NUMA GDR 性能差 | 自动 bounce buffer |
| NVLS multicast | NVSwitch 硬件 reduce | 节点内最优 |
| Rail-Optimized NIC | 均衡 NIC 流量 | 线性扩展带宽 |
| IPC 内存映射 | 跨进程 P2P | 零额外拷贝 |
| 8-step pipeline | 隐藏网络延迟 | 带宽利用率 >95% |


## 11. IB Transport 深入 (net_ib/)

### 11.1 IB Verbs 调用链

```
NCCL NET transport 对 IB 的调用顺序:
  初始化:
    ibv_open_device()       → 打开 HCA
    ibv_alloc_pd()          → 分配 Protection Domain
    ibv_create_cq()         → 创建 Completion Queue
    ibv_create_qp(RC)       → 创建 Reliable Connected QP
    ibv_modify_qp(INIT→RTR→RTS) → QP 状态机迁移
    
  数据传输:
    ibv_reg_mr(gpu_buf)     → 注册 GPU 内存 (GDR)
    ibv_post_send(RDMA_WRITE) → 发起 RDMA 写
    ibv_poll_cq()           → 等待完成

关键: ibv_reg_mr 对 GPU 内存的注册
  → 需要 nvidia_peermem 内核模块
  → 允许 NIC 通过 PCIe 直接 DMA 到 GPU 显存
  → 无此模块时 fallback 到 bounce buffer
```

### 11.2 多 QP 与多 Rail

```
你的集群每个 rank 的连接结构:
  
  本地 GPU0 → 远端 Rank (4节点×8GPU = 32个 peer rank):
    QP_0: GPU0 → mlx5_0 → IB → mlx5_0(node1) → GPU0(node1)
    QP_1: GPU0 → mlx5_0 → IB → mlx5_0(node2) → GPU0(node2)
    QP_2: GPU0 → mlx5_0 → IB → mlx5_0(node3) → GPU0(node3)
    
  每 channel 一个 QP → 8 channels × 3 remote nodes = 24 QP per GPU

环境变量:
  NCCL_IB_QPS_PER_CONNECTION=1   (默认, 每连接 1 QP)
  NCCL_IB_TC=0                   (Traffic Class, 优先级)
  NCCL_IB_SL=0                   (Service Level)
  NCCL_IB_GID_INDEX=3            (RoCEv2 用, IB 不需要)
```

### 11.3 IB 带宽调优建议

```
你的集群优化策略:
  1. 确保 nvidia_peermem 加载 → GDR 生效
     lsmod | grep nvidia_peermem
     
  2. GPU-NIC affinity 对齐 → PATH_PIX (同 PCIe switch)
     nvidia-smi topo -m  → 确认 GPU-NIC 映射
     
  3. Adaptive Routing 开启 → 避免 IB 热点
     (交换机侧配置)
     
  4. PCI relaxed ordering → 提升 GDR 吞吐
     NCCL_IB_PCI_RELAXED_ORDERING=1
```

