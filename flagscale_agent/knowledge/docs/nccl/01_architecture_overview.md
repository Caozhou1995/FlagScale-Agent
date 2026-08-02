# N01: NCCL 架构总览 — 深度源码分析

> 源码版本: NCCL (2025, /workspace/deps/nccl)
> 核心文件: src/init.cc (3522行), src/include/comm.h (911行), src/include/transport.h (233行)

## 1. NCCL 是什么？解决什么问题？

NCCL (NVIDIA Collective Communication Library) 是 NVIDIA 的多 GPU 集合通信库。

**核心问题**: N 张 GPU 之间如何高效执行 AllReduce / AllGather / ReduceScatter 等集合操作？

**WHY 不直接用 MPI？**
- MPI 设计于 CPU 时代，数据在主存
- GPU 通信需要感知: NVLink / NVSwitch / PCIe / IB / GDR 等异构互联
- MPI 无法利用 GPU 直接传输 (GPUDirect RDMA / NVLink P2P)
- NCCL 在 GPU 上运行 kernel，实现 zero-copy 通信

## 2. 全局架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         用户 API 层                                  │
│  ncclAllReduce / ncclAllGather / ncclReduceScatter / ncclSend/Recv  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│                        Group / Enqueue 层                            │
│  任务打包、kernel 参数准备、work FIFO 管理                            │
│  src/group.cc (913行), src/enqueue.cc (3246行)                      │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│                        算法 / 协议选择层                              │
│  Ring / Tree / CollNetDirect / CollNetChain / NVLS / NVLSTree / PAT │
│  协议: LL (Low Latency) / LL128 / Simple                            │
│  src/include/comm.h L53-55                                          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│                        Channel / Graph 层                            │
│  通信通道抽象、拓扑图构建与搜索、路径选择                              │
│  src/graph/ 目录, src/channel.cc                                    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│                        Transport 层                                  │
│  P2P (NVLink/PCIe) / SHM (共享内存) / NET (IB/Socket) / CollNet     │
│  src/transport/ 目录, src/transport.cc                               │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│                        Proxy 层 (CPU 端)                             │
│  IB Verbs 操作、GDR 注册、异步 progress thread                       │
│  src/proxy.cc (2151行)                                              │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│                        Device Kernel 层 (GPU 端)                     │
│  实际的 reduce/copy kernel、协议实现                                  │
│  src/device/ 目录                                                    │
└─────────────────────────────────────────────────────────────────────┘
```

## 3. 核心数据结构

### 3.1 ncclComm — 通信器 (src/include/comm.h L523-797)

ncclComm 是 NCCL 最核心的结构体，包含一次集合通信所需的全部状态:

```c
struct ncclComm {                          // comm.h L523
  uint64_t startMagic;                     // L524: 用于检测内存损坏
  struct ncclMemoryStack memPermanent;     // L525: 生命周期=comm 的内存池
  struct ncclMemoryStack memScoped;        // L525: 临时作用域内存

  // === 拓扑与对等体信息 ===
  struct ncclChannel channels[MAXCHANNELS]; // L534: 通信通道数组 (最多32)
  struct ncclPeerInfo* peerInfo;           // L535: 所有 rank 的设备信息
  struct ncclTopoSystem* topo;             // L536: 系统拓扑图
  struct ncclTopoGraph graphs[NCCL_NUM_ALGORITHMS]; // L559: 各算法的拓扑图

  // === 基本标识 ===
  int rank;                                // L570: 本 rank 在 comm 中的编号
  int nRanks;                              // L571: comm 总 rank 数
  int cudaDev;                             // L572: 本 rank 的 CUDA 设备号
  int64_t busId;                           // L576: PCI Bus ID

  // === 节点拓扑 ===
  int node;                                // L583: 本 rank 所在节点编号
  int nNodes;                              // L584: 总节点数
  int localRank;                           // L585: 节点内本地 rank
  int localRanks;                          // L586: 节点内总 rank 数
  int MNNVL;                               // L595: 是否启用多节点 NVLink

  // === Channel 配置 ===
  int nChannels;                           // L612: 连接用 channel 数
  int collChannels;                        // L613: 集合通信用 channel 数
  int nvlsChannels;                        // L614: NVLS 用 channel 数
  int p2pnChannels;                        // L620: P2P 用 channel 数

  // === 性能调优 ===
  int buffSizes[NCCL_NUM_PROTOCOLS];       // L629: 各协议 buffer 大小
  float latencies[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];  // L641
  float bandwidths[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS]; // L642

  // === 网络 ===
  ncclNet_t* ncclNet;                      // L543: 网络插件接口
  ncclCollNet_t* ncclCollNet;              // L552: CollNet 插件接口
  bool useGdr;                             // L781: 是否使用 GPUDirect RDMA

  // === NVLS (NVLink SHARP) ===
  int nvlsSupport;                         // L697: NVLS 支持标志
  struct ncclNvlsSharedRes* nvlsResources; // L700: NVLS 共享资源

  // === Proxy ===
  struct ncclProxyState* proxyState;       // L683: CPU 端 proxy 状态

  uint64_t endMagic;                       // L796: 结尾 magic (检测溢出)
};
```

**WHY ncclComm 这么大 (900+ 行)?**
因为它是 "通信世界" 的完整快照 — 包含拓扑、路径、buffer、tuning 参数。
初始化一次后，后续每次集合通信只需要查表，不再重新探测。


### 3.2 ncclChannel — 通信通道 (comm.h L150-172)

Channel 是 NCCL 并行化通信的基本单位。多个 channel 并行工作以利用全部互联带宽。

```c
struct ncclChannel {                       // comm.h L150
  struct ncclChannelPeer** peers;          // L151: 对等体连接信息 (host)
  struct ncclDevChannelPeer** devPeers;    // L152: 对等体连接信息 (device)
  struct ncclRing ring;                    // L155: Ring 拓扑中的前驱/后继
  struct ncclTree tree;                    // L157: Tree 拓扑中的父/子节点
  struct ncclDirect collnetDirect;         // L160: CollNet 直连
  struct ncclNvls nvls;                    // L162: NVLS 通道信息
  int id;                                  // L164: channel 编号
};
```

**Channel 数量与带宽的关系**:
```
单 channel 带宽 = min(单链路带宽, 协议开销)
总带宽 = nChannels × 单 channel 带宽

例: H100 NVSwitch, Ring AllReduce:
  理论: 900 GB/s / (2*(n-1)/n) ≈ 450 GB/s bus bandwidth
  实际: 8 channels × ~55 GB/s/channel ≈ 440 GB/s
```

**WHY 需要多 Channel？**
- 单 channel 的 pipeline 深度有限，不能打满互联带宽
- 多 channel 相当于多条平行流水线
- 不同 channel 可以走不同物理路径 (multi-rail)

### 3.3 ncclTransport — 传输层抽象 (transport.h L136-142)

```c
#define NTRANSPORTS 4                      // transport.h L16
#define TRANSPORT_P2P 0                    // NVLink / PCIe P2P
#define TRANSPORT_SHM 1                    // 共享内存 (进程间)
#define TRANSPORT_NET 2                    // IB / Socket / 网络
#define TRANSPORT_COLLNET 3                // CollNet (SHARP)

struct ncclTransport {                     // transport.h L136
  const char name[8];                      // "P2P", "SHM", "NET", "COLLNET"
  ncclResult_t (*canConnect)(...);         // L138: 判断两个 peer 能否用此 transport
  struct ncclTransportComm send;           // L140: 发送端操作集
  struct ncclTransportComm recv;           // L141: 接收端操作集
};

struct ncclTransportComm {                 // transport.h L117
  ncclResult_t (*setup)(...);              // 建立连接
  ncclResult_t (*connect)(...);            // 完成连接握手
  ncclResult_t (*free)(...);               // 释放连接
  ncclResult_t (*proxySharedInit)(...);    // proxy 共享初始化
  ncclResult_t (*proxySetup)(...);         // proxy 端 setup
  ncclResult_t (*proxyConnect)(...);       // proxy 端 connect
  ncclResult_t (*proxyProgress)(...);      // proxy 进度推进 (轮询)
};
```

**Transport 选择优先级** (init.cc 中 canConnect 判断):
```
1. P2P (NVLink)    — 同节点、有 NVLink 直连时优先
2. P2P (PCIe)      — 同节点、无 NVLink 但有 PCIe P2P
3. SHM             — 同节点、同进程间的 fallback
4. NET (IB)        — 跨节点，使用 IB Verbs + GDR
5. CollNet (SHARP) — 跨节点，交换机内计算 (需要硬件支持)
```

### 3.4 ncclPeerInfo — 对等体信息 (transport.h L43-65)

每个 rank 在初始化时交换自己的设备信息:

```c
struct ncclPeerInfo {                      // transport.h L43
  int rank;                                // 全局 rank 编号
  int cudaDev;                             // CUDA 设备号
  int nvmlDev;                             // NVML 设备号
  int gdrSupport;                          // 是否支持 GPUDirect RDMA
  uint64_t hostHash;                       // 主机名哈希 (判断同节点)
  uint64_t pidHash;                        // 进程 ID 哈希 (判断同进程)
  int64_t busId;                           // PCI Bus ID
  cudaUUID_t gpuUuid;                      // GPU UUID
  nvmlGpuFabricInfoV_t fabricInfo;         // MNNVL fabric 信息
  int mloPart;                             // MIG 分区号 (-1 无)
};
```

**WHY hostHash 和 pidHash？**
- hostHash: 判断两个 rank 是否在同一物理机 (决定用 P2P/SHM vs NET)
- pidHash: 判断两个 rank 是否在同一进程 (决定能否直接访问 comm 指针)

## 4. 初始化流程 — ncclCommInitRank

### 4.1 调用链总览

```
用户调用: ncclCommInitRank(&comm, nranks, commId, myrank)
                │
                ▼
        ncclCommInitRankDev()             // init.cc L2477
                │
                ▼ (异步 Job)
        ncclCommInitRankFunc()            // init.cc L1831
                │
                ├── bootstrapInit()       // 建立 TCP 控制面连接
                │
                └── initTransportsRank()  // init.cc L965 ★核心★
                     │
                     ├── AllGather1: 交换 peerInfo
                     │
                     ├── ncclTopoGetSystem()     // 拓扑检测
                     ├── ncclTopoComputePaths()  // 路径计算
                     ├── ncclTopoTrimSystem()    // 裁剪无用节点
                     ├── ncclTopoSearchInit()    // 初始化搜索
                     │
                     ├── ncclTopoCompute(Ring)   // 搜索 Ring 图
                     ├── ncclTopoCompute(Tree)   // 搜索 Tree 图
                     ├── ncclTopoCompute(NVLS)   // 搜索 NVLS 图
                     │
                     ├── AllGather2: 交换 graphInfo + topoRanks
                     │
                     ├── ncclTransportRingConnect()  // 建立 Ring 连接
                     ├── ncclTransportTreeConnect()  // 建立 Tree 连接
                     └── ncclNvlsSetup()            // 建立 NVLS
```

### 4.2 AllGather1: 交换 PeerInfo (init.cc L1034-1068)

```c
// init.cc L1034-1037
// AllGather1: 每个 rank 广播自己的设备信息
NCCLCHECKGOTO(ncclCalloc(&comm->peerInfo, nranks + 1), ret, fail);
NCCLCHECKGOTO(fillInfo(comm, comm->peerInfo + rank, comm->commHash), ret, fail);
NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, comm->peerInfo, sizeof(struct ncclPeerInfo)), ret, fail);
```

**AllGather1 之后，每个 rank 知道**:
- 所有 rank 的 GPU 型号、PCI 地址、所在主机
- 哪些 rank 在同一节点 (hostHash 相同)
- 哪些 rank 支持 GDR

### 4.3 拓扑检测与图搜索 (init.cc L1131-1210)

```c
// init.cc L1141-1153
NCCLCHECKGOTO(ncclTopoGetSystem(comm, &comm->topo), ret, fail);   // 从 sysfs 构建系统拓扑
NCCLCHECKGOTO(ncclTopoComputePaths(comm->topo, comm), ret, fail); // 计算 GPU-GPU、GPU-NIC 路径
NCCLCHECKGOTO(ncclTopoTrimSystem(comm->topo, comm), ret, fail);   // 去掉不可达 GPU/NIC
NCCLCHECKGOTO(ncclTopoComputePaths(comm->topo, comm), ret, fail); // 重新计算路径
NCCLCHECKGOTO(ncclTopoSearchInit(comm->topo), ret, fail);         // 初始化搜索参数

// init.cc L1173-1187: 搜索 Ring 和 Tree 拓扑图
ringGraph->pattern = NCCL_TOPO_PATTERN_RING;
NCCLCHECKGOTO(ncclTopoCompute(comm->topo, ringGraph), ret, fail);

treeGraph->pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
NCCLCHECKGOTO(ncclTopoCompute(comm->topo, treeGraph), ret, fail);
```

**WHY 先 Ring 再 Tree？**
- Ring 的 nChannels 作为 Tree 的 minChannels/maxChannels
- Tree 的 channel 数与 Ring 对齐，共享连接资源

## 5. 算法与协议

### 5.1 七种算法 (init.cc L53-54)

```c
const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS] = {
  "Tree", "Ring", "CollNetDirect", "CollNetChain", "NVLS", "NVLSTree", "PAT"
};
```

| 算法 | 适用场景 | 通信模式 | 带宽利用率 |
|------|----------|----------|-----------|
| Ring | 大数据量 AllReduce | 环形流水线 | (N-1)/N (最优) |
| Tree | 小数据量或延迟敏感 | 二叉树递归 | 1/2 (带宽换延迟) |
| CollNetDirect | IB SHARP 硬件 | 交换机内 reduce | N/A (硬件加速) |
| CollNetChain | IB SHARP + 链式 | 链式 offload | ~Ring |
| NVLS | NVSwitch 多播 | GPU multicast | ~1.0 (最优) |
| NVLSTree | NVLS + 跨节点 Tree | 节点内 NVLS + 跨节点 | 混合 |
| PAT | Parallel Aggregation Tree | 新增并行聚合 | TBD |

**WHY NVLS 比 Ring 好？** (H100 NVSwitch 场景)
```
Ring AllReduce 1GB, 8 GPU:
  每 GPU 发送: 1GB * (N-1)/N * 2 = 1.75 GB (ReduceScatter + AllGather)
  时间: 1.75GB / 450GB/s = 3.9 ms

NVLS AllReduce 1GB, 8 GPU:
  所有 GPU 同时写入 multicast 地址，NVSwitch 做 in-network reduce
  每 GPU 发送: 1GB (一次写入)
  时间: 1GB / 450GB/s = 2.2 ms
  
加速比: ~1.8x
```

### 5.2 三种协议 (init.cc L55)

```c
const char* ncclProtoStr[NCCL_NUM_PROTOCOLS] = {"LL", "LL128", "Simple"};
```

| 协议 | 全称 | 特点 | 适用大小 |
|------|------|------|----------|
| LL | Low Latency | 4B 数据 + 4B flag (一半带宽换零等待) | < 数 KB |
| LL128 | Low Latency 128B | 120B 数据 + 8B flag (更高效) | 数 KB ~ 数百 KB |
| Simple | Simple | 大块传输 + 显式同步 (信号量) | > 数百 KB |

**LL 协议原理**:
```
┌─────────────────────────────────────────┐
│ 传统 (Simple): 写数据 -> 写 flag -> 读 flag -> 读数据    │
│   延迟 = 数据传输 + flag 同步 + 轮询开销                  │
│                                                          │
│ LL 协议: 每 8 字节 = [4B data | 4B flag]                 │
│   接收方直接轮询 data+flag 的 8B word                     │
│   一旦 flag 位翻转，data 立即可用 (零额外延迟)            │
│   代价: 有效带宽减半 (一半空间用于 flag)                   │
└─────────────────────────────────────────┘
```

### 5.3 算法选择机制 (comm.h L639-643)

```c
// comm.h L639-643: Tuner 参数
ncclTunerConstants_t tunerConstants;
ssize_t threadThresholds[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
float latencies[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
float bandwidths[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
int maxThreads[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
```

NCCL 为每种 (Function × Algorithm × Protocol) 组合预计算:
- latency: 启动延迟 (us)
- bandwidth: 稳态带宽 (GB/s)

运行时根据数据大小选择使 `time = latency + size/bandwidth` 最小的组合。

## 6. 通信执行流程

### 6.1 一次 AllReduce 的完整路径

```
用户: ncclAllReduce(sendbuf, recvbuf, count, datatype, op, comm, stream)
         │
         ▼
    ncclEnqueueCheck()                    // enqueue.cc
         │ 检查参数、选择算法/协议
         ▼
    ncclKernelPlan 生成                    // 描述 kernel 要做什么
         │ 包含: 算法、通道分配、buffer 指针、chunk 大小
         ▼
    写入 workFifo                          // comm->workFifoBuf
         │ kernel 从 FIFO 读取工作描述
         ▼
    Launch CUDA Kernel                     // 如果需要新 kernel
         │ 或复用已在运行的 persistent kernel
         ▼
    ┌────┴────────────────────────────────────────────────┐
    │              GPU Kernel 执行                         │
    │                                                     │
    │  [Ring AllReduce 为例, 8 GPU, Channel 0]:           │
    │                                                     │
    │  Phase 1: ReduceScatter (7 步)                     │
    │    Step 0: GPU[i] 发送 chunk[i] 给 GPU[i+1]       │
    │    Step 1: GPU[i] 接收 chunk 并 reduce，发给下一个  │
    │    ...                                              │
    │    Step 6: 每个 GPU 得到 1/8 的全局 reduce 结果     │
    │                                                     │
    │  Phase 2: AllGather (7 步)                          │
    │    Step 0: GPU[i] 发送 reduced chunk 给 GPU[i+1]   │
    │    ...                                              │
    │    Step 6: 每个 GPU 得到完整的 reduce 结果          │
    │                                                     │
    │  传输方式取决于 Transport:                           │
    │    节点内: NVLink 直接 load/store (P2P)             │
    │    跨节点: 通知 CPU Proxy -> IB RDMA Write         │
    └─────────────────────────────────────────────────────┘
```

### 6.2 CPU Proxy 在跨节点通信中的作用

```
┌──────── GPU Kernel ────────┐     ┌──────── CPU Proxy Thread ────────┐
│                            │     │                                   │
│ 1. 将数据写入 send buffer  │     │                                   │
│ 2. 设置 tail 指针          │────>│ 3. 轮询 tail，发现新数据           │
│                            │     │ 4. 调用 ibv_post_send() RDMA Write│
│                            │     │ 5. 等待完成 (ibv_poll_cq)          │
│ 7. 看到 head 指针更新      │<────│ 6. 更新 head 指针                  │
│ 8. 继续下一轮              │     │                                   │
└────────────────────────────┘     └───────────────────────────────────┘

WHY 需要 CPU Proxy？
- GPU kernel 不能直接调用 IB Verbs (用户态系统调用)
- GDR 虽然让 NIC 直接读 GPU 内存，但提交 WR 仍需 CPU
- Proxy thread 持续轮询，延迟 ~2-5 us (CPU)
```

## 7. 关键环境变量与 PARAM 机制

### 7.1 NCCL_PARAM 宏 (init.cc L57-68)

```c
// init.cc L57-68: 环境变量定义
NCCL_PARAM(GroupCudaStream, "GROUP_CUDA_STREAM", NCCL_GROUP_CUDA_STREAM);
NCCL_PARAM(CheckPointers, "CHECK_POINTERS", 0);
NCCL_PARAM(CommBlocking, "COMM_BLOCKING", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(RuntimeConnect, "RUNTIME_CONNECT", 1);
NCCL_PARAM(CollnetEnable, "COLLNET_ENABLE", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(NvlsChannels, "NVLS_NCHANNELS", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(GdrCopyEnable, "GDRCOPY_ENABLE", 0);        // L130
```

NCCL_PARAM 宏展开为一个函数 `ncclParam<Name>()`，首次调用时读取 `NCCL_<NAME>` 环境变量，
之后缓存。整个 NCCL 有 100+ 个可调参数。

### 7.2 关键调优环境变量

| 变量 | 默认 | 作用 | 你的建议值 |
|------|------|------|-----------|
| NCCL_ALGO | 自动 | 强制算法 (Ring/Tree/NVLS) | 不设 (自动最优) |
| NCCL_PROTO | 自动 | 强制协议 (LL/LL128/Simple) | 不设 |
| NCCL_NCHANNELS_PER_PEER | 自动 | P2P channel 数 | 不设 |
| NCCL_NVLS_NCHANNELS | 自动 | NVLS channel 数 | 不设 |
| NCCL_BUFFSIZE | 4MB | Simple 协议 buffer 大小 | 默认或 8MB |
| NCCL_IB_HCA | 自动 | 指定 IB 网卡 | =mlx5_101:1,mlx5_102:1,... |
| NCCL_IB_GID_INDEX | 0 | IB GID 索引 (RoCE 需要) | 不适用 (你用 IB) |
| NCCL_NET_GDR_LEVEL | 自动 | GDR 启用级别 | 不设 (自动) |
| NCCL_P2P_LEVEL | 自动 | P2P 类型 (NVL/PIX/...) | 不设 |
| NCCL_CROSS_NIC | 自动 | 是否跨 NIC 通信 | 不设 (1:1 不需要) |
| NCCL_TOPO_DUMP_FILE | 无 | 导出拓扑 XML 文件 | 调试时设置 |
| NCCL_DEBUG | WARN | 日志级别 | INFO (调试时) |
| NCCL_DEBUG_SUBSYS | 全部 | 日志子系统筛选 | INIT,GRAPH,TUNING |

## 8. 与训练框架的集成

### 8.1 PyTorch / Megatron 如何调用 NCCL

```python
# PyTorch 内部:
torch.distributed.init_process_group(backend="nccl")
# -> 调用 ncclCommInitRank() 创建 comm

# AllReduce:
torch.distributed.all_reduce(tensor, group=group)
# -> ncclAllReduce(tensor.data_ptr(), ..., comm, stream)

# Megatron-LM 的 parallel_state.py 创建多个 comm:
# - TP group: 8 ranks (节点内)
# - DP group: 4 ranks (跨节点)
# - PP group: 4 ranks (跨节点)
# 每个 group 独立的 ncclComm，不同拓扑路径
```

### 8.2 NCCL 与 Megatron 并行策略的映射

```
Megatron parallel_state          NCCL comm 特性
─────────────────────────────────────────────────────────
TP group (8 ranks, 同节点)  →  Transport: P2P/NVLink
                                Algorithm: NVLS (NVSwitch)
                                Channel: 8+
                                带宽: ~450 GB/s

DP group (4 ranks, 跨节点)  →  Transport: NET/IB
                                Algorithm: Ring (大) / Tree (小)
                                Channel: 8 (multi-rail)
                                带宽: ~380 GB/s

PP group (4 ranks, 跨节点)  →  只用 ncclSend/Recv (P2P)
                                Transport: NET/IB
                                Channel: 1-2
                                带宽: ~50 GB/s (单口)
```

## 9. 源码目录结构

```
nccl/src/
├── init.cc              (3522) ★ 初始化主流程 (ncclCommInitRank, initTransportsRank)
├── group.cc             (913)  任务分组 (ncclGroupStart/End)
├── enqueue.cc           (3246) ★ 任务入队 (算法选择, kernel plan 生成)
├── collectives.cc       (398)  集合通信 API 入口
├── transport.cc         (491)  Transport 连接管理
├── channel.cc           (185)  Channel 初始化
├── bootstrap.cc         (1349) ★ 控制面网络 (TCP AllGather)
├── proxy.cc             (2151) ★ CPU Proxy 线程 (IB 操作)
├── graph/               拓扑检测与图搜索 (下一章详述)
├── transport/           各 Transport 实现 (P2P/SHM/NET)
├── device/              GPU Kernel 代码
├── include/
│   ├── comm.h           (911)  ★ ncclComm 结构体
│   ├── transport.h      (233)  Transport/PeerInfo 定义
│   ├── graph.h          (219)  拓扑图结构体
│   ├── device.h               Device 端结构体
│   └── ...
└── plugin/              网络插件接口
```

## 10. 本章总结与后续章节预告

| 维度 | 核心发现 |
|------|----------|
| 架构层次 | 7 层: API → Group → 算法 → Channel → Transport → Proxy → Kernel |
| 核心对象 | ncclComm (通信世界) + ncclChannel (并行通道) + ncclTransport (传输抽象) |
| 初始化 | 2 次 AllGather + 拓扑检测 + 图搜索 + 连接建立 |
| 算法选择 | 7 种算法 × 3 种协议，运行时 latency+bw model 选最优 |
| Transport | P2P(NVLink) > SHM > NET(IB) > CollNet，按拓扑自动选 |
| 调优入口 | 100+ NCCL_PARAM 环境变量，大部分无需手动设置 |

**后续章节**:
- N02: 拓扑检测与路径选择 (ncclTopoGetSystem, graph/topo.cc)
- N03: 传输层 (P2P/SHM/NET 实现, GDR)
- N04: 集合通信算法 (Ring/Tree/NVLS kernel 实现)
- N05: 协议与流控 (LL/LL128/Simple)
- N06: 调优与性能分析 (你的 4 机配置)
