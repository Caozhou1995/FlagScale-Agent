# N02: 拓扑检测与路径选择 — 深度源码分析

> 核心文件: src/graph/topo.cc (2222行), src/graph/paths.cc (1107行), src/graph/search.cc (1477行)
> 头文件: src/graph/topo.h (295行), src/include/graph.h (219行)

## 1. 本章解决什么问题？

NCCL 初始化时需要回答:
- 系统中有哪些 GPU、NIC、CPU、NVSwitch？
- 它们之间的物理连接方式和带宽是什么？
- 对于 AllReduce/AllGather 等集合操作，最优的通信路径（Ring/Tree）是什么？

这是 NCCL 中**最复杂的子系统** — topo.cc 一个文件就 2222 行。

## 2. 拓扑模型：系统图

### 2.1 节点类型 (topo.h L42-51)

```c
#define NCCL_TOPO_NODE_TYPES 10
#define GPU 0    // GPU 设备
#define PCI 1    // PCI 交换机
#define NVS 2    // NVSwitch
#define CPU 3    // CPU (实际代表 NUMA 域)
#define NIC 4    // 网卡 (物理)
#define NET 5    // 网络设备 (逻辑，含 plugin dev 号)
#define GIN 6    // GIN (GPU Interconnect Network, 新一代)
#define RMA 7    // RMA 设备
#define DEV 8    // 物理 GPU 设备 (MIG 分区的父节点)
#define CXB 9    // C2C Cross-Bridge (Grace Hopper 的 C2C 总线)
```

**WHY 区分 GPU 和 DEV？**
MIG (Multi-Instance GPU) 场景下，一个物理 GPU (DEV) 可以分成多个逻辑 GPU。
DEV 节点是分区 GPU 的父节点，代表物理设备。

### 2.2 链路类型 (topo.h L55-65)

```c
#define LINK_LOC 0    // 本地 (同一设备)
#define LINK_NVL 1    // NVLink
#define LINK_C2C 3    // C2C (Grace Hopper CPU-GPU 直连)
#define LINK_PCI 4    // PCI Express
#define LINK_SYS 9    // 跨 NUMA (QPI/UPI/xGMI)
#define LINK_NET 10   // 网络
```

### 2.3 路径类型 (graph.h L117-154) — 从好到差排序

```c
#define PATH_LOC 0    // 本地 (自身)
#define PATH_NVL 1    // NVLink 直连
#define PATH_NVB 2    // NVLink 经中间 GPU (NVBridge)
#define PATH_C2C 3    // C2C 直连 (Grace Hopper)
#define PATH_PIX 4    // 同一 PCIe switch 下
#define PATH_PXB 5    // 跨多个 PCIe switch (不过 CPU)
#define PATH_P2C 6    // GPU->C2C->CPU->PCIe->NIC
#define PATH_PXN 7    // GPU->NVLink->中间GPU->NIC (PXN)
#define PATH_PHB 8    // 跨 PCIe Host Bridge (过 CPU)
#define PATH_SYS 9    // 跨 NUMA (CPU 间互联)
#define PATH_NET 10   // 网络 (跨节点)
#define PATH_DIS 11   // 不可达
```

**关键: PATH_PXN (路径类型 7)**
```
普通跨 NUMA GPU-NIC 路径:
  GPU0 (NUMA0) -> PCIe -> CPU0 -> UPI -> CPU1 -> PCIe -> NIC (NUMA1)
  路径类型: PATH_SYS, 带宽受限于 UPI

PXN 优化路径:
  GPU0 (NUMA0) -> NVLink -> GPU4 (NUMA1) -> PCIe -> NIC (NUMA1, PIX)
  路径类型: PATH_PXN, 带宽 = min(NVLink, PCIe)

WHY PXN 更好？
  NVLink 带宽 (450 GB/s) >> UPI 带宽 (22 GB/s)
  数据先通过 NVLink 到达 NIC 所在 NUMA 的 GPU，再走本地 PCIe
```

### 2.4 核心数据结构

```c
struct ncclTopoNode {                      // topo.h L103
  int type;                                // 节点类型 (GPU/PCI/NVS/CPU/NIC/...)
  int64_t id;                              // 全局唯一 ID (编码 systemId + localId)
  union { ... } gpu/dev/net/cpu/pci;       // 类型特定数据
  int nlinks;                              // 出边数量
  struct ncclTopoLink links[576];          // L147: 出边数组 (最多576条!)
  struct ncclTopoLinkList* paths[10];      // L149: 预计算的到各类型节点的最短路径
  uint64_t used;                           // L151: 搜索时标记已使用
};

struct ncclTopoLink {                      // topo.h L71
  int type;                                // 链路类型 (LINK_NVL/LINK_PCI/...)
  float bw;                                // 带宽 (GB/s)
  struct ncclTopoNode* remNode;            // 远端节点指针
};

struct ncclTopoLinkList {                  // topo.h L80
  struct ncclTopoLink* list[NCCL_TOPO_MAX_HOPS]; // 路径上的链路序列
  int count;                               // 跳数
  float bw;                                // 路径瓶颈带宽
  int type;                                // 路径类型 (PATH_NVL/PATH_PIX/...)
};

struct ncclTopoSystem {                    // topo.h L159
  int systemId;                            // 系统编号 (多节点时区分)
  struct ncclTopoNodeSet nodes[10];        // 各类型节点集合
  float maxBw;                             // 系统最大单 channel 带宽
  float totalBw;                           // 系统总带宽
  int inter;                               // 是否跨节点
};
```

### 2.5 带宽常量 (topo.h L17-35)

```c
#define LOC_BW 5000.0                      // 本地 (无限大，象征值)
#define SM90_NVLINK_BW 20.6                // H100 每条 NVLink 带宽 (GB/s)
#define SM100_NVLINK_BW 40.1               // B200 每条 NVLink 带宽 (GB/s)
#define PCI_BW 12.0                        // PCIe Gen3 x16
#define AMD_BW 16.0                        // AMD xGMI 带宽
#define SKL_QPI_BW 10.0                    // Intel Skylake QPI
#define SRP_QPI_BW 22.0                    // Intel Sapphire Rapids UPI
#define ERP_QPI_BW 40.0                    // Intel Emerald Rapids UPI
#define NET_BW 12.0                        // 100Gbit 基准网络带宽

// topo.h L274-281: 根据 compute capability 返回 NVLink 带宽
static float ncclTopoNVLinkBw(int cudaCompCap) {
  return cudaCompCap >= 100 ? SM100_NVLINK_BW :  // B200: 40.1 GB/s/link
         cudaCompCap >= 90  ? SM90_NVLINK_BW :   // H100: 20.6 GB/s/link
         cudaCompCap >= 80  ? SM80_NVLINK_BW :   // A100: 20.0 GB/s/link
         ...
}
```

**你的 H100 (SM90): 20.6 GB/s × 18 links = 370.8 GB/s 单向 per GPU pair**
(实测 ~450 GB/s 是因为 NVSwitch 的多路径聚合效果)


## 3. 拓扑发现流程 (topo.cc)

### 3.1 入口: ncclTopoGetSystem (init.cc L965 调用)

```
initTransportsRank()
  └─ ncclTopoGetSystem(&system)         // graph.h L24
       ├─ ncclTopoFillGpu()             // 从 CUDA API 获取 GPU 信息
       │    ├─ cudaDeviceGetAttribute() → 获取 CC, 内存大小
       │    ├─ 获取 PCI bus ID → 用于定位拓扑树中位置
       │    └─ nvmlDeviceGetNvLinkState() → 检测活跃 NVLink
       ├─ ncclTopoFillNet()             // 从 net plugin 获取网卡
       │    └─ ncclNet->getProperties() → 获取 NIC 的 pciPath, speed, port
       ├─ ncclTopoConnectNodes()        // 根据 PCI 地址建立边
       │    └─ 解析 /sys/bus/pci/devices 层次关系
       └─ ncclTopoConnectNvLinks()      // 建立 NVLink 连接边
            └─ 通过 NVML 查询每条 link 的远端 GPU
```

### 3.2 PCI 拓扑树的构建 (topo.cc L680-800)

NCCL 通过 `/sys/bus/pci/devices/<BDF>/` 路径来确定设备层次:

```c
// topo.cc L120: ncclTopoCudaPath
ncclResult_t ncclTopoCudaPath(int cudaDev, char** path) {
  // 通过 /proc/driver/nvidia/gpus/<uuid>/information 获取 PCI BDF
  // 然后 realpath("/sys/bus/pci/devices/<BDF>") 得到完整路径
  // 路径深度 → 确定 PCIe 层次
}
```

**构建逻辑:**
```
解析路径: /sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0/.../0000:bb:00.0
                       └─ root      └─ switch1      └─ switch2     └─ GPU

每一层 PCI 地址 → 一个 PCI 类型节点
两个设备共享路径前缀 → 它们在同一 PCI switch 下
前缀越长 → PATH_PIX (直连); 前缀短 → PATH_PXB/PATH_PHB
```

### 3.3 NVLink/NVSwitch 探测 (topo.cc L400-500)

```c
// 对每个 GPU 的每条 NVLink (H100 有 18 条):
for (int l = 0; l < NVML_NVLINK_MAX_LINKS; l++) {
  nvmlDeviceGetNvLinkState(dev, l, &isActive);     // 链路是否激活
  nvmlDeviceGetNvLinkRemotePciInfo(dev, l, &pci);  // 远端 PCI 地址
  // 远端是 GPU → 建立 GPU-GPU 的 NVLink 边
  // 远端是 NVSwitch → 建立 GPU-NVSwitch 的 NVLink 边
  ncclTopoConnectNodes(gpuNode, remNode, LINK_NVL, nvlinkBw);
}
```

**你的 H100 集群:**
```
NVSwitch 拓扑 (8 GPU + NVSwitch):
  GPU0 ──NVL(18×20.6)──┐
  GPU1 ──NVL(18×20.6)──┤
  GPU2 ──NVL(18×20.6)──┤
  ...                   ├── NVSwitch (全连接)
  GPU7 ──NVL(18×20.6)──┘

每个 GPU 到 NVSwitch: 18 links × 20.6 GB/s = 370.8 GB/s
任意 GPU pair 通过 NVSwitch: 全带宽直连
```

## 4. BFS 最短路径计算 (paths.cc)

### 4.1 核心: ncclTopoSetPaths (paths.cc L38-130)

```c
static ncclResult_t ncclTopoSetPaths(struct ncclTopoNode* baseNode,
                                     struct ncclTopoSystem* system) {
  // 经典 BFS — 从 baseNode 出发，找到到所有其它节点的最短(最高带宽)路径
  
  struct ncclTopoNodeList nodeList, nextNodeList;
  nodeList.count = 1; nodeList.list[0] = baseNode;
  
  // baseNode 到自己: PATH_LOC, bw=LOC_BW(5000)
  struct ncclTopoLinkList* basePath = baseNode->paths[baseNode->type] + ...;
  basePath->count = 0; basePath->bw = LOC_BW; basePath->type = PATH_LOC;
  
  while (nodeList.count) {               // BFS 层次遍历
    nextNodeList.count = 0;
    for (int n = 0; n < nodeList.count; n++) {
      struct ncclTopoNode* node = nodeList.list[n];
      struct ncclTopoLinkList* path = node->paths[baseNode->type] + baseIdx;
      
      for (int l = 0; l < node->nlinks; l++) {    // 遍历出边
        struct ncclTopoLink* link = node->links + l;
        struct ncclTopoNode* remNode = link->remNode;
        
        // 计算经过当前边到达 remNode 的带宽
        float bw = std::min(path->bw, link->bw);  // 瓶颈带宽
        int newType = cyclePathType(path, link);   // 计算路径类型
        
        // 更新: 如果新路径更好 (类型更好或同类型带宽更高)
        struct ncclTopoLinkList* remPath = remNode->paths[baseNode->type]+baseIdx;
        if (newType < remPath->type ||
            (newType == remPath->type && bw > remPath->bw)) {
          // 更新最短路径
          remPath->bw = bw;
          remPath->type = newType;
          remPath->count = path->count + 1;
          // 记录路径上的每一跳
          for (int i = 0; i < path->count; i++) remPath->list[i+1] = path->list[i];
          remPath->list[0] = link;   // 注意: 反向记录
        }
      }
    }
    memcpy(&nodeList, &nextNodeList, sizeof(nodeList));
  }
}
```

**WHY BFS 而不是 Dijkstra？**
路径质量由**类型**决定 (NVL > PIX > PXB > PHB > SYS)，带宽只是同类型内的次要排序。
BFS 天然按"跳数"展开 → 类型单调不递减 → 第一次到达就是最优路径。

### 4.2 路径类型推导 (paths.cc, ncclTopoSetPaths 内部)

```
路径类型升级规则:
  当前路径类型 + 新边类型 → 新路径类型

  LOC + NVL → NVL     (从自身出发走 NVLink)
  NVL + NVL → NVB     (NVLink 经中间 GPU, NVBridge)
  LOC + PCI → PIX     (单跳 PCIe)
  PIX + PCI → PXB     (多跳 PCIe, 同 root complex)
  PXB + CPU → PHB     (过 CPU/PCIe Host Bridge)
  PHB + SYS → SYS     (跨 NUMA)
```

### 4.3 ncclTopoComputePaths — 顶层调用 (paths.cc L721)

```c
ncclResult_t ncclTopoComputePaths(struct ncclTopoSystem* system, struct ncclComm* comm) {
  // 对每种类型的每个节点调用 ncclTopoSetPaths
  for (int t = 0; t < NCCL_TOPO_NODE_TYPES; t++) {
    for (int n = 0; n < system->nodes[t].count; n++) {
      NCCLCHECK(ncclTopoSetPaths(system->nodes[t].nodes + n, system));
    }
  }
  // 之后所有节点对之间都有了最短路径信息
}
```

## 5. 图搜索：找最优通信模式 (search.cc)

### 5.1 核心: ncclTopoCompute (graph.h L190)

```c
ncclResult_t ncclTopoCompute(struct ncclTopoSystem* system, struct ncclTopoGraph* graph);
// 对每种拓扑模式 (ring/tree/nvls/collnet) 搜索最优 channel 配置
```

搜索过程：
```
输入:
  - graph->pattern: 要搜索的模式 (RING=4, TREE=3, NVLS=5, ...)
  - graph->minChannels / maxChannels: channel 数量范围
  
输出:
  - graph->nChannels: 找到的 channel 数
  - graph->bwIntra: 节点内带宽
  - graph->bwInter: 节点间带宽
  - graph->intra[]: 节点内 GPU 排列顺序
  - graph->inter[]: 节点间连接方式
```

### 5.2 Ring 搜索策略

```
Ring Pattern (8 GPU NVSwitch 示例):
  目标: 找到经过所有 GPU 的环路, 最大化瓶颈带宽

  NVSwitch 场景:
    任意排列都可以达到 NVLink 满带宽
    NCCL 选择: 考虑 NIC affinity 的最优排列
    
  无 NVSwitch 场景:
    需要找 Hamiltonian cycle 使得最短边带宽最大
    搜索空间 = N! (GPU 排列), 用启发式剪枝
```

### 5.3 Tree 搜索策略

```
Tree Pattern (跨节点):
  balanced_tree: 
    GPU0(parent+child0) ──NIC──┐
    GPU1(child1) ─────NVLink───┘  (节点内)
    
  WHY balanced_tree 而非纯 tree？
    将 NIC 流量分散到多个 GPU → 避免单 GPU 成为带宽瓶颈
    
  split_tree:
    GPU0(parent) ──NIC──  ← 发送
    GPU1(child0+child1) ──NVLink── ← 聚合  (节点内)
```

### 5.4 NVLS (NVLink SHARP) 搜索

```
NVLS Pattern:
  利用 NVSwitch 的硬件 Reduction 能力 (NVLS = NVLink SHARP)
  所有 GPU 同时写入 NVSwitch 的 multicast 地址
  NVSwitch 在硬件上完成 reduce → 广播结果

  搜索: 验证所有 GPU 都连接到同一组 NVSwitch
  如果成立 → 启用 NVLS 算法 (比 Ring 更优)
```

## 6. P2P 路径判定 (paths.cc L298-435)

### 6.1 ncclTopoCheckP2p

```c
ncclResult_t ncclTopoCheckP2p(struct ncclComm* comm, struct ncclTopoSystem* system,
                              int rank1, int rank2, int* p2p, int* read,
                              int* intermediateRank, int* cudaP2p) {
  // 决策:
  // 1. GPU pair 之间路径类型是什么?
  // 2. 是否可以走 P2P (CUDA IPC)?
  // 3. 是否应该用 GDR Read (vs Write)?
  // 4. 如果路径太差, 是否需要 PXN 中间节点?
  
  int pathType = system->nodes[GPU].nodes[gpu1].paths[GPU][gpu2].type;
  
  if (pathType <= PATH_PXB) {
    *p2p = 1;        // PCIe P2P 可行
    *cudaP2p = 1;
  } else if (pathType <= PATH_PHB && pxnLevel) {
    *p2p = 1;        // 启用 PXN proxy
    *intermediateRank = findPxnIntermediate(comm, rank1, rank2);
  }
}
```

### 6.2 PXN (PCIe x NVLink) 代理机制

```
问题: GPU0 (NUMA0) 要发送到 NIC (NUMA1), 但 PCIe 跨 NUMA 带宽只有 22 GB/s (UPI)

PXN 解决方案:
  GPU0 ──NVLink(370GB/s)──> GPU4 (NUMA1) ──PCIe(32GB/s)──> NIC (NUMA1)
  
  GPU4 作为 proxy:
    1. GPU0 通过 NVLink 把数据写入 GPU4 的 buffer
    2. GPU4 的 proxy thread 将 buffer 数据通过本地 PCIe 发送到 NIC
    
  收益: 避免 UPI 瓶颈, 有效带宽 = min(NVLink, PCIe) = 32 GB/s >> 22 GB/s
```

## 7. GDR (GPUDirect RDMA) 判定 (paths.cc L468-550)

```c
ncclResult_t ncclTopoCheckGdr(struct ncclTopoSystem* system, int rank,
                              int64_t netId, int read, enum ncclTopoGdrMode* gdrMode) {
  // 判断 GPU-NIC 之间是否适合启用 GPUDirect RDMA
  
  int pathType = gpuNode->paths[NET][netIndex].type;
  
  // GDR 适用条件:
  // 1. GPU-NIC 路径 ≤ PATH_PXB (同一 PCIe root complex 下)
  // 2. 或 C2C 路径 (Grace Hopper)
  // 3. NIC 支持 GPUDirect (peer_access 属性)
  
  if (pathType <= PATH_PXB) {
    *gdrMode = ncclTopoGdrModePci;    // PCIe P2P GDR
  } else if (pathType == PATH_C2C) {
    *gdrMode = ncclTopoGdrModeDefault; // C2C GDR
  }
  
  // WHY 只在 PATH_PXB 以内启用?
  // 跨 Host Bridge (PHB/SYS) 的 GDR 需要 CPU 中转 → 性能反而更差
  // 此时用 bounce buffer (先到 CPU 内存再到 NIC) 更好
}
```

## 8. 与 initTransportsRank 的集成 (init.cc L965-1200)

```
ncclCommInitRank()
  └─ initTransportsRank()
       ├─ ncclTopoGetSystem()           // [本章] 建拓扑图
       ├─ ncclTopoComputePaths()        // [本章] BFS 最短路径
       ├─ ncclTopoSearchInit()          // 初始化搜索参数
       ├─ ncclTopoCompute(ringGraph)    // [本章] 搜索最优 Ring
       ├─ ncclTopoCompute(treeGraph)    // [本章] 搜索最优 Tree  
       ├─ ncclTopoCompute(nvlsGraph)    // [本章] 搜索 NVLS
       ├─ ncclTopoCompute(collnetGraph) // 搜索 CollNet (SHARP)
       ├─ ncclTopoPreset()              // 根据搜索结果设置 channel
       ├─ allGather(topoRanks)          // 交换各 rank 的拓扑排名
       └─ ncclTopoPostset()             // 最终连接所有 channel
```

## 9. 关键环境变量控制

| 环境变量 | 作用 | 默认值 |
|---------|------|--------|
| NCCL_TOPO_FILE | 外部拓扑 XML 文件 | 自动检测 |
| NCCL_TOPO_DUMP_FILE | 导出检测到的拓扑 | 无 |
| NCCL_P2P_LEVEL | P2P 通信级别控制 | 自动 |
| NCCL_P2P_DISABLE | 禁用 P2P | 0 |
| NCCL_SHM_DISABLE | 禁用共享内存 | 0 |
| NCCL_NET_GDR_LEVEL | GDR 启用级别 | 自动 |
| NCCL_PXN_DISABLE | 禁用 PXN | 0 |
| NCCL_CROSS_NIC | 允许跨 NIC 通信 | 0 |
| NCCL_MIN_NCHANNELS | 最小 channel 数 | 自动 |
| NCCL_MAX_NCHANNELS | 最大 channel 数 | 自动 |

## 10. 你的集群拓扑分析

```
4 节点 × 8 GPU/节点, 8× CX7 NDR400 IB 网卡

节点内拓扑 (NCCL 视角):
  ┌──────────────────────── NVSwitch ────────────────────────┐
  │  ↕NVL(18×20.6)  ↕NVL  ↕NVL  ↕NVL  ↕NVL  ↕NVL  ↕NVL  ↕NVL │
  │ GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7         │
  │  ↕PCIe  ↕PCIe  ↕PCIe  ↕PCIe  ↕PCIe  ↕PCIe  ↕PCIe  ↕PCIe  │
  │ NIC0  NIC1  NIC2  NIC3  NIC4  NIC5  NIC6  NIC7         │
  └──────────────────────────────────────────────────────────┘

NCCL 路径判定:
  GPU-GPU: PATH_NVL (所有 pair 经 NVSwitch, 20.6×18=370.8 GB/s)
  GPU-NIC: PATH_PIX (每个 GPU 有 local 绑定 NIC, 32 GB/s PCIe Gen5)
  
节点间 (IB fabric):
  路径: GPU → PCIe → NIC → IB(NDR400=50GB/s) → switch → NIC → PCIe → GPU
  NCCL 使用 Rail-Optimized topology:
    8 个 channel, 每个 channel 用对应 rail 的 NIC
    跨节点带宽: 8 × 50 GB/s = 400 GB/s 双向

NCCL 算法选择预测:
  小消息: Tree (低延迟)
  大消息: 
    节点内 AllReduce → NVLS (NVSwitch 硬件 reduce)
    跨节点 AllReduce → Ring (8-channel, 50 GB/s/channel)
```

## 11. 设计洞察总结

| 设计点 | 动机 | 收益 |
|--------|------|------|
| BFS 而非 Dijkstra | 路径类型优先于带宽 | 简单+正确 |
| 预计算所有路径 | 运行时 O(1) 查询 | 集合操作零开销 |
| PXN 代理 | 跨 NUMA 避免 UPI 瓶颈 | 带宽提升 50%+ |
| 多 Pattern 搜索 | Ring/Tree/NVLS 各有适用场景 | 自动选最优 |
| NVSwitch 特判 | all-to-all 连接简化搜索 | 避免 NP-hard |
| Rail-Optimized | 利用对称拓扑均衡 NIC 流量 | 无网卡空闲 |

