# N06: 性能调优与环境变量 — 深度源码分析

> 核心文件: src/graph/tuning.cc, src/param.cc, src/include/nccl_tuner.h
> 参考: src/init.cc (环境变量读取), src/proxy.cc (proxy 调优参数)

## 1. 本章解决什么问题？

NCCL 有 100+ 环境变量可调, 本章:
- 分类梳理所有关键环境变量
- 结合源码解释每个参数的生效路径
- 给出你的 4×8 H100 集群的推荐配置
- 提供系统性能诊断方法论

## 2. NCCL 性能模型 (tuning.cc)

### 2.1 Tuning 模型结构

```c
// tuning.cc 中的性能模型:
// 对每种 (collective, algorithm, protocol) 维护 bandwidth 和 latency 参数

struct ncclTopoTuning {
  float bwIntra[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];  // 节点内带宽
  float bwInter[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];  // 跨节点带宽
  float lat[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];      // 基础延迟
};

// 时间预测:
float time = lat + nBytes / bw;
// 其中 bw 考虑算法系数:
//   Ring: bw_eff = bw × N/(N-1)    (理论最大)
//   Tree: bw_eff = bw / 2          (每节点全量数据)
```

### 2.2 Tuner Plugin 接口 (nccl_tuner.h)

```c
// NCCL 支持外部 tuner plugin 覆盖内置 tuning:
typedef struct {
  const char* name;
  ncclResult_t (*init)(size_t nRanks, size_t nNodes, ncclDebugLogger_t logger, void** context);
  ncclResult_t (*getCollInfo)(void* context, ncclFunc_t coll, size_t nBytes,
                              int nChannels, int nAlgos, int* algo, int* proto, int* nChannelsOut);
  ncclResult_t (*destroy)(void* context);
} ncclTuner_v3_t;

// 加载: NCCL_TUNER_PLUGIN=path/to/tuner.so
// WHY 外部 tuner?
//   不同集群拓扑最优参数不同 → 允许运营商定制
//   AWS/Azure 都提供自己的 NCCL tuner plugin
```

## 3. 环境变量分类详解

### 3.1 算法与协议控制

| 变量 | 值域 | 默认 | 源码位置 | 说明 |
|------|------|------|----------|------|
| NCCL_ALGO | Tree,Ring,NVLS,CollNet | 自动 | enqueue.cc L2123 | 强制算法 |
| NCCL_PROTO | Simple,LL,LL128 | 自动 | enqueue.cc L2123 | 强制协议 |
| NCCL_NVLS_ENABLE | 0/1 | 1 | transport/nvls.cc | 启用 NVLS |
| NCCL_COLLNET_ENABLE | 0/1 | 0 | transport/coll_net.cc | 启用 SHARP |
| NCCL_TUNER_PLUGIN | path | 无 | init.cc | 外部 tuner |

```
你的集群推荐:
  NCCL_NVLS_ENABLE=1     (H100 有 NVSwitch, 必须启用)
  NCCL_COLLNET_ENABLE=1  (如果交换机支持 SHARP)
  NCCL_ALGO/PROTO: 不设置 → 让 NCCL 自动选择
```

### 3.2 网络与 Transport 控制

| 变量 | 值域 | 默认 | 说明 |
|------|------|------|------|
| NCCL_IB_HCA | 设备列表 | 全部 | 指定使用的 IB 设备 |
| NCCL_IB_GID_INDEX | int | 0 | RoCE GID 选择 (IB 无需) |
| NCCL_IB_TC | 0-255 | 0 | Traffic Class (QoS) |
| NCCL_IB_SL | 0-15 | 0 | Service Level |
| NCCL_IB_TIMEOUT | 1-31 | 18 | QP 超时 (2^timeout × 4.096μs) |
| NCCL_IB_RETRY_CNT | 0-7 | 7 | 重试次数 |
| NCCL_IB_PCI_RELAXED_ORDERING | 0/1 | 0 | PCIe Relaxed Ordering |
| NCCL_IB_QPS_PER_CONNECTION | int | 1 | 每连接 QP 数 |
| NCCL_NET_GDR_LEVEL | LOC~SYS | PXB | GDR 启用级别 |
| NCCL_NET_GDR_READ | 0/1 | 1 | 允许 GDR read |
| NCCL_CROSS_NIC | 0/1/2 | 0 | 跨 NIC 通信 |
| NCCL_SOCKET_IFNAME | 接口名 | 全部 | Bootstrap 网卡 |

```
你的集群推荐:
  NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,...,mlx5_7:1  (明确绑定 8 个 NIC)
  NCCL_IB_PCI_RELAXED_ORDERING=1              (提升 GDR 吞吐 ~5%)
  NCCL_NET_GDR_LEVEL=PXB                     (默认即可, GPU-NIC 同 switch)
  NCCL_IB_TIMEOUT=22                         (大集群适当加大)
  NCCL_CROSS_NIC=0                           (保持 rail-optimized)
```

### 3.3 Buffer 与 Channel 控制

| 变量 | 值域 | 默认 | 说明 |
|------|------|------|------|
| NCCL_BUFFSIZE | bytes | 4MB | 每 channel buffer |
| NCCL_MAX_NCHANNELS | 1-32 | 32 | 最大 channel |
| NCCL_MIN_NCHANNELS | 1-32 | 1 | 最小 channel |
| NCCL_NTHREADS | 64-1024 | 512 | kernel 线程数 |
| NCCL_P2P_NET_CHUNKSIZE | bytes | 512KB | chunk 大小 |
| NCCL_STEPS | int | 8 | 流水线深度 |

```
你的集群推荐:
  NCCL_BUFFSIZE=8388608                (8MB, 大模型 gradient 更大)
  NCCL_MAX_NCHANNELS=8                 (= NIC 数量, 避免过多 channel 争抢 SM)
  NCCL_MIN_NCHANNELS=8                 (保持对称, 所有 NIC 都用)
  NCCL_P2P_NET_CHUNKSIZE=1048576       (1MB, NDR400 带宽大)
  
  WHY BUFFSIZE=8MB?
    默认 4MB → 8 steps × 512KB chunk → 4MB in-flight
    NDR400 带宽 50GB/s, RTT~6μs → BDP = 50×6×10^-6 = 300KB
    4MB >> BDP → 充分流水线
    但大模型单次 allreduce 数据量大 → 更大 buffer 减少 kernel 重入
```

### 3.4 P2P 与拓扑控制

| 变量 | 值域 | 默认 | 说明 |
|------|------|------|------|
| NCCL_P2P_DISABLE | 0/1 | 0 | 禁用 P2P |
| NCCL_P2P_LEVEL | LOC~SYS | 自动 | P2P 可达范围 |
| NCCL_PXN_DISABLE | 0/1 | 0 | 禁用 PXN |
| NCCL_SHM_DISABLE | 0/1 | 0 | 禁用 SHM transport |
| NCCL_TOPO_FILE | path | 自动检测 | 外部拓扑文件 |
| NCCL_TOPO_DUMP_FILE | path | 无 | 导出拓扑 |

```
你的集群推荐:
  NCCL_P2P_DISABLE=0           (NVSwitch P2P 必须启用)
  NCCL_PXN_DISABLE=0           (保持 PXN 优化可用)
  NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml  (首次运行导出, 用于debug)
```

### 3.5 调试与 Profiling

| 变量 | 值域 | 默认 | 说明 |
|------|------|------|------|
| NCCL_DEBUG | VERSION/WARN/INFO/TRACE | WARN | 日志级别 |
| NCCL_DEBUG_SUBSYS | INIT,NET,... | ALL | 子系统过滤 |
| NCCL_DEBUG_FILE | path | stderr | 日志输出文件 |
| NCCL_PROFILE | 0/1 | 0 | 启用 profiling |

```
调试时:
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
  → 输出初始化、网络连接、拓扑搜索的详细信息

性能分析时:
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL
  → 输出每次集合操作的算法选择和耗时
```

## 4. 性能诊断方法论

### 4.1 诊断工具: nccl-tests

```bash
# 安装 nccl-tests:
git clone https://github.com/NVIDIA/nccl-tests
make MPI=1 CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/lib/x86_64-linux-gnu

# 基础性能测试 (8 GPU 节点内):
mpirun -np 8 ./all_reduce_perf -b 8 -e 8G -f 2 -g 1

# 跨节点测试 (32 GPU):
mpirun -np 32 -hostfile hosts ./all_reduce_perf -b 8 -e 8G -f 2 -g 1

# 解读输出:
#  size(B)  count  type  redop  time(us)  algbw(GB/s)  busbw(GB/s)
#  8        2      float sum    28.5      0.00         0.00
#  ...
#  1G       250M   float sum    2400      416.67       729.17

# busbw = 实际bus带宽 = algbw × 2×(N-1)/N  (Ring AllReduce)
# algbw = 算法带宽 = dataSize / time
```

### 4.2 性能基准 (你的集群)

```
期望性能 (nccl-tests all_reduce_perf, 大消息):
  
  节点内 8 GPU (NVSwitch):
    busbw 期望: 380-420 GB/s (NVLS 启用时)
    如果 < 350 GB/s → 检查 NVLink 状态
    
  跨节点 32 GPU:
    busbw 期望: 180-200 GB/s  (8 NIC × 50 GB/s × 算法系数)
    algbw 期望: 40-45 GB/s/GPU
    如果 < 150 GB/s → 检查 IB/GDR 配置

  延迟 (小消息 8B):
    节点内: < 10μs
    跨节点: < 15μs
```

### 4.3 常见性能问题诊断

```
问题 1: 跨节点带宽低于预期
  诊断: NCCL_DEBUG=INFO → 检查 "Channel" 日志
    → 确认 nChannels=8 (应等于 NIC 数量)
    → 确认每个 channel 用不同 NIC
  可能原因:
    a) NIC 未全部使用 → NCCL_IB_HCA 配置错误
    b) GDR 未生效 → 检查 nvidia_peermem 模块
    c) PCIe 带宽限制 → GPU-NIC 不在同一 switch

问题 2: 节点内带宽低于预期
  诊断: nvidia-smi nvlink -s → 检查 NVLink 利用率
  可能原因:
    a) NVLink 故障 → nvidia-smi nvlink -e (error 计数)
    b) NVLS 未启用 → NCCL_DEBUG=INFO 检查 "NVLS" 字样
    c) GPU 时钟降频 → nvidia-smi -q -d PERFORMANCE

问题 3: 启动延迟过高 (训练 iter 时间波动)
  诊断: nsys profile → 看 NCCL kernel launch 时间
  可能原因:
    a) Lazy connection → 前几次 iter 慢 → warmup
    b) kernel launch → 用 CUDA Graph 或 persistent kernel
    c) proxy thread CPU 争抢 → 绑核
```

## 5. Proxy 线程绑核优化

### 5.1 CPU Affinity 设置

```c
// proxy.cc 中的 CPU affinity 逻辑:
ncclResult_t ncclProxyCreate(struct ncclComm* comm) {
  // 获取 NIC 所在 NUMA 的 CPU core
  ncclTopoGetCpuAffinity(comm->topo, comm->rank, &affinity);
  // 设置 proxy 线程的 CPU affinity
  pthread_setaffinity_np(proxyThread, sizeof(affinity), &affinity);
}

// WHY proxy 需要绑核?
// proxy 做 busy-polling → 需要独占 CPU core
// 如果 proxy 被调度到其他 core → context switch 引入数十μs 延迟
// 绑到 NIC 同 NUMA 的 core → 减少跨 NUMA 内存访问
```

### 5.2 你的集群绑核方案

```
8 GPU + 8 NIC, 假设双路 CPU (NUMA0: GPU0-3+NIC0-3, NUMA1: GPU4-7+NIC4-7):

GPU0 proxy → bind to NUMA0 core (如 core 0)
GPU1 proxy → bind to NUMA0 core (如 core 1)
...
GPU4 proxy → bind to NUMA1 core (如 core 32)
...

确认方式:
  taskset -c -p <proxy_pid>
  或 NCCL_DEBUG=INFO 查看 "Set CPU affinity" 日志
```

## 6. GDR (GPUDirect RDMA) 深入

### 6.1 GDR 生效条件

```
前提:
  1. nvidia_peermem 内核模块加载
     → lsmod | grep nvidia_peermem
  2. GPU 和 NIC 在同一 PCIe tree (PATH ≤ PXB)
     → nvidia-smi topo -m 确认
  3. NCCL_NET_GDR_LEVEL 允许
     → 默认 PXB: 同 root complex 才 GDR

数据路径对比:
  GDR:    GPU ←→ NIC (PCIe P2P DMA, 零 CPU 参与)
  Bounce: GPU → CPU(memcpy) → NIC (CPU 转发)
  
  GDR 带宽: ~25 GB/s (PCIe Gen5 x16 单向)
  Bounce 带宽: ~12 GB/s (受限于 CPU 内存带宽 + memcpy)
  GDR 延迟: ~1.5μs
  Bounce 延迟: ~5μs (多一次 D2H + H2N)
```

### 6.2 GDR 性能诊断

```bash
# 确认 GDR 生效:
NCCL_DEBUG=INFO mpirun ... ./all_reduce_perf -b 1G -e 1G
# 看输出中:
#   "NET/IB : GPU Direct RDMA Enabled for ..."  → GDR 生效
#   "NET/IB : Using bounce buffers"             → GDR 未生效

# 强制关闭 GDR 对比:
NCCL_NET_GDR_LEVEL=LOC mpirun ... ./all_reduce_perf -b 1G -e 1G
# 对比带宽差异 → 量化 GDR 收益
```

## 7. SHARP (CollNet) 配置

### 7.1 SHARP 启用条件

```
硬件: Mellanox/NVIDIA IB 交换机 (SB7800/QM8700/QM9700)
软件: HPC-X 工具包中的 SHARP daemon (sharpd)
配置: 交换机侧启用 SHARP aggregation manager (AM)

启用:
  NCCL_COLLNET_ENABLE=1
  SHARP_COLL_ENABLE_SAT=1           (SHARP Aggregation Tree)
  SHARP_COLL_LOG_LEVEL=3            (调试用)
  
你的集群: 有 SHARP 支持 (QM9700 NDR 交换机)
  启用后对 Tree AllReduce 有显著加速 (小消息延迟降低 50%)
```

### 7.2 SHARP 性能特点

```
SHARP 优势: 跨节点小消息 AllReduce 延迟极低
  无 SHARP: GPU→NIC→switch→NIC→GPU (数据过交换机不做reduce)
  有 SHARP: GPU→NIC→switch(reduce)→NIC→GPU (交换机完成reduce)

  32 GPU AllReduce 8B:
    Ring: ~20μs (30 步 × ~0.7μs)
    Tree: ~12μs (10 步 × ~1.2μs)
    SHARP: ~6μs (2 步 × ~3μs, 但只需 2 步!)

SHARP 限制:
  - 只支持 sum/min/max (不支持 custom reduce)
  - buffer 数量有限 (~256 个并发 SHARP group)
  - FP16/BF16 reduce 精度可能有差异
  - 需要 AM 健康 (单点故障 fallback 到 Tree)
```

## 8. NCCL + Megatron-LM 集成调优

### 8.1 典型训练场景参数

```bash
# 你的集群: 4 节点 × 8 GPU = 32 GPU
# 典型配置: TP=8 (节点内), PP=1, DP=4 (跨节点)

# TP 通信 (节点内 AllReduce):
#   频率高, 大消息 → NVLS 最优 → 无需干预

# DP 通信 (跨节点 AllReduce):  
#   梯度同步, 极大消息 → Ring 最优
#   推荐环境变量:
export NCCL_BUFFSIZE=8388608            # 8MB buffer
export NCCL_MAX_NCHANNELS=8             # 8 channel = 8 NIC
export NCCL_MIN_NCHANNELS=8
export NCCL_IB_PCI_RELAXED_ORDERING=1   # GDR 优化
export NCCL_NVLS_ENABLE=1               # 节点内 NVLS
export NCCL_NET_GDR_LEVEL=PXB
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1
```

### 8.2 通信-计算重叠配置

```
Megatron-LM 支持的重叠策略:
  1. TP 重叠: ReduceScatter + GEMM + AllGather
     → NCCL 异步操作 + 独立 CUDA stream
     
  2. DP 重叠: gradient AllReduce + forward of next microbatch
     → 需要 PP schedule 配合

NCCL 侧配置:
  NCCL_MAX_NCHANNELS: 控制通信占用的 SM 数
    channels=8, nWarps=16 → 8×512 threads = 32 SMs (占 24%)
    剩余 100 SMs 做计算 → 重叠效率高
    
  如果 channels 过多 (如 32) → 通信占 96 SMs → 计算被严重影响
  如果 channels 过少 (如 2) → 通信带宽不足 → 通信时间长
  
  最佳平衡: channels = NIC 数量 (你的: 8)
```

## 9. 完整推荐配置

```bash
# /workspace/nccl_env.sh — 你的 4×8 H100 集群 NCCL 环境变量

# === 基础 ===
export NCCL_DEBUG=WARN                        # 生产环境只输出警告
export NCCL_DEBUG_FILE=/tmp/nccl_%h_%p.log    # 日志到文件

# === 算法 ===
export NCCL_NVLS_ENABLE=1                     # 启用 NVLink SHARP
export NCCL_COLLNET_ENABLE=1                  # 启用 IB SHARP (如果可用)

# === 网络 ===
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7
export NCCL_NET_GDR_LEVEL=PXB
export NCCL_CROSS_NIC=0

# === Buffer/Channel ===
export NCCL_BUFFSIZE=8388608                  # 8MB
export NCCL_MAX_NCHANNELS=8
export NCCL_MIN_NCHANNELS=8
export NCCL_P2P_NET_CHUNKSIZE=1048576         # 1MB

# === 高级优化 ===
export NCCL_PXN_DISABLE=0                     # 保持 PXN
export CUDA_DEVICE_MAX_CONNECTIONS=32         # 增加 CUDA 连接数

# === SHARP (如果可用) ===
export SHARP_COLL_ENABLE_SAT=1
export SHARP_COLL_LOG_LEVEL=0
```

## 10. 设计洞察总结

| 调优维度 | 关键参数 | 你的集群最优值 | 原理 |
|----------|----------|----------------|------|
| 节点内算法 | NVLS_ENABLE | 1 | NVSwitch 硬件 reduce |
| 跨节点带宽 | MAX_NCHANNELS | 8 (= NIC数) | 充分利用所有 NIC |
| GDR 性能 | PCI_RELAXED_ORDERING | 1 | 放宽 PCIe 排序约束 |
| 网络可靠性 | IB_TIMEOUT, RETRY | 22, 7 | 大集群拥塞容错 |
| 通信-计算重叠 | NCHANNELS | 8 (不多不少) | SM 资源平衡 |
| buffer 效率 | BUFFSIZE | 8MB | 大消息流水线化 |

