# 第0章：集群通信拓扑检测方法论

## 1. 为什么需要拓扑检测？

分布式训练的通信性能取决于**硬件拓扑**：
- GPU 之间怎么连？（NVLink / PCIe / NVSwitch）
- GPU 和网卡怎么对应？（NUMA 亲和性）
- 节点之间经过几跳交换机？（fat-tree 层级）
- 带宽瓶颈在哪？（网卡带宽 vs 交换机带宽 vs NVLink 带宽）

NCCL 自身也会做拓扑检测（后续源码分析会详述），但我们需要**手动检测**来：
1. 验证 NCCL 的选择是否最优
2. 定位性能瓶颈
3. 指导并行策略配置（TP 放节点内 NVLink，DP 走节点间 IB）

## 2. 检测层次与工具总览

```
┌───────────────┬─────────────────────┬─────────────────────────┐
│ 层次          │ 检测什么            │ 工具                    │
├───────────────┼─────────────────────┼─────────────────────────┤
│ GPU 层        │ 型号/显存/数量      │ nvidia-smi              │
│ GPU 互联层    │ NVLink/NVSwitch     │ nvidia-smi topo/nvlink  │
│ PCIe 拓扑层   │ GPU-NIC NUMA 亲和   │ lspci + sysfs          │
│ 网卡层        │ IB/RoCE 型号/速率   │ ibv_devinfo + sysfs    │
│ 链路层        │ 端口状态/错误计数   │ perfquery              │
│ 交换机层      │ 几级交换/过载比     │ ibnetdiscover          │
│ 端到端验证    │ 实际带宽/延迟       │ nccl-tests / ib_*_bw   │
└───────────────┴─────────────────────┴─────────────────────────┘
```

## 3. 逐层检测详解

### 3.1 GPU 层 - nvidia-smi

**目标**: 确认 GPU 型号、数量、PCI 地址

```bash
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total --format=csv,noheader
```

**你的实际结果**:
```
0, NVIDIA H100 80GB HBM3, 00000000:18:00.0
1, NVIDIA H100 80GB HBM3, 00000000:2A:00.0
2, NVIDIA H100 80GB HBM3, 00000000:3A:00.0
3, NVIDIA H100 80GB HBM3, 00000000:5D:00.0
4, NVIDIA H100 80GB HBM3, 00000000:9A:00.0
5, NVIDIA H100 80GB HBM3, 00000000:AB:00.0
6, NVIDIA H100 80GB HBM3, 00000000:BA:00.0
7, NVIDIA H100 80GB HBM3, 00000000:DB:00.0
```

**解读**: 8 张 H100 80GB，PCI 地址分两组 (0x18-0x5D 和 0x9A-0xDB)，
暗示两个 NUMA node 各挂 4 张 GPU。

### 3.2 GPU 互联层 - nvidia-smi topo

**目标**: 确认 GPU 之间的互联方式

```bash
nvidia-smi topo -m
```

**关键标记含义**:

| 标记 | 含义 | 典型带宽 |
|------|------|----------|
| NV18 | NVLink 18条链路 | 18x25GB/s = 450GB/s 单向 |
| NV12 | NVLink 12条 | A100 级别, 300GB/s 单向 |
| PIX | 同一 PCIe switch 下 | ~64GB/s (PCIe Gen5 x16) |
| PHB | 同一 PCIe Host Bridge | ~64GB/s |
| NODE | 同一 NUMA, 不同 PCIe 树 | ~64GB/s, 延迟更高 |
| SYS | 跨 NUMA (跨 CPU socket) | ~64GB/s, 需过 UPI/QPI |

**你的结果**: 所有 GPU 对之间都是 NV18 -- 说明有 NVSwitch 全互联。
这是 DGX H100 / HGX H100 的标准配置。每对 GPU 间带宽约 900 GB/s 双向。

```bash
# 进一步确认 NVLink 带宽
nvidia-smi nvlink --status
# 输出每条 link 的速率 (26.562 GB/s per link)
# 总计: 18 links x 26.562 = 478 GB/s 单向
```

**WHY NVSwitch 重要？**
- 无 NVSwitch: GPU 只能与相邻 GPU 通过 NVLink 点对点通信
- 有 NVSwitch: 任意 GPU 对之间都有全带宽直连，AllReduce 可以用 NVLink 完成
- NCCL 检测到 NVSwitch 后会选择 NVLS (NVLink SHARP) 算法，比 ring 快很多

### 3.3 PCIe/NUMA 拓扑 - GPU 与网卡的亲和性

**目标**: 确认哪张网卡离哪张 GPU 最近

**为什么重要？** GPU 通过网卡发数据时，数据路径是：
```
GPU -> (PCIe) -> CPU Memory -> (PCIe) -> NIC -> 网络
      或
GPU -> (GPUDirect RDMA, 绕过CPU) -> NIC -> 网络
```

最优情况：GPU 和 NIC 在同一 PCIe switch 下 (PIX)，
GPUDirect RDMA 可以让数据直接从 GPU 显存 -> NIC，不经过 CPU。

**检测方法 1: nvidia-smi topo -m 的 NIC 列**
```bash
nvidia-smi topo -m
# 看 GPU 行与 NIC 列的交叉值:
#   PIX = 最优 (同 PCIe switch)
#   NODE = 次优 (同 NUMA, 不同 PCIe 树)
#   SYS = 最差 (跨 NUMA)
```

**检测方法 2: 通过 sysfs 手动匹配 PCI 地址**
```bash
# 列出所有 IB 网卡的 PCI 地址和 NUMA 节点
for dev in /sys/class/infiniband/*/device; do
  nic=$(echo $dev | grep -oP 'mlx5_\d+')
  pci=$(readlink -f $dev | grep -oP '[0-9a-f]+:[0-9a-f]+\.\d+' | tail -1)
  numa=$(cat $dev/numa_node 2>/dev/null)
  echo "$nic -> PCI $pci, NUMA $numa"
done
```

**你的结果**:
```
NUMA 0:                                  NUMA 1:
  GPU0 (18:00.0) <-> mlx5_101 (19:00.0) PIX    GPU4 (9A:00.0) <-> mlx5_105 (9B:00.0) PIX
  GPU1 (2A:00.0) <-> mlx5_102 (29:00.0) PIX    GPU5 (AB:00.0) <-> mlx5_106 (AA:00.0) PIX
  GPU2 (3A:00.0) <-> mlx5_103 (3B:00.0) PIX    GPU6 (BA:00.0) <-> mlx5_107 (BB:00.0) PIX
  GPU3 (5D:00.0) <-> mlx5_104 (5C:00.0) PIX    GPU7 (DB:00.0) <-> mlx5_108 (DA:00.0) PIX
                  + mlx5_201 (53:00.0) 第9口
```

**解读**:
- 每张 GPU 有一张专属 NIC，且在同一 PCIe switch 下 (PIX)
- 这是最优拓扑：NCCL 可以为每张 GPU 分配独立网络通道 (8-rail)
- 第 9 张网卡 mlx5_201 可能是管理口或备用口
- PCI 地址相邻 (18:00 vs 19:00) 证明在同一 PCIe switch 下


### 3.4 网卡层 - IB/RoCE 设备信息

**目标**: 确认网卡型号、固件版本、端口速率、链路层类型

**工具 1: lspci (PCI 设备枚举)**
```bash
lspci | grep -i "mellanox\|infiniband\|connectx"
```

**你的结果**:
```
19:00.0 Infiniband controller: Mellanox Technologies MT2910 Family [ConnectX-7]
29:00.0 Infiniband controller: Mellanox Technologies MT2910 Family [ConnectX-7]
... (共 9 张 ConnectX-7)
```

**解读**: ConnectX-7 是目前最高端的 IB 网卡，支持 NDR 400Gb/s。

**工具 2: sysfs 详细信息**
```bash
# 查看每张网卡的固件、端口状态、速率
for dev in /sys/class/infiniband/*/; do
  devname=$(basename "$dev")
  fw=$(cat "$dev/fw_ver" 2>/dev/null)
  for port in "$dev/ports/"/*/; do
    state=$(cat $port/state 2>/dev/null)
    rate=$(cat $port/rate 2>/dev/null)
    link=$(cat $port/link_layer 2>/dev/null)
    echo "$devname: FW=$fw, State=$state, Rate=$rate, Layer=$link"
  done
done
```

**你的结果**:
```
mlx5_101: FW=28.39.3004, State=4: ACTIVE, Rate=400 Gb/sec (4X NDR), Layer=InfiniBand
mlx5_102: FW=28.39.3004, State=4: ACTIVE, Rate=400 Gb/sec (4X NDR), Layer=InfiniBand
... (全部 9 张，状态相同)
```

**关键指标解读**:
- `State=4: ACTIVE`: 链路正常 UP (1=Down, 2=Init, 3=Armed, 4=Active)
- `Rate=400 Gb/sec (4X NDR)`: 4 条 NDR lane，每条 100Gb/s，总计 400Gb/s
- `Layer=InfiniBand`: 使用 IB 协议 (vs RoCE 使用以太网)

**工具 3: ibv_devinfo (更详细的设备能力)**
```bash
ibv_devinfo    # 列出所有设备能力
ibv_devinfo -d mlx5_101 -v  # 单个设备详细信息
```

输出中关注:
- `max_qp`: 最大 QP 数量 (131072) -- 决定能开多少并行通信通道
- `max_qp_wr`: 每个 QP 的 WR 深度 (32768) -- 影响通信 pipeline 深度
- `max_mr_size`: 最大内存注册区域 -- GPUDirect RDMA 需要
- `device_cap_flags`: 设备能力标志

**IB vs RoCE 的区别**:

| 维度 | InfiniBand | RoCE v2 |
|------|-----------|---------|
| 链路层 | IB L2 | Ethernet |
| 传输层 | IB Transport | UDP/IP |
| 拥塞控制 | Credit-based (无丢包) | ECN + PFC (可能丢包) |
| 延迟 | ~1 us | ~2-3 us |
| 需要交换机 | IB 专用交换机 | 普通以太网交换机 |
| NCCL 表现 | 更稳定、带宽更可预测 | 需要调 PFC/ECN 参数 |

你的集群用 InfiniBand，这是最优选择 -- credit-based flow control 保证零丢包。

### 3.5 链路层 - 端口性能计数器

**目标**: 检查是否有链路错误、丢包、CRC 错误

```bash
# 安装 infiniband-diags (如果没有)
apt-get install -y infiniband-diags

# 查看本地端口计数器
perfquery
```

**关注的错误计数器**:
```
PortRcvErrors............: 0      # 接收错误 (应该为 0)
PortRcvRemotePhysicalErrors: 0   # 远端物理错误
SymbolErrorCounter.......: 0      # 符号错误 (线缆/光模块问题)
LinkErrorRecoveryCounter.: 0      # 链路恢复次数
LinkDownedCounter........: 0      # 链路断开次数
PortXmitDiscards.........: 0      # 发送丢弃 (拥塞指标)
VL15Dropped..............: 0      # 管理包丢弃
```

**如果看到非零错误**: 可能是光模块老化、线缆弯折、交换机端口故障。
即使少量错误也会导致 NCCL 性能抖动（因为要重传）。

### 3.6 交换机层 - ibnetdiscover (核心！)

**目标**: 发现整个 IB fabric 的交换机拓扑

```bash
ibnetdiscover
```

这个命令从本节点出发，通过 SMP (Subnet Management Protocol) 遍历整个网络，
输出所有交换机和主机的连接关系。

**输出格式解读**:
```
Switch  65 "S-fc6a1c030057aa00"  # "401-M01-40U-SU7-C-leaf-49"
[1]  "H-58a2e103000748ec"[1]    # "gpu01 mlx5_gdr_0" 4xNDR    <- 主机端口
[34] "S-fc6a1c0300466900"[49]   # "402-H18-38U-C-spine-02"    <- 上联到 spine
```

含义:
- `Switch 65`: 65 端口交换机 (QM9700 NDR)
- `"leaf-49"`: 这是第 49 号 leaf 交换机
- `[1] "H-..."`: 端口 1 连接到一个 Host (主机网卡)
- `[34] "S-..."`: 端口 34 连接到另一个 Switch (spine)
- `4xNDR`: 4 条 NDR lane = 400 Gb/s

**你的 fabric 拓扑**:
```
发现的设备:
- 64 个 Leaf 交换机 (命名: leaf-01 到 leaf-64)
- 32 个 Spine 交换机 (命名: spine-01 到 spine-32)
- 2117 个 HCA (主机网卡)

每个 Leaf 交换机:
- 33 个端口连接主机 (下联)
- 31 个端口连接 Spine (上联)
- 1 个管理端口

每个 Spine 交换机:
- 64 个端口连接 Leaf (每个 Leaf 1-2 条)
```

### 3.7 Fat-Tree 拓扑分析

**什么是 Fat-Tree？**
```
            Spine 层 (32 台交换机)
           /  |  |  |  |  |  \
          /   |  |  |  |  |   \
    Leaf-1  Leaf-2 ... Leaf-63  Leaf-64
     |  |    |  |              |  |
   HCA HCA  HCA HCA         HCA HCA
```

**过载比 (Oversubscription Ratio) 计算**:
```
每个 Leaf:
  下联带宽 = 33 端口 x 400 Gb/s = 13.2 Tb/s
  上联带宽 = 31 端口 x 400 Gb/s = 12.4 Tb/s
  过载比 = 13.2 / 12.4 = 1.065 : 1
```

**解读**: 过载比 1.065:1 接近 1:1，这是几乎无过载的网络。
意味着即使所有主机同时全速通信，交换机也不会成为瓶颈。

**跳数分析**:
```
同一 Leaf 下的两台机器: 1 跳 (Host -> Leaf -> Host)
不同 Leaf 下的两台机器: 3 跳 (Host -> Leaf -> Spine -> Leaf -> Host)
```

延迟差异: 1 跳 ~1us vs 3 跳 ~2-3us (对大包传输影响不大，对小包 latency 有影响)


## 4. 端到端性能验证

### 4.1 IB 原始带宽测试

```bash
# 安装 perftest (IB 性能测试工具)
apt-get install -y perftest

# 节点间点对点带宽 (需要两个节点配合)
# 节点 A (server):
ib_write_bw -d mlx5_101 --report_gbits

# 节点 B (client):
ib_write_bw -d mlx5_101 <节点A_IP> --report_gbits

# 预期结果: ~390 Gb/s (接近 400Gb/s 线速)
```

**如果带宽远低于预期**: 检查 MTU (应为 4096)、PFC 配置、交换机 buffer。

### 4.2 NCCL 集合通信测试 (nccl-tests)

```bash
# 编译 nccl-tests
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1 MPI_HOME=/opt/hpcx/ompi NCCL_HOME=/path/to/nccl CUDA_HOME=/usr/local/cuda

# 单机 8 卡 AllReduce
./build/all_reduce_perf -b 1M -e 1G -f 2 -g 8

# 多机 AllReduce (通过 mpirun)
mpirun -np 32 --hostfile hosts \
  -x NCCL_DEBUG=WARN \
  -x NCCL_IB_GID_INDEX=3 \
  ./build/all_reduce_perf -b 1M -e 1G -f 2
```

**预期结果 (8xH100 单机)**:
```
#       size    time   algbw   busbw
   1048576    0.01    92.6    161.9   # 1MB
  67108864    0.09   709.4    620.7   # 64MB
 134217728    0.18   730.1    639.0   # 128MB
1073741824    1.41   760.5    665.4   # 1GB
```

algbw = 算法带宽 (总数据量/时间)
busbw = 总线带宽 (考虑 ring 因子后的有效带宽)

**H100 8 卡 NVSwitch 理论上限**: busbw ~900 GB/s (NVLink 带宽)

**预期结果 (4 机 32 卡)**:
```
# 跨节点部分受限于 IB 带宽
# 8 口 x 400Gb/s = 3.2Tb/s = 400GB/s 每节点
# AllReduce busbw 预期: ~350-380 GB/s (接近 IB 线速)
```

### 4.3 GPUDirect RDMA 验证

```bash
# 检查 GDR (GPU Direct RDMA) 是否可用
cat /proc/driver/nvidia/params | grep -i peer
# 应该看到 NVreg_RegistryDwords ... PeerMappingOverride=0x1

# 或者通过 NCCL debug log
NCCL_DEBUG=INFO ./build/all_reduce_perf -b 128M -e 128M -g 8 2>&1 | grep -i "gdrdma\|GPU Direct"
# 应该看到: NET/IB : Using [0]mlx5_101:1/IB/... ; GDR: enabled
```

**WHY GPUDirect RDMA 重要？**
```
无 GDR:  GPU -> PCIe -> CPU内存 -> PCIe -> NIC  (额外一次内存拷贝)
有 GDR:  GPU -> PCIe -> NIC                     (直接传输)

延迟减少: ~5us -> ~2us
带宽提升: 因为省了一次 PCIe 往返
```

## 5. NCCL Debug 日志分析

### 5.1 获取 NCCL 拓扑信息

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING \
  python -c "import torch.distributed; ..." 2>&1 | head -100
```

**关注的输出**:
```
# 拓扑检测
NCCL INFO Trees [0] ... -1/-1/-1->0->1
NCCL INFO Channel 00/08 : 0[0] -> 1[1] via P2P/NVLink

# 算法选择
NCCL INFO Algorithm: Ring / Tree / NVLink SHARP / CollNet
NCCL INFO Protocol: Simple / LL / LL128

# 传输方式
NCCL INFO NET/IB : Using [0]mlx5_101:1/IB ; GDR: enabled
NCCL INFO Channel 00 : net send/recv via NIC 0 (mlx5_101)
```

### 5.2 关键环境变量（调试用）

```bash
# 打印详细拓扑
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING,NET

# 导出拓扑文件 (NCCL 检测到的完整拓扑)
NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml

# 指定拓扑文件 (调试时用预制拓扑)
NCCL_TOPO_FILE=/path/to/topo.xml
```

## 6. 你的集群完整拓扑图

```
                    ┌─────────────────────────────────────────────┐
                    │        32x Spine (QM9700 NDR 400G)          │
                    │  spine-01, spine-02, ... spine-32           │
                    │  每台 64 口，全部连 leaf                      │
                    └──────────┬──────────────┬───────────────────┘
                               │   31条上联    │
                    ┌──────────┴──┐    ┌──────┴────────┐
                    │  64x Leaf    │    │               │
                    │  leaf-01~64  │    │  ...          │
                    │  65口/台     │    │               │
                    └──┬──┬──┬──┬─┘    └───────────────┘
                       │  │  │  │  33条下联
              ┌────────┴──┴──┴──┴─────────────────────────────┐
              │              你的 4 台节点                      │
              │                                               │
    ┌─────────┴─────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
    │      Node 1       │  │    Node 2      │  │    Node 3      │  │    Node 4      │
    │                   │  │                │  │                │  │                │
    │  ┌─NVSwitch────┐  │  │  ┌─NVSwitch──┐ │  │  (同结构)      │  │  (同结构)      │
    │  │ GPU0 GPU1   │  │  │  │           │ │  │                │  │                │
    │  │ GPU2 GPU3   │  │  │  │   8xH100  │ │  │                │  │                │
    │  │ GPU4 GPU5   │  │  │  │  NV18全联  │ │  │                │  │                │
    │  │ GPU6 GPU7   │  │  │  │           │ │  │                │  │                │
    │  └─────────────┘  │  │  └───────────┘ │  │                │  │                │
    │                   │  │                │  │                │  │                │
    │  GPU0-mlx5_101 ───┼──┼────────────────┼──┼──> IB Leaf     │  │                │
    │  GPU1-mlx5_102 ───┼──┼────────────────┼──┼──> IB Leaf     │  │                │
    │  ...              │  │  ...           │  │                │  │                │
    │  GPU7-mlx5_108 ───┼──┼────────────────┼──┼──> IB Leaf     │  │                │
    │  (8x400G = 3.2T) │  │  (8x400G)     │  │  (8x400G)     │  │  (8x400G)     │
    └───────────────────┘  └────────────────┘  └────────────────┘  └────────────────┘

带宽层次:
  节点内 GPU-GPU (NVLink):  900 GB/s 双向 (任意 pair)
  节点间单口 (IB NDR):      50 GB/s 单向 (400Gb/s)
  节点间总计 (8-rail):      400 GB/s 单向 (3.2Tb/s)
  比值: 节点内/节点间 = 900/400 = 2.25x
```

## 7. 基于拓扑的训练配置建议

### 7.1 并行策略与硬件映射

| 并行维度 | 推荐位置 | 通信量 | 利用的互联 |
|----------|----------|--------|-----------|
| TP (张量) | 节点内 8 卡 | 大,延迟敏感 | NVLink 900GB/s |
| PP (流水线) | 跨节点 | 小 (P2P) | IB 单口 50GB/s |
| DP (数据) | 跨节点 | 中 (AllReduce) | IB 8-rail 400GB/s |
| EP (专家) | 跨节点 | 大 (All2All) | IB 8-rail 400GB/s |
| CP (上下文) | 节点内或跨节点 | 中 (Ring) | NVLink 或 IB |

### 7.2 WHY TP 必须放节点内？

TP AllReduce 在每个 Transformer 层的 Attention 和 MLP 后各执行一次，
对于 seq_len=4096, hidden=8192 的模型:
- 单次 AllReduce 数据量: 4096 x 8192 x 2B = 64 MB
- 每层 2 次 = 128 MB/层
- 80 层模型 = 10 GB/step

NVLink 完成 64MB AllReduce: 64MB / 900GB/s = 0.07ms
IB 完成 64MB AllReduce: 64MB / 400GB/s = 0.16ms + 额外延迟

延迟差 2-3x，且每层都有，累积后严重影响吞吐。

## 8. 检测命令速查表

```bash
# === 一键检测脚本 ===

echo "=== GPU ==="
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader

echo "=== GPU Interconnect ==="
nvidia-smi topo -m

echo "=== NVLink ==="
nvidia-smi nvlink --status | head -20

echo "=== IB NICs ==="
for dev in /sys/class/infiniband/*/; do
  devname=$(basename "$dev")
  fw=$(cat "$dev/fw_ver")
  port="$dev/ports/1/"
  echo "$devname: FW=$fw State=$(cat $port/state) Rate=$(cat $port/rate) Layer=$(cat $port/link_layer)"
done

echo "=== GPU-NIC Affinity ==="
for dev in /sys/class/infiniband/*/device; do
  nic=$(echo $dev | grep -oP 'mlx5_\d+')
  numa=$(cat $dev/numa_node)
  echo "$nic: NUMA $numa"
done

echo "=== NUMA ==="
numactl --hardware 2>/dev/null | head -10

echo "=== IB Fabric (需要 infiniband-diags) ==="
echo "Leaf switches: $(ibnetdiscover 2>/dev/null | grep -c 'leaf')"
echo "Spine switches: $(ibnetdiscover 2>/dev/null | grep -c 'spine')"
echo "Total HCAs: $(ibnetdiscover 2>/dev/null | grep -c '^Ca')"
```

## 9. 常见问题诊断

| 现象 | 可能原因 | 检测方法 |
|------|----------|----------|
| NCCL 带宽远低于理论值 | GDR 未启用 | NCCL_DEBUG=INFO 看 GDR: disabled |
| 跨节点延迟高 | 走了 SYS 路径 | nvidia-smi topo 检查 NIC-GPU 亲和 |
| 间歇性性能抖动 | 链路 CRC 错误 | perfquery 看错误计数器 |
| AllReduce 只用了部分网卡 | NCCL 未检测到所有 NIC | NCCL_IB_HCA 手动指定 |
| 跨 leaf 性能差 | 交换机过载 | ibnetdiscover 计算过载比 |

## 10. 本章总结

通过以上检测，我们掌握了你的集群完整通信拓扑:
- **节点内**: 8xH100 通过 NVSwitch 全互联 (NV18, 900GB/s per pair)
- **GPU-NIC**: 1:1 PIX 亲和 (最优 GPUDirect RDMA 路径)
- **网卡**: 9x ConnectX-7 NDR 400Gb/s (8 数据口 + 1 管理/备用)
- **网络**: 2 层 fat-tree (64 leaf + 32 spine)，过载比 1.065:1
- **集群规模**: 大集群 (2117 HCA)，你使用 4 节点 32 GPU
- **节点间带宽**: 8 x 400Gb/s = 3.2Tb/s = 400GB/s per node

这些信息是后续 NCCL 源码分析和调优的基础。
NCCL 在初始化时会自动执行类似的检测（通过 PCI sysfs + NVML），
我们接下来的章节会分析 NCCL 如何利用这些拓扑信息选择最优通信算法。

