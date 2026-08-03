# 硬件与通信基础设施分析标准

> 目标：建立从物理拓扑到软件栈的完整性能模型，量化每条数据路径的带宽极限与实测基线。本标准仅覆盖硬件级分析，产出作为基础输入提供给上层（模型并行策略选择、训练任务配置）使用。

## 一、核心原则

1. **数据驱动**：所有结论必须有实测数据支撑，不能仅依赖datasheet理论值
2. **分层递进**：物理层→驱动层→软件栈→通信性能，逐层分析
3. **瓶颈定位**：每层分析的最终目的是回答"当前瓶颈在哪，如何消除"
4. **可复现**：所有检测命令、工具版本、环境条件完整记录
5. **对比基线**：实测值必须与理论峰值对比，给出利用率百分比
6. **时效性**：硬件分析结果需标注检测时间，driver/firmware升级后需重新验证

## 二、分析层次与产出要求

### 2.1 Layer 1：物理拓扑与带宽极限

**分析目标**：画出完整的bandwidth hierarchy图，标注每条路径的理论峰值。

| 检测维度 | 关键指标 | 工具/方法 | 产出格式 |
|----------|----------|-----------|----------|
| GPU互联 | NVLink代际、链路数、单向带宽、聚合带宽 | nvidia-smi topo -m, nvbandwidth | 拓扑矩阵 + 带宽表 |
| GPU-CPU | PCIe代际、lane数、实测HtoD/DtoH带宽 | lspci, nvbandwidth, cuda bandwidthTest | 带宽曲线(msg_size vs BW) |
| 节点间网络 | 网卡型号/数量/速率、交换机层级 | ibstat, ibdev2pci, show_gids | 网络拓扑图 |
| 存储I/O | NVMe型号/数量、顺序读写带宽 | fio, dd | IOPS + BW表 |
| NUMA拓扑 | socket数、核心分布、GPU-NUMA亲和 | numactl, lstopo, nvidia-smi topo | NUMA亲和矩阵 |
| 物理布局 | 机柜位置、线缆连接、散热条件 | 人工记录 | 物理拓扑图 |

**必须回答的问题**：
- 节点内GPU间通信的理论峰值是多少？（如 8×NVLink4 = 900 GB/s bidirectional per GPU pair）
- 节点间通信的理论峰值是多少？（如 8×400G IB = 400 GB/s aggregate per node）
- Intra/Inter带宽比是多少？（决定TP应放节点内还是可以跨节点）
- 存储带宽是否足以支撑数据加载？（不能让数据IO成为训练瓶颈）

### 2.2 Layer 2：驱动与固件能力

**分析目标**：确认硬件能力被正确暴露，无降级或禁用。

| 检测维度 | 关键指标 | 异常信号 |
|----------|----------|----------|
| GPU Driver | 版本、persistence mode、fabric manager状态 | 版本过旧不支持新CUDA特性 |
| NVLink状态 | 每条link的协商速率、error count | link down/degraded、error累积 |
| IB/RoCE固件 | firmware版本、port state、physical state | LinkDown、速率降级（如400G降为200G） |
| IOMMU/ACS | 是否影响GPU P2P direct access | P2P被block导致走CPU bounce |
| ECC | 开启状态、correctable/uncorrectable error count | uncorrectable error > 0需更换 |
| PCIe | 协商speed/width、AER错误 | x16降为x8、Gen5降为Gen4 |
| 时钟 | GPU boost clock、memory clock、power limit | 被thermal throttle或power cap降频 |

**必须回答的问题**：
- 所有NVLink是否全部active且协商到最高速率？
- IB端口是否全部LinkUp且在标称速率？
- 是否有系统配置（IOMMU、ACS）阻止了最优数据路径？
- GPU是否工作在最大性能状态（未被throttle）？

### 2.3 Layer 3：软件栈版本与能力匹配

**分析目标**：确保软件栈充分利用硬件代际特性，无版本不匹配导致的性能损失。

| 组件 | 关键检查项 | 最优实践 |
|------|-----------|----------|
| CUDA Toolkit | 版本是否发挥GPU代际特性（H100: TMA, WGMMA, FP8） | ≥CUDA 12.x for H100 |
| NCCL | 版本、algo/proto选择、NVLink/IB路径是否正确使用 | 使用与driver匹配的最新版 |
| cuDNN | 版本、是否启用auto-tune | fusion + auto-tune enabled |
| cuBLAS | 版本、默认math mode (TF32) | TF32 enabled for H100 |
| IB驱动(MLNX_OFED) | 版本、GDR支持 | GPUDirect RDMA enabled |
| TransformerEngine | 版本、FP8 recipe配置 | 匹配CUDA版本 |
| PyTorch | 版本、CUDA backend编译版本 | 与toolkit匹配 |

**版本兼容性矩阵**（示例）：
```
GPU: H100 80GB
Driver: ≥535.x (推荐550+)
CUDA: 12.1-12.8
NCCL: ≥2.18 (推荐2.21+)
cuDNN: ≥8.9 (推荐9.x)
MLNX_OFED: ≥5.8 (推荐23.x)
```

**必须回答的问题**：
- 软件栈是否充分利用了硬件代际新特性？（如H100的TMA/WGMMA/FP8）
- NCCL是否选择了最优的通信算法和协议？
- GPUDirect RDMA是否生效？（节点间通信是否bypass CPU）
- 是否存在版本冲突导致fallback到低性能路径？

### 2.4 Layer 4：通信性能实测基线

**分析目标**：建立不同通信模式的性能基线，作为并行策略选择的定量依据。

| 测试场景 | 测试工具 | 关键指标 | 对比基线 |
|----------|----------|----------|----------|
| 节点内AllReduce | nccl-tests (all_reduce_perf) | busbw vs msg_size曲线 | NVLink理论带宽 |
| 节点间AllReduce | nccl-tests (多节点) | busbw vs msg_size曲线 | IB理论带宽 |
| P2P带宽 | nccl-tests (sendrecv) / nvbandwidth | 单向/双向带宽 | NVLink/PCIe理论 |
| AlltoAll | nccl-tests / custom | EP通信模拟 | 网络bisection BW |
| Latency | nccl-tests (-b 1 -e 1) | 最小延迟 | us级基线 |
| Overlap测试 | 自定义benchmark | compute/comm重叠比 | 目标>80%重叠 |

**关键产出**：
```
# 通信性能摘要示例
intra_node_allreduce_peak: 850 GB/s busbw (94% of NVLink theoretical)
inter_node_allreduce_peak: 380 GB/s busbw (95% of IB theoretical)
intra_node_p2p: 450 GB/s unidirectional
allreduce_latency_8B: 12 us (intra) / 45 us (inter)
compute_comm_overlap: 85% achievable
```

**必须回答的问题**：
- 节点内通信效率（实测/理论）是多少？低于90%需排查原因
- 节点间通信效率是多少？
- Intra/Inter性能比是多少？（决定跨节点并行的代价）
- 通信延迟对小消息场景（如PP bubble）的影响？
- Compute-communication overlap的实际可达比例？

### 2.5 产出规范与下游接口

**本标准的产出边界**：硬件分析到Layer 4（通信性能实测基线）为止。以下内容属于本标准的产出物，供下游任务消费：

**硬件分析交付物**：
```yaml
# 硬件能力摘要（供并行策略选择使用）
cluster:
  nodes: N
  gpus_per_node: 8
  gpu_model: H100 80GB
  gpu_peak_flops_bf16: 989 TFLOPS  # per GPU

intra_node:
  interconnect: NVLink4
  topology: full_mesh / nvswitch
  allreduce_busbw: XXX GB/s  # 实测峰值
  p2p_bw: XXX GB/s           # 实测单向
  latency: XX us             # 8B最小延迟

inter_node:
  network: IB_NDR_400G / RoCE
  nics_per_node: N
  allreduce_busbw: XXX GB/s  # 实测峰值
  latency: XX us
  gdr_enabled: true/false

bandwidth_ratio:
  intra_inter: X.Xx          # 节点内/节点间带宽比（核心决策指标）

software_stack:
  driver: XXX
  cuda: XX.X
  nccl: X.XX.X
  gdr: enabled/disabled
  issues: [...]              # 已知问题列表
```

**下游消费方**：
- 模型并行策略选择（结合模型参数量、层数、MoE结构）
- 训练配置生成（TP/PP/DP/EP/CP具体数值）
- 性能预估（通信时间 = 数据量 / 实测带宽）
- 问题诊断（实际throughput低时对照基线排查）

**不属于本标准的内容**：
- MFU计算（需要具体模型信息）
- 并行策略推荐（需要模型+硬件联合分析）
- 训练超参选择（需要模型+数据+硬件三方信息）

## 三、文档结构模板

一份完整的硬件分析文档应包含以下节：

```markdown
# [集群名称] 硬件与通信分析报告

## 1. 集群概览
- 节点数、GPU型号/数量、网络类型
- 用途定位（训练/推理/混合）

## 2. 物理拓扑
- 节点内GPU互联拓扑图
- 节点间网络拓扑图
- NUMA亲和关系

## 3. 驱动与固件状态
- 版本矩阵（driver/CUDA/NCCL/IB）
- 异常项与修复建议

## 4. 带宽基线数据
- 节点内通信性能（AllReduce/P2P曲线）
- 节点间通信性能
- 存储I/O性能

## 5. 软件栈评估
- 版本兼容性检查
- 关键配置项（GDR/SHARP/NUMA binding）
- 优化建议

## 6. 硬件能力摘要（YAML格式，供下游任务使用）
- 结构化的带宽/延迟/拓扑数据
- 已知问题和限制

## 7. 问题清单与行动项
- 已发现的性能gap
- 修复优先级排序
- 验证方案
```

## 四、质量检查清单

### 完备性
- [ ] 五层分析是否全部覆盖
- [ ] 每层是否回答了"必须回答的问题"
- [ ] 是否有实测数据支撑结论
- [ ] 理论值与实测值的比率是否计算

### 准确性
- [ ] 工具版本是否记录
- [ ] 测试条件是否完整（msg_size范围、迭代次数、warmup）
- [ ] 多次测量是否取中位数/P95
- [ ] 异常值是否排查原因

### 可行动性
- [ ] 是否给出具体的优化建议
- [ ] 建议是否有优先级排序
- [ ] 是否说明预期收益（如"修复后预计AllReduce提升15%"）
- [ ] 是否有验证方案确认优化效果

### 时效性
- [ ] 检测时间是否标注
- [ ] 哪些结果会因升级而失效
- [ ] 下次复检触发条件是什么（如driver升级后、新节点加入后）

## 五、与其他标准的关系

| 标准 | 关系 | 说明 |
|------|------|------|
| 01_infrastructure_analysis | 被依赖 | 软件框架如何使用硬件能力（如Megatron的TP实现如何映射到NVLink） |
| 02_implementation_analysis | 被依赖 | 模型实现的并行策略选择依赖硬件分析结论 |
| 03_paper_research | 参考 | 论文中的infra配置需对照自身硬件评估可行性 |
| topo-detect skill | 执行层 | 本标准定义"分析什么"，skill定义"怎么跑命令" |
| know-cluster-infra | 产出存储 | 分析结果作为知识存入集群基础设施知识库 |

## 六、常见陷阱

1. **只看datasheet不实测**：理论900 GB/s NVLink，实际可能因driver bug/config错误只跑到600
2. **忽略NUMA亲和**：GPU 0-3在socket 0，GPU 4-7在socket 1，跨socket访问延迟翻倍
3. **NCCL版本不匹配**：旧NCCL不认新NVLink拓扑，退化为PCIe路径
4. **GDR未生效**：节点间通信走CPU bounce，延迟增加50%+
5. **PCIe降级未察觉**：x16降为x8，带宽减半但不报error
6. **ECC silent error**：correctable ECC error累积暗示硬件退化
7. **Power/Thermal throttle**：持续高负载下GPU降频，throughput下降10-20%
8. **交换机过订比**：leaf-spine架构中spine带宽不足导致跨rack通信降级
