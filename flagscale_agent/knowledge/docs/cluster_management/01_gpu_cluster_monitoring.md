# 万卡级GPU集群监控与管理方案

## 1. 背景与挑战

### 1.1 万卡集群的规模特征

| 指标 | 4机验证环境 | 千卡生产环境 | 万卡超算环境 |
|------|-------------|--------------|--------------|
| 节点数 | 4 | 128 | 1280+ |
| GPU总数 | 32 | 1024 | 10240+ |
| IB网卡数 | 32 | 1024 | 10240+ |
| 交换机数 | ~96 | ~200 | 2000+ |
| 故障频率 | 低 | 每天数次 | 每小时数次 |

### 1.2 核心挑战

**挑战一：故障检测的时效性**

万卡训练中，单卡故障会导致整个训练任务挂起。若无实时监控，从故障发生到人工发现可能需要数十分钟，浪费大量GPU时。

- 单卡ECC错误 → 静默数据损坏 → 训练loss异常 → 回溯困难
- 网卡抖动 → AllReduce超时 → NCCL watchdog timeout → 全部进程死亡
- 温度过高 → GPU降频 → 训练吞吐突降但不会crash

**挑战二：状态收集的可扩展性**

串行SSH轮询在大规模集群不可行：

```
4节点串行SSH: ~2秒完成
128节点串行SSH: ~60秒（部分超时）
1280节点串行SSH: 完全不可行
```

**挑战三：定位问题的复杂性**

万卡训练涉及多层级：
- 应用层：loss spike、梯度爆炸
- 框架层：NCCL timeout、OOM
- 硬件层：GPU掉卡、IB链路flap、交换机故障

需要将各层指标关联分析才能快速定位根因。

**挑战四：资源利用率的可见性**

没有全局视图时无法回答：
- 哪些节点GPU利用率持续为0（空闲浪费）？
- 哪些任务显存碎片化严重？
- IB带宽是否是瓶颈？

## 2. 解决方案架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                   用户浏览器 / VSCode                          │
│                 Grafana Dashboard (:3000)                     │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP查询
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Prometheus (:9090)                           │
│            时序数据库 + 告警规则引擎                             │
│       每10s主动Pull各节点Exporter的/metrics端点                 │
└─────┬────────────┬────────────┬────────────┬────────────────┘
      │            │            │            │
      ▼            ▼            ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ Node 1   │ │ Node 2   │ │ Node 3   │ │ Node N   │
│ DCGM-Exp │ │ DCGM-Exp │ │ DCGM-Exp │ │ DCGM-Exp │
│ (:9400)  │ │ (:9400)  │ │ (:9400)  │ │ (:9400)  │
│    ↕     │ │    ↕     │ │    ↕     │ │    ↕     │
│ nv-host  │ │ nv-host  │ │ nv-host  │ │ nv-host  │
│ engine   │ │ engine   │ │ engine   │ │ engine   │
│    ↕     │ │    ↕     │ │    ↕     │ │    ↕     │
│ 8×H100   │ │ 8×H100   │ │ 8×H100   │ │ 8×H100   │
└──────────┘ └──────────┘ └──────────┘ └──────────┘
```

### 2.2 设计原则

| 原则 | 说明 | 万卡适配 |
|------|------|----------|
| Pull模型 | Prometheus主动拉取，非Push | 避免数据洪泛，控制采集频率 |
| 去中心化采集 | 每节点独立Exporter | 单节点故障不影响其他 |
| 分层解耦 | 采集/存储/展示分离 | 可独立水平扩展 |
| 共享存储 | 脚本/配置放共享卷 | 一处修改全局生效 |

### 2.3 数据流

```
GPU硬件 → DCGM(nv-hostengine) → DCGM Exporter(Python HTTP)
         → Prometheus(TSDB存储) → Grafana(可视化/告警)
```

## 3. 各组件详解

### 3.1 DCGM (Data Center GPU Manager)

**角色**：GPU硬件指标的标准采集接口

**为什么不直接用nvidia-smi？**

| 对比项 | nvidia-smi | DCGM |
|--------|-----------|------|
| 采集方式 | fork进程+解析文本 | C API + 守护进程 |
| 采集开销 | 高（每次~50ms） | 低（持续监听，~1ms） |
| 支持指标 | ~20个基础指标 | 1000+专业指标 |
| ECC/XID | 仅当前计数 | 历史记录+事件触发 |
| 健康诊断 | 无 | Level 1-4诊断 |
| 多GPU效率 | 串行查询 | 批量采集 |

**关键命令**：
```bash
nv-hostengine -b ALL          # 守护进程绑定所有GPU
dcgmi discovery -l             # 列出GPU拓扑
dcgmi dmon -e 155,150,203,204  # 实时监控(功耗/温/GPU利用率/显存利用率)
dcgmi diag -r 3                # Level 3健康诊断(含stress test)
```

**万卡价值**：统一的GPU健康基线，XID错误事件可触发自动隔离故障节点。

### 3.2 DCGM Exporter (轻量Python实现)

**角色**：将DCGM指标转换为Prometheus可抓取的HTTP端点

**设计决策**：

我们选择自研轻量Python Exporter而非官方Go版本，原因：
1. 官方dcgm-exporter需要Docker-in-Docker或独立容器部署
2. 我们已在容器内有DCGM，直接调用`dcgmi dmon`最简单
3. Python脚本放共享存储，4台同时可用，零部署成本
4. 38行代码，维护成本极低

**实现核心**：
```python
# <shared_storage>/scripts/dcgm_exporter.py
def collect():
    """调用dcgmi dmon采集一次数据，解析为Prometheus文本格式"""
    r = subprocess.run(["dcgmi","dmon","-e","155,150,203,204,252,253","-c","1"],
                       capture_output=True, text=True, timeout=10)
    # 解析输出行，跳过header，生成 metric{gpu="N"} value 格式
    ...
```

**暴露指标**：
| Metric | 含义 | 运维价值 |
|--------|------|----------|
| dcgm_power_watts | GPU功耗 | 检测降频/过载 |
| dcgm_gpu_temp_c | GPU温度 | 预警散热故障 |
| dcgm_gpu_util | SM利用率 | 发现空闲/卡住 |
| dcgm_mem_util | 显存带宽利用率 | 识别Memory-bound |
| dcgm_fb_used_mib | 已用显存 | 预警OOM |
| dcgm_fb_free_mib | 可用显存 | 容量规划 |

**万卡扩展**：每节点一个exporter进程，开销<5MB内存，无跨节点依赖。

### 3.3 Prometheus

**角色**：中心化时序数据库 + 服务发现 + 告警引擎

**为什么选Prometheus而非InfluxDB/Victoria？**
- GPU/HPC生态标准（NVIDIA官方推荐）
- PromQL查询语言强大（聚合、分位数、函数计算）
- 原生支持告警规则（Alertmanager）
- Pull模型天然限流

**配置核心**：
```yaml
# /etc/prometheus/prometheus.yml
scrape_configs:
  - job_name: 'dcgm_gpu'
    scrape_interval: 10s
    static_configs:
      - targets:
        - '<node-1>:9400'
        - '<node-2>:9400'
        - '<node-3>:9400'
        - '<node-3>87:9400'
```

**万卡扩展方案**：

```
                ┌── Prometheus联邦 ──┐
                │   Global (汇总)    │
                └──┬─────┬────┬────┘
                   │     │    │
         ┌─────────┤     │    ├─────────┐
         ▼         ▼     ▼    ▼         ▼
    Prom-Shard1  Shard2  Shard3  ...  ShardN
    (节点1-128) (129-256) ...       (1153-1280)
```

每个Shard负责~128节点，全局Prometheus做联邦聚合查询。

### 3.4 Grafana

**角色**：数据可视化 + 告警通知 + 团队协作

**核心价值**：
1. 将原始时序数据转化为人类可理解的图表
2. 预置告警规则（温度>85°C、利用率=0超过5min）
3. 支持多团队不同视角（运维全局 vs 研究员单任务）

**Dashboard面板设计**：
| 面板 | PromQL示例 | 用途 |
|------|------------|------|
| 集群GPU利用率热力图 | `dcgm_gpu_util` | 一眼识别空闲/过载节点 |
| 温度分布 | `quantile(0.95, dcgm_gpu_temp_c)` | 散热异常预警 |
| 显存水位 | `dcgm_fb_used / (used + free)` | OOM预警 |
| 功耗趋势 | `rate(dcgm_power_watts[5m])` | 降频检测 |

**访问方式**：
- VSCode端口转发：`localhost:3000` → 容器`:3000`
- 默认登录：admin / admin
- Dashboard：GPU Cluster Monitor

### 3.5 Slurm (作业调度器)

**角色**：多用户GPU资源分配与任务队列管理

**为什么需要Slurm？**

裸机直接跑训练的问题：
- 多人抢GPU → 冲突、显存不够
- 无法排队 → 反复手动检查空闲
- 无审计 → 不知道谁用了多少资源

**我们的Slurm配置**：
```conf
# /etc/slurm/slurm.conf
ClusterName=flagscale-cluster
SlurmctldHost=node-01
SchedulerType=sched/backfill    # 回填调度，最大化利用率
SelectType=select/cons_tres     # 可消耗资源选择（含GPU）
GresTypes=gpu                   # 声明GPU为可调度资源

NodeName=node-01 Gres=gpu:8 CPUs=<N> State=UNKNOWN
NodeName=node-02  Gres=gpu:8 CPUs=<N> State=UNKNOWN
NodeName=node-03   Gres=gpu:8 CPUs=<N> State=UNKNOWN
NodeName=node-04 Gres=gpu:8 CPUs=<N> State=UNKNOWN
```

**万卡场景的Slurm角色**：
| 功能 | 说明 |
|------|------|
| 资源隔离 | 按项目/团队分partition，保证公平 |
| 排队调度 | FIFO + backfill，最大化GPU利用率 |
| 故障隔离 | drain故障节点，自动跳过 |
| 弹性训练 | 配合FlagScale支持节点增减 |
| 计费审计 | 记录每个job的GPU×时 |

## 4. 万卡扩展：从4机到1280机

### 4.1 采集层扩展

| 规模 | 方案 | 延迟 |
|------|------|------|
| 4节点 | 单Prometheus直接Pull | <1s |
| 128节点 | 单Prometheus + 服务发现 | <5s |
| 512节点 | 2-3个Prometheus Shard + 联邦 | <10s |
| 1280节点 | 10个Shard + Thanos/VictoriaMetrics全局查询 | <15s |

### 4.2 服务发现（替代静态配置）

4节点可以手写targets，万卡必须自动发现：

```yaml
# 基于文件的服务发现
scrape_configs:
  - job_name: 'dcgm_gpu'
    file_sd_configs:
      - files: ['/etc/prometheus/targets/*.json']
        refresh_interval: 30s
```

配合脚本自动从Slurm生成targets：
```bash
sinfo -N -o "%N" | while read node; do
  echo "{\"targets\":[\"${node}:9400\"],\"labels\":{\"node\":\"${node}\"}}"
done > /etc/prometheus/targets/gpu_nodes.json
```

### 4.3 告警层

万卡集群必须有自动化告警，人工盯屏不现实：

```yaml
# /etc/prometheus/alert_rules.yml
groups:
  - name: gpu_alerts
    rules:
      - alert: GPUTemperatureHigh
        expr: dcgm_gpu_temp_c > 83
        for: 2m
        labels: {severity: warning}

      - alert: GPUUtilizationZero
        expr: dcgm_gpu_util == 0
        for: 10m
        labels: {severity: info}
        annotations:
          summary: "GPU空闲超过10分钟"

      - alert: GPUMemoryAlmostFull
        expr: dcgm_fb_free_mib < 1000
        for: 1m
        labels: {severity: critical}
```

### 4.4 存储层

| 规模 | 数据量/天 | 存储方案 |
|------|-----------|----------|
| 32卡 | ~50MB | 本地Prometheus TSDB |
| 1024卡 | ~1.5GB | Prometheus + 30天保留 |
| 10240卡 | ~15GB | Thanos + 对象存储(S3/MinIO) |

## 5. 我们的4机实践总结

### 5.1 部署拓扑

```
node-01 (管理节点):
  ├── slurmctld (调度器主进程)
  ├── slurmd (本地计算守护)
  ├── Prometheus (:9090)
  ├── Grafana (:3000)
  ├── nv-hostengine + dcgm_exporter (:9400)
  └── 8× H100 GPU

node-02 / node-03 / node-04 (计算节点):
  ├── slurmd
  ├── nv-hostengine + dcgm_exporter (:9400)
  └── 8× H100 GPU

共享存储 /workspace:
  ├── scripts/dcgm_exporter.py (所有节点共用)
  ├── tools/prometheus_data/ (时序数据)
  └── miniconda3/ (conda环境)
```

### 5.2 环境关键配置

| 配置项 | 值 | 原因 |
|--------|-----|------|
| SSH端口 | 2222 | 容器`--network=host`，22被宿主机占 |
| 免密方式 | 共享存储中转密钥 | 一处生成，cp到各容器 |
| Conda | 共享存储安装 | 4台共享同一环境，无需重复安装 |
| DCGM | 各容器apt安装 | 需要本地nv-hostengine守护进程 |
| Slurm | 各容器apt安装 | 需要本地slurmd进程 |
| cgroup | 手动enable控制器 | 容器内无systemd |

### 5.3 容器启动标准命令

```bash
docker run -d --gpus all --network=host \
  --ipc=host --ulimit memlock=-1 \
  -v <shared_storage>:<shared_storage> \
  --cap-add=SYS_ADMIN \
  --name flagscale \
  <image>
```

关键参数说明：
- `--network=host`：IB/RDMA需要主机网络栈
- `--ipc=host`：NCCL需要共享内存
- `--cap-add=SYS_ADMIN`：Slurm cgroup需要
- `-v <shared_storage>:<shared_storage>`：共享存储挂载

### 5.4 服务启动清单（容器重启后执行）

```bash
# 每个容器内：
/usr/sbin/sshd -p 2222                    # SSH
nv-hostengine -b ALL                       # DCGM
munged --force                             # Slurm认证
slurmd -N $(hostname -s)                   # Slurm计算守护
python3 <shared_storage>/scripts/dcgm_exporter.py 9400 &  # Metrics

# 仅管理节点(118)：
slurmctld                                  # Slurm调度器
prometheus --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=<shared_storage>/tools/prometheus_data &
grafana-server --homepath=/usr/share/grafana &
```

## 6. 与训练框架的协同

### 6.1 FlagScale训练监控集成

训练运行时，监控栈可以回答：
| 问题 | 指标来源 | 判断方法 |
|------|----------|----------|
| 训练是否卡住？ | GPU利用率 | 全卡突降到0 |
| 是否有慢节点？ | GPU利用率方差 | 某节点持续偏低 |
| 是否通信瓶颈？ | GPU利用率+IB带宽 | 高利用率但吞吐低 |
| 是否显存泄漏？ | FB_used趋势 | 持续增长不回落 |
| 硬件是否降频？ | 功耗+温度 | 温度>80且功耗突降 |

### 6.2 训练失败自动处理（万卡必备）

```
训练NCCL timeout
  → Prometheus检测到某节点GPU util突降
  → 关联IB指标发现该节点网卡error count上升
  → Alertmanager通知 + 自动drain节点
  → Slurm重调度训练到健康节点
  → FlagScale从最近checkpoint恢复
```

## 7. 总结

本方案以**DCGM + Prometheus + Grafana + Slurm**四件套为核心，实现了：

1. **实时可观测性**：10秒粒度的全集群GPU状态
2. **可扩展架构**：从4机验证到万卡生产，架构不变只加Shard
3. **低部署成本**：利用共享存储，38行Python搞定Exporter
4. **故障快速定位**：分层指标关联，从现象到根因
5. **自动化运维**：告警规则 + 故障隔离 + 弹性恢复

从4机小集群开始验证，逐步扩展到万卡，核心思路不变：**每层只做一件事，通过标准接口解耦**。
