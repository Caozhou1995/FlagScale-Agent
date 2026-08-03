# PyTorch Distributed 源码深度分析 — 第1章：ProcessGroup 与集合通信

## 1. 设计动机

### 1.1 为什么需要 torch.distributed？

**WHY 不直接调用 NCCL API？** 需要统一抽象层支持多后端（NCCL/Gloo/MPI/UCC/XCCL），
提供 Python 接口与 Autograd 集成，管理进程组生命周期。

```
torch.distributed 在技术栈中的位置:
┌──────────────────────────────────────────────────┐
│  用户代码 / DDP / FSDP / Pipeline Parallel       │
├──────────────────────────────────────────────────┤
│  torch.distributed (distributed_c10d.py: 7915行) │
│  init_process_group / all_reduce / all_gather    │
├──────────────────────────────────────────────────┤
│  ProcessGroup C++ 抽象层                          │
│  ProcessGroupNCCL / ProcessGroupGloo / ...        │
├──────────────────────────────────────────────────┤
│  NCCL / Gloo / MPI / UCC / XCCL                 │
├──────────────────────────────────────────────────┤
│  IB Verbs / Socket / NVLink / PCIe               │
└──────────────────────────────────────────────────┘
```

### 1.2 核心设计约束

| 约束 | 解决方案 | 源码体现 |
|------|---------|---------|
| 多后端统一 | Backend 抽象 + 运行时注册 | Backend class L436 |
| 异步执行 | Work 对象 + wait() | isend/irecv 返回 Work |
| 进程组管理 | _World 全局状态 | _World class L1156 |
| Lazy 初始化 | 延迟到首次通信 | _create_nccl_process_group |
| 设备绑定 | device_id 参数 | init_process_group L2228 |

## 2. init_process_group 初始化 (L2228-2340)

### 2.1 函数签名

```python
# torch/distributed/distributed_c10d.py L2228-2240
def init_process_group(
    backend: str | None = None,        # "nccl", "gloo", "mpi", None(自动检测)
    init_method: str | None = None,    # "env://", "tcp://host:port", "file:///path"
    timeout: timedelta | None = None,  # NCCL: 10min, Others: 30min
    world_size: int = -1,
    rank: int = -1,
    store: Store | None = None,        # 与 init_method 互斥
    group_name: str = "",              # deprecated
    pg_options: object | None = None,  # NCCL 选项 (高优先级流等)
    device_id: torch.device | int | None = None,  # 即时初始化 + ncclCommSplit
    _ranks: list[int] | None = None,
    enable_reconfigure: bool = False,  # 容错模式
) -> None:
```

### 2.2 初始化流程

```
init_process_group 执行流:
┌─────────────────────────────────────────────────────────────┐
│ 1. 解析 backend → BackendConfig                             │
│    - 自动检测: 检查 device_id / CUDA 可用性 → 选择 nccl    │
│    - 手动指定: "nccl" / "gloo" / 自定义注册后端             │
├─────────────────────────────────────────────────────────────┤
│ 2. 创建 Store (进程间 KV 存储)                              │
│    - env://: 环境变量 MASTER_ADDR/PORT → TCPStore           │
│    - tcp://: 直接创建 TCPStore                              │
│    - file://: FileStore (NFS 共享文件系统)                   │
├─────────────────────────────────────────────────────────────┤
│ 3. Store-based barrier (确保所有进程就绪)                    │
│    - 每个 rank 写入 key, 等待 world_size 个 key 出现        │
├─────────────────────────────────────────────────────────────┤
│ 4. 创建 ProcessGroup 实例                                   │
│    - _create_nccl_process_group() → ProcessGroupNCCL       │
│    - 设置 _World 默认组                                     │
├─────────────────────────────────────────────────────────────┤
│ 5. (可选) 即时初始化 (device_id != None)                    │
│    - 立即调用 ncclCommInitRank → 验证网络连通性            │
│    - 后续 new_group() 可使用 ncclCommSplit                  │
└─────────────────────────────────────────────────────────────┘
```

**WHY device_id 参数？** 默认 lazy 初始化在首次通信时才创建 NCCL communicator。
如果此时网络有问题，错误信息晦涩。device_id 强制即时初始化 → 启动时就能发现问题。

## 3. Backend 注册机制 (L436-620)

### 3.1 Backend 类

```python
# distributed_c10d.py L436
class Backend(str):
    UNDEFINED = "undefined"
    GLOO = "gloo"
    NCCL = "nccl"
    UCC = "ucc"
    MPI = "mpi"
    XCCL = "xccl"
    
    # 第三方后端注册
    _plugins: dict[str, _BackendPlugin] = {}
    
    @classmethod
    def register_backend(cls, name, func, extended_api=False, devices=None):
        """注册自定义后端 (如 torch-ccl for Intel)"""
        cls._plugins[name] = _BackendPlugin(func, extended_api, devices)
```

### 3.2 NCCL ProcessGroup 创建 (L648-685)

```python
# distributed_c10d.py L648
def _create_nccl_process_group(
    store, rank, world_size, pg_options, group_name, timeout, device_id, ...
):
    """创建 ProcessGroupNCCL 实例"""
    # pg_options: ProcessGroupNCCL.Options
    #   - is_high_priority_stream: 使用高优先级 CUDA stream
    #   - config: ncclConfig_t (minCTAs, maxCTAs, net 配置等)
    pg = ProcessGroupNCCL(store, rank, world_size, pg_options)
    if device_id is not None:
        pg._set_default_device(device_id)
        pg.eager_connect_single_device(device_id)  # 即时初始化
    return pg
```

## 4. 集合通信 API

### 4.1 all_reduce (L3717-3848)

```python
# distributed_c10d.py L3736 (overloaded)
@overload
def all_reduce(tensor: torch.Tensor, op=ReduceOp.SUM, 
               group=None, async_op=False) -> Work | None: ...

# 执行流:
# 1. 获取 ProcessGroup (默认 or 指定)
# 2. 调用 pg.allreduce([tensor], AllreduceOptions(op))
# 3. 返回 Work 对象 (async_op=True) 或等待完成
```

### 4.2 集合通信操作总览

| API | 语义 | 输入 → 输出 | NCCL 对应 |
|-----|------|------------|-----------|
| all_reduce | 全部归约 | [T] → [ReduceOp(T)] | ncclAllReduce |
| all_gather | 全部收集 | [Ti] → [T0,T1,...,Tn] | ncclAllGather |
| reduce_scatter | 归约分发 | [T0,...,Tn] → [ReduceOp(Ti)] | ncclReduceScatter |
| broadcast | 广播 | src:[T] → all:[T] | ncclBroadcast |
| send/recv | 点对点 | src → dst | ncclSend/ncclRecv |
| batch_isend_irecv | 批量P2P | 批量发送接收 | ncclGroupStart/End |
| barrier | 同步屏障 | 所有进程同步 | ncclAllReduce(1byte) |
| all_to_all | 全交换 | [Ti_j] → [Tj_i] | ncclSend+ncclRecv |

### 4.3 Functional Collectives (_functional_collectives.py)

```python
# torch/distributed/_functional_collectives.py
# 支持 torch.compile 的函数式集合通信

def all_reduce(self: torch.Tensor, reduceOp: str, group: ...) -> torch.Tensor:
    """返回新 tensor (非 inplace)，支持 autograd 和 torch.compile"""
    # 使用 torch.library.custom_op 注册
    # 区别于 distributed_c10d.all_reduce (inplace 修改 tensor)
```

**WHY Functional Collectives？** torch.compile 需要纯函数语义（无 side effect），
传统 inplace 集合通信打破了 FX graph 的函数式假设。

## 5. _World 全局状态管理 (L1156-1290)

```python
# distributed_c10d.py L1156
class _World:
    """全局分布式状态单例"""
    _default_pg: ProcessGroup | None = None     # 默认进程组
    _pg_map: dict[ProcessGroup, tuple] = {}      # PG → (backend, store) 映射
    _pg_names: dict[ProcessGroup, str] = {}      # PG → name
    _group_count: int = 0                        # 已创建组数
    _tags_to_pg: dict[str, list[ProcessGroup]]   # tag → PG 列表
    _pg_to_tag: dict[ProcessGroup, str]          # PG → tag
```

## 6. new_group 子组创建 (L1693+)

```python
# 创建子组 (如 TP/PP/DP 组):
tp_group = torch.distributed.new_group(ranks=[0,1,2,3])
dp_group = torch.distributed.new_group(ranks=[0,4])

# 当 device_id 已设置时, 使用 ncclCommSplit:
# WHY ncclCommSplit? 比 ncclCommInitRank 快 10-100×
# 因为 Split 复用已有连接, 不需要重新建立 IB 连接
```

## 7. P2POp 批量操作 (L1046-1155)

```python
# distributed_c10d.py L1046
class P2POp:
    """点对点操作描述"""
    def __init__(self, op, tensor, peer, group=None, tag=0):
        self.op = op        # isend 或 irecv
        self.tensor = tensor
        self.peer = peer    # 目标/源 rank
        self.group = group
        self.tag = tag

# 批量 P2P (Pipeline Parallel 核心):
ops = [P2POp(dist.isend, send_tensor, dst_rank),
       P2POp(dist.irecv, recv_tensor, src_rank)]
works = dist.batch_isend_irecv(ops)
# 内部: ncclGroupStart() → 多个 ncclSend/ncclRecv → ncclGroupEnd()
```

## 8. 超时与错误处理

```
NCCL 超时机制:
  default_timeout = 10 minutes (NCCL), 30 minutes (others)
  
  超时后行为:
  1. 异步中止 collective → ProcessGroupNCCL::abort()
  2. ncclCommAbort() → 销毁 communicator
  3. 进程 crash (因为后续 CUDA 操作可能访问损坏数据)
  
  WHY 必须 crash?
  - NCCL 操作是异步的
  - 超时意味着某些 rank 已 hang
  - 继续执行可能导致 silent data corruption
  
  环境变量:
  TORCH_NCCL_BLOCKING_WAIT=1  → 阻塞等待 (而非异步)
  NCCL_TIMEOUT=1800            → 自定义超时秒数
```

## 9. 总结

| 组件 | 职责 | 源码行数 | 关键设计 |
|------|------|---------|---------|
| init_process_group | 初始化通信 | L2228-2340 | Store + Barrier + Lazy/Eager |
| Backend | 后端抽象 | L436-620 | 注册式扩展 |
| _World | 全局状态 | L1156-1290 | 单例 + PG 注册表 |
| Collectives | 集合通信 | L3646-6275 | Work 异步模型 |
| P2POp | 点对点 | L1046-1155 | ncclGroup 批量 |
| Functional | torch.compile | _functional_collectives | 纯函数语义 |

## 10. ProcessGroupNCCL 内部机制

### 10.1 Stream 管理

```
ProcessGroupNCCL 内部 CUDA Stream 策略:
┌─────────────────────────────────────────────────────────┐
│ 每个 (device, op_type) 有独立 NCCL Stream               │
│                                                         │
│  Default stream ──→ record event ──→ NCCL stream 等待   │
│                                            ↓            │
│                                     执行 NCCL 操作      │
│                                            ↓            │
│  Default stream ←── NCCL stream record event ←──────   │
│                                                         │
│  WHY 独立 stream?                                       │
│  - NCCL 操作可能长时间阻塞                               │
│  - 不能阻塞计算 stream                                  │
│  - 允许通信与计算 overlap                                │
└─────────────────────────────────────────────────────────┘
```

### 10.2 Watchdog 线程

```python
# torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp
# Watchdog 独立线程检测 hang:
#   - 每隔 timeout/2 检查一次
#   - 如果某 Work 超时 → desyncReport
#   - 收集 flight recorder 信息 → dump debug info
#   - 调用 ncclCommAbort

# 环境变量:
TORCH_NCCL_ENABLE_MONITORING=1    # 启用监控
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC  # 心跳超时
TORCH_NCCL_DUMP_ON_TIMEOUT=1      # 超时时 dump 调试信息
```

### 10.3 Flight Recorder (通信记录器)

```
Flight Recorder 记录每次集合通信的:
  - op_type, tensor_size, dtype, group_name
  - start_time, end_time (或 pending)
  - seq_num (序列号, 用于检测 desync)
  
  用途:
  1. 超时诊断: 哪个 op 在哪个 rank hang 了
  2. Desync 检测: rank 间 seq_num 不匹配 → 某 rank 跳过了 collective
  3. 性能分析: 统计各 op 耗时

  启用: TORCH_NCCL_TRACE_BUFFER_SIZE=1000 (记录最近 N 次通信)
```

## 11. Store 实现

### 11.1 TCPStore (最常用)

```
TCPStore 架构:
┌──────────────────────────────────────────┐
│  Rank 0: TCPStore server (监听端口)       │
│    - HashMap<string, vector<uint8_t>>    │
│    - 支持: set/get/wait/add/check        │
├──────────────────────────────────────────┤
│  Rank 1..N-1: TCPStore client             │
│    - 连接 server                          │
│    - 支持同样的 KV 操作                    │
└──────────────────────────────────────────┘

# WHY TCPStore 而非 Redis/etcd?
# - 零外部依赖
# - 轻量级 (单进程内嵌)
# - 足够满足 barrier/rendezvous 需求
# - 仅用于初始化阶段, 不在数据路径上
```

### 11.2 Store-based Barrier

```python
# distributed_c10d.py _store_based_barrier:
def _store_based_barrier(rank, store, group_name, world_size, timeout):
    """所有 rank 写入 key, 等待 world_size 个 key 出现"""
    store_key = f"store_based_barrier_key:{group_name}"
    store.add(store_key, 1)           # 原子 +1
    # 忙等待直到 counter == world_size
    while store.add(store_key, 0) < world_size:
        time.sleep(0.01)
```

## 12. 与 Megatron 的集成

```
Megatron-LM 使用 torch.distributed 的模式:

1. 初始化:
   megatron/core/parallel_state.py 调用:
   torch.distributed.init_process_group(backend="nccl")
   
2. 创建子组:
   tp_group = torch.distributed.new_group(tp_ranks)
   pp_group = torch.distributed.new_group(pp_ranks)
   dp_group = torch.distributed.new_group(dp_ranks)
   
3. 集合通信:
   - TP: all_reduce (tensor parallel gradient sync)
   - DP: all_reduce / reduce_scatter (data parallel)
   - PP: batch_isend_irecv (pipeline bubble)
   - CP: all_to_all (context parallel KV exchange)
   
4. 关键优化:
   - _coalescing_manager: 合并多个小 tensor all_reduce
   - async_op=True: 通信与计算 overlap
   - process_group 复用: 避免重复创建
```

## 13. 与 NCCL 源码的对应关系

| torch.distributed API | C++ ProcessGroupNCCL | NCCL API |
|-----------------------|---------------------|----------|
| all_reduce(t, SUM) | collective(allreduce) | ncclAllReduce |
| all_gather_into_tensor | collective(allgather) | ncclAllGather |
| reduce_scatter_tensor | collective(reduce_scatter) | ncclReduceScatter |
| broadcast(t, src) | collective(broadcast) | ncclBroadcast |
| batch_isend_irecv | groupStart/End + p2p | ncclSend/Recv |
| barrier() | allreduce(1 byte) | ncclAllReduce |
| new_group(ranks) | ncclCommSplit / ncclCommInitRank | ncclComm* |

