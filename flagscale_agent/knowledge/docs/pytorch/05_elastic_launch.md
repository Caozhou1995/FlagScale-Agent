# PyTorch Distributed 源码深度分析 — 第5章：Elastic Launch 与 torchrun

## 1. 设计动机

### 1.1 WHY torchrun 替代 torch.distributed.launch？

```
早期 (torch.distributed.launch):
  python -m torch.distributed.launch --nproc_per_node=8 train.py
  
  问题:
  1. 固定 world_size: 无法动态增减节点
  2. 无容错: 一个进程挂 → 所有进程挂
  3. 手动重启: 失败后需要人工干预
  4. 无 health check: 不知道进程是否 hang
  
torchrun (torch.distributed.elastic):
  torchrun --nproc_per_node=8 --nnodes=4 --rdzv_backend=c10d train.py
  
  WHY 弹性?
  1. 自动重启失败进程
  2. 支持节点动态加入/退出
  3. Rendezvous 协调多节点
  4. 心跳监控 + 超时检测
```

### 1.2 架构

```
torchrun 架构:
┌──────────────────────────────────────────────────────┐
│ torchrun CLI (entry point)                           │
├──────────────────────────────────────────────────────┤
│ LaunchAgent (本节点进程管理)                          │
│  - 启动 nproc_per_node 个 worker                     │
│  - 监控 worker 存活                                  │
│  - 处理失败 (重启 or 上报)                           │
├──────────────────────────────────────────────────────┤
│ RendezvousHandler (多节点协调)                        │
│  - 等待所有节点就绪                                  │
│  - 分配 rank / world_size                            │
│  - 处理节点加入/退出                                 │
├──────────────────────────────────────────────────────┤
│ Worker Processes (用户训练代码)                        │
│  - 通过环境变量获取: RANK, WORLD_SIZE, MASTER_ADDR   │
│  - 调用 init_process_group                           │
└──────────────────────────────────────────────────────┘
```

## 2. Rendezvous (会合) 机制

### 2.1 WHY Rendezvous？

```
多节点训练的同步问题:
─────────────────────
Node 0 启动于 T=0
Node 1 启动于 T=5s
Node 2 启动于 T=12s
Node 3 启动于 T=30s (慢节点/调度延迟)

如何让所有节点同时开始训练?
→ Rendezvous: 等待到最少 min_nodes 就绪, 达成一致后开始

Rendezvous 参数:
  --nnodes=4                    # min=max=4 (固定规模)
  --nnodes=2:4                  # min=2, max=4 (弹性规模)
  --rdzv_backend=c10d           # 使用 TCPStore 协调
  --rdzv_endpoint=master:29400  # coordinator 地址
```

### 2.2 C10d Rendezvous 流程

```
C10d (TCPStore) Rendezvous 流程:
═══════════════════════════════════════════════
Step 1: 各节点连接 TCPStore (rdzv_endpoint)
        Node i → store.set(f"node_{i}", "joining")
        
Step 2: 等待参与者
        wait until store.num_keys("node_*") >= min_nodes
        
Step 3: Barrier (超时机制)
        if 在 rdzv_timeout 内达到 max_nodes → 开始
        elif 达到 min_nodes → 等待 last_call_timeout 后开始
        else → 失败
        
Step 4: 分配 rank
        排序节点 → 分配 group_rank (节点编号)
        每节点内: local_rank = 0..nproc_per_node-1
        global_rank = group_rank * nproc_per_node + local_rank
        
Step 5: 返回 RendezvousInfo
        {world_size, rank, master_addr, master_port, store}
```

### 2.3 Rendezvous Backend 选项

| Backend | 实现 | 适用场景 | 外部依赖 |
|---------|------|---------|---------|
| c10d | TCPStore | 通用 (推荐) | 无 |
| etcd | etcd v2 | 大规模弹性 | etcd 服务 |
| etcd-v2 | etcd v2 HTTP | 同上 | etcd 服务 |
| static | 固定配置 | 调试/小规模 | 无 |

## 3. LaunchAgent (torch.distributed.elastic.agent)

### 3.1 LocalElasticAgent

```python
# torch/distributed/elastic/agent/server/local_elastic_agent.py
class LocalElasticAgent(SimpleElasticAgent):
    """单节点上的 worker 管理器"""
    
    def _start_workers(self, worker_group):
        """启动 worker 子进程"""
        for local_rank in range(nproc_per_node):
            env = {
                "LOCAL_RANK": str(local_rank),
                "RANK": str(global_rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": str(master_port),
                "LOCAL_WORLD_SIZE": str(nproc_per_node),
                "GROUP_RANK": str(group_rank),
            }
            proc = subprocess.Popen(cmd, env=env)
```

### 3.2 Worker 生命周期

```
Worker 状态机:
─────────────────
  INIT → HEALTHY → (error) → FAILED
                → (complete) → SUCCEEDED
                → (timeout) → UNKNOWN
  
Agent 处理逻辑:
  loop:
    monitor workers (每 30s)
    if any worker FAILED:
      if restarts < max_restarts:
        stop_all_workers()
        rendezvous()  # 重新协调
        start_workers()
        restarts += 1
      else:
        raise WorkerGroupFailure
    if all workers SUCCEEDED:
      return SUCCESS
```

## 4. 环境变量传递

```
torchrun 设置的环境变量 (Worker 可读取):
┌─────────────────────┬────────────────────────────────────┐
│ 变量                │ 含义                                │
├─────────────────────┼────────────────────────────────────┤
│ RANK                │ 全局 rank (0..world_size-1)         │
│ LOCAL_RANK          │ 节点内 rank (0..nproc-1)            │
│ WORLD_SIZE          │ 总进程数                            │
│ LOCAL_WORLD_SIZE    │ 节点内进程数 (nproc_per_node)       │
│ MASTER_ADDR         │ rank 0 的地址                       │
│ MASTER_PORT         │ rank 0 的端口                       │
│ GROUP_RANK          │ 节点编号 (0..nnodes-1)              │
│ TORCHELASTIC_RUN_ID │ 本次运行的唯一 ID                   │
└─────────────────────┴────────────────────────────────────┘
```

## 5. 容错机制

### 5.1 自动重启

```
容错场景:
──────────
1. Worker OOM → 被 OS kill → Agent 检测到退出码
2. NCCL 超时 → Worker abort → Agent 检测
3. 硬件错误 → ECC error → 进程 crash

Agent 行为:
  --max_restarts=3 (默认)
  每次重启:
    1. Kill 所有 worker (确保干净状态)
    2. 重新 rendezvous (获取新 rank 分配)
    3. 重新启动 worker
    4. Worker 从 checkpoint 恢复训练
    
  WHY kill 所有 worker?
  - NCCL communicator 已损坏, 无法部分恢复
  - rank 分配可能变化 (弹性场景)
  - 保证一致性: 所有 rank 同时从 checkpoint 恢复
```

### 5.2 与训练代码的配合

```python
# 用户代码需要支持容错:
def train():
    init_process_group(...)
    
    # 关键: 从 checkpoint 恢复
    if checkpoint_exists():
        model.load_state_dict(torch.load(ckpt_path))
        start_iter = load_iter()
    else:
        start_iter = 0
    
    for iter in range(start_iter, max_iters):
        train_step(...)
        
        # 定期保存 checkpoint (容错恢复点)
        if iter % save_interval == 0:
            save_checkpoint(model, optimizer, iter)
```

## 6. torchrun CLI 参数

```bash
torchrun \
  --nnodes=4 \                      # 节点数 (或 min:max)
  --nproc_per_node=8 \              # 每节点进程数
  --rdzv_backend=c10d \             # rendezvous 后端
  --rdzv_endpoint=master:29400 \    # coordinator 地址
  --rdzv_id=my_job \                # 任务唯一标识
  --max_restarts=3 \                # 最大重启次数
  --monitor_interval=30 \           # 健康检查间隔 (秒)
  --start_method=spawn \            # 进程启动方式
  --role=trainer \                  # worker 角色
  --redirects=3 \                   # stdout/stderr 重定向
  --log_dir=/logs \                 # 日志目录
  train.py --arg1 val1              # 用户脚本
```

## 7. 与 FlagScale 的集成

```
FlagScale 使用 torchrun 的模式:
──────────────────────────────
run:
  runner:
    backend: torchrun               # 底层使用 torchrun
    nnodes: 4
    nproc_per_node: 8
    hostfile: /path/to/hostfile    # FlagScale 扩展: 主机列表
    
FlagScale 在 torchrun 之上提供:
1. hostfile 解析 → 自动设置 MASTER_ADDR
2. SSH 远程启动 → 多节点协调
3. 日志收集 → 按 rank 归档
4. 环境变量注入 → NCCL 调优参数
```

## 8. 总结

| 组件 | 职责 | 源码位置 |
|------|------|---------|
| torchrun | CLI 入口 | distributed/run.py |
| LaunchAgent | Worker 管理 | elastic/agent/server/ |
| Rendezvous | 多节点协调 | elastic/rendezvous/ |
| HealthCheck | 心跳监控 | elastic/agent/server/api.py |
| WorkerGroup | 进程组状态 | elastic/agent/server/api.py |

```
关键设计决策:
┌────────────────────────────────────────────────────┐
│ 1. 全杀全起: 避免 partial recovery 的一致性问题   │
│ 2. TCPStore rdzv: 零依赖, 够简单够可靠            │
│ 3. 环境变量传递: 用户代码无需感知 elastic          │
│ 4. max_restarts 限制: 防止无限重启循环             │
│ 5. Checkpoint 协作: 容错依赖定期保存               │
└────────────────────────────────────────────────────┘
```

## 9. Multiprocessing 启动方式

### 9.1 start_method 选择

```
进程启动方式:
────────────────────────────────
spawn (默认, 推荐):
  - 全新进程, 不继承父进程状态
  - 安全: 无共享内存泄漏风险
  - 代价: 需要 pickle 所有参数
  - CUDA 兼容: 子进程独立初始化 CUDA context
  
fork:
  - 复制父进程 (COW)
  - 快速启动
  - 危险: CUDA context 无法 fork! 
  - 会导致 segfault 或静默错误
  
forkserver:
  - 折中方案
  - 少数场景使用

WHY spawn 是 CUDA 训练唯一安全选择?
  fork 后子进程继承 CUDA context
  但 CUDA driver 不支持跨进程共享 context
  任何 CUDA 操作都会 crash
```

### 9.2 多节点启动流程

```
4 节点启动时序:
═══════════════════════════════════════════════

[T=0] 调度系统在 4 节点上各启动 torchrun
  Node 0: torchrun --nnodes=4 --rdzv_endpoint=node0:29400
  Node 1: torchrun --nnodes=4 --rdzv_endpoint=node0:29400
  Node 2: torchrun --nnodes=4 --rdzv_endpoint=node0:29400
  Node 3: torchrun --nnodes=4 --rdzv_endpoint=node0:29400

[T=0~30s] Rendezvous 阶段
  Node 0 创建 TCPStore server (port 29400)
  Node 1,2,3 连接 TCPStore
  所有节点 barrier 通过 → 分配 group_rank

[T=30s+] Worker 启动
  每节点启动 8 个 worker 进程
  设置环境变量 (RANK, WORLD_SIZE 等)
  
[T=30s+] Worker 初始化
  每个 worker 调用 init_process_group
  使用 MASTER_ADDR:MASTER_PORT 创建 TCPStore
  创建 NCCL communicator
  开始训练
```

## 10. 错误处理与日志

### 10.1 错误分类

```
Worker 失败类型:
┌──────────────────┬────────────┬──────────────────────────┐
│ 退出码           │ 类型       │ Agent 处理                │
├──────────────────┼────────────┼──────────────────────────┤
│ 0                │ 正常退出   │ 标记 SUCCEEDED            │
│ 1                │ 用户错误   │ 重启 (if restarts < max)  │
│ -9 (SIGKILL)    │ OOM kill   │ 重启 + 日志警告           │
│ -11 (SIGSEGV)   │ 段错误     │ 重启 + dump core          │
│ 137             │ OOM/kill   │ 同 SIGKILL                │
│ timeout         │ Hang 检测  │ Kill all + 重启           │
└──────────────────┴────────────┴──────────────────────────┘
```

### 10.2 日志结构

```
--log_dir=/logs 时的目录结构:
/logs/
├── {run_id}/
│   ├── 0/                  # local_rank=0
│   │   ├── stdout.log
│   │   └── stderr.log
│   ├── 1/                  # local_rank=1
│   │   ├── stdout.log
│   │   └── stderr.log
│   ├── ...
│   └── 7/                  # local_rank=7
│       ├── stdout.log
│       └── stderr.log
```

## 11. 性能影响

```
torchrun 本身的开销:
─────────────────────
1. 进程启动: ~5-10s (spawn + CUDA init)
2. Rendezvous: ~1-30s (取决于节点启动时差)
3. NCCL init: ~5-60s (取决于网络)
4. 运行时监控: <0.1% CPU (轻量 poll)

优化建议:
- 使用 --start_method=spawn (避免 fork CUDA bug)
- 设置合理 --rdzv_timeout (避免等太久或太快放弃)
- 定期 checkpoint (容错恢复点, 减少重训代价)
- 使用 --redirects=3 收集日志 (方便诊断)
```
