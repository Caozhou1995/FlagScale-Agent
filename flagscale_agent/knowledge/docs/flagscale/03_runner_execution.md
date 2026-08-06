# FlagScale Runner 执行链路 深度源码分析

## 1. 概述与设计动机

Runner 是 FlagScale 的任务执行引擎——从 CLI 命令到训练进程启动的完整链路。设计目标是将 **配置解析 → Runner 选择 → 脚本生成 → 多节点调度** 解耦为独立阶段。

**核心问题**：大模型训练启动涉及多层抽象（Hydra config、torchrun 参数、SSH 调度、环境变量注入），如何在保持灵活性的同时确保可复现性？

**解决方案**：所有中间产物（生成的 bash 脚本、展平的参数）持久化到 `scripts/` 目录，任何一次训练都可以通过回放脚本精确复现。

## 2. 源码定位

| 组件 | 路径 | 行号 | 职责 |
|------|------|------|------|
| CLI 入口 | `flagscale/cli.py:132-170` | 39 | typer 命令，构造 sys.argv |
| Hydra 入口 | `flagscale/run.py:148` | 1 | `@hydra.main` 装饰器 |
| Runner 分发 | `flagscale/run.py:70-102` | 33 | 新旧架构路由 |
| Action 执行 | `flagscale/run.py:119-146` | 28 | run/dryrun/stop/query 分发 |
| 配置补全 | `runner_train.py:93-163` | 70 | 路径解析、默认值补全 |
| torchrun 命令构建 | `runner_train.py:166-230` | 65 | 组装 torchrun 参数 |
| 脚本生成 | `runner_train.py:233-340` | 108 | 生成 host_N_IP_run.sh |
| 多节点调度 | `runner_train.py:502-596` | 95 | multiprocessing.Pool 并发 SSH |

## 3. 完整执行时序图

```
User                  CLI (cli.py)        Hydra (run.py)      Runner              Remote Host
 │                        │                    │                 │                     │
 │  flagscale run ...     │                    │                 │                     │
 │───────────────────────>│                    │                 │                     │
 │                        │  sys.argv = [      │                 │                     │
 │                        │    "run.py",       │                 │                     │
 │                        │    "--config-path",│                 │                     │
 │                        │    "--config-name",│                 │                     │
 │                        │    "action=run"    │                 │                     │
 │                        │  ]                 │                 │                     │
 │                        │───────────────────>│                 │                     │
 │                        │                    │  Hydra compose  │                     │
 │                        │                    │  (merge L1+L2)  │                     │
 │                        │                    │                 │                     │
 │                        │                    │  os.chdir(orig) │                     │
 │                        │                    │  (L157)         │                     │
 │                        │                    │                 │                     │
 │                        │                    │  get_runner()   │                     │
 │                        │                    │───────────────> │                     │
 │                        │                    │                 │  _prepare()         │
 │                        │                    │                 │  ├─ _update_config  │
 │                        │                    │                 │  │   (补全路径)      │
 │                        │                    │                 │  ├─ _get_args       │
 │                        │                    │                 │  │   (展平config)   │
 │                        │                    │                 │  └─ parse_hostfile  │
 │                        │                    │                 │                     │
 │                        │                    │  execute_action │                     │
 │                        │                    │───────────────> │                     │
 │                        │                    │                 │  runner.run()       │
 │                        │                    │                 │                     │
 │                        │                    │                 │  for each node:     │
 │                        │                    │                 │  ├─ _run_each()     │
 │                        │                    │                 │  │  ├─ build cmd    │
 │                        │                    │                 │  │  ├─ gen script   │
 │                        │                    │                 │  │  └─ SSH execute ─────────────>│
 │                        │                    │                 │  │                  │  bash run.sh
 │                        │                    │                 │  │                  │  └─ torchrun
 │                        │                    │                 │  └─ (parallel)      │     └─ train
 │                        │                    │                 │                     │
```

## 4. 核心阶段详解

### 4.1 阶段 1：CLI → Hydra

**入口**：`flagscale/cli.py:71` (`run_task` 函数)

```python
# cli.py:71-93
def run_task(config_path, config_name, action, extra_args=None):
    from flagscale.run import main as run_main
    args = ["run.py", f"--config-path={config_path}", f"--config-name={config_name}", f"action={action}"]
    if extra_args:
        args.extend(extra_args)
    sys.argv = args       # 伪装命令行参数
    run_main()            # 触发 @hydra.main
```

**设计动机**：Hydra 从 `sys.argv` 解析参数，通过替换 argv 实现从 typer CLI 到 Hydra 的桥接。

### 4.2 阶段 2：Runner 选择

**入口**：`flagscale/run.py:70-102` (`get_runner` 函数)

路由逻辑：
```
FLAGSCALE_USE_V1=1 (默认)
  → Runner(config)                    # 新架构：Backend + Launcher
  
FLAGSCALE_USE_V1=0
  → LEGACY_RUNNER_MAP[task_type]      # 旧架构：SSHTrainRunner 等
```

**当前状态**（`run.py:36-37`）：
- `FLAGSCALE_USE_V1 = os.environ.get("FLAGSCALE_USE_V1", "1")` — **默认走新架构**
- 但训练的新架构 `Runner` 内部仍用 `SSHTrainRunner` 的 Backend（MegatronBackend → 同代码）

### 4.3 阶段 3：配置补全

**入口**：`runner_train.py:93-163` (`_update_config_train`)

补全的路径（全部基于 `exp_dir`）：
```
exp_dir/                          ← experiment.exp_dir
├── checkpoints/                  ← system.checkpoint.save (默认)
├── logs/                         ← system.logging.log_dir (默认)
│   ├── scripts/                  ← logging.scripts_dir
│   ├── pids/                     ← logging.pids_dir
│   ├── details/                  ← logging.details_dir
│   └── straggler/                ← logging.straggler_dir
├── tensorboard/                  ← logging.tensorboard_dir (默认)
└── wandb/                        ← logging.wandb_save_dir (默认)
```

### 4.4 阶段 4：torchrun 命令构建

**入口**：`runner_train.py:166-230` (`_get_runner_cmd_train`)

构建的 torchrun 参数：
```bash
torchrun \
  --nnodes=<N> \
  --node-rank=<rank> \
  --nproc-per-node=<gpus> \
  --rdzv-backend=static \
  --rdzv-endpoint=<master_ip>:<port> \
  --rdzv-id=<unique_id> \
  --log-dir=<details_dir>/host_<rank>_<ip>/<timestamp> \
  --redirects=3 \
  --tee=3 \
  <entrypoint> <展平的训练参数>
```

**log-dir 路径构建逻辑**（`runner_train.py:178-185`）：
```python
log_dir = logging_config.details_dir                          # 基础路径
if no_shared_fs:
    log_dir = os.path.join(log_dir, "host")                   # 无共享 FS：统一目录名
else:
    log_dir = os.path.join(log_dir, f"host_{node_rank}_{host}")  # 有共享 FS：含节点信息
log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d_%H%M%S.%f"))  # 时间戳
```

### 4.5 阶段 5：脚本生成

**入口**：`runner_train.py:233-340` (`_generate_run_script_train`)

生成的 bash 脚本结构：
```bash
#!/bin/bash
# 1. 创建目录
mkdir -p <log_dir> <details_dir> <scripts_dir> <pids_dir>

# 2. before_start 命令（用户自定义环境准备）
export PATH=/workspace/envs/flagscale-train/bin:$PATH && ulimit -n 1048576

# 3. cd 到 FlagScale 根目录
cd <pkg_dir>

# 4. 执行 torchrun（后台或前台）
<export_envs> torchrun <runner_args> <entrypoint> <user_args> > <host.output> 2>&1 &
echo $! > <pid_file>
```

### 4.6 阶段 6：多节点调度

**入口**：`runner_train.py:502-596` (`SSHTrainRunner.run`)

```python
# 并发调度所有节点（runner_train.py:543-551）
num_processes = min(nnodes, multiprocessing.cpu_count())
with multiprocessing.Pool(processes=num_processes) as pool:
    tasks = []
    for node_rank, (host, resource_info) in enumerate(self.resources.items()):
        if node_rank >= nnodes:
            break
        tasks.append((self._run_each, node_rank, host, ...))
    pool.starmap(run_node, tasks)
```

远程执行三步（`runner_train.py:471-500`）：
1. `ssh host "mkdir -p <scripts_dir>"` — 确保目录存在
2. `scp run.sh host:<scripts_dir>/` — 传送脚本（仅 no_shared_fs）
3. `ssh host "bash <run_script>"` — 执行

## 5. 新旧架构对比表

| 维度 | 新架构 (FLAGSCALE_USE_V1=1) | 旧架构 (FLAGSCALE_USE_V1=0) |
|------|---------------------------|---------------------------|
| 入口 | `Runner(config)` → Factory 分发 | `SSHTrainRunner(config)` 直接实例化 |
| 训练实际路径 | MegatronBackend + SshLauncher | SSHTrainRunner 内部方法 |
| 推理/部署 | 完全走 Backend+Launcher | SSHInferenceRunner/SSHServeRunner |
| 新增任务 | 注册 Backend + Launcher | 新增 Runner 子类 |
| 默认状态 | **启用**（环境变量默认 "1"） | 需手动 `FLAGSCALE_USE_V1=0` |
| 代码位置 | runner_base.py + runner_factory.py | runner_train.py 内部 |

## 6. 边界条件与约束

1. **cwd 恢复**：Hydra 会 chdir 到 output dir，`run.py:157` 立即 `os.chdir(hydra.utils.get_original_cwd())` 恢复，否则所有相对路径失效
2. **rdzv_id 唯一性**：使用 `datetime.now().strftime("%Y%m%d_%H%M%S.%f")` 确保每次启动唯一
3. **前台 vs 后台**：`background=False` 时 node_rank=0 的 SSH 开启 `stream_output=True`，直接 tee 到控制台
4. **dryrun 模式**：只生成脚本到 `scripts/`，不实际执行 SSH
5. **Cloud Runner**：`runner.type=cloud` 时走 `CloudTrainRunner`，不经过 SSH/hostfile

## 7. 调优指南

- **调试启动问题**：`action=dryrun` 查看生成的脚本，确认 torchrun 参数和环境变量
- **无共享文件系统**：设 `runner.no_shared_fs=true`，Runner 自动 SCP 脚本到远端
- **自定义 torchrun 参数**：在 Level 1 的 `experiment.runner` 中添加，会被透传
- **监控集成**：`runner.enable_monitoring=true` 启用 `MonitorService` 弹性监控
