# FlagScale 训练 Runner 详解 深度源码分析

## 1. 概述

训练 Runner 负责将 config 转化为可在多节点执行的训练进程。核心产物是一组 bash 脚本（run/stop），通过 SSH 分发到各节点执行。

**源码定位**：`flagscale/runner/runner_train.py`

## 2. 输出目录结构

Runner 自动在 `experiment.exp_dir` 下构建以下目录树。目录结构由两层协作生成：
1. **FlagScale Runner** 构建到 `{timestamp}` 层并传给 torchrun 的 `--log-dir`
2. **torchrun (Elastic Launch)** 在其下创建 `{rdzv_id}/attempt_{N}/{local_rank}/` 结构

```
{exp_dir}/                                    ← experiment.exp_dir
├── checkpoints/                              ← system.checkpoint.save (默认)
│   ├── iter_0001000/
│   └── iter_0005000/
├── logs/                                     ← system.logging.log_dir
│   ├── host_0_10.0.0.1.output                ← 主进程 stdout/stderr 汇总（脚本重定向）
│   ├── host_1_10.0.0.2.output                ← 第二节点汇总
│   ├── details/                              ← system.logging.details_dir
│   │   ├── host_0_10.0.0.1/                  ← FlagScale 按 node_rank + IP 创建
│   │   │   └── 20250115_143022.123456/       ← FlagScale 按时间戳创建（每次 launch 唯一）
│   │   │       └── default/                  ← torchrun 按 rdzv_id 创建（默认 "default"）
│   │   │           ├── attempt_0/            ← torchrun 容错：第一次启动
│   │   │           │   ├── 0/                ← local_rank 0
│   │   │           │   │   ├── stdout.log
│   │   │           │   │   ├── stderr.log
│   │   │           │   │   └── error.json
│   │   │           │   ├── 1/                ← local_rank 1
│   │   │           │   │   ├── stdout.log
│   │   │           │   │   ├── stderr.log
│   │   │           │   │   └── error.json
│   │   │           │   └── .../              ← 每 GPU 一个子目录
│   │   │           ├── attempt_1/            ← torchrun 容错：第一次自动重启
│   │   │           │   └── .../
│   │   │           └── attempt_2/            ← torchrun 容错：第二次自动重启
│   │   │               └── .../
│   │   └── host_1_10.0.0.2/
│   │       └── 20250115_143022.789012/
│   │           └── default/
│   │               └── attempt_0/
│   │                   └── .../
│   ├── scripts/                              ← system.logging.scripts_dir
│   │   ├── host_0_10.0.0.1_run.sh            ← 生成的启动脚本
│   │   ├── host_0_10.0.0.1_stop.sh           ← 生成的停止脚本
│   │   ├── host_1_10.0.0.2_run.sh
│   │   └── host_1_10.0.0.2_stop.sh
│   ├── pids/                                 ← system.logging.pids_dir
│   │   ├── host_0_10.0.0.1.pid
│   │   └── host_1_10.0.0.2.pid
│   └── straggler/                            ← system.logging.straggler_dir
├── tensorboard/                              ← system.logging.tensorboard_dir
└── wandb/                                    ← system.logging.wandb_save_dir
```

**关键细节**：
- `host_{node_rank}_{ip}` 命名模式在有共享 FS 时使用（`no_shared_fs=false`）
- 无共享 FS 时统一用 `host` 目录名（各节点本地存储）
- **时间戳层**（`20250115_143022.123456`）由 FlagScale Runner 生成，每次 launch 唯一
- **rdzv_id 层**（默认 `"default"`）由 torchrun 创建，用于标识一次 elastic job
- **attempt_N 层**由 torchrun 容错机制管理：`max_restarts` 控制最大重启次数，每次重启创建新的 `attempt_{N}` 目录
- torchrun 的 `--redirects=3 --tee=3` 表示 stdout 和 stderr 都重定向到日志文件并同时输出到终端
- 同一次 launch（同一时间戳）内的多次 torchrun attempt 是**容错自动重启**，不是用户手动重试

**两种 "attempt" 的区别**：
| 概念 | 粒度 | 产生方式 | 目录标识 |
|------|------|----------|----------|
| torchrun attempt | 同一 launch 内的容错重启 | 自动（Worker crash → Agent 重启） | `attempt_0`, `attempt_1` |
| 实验 attempt（用户） | 不同次 launch（改了配置重跑） | 手动（用户修改 config 后重新启动） | 不同时间戳目录 |

## 3. 路径构建源码

### 3.1 目录补全 (`_update_config_train`, L93-163)

```python
# runner_train.py:93-163 (核心逻辑)
def _update_config_train(config):
    exp_dir = config.experiment.exp_dir

    # 日志根目录
    if not logging_config.get("log_dir"):
        logging_config.log_dir = os.path.join(exp_dir, "logs")

    # 子目录
    logging_config.details_dir = os.path.join(logging_config.log_dir, "details")
    logging_config.scripts_dir = os.path.join(logging_config.log_dir, "scripts")
    logging_config.pids_dir = os.path.join(logging_config.log_dir, "pids")
    logging_config.straggler_dir = os.path.join(logging_config.log_dir, "straggler")

    # Checkpoint 路径
    if not checkpoint_config.get("save"):
        checkpoint_config.save = os.path.join(exp_dir, "checkpoints")

    # Tensorboard
    if not logging_config.get("tensorboard_dir"):
        logging_config.tensorboard_dir = os.path.join(exp_dir, "tensorboard")
```

### 3.2 torchrun log-dir 构建 (`_get_runner_cmd_train`, L166-230)

```python
# runner_train.py:177-218 (关键逻辑)
rdzv_id = runner_config.get("rdzv_id", "default")       # 默认 "default"
log_dir = runner_config.get("log_dir", logging_config.details_dir)
no_shared_fs = runner_config.get("no_shared_fs", False)
if no_shared_fs:
    log_dir = os.path.join(log_dir, "host")
else:
    log_dir = os.path.join(log_dir, f"host_{node_rank}_{host}")
log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d_%H%M%S.%f"))
# ...
# 最终传给 torchrun:
runner_args["log_dir"] = log_dir if backend == "torchrun" else os.path.join(log_dir, rdzv_id)
# torchrun 内部再创建: {log_dir}/{rdzv_id}/attempt_{restart_count}/{local_rank}/
```

**完整路径推导**：
```
FlagScale 构建:  {details_dir}/host_{node_rank}_{ip}/{timestamp}/
torchrun 追加:   {rdzv_id}/attempt_{N}/{local_rank}/stdout.log
最终完整路径:    {details_dir}/host_0_10.0.0.1/20250115_143022.123456/default/attempt_0/0/stdout.log
```

**torchrun 容错机制**（源码：`torch/distributed/elastic/multiprocessing/api.py:283-315`）：
- `restart_count` 从环境变量 `TORCHELASTIC_RESTART_COUNT` 获取
- 每次容错重启时 `shutil.rmtree(attempt_log_dir)` 后重建——即每个 attempt 目录是**覆盖写**
- `max_restarts` 参数（runner config 中配置）控制最大容错重启次数

### 3.3 host_output 文件路径 (`_generate_run_script_train`, L240-255)

```python
# runner_train.py:245-249
no_shared_fs = config.experiment.runner.get("no_shared_fs", False)
if no_shared_fs:
    host_output_file = os.path.join(logging_config.log_dir, "host.output")
else:
    host_output_file = os.path.join(logging_config.log_dir, f"host_{node_rank}_{host}.output")
```

## 4. 生成的 bash 脚本结构

### 4.1 run 脚本 (`host_{rank}_{ip}_run.sh`)

```bash
#!/bin/bash
# === 目录创建 ===
mkdir -p {log_dir} {details_dir} {scripts_dir} {pids_dir}

# === before_start 命令（用户自定义） ===
export PATH=/workspace/envs/train/bin:$PATH && ulimit -n 1048576

# === cd 到 FlagScale 根目录 ===
cd {pkg_dir}

# === 环境变量注入 ===
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4

# === torchrun 命令 ===
torchrun \
  --nnodes=2 \
  --node-rank=0 \
  --nproc-per-node=8 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=10.0.0.1:29500 \
  --rdzv-id=20250115_143022.123456 \
  --log-dir={details_dir}/host_0_10.0.0.1/20250115_143022.123456 \
  --redirects=3 \
  --tee=3 \
  flagscale/train/megatron/train_gpt.py \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 28 \
    --hidden-size 1024 \
    --bf16 \
    --data-path /data/train_text_document \
    ... \
  > {log_dir}/host_0_10.0.0.1.output 2>&1 &

echo $! > {pids_dir}/host_0_10.0.0.1.pid
```

### 4.2 stop 脚本 (`host_{rank}_{ip}_stop.sh`)

```bash
#!/bin/bash
pkill -P $(cat {pids_dir}/host_0_10.0.0.1.pid) 2>/dev/null
kill $(cat {pids_dir}/host_0_10.0.0.1.pid) 2>/dev/null
# after_stop 命令（用户自定义）
```

## 5. 多节点调度流程

### 5.1 Hostfile 解析

```
# hostfile 格式
10.0.0.1 slots=8
10.0.0.2 slots=8
```

**源码**：`runner_train.py:380-420` (`_parse_hostfile`)

### 5.2 并发 SSH 执行

```python
# runner_train.py:543-551 (简化)
num_processes = min(nnodes, multiprocessing.cpu_count())
with multiprocessing.Pool(processes=num_processes) as pool:
    for node_rank, (host, _) in enumerate(resources.items()):
        # 对每个节点：
        # 1. 生成 run 脚本
        # 2. 如 no_shared_fs：SCP 脚本到远端
        # 3. SSH 执行 bash run.sh
        pool.apply_async(run_node, args=(node_rank, host, ...))
    pool.close()
    pool.join()
```

### 5.3 SSH 命令模板

```bash
# 共享 FS（默认）：脚本已通过共享存储可见
ssh -p {ssh_port} {host} "bash {scripts_dir}/host_{rank}_{host}_run.sh"

# 无共享 FS：先传送脚本
scp -P {ssh_port} {run_script} {host}:{remote_scripts_dir}/
ssh -p {ssh_port} {host} "bash {remote_scripts_dir}/host_{rank}_{host}_run.sh"
```

## 6. 日志定位指南

### 6.1 找到最新训练日志

```bash
# 方法 1：按时间戳排序
ls -lt {exp_dir}/logs/details/host_0_*/  # 最新的时间戳目录在最前

# 方法 2：找到 loss 所在 rank（最后 PP stage，rank = tp*pp - tp 或更高）
cat {exp_dir}/logs/details/host_0_10.0.0.1/20250115_143022.123456/7/stdout.log | grep "iteration"

# 方法 3：host.output 汇总文件
cat {exp_dir}/logs/host_0_10.0.0.1.output
```

### 6.2 crash 诊断

```bash
# 检查所有 rank 的 stderr
for f in {exp_dir}/logs/details/host_*/*/*/stderr.log; do
    if [ -s "$f" ]; then echo "=== $f ===" && tail -20 "$f"; fi
done
```

## 7. 边界条件

1. **时间戳唯一性**：同一秒内多次启动可能冲突（微秒 `%f` 降低概率但不消除）
2. **rdzv_id**：默认值 "default"——多个训练用同一 master 时必须设不同 rdzv_id
3. **background 模式**：默认 `background=True`，SSH 后台执行并立即返回；`background=False`（test 模式）前台执行，node_rank=0 流式输出
4. **redirects=3**：torchrun 的 `3` 表示 `stdout|stderr` 都重定向（bit flag: 1=stdout, 2=stderr, 3=both）
5. **tee=3**：同时输出到终端和文件（bit flag 同上）
