# FlagScale Hydra 两级 Config 体系 深度源码分析

## 1. 概述与设计动机

FlagScale 使用 Hydra 框架管理配置，采用**两级 YAML**结构：顶级实验配置（experiment config）通过 `defaults` 引用任务配置（task config）。

**解决的问题**：
- 同一模型不同规模（0.6B/10B/32B）共享实验元数据，只切换模型参数
- 配置的组合爆炸：模型规模 × 并行策略 × 硬件环境
- 命令行参数散乱、不可版本化

**核心设计思想**：将"不变量"（实验框架：task type、entrypoint、envs）与"变量"（模型参数、并行度）分离到两级文件中。

## 2. 源码定位

| 组件 | 路径 | 行数 | 职责 |
|------|------|------|------|
| Hydra 入口 | `flagscale/runner/__init__.py` | 13 | CLI 命令注册，触发 Hydra compose |
| Config 展平 | `flagscale/runner/utils.py:489-510` | 22 | 递归 dict → `--key value` 列表 |
| 训练 Config 补全 | `flagscale/runner/runner_train.py:93-163` | 70 | 补全路径、设默认值 |
| Examples 配置 | `examples/*/conf/` | — | 所有模型的两级配置示例 |

## 3. 两级配置架构

### 3.1 文件组织

```
examples/{model}/conf/
├── train.yaml              # Level 1: 实验配置（顶级）
├── train/
│   ├── 0_6b.yaml           # Level 2: 任务配置（按规模）
│   ├── 10b.yaml
│   └── 32b.yaml
├── serve.yaml              # Level 1: 部署实验配置
├── serve/
│   └── 8b.yaml             # Level 2: 部署任务配置
└── ...
```

### 3.2 Level 1（实验配置）结构

```yaml
# examples/qwen3/conf/train.yaml
defaults:
  - _self_
  - train: 10b              # ← 引用 train/10b.yaml 为 "train" 节点

experiment:                  # 实验元数据
  exp_name: Qwen3-10b-Train
  seed: 42
  exp_dir: /path/to/output
  task:
    type: train              # 任务类型
    backend: megatron        # 后端
    entrypoint: flagscale/train/megatron/train_gpt.py
  runner:                    # 调度配置
    hostfile: null
    no_shared_fs: false
  cmds:
    before_start: export PATH=...   # 环境准备命令
  envs:                      # 环境变量
    CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"

action: run                  # 动作：run / stop / query / dryrun

hydra:
  run:
    dir: ${experiment.exp_dir}/hydra
```

### 3.3 Level 2（任务配置）结构

训练任务配置分为三个固定 section：

```yaml
# examples/qwen3/conf/train/0_6b.yaml
system:                      # 系统/并行配置
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
  sequence_parallel: true
  use_distributed_optimizer: true
  precision:
    bf16: true
  logging:
    log_interval: 1
  checkpoint:
    save_interval: ${experiment.save_steps}  # ← OmegaConf 跨级插值

model:                       # 模型架构参数
  num_layers: 28
  hidden_size: 1024
  num_attention_heads: 16
  seq_length: 4096
  # 训练超参也在 model 下（Megatron 约定）
  micro_batch_size: 1
  global_batch_size: 8
  train_iters: 20
  optimizer:
    weight_decay: 0.1
    lr_scheduler:
      lr: 3.0e-3

data:                        # 数据配置
  data_path: /path/to/data
  tokenizer:
    tokenizer_type: QwenTokenizerFS
    vocab_size: 151936
```

## 4. Config 合并与展平机制

### 4.1 Hydra 合并规则

Hydra 的 `defaults` 列表按顺序合并。`_self_` 的位置决定优先级：

```yaml
defaults:
  - _self_          # 本文件的值作为 base
  - train: 10b      # train/10b.yaml 的值覆盖本文件中 "train:" 节点
```

合并后的 config 结构：
```
root
├── experiment: {...}    # 来自 Level 1
├── train:               # 来自 Level 2 的内容被挂载到此节点
│   ├── system: {...}
│   ├── model: {...}
│   └── data: {...}
├── action: run
└── hydra: {...}
```

### 4.2 OmegaConf 插值

支持跨节点引用（`runner_train.py` 展平前由 OmegaConf resolve）：

```yaml
# Level 2 引用 Level 1 的值
system:
  checkpoint:
    save_interval: ${experiment.save_steps}   # → 10000
    load: ${experiment.load}                  # → null
model:
  seed: ${experiment.seed}                    # → 42
system:
  no_shared_fs: ${experiment.runner.no_shared_fs}  # → false
```

**解析时机**：`OmegaConf.to_container(config, resolve=True)` 在 `_get_args_megatron` 中触发（`runner_train.py:56`）。

### 4.3 展平为命令行参数

**源码**：`flagscale/runner/utils.py:489-510`

```python
def flatten_dict_to_args(config_dict, ignore_keys=[], do_dash_replace=True):
    """递归展平嵌套 dict 为 ['--key', 'value', ...] 列表"""
    args = []
    for key, value in config_dict.items():
        if key in ignore_keys:
            continue
        if do_dash_replace:
            key = key.replace("_", "-")      # tensor_model_parallel_size → --tensor-model-parallel-size
        if isinstance(value, dict):
            args.extend(flatten_dict_to_args(value, ignore_keys))  # 递归展平
        elif isinstance(value, list):
            args.append(f"--{key}")
            for v in value:
                args.append(f"{v}")          # --data-path v1 v2 v3
        elif isinstance(value, bool):
            if value:
                args.append(f"--{key}")      # --bf16 (无 value)
            # False → 不输出（即不设该 flag）
        else:
            args.append(f"--{key}")
            args.append(f"{value}")           # --hidden-size 1024
    return args
```

**关键行为**：
- 下划线 → 短横线（`hidden_size` → `--hidden-size`）
- 嵌套 dict 递归展平（`precision.bf16: true` → `--bf16`）
- bool=False 被跳过（不传该参数）
- 列表值展开为多个位置参数
- `ignore_keys` 过滤内部管理字段

**被忽略的字段**（`runner_train.py:64`）：
```python
ignore_keys = ["log_dir", "details_dir", "scripts_dir", "pids_dir", "straggler_dir"]
```

### 4.4 展平数据流

```
Level 2 YAML (train/0_6b.yaml)
    │
    │  Hydra compose
    ▼
config.train = {system: {...}, model: {...}, data: {...}}
    │
    │  _get_args_megatron (runner_train.py:50-68)
    ▼
config_dict = OmegaConf.to_container(config.train, resolve=True)
    │
    │  merge system + model + data into flat dict
    ▼
new_config_dict = {tensor_model_parallel_size:1, num_layers:28, data_path:..., ...}
    │
    │  flatten_dict_to_args (utils.py:489)
    ▼
["--tensor-model-parallel-size", "1", "--num-layers", "28", "--bf16", "--data-path", "/path", ...]
```

## 5. 边界条件与约束

### 5.1 展平的陷阱

| 陷阱 | 说明 | 影响 |
|------|------|------|
| 键名冲突 | system 和 model 有同名键时，后 update 覆盖前者 | `system.seed` 被 `model.seed` 覆盖 |
| 嵌套 dict 递归 | `optimizer.lr_scheduler.lr` 展平为 `--lr`（丢失层级） | 若两个嵌套都有 `lr` 键则冲突 |
| Bool False | `bf16: false` 不会生成 `--no-bf16` | Megatron 的 store_true 参数无法显式关闭 |
| List 顺序 | 列表展开为位置参数，Megatron 按顺序解析 | `data_path` 多值顺序必须正确 |

### 5.2 跨级插值约束

- 插值目标必须在合并后的完整 config 中存在
- 循环引用会导致 OmegaConf 抛 `InterpolationResolutionError`
- `${experiment.xxx}` 只能引用 Level 1 中定义的字段

### 5.3 命令行覆盖

Hydra 支持 CLI 覆盖任何字段：
```bash
flagscale run --config-path=examples/qwen3/conf --config-name=train \
  train.model.train_iters=100 \
  experiment.exp_dir=/tmp/test \
  train=0_6b                    # 切换 Level 2 文件
```

## 6. 配置切换对比表

| 切换目标 | 修改位置 | 示例 |
|----------|----------|------|
| 模型规模 | `defaults: - train: XXX` 或 CLI `train=XXX` | `train: 0_6b` → `train: 10b` |
| 并行策略 | Level 2 的 `system` 节 | `tensor_model_parallel_size: 4` |
| 实验目录 | Level 1 的 `experiment.exp_dir` | 换路径 |
| 节点数 | Level 1 的 `experiment.runner.hostfile` | 指向新 hostfile |
| 环境变量 | Level 1 的 `experiment.envs` | `CUDA_VISIBLE_DEVICES: "0,1"` |
| 数据集 | Level 2 的 `data` 节 | `data_path: /new/path` |

## 7. 调优指南

- **新增模型规模**：在 `conf/train/` 下新建 YAML，只写 system/model/data
- **快速实验**：不改文件，用 CLI 覆盖 `train.model.train_iters=5`
- **多节点切换**：只需修改 Level 1 的 hostfile + envs，Level 2 不动
- **避免键名冲突**：不要在 system 和 model 中使用同名 key
- **调试展平结果**：`flagscale run ... action=dryrun` 查看生成的 bash 脚本中的完整命令
