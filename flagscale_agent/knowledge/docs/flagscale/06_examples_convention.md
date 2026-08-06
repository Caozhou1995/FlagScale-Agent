# FlagScale Examples 目录规范 深度源码分析

## 1. 概述

`examples/` 目录是 FlagScale 的模型配置仓库，包含 38+ 个模型系列的训练/推理/部署配置。每个模型目录遵循统一的配置规范。

**核心价值**：用户通过 `flagscale run --config-path examples/<model>/conf --config-name train` 即可启动训练，无需理解底层 Megatron 参数。

## 2. 标准目录结构

```
examples/{model}/
├── conf/                           # 配置目录（必须）
│   ├── train.yaml                  # Level 1: 训练实验配置
│   ├── train/                      # Level 2: 训练任务配置
│   │   ├── 0_6b.yaml              # 按规模命名
│   │   ├── 10b.yaml
│   │   └── 32b.yaml
│   ├── serve.yaml                  # Level 1: 部署实验配置（可选）
│   ├── serve/                      # Level 2: 部署任务配置（可选）
│   │   └── 8b.yaml
│   ├── inference_fl.yaml           # Level 1: 推理配置（可选）
│   ├── inference/                  # Level 2: 推理任务配置（可选）
│   ├── rl.yaml                     # Level 1: 强化学习配置（可选）
│   └── hostfile.txt                # 节点列表（可选）
├── README.md                       # 模型说明文档（可选）
└── src/                            # 模型特有源码（可选，少用）
```

## 3. 实际模型分析

### 3.1 覆盖的模型系列（截至当前）

| 类别 | 模型 | 支持任务 |
|------|------|----------|
| **通用 LLM** | qwen2/2.5/3/3.5, llama2/3, deepseek_v3/r1, kimi_k2, grok2, rwkv | train, serve |
| **多模态** | qwen2_5_vl, qwen3_vl, llava1_5, llava_onevision, emu3/3.5, minicpm_v_4, minicpm_o_2.6 | train, serve |
| **图像生成** | flux_1_dev, openjourney, wan2_1, qwen_image | train |
| **具身智能** | gr00t_n1_5, pi0/pi0_5, robobrain/2/2_5/x0/x0_5, qwen_gr00t | train |
| **推理模型** | qwen3_o, qwq, deepseek_r1, deepseek_r1_distill_qwen | train, rl |
| **MoE** | deepseek_v3, mixtral, kimi_k2, ernie45 | train |

### 3.2 配置变体命名规范

| 模式 | 示例 | 含义 |
|------|------|------|
| `{size}.yaml` | `0_6b.yaml`, `10b.yaml` | 模型规模 |
| `{size}_finetune.yaml` | `16b_a3b_finetune.yaml` | 微调配置 |
| `train_te_fl.yaml` | — | 使用 TransformerEngine |
| `train_engram.yaml` | — | 使用 Energon 数据管线 |
| `train_auto_tuner.yaml` | — | 自动调优实验 |
| `train_hetero.yaml` | — | 异构并行 |
| `serve_atmb.yaml` | — | ATMB 部署策略 |

### 3.3 典型 Level 1 配置对比

| 字段 | qwen3/train.yaml | deepseek_v3/train.yaml |
|------|-------------------|------------------------|
| `task.type` | train | train |
| `task.backend` | megatron | megatron |
| `task.entrypoint` | flagscale/train/megatron/train_gpt.py | flagscale/train/megatron/train_gpt.py |
| `runner.hostfile` | null（单机） | conf/hostfile.txt |
| `envs` | 基础 CUDA 设置 | 含 NCCL 调优参数 |
| `cmds.before_start` | conda activate | conda activate + ulimit |

## 4. Level 1 必填字段清单

```yaml
experiment:
  exp_name: <str>               # 实验名称（必填）
  exp_dir: <path>               # 输出根目录（必填）
  seed: <int>                   # 随机种子
  task:
    type: train                 # 任务类型（必填）
    backend: megatron           # 后端（必填）
    entrypoint: <path>          # 训练脚本路径（必填）
  runner:
    hostfile: <path|null>       # 多节点时必填
    no_shared_fs: <bool>        # 无共享存储时设 true

defaults:
  - _self_
  - train: <config_name>        # 引用 Level 2 文件（必填）

action: run                     # 默认动作
```

## 5. Level 2 最小可用模板

```yaml
system:
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
  precision:
    bf16: true
  logging:
    log_interval: 1
  checkpoint:
    save_interval: 1000

model:
  num_layers: 28
  hidden_size: 1024
  num_attention_heads: 16
  seq_length: 4096
  micro_batch_size: 1
  global_batch_size: 8
  train_iters: 100
  optimizer:
    weight_decay: 0.1
    lr_scheduler:
      lr: 1.0e-4
      min_lr: 1.0e-5
      lr_warmup_iters: 10
      lr_decay_style: cosine
      lr_decay_iters: 100

data:
  data_path: <actual_path_to_processed_data>
  split: "1"
  tokenizer:
    tokenizer_type: <type>
    tokenizer_path: <actual_path>
    vocab_size: <int>
```

## 6. 新增模型配置步骤

1. 创建目录：`mkdir -p examples/{model}/conf/train/`
2. 创建 Level 1：`examples/{model}/conf/train.yaml`（从类似模型复制）
3. 创建 Level 2：`examples/{model}/conf/train/{size}.yaml`
4. 修改 Level 1 的 `defaults` 引用正确的 Level 2
5. 验证：`flagscale run -p examples/{model}/conf -n train -a dryrun`
6. 检查生成的脚本：`cat {exp_dir}/logs/scripts/host_0_*_run.sh`

## 7. 常见错误

| 错误 | 原因 | 修复 |
|------|------|------|
| `MissingMandatoryValue` | Level 2 用了 `${experiment.xxx}` 但 Level 1 没定义 | 在 Level 1 补上字段 |
| `Could not load train/xxx` | defaults 引用的文件名不存在 | 确认 `train/` 下有对应 YAML |
| `KeyError: 'train'` | Level 1 没有 `defaults: - train: xxx` | 添加正确的 defaults |
| 参数未生效 | 键名放错 section（如 lr 放到 system） | 参照第 4 章字段归属表 |
