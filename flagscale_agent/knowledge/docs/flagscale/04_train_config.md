# FlagScale 训练 Config 字段全表 深度源码分析

## 1. 概述

FlagScale 训练的 Level 2 配置（task config）分为三个固定 section：`system`、`model`、`data`。这三个 section 最终被 merge 并展平为 Megatron 的 `--key value` 命令行参数。

**源码定位**：`flagscale/runner/runner_train.py:50-68` (`_get_args_megatron`)

```python
# runner_train.py:50-68 (简化)
def _get_args_megatron(config):
    config_dict = OmegaConf.to_container(config.train, resolve=True)
    system = config_dict.get("system", {})
    model = config_dict.get("model", {})
    data = config_dict.get("data", {})
    # merge 顺序：system 先，model 后（model 覆盖 system 同名键）
    new_config_dict = {}
    new_config_dict.update(system)
    new_config_dict.update(model)
    new_config_dict.update(data)
    # 移除内部管理字段
    ignore_keys = ["log_dir", "details_dir", "scripts_dir", "pids_dir", "straggler_dir"]
    return flatten_dict_to_args(new_config_dict, ignore_keys=ignore_keys)
```

**关键约束**：三个 section 的键名**不得冲突**——后 update 覆盖前者。

## 2. Section 归属规则

| Section | 归属内容 | 设计原则 |
|---------|----------|----------|
| `system` | 并行策略、分布式优化、精度、日志、检查点 | **与模型架构无关**的系统级参数 |
| `model` | 架构参数、训练超参（lr/batch/iters）、优化器 | **模型特有**的参数 |
| `data` | 数据路径、tokenizer、数据分割 | **数据管线**参数 |

**误放检测规则**（基于 merge 顺序推导）：
- `seed` → 通常在 `model` 中（Megatron 约定），但也出现在 `system` 中 → **应统一放 model**
- `num_workers` → 出现在 `system` 中（并行 workers），但也是 data loader 参数 → **FlagScale 约定放 system**
- `finetune` → 训练模式标记，放 `system`
- `micro_batch_size` / `global_batch_size` → 放 `model`（Megatron 约定）

## 3. system 字段详解

### 3.1 并行策略

| 字段 | 类型 | 默认值 | 说明 | Megatron 参数名 |
|------|------|--------|------|----------------|
| `tensor_model_parallel_size` | int | 1 | TP 度 | `--tensor-model-parallel-size` |
| `pipeline_model_parallel_size` | int | 1 | PP 度 | `--pipeline-model-parallel-size` |
| `context_parallel_size` | int | 1 | CP 度 | `--context-parallel-size` |
| `expert_model_parallel_size` | int | 1 | EP 度（MoE） | `--expert-model-parallel-size` |
| `sequence_parallel` | bool | false | SP（TP 内 sequence 切分） | `--sequence-parallel` |
| `decoder_first_pipeline_num_layers` | int | null | PP 首段层数（非均分） | `--decoder-first-pipeline-num-layers` |

### 3.2 分布式优化

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_distributed_optimizer` | bool | false | ZeRO-1 优化器分片 |
| `overlap_grad_reduce` | bool | false | 梯度 all-reduce 与计算 overlap |
| `overlap_param_gather` | bool | false | 参数 gather 与计算 overlap |

### 3.3 精度配置

嵌套在 `system.precision` 下：

| 字段 | 类型 | 说明 | 展平后参数 |
|------|------|------|-----------|
| `bf16` | bool | 启用 BF16 混合精度 | `--bf16` |
| `fp16` | bool | 启用 FP16 混合精度 | `--fp16` |
| `attention_softmax_in_fp32` | bool | softmax 用 FP32 | `--attention-softmax-in-fp32` |
| `accumulate_allreduce_grads_in_fp32` | bool | 梯度聚合用 FP32 | `--accumulate-allreduce-grads-in-fp32` |

### 3.4 日志配置

嵌套在 `system.logging` 下：

| 字段 | 类型 | 说明 |
|------|------|------|
| `log_interval` | int | 每 N 步打印 loss |
| `tensorboard_log_interval` | int | TB 记录间隔 |
| `log_throughput` | bool | 打印 tokens/sec |
| `log_params_norm` | bool | 打印参数范数 |
| `log_num_zeros_in_grad` | bool | 打印梯度零值数 |
| `log_memory_to_tensorboard` | bool | 记录显存到 TB |
| `log_timers_to_tensorboard` | bool | 记录 timer 到 TB |

**注意**：`log_dir`、`details_dir`、`scripts_dir`、`pids_dir`、`straggler_dir` 由 Runner 自动填充，不需要用户设置。

### 3.5 检查点配置

嵌套在 `system.checkpoint` 下：

| 字段 | 类型 | 说明 |
|------|------|------|
| `save_interval` | int | 每 N 步保存 |
| `load` | str/null | 加载 checkpoint 路径 |
| `ckpt_format` | str | 格式（torch/torch_dist） |
| `no_save_optim` | bool | 不保存优化器状态 |
| `no_save_rng` | bool | 不保存 RNG 状态 |
| `no_load_optim` | bool | 不加载优化器状态 |

### 3.6 其他系统参数

| 字段 | 类型 | 说明 |
|------|------|------|
| `no_shared_fs` | bool | 无共享文件系统标记 |
| `num_workers` | int | dataloader workers |
| `disable_bias_linear` | bool | 线性层无 bias |
| `reset_position_ids` | bool | 重置 position id（多文档打包） |
| `reset_attention_mask` | bool | 重置 attention mask |
| `qk_layernorm` | bool | QK 层归一化 |
| `finetune` | bool | 微调模式 |

## 4. model 字段详解

### 4.1 架构参数

| 字段 | 类型 | 说明 |
|------|------|------|
| `num_layers` | int | Transformer 层数 |
| `hidden_size` | int | 隐藏维度 |
| `ffn_hidden_size` | int | FFN 中间维度 |
| `num_attention_heads` | int | 注意力头数 |
| `kv_channels` | int | KV 通道数（head_dim） |
| `group_query_attention` | bool | 启用 GQA |
| `num_query_groups` | int | GQA key/value 头数 |
| `seq_length` | int | 序列长度 |
| `max_position_embeddings` | int | 最大位置编码长度 |
| `norm_epsilon` | float | 归一化 epsilon |
| `normalization` | str | 归一化类型（RMSNorm/LayerNorm） |
| `swiglu` | bool | 启用 SwiGLU |
| `use_rotary_position_embeddings` | bool | 启用 RoPE |
| `rotary_base` | int | RoPE base frequency |
| `position_embedding_type` | str | 位置编码类型（rope/learned） |
| `no_position_embedding` | bool | 不使用绝对位置编码 |
| `no_rope_fusion` | bool | 不融合 RoPE 计算 |
| `untie_embeddings_and_output_weights` | bool | 输入输出不共享权重 |
| `transformer_impl` | str | 实现后端（transformer_engine/local） |
| `attention_dropout` | float | 注意力 dropout |
| `hidden_dropout` | float | 隐藏层 dropout |
| `init_method_std` | float | 初始化标准差 |
| `clip_grad` | float | 梯度裁剪阈值 |

### 4.2 训练超参

| 字段 | 类型 | 说明 |
|------|------|------|
| `seed` | int | 随机种子 |
| `micro_batch_size` | int | 每 GPU 每微步 batch |
| `global_batch_size` | int | 全局 batch size |
| `train_iters` | int | 总训练步数 |
| `eval_iters` | int | 验证步数 |
| `eval_interval` | int | 验证间隔 |

### 4.3 优化器配置

嵌套在 `model.optimizer` 下：

| 字段 | 类型 | 说明 |
|------|------|------|
| `weight_decay` | float | 权重衰减 |
| `adam_beta1` | float | Adam β₁ |
| `adam_beta2` | float | Adam β₂ |
| `lr_scheduler.lr` | float | 学习率 |
| `lr_scheduler.min_lr` | float | 最小学习率 |
| `lr_scheduler.lr_warmup_iters` | int | warmup 步数 |
| `lr_scheduler.lr_decay_style` | str | 衰减策略（cosine/linear） |
| `lr_scheduler.lr_decay_iters` | int | 衰减持续步数 |

## 5. data 字段详解

| 字段 | 类型 | 说明 |
|------|------|------|
| `data_path` | str/list | 数据文件路径（.bin/.idx 前缀，或多个路径 blend） |
| `split` | str | 训练/验证/测试比例（如 "949,50,1"） |
| `no_mmap_bin_files` | bool | 不使用 mmap（适合网络存储） |
| `tokenizer.tokenizer_type` | str | Tokenizer 类型 |
| `tokenizer.tokenizer_path` | str | Tokenizer 模型路径 |
| `tokenizer.vocab_size` | int | 词表大小 |
| `tokenizer.make_vocab_size_divisible_by` | int | 词表对齐（TP 友好） |

## 6. 展平后的键名冲突检测

由于 system/model/data 三个 dict 按顺序 update，同名键会被覆盖。已知冲突：

| 键名 | 出现位置 | 实际生效 | 风险 |
|------|----------|----------|------|
| `seed` | system 和 model | model（后 update） | 低，一般只放 model |
| `num_workers` | system | — | 无冲突 |
| `no_shared_fs` | system | 被 ignore_keys 过滤 | 无（不传给 Megatron） |

## 7. 验证规则总结

1. **必填字段**：`model.num_layers`, `model.hidden_size`, `model.num_attention_heads`, `model.seq_length`, `model.micro_batch_size`, `data.data_path`
2. **互斥字段**：`precision.bf16` 和 `precision.fp16` 不能同时为 true
3. **约束关系**：
   - `global_batch_size % (micro_batch_size × dp_size × gradient_accumulation_steps) == 0`
   - `num_attention_heads % tensor_model_parallel_size == 0`
   - `num_layers % pipeline_model_parallel_size == 0`（除非设了 `decoder_first_pipeline_num_layers`）
   - `num_query_groups % tensor_model_parallel_size == 0`（GQA 时）
4. **内部管理字段不要设**：`log_dir`, `details_dir`, `scripts_dir`, `pids_dir`, `straggler_dir`（Runner 自动填充）
