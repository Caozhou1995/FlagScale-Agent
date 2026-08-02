# 模型训练实现分析标准

## 适用范围

对**已有开源实现**的模型训练代码进行调研分析时，按本标准组织文档。

分析对象涵盖三种来源体系：
- **FlagScale/Megatron-LM-FL 体系** — 主要目标框架，分析其中的模型实现
- **DeepSpeed 体系** — 分析已有 DS 实现，为迁移到 FlagScale 做准备
- **PyTorch 原生体系** (FSDP/torchtitan/HuggingFace Trainer) — 分析原生实现作为参考

目的：
- 理解模型在特定框架下如何跑起来
- 为跨体系迁移提供完整技术规格
- 为复现训练结果提供足够信息

## 一、模型架构层（"是什么" — 摘要性质，为实现分析提供上下文）

> 注：详细的模型架构分析（设计理由、对比前代、创新点深度探讨）属于论文调研范畴（见 `03_paper_research.md`）。本节只摘录实现分析所需的关键参数。

### 1.1 参数总表

必须包含一张完整参数表，满足"仅凭此表即可写出 config yaml"的标准：

| 类别 | 必填项 |
|------|--------|
| 规模 | 总参数量、激活参数量（MoE）、层数、隐藏维度 |
| 注意力 | head数、KV head数(GQA)、kv_channels、注意力变体类型 |
| FFN | 中间维度、激活函数、是否 gated (SwiGLU) |
| MoE | 专家数、TopK、router类型、共享专家、负载均衡策略 |
| 位置编码 | 类型(RoPE/mRoPE/ALiBi)、base、percent、特殊分段 |
| 归一化 | 类型(RMSNorm/LayerNorm)、epsilon、是否zero-centered |
| 词表 | vocab_size、特殊token ID、tokenizer类型 |
| 序列 | max_position_embeddings、训练seq_length |
| 其他 | MTP层数、Vision encoder参数（如多模态）、tie_embeddings |

### 1.2 架构创新点

- 与前代模型/baseline 的**具体差异**（不是泛泛描述）
- 设计动机（为什么这样做，解决什么问题）
- 论文/技术报告出处

### 1.3 计算图（Forward Path）

描述一个 token 从输入到输出的完整数据流：

```
输入 → Embedding → [Layer × N] → Final LayerNorm → Output Head → Loss
                      ↓
              每层内部：
              Input → PreNorm → Attention/GDN → Residual
                                    ↓
                    → PreNorm → MoE/FFN → Residual
```

关键信息：
- 每个模块的 input/output shape（用符号表示：`[s, b, h]`）
- 哪些模块有 residual connection
- 哪些模块在特定条件下跳过

### 1.4 类继承/组合关系

```
TopModel
├── SubModule A (来源、职责)
├── SubModule B
│   ├── Component B1
│   └── Component B2
└── SubModule C
```

标注：
- 每个类的来源文件
- 是继承还是组合
- 哪些是复用（从其他模型 import）

## 二、工程实现层（"怎么做的"）

### 2.1 并行适配分析

| 并行维度 | 分析内容 |
|----------|----------|
| TP | 切分点在哪（QKV/FC1 列切、Proj/FC2 行切）、非标准切分（GDN、MoE router） |
| PP | boundary 如何划分、vision encoder 是否参与 PP、first/last stage 特殊逻辑 |
| EP | expert dispatch 方式（AllToAll/AllGather）、共享专家是否参与 EP |
| CP | 对位置编码的影响、attention mask 如何切分、Ring/Ulysses 选择 |
| SP | 哪些模块启用 sequence parallel、LayerNorm 输入格式 |

**必须回答**："如果 TP 从 2 改到 4，需要改什么？有什么约束？"

### 2.2 精度与性能关键路径

- 哪些算子强制走 FP32（router softmax、attention softmax、loss）
- 哪些层使用 Grouped GEMM / Fused kernel
- Recompute 策略（full/selective、哪些层、granularity）
- 通信重叠配置（tp_comm_overlap、overlap_grad_reduce）

### 2.3 数据接口契约

模型的 `forward()` 签名及每个参数的含义：

```python
def forward(
    input_ids: [b, s],           # token IDs
    position_ids: [b, s],        # 位置ID（特殊格式如 mRoPE [3,b,s] 需单独说明）
    labels: [b, s],              # 训练target
    loss_mask: [b, s],           # 哪些位置计算loss
    attention_mask: ...,         # mask 格式（因模型而异）
    packed_seq_params: ...,      # packing 参数（可选）
    ...
) -> loss or logits
```

与 `get_batch` 的对应关系 — 数据管道产出什么，模型消费什么。

### 2.4 Checkpoint 权重映射

HuggingFace ↔ Megatron 的权重名称和变换规则：

| HF 权重名 | Megatron 权重名 | 变换 |
|-----------|----------------|------|
| `model.layers.{i}.self_attn.q_proj.weight` | `decoder.layers.{i}.self_attention.linear_qkv.weight` | QKV concat |
| `model.layers.{i}.mlp.gate_proj.weight` | `decoder.layers.{i}.mlp.linear_fc1.weight` | gate+up concat |
| ... | ... | ... |

必须说明：
- 是否需要转置
- 是否有 zero-centered gamma 调整
- MoE expert 如何分片存储
- TE extra_state 键有哪些

### 2.5 已知限制与约束

- 哪些并行组合当前版本不支持（需实测确认，列出验证过的组合）
- 配置参数间的隐含约束（如 num_heads % TP == 0）
- 推理 vs 训练的差异（如 inference_params 未实现）
- 已知 bug 或 TODO

## 三、标准化检查清单

完成调研后，逐项自检：

| # | 检查项 | 判定标准 |
|---|--------|----------|
| 1 | 能否仅凭文档重建 config yaml | 参数表完备，无需读代码 |
| 2 | 能否说清每层的 input/output shape | 计算图有 shape 标注 |
| 3 | 能否回答"改 TP=4 需要什么" | 并行适配分析到位 |
| 4 | 能否定位"loss 不降"查哪里 | 精度路径明确 |
| 5 | 能否写出 checkpoint 转换脚本 | 权重映射完整 |
| 6 | 能否判断某配置会不会 OOM | MoE/recompute 分析足够 |
| 7 | 是否标注了源码文件和行号 | 可追溯 |
| 8 | 创新点是否有动机解释 | 不只是"what"还有"why" |

## 四、文档模板

```markdown
# {模型名} 实现分析

## 1. 模型概览
(参数总表)

## 2. 架构创新点
(与baseline差异、设计动机)

## 3. 源码结构
(文件列表、类关系图)

## 4. 计算图
(Forward path, shape flow)

## 5. 并行适配
(TP/PP/EP/CP/SP 切分分析)

## 6. 精度与性能
(FP32路径、recompute、fused ops)

## 7. 数据接口
(forward签名、get_batch契约)

## 8. Checkpoint 映射
(HF↔Megatron 权重表)

## 9. 已知限制
(约束、TODO、坑)
```

## 五、与通用质量标准的关系

本标准是 `01_infrastructure_analysis.md` 在模型调研场景下的具体化：
- 01 定义"好的源码分析文档"的通用原则（设计动机优先、代码路径可追溯、性能分析等）
- 02（本文）定义模型实现调研这一特定场景下，具体需要分析哪些维度、产出什么内容
