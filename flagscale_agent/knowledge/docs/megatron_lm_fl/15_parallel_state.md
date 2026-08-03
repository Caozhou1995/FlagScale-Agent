# 第15章：parallel_state 进程组管理 深度源码分析

## 1. 概述与设计动机

### 1.1 解决什么问题

分布式训练中，不同并行维度（TP/PP/DP/CP/EP）需要各自独立的通信组。
`parallel_state.py` 是 Megatron-LM-FL 中管理所有 NCCL 进程组的**中央注册表**，
为上层模块提供统一的组查询接口。

### 1.2 核心设计思想

- **正交分解**：将 N 个 GPU 视为多维张量 `[TP, CP, EP, DP, PP]`，每个维度对应一类并行
- **RankGenerator 抽象**：通过掩码运算自动生成任意维度组合的 rank 列表
- **全局单例模式**：所有组存储在模块级全局变量中，通过 getter 函数访问

### 1.3 为什么不用 PyTorch 内置的组管理？

PyTorch `dist.new_group()` 只创建组，不提供：
- 多维正交分组的自动生成逻辑
- 组之间的逻辑关系（如 DP+CP 组 = DP 组 × CP 组）
- PP stage 判断（first/last stage）
- Expert 并行的独立命名空间

## 2. 源码定位

| 组件 | 文件路径 | 行数 | 职责 |
|------|----------|------|------|
| parallel_state | `megatron/core/parallel_state.py` | 2838 | 全部进程组管理 |
| initialize 调用处 | `megatron/training/initialize.py` | - | 启动时调用 |
| FlagScale 异构扩展 | `megatron/plugin/hetero/parallel_context.py` | - | 异构并行适配 |

## 3. 架构总览

### 3.1 全局变量注册表 (L36-175)

```
模块级全局变量（进程组注册表）:
┌─────────────────────────────────────────────────┐
│ 基础并行组                                       │
│  _TENSOR_MODEL_PARALLEL_GROUP     (TP)          │
│  _PIPELINE_MODEL_PARALLEL_GROUP   (PP)          │
│  _DATA_PARALLEL_GROUP             (DP)          │
│  _CONTEXT_PARALLEL_GROUP          (CP)          │
│                                                 │
│ 组合并行组                                       │
│  _MODEL_PARALLEL_GROUP            (TP×PP)       │
│  _TENSOR_AND_DATA_PARALLEL_GROUP  (TP×DP)       │
│  _TENSOR_AND_CONTEXT_PARALLEL_GROUP (TP×CP)     │
│  _DATA_PARALLEL_GROUP_WITH_CP     (DP×CP)       │
│                                                 │
│ Expert 专用组                                    │
│  _EXPERT_MODEL_PARALLEL_GROUP     (EP)          │
│  _EXPERT_TENSOR_PARALLEL_GROUP    (ETP)         │
│  _EXPERT_DATA_PARALLEL_GROUP      (EDP)         │
│                                                 │
│ 辅助组                                          │
│  _EMBEDDING_GROUP  (PP first+last stage)        │
│  _POSITION_EMBEDDING_GROUP                      │
│  _INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP    │
└─────────────────────────────────────────────────┘
```

### 3.2 核心数据流

```
initialize_model_parallel(tp=T, pp=P, dp=D, cp=C, ep=E, order="tp-cp-ep-dp-pp")
    │
    ├─ 创建 decoder_rank_generator = RankGenerator(tp=T, ep=1, dp=D*C, pp=P, cp=C)
    │       └─ world_size = T * D * P * C
    │
    ├─ 创建 expert_rank_generator  = RankGenerator(tp=ETP, ep=E, dp=EDP, pp=P, cp=1)
    │
    ├─ 对每种并行类型调用 rank_generator.get_ranks(token):
    │   ├─ get_ranks('tp')    → TP 组列表
    │   ├─ get_ranks('dp')    → DP 组列表
    │   ├─ get_ranks('pp')    → PP 组列表
    │   ├─ get_ranks('cp')    → CP 组列表
    │   ├─ get_ranks('dp-cp') → DP×CP 组合组列表
    │   ├─ get_ranks('tp-pp') → Model Parallel 组列表
    │   └─ ...
    │
    └─ 对每个 rank 列表调用 create_group(ranks, ...) → NCCL 进程组
        └─ 当前 rank 在列表中时，赋值给对应全局变量
```

## 4. RankGenerator 核心算法

### 4.1 设计动机 (WHY)

**为什么不硬编码各种组？**

硬编码方式需要为每种并行组合写一套 rank 计算逻辑。
5 个维度的组合有 2^5 - 1 = 31 种可能的组，维护成本极高。

RankGenerator 用**正交掩码算法**统一生成所有组：
- 输入：各维度 size + 排列顺序 + 目标组 token
- 输出：该组的所有 rank 列表

### 4.2 实现分析 (parallel_state.py:L476-551)

```python
class RankGenerator:
    def __init__(self, tp, ep, dp, pp, cp, order, rank_offset=0):
        # 约束检查：EP 和 CP 互斥（不能同时 > 1）
        assert ep == 1 or cp == 1  # (L483-485)
        
        self.world_size = tp * dp * pp * cp * ep
        
        # 解析 order 字符串为有序 size 列表
        # order="tp-cp-ep-dp-pp" → ordered_size=[tp, cp, ep, dp, pp]
        self.ordered_size = [name_to_size[token] for token in order.split("-")]
    
    def get_mask(self, order, token):
        """将 token 字符串转为布尔掩码 (L520-533)"""
        # token="dp-cp" + order="tp-cp-ep-dp-pp"
        # → mask=[False, True, False, True, False]
        
    def get_ranks(self, token):
        """获取指定并行类型的 rank 组列表 (L535-551)"""
        mask = self.get_mask(self.order, token)
        ranks = generate_masked_orthogonal_rank_groups(
            self.world_size, self.ordered_size, mask)
        # 加上 rank_offset（用于异构场景）
        return ranks
```

### 4.3 generate_masked_orthogonal_rank_groups 算法 (L278-370)

**核心数学**：

给定 `parallel_size = [s0, s1, ..., sN]` 和 `mask = [m0, m1, ..., mN]`：
- mask=True 的维度：这些维度的 rank 组成一个组
- mask=False 的维度：这些维度的不同组合产生不同的组

```
公式：global_rank = Σ(rank_i * Π(size_j for j < i))

示例：parallel_size=[2,3,4], mask=[False,True,False]
  组索引 = tp_rank + pp_rank * tp_size (mask=False 的维度组合)
  组内容 = 遍历 dp_rank ∈ [0, dp_size) (mask=True 的维度)
  
  组[0] = {0, 2, 4}   (tp=0, pp=0, dp=0,1,2)
  组[1] = {1, 3, 5}   (tp=1, pp=0, dp=0,1,2)
  ...
  组[7] = {19, 21, 23} (tp=1, pp=3, dp=0,1,2)
```

### 4.4 时序图：8 GPU, TP=2, PP=2, DP=2, order="tp-dp-pp"

```
GPU:     0    1    2    3    4    5    6    7
         ├────┤    ├────┤    ├────┤    ├────┤
TP组:    [0,1]     [2,3]     [4,5]     [6,7]

         ├─────────┤         ├─────────┤
DP组:    [0,2]  [1,3]       [4,6]  [5,7]

         ├───────────────────┤
PP组:    [0,4]  [1,5]  [2,6]  [3,7]

MP组:    [0,1,4,5]    [2,3,6,7]   (TP×PP)
```

**rank 编码公式** (order="tp-dp-pp"):
```
global_rank = tp_rank + dp_rank * 2 + pp_rank * 4
  GPU0: tp=0, dp=0, pp=0
  GPU1: tp=1, dp=0, pp=0
  GPU2: tp=0, dp=1, pp=0
  GPU3: tp=1, dp=1, pp=0
  GPU4: tp=0, dp=0, pp=1
  GPU5: tp=1, dp=0, pp=1
  GPU6: tp=0, dp=1, pp=1
  GPU7: tp=1, dp=1, pp=1
```

## 5. initialize_model_parallel 详解

### 5.1 函数签名 (L577-602)

```python
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    expert_tensor_parallel_size: Optional[int] = None,
    num_distributed_optimizer_instances: int = 1,
    order: str = "tp-cp-ep-dp-pp",
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
    ...
)
```

### 5.2 组创建顺序 (L870-1200)

```
创建顺序（有依赖关系）:
1. DP+CP 组合组        get_ranks('dp-cp')     (L870-975)
2. Hybrid CP 组        create_hybrid_dp_cp     (L1001-1011)
3. DP 组               get_ranks('dp')         (L1014-1030)
4. CP 组               get_ranks('cp')         (L1032-1059)
5. Model Parallel 组   get_ranks('tp-pp')      (L1061-1074)
6. TP 组               get_ranks('tp')         (L1076-1091)
7. PP 组 + Embedding   get_ranks('pp')         (L1158-1200)
8. TP+CP 组合组        get_ranks('tp-cp')      (后续)
9. TP+DP 组合组        get_ranks('tp-dp')      (后续)
10. Expert 相关组      expert_rank_generator    (后续)
```

### 5.3 设计决策：为什么 DP+CP 先于 DP？

**原因**：Context Parallel 中权重梯度需要在 DP×CP 组内 all-reduce。
创建组合组时需要知道完整的 rank 列表。先创建 DP+CP 组，再从中提取纯 DP 子组。

### 5.4 Expert 组的独立 RankGenerator (L836-848)

```python
expert_rank_generator = RankGenerator(
    tp=expert_tensor_parallel_size,   # ETP（默认=TP）
    ep=expert_model_parallel_size,    # EP
    dp=expert_data_parallel_size,     # EDP = world / (ETP * EP * PP)
    pp=pipeline_model_parallel_size,
    cp=1,                             # Expert 不参与 CP
    order=order
)
```

**WHY 单独的 generator？**
Expert 层的并行策略可以独立于 Dense 层：
- Dense 层：TP=8, EP=1
- Expert 层：ETP=2, EP=4 (4路专家并行，每个专家内2路张量并行)

两者共享 PP 维度，但 TP/DP 维度独立。

### 5.5 FlagScale 扩展

#### 5.5.1 DualPipeV (L83-85, L601)

```python
_DUALPIPEV_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
create_dualpipev_parallel_size: bool = False  # 参数
```

DualPipeV 需要额外的 PP world size 信息来支持双向流水线。

#### 5.5.2 Engram Embedding Parallel (L849-868)

```python
engram_rank_generator = RankGenerator(
    tp=engram_embedding_parallel_size,
    ep=1, dp=engram_dp_size, pp=pipeline_model_parallel_size, cp=1
)
```

用于 Engram 模型的独立 embedding 并行维度。

#### 5.5.3 Platform 抽象 (L22-23)

```python
cur_platform = get_platform()  # 替代硬编码 torch.cuda
# 使得代码可适配非 NVIDIA 硬件
```

## 6. 组查询 API 分析

### 6.1 基础 Getter 函数 (L1700-2000)

| 函数 | 返回 | 源码位置 |
|------|------|----------|
| `get_tensor_model_parallel_group()` | TP NCCL 组 | L1795 |
| `get_pipeline_model_parallel_group()` | PP NCCL 组 | L1808 |
| `get_data_parallel_group(with_cp)` | DP/DP+CP 组 | L1761 |
| `get_context_parallel_group()` | CP NCCL 组 | L1771 |
| `get_expert_model_parallel_group()` | EP NCCL 组 | L2255 |

### 6.2 Rank/World Size 查询 (L1813-2000)

```python
def get_tensor_model_parallel_world_size():  # L1813
    """TP 组内 GPU 数量"""
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())

def get_pipeline_model_parallel_rank(group=None):  # L1879 (FlagScale Add)
    """当前 rank 在 PP 组内的位置"""
    # FlagScale: 支持传入自定义 group（异构并行场景）
```

### 6.3 Pipeline Stage 判断 (L1893-1940)

```python
def is_pipeline_first_stage(ignore_virtual=True, vp_stage=None, 
                            group=None, ignore_dualpipev=True):  # L1893
    """判断当前 rank 是否为第一个 PP stage"""
    # Virtual Pipeline: vp_stage=0 才是 first
    # DualPipeV: 需要额外判断 dualpipev_stage
    if not ignore_virtual:
        if not ignore_dualpipev and _DUALPIPEV_PIPELINE_MODEL_PARALLEL_WORLD_SIZE:
            return get_virtual_pipeline_model_parallel_rank() == dualpipev_stage
        return get_virtual_pipeline_model_parallel_rank() == (vp_stage or 0)
    return get_pipeline_model_parallel_rank(group=group) == 0
```

### 6.4 PP 邻居查询 (L2114-2141)

```python
def get_pipeline_model_parallel_next_rank(group=None):  # L2114
    """获取 PP 组中下一个 rank（用于 P2P send）"""
    ranks = _PIPELINE_GLOBAL_RANKS
    rank_in_pipeline = get_pipeline_model_parallel_rank(group=group)
    return ranks[(rank_in_pipeline + 1) % world_size]

def get_pipeline_model_parallel_prev_rank(group=None):  # L2128
    """获取 PP 组中上一个 rank（用于 P2P recv）"""
```

## 7. 性能与通信量化分析

### 7.1 进程组数量

给定 world_size=W, TP=T, PP=P, DP=D, CP=C, EP=E:

| 组类型 | 组数量 | 每组 size |
|--------|--------|-----------|
| TP | W/T | T |
| PP | W/P | P |
| DP | W/D | D |
| CP | W/C | C |
| DP+CP | W/(D*C) | D*C |
| Model(TP×PP) | W/(T*P) | T*P |
| EP | W/(E*ETP) | E |
| EDP | W/EDP_size | EDP_size |

### 7.2 NCCL 通信器开销

每个进程组对应一个 NCCL communicator：
- 初始化开销：~50-200ms per group (ring/tree topology 建立)
- 内存开销：~10-50MB per communicator (内部 buffer)
- 典型场景 (TP=8, PP=4, DP=4, CP=2, EP=4)：
  - 基础组：5 个 communicator
  - 组合组：~5 个
  - Expert 组：~4 个
  - 总计：~14 个 communicator / rank
  - 内存：~14 * 30MB ≈ 420MB / GPU

### 7.3 order 参数对通信拓扑的影响

```
order="tp-dp-pp" (默认):
  TP 组内 rank 相邻 → NVLink 通信（带宽最高）
  DP 组跨 TP 组 → 可能跨节点

order="tp-pp-dp":
  PP 组内 rank 比 DP 更"近" → 适用于 PP 通信密集场景
  
推荐：
  - TP 放最内层（NVLink）
  - CP 紧邻 TP（同节点 NVSwitch）
  - PP 和 DP 放外层（可跨节点）
```

## 8. 设计决策对比表

| 维度 | 全局变量单例 | 对象封装 (OOP) | Megatron 选择理由 |
|------|-------------|---------------|------------------|
| 访问方式 | `get_tp_group()` | `state.tp_group` | 全局函数简单直接 |
| 初始化 | 一次 `initialize_model_parallel` | 构造函数 | 避免传递对象引用 |
| 线程安全 | 不安全（单进程假设） | 可加锁 | 训练场景单线程够用 |
| 可测试性 | 差（全局状态） | 好 | 用 `destroy_model_parallel()` 重置 |
| 扩展性 | 加全局变量 + getter | 加属性 | 2838 行的代价 |

| 维度 | RankGenerator 掩码算法 | 手动枚举组 | 选择理由 |
|------|----------------------|-----------|---------|
| 代码量 | ~80 行核心 | ~500 行 per 配置 | 大幅减少代码 |
| 正确性 | 数学保证正交性 | 容易遗漏边界 | 不易出错 |
| 灵活性 | 任意维度组合 | 每种需手写 | order 参数即可切换 |
| 约束 | EP∩CP 互斥 | 无显式约束 | assert 强制检查 |

## 9. 边界条件与约束

### 9.1 维度约束 (L483-485, L806-825)

```python
# EP 和 CP 互斥（一个 RankGenerator 内）
assert ep == 1 or cp == 1

# world_size 必须整除
assert world_size % (tp * pp * dp * cp) == 0

# Expert TP 默认等于 Dense TP
expert_tensor_parallel_size = expert_tensor_parallel_size or tensor_model_parallel_size

# Expert DP 自动计算
expert_data_parallel_size = world_size // (ETP * EP * PP)
```

### 9.2 与其他特性的互斥

| 特性A | 特性B | 关系 | 原因 |
|-------|-------|------|------|
| CP > 1 | EP > 1 | 同一 RankGenerator 内互斥 | 组合爆炸，分离处理 |
| Virtual PP | DualPipeV | 共存但需额外判断 | stage 编号冲突 |
| Hierarchical CP | 普通 CP | 替代关系 | 两级 CP 组替代单级 |
| UCC backend | CUDA_DEVICE_MAX_CONNECTIONS=1 | 互斥 | UCC 需多连接并发 |

### 9.3 destroy 与重新初始化 (L2661+)

```python
def destroy_model_parallel():
    """重置所有全局变量为 None"""
    # 用于测试场景重新初始化
    # 注意：不会销毁 NCCL communicator（由 PyTorch 管理 GC）
```

## 10. 配置建议与调优指南

### 10.1 order 参数选择

| 硬件拓扑 | 推荐 order | 理由 |
|----------|-----------|------|
| 8×H100 NVSwitch 单节点 | tp-cp-dp-pp | TP+CP 走 NVLink |
| 多节点 IB | tp-cp-ep-dp-pp | TP 最内层 NVLink |
| PP 通信密集 | tp-pp-dp | PP 尽量同节点 |

### 10.2 NCCL Communicator 调优

```yaml
# nccl_communicator_config.yaml
tp:
  min_ctas: 1
  max_ctas: 32
  cga_cluster_size: 2
dp:
  min_ctas: 1
  max_ctas: 16
pp:
  min_ctas: 1
  max_ctas: 2    # PP P2P 小消息，少 CTA 足够
```

### 10.3 high_priority_stream_groups

```python
# 推荐将通信密集组设为高优先级
initialize_model_parallel(
    ...,
    high_priority_stream_groups=['tp', 'cp']  # TP/CP 通信不被计算 kernel 阻塞
)
```

### 10.4 常见陷阱

1. **忘记 `destroy_model_parallel()`**：pytest 多 case 共用进程时，第二个 case 会 assert 失败
2. **order 与硬件不匹配**：TP 跨节点导致 all-reduce 走 IB 而非 NVLink
3. **Expert TP ≠ Dense TP 时**：需确保 ETP 整除 TP 组内 GPU 数
4. **CP > 1 时忘记开 SP**：非 attention 层需要 Sequence Parallel 处理序列维度
