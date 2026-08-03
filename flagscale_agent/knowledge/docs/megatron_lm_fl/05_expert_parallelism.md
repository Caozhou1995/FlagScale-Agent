# 05 - Expert Parallelism (EP) & Mixture-of-Experts 源码深度分析

## 源码位置

| 文件 | 行数 | 核心功能 |
|------|------|----------|
| `megatron/core/transformer/moe/moe_layer.py` | 663 | `MoELayer` 主体: route→dispatch→expert_compute→combine 四阶段 pipeline |
| `megatron/core/transformer/moe/token_dispatcher.py` | 1547 | Token 调度器: `MoEAlltoAllTokenDispatcher`(L357), `MoEFlexTokenDispatcher`(L1347), `_DeepepManager`(L1109) |
| `megatron/core/transformer/moe/router.py` | 928 | `TopKRouter`(L144): gating→scoring→topk→aux_loss→bias, `_hash_routing`(L623) |
| `megatron/core/transformer/moe/experts.py` | ~600 | `GroupedMLP` (fused grouped GEMM) / `SequentialMLP` (逐 expert 循环) |
| `megatron/core/parallel_state.py` | ~2800 | EP group 创建, `expert_model_parallel_size` 维度 |

---

## 1. MoE 层架构与执行流程

### 1.1 MoELayer 四阶段 Pipeline (moe_layer.py L532-633)

```python
# moe_layer.py L532
def forward(self, hidden_states, ...):
    """
    四阶段执行:
    1. route():       Router 计算 → probs, routing_map
    2. dispatch():    Token 分发 → AlltoAll 通信
    3. expert_compute(): 本地 expert 计算
    4. combine():     结果收集 → AlltoAll 逆向 + unpermute
    """
    # Stage 1: 路由
    probs, routing_map = self.route(hidden_states, padding_mask, input_ids)
    
    # Stage 1.5: 预处理 (latent projection if enabled)
    hidden_states, probs, residual = self.preprocess(hidden_states, probs, routing_map)
    
    # Stage 2: 分发 (AlltoAll / DeepEP)
    hidden_states, probs = self.dispatch(hidden_states, probs)
    
    # Stage 2.5: Shared expert (如果不 overlap, 在此计算)
    shared_expert_output = self.shared_experts_compute(hidden_states_original)
    
    # Stage 3: Expert 计算
    output, mlp_bias = self.routed_experts_compute(hidden_states, probs)
    
    # Stage 4: 合并
    output = self.combine(output)
    
    # Stage 5: 后处理 (latent de-projection + shared expert 加和)
    output = self.postprocess(output, shared_expert_output)
    return output
```

### 1.2 完整数据流 ASCII 图

```
Input: [S, B, H]
  │
  ├─── TopKRouter ─────────────────────────────────────────────────────┐
  │    gating: [S×B, H] × [H, E] → logits [S×B, E]                   │
  │    score_fn: softmax/sigmoid → scores [S×B, E]                     │
  │    topk: scores → probs [S×B, K], routing_map [S×B, E] (bool)     │
  │    aux_loss: attach autograd hook                                   │
  ├────────────────────────────────────────────────────────────────────┘
  │
  ├─── [Optional] Latent Projection ──────────────────────────────────┐
  │    fc1_latent: [S×B, H] → [S×B, latent_size]  (降维减少通信量)     │
  ├────────────────────────────────────────────────────────────────────┘
  │
  ├─── dispatch_preprocess ────────────────────────────────────────────┐
  │    permute(): 按 expert_id 排序 tokens                             │
  │    计算 input_splits/output_splits (EP 各 rank 的 token 数)        │
  ├────────────────────────────────────────────────────────────────────┘
  │
  ├─── token_dispatch (AlltoAll) ──────────────────────────────────────┐
  │    all_to_all(ep_group, tokens, output_splits, input_splits)       │
  │    ← 跨 EP rank 交换 tokens                                       │
  ├────────────────────────────────────────────────────────────────────┘
  │
  ├─── dispatch_postprocess ───────────────────────────────────────────┐
  │    [If TP>1] all-gather tokens (TP group)                          │
  │    sort_chunks_by_local_experts: 按本地 expert 排序                 │
  │    [If shared_expert_overlap] shared_experts.fc1_forward()         │
  ├────────────────────────────────────────────────────────────────────┘
  │
  ├─── Expert Computation ─────────────────────────────────────────────┐
  │    GroupedMLP: fused grouped GEMM (gate+up → act → down)           │
  │    或 SequentialMLP: for e in experts: e(tokens_for_e)             │
  ├────────────────────────────────────────────────────────────────────┘
  │
  ├─── combine_preprocess ─────────────────────────────────────────────┐
  │    unsort by local experts                                         │
  │    [If TP>1] reduce-scatter (TP group)                             │
  ├────────────────────────────────────────────────────────────────────┘
  │
  ├─── token_combine (AlltoAll reverse) ───────────────────────────────┐
  │    all_to_all(ep_group, output, input_splits, output_splits)       │
  │    ← 结果返回原始 rank                                             │
  ├────────────────────────────────────────────────────────────────────┘
  │
  ├─── combine_postprocess ────────────────────────────────────────────┐
  │    [If shared_expert_overlap] shared_experts.fc2_forward()         │
  │    unpermute(): 恢复原始 token 顺序                                │
  │    output += shared_expert_output                                   │
  ├────────────────────────────────────────────────────────────────────┘
  │
  └─── [Optional] fc2_latent_proj: [S×B, latent] → [S×B, H]
  
Output: [S, B, H]
```

---

## 2. TopKRouter 路由机制 (router.py L144-763)

### 2.1 完整路由流程

```python
# router.py L664-763: TopKRouter.routing()
def routing(self, logits, padding_mask=None, input_ids=None):
    # Step 1: reshape
    logits = logits.view(-1, num_moe_experts)  # [num_tokens, E]
    
    # Step 2: Z-Loss (抑制 logits 过大, 稳定训练)
    logits = self.apply_z_loss(logits, padding_mask)
    
    # Step 3: 路由计算 (三种模式)
    if self.is_hash_layer:
        # DSv4-Pro hash routing: tid2eid 预训练查找表
        probs, routing_map = self._hash_routing(logits, input_ids)
    elif self.routing_type == "sinkhorn":
        # Sinkhorn: 迭代使 routing matrix 行列和均匀
        probs, routing_map = self.sinkhorn_load_balancing(logits)
    else:
        # 标准 TopK: softmax/sigmoid → top-k selection
        probs, routing_map = topk_routing_with_score_function(
            logits, self.topk,
            use_pre_softmax=config.moe_router_pre_softmax,
            num_groups=config.moe_router_num_groups,      # grouped routing
            group_topk=config.moe_router_group_topk,       # group-level topk
            scaling_factor=config.moe_router_topk_scaling_factor,
            score_function=self.score_function,  # "softmax" / "sigmoid"
            expert_bias=self.expert_bias,        # 动态偏置修正
        )
    
    # Step 4: Token Dropping (容量限制)
    if config.moe_expert_capacity_factor is not None:
        probs, routing_map = apply_router_token_dropping(
            probs, routing_map, capacity_factor=..., drop_policy=...
        )
    
    # Step 5: Aux Loss (训练时)
    if self.training and self.is_aux_loss_enabled():
        probs = self._apply_aux_loss(probs, scores, routing_map)
        probs = self._apply_seq_aux_loss(probs, scores, routing_map, seq_len, bsz)
        probs = self._apply_global_aux_loss(probs, scores, routing_map)
    
    # Step 6: Expert Bias 更新
    self._apply_expert_bias(routing_map, padding_mask)
    
    return probs, routing_map  # probs: [tokens, K], routing_map: [tokens, E] bool
```


### 2.2 负载均衡策略详解

| 策略 | 源码位置 | 机制 | 数学表达 |
|:-----|:---------|:-----|:---------|
| `aux_loss` | L321-360 | Switch Transformer 标准辅助损失 | `L_aux = α × E × Σ(f_i × P_i)`, f_i=expert负载, P_i=路由概率 |
| `seq_aux_loss` | L361-415 | 序列级别辅助损失 (per sequence) | 在每个 sequence 内独立计算 aux loss |
| `global_aux_loss` | L416-461 | 跨 microbatch 全局 EMA 统计 | 使用 `global_tokens_per_expert` 累积, 跨步平滑 |
| `z_loss` | L533-587 | 抑制 router logits 过大 | `L_z = β × mean(log(Σexp(logits))²)` |
| `sinkhorn` | L266-297 | Sinkhorn-Knopp 迭代均衡 | 交替归一化行列, 强制双随机矩阵 |
| `expert_bias` | L610-622 | 动态偏置修正 | 统计 `local_tokens_per_expert`, 对低负载 expert 加正偏置 |
| `hash_routing` | L623-663 | DSv4-Pro 预训练查找表 | `tid2eid[token_id]` 直接查表, 无需 learned gating |

### 2.3 Grouped TopK Routing (DeepSeek-V3 风格)

```python
# topk_routing_with_score_function 支持的参数:
# num_groups: 将 experts 分为 G 组
# group_topk: 先选 top-G_k 组, 再在组内选 top-k experts
# scaling_factor: 归一化 topk 权重

# 流程:
# 1. logits [tokens, E] → reshape [tokens, G, E/G]
# 2. group_scores = max(scores_per_group) → top-G_k 组选择
# 3. 在选中的组内: top-k expert selection
# 4. scaling: probs *= scaling_factor / sum(probs)
```

### 2.4 Hash Routing (DSv4-Pro, router.py L623-663)

```python
def _hash_routing(self, logits, input_ids):
    """DSv4-Pro: 前 N 层使用预训练的 token_id → expert_id 映射表"""
    # tid2eid: [vocab_size, topk] — 预训练时学习的固定映射
    input_ids_flat = input_ids.view(-1)
    expert_ids = self.tid2eid[input_ids_flat]  # [num_tokens, topk]
    
    # 构造 routing_map (one-hot)
    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool)
    routing_map.scatter_(1, expert_ids, True)
    
    # probs 从 softmax(logits) 中提取对应位置
    scores = F.softmax(logits, dim=-1)
    probs = scores.gather(1, expert_ids)  # [num_tokens, topk]
    return probs, routing_map
```

**设计意义**: Hash routing 消除了 gating network 的动态路由开销, 适用于训练初期 (前 N 层) 的确定性分配。

---

## 3. Token Dispatcher 深度分析

### 3.1 AlltoAll Dispatcher 七步工作流 (token_dispatcher.py L357-865)

```
完整工作流 (docstring L362-368):
(1) preprocess:      计算 tokens_per_expert, input/output splits
(2) dispatch_preprocess: permute tokens 按 expert_id 排序
(3) token_dispatch:  A2A(EP group) 跨 rank 交换
(4) dispatch_postprocess: AG(TP group) + sort_by_local_experts
(5) combine_preprocess:   unsort + RS(TP group)
(6) token_combine:   A2A(EP group) 逆向收回
(7) combine_postprocess:  unpermute + reshape + shared_expert 加和
```

### 3.2 Permutation 详解

```python
# dispatch_preprocess (L605-660):
# Permutation 1: 将 tokens 按 routing_map 中 expert 分配排序
permutated_local_input_tokens, permuted_probs, reversed_mapping, _, _ = permute(
    hidden_states,        # [num_tokens, H]
    self.routing_map,     # [num_tokens, E] bool
    probs=probs,          # [num_tokens, K]
    num_out_tokens=self.num_out_tokens,
    fused=config.moe_permute_fusion,  # FlagScale: fused permute kernel
    drop_and_pad=self.drop_and_pad,
)
# 结果: tokens 按 [expert_0_tokens, expert_1_tokens, ..., expert_E_tokens] 排列
# reversed_mapping: 用于 combine 阶段恢复原始顺序

# dispatch_postprocess (L691-760):
# Permutation 2: 将 all-to-all 后的 tokens 按 LOCAL expert 排序
# 因为 A2A 后的布局是 [from_rank_0, from_rank_1, ...], 每段内含多个 expert
# 需要 transpose 为 [expert_0_all_ranks, expert_1_all_ranks, ...]
if self.num_local_experts > 1:
    global_input_tokens = sort_chunks_by_idxs(
        global_input_tokens,
        self.num_global_tokens_per_local_expert.ravel(),
        self.sort_input_by_local_experts,  # 预计算的 sort 索引
    )
```

### 3.3 AlltoAll 通信细节

```python
# token_dispatch (L662-689):
def token_dispatch(self, permutated_local_input_tokens, permuted_probs):
    # input_splits: [ep_size] — 本 rank 发给各 EP rank 的 token 数
    # output_splits: [ep_size] — 本 rank 从各 EP rank 接收的 token 数
    global_input_tokens = all_to_all(
        self.ep_group,
        permutated_local_input_tokens,  # [sum(input_splits), H]
        self.output_splits,             # 接收 split sizes
        self.input_splits               # 发送 split sizes
    )
    # 结果: global_input_tokens 包含所有应由本 rank 处理的 tokens
    return global_input_tokens, global_probs
```

### 3.4 TP×EP 联合通信模式

```
场景: TP=2, EP=4, num_experts=8, num_local_experts=2

dispatch 阶段:
  Step 1: Permute (按 expert_id 排序)
  Step 2: AlltoAll (EP group): tokens → 目标 EP rank
  Step 3: AllGather (TP group): 恢复 hidden 维完整性
           [tokens, H/TP] → [tokens, H]  (TP ranks 各持有部分 hidden)
  Step 4: Sort by local experts

combine 阶段 (逆序):
  Step 1: Unsort
  Step 2: ReduceScatter (TP group): [tokens, H] → [tokens, H/TP]
  Step 3: AlltoAll reverse (EP group): 结果返回原 rank
  Step 4: Unpermute
```

### 3.5 CUDA DtoH 同步优化 (L438-601)

```python
# 关键优化: AlltoAll 需要 CPU 端的 split sizes, 但计算在 GPU 上
# 直接 DtoH 同步会阻塞 → 使用专用 stream 异步拷贝

cuda_dtoh_stream = cur_platform.Stream()  # FlagScale: 平台抽象

# 5 个同步点 (按优先级):
# "before_permutation_1" → 最早同步 (最安全)
# "before_ep_alltoall"   → AlltoAll 前同步 (CUDAGraph 需要)
# "before_permutation_2" → 第二次 permute 前
# "before_finish"        → 最终输出前
# "no_sync"              → 不需要同步 (drop_and_pad 模式)

# 异步 DtoH 拷贝:
with cur_platform.stream(self.cuda_dtoh_stream):
    tokens_per_expert = maybe_move_tensor_to_cpu(tokens_per_expert)
    self.input_splits = maybe_move_tensor_to_cpu(self.input_splits, as_numpy=True)
    self.output_splits = maybe_move_tensor_to_cpu(self.output_splits, as_numpy=True)
# 在 sync point 处: cuda_dtoh_stream.synchronize()
```

### 3.6 DeepEP 集成 (_DeepepManager, L1109-1300)

```python
class _DeepepManager(_DispatchManager):
    """DeepEP: 高性能 EP 通信后端 (RDMA + pipeline)"""
    
    def __init__(self, group, num_local_experts, ...):
        # 两种 buffer, 按 batch size 自动选择:
        self.buffer_low_latency = deepep.Buffer(
            group, num_nvl_bytes, num_rdma_bytes  # RDMA 直传
        )
        self.buffer_high_throughput = deepep.Buffer(
            group, num_nvl_bytes, num_rdma_bytes  # 流水线通信
        )
    
    def dispatch(self, hidden_states, topk_idx, topk_weight, ...):
        # 根据 num_tokens 选择模式:
        if num_tokens < threshold:
            # Low latency: RDMA one-shot, 适合推理/小 batch
            return self.buffer_low_latency.dispatch(...)
        else:
            # High throughput: 多 chunk 流水线, 适合训练/大 batch
            return self.buffer_high_throughput.dispatch(...)
    
    def combine(self, hidden_states, ...):
        # 逆向操作, 同样自动选择 buffer
        return self.buffer.combine(...)
```

### 3.7 Flex Dispatcher (L1347-1547)

```python
class MoEFlexTokenDispatcher(MoETokenDispatcher):
    """
    统一 TP×EP 为单一通信域:
    - 将 routing_map 从 [tokens, experts] 展开为 [tokens, world_size, local_experts]
    - 通信策略与 TP/EP 划分解耦
    - 支持 DeepEP / HybridEP 后端
    """
    def __init__(self, ...):
        assert self.tp_size * self.ep_size > 1
        # 初始化后端 dispatch manager
        if config.moe_flex_dispatcher_backend == "deepep":
            self.dispatch_manager = _DeepepManager(...)
        elif config.moe_flex_dispatcher_backend == "hybridep":
            self.dispatch_manager = _HybridEPManager(...)
```

---

## 4. Shared Expert 与通信-计算 Overlap

### 4.1 Overlap 机制 (moe_layer.py L320-321, token_dispatcher.py L638, L704, L844-864)

```
┌─────────────────────────────────────────────────────────────────┐
│ 时序图: shared_expert_overlap=True                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Main Stream:   dispatch_preprocess → AlltoAll → dispatch_post → │
│                 Expert Compute → combine_pre → AlltoAll_rev →     │
│                                                                   │
│  Comm Stream:   ─────────────── shared_expert.fc1() ──────────── │
│                 ─────────────── shared_expert.fc2() ──────────── │
│                                                                   │
│  Sync Point:    combine_postprocess 中 get_output() 同步         │
├─────────────────────────────────────────────────────────────────┤
│ 关键调用链:                                                       │
│   dispatch_preprocess L638:                                       │
│     self.shared_experts.pre_forward_comm(hidden_states)           │
│     → 启动 shared expert fc1 计算 (在 comm stream)               │
│                                                                   │
│   dispatch_postprocess L704:                                      │
│     self.shared_experts.linear_fc1_forward_and_act(...)           │
│     → 完成 fc1 + activation (与 A2A 通信 overlap)                │
│                                                                   │
│   combine_postprocess L844-864:                                   │
│     self.shared_experts.linear_fc2_forward(...)                   │
│     self.shared_experts.post_forward_comm()                       │
│     shared_expert_output = self.shared_experts.get_output()       │
│     output += shared_expert_output  ← 最终合并                   │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 backward_dw 分离 (moe_layer.py L634-654)

```python
def backward_dw(self, routed_experts=True, shared_experts=False):
    """权重梯度计算与 activation 梯度解耦, 支持 fine-grained overlap"""
    if routed_experts:
        self.experts.backward_dw()
        if self.config.moe_latent_size:
            # fc2_latent_proj 在 comm stream 执行 (与 EP 通信 overlap)
            comm_stream = get_comm_stream()
            with torch.cuda.stream(comm_stream):
                self.fc2_latent_proj.backward_dw()
    if shared_experts:
        if self.use_shared_expert and not self.shared_expert_overlap:
            self.shared_experts.backward_dw()
```

---

## 5. Latent Projection — 通信量压缩 (moe_layer.py L246-276)

```python
# 设计动机: AlltoAll 通信量 ∝ hidden_size
# Latent projection: H → latent_size (降维), 通信后再 latent_size → H (升维)
# 通信量减少: hidden_size / latent_size 倍

if self.config.moe_latent_size:
    # 降维 (dispatch 前)
    self.fc1_latent_proj = TELinear(hidden_size, moe_latent_size, ...)
    # 升维 (combine 后)  
    self.fc2_latent_proj = TELinear(moe_latent_size, hidden_size, ...)

# 数据流:
# [S×B, H] → fc1_latent → [S×B, latent] → AlltoAll → Expert → AlltoAll → fc2_latent → [S×B, H]
# 通信量: 2 × S×B × latent_size × dtype (vs 原始 2 × S×B × H × dtype)
```

---

## 6. 通信量量化分析

### 6.1 AlltoAll 通信量

```
配置: E=64, EP=8, TP=2, S=4096, B=1, H=4096, K=2, BF16

单次 dispatch AlltoAll:
  每 rank 发出: num_tokens × K / EP × H × 2B = 4096 × 2 / 8 × 4096 × 2 = 8MB
  全 EP group 总通信: 8MB × EP = 64MB

完整 forward (dispatch + combine): 64MB × 2 = 128MB
完整 forward + backward: 128MB × 2 = 256MB

TP AllGather (dispatch_postprocess):
  tokens_received × H × 2B = 1024 × 4096 × 2 = 8MB per rank
  
总通信 (per MoE layer): ~280MB
NVLink (intra-node EP): 280MB / 900GB/s ≈ 0.3ms
IB (inter-node EP): 280MB / 50GB/s ≈ 5.6ms  ← 跨节点 EP 代价大!
```

### 6.2 Latent Projection 的通信收益

```
Without latent (H=4096): AlltoAll 通信 = 256MB/layer
With latent (latent_size=1024): AlltoAll 通信 = 64MB/layer  → 节省 75%!

代价: 2 × matmul(S×B, H, latent_size) ≈ 2 × 4096 × 4096 × 1024 × 2 FLOPS
     = 68 GFLOPS ≈ 0.2ms on H100 → 远小于通信节省
```

### 6.3 Shared Expert Overlap 收益

```
Without overlap:
  Timeline: [AlltoAll dispatch] → [Expert compute] → [AlltoAll combine] → [Shared Expert]
  Total: T_comm + T_expert + T_comm + T_shared

With overlap:
  Timeline: [AlltoAll dispatch + Shared fc1] → [Expert compute] → [AlltoAll combine + Shared fc2]
  Total: max(T_comm, T_shared_fc1) + T_expert + max(T_comm, T_shared_fc2)
  
  典型节省: T_shared ≈ T_comm → 总时间减少 ~30%
```

---

## 7. Expert 计算实现

### 7.1 GroupedMLP vs SequentialMLP

| 维度 | GroupedMLP | SequentialMLP |
|:-----|:-----------|:--------------|
| **实现** | Fused grouped GEMM (CUTLASS) | `for e in experts: e(tokens_e)` |
| **Kernel 次数** | 1 次 grouped_gemm | num_local_experts × 2 次 matmul |
| **GPU 利用率** | 高 (一次大 GEMM) | 低 (多次小 GEMM, launch overhead) |
| **内存** | 需要 padding to max_tokens | 按需分配 |
| **适用** | 训练 (tokens 多, experts 多) | debug, 小模型 |

### 7.2 GroupedMLP 数据流

```
Input: [total_tokens_for_all_local_experts, H]
  tokens_per_expert: [num_local_experts] — 每个 expert 的 token 数

Gate+Up: grouped_gemm(input, [W_gate, W_up] × num_experts, tokens_per_expert)
  → [total_tokens, intermediate_size × 2]  (SwiGLU)

Activation: SiLU(gate) × up → [total_tokens, intermediate_size]

Down: grouped_gemm(activated, [W_down] × num_experts, tokens_per_expert)
  → [total_tokens, H]
```

---

## 8. FlagScale 扩展点

| 扩展位置 | 修改内容 | 目的 |
|:---------|:---------|:-----|
| router.py L215,224,238,244 | `cur_platform.current_device()` 替代 `torch.cuda.current_device()` | 多硬件支持 |
| token_dispatcher.py L414-416 | `cur_platform.device_name()` 用于 permute_idx 设备 | Ascend/ROCm 兼容 |
| token_dispatcher.py L458 | `cur_platform.Stream()` 替代 `torch.cuda.Stream()` | DtoH stream 平台抽象 |
| token_dispatcher.py L887-891 | `cur_platform.current_stream()` / `cur_platform.stream()` | Stream 管理平台化 |
| moe_layer.py (通过 submodules) | Builder pattern 注入 | 允许替换 expert/router 实现 |

---

## 9. 配置参数速查

| 参数 | 默认值 | 说明 | 影响范围 |
|:-----|:-------|:-----|:---------|
| `--num-experts` | — | 总 expert 数量 | router, dispatcher |
| `--expert-model-parallel-size` | 1 | EP 并行度 | parallel_state |
| `--moe-router-topk` | 2 | 每 token 选取 expert 数 | router |
| `--moe-router-load-balancing-type` | aux_loss | 负载均衡策略 | router |
| `--moe-router-score-function` | softmax | 评分函数: softmax/sigmoid | router |
| `--moe-router-pre-softmax` | False | 先 softmax 再 topk (vs 先 topk 再 softmax) | router |
| `--moe-router-num-groups` | 0 | Grouped routing 组数 (DeepSeek-V3) | router |
| `--moe-router-group-topk` | 0 | 组级 topk 数 | router |
| `--moe-router-topk-scaling-factor` | None | TopK 权重缩放因子 | router |
| `--moe-router-enable-expert-bias` | False | 动态偏置修正 | router |
| `--moe-n-hash-layers` | 0 | 使用 hash routing 的前 N 层 | router |
| `--moe-aux-loss-coeff` | 0.01 | 辅助损失系数 | router |
| `--moe-z-loss-coeff` | None | Z-Loss 系数 | router |
| `--moe-expert-capacity-factor` | None | Token dropping 容量因子 | router, dispatcher |
| `--moe-token-dispatcher-type` | alltoall | 调度器: alltoall/allgather/flex | dispatcher |
| `--moe-flex-dispatcher-backend` | deepep | Flex 后端: deepep/hybridep | dispatcher |
| `--moe-grouped-gemm` | False | 启用 fused grouped GEMM | experts |
| `--moe-shared-expert-intermediate-size` | — | 共享专家 FFN 中间层大小 | moe_layer |
| `--moe-shared-expert-gate` | False | 共享专家是否有 gate | moe_layer |
| `--overlap-moe-expert-parallel-comm` | False | EP 通信与 shared expert overlap | dispatcher |
| `--moe-latent-size` | None | Latent projection 降维大小 | moe_layer |
| `--moe-permute-fusion` | False | 融合 permute kernel | dispatcher |
| `--moe-pad-expert-input-to-capacity` | False | Pad to capacity (CUDAGraph 友好) | dispatcher |
| `--moe-router-fusion` | False | 融合 router 计算 | router |

---

## 10. 设计决策与权衡

| 决策 | 选择 | 替代方案 | 理由 |
|:-----|:-----|:---------|:-----|
| 默认 dispatcher | AlltoAll | AllGather | A2A 通信量 ∝ 1/EP, AG 通信量 ∝ (EP-1)/EP |
| Expert 计算 | GroupedMLP (fused) | SequentialMLP | 减少 kernel launch, 提高 GPU 利用率 |
| Shared expert overlap | 与 A2A 并行 | 串行计算 | 隐藏 shared expert 计算于通信中 |
| Token permute | 独立 step | 融合入 A2A | 清晰的阶段划分, 便于 debug; 可选 fused mode |
| DtoH 同步 | 异步 stream + 延迟 sync | 同步 DtoH | 避免 GPU stall, 重叠 DtoH 与计算 |
| Latent projection | 可选降维 | 全 hidden 通信 | 跨节点 EP 时通信节省显著 |
| Router score | softmax (默认) | sigmoid | softmax 保证 sum=1; sigmoid 适合 topk>2 |
| Hash routing | 前 N 层 | 全部 learned | 前几层路由稳定, 减少计算; 后续层 learned 更灵活 |
| Expert bias | 在线统计 + 动态调整 | 静态均衡 | 适应训练动态, 无需 capacity dropping |
| Flex dispatcher | 统一 TP×EP | 分离 TP/EP 通信 | 简化跨域通信, 支持更灵活的后端切换 |

---

## 11. EP vs 其他并行策略的选择

```
决策树:

模型是否为 MoE?
├─ No (Dense) → 不需要 EP, 使用 TP+PP+DP
└─ Yes (MoE) →
    num_experts ≥ EP_size?
    ├─ No → 使用小 EP 或 TP (expert 内切分)
    └─ Yes →
        训练 or 推理?
        ├─ 训练 (大 batch) →
        │   节点内 EP: AlltoAll (NVLink)
        │   跨节点 EP: DeepEP high_throughput / Latent projection
        │   overlap: shared_expert + A2A
        └─ 推理 (小 batch) →
            DeepEP low_latency (RDMA one-shot)
            或 AllGather (如果 EP 很小)
```

---

## 12. 扩展阅读

- Switch Transformer: "Switch Transformers: Scaling to Trillion Parameter Models" (Fedus et al., 2022) — aux_loss 起源
- GShard: "GShard: Scaling Giant Models with Conditional Computation" (Lepikhin et al., 2020) — AlltoAll EP
- DeepSeek-V3: "DeepSeek-V3 Technical Report" (2024) — grouped routing, hash routing, expert bias
- DeepEP: "DeepEP: an efficient expert-parallel communication library" (2024) — RDMA/pipeline 优化
- Megablocks: "MegaBlocks: Efficient Sparse Training with Mixture-of-Experts" (Gale et al., 2022) — grouped GEMM
