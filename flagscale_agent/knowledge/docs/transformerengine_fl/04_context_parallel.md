# 第四章：Context Parallelism (CP) 通信系统深度源码分析

## 1. 概述与源文件定位

| 功能模块 | 源文件路径 | 行数 | 核心类/函数 |
|---------|-----------|------|------------|
| CP主入口 | `dot_product_attention/context_parallel.py` | 4365 | `attn_forward_func_with_cp` (L3934) |
| Autograd封装 | 同上 | - | `AttnFuncWithCP` (L1262) |
| P2P Ring通信 | 同上 | - | `flash_attn_p2p_communicate` (L62) |
| P2P前向 | 同上 | - | `cp_p2p_fwd_flash_attn` (L913) |
| A2A通信 | 同上 | - | `cp_a2a_fwd_flash_attn` |
| DPA主模块 | `dot_product_attention/__init__.py` | ~1628 | `DotProductAttention` |

### 1.1 设计动机

长上下文训练（128K-1M tokens）的核心挑战是Attention的O(S²)内存和计算复杂度。当S=128K时，单GPU无法存放完整的QKV张量和attention矩阵。CP将序列维度切分到多个GPU，每个GPU只处理S/cp_size长度的局部序列，通过通信获取远端KV完成完整Attention计算。

### 1.2 CP vs SP 本质区别

| 维度 | Sequence Parallelism (SP) | Context Parallelism (CP) |
|------|--------------------------|--------------------------|
| 切分位置 | LayerNorm/Dropout区域 | Attention内部QKV |
| 通信原语 | AllGather + ReduceScatter | Ring P2P / AlltoAll |
| 计算模式 | 非Attention区域并行 | Attention本身并行 |
| 内存节省 | 激活内存/TP倍 | Attention内存/CP倍 |
| 通信量 | O(S·H) | O(S·H·(CP-1)/CP) |
| 依赖 | TP > 1 | 独立于TP |

## 2. DualChunkSwap 序列重排策略

### 2.1 核心思想

朴素的序列均分（前半给GPU0，后半给GPU1）导致causal mask下负载严重不均：GPU0几乎不需要远端KV，而最后一个GPU需要所有远端KV。

DualChunkSwap策略：将每个序列分为 `cp_size * 2` 个chunk，交错分配给各GPU，使得每个GPU都同时持有序列的"前段"和"后段"tokens，从而平衡causal mask下的计算量。

### 2.2 重排示例（源码L3981-4020）

以 S=12, causal mask, cp_size=2 为例：

**重排前（朴素分配）：**
```
GPU0: tokens [0,1,2,3,4,5]   → 只需本地KV，计算轻
GPU1: tokens [6,7,8,9,10,11] → 需要GPU0全部KV，计算重
```

**DualChunkSwap重排后：**
```
GPU0: tokens [0,1,2,9,10,11]  → 前3个+后3个
GPU1: tokens [3,4,5,6,7,8]    → 中间6个
```

此时GPU0持有序列首尾，GPU1持有中间段。causal mask下：
- GPU0的token 9,10,11 需attend到GPU1的token 3-8 → 有通信需求
- GPU1的token 6,7,8 需attend到GPU0的token 0,1,2 → 有通信需求
- 双方通信量接近平衡

### 2.3 THD格式（变长序列Pack）

当 `qkv_format="thd"` 时，多个不等长序列pack到同一个batch中。DualChunkSwap对每个序列独立执行chunk划分，确保不同序列的tokens不混淆：

```python
# 源码注释(L3998-4020)示意：
# batch_size=2, seq_lens=[8,4], cp_size=2
# 序列0: 8tokens → 4chunks(各2tokens) → GPU0得chunk0+chunk3, GPU1得chunk1+chunk2  
# 序列1: 4tokens → 4chunks(各1token) → GPU0得chunk0+chunk3, GPU1得chunk1+chunk2
```

## 3. 通信模式详解

### 3.1 P2P Ring模式 (`cp_comm_type="p2p"`)

#### 3.1.1 核心通信函数 `flash_attn_p2p_communicate` (L62-105)

```python
def flash_attn_p2p_communicate(rank, send_tensor, recv_tensor, cp_group, *, send_dst, recv_src):
    """异步P2P发送/接收KV或dKV张量"""
    send_op = torch.distributed.P2POp(
        torch.distributed.isend, send_tensor, send_dst, group=cp_group
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv, recv_tensor, recv_src, group=cp_group
    )
    # batch_isend_irecv合并发送和接收，最大化带宽利用
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    return reqs  # 异步handle, 后续wait()
```

**设计要点：**
- 使用 `batch_isend_irecv` 将send+recv打包为单次NCCL操作
- 环形拓扑：每个rank向右发送本地KV，从左接收远端KV
- 异步执行：通信与下一步计算可overlap

#### 3.1.2 前向Ring Attention `cp_p2p_fwd_flash_attn` (L913-1020)

```
执行流程（cp_size=4, rank=0为例）:
┌─────────────────────────────────────────────────────────┐
│ Step 0: local_attn(Q_local, KV_local) → out_0, lse_0   │
│         同时: send KV_local→rank1, recv KV_from_rank3   │
├─────────────────────────────────────────────────────────┤
│ Step 1: local_attn(Q_local, KV_from_rank3) → out_1     │
│         Online Softmax合并: out = merge(out_0, out_1)   │
│         同时: send KV_from_rank3→rank1, recv KV_from_2  │
├─────────────────────────────────────────────────────────┤
│ Step 2: local_attn(Q_local, KV_from_rank2) → out_2     │
│         Online Softmax合并: out = merge(out, out_2)     │
│         (最后一步, 无需继续通信)                          │
└─────────────────────────────────────────────────────────┘
```

#### 3.1.3 Online Softmax合并算法

Ring Attention的关键是无需全局attention矩阵即可正确计算softmax。利用log-sum-exp (LSE)增量更新：

```python
# 伪代码 - 合并第i步的局部attention结果
def online_softmax_merge(out_prev, lse_prev, out_new, lse_new):
    """
    out_prev: 之前所有步累积的输出 [B,H,S/cp,D]
    lse_prev: 之前的log-sum-exp [B,H,S/cp,1]  
    out_new:  当前步局部attention输出
    lse_new:  当前步局部log-sum-exp
    """
    lse_max = torch.maximum(lse_prev, lse_new)
    exp_prev = torch.exp(lse_prev - lse_max)
    exp_new = torch.exp(lse_new - lse_max)
    
    # 加权合并
    out = (out_prev * exp_prev + out_new * exp_new) / (exp_prev + exp_new)
    lse = lse_max + torch.log(exp_prev + exp_new)
    return out, lse
```

**数值稳定性**: 通过减去max(lse)避免exp溢出，这是FlashAttention的核心技巧扩展到分布式场景。

### 3.2 AlltoAll模式 (`cp_comm_type="a2a"`)

#### 3.2.1 设计思想

P2P Ring模式需要cp_size-1步串行通信，延迟随cp_size线性增长。AlltoAll模式将KV一次性全交换，用带宽换延迟：

```
P2P Ring: 延迟 = (cp_size-1) × (通信+计算)，但每步通信量小
AlltoAll:  延迟 = 1次AlltoAll + 1次本地计算，但AlltoAll通信量大
```

#### 3.2.2 A2A执行流程

```
Step 1: AlltoAll交换 - 每个rank将KV切片发送给所有其他rank
        输入: KV_local [B, S/cp, H, D]
        输出: KV_all [B, S, H/cp, D]  (head维度切分)
        
Step 2: 本地FlashAttention - 完整序列，部分head
        Q_local [B, S/cp, H, D] × KV_all [B, S, H/cp, D]
        
Step 3: AlltoAll还原 - 将输出从head切分还原为seq切分
        输入: Out [B, S, H/cp, D]
        输出: Out_local [B, S/cp, H, D]
```

#### 3.2.3 A2A vs P2P 适用场景

| 条件 | 推荐模式 | 原因 |
|------|---------|------|
| cp_size ≤ 8 | P2P | 步数少，通信可与计算overlap |
| cp_size > 8 | A2A | 避免多步串行延迟 |
| NVLink互联 | P2P | 带宽充足，延迟低 |
| 跨节点IB互联 | A2A | 减少通信轮次 |
| H/cp_size < 4 | P2P | A2A按head切分，head太少效率低 |

### 3.3 分层模式 (`cp_comm_type="a2a+p2p"`)

#### 3.3.1 架构设计（L4030-4049）

分层CP结合两种模式的优势：节点内用A2A（高带宽NVLink），节点间用P2P Ring（适配IB）：

```python
# 源码逻辑(L4030-4045):
if cp_comm_type == "a2a+p2p":
    cp_group = [a2a_cp_group, p2p_cp_group]  # 两级group
    # 退化处理：
    if world_size(a2a_cp_group) == 1:  # 节点内只有1个rank
        cp_group = p2p_cp_group; cp_comm_type = "p2p"
    elif world_size(p2p_cp_group) == 1:  # 只有1个节点
        cp_group = a2a_cp_group; cp_comm_type = "a2a"
```

#### 3.3.2 分层通信时序

```
假设: 2节点×4GPU, a2a_group=[0,1,2,3](节点内), p2p_group=[0,4](节点间)

Phase 1 (节点内A2A):
  GPU0-3: AlltoAll交换KV的head维度切片 → 每GPU得到完整seq的1/4 heads
  
Phase 2 (节点间P2P Ring):
  GPU0↔GPU4: P2P环形传递远端节点的KV chunk
  每步: 本地FlashAttn + online softmax merge
  
Phase 3 (节点内A2A还原):
  GPU0-3: AlltoAll将输出从head切分还原为seq切分
```

**限制（源码L4035-4036）：**
- 不支持 `thd` 格式（变长pack）
- 不支持 attention bias

## 4. AttnFuncWithCP Autograd封装 (L1262+)

### 4.1 Forward流程

```python
class AttnFuncWithCP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, is_training, q, k, v, ..., cp_comm_type, ...):
        # 1. 根据cp_comm_type分发到对应实现
        if cp_comm_type == "p2p":
            out, softmax_lse = cp_p2p_fwd_flash_attn(...)
        elif cp_comm_type == "a2a":
            out, softmax_lse = cp_a2a_fwd_flash_attn(...)
        elif cp_comm_type == "a2a+p2p":
            out, softmax_lse = cp_a2a_p2p_fwd_flash_attn(...)
        
        # 2. 保存反向传播所需张量
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.cp_comm_type = cp_comm_type
        ctx.cp_group = cp_group
        
        return out
```

### 4.2 Backward流程（P2P模式）

反向传播同样采用Ring结构，但传递的是dKV而非KV：

```
Step 0: local_attn_bwd(dOut, Q, KV_local, Out, LSE) → dQ_0, dKV_0
        同时: send KV_local→right, recv KV_from_left
        
Step i: local_attn_bwd(dOut, Q, KV_from_step_i, Out, LSE) → dQ_i, dKV_i
        dQ_accum += dQ_i
        同时: send dKV_i→left (返还给KV的源rank)
        
最终: 每个rank得到完整的dQ(本地累加), dK, dV(从远端接收累加)
```

### 4.3 FP8 Attention支持 (L1380+)

CP系统支持FP8 attention（源码L1050-1080确认）：

```python
# FP8 quantization在CP通信前执行
if fp8:
    # KV在发送前量化为FP8，减少通信量50%
    kv_fp8 = quantizers["kv"](kv)  # BF16 → E4M3
    # 接收后保持FP8直接输入FlashAttention FP8 kernel
    # 避免重复量化/反量化开销
```

## 5. 通信量分析

### 5.1 P2P Ring模式

```
每步通信量: 2 × B × (S/cp_size) × H × D × sizeof(dtype)  [发送K和V]
总通信步数: cp_size - 1
总通信量:   2 × B × S × H × D × (cp_size-1)/cp_size × sizeof(dtype)

示例: B=1, S=128K, H=128, D=128, cp=8, BF16
单步: 2 × 1 × 16K × 128 × 128 × 2B = 1GB
总量: 1GB × 7 = 7GB
```

### 5.2 AlltoAll模式

```
AlltoAll通信量: B × S × H × D × sizeof(dtype) × (cp_size-1)/cp_size
执行2次(前向KV交换 + 输出还原)
总通信量: 2 × B × S × H × D × (cp_size-1)/cp_size × sizeof(dtype)

与P2P总量相同，但仅需1轮通信（延迟更低）
```

### 5.3 FP8通信优化

启用FP8 attention后，KV通信量减半：
```
FP8通信量 = BF16通信量 / 2
示例: 7GB → 3.5GB (P2P Ring, cp=8)
```

## 6. 与FlashAttention内核的集成

### 6.1 Backend选择逻辑

CP系统不直接实现attention计算，而是调用底层backend：

```python
# 简化逻辑 (context_parallel.py内部)
def local_flash_attn(q, k, v, causal, ...):
    if use_fused_attention:
        # cuDNN Fused Attention (支持更多mask类型)
        return fused_attn_fwd(q, k, v, ...)
    elif use_flash_attn_3:
        # Flash Attention 3 (Hopper优化)
        return flash_attn_func_v3(q, k, v, ...)
    else:
        # Flash Attention 2 (默认)
        return flash_attn_func(q, k, v, ...)
```

### 6.2 Causal Mask处理

DualChunkSwap重排后，每个GPU持有不连续的token位置。CP系统需要特殊处理causal mask：

```
GPU0持有tokens [0,1,2,9,10,11], GPU1持有tokens [3,4,5,6,7,8]

当GPU0的Q(pos=9)与GPU1传来的KV(pos=3-8)计算attention时：
- pos9应attend到pos3-8的所有token（causal: 9>3,4,5,6,7,8 均为True）
- 因此这一步不需要mask

当GPU1的Q(pos=3)与GPU0传来的KV(pos=9,10,11)计算attention时：
- pos3不应attend到pos9-11（causal: 3<9,10,11 均为False）
- 这一步需要完整mask掉
```

CP系统在每个ring step中根据当前Q和远端KV的相对位置关系决定是否需要causal mask。

## 7. 与Megatron-LM集成

### 7.1 配置接口

```yaml
# Megatron训练配置
model:
  context_parallel_size: 8        # CP并行度
  cp_comm_type: "p2p"             # 或 "a2a", "a2a+p2p"  
  # hierarchical CP需要额外配置:
  # cp_comm_type: "a2a+p2p"
  # 自动根据网络拓扑划分a2a_group(节点内)和p2p_group(节点间)
```

### 7.2 数据加载器Token重排

序列重排在数据加载阶段执行一次（而非每层执行），参考Megatron的 `get_batch_on_this_cp_rank`：

```python
# megatron/core/utils.py#L1725
def get_batch_on_this_cp_rank(batch):
    """根据当前CP rank重排batch中的tokens"""
    cp_size = get_context_parallel_world_size()
    cp_rank = get_context_parallel_rank()
    # DualChunkSwap: 取chunk[cp_rank]和chunk[2*cp_size-1-cp_rank]
    seq_len = batch['tokens'].shape[1]
    chunk_size = seq_len // (cp_size * 2)
    # 第一个chunk: 序列前段
    chunk1 = batch['tokens'][:, cp_rank*chunk_size:(cp_rank+1)*chunk_size]
    # 第二个chunk: 序列后段(镜像位置)
    mirror_rank = 2*cp_size - 1 - cp_rank
    chunk2 = batch['tokens'][:, mirror_rank*chunk_size:(mirror_rank+1)*chunk_size]
    batch['tokens'] = torch.cat([chunk1, chunk2], dim=1)
    return batch
```

## 8. 设计决策总结

| 设计选择 | 方案 | 理由 |
|---------|------|------|
| 序列重排策略 | DualChunkSwap | causal mask下负载均衡 |
| 通信-计算overlap | Ring步间异步P2P | 隐藏通信延迟 |
| Softmax正确性 | Online合并(LSE增量) | 无需全局attention矩阵 |
| 多模式支持 | p2p/a2a/a2a+p2p | 适配不同拓扑和规模 |
| FP8集成 | 通信前量化 | 通信量减半 |
| THD格式支持 | 每序列独立chunk | 变长序列pack兼容 |
| 退化处理 | 自动降级到单模式 | a2a+p2p中某级size=1时简化 |
| 反向传播 | 同构Ring传递dKV | 与前向对称，减少实现复杂度 |

## 9. 性能调优建议

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. cp_size选择: 优先使S/cp_size能整除flash_attn的block_size(128) │
│ 2. 序列长度: 必须能被 cp_size*2 整除(DualChunkSwap要求)           │
│ 3. 节点内优先A2A: NVLink带宽900GB/s远高于IB 400Gb/s              │
│ 4. FP8 attention: H100+上开启可同时加速计算和减少通信              │
│ 5. 与SP组合: SP处理非Attention区域，CP处理Attention内部          │
│ 6. 避免H/cp_size过小: A2A模式按head切分，每GPU至少4个head         │
└─────────────────────────────────────────────────────────────────┘
```
