# FlashAttention 源码深度分析 — 第4章：反向传播与梯度计算

## 1. 设计动机

### 1.1 Attention 反向传播的核心挑战

**WHY FlashAttention 反向更复杂？** 标准反向需要 O(N²) 的 P 矩阵（softmax output），
而 FlashAttention 前向时已经丢弃了 P。反向必须"重计算" P，同时保持 tiling 结构。

```
标准 Attention 反向 (存储 P):
  dV = P^T × dO              ← 需要完整 P (N×N)
  dP = dO × V^T
  dS = P ⊙ (dP - rowsum(dP ⊙ P))  ← softmax 反向
  dQ = dS × K
  dK = dS^T × Q

FlashAttention 反向 (重计算 P):
  对每个 (Q-block, K-block) pair:
    1. 重新计算 S_ij = Q_i × K_j^T
    2. 从 LSE 恢复 P_ij = exp(S_ij - LSE_i)
    3. 逐 block 累积 dQ, dK, dV
```

### 1.2 反向 IO 复杂度

```
标准反向: O(N² + Nd) HBM IO (需读/写 P, dP)
Flash 反向: O(N²d²/M) HBM IO (与前向相同量级)
  - 代价: 多一次 QK^T 矩阵乘 (重计算 S)
  - 收益: 省去 O(N²) 的 P 存储与读取
```

## 2. 反向参数结构 (flash.h L172-214)

```cpp
// hopper/flash.h L172-214
struct Flash_bwd_params : public Flash_fwd_params {
    // 梯度输入
    void *__restrict__ do_ptr;          // dO: (B, N, H, d) 输出梯度
    
    // 梯度输出
    void *__restrict__ dq_ptr;          // dQ: (B, N, H, d)
    void *__restrict__ dk_ptr;          // dK: (B, N_k, H_k, d)
    void *__restrict__ dv_ptr;          // dV: (B, N_k, H_k, d)
    
    // Float32 累加缓冲 (避免 fp16 原子加精度损失)
    void *__restrict__ dq_accum_ptr;    // dQ 累加: (B, N, H, d) fp32
    void *__restrict__ dk_accum_ptr;    // dK 累加 (split 模式)
    void *__restrict__ dv_accum_ptr;    // dV 累加 (split 模式)
    
    // 同步信号量
    int *__restrict__ dq_semaphore;     // dQ 原子累加同步
    int *__restrict__ dk_semaphore;     // dK 同步
    int *__restrict__ dv_semaphore;     // dV 同步
    
    // softmax 辅助
    void *__restrict__ dsoftmax_sum;        // D = rowsum(dO ⊙ O)
    void *__restrict__ softmax_lse_log2_ptr; // LSE 的 log2 形式
    
    bool deterministic;                 // 确定性模式
    index_t dq_accum_split_stride;      // split 间 dQ 累加步长
};
```

**WHY `dq_accum_ptr` 用 float32？** 多个 K-block 需要向同一 dQ 位置做原子累加。
FP16 原子加有 ~1% 相对误差，在 N/Bc 次累加后误差放大到不可接受。Float32 保证精度。

## 3. 反向 Kernel 架构 (flash_bwd_kernel_sm90.h)

### 3.1 类结构 (L25-62)

```cpp
// hopper/flash_bwd_kernel_sm90.h L25-62
template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class FlashAttnBwdSm90 {
    // 三种 WGMMA 操作
    using TiledMmaSdP = ...;   // S/dP 计算: Q×K^T 和 dO×V^T
    using TiledMmadKV = ...;   // dK/dV 计算: dS^T×Q 和 P^T×dO
    
    // Swap AB 优化标志
    static constexpr bool dKV_swapAB = CollectiveMainloop::dKV_swapAB;
    
    // Thread 组织 (与前向相同: Producer + Consumer)
    static constexpr uint32_t NumLoadWarpGroups = 1;
    static constexpr uint32_t NumMmaWarpGroups = ...;
};
```

### 3.2 反向 Mainloop 模板参数 (mainloop_bwd L28-34)

```cpp
// hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp L28-34
template <int Stages, int Stages_dO, int Stages_dS,
          class ClusterShape_, class TileShape_MNK_,
          class Element_, class ElementAccum_, class ArchTag_,
          bool Is_causal_, bool Is_local_, bool Has_softcap_,
          bool Varlen_, bool Deterministic,
          bool SdP_swapAB_, bool dKV_swapAB_, bool dQ_swapAB_,
          int NumMmaWarpGroups=2, int AtomLayoutMSdP=1,
          int AtomLayoutNdKV=2, int AtomLayoutMdQ=1,
          bool Mma_dP_is_RS=false>
struct CollectiveMainloopBwdSm90 { ... };
```

**WHY 需要 3 种 Pipeline Stages？**
- `Stages`: Q/K 数据 pipeline 深度
- `Stages_dO`: dO 数据 pipeline 深度 (可以比 Q/K 浅)
- `Stages_dS`: dS 中间结果 pipeline 深度

dO 可以用更浅的 pipeline 因为它的 reuse pattern 不同于 Q/K。

## 4. 反向算法流程

### 4.1 预处理: D = rowsum(dO ⊙ O) (flash_bwd_preprocess_kernel.h)

```
WHY 需要 D？ Softmax 反向公式:
  dS_ij = P_ij * (dP_ij - D_i)
  其中 D_i = Σ_j P_ij * dP_ij = Σ_j (dO_ij * O_ij) (因为 O = P×V)

D 只依赖 dO 和 O (不依赖 K/V), 可以预计算:
  D: (batch, nheads, seqlen_q) — 与 LSE 形状相同
```

### 4.2 主循环结构 (外层遍历 K-block)

```
反向与前向的遍历方向不同:

前向: 固定 Q-block, 遍历所有 K-blocks (行主序)
反向: 固定 K-block, 遍历所有 Q-blocks (列主序)

原因: dK 和 dV 需要累积所有 Q-block 的贡献
  dK_j = Σ_i dS_ij^T × Q_i  (对所有 Q-block i 求和)
  dV_j = Σ_i P_ij^T × dO_i  (对所有 Q-block i 求和)

固定 K-block j, 遍历 Q-block i:
  → dK_j 和 dV_j 在寄存器中累加, 只需最后写一次 HBM
  → dQ 需要原子累加 (多个 K-block 写同一 dQ)
```

### 4.3 单步计算 (对每个 Q-block i × K-block j)

```
┌─────────────────────────────────────────────────────────┐
│ 1. 重计算 S_ij = Q_i × K_j^T                (WGMMA)    │
│ 2. Apply mask (causal/local)                            │
│ 3. P_ij = exp2(S_ij * scale_log2 - LSE_i * scale_log2) │
│ 4. dP_ij = dO_i × V_j^T                     (WGMMA)    │
│ 5. dS_ij = P_ij ⊙ (dP_ij - D_i)            (元素级)   │
│ 6. dV_j += P_ij^T × dO_i                    (WGMMA)    │
│ 7. dK_j += dS_ij^T × Q_i                    (WGMMA)    │
│ 8. dQ_i += dS_ij × K_j  → atomic add to dq_accum      │
└─────────────────────────────────────────────────────────┘
```

## 5. swapAB 优化策略 (mainloop_bwd L:SdP_swapAB, dKV_swapAB, dQ_swapAB)

### 5.1 为什么需要 swap？

```
WGMMA 指令约束:
  - SS mode: 两个操作数都在 SMEM, layout 有 swizzle 要求
  - RS mode: A 在 Register, B 在 SMEM → A 必须是"行"操作数

问题: 反向有 4 种 GEMM, 每种对操作数布局要求不同:
  (1) S = Q × K^T     → Q 是行, K^T 是列
  (2) dP = dO × V^T   → dO 是行, V^T 是列
  (3) dK = dS^T × Q   → dS^T 是行, Q 是列
  (4) dV = P^T × dO   → P^T 是行, dO 是列

swapAB 的含义: 将 A×B 改为 (B^T × A^T)^T
  → 交换两个操作数在 SMEM 中的角色
  → 允许复用相同 SMEM layout 做不同 GEMM
```

### 5.2 RS vs SS 模式选择

```cpp
// mainloop_bwd L:Mma_dKV_is_RS, Mma_dQ_is_RS
// 条件: 当 AtomLayout 配置满足特定条件时, 可用 RS 模式

static constexpr bool Mma_dKV_is_RS = 
    AtomLayoutMSdP == 1 && AtomLayoutNdKV == NumMmaWarpGroups 
    && SdP_swapAB && !dKV_swapAB;
    
// RS mode: P^T/dS^T 直接留在寄存器中做下一步 GEMM
// → 省去一次 SMEM 写+读 (~50 cycles latency)
```

## 6. dQ 原子累加机制

### 6.1 问题分析

```
dQ_i = Σ_j (dS_ij × K_j)  — 多个 K-block 产生的 partial dQ 需累加

方案对比:
┌──────────────────┬──────────────┬──────────────┬─────────────┐
│ 方案             │ 精度         │ 性能         │ 确定性      │
├──────────────────┼──────────────┼──────────────┼─────────────┤
│ FP16 全局原子加  │ 差 (1% err) │ 最快         │ 非确定      │
│ FP32 accum buffer│ 好           │ 需额外空间   │ 取决于顺序  │
│ Split + reduce   │ 精确         │ 需 reduce    │ 可确定      │
└──────────────────┴──────────────┴──────────────┴─────────────┘

FlashAttention 选择: FP32 accum buffer + semaphore
```

### 6.2 Semaphore 同步流程

```
dq_semaphore 使用方式:
  
K-block 0 处理完 → atomicAdd(dq_accum, partial_dQ_0); atomicAdd(semaphore, 1)
K-block 1 处理完 → atomicAdd(dq_accum, partial_dQ_1); atomicAdd(semaphore, 1)
...
K-block last 处理完 → atomicAdd semaphore 达到 total_k_blocks
  → 触发 final kernel: dQ = cast_fp16(dq_accum)

确定性模式 (deterministic=True):
  - 不用 atomicAdd, 而是按固定 K-block 顺序串行累加
  - 使用 dq_accum_split_stride 存储每个 split 的独立 partial
  - 最后按顺序 reduce → 结果完全可复现
```

## 7. Softmax 反向的 Fused 实现

### 7.1 数学公式

```
给定:
  P_ij = softmax(S_ij) = exp(S_ij - LSE_i)  (从 LSE 恢复)
  D_i = rowsum(dO_i ⊙ O_i)                  (预计算)

Softmax 梯度:
  dS_ij = P_ij * (dP_ij - D_i)
  
其中 dP_ij = dO_i × V_j^T (通过 WGMMA 计算)

Fused 实现: 不显式存储 P_ij 和 dP_ij
  1. 计算 S_ij (WGMMA)
  2. P_ij = exp2f(S_ij * scale_log2 - lse_scaled)  (in-register)
  3. 计算 dO×V^T = dP_ij (WGMMA, 结果在 register)
  4. dS_ij = P_ij * (dP_ij - D_i)  (element-wise, in-register)
  → P 和 dP 都不写回 HBM, 全程在寄存器/SMEM 中
```

### 7.2 Softcap 对反向的影响

```
前向: S' = softcap * tanh(S / softcap)
反向: dS_original = dS' * softcap * (1 - tanh²(S / softcap)) / softcap
                   = dS' * (1 - (S'/softcap)²)

实现: 重计算 S_ij 后先应用 softcap, 再计算 P_ij
  → 额外的 tanh 计算 (约 4-5 条指令) + 反向修正
```

## 8. 反向 Pipeline 时序

```
反向 Pipeline (3-stage: Q/dO → S/dP → dK/dV):

Time ───────────────────────────────────────────────────→

Producer:  [TMA Q0,dO0] [TMA Q1,dO1] [TMA Q2,dO2] ...
                │              │              │
Consumer:  wait ─[S=QK^T]─[dP=dOV^T]─[dS=P*(dP-D)]─[dKV+=...]─ ...
                                              │
                 ← K,V 保持在 SMEM (外层固定) →

注意: K 和 V 在外层循环加载后保持不动
     Q 和 dO 在内层循环每步更新
```

## 9. 确定性模式实现

### 9.1 非确定性来源

```
来源 1: 浮点加法顺序
  dQ_i = partial_0 + partial_1 + ... + partial_k
  不同 CTA 完成时间不同 → 加法顺序不固定 → 结果有 bit-level 差异

来源 2: Warp 调度
  同一 warp 内的线程执行顺序可能变化
  → reduce_sum 结果可能有 ULP 级差异

来源 3: Pipeline 异步性
  TMA 完成顺序可能变化 → 处理顺序变化
```

### 9.2 确定性保证机制

```
deterministic=True 时:
  1. 禁用 work-stealing (固定 tile → CTA 映射)
  2. dQ 累加使用 ordered reduce:
     - 每个 K-block split 写入独立 buffer
     - 最后按固定顺序 reduce (K-block 0 → 1 → 2 → ...)
  3. 代价: ~10% 性能下降 + 额外内存 (num_splits × dQ_size)
```

## 10. 内存占用分析

```
反向额外内存 (相比前向):
┌──────────────────┬─────────────────────┬──────────────┐
│ 缓冲区           │ 大小                │ 生命周期     │
├──────────────────┼─────────────────────┼──────────────┤
│ dq_accum         │ B×N×H×d×4 (fp32)   │ 整个反向     │
│ dsoftmax_sum (D) │ B×H×N×4 (fp32)     │ 预计算 → 主循环│
│ softmax_lse      │ B×H×N×4 (fp32)     │ 前向保存     │
│ dQ (output)      │ B×N×H×d×2 (fp16)   │ 输出         │
│ dK, dV (output)  │ 2×B×N_k×H_k×d×2    │ 输出         │
└──────────────────┴─────────────────────┴──────────────┘

总额外内存 ≈ B×N×H×d×6 bytes (dq_accum 主导)
对比标准 Attention: B×N²×2 bytes (存储 P)
当 N > 3Hd 时, Flash 反向内存更优 (例: N=8192, H=32, d=128 → 3Hd=12288)
```

## 11. Python 层反向集成 (flash_attn_interface.py)

```python
# flash_attn/flash_attn_interface.py L510-540
@staticmethod
def backward(ctx, dout, *args):
    q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
    # 预分配 dqkv (打包格式)
    dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
    # Pad dout (与前向对齐)
    if head_size_og % 8 != 0:
        dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])
    # 调用 C++ 反向 kernel
    _wrapped_flash_attn_backward(
        dout_padded, q, k, v, out, softmax_lse,
        dqkv[:,:,0], dqkv[:,:,1], dqkv[:,:,2],  # dQ, dK, dV 输出
        ctx.dropout_p, ctx.softmax_scale, ctx.causal,
        ctx.window_size[0], ctx.window_size[1],
        ctx.softcap, ctx.alibi_slopes, ctx.deterministic,
        rng_state=rng_state)
    return dqkv[..., :dout.shape[-1]], None, None, ...
```

## 12. 总结

| 技术要点 | 实现方式 | 源码位置 |
|---------|---------|---------|
| P 重计算 | exp2(S - LSE) | mainloop_bwd, softmax.h |
| D 预计算 | rowsum(dO⊙O) | flash_bwd_preprocess_kernel.h |
| dQ 原子累加 | fp32 buffer + semaphore | Flash_bwd_params L177-208 |
| K-block 外循环 | dK/dV 寄存器累加 | mainloop_bwd 主循环 |
| 确定性模式 | ordered reduce | deterministic flag |
| swapAB 优化 | 复用 SMEM layout | SdP_swapAB_, dKV_swapAB_ |
| 3种 Pipeline | Q/dO, S/dP, dS 分离 | Stages, Stages_dO, Stages_dS |
