# FlashAttention 源码深度分析 — 第2章：Online Softmax 与 Tiling 算法

## 1. 设计动机

### 1.1 核心矛盾

标准 Softmax 需要两趟遍历所有 N 个元素（pass 1: max, pass 2: sum_exp），而 FlashAttention
要求逐 block 流式处理 K/V tiles。

**WHY Online Softmax？** 在不知道全局 max 的情况下，必须能够"回溯修正"之前 block 的输出结果。
Online Softmax 通过维护 running max + running sum 实现单趟精确计算。

### 1.2 数学等价性保证

```
定理：Online Softmax 与标准 Softmax 数值等价（不是近似）
证明关键: 对于任意分块 B1, B2, ..., Bk:
  - m_k = max(m_{k-1}, max(B_k))
  - l_k = l_{k-1} * exp(m_{k-1} - m_k) + Σ_{i∈Bk} exp(B_k[i] - m_k)
  - O_k = O_{k-1} * (l_{k-1}/l_k * exp(m_{k-1} - m_k)) + Σ exp(B_k-m_k)/l_k × V_k
  
最终等价于: O = softmax(S) × V (标准定义)
```

## 2. Softmax 结构体实现 (hopper/softmax.h L92-168)

### 2.1 数据结构

```cpp
// hopper/softmax.h L92-99
template <int kNRows, int Max_offset=0>
struct Softmax {
    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max;   // 每行当前最大值 (running max)
    TensorT row_sum;   // 每行当前归一化因子 (running sum)
    float const softmax_scale_log2;  // log2(1/sqrt(d)) — 使用 exp2 替代 exp
    
    // 构造: 接收 scale 的 log2 形式
    CUTLASS_DEVICE Softmax(float const softmax_scale_log2_) : ...
};
```

**WHY `softmax_scale_log2`？** GPU 硬件有专用 `exp2f()` 指令（1 cycle），而 `expf()` 需要
多条指令模拟。使用 log2 缩放因子 + `exp2f` 比 `expf(x * scale)` 更快。

### 2.2 三步流程

```
对每个 K-block (Bc 列):
┌──────────────────────────────────────────────────────┐
│ Step 1: max_get_scale()  — 计算新 max, 返回 rescale │
│ Step 2: online_softmax() — exp2f 并累加 row_sum     │
│ Step 3: rescale_o()      — 修正之前累积的 O         │
└──────────────────────────────────────────────────────┘
最终: finalize() — quad_allreduce + 计算 LSE
```

## 3. max_get_scale 实现 (L101-124)

### 3.1 首块 vs 后续块

```cpp
// hopper/softmax.h L101-124
template<bool Is_first, bool Check_inf=false, typename Tensor0>
__forceinline__ __device__ TensorT max_get_scale(Tensor0 &acc_s) {
    Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
    TensorT scores_scale;
    
    if constexpr (Is_first) {
        // 首块: 直接求 max，scale=1（无需修正）
        reduce_max<true>(scores, row_max);
        cute::fill(scores_scale, 1.f);
    } else {
        // 后续块: 保存旧 max, 更新 max, 计算修正因子
        Tensor scores_max_prev = make_fragment_like(row_max);
        cute::copy(row_max, scores_max_prev);
        reduce_max<false>(scores, row_max);  // max(old_max, new_block_max)
        
        for (int mi = 0; mi < size(row_max); ++mi) {
            float scores_max_cur = row_max(mi);
            // 修正因子: exp2(old_max - new_max) — 用于 rescale O 和 row_sum
            scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
            row_sum(mi) *= scores_scale(mi);  // 修正 running sum
        }
    }
    return scores_scale;  // 返回给 rescale_o 使用
};
```

**数值稳定性关键**：`scores_max_prev - scores_max_cur` 总是 ≤ 0，所以 `exp2f(...)` ∈ [0, 1]，
不会溢出。

### 3.2 reduce_max 实现 (L50-54)

```cpp
// hopper/softmax.h L50-54
template<bool zero_init=true>
__device__ __forceinline__ void reduce_max(Tensor const& tensor, Tensor& max) {
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
    // reduce_ = thread_reduce_ + quad_allreduce_
    // quad_allreduce_: 4线程 shuffle 求全局 max (warp内)
}
```

```
Quad Allreduce 过程 (4-thread group):
Thread 0: max_local=5.2  ─┐
Thread 1: max_local=3.8  ─┼─→ shuffle → all get max=7.1
Thread 2: max_local=7.1  ─┤
Thread 3: max_local=2.4  ─┘
```

## 4. scale_apply_exp2 实现 (L64-88)

### 4.1 Fused Scale + Exp2

```cpp
// hopper/softmax.h L64-88
template <bool Scale_max=true, bool Check_inf=true, int Max_offset=0>
__forceinline__ __device__ void scale_apply_exp2(
    Tensor &tensor, Tensor const &max, const float scale) {
    
    static constexpr float max_offset = float(Max_offset);
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // 处理 -inf 情况（被 mask 的位置）
        const float max_scaled = Check_inf
            ? (max(mi) == -INFINITY ? 0.f : max(mi) * scale - max_offset)
            : max(mi) * scale - max_offset;
        for (int ni = 0; ni < size<1>(tensor); ++ni) {
            // exp2f(x * log2(e) * scale - max * log2(e) * scale)
            // = exp(x * scale - max * scale) = exp((x - max) * scale)
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
        }
    }
}
```

**WHY `Max_offset`？** FP8 推理时的范围扩展技巧（L67-68 注释）：
将 exp2 结果从 [0,1] 扩展到 [0,256]，充分利用 FP8 E4M3 的动态范围，减少 underflow。

### 4.2 编译器优化提示 (L82-85 注释)

```cpp
// "This allows the compiler to use the ffma instruction instead of
//  fadd and fmul separately"
tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
// 编译器可将 x * scale - max_scaled 融合为单条 FFMA 指令
```

## 5. online_softmax 主函数 (L127-135)

```cpp
// hopper/softmax.h L127-135
template<bool Is_first, bool Check_inf=false>
__forceinline__ __device__ void online_softmax(Tensor0 &acc_s) {
    Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
    // 1. 对 scores 原地应用 exp2(S - max)
    scale_apply_exp2<true, Check_inf, Max_offset>(scores, row_max, softmax_scale_log2);
    // 2. 累加到 row_sum（线程内先 reduce，不做 warp reduce）
    reduce_sum<Is_first, /*warp_reduce=*/false>(scores, row_sum);
};
```

**WHY `warp_reduce=false`？** 延迟 warp 通信到 finalize() 阶段。
中间每个 block 的 row_sum 不需要跨线程精确，只要最终 finalize 时做一次即可，
减少了 N/Bc 次不必要的 warp shuffle。

## 6. rescale_o 输出修正 (L157-166)

```cpp
// hopper/softmax.h L157-166
template<typename Tensor1>
__forceinline__ __device__ void rescale_o(Tensor1 &acc_o, TensorT const &scores_scale) {
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
            acc_o_rowcol(mi, ni) *= scores_scale(mi);  // O_old *= exp(m_old - m_new) / (l_new)
        }
    }
};
```

**执行时序**：
```
Block j 处理流程:
1. scores_scale = max_get_scale(acc_s)  // 更新 max, 计算 rescale
2. rescale_o(acc_o, scores_scale)       // 修正之前的 O 累积
3. online_softmax(acc_s)                // exp2 + 累加 row_sum
4. acc_o += P_block × V_block (WGMMA)  // 新 block 贡献加入 O
```

## 7. finalize 最终归一化 (L137-154)

```cpp
// hopper/softmax.h L137-154
__forceinline__ __device__ TensorT finalize(float const final_scale=1.f) {
    SumOp<float> sum_op;
    // 此时才做 warp 内 allreduce（收集4线程的 partial sum）
    quad_allreduce_(row_sum, row_sum, sum_op);
    
    TensorT scores_scale;
    for (int mi = 0; mi < size(row_sum); ++mi) {
        float sum = row_sum(mi);
        float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;
        scores_scale(mi) = inv_sum * final_scale;
        
        // 计算 LSE = log(sum) + max * scale
        // 这就是保存到 softmax_lse 的值，用于反向重计算
        if constexpr (Max_offset != 0) {
            sum *= 1.f / float(1 << Max_offset);  // 消除 FP8 offset
        }
        row_sum(mi) = (sum == 0.f || sum != sum)
            ? -INFINITY
            : row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum);
    }
    return scores_scale;  // 最终 rescale O *= 1/sum
};
```

**关键细节**：
- `sum != sum` 检测 NaN（IEEE 754: NaN != NaN）
- `row_sum(mi)` 被复用为 LSE 输出（节省寄存器）
- `softmax_scale_log2 * M_LN2 = log2(scale) * ln(2) = ln(scale)` ← 将 log2 域转回 ln 域

### 7.1 LSE 公式推导

```
LSE = log(Σ exp(S_i * scale))
    = log(Σ exp((S_i - max) * scale) * exp(max * scale))
    = max * scale + log(Σ exp((S_i - max) * scale))
    = max * scale + log(row_sum)

代码中: row_max(mi) * (softmax_scale_log2 * M_LN2) + __logf(sum)
       = row_max * log2(scale) * ln(2) + ln(sum)
       = row_max * ln(scale) + ln(sum)  ✓ (注意 scale = 1/sqrt(d))
```

## 8. Tiling 策略与主循环交互

### 8.1 前向 Tiling 模式

```
Q tiles (Br行) × K tiles (Bc列) 的 Outer Loop:

for each Q-tile i (rows [i*Br, (i+1)*Br]):
  Load Q[i*Br:(i+1)*Br, :] to SRAM
  Initialize: O=0, m=-inf, l=0
  
  for each K-tile j (cols [j*Bc, (j+1)*Bc]):
    Load K[j*Bc:(j+1)*Bc, :] to SRAM (via TMA, double-buffered)
    Load V[j*Bc:(j+1)*Bc, :] to SRAM
    
    ┌─ Step 1: S_ij = Q_i × K_j^T (WGMMA, Br×Bc)
    │  Step 2: Apply causal mask (if needed)
    │  Step 3: scores_scale = max_get_scale(S_ij)
    │  Step 4: rescale_o(O, scores_scale)
    │  Step 5: online_softmax(S_ij)  → P_ij in-place
    └─ Step 6: O += P_ij × V_j (WGMMA, Br×d)
  
  final_scale = finalize()
  O *= final_scale  → 完成归一化
  Store O to HBM, Store LSE to HBM
```

### 8.2 Causal Mask 对 Tiling 的影响

```
Causal Mask 分块示意 (seqlen=8, Br=Bc=2):

     K blocks: [0,1] [2,3] [4,5] [6,7]
Q[0,1]:        ████   skip  skip  skip     ← 只需 1 block
Q[2,3]:        ████   ████  skip  skip     ← 只需 2 blocks
Q[4,5]:        ████   ████  ████  skip     ← 只需 3 blocks
Q[6,7]:        ████   ████  ████  ████    ← 需要 4 blocks

优化: Q-tile i 最多遍历 K-tiles [0..i]，总计算量减半
```

### 8.3 Split-KV 策略 (长 KV 并行化)

```
当 seqlen_k 很长时，单个 SM 处理完整 K 循环太慢:

标准模式: SM0 处理 Q[0:Br] × K[0:seqlen_k] (串行遍历所有K blocks)
Split模式: 将 K 分成 num_splits 段，多个 SM 并行:
  SM0: Q[0:Br] × K[0:split_size]        → O_partial_0, LSE_0
  SM1: Q[0:Br] × K[split_size:2*split]  → O_partial_1, LSE_1
  SM2: Q[0:Br] × K[2*split:3*split]     → O_partial_2, LSE_2
  
Combine kernel (flash_fwd_combine.cu):
  O_final = Σ (O_partial_i * exp(LSE_i - LSE_max)) / Σ exp(LSE_i - LSE_max)
```

**WHY Split-KV?** 当 batch_size 小但 seqlen_k 很长时（如 128K context），
可用 SM 远多于 Q-tiles 数量，Split-KV 提高 SM 利用率。

## 9. 数值精度分析

### 9.1 exp2f 精度

| 操作 | 精度 (ULP) | 影响 |
|------|-----------|------|
| exp2f (CUDA) | 1 ULP | 比 expf (2 ULP) 更精确 |
| __logf | ~1 ULP (快速) | 仅用于 LSE 计算 |
| FFMA (fp32) | 0.5 ULP | 无 round-off 累加 |

### 9.2 累加顺序的确定性问题

```
非确定性来源:
  - 不同 warp 完成顺序 → quad_allreduce 结果可能有 bit 差异
  - Split-KV combine 中浮点加法顺序

确定性模式 (deterministic=True):
  - 强制固定 K-block 遍历顺序
  - dQ 累加使用 semaphore 保序
  - 代价: ~5-10% 性能下降
```

## 10. 与标准实现的 IO 复杂度对比

```
设: N=seqlen, d=headdim, M=SRAM_size, Br=Bc=O(√M)

标准 Attention IO:
  Step 1 (QK^T):  Read Q(Nd) + K(Nd), Write S(N²)     → Θ(Nd + N²)
  Step 2 (softmax): Read S(N²), Write P(N²)            → Θ(N²)
  Step 3 (PV):    Read P(N²) + V(Nd), Write O(Nd)     → Θ(N² + Nd)
  Total: Θ(N² + Nd) HBM accesses

FlashAttention IO:
  Outer loop: N/Br Q-blocks, Inner: N/Bc K-blocks
  Per inner iteration: Load K(Bc×d) + V(Bc×d) from HBM = O(Bcd)
  Total inner per Q-block: (N/Bc) × Bcd = Nd
  Total: (N/Br) × Nd = N²d/Br
  With Br = O(M/d): Total = O(N²d²/M)

当 M ≥ d²: FlashAttention IO = O(N²d²/M) ≤ O(N²) ← 严格优于标准方法
H100 SRAM=228KB, d=128: M/d² ≈ 14 → 约 14× IO 减少
```

## 11. 总结

| 技术要点 | 实现方式 | 源码位置 |
|---------|---------|---------|
| Running max/sum | Softmax struct | softmax.h L92-168 |
| exp2f 替代 expf | scale_apply_exp2 | softmax.h L64-88 |
| 延迟 warp reduce | online_softmax + finalize 分离 | softmax.h L127-154 |
| O 修正 | rescale_o | softmax.h L157-166 |
| NaN/Inf 保护 | Check_inf 模板参数 | softmax.h L77, L116 |
| FP8 范围扩展 | Max_offset 模板参数 | softmax.h L67-69 |
| LSE 复用输出 | row_sum 被覆写为 LSE | softmax.h L151 |
