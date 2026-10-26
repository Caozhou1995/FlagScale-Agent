# FlagGems 深度源码分析 · 第05章：fused/ 推理融合算子与 FLA 训练 kernel

> **知识组**: know-flaggems ｜ **源码**: FlagOpen/FlagGems @55a3ceda（快照 `/tmp/flaggems_repo`）
> **本章范围**: `src/flag_gems/fused/` —— 61 个顶层 `.py` + `FLA/` + `mhc/` + `DSA/` 三个子目录
> **上游关联**: fla-org/flash-linear-attention（FLA）与 Qwen3.6 GDN 训练路径
> **写作约定**: 所有 `file:line` 锚点均附**原文逐行引用**，行号已逐条对照快照源码核验

---

## 5.1 fused/ 目录是什么：与 ops/ 的角色分工

FlagGems 的算子体分两支（第01章 §1.3 结构总表的深化）：

- **`ops/`**（910 文件，~5243 个 def）：单算子单文件，`add/rms_norm/softmax...`，经
  `torch.library.Library("aten","IMPL")` 注册进 **`_FULL_CONFIG`** 全量注册表，
  由 `flag_gems.enable()` 批量挂上 ATen dispatch。
- **`fused/`**（61 个顶层文件 + 3 子目录）：**面向大模型推理/训练关键路径的融合算子包**，
  供上层（vLLM patch 层、FlagScale 训练侧、模块封装）**按名字直接 import 调用**，
  不走 `_FULL_CONFIG` 注册——这是本章的核心结构事实：

```
$ grep -rc "_FULL_CONFIG" src/flag_gems/fused/__init__.py  → 0（全目录 0 命中）
$ wc -l src/flag_gems/fused/__init__.py                    → 200 行显式 re-export 层
$ sed -n '21p;40p;44p' src/flag_gems/fused/__init__.py
    from flag_gems.fused.chunk_gated_delta_rule import chunk_gated_delta_rule
    from flag_gems.fused.FLA import (
    from flag_gems.fused.flash_mla import flash_mla
$ sed -n '25p'  src/flag_gems/__init__.py
    from flag_gems.fused import *  # noqa: F403
```

即：`fused/__init__.py`（200 行）把各算子**显式 re-export**（:15 起 `from flag_gems.fused.X import ...`，
覆盖 DSA/:40 附近与 FLA/ 入口），`flag_gems/__init__.py:25` 再 `import *` 透出到包顶层——
但注册表（`runtime/op_registrar.py:149`：
`f"Key '{key}' was not found in flag_gems._FULL_CONFIG; "` 的报错路径）从不检查它们。
调度链 evidence：`enable()` 的白名单/黑名单都只在 `_FULL_CONFIG` 里查键（op_registrar.py:131
`# attribute, look the key up in the authoritative _FULL_CONFIG to`），fused 算子既不在
`include` 也不受 `exclude` 影响——**FLA 训练 kernel 不在 `_FULL_CONFIG` 中**（任务要求注明），
消费方式是显式 import，例如 `chunk_gated_delta_rule.py:19`：

> `from flag_gems.fused.FLA import chunk_gated_delta_rule_fwd`

### 5.1.1 61 个顶层文件清单（按功能分组，量化）

（分组依据 `ls src/flag_gems/fused/*.py | wc -l` = 61 实测；行数抽查自 wc）

| 组 | 数量 | 代表文件 |
|---|---|---|
| 注意力/MLA | 9 | flash_mla.py(247行), flashmla_sparse.py, flash_mla_with_kvcache{,_model1}.py, concat_and_cache_mla.py, sparse_attention.py, top_k_per_row_{prefill,decode}.py |
| MoE | 9 | fused_moe.py(2300行), fused_marlin_moe.py, fp8_fp4_mega_moe.py, moe_align_block_size.py, moe_sum.py, grouped_topk.py, topk_softmax.py |
| GDN/线性注意力 | 1+子目录 | chunk_gated_delta_rule.py(248行) + FLA/15 文件 |
| DeepSeek-V4 前处理 | 9 | deepseek_v4_attention_fused_q_kv_rmsnorm.py, fused_deepseek_v4_{qnorm_rope_kv_rope_insert,qnorm_rope_kv_rope_quant_insert}.py, stage_deepseek_v4_mega_moe_inputs.py 等 |
| FP8/FP4 量化 | 7 | fp8_fp4_{mega_moe,mqa_logits,paged_mqa_logits}.py, act_quant.py, fused_indexer_q_rope_quant.py, fused_inv_rope_fp8_quant.py |
| 激活融合 | 6 | silu_and_mul.py(69行), swiglu.py, geglu.py, reglu.py, gelu_and_mul.py, silu_and_mul_with_clamp.py |
| MLA KV cache | 3 | reshape_and_cache{,_flash}.py, cp_gather_indexer_k_quant_cache.py |
| 其它 | 余量 | add_rms_norm.py, fused_add_rms_norm.py, skip_layernorm.py, post_layer_norm_residual.py, cross_entropy_loss.py, rwkv_ka_fusion.py, rwkv_mm_sparsity.py, mrope.py, weight_norm.py, matmuladd.py, matmul_bias_activation.py, outer.py, bincount.py, instance_norm.py, beam_search_score.py, apply_repetition_penalties.py, pack_seq.py, unpack_seq.py, topk_softplus_sqrt.py, DSA/(4文件), mhc/(5文件) |

三个子目录：
- `FLA/`：**15 个文件**，chunked GatedDeltaNet 训练 kernel（源自 fla-org/flash-linear-attention，见 §5.3）
- `mhc/`：5 文件（`mhc_pre/mhc_post/mhc_bwd/hc_head_fused_kernel/hc_split_sinkhorn`）——
  超连接（Manifold/MHC 类结构）预/后处理与 Sinkhorn 归一化 kernel
- `DSA/`：4 文件（`bin_topk/indexer_k_tiled/sparse_mla` 等）——DeepSeek V3.2 式稀疏注意力索引路径

---

## 5.2 chunk_gated_delta_rule.py：公开入口与三级调度

`fused/chunk_gated_delta_rule.py`（248 行，FlagOS Apache-2.0 头 + 无 FLA 头）是 GDN
（Gated DeltaNet）前向的公开入口。证据链：

- `chunk_gated_delta_rule.py:155` 函数定义（原文）：

> `def chunk_gated_delta_rule(`

- `chunk_gated_delta_rule.py:177-178` 硬性限定（原文）：

> `    if BT != 64:`
> `        raise ValueError("chunk_gated_delta_rule currently supports only BT=64")`

入口做**三级调度**（自上而下短路）：

1. **direct 快路径**（短序列）：`chunk_gated_delta_rule.py:193-200` 条件组合
   （原文 :196-199 逐行为 `and T <= 128` / `and K <= 128` / `and V <= 128` / `and H % Hg == 0`），
   命中则 `chunk_gated_delta_rule.py:215`（原文）调
   `            o, final_state = chunk_gated_delta_rule_direct_fwd(`（单 kernel 吞掉整段递推）；
2. **q/k L2 归一化**（Qwen3.6 GDN 需要）：`chunk_gated_delta_rule.py:230-232`（原文）：

> `    if use_qk_l2norm_in_kernel:`
> `        q_seq = _l2_normalize_last_dim(q_seq)`
> `        k_seq = _l2_normalize_last_dim(k_seq)`

   其实现是本文件内 `@libentry() @triton.jit` 的 `_l2_normalize_last_dim_kernel`（:29），
   逐行做 `values * (1/sqrt(sum(v^2)))`（:51，`1e-6` 下限防除零）；
3. **chunk 主路径**：`chunk_gated_delta_rule.py:234`（原文）——
   `    _, o, _, final_state, _, _, _ = chunk_gated_delta_rule_fwd(`
   （转 FLA/ 管线，§5.3）。

布局契约（`_validate_inputs` 位于 :71；约束行原文 :97-98）：

> `if H % Hg != 0:`
> `    raise ValueError("the q/k head count must divide the v head count")`

即 **q/k 头数必须整除 v 头数**——FLA 包里 `Hg`（q/k 头数）与 `H`（v 头数）分轨的原因，
Qwen3.6 GDN 的 q/k 头数 < v 头数正是靠这个约定喂进来（§5.4）。

## 5.3 FLA/：chunked GatedDeltaNet 训练 kernel（源自 flash-linear-attention）

### 5.3.1 上游出处证据（离线可核验）

GitHub 直接访问被代理阻断（403 / 代理握手超时，两种拼写均试过），出处改用**源内证据链**：

1. **MIT 版权头**（FLA/ 15 文件中 13 个标注复制来源）——`FLA/chunk_delta_h.py:1-4`（原文）：

> `# This file contains code copied from the flash-linear-attention project.`
> `# The original source code was licensed under the MIT license and included`
> `# the following copyright notice:`
> `# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang`

   Songlin Yang / Yu Zhang 正是 fla-org/flash-linear-attention 的作者，MIT 许可一致。
2. **上游差异的代码内说明**——`FLA/chunk_delta_h.py:291-292`（原文）：

> `    # This kernel is slightly different from fla to support Q/K with different head numbers.`
> `    # In fla, Q/K always have the same head number, so Hg is always equal to H.`

3. **FlagGems 原创融合件**（Apache-2.0 头，无 FLA 版权头）证明本目录是深度改造而非纯拷贝：
   - `FLA/fused_cumsum_kkt_solve_tril.py:1-2`（原文）：
     `# Copyright (c) 2025 FlagGems. All rights reserved.` ＋
     `# Fused cumsum + KKT + solve_tril for chunk_gated_delta_rule. Returns g_out, A_inv; w_u is separate.`
   - `FLA/chunk_fused_tail_vblock.py:15`（原文）：
     `# V-blocked fused tail for the official K=V=BT=64 chunk_gated_delta_rule path.`
   - 入口 `chunk_gated_delta_rule.py` 为 FlagOS Apache-2.0 头（:1-13），非 FLA 拷贝。

### 5.3.2 chunk 主路径管线（kernel 调用链，锚点已核验）

`FLA/chunk.py:47` 起 `chunk_gated_delta_rule_fwd(...)` 串联四条主调用（chunk.py:92/:100/:129/:139）：

> `    g, A = chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril(`（chunk.py:92）
> `    w, u = recompute_w_u_fwd(`（chunk.py:100）
> `    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(`（chunk.py:129）
> `    o = chunk_fwd_o(`（chunk.py:139）

chunk 大小自适应（原文，chunk.py:41-44）：

> `def _chunk_size_for_sequence(T: int, is_varlen: bool) -> int:`
> `    if is_varlen:`
> `        return 64`
> `    return min(64, max(16, 1 << (T - 1).bit_length()))`

配 ASCII 数据流图（方括号内为 kernel 所在文件:行）：

```
 q,k,v,beta,g ──► ┌────────────────────────────────────────────────────┐
 (head_first可选)  │ P1 cumsum(g) + KKT + solve_tril                    │
                  │  [fused_cumsum_kkt_solve_tril.py:35 1-kernel融合]   │
                  └───────┬────────────────────────────────────────────┘
                     g(cumsum), A, A_inv
                          │
                  ┌───────▼──────────────┐     ┌────────────────────────┐
                  │ P2 WY: recompute_w_u │     │ P3 状态递推             │
                  │  [wy_fast.py:27]     ├────►│ [chunk_delta_h.py:41]  │
                  └──────────────────────┘  w,u│  h[NT,K,V], v_new      │
                                               └───────┬────────────────┘
                  ┌────────────────────────────────────┼──────────────┐
                  │ P4 输出: chunk_fwd_o [chunk_o.py:39] ◄─────────────┘ h,v_new
                  │   o = q@h*exp(g) + intra-chunk(q@A@v)*scale
                  └──► o, final_state  (备选: fused_tail_vblock 单kernel合并P3+P4)
```

各阶段职责（机制级）：

- **P1**：块内 `tl.cumsum(b_g, axis=0)` 得 g 的 chunk 内前缀和；KKT 产生
  `A = tril(beta·K·Kᵀ · exp(g_i-g_j))`（因果下三角掩码），solve_tril 求 A⁻¹。
  上游 fla 把这三步拆成独立 kernel（chunk_scaled_dot_kkt.py:34
  `def chunk_scaled_dot_kkt_fwd_kernel(` 与 solve_tril.py:36
  `def solve_tril_16x16_kernel(` 均保留在树内供对照/单独复用），FlagGems 融合成一个。
- **P2 WY 表示**：`wy_fast.py:27` `recompute_w_u_fwd_kernel` 按 `b_u = tl.dot(b_A, b_vb, allow_tf32=False)`（:91）、
  `b_w = tl.dot(b_A, b_kb)`（:113）产出 WY 表示的 `w,u`。
- **P4 块输出**：`chunk_o.py:39` `chunk_fwd_kernel_o` —— 块间项
  `b_o += tl.dot(b_q, b_h)`（chunk_o.py:108）＋块内项 `b_A += tl.dot(b_q, b_k)`（:110，
  经门差 `b_A = b_A * exp(b_g[:, None] - b_g[None, :])` :117 与下三角掩码 :121-122），
  合并 `b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale`（chunk_o.py:134）。

### 5.3.3 chunk_delta_h.py：状态递推核心 kernel（342 行，逐段精读）

任务指定锚点核验通过——`chunk_delta_h.py:41`（原文）：

> `def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(`

**装饰器栈**（:19-40）：`@libentry()`（FlagGems 自有 kernel 分发，第01章 §3）→
`@triton.heuristics` 6 个编译期开关（USE_G/USE_GK/USE_INITIAL_STATE/STORE_FINAL_STATE/
SAVE_NEW_VALUE/IS_VARLEN，:22-27）→ `@libtuner` 12 配置调优（BV∈{32,64}×warps[2,4]×
stages[2,3,4]，key 含 `"IS_VARLEN"`，:30-38）→ `@triton.jit(do_not_specialize=["T"])`（:40）。

**grid 与状态切分**（原文 :67、:83）：

> `    i_v, i_nh = tl.program_id(0), tl.program_id(1)`
> `    b_h1 = tl.zeros([64, BV], dtype=tl.float32)`

状态 `h[K,V]` 按 K 维切最多 4 个 64 高块（b_h1..b_h4，:83-89），V 维由 grid 第 0 维切
BV 列；grid `(cdiv(V,BV), N*H)`（:320-321）——每程序负责一个 (v列块, 头) 的整条
chunk 序列递推，`"blockdim64"` 即 K 块 64 之意。K≤256 硬约束在
chunk_delta_h.py:311：`assert K <= 256, "current kernel does not support head dimension larger than 256."`

**主循环 `for i_t in range(NT)`**（:128 起）四步（每步原文锚点）：

1. **存旧状态**：`tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))`（:132）；
2. **残差 v_new = u − w@h**（原文 :153/:175）：

> `        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))`
> `        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v`

   delta-rule 的"减去已解释部分"：`w@h` 是当前状态对 u 的预测，差值 `b_v`=v_new
   是**新信息**（SAVE_NEW_VALUE 时 :177-181 存回，供 P4 消费）；
3. **门衰减**（:184-199）：块尾门 `b_g_last` 与块内门差 `exp(b_g_last - b_g)`
   逐 token 重加权（:191），状态整体乘 `b_g_last`（:193）；GK 向量门按 K 维 64 段
   乘 `exp(b_gk_last)`（USE_GK，:201-232）；
4. **状态更新**（原文 :239）：

> `        b_h1 += tl.dot(b_k, b_v)`

   （K>64/128/192 时 b_h2..b_h4 在 :245/:251/:257 逐段追加同样的 dot）。

**包装器** `chunk_gated_delta_rule_fwd_h`（:279-342）：形状展开、`assert K<=256`（:311）、
`h = k.new_empty(B, NT, H, K, V)`（:313）、grid 定义（:320-321）。
入口侧佐证锚点（均已核）：`chunk_gated_delta_rule.py:19`（原文
`from flag_gems.fused.FLA import chunk_gated_delta_rule_fwd`，显式 import 消费）；
`:29` `def _l2_normalize_last_dim_kernel(`；`:51` 归一化实现行（`1e-6` 防除零）；
`:1`/`:13` 界定 Apache 头范围（`:1` `# Copyright 2026 FlagOS Contributors`）。

### 5.3.4 FlagGems 原创：单 kernel 融合 P1（fused_cumsum_kkt_solve_tril.py）

`FLA/fused_cumsum_kkt_solve_tril.py`（460 行，`cat -n` 尾行实测；Apache-2.0）把上游
三个独立 kernel（cumsum → KKT → solve_tril）压进**一个** launch：

> `# Copyright (c) 2025 FlagGems. All rights reserved.`
> `# Fused cumsum + KKT + solve_tril for chunk_gated_delta_rule. Returns g_out, A_inv; w_u is separate.`

kernel（:35 起）体内三段注释即三阶段（原文）：`# ---------- cumsum ----------`（:70）、
`# ---------- KKT (write L to A) ----------`（:81）、
`# ---------- solve_tril (read A, write A_inv) ----------`（:111）。
收益：上游三 kernel 间 `A[B,T,H,BT]` 大张量需两轮 HBM round-trip，融合后 A 只活在
寄存器/SMEM——GDN 训练 BT×BT 小矩阵密集，这是实打实的省流。

**solve_tril 机制**（独立文件 `FLA/solve_tril.py` 保留对照）：BT×BT 下三角求逆拆成
**16×16 块串行前代 + 2×2 分块合并**：
- `solve_tril_16x16_kernel`（solve_tril.py:36）：块内 `for i in range(2, min(16, T - i_t * 16)):`
  逐行前代（:82），TMA 支持时走 `make_tensor_descriptor`（:77）；
- `merge_16x16_to_32x32_inverse_kernel`（solve_tril.py:112）：2×2 分块求逆，块间耦合项
  仅 `b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21, ...), b_Ai_11, ...)`（体内双 dot），
  精度经 `FLA_TRIL_PRECISION` env 控制（"ieee"/"tf32"/"tf32x3"，solve_tril.py:18-19）。

### 5.3.5 其余 FLA kernel 与朴素对照实现

- **chunk_fused_tail_vblock.py**（184 行）：FlagOS 原创的 K=V=BT=64 专用融合（:15
  `# V-blocked fused tail for the official K=V=BT=64 chunk_gated_delta_rule path.`）。
  `can_use_fused_tail_vblock`（:26）要求**非 varlen 且有 initial_state 且要 final_state**、
  `(K,V)=(64,64)`、`T%64==0`、fp16/bf16 全连续（:38-57）才启用；kernel
  `_chunk_gated_delta_rule_fused_tail_vblock_kernel`（:62）把 P3 状态递推与 P4 输出
  合并：`residual = u - w@h_acc`（:117）、`q_h = dot(q, h_acc)`（:119）、
  状态更新 `h_acc = h_acc*exp(g_last) + dot(k_t, residual*exp(g_last-g))`（:135-137）。
  chunk.py:108-128 在 `SUPPRESS_LEVEL < 3` 时优先走这条备选路径。

- **fused_recurrent.py**（686 行，MIT 头同 FLA 系）：**逐 token 递推**路径——
  `fused_recurrent_gated_delta_rule_fwd_kernel`（:188）与 Qwen3-Next 专用变体
  `fused_recurrent_gated_delta_rule_fwd_sp_for_qwen3_next_kernel`（:30），
  包装器 `fused_recurrent_gated_delta_rule_fwd`（:496）。解码阶段逐 token 更新状态
  无需 chunk 开销，与 chunk 管线（prefill）互补；二者均由 `FLA/__init__.py`
  `__all__` 导出。

- **naive_chunk_gated_delta_rule_fwd**：`FLA/chunk.py:155` 是逐 token 参考实现，
docstring 给出递推式（chunk.py:159-160，原文）：

> `        S_t = exp(g_t) * S_{t-1} + beta_t * k_t^T (v_t - k_t @ S_{t-1})`
> `        o_t = q_t @ S_t * scale`

这正是 §5.3.3 伪代码的标量形式——chunk 管线即它的 BT×BT 分块矩阵化。
## 5.4 FLA 与上游 fla / Qwen3.6 GDN 的关系（含 WHY 与差异清单）

### 5.4.1 语义对位：Gated Delta Rule 与 Qwen3.6 GDN

Gated DeltaNet（线性注意力，Songlin Yang 等提出的 DeltaNet 加门控变体）以**可学习的
遗忘门 g 与 delta 修正规则**替代 softmax 注意力的二次注意力矩阵，状态为固定
`[K,V]` 矩阵，推理时 KV cache 退化为**常数大小状态**——这是 Qwen3.6 系列采用
GDN+标准注意力混合（4:1 分层，见 qwen3.6 模型分析）的算力动机。
FlagGems 把这条训练路径整体搬进 `fused/FLA/`，供 FlagScale/Megatron 侧
`chunk_gated_delta_rule` 前向调用；kernel 头部 `Hg`（q/k 头数）与 `H`（v 头数）
分离正是为 GQA 式 GDN 服务（§5.3.1 证据 2）。

### 5.4.2 FlagGems 相对上游 fla 的改动（代码内可证）

| # | 差异 | 证据（file:line 原文见前/后文） | 动机 |
|---|---|---|---|
| 1 | Q/K 头数与 V 头数解耦（Hg≠H） | chunk_delta_h.py:291-292；:94 `k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)` | Qwen3.6 GDN q/k 头数<v 头数 |
| 2 | P1 三 kernel 融合为一 | fused_cumsum_kkt_solve_tril.py:1-2 Apache 头 | 省 HBM round-trip |
| 3 | fused tail vblock（P3+P4 单 kernel） | chunk_fused_tail_vblock.py:15 | K=V=BT=64 官方形状再省两轮 |
| 4 | chunk_size 自适应 16..64 | chunk.py:41-44 `_chunk_size_for_sequence` | 短序列少浪费 |
| 5 | FlagGems 基建接管调优 | chunk_delta_h.py:14 `from flag_gems.utils import libentry, libtuner` | 复用 LibEntry/LibTuner（第01章 §3） |
| 6 | 设备探测换 flag_gems.runtime | FLA/utils.py:26-36 `_detect_nvidia_hopper`（vendor_name+capability） | 多 vendor 后端一致 |
| 7 | 入口公开封装+校验+direct 分流 | chunk_gated_delta_rule.py:155（def）/ :71（`def _validate_inputs(`）/ :177-178（BT=64 强制）/ :215（direct_fwd 调用） | 接 Qwen3.6 GDN 布局(head_first) |

### 5.4.3 WHY：为什么放 `fused/` 而不进 `ops/`

(a) 这些是**多算子组合的管线**，不是单 ATen op——ATen 没有对应符号可挂；
(b) 它们是**训练侧专属**路径（GDN 前向），而 `enable()` 的语义是"替换单 op"，
    挂进 `_FULL_CONFIG` 会被 vendor include/exclude 策略误伤；
(c) vLLM/FlagScale 消费时是**显式调用**（vLLM patch 层按名 import），不需要
    dispatch 魔法。代价：失去 `_FULL_CONFIG` 的 exclude/precision-check 编排，
    所以 FLA 自带 env 开关（`GDN_RECOMPUTE_SUPPRESS_LEVEL`/`FLA_GDN_FIX_BT`/
    `FLA_USE_CUDA_GRAPH`，FLA/utils.py:20-23，原文锚点见 §5.3.5 表）与
    `FLA_USE_FAST_OPS`（triton_ops_helper.py:14-22 exp 快/慢实现切换）。

## 5.5 fused_moe.py（2300 行）：Triton MoE GEMM 主干

结构图（grep 实测 def 行号，抽查已核）：

```
 get_moe_configs(:166)/try_get_optimal_moe_config(:206)   ← 从 json 选调优配置
        │
 fused_experts_impl(:1826)                                 ← 主编排
   ├─ moe_kernel_quantize_input(:754)                      ← fp8/int8 量化入口
   │    ├─ _fp8_quantize(:634) / _int8_quantize(:706)
   ├─ dispatch_fused_moe_kernel(:1735)                     ← 选 fused_moe_kernel 或 wna16
   │    ├─ fused_moe_kernel(:1021)        @triton.jit 主 GEMM
   │    └─ fused_moe_kernel_gptq_awq(:829)                 ← GPTQ/AWQ 权重量化变体
   ├─ apply_moe_activation(:585)（MoEActivation 枚举 :543: silu/geglu/...）
   └─ invoke_fused_moe_triton_kernel(:1607)/invoke_fused_moe_wna16_triton_kernel(:1517)
```

机制要点：分组 GEMM（expert 维当 batch）、`moe_align_block_size` 排布 token×expert、
激活由 `MoEActivation` 枚举驱动（:543）与 `fused_marlin_moe.py`/`fp8_fp4_mega_moe.py`
形成 WNA16/FP8-FP4 三档算力档位。与 vLLM 的关系：`patch_vllm_all.py` 的 `_moe_C`
Library 注入（第01章 §5.3）使 `torch.ops._moe_C.topk_softmax/moe_align_block_size`
落到 FlagGems 实现，`fused_moe_kernel` 是其中的计算核心。

## 5.6 其它代表性融合算子（量化+WHY）

- **silu_and_mul.py**（69 行，全文件已读）：`@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])`
  （silu_and_mul.py:26，原文）+ `def silu_and_mul_kernel(x, y):`（:28）+
  `    x_silu = tl.fdiv(x_fp32, (1.0 + tl.exp(-x_fp32)))`（:30）+ `    return x_silu * y`（:31）。
  一个 pointwise kernel 吃掉 `silu(x)*y`，vLLM `SiluAndMul.forward_cuda` 被
  patch 成此实现（第01章 §5.3 module_patches）；同目录 `swiglu/geglu/reglu/gelu_and_mul`
  同范式。
- **flash_mla.py**（247 行）：MLA 注意力主 kernel，配合
  `reshape_and_cache{,_flash}.py`（KV 写 cache）与 `concat_and_cache_mla.py`
  （DeepSeek MLA 的 KV 拼接 cache）覆盖 MLA 推理全链；`flashmla_sparse.py`+
  `DSA/` 支撑 V3.2 稀疏注意力（`top_k_per_row_{prefill,decode}.py` 选 top-k 索引）。
- **deepseek_v4 系列**（9 文件）：`deepseek_v4_attention_fused_q_kv_rmsnorm.py`、
  `fused_deepseek_v4_qnorm_rope_kv_rope_{insert,quant_insert}.py`、
  `stage_deepseek_v4_mega_moe_inputs.py`、`fp8_fp4_mega_moe.py` 等——把
  QK Norm→RoPE→FP8 量化→cache 插入 的 V4 前处理链逐级压成单 kernel；
  `fp8_fp4_mega_moe.py` 是 FP8/FP4 混合精度的 MoE 巨核。
- **rwkv_ka_fusion.py / rwkv_mm_sparsity.py**：RWKV 训练侧融合，与 FLA 同属线性注意力家族。

## 5.7 mhc/ 与 DSA/ 概览

- `mhc/`（5 文件）：`mhc_pre.py/mhc_post.py/mhc_bwd.py`（超连接层前/后/反向）、
  `hc_head_fused_kernel.py`（头融合）、`hc_split_sinkhorn.py`（Sinkhorn 归一化，
  行随机矩阵投影）——manifold 类超连接结构的训练 kernel 集。
- `DSA/`（4 文件）：`bin_topk.py`（bin 化 top-k 索引）、`indexer_k_tiled.py`
  （分块 K 索引器）、`sparse_mla.py`（稀疏 MLA）——DeepSeek V3.2 DSA 稀疏注意力
  的索引+稀疏计算路径，与顶层 `cp_gather_indexer_k_quant_cache.py`、
  `indexer_k_quant_and_cache.py` 配套。

## 5.8 端到端调用时序（伪代码，聚合前文机制）

```
# ---- 训练侧 (FlagScale, GDN 层前向) ----
layer.forward(q,k,v,beta,g):
    o, S = flag_gems.fused.chunk_gated_delta_rule(
        q,k,v,beta,g, BT=64, head_first=..., use_qk_l2norm_in_kernel=True)
    # 内部: direct? -> direct_fwd(短序列单kernel)
    # 否则: fused_cumsum_kkt_solve_tril -> recompute_w_u_fwd
    #       -> fused_tail_vblock(可选) 或  chunk_delta_h -> chunk_fwd_o
    out = attn_mix(o, softmax_attn_out)   # Qwen3.6: GDN层:标准层 = 4:1 混合

# ---- 推理侧 (vLLM worker, USE_FLAGGEMS=1) ----
apply_gems_patches_to_vllm():       # patch_vllm_all.py:643
    SiluAndMul.forward_cuda  = flag_gems.fused.silu_and_mul     # module patch
    torch.ops._moe_C.*       = flag_gems fused moe kernels      # Library patch
    torch.ops._C.rms_norm... = flag_gems normalization kernels
```

## 5.8.1 核心结论

1. `fused/` 是 FlagGems 的"**手工融合层**"：不进 `_FULL_CONFIG`（`fused/__init__.py` grep 0 命中），
   200 行 `__init__.py` 显式 re-export（:15 起），经 `flag_gems/__init__.py:25`
   `import *` 透出，由 vLLM patch 与 FlagScale 训练侧**显式消费**——
   与 `ops/` 的 ATen 自动替换形成互补双轨。
2. `FLA/` 是 fla-org/flash-linear-attention 的**许可兼容移植+深度改造**：
   MIT 头（chunk_delta_h.py:1-4）+ FlagGems Apache 融合件（fused_cumsum_kkt_solve_tril.py:1-2、
   chunk_fused_tail_vblock.py:15）+ Hg≠H 解耦（chunk_delta_h.py:291-292），
   服务 Qwen3.6 GDN 训练。
3. 性能哲学：GDN 管线每个中间张量（A、A_inv、g、w/u、h、v_new）要么融合进单 kernel
   要么降为 BT×BT 小块，HBM 流量是第一优化对象；调优统一交给 LibEntry/LibTuner。
