# tb2.1 失败轨迹 RSI — dna-assembly 10-27 盘点

## Oracle 盘点（grep result.json，已执行）
- dna-assembly 12 局：03:51 __nUDknvc = **1.0**；其余 11 局 = 0.0
- dna-insert 01:30 局 = 1.0；03:51 局 = 0.0（反向失败）

## 判分器死因（全部失败局同一条断言）
```
/tests/test_outputs.py: parse_bsai_primer
  assert i >= 1, "Primer must have clamp of at least 1 nucleotide before BsaI site."
LOSE primer = 'ggtctcatgaggatcccgggaattctcgagta'   (clamp=0 → 死)
```

## 成功局 vs 失败局的构造差异（决定性）
| | 03:51 WIN (1.0) | 22:23 LOSE (0.0) |
|---|---|---|
| 5' clamp | `AATG` + GGTCTC + N + OV（4nt clamp） | GGTCTC + N + OV（无 clamp） |
| NEB 建议（extra 5' bases） | **加**（AATG 就是 extra bases） | **不加**（判 optional） |
| Tm 双 reading 验证 | 同时测 primer[15:] 和 primer[11:]，两者都过 | 只按自己模拟的判分器剥离假设算 |
| 死因 | — | clamp 断言 assert 0 >= 1 |

## 根因链（轨迹原文证据）
1. 22:23 局 L965-973：agent 知道 NEB 建议加 5' 额外碱基；
2. L2704：它**模拟判分器行为**（"grader would most likely strip first 11 chars... adding 4 extra bases would break the naive strip"）→ 用这个无证据的赌注否定了 NEB 建议；
3. 交付无 clamp 引物 → 判分器 `assert i >= 1` 必死。
4. 03:51 成功局：加了 clamp，且 Tm 在两种剥离假设（15:/11:）下都过 —— **union hedge**，天然免疫判分器解析方式。

## 失败模式分类
- **F1（top）：无证据的判分器模拟否定了廉价防御** —— agent 在判分器不可见时，用自己的猜测代替满足所有 defensible readings。可修点：harness prompt 层把"丢弃决策需 falsifier；不可观测时构造 union"绑定到具体判例。
- F2：计划层门（instrument_input/named-standard）已落地但没拦住 F1 —— 门在 plan 阶段，F1 死在执行中段的构造决策。
