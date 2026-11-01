# dna-assembly（Golden Gate / BsaI-HF v2）判分契约事实卡

> 来源：86 局历史轨迹 + 判分器断言字节 + 获胜局/失败局对照（jobs/2026-11-0*）。
> 用途：容器无网络、无法按任务说明书"check NEB's requirements"时，以此卡代替外部求证。

## 任务原文给的关键提示（必读）

任务说明书原文：*"If you aren't familiar with BsaI-HF v2 make sure to check that
the enzyme cut-sites you design satisfy NEB's requirements."*
—— 这句话是判分契约的路标：**判分器会按 NEB 规则校验酶切位点**，任何你无法从
NEB 文档核实的约定都是隐性死亡点。

## 判分器已知的硬条款（从断言原文逐字取证）

1. **5' clamp ≥ 1 nt**：判分器 `parse_bsai_primer` 执行
   `i = primer.find("ggtctc"); assert i >= 1, "Primer must have clamp of at least
   1 nucleotide before BsaI site"`。
   - 机制：NEB BsaI-HF v2 需要识别位点 **5' 端外侧** ≥1 个碱基才能有效切割。
   - **方向是"位点之前"**：`AGGTCTC...` 合法（i=1）；`GGTCTC...` 开头即死（i=0）。
   - 这是本任务 86 局中 36 例的头号死法（census: reward=0 的 74 局里 36 例）。
   - 最小修复已实测：在 GGTCTC 前加任意 1 碱基即转绿（find→1，assert PASS）。
2. **spacer/overhang 方向**：标准写法 `5'-[1nt clamp]-GGTCTC-[1nt spacer]-[4nt
   overhang]-[annealing region]`。spacer 在位点**之后**、clamp 在位点**之前**，
   两者是不同的碱基，不要合并也不要放错侧。
3. **Tm 三条款**：annealing 区 15–45 nt；单体 Tm∈[58,72]°C；每对 fwd/rev ΔTm≤5°C。
   Tm 只按 annealing 段计算（不含尾巴），oligotm 参数
   `-tp 1 -sc 1 -mv 50 -dv 2 -n 0.8 -d 500` 为 ground truth。
4. **产物形式**：`primers.fasta`，header `>TEMPLATENAME_DIR`（fwd/rev），
   无空行，最少 primer 对数。

## 历史败局签名速查

| 签名 | 例数(86局) | 死因 |
|---|---|---|
| clamp i=0（GGTCTC 开头） | 36 | 位点前无碱基 |
| ΔTm>5 | 11 | overhang 并入退火段后真实 Tm 漂移 3–10° |
| 文件不存在 | 9 | 没写 /app/primers.fasta |
| fwd Tm 越界 | 7 | 只算 binding 区漏 4nt overhang 的退火贡献 |

## 设计自检清单（交付前逐条跑）

```python
p = <fwd_primer>          # 全小写
i = p.find("ggtctc")
assert i >= 1, "clamp 缺失"
```
并确认 rev 侧对应位点 `gagacc` 同样满足其自身 clamp 方向。
