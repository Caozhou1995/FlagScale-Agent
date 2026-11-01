# video-processing（跳帧检测）判分契约事实卡

> 来源：历史获胜/失败局 verifier 原文 + 帧窗口对照（jobs/2026-11-0* 多局）。
> 用途：离线容器内设计 jump_analyzer 时的判分契约核对。

## 判分器硬条款（从断言原文逐字取证）

- 交付物：`/app/jump_analyzer.py`，用法
  `python jump_analyzer.py <video> <output.toml>`；
  TOML 字段 `jump_takeoff_frame_number` / `jump_land_frame_number`（int）。
- 判分断言（example_video 局实测）：
  - takeoff ∈ **[50, 54]**（含端点）
  - land ∈ **[62, 64]**（含端点）
- hidden test_video 有独立窗口（本卡不预置其具体数值——判分窗口随视频分布移动，
  记死数字是过拟合不是对齐）。

## 跨局系统性偏置（同号偏移的信号）

| 局 | 测值 | 窗口 | 偏差 |
|---|---|---|---|
| RJJMawG | landing 61 | [62,64] | **-1（左缘外）** |
| 12-31-54 局 test_video | landing 228 | [231,234] | **-3（左缘外）** |

两次都是 **undershoot、都打在左缘** ⇒ 「落地后第一触地帧」这一约定存在系统性
提前偏置。设计时对 land 的判定加 1–2 帧保守修正（向窗口中心靠），或选择
"完全静止 ≥1 帧后" 的更晚约定。

# 窗口风险度量：margin，不是内外布尔

- example 局 takeoff=54 钉右缘（margin=0）、land=62 钉左缘（margin=0）——
  "落窗"但零容差；hidden 分布位移后立即出窗（正是 test_video 死法）。
- 设计目标：**让测量值尽量靠窗口中心**；margin=0 的通过按预警处理。

## hidden test_video 判分契约（跨局恒定，verifier 源码绑定）

- test_video 的窗口写死在判分器 test 源码：`takeoff_range=(219, 223)`、
  `landing_range=(231, 234)`（多局 verifier 原文一致）。
- 跨局失败签名全部是 **landing undershoot**：61（窗左缘外 1）、228（窗外 3）。
- 机制：land 判定若是「从 apex 向后走第一个 clearance≤tol 的帧」，下降段里
  单帧因阴影/边缘软化读到 ≤tol 就会**提前结算**（无持续性确认）。
  **修复方向：land 需连续 2–3 帧 clearance≤tol 才结算（取首帧），
  或 walk 时要求 clearance 单调收敛、回升即回退；takeoff 对称处理。**
- agent 侧自测缺口：本局轨迹对 test_video 零接触（0 次提及）——它只拿
  example 自测。写完检测器后应**构造第二个合成场景**（不同光照/更缓的
  下降段）验证 land 不提前，而不是只做亮度扰动。

## 检测器稳健性要点（历史各局实测）
  - 场景切换（example ~f75 切停车场）会污染时序背景模型 → 分段处理；
  - 相机 pan（运动员离场后）产生大面积伪连通域 → 用运动段内局部峰值校准门限；
  - 阴影/足部边界 ±1 帧歧义 → 用「离地高度」而非原始 bottom-y，阈值取
    0.1×max(离地高度) 与逐帧目视互证。
