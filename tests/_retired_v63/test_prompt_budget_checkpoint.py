"""G12: 预算感知的早收敛检查点（压步数轴）。

实测依据（analysis/agent_time_decomposition.py，60 个 win）：
  REAL 组步数 med=69，BORROWED 组 med=111（1.61x）；每步延迟 10.71s vs 14.06s（1.31x）。
  ⇒ 总耗时 2.04x = 1.61x × 1.31x。压步数是两轴之一。

当前 prompt 的 TIME BUDGET 段只说「momentum / front-load」，**没有**要求 agent
在运行中周期性核对「已用预算比例 vs 已完成的必做项比例」。缺这一条时，agent 会
一路探索到被墙撞停（BORROWED 组的典型形状：595 步仍未收敛）。

本测试锁定该条款必须存在的三个可判别要素：
  1. 一个**周期性**的检查点（不是只在开始/结束各一次）
  2. 判据是**预算比例 vs 必做项比例**的对比（不是"还剩多少时间"这种模糊说法）
  3. 触发后的**具体动作限制**（禁止新增探索性调用，只允许收尾/验证）
"""

CASES = {
    "periodic_checkpoint": [
        "periodic",
        "checkpoint",
        "every",
    ],
    "budget_vs_required_ratio": [
        "budget",
        "required",
        "ratio",
    ],
    "restriction_action": [
        "exploratory",
        "verification",
        "forbid",
        "only",
    ],
}


def _prompt() -> str:
    import pathlib

    p = pathlib.Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
    return p.read_text()


def test_budget_aware_checkpoint_clause_exists():
    txt = _prompt()
    low = txt.lower()
    # 必须有一条同时含「周期」+「预算比例 vs 必做项」+「限制探索」三段语义的条款
    clauses = [ln for ln in txt.splitlines() if ln.strip().startswith("- ")]
    hit = [
        ln
        for ln in clauses
        if "periodic" in ln.lower()
        and "budget" in ln.lower()
        and ("required" in ln.lower() or "must-do" in ln.lower())
    ]
    assert hit, (
        "no clause states a PERIODIC budget-vs-required checkpoint; "
        f"{len(clauses)} bullet clauses present"
    )


def test_checkpoint_names_the_ratio_test():
    low = _prompt().lower()
    assert "ratio" in low or "proportion" in low, (
        "clause must compare USED BUDGET RATIO against COMPLETED REQUIRED RATIO, "
        "not merely 'how much time is left'"
    )


def test_checkpoint_restricts_action_after_trigger():
    low = _prompt().lower()
    assert "exploratory" in low, (
        "clause must forbid new EXPLORATORY calls once the budget/required ratio "
        "turns adverse — otherwise the checkpoint has no teeth"
    )


def test_no_task_identifying_words_in_new_clause():
    """防泄漏：新条款不得引入任何任务名/任务包特定词。"""
    low = _prompt().lower()
    for leak in (
        "terminal-bench",
        "extract-elf",
        "sanitize-git",
        "video-processing",
        "raman",
        "protein-assembly",
        "pytorch-model-cli",
        "mips",
        "gcode",
        "dna-assembly",
    ):
        assert leak not in low, f"leak: {leak}"


def test_existing_time_budget_clause_still_intact():
    """回归：原有 TIME BUDGET 的 front-load / momentum 要求不得被新条款挤掉。"""
    low = _prompt().lower()
    assert "front-load" in low or "front load" in low
    assert "momentum" in low
