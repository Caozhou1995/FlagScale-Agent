"""G12: 预算感知的早收敛检查点（压步数轴）。

实测依据（analysis/agent_time_decomposition.py，60 个 win）：
  REAL 组步数 med=69，BORROWED 组 med=111（1.61x）；每步延迟 10.71s vs 14.06s（1.31x）。
  ⇒ 总耗时 2.04x = 1.61x × 1.31x。压步数是两轴之一。

当前 prompt §7 用预算分数触发承载该纪律（当前 prompt 中该条款不再是 "- " bullet）：
「At four fifths of budget, stop exploring — finish and verify what is delivered」。
缺这一条时，agent 会
一路探索到被墙撞停（BORROWED 组的典型形状：595 步仍未收敛）。

本测试锁定该条款必须存在的三个可判别要素：
  1. 一个**周期性**的检查点（不是只在开始/结束各一次）
  2. 判据是**预算比例 vs 必做项比例**的对比（不是"还剩多少时间"这种模糊说法）
  3. 触发后的**具体动作限制**（禁止新增探索性调用，只允许收尾/验证）
"""

CASES = {
    "budget_fraction_checkpoint": [
        "four fifths of budget",
        "stop exploring",
        "finish and verify",
    ],
    "budget_vs_required_ratio": [
        "budget",
        "four fifths",
        "runs out",
    ],
    "restriction_action": [
        "stop exploring",
        "finish",
        "verify",
    ],
}


def _prompt() -> str:
    import pathlib

    p = pathlib.Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
    return p.read_text()


def test_budget_aware_checkpoint_clause_exists():
    txt = _prompt()
    # 必须有一条同时含「预算分数触发」+「停止探索」+「只收尾/验证」三段语义的条款
    # （该条款已不再是 "- " bullet，措辞已变为 §7 的 budget-fraction 触发）
    clauses = [ln for ln in txt.splitlines() if ln.strip()]
    hit = [
        ln
        for ln in clauses
        if "four fifths of budget" in ln.lower()
        and "stop exploring" in ln.lower()
        and ("finish" in ln.lower() or "verify" in ln.lower())
    ]
    assert hit, (
        "no clause states a budget-fraction checkpoint that stops exploring and "
        f"restricts the remainder to finishing/verifying; "
        f"{len(clauses)} non-empty lines present"
    )


def test_checkpoint_names_the_ratio_test():
    low = _prompt().lower()
    assert "four fifths of budget" in low and "stop exploring" in low, (
        "clause must pin the budget FRACTION consumed ('four fifths') as the "
        "checkpoint trigger, not merely 'how much time is left'"
    )


def test_checkpoint_restricts_action_after_trigger():
    low = _prompt().lower()
    assert "stop exploring" in low and "finish and verify" in low, (
        "clause must restrict work after the budget-fraction trigger to "
        "finishing and verifying what is delivered — otherwise the checkpoint "
        "has no teeth"
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
    assert (
        "at the delivery path early" in low or "front-load" in low
    ), "land-early (front-load) discipline must stay"
    assert (
        "momentum" in low or "budget runs out" in low
    ), "TIME-BUDGET discipline (momentum / keep-claiming until budget runs out) must stay"
