"""Pin: 自检读 FAIL 时的对称条款必须存在且在 PASS 方向之后。

provenance: dna-assembly__d3u573a (job 2026-11-27__13-29-30) 真实失败形态 —
自建 PCR/digest/ligation 模拟器报 FAIL（其 anneal 搜索有 bug），agent 未校准模拟器
即反复重写设计脚本 8 次（gg.py→final.py→design2→f3→f4→f5→g.py→below），
最后在 87% 预算处放弃并标 NOT VERIFIED；判分器实际失败点 (overhang 几何, :178)
与它自建模拟器报的错不在同一处。
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from flagscale_agent.react import prompt as P  # noqa: E402

TXT = P.SYSTEM_PROMPT_STATIC
FLAT = TXT.lower()


def test_clause_present():
    assert "calibrate it on a known-good input" in FLAT, (
        "缺少「自检读 FAIL 时先校准检查器」的反方向条款")


def test_clause_names_the_action_not_just_the_virtue():
    # 必须点名动作：rewriting the deliverable
    assert "before rewriting the deliverable" in FLAT


def test_clause_follows_the_pass_side_rule():
    i = FLAT.find("a self-check you wrote is not the grader")
    j = FLAT.find("reverse it too")
    assert i != -1 and j != -1 and i < j, (
        f"反方向条款应紧接在 PASS 方向之后: pass@{i} new@{j}")


def test_budget_not_raised():
    assert len(TXT) <= P.PROMPT_BUDGET, (len(TXT), P.PROMPT_BUDGET)
