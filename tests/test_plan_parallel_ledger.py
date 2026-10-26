# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The time ledger must know the difference between a serial sum and a chain.

Regression cover for the defect where the ledger added every step estimate end
to end, so a plan with independent branches was reported as "over budget: cut
scope" when the same work fit inside the wall once the branches ran at the same
time. Three surfaces are covered:

  * `_coerce_deps`      -- caller-supplied dependencies are stored, not ignored
  * `_critical_path_secs` -- serial sum vs longest chain, and WHICH step to shrink
  * the rendered verdict -- fan-out first when a parallel saving exists, scope
                            cut only when the chain is genuinely too long
"""

import os
import tempfile
import time

import pytest

from flagscale_agent.react.plan import (
    TaskPlan,
    _coerce_deps,
    _critical_path_secs,
)


def _plan(tmp_path=None):
    d = tmp_path or tempfile.mkdtemp()
    p = TaskPlan(os.path.join(str(d), "p.yaml"))
    p._budget_sec = 900.0
    return p


def _ledger(plan_obj, steps, budget=900.0, reserve=135.0):
    doc = {
        "created": time.time(),
        "steps": steps,
        "time_budget": {
            "budget_sec": budget,
            "reserve_sec": reserve,
            "started": time.time(),
        },
    }
    return "\n".join(plan_obj._time_ledger_lines(doc))


# --------------------------------------------------------------------------
# _coerce_deps: defaults must keep the historical serial chain
# --------------------------------------------------------------------------

def test_coerce_deps_defaults_to_chain_when_absent():
    assert _coerce_deps(None, [4]) == [4]


def test_coerce_deps_preserves_explicit_parallel_shape():
    # Two steps declaring the same predecessor is how parallelism is expressed.
    assert _coerce_deps([1], [3]) == [1]


def test_coerce_deps_empty_list_is_a_real_answer_not_a_fallback():
    # [] means "can start immediately" and must survive.
    assert _coerce_deps([], [3]) == []


def test_coerce_deps_drops_junk_and_duplicates():
    assert _coerce_deps([1, 1, "2", 0, -5, None, True], [9]) == [1, 2]


def test_coerce_deps_unusable_nonempty_falls_back_to_chain():
    # Non-empty but nothing usable -> the safe serial default, never a bogus graph.
    assert _coerce_deps(["x", {}], [7]) == [7]


def test_create_stores_caller_dependencies(tmp_path):
    p = _plan(tmp_path)
    created = p.create(
        "t",
        [
            {"title": "A", "est_sec": 60},
            {"title": "B", "est_sec": 30, "depends_on": [1]},
            {"title": "C", "est_sec": 40, "depends_on": [1]},
        ],
    )
    deps = [s.get("depends_on") for s in created["steps"][:3]]
    assert deps == [[], [1], [1]]


def test_create_default_still_serial_chain(tmp_path):
    p = _plan(tmp_path)
    created = p.create("t", ["a", "b", "c"])
    deps = [s.get("depends_on") for s in created["steps"][:3]]
    assert deps == [[], [1], [2]]


# --------------------------------------------------------------------------
# _critical_path_secs: the measurement itself
# --------------------------------------------------------------------------

def test_critical_path_two_branches_saves_time():
    steps = [
        {"id": 1, "est_sec": 60, "depends_on": []},
        {"id": 2, "est_sec": 30, "depends_on": [1]},
        {"id": 3, "est_sec": 40, "depends_on": [1]},
        {"id": 4, "est_sec": 20, "depends_on": [2, 3]},
    ]
    r = _critical_path_secs(steps)
    assert r["serial"] == 150.0
    assert r["critical"] == 120.0
    assert r["gain"] == 30.0
    assert r["parallel"] == [2, 3]


def test_critical_path_pure_chain_has_zero_gain():
    steps = [
        {"id": 1, "est_sec": 60, "depends_on": []},
        {"id": 2, "est_sec": 30, "depends_on": [1]},
        {"id": 3, "est_sec": 40, "depends_on": [2]},
    ]
    r = _critical_path_secs(steps)
    assert r["serial"] == r["critical"] == 130.0
    assert r["gain"] == 0.0
    assert r["parallel"] == []


def test_constraint_is_the_step_to_shrink_not_the_last_to_finish():
    # The tail step finishes last but is cheap: cutting it saves only 20s.
    # The 500s step bounds the clock and is what a scope cut must target.
    steps = [
        {"id": 1, "est_sec": 100, "depends_on": []},
        {"id": 2, "est_sec": 500, "depends_on": [1]},
        {"id": 3, "est_sec": 20, "depends_on": [2]},
    ]
    r = _critical_path_secs(steps)
    assert r["constraint"] == 2


def test_critical_path_tolerates_cycle_without_hanging():
    steps = [
        {"id": 1, "est_sec": 10, "depends_on": [2]},
        {"id": 2, "est_sec": 10, "depends_on": [1]},
    ]
    r = _critical_path_secs(steps)
    assert r["critical"] >= 10.0


def test_critical_path_ignores_dangling_and_self_deps():
    steps = [
        {"id": 1, "est_sec": 10, "depends_on": [9, 1]},
        {"id": 2, "est_sec": 20, "depends_on": []},
    ]
    r = _critical_path_secs(steps)
    assert r["critical"] == 20.0
    assert r["parallel"] == [1, 2]


# --------------------------------------------------------------------------
# Rendered verdict: the wording must point at the right action
# --------------------------------------------------------------------------

def test_parallel_plan_that_fits_after_fanout_is_not_told_to_cut_scope():
    p = _plan()
    steps = [
        {"id": 1, "title": "probe", "est_sec": 120, "depends_on": [],
         "status": "pending"},
        {"id": 2, "title": "oracle", "est_sec": 300, "depends_on": [1],
         "status": "pending"},
        {"id": 3, "title": "impl", "est_sec": 400, "depends_on": [1],
         "status": "pending"},
        {"id": 4, "title": "verify", "est_sec": 200, "depends_on": [2, 3],
         "status": "pending"},
    ]
    text = _ledger(p, steps)
    # serial 1020 > usable 765, but critical path 720 fits -> no scope cut
    assert "关键路径" in text
    assert "砍范围" not in text
    assert "第2、3步" in text


def test_genuine_chain_is_named_as_such_with_the_bottleneck():
    p = _plan()
    steps = [
        {"id": 1, "title": "a", "est_sec": 300, "depends_on": [],
         "status": "pending"},
        {"id": 2, "title": "b", "est_sec": 600, "depends_on": [1],
         "status": "pending"},
        {"id": 3, "title": "c", "est_sec": 120, "depends_on": [2],
         "status": "pending"},
    ]
    text = _ledger(p, steps)
    # 1020s of work against a 900s wall: this one genuinely does not fit.
    assert "依赖链是真的" in text
    assert "第 2 步" in text  # the 600s step, not the tail


def test_scope_cut_wording_never_names_verification_as_the_cheap_win():
    p = _plan()
    steps = [
        {"id": 1, "title": "a", "est_sec": 900, "depends_on": [],
         "status": "pending"},
        {"id": 2, "title": "verify", "est_sec": 200, "depends_on": [1],
         "status": "pending"},
    ]
    text = _ledger(p, steps)
    assert "依赖链是真的" in text or "链是真的" in text


# --------------------------------------------------------------------------
# Held-out shapes (not the sample this was developed on)
# --------------------------------------------------------------------------

def test_critical_path_three_parallel_branches():
    s = [
        {"id": 1, "est_sec": 100, "depends_on": []},
        {"id": 2, "est_sec": 400, "depends_on": [1]},
        {"id": 3, "est_sec": 250, "depends_on": [1]},
        {"id": 4, "est_sec": 150, "depends_on": [1]},
        {"id": 5, "est_sec": 80, "depends_on": [2, 3, 4]},
    ]
    r = _critical_path_secs(s)
    assert r["serial"] == 980.0
    assert r["critical"] == 580.0        # 100 + 400 + 80
    assert r["gain"] == 400.0
    assert r["constraint"] == 2          # the longest branch, not the 80s tail
    assert sorted(r["parallel"]) == [2, 3, 4]


def test_critical_path_two_independent_roots():
    s = [
        {"id": 1, "est_sec": 50, "depends_on": []},
        {"id": 2, "est_sec": 500, "depends_on": []},
        {"id": 3, "est_sec": 50, "depends_on": [1]},
    ]
    r = _critical_path_secs(s)
    assert r["serial"] == 600.0
    assert r["critical"] == 500.0        # the lone 500s root dominates
    assert r["free"] == [1, 2]
