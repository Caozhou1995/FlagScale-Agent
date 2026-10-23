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

"""Regression tests for the CONSUMER-DRIVEN SELF-TEST delivery clause.

Empirical basis (4/4 deep-read losing trials of the 2026-10-23 loss-18
cohort): every examined failure is a DELIVERY-CONTRACT regression, not a
capability gap — the agent verified its artifact under a self-built driver
(a muted console, a demo input shape, a local-path transport) while the
grader drove the SAME artifact through ITS OWN channel (re-running the
product and grepping stdout, its own input shape and verbatim strings,
per-item standard parsing, its own protocol/auth endpoint). The clause
teaches: enumerate every channel the consumer will drive, run the final
self-test through EACH one literally, and never ship a version whose
consumer-visible output was silenced for a smoke test.

Prompt-only guidance: pin the clause in SYSTEM_PROMPT_STATIC (a future
edit must not silently drop it) and assert it leaks no task-specific
grader detail (same hard rule as the preserve-by-default tests).
"""
import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def test_consumer_driven_selftest_clause_present():
    """The load-bearing behaviors must each be stated."""
    p = SYSTEM_PROMPT_STATIC
    low = p.lower()
    # 1. drive the artifact the way the consumer will, not a stand-in
    assert re.search(r"(channel|protocol|transport|driver)", low), \
        "consumer-driven: channel/protocol guidance missing"
    # 2. re-run the delivered artifact and inspect its consumer-visible output
    assert re.search(r"(re-?run|replay)[^.]{0,120}(deliver|artifact|script)", low), \
        "consumer-driven: re-run-and-inspect guidance missing"
    # 3. never ship a version with consumer-visible output silenced
    assert "silent" in low or "silenc" in low or "quiet" in low, \
        "consumer-driven: no-silencing guidance missing"


def test_consumer_driven_clause_is_generic_no_task_leak():
    """The clause is general posture: no task's file names, protocols,
    signal names, or grader expressions may appear."""
    p = SYSTEM_PROMPT_STATIC
    for bad in ("mcmc-sampling", "htpasswd", "receivepack", "sigint,",
                "mailbox", "expect script", "analysis.R"):
        assert bad not in p, f"consumer-driven clause leaked {bad!r}"


def test_consumer_driven_clause_sits_in_before_declaring_done_block():
    """Placement contract: the clause must live with the delivery-discipline
    clauses (the 'Before declaring done' block)."""
    p = SYSTEM_PROMPT_STATIC
    anchor = "PRESERVE BY DEFAULT"
    assert anchor in p
    idx_anchor = p.index(anchor)
    idx_next_block = p.index("Your own examples are not a held-out test set", idx_anchor)
    clause = p[idx_anchor:idx_next_block]
    assert "consumer" in clause.lower(), \
        "consumer-driven bullet not found in the Before-declaring-done block"
