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

"""Anti-decay tests for the §8 READ-FIRST memory clause.

Why this file exists (measured 2026-11-09): §8 stated only the WRITE half of the
memory doctrine — "write the moment you learn anything durable" — and said
nothing about WHEN to READ. The observed consequence was an agent that stopped
consulting memory at the start of a task and re-derived conclusions the store
already held (E3's exact failure), so the read half was added.

Presence-only assertions would lock a token shape and not the doctrine, which is
how the previous decay (test_prompt_static_clause_anti_decay.py's docstring)
slipped through. Each test below pins a RELATION the clause must keep: the
trigger, the query construction, the claim-not-answer grading, the miss rule,
and the cost asymmetry that justifies the habit. Softening any one of them —
"consider memory", "if convenient", dropping the cost argument — turns the
matching test red.

The clause is scoped by slice, never asserted over the whole prompt, because
other sections supply the same words ("memory" appears throughout §2/§9).
"""

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC, prompt_len, PROMPT_BUDGET


def _section8() -> str:
    """The §8 slice of the static prompt, whitespace-normalised, lowercased."""
    raw = SYSTEM_PROMPT_STATIC
    start = raw.index("════ 8.")
    end = raw.index("════ 9.")
    return " ".join(raw[start:end].split()).lower()


def test_read_half_exists_and_is_imperative():
    """The clause must COMMAND a read, not suggest one."""
    s = _section8()
    assert "read first" in s, "§8 lost its READ-FIRST clause"
    # Banned softenings — the shapes a 'be more mindful of memory' rewrite takes.
    for banned in ("consider reading memory", "if convenient, read",
                   "you may want to check memory", "feel free to search memory"):
        assert banned not in s, f"§8 read clause was softened to {banned!r}"


def test_read_trigger_names_both_the_start_and_the_boundary():
    """WHEN to read: once at the start, and again where the work changes KIND.

    Measured motivation: a single "before you start" rule decays into a single
    lookup; the losses were at boundaries (first deploy, first launch, first
    read of an unfamiliar spec), where the field — and therefore the relevant
    recorded death — changes.
    """
    s = _section8()
    assert "before the first substantive action" in s
    assert "wherever the kind of work changes" in s
    assert "first deploy" in s and "first launch" in s


def test_query_is_built_from_the_task_s_own_nouns():
    """HOW to query: nouns from the task, and a warning against title-only.

    A keyword of the task's title alone retrieves the harness's own bookkeeping
    rather than the domain's recorded deaths.
    """
    s = _section8()
    assert "memory_list(keyword=...)" in s
    assert "2-4 of the task's own nouns" in s
    assert "never the title alone" in s
    assert "memory_read in full" in s, "the read must be the full domain prefix, not a skim"


def test_a_hit_is_a_claim_not_an_answer():
    """GRADING: a recalled entry is re-verified against the live tree.

    Without this the clause reintroduces the opposite failure — trusting a stale
    entry over the artifact on disk, which Section 5 forbids.
    """
    s = _section8()
    assert "a hit is a claim" in s
    assert "re-verify against the live tree" in s
    assert "not an answer" in s


def test_a_miss_is_not_evidence_of_absence():
    """A zero-hit count is not proof nothing is recorded (the store truncates)."""
    s = _section8()
    assert "a miss is not absence" in s
    assert "truncate at the limit" in s
    assert "widen before concluding nothing is recorded" in s


def test_the_cost_asymmetry_is_stated():
    """The justification, not just the instruction — a rule with no reason decays."""
    s = _section8()
    assert "a recall costs seconds" in s
    assert "the failure it prevents costs the run" in s


def test_write_half_survives_the_rewrite():
    """The original write doctrine must not have been traded away for the read half."""
    s = _section8()
    assert "write the moment you learn anything durable" in s
    assert "persists on disk across evictions" in s
    assert "a wrong entry is worse than none" in s
    assert "step notes are append-only" in s


def test_prompt_still_fits_the_budget():
    """The read clause was paid for by compressing §8 prose, never by raising the cap."""
    n = prompt_len()
    assert n <= PROMPT_BUDGET, f"prompt is {n} chars, budget {PROMPT_BUDGET}"
    # The ratchet now lives in ONE place (prompt.PROMPT_BUDGET, which carries the
    # user-ruling history in its comment). Repeating the number here meant every
    # ruled raise broke this test for no reason and invited a second, divergent
    # copy of the value. Pin the property instead: a raise must be recorded, and
    # the prompt must still fit inside it.
    assert PROMPT_BUDGET >= 41_000, (
        "the budget is a user-ruled anti-decay ratchet: it may be raised by an "
        "explicit ruling, never quietly lowered"
    )
    assert n <= PROMPT_BUDGET
