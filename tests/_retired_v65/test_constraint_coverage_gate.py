"""M10: per-constraint acceptance coverage.

The mechanism splits the work deliberately:

  * READING the task statement is SEMANTIC. Requirements are stated in free
    prose, so the judge (an LLM) enumerates them and decides which claim
    addresses which requirement. Nothing here pattern-matches wording -- a
    paraphrase of the same measurement still counts, and a differently-worded
    statement of the same requirement yields the same answer.
  * ADMISSIBILITY is MECHANICAL and stays in the guard: a claim only counts if
    it carries a command and an `observed` value that really occurs in the
    session's tool-output corpus. So entries cannot be fabricated.

Every judge interaction is faked here on purpose (these tests must not call a
provider). The fakes implement the same contract the real judge does, and the
provider-facing behaviour is verified separately against a live model.
"""
import pytest

from flagscale_agent.react.guard import verification as V


DNA = """Requirements:
- The annealed part of the primers should have a length between 15 and 45 nucleotides.
- Have a melting temperature between 58 and 72 degrees celsius.
- Each primer pair should have a melting temperature at most 5 degrees celsius apart.
"""

PROSE = """Retrieve the best model for the task and write the answer out.
Use your judgement.
"""

CORPUS_RAW = "marker\nREAL_OUT_A here\nREAL_OUT_B\nREAL_OUT_C\n"


class FakeJudge:
    """Stands in for Judge with the same public contract, no provider."""

    def __init__(self, constraints=None, coverage=None, boom=False):
        self._constraints = constraints or []
        self._coverage = coverage or {}
        self._boom = boom
        self.calls = []

    def constraints(self, task_text):
        if self._boom:
            raise RuntimeError("provider down")
        self.calls.append(("constraints", task_text))
        return list(self._constraints)

    def constraint_coverage(self, quotes, constraints, task_text=""):
        if self._boom:
            raise RuntimeError("provider down")
        self.calls.append(("coverage", list(quotes)))
        out = []
        for i, c in enumerate(constraints):
            for q in quotes:
                if q in self._coverage.get(c, ()):
                    out.append(i)
                    break
        return out


def _corpus():
    return V._normalize_for_grounding(CORPUS_RAW)


def _act(rows):
    return {"command": "c", "observed": "REAL_OUT_B", "constraints": rows}


REQS = [
    "The annealed part of the primers should have a length between 15 and 45 nucleotides.",
    "Have a melting temperature between 58 and 72 degrees celsius.",
    "Each primer pair should have a melting temperature at most 5 degrees celsius apart.",
]

ROWS_FULL = [
    {"quote": "len(anneal) in 15..45", "command": "c1", "observed": "REAL_OUT_A"},
    {"quote": "Tm in 58..72", "command": "c2", "observed": "REAL_OUT_B"},
    {"quote": "Tm delta under 5", "command": "c3", "observed": "REAL_OUT_C"},
]


class TestNoPatternMatching:
    """The census of requirements must never be derived from the text's shape."""

    def test_enumeration_comes_from_the_judge_not_from_the_text(self):
        j = FakeJudge(constraints=REQS)
        assert V._stated_constraints(DNA, j) == REQS

    def test_a_statement_with_no_numeric_shape_is_still_read(self):
        # This statement has no comparators at all; a shape rule returns [].
        # The semantic reader returns what it actually states.
        reqs = ["The answer must be written to /app/result.txt."]
        j = FakeJudge(constraints=reqs)
        assert V._stated_constraints(PROSE, j) == reqs

    def test_no_judge_means_no_obligations(self):
        assert V._stated_constraints(DNA, None) == []

    def test_judge_failure_does_not_manufacture_obligations(self):
        assert V._stated_constraints(DNA, FakeJudge(boom=True)) == []


class TestCoverage:
    def test_single_claim_covers_only_its_own_requirement(self):
        j = FakeJudge(constraints=REQS,
                      coverage={REQS[0]: ["len(anneal) in 15..45"]})
        missing = V._constraint_coverage_missing(
            _act(ROWS_FULL[:1]), DNA, _corpus(), judge=j)
        assert REQS[0] not in missing
        assert missing == REQS[1:]

    def test_paraphrased_claim_per_requirement_covers_everything(self):
        j = FakeJudge(constraints=REQS, coverage={
            REQS[0]: ["len(anneal) in 15..45"],
            REQS[1]: ["Tm in 58..72"],
            REQS[2]: ["Tm delta under 5"],
        })
        assert V._constraint_coverage_missing(
            _act(ROWS_FULL), DNA, _corpus(), judge=j) == []

    def test_claim_about_a_different_quantity_covers_nothing(self):
        j = FakeJudge(constraints=REQS, coverage={})
        rows = [{"quote": "optimized salt to 50 mM",
                 "command": "c", "observed": "REAL_OUT_A"}]
        assert V._constraint_coverage_missing(
            _act(rows), DNA, _corpus(), judge=j) == REQS


class TestAdmissibilityStaysMechanical:
    def test_fabricated_observation_is_dropped_before_the_judge_sees_it(self):
        j = FakeJudge(constraints=REQS, coverage={
            REQS[0]: ["len(anneal) in 15..45"],
            REQS[1]: ["Tm in 58..72"],
            REQS[2]: ["Tm delta under 5"],
        })
        rows = [
            {"quote": "len(anneal) in 15..45", "command": "c1",
             "observed": "NEVER_RAN_XYZ"},
            ROWS_FULL[1], ROWS_FULL[2],
        ]
        missing = V._constraint_coverage_missing(
            _act(rows), DNA, _corpus(), judge=j)
        assert REQS[0] in missing  # its only claim was never actually observed
        assert REQS[1] not in missing and REQS[2] not in missing
        # the fabricated quote was filtered out before the judge was consulted
        assert all(q != "len(anneal) in 15..45"
                   for kind, q in j.calls if kind == "coverage" for q in q)

    def test_row_missing_a_command_or_observation_is_not_a_claim(self):
        j = FakeJudge(constraints=REQS, coverage={REQS[0]: ["no command"]})
        rows = [{"quote": "no command", "observed": "REAL_OUT_A"}]
        assert V._constraint_coverage_missing(
            _act(rows), DNA, _corpus(), judge=j) == REQS

    def test_unavailable_corpus_fails_open(self):
        j = FakeJudge(constraints=REQS)
        assert V._constraint_coverage_missing(_act(ROWS_FULL), DNA, None,
                                             judge=j) == []


class TestGateInertness:
    def test_below_threshold_the_gate_does_not_apply(self):
        j = FakeJudge(constraints=REQS[:2])
        assert V._constraint_coverage_missing(_act([]), DNA, _corpus(),
                                             judge=j) == []

    def test_judge_failure_never_blocks(self):
        j = FakeJudge(constraints=REQS, boom=True)
        assert V._constraint_coverage_missing(_act([]), DNA, _corpus(),
                                              judge=j) == []

    def test_no_judge_never_blocks(self):
        assert V._constraint_coverage_missing(_act([]), DNA, _corpus(),
                                              judge=None) == []


class TestBlockMessage:
    def test_message_names_each_missing_requirement_and_a_copyable_shape(self):
        msg = V._completion_constraint_coverage_missing(REQS[1:])
        for r in REQS[1:]:
            assert r in msg
        for key in ("constraints", "quote", "command", "observed"):
            assert key in msg


# ── M12: the copy-paste starters in the message must themselves be valid ────

def test_message_carries_copy_paste_starters():
    """The missing-field message must contain complete, copyable examples.

    A description of the shape forces the model to compose the object; a full
    starter it can copy and edit removes that failure mode. This test pins the
    presence of all three starters so the guidance cannot silently regress.
    """
    from flagscale_agent.react.guard.verification import _DELIVERABLE_READING_MISSING
    assert "COPY-PASTE STARTERS" in _DELIVERABLE_READING_MISSING
    for marker in ('deliverable_reading={"deliverable": "none"}',
                   '"shape": "union"',
                   '"shape": "intersection"'):
        assert marker in _DELIVERABLE_READING_MISSING, marker


def test_copy_paste_starters_are_accepted_by_the_real_checker():
    """Each starter, filled in, must PASS the guard — guidance must not lie."""
    from flagscale_agent.react.guard import verification as V
    corpus = " ".join(
        "files differ\nPASS rescale\nnormalized ref observed\n"
        "normalized cand observed\n".split()).lower()
    starters = [
        {"deliverable": "none"},
        {"deliverable": "/app/out.txt", "grader_visible": False,
         "readings": [{"reading": "raw bytes", "source": "q",
                       "acceptance_test": "cmp a b",
                       "normalized_side": "neither"}],
         "hedge": {"shape": "union", "output": "x",
                   "evidence": {"command": "cmp a b",
                                "observed": "files differ"},
                   "extras_direction": "ignored"}},
        {"deliverable": "/app/out.npy", "grader_visible": False,
         "readings": [{"reading": "rescale", "source": "q1",
                       "acceptance_test": "c1", "normalized_side": "both"},
                      {"reading": "raw", "source": "q2",
                       "acceptance_test": "c2", "normalized_side": "candidate"}],
         "hedge": {"shape": "intersection", "output": "y",
                   "evidence": {"command": "c1 && c2",
                                "observed": "PASS rescale"},
                   "extras_direction": "penalized",
                   "direction_evidence": {
                       "reference": {"command": "n1",
                                     "observed": "normalized ref observed"},
                       "candidate": {"command": "n2",
                                     "observed": "normalized cand observed"}}}},
    ]
    for dr in starters:
        res = V._check_deliverable_reading_field(dr, task_text="", observed=corpus)
        assert res is None, f"starter rejected: {res.reason if res else ''}"
