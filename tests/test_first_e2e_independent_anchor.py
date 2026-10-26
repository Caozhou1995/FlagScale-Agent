"""Regression: the 75% cold-consumer checkpoint must demand an INDEPENDENT anchor.

WHY (measured, 2026-10-27): a calibrated classifier over 2632 trials showed
that 72.7% of ALL losses are "artifact exists, runs, produces WRONG CONTENT"
(G_artifact_wrong 440 + A_timeout_agent/artifact_present_wrong_content 260),
while "never delivered" is only ~30%. The prior mechanism message asked the
agent to "RUN it and READ the result" — a check that CANNOT fail: an artifact
re-reading its own output agrees with itself wherever the shared idea is wrong.
The clearest measured instance (gpt2-codegolf): the winning trial built an
independent reference and diffed token-by-token; the losing trials ran their own
artifact, saw plausible output, and shipped.

So the message must name the discriminating question, and must accept an honest
confession of a self-only anchor instead of forcing the agent to phrase a
non-check as evidence.
"""

from flagscale_agent.react.guard.first_e2e_run import _MSG_75


def test_75_message_demands_independent_anchor():
    assert "INDEPENDENT" in _MSG_75
    assert "agrees with itself" in _MSG_75


def test_75_message_keeps_run_requirement():
    # The run requirement must survive: a message that only demands "independent
    # evidence" without requiring the artifact to actually be exercised loses the
    # run-late protection this rung was created for.
    assert "Run it NOW" in _MSG_75
    assert "COLD-CONSUMER" in _MSG_75


def test_75_message_accepts_honest_self_anchor_confession():
    # Forcing the agent to dress a self-check as independent evidence would make
    # the gate a phrasing exercise. The message must offer the honest exit.
    assert "looked plausible" in _MSG_75
    assert "self-check" in _MSG_75


def test_75_message_names_falsifiable_anchor_kinds():
    # Generic wording only: the anchors named must be classes of evidence, never
    # a task-specific artifact (no leakage of any task's answer). The
    # verbatim-named-value anchor covers the measured failure class "right
    # numbers under the wrong name" — a wrong name passes every numeric
    # comparison and every self-consistent run, so only an external anchor
    # (the task text's own words) can catch it.
    assert "reference artifact the task or" in _MSG_75
    assert "different route" in _MSG_75
    assert "never asserts" in _MSG_75
    assert "named values" in _MSG_75
    assert "verbatim" in _MSG_75
