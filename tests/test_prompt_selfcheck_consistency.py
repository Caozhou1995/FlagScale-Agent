

def test_prompt_states_quantifier_must_match_criterion():
    """Falsifier: a grader with an all-match criterion cannot be satisfied by an
    aggregate metric. Measured case: 0.9207 per-image accuracy => ~4/50 wrong =>
    50-of-50 all-match fails by construction. The prompt must tell the agent to
    report the criterion's own counting form, not an average."""
    from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC as S
    assert "Copy the CRITERION's QUANTIFIER" in S
    assert "50-of-50" in S
    assert "per-item pass/fail counts" in S
