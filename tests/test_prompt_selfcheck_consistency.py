

def test_prompt_states_quantifier_must_match_criterion():
    """Falsifier: an all-match criterion cannot be satisfied by an aggregate metric.

    A measured 99% per-item accuracy still implies failures on an all-match check
    over many items, so reporting an average proves nothing. The prompt must tell
    the agent to report the criterion's own counting form — stated generically, with
    no reference to any particular task's item count.
    """
    from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC as S
    assert "Copy the CRITERION's QUANTIFIER" in S
    assert "all N of N" in S
    assert "no aggregate score can certify" in S
    # must NOT carry a concrete item count from a graded task
    for leaked in ("50-of-50", "0.9207", "4/50"):
        assert leaked not in S, f"item-count leak in prompt: {leaked}"
