"""Pin the MEANS-FIRST judgement: scriptable reasoning must go to a script, not to prose thinking."""
import importlib
import flagscale_agent.react.prompt as P


def _clause():
    importlib.reload(P)
    return P.SYSTEM_PROMPT_STATIC


def test_means_first_clause_present():
    c = _clause()
    assert "MEANS FIRST" in c, "MEANS FIRST judgement missing"


def test_means_first_names_the_mechanical_classes():
    c = _clause()
    for word in ("arithmetic", "bit layout", "offset derivation"):
        assert word in c, f"class {word!r} not named"


def test_means_first_says_write_a_script_not_hand_derive():
    c = _clause()
    assert "do NOT hand-derive it in thinking" in c
    assert "write a 10-line script" in c


def test_means_first_states_the_cost_asymmetry():
    c = _clause()
    assert "visible wrong answer costs one call" in c
    assert "invisible one costs the task" in c


def test_means_first_sits_inside_s3_test():
    """It must be attached to S3 TEST so it arrives where the method choice is made."""
    c = _clause()
    i3 = c.find("S3 TEST")
    im = c.find("MEANS FIRST")
    i4 = c.find("S4 READ")
    assert 0 < i3 < im < i4, f"S3={i3} MEANS={im} S4={i4}"


def test_prompt_still_within_budget():
    importlib.reload(P)
    assert P.prompt_len() <= P.PROMPT_BUDGET
