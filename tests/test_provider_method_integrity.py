"""Structural integrity: every method the provider body references must exist.

Origin (measured 2026-11-26): an edit whose `old_string` started at the
PRECEDING method's `return True` and whose replacement omitted the `def` line
left a whole function body dangling AFTER that return. The file still compiled
(py_compile: OK) and all 114 behavioural tests stayed green, because nothing
executed the orphaned body -- but `_strip_all_thinking` had stopped being a
method, and `_degraded_kwargs` calls it on a live recovery path. A code path
that only runs when the gateway misbehaves would have raised AttributeError at
the exact moment it was needed.

So this test asserts the CLASS, not the instance: for each provider class, every
`self.<name>(` referenced in its source must be an attribute of the class.
"""
import re

import pytest

from flagscale_agent.react.providers.anthropic_provider import AnthropicProvider
from flagscale_agent.react.providers.openai_provider import OpenAIProvider


def _self_calls(cls):
    """Every `self.foo(` name in the class's own source, in definition order."""
    import inspect
    src = inspect.getsource(cls)
    names = set(re.findall(r"\bself\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", src))
    return src, names


@pytest.mark.parametrize("cls,floor,known", [
    (AnthropicProvider, 18, "_build_kwargs"),
    (OpenAIProvider, 6, "_build_kwargs"),
])
def test_every_self_call_resolves_to_a_real_method(cls, floor, known):
    src, names = _self_calls(cls)
    # Anti-vacuity, twice over: a count floor sized per class (the OpenAI
    # provider is deliberately small) AND a named method that must appear, so a
    # scanner that silently read nothing cannot pass by finding "no missing
    # names" in an empty set.
    assert len(names) >= floor, (
        f"scanner is not reading the class body (only {len(names)} self-calls "
        f"found, expected >= {floor}) -- an empty set would pass vacuously")
    assert known in names, (
        f"scanner did not find {known!r}; it is not parsing {cls.__name__}")
    missing = sorted(n for n in names if not hasattr(cls, n))
    assert not missing, (
        f"{cls.__name__} references self.<name>( ) that does not exist: {missing}. "
        "A function body was probably orphaned after a `return` by an edit whose "
        "replacement dropped the `def` line -- it compiles and behaves as dead "
        "code until the recovery path that needs it fires.")


@pytest.mark.parametrize("cls", [AnthropicProvider, OpenAIProvider])
def test_no_dead_code_after_a_bare_return(cls):
    """A triple-quoted string or statement block sitting after `return` is dead.

    Cheap, narrow detector: a line that is exactly `return <expr>` followed by a
    triple-quoted string at the same or deeper indent means a body lost its def.
    """
    import inspect
    src = inspect.getsource(cls)
    lines = src.splitlines()
    offenders = []
    for i, line in enumerate(lines):
        if re.match(r"\s*return\b[^(\[{'\"].*$", line) and "return" in line:
            indent = len(line) - len(line.lstrip())
            for j in range(i + 1, min(i + 3, len(lines))):
                nxt = lines[j]
                if not nxt.strip():
                    continue
                nind = len(nxt) - len(nxt.lstrip())
                if nind >= indent and nxt.lstrip().startswith(('"""', "'''")):
                    offenders.append((i + 1, nxt.strip()[:40]))
                break
    assert not offenders, (
        f"possible orphaned function body (docstring after return) at "
        f"{cls.__name__} source offsets {offenders}")


def test_the_recovery_path_that_needed_it_actually_works():
    """Direct positive check: the method that was missing, and its caller."""
    p = AnthropicProvider.__new__(AnthropicProvider)
    assert hasattr(p, "_strip_all_thinking")
    msgs = [
        {"role": "assistant",
         "content": [{"type": "thinking", "thinking": "x"}, {"type": "text", "text": "hi"}]},
        {"role": "user", "content": "q"},
    ]
    out = p._strip_all_thinking(msgs)
    assert out[0]["content"] == [{"type": "text", "text": "hi"}], out
    assert out[1] == msgs[1], "non-assistant turns must pass through untouched"
    # and the caller that exposed the bug end-to-end
    assert p._degraded_kwargs({"messages": msgs})["messages"][0]["content"] == [
        {"type": "text", "text": "hi"}]
