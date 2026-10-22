"""Guard messages must not contain strings harbor classifies as provider API errors.

LIVE DEFECT (2026-10-22, full run): two trials were recorded as

    harbor.agents.installed.base.ApiRateLimitError: Command failed (exit -9)

with NO rate limit anywhere in the run. The only match for harbor's classifier
regex `rate.?limit` was this agent-visible guard text, echoed into the agent's
own output:

    "If the foreground wait is genuinely correct here (e.g. a rate-limit cooldown"

harbor's _classify_exec_error() greps the agent phase's stdout/stderr for
ERROR_PATTERNS and maps `rate.?limit` -> ApiRateLimitError. Because the guard
prints that phrase, an ordinary exit -9 (SIGKILL) got mislabelled as a provider
rate limit — which flips the exception_stats bucket, defeats any
`--retry-include ApiRateLimitError` policy, and makes a capability/infra failure
look like a quota problem.

The rule this pins: agent-visible harness text must not contain substrings that
the evaluation harness itself interprets as infrastructure errors.
"""
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUARD_DIR = os.path.join(REPO, "flagscale_agent", "react", "guard")

# The classifier patterns from harbor/agents/installed/base.py that a guard
# message could plausibly trip. Kept narrow and literal: these are the ones a
# prose guard would accidentally use.
HARBOR_ERROR_PATTERNS = [
    (r"rate.?limit", "ApiRateLimitError"),
    (r"too many requests", "ApiRateLimitError"),
    (r"specified API usage limits", "ApiUsageLimitError"),
    (r"You've hit your usage limit", "ApiUsageLimitError"),
    (r"SSL_ERROR_SYSCALL", "NetworkConnectionError"),
    (r"SSL_connect", "NetworkConnectionError"),
    (r"Could not resolve host", "NetworkConnectionError"),
    (r"Connection refused", "NetworkConnectionError"),
    (r"Connection timed out", "NetworkConnectionError"),
    (r"Request timed out", "NetworkConnectionError"),
]


def _guard_sources():
    for name in sorted(os.listdir(GUARD_DIR)):
        if name.endswith(".py"):
            yield name, open(os.path.join(GUARD_DIR, name), errors="replace").read()


def test_no_guard_message_trips_harbor_api_error_classifier():
    offenders = []
    for name, src in _guard_sources():
        for pat, exc in HARBOR_ERROR_PATTERNS:
            for m in re.finditer(pat, src, re.I):
                line = src[:m.start()].count("\n") + 1
                offenders.append(f"{name}:{line} matches {pat!r} -> {exc}")
    assert not offenders, (
        "agent-visible guard text would be misclassified by harbor as an "
        "infrastructure/API error:\n  " + "\n  ".join(offenders))


def test_the_known_offending_phrase_is_gone():
    """Pin the exact regression: the cooldown phrase that caused 2 mislabels."""
    src = open(os.path.join(GUARD_DIR, "longtimeshell.py"), errors="replace").read()
    assert "rate-limit cooldown" not in src
    assert not re.search(r"rate.?limit", src, re.I), \
        "longtimeshell.py must not contain any rate-limit phrasing"


def test_guard_still_explains_the_legitimate_override():
    """Removing the phrase must not remove the guidance it carried."""
    src = open(os.path.join(GUARD_DIR, "longtimeshell.py"), errors="replace").read()
    assert "_override_reason" in src
    assert "genuinely correct" in src
    assert "throttle" in src
