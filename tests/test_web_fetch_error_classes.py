"""§4.1 contract: web_fetch must CLASSIFY a failure, not just report one.

Thirty of 89 batch trials never called web_fetch at all, and the runs that did
call it burned real budget on a loop the tool made invisible: a proxy that
502s the CONNECT tunnel looks identical to a site refusing the request, so the
agent retried the same URL, then tried another domain, and again. The fix is
not "report the error better" prose — it is a distinct message per failure
class, because 5xx / 403 / DNS have OPPOSITE fixes.

Per the user's ruling on the plan's §4.1:
  * change 1 (error-class split)  — implemented
  * change 2 (alternate paths)    — REJECTED as written (we do not know the
    correct alternate endpoint); only the proxy-bypass FORM is hinted
  * change 3 (block after 2 tries) — REJECTED as written; instead the FIRST
    failure carries a repetition reminder, and nothing is ever blocked
  * change 4 (offline degradation) — implemented, keyed on DISTINCT hosts

Every test below injects the exception, so it runs with no network.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from flagscale_agent.react.tools.web_fetch import WebFetchTool


def _http_error(code: int) -> requests.HTTPError:
    resp = MagicMock()
    resp.status_code = code
    return requests.HTTPError(response=resp)


def _fetch(tool: WebFetchTool, url: str, exc: Exception) -> str:
    with patch("requests.get", side_effect=exc):
        return tool.execute(url=url)


# ── change 1: the three classes are told apart ──────────────────────────────


@pytest.mark.parametrize("code", [500, 502, 503, 504])
def test_gateway_5xx_is_reported_as_upstream_not_site_refusal(code: int) -> None:
    """5xx is the intermediary failing, NOT the site refusing us."""
    out = _fetch(WebFetchTool(), "https://data.rcsb.org/rest/v1/x", _http_error(code))
    assert out.startswith(f"[WEB_FETCH_UPSTREAM_{code}]")
    # The wrong reading this guard exists to prevent, in the agent's own words.
    assert "network is blocked" in out.lower()  # …only inside the "NOT" clause
    assert "NOT 'network is blocked'" in out


@pytest.mark.parametrize("code", [403, 404, 429])
def test_client_4xx_is_reported_as_site_refusal(code: int) -> None:
    """4xx means the path works and this request was refused."""
    out = _fetch(WebFetchTool(), "https://www.rcsb.org/fasta/entry/2H5Q", _http_error(code))
    assert out.startswith(f"[WEB_FETCH_HTTP_{code}]")
    assert "REACHABLE" in out
    assert "UPSTREAM" not in out, "a 4xx must not be classified as a proxy failure"


def test_proxy_refusal_is_its_own_class() -> None:
    out = _fetch(
        WebFetchTool(),
        "https://www.rcsb.org/fasta/entry/2H5Q",
        requests.exceptions.ProxyError("CONNECT tunnel failed, 502"),
    )
    assert out.startswith("[WEB_FETCH_PROXY_REFUSED]")


def test_dns_failure_is_its_own_class() -> None:
    out = _fetch(
        WebFetchTool(),
        "https://nope.invalid/x",
        requests.ConnectionError("Name or service not known"),
    )
    assert out.startswith("[WEB_FETCH_NETWORK_ERROR]")


# ── change 2: hint the proxy-bypass FORM, not a substitute endpoint ─────────


def test_gateway_failure_hints_bypass_form_without_inventing_an_endpoint() -> None:
    """The ruling: we do not know the right alternate path, so do not guess one.

    What we DO know is the form that turned a 502 into a 200 in the real run,
    so that is what the message carries.
    """
    out = _fetch(WebFetchTool(), "https://data.rcsb.org/rest/v1/x", _http_error(502))
    assert "proxies={}" in out
    assert "ProxyHandler({})" in out
    # No invented mirror / alternate-endpoint list: naming one would be a guess
    # presented with the authority of a tool hint.
    assert "rcsb.org/fasta" not in out
    assert "mirror endpoint:" not in out


def test_proxy_refusal_carries_the_bypass_form() -> None:
    out = _fetch(
        WebFetchTool(),
        "https://u.example/x",
        requests.exceptions.ProxyError("CONNECT tunnel 502"),
    )
    assert "ProxyHandler({})" in out


# ── change 3: remind on the FIRST failure; never block ─────────────────────


def test_first_failure_already_carries_the_reminder() -> None:
    """Per the ruling this is a nudge at attempt 1, not a block at attempt 2."""
    out = _fetch(WebFetchTool(), "https://u1.example/x", _http_error(502))
    assert "FAILED 1 time" in out
    assert "change one axis" in out


def test_second_failure_escalates_the_wording_but_still_returns() -> None:
    tool = WebFetchTool()
    first = _fetch(tool, "https://u1.example/x", _http_error(502))
    second = _fetch(tool, "https://u1.example/x", _http_error(502))
    assert "FAILED 1 time" in first
    assert "FAILED 2 times" in second
    # Never a refusal: both calls returned a message, neither raised, and the
    # second one is still a normal error body the agent may act on.
    assert second.startswith("[WEB_FETCH_UPSTREAM_502]")


def test_reminder_is_per_url_not_global() -> None:
    tool = WebFetchTool()
    _fetch(tool, "https://a.example/x", _http_error(502))
    other = _fetch(tool, "https://b.example/x", _http_error(502))
    assert "FAILED 1 time" in other, "a different URL starts at 1, not 2"


# ── change 4: distinct-host count triggers the offline-degradation nudge ───


def test_offline_nudge_appears_only_after_three_distinct_hosts() -> None:
    tool = WebFetchTool()
    outs = [
        _fetch(tool, f"https://h{i}.example/x", _http_error(502)) for i in range(3)
    ]
    assert "DISTINCT hosts" not in outs[0]
    assert "DISTINCT hosts" not in outs[1]
    assert "3 DISTINCT hosts" in outs[2]
    assert "NOT VERIFIED" in outs[2]


def test_offline_nudge_counts_hosts_not_repeated_paths() -> None:
    """Three URLs on ONE host is one dead host, not three — no nudge."""
    tool = WebFetchTool()
    outs = [
        _fetch(tool, f"https://same.example/p{i}", _http_error(502)) for i in range(3)
    ]
    assert all("DISTINCT hosts" not in o for o in outs)
    # Each of those is a DIFFERENT URL, so each is its own first failure; the
    # counter is per-URL by design (re-issuing the same URL is the empty loop
    # worth flagging, exploring a different path on a live host is not).
    assert all("FAILED 1 time" in o for o in outs)


def test_offline_nudge_does_not_claim_the_network_is_gone() -> None:
    """It must read as "treat it as unavailable for this task", not as a fact."""
    tool = WebFetchTool()
    for i in range(3):
        out = _fetch(tool, f"https://h{i}.example/x", _http_error(502))
    assert "Treat external reachability as unavailable" in out
    assert "is unavailable" not in out.replace("Treat external reachability as unavailable", "")
