#!/usr/bin/env python3
"""Leak probe + final clause read-back for Steps A+B.

Checks (1) no trial-specific text leaked into either new clause, (2) the clauses
are present verbatim in the STATIC prompt the agent actually receives,
(3) md5 of the two changed files.
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path("/nfs/caozhou/fa_eval/FlagScale-Agent")
sys.path.insert(0, str(ROOT))

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC as S
from flagscale_agent.react.tools.plan_create import _FRAME_STEP_ACCEPTANCE as A

frame = "\n".join(A)
full = frame + "\n" + S
# normalise whitespace: the acceptance items are wrapped Python literals, so a
# phrase can straddle a line break. The regression tests normalise too -- the
# probe must measure the same thing they do, or it judges a different string.
frame_norm = " ".join(frame.lower().split())
S_norm = " ".join(S.lower().split())
full_norm = frame_norm + "\n" + S_norm

BANNED = [
    # trial identifiers / task names
    "mcmc", "stan_model_sampling", "filter-js", "filter_js", "dna-assembly",
    "video-processing", "git-multibranch", "sam-cell-seg", "mailman",
    # trial-specific artifacts / numbers
    "BeautifulSoup", "html.parser", "refresh=0", "2793", "1800s",
    "stolen_A1", "randn(30",
]
hits = [b for b in BANNED if b in full]
print("leak_probe_hits:", hits if hits else "0 (clean)")

anchors = {
    "Acceptance-facet manifest": "acceptance-facet manifest" in frame_norm,
    "facet-in-static-prompt": "acceptance-facet manifest" in S_norm,
    "Reading ledger": "reading ledger" in frame_norm,
    "ledger-in-static-prompt": "reading ledger" in S_norm,
    "external anchoring relation": "fixed by the clause count" in frame_norm,
    "top-2 coverage action": ("cover the top-2" in frame_norm
                              and "in the same run" in frame_norm),
}
for k, v in anchors.items():
    print(f"anchor[{k}] = {v}")

for rel in ("flagscale_agent/react/prompt.py",
            "flagscale_agent/react/tools/plan_create.py"):
    p = ROOT / rel
    print(f"md5[{rel}] = {hashlib.md5(p.read_bytes()).hexdigest()[:8]}")

bad = hits or not all(anchors.values())
print("\nRESULT:", "FAIL" if bad else "PASS")
sys.exit(1 if bad else 0)
