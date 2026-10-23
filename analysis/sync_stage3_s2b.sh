#!/usr/bin/env bash
# Sync the two changed harness files to the deployed stage3 tree, with md5
# guards in BOTH directions (repo must match what we expect; stage3 must end up
# byte-identical), plus an import check from the stage3 path.
set -u
REPO=/nfs/caozhou/fa_eval/FlagScale-Agent
STG=/nfs/caozhou/glm/agent_stage3
EXPECT_PROMPT=97a0fe570b0e7c7bd2fafb8fd5040569
EXPECT_PLAN=d5ee36598b63e168906607ee12890f8d

fail() { echo "FAIL: $*"; exit 1; }

for f in flagscale_agent/react/prompt.py flagscale_agent/react/tools/plan_create.py; do
  [ -f "$REPO/$f" ] || fail "missing repo file $f"
  [ -f "$STG/$f" ]  || fail "missing stage3 file $f"
done

got_p=$(md5sum "$REPO/flagscale_agent/react/prompt.py" | cut -d' ' -f1)
got_l=$(md5sum "$REPO/flagscale_agent/react/tools/plan_create.py" | cut -d' ' -f1)
[ "$got_p" = "$EXPECT_PROMPT" ] || fail "repo prompt.py md5 $got_p != $EXPECT_PROMPT"
[ "$got_l" = "$EXPECT_PLAN"   ] || fail "repo plan_create.py md5 $got_l != $EXPECT_PLAN"

# Refuse to deploy while a batch is running (mid-run deploy = mixed cohorts).
if pgrep -af "harbor|tb_adapter.*run" | grep -v pgrep > /dev/null 2>&1; then
  fail "a batch appears to be running; refusing mid-run deploy"
fi

cp -f "$REPO/flagscale_agent/react/prompt.py" "$STG/flagscale_agent/react/prompt.py" || fail "cp prompt"
cp -f "$REPO/flagscale_agent/react/tools/plan_create.py" "$STG/flagscale_agent/react/tools/plan_create.py" || fail "cp plan_create"

got_p2=$(md5sum "$STG/flagscale_agent/react/prompt.py" | cut -d' ' -f1)
got_l2=$(md5sum "$STG/flagscale_agent/react/tools/plan_create.py" | cut -d' ' -f1)
[ "$got_p2" = "$EXPECT_PROMPT" ] || fail "stage3 prompt.py md5 $got_p2 != repo"
[ "$got_l2" = "$EXPECT_PLAN"   ] || fail "stage3 plan_create.py md5 $got_l2 != repo"

echo "stage3 prompt.py     = $got_p2  (== repo)"
echo "stage3 plan_create.py= $got_l2  (== repo)"

# FAR END: import from the STAGE3 tree (not the repo) and assert the clauses are
# present in what the deployed agent will actually read.
cd "$STG" || fail "cd stage3"
python - <<'PY' || fail "stage3 far-end clause check"
from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC as S
from flagscale_agent.react.tools.plan_create import _FRAME_STEP_ACCEPTANCE as A
frame = " ".join(" ".join(A).lower().split())
s = " ".join(S.lower().split())
assert "acceptance-facet manifest" in frame, "facet clause missing in stage3 FRAME"
assert "acceptance-facet manifest" in s, "facet clause missing in stage3 STATIC prompt"
assert "reading ledger" in frame, "ledger clause missing in stage3 FRAME"
assert "reading ledger" in s, "ledger clause missing in stage3 STATIC prompt"
assert "fixed by the clause count" in frame, "anchoring relation missing"
assert "cover the top-2" in frame and "in the same run" in frame, "coverage action missing"
print("stage3 far-end clause check: PASS (all 6 anchors present in the deployed tree)")
PY
echo "SYNC OK"
