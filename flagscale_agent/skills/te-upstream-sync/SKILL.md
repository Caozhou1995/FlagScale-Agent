---
description: Sync TransformerEngine-FL fork with an upstream NVIDIA/TransformerEngine
  release. Version-agnostic (base/fork/target refs), evidence-gated workflow covering
  merge conflict resolution, plugin API audit, runtime patch preservation, build/submodule
  integration, CI/CD audit, multi-backend test matrix, and finalize-to-main. Fuses
  V2 (PR#103) decomposed skeleton with V1 (PR#67) hard-won vendor field experience.
name: te-upstream-sync
parameters:
  - name: base
    description: upstream ref the fork last synced from (e.g. release_v2.x)
    default: <BASE_REF>
  - name: fork
    description: current FlagScale fork HEAD (e.g. origin/main)
    default: <FORK_REF>
  - name: target
    description: upstream ref to sync to (e.g. release_v2.y or a tag)
    default: <TARGET_REF>
---

## What this skill does

Sync the FlagScale fork of TransformerEngine (TransformerEngine-FL) up to a new
upstream NVIDIA release while preserving all FlagScale multi-backend plugin work.
Three refs drive everything -- nothing is hardcoded to a specific version:

- `base` -- the upstream commit the fork last merged from (the common ancestor).
- `fork` -- current fork HEAD carrying FlagScale customizations.
- `target` -- the upstream commit/tag to sync up to.

The delta the fork must preserve = `base..fork`. The delta upstream added =
`base..target`. The job is to replay the fork delta onto target without dropping
FlagScale plugin backends or regressing upstream fixes.

## Critical Rules

1. **Version-agnostic** -- never hardcode a release number or date in commands or
   decisions. Everything is derived from `base`/`fork`/`target` refs.
2. **Probe, never assume paths** -- before referencing any file/dir in a command,
   confirm it exists with `git ls-tree <ref> -- <path>` or `ls`. (V1 shipped a
   cicd doc + validate script referencing `common/plugin/*.h`, `common/cuda_patches/`,
   `tests/unit/` that never existed. See `pitfall/skill/doc_vs_reality_drift`.)
3. **Real layout** -- FlagScale plugin work lives in `transformer_engine/plugin/`:
   - `core/ops.py` -- plugin base op dispatch (Python).
   - `core/backends/vendor/<name>/` -- per-vendor backend implementations.
   - `core/backends/flagos/` and `core/backends/reference/` -- FlagOS + reference impls.
   C++ bindings live in `transformer_engine/pytorch/csrc/`.
4. **Dynamic backend discovery** -- enumerate backends from
   `transformer_engine/plugin/core/backends/vendor/` at runtime. Never paste a
   fixed vendor list into logic; the set changes over releases.
5. **Evidence gate** -- every stage emits artifacts to a persistent dir under
   `/share/.../temp/te-upstream-<target>/` (NOT `/tmp`). A stage is "done" only
   when its artifact exists and is inspected.
6. **P0 needs approval** -- dropping/altering any FlagScale backend op, or any
   change that could change numerics, is P0: record in decisions ledger and get
   sign-off before merging.
7. **Blocked != pass** -- a backend test that cannot run (no hardware) is BLOCKED,
   never counted as passing. Only NVIDIA CUDA is available as local ground truth.
8. **One conflict class at a time** -- resolve, build, verify, then next. Never
   batch unverified conflict resolutions.
9. **Squash before PR** -- final sync is one clean commit onto a new branch.
10. **Re-parent onto fork HEAD** -- the squash commit's parent MUST be `{fork}` HEAD,
    not the upstream target. Use `git commit-tree -p {fork}` (Stage 8). Skipping this
    causes GitHub to find an ancient merge-base and report hundreds of false conflicts
    on the PR. See `pitfall/te_upstream/pr_conflict_reparent_fix`.

---

## Stage 0: Orientation and ref resolution

Resolve all three refs to commits and set the artifact dir. Never proceed until
`base`, `fork`, `target` all resolve.

```bash
REPO=$(git -C . rev-parse --show-toplevel)
for r in {base} {fork} {target}; do
  git -C "$REPO" rev-parse --verify "$r^{commit}" || { echo "unresolved ref: $r"; exit 1; }
done
ART=/share/$(whoami)/flagos/temp/te-upstream-{target}
mkdir -p "$ART"
```

Record resolved commits to memory (`fact/te_upstream/refs`). Confirm the working
tree is clean before touching anything.

## Stage 1: Classify the fork delta

Compute what the fork changed on top of `base` and bucket every touched path into
preserve / adapt / drop. This is the decision ledger that drives the whole sync.

```bash
git -C "$REPO" diff --name-status {base} {fork} > "$ART/fork-delta.txt"
# dynamic backend discovery -- do NOT hardcode vendor names
git -C "$REPO" ls-tree -r --name-only {fork} \
  -- transformer_engine/plugin/core/backends/vendor/ \
  | awk -F/ 'NF>5 && $6!="__init__.py"{print $6}' | sort -u > "$ART/backends.txt"
```

For each changed path emit a row to `$ART/decisions.tsv` with columns:
`path  bucket(preserve|adapt|drop)  reason  owner  test  status  evidence`.
Plugin dir changes are almost always **preserve**; upstream files the fork edited
are **adapt**; anything the fork added that upstream now provides natively is a
**drop** candidate (P0 -- needs approval).

## Stage 2: Integrate upstream, resolve conflicts

Merge (or rebase-replay) `target` and resolve conflicts one class at a time.

```bash
git -C "$REPO" checkout -b sync/upstream-{target} {fork}
git -C "$REPO" merge --no-commit --no-ff {target} 2>&1 | tee "$ART/merge.log"
git -C "$REPO" diff --name-only --diff-filter=U > "$ART/conflicts.txt"
```

Resolve conflict files grouped by area (build, csrc, pytorch modules, plugin). For
each resolution note the class + rationale in `$ART/conflict-notes.md`. Keep the
plugin dir intact; when upstream refactors a base class the plugin subclasses,
adapt the subclass rather than reverting upstream.

## Stage 3: Audit the plugin API surface

Verify the FlagScale plugin ops still line up with upstream C++ bindings after the
merge. Three sets must stay consistent:

- **bindings** -- `.def("...")` names in `transformer_engine/pytorch/csrc/`.
- **plugin base** -- `def <name>` in `transformer_engine/plugin/core/ops.py`.
- **registered ops** -- `op_name = "..."` under `core/backends/`.

```bash
git -C "$REPO" grep -hE '\.def\(\s*"[A-Za-z0-9_]+' {target} \
  -- transformer_engine/pytorch/csrc/ | grep -oE '"[A-Za-z0-9_]+' | tr -d '"' | sort -u > "$ART/bindings.txt"
git -C "$REPO" grep -hE '^\s*def\s+\w+' {fork} \
  -- transformer_engine/plugin/core/ops.py | grep -oE 'def\s+\w+' | awk '{print $2}' | sort -u > "$ART/plugin-base.txt"
```

Build a matrix (`$ART/api-matrix.tsv`): `symbol  binding  plugin_base  registered  cuda_impl  disposition`.
A binding present upstream but missing in plugin base after refactor = adapt task.
A registered op whose binding vanished upstream = P0 investigation.

**Also run the reverse check** -- new bindings added in `target` that are absent from
`vendor/cuda/cuda.py` are implementation gaps. Stage 2 conflict resolution preserves
`cuda.py` from `{fork}` (correct), but does not automatically add methods for new
`target` bindings. These gaps cause silent `AttributeError` at dispatch time, not at import.

```bash
# New bindings in target not present in cuda.py
comm -23 \
  <(grep -hoE '\.def\("([A-Za-z0-9_]+)"' \
      "$REPO"/transformer_engine/pytorch/csrc/extensions/pybind.cpp | \
      grep -oE '"[A-Za-z0-9_]+"' | tr -d '"' | sort -u) \
  <(grep -oE '^\s*def ([A-Za-z0-9_]+)' \
      "$REPO"/transformer_engine/plugin/core/backends/vendor/cuda/cuda.py | \
      awk '{print $2}' | sort -u) \
  > "$ART/cuda-impl-gaps.txt"
cat "$ART/cuda-impl-gaps.txt"   # every line here = missing method in cuda.py
```

For each gap: implement the method in `cuda.py` following the existing pattern
(`tex = self._get_tex(); return tex.<binding>(...)`). Also check for **new parameters**
on existing bindings by diffing `pybind.cpp` `{base}..{target}` -- a new `py::arg` on
an existing `.def()` must be added to the corresponding `cuda.py` method signature too.

## Stage 4: Preserve runtime patches

FlagScale carries runtime shims that adapt upstream code for non-NVIDIA backends
(e.g. lazy `transformer_engine_torch` alias resolution, staged imports so a missing
vendor extension doesn't break import). After merge, re-verify these still apply.

- Diff the fork's runtime-patch touchpoints (`git diff {base} {fork} --
  transformer_engine/pytorch/`) and confirm each still lands cleanly on target.
- Confirm `import transformer_engine` succeeds with no vendor extension built
  (CPU-only), proving the staged-import guard survived the merge.
- Any patch that no longer applies because upstream changed the target lines is an
  **adapt** row -- re-derive the shim against the new upstream code, do not force
  the old diff.

## Stage 5: Integrate build and submodules

Reconcile build system and the three submodules with target.

```bash
git -C "$REPO" diff {base} {target} -- .gitmodules setup.py build_tools/ > "$ART/build-delta.txt"
git -C "$REPO" submodule status > "$ART/submodule-status.txt"
```

Discover submodules dynamically -- do not hardcode the list (v2.17 added
`3rdparty/nccl` for EP support, earlier versions did not have it):

```bash
git -C "$REPO" submodule status | awk '{print $2}' | sort > "$ART/submodule-list.txt"
```

Update each to the commit target expects; record old->new SHAs in the ledger. A
cudnn-frontend or cutlass bump can change kernel availability -- flag for the test
matrix.

## Stage 6: Audit CI/CD (added in V2 -- V1 predates CI/CD)

Verify GitHub Actions still cover every discovered backend and contain no broken
local references or silent failure masks.

```bash
ls "$REPO/.github/workflows"/all_tests_*.yml
ls "$REPO/.github/configs"/*.yml
# broken local action references
git -C "$REPO" grep -hoE 'uses:\s*\./[^ #]+' HEAD -- .github/workflows/ | sort -u > "$ART/local-uses.txt"
# silent failure masks
git -C "$REPO" grep -nE '\|\|\s*true|continue-on-error:\s*true' HEAD -- .github/workflows/ > "$ART/failure-masks.txt"
```

Cross-check each backend in `$ART/backends.txt` against a matching workflow
(`all_tests_<backend>.yml`) and config (`.github/configs/<backend>.yml`). A backend
with a plugin dir but no workflow = coverage gap (record it). Any `|| true` /
`continue-on-error: true` that hides a real test = flag. This stage exists because
V1's helper docs drifted from reality; here we detect drift automatically.

## Stage 7: Run the multi-backend test matrix

Only NVIDIA CUDA runs locally as ground truth. Everything else is BLOCKED unless
the corresponding hardware/CI is reachable.

- CUDA unit + integration: run via the repo's own workflow scripts
  (`.github/workflows/te-plugin-tests.yml`, `all_tests_cuda.yml`), stream to
  `$ART/test-cuda.log`.
- For each non-CUDA backend, mark BLOCKED with the reason in the matrix; do not
  guess pass. If CI for that backend can be triggered, capture the run URL as
  evidence instead.
- Update `decisions.tsv` status column: pass / blocked / fail + evidence path.

## Stage 8: Finalize to main and open PR

Only when the ledger has no open P0 and CUDA is green.

**Step 1 — squash via commit-tree, parent MUST be `{fork}`**

Do NOT use `git add -A && git commit` directly. That produces a commit whose
parent is the upstream target (or a merge node), so GitHub's 3-way merge finds
an ancient merge-base and explodes into hundreds of false conflicts.
Instead, capture the final tree and re-parent explicitly onto `{fork}`:

```bash
git -C "$REPO" add -A
# Commit working state as a scratch commit, then re-parent
TREE=$(git -C "$REPO" rev-parse HEAD^{tree})
NEW=$(git -C "$REPO" commit-tree "$TREE" -p {fork} \
        -m "sync: TransformerEngine-FL {base}..{fork} onto upstream {target}")
git -C "$REPO" reset --hard "$NEW"
```

Verify the parent is correct before pushing:
```bash
git -C "$REPO" log --oneline -2   # HEAD parent should be {fork} HEAD
git -C "$REPO" merge-base HEAD {fork}   # must equal $(git rev-parse {fork})
```

**Step 2 — simulate GitHub's 3-way merge before pushing**

```bash
MERGE_BASE=$(git -C "$REPO" merge-base HEAD {fork})
# Expect zero conflict lines in the output:
git -C "$REPO" merge-tree "$MERGE_BASE" {fork} HEAD | grep -c '^<<<<<<<' \
  && echo "CONFLICTS FOUND -- fix before pushing" || echo "clean"
```

Only proceed if the conflict count is 0.

**Step 3 — push to personal fork, open PR to official repo**

The branch is pushed to the **personal fork** (e.g. `origin` = Caozhou1995/TransformerEngine-FL).
The PR target is the **official fork** (e.g. flagos-ai/TransformerEngine-FL:main).
Never push the sync branch directly to the official repo.

Because re-parenting rewrites history, a force push is required. Use
`--force-with-lease` (fetch first to update the tracking ref):

```bash
git -C "$REPO" fetch origin sync/upstream-{target}   # update tracking ref
git -C "$REPO" push origin sync/upstream-{target} --force-with-lease
```

Open the PR with: summary of upstream delta, list of preserved plugin
backends, adapt/drop decisions with rationale, test matrix (pass/blocked/fail), and
submodule SHA bumps. Attach the artifact dir contents as evidence.

---

## Appendix A: Vendor field traps (distilled from V1 / PR#67)

These are real breakages seen in past syncs. The backend set is discovered
dynamically (Stage 1); the traps below are what to look for per backend, not a
fixed roster. Confirm each backend still exists under
`transformer_engine/plugin/core/backends/vendor/` before applying its lore.

- **Op-count gap (seen on hygon)** -- a vendor backend may register fewer ops than
  the plugin base defines. After merge, diff registered ops vs `core/ops.py` base
  methods per backend; a shrinking count usually means an upstream op was renamed
  and the vendor register map wasn't updated. Treat as adapt, not drop.
- **Lazy tex module (seen on enflame)** -- some backends resolve
  `transformer_engine_torch` (the `tex` extension) lazily so import works without
  the compiled extension. Upstream refactors that eagerly import `tex` at module
  top-level break this. Preserve the lazy/staged import guard; re-derive it if
  upstream moved the import site.
- **AttentionParams field drift** -- upstream periodically adds/removes fields on
  attention param structs. A vendor attention impl passing positional args, or
  reading a removed field, silently breaks. After merge, diff the attention param
  definition `base..target` and reconcile every vendor `attention/` impl field by
  field.
- **No `*args`/`**kwargs` passthrough** -- FlagScale forbids swallowing args with
  `*args`/`**kwargs` in plugin op signatures, because it hides upstream signature
  changes. Keep explicit signatures so a mismatch surfaces as a TypeError at the
  boundary instead of silently mis-dispatching.

## Appendix B: Anti-patterns to avoid (why V1 needed a rewrite)

- **Do not hardcode a release number or date** anywhere in the skill or in
  generated commands. Derive from refs.
- **Do not reference paths you have not confirmed.** V1 shipped a `cicd-pipeline.md`
  and a `validate_plugin.sh` referencing `transformer_engine/common/plugin/*.h`,
  `common/cuda_patches/*.patch`, and `tests/unit/` -- none of which exist. Always
  `git ls-tree`/`ls` first (Rule 2). Stage 6 automates this drift check.
- **Do not count a blocked backend as a pass.** No hardware = BLOCKED with reason.
- **Do not squash without re-parenting.** `git commit-tree -p {fork}` is mandatory.
  If you use plain `git commit` after a merge, the squash commit's parent becomes
  the upstream target (or a merge node). GitHub then diffs from an ancient merge-base
  (~v0.1) and reports hundreds of false conflicts on the PR. The fix is to re-parent
  after the fact: `TREE=$(git rev-parse HEAD^{tree}); NEW=$(git commit-tree $TREE -p {fork} -m "..."); git reset --hard $NEW`.
  See `pitfall/te_upstream/pr_conflict_reparent_fix`.

## Artifact layout

All under `$ART = /share/.../temp/te-upstream-<target>/`:

```
fork-delta.txt         backends.txt          decisions.tsv
merge.log conflicts.txt         conflict-notes.md
bindings.txt           plugin-base.txt       api-matrix.tsv
build-delta.txt        submodule-status.txt
local-uses.txt         failure-masks.txt     test-cuda.log
```

The `decisions.tsv` ledger is the source of truth: no open P0 rows -> eligible to
finalize (Stage 8).