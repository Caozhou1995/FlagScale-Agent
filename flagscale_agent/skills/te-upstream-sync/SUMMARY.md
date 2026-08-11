# Te-Upstream-Sync -- Summary

Sync the FlagScale fork of TransformerEngine (TransformerEngine-FL) up to a new
upstream NVIDIA release while preserving all multi-backend plugin work. Version-agnostic
and evidence-gated. Fuses the V2 (PR#103) decomposed, reusable skeleton with V1
(PR#67) hard-won vendor field experience.

**Load when**: TransformerEngine-FL needs to merge a newer upstream
NVIDIA/TransformerEngine release, or when resolving the fork-vs-upstream delta for a
version bump.

**Three refs drive everything** (never hardcode a version):
- `base` -- upstream commit the fork last synced from (common ancestor)
- `fork` -- current fork HEAD carrying FlagScale customizations
- `target` -- upstream commit/tag to sync up to

Preserve delta = `base..fork`; upstream delta = `base..target`.

**Full pipeline**: Stage 0 orientation + ref resolution -- Stage 1 classify fork
delta (preserve/adapt/drop ledger, dynamic backend discovery) -- Stage 2 integrate
upstream + resolve conflicts one class at a time -- Stage 3 audit plugin API surface
(bindings vs plugin base vs registered ops matrix) -- Stage 4 preserve runtime
patches (lazy tex, staged import) -- Stage 5 integrate build + submodules
(googletest/cudnn-frontend/cutlass) -- Stage 6 audit CI/CD (backend coverage, broken
local uses, failure masks) -- Stage 7 multi-backend test matrix (CUDA ground truth,
others BLOCKED) -- Stage 8 finalize to main + squashed PR.

**Key principles**:
- Version-agnostic: derive everything from base/fork/target refs, never hardcode a
  release number or date
- Probe before referencing any path (`git ls-tree`/`ls`) -- V1 shipped docs pointing
  at nonexistent paths; Stage 6 automates drift detection
- Real layout: plugin work in `transformer_engine/plugin/` (core/ops.py, core/backends/
  {vendor,flagos,reference}); C++ bindings in `transformer_engine/pytorch/csrc/`
- Dynamic backend discovery from `core/backends/vendor/` -- never hardcode vendor list
- Evidence gate: artifacts to `/share/.../temp/te-upstream-<target>/`, never `/tmp`;
  a stage is done only when its artifact exists and is inspected
- P0 (drop/alter a backend op, any numeric change) needs approval before merge
- Blocked != pass: no hardware = BLOCKED with reason, never counted as passing
- Squash into one clean commit on a new branch; never push to main directly

**Vendor field traps** (Appendix A): op-count gaps (hygon), lazy tex module
(enflame), AttentionParams field drift, no `*args`/`**kwargs` passthrough.
