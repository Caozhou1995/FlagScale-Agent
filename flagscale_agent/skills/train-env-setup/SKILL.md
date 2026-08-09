---
description: Set up FlagScale training environment on GPU servers. Install conda env,
  FlagScale, and all FL-customized dependencies. PyTorch installs via official whl
  matching the driver's max CUDA version. Megatron-LM-FL, TransformerEngine-FL, Apex,
  and Flash-Attention MUST ALL be built from source — pre-built whls are NOT acceptable
  because they may not match the system CUDA. Source builds guarantee binary compatibility
  with the actual hardware. Handles CUDA compatibility detection, multi-node deployment,
  and Docker image setup.
name: train-env-setup
---

<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# FlagScale Training Environment Setup

Set up a complete FlagScale training environment on a GPU server. All dependencies use FL-customized versions.

## Strategy

Environment setup is a constraint satisfaction problem. Collect ALL constraints first, solve for compatible versions, then install once.

### CRITICAL: Source-of-truth principle

**NEVER reference or inspect existing environments when determining what to install.** Existing environments (even `flagscale-train`, even on the same machine) may have different hardware, editable installs pointing to other workspaces, patched packages, or stale versions. They tell you NOTHING useful about what the CURRENT environment needs.

The ONLY valid sources of truth for dependency versions are:
1. FlagScale's own `requirements/*.txt`, `setup.py`, `setup.cfg`, `pyproject.toml`
2. The upstream repos of FL-customized dependencies: Megatron-LM-FL, TransformerEngine-FL
3. The actual hardware (driver version, GPU type) — queried fresh with nvidia-smi

**Do NOT run `pip list`, `conda list`, `pip show` in any existing environment.** Do NOT look at what another environment has installed. These are irrelevant and misleading.

### General rules

1. ALL installs go into the target conda environment — NEVER install into base or current environment. Use `conda run -n <env> pip install ...` for every pip command. To check dependency versions without installing, read setup.cfg/pyproject.toml from the source repo or use `pip index versions <pkg>`.
2. PyTorch installs via official whl — choose the CUDA tag that matches the system's actual CUDA version (from nvcc --version), NOT what FlagScale's train.txt specifies. Verify wheel availability with `pip index versions` first.
3. Megatron-LM-FL, TransformerEngine-FL, Apex, and Flash-Attention MUST ALL be built from source. Pre-built whls (including from FlagScale PyPI) are NOT acceptable — they are compiled against a specific CUDA version that may not match the system. Source builds are the ONLY way to guarantee binary compatibility with the actual hardware. Never install from generic PyPI (pypi.org) either — those packages are either wrong (apex) or missing FL customizations
4. Never modify dependency source code to work around errors — report to user
5. **After EVERY pip install, VERIFY the import works.** DO NOT assume a successful pip exit code means the package is usable. Immediately test: `python -c "import <package>; print(<package>.__version__)"`. For large packages (torch, flash-attn, apex), if `import` hangs >10s, the install is corrupt and must be redone. On NFS/shared storage, use `timeout 15 python -c "import <package>"` to catch hangs quickly without blocking the session.
6. **Auto-fetch FL dependencies**: When Megatron-LM-FL or TransformerEngine-FL source code is needed (for analysis, compilation, or debugging) and is not available under YOUR workspace root, clone it fresh into `{deps_dir}` — don't ask the user. NEVER search for or reuse copies from other users' directories. Repos: `https://github.com/flagos-ai/Megatron-LM-FL.git`, `https://github.com/flagos-ai/TransformerEngine-FL.git` (use `--recursive` for TE-FL)
7. **ALL FL-customized dependencies are MANDATORY.** Do NOT skip Megatron-LM-FL, TransformerEngine-FL, Apex, or Flash-Attention. These are not optional — FlagScale training will fail or produce incorrect results without them. If one is difficult to install, try the source build fallback. Only skip a dependency if the user explicitly requests it after being warned of the consequences.
8. **If the user asks to create a new environment, create a new environment.** Do not reuse an existing one, even if it appears to have the right packages. Existing environments may have editable installs pointing to other workspaces, patched packages, or stale versions. A fresh environment is the only way to guarantee a clean, reproducible baseline. If you believe reusing is genuinely better, explain why and ask — but do not silently substitute.
9. **NEVER copy packages between environments using `cp -r` from site-packages.** This bypasses pip's metadata tracking — pip won't know the package exists, so dependency resolution, upgrades, and uninstalls all break silently. Always install via `pip install` (from wheel, PyPI, or source build). If a prebuilt wheel isn't available, build from source — it takes longer but produces a properly registered package.
10. **Prefer shared storage for conda environments.** If the working directory is under a shared filesystem (e.g., `/share/`, `/mnt/share/`, `/mnt/cfs/`), create the conda environment with `--prefix <shared_path>/envs/<name>` instead of `-n <name>`. This ensures all nodes can access the same environment in multi-node training without duplication. Use `--prefix` for ALL subsequent `conda run` commands targeting this environment. Only use `-n` if no shared storage is available.
11. **Conda envs and pip packages MUST go on shared storage, not local paths.** Even if `/tmp` or local disk has more space or is faster, the conda environment prefix and pip install target MUST be on shared storage (e.g., `/share/.../envs/<name>`). The only exception is `TMPDIR` for pip's temporary build cache — that can point to local storage to speed up compilation, but the final installed packages must land in the shared prefix.

## Step 0: Determine Dependency Source Directory

**Before anything else, determine `deps_dir` — the directory for cloning and building source dependencies.**

1. If workspace-layout skill has been loaded and `workspace_root` is known (from memory or detection), set `deps_dir = <workspace_root>/code/deps/`. This ensures all nodes in multi-node training can access the same builds.
2. If shared storage is available but workspace_root is not yet set, detect it now (see workspace-layout Step 1) and use it.
3. Only if NO shared storage is available, fall back to a local path.

**Summary**: `deps_dir` is always on shared storage when available. Never hardcode `/opt/flagscale/deps` — this path is local to one node and invisible to others.

Record `deps_dir` in memory after determining it.

## Step 1: Constraint Collection (NO installs in this step)

Collect ALL version constraints before installing anything. Do NOT look at existing environments.

### 1a. Hardware constraint — driver → max CUDA

First, detect the accelerator type:

```bash
# Try NVIDIA first
nvidia-smi --query-gpu=driver_version,name,compute_cap,memory.total --format=csv,noheader | head -1 && echo "GPU_COUNT=$(nvidia-smi -L | wc -l)"
nvcc --version 2>/dev/null || echo "nvcc not found"

# If nvidia-smi fails, detect non-NVIDIA accelerators
# Huawei Ascend NPU:
npu-smi info 2>/dev/null || true
# Other vendors: add detection commands as needed
```

**Hardware type determination:**
- If `nvidia-smi` succeeds → **NVIDIA GPU platform**. Follow the standard CUDA-based flow below.
- If `nvidia-smi` fails but a non-NVIDIA accelerator is detected (e.g., `npu-smi info` succeeds) → **Non-NVIDIA accelerator platform**. Mark `IS_NON_NVIDIA=true`. The rest of the installation follows the same steps as NVIDIA, with one exception: TransformerEngine-FL must be built with `TE_FL_SKIP_CUDA=1` (see Step 4b).

The `GPU_COUNT=` line gives the exact GPU count. Use that number in all subsequent references — never count nvidia-smi output lines manually.

Driver → max CUDA version (for PyTorch wheel selection, NVIDIA platforms only):
- Driver 580.x → CUDA ≤ 13.0 → wheels: cu118, cu121, cu124, cu126, cu128, cu130
- Driver 570.x → CUDA ≤ 12.8 → wheels: cu118, cu121, cu124, cu126, cu128
- Driver 560.x → CUDA ≤ 12.6 → wheels: cu118, cu121, cu124, cu126
- Driver 550.x → CUDA ≤ 12.4 → wheels: cu118, cu121, cu124
- Driver 535.x → CUDA ≤ 12.4 → wheels: cu118, cu121, cu124
- Driver 530.x → CUDA ≤ 12.1 → wheels: cu118, cu121
- Driver 520.x → CUDA ≤ 11.8 → wheels: cu118

### CRITICAL: CUDA version alignment for source builds

PyTorch whl bundles its own CUDA runtime, so `torch+cu128` can run on a system with driver 535.x (CUDA 12.4 compatible). **However**, source-building Apex/TE-FL/Flash-Attention uses the system `nvcc` compiler. If system nvcc version ≠ PyTorch's CUDA version, builds will fail or produce incompatible binaries.

**MANDATORY resolution strategy:**

1. **PyTorch CUDA tag MUST match the system's actual CUDA version (from nvcc --version or /usr/local/cuda/version.txt), NOT what FlagScale's requirements/cuda/base.txt specifies.** FlagScale's requirements may be out of date relative to your hardware. First check if a PyTorch wheel exists for the system's exact CUDA major.minor version using `pip index versions torch --extra-index-url https://download.pytorch.org/whl/cu<MAJOR><MINOR>`. E.g., system has CUDA 13.0 → check cu130 → if `torch==2.9.0+cu130` exists, use it. This eliminates all version conflicts between bundled libcudart and system libcudart.

2. **If NO wheel exists for the exact system CUDA version**: Fall back to the closest lower version. E.g., system has CUDA 13.1 but only cu130 wheels exist → use cu130. This is safe because CUDA is forward-compatible.

3. **If system nvcc is missing or wrong version**: Install the CUDA toolkit that matches the chosen PyTorch CUDA tag. E.g., torch+cu124 → install CUDA 12.4 toolkit, then set `CUDA_HOME=/usr/local/cuda-12.4` for all source builds.

4. **If system nvcc version > PyTorch's CUDA version** (e.g., only nvcc 13.0 available but NO cu130 torch wheel exists, so torch is cu128): This is common on cutting-edge systems where the driver/toolkit is newer than what PyTorch supports. In this case, create an **nvcc version shim** — a wrapper that reports the torch-compatible version while using the real nvcc for compilation. CUDA is forward-compatible (nvcc 13.0 can compile code targeting CUDA 12.8 without issues). **However, prefer finding a cu130 wheel first — the shim is a last resort for when no matching wheel exists.**

   **Shim creation procedure:**
   ```bash
   # Determine torch's CUDA version
   TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)")  # e.g., "12.8"
   
   # Create shim directory
   SHIM_DIR={deps_dir}/cuda-${TORCH_CUDA}-shim
   mkdir -p $SHIM_DIR/bin
   
   # Create nvcc wrapper that reports matching version
   cat > $SHIM_DIR/bin/nvcc << EOF
   #!/bin/bash
   if [[ "\$*" == *"--version"* ]]; then
       echo "nvcc: NVIDIA (R) Cuda compiler driver"
       echo "Cuda compilation tools, release ${TORCH_CUDA}, V${TORCH_CUDA}.0"
       exit 0
   fi
   exec /usr/local/cuda/bin/nvcc "\$@"
   EOF
   chmod +x $SHIM_DIR/bin/nvcc
   
   # Symlink libraries and headers from real CUDA
   for dir in lib64 include targets; do
       ln -sf /usr/local/cuda/$dir $SHIM_DIR/$dir
   done
   
   # Use this for ALL source builds
   export CUDA_HOME=$SHIM_DIR
   ```
   
   **When to use the shim vs installing a matching toolkit:**
   - Shim (preferred): When nvcc is only 1 major version ahead (e.g., 13.0 vs 12.8). CUDA forward compatibility guarantees correct compilation.
   - Install matching toolkit: When the version gap is large, or when the shim approach produces linking errors (rare).
   - NEVER modify dependency source code (Apex/TE/Flash-Attention) to bypass version checks.

4. **ALL four FL dependencies (Megatron-LM-FL, TransformerEngine-FL, Apex, Flash-Attention) MUST be built from source.** Pre-built whls from FlagScale PyPI are NOT acceptable when driver/CUDA versions don't match FlagScale's default requirements — source builds guarantee binary compatibility with the actual hardware.

**Decision rule**: If `nvcc --version` reports a different major.minor than `torch.version.cuda`, you MUST resolve this BEFORE attempting any source build. Resolution options: (a) install matching CUDA toolkit, (b) create nvcc shim (if nvcc > torch CUDA), (c) choose a different PyTorch CUDA tag. Do NOT bypass version checks by modifying dependency source code.

**IMPORTANT: Use CONSISTENT CUDA_HOME for ALL source builds.** Once you determine the correct `CUDA_HOME` (real toolkit path or shim path), use it for ALL four dependency builds. Do NOT use different `CUDA_HOME` values for different packages — this causes ABI mismatches where one package's `.so` expects a different symbol signature than what torch provides. The most common symptom is `undefined symbol` errors on import (e.g., `_ZN3c104cuda...` with wrong argument types).

**Pre-build verification step** (run AFTER installing PyTorch, BEFORE building any extension):
```bash
NVCC_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+')
TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)")
echo "nvcc version: $NVCC_VER"
echo "torch CUDA:   $TORCH_CUDA"
if [ "$NVCC_VER" != "$TORCH_CUDA" ]; then
    echo "⚠️  MISMATCH DETECTED — must resolve before source builds"
    echo "Options: (1) install CUDA $TORCH_CUDA toolkit, (2) create nvcc shim if nvcc > torch"
fi
```

**When driver doesn't match FlagScale's default CUDA requirement** (e.g., FlagScale's train.txt specifies torch+cu128 but driver only supports cu124):
- Install PyTorch matching the DRIVER (e.g., torch+cu124), NOT what train.txt says
- Then source-build ALL four FL dependencies against that PyTorch version
- This is the ONLY reliable path — never force a higher CUDA version than the driver supports

### 1b. FlagScale framework constraint — read from source (REFERENCE ONLY for non-PyTorch deps)

Read FlagScale's own dependency declarations (NOT from any installed environment):

```bash
cat requirements.txt
cat requirements/cuda/train.txt
cat requirements/cuda/base.txt
cat setup.py
```

**CRITICAL: FlagScale's train.txt torch version is REFERENCE ONLY, NOT authoritative.**
FlagScale's `requirements/cuda/base.txt` may specify e.g. `torch==2.9.0` with `--extra-index-url .../whl/cu128`. **IGNORE this CUDA tag for PyTorch installation.** The actual PyTorch CUDA tag is determined by the system's installed CUDA version (nvcc --version). Steps:
1. Detect system CUDA: `nvcc --version` or `cat /usr/local/cuda/version.txt`
2. Check if wheel exists: `pip index versions torch --extra-index-url https://download.pytorch.org/whl/cu<MAJOR><MINOR>`
3. Use FlagScale's torch VERSION number (e.g., 2.9.0) but with the SYSTEM's CUDA tag (e.g., cu130 for CUDA 13.0)
4. E.g., system CUDA 13.0 + FlagScale specifies torch==2.9.0 → install `torch==2.9.0+cu130` from `https://download.pytorch.org/whl/cu130`

Use FlagScale requirements ONLY to determine:
- Python version requirement
- Non-PyTorch dependency versions (pydantic, hydra, etc.)
- Which FL-customized dependencies are needed (Megatron-LM-FL, TE-FL, etc.)

**IMPORTANT**: `requirements/cuda/train.txt` may contain `megatron_core @ https://...whl` or `transformer_engine @ https://...whl` URLs. These whl URLs point to the official FlagScale PyPI. **However, do NOT install from these whls** — they are compiled against a specific CUDA version that may not match your system. Instead, use the version information to identify the correct source branch/tag, then build from source. Do NOT install megatron_core or transformer_engine from generic PyPI (pypi.org) either.

Also fetch the setup configs of the two FL forks to check their torch/python requirements:

```bash
# Megatron-LM-FL: check setup.py for torch/python_requires
web_fetch https://raw.githubusercontent.com/flagos-ai/Megatron-LM-FL/main/setup.py
# TransformerEngine-FL: check setup.py for torch/python/minor version requirements
web_fetch https://raw.githubusercontent.com/flagos-ai/TransformerEngine-FL/main/setup.py
```

### 1c. FL-customized dependency analysis (as important as PyTorch itself)

FlagScale requires four FL-customized / special packages. ALL four are MANDATORY and ALL MUST be built from source:

| Package | Source | Install method |
|---------|--------|----------------|
| Megatron-LM-FL | flagos-ai GitHub | source build ONLY (`git clone` + `pip install --no-build-isolation .`) |
| TransformerEngine-FL | flagos-ai GitHub | source build ONLY (`git clone --recursive` + `NVTE_FRAMEWORK=pytorch pip install --no-build-isolation .`) |
| Apex | NVIDIA GitHub | source build ONLY (`APEX_CUDA_EXT=1 pip install --no-build-isolation .`) |
| Flash-Attention | Dao-AILab GitHub | source build ONLY (`--no-deps --no-build-isolation .`) |

**Why source build is mandatory**: Pre-built whls are compiled against a specific CUDA version. When the system driver/CUDA doesn't match FlagScale's default (which is common), pre-built whls produce silent runtime errors or segfaults. Source builds compile against the ACTUAL system CUDA toolkit, guaranteeing binary compatibility.

For each, analyze:
- **Megatron-LM-FL**: MUST build from source. Clone from GitHub and build with `pip install --no-build-isolation .`. Never use pre-built whls from FlagScale PyPI — they may not match the system CUDA.
- **TransformerEngine-FL**: MUST build from source. Requires `--recursive` clone for submodules. Build with `NVTE_FRAMEWORK=pytorch pip install --no-build-isolation .`. On non-NVIDIA accelerator platforms, prepend `TE_FL_SKIP_CUDA=1` to skip CUDA kernel compilation.
- **Apex**: MUST build from source. Must compile with `APEX_CUDA_EXT=1` matching PyTorch's CUDA version. Check that the nvcc toolkit version matches torch.version.cuda (not just driver CUDA version).
- **Flash-Attention**: MUST build from source. The version must match the installed PyTorch version. Use `--no-deps` to prevent pip from upgrading PyTorch. Check: GPU compute capability ≥ 8.0 required for flash-attn v2.x.

### 1d. Solve — write the FULL compatibility table

Write a COMPLETE compatibility table covering ALL components. Do NOT skip to Step 2 until this table is written and verified.

**CRITICAL: Determine PyTorch version BEFORE writing the table**

Do NOT write "torch_ver+cuXXX" as a placeholder. You MUST determine the EXACT PyTorch version that exists for the driver's max CUDA tag:

1. From Step 1a, you know the driver's max CUDA (e.g., Driver 535.x → max cu124)
2. From Step 1b, you know the FlagScale-preferred torch version (e.g., `torch==2.9.0` from requirements)
3. **Query PyPI to check if FlagScale's version has a wheel for your CUDA tag**:
   ```bash
   pip install torch==<flagscale_version>+<your_cu_tag> --extra-index-url https://download.pytorch.org/whl/<your_cu_tag> --dry-run 2>&1 | head -10
   ```
4. **Decision logic**:
   - If `torch==<flagscale_version>+<your_cu_tag>` EXISTS → use it (ideal: matches both FlagScale and driver)
   - If it does NOT exist → find the LATEST torch version that HAS a `+<your_cu_tag>` wheel from the available versions list in the error output
5. Write the EXACT version in the table (e.g., `torch==2.6.0+cu124`)

**KEY INSIGHT**: PyTorch does NOT ship all CUDA tags for every version. Newer CUDA tags (cu126, cu128) are added as older ones (cu118, cu121, cu124) are dropped. Always verify the exact combination exists before attempting install. The `pip install --dry-run` error message conveniently lists all available versions with their tags — use that list to find the latest compatible version.

**Example decision flow**:
- Driver 535.x → max CUDA 12.4 → need cu124 wheels
- FlagScale train.txt says `torch==2.9.0+cu128` → try `torch==2.9.0+cu124`
- Query PyPI: `pip install torch==2.9.0+cu124 --dry-run` → NOT FOUND (2.9.0 only ships cu126/cu128)
- From error output, find latest version with cu124: `2.6.0+cu124`
- Write in table: `torch==2.6.0+cu124`

**Example when FlagScale version IS available**:
- Driver 570.x → max CUDA 12.8 → need cu128 wheels
- FlagScale train.txt says `torch==2.9.0+cu128` → try `torch==2.9.0+cu128`
- Query PyPI: EXISTS
- Write in table: `torch==2.9.0+cu128`

This eliminates trial-and-error in Step 3a — you install exactly what you determined here.

```
COMPATIBILITY ANALYSIS TABLE
============================
Hardware: N×GPU_TYPE, Driver DRI_VER → max CUDA CUDA_MAX
FlagScale requirements:
  Python: py_req
  PyTorch: torch_req (from requirements, may differ from what we'll install)
  CUDA toolkit required: cuda_toolkit_needed

| # | Component | Required Version | Install Method | Notes |
|---|-----------|-----------------|---------------|-------|
| 1 | Conda env | python=py_ver | conda create --prefix | path: <shared>/envs/env_name (or -n if no shared storage) |
| 2 | PyTorch | torch==X.Y.Z+cuXXX | pip (whl) | EXACT version from PyPI query; cuXXX matches driver's max CUDA; --extra-index-url https://download.pytorch.org/whl/cuXXX |
| 3 | FlagScale | editable | pip -e ".[cuda-train]" | from project root, --no-deps |
| 4 | Megatron-LM-FL | latest | SOURCE BUILD | git clone + pip install --no-build-isolation . |
| 5 | TransformerEngine-FL | latest | SOURCE BUILD | git clone --recursive + NVTE_FRAMEWORK=pytorch pip install --no-build-isolation . (add `TE_FL_SKIP_CUDA=1` on non-NVIDIA platforms) |
| 6 | Apex | master | SOURCE BUILD | git clone NVIDIA/apex + APEX_CUDA_EXT=1 |
| 7 | Flash-Attention | fa_ver | SOURCE BUILD | --no-deps --no-build-isolation to protect PyTorch |
```

CRITICAL CHECKLIST before proceeding:
- [ ] PyTorch version is EXACT (e.g., `torch==2.6.0+cu124`), NOT a placeholder
- [ ] PyTorch CUDA tag matches driver's max supported CUDA (NOT FlagScale's default if they differ)
- [ ] PyTorch version was verified to exist on PyPI (via `pip index versions torch`)
- [ ] All versions in the table are derived from FlagScale source files (NOT existing envs)
- [ ] Shared storage checked — conda env path uses --prefix on shared FS if available
- [ ] CUDA toolkit/nvcc version alignment resolved — one of: (a) nvcc matches torch CUDA, (b) shim created for nvcc > torch, (c) matching toolkit to be installed
- [ ] `CUDA_HOME` path determined and will be used consistently for ALL source builds
- [ ] GPU compute capability ≥ required by flash-attn
- [ ] Build-time dependencies noted: pybind11, cmake, ninja, packaging
- [ ] Megatron-LM-FL will be built from source (NO whl, NO PyPI)
- [ ] TransformerEngine-FL will be built from source (NO whl, NO PyPI); on non-NVIDIA platforms, uses `TE_FL_SKIP_CUDA=1`
- [ ] Apex build flags include APEX_CUDA_EXT=1 (environment variable, NOT --global-option)
- [ ] Flash-attn install uses --no-deps

Present the table and ASK FOR CONFIRMATION. Do NOT proceed to Step 2 until the user confirms.
After confirmation, annotate your response with [ENV_COMPAT_ANALYZED].

## Step 2: Conda Environment

### 2a. Check shared storage FIRST

**CRITICAL**: If the current working directory is under a shared filesystem (e.g., `/share/`, `/mnt/share/`, `/mnt/cfs/`), create the conda environment on the shared storage — NOT on the local node. This ensures all nodes in multi-node training can access the same environment without duplication.

```bash
# Check if we're on shared storage
df -h . | grep -E '^[^/]' | head -5

# Check available shared mount points
ls -d /share /mnt/share /mnt/cfs /mnt/dfs 2>/dev/null
```

If shared storage is found (e.g., `/share/project/...`), use `--prefix` instead of `--name`:

```bash
# Create env in shared storage — use --prefix with full path
conda create --prefix /share/project/<path>/envs/{env_name} python={python_version} -y

# For all subsequent commands, use --prefix (not -n):
conda run --prefix /share/project/<path>/envs/{env_name} <command>
```

If NO shared storage is found, fall back to `-n`:

```bash
conda create -n {env_name} python={python_version} -y
# In non-interactive shells (agent), use: conda run -n {env_name} <command>
# In interactive shells (user), use: conda activate {env_name}
```

### 2b. Verify

```bash
python --version
```

## Step 3: Install FlagScale

### 3a. Pin PyTorch FIRST (before installing FlagScale)

**PyTorch installs via official whl** — PyTorch has excellent coverage of CUDA versions (cu118, cu121, cu124, cu126, cu128), so a pre-built wheel is always available. No source build needed.

**CRITICAL**: Use the EXACT version determined in Step 1d. Do NOT re-derive or guess the version here. The version was already verified to exist on PyPI during Step 1d.

**CRITICAL**: The PyTorch CUDA tag is determined by the DRIVER, not by FlagScale. If FlagScale's train.txt says `torch==2.9.0+cu128` but your driver only supports cu124, you must use the latest torch that has a cu124 wheel (e.g., `torch==2.6.0+cu124`). NEVER try to install a torch version+tag combination that doesn't exist on PyPI — always verify with `--dry-run` first.

**CRITICAL**: `pip install -e ".[cuda-train]"` will pull in ALL requirements, including PyTorch from the requirements files. If those requirements specify a different CUDA version than what your driver supports, pip will silently upgrade PyTorch and all CUDA libraries. This is the #1 cause of wasted time in environment setup.

**Always pin PyTorch before FlagScale install:**

```bash
# Install exact PyTorch version determined in Step 1d — cu_tag matches driver's max CUDA
pip install torch=={exact_version_from_step_1d} torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/{cu_tag}
# Verify CUDA version is correct
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

### 3b. Clone and Install FlagScale

FlagScale-Agent is an independent repository. FlagScale itself must be cloned separately.

**CRITICAL: Workspace Isolation** — If FlagScale source code is not already present under YOUR workspace root (`{workspace_root}/code/FlagScale`), clone it fresh. Do NOT search for or reuse copies from other users' directories. Other copies may be at different versions, have local patches, or have editable installs that would break if you modify them. Always clone your own copy.

```bash
# Clone FlagScale into YOUR workspace code directory
mkdir -p {workspace_root}/code
git clone --depth 1 https://github.com/FlagOpen/FlagScale.git {workspace_root}/code/FlagScale
cd {workspace_root}/code/FlagScale
```

Then install in editable mode:

```bash
pip install -e ".[cuda-train]"
```

**If pip tries to upgrade PyTorch during this step**, abort and use the two-phase approach:
```bash
# Phase 1: install FlagScale without deps
pip install --no-deps -e .
# Phase 2: install remaining deps from requirements (PyTorch already pinned, won't change)
pip install -r requirements/cuda/train.txt
```

This ensures PyTorch stays at the pinned version.

Verify:
```bash
flagscale --help
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
# CRITICAL: confirm torch version did NOT change from what was installed in 3a
```

**Important**: `requirements/cuda/train.txt` includes a megatron-core whl from the FlagScale PyPI. Do NOT use this whl — it is compiled against a specific CUDA version that may not match your system. Always build Megatron-LM-FL from source in Step 4. If `pip install -e ".[cuda-train]"` installs the whl version, it will be overwritten by the source build in Step 4.

## Step 4: FL-Customized Dependencies (ALL SOURCE BUILDS)

These are FlagScale's customized forks. ALL MUST be built from source — pre-built whls are NOT acceptable. Install order matters — Megatron-LM-FL first, then the rest.

### 4a. Megatron-LM-FL (MANDATORY source build)

**Always build from source** — pre-built whls are NOT acceptable regardless of source.

```bash
mkdir -p {deps_dir}
git clone https://github.com/flagos-ai/Megatron-LM-FL.git {deps_dir}/Megatron-LM-FL
cd {deps_dir}/Megatron-LM-FL
pip install --no-build-isolation . -v
```

Verify:
```bash
python -c "from megatron.plugin.platform import get_platform; print('OK:', get_platform())"
```

### 4b. TransformerEngine-FL (MANDATORY source build)

**Always build from source** — pre-built whls are NOT acceptable regardless of source.

**Build-time dependencies** (install these FIRST if not already present):
```bash
pip install pybind11 cmake ninja
```

**CRITICAL: Use the same `CUDA_HOME` determined in Step 1a for ALL builds.** If you created an nvcc shim, set `CUDA_HOME` to the shim directory. If you installed a matching CUDA toolkit, set it to that path. Inconsistent `CUDA_HOME` between TE-FL and other builds causes ABI mismatches (undefined symbol errors on import).

**For NVIDIA GPU platforms:**
```bash
pip install nvidia-mathdx --extra-index-url https://pypi.nvidia.com
git clone --recursive https://github.com/flagos-ai/TransformerEngine-FL.git {deps_dir}/TransformerEngine-FL
cd {deps_dir}/TransformerEngine-FL
SM_ARCH=$(python -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}.{cc[1]}')")
CUDA_HOME=$CUDA_HOME TORCH_CUDA_ARCH_LIST="$SM_ARCH" NVTE_FRAMEWORK=pytorch pip install --no-build-isolation . -v
```

**For non-NVIDIA accelerator platforms (e.g., Huawei Ascend NPU):**
```bash
git clone --recursive https://github.com/flagos-ai/TransformerEngine-FL.git {deps_dir}/TransformerEngine-FL
cd {deps_dir}/TransformerEngine-FL
TE_FL_SKIP_CUDA=1 NVTE_FRAMEWORK=pytorch pip install --no-build-isolation . -v
```

The `TE_FL_SKIP_CUDA=1` flag tells the build system to skip CUDA-specific compilation (cuDNN, cuBLAS kernels, etc.) that would fail on non-NVIDIA hardware. The Python-level TE interfaces remain available for the vendor's backend to hook into.

Note: Source build takes 10-30 minutes. Do NOT interrupt or ask for confirmation during compilation — just wait for it to finish.

Verify:
```bash
python -c "import transformer_engine; print('TE version:', transformer_engine.__version__)"
```

### 4c. NVIDIA Apex (source build)

**WARNING: The PyPI package named `apex` is a Pyramid web framework — NOT NVIDIA Apex.** Never run `pip install apex` from PyPI. NVIDIA Apex must always be built from source.

**Build-time dependencies** (install these FIRST if not already present):
```bash
pip install packaging
```

**IMPORTANT: `--global-option` is REMOVED in pip 25+.** The old pattern `pip install --global-option="--cpp_ext" --global-option="--cuda_ext"` no longer works. Apex now uses environment variables instead: `APEX_CPP_EXT=1 APEX_CUDA_EXT=1`. Always use environment variables, never `--global-option`.

```bash
git clone --depth 1 https://github.com/NVIDIA/apex.git {deps_dir}/apex
cd {deps_dir}/apex

# Detect current GPU compute capability — only compile for this architecture
SM_ARCH=$(python -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}.{cc[1]}')")

CUDA_HOME=$CUDA_HOME TORCH_CUDA_ARCH_LIST="$SM_ARCH" NVCC_APPEND_FLAGS='--threads 4' \
    APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 \
    pip install --no-build-isolation --no-deps -e . -v
```

Verify:
```bash
python -c "from apex.optimizers import FusedAdam; from apex.multi_tensor_apply import multi_tensor_applier; print('Apex CUDA extensions OK')"
```

**Common issue**: CUDA version mismatch between system nvcc and PyTorch's CUDA. If Apex build fails with version check error, go back to Step 1a "CUDA version alignment for source builds" and resolve the mismatch (likely by creating an nvcc shim or installing matching CUDA toolkit). Do NOT modify Apex source code to bypass the check.

**Two version checks that can fail:**
1. **Apex's own `setup.py` check** (`check_cuda_torch_binary_vs_bare_metal`): Compares nvcc version against `torch.version.cuda`. Triggered when nvcc major.minor ≠ torch CUDA.
2. **PyTorch's `cpp_extension.py` check** (`_check_cuda_version`): Compares `CUDA_HOME/bin/nvcc --version` against `torch.version.cuda`. Fails if major version differs.

Both checks are resolved by ensuring `CUDA_HOME` points to an nvcc that reports the correct version (either a real matching toolkit, or the shim from Step 1a).

**IMPORTANT: Pure-Python vs CUDA Extensions**

Apex has two install modes:
- **Full install** (with `APEX_CUDA_EXT=1`): Compiles CUDA extensions for fused kernels. Required for `gradient_accumulation_fusion`, fused Adam, fused layer norm, etc.
- **Pure-Python install** (without CUDA flags or `pip install apex`): Only Python wrappers, NO fused kernels. Many Megatron features silently fall back to slower paths or fail with `RuntimeError: ... requires APEX CUDA extensions`.

**If you see `gradient_accumulation_fusion requires APEX CUDA extensions`**: Apex was installed in pure-Python mode. You must either:
1. Reinstall with CUDA extensions (recommended): use the build command above with `APEX_CUDA_EXT=1`
2. OR disable ALL fusion flags at once: `gradient_accumulation_fusion: false`, `bias_gelu_fusion: false`, `bias_swiglu_fusion: false` — and note the performance impact

Never disable just one fusion flag — if APEX CUDA extensions are missing, ALL fused kernels are unavailable.

### 4d. Flash-Attention 2

**CRITICAL**: Always use `--no-deps` when installing flash-attn. Without it, pip may upgrade PyTorch to an incompatible version, causing cascading failures (triton mismatch, CUDA version conflicts). The PyTorch version was already pinned in Step 3 — do not let flash-attn override it.

**CRITICAL**: Only compile for the current GPU's SM architecture. Flash-attn defaults to compiling ALL supported architectures (sm_80, sm_86, sm_89, sm_90, ...), which takes 30-60 minutes and is completely unnecessary — you only need the architecture of the GPUs on this machine. Set `TORCH_CUDA_ARCH_LIST` to the detected compute capability. This reduces compile time to 5-10 minutes.

**Build-time dependencies** (install these FIRST if not already present):
```bash
pip install packaging ninja
```

```bash
git clone --branch v2.8.1 --depth 1 https://github.com/Dao-AILab/flash-attention.git {deps_dir}/flash-attention
cd {deps_dir}/flash-attention

# Detect current GPU compute capability — ONLY compile for this architecture
SM_ARCH=$(python -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}.{cc[1]}')")
echo "Building flash-attn for SM $SM_ARCH only (skipping other architectures)"

CUDA_HOME=$CUDA_HOME TORCH_CUDA_ARCH_LIST="$SM_ARCH" FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 \
    pip install --no-build-isolation --no-deps . -v
```

**CUDA toolkit vs driver version**: Flash-attn compilation requires the CUDA **toolkit** version (nvcc) to match PyTorch's CUDA version, NOT the driver version. Check with `nvcc --version` (toolkit) vs `nvidia-smi` (driver). If nvcc is missing or wrong version, install the matching CUDA toolkit or use the nvcc shim from Step 1a.

After installing, verify PyTorch was NOT changed:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```
If the version differs from what was installed in Step 3, flash-attn broke the environment. Uninstall flash-attn, reinstall the correct PyTorch, and retry with `--no-deps`.

Verify:
```bash
python -c "import flash_attn; print('Flash-Attention version:', flash_attn.__version__)"
```

### 4e. Troubleshooting: Rebuild after ABI mismatch

If `import` of a CUDA extension fails with `undefined symbol` errors (e.g., `_ZN3c104cuda...`), the most likely cause is ABI mismatch — the extension was compiled against different C++ headers than the runtime torch library provides.

**Diagnosis:**
```bash
# Check the missing symbol
python -c "import transformer_engine" 2>&1 | grep "undefined symbol"
# Look up whether torch provides it with the same signature
nm -D $(python -c "import torch; print(torch.__file__.replace('__init__.py', 'lib/libc10_cuda.so'))") | grep "<symbol_name>"
```

If the symbol exists but with a different mangled name (e.g., `...EiPKcS2_ib` vs `...EiPKcS2_jb`), the argument types differ between compile-time headers and runtime library. This means the extension was compiled with wrong CUDA_HOME.

**Resolution:**
```bash
cd {deps_dir}/<package>
# 1. Clean ALL build artifacts
rm -f *.so
rm -rf build/ *.egg-info dist/
# 2. Rebuild with correct CUDA_HOME (must match torch.version.cuda)
CUDA_HOME=<correct_path> TORCH_CUDA_ARCH_LIST="<sm_arch>" <other_flags> \
    pip install --no-build-isolation --no-deps -e . -v
# 3. Verify import
python -c "import <package>"
```

**Key lesson**: ALL extensions in the environment must use the SAME `CUDA_HOME`. If one was built with `/usr/local/cuda-13.0` and another with the shim reporting 12.8, their ABI expectations may conflict. Always set `CUDA_HOME` once (in Step 1a) and reuse it consistently.

## Step 5: Final Verification

Run a comprehensive check:

```bash
python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'GPUs: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}')

from megatron.plugin.platform import get_platform
print(f'Megatron platform: {get_platform()}')

import transformer_engine
print(f'TransformerEngine: {transformer_engine.__version__}')

import apex
print('Apex: OK')

import flash_attn
print(f'Flash-Attention: {flash_attn.__version__}')

print('All dependencies ready!')
"
```

**Post-install verification gate — do NOT proceed to training or model porting until ALL checks pass:**

| Check | Command | Pass Criteria |
|-------|---------|---------------|
| PyTorch CUDA | `python -c "import torch; assert torch.cuda.is_available()"` | No error |
| PyTorch version unchanged | Compare against version from Step 3 | Exact match |
| Megatron-LM-FL | `python -c "from megatron.plugin.platform import get_platform"` | No ImportError |
| TransformerEngine-FL | `python -c "import transformer_engine"` | No ImportError |
| Apex | `python -c "import apex"` | No ImportError |
| Flash-Attention | `python -c "import flash_attn"` | No ImportError |

If ANY check fails, fix it before moving on. Do not proceed with "we'll fix it later" — dependency issues compound during training and are much harder to debug.

### 5b. Package provenance check

Verify that each FL dependency is installed from the correct source — not from a different workspace or stale editable install:

```bash
pip show megatron-core transformer-engine apex flash-attn 2>/dev/null | grep -E "^(Name|Location|Editable)"
```

For each package:
- If `Editable project location` is shown, verify it points to a directory within the CURRENT workspace (not a different `/workspace/X/` directory)
- If the editable path points to a different workspace, the installed code won't match the code you'll read for debugging — reinstall from the correct source tree within your workspace
- For non-editable installs, verify the `Location` is inside the target conda environment's `site-packages/`

**Cross-workspace editable installs are NEVER acceptable.** Even if two directories are at the same git commit today, they can diverge silently. If the dependency source doesn't exist in your workspace, clone it locally first (`git clone <repo> /workspace/<your_workspace>/<dep>/`), then editable-install from the local clone.

This check prevents the most insidious debugging trap: reading source code from one directory while the runtime uses code from a completely different directory.

## Step 6: Multi-Node Deployment

When setting up multiple nodes for distributed training:

1. Ensure the same conda environment and dependencies are installed on ALL nodes
2. Verify passwordless SSH between nodes:
   ```bash
   ssh -o BatchMode=yes <other_node> hostname
   ```
3. Verify NCCL connectivity between nodes:
   ```bash
   # On each node, check IB/RoCE NICs are up
   ibstat 2>/dev/null || rdma link show 2>/dev/null || echo "No RDMA detected"
   ```
4. Set consistent NCCL environment variables across all nodes:
   ```bash
   export NCCL_IB_DISABLE=0        # Enable IB if available
   export NCCL_NET_GDR_LEVEL=5     # GPUDirect RDMA level
   export NCCL_SOCKET_IFNAME=eth0  # Fallback interface (adjust to actual)
   ```
5. Verify shared filesystem is mounted at the same path on all nodes (for checkpoints and data)

## Error Handling Rules

1. **Network errors** (git clone fails, pip timeout): Tell user to configure proxy. Do NOT try alternative URLs or workarounds.
2. **Build errors** (compilation fails): Report the exact error to user. Do NOT modify dependency source code.
3. **Version mismatch**: Report versions found and let user decide. Do NOT skip version checks by patching code.
4. **Successful builds**: Proceed to next step automatically. Do NOT ask user to confirm after each successful install.

## Alternative: Docker Image

If source builds are too complex, recommend the official training Docker image:

```bash
docker pull harbor.baai.ac.cn/flagscale/flagscale-train:dev-cu128-py3.12-20260319182856
docker run -itd --gpus all --shm-size=500g --name <name> harbor.baai.ac.cn/flagscale/flagscale-train:dev-cu128-py3.12-20260319182856 /bin/bash
docker exec -it <name> /bin/bash
# In non-interactive shells (agent), use: conda run -n flagscale-train <command>
```

This image has all dependencies pre-installed.

## Download Best Practices

- Always use `wget -c` (resume) instead of plain `wget` for large files.
- For files > 1GB, verify size after download: `ls -lh <file>`.
- Use proxy when available: check `echo $HTTP_PROXY` before downloading.
- For git clone on large repos, use `--depth 1` to avoid fetching full history.
- If a download fails, resume instead of deleting and re-downloading.
- Run large downloads as separate commands, not chained with `&&` or `&`, so failures are isolated.

---

## Related Skills

- `topo-detect` — detect hardware topology after environment setup
- `train-config` — generate training configuration files
- `train-run` — launch training after environment is ready
