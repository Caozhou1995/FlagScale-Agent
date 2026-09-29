# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ResourceProbeGuard — structured resource profile for sizing decisions,

Why: parallelism sizing decisions need to be anchored on CONSTRAINT signals
(cgroup cpu.max / cpuset / memory.max|high / pids.max), not on observation
signals (nproc, /proc/meminfo). Observation signals report the HOST;
constraint signals bound the actual execution slice. Sizing from observations
alone over-commits by orders of magnitude whenever the container's ceiling is
below the host's capacity (empty cpu.max + small effective share, or a static
cpuset), which shows up as wall-clock blowouts in compile/test fan-out.

Principle: anchor constraints first; observers are corroboration only. Reads
happen from THIS process's own cgroup view — the process's own execution
domain — never from an outside vantage.

This guard injects a FRESH constraint profile on EVERY command that sizes
parallelism (nproc / cpu_count / -jN / xargs -P / worker
counts) or launches a background job (its parallelism is decided at launch).
The profile carries per-field `source` so every number is auditable, and
`confidence` so the agent can downgrade itself. When NO constraint is readable
(stub cgroup mounts, hardened runtimes, kubelet static policies), the profile
says so and advertises the step-ladder fallback (1 -> 2 -> 4 with measured
throughput) instead of a host number. Nesting: when the command wraps a
docker/nerdctl/podman run/exec, the reminder points the probe INSIDE the
target container without auto-executing anything.

Never blocks — inject only, and re-injects on every trigger: the agent may
probe, exec into, or wrap a DIFFERENT container within one session, so a stale
one-shot profile could silently describe the wrong execution domain. The
repeat cost is a bounded advisory reminder, not an error.
"""

from __future__ import annotations

import json
import os
import re

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict

# Read vantage, injectable for tests: where /proc/self/cgroup is read from and
# which mount the 0:: path is resolved against.
_SELF_CGROUP = "/proc/self/cgroup"
_CGROUP_BASE = "/sys/fs/cgroup"

# ---------------------------------------------------------------------------
# Parsers — granularity floor set by guard/compile_redirect.py (cpu.max three
# states, v1 cfs fallback, nproc min-clamp); extended with cpuset, memory.high,
# swap.max and pids.max so coverage never regresses below the existing
# implementation.
# ---------------------------------------------------------------------------


def _read_text(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except (OSError, UnicodeDecodeError, ValueError):
        return ""


def _read_cpu_max(root: str) -> tuple[float | None, str]:
    """Quota in cores. State spelled out in the source string: ok / unlimited / unreadable."""
    txt = _read_text(os.path.join(root, "cpu.max"))  # v2: "<quota> <period>" or "max <period>"
    if txt:
        parts = txt.split()
        if parts and parts[0] == "max":
            return None, "unlimited(v2 cpu.max=max)"
        if len(parts) >= 2:
            try:
                quota, period = int(parts[0]), int(parts[1])
            except ValueError:
                return None, "unreadable(v2 cpu.max malformed)"
            if quota > 0 and period > 0:
                return quota / period, f"cgroup v2 cpu.max ({parts[0]}/{parts[1]})"
            if quota <= 0:
                return None, "unlimited(v2 cpu.max quota<=0)"
            return None, "unreadable(v2 cpu.max period<=0)"
        return None, "unreadable(v2 cpu.max malformed)"
    # v1 fallback: cfs_quota_us / cfs_period_us
    q_txt = _read_text(os.path.join(root, "cpu/cpu.cfs_quota_us"))
    if q_txt:
        try:
            quota = int(q_txt)
        except ValueError:
            return None, "unreadable(v1 cfs_quota malformed)"
        if quota <= 0:
            return None, "unlimited(v1 cfs_quota<=0)"
        p_txt = _read_text(os.path.join(root, "cpu/cpu.cfs_period_us"))
        try:
            period = int(p_txt) if p_txt else 100000
        except ValueError:
            period = 100000
        if period <= 0:
            period = 100000
        return quota / period, f"cgroup v1 cfs ({quota}/{period})"
    return None, "unreadable(no v2 cpu.max, no v1 cfs files)"


def _read_cpuset(root: str) -> tuple[list[int] | None, str]:
    """Effective CPU affinity list. Unset/empty = unrestricted (None), not zero."""
    # v2 unified hierarchy: files at the cgroup root. v1: under the cpuset
    # controller dir, where the effective file is named cpuset.effective_cpus.
    # .effective (actual affinity) before .cpus (configured).
    for rel in ("cpuset.cpus.effective", "cpuset.effective_cpus",
                "cpuset/cpuset.cpus.effective", "cpuset/cpuset.effective_cpus",
                "cpuset.cpus", "cpuset/cpuset.cpus"):
        txt = _read_text(os.path.join(root, rel))
        if txt:
            break
    if not txt:
        return None, "unreadable(no cpuset files)"
    if txt in ("-1", ""):
        return None, f"unlimited(cpuset={txt})"
    cpus = _parse_cpu_list(txt)
    if cpus is None:
        return None, f"unreadable(cpuset={txt!r} unparsed)"
    if not cpus:
        return None, f"unlimited(cpuset empty -> unrestricted)"
    return cpus, f"cgroup cpuset ({txt})"


def _parse_cpu_list(txt: str) -> list[int] | None:
    """Parse '0-7,16-23' style lists. Returns None on malformed input."""
    cpus: list[int] = []
    for chunk in txt.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo_hi = chunk.split("-", 1)
            try:
                lo, hi = int(lo_hi[0]), int(lo_hi[1])
            except (ValueError, IndexError):
                return None
            if lo < 0 or hi < lo:
                return None
            cpus.extend(range(lo, hi + 1))
        else:
            try:
                cpus.append(int(chunk))
            except ValueError:
                return None
    return cpus


def _read_memory_limit(root: str) -> tuple[float | None, float | None, str]:
    """(limit_gib, swap_gib, source). v2 memory.max / v1 memory.limit_in_bytes; 'max'/'-1' = unlimited."""
    limit_txt = _read_text(os.path.join(root, "memory.max"))
    limit_name = "memory.max"
    if not limit_txt:
        limit_txt = _read_text(os.path.join(root, "memory/memory.limit_in_bytes"))
        limit_name = "memory/memory.limit_in_bytes(v1)"
    limit = _parse_mem_value(limit_txt)
    swap_txt = _read_text(os.path.join(root, "memory.swap.max"))
    swap_name = "memory.swap.max"
    if not swap_txt:
        swap_txt = _read_text(os.path.join(root, "memory/memory.memsw.limit_in_bytes"))
        # v1 memsw is the RAM+SWAP TOTAL (not swap alone) — label it so the
        # value stays auditable as the sum rather than a swap-only figure.
        swap_name = "memory/memory.memsw.limit_in_bytes(v1, RAM+swap total)"
    swap = _parse_mem_value(swap_txt)
    src = f"cgroup {limit_name}={limit_txt or 'absent'} {swap_name}={swap_txt or 'absent'}"
    return limit, swap, src


def _read_memory_high(root: str) -> tuple[float | None, str]:
    """Soft limit; throttle pressure starts here before OOM. v2 only (no v1 equivalent)."""
    txt = _read_text(os.path.join(root, "memory.high"))
    if not txt:
        return None, "absent(no memory.high)"
    val = _parse_mem_value(txt)
    return val, f"cgroup memory.high={txt}"


def _parse_mem_value(txt: str) -> float | None:
    """Bytes (or 'max'/'-1') -> GiB, or None for unlimited/unreadable.

    v1 kernels write huge sentinel values (2^63, 2^64-1) for "unlimited";
    anything >= 2^60 (1 EiB — beyond any real memory limit) is treated as
    unlimited instead of an astronomic GiB figure."""
    if not txt:
        return None
    if txt in ("max", "-1"):
        return None
    try:
        raw = int(txt)
    except ValueError:
        return None
    if raw >= 2 ** 60:
        return None
    return raw / (1024 ** 3)


def _read_pids_max(root: str) -> tuple[int | None, str]:
    """v2 pids.max / v1 pids/pids.max. 'max' = unlimited."""
    txt = _read_text(os.path.join(root, "pids.max"))
    if not txt:
        txt = _read_text(os.path.join(root, "pids/pids.max"))
    if not txt:
        return None, "absent(no pids.max)"
    if txt == "max":
        return None, f"unlimited(pids.max=max)"
    try:
        val = int(txt)
    except ValueError:
        return None, f"unreadable(pids.max={txt!r})"
    return val, f"cgroup pids.max={txt}"


def _read_nproc() -> tuple[int | None, str]:
    """CPU count via the kernel affinity mask when available — a true
    constraint (the scheduler cannot schedule beyond it), else the host CPU
    count (observation). Serves only as an upper-bound estimate and as a
    clamp on the measured constraint; cgroup files remain the anchor."""
    try:
        aff = getattr(os, "sched_getaffinity", None)
        if aff is not None:
            try:
                return len(aff(0)), "sched_getaffinity(0) (kernel affinity mask)"
            except OSError:
                pass
        count = os.cpu_count()
        return (count, "/proc via os.cpu_count()" if count else "unreadable(cpu_count None)")
    except Exception:  # pragma: no cover - defensive
        return None, "unreadable(cpu_count failed)"


def _effective_cpu(cgroup_root: str) -> tuple[float | None, str]:
    """min(cgroup quota, cpuset size) — nested hierarchy takes the minimum
    (cgroup v2 semantics: an inner limit cannot exceed the ancestor's).

    When no CPU constraint file is readable (unlimited/unreadable states), the
    host nproc is used as an UPPER-BOUND estimate, not as a measured constraint:
    a process running in the host cgroup genuinely gets the whole machine, while
    an unreadable cgroup mount leaves nproc as the only available (possibly
    inflated) ceiling. Callers must treat the source string, not just the
    number, to distinguish the two."""
    quota, quota_src = _read_cpu_max(cgroup_root)
    cpus, cpuset_src = _read_cpuset(cgroup_root)
    nproc, _ = _read_nproc()
    parts = [quota_src, cpuset_src]
    candidates = []
    if quota is not None:
        candidates.append(quota)
    if cpus is not None:
        candidates.append(float(len(cpus)))
    if not candidates:
        # No constraint file readable. nproc is the only ceiling available:
        # reported as an ESTIMATE (upper bound), never as a measured constraint.
        if nproc is not None and nproc > 0:
            est = float(nproc)
            return est, ("; ".join(parts)
                         + f" -> estimated-from-nproc={nproc:g} (upper bound)")
        return None, "; ".join(parts) + " -> no CPU constraint readable"
    eff = min(candidates)
    if nproc is not None and nproc > 0 and eff > nproc:
        eff = float(nproc)
        parts.append(f"clamped to nproc={nproc}")
    return eff, "; ".join(parts) + f" -> effective={eff:g}"


def _cgroup_root() -> str:
    """Root of THIS process's own cgroup hierarchy (its execution domain).

    Vantage order: for cgroup v2 the path encoded in /proc/self/cgroup
    ("0::/path") is THIS process's own slice — limits live at the leaf, so it is
    probed FIRST (dir existence only: absent files there are an honest
    "unreadable", while reading the mount root would silently report the
    PARENT's limits). Pure-v1 layouts (no v2 line) fall back to the mount root,
    which is the same vantage the granularity-floor predecessor used — v1 leaf
    paths encoded in /proc/self/cgroup lines are not resolved here."""
    try:
        with open(_SELF_CGROUP, "r") as f:
            for line in f:
                parts = line.rstrip("\n").split(":")
                if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
                    cand = os.path.join(_CGROUP_BASE, parts[2].lstrip("/"))
                    if os.path.isdir(cand):
                        return cand
    except OSError:
        pass
    return _CGROUP_BASE


def _build_profile() -> dict:
    """Structured resource profile — every field carries `source` (auditable)."""
    root = _cgroup_root()
    kind, evidence = _detect_exec_domain()
    eff_cpu, cpu_src = _effective_cpu(root)
    nproc, nproc_src = _read_nproc()
    mem_lim, swap, mem_src = _read_memory_limit(root)
    mem_high, high_src = _read_memory_high(root)
    pids_max, pids_src = _read_pids_max(root)

    cpu_est = "estimated-from-nproc" in cpu_src
    cpu_constrained = eff_cpu is not None and not cpu_est
    suggested = None
    conflicts: list[str] = []
    if cpu_constrained and eff_cpu is not None:
        suggested = max(1, int(eff_cpu))
    if "clamped to nproc" in cpu_src:
        conflicts.append(
            "measured CPU constraint exceeds visible nproc; workers capped at nproc"
        )
    return {
        "exec_domain": {"kind": kind, "evidence": evidence},
        "cpu": {
            "effective": eff_cpu,
            "source": cpu_src,
            "nproc": nproc,
            "nproc_source": nproc_src,
            "estimated_upper_bound": cpu_est,
            "suggested_max_cpu": suggested,
            "conflicts": conflicts,
            "note": "nproc = observation (host report, corroboration); effective = constraint anchor",
        },
        "memory_gib": {
            "limit": mem_lim,
            "swap": swap,
            "high": mem_high,
            "source": f"{mem_src}; {high_src}",
        },
        "pids": {"max": pids_max, "source": pids_src},
        "confidence": "high" if cpu_constrained else "low",
        "ladder_available": not cpu_constrained,
    }


def _detect_exec_domain() -> tuple[str, list[str]]:
    evidence: list[str] = []
    if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
        evidence.append("/.dockerenv or /run/.containerenv present")
        kind = "container"
    else:
        kind = "unknown"
    try:
        with open("/proc/1/cgroup", "r") as f:
            c1 = f.read().strip()
    except OSError:
        c1 = ""
    if c1:
        evidence.append(f"/proc/1/cgroup={c1}")
        if kind == "unknown":
            if "/docker/" in c1 or "/lxc/" in c1 or "kubepods" in c1:
                kind = "container"
            else:
                kind = "host"
    else:
        evidence.append("/proc/1/cgroup unreadable")
    return kind, evidence


# ---------------------------------------------------------------------------
# Trigger: a command that SIZES parallelism or LAUNCHES a background job.
# Background-job launch is the practical proxy for a >60s command: its
# parallelism is decided at launch time, before any measurement can veto it.
# ---------------------------------------------------------------------------

_SIZING_RE = re.compile(
    r"(?:\bnproc\b)|(?:cpu_count)|(?:multiprocessing\.cpu)|(?:\bworkers?\s*[=:])"
    r"|(?:-j\s*\d)|(?:-j\b)|(?:xargs\s+[^|;&]*-P)|(?:\bparallel\s+[^|;&]*-P)"
    r"|(?:--parallel(?:ism)?[=\s])"
    r"|(?:make\s+[^|;&]*-j)|(?:\bnum[-_]workers)|(?:\bmax[-_]workers)"
    r"|(?:--?workers?\s+\d)|(?:\bfork\s*\()",
    re.IGNORECASE,
)

_BG_LAUNCH_RE = re.compile(r"background\s*=\s*true", re.IGNORECASE)

_CONTAINER_WRAP_RE = re.compile(
    r"\b(?:docker|nerdctl|podman|apptainer|singularity)\s+(?:run|exec)\b"
)


class ResourceProbeGuard(Guard):
    """Inject-only, on EVERY sizing/launch command — no latch.

    Fresh profile per trigger: a session can span several execution domains
    (outer container -> docker exec -> nested jobs), and one-shot state would
    serve a stale profile for the later ones. Statelessness also makes the
    base-class no-op reset_turn exactly correct — nothing per-turn to clear.
    """

    name = "resource_probe"
    priority = 9  # after compile_redirect (8), before ip_port (12)

    # Stateless: no __init__ state, no per-turn state — base no-op
    # reset_turn inherited unchanged.

    # -- trigger ------------------------------------------------------------

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name != "shell":
            return None
        command = str(ctx.tool_args.get("command", ""))
        if not command:
            return None
        # background is a SEPARATE tool_args key in the real tool contract
        # (see longtimeshell.py L190), never part of the command text — the
        # regex is a defensive fallback for serialized variants only.
        launched_bg = bool(ctx.tool_args.get("background", False))
        if not (launched_bg or _SIZING_RE.search(command)
                or _BG_LAUNCH_RE.search(command)):
            return None
        return GuardVerdict.inject(
            message=self._render(command),
            reason="parallelism_sizing_or_background_launch",
            category="resource_probe",
            # If a block fires in the same resolve, this recon data must ride
            # along in the block message — the registry drops plain injects
            # when co-firing with a block, and the agent should get its
            # execution-domain numbers on the first (blocked) attempt.
            attach_on_block=True,
        )

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        return None

    # -- message ------------------------------------------------------------

    def _render(self, command: str) -> str:
        probe_error = ""
        try:
            profile = _build_profile()
            profile_txt = json.dumps(profile, ensure_ascii=False, indent=2)
        except Exception as exc:  # profile must never crash the tool loop
            probe_error = repr(exc)
            profile_txt = json.dumps({"error": f"profile build failed: {probe_error}"})
        lines = [
            "[ResourceProbeGuard] This command sizes parallelism or launches a background "
            "job — its resource ceiling is being decided NOW. Structured profile of THIS "
            "process's execution domain (constraints anchored first; observation signals "
            "are corroboration only):",
            profile_txt,
        ]
        conf = ""
        try:
            conf = json.loads(profile_txt).get("confidence", "")
        except Exception:
            pass
        if conf == "low":
            lines.append(
                "confidence=low: no CPU constraint readable (stub cgroup mounts or "
                "hardened runtime are common). DO NOT fall back to the host number. "
                "Use the step-ladder instead: start at 1 worker, then 2, then 4 — keep "
                "each step only if measured throughput actually improves. nproc in this "
                "state reports the HOST, not your slice."
            )
        if _CONTAINER_WRAP_RE.search(command):
            lines.append(
                "This command wraps docker/nerdctl/podman run|exec: the profile above "
                "describes the OUTER cgroup view. The inner container may carry its own "
                "--cpus/--memory caps — probe INSIDE it (e.g. append 'cat "
                "/sys/fs/cgroup/cpu.max && nproc' to the wrapped command) before sizing "
                "workloads there. Nothing was auto-executed."
            )
        if probe_error:
            lines.append(
                "[profile build failed: constraint probe unavailable] Do NOT size "
                "from the host nproc — step up a ladder: 1 worker, then 2, then 4, "
                "keeping each step only if measured throughput improves."
            )
        lines.append(
            "Sizing rules: anchor workers on cpu.effective (constraint), not nproc "
            "(observation). Before killing a long-running job you own, apply the "
            "kill-discipline floor: >=2 samples across >=10s or measured progress "
            "evidence; SIGTERM before SIGKILL. Advisory only — no action taken against your command."
        )
        return "\n\n".join(lines)
