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

"""inspect_checkpoint tool — deep checkpoint inspection with shape/dtype/numerical verification."""

import os
import re

from flagscale_agent.react.tools.base import Tool


# dtype name -> array module typecode. Assumed by the caller, never sniffed:
# a headerless dump carries no dtype tag, so the tool must say so out loud.
_RAW_DTYPE_CODE = {
    "float32": "f", "float16": "e", "float64": "d",
    "int32": "i", "int64": "q", "int8": "b", "uint8": "B",
}


def _raw_window_stats(mv, code, start_elem, count):
    """min/max/mean/std over a BOUNDED window of a memoryview (stdlib only).

    Reads only ``count`` elements — never the whole file. A 497MB dump would
    otherwise allocate ~2GB as Python floats, which is the unmeasured-
    accumulation failure this codebase warns about.
    """
    import array
    import sys
    itemsize = array.array(code).itemsize
    total_elems = len(mv) // itemsize
    n = min(count, total_elems - start_elem)
    if n <= 0:
        return None
    a = array.array(code)
    a.frombytes(mv[start_elem * itemsize:(start_elem + n) * itemsize])
    if sys.byteorder != "little":
        a.byteswap()
    lo = hi = None
    total = 0.0
    total_sq = 0.0
    kept = 0
    for v in a:
        v = float(v)
        if v != v:  # NaN never compares; excluded from the stats
            continue
        if lo is None:
            lo = hi = v
        elif v < lo:
            lo = v
        elif v > hi:
            hi = v
        total += v
        total_sq += v * v
        kept += 1
    if not kept:
        return None
    mean = total / kept
    var = max(total_sq / kept - mean * mean, 0.0)
    return {"n": kept, "min": lo, "max": hi, "mean": mean, "std": var ** 0.5}


def inspect_raw_binary(path, dtype="float32", window=4096, windows=8):
    """Inspect a headerless raw numeric dump using ONLY the standard library.

    MEASURED GAP (2026-11-27, gpt2-codegolf): the task ships `gpt2-124M.ckpt`,
    497,759,232 bytes of raw little-endian float32 (numel 124,439,808 = the
    published GPT-2 small parameter count exactly), with no header, no key
    names, no shapes. The container has neither torch nor numpy, so
    `inspect_checkpoint` — which this file's core path imports at its first
    line — was dead on arrival in the one environment the task creates. The
    agent's own post-mortem: "I had no reference logits to diff against (no
    numpy, no network)" and it burned four of seven correction rounds hunting
    layer-stride and orientation bugs "by arithmetic rather than by an oracle".

    This path is what a headerless dump CAN support, and it says plainly what
    it cannot: element count constrains the TOTAL, not the ORDER.
    """
    import array
    import json
    import mmap
    import sys

    code = _RAW_DTYPE_CODE.get(dtype)
    if code is None:
        return (f"ERROR: unsupported dtype {dtype!r}. Choose one of: "
                f"{', '.join(sorted(_RAW_DTYPE_CODE))}")

    size = os.path.getsize(path)
    itemsize = array.array(code).itemsize
    if size % itemsize:
        return (f"ERROR: {size} bytes is not a multiple of {itemsize} (dtype "
                f"{dtype}) — probably not a raw {dtype} dump. Try another "
                f"dtype, or the file has a header/container format.")
    numel = size // itemsize

    out = [f"## Raw binary: {os.path.basename(path)}",
           f"Bytes: {size} | dtype: {dtype} (assumed) | itemsize: {itemsize} "
           f"| numel: {numel}",
           f"Endianness read: {'little' if sys.byteorder == 'little' else 'big'} "
           f"(host order). A wrong-endian read garbles the ieee754 exponent and "
           f"shows up as absurd magnitudes — check the ranges below if unsure."]

    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            mv = memoryview(mm)
            n_win = max(1, min(int(windows or 8), 64))
            span = max(1, numel // n_win)
            out.append("")
            out.append(f"Evenly spaced windows ({n_win} x {span} elements apart, "
                       f"each stat over <= {window} elements):")
            out.append("  win   elem_offset     byte_offset     min            max            mean           std")
            for w in range(n_win):
                st = w * span
                s = _raw_window_stats(mv, code, st, min(span, window or span))
                if s is None:
                    out.append(f"  {w:<4}  {st:<14}  {st*itemsize:<14}  (all-NaN or empty)")
                    continue
                out.append(f"  {w:<4}  {st:<14}  {st*itemsize:<14}  "
                           f"{s['min']:<14.6g} {s['max']:<14.6g} "
                           f"{s['mean']:<14.6g} {s['std']:<14.6g}")
            head = _raw_window_stats(mv, code, 0, min(numel, 4096))
            tail = _raw_window_stats(mv, code, max(0, numel - 4096), 4096)
            mv.release()

    if head:
        out.append("")
        out.append(f"First 4096 elems: min={head['min']:.6g} max={head['max']:.6g} "
                   f"mean={head['mean']:.6g} std={head['std']:.6g}")
    if tail:
        out.append(f"Last  4096 elems: min={tail['min']:.6g} max={tail['max']:.6g} "
                   f"mean={tail['mean']:.6g} std={tail['std']:.6g}")
    out.append("")
    out.append("WHAT THIS DOES NOT TELL YOU: a flat dump carries no key names, no "
               "offsets and no shapes. Element count constrains the TOTAL, not "
               "the ORDER. Any layout claim made from this report is a "
               "hypothesis — confirm it against an independent statistic of a "
               "specific tensor before building on it.")
    out.append("MACHINE_READABLE: " + json.dumps(
        {"numel": numel, "dtype": dtype, "bytes": size, "mode": "raw"}))
    return "\n".join(out)


def _safe_torch_load(path: str):
    """Load a PyTorch checkpoint safely — try weights_only first, fall back."""
    import logging as _logging
    _log = _logging.getLogger(__name__)
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        _log.warning("weights_only=True failed for %s, falling back to weights_only=False. "
                     "This is needed for checkpoints with custom objects.", path)
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_state_dict(path: str):
    """Load state_dict from a checkpoint file."""
    import torch
    if path.endswith(".safetensors"):
        from safetensors import safe_open
        sd = {}
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                sd[key] = f.get_tensor(key)
        return sd
    data = _safe_torch_load(path)
    if isinstance(data, dict):
        for k in ("model", "state_dict", "module"):
            if k in data:
                return data[k]
        if all(isinstance(v, torch.Tensor) for v in list(data.values())[:5]):
            return data
    return None


def inspect_checkpoint(path, reference_path="", expected_keys="", sample_count=20,
                       raw_dtype="float32", raw_window=4096, raw_windows=8):
    """Core inspection logic. Returns formatted report string.

    Two modes, selected automatically:

    * TENSOR mode — torch/safetensors present and the file is a real container
      format. Full key/shape/dtype/anomaly report (the original behaviour).
    * RAW mode — a headerless numeric dump, or an interpreter with no torch.
      The standard-library path. This is the mode that keeps the tool usable
      inside the task container, which has neither torch nor numpy.
    """
    if not os.path.isfile(path):
        return f"ERROR: File not found: {path}"

    try:
        import torch  # noqa: F401  (presence probe, not a hard dependency)
        _have_torch = True
    except Exception:
        _have_torch = False

    if not _have_torch:
        # No torch in THIS interpreter: a raw dump is all we can honour, and it
        # is also exactly what the container-side tasks ship. Say why, so the
        # caller does not mistake the mode for a failure.
        report = inspect_raw_binary(path, dtype=raw_dtype,
                                    window=raw_window, windows=raw_windows)
        return ("[MODE: raw — torch is not importable in this interpreter, so the "
                "tensor-container path is unavailable. Falling back to the "
                "standard-library raw-dump reader.]\n\n" + report)

    import random

    try:
        state_dict = _load_state_dict(path)
    except Exception as e:
        # torch IS present but could not parse this file. A headerless numeric
        # dump is the expected cause on this box (tasks ship raw float32).
        # Fall back ONLY when the size is a clean multiple of the element
        # width — otherwise a genuinely corrupt checkpoint would be silently
        # re-read as if it were raw, hiding the corruption.
        import array as _array
        _code = _RAW_DTYPE_CODE.get(raw_dtype)
        if _code and os.path.getsize(path) % _array.array(_code).itemsize == 0:
            report = inspect_raw_binary(path, dtype=raw_dtype,
                                        window=raw_window, windows=raw_windows)
            return (f"[MODE: raw — torch could not load this file as a container "
                    f"({type(e).__name__}: {e}). Its size IS a clean multiple of "
                    f"the {raw_dtype} element width, so it is being read as a "
                    f"headerless dump. If it was meant to be a real container, "
                    f"that is a corruption, not a layout.]\n\n" + report)
        return f"ERROR: Failed to load checkpoint: {type(e).__name__}: {e}"

    if state_dict is None:
        data = _safe_torch_load(path)
        keys = list(data.keys())[:20] if isinstance(data, dict) else type(data).__name__
        return f"ERROR: Cannot find state_dict. Top-level keys: {keys}"

    out = []
    out.append(f"## Checkpoint: {os.path.basename(path)}")
    out.append(f"Keys: {len(state_dict)} | Size: {os.path.getsize(path) / 1024 / 1024:.1f} MB")

    # Dtype distribution
    dtype_counts = {}
    anomalies = []
    for key, tensor in state_dict.items():
        dt = str(tensor.dtype)
        dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
        if tensor.numel() == 0:
            anomalies.append(f"  EMPTY: {key} shape={tuple(tensor.shape)}")
        elif tensor.is_floating_point():
            if torch.isnan(tensor).any():
                anomalies.append(f"  NaN: {key} shape={tuple(tensor.shape)}")
            elif torch.isinf(tensor).any():
                anomalies.append(f"  Inf: {key} shape={tuple(tensor.shape)}")
            elif tensor.abs().max().item() == 0.0:
                anomalies.append(f"  ALL-ZERO: {key} shape={tuple(tensor.shape)}")

    out.append("\nDtypes: " + ", ".join(f"{dt}={c}" for dt, c in sorted(dtype_counts.items(), key=lambda x: -x[1])))

    if anomalies:
        out.append(f"\nANOMALIES ({len(anomalies)}):")
        for a in anomalies[:15]:
            out.append(a)
        if len(anomalies) > 15:
            out.append(f"  ... +{len(anomalies) - 15} more")
    else:
        out.append("\nAnomalies: None")

    # Sample statistics
    keys_list = list(state_dict.keys())
    n = min(sample_count, len(keys_list))
    sample_keys = keys_list[:n] if len(keys_list) <= sample_count else random.sample(keys_list, n)
    out.append(f"\nSample ({n} tensors):")
    for key in sorted(sample_keys):
        t = state_dict[key]
        if t.is_floating_point() and t.numel() > 0:
            tf = t.float()
            out.append(
                f"  {key}: {tuple(t.shape)} {t.dtype} "
                f"mean={tf.mean().item():.6f} std={tf.std().item():.6f}"
            )
        else:
            out.append(f"  {key}: {tuple(t.shape)} {t.dtype}")

    # Expected keys check
    if expected_keys:
        out.append("\nExpected keys:")
        for pat in (p.strip() for p in expected_keys.split(",")):
            matches = [k for k in state_dict if re.search(pat, k)]
            out.append(f"  {'OK' if matches else 'MISSING'}: '{pat}' → {len(matches)} keys")

    # Reference comparison
    if reference_path and os.path.isfile(reference_path):
        out.append(f"\nReference comparison vs {os.path.basename(reference_path)}:")
        try:
            ref_dict = _load_state_dict(reference_path)
            if ref_dict is not None:
                ref_keys = set(ref_dict)
                cur_keys = set(state_dict)
                only_ref = ref_keys - cur_keys
                only_cur = cur_keys - ref_keys
                common = ref_keys & cur_keys
                out.append(f"  Ref={len(ref_dict)} Cur={len(state_dict)} Common={len(common)}")
                if only_ref:
                    out.append(f"  Only in ref ({len(only_ref)}):")
                    for k in sorted(only_ref)[:8]:
                        out.append(f"    - {k}")
                if only_cur:
                    out.append(f"  Only in cur ({len(only_cur)}):")
                    for k in sorted(only_cur)[:8]:
                        out.append(f"    + {k}")
                mismatches = []
                for k in sorted(common):
                    r, c = ref_dict[k], state_dict[k]
                    if r.shape != c.shape:
                        mismatches.append(f"  SHAPE: {k} ref={tuple(r.shape)} cur={tuple(c.shape)}")
                    elif r.dtype != c.dtype:
                        mismatches.append(f"  DTYPE: {k} ref={r.dtype} cur={c.dtype}")
                if mismatches:
                    out.append(f"  Mismatches ({len(mismatches)}):")
                    for m in mismatches[:10]:
                        out.append(m)
                else:
                    out.append(f"  All {len(common)} common keys match shape and dtype")
        except Exception as e:
            out.append(f"  ERROR: {e}")

    # Verdict
    issues = len(anomalies)
    if expected_keys:
        issues += sum(1 for p in expected_keys.split(",")
                      if not any(re.search(p.strip(), k) for k in state_dict))
    out.append(f"\nVerdict: {'ISSUES FOUND — fix before training' if issues else '[OK] checkpoint valid'}")

    return "\n".join(out)


class InspectCheckpointTool(Tool):
    name = "inspect_checkpoint"
    description = (
        "Inspect a checkpoint file. TWO modes, auto-selected: "
        "(1) tensor containers (.pt/.bin/.safetensors) — key count, shapes, dtypes, "
        "anomalies (all-zero/NaN/Inf), sampled tensor stats, optional reference diff; "
        "(2) HEADERLESS RAW numeric dumps (e.g. a .ckpt that is plain float32 with no "
        "header) — total element count plus windowed min/max/mean/std, using only the "
        "Python standard library, so it still works when torch and numpy are absent. "
        "For a raw dump it reports element count and per-window statistics; it does "
        "NOT report key names, offsets or shapes, because a flat file carries none. "
        "Use this as a numeric oracle before guessing tensor layouts or layer strides."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to checkpoint file to inspect.",
            },
            "reference_path": {
                "type": "string",
                "description": "Optional: reference checkpoint to compare shapes/dtypes against.",
            },
            "expected_keys": {
                "type": "string",
                "description": "Optional: comma-separated key patterns (regex) to verify exist.",
            },
            "sample_count": {
                "type": "integer",
                "description": "Number of tensors to sample for statistics. Default: 20.",
            },
            "raw_dtype": {
                "type": "string",
                "description": ("Element type ASSUMED when reading a headerless dump: "
                                "float32 (default), float16, float64, int32, int64, "
                                "int8, uint8. A headerless file carries no dtype tag, "
                                "so this is your stated assumption, not a detection."),
            },
            "raw_window": {
                "type": "integer",
                "description": "Elements per statistics window for a raw dump. Default: 4096.",
            },
            "raw_windows": {
                "type": "integer",
                "description": "Number of evenly spaced windows across a raw dump. Default: 8.",
            },
        },
        "required": ["path"],
    }

    def execute(self, **kwargs) -> str:
        return inspect_checkpoint(
            path=kwargs["path"],
            reference_path=kwargs.get("reference_path", ""),
            expected_keys=kwargs.get("expected_keys", ""),
            sample_count=kwargs.get("sample_count", 20),
            raw_dtype=kwargs.get("raw_dtype") or "float32",
            raw_window=kwargs.get("raw_window") or 4096,
            raw_windows=kwargs.get("raw_windows") or 8,
        )
