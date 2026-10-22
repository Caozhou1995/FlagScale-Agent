#!/usr/bin/env python3
"""convention_discrimination.py -- decide whether ONE observation can pick a convention.

WHY THIS EXISTS
---------------
A recurring failure: several candidate conventions (a scaling, a normalisation, an
axis order, an offset, a rounding rule) all produce the SAME output on the only
example you can see. That agreement feels like confirmation but carries zero
information -- it cannot separate the candidates, so whichever one you adopt is
an unverified guess. Worse, agents then break the tie with a QUALITY proxy
(accuracy against labels, logit margin, confidence, a benchmark score), which
measures something else entirely and frequently points at the wrong candidate.

This tool answers one question mechanically:
  "Given a set of candidate conventions and a set of observations (inputs), do any
   of those observations actually SEPARATE the candidates?"

If yes, it prints the separating observations -- run those through the consumer's
own reference path and the answer falls out. If no, it says so, and the only
defensible move left is the minimal-uncertainty default (add no unrequested
transformation).

Pure stdlib. No third-party dependency, no network, no task knowledge.

USAGE
  # candidates from a JSON file: {"name": "python expression over x", ...}
  python3 convention_discrimination.py --spec spec.json --inputs inputs.npy
  python3 convention_discrimination.py --demo     # built-in self-check
"""
import argparse
import json
import sys

try:
    import numpy as np
except ImportError:  # numpy is a hard dependency of this package
    print("FATAL: numpy not importable; run with the agent's own interpreter",
          file=sys.stderr)
    raise


def apply_convention(expr, x):
    """Evaluate one convention expression against an input array."""
    env = {"x": x, "np": np, "abs": abs, "min": min, "max": max}
    return eval(expr, {"__builtins__": {}}, env)  # noqa: S307 - sandboxed namespace


def apply_collapse(expr, v):
    """Map a convention's output to the FINAL observable the consumer compares.

    This matters: two conventions can differ at the input level yet collapse to
    the SAME observable after the pipeline (same label, same file bytes, same
    rounded value). Discriminability must be judged on the observable, never on
    the intermediate array -- otherwise the tool reports a separation that the
    consumer cannot see.
    """
    if not expr:
        return v
    env = {"v": v, "np": np, "abs": abs, "min": min, "max": max}
    return eval(expr, {"__builtins__": {}}, env)  # noqa: S307 - sandboxed namespace


def discriminability(conventions, inputs, tol=1e-9, collapse=None):
    """For each input, compute every convention's FINAL observable and report separation.

    Returns a list of (input_index, n_distinct_outputs, outputs_by_convention).

    An input with n_distinct == 1 cannot discriminate anything -- BUT only when
    every candidate actually evaluated. A candidate that RAISED produced no
    observable at all: it is UNKNOWN, not "agrees with the others". Counting it
    as a value would let N-1 broken candidates collapse onto one identical
    `<error: ...>` string and manufacture a false NOT SEPARABLE verdict, which is
    the exact opposite of the truth (the spec is invalid, so nothing was learned).
    Such a row is reported with .errored populated and the caller refuses to draw
    a conclusion from it.
    """
    rows = []
    for i, x in enumerate(inputs):
        outs = {}
        errored = {}
        for name, expr in conventions.items():
            try:
                v = apply_convention(expr, x)
                v = apply_collapse(collapse, v)
            except Exception as e:  # a candidate that cannot run is not comparable
                errored[name] = f"{type(e).__name__}: {e}"
                continue
            arr = np.asarray(v, dtype=np.float64).ravel()
            # coarse-quantise so float noise does not fake a separation
            key = tuple(np.round(arr, int(-np.log10(tol))) if tol > 0 else arr)
            outs[name] = key
        # Separability is judged ONLY over candidates that produced an observable.
        distinct = len(set(outs.values())) if len(outs) >= 2 else 1
        rows.append((i, distinct, outs, errored))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", help="JSON: {\"candidates\": {name: expr}, ...}")
    ap.add_argument("--inputs", help=".npy of shape (n, ...) -- one input per row")
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--collapse", default=None,
                    help="expression mapping a convention's output `v` to the FINAL "
                         "observable the consumer compares (e.g. 'np.argmax(v)'). "
                         "Discriminability is judged on this, not on v.")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        rng = np.random.RandomState(0)
        inputs = rng.randint(0, 256, size=(6, 4)).astype(np.float64)
        conventions = {
            "raw": "x",
            "div255": "x / 255.0",
            "mean_std": "(x / 255.0 - 0.5) / 0.2",
        }
        print("DEMO: candidates =", list(conventions))
        rows = discriminability(conventions, inputs, args.tol)
        sep = [r for r in rows if r[1] > 1]
        for idx, d, _, _ in rows:
            print(f"  input {idx}: {d} distinct output(s)")
        print(f"VERDICT: {'SEPARABLE' if sep else 'NOT SEPARABLE'} "
              f"({len(sep)}/{len(rows)} inputs separate the candidates)")
        return 0

    if not args.spec or not args.inputs:
        ap.error("--spec and --inputs are required (or use --demo)")
    spec = json.load(open(args.spec))
    conventions = spec["candidates"]
    inputs = np.load(args.inputs)
    if inputs.ndim == 1:
        inputs = inputs[None, :]
    rows = discriminability(conventions, inputs, args.tol, args.collapse)
    sep = [r for r in rows if r[1] > 1]
    errored_any = {n for _, _, _, e in rows for n in e}
    for idx, d, _, err in rows:
        suffix = f"  [ERR: {len(err)} candidate(s) did not evaluate]" if err else ""
        print(f"input {idx}: {d} distinct{suffix}")
    if errored_any:
        print(f"WARNING: these candidates raised and produced NO observable: "
              f"{sorted(errored_any)}")
        print("A candidate that raises was NOT compared -- it is unknown, not "
              "agreeing. Any input above where a candidate raised is "
              "INCONCLUSIVE for that candidate; fix the expression and re-run "
              "before drawing a verdict.")
    if sep:
        print(f"VERDICT: SEPARABLE -- these inputs separate the candidates: "
              f"{[r[0] for r in sep]}")
        print("To decide, run those inputs through the CONSUMER'S OWN reference")
        print("path -- the pipeline that produced the values the checker compares")
        print("against: the author's reference/solve program, a shipped sample")
        print("output, or a golden artifact.")
        print("A measurement against GROUND-TRUTH LABELS is NOT that path. Accuracy,")
        print("confidence, margin or any benchmark score measures how GOOD a")
        print("candidate is; the checker measures whether it is EQUIVALENT to the")
        print("reference pipeline, which may be weaker or differently configured.")
        print("If the reference path is not available, the axis is UNDECIDED: fall")
        print("back to the minimal-uncertainty default (add no unrequested")
        print("transformation) and record the residual risk. Never take the")
        print("highest-scoring candidate.")
        return 0
    print("VERDICT: NOT SEPARABLE -- no visible observation distinguishes the "
          "candidates.")
    print("Do NOT treat their agreement as confirmation, and do NOT break the tie "
          "with a quality metric (accuracy / confidence / margin).")
    print("Use the minimal-uncertainty default: the candidate that adds no "
          "unrequested transformation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
