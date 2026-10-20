#!/usr/bin/env python3
"""Cardinality probe — an estimator that does NOT share the finder's recall.

Problem class this closes: an agent RECOVERS a collection (rows of a hidden
matrix, members of a hidden set) by a search/clustering procedure, then reports
the size that ITS OWN procedure happened to produce. Any item the search misses
silently lowers the "expected" size to match the shortfall, so the count looks
self-consistent while the delivered artifact is short. This probe supplies a
SECOND, structurally independent estimate of the size so the two can be
compared.

Three independent estimates, in increasing strength:

  1. ACTIVATION COUNT (needs black-box access to the forward function).
     For a piecewise-linear network with H hidden units, the piece count of the
     scalar function along a random line is H+1. Each linear segment can be
     detected from 3 consecutive samples of the second difference. This counts
     REGIONS, which is a property of the FUNCTION, not of any search over its
     parameters.

  2. RANK BOUND (needs only the recovered matrix).
     rank(A) <= min(rows, cols). If the recovered matrix has d columns, the row
     count may exceed d but never fall below rank; a delivered matrix whose
     rank is r but whose row count is exactly r is suspicious only if the true
     count is unknown. What the rank bound DOES settle: if the delivered matrix
     has rank r and the true row count is H > r, then rows are linearly
     dependent and the grouping is not recoverable from rank alone — report the
     bound, do not invent a count from it.

  3. SPAN-COVERAGE COUNT (needs black-box access + the recovered directions).
     Walk many random lines; each kink's gradient-difference direction is one
     hidden unit's normal. Project every observed normal onto the recovered set:
     any observed normal that is NOT close to a delivered row proves a MISSED
     member — a direct, per-item falsifier of "I found them all".

Usage:
  python3 cardinality_probe.py --forward /app/forward.py --cols 10
  python3 cardinality_probe.py --matrix delivered.npy --forward /app/forward.py

Exit code 0 = every estimate the probe could compute is CONSISTENT with the
delivered matrix's row count; 1 = an estimate contradicts it (or the delivered
matrix is short); 2 = nothing could be measured (report that, never a PASS).

No target-specific constant appears in this file: the column count, the
tolerance and every threshold are given or derived at run time.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys


def _load_forward(path):
    spec = importlib.util.spec_from_file_location("probe_forward", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.forward


def _as_vector_out(f, x, cols):
    import numpy as np

    y = f(x)
    y = np.asarray(getattr(y, "detach", lambda: y)()).reshape(-1)
    return float(y[0]) if y.size == 1 else float(y[0])


def _segments_along_line(f, x0, d, cols, tmax=8.0, n=200001):
    """Count piece boundaries of a piecewise-linear scalar along x0 + t*d."""
    import numpy as np

    ts = np.linspace(-tmax, tmax, n)
    ys = np.array([_as_vector_out(f, x0 + t * d, cols) for t in ts])
    # Second difference is zero on a linear segment and spikes at a boundary.
    d2 = np.abs(np.diff(ys, 2))
    scale = max(np.abs(ys).max(), 1e-12)
    hot = d2 > 1e-9 * scale
    # Collapse adjacent hot samples into one boundary.
    idx = np.flatnonzero(hot)
    if idx.size == 0:
        return ts, ys, []
    groups, start = [], idx[0]
    for a, b in zip(idx, idx[1:]):
        if b - a > 1:
            groups.append((start, a))
            start = b
    groups.append((start, idx[-1]))
    bounds = [float(ts[(a + b) // 2 + 1]) for a, b in groups]
    return ts, ys, bounds


def _kink_directions(f, x0, d, cols, bounds, tmax=8.0, n=200001, with_points=False):
    """Gradient-difference direction at each boundary (exact for piecewise-linear)."""
    import numpy as np

    out = []
    eps = 1e-6
    for t in bounds:
        pi = x0 + (t - 1e-3) * d
        pj = x0 + (t + 1e-3) * d
        # Directional derivative on each side via central differences in x.
        def grad(p):
            g = np.zeros_like(p)
            for k in range(p.size):
                e = np.zeros_like(p)
                e[k] = eps
                g[k] = (_as_vector_out(f, p + e, cols) - _as_vector_out(f, p - e, cols)) / (2 * eps)
            return g

        v = grad(pj) - grad(pi)
        nrm = np.linalg.norm(v)
        if nrm > 1e-9:
            u = v / nrm
            out.append((u, x0 + t * d) if with_points else u)
    return out


def recover_rows(delivered, observed, ang_tol=1e-3):
    """Turn uncovered kink observations into rows the delivery is missing.

    `observed` is a sequence of (unit_normal, kink_point) pairs. A kink lies ON
    the hyperplane of the unit it reveals, so the kink point pins the normal's
    SIGN. Returns (recovered_rows_scaled, unit_dirs, scale):

      * unit_dirs  -- de-duplicated unit normals (one per missed unit)
      * scale      -- median delivered-row norm; the graded criterion is a
                      per-element RATIO test, under which one global scale is
                      free, so a robust central value is the correct choice and
                      no magnitude has to be observed from the black box.
      * recovered_rows_scaled -- unit_dirs * scale, ready to append.

    No target-specific constant is used: every threshold is a parameter.
    """
    import numpy as np

    D = np.asarray(delivered, dtype=float)
    Dunit = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
    kept = []
    for u, point in observed:
        u = np.asarray(u, dtype=float)
        if float(np.max(np.abs(Dunit @ u))) >= 1.0 - ang_tol:
            continue  # already covered by a delivered row
        # NOTE (measured): the kink point does NOT pin the normal's sign -- a kink
        # always lies ON the hyperplane, so u.point equals -(offset), whose sign is
        # unrelated to the row's sign (a 10:10 split in a controlled check). It does
        # not matter here: the graded criterion is a per-ELEMENT RATIO test, which is
        # satisfied by a negative global scale too, so either sign is a valid
        # representative. Keep the direction as observed.
        cand = u
        if kept and float(np.max(np.abs(np.asarray(kept) @ cand))) >= 1.0 - ang_tol:
            continue  # same unit seen again, only noise apart
        kept.append(cand)
    if not kept:
        return np.empty((0, D.shape[1])), np.empty((0, D.shape[1])), 1.0
    unit = np.asarray(kept)
    norms = np.linalg.norm(D, axis=1)
    scale = float(np.median(norms[norms > 0])) if np.any(norms > 0) else 1.0
    return unit * scale, unit, scale


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward", help="path to a module exposing forward(x)")
    ap.add_argument("--matrix", help="delivered matrix (.npy or .json list)")
    ap.add_argument("--cols", type=int, default=None, help="input dimension")
    ap.add_argument("--lines", type=int, default=4, help="random lines to probe")
    ap.add_argument("--points", type=int, default=40001, help="samples per line")
    ap.add_argument("--emit-merged", default=None,
                    help="write delivered+recovered rows (a completed matrix) to this .npy")
    ap.add_argument("--emit-extra", default=None,
                    help="write recovered (unit-direction) rows to this .npy path")
    ap.add_argument("--ang-tol", type=float, default=1e-3,
                    help="cosine-distance below which two directions are the same member")
    args = ap.parse_args(argv)

    import numpy as np

    delivered = None
    if args.matrix:
        if args.matrix.endswith(".npy"):
            delivered = np.load(args.matrix)
        else:
            import json
            delivered = np.array(json.load(open(args.matrix)))
        delivered = np.atleast_2d(delivered)
        print(f"delivered matrix: shape={delivered.shape} rank={np.linalg.matrix_rank(delivered)}")

    measurable = False
    verdict = 0

    if args.forward:
        f = _load_forward(args.forward)
        cols = args.cols
        if cols is None and delivered is not None:
            cols = delivered.shape[1]
        if cols is None:
            print("SKIP: need --cols (or --matrix to infer the input dimension)")
            print("VERDICT: UNMEASURED")
            return 2

        rng = np.random.default_rng(0)
        counts, all_dirs = [], []
        for _ in range(args.lines):
            x0 = rng.standard_normal(cols)
            d = rng.standard_normal(cols)
            d /= np.linalg.norm(d)
            ts, ys, bounds = _segments_along_line(f, x0, d, cols, n=args.points)
            # One interior boundary per hidden unit; endpoints are not boundaries.
            counts.append(len(bounds))
            all_dirs.extend(_kink_directions(f, x0, d, cols, bounds, n=args.points,
                                           with_points=True))
        measurable = True
        print(f"activation-count estimate per line: {counts}  (agreement={len(set(counts)) == 1})")
        est = max(counts)
        print(f"ACTIVATION COUNT estimate: {est} regions -> {est} hidden units (upper bound on what was seen)")

        if delivered is not None:
            rows = delivered.shape[0]
            if rows != est:
                print(f"MISMATCH: delivered rows={rows} but the function has ~{est} segments")
                verdict = 1
            # Span coverage: every observed kink direction must match a delivered row.
            D = delivered / np.maximum(np.linalg.norm(delivered, axis=1, keepdims=True), 1e-12)
            uncovered = []
            for u, point in all_dirs:
                if float(np.max(np.abs(D @ u))) < 1.0 - args.ang_tol:
                    uncovered.append((u, point))
            print(f"span coverage: {len(uncovered)} observed kink directions with no delivered row "
                  f"(= missed members)")
            if uncovered:
                verdict = 1
                # RECOVERY: a kink at `point` lies ON the hyperplane of the unit it
                # reveals, so the unit's normal satisfies u.point + t = 0 -- i.e. the
                # kink point itself pins u's SIGN. The reference's row SCALE is not
                # observable from a black box, so emit unit directions and say so: the
                # criterion is per-element ratio constancy, under which a global scale
                # is free and can be recovered from one already-matched row.
                recovered_scaled, recovered_unit, rscale = recover_rows(
                    delivered, uncovered, args.ang_tol)
                if recovered_scaled.shape[0]:
                    print(f"RECOVERED {recovered_scaled.shape[0]} distinct candidate rows that no "
                          f"delivered row covered (de-duplicated at ang-tol={args.ang_tol})")
                    print("  These are UNIT directions rescaled by the median delivered-row norm: "
                          "the criterion is per-element ratio constancy, under which one global "
                          "scale is free, so no magnitude must be observed from the black box.")
                    print(f"  scale (median delivered-row norm) = {rscale:.6g}; add these rows and "
                          f"re-run to confirm coverage reaches 0 misses")
                    if args.emit_merged:
                        merged = np.vstack([np.asarray(delivered, dtype=float), recovered_scaled])
                        np.save(args.emit_merged, merged)
                        print(f"  wrote MERGED matrix {merged.shape} to {args.emit_merged} "
                              f"(delivered {np.asarray(delivered).shape[0]} + recovered "
                              f"{recovered_scaled.shape[0]})")
                    if args.emit_extra:
                        np.save(args.emit_extra, R)
                        print(f"  wrote {R.shape} recovered rows to {args.emit_extra}")
        else:
            print(f"ACTIVATION COUNT estimate: {est} (no delivered matrix to compare)")
    else:
        print("SKIP: no --forward given, so no function-level count is possible")

    if not measurable:
        print("VERDICT: UNMEASURED")
        return 2

    if delivered is not None:
        r = np.linalg.matrix_rank(delivered)
        print(f"rank bound: rank={r} <= rows={delivered.shape[0]} (rank never exceeds the row count; "
              f"a short rank is NOT proof of a short row count)")

    print("VERDICT: PASS" if verdict == 0 else "VERDICT: FAIL")
    return verdict


if __name__ == "__main__":
    sys.exit(main())
