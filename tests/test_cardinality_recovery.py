"""The cardinality probe must not only REPORT a short delivered set -- it must
make the shortfall recoverable, via the tool's OWN code path.

A test that re-implements the recovery logic covers nothing: a mutation in the
tool then leaves the test green. So these tests call
`cardinality_probe.recover_rows` directly.

Motivation (measured, zero-leak rounds): model-extraction agents converge on
their OWN clustered count and self-verify cleanly against that same count, then
deliver a short matrix. The criterion the graded tasks use is per-element ratio
constancy, i.e. equality up to ONE global scale per row.
"""
import numpy as np

from flagscale_agent.tools.cardinality_probe import (
    _kink_directions,
    _segments_along_line,
    recover_rows,
)


def _make_world(seed=0, hidden=6, dim=4, scale=0.3):
    rng = np.random.default_rng(seed)
    A1 = rng.standard_normal((hidden, dim)) * scale
    b1 = rng.standard_normal(hidden) * scale
    A2 = rng.standard_normal(hidden) * scale

    def fwd(x):
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        h = np.maximum(x @ A1.T + b1, 0.0)
        y = h @ A2
        return float(y[0]) if single else y

    return A1, fwd


def _observations(fwd, cols, n_lines=24, seed=7):
    rng = np.random.default_rng(seed)
    obs = []
    for _ in range(n_lines):
        x0 = rng.standard_normal(cols)
        d = rng.standard_normal(cols)
        d = d / np.linalg.norm(d)
        _ts, _ys, bounds = _segments_along_line(fwd, x0, d, cols, n=20001)
        obs.extend(_kink_directions(fwd, x0, d, cols, bounds, n=20001, with_points=True))
    return obs


def _matches(ref, mat, tol=1e-4):
    """The graded relation: some row of `mat` equals `ref` up to ONE global scale."""
    for row in np.asarray(mat):
        if np.any(row == 0):
            continue
        ratio = ref / row
        if not np.all(np.isfinite(ratio)):
            continue
        if float(np.max(np.abs(ratio - ratio.mean()))) < tol:
            return True
    return False


def test_recover_rows_completes_a_short_delivery():
    A1, fwd = _make_world()
    hidden = A1.shape[0]
    delivered = A1[: hidden - 2]          # deliberately short by 2 rows
    obs = _observations(fwd, A1.shape[1])

    recovered, unit, scale = recover_rows(delivered, obs, ang_tol=1e-2)
    assert unit.shape[0] == 2, f"expected 2 missed units, got {unit.shape[0]}"
    assert scale > 0

    merged = np.vstack([delivered, recovered])
    before = sum(_matches(A1[i], delivered) for i in range(hidden))
    after = sum(_matches(A1[i], merged) for i in range(hidden))
    assert before == delivered.shape[0]
    assert after == hidden, f"recovery must complete the set: {after}/{hidden}"


def test_recovered_normal_matches_a_true_row_up_to_sign():
    """Mutation guard for the DIRECTION itself.

    Measured, and the reason the tool keeps the observed direction instead of
    "fixing" a sign: a kink always lies ON its hyperplane, so u.point equals
    -(offset), whose sign is uncorrelated with the row's sign (10:10 in a
    controlled check). It is harmless because the graded criterion is a
    per-ELEMENT RATIO test, which a NEGATIVE global scale also satisfies - so
    the requirement is agreement up to sign, not a fixed sign.
    """
    A1, fwd = _make_world()
    hidden = A1.shape[0]
    delivered = A1[: hidden - 2]
    obs = _observations(fwd, A1.shape[1])
    unit = recover_rows(delivered, obs, ang_tol=1e-2)[1]

    A1n = A1 / np.linalg.norm(A1, axis=1, keepdims=True)
    for u in unit:
        best = max(abs(float(u @ A1n[i])) for i in range(hidden))
        assert best > 0.99, f"recovered normal is not a true row direction (|cos| {best})"
    # and the ratio criterion accepts the recovered row as-is, scaled per-row
    rec = unit * np.linalg.norm(delivered, axis=1).mean()
    for r in rec:
        assert any(_matches(A1[i], r[None, :]) for i in range(hidden)), "recovered row does not satisfy the ratio criterion"


def test_no_phantom_rows_when_delivery_is_complete():
    A1, fwd = _make_world(seed=5)
    obs = _observations(fwd, A1.shape[1], seed=13)
    recovered, unit, _scale = recover_rows(A1, obs, ang_tol=1e-2)
    assert unit.shape[0] == 0, f"complete delivery must recover nothing, got {unit.shape[0]}"
    assert recovered.shape[0] == 0
