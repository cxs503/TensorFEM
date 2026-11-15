"""GAP-PP-1 regression tests: envelope control of the acceptance sweep and the
ill-conditioning of the FEM "ratio of maxima".

These tests pin the two findings of
``tools/verify_gap_pp1_delta_caliber.py`` / ``docs/VERIFY-GAP-PP-1-delta-caliber.md``:

1. ``scan_contact_configurations`` now *controls* the small-deflection envelope
   (``target_w_over_t``): the acceptance sweep is reported at ``w/t = 0.2``
   instead of the historical ``w/t = 5..17``.  A future default change cannot
   silently move the sweep back outside the theory's validity.
2. The FEM "peak over element centroids" ratio between the non-uniform and the
   uniform patch is **not mesh converged** (it moves by tens of percentage
   points between nx = 8 and nx = 12).  The test asserts that it is still
   unstable -- i.e. it guards the *documented* reason why the quoted Delta must
   come from the analytic Navier path.  If someone ever makes the FEM peak a
   converged quantity, this test fails loudly and the doc must be updated.
"""
from __future__ import annotations

import math

import pytest

import tensorfem.patch_from_contact as P

SIDE = 1.0
E = 2.06e11
NU = 0.3
T = 0.01
K = 1.0e7
LIMIT = 0.5

MILD = {
    "name": "mild_nonuniform",
    "peak_pressure": 5.0e6,
    "sigma_x": 0.15 * SIDE,
    "sigma_y": 0.15 * SIDE,
    "patch_width": 0.3 * SIDE,
    "patch_length": 0.3 * SIDE,
}


def _scan(nx: int, **kw) -> dict:
    return P.scan_contact_configurations(
        plate_nx=nx, plate_side=SIDE, young=E, poisson=NU, thickness=T,
        winkler_k=K, **kw,
    )


def test_scan_default_lands_inside_small_deflection_envelope():
    """The default sweep is envelope controlled at w/t = target."""
    scan = _scan(8)
    summary = scan["summary"]
    assert summary["target_w_over_t"] == pytest.approx(0.2)
    assert summary["all_in_small_deflection_envelope"] is True
    assert summary["n_in_envelope"] == summary["n_configurations"] >= 3
    for cfg in scan["configurations"]:
        assert cfg["in_small_deflection_envelope"] is True
        # the scaling is exact (linear model), not a search: w/t lands on target
        assert cfg["w_over_t_nonuniform"] == pytest.approx(0.2, rel=1e-9)
        assert cfg["w_over_t_nonuniform"] <= LIMIT
        # ... and the applied pressure really is the scaled one
        assert cfg["pressure_scale_factor"] != 1.0
        assert cfg["peak_pressure_MPa"] == pytest.approx(
            cfg["nominal_peak_pressure_MPa"] * cfg["pressure_scale_factor"], rel=1e-12
        )


def test_nominal_mode_is_flagged_outside_envelope():
    """The historical nominal-pressure sweep is reproducible AND flagged invalid."""
    scan = _scan(8, target_w_over_t=None)
    summary = scan["summary"]
    assert summary["all_in_small_deflection_envelope"] is False
    assert summary["max_w_over_t"] > LIMIT
    for cfg in scan["configurations"]:
        assert cfg["in_small_deflection_envelope"] is False
        assert cfg["pressure_scale_factor"] == 1.0
        assert cfg["peak_pressure_MPa"] == pytest.approx(
            cfg["nominal_peak_pressure_MPa"]
        )
    # the historical nominal pressures are the configured ones
    assert scan["configurations"][0]["nominal_peak_pressure_MPa"] == pytest.approx(5.0)


def test_delta_ratio_is_load_invariant_in_the_linear_model():
    """Two different in-envelope working points give the same Delta%."""
    low = _scan(8, target_w_over_t=0.1)
    high = _scan(8, target_w_over_t=0.4)
    for a, b in zip(low["configurations"], high["configurations"]):
        assert a["w_over_t_nonuniform"] == pytest.approx(0.1, rel=1e-9)
        assert b["w_over_t_nonuniform"] == pytest.approx(0.4, rel=1e-9)
        da = a["delta_nonuniform_over_uniform_pct"]["sigma_x"]
        db = b["delta_nonuniform_over_uniform_pct"]["sigma_x"]
        assert da == pytest.approx(db, rel=1e-9)


def test_fem_ratio_of_maxima_is_not_mesh_converged():
    """Documented defect: the FEM peak ratio moves by >10 pp between nx=8 and 12.

    Measured shift on this plate/config: ~34 percentage points (and it changes
    sign for the 'moderate'/'sharp' configurations).  The quoted Delta therefore
    comes from the analytic Navier path
    (``tools/verify_gap_pp1_delta_navier.py``), not from a FEM peak hunt.
    """
    coarse = _scan(8, configurations=[MILD])
    finer = _scan(12, configurations=[MILD])
    d8 = coarse["configurations"][0]["delta_nonuniform_over_uniform_pct"]["sigma_x"]
    d12 = finer["configurations"][0]["delta_nonuniform_over_uniform_pct"]["sigma_x"]
    assert math.isfinite(d8) and math.isfinite(d12)
    assert abs(d12 - d8) > 10.0, (
        "the FEM peak ratio became mesh stable; revisit "
        "docs/VERIFY-GAP-PP-1-delta-caliber.md and the analytic-path claim"
    )
    # both are still inside the envelope, so the shift is not an envelope artefact
    for scan in (coarse, finer):
        assert scan["summary"]["all_in_small_deflection_envelope"] is True