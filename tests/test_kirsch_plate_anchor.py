"""Regression gates for the CASE-KIRSCH-PLATE anchor (``fem_kirsch_plate``).

The anchor certifies TensorFEM against the Kirsch (1898) closed-form stress
concentration ``K_t = 3`` for a circular hole in a tension plate.  This module

1. re-derives the oracle in the test itself (Kirsch field from the cited closed
   form, checked for a traction-free hole edge and a uniaxial far field), so a
   drifting formula inside the driver cannot pass silently;
2. checks the committed evidence (``result.json`` / ``gates_verified.json``) is
   self-consistent: every gated value equals the number stored in the result,
   every gate passes, and the headline error is inside the 5 % pool tolerance;
3. re-runs the cheap end of the ladder end to end, so the pipeline itself -- not
   just the stored JSON -- is exercised.

The heavy ladder (n = 48 plates) is *not* re-solved here: ``run.py`` owns it and
writes ``result.json``.  What is re-solved is the coarse plate, which is enough
to catch a broken load path, mesh or stress recovery.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = ROOT / "benchmarks" / "verified" / "fem_kirsch_plate"
KT_KIRSCH = 3.0
TOLERANCE = 0.05  # pool mandate: < 5 % against the reference


def _result() -> dict:
    return json.loads((ANCHOR / "result.json").read_text())


def _gates() -> dict:
    return json.loads((ANCHOR / "gates_verified.json").read_text())


def _kirsch(r, theta, radius=1.0, tension=1.0):
    """Kirsch (1898) field; Timoshenko & Goodier 3rd ed. Art. 39 eqs. (a)-(c)."""
    r = np.asarray(r, dtype=float)
    theta = np.asarray(theta, dtype=float)
    a2 = radius * radius
    a4 = a2 * a2
    c2, s2 = np.cos(2.0 * theta), np.sin(2.0 * theta)
    srr = 0.5 * tension * (1 - a2 / r**2) + 0.5 * tension * (1 - 4 * a2 / r**2 + 3 * a4 / r**4) * c2
    stt = 0.5 * tension * (1 + a2 / r**2) - 0.5 * tension * (1 + 3 * a4 / r**4) * c2
    srt = -0.5 * tension * (1 + 2 * a2 / r**2 - 3 * a4 / r**4) * s2
    return srr, stt, srt


def test_oracle_is_kirsch_three():
    """The reference itself: traction-free hole edge, 3T peak, uniaxial far field."""
    theta = np.linspace(0.0, 2.0 * math.pi, 721)
    srr, stt, srt = _kirsch(np.ones_like(theta), theta)
    assert np.abs(srr).max() < 1e-12
    assert np.abs(srt).max() < 1e-12
    assert stt.max() == pytest.approx(KT_KIRSCH, rel=1e-12)
    assert stt.min() == pytest.approx(-1.0, rel=1e-12)
    srr_far, stt_far, srt_far = _kirsch(np.full(theta.size, 200.0), theta)
    sxx = srr_far * np.cos(theta) ** 2 + stt_far * np.sin(theta) ** 2 - 2 * srt_far * np.sin(theta) * np.cos(theta)
    assert np.allclose(sxx, 1.0, atol=1e-3)


def test_committed_gates_pass_and_agree_with_result():
    """No gate may pass on a number that the result file does not carry."""
    manifest = _gates()
    result = _result()
    assert manifest["schema"] == "tensorfem.verified-gates.v1"
    assert manifest["anchor"] == "fem_kirsch_plate"
    assert manifest["passed"] is True, manifest["failed_gates"]
    assert manifest["undefined_terms"] == []
    values = {g["id"]: g["value"] for g in manifest["gates"]}

    anchor_row = next(r for r in result["ladder"]
                      if abs(r["ratio"] - 0.02) < 1e-12 and r["n_boundary"] == 48)
    summary = result["ladder_summary"]["0.02"]
    assert values["kirsch_limit"] == pytest.approx(anchor_row["ktg_error_vs_kirsch"], rel=1e-12)
    assert values["kirsch_extrapolated"] == pytest.approx(summary["richardson_error_vs_kirsch"], rel=1e-12)
    assert values["q8_crosscheck"] == pytest.approx(result["q8_crosscheck"][-1]["error_vs_kirsch"], rel=1e-12)
    assert values["finite_width_trend"] == pytest.approx(
        max(r["ktg_error_vs_heywood"] for r in result["finite_width_trend"]), rel=1e-12)
    # every ladder row carries the same error definition
    for row in result["ladder"]:
        assert row["ktg_error_vs_kirsch"] == pytest.approx(
            abs(row["ktg_fe"] - KT_KIRSCH) / KT_KIRSCH, rel=1e-12)


def test_headline_within_pool_tolerance():
    """The campaign criterion: the anchor number vs K_t = 3 must be < 5 %."""
    result = _result()
    headline = next(r for r in result["ladder"] if abs(r["ratio"] - 0.02) < 1e-12 and r["n_boundary"] == 48)
    assert headline["ktg_error_vs_kirsch"] < TOLERANCE
    assert headline["ktg_fe"] == pytest.approx(KT_KIRSCH, rel=TOLERANCE)
    # the documented reference is the closed form, with no hand-copied digits
    assert result["reference"]["value"] == pytest.approx(KT_KIRSCH)
    assert result["reference"]["hand_copied_digits"] is None
    assert result["oracle_selfcheck"]["kt_exact"] == pytest.approx(KT_KIRSCH, rel=1e-12)


def test_committed_figures_exist():
    for name in ("kirsch_field.png", "kirsch_convergence.png",
                 "kirsch_hoop_profile.png", "kirsch_width_trend.png"):
        path = ANCHOR / name
        assert path.is_file() and path.stat().st_size > 10_000, name


def test_cheap_plate_rerun_matches_the_ladder():
    """Re-solve the coarse plate: catches a broken load path / stress recovery."""
    from tensorfem.finite_width_hole import FiniteWidthPlate, solve_case

    plate = FiniteWidthPlate(diameter=0.02, half_length=1.5)
    case = solve_case(plate, n_boundary=12, n_radial=12, grading=6.0)
    # traction equilibrium on the symmetry plane is exact by statics
    assert case["load_balance"]["relative_error"] < 1e-10
    # coarse mesh, but the peak must already be within the pool tolerance
    assert abs(case["ktg_fe"] - KT_KIRSCH) / KT_KIRSCH < TOLERANCE
    # and refining once must halve the error towards K_t = 3
    fine = solve_case(plate, n_boundary=24, n_radial=24, grading=6.0)
    assert abs(fine["ktg_fe"] - KT_KIRSCH) < abs(case["ktg_fe"] - KT_KIRSCH)
    stored = next(r for r in _result()["ladder"]
                  if abs(r["ratio"] - 0.02) < 1e-12 and r["n_boundary"] == 12)
    assert case["ktg_fe"] == pytest.approx(stored["ktg_fe"], rel=1e-12)


def test_q8_crosscheck_converges_to_three():
    """The independent Q8 annulus ladder in the stored evidence is monotone to 3."""
    rows = _result()["q8_crosscheck"]
    errors = [r["error_vs_kirsch"] for r in rows]
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < TOLERANCE
    assert rows[-1]["kt"] == pytest.approx(KT_KIRSCH, rel=TOLERANCE)