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
    """Every declared figure must exist at a *publisher-safe* (tracked) path.

    The mission publisher drops ``*.png`` outside ``docs/``, so a figure written next to
    ``run.py`` never reaches the repository: asserting on that copy made this gate
    permanently red on any clean checkout (finding of this round).  The driver now writes
    the figures into ``docs/assets/kirsch_plate/`` and ``result.json`` declares those
    paths, so the assertion is on the file the repository actually ships.
    """
    result = _result()
    declared = result["figures"]
    assert len(declared) == 5, declared
    for relative in declared:
        path = ROOT / relative
        assert relative.startswith("docs/"), f"figure outside docs/ would be filtered: {relative}"
        assert path.is_file() and path.stat().st_size > 10_000, relative


def test_interior_field_gate_reproduces_kirsch_away_from_the_edge():
    """The field-level gate: recovered stress vs the full Kirsch field at interior points."""
    manifest = _gates()
    result = _result()
    study = result["interior_field"]
    summary = study["summary"]
    gates = {g["id"]: g for g in manifest["gates"]}
    gate = gates["interior_field_recovered"]
    assert gate["passed"] is True
    assert gate["threshold"] == manifest["criteria"]["interior_field_rel_error_max"] == 0.05
    assert gate["value"] == pytest.approx(summary["max_error_recovered"], rel=1e-12)
    assert summary["max_error_recovered"] < TOLERANCE
    # non-vacuous: the coarse level of the same probe set must break the 5 % gate,
    # otherwise the gate would not be able to discriminate a bad mesh
    assert summary["coarse_max_error_recovered"] > TOLERANCE
    refinement = gates["interior_field_refinement"]
    assert refinement["passed"] is True
    assert refinement["value"] == pytest.approx(summary["refinement_ratio"], rel=1e-12)
    assert summary["refinement_ratio"] >= manifest["criteria"]["interior_field_refinement_factor_min"]
    # probes sit strictly outside the hole and off the two symmetry axes
    for level in study["levels"]:
        rows = study["probes"][str(level)]["probes"]
        assert len(rows) == len(study["radii_over_a"]) * len(study["rays_degrees"])
        for row in rows:
            assert row["r_over_a"] > 1.0
            assert min(abs(row["theta_degrees"] - axis) for axis in (0.0, 90.0, 180.0)) > 1e-6
            # the stored "exact" column must be the cited closed form, re-derived here
            exact = _kirsch(row["r_over_a"], math.radians(row["theta_degrees"]))
            assert row["exact"][0] == pytest.approx(float(exact[0]), rel=1e-12, abs=1e-14)
            assert row["exact"][1] == pytest.approx(float(exact[1]), rel=1e-12, abs=1e-14)
            assert row["exact"][2] == pytest.approx(float(exact[2]), rel=1e-12, abs=1e-14)
            for component, index in (("radial", 0), ("hoop", 1), ("shear", 2)):
                denominator = max(abs(row["exact"][index]), study["absolute_floor_over_T"])
                expected = abs(row["recovered"][index] - row["exact"][index]) / denominator
                assert row["error_recovered"][component] == pytest.approx(expected, rel=1e-12)


def test_symmetry_axis_shear_residual_is_bounded():
    """``tau_rt`` is exactly zero on both symmetry axes; the recovery leaves O(1 %) of T."""
    manifest = _gates()
    study = _result()["interior_field"]
    gate = next(g for g in manifest["gates"] if g["id"] == "symmetry_axis_shear")
    assert gate["passed"] is True
    assert gate["value"] == pytest.approx(study["symmetry_axis_shear_max_abs"], rel=1e-12)
    assert gate["value"] <= 0.05
    for row in study["symmetry_axis_shear"]:
        assert row["exact_shear"] == pytest.approx(0.0, abs=1e-12)


def test_probe_pipeline_is_live_not_just_json():
    """Re-run the probe on the coarse plate: the stored level must be reproduced."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "kirsch_plate_driver", ANCHOR / "run.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)

    from tensorfem.finite_width_hole import FiniteWidthPlate, solve_case

    level = 24
    plate = FiniteWidthPlate(diameter=0.02, half_length=1.5)
    case = solve_case(plate, n_boundary=level, grading=6.0)
    context = driver.probe_context(case, plate)
    stored = _result()["interior_field"]["probes"][str(level)]["probes"]
    row = next(r for r in stored
               if abs(r["r_over_a"] - 1.25) < 1e-12 and abs(r["theta_degrees"] - 45.0) < 1e-9)
    fresh = driver.field_probe(context, 1.25, math.radians(45.0))
    assert fresh["recovered"] == pytest.approx(row["recovered"], rel=1e-9, abs=1e-12)
    assert fresh["raw"] == pytest.approx(row["raw"], rel=1e-9, abs=1e-12)


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

def test_figures_tracked_mirror():
    """Every declared figure needs a version-controlled home under ``docs/``.

    The mission publisher drops ``*.png`` outside ``docs/`` (tp_pusher ``_EXCLUDE2``),
    so the figures declared in ``result.json`` were absent from main after the merge
    (audit m15393, finding F3).  The mirrors under ``docs/assets/kirsch_plate/`` are
    regenerated byte-identically by ``run.py``.
    """
    mirror = ROOT / "docs" / "assets" / "kirsch_plate"
    names = [Path(p).name for p in _result()["figures"]]
    assert names, "result.json must declare the anchor figures"
    for name in names:
        f = mirror / name
        assert f.is_file() and f.stat().st_size > 1000, f"missing figure mirror: {f}"
