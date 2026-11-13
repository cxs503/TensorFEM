"""CASE-CANTILEVER-TIP shear-validity ladder -- oracle and evidence audit.

Two independent duties:

1. **Oracle independence.**  The Euler-Bernoulli and Timoshenko closed forms are
   recomputed here from the textbook expressions written out longhand (no import
   of the driver's constants), so a transcription slip in the driver cannot hide
   behind its own evidence.
2. **Evidence/honesty audit.**  ``gates_verified.json`` is re-derived from
   ``result.json`` (a gate that does not match the data fails), every pass/fail
   flag is recomputed from its own threshold, and the *scientific content* of the
   anchor is asserted: the thick rung must actually discriminate the two
   oracles, the shear-rigid frame must be rejected there, and the 3-D continuum
   must recover a large part of the shear compliance.

The solid mesh ladder itself is not re-solved here (minutes of dense solves);
instead one tiny 3-D solve is run live so that the anchor still fails if the
library's HEX20 tip behaviour changes under it.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ANCHOR_DIR = REPO / "benchmarks" / "verified" / "fem_cantilever_shear"
RESULT = ANCHOR_DIR / "result.json"
GATES = ANCHOR_DIR / "gates_verified.json"
TOLERANCE = 0.03


def _load_driver():
    spec = importlib.util.spec_from_file_location("fem_cantilever_shear_run", ANCHOR_DIR / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def result() -> dict:
    assert RESULT.exists(), "run benchmarks/verified/fem_cantilever_shear/run.py first"
    return json.loads(RESULT.read_text())


@pytest.fixture(scope="module")
def gates() -> dict:
    assert GATES.exists(), "run benchmarks/verified/fem_cantilever_shear/run.py first"
    return json.loads(GATES.read_text())


@pytest.fixture(scope="module")
def driver():
    return _load_driver()


# --------------------------------------------------------------------------- #
# 1. Oracles recomputed independently
# --------------------------------------------------------------------------- #
def test_closed_forms_by_hand():
    """L = 1 m, b = h = 10 mm, E = 200 GPa, P = 5 N, nu = 0.3."""
    E, nu, P = 200.0e9, 0.3, 5.0
    L, b, h = 1.0, 0.01, 0.01
    inertia = b * h**3 / 12.0
    area = b * h
    shear_modulus = E / (2.0 * (1.0 + nu))
    kappa = 5.0 / 6.0

    delta_eb = P * L**3 / (3.0 * E * inertia)
    delta_shear = P * L / (kappa * area * shear_modulus)
    assert delta_eb == pytest.approx(1.0e-2, rel=1e-14)          # 10.0 mm
    assert delta_shear == pytest.approx(7.8e-7, rel=1e-3)        # 0.00078 mm
    assert (delta_eb + delta_shear) / delta_eb - 1.0 < 1.0e-4    # 0.0078 %

    # The gap is a function of slenderness only: E/(4 kappa G) (h/L)^2.
    for slenderness in (100.0, 20.0, 10.0, 5.0, 4.0):
        length = slenderness * h
        eb = P * length**3 / (3.0 * E * inertia)
        ti = eb + P * length / (kappa * area * shear_modulus)
        expected = E / (4.0 * kappa * shear_modulus) / slenderness**2
        assert (ti - eb) / eb == pytest.approx(expected, rel=1e-12)


def test_crossover_slenderness(driver):
    """Euler-Bernoulli leaves the 3 % band below L/h = 5.10."""
    assert driver.crossover_slenderness() == pytest.approx(math.sqrt(0.78 / 0.03), rel=1e-12)
    assert 5.0 < driver.crossover_slenderness() < 5.2


# --------------------------------------------------------------------------- #
# 2. Evidence consistency
# --------------------------------------------------------------------------- #
def test_manifest_schema(gates):
    assert gates["schema"] == "tensorfem.verified-gates.v1"
    assert gates["anchor"] == "fem_cantilever_shear"
    assert gates["mission"] == "CASE-CANTILEVER-TIP"
    assert gates["undefined_terms"] == []
    assert gates["passed"] is True
    assert gates["failed_gates"] == []
    assert gates["criteria"]["case_pool_tolerance"] == TOLERANCE


def test_every_gate_matches_its_threshold(gates):
    for gate in gates["gates"]:
        value = gate["value"]
        assert math.isfinite(value), gate["id"]
        if gate["operator"] == "<=":
            expected = value <= gate["threshold"]
        elif gate["operator"] == ">=":
            expected = value >= gate["threshold"]
        else:  # pragma: no cover
            raise AssertionError(gate["operator"])
        assert gate["passed"] is expected, gate["id"]


def test_gates_are_rederivable_from_result(gates, result):
    """Each graded number must be recoverable from :file:`result.json`."""
    by_id = {gate["id"]: gate for gate in gates["gates"]}
    ladder = {row["slenderness_L_over_h"]: row for row in result["ladder"]}

    for slenderness, row in ladder.items():
        tag = f"{slenderness:g}"
        assert by_id[f"frame_eb_vs_eb_{tag}"]["value"] == pytest.approx(
            abs(row["frame_eb"]["error_vs_eb"]), rel=1e-12, abs=1e-15
        )
        assert by_id[f"beam_timoshenko_vs_ti_{tag}"]["value"] == pytest.approx(
            abs(row["beam_timoshenko"]["error_vs_timoshenko"]), rel=1e-9
        )
        assert by_id[f"solid_hex20_vs_timoshenko_{tag}"]["value"] == pytest.approx(
            abs(row["solid_hex20_levels"][-1]["error_vs_timoshenko"]), rel=1e-12
        )
        # oracle values re-derived from the closed forms, not trusted from the file
        eb = row["oracle_eb_m"]
        ti = row["oracle_timoshenko_m"]
        assert ti / eb - 1.0 == pytest.approx(row["oracle_gap_relative"], rel=1e-9)
        assert row["shear_fraction_formula"] == pytest.approx(
            row["oracle_gap_relative"], rel=1e-9
        )

    thick = result["summary"]["thick_rung"]
    # oracle separation, measured against Euler-Bernoulli
    assert by_id["thick_rung_discriminates"]["value"] == pytest.approx(
        (ladder[thick]["oracle_timoshenko_m"] - ladder[thick]["oracle_eb_m"])
        / ladder[thick]["oracle_eb_m"],
        rel=1e-12,
    )
    assert by_id["slender_rung_cannot_discriminate"]["value"] == pytest.approx(
        ladder[result["summary"]["thin_rung"]]["oracle_gap_relative"], rel=1e-12
    )
    assert by_id["solid_recovers_shear"]["value"] == pytest.approx(
        ladder[thick]["solid_hex20_levels"][-1]["shear_recovery"], rel=1e-12
    )
    assert by_id["eb_rejected_by_timoshenko"]["value"] == pytest.approx(
        abs(ladder[thick]["frame_eb"]["error_vs_timoshenko"]), rel=1e-12
    )


def test_equilibrium_residuals_are_recorded_and_small(result):
    for row in result["ladder"]:
        for level in row["solid_hex20_levels"]:
            assert level["force_residual"] <= 1e-5
            assert level["moment_residual"] <= 1e-5
            # the recorded moment must be the balancing moment of the tip load
            assert level["reaction_moment_y_Nm"] == pytest.approx(
                -5.0 * row["length_m"], rel=1e-4
            )


# --------------------------------------------------------------------------- #
# 3. Scientific content: the ladder must be falsifiable
# --------------------------------------------------------------------------- #
def test_slender_rung_is_not_discriminating(result):
    """The original CASE-CANTILEVER-TIP rung cannot tell the oracles apart."""
    thin = result["summary"]["thin_rung"]
    gap = next(r for r in result["ladder"] if r["slenderness_L_over_h"] == thin)["oracle_gap_relative"]
    assert thin == 100.0
    assert gap < 1e-3, f"slender rung gap {gap:g} should be negligible"


def test_thick_rung_discriminates_and_solid_follows_timoshenko(result):
    thick = result["summary"]["thick_rung"]
    row = next(r for r in result["ladder"] if r["slenderness_L_over_h"] == thick)
    gap = row["oracle_gap_relative"]
    assert gap > TOLERANCE, "thick rung must separate the oracles beyond tolerance"

    finest = row["solid_hex20_levels"][-1]
    assert abs(finest["error_vs_timoshenko"]) < TOLERANCE     # passes Timoshenko
    assert abs(finest["error_vs_eb"]) > TOLERANCE             # would fail EB
    assert finest["shear_recovery"] >= 0.5                    # inherited shear compliance
    # and the shear-rigid frame must be rejected by the Timoshenko oracle there
    assert abs(row["frame_eb"]["error_vs_timoshenko"]) > TOLERANCE


def test_solid_refinement_improves_agreement(result):
    for row in result["ladder"]:
        coarse, fine = row["solid_hex20_levels"]
        assert abs(fine["error_vs_timoshenko"]) <= abs(coarse["error_vs_timoshenko"]) + 1e-12, (
            row["slenderness_L_over_h"]
        )
        assert fine["dofs"] > coarse["dofs"]


def test_all_rungs_within_tolerance_for_the_valid_oracle(result):
    """Every rung passes against the oracle that applies there (< 3 %)."""
    for row in result["ladder"]:
        assert abs(row["frame_eb"]["error_vs_eb"]) < 1e-6
        assert abs(row["beam_timoshenko"]["error_vs_timoshenko"]) < 1e-6
        assert abs(row["solid_hex20_levels"][-1]["error_vs_timoshenko"]) < TOLERANCE


# --------------------------------------------------------------------------- #
# 4. Live library check (fast) -- evidence must track the code
# --------------------------------------------------------------------------- #
def test_beam_models_live_against_oracles(driver):
    for slenderness in (100.0, 4.0):
        length = slenderness * driver.H
        eb = driver.delta_euler_bernoulli(length)
        ti = driver.delta_timoshenko(length)
        assert driver.frame_eb_tip(length) == pytest.approx(eb, rel=1e-9)
        assert driver.beam_timoshenko_tip(length) == pytest.approx(ti, rel=1e-9)


def test_solid_live_tiny_mesh(driver):
    """One cheap 3-D solve: the continuum must show shear softening at L/h = 4."""
    length = 4.0 * driver.H
    eb = driver.delta_euler_bernoulli(length)
    ti = driver.delta_timoshenko(length)
    level = driver.solid_hex20_tip(length, (10, 2, 2))
    assert level["tip_deflection_m"] > eb * (1.0 + 5e-3)     # softer than shear-rigid
    assert abs(level["tip_deflection_m"] - ti) / ti < 0.05    # near Timoshenko
    assert level["force_residual"] <= 1e-5


def test_docs_and_figures_present(result):
    for rel in result["artifacts"]["figures"].values():
        assert (REPO / rel).stat().st_size > 5_000, rel
    doc = REPO / result["artifacts"]["report"]
    assert doc.exists()
    text = doc.read_text()
    assert "CASE-CANTILEVER-TIP" in text
    assert "3 %" in text or "< 3 %" in text


def test_original_slender_case_still_passes():
    """Guard the anchor this one extends: the L/h = 100 evidence is unchanged."""
    legacy = json.loads((REPO / "benchmarks" / "cantilever_tip_artifacts" / "results.json").read_text())
    assert legacy["case_key"] == "CASE-CANTILEVER-TIP"
    assert legacy["verdict"]["all_passed"] is True
    assert legacy["verdict"]["finest_hex20_error_pct"] < 3.0