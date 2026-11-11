"""Tests of the W23 multi-station hull-girder ULS scan.

The invariants asserted here are the ones the W23 criteria depend on:

* the bending condition is paired with the *compressed* extreme fibre (the W22
  defect this module fixes), verified against the engine's strain definition
  ``eps = kappa (y - y_na)`` rather than against a docstring;
* the station sections respect the declared particulars (``B``, taper range);
* the ULS arithmetic of every case reproduces the clause
  ``gamma_s M_sw + gamma_w M_wv`` to better than ``1e-6``, and the verdict
  follows from the utilisation;
* the grid-convergence estimator recovers a known order and GCI on a synthetic
  second-order sequence;
* the CSR factor register is honestly blocked (no invented factor value).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorfem.hull_girder_csr_w22 import CASE_CATALOGUE, build_scantlings
from tensorfem.hull_girder_smith_collapse import solve_moment_curvature
from tensorfem.hull_girder_uls_scan import (
    CONDITION_COMPRESSED_FIBRE,
    ATTRIBUTION_THRESHOLD,
    STATION_FRACTIONS,
    STEP_LEVELS,
    analyse_condition,
    curvature_sign,
    grid_convergence,
    recompute_residual,
    run_uls_scan,
    scantling_taper,
    section_fibers,
    station_members,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTER = ROOT / "benchmarks" / "hull_girder_uls" / "csr_factor_register.json"


@pytest.fixture(scope="module")
def midship_members():
    case = CASE_CATALOGUE[0]
    _, closure = build_scantlings(case)
    return station_members(case, 0.5, deck_thickness=closure["deck_thickness_final"])[0]


@pytest.fixture(scope="module")
def small_study():
    """A valid but cheap scan (5 sections, 3 levels per axis)."""
    return run_uls_scan(stations=(0.25, 0.35, 0.45, 0.55, 0.65),
                        fibre_levels=(2, 4, 8), step_levels=(80, 160, 320),
                        repo_root=ROOT)


# ---------------------------------------------------------------------------
# pairing of condition and compressed fibre
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("condition,expected_sign", [("sagging", -1.0), ("hogging", 1.0)])
def test_curvature_sign_matches_compressed_fibre(condition, expected_sign):
    assert curvature_sign(condition) == expected_sign
    assert CONDITION_COMPRESSED_FIBRE[condition] == ("deck" if expected_sign < 0 else "bottom")


def test_charged_fibre_is_actually_in_compression(midship_members):
    """Direct strain check with the engine's own convention ``eps = kappa (y - y_na)``."""
    fibers = section_fibers(midship_members, strips=4)
    y_top = max(fiber.y for fiber in fibers)
    y_bottom = min(fiber.y for fiber in fibers)
    for condition in ("sagging", "hogging"):
        sign = curvature_sign(condition)
        extent = max(abs(fiber.y - sum(f.y for f in fibers) / len(fibers)) for fiber in fibers)
        kappa = sign * 0.2 * (235.0e6 / 206.0e9) / extent
        result = solve_moment_curvature(fibers, (0.0, 0.5 * kappa, kappa))
        y_na = result.points[1].neutral_axis
        strain_top = kappa * (y_top - y_na)
        strain_bottom = kappa * (y_bottom - y_na)
        if CONDITION_COMPRESSED_FIBRE[condition] == "deck":
            assert strain_top < 0.0, "deck must be in compression for sagging"
            assert strain_bottom > 0.0
        else:
            assert strain_bottom < 0.0, "bottom must be in compression for hogging"
            assert strain_top > 0.0
        # and the moment must carry the sign of the curvature
        assert math.copysign(1.0, result.points[1].moment) == sign


def test_capacity_ordering_follows_the_compressed_fibre(midship_members):
    """The deck-compression capacity is the smaller one for this tanker section."""
    sagging = analyse_condition(midship_members, condition="sagging", strips=6, steps=160)
    hogging = analyse_condition(midship_members, condition="hogging", strips=6, steps=160)
    assert sagging["M_u_Nm"] < hogging["M_u_Nm"]
    assert sagging["compressed_fibre"] == "deck" and hogging["compressed_fibre"] == "bottom"


# ---------------------------------------------------------------------------
# geometry of the station sections
# ---------------------------------------------------------------------------
def test_station_sections_respect_declared_particulars():
    case = CASE_CATALOGUE[0]
    _, closure = build_scantlings(case)
    for x_fraction in STATION_FRACTIONS:
        members, station = station_members(case, x_fraction,
                                           deck_thickness=closure["deck_thickness_final"])
        assert station["station_breadth_m"] <= case.B + 1e-9
        assert station["station_breadth_m"] > 0.0
        assert 0.80 - 1e-12 <= station["taper_factor"] <= 1.0 + 1e-12
        assert sum(member.total_area for member in members) > 0.0
        assert all(member.plate_thickness > 0.0 for member in members)


def test_taper_is_flat_in_the_parallel_body_and_symmetric():
    assert scantling_taper(0.5) == 1.0
    assert scantling_taper(0.4) == pytest.approx(scantling_taper(0.6))
    assert scantling_taper(0.2) == pytest.approx(scantling_taper(0.8))
    assert scantling_taper(0.2) < scantling_taper(0.4)
    with pytest.raises(ValueError):
        scantling_taper(0.0)


# ---------------------------------------------------------------------------
# ULS arithmetic and criteria
# ---------------------------------------------------------------------------
def test_clause_arithmetic_and_verdict(small_study):
    assert len(small_study["cases"]) == 10
    assert len({row["station_x_fraction"] for row in small_study["cases"]}) == 5
    for row in small_study["cases"]:
        lhs = row["gamma_s"] * row["M_sw_component_Nm"] + row["gamma_w"] * row["M_wv_component_Nm"]
        assert lhs == pytest.approx(row["lhs_Nm"], rel=1e-12)
        assert row["utilisation"] == pytest.approx(lhs / row["M_u_smith_Nm"], rel=1e-12)
        assert row["verdict"] == ("PASS" if row["utilisation"] <= 1.0 else "FAIL")
        assert recompute_residual(row) < 1e-6
        assert row["gamma_admissible"] == pytest.approx(row["M_u_smith_Nm"] / lhs)


def test_elastic_and_smith_capacities_are_reported_per_case(small_study):
    for row in small_study["cases"]:
        assert row["M_y_elastic_Nm"] > 0.0
        ratio = row["M_u_smith_Nm"] / row["M_y_elastic_Nm"]
        assert ratio == pytest.approx(row["M_u_over_M_y"], rel=1e-12)
        if abs(ratio - 1.0) > ATTRIBUTION_THRESHOLD:
            assert row["attribution"]["attribution_required"] is True
            assert row["dominant_mechanism"] != "unknown"
            assert row["controlling_member"]["member"]


def test_criteria_verdicts_are_honest(small_study):
    criteria = {item["id"]: item for item in small_study["criteria"]}
    assert criteria["W23-C1"]["verdict"] == "PASS"
    assert criteria["W23-C2a"]["verdict"] == "PASS"
    # the CSR factor table is not available in this environment: must stay FAIL
    assert criteria["W23-C2b"]["verdict"] == "FAIL"
    assert criteria["W23-C3"]["verdict"] == "PASS"
    assert criteria["W23-C4"]["verdict"] == "PASS"
    assert criteria["W23-C2a"]["value"]["max_residual"] < 1e-6


def test_fewer_than_five_sections_is_rejected():
    with pytest.raises(ValueError):
        run_uls_scan(stations=(0.3, 0.4, 0.5, 0.6), repo_root=ROOT)


def test_fewer_than_three_grid_levels_is_rejected():
    with pytest.raises(ValueError):
        run_uls_scan(stations=STATION_FRACTIONS, fibre_levels=(4, 8),
                     step_levels=STEP_LEVELS, repo_root=ROOT)


# ---------------------------------------------------------------------------
# grid convergence estimator
# ---------------------------------------------------------------------------
def test_gci_recovers_a_known_second_order_sequence():
    """f(h) = 1 + h^2 sampled at h = 1, 1/2, 1/4: order 2, GCI = 1.25 * 5.88%."""
    values = [1.0 + h ** 2 for h in (1.0, 0.5, 0.25)]
    result = grid_convergence(*values)
    assert result["order_observed"] == pytest.approx(2.0, rel=1e-12)
    assert result["gci_fine_percent"] == pytest.approx(7.35, rel=1e-3)
    assert result["verdict"] == "FAIL"  # honest: 7.35% > 5%


def test_gci_flags_oscillation_instead_of_inventing_an_order():
    result = grid_convergence(1.0, 0.90, 0.95)
    assert result["behaviour"] == "oscillatory"
    assert result["order_used"] is None
    assert result["gci_fine_percent"] == pytest.approx(100 * 1.25 * 0.05 / 0.95)


# ---------------------------------------------------------------------------
# factor register honesty
# ---------------------------------------------------------------------------
def test_csr_factor_register_is_blocked_and_holds_no_values():
    payload = json.loads(REGISTER.read_text())
    assert payload["status"] == "BLOCKED"
    assert payload["acquisition_attempts"]
    ids = {factor["id"] for factor in payload["factors"]}
    assert {"gamma_s", "gamma_w", "gamma_R"} <= ids
    for factor in payload["factors"]:
        assert factor["status"] == "BLOCKED"
        assert factor["printed_values"] is None
        assert factor["usable_for_compliance_claim"] is False


def test_study_states_the_unit_factor_status(small_study):
    assert small_study["settings"]["gamma_status"].startswith("BLOCKED")
    for row in small_study["cases"]:
        assert row["gamma_s"] == 1.0 and row["gamma_w"] == 1.0