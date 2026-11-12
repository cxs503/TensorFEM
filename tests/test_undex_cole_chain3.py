"""Invariants of the W24 / chain-3 UNDEX artifacts (Cole anchor closure).

These tests do not re-run the FEM: they re-derive every verdict from the raw
numbers stored in the artifacts, so a hand-edited PASS cannot survive them.

1. the anchor is honestly classified WITHHELD and no citation is fabricated;
2. the declared laws are re-derivable from their own values (coefficients and
   exponents recovered by regression to machine precision);
3. the household-level result of the wall-peak comparison is structural
   (peak == 2 p_m for every case), which is exactly why a PASS against the
   doubled reference must never be read as validation;
4. every PASS/FAIL label in fem_vs_cole.csv follows from the raw deviation and
   the 15 % band, and failing rows are present (no cherry-picking);
5. the grid-convergence file keeps "not demonstrated" distinct from "converged"
   and its verdict is consistent with the per-quantity statuses;
6. the energy ledger is inside 5 %.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
UNDEX = REPO / "benchmarks" / "undex_shock"
ANCHOR = UNDEX / "cole_anchor.json"
CSVF = UNDEX / "fem_vs_cole.csv"
GCI = UNDEX / "grid_convergence.json"
TOL = 0.15
GCI_LIMIT = 0.10
ENERGY_LIMIT = 0.05


@pytest.fixture(scope="module")
def anchor():
    return json.loads(ANCHOR.read_text())


@pytest.fixture(scope="module")
def rows():
    if not CSVF.exists():
        pytest.skip("run scripts/w24_undex_fem_vs_cole.py first")
    with CSVF.open() as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def gcidx():
    if not GCI.exists():
        pytest.skip("run scripts/w24_undex_fem_vs_cole.py first")
    return json.loads(GCI.read_text())


# ---------------------------------------------------------------------------
# 1. anchor honesty
# ---------------------------------------------------------------------------
def test_anchor_is_withheld_and_not_fabricated(anchor):
    assert anchor["acquisition_status"] == "WITHHELD"
    assert anchor["usable_for_physics_validation"] is False
    assert anchor["usable_for_transcription_fidelity_check"] is True
    for key in ("K", "A", "K_theta", "decay_exponent", "K_b"):
        e = anchor["citation_ledger"][key]
        assert e["citation_status"] == "WITHHELD"
        assert e["chapter"] is None and e["table"] is None and e["page"] is None
        assert e["required_deposition"], f"{key} must list what a deposit needs"
    assert anchor["citation_ledger"]["bubble_exponent"]["citation_status"] \
        == "DERIVED_NOT_NEEDED"


def test_withheld_ledger_lists_the_shopping_list(anchor):
    led = anchor["withheld_ledger"]
    assert led["verdict"] == "WITHHELD"
    assert set(led["constants_withheld"]) >= {"K", "A", "K_theta", "K_b"}
    assert len(led["deposition_shopping_list"]) >= 4


# ---------------------------------------------------------------------------
# 2. the laws are recoverable from their own values
# ---------------------------------------------------------------------------
def test_independent_recompute_closes(anchor):
    ir = anchor["independent_recompute"]
    assert ir["anchor_selfcheck_max_rel_dev"] < 1e-9
    P = anchor["parameters"]
    assert abs(ir["peak_fit"]["slope"] - P["A"]["value"]) / P["A"]["value"] < 1e-9
    assert abs(ir["peak_fit"]["prefactor"] - P["K"]["value"]) / P["K"]["value"] < 1e-9
    assert abs(ir["decay_fit"]["slope_reduced_range"]
               - P["decay_exponent"]["value"]) < 1e-9
    assert abs(ir["bubble_head_fit"]["slope"] + 5.0 / 6.0) < 1e-9
    assert abs(ir["bubble_weight_fit"]["slope"] - 1.0 / 3.0) < 1e-9


def test_head_convention_cannot_move_a_verdict(anchor):
    rel = anchor["unit_conventions"]["head_offset_sensitivity"]["max_rel_dev_in_T"]
    assert rel < 0.02, "competing head conventions must stay far inside 15 %"


def test_unit_conversion_of_K_is_arithmetic_only(anchor):
    uc = anchor["unit_conventions"]["implied_imperial_constants"]
    K = anchor["parameters"]["K"]["value"]
    A = anchor["parameters"]["A"]["value"]
    ref = K * (0.45359237 ** (1.0 / 3.0) / 0.3048) ** A / 6894.757293168
    assert abs(uc["K_psi_lb_ft"] - ref) / ref < 1e-9


# ---------------------------------------------------------------------------
# 3./4. FEM vs Cole table
# ---------------------------------------------------------------------------
def test_peak_vs_free_field_is_structural(rows):
    """Wall peak == 2 p_m for every case: the comparison carries no validation."""
    for r in rows:
        assert abs(float(r["fem_wall_peak_over_cole_free_field"]) - 2.0) < 1e-9
        assert abs(float(r["dev_peak_vs_cole_free_field"]) - 1.0) < 1e-9
        assert r["crit_peak_free_field"] == "FAIL"
        assert r["crit_peak_rigid_tautological"].startswith("PASS_TAUTOLOGICAL")


def test_verdicts_follow_from_raw_numbers(rows):
    assert len(rows) >= 4
    for r in rows:
        for col, dev_col in (("crit_impulse_like_for_like",
                              "dev_impulse_vs_cole_incident"),
                             ("crit_rigid_limit",
                              "dev_rigid_limit_impulse_vs_2x_cole"),
                             ("crit_energy", "energy_residual_full_worst_level")):
            expect = "PASS" if float(r[dev_col]) <= TOL else "FAIL"
            if col == "crit_energy":
                expect = ("PASS" if float(r[dev_col]) < ENERGY_LIMIT else "FAIL")
            assert r[col] == expect, (r["case_id"], col, r[dev_col], r[col])
    # honest reporting: failures must be present, not filtered out
    assert any(r["crit_impulse_like_for_like"] == "FAIL" for r in rows), \
        "chain-3 documents that the delivered impulse departs from the " \
        "free-field Cole impulse; a table with no FAIL is suspicious"


def test_cole_inputs_match_the_anchor(anchor, rows):
    K = anchor["parameters"]["K"]["value"]
    A = anchor["parameters"]["A"]["value"]
    for r in rows:
        W, R = float(r["weight_kg"]), float(r["range_m"])
        expect = K * (W ** (1.0 / 3.0) / R) ** A
        got = float(r["cole_peak_pressure_mpa"]) * 1e6
        assert abs(got - expect) / expect < 1e-9
        # same-window Cole incident impulse must be the closed form
        pm = got
        th = float(r["cole_decay_time_us"]) * 1e-6
        t = float(r["impulse_window_s"])
        ref = pm * th * (1.0 - math.exp(-t / th))
        assert abs(float(r["cole_incident_impulse_window_pa_s"]) - ref) / ref < 1e-9


def test_rigid_limit_recovers_doubling(rows):
    """A x1e4 stiffened panel is a *limit*, not the exact rigid wall.

    Its wall peak still sits a couple of per cent below 2 p_m (residual motion
    on the finest dofs), so the impulse band is the meaningful assertion; the
    peak is required only to be close, and the small offset is recorded in the
    csv rather than asserted away.
    """
    for r in rows:
        assert float(r["dev_rigid_limit_impulse_vs_2x_cole"]) < 0.01
        assert float(r["dev_rigid_limit_peak_vs_2pm"]) < 0.05
        assert float(r["dev_rigid_limit_impulse_vs_2x_cole"]) \
            < float(r["dev_impulse_vs_cole_rigid_2x"])


# ---------------------------------------------------------------------------
# 5./6. grid convergence and energy
# ---------------------------------------------------------------------------
def test_gci_statuses_are_not_score_zero(gcidx):
    statuses = set()
    for case in gcidx["cases"].values():
        for q, g in case["gci"].items():
            statuses.add(g["status"])
            if g["status"] == "NOT_DEMONSTRATED":
                assert not g["available"]
                assert g["status"] != "CONVERGED_UNDER_LIMIT"
            if g["status"] == "CONVERGED_UNDER_LIMIT":
                assert g["available"] and g["gci_fine"] < GCI_LIMIT
    assert statuses <= {"CONVERGED_UNDER_LIMIT", "ABOVE_LIMIT",
                        "NOT_DEMONSTRATED"}
    c4 = gcidx["criterion4"]
    expect = "PASS" if (not c4["above_limit"] and not c4["not_demonstrated"]) \
        else "FAIL"
    assert c4["verdict"] == expect


def test_energy_ledger_inside_band(gcidx):
    for cid, case in gcidx["cases"].items():
        led = case["energy_ledger"]
        assert led["residual_full_record_worst_level"] < ENERGY_LIMIT, cid
        assert led["verdict"] == "PASS"
    assert gcidx["criterion3"]["verdict"] == "PASS"


def test_three_levels_and_observed_order_reported(gcidx):
    assert len(gcidx["levels"]) == 3
    assert abs(gcidx["refinement_ratio"] - 1.5) < 1e-12
    for case in gcidx["cases"].values():
        for g in case["gci"].values():
            if g["available"]:
                assert "observed_order" in g