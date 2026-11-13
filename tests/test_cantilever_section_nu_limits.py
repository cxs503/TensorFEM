#!/usr/bin/env python3
"""Regression gates for CASE-CANTILEVER-TIP section / Poisson study.

The study itself (``benchmarks/verified/fem_cantilever_shear/section_nu_limits.py``)
is a few minutes of dense solves, so the suite does three cheap things:

1. re-derives the closed-form oracles from scratch (no artifact needed) and
   checks the identities the document leans on;
2. re-reads the recorded artifacts and recomputes every derived column from the
   raw deflections, so a hand-edited or drifted number fails the build;
3. re-runs the cheapest recorded 3-D point live and compares it to the artifact.

Run: ``python3 -m pytest tests/test_cantilever_section_nu_limits.py -q``
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "verified"))
sys.path.insert(0, str(REPO / "benchmarks" / "verified" / "fem_cantilever_shear"))

import section_nu_limits as S  # noqa: E402

RESULT = S.OUT_JSON
GATES = S.OUT_GATES
DOC = S.DOC


def _payload() -> dict:
    if not RESULT.exists():
        pytest.skip(f"{RESULT.name} not generated yet (run section_nu_limits.py)")
    return json.loads(RESULT.read_text())


def _gate_payload() -> dict:
    if not GATES.exists():
        pytest.skip(f"{GATES.name} not generated yet (run section_nu_limits.py)")
    return json.loads(GATES.read_text())


# --------------------------------------------------------------------------- #
# 1. Closed forms, re-derived here
# --------------------------------------------------------------------------- #
def test_exact_2d_field_is_timoshenko_with_two_thirds():
    length = S.SLENDERNESS * S.H
    for nu in (0.0, 0.15, 0.30, 0.45):
        exact = S.delta_2d_analytic(length, nu)
        timo = S.delta_ti(length, 2.0 / 3.0, nu)
        assert exact == pytest.approx(timo, rel=1e-13), nu
        # and the module-backed field agrees to its float64 evaluation floor
        module = S.delta_airy_module(length, nu)
        assert module == pytest.approx(exact, rel=1e-6), nu


def test_cowper_factor_matches_published_values():
    # 0.8496 is the published 4-digit rounding of 10*1.3/(12+3.3) = 0.849673
    assert S.kappa_cowper(0.3) == pytest.approx(0.8496, abs=5e-4)
    assert S.kappa_cowper(0.0) == pytest.approx(5.0 / 6.0, abs=1e-15)
    # monotone increasing, but by less than 3 % over the admissible range
    assert S.kappa_cowper(0.45) > S.kappa_cowper(0.0)
    assert S.kappa_cowper(0.45) / S.kappa_cowper(0.0) - 1.0 < 0.03


def test_strip_limits_bracket_plane_stress():
    length = S.SLENDERNESS * S.H
    strain = S.delta_exact_strip(length, 0.30, 4 * S.H, S.H, plane_strain=True)
    stress = S.delta_exact_strip(length, 0.30, 4 * S.H, S.H, plane_strain=False)
    assert strain is not None and stress is not None
    assert strain < stress                      # plane strain is the stiffer limit
    # the substitution nu -> nu/(1-nu) leaves the admissible range above 1/3
    assert S.delta_exact_strip(length, 0.45, 4 * S.H, S.H, plane_strain=True) is None


def test_shear_share_and_apparent_kappa_are_derived_not_recorded():
    payload = _payload()
    for row in payload["studies"]["nu_study"] + payload["studies"]["width_study"]:
        oracles = row["oracles"]
        shear = oracles["delta_ti_5_6_m"] - oracles["delta_eb_m"]
        assert shear > 0.0
        for level in row["levels"]:
            delta = level["tip_deflection_m"]
            share = (delta - oracles["delta_eb_m"]) / shear
            assert level["shear_share_vs_5_6"] == pytest.approx(share, rel=1e-12)
            assert level["kappa_apparent_orientation_only"] == pytest.approx(
                5.0 / 6.0 / share, rel=1e-12)
            for key, oracle_key in (("error_vs_ti_5_6", "delta_ti_5_6_m"),
                                    ("error_vs_ti_2_3", "delta_ti_2_3_m"),
                                    ("error_vs_ti_cowper", "delta_ti_cowper_m"),
                                    ("error_vs_exact_plane_stress",
                                     "delta_exact_plane_stress_m"),
                                    ("error_vs_eb", "delta_eb_m")):
                expected = (delta - oracles[oracle_key]) / oracles[oracle_key]
                assert level[key] == pytest.approx(expected, rel=1e-12), (key, level["mesh"])


# --------------------------------------------------------------------------- #
# 2. Recorded gates
# --------------------------------------------------------------------------- #
def test_gate_artifact_is_self_consistent():
    gate_payload = _gate_payload()
    payload = _payload()
    gates = gate_payload["gates"]
    assert [g["id"] for g in gates] == [g["id"] for g in payload["gates"]]
    assert len({g["id"] for g in gates}) == len(gates), "duplicate gate ids"
    for gate in gates:
        if gate["operator"] == "<=":
            recomputed = gate["value"] <= gate["threshold"]
        elif gate["operator"] == ">=":
            recomputed = gate["value"] >= gate["threshold"]
        else:                                   # pragma: no cover - schema guard
            raise AssertionError(f"unsupported operator {gate['operator']}")
        assert bool(gate["passed"]) is bool(recomputed), gate["id"]
    assert gate_payload["criteria"]["case_pool_tolerance"] == S.TOLERANCE
    assert gate_payload["criteria"]["equilibrium_max"] == S.EQUILIBRIUM_MAX
    assert gate_payload["passed"] == sum(1 for g in gates if g["passed"])


def test_documented_oracles_are_verifiable_gates():
    gates = {g["id"]: g for g in _gate_payload()["gates"]}
    must_pass = (
        "exact2d_is_ti_2_3",
        "airy_module_matches_closed_form",
        "cowper_value_at_nu_0p3",
        "cowper_limit_at_nu_0_is_5_6",
        "force_equilibrium_all_solves",
        "moment_equilibrium_all_solves",
        "mesh_sensitivity_bound",
    )
    for gate_id in must_pass:
        assert gate_id in gates, gate_id
        assert gates[gate_id]["passed"], (gate_id, gates[gate_id]["value"],
                                          gates[gate_id]["threshold"])


def test_anchor_declared_point_keeps_the_case_pool_verdict():
    """The case entry is `nu = 0.3`, `b/h = 1`: that point must stay inside 3 %.

    The sweep deliberately goes outside the declared material and section, and
    the artifact records where the tolerance breaks; that is the finding, not a
    regression.
    """
    payload = _payload()
    declared = [row for row in payload["studies"]["nu_study"]
                if abs(row["poisson"] - 0.30) < 1e-12]
    assert declared
    level = declared[0]["levels"][1]
    assert abs(level["error_vs_ti_5_6"]) < S.TOLERANCE
    # the wide / high-Poisson corners must be recorded as out of tolerance
    payload_gate = {g["id"]: g for g in payload["gates"]}["case_pool_tolerance_vs_ti_5_6"]
    assert payload_gate["value"] > S.TOLERANCE


def test_mesh_pairs_are_recorded_for_every_point():
    payload = _payload()
    for row in payload["studies"]["nu_study"] + payload["studies"]["width_study"]:
        meshes = [tuple(sorted(level["mesh"].items())) for level in row["levels"]]
        assert len(meshes) == 2 and meshes[0] != meshes[1]


# --------------------------------------------------------------------------- #
# 3. One live solve, and the document
# --------------------------------------------------------------------------- #
def test_cheapest_recorded_point_reproduces_live():
    payload = _payload()
    live = payload["studies"]["live_recompute"]
    assert live is not None
    length = payload["studies"]["length_m"]
    oracles = None
    for row in payload["studies"]["nu_study"]:
        if abs(row["poisson"] - S.LIVE_NU) < 1e-12:
            oracles = row["oracles"]
    assert oracles is not None
    redo = S.grade_point(length, S.B, S.H, S.LIVE_NU,
                         (live["mesh"]["axial"], live["mesh"]["cross_y"],
                          live["mesh"]["cross_z"]), oracles)
    assert redo["tip_deflection_m"] == pytest.approx(live["tip_deflection_m"], rel=1e-9)
    assert redo["force_residual"] <= S.EQUILIBRIUM_MAX
    assert redo["moment_residual"] <= S.EQUILIBRIUM_MAX


def test_document_is_generated_from_the_artifact():
    payload = _payload()
    if not DOC.exists():
        pytest.skip("document not generated yet")
    text = DOC.read_text()
    assert "Every number below is injected from" in text
    for row in payload["studies"]["width_study"]:
        level = row["levels"][1]
        assert f"{level['tip_deflection_m']:.6e}" in text, row["width_over_height"]
    failures = [g["id"] for g in payload["gates"] if not g["passed"]]
    for gate_id in failures:
        assert gate_id in text
    assert f"{payload['passed_count']} / {len(payload['gates'])} gates PASS" in text