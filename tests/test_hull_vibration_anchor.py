"""Tests for the external-anchor audit kernel of ``hull-vibration-analytic-kernel``.

The tests enforce the *discipline*, not a wished-for outcome:

* the artifact on disk is exactly what the script produces (no hand-edited numbers),
* the citation promotion gate refuses every currently registered candidate,
* the same-口径 conversion chain is recomputable below the 1e-9 residual gate,
* and the honest verdicts (C1 withheld, C4 cross-registration inconsistency)
  cannot silently flip to PASS without new evidence.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "benchmarks" / "anchors" / "hull_vibration_anchor.py"
ARTIFACT = REPO / "benchmarks" / "anchors" / "hull-vibration-analytic-kernel.json"


def load_module():
    spec = importlib.util.spec_from_file_location("hull_vibration_anchor", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return load_module()


@pytest.fixture(scope="module")
def artifact():
    assert ARTIFACT.exists(), f"missing artifact: run {MODULE_PATH.name} first"
    return json.loads(ARTIFACT.read_text())


def test_artifact_matches_regenerated_payload(mod, artifact):
    """The JSON must be script-generated; a hand edit breaks this test."""
    fresh = mod.build_artifact(write=False, figures=False)
    on_disk = dict(artifact)
    fresh = dict(fresh)
    on_disk.pop("figures", None)
    fresh.pop("figures", None)
    assert json.dumps(on_disk, sort_keys=True) == json.dumps(fresh, sort_keys=True)


def test_criteria_are_complete(artifact):
    ids = [c["id"] for c in artifact["criteria"]]
    assert ids == ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
    for c in artifact["criteria"]:
        assert c["claim"] and c["threshold"] and "measured" in c
        assert c.get("verdict", c["status"]) in {"PASS", "FAIL", "WITHHELD"}


def test_anchor_stays_withheld_without_verbatim_source(artifact):
    anchor = artifact["external_anchor"]
    assert anchor["verification_status"] == "WITHHELD"
    assert anchor["n_verified_records"] == 0
    c1 = next(c for c in artifact["criteria"] if c["id"] == "C1")
    assert c1["status"] == "WITHHELD"
    assert c1["verdict"] == "FAIL"
    assert artifact["overall_status"] == "FAIL"
    # the gap must be stated, not hidden
    assert "WITHHELD" in artifact["gap_statement"]
    assert artifact["stance"]["independent_anchor"] is None


def test_promotion_gate_refuses_uncited_candidates(mod):
    records = mod.candidate_registry()
    assert records
    for rec in records:
        with pytest.raises(mod.AnchorNotVerifiable):
            mod.promotion_gate(rec)
    c5 = next(c for c in mod.build_artifact(write=False, figures=False)["criteria"]
              if c["id"] == "C5")
    assert c5["status"] == "PASS"
    assert c5["measured"] == len(records)


def test_no_candidate_value_is_used_as_reference(artifact):
    assert artifact["external_anchor"]["candidate_values_used_as_reference"] == []
    for rec in artifact["external_anchor"]["candidate_registrations"]:
        assert rec["usable_for_compliance_claim"] is False
        assert rec["status"] == "UNCITED_CANDIDATE"
        assert rec["missing_citation_pillars"]


def test_conversion_chain_is_recomputable(mod):
    freqs = [1.0, 2.81955969, 7.5, 15.2375792, 80.0]
    worst, rows = mod.tone_round_trip_residual(freqs)
    assert worst <= mod.GATE_RECONCILE_RESIDUAL
    assert len(rows) == len(freqs)
    # an explicit hand-computable anchor: 0.01 m/s^2 at 7.5 Hz -> 0.2122 mm/s
    row = rows[2]
    assert row["f_hz"] == pytest.approx(7.5)
    assert row["v_rms_mm_s"] == pytest.approx(0.01 / (2 * 3.141592653589793 * 7.5) * 1000.0,
                                              rel=1e-12)
    back = mod.velocity_to_acceleration(row["v_rms_mm_s"], row["f_hz"])
    assert float(back) == pytest.approx(row["a_rms_m_s2"], rel=1e-12)


def test_dimensional_homogeneity_has_no_violation(mod):
    check = mod.check_dimensional_homogeneity()
    assert check["n_violations"] == 0
    assert check["status"] == "PASS"
    assert check["chains"]["velocity_to_acceleration"]["closed"] is True
    assert check["chains"]["acceleration_to_velocity"]["derived_dimension"] == "L^1 T^-1"


def test_cross_registration_inconsistency_is_reported(mod):
    """The two candidate ladders cannot both be right; the gate must catch it."""
    accel = {"A": 0.2, "B": 0.4, "C": 0.8}
    vel = {"A": 4.0, "B": 6.0, "C": 8.0}
    worst, rows = mod.registration_consistency([1.0, 7.5, 80.0], accel, vel)
    assert worst > mod.GATE_REGISTRATION_CONSISTENCY
    # near the blade frequency the two registrations nearly coincide (<10 %) ...
    near = rows[0]["deviation_at_each_frequency"]["7.500"]
    assert near < 0.10
    # ... and at the band edges they differ by an order of magnitude
    assert rows[0]["deviation_at_each_frequency"]["80.000"] > 5.0


def test_conversion_bandwidth_validity_is_monotone(mod):
    rows = mod.bandwidth_validity(7.5, [0.01, 0.05, 0.2])
    errs = [r["relative_error"] for r in rows]
    assert errs[0] < errs[1] < errs[2]
    assert errs[0] < 1e-3
    case_error = mod.build_artifact(write=False, figures=False)[
        "case_value_in_standard_kou_jing"]["applicability_error_at_case_bandwidth"]
    assert case_error <= mod.GATE_REGISTRATION_CONSISTENCY


def test_case_value_is_traceable_to_the_kernel(artifact):
    case = artifact["case_value_in_standard_kou_jing"]
    assert case["f_blade_hz"] == pytest.approx(7.5)
    expected = case["a_w_m_s2"] / (2 * 3.141592653589793 * 7.5) * 1000.0
    assert case["v_w_mm_s_single_frequency_conversion"] == pytest.approx(expected, rel=1e-12)
    assert artifact["case_conventions"]["source"].endswith("run_hull_girder_vibration_chain "
                                                          "(defaults, n_elements=20)")