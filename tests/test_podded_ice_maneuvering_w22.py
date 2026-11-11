"""Tests for the W22 / m14511 podded-ice manoeuvring study (TensorFEM).

Layers:
1. model/physics invariants that must hold regardless of the (declared) coefficient
   values: geometry gate, anchor cross-check, IMO-limit injection, indicator
   definitions, overshoot convention, time-step convergence;
2. the committed artefact: schema, provenance hashes, criterion table.

The tests are deliberately fast (coarse dt, short horizons); the production run
lives in ``benchmarks/podded_ice_maneuvering_w22.py``.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmarks" / "podded_ice_maneuvering_w22.py"
ANCHOR = REPO / "benchmarks" / "anchors" / "msc137-simman-w22.json"
ART = REPO / "docs" / "w22_podded_ice" / "turning_zigzag.json"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("w22_maneuvering", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["w22_maneuvering"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def anchor(mod):
    return mod.load_anchor()


@pytest.fixture(scope="module")
def models(mod, anchor):
    return mod.build_models(anchor)


# --------------------------------------------------------------------------- #
# 1. anchor file
# --------------------------------------------------------------------------- #
def test_anchor_transcribes_all_imo_criteria(anchor):
    by_id = {a["id"]: a for a in anchor["anchors"]}
    expected = {
        "AN-MSC137-01": 4.5,
        "AN-MSC137-02": 5.0,
        "AN-MSC137-03": 10.0,
        "AN-MSC137-04": 20.0,
        "AN-MSC137-05": 25.0,
        "AN-MSC137-06": 40.0,
        "AN-MSC137-07": 100.0,
    }
    for anchor_id, value in expected.items():
        assert by_id[anchor_id]["value"] == value
        assert by_id[anchor_id]["unit"]                      # unit recorded
        assert by_id[anchor_id]["comparator"] in {"<=", ">=", "=", "in-range"}
        assert by_id[anchor_id]["provenance_grade"] in {"A", "B", "C", "MISSING"}


def test_anchor_does_not_invent_simman_numbers(anchor):
    """SIMMAN KVLCC2 numeric baselines must be MISSING, not memory-filled."""
    by_id = {a["id"]: a for a in anchor["anchors"]}
    for anchor_id in ("AN-SIMMAN-03", "AN-SIMMAN-04", "AN-SIMMAN-05"):
        assert by_id[anchor_id]["value"] is None
        assert by_id[anchor_id]["status"] == "MISSING"
        assert by_id[anchor_id]["provenance_grade"] == "MISSING"
    assert anchor["blocked"]["blocked_fields"] == ["AN-SIMMAN-03", "AN-SIMMAN-04",
                                                   "AN-SIMMAN-05"]
    completeness = anchor["completeness"]
    assert completeness["ratio"] < 1.0
    assert completeness["criterion_1_verdict"] == "FAIL"


def test_anchor_source_files_are_hashed(anchor):
    """Every quoted source carries a sha256; hashed files listed here are re-checked."""
    refs = [a["source"] for a in anchor["anchors"] if "file" in a.get("source", {})]
    refs += list(anchor["source_registry"])
    assert refs
    for ref in refs:
        assert len(ref["sha256"]) == 64
        path = Path(ref["file"])
        if not path.exists():        # another checkout of the ecosystem -> not an error
            continue
        import hashlib
        assert hashlib.sha256(path.read_bytes()).hexdigest() == ref["sha256"]


# --------------------------------------------------------------------------- #
# 2. geometry gate + anchor cross-check
# --------------------------------------------------------------------------- #
def test_geometry_gate_accepts_kvlcc2(mod, anchor):
    hull = mod.hull_from_anchor(anchor)
    assert hull.length == 320.0 and hull.beam == 58.0 and hull.draft == 20.8
    assert abs(hull.cb - 0.8098) < 1e-12
    closure = abs(hull.displacement_t_from_geometry() - hull.displacement_t) / hull.displacement_t
    assert closure < 0.03


def test_geometry_gate_rejects_inconsistent_displacement(mod):
    with pytest.raises(ValueError):
        mod.Hull(length=320.0, beam=58.0, draft=20.8, cb=0.8098, speed=7.97,
                 wetted_surface=27000.0, displacement_t=200000.0)


def test_geometry_gate_rejects_absurd_block_coefficient(mod):
    with pytest.raises(ValueError):
        mod.Hull(length=320.0, beam=58.0, draft=20.8, cb=0.99, speed=7.97,
                 wetted_surface=27000.0)


def test_froude_number_matches_simman_anchor(mod, anchor):
    hull = mod.hull_from_anchor(anchor)
    checks = mod.anchor_cross_checks(hull, anchor)
    assert checks["froude_number"]["verdict"] == "PASS"
    assert checks["froude_number"]["rel_diff_pct"] < 1.0
    assert checks["L_over_V_s"] > 10.0            # -> 10/10 limit takes the >=10 s branch


# --------------------------------------------------------------------------- #
# 3. model / indicator definitions
# --------------------------------------------------------------------------- #
def test_self_propulsion_balance(mod, models):
    """The solved rpm must make thrust*(1-t_p) equal the design resistance."""
    m = models["conventional"]
    balance = (1.0 - m.p.t_p) * m.thrust_design / m.resistance_design
    assert abs(balance - 1.0) < 1e-6


def test_kt_line_is_monotone_and_positive(mod, models):
    m = models["conventional"]
    kts = [m.kt(u) for u in (0.0, 2.0, 4.0, 6.0, 7.97)]
    assert all(k > 0 for k in kts)
    assert all(a > b for a, b in zip(kts, kts[1:]))


def test_rudder_damps_yaw_for_both_configurations(mod, models):
    """A helm-hold at zero angle must not amplify a yaw disturbance."""
    for cfg in ("conventional", "podded"):
        m = models[cfg]
        state = m.initial_state()
        state[2] = 0.01                      # 0.01 rad/s yaw disturbance
        f0 = m.steering_forces(*state[:3], 0.0)[2]
        assert f0 <= 0.0, f"{cfg}: steering force must damp a free yaw at zero helm"


def test_turning_circle_indicators_are_consistent(mod, models):
    t = mod.turning_circle(models["conventional"], 35.0, dt=2.0, t_end=1200.0)
    assert t["reached_90deg"] and t["reached_180deg"]
    assert t["advance_over_L"] > 0.0
    assert t["transfer_over_L_at_90deg"] > 0.0
    assert t["tactical_diameter_over_L"] > 0.0
    # D_T is measured at 180 deg; it cannot exceed twice the maximum transfer
    assert t["tactical_diameter_over_L"] <= t["tactical_diameter_over_L_max_transfer"] + 1e-9
    # physical sanity: the turn costs speed
    assert t["u_over_u0_at_end"] < 1.0


def test_turning_circle_reports_no_180deg_without_enough_time(mod, models):
    t = mod.turning_circle(models["conventional"], 35.0, dt=2.0, t_end=60.0)
    assert t["reached_180deg"] is False
    assert t["tactical_diameter_over_L"] is None
    assert t["u_ss_status"] in {"STEADY", "NO_STEADY_TURN"}


def test_zigzag_overshoot_measured_from_this_reversals_trigger(mod, models):
    z = mod.zigzag(models["conventional"], 10.0, dt=2.0, t_end=1200.0)
    assert len(z["overshoots"]) >= 2
    for entry in z["overshoots"]:
        # trigger sits at +-10 deg by construction of the event, not by luck
        assert abs(abs(entry["trigger_heading_deg"]) - 10.0) < 1e-6
        sign = math.copysign(1.0, entry["trigger_heading_deg"])
        assert entry["overshoot_deg"] == pytest.approx(
            sign * (entry["peak_heading_deg"] - entry["trigger_heading_deg"]), abs=1e-9)
        assert entry["overshoot_deg"] > -1e-6


def test_zigzag_overshoot_of_a_linear_model_scales_with_helm(mod, models):
    """Monotone response check: the 20/20 overshoots exceed the 10/10 ones."""
    m = models["conventional"]
    z10 = mod.zigzag(m, 10.0, dt=2.0, t_end=1200.0)
    z20 = mod.zigzag(m, 20.0, dt=2.0, t_end=1200.0)
    assert z20["first_overshoot_deg"] > z10["first_overshoot_deg"]


def test_time_step_convergence_of_advance_below_two_percent(mod, anchor):
    conv = mod.convergence(anchor, cfg="conventional", dts=(2.0, 1.0))
    halving = conv["advance_over_L"]["halving"][0]
    assert halving["rel_change_pct"] < 2.0


def test_imo_limits_come_from_the_anchor_file(mod, anchor):
    limits = mod.imo_limits(anchor)
    assert limits["advance_over_L"] == 4.5
    assert limits["tactical_diameter_over_L"] == 5.0
    assert limits["zz10_first_lt10s"] == 10.0 and limits["zz10_first_ge10s"] == 20.0
    assert limits["zz20_first"] == 25.0 and limits["zz20_second"] == 40.0


def test_imo_verdicts_flag_a_failure_when_above_the_limit(mod, models, anchor):
    limits = mod.imo_limits(anchor)
    fake_turn = {"advance_over_L": 9.0, "tactical_diameter_over_L": 9.0}
    fake_zz = {"first_overshoot_deg": 30.0, "second_overshoot_deg": 50.0}
    rows = mod.imo_verdicts(models["conventional"], fake_turn, fake_zz, fake_zz, limits)
    by_q = {r["quantity"]: r for r in rows}
    assert by_q["turning.advance_over_L"]["verdict"] == "FAIL"
    assert by_q["turning.tactical_diameter_over_L"]["verdict"] == "FAIL"
    assert by_q["zigzag.10_10.first_overshoot_deg"]["verdict"] == "FAIL"
    assert by_q["zigzag.20_20.second_overshoot_deg"]["verdict"] == "FAIL"


def test_imo_verdicts_use_the_lv_branch(mod, models, anchor):
    limits = mod.imo_limits(anchor)
    hull = models["conventional"].hull
    fake_turn = {"advance_over_L": 1.0, "tactical_diameter_over_L": 1.0}
    fake_zz = {"first_overshoot_deg": 15.0, "second_overshoot_deg": 15.0}
    rows = mod.imo_verdicts(models["conventional"], fake_turn, fake_zz, fake_zz, limits)
    row = {r["quantity"]: r for r in rows}["zigzag.10_10.first_overshoot_deg"]
    if hull.L_over_V >= 10.0:
        assert row["limit"] == limits["zz10_first_ge10s"] and row["verdict"] == "PASS"
    else:
        assert row["limit"] == limits["zz10_first_lt10s"] and row["verdict"] == "FAIL"


# --------------------------------------------------------------------------- #
# 4. committed artefact
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def artefact():
    if not ART.exists():
        pytest.skip("run benchmarks/podded_ice_maneuvering_w22.py first")
    return json.loads(ART.read_text(encoding="utf-8"))


def test_artifact_provenance_hashes(artefact):
    import hashlib
    prov = artefact["provenance"]
    assert hashlib.sha256(ANCHOR.read_bytes()).hexdigest() == prov["anchor_sha256"]
    assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == prov["script_sha256"]
    assert len(prov["git_commit"]) == 40


def test_artifact_has_all_four_criteria_with_verdicts(artefact):
    crit = {c["id"]: c for c in artefact["criteria"]}
    assert set(crit) == {1, 2, 3, 4}
    for c in crit.values():
        assert c["verdict"] in {"PASS", "FAIL", "WITHHELD"}
        assert c["measured"] is not None
        assert c["evidence"]
    assert crit[1]["verdict"] == "FAIL"          # anchor completeness (honest FAIL)
    assert crit[3]["verdict"] == "WITHHELD"      # no SIMMAN baseline offline
    assert crit[4]["verdict"] == "PASS"          # dt halving < 2%


def test_artifact_run_rows_are_complete(artefact):
    for cfg in ("conventional", "podded"):
        run = artefact["runs"][cfg]
        for key in ("turning_35deg", "turning_20deg", "zigzag_10_10", "zigzag_20_20"):
            assert key in run
        for row in run["imo_verdicts"]:
            assert row["verdict"] in {"PASS", "FAIL", "NOT_REACHED"}
            assert row["unit"] and row["evidence"]
            if row["verdict"] != "NOT_REACHED":
                assert row["value"] is not None and row["limit"] is not None


def test_artifact_comparison_is_like_for_like(artefact):
    cmp_adv = artefact["comparison_podded_vs_conventional"]["advance_over_L"]
    assert cmp_adv["conventional"] > 0 and cmp_adv["podded"] > 0
    assert cmp_adv["rel_diff_pct"] == pytest.approx(
        100.0 * (cmp_adv["podded"] - cmp_adv["conventional"]) / cmp_adv["conventional"],
        rel=1e-9)
    assert "internal comparison" in artefact["comparison_podded_vs_conventional"]["note"]


def test_artifact_declares_its_evidence_grade(artefact):
    model = artefact["model"]
    assert "WITHHELD" in model["model_status"]
    assert model["declared_constants"]["hull_derivatives"]
    assert "ice_resistance_fraction" in model["declared_constants"]