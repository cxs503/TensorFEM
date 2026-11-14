"""Tests for the out-of-sample Hertz verification harness (case pool id 104).

These tests never re-run the solver (each solve is ~40 s); they grade the
*harness contract* and, when the artifact exists, the artifact's internal
consistency against the shipped certificate.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ANCHOR = os.path.join(HERE, "..", "benchmarks", "verified",
                      "fem_hertz_contact_outofsample")
GATES = os.path.join(ANCHOR, "gates_outofsample.json")
CERT = os.path.join(HERE, "..", "benchmarks", "verified",
                    "fem_hertz_contact_loadcontrol", "gates_verified.json")

sys.path.insert(0, ANCHOR)


def _harness():
    import run as harness_mod  # noqa: F401  (benchmarks/verified/.../run.py)
    return harness_mod


def test_criteria_declared_before_run():
    h = _harness()
    assert h.CRITERIA["contact_radius_pct"] == 3.0
    assert h.CRITERIA["indentation_pct"] == 3.0
    assert h.CRITERIA["replication_tol_pp"] <= 0.02
    # out-of-sample points must be disjoint from the certified band
    assert not set(h.OUT_OF_SAMPLE) & set(h.REPLICATION)
    assert min(h.OUT_OF_SAMPLE) < 0.05 < 0.10 < max(h.OUT_OF_SAMPLE)


def test_envelope_statement_reports_failures():
    h = _harness()
    txt = h._envelope_statement({0.03: True, 0.15: False}, {0.02: False})
    assert "fails at a/R 0.15" in txt
    assert "passes at a/R 0.03" in txt
    assert "outside 3%" in txt


def test_load_for_radius_ratio_inverts_hertz():
    h = _harness()
    from tensorfem.hertz_halfspace_loadcontrol import hertz_load_reference

    for ratio in h.REPLICATION + h.OUT_OF_SAMPLE:
        load = h.load_for_radius_ratio(ratio)
        ref = hertz_load_reference(load, h.R, h.E, h.NU)
        assert ref.contact_radius == pytest.approx(ratio * h.R, rel=1e-12)


AUX_PREFIXES = ("refinement_stability", "penetration", "kinematic_consistency")


@pytest.mark.skipif(not os.path.exists(GATES), reason="out-of-sample run not executed")
def test_artifact_pool_gates_pass():
    """Only the case-pool metrics are required to pass.

    Auxiliary robustness gates (estimator refinement stability, ALM
    penetration, Hertz kinematic identity) are graded too, and any failure
    must be *declared* in ``auxiliary_failures`` -- never dropped silently.
    """
    with open(GATES) as fh:
        doc = json.load(fh)
    assert doc["schema"] == "tensorfem.verified-gates.v1"
    assert doc["pool_gates_all_passed"] is True
    assert doc["passed"] == len(doc["gates"]) - len(doc["failed_gates"])
    for gid in doc["failed_gates"]:
        assert gid.removeprefix("oos_").startswith(AUX_PREFIXES), gid
        assert gid in doc["auxiliary_failures"], gid
        assert doc["auxiliary_failures"][gid]["scope"]
    assert not any(gid.startswith("replication_") for gid in doc["failed_gates"])
    ids = {g["id"] for g in doc["gates"]}
    # every out-of-sample point is graded on both pool metrics
    for ratio in (0.03, 0.075, 0.15):
        key = f"{round(ratio * 1000):03d}"
        assert f"oos_contact_radius_aR{key}" in ids
        assert f"oos_indentation_aR{key}" in ids


@pytest.mark.skipif(not os.path.exists(GATES), reason="out-of-sample run not executed")
def test_artifact_reproduces_certificate():
    with open(GATES) as fh:
        doc = json.load(fh)
    with open(CERT) as fh:
        cert = {g["id"]: g for g in json.load(fh)["gates"]}
    unrefined = [g for g in doc["gates"] if g["id"].startswith("replication_")]
    assert len(unrefined) == 4
    for g in unrefined:
        assert g["passed"], (g["id"], g["value"])
        assert g["value"] <= 0.02
    # the certificate numbers quoted as references must be the frozen ones
    for ratio, tag in ((0.05, "aR005"), (0.10, "aR010")):
        row = next(r for r in doc["rows"]
                   if abs(r["a_over_R"] - ratio) < 1e-12
                   and abs(r["h_over_a"] - 0.0075) < 1e-12)
        assert abs(row["error_pct"]["contact_radius_linear"]) == pytest.approx(
            cert[f"contact_radius_{tag}"]["value"], abs=0.02)
        assert abs(row["error_pct"]["indentation"]) == pytest.approx(
            cert[f"indentation_{tag}"]["value"], abs=0.02)


@pytest.mark.skipif(not os.path.exists(GATES), reason="out-of-sample run not executed")
def test_artifact_envelope_matches_gate_verdicts():
    with open(GATES) as fh:
        doc = json.load(fh)
    env = doc["envelope"]
    for ratio, passed in env["out_of_sample_pass"].items():
        key = f"{round(float(ratio) * 1000):03d}"
        ga = next(g for g in doc["gates"] if g["id"] == f"oos_contact_radius_aR{key}")
        gd = next(g for g in doc["gates"] if g["id"] == f"oos_indentation_aR{key}")
        assert passed == (ga["passed"] and gd["passed"])
    # the documented low-load control must be recorded, gated or not
    assert len(doc["controls"]) == 2  # two mesh levels at a/R = 0.02
    assert all("inside_3pct" in c for c in doc["controls"])