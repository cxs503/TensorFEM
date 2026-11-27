"""Guards for the springing-whipping-hydroelastic domain-experiment artifact.

The tests encode the discipline of this experiment:

* the artifact must exist at the declared path and carry the required schema
  (anchor reference / case caliber / geometry gate / cases / criteria /
  damping sensitivity / verdict / reproduce);
* every case must be reported (no cherry-picking of passing rows);
* the peak / dominant-frequency gates are machine-judged against the 15 %
  limit and the honest boundary (external SNAME anchor NOT_OBTAINED,
  claim_eligible = false) must be stated;
* the anchor values used by the experiment must be traceable to the T1
  artifact (no number typed from memory).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = REPO_ROOT / "benchmarks" / "springing-whipping-hydroelastic" / "comparison.json"
T1_ANCHOR_PATH = REPO_ROOT / "benchmarks" / "anchors" / "springing-whipping-hydroelastic.json"


@pytest.fixture(scope="module")
def artifact():
    assert ARTIFACT_PATH.exists(), (
        "artifact missing: run PYTHONPATH=src python "
        "benchmarks/springing-whipping-hydroelastic/run_comparison.py"
    )
    return json.loads(ARTIFACT_PATH.read_text())


@pytest.fixture(scope="module")
def t1_anchor():
    return json.loads(T1_ANCHOR_PATH.read_text())


def test_artifact_has_required_schema(artifact):
    for key in ("task_key", "topic", "experiment", "repository", "branch",
                "anchor_reference", "case_caliber", "geometry_gate",
                "wet_modal_model", "cases", "criteria", "damping_sensitivity",
                "verdict", "limitations", "reproduce"):
        assert key in artifact, key
    assert artifact["task_key"] == "paper-exp-springing-whipping-hydroelastic-2"
    assert artifact["repository"] == "TensorFEM"


def test_all_cases_reported_and_gates_machine_judged(artifact):
    """≥3 cases, all reported; every case carries a machine-judged PASS/FAIL."""
    cases = artifact["cases"]
    assert len(cases) >= 3, f"need >= 3 cases, got {len(cases)}"
    # every case must have both a peak and a frequency gate row
    judged = {c["case"] for c in artifact["criteria"]}
    assert judged == {c["case"] for c in cases}, "a case is missing from the criteria table"
    for c in artifact["criteria"]:
        assert c["criterion_peak"] in ("PASS", "FAIL")
        assert c["criterion_frequency"] in ("PASS", "FAIL")
        assert c["limit_pct"] == 15.0


def test_peak_and_frequency_gates_within_15pct(artifact):
    """The 15 % gate is the task's quantitative criterion."""
    for c in artifact["criteria"]:
        assert c["peak_rel_dev_pct"] <= 15.0, (
            f"{c['case']} peak deviation {c['peak_rel_dev_pct']:.2f}% exceeds 15%"
        )
        assert c["freq_rel_dev_pct"] <= 15.0, (
            f"{c['case']} frequency deviation {c['freq_rel_dev_pct']:.2f}% exceeds 15%"
        )


def test_damping_sensitivity_reported(artifact):
    """Damping-ratio sensitivity must be reported (not just the passing case)."""
    ds = artifact["damping_sensitivity"]
    assert len(ds["zeta_values"]) >= 3
    assert len(ds["M_peak_nm"]) == len(ds["zeta_values"])
    assert len(ds["daf"]) == len(ds["zeta_values"])
    # DAF must be consistent with the analytic 1/(2 zeta) within a factor ~1.1
    for z, daf, daf_an in zip(ds["zeta_values"], ds["daf"], ds["daf_analytic_1_over_2zeta"]):
        assert abs(daf / daf_an - 1.0) < 0.1, (
            f"zeta={z}: DAF {daf:.2f} vs analytic {daf_an:.2f} inconsistent"
        )


def test_anchor_values_traceable_to_t1(artifact, t1_anchor):
    """The experiment's anchor values must come from the T1 artifact, not memory."""
    used = artifact["anchor_reference"]["values_used"]
    cal = t1_anchor["case_caliber"]["quantities"]
    fix = t1_anchor["dry_run_fixture"]
    assert used["L_m"] == cal["L"]["value"]
    assert used["B_m"] == cal["B"]["value"]
    assert used["T_m"] == cal["T"]["value"]
    assert used["rho_kg_m3"] == cal["rho"]["value"]
    assert used["V_m_s"] == cal["V"]["value"]
    assert used["q_pa"] == fix["derived_case_caliber_values"]["q_pa"]
    assert used["p_rms_pa"] == fix["derived_case_caliber_values"]["p_rms_pa"]
    assert used["cp_rms_star"] == fix["cp_rms_star"]
    assert used["amplitude_A"] == fix["amplitude_A"]
    # the honest boundary must be stated
    assert artifact["anchor_reference"]["claim_eligible"] is False
    assert "NOT_OBTAINED" in artifact["anchor_reference"]["note"] or \
           "NOT_OBTAINED" in artifact["verdict"]["kind"]


def test_geometry_gate_passes(artifact):
    """Geometry must be verified (Cb / L / B) before any modal or time-domain number."""
    ga = artifact["geometry_gate"]
    assert ga["pass"] is True
    assert ga["area_rel_error"] < 1e-9
    assert ga["inertia_rel_error"] < 1e-9


def test_verdict_honest_boundary(artifact):
    """The verdict must not overclaim: external anchor is WITHHELD, not validated."""
    v = artifact["verdict"]
    assert v["external_anchor_status"] == "WITHHELD"
    assert "claim_eligible=false" in v["kind"] or "claim_eligible = false" in v["kind"]
