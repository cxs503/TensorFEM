"""Certificate test for the pressurised-sphere membrane anchor.

The anchor shipped a legacy certificate (no ``schema``/``anchor``/``passed``
envelope, prose criteria, two vector-valued gates), so
``tools/verified_anchor_audit.py`` filed it as ``E_SCHEMA`` +
``E_PASS_WITHOUT_CRITERION``: the PASS could be read but not recomputed.
``benchmarks/verified/fem_sphere_pressure_membrane/emit_certificate.py``
re-emits the certificate from ``result.json`` alone.

These tests defend the *substance* of that repair, not the file format:

* every gate value is re-derived here **independently** of the emitter (own
  arithmetic, straight from the raw Gauss-point arrays in ``result.json``);
* every verdict is recomputed from its own ``value``/``operator``/``threshold``;
* the 3 % acceptance tolerance is the binding one, unchanged from the case;
* the falsified prediction is still in the record, with its numbers, and is
  structurally unable to flip the gating verdict;
* the anchor is out of the shrink-only debt registry.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANCHOR_DIR = REPO_ROOT / "benchmarks" / "verified" / "fem_sphere_pressure_membrane"
RESULT_JSON = ANCHOR_DIR / "result.json"
GATES_JSON = ANCHOR_DIR / "gates_verified.json"
EMITTER = ANCHOR_DIR / "emit_certificate.py"
AUDIT_TOOL = REPO_ROOT / "tools" / "verified_anchor_audit.py"

TOLERANCE = 0.03


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def emitter():
    return _load(EMITTER, "sphere_emit_certificate")


@pytest.fixture(scope="module")
def result():
    return json.loads(RESULT_JSON.read_text())


@pytest.fixture(scope="module")
def certificate():
    return json.loads(GATES_JSON.read_text())


@pytest.fixture(scope="module")
def gates(certificate):
    return {gate["id"]: gate for gate in certificate["gates"]}


def _ladder_row(result: dict, geometry: str, facets: int) -> dict:
    for row in result["convergence"]["ladder"][geometry]:
        if int(row["facets"]) == facets:
            return row
    raise AssertionError(f"no {geometry} row at facets={facets}")


# --------------------------------------------------------------------------- #
# 1. the certificate is a pure function of the artifact
# --------------------------------------------------------------------------- #
def test_certificate_matches_artifact_derived_regeneration(emitter):
    assert GATES_JSON.read_text() == emitter.render(), (
        "gates_verified.json drifted from result.json; re-run "
        "benchmarks/verified/fem_sphere_pressure_membrane/emit_certificate.py --write"
    )


def test_certificate_has_canonical_envelope(certificate):
    assert certificate["schema"] == "tensorfem.verified-gates.v1"
    assert certificate["anchor"] == ANCHOR_DIR.name
    assert isinstance(certificate["passed"], bool)
    assert certificate["gates"], "a certificate with no gates certifies nothing"


def test_no_gate_is_a_prose_or_vector_verdict(certificate):
    """The legacy defect: a PASS whose criterion was only prose or a vector."""
    for gate in certificate["gates"]:
        assert isinstance(gate["value"], (int, float)) and not isinstance(gate["value"], bool)
        assert isinstance(gate["threshold"], (int, float))
        assert gate["operator"] in {"<", "<=", ">", ">=", "=="}
        assert isinstance(gate["passed"], bool)


# --------------------------------------------------------------------------- #
# 2. independent recomputation from the raw artifact (not via the emitter)
# --------------------------------------------------------------------------- #
def test_hoop_stress_gate_recomputes_from_raw_gauss_points(result, gates):
    row = _ladder_row(result, "exact_meridian", 8)
    reference = row["sigma_ref_pa"]
    independently = max(abs(v / reference - 1.0) for v in row["sigma_theta_pa"])
    assert gates["G1.sigma_theta_vs_pR_2t"]["value"] == pytest.approx(independently, rel=1e-12)
    assert independently < TOLERANCE


def test_meridional_stress_gate_recomputes_from_raw_gauss_points(result, gates):
    row = _ladder_row(result, "exact_meridian", 8)
    reference = row["sigma_ref_pa"]
    independently = max(abs(v / reference - 1.0) for v in row["sigma_phi_pa"])
    assert gates["G2.sigma_phi_vs_pR_2t"]["value"] == pytest.approx(independently, rel=1e-12)


def test_displacement_gate_recomputes_from_artifact_fields(result, gates):
    row = _ladder_row(result, "exact_meridian", 8)
    independently = abs(row["w_m"] / row["w_ref_m"] - 1.0)
    assert gates["G4.radial_displacement_vs_closed_form"]["value"] == pytest.approx(independently, rel=1e-12)


def test_negative_control_recomputes_from_raw_chord_points(result, gates):
    """G9 must stay a *measured* failure of the chord geometry, not a declaration."""
    row = _ladder_row(result, "chord_control", 8)
    independently = max(abs(v / row["sigma_ref_pa"] - 1.0) for v in row["sigma_theta_pa"])
    gate = gates["G9.chord_control_fails"]
    assert gate["value"] == pytest.approx(independently, rel=1e-12)
    assert independently > TOLERANCE


def test_poisson_invariance_gate_recomputes_over_the_nu_sweep(result, gates):
    """Binding member = the worst per-run error recorded over the nu sweep."""
    rows = result["poisson_sensitivity"]["rows"]
    independently = max(row["sigma_theta_max_rel_error"] for row in rows)
    assert gates["G7.poisson_invariance"]["value"] == pytest.approx(independently, rel=1e-12)
    # cross-check the sweep's own ratios: every nu must sit on the same membrane state
    assert max(abs(row["sigma_theta_over_ref"] - 1.0) for row in rows) < TOLERANCE
    assert len(rows) >= 4


def test_reference_validity_gate_recomputes_at_case_t_over_r(result, gates):
    case_t = result["parameters"]["t_over_R"]
    row = min(result["thickness_sensitivity"]["rows"], key=lambda r: abs(r["t_over_R"] - case_t))
    gate = gates["G8a.reference_valid_at_case_point"]
    assert gate["value"] == pytest.approx(row["thin_vs_lame_rel"], rel=1e-12)
    assert gate["value"] < TOLERANCE


def test_flat_facet_control_gate_recomputes_from_result(result, gates):
    gate = gates["G10.flat_facet_shell4_control_fails"]
    assert gate["value"] == pytest.approx(result["flat_facet_control"]["N_mean_rel_error"], rel=1e-12)
    assert gate["value"] > 1.0, "the flat-facet control must miss the tolerance by orders of magnitude"


# --------------------------------------------------------------------------- #
# 3. verdicts are recomputable and the tolerance is unchanged
# --------------------------------------------------------------------------- #
def test_every_verdict_follows_from_its_own_numbers(certificate):
    import operator as op

    table = {"<": op.lt, "<=": op.le, ">": op.gt, ">=": op.ge, "==": op.eq}
    for gate in certificate["gates"]:
        expected = table[gate["operator"]](gate["value"], gate["threshold"])
        assert gate["passed"] is expected, f"{gate['id']} declares {gate['passed']} over its own numbers"
    assert certificate["passed"] is all(gate["passed"] for gate in certificate["gates"])
    assert certificate["failed_gates"] == [
        gate["id"] for gate in certificate["gates"] if not gate["passed"]
    ]


def test_accuracy_gates_keep_the_three_percent_bar(certificate):
    accuracy = [gate for gate in certificate["gates"] if gate["kind"] == "accuracy"]
    assert accuracy, "the case must keep at least one accuracy gate"
    assert all(gate["threshold"] == TOLERANCE for gate in accuracy)
    assert certificate["tolerance"] == TOLERANCE


def test_controls_can_only_pass_by_missing_the_bar(certificate):
    """A negative control that passes when it *meets* the tolerance is meaningless."""
    controls = [gate for gate in certificate["gates"] if gate["kind"] == "control-negative"]
    assert {gate["id"] for gate in controls} == {
        "G9.chord_control_fails",
        "G10.flat_facet_shell4_control_fails",
    }
    for gate in controls:
        assert gate["operator"] == ">=" and gate["value"] > gate["threshold"]


# --------------------------------------------------------------------------- #
# 4. the falsified prediction is preserved and cannot be laundered
# --------------------------------------------------------------------------- #
def test_falsified_prediction_survives_the_repair(certificate):
    predictions = {p["id"]: p for p in certificate["predictions"]}
    prediction = predictions["P1.reference_band_reaches_t_over_R_0.05"]
    assert prediction["outcome"] == "falsified"
    assert prediction["predicted_threshold"] == 0.05
    assert prediction["measured_band_upper_t_over_R"] < prediction["predicted_threshold"]
    assert prediction["measured_passed"] is False
    assert certificate["failed_predictions"] == [prediction["id"]]


def test_falsified_prediction_is_not_a_gate_and_cannot_flip_the_verdict(certificate):
    gate_ids = {gate["id"] for gate in certificate["gates"]}
    assert not any("band_reaches" in gate_id for gate_id in gate_ids)
    assert certificate["passed"] is True


def test_falsified_prediction_keeps_its_measured_deviations(certificate, result):
    """The band endpoints must be re-derivable from the deviation table."""
    prediction = certificate["predictions"][0]
    rows = prediction["measured_deviations"]
    assert len(rows) == len(result["thickness_sensitivity"]["rows"])
    within = [row["t_over_R"] for row in rows if row["thin_vs_lame_rel"] <= TOLERANCE]
    assert within, "no thickness in the sweep keeps the reference within tolerance"
    assert prediction["measured_band_lower_t_over_R"] == pytest.approx(min(within), rel=1e-12)
    assert prediction["measured_band_upper_t_over_R"] == pytest.approx(max(within), rel=1e-12)
    assert prediction["deviation_scaling_slope"] == pytest.approx(
        result["thickness_sensitivity"]["thin_vs_lame_loglog_slope"], rel=1e-12
    )


# --------------------------------------------------------------------------- #
# 5. the audit agrees, and the debt registry shrank
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def audit_tool():
    return _load(AUDIT_TOOL, "verified_anchor_audit_for_sphere")


def test_audit_certifies_the_anchor_without_defects(audit_tool):
    record = audit_tool.audit_anchor(ANCHOR_DIR, audit_tool.tracked_files())
    assert record.certified, f"defects={record.defects} warnings={record.warnings}"
    assert record.defects == []
    assert record.grade == "certified"
    assert record.declared_overall is True
    assert record.recomputed_overall is True
    assert record.failed_gates == 0


def test_anchor_is_out_of_the_debt_registry(audit_tool):
    debt = json.loads(audit_tool.DEBT_JSON.read_text())
    assert ANCHOR_DIR.name not in debt["anchors"], "a repaired anchor still listed as debt"