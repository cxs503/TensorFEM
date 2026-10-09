import copy
import hashlib
import json

import pytest

from tensorfem.shell_sparse_scaling import (
    ShellSparseScalingPoint, sparse_scaling_qualification,
)
from tensorfem.nonlinear_shell_robustness import run_nonlinear_shell_robustness
from tensorfem.v1_evidence_aggregator import (
    aggregate_v1_evidence, validate_v1_evidence_aggregate,
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_hashed(path, payload):
    payload = dict(payload)
    payload["evidence_sha256"] = hashlib.sha256(canonical(payload).encode()).hexdigest()
    path.write_text(json.dumps(payload))


def release_payload(directory):
    executions = {}
    for name in ("full_regression", "api_audit", "offline_install"):
        log = directory/f"{name}.log"; log.write_text(name)
        executions[name] = {
            "passed": True, "returncode": 0, "elapsed_seconds": .1,
            "command": ["python", name], "log_path": log.name,
            "output_sha256": hashlib.sha256(name.encode()).hexdigest()}
    return {
        "schema": "tensorfem.release-quality-evidence/1.0", "passed": True,
        "full_regression": True, "offline_install": True, "api_audit": True,
        "executions": executions,
    }


def sparse_report(path):
    point = ShellSparseScalingPoint(
        42, 10882, 566460, 80., 2., .3, 7, 1e-10, 1e-6,
        6_800_000, 80_000_000, 1.5, .03, 1e-15,
        6_800_000, 99_000_000, 947_000_000)
    path.write_text(json.dumps(sparse_scaling_qualification([point])))


def test_missing_sources_are_blocked_not_inferred(tmp_path):
    report = aggregate_v1_evidence(
        sparse_report=None, release_report=None, contact_report=None,
        panel_directories={})
    assert not report["readiness"]["ready_for_1_0"]
    assert set(report["readiness"]["blockers"]) == {
        "general_double_deformable_contact_3d", "nonlinear_shell_robustness",
        "release_quality", "shell_sparse_scalability",
        "marine_panel_4_8_12_peak_postpeak",
    }
    assert validate_v1_evidence_aggregate(report) == report


def test_verified_partial_sources_map_scope_and_stay_blocked(tmp_path):
    sparse = tmp_path/"sparse.json"; sparse_report(sparse)
    release = tmp_path/"release.json"
    write_hashed(release, release_payload(tmp_path))
    contact = tmp_path/"contact.json"
    write_hashed(contact, {
        "schema": "tensorfem.curved-surface-contact3d-qualification/1.0",
        "passed": True, "maximum_relative_error": .01,
        "rollback_exact": True,
        "general_surface_to_surface": "qualified_curved_frictionless_subset"})
    report = aggregate_v1_evidence(
        sparse_report=sparse, release_report=release, contact_report=contact,
        panel_directories={})
    gates = report["readiness"]["gates"]
    assert gates["shell_sparse_scalability"]["status"] == "qualified"
    assert gates["release_quality"]["status"] == "qualified"
    assert gates["general_double_deformable_contact_3d"]["status"] == "blocked"
    assert not report["readiness"]["ready_for_1_0"]


def test_source_and_aggregate_tampering_fail_closed(tmp_path):
    release = tmp_path/"release.json"
    write_hashed(release, release_payload(tmp_path))
    payload = json.loads(release.read_text()); payload["offline_install"] = False
    release.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        aggregate_v1_evidence(sparse_report=None, release_report=release,
                              contact_report=None, panel_directories={})
    clean = aggregate_v1_evidence(sparse_report=None, release_report=None,
                                  contact_report=None, panel_directories={})
    broken = copy.deepcopy(clean); broken["readiness"]["ready_for_1_0"] = True
    with pytest.raises(ValueError, match="aggregate hash mismatch"):
        validate_v1_evidence_aggregate(broken)


def test_nonlinear_evidence_is_independent_hashed_and_semantic(tmp_path):
    nonlinear = tmp_path/"nonlinear.json"
    report = run_nonlinear_shell_robustness()
    assert report["passed"]
    assert report["maximum_plastic_strain"] > 0.
    assert report["plastic_history_relative_error"] < .01
    nonlinear.write_text(json.dumps(report))
    aggregate = aggregate_v1_evidence(
        sparse_report=None, release_report=None, contact_report=None,
        nonlinear_shell_report=nonlinear, panel_directories={})
    assert aggregate["readiness"]["gates"]["nonlinear_shell_robustness"][
        "status"] == "qualified"

    forged = dict(report); forged["passed"] = False
    nonlinear.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="hash mismatch"):
        aggregate_v1_evidence(
            sparse_report=None, release_report=None, contact_report=None,
            nonlinear_shell_report=nonlinear, panel_directories={})


def test_semantically_forged_nonlinear_and_release_fail_closed(tmp_path):
    nonlinear = tmp_path/"nonlinear.json"
    report = run_nonlinear_shell_robustness()
    report["passed"] = False
    write_hashed(nonlinear, {key: value for key, value in report.items()
                             if key != "evidence_sha256"})
    with pytest.raises(ValueError, match="status is inconsistent"):
        aggregate_v1_evidence(
            sparse_report=None, release_report=None, contact_report=None,
            nonlinear_shell_report=nonlinear, panel_directories={})

    release = tmp_path/"release.json"
    payload = release_payload(tmp_path); del payload["executions"]
    write_hashed(release, payload)
    with pytest.raises(ValueError, match="no executable gate records"):
        aggregate_v1_evidence(
            sparse_report=None, release_report=release, contact_report=None,
            panel_directories={})

    payload = release_payload(tmp_path)
    write_hashed(release, payload)
    (tmp_path/"api_audit.log").write_text("tampered")
    with pytest.raises(ValueError, match="execution log hash mismatch"):
        aggregate_v1_evidence(
            sparse_report=None, release_report=release, contact_report=None,
            panel_directories={})


def test_finite_strain_contact_report_preserves_subset_scope(tmp_path):
    contact = tmp_path/"finite-contact.json"
    body = {
        "schema": "tensorfem.curved-finite-strain-friction-qualification/1.0",
        "passed": True, "rollback_exact": True,
        "mesh_sequence": [{"coulomb_relative_error": 2e-4}],
        "master_slave_interchange_relative_error": 3e-4,
    }
    digest = hashlib.sha256(canonical(body).encode()).hexdigest()
    contact.write_text(json.dumps({**body, "evidence_sha256": digest,
                                   "wall_time_seconds": 1.25}))
    report = aggregate_v1_evidence(
        sparse_report=None, release_report=None, contact_report=contact,
        panel_directories={})
    gate = report["readiness"]["gates"][
        "general_double_deformable_contact_3d"]
    assert gate == {"status": "blocked",
                    "reason": "incomplete_or_out_of_scope"}
    assert report["sources"]["general_double_deformable_contact_3d"][
        "schema"] == body["schema"]
