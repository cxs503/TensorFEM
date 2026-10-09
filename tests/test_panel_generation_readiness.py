import hashlib
import json

from tensorfem.panel_generation_readiness import (
    aggregate_panel_mesh_readiness, build_peak_step_sensitivity_evidence,
    panel_generation_readiness,
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def test_readiness_adapter_is_hashed_and_never_claims_v1(tmp_path):
    events = tmp_path/"events"; events.mkdir()
    checkpoint = tmp_path/"generation.pt"; checkpoint.write_bytes(b"generation")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    point = {"force_n": 12., "yielded_fraction": .1,
             "equilibrium_relative_norm": 1e-10,
             "energy_balance_gate": {"passed": True},
             "controller_incremental_energy_balance_gate": {"passed": True}}
    event = {"generation_committed": 4, "prior_event_sha256": None,
             "terminal_point": point, "elapsed_seconds": 2.}
    event["event_sha256"] = hashlib.sha256(canonical(event).encode()).hexdigest()
    (events/"g000004.json").write_text(json.dumps(event))
    manifest = {"accepted_points": 4, "divisions": 4,
                "checkpoint_file": checkpoint.name,
                "checkpoint_sha256": checkpoint_hash, "peak_force_n": 12.,
                "peak_confirmed": False, "post_peak_observed": False}
    (tmp_path/"chunked-test.json").write_text(json.dumps(manifest))
    status = {"schema": "tensorfem.panel-generation-status/1.0",
              "state": "target_reached", "current_generation": 4,
              "target_generation": 4, "last_event_sha256": event["event_sha256"],
              "reason": None, "updated_unix_seconds": 1.}
    status["status_sha256"] = hashlib.sha256(canonical(status).encode()).hexdigest()
    (tmp_path/"generation-status.json").write_text(json.dumps(status))
    result = panel_generation_readiness(tmp_path)
    assert result["integrity_passed"] is True
    assert result["qualification_state"] == "verified_monotonic_prefix"
    assert result["recent_energy_gates_passed"] is True
    assert result["v1_ready"] is False
    assert len(result["evidence_sha256"]) == 64


def test_readiness_keeps_failed_attempt_outside_committed_head(tmp_path):
    events = tmp_path/"events"; events.mkdir()
    checkpoint = tmp_path/"generation.pt"; checkpoint.write_bytes(b"generation")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    point = {"force_n": 12., "yielded_fraction": .1,
             "equilibrium_relative_norm": 1e-10,
             "energy_balance_gate": {"passed": True},
             "controller_incremental_energy_balance_gate": {"passed": True}}
    committed = {"generation_requested": 7, "generation_committed": 7,
                 "status": "executed", "commit_advanced": True,
                 "prior_event_sha256": None, "terminal_point": point}
    committed["event_sha256"] = hashlib.sha256(
        canonical(committed).encode()).hexdigest()
    (events/"g000007.json").write_text(json.dumps(committed))
    failed = {"generation_requested": 8, "generation_committed": 7,
              "status": "incomplete", "commit_advanced": False,
              "prior_event_sha256": committed["event_sha256"],
              "terminal_point": point}
    failed["event_sha256"] = hashlib.sha256(canonical(failed).encode()).hexdigest()
    (events/"g000008.json").write_text(json.dumps(failed))
    manifest = {"accepted_points": 7, "divisions": 4,
                "checkpoint_file": checkpoint.name,
                "checkpoint_sha256": checkpoint_hash, "peak_force_n": 12.,
                "peak_confirmed": False, "post_peak_observed": False}
    (tmp_path/"chunked-test.json").write_text(json.dumps(manifest))
    status = {"schema": "tensorfem.panel-generation-status/1.0",
              "state": "stopped", "current_generation": 7,
              "target_generation": 10,
              "last_event_sha256": committed["event_sha256"],
              "last_attempt_event_sha256": failed["event_sha256"],
              "reason": "wall", "updated_unix_seconds": 1.}
    status["status_sha256"] = hashlib.sha256(canonical(status).encode()).hexdigest()
    (tmp_path/"generation-status.json").write_text(json.dumps(status))
    result = panel_generation_readiness(tmp_path)
    assert result["generation"] == 7
    assert result["last_event_sha256"] == committed["event_sha256"]
    assert result["last_attempt_event_sha256"] == failed["event_sha256"]


def _readiness(divisions, peak, *, post_peak=True):
    sensitivity = build_peak_step_sensitivity_evidence(
        divisions=divisions, source_checkpoint_sha256="a"*64,
        source_checkpoint_file="source.pt", source_manifest_sha256="c"*64,
        refined_manifest_file="refined.json", refined_manifest_sha256="d"*64,
        baseline_maximum_step=2e-4, refined_maximum_step=5e-5,
        baseline_peak_force_n=peak, refined_peak_force_n=peak*.995,
        energy_balance_passed=True, restart_state_exact=True)
    clean = {"schema": "tensorfem.panel-generation-readiness/1.0",
             "integrity_passed": True, "divisions": divisions,
             "arc_metric": "dimensionally_scaled",
             "source_manifest_sha256": str(divisions)*64,
             "peak_force_n": peak, "peak_confirmed": True,
             "post_peak_observed": post_peak,
             "peak_step_sensitivity": sensitivity,
             "peak_step_sensitivity_error": None}
    return {**clean, "evidence_sha256": hashlib.sha256(
        canonical(clean).encode()).hexdigest()}


def test_mesh_readiness_aggregate_is_strictly_blocked_until_all_postpeak():
    blocked = aggregate_panel_mesh_readiness({
        4: _readiness(4, 509000.), 8: _readiness(8, 505000., post_peak=False),
    })
    assert blocked["status"] == "blocked" and not blocked["passed"]
    assert blocked["rejected"] == {8: "post_peak_not_observed",
                                    12: "missing_readiness"}

    passed = aggregate_panel_mesh_readiness({
        4: _readiness(4, 509000.), 8: _readiness(8, 505000.),
        12: _readiness(12, 503000.),
    })
    assert passed["status"] == "passed" and passed["passed"]
    assert len(passed["report_sha256"]) == 64


def test_mesh_readiness_requires_local_peak_step_sensitivity():
    item = _readiness(4, 500000.)
    clean = {key: value for key, value in item.items()
             if key not in {"evidence_sha256", "peak_step_sensitivity"}}
    clean["peak_step_sensitivity"] = None
    clean["peak_step_sensitivity_error"] = "missing_peak_step_sensitivity"
    item = {**clean, "evidence_sha256": hashlib.sha256(
        canonical(clean).encode()).hexdigest()}
    result = aggregate_panel_mesh_readiness({4: item})
    assert result["rejected"][4] == "missing_peak_step_sensitivity"


def test_peak_step_sensitivity_uses_strict_three_percent_gate():
    passed = build_peak_step_sensitivity_evidence(
        divisions=8, source_checkpoint_sha256="b"*64,
        source_checkpoint_file="source.pt", source_manifest_sha256="c"*64,
        refined_manifest_file="refined.json", refined_manifest_sha256="d"*64,
        baseline_maximum_step=2e-4, refined_maximum_step=5e-5,
        baseline_peak_force_n=1e6, refined_peak_force_n=0.98e6,
        energy_balance_passed=True, restart_state_exact=True)
    assert passed["passed"] is True
    failed = build_peak_step_sensitivity_evidence(
        divisions=8, source_checkpoint_sha256="b"*64,
        source_checkpoint_file="source.pt", source_manifest_sha256="c"*64,
        refined_manifest_file="refined.json", refined_manifest_sha256="d"*64,
        baseline_maximum_step=2e-4, refined_maximum_step=5e-5,
        baseline_peak_force_n=1e6, refined_peak_force_n=0.96e6,
        energy_balance_passed=True, restart_state_exact=True)
    assert failed["passed"] is False
