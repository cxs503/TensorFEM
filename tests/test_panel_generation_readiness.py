import hashlib
import json

from tensorfem.panel_generation_readiness import panel_generation_readiness


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
    manifest = {"accepted_points": 4, "checkpoint_file": checkpoint.name,
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

