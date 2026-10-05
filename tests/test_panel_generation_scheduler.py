import hashlib
import json

from scripts.run_panel_generation_path import canonical, latest_event_hash
from scripts.watch_panel_generation_path import verify_snapshot


def test_generation_event_chain_verifies_and_rejects_tampering(tmp_path):
    events = tmp_path/"events"
    events.mkdir()
    value = {"schema": "tensorfem.panel-generation-event/1.0",
             "generation_committed": 4, "prior_event_sha256": None}
    value["event_sha256"] = hashlib.sha256(canonical(value).encode()).hexdigest()
    (events/"g000004.json").write_text(json.dumps(value))
    assert latest_event_hash(events) == value["event_sha256"]
    value["generation_committed"] = 3
    (events/"g000004.json").write_text(json.dumps(value))
    try:
        latest_event_hash(events)
    except ValueError as error:
        assert "integrity mismatch" in str(error)
    else:
        raise AssertionError("tampered event was accepted")


def test_watcher_verifies_status_checkpoint_and_commit_alignment(tmp_path):
    events = tmp_path/"events"
    events.mkdir()
    checkpoint = tmp_path/"generation.pt"
    checkpoint.write_bytes(b"trusted generation")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    event = {"schema": "tensorfem.panel-generation-event/1.0",
             "generation_committed": 7, "prior_event_sha256": None,
             "checkpoint_sha256": checkpoint_hash}
    event["event_sha256"] = hashlib.sha256(canonical(event).encode()).hexdigest()
    (events/"g000007.json").write_text(json.dumps(event))
    manifest = {"accepted_points": 7, "checkpoint_file": checkpoint.name,
                "checkpoint_sha256": checkpoint_hash}
    (tmp_path/"chunked-test.json").write_text(json.dumps(manifest))
    status = {"schema": "tensorfem.panel-generation-status/1.0",
              "state": "running", "current_generation": 7,
              "target_generation": 10, "last_event_sha256": event["event_sha256"],
              "reason": None, "updated_unix_seconds": 1.}
    status["status_sha256"] = hashlib.sha256(canonical(status).encode()).hexdigest()
    (tmp_path/"generation-status.json").write_text(json.dumps(status))
    assert verify_snapshot(tmp_path)["passed"] is True

    # Re-hashing a semantically detached status must not make it acceptable.
    detached = json.loads((tmp_path/"generation-status.json").read_text())
    detached["last_event_sha256"] = "0"*64
    detached.pop("status_sha256")
    detached["status_sha256"] = hashlib.sha256(canonical(detached).encode()).hexdigest()
    (tmp_path/"generation-status.json").write_text(json.dumps(detached))
    try:
        verify_snapshot(tmp_path)
    except ValueError as error:
        assert "status/event hash-link mismatch" in str(error)
    else:
        raise AssertionError("detached but re-hashed status was accepted")

    # Restore the linked status before testing checkpoint tampering.
    status["status_sha256"] = hashlib.sha256(canonical(
        {key: value for key, value in status.items() if key != "status_sha256"}
    ).encode()).hexdigest()
    (tmp_path/"generation-status.json").write_text(json.dumps(status))

    checkpoint.write_bytes(b"tampered")
    try:
        verify_snapshot(tmp_path)
    except ValueError as error:
        assert "checkpoint integrity mismatch" in str(error)
    else:
        raise AssertionError("tampered checkpoint was accepted")
