import hashlib
import json

from scripts.run_panel_generation_path import canonical, latest_event_hash


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

