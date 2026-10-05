#!/usr/bin/env python3
"""Print the last immutable panel-generation event and process status."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from run_panel_generation_path import canonical, latest_event_hash
except ImportError:  # imported as ``scripts.watch_panel_generation_path``
    from scripts.run_panel_generation_path import canonical, latest_event_hash


def verify_snapshot(root: Path) -> dict[str, object]:
    """Verify all published hashes while tolerating the atomic commit window."""
    events = sorted((root/"events").glob("g*.json"))
    manifests = sorted(root.glob("chunked-*.json"))
    status_path = root/"generation-status.json"
    if len(manifests) != 1 or not status_path.exists():
        raise ValueError("panel scheduler snapshot is incomplete")
    status = json.loads(status_path.read_text())
    recorded_status = status.pop("status_sha256", None)
    if recorded_status != hashlib.sha256(canonical(status).encode()).hexdigest():
        raise ValueError("panel scheduler status integrity mismatch")
    event_hash = latest_event_hash(root/"events")
    manifest = json.loads(manifests[0].read_text())
    checkpoint = root/str(manifest["checkpoint_file"])
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_hash != manifest.get("checkpoint_sha256"):
        raise ValueError("panel scheduler checkpoint integrity mismatch")
    event_generation = int(status["current_generation"])
    if events:
        last = json.loads(events[-1].read_text())
        event_generation = int(last["generation_committed"])
        if event_hash != last["event_sha256"]:
            raise ValueError("panel scheduler terminal event mismatch")
    manifest_generation = int(manifest["accepted_points"])
    # Manifest publication precedes event publication.  A watcher in that
    # millisecond-scale window may observe exactly one pending event.
    if manifest_generation-event_generation not in (0, 1):
        raise ValueError("panel scheduler manifest/event generation mismatch")
    if int(status["current_generation"]) > manifest_generation:
        raise ValueError("panel scheduler status is ahead of committed state")
    status_generation = int(status["current_generation"])
    status_event_hash = status.get("last_event_sha256")
    if status_generation == event_generation:
        expected_status_event_hash = event_hash
    elif (events and status_generation == event_generation-1
          and manifest_generation == event_generation):
        # Event publication succeeded and the scheduler has not yet entered
        # the next loop to refresh status.  The status must reference exactly
        # the predecessor recorded by that newly published event.
        expected_status_event_hash = last.get("prior_event_sha256")
    else:
        raise ValueError("panel scheduler status/event generation mismatch")
    if status_event_hash != expected_status_event_hash:
        raise ValueError("panel scheduler status/event hash-link mismatch")
    return {"passed": True, "status_schema": status.get("schema"),
            "manifest_generation": manifest_generation,
            "event_generation": event_generation,
            "event_publication_pending": manifest_generation == event_generation+1,
            "last_event_sha256": event_hash,
            "checkpoint_sha256": checkpoint_hash}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=Path)
    args = parser.parse_args()
    events = sorted((args.cache_dir/"events").glob("g*.json"))
    manifests = sorted(args.cache_dir.glob("chunked-*.json"))
    result = {"event_count": len(events)}
    status_path = args.cache_dir/"generation-status.json"
    if status_path.exists():
        result["scheduler"] = json.loads(status_path.read_text())
    if manifests:
        manifest = json.loads(manifests[0].read_text())
        result.update({name: manifest.get(name) for name in (
            "status", "accepted_points", "peak_force_n", "peak_confirmed",
            "post_peak_observed", "checkpoint_file", "checkpoint_sha256",
        )})
    if events:
        event = json.loads(events[-1].read_text())
        result["last_event"] = {name: event.get(name) for name in (
            "generation_requested", "generation_committed", "status", "error",
            "elapsed_seconds", "event_sha256",
        )}
    result["integrity"] = verify_snapshot(args.cache_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
