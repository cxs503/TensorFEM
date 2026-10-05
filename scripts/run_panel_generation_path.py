#!/usr/bin/env python3
"""Resume a panel path one immutable generation at a time."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from tensorfem.marine_panel_execution import execute_panel_chunked_job


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    os.replace(temporary, path)


def publish_status(path: Path, *, state: str, current: int, target: int,
                   event_sha256: str | None, reason: str | None = None) -> None:
    value = {"schema": "tensorfem.panel-generation-status/1.0",
             "state": state, "current_generation": current,
             "target_generation": target, "last_event_sha256": event_sha256,
             "reason": reason, "updated_unix_seconds": time.time()}
    value["status_sha256"] = hashlib.sha256(canonical(value).encode()).hexdigest()
    atomic_json(path, value)


def latest_event_hash(events: Path) -> str | None:
    paths = sorted(events.glob("g*.json"))
    previous = None
    for path in paths:
        value = json.loads(path.read_text())
        recorded = value.pop("event_sha256")
        expected = hashlib.sha256(canonical(value).encode()).hexdigest()
        expected_name = f"g{int(value['generation_committed']):06d}.json"
        if (recorded != expected or value.get("prior_event_sha256") != previous
                or path.name != expected_name):
            raise ValueError("panel generation event-chain integrity mismatch")
        previous = recorded
    return previous


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--divisions", type=int, required=True)
    parser.add_argument("--normalized-step", type=float, required=True)
    parser.add_argument("--maximum-solver-step", type=float, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--point-wall-seconds", type=float, default=300.)
    parser.add_argument("--stop-at-post-peak", action="store_true")
    args = parser.parse_args()
    events = args.cache_dir/"events"
    events.mkdir(parents=True, exist_ok=True)
    previous_hash = latest_event_hash(events)
    status_path = args.cache_dir/"generation-status.json"
    while True:
        manifests = sorted(args.cache_dir.glob("chunked-*.json"))
        if len(manifests) != 1:
            raise RuntimeError("cache must contain exactly one committed manifest")
        current = json.loads(manifests[0].read_text())
        accepted = int(current["accepted_points"])
        publish_status(status_path, state="running", current=accepted,
                       target=args.target, event_sha256=previous_hash)
        if accepted >= args.target or (args.stop_at_post_peak
                                        and current.get("post_peak_observed")):
            publish_status(
                status_path,
                state=("post_peak" if current.get("post_peak_observed")
                       else "target_reached"),
                current=accepted, target=args.target, event_sha256=previous_hash,
            )
            break
        requested = accepted+1
        started = time.monotonic()
        result = execute_panel_chunked_job(
            args.divisions, args.normalized_step, steps=requested,
            cache_dir=args.cache_dir, chunk_size=1, resume=True,
            maximum_wall_seconds=args.point_wall_seconds,
            maximum_solver_step=args.maximum_solver_step,
            line_search="backtracking",
        )
        event = {
            "schema": "tensorfem.panel-generation-event/1.0",
            "generation_requested": requested,
            "generation_committed": int(result.get("accepted_points", accepted)),
            "status": result.get("status"),
            "error": result.get("error"),
            "elapsed_seconds": time.monotonic()-started,
            "job_key": result.get("job_key"),
            "checkpoint_file": result.get("checkpoint_file"),
            "checkpoint_sha256": result.get("checkpoint_sha256"),
            "peak_force_n": result.get("peak_force_n"),
            "peak_confirmed": result.get("peak_confirmed"),
            "post_peak_observed": result.get("post_peak_observed"),
            "terminal_point": (result.get("point_history") or [None])[-1],
            "terminal_chunk": (result.get("chunks") or [None])[-1],
            "prior_event_sha256": previous_hash,
        }
        event["event_sha256"] = hashlib.sha256(canonical(event).encode()).hexdigest()
        atomic_json(events/f"g{requested:06d}.json", event)
        previous_hash = event["event_sha256"]
        if (result.get("status") != "executed"
                or int(result.get("accepted_points", accepted)) < requested):
            publish_status(status_path, state="stopped",
                           current=int(result.get("accepted_points", accepted)),
                           target=args.target, event_sha256=previous_hash,
                           reason=str(result.get("error")))
            break


if __name__ == "__main__":
    main()
