#!/usr/bin/env python3
"""Resume a panel path one immutable generation at a time."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
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
                   event_sha256: str | None,
                   attempt_event_sha256: str | None = None,
                   reason: str | None = None) -> None:
    value = {"schema": "tensorfem.panel-generation-status/1.0",
             "state": state, "current_generation": current,
             "target_generation": target, "last_event_sha256": event_sha256,
             "last_attempt_event_sha256": attempt_event_sha256 or event_sha256,
             "reason": reason, "updated_unix_seconds": time.time()}
    value["status_sha256"] = hashlib.sha256(canonical(value).encode()).hexdigest()
    atomic_json(path, value)


_EVENT_NAME = re.compile(r"^g(?P<generation>\d{6})(?:_attempt(?P<attempt>\d{3}))?\.json$")


def ordered_event_paths(events: Path) -> list[Path]:
    parsed = []
    for path in events.glob("g*.json"):
        match = _EVENT_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError("panel generation event filename is invalid")
        parsed.append((int(match.group("generation")),
                       int(match.group("attempt") or 1), path))
    parsed.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in parsed]


def latest_event_hashes(events: Path) -> tuple[str | None, str | None]:
    paths = ordered_event_paths(events)
    previous = None
    committed = None
    committed_generation = 0
    attempts: dict[int, int] = {}
    for path in paths:
        value = json.loads(path.read_text())
        recorded = value.pop("event_sha256")
        expected = hashlib.sha256(canonical(value).encode()).hexdigest()
        requested = int(value.get("generation_requested",
                                  value["generation_committed"]))
        base_name = f"g{requested:06d}.json"
        retry_prefix = f"g{requested:06d}_attempt"
        match = _EVENT_NAME.fullmatch(path.name)
        attempt = int(match.group("attempt") or 1) if match else -1
        if attempt != attempts.get(requested, 0)+1:
            raise ValueError("panel generation attempt sequence integrity mismatch")
        attempts[requested] = attempt
        if value.get("attempt_id", attempt) != attempt:
            raise ValueError("panel generation attempt identity mismatch")
        if (recorded != expected or value.get("prior_event_sha256") != previous
                or not (path.name == base_name or (
                    path.name.startswith(retry_prefix) and path.suffix == ".json"))):
            raise ValueError("panel generation event-chain integrity mismatch")
        previous = recorded
        generation = int(value["generation_committed"])
        advanced = bool(value.get(
            "commit_advanced",
            ("generation_requested" not in value)
            or (value.get("status") == "executed" and generation == requested),
        ))
        if advanced:
            if generation <= committed_generation:
                raise ValueError("panel generation committed generations are not monotonic")
            committed, committed_generation = recorded, generation
    return previous, committed


def latest_event_hash(events: Path) -> str | None:
    """Backward-compatible audit-chain head (including failed attempts)."""
    return latest_event_hashes(events)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--divisions", type=int, required=True)
    parser.add_argument("--normalized-step", type=float, required=True)
    parser.add_argument("--maximum-solver-step", type=float, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--point-wall-seconds", type=float, default=300.)
    parser.add_argument("--stop-at-post-peak", action="store_true")
    parser.add_argument("--automatic-step-control", action="store_true",
                        help="apply the hashed continuation step controller")
    parser.add_argument("--minimum-solver-step-divisor", type=int, default=128,
                        help="minimum arc step is nominal/divisor (robust mode may use 1024)")
    args = parser.parse_args()
    events = args.cache_dir/"events"
    events.mkdir(parents=True, exist_ok=True)
    previous_hash, committed_hash = latest_event_hashes(events)
    status_path = args.cache_dir/"generation-status.json"
    while True:
        manifests = sorted(args.cache_dir.glob("chunked-*.json"))
        if len(manifests) != 1:
            raise RuntimeError("cache must contain exactly one committed manifest")
        current = json.loads(manifests[0].read_text())
        accepted = int(current["accepted_points"])
        publish_status(status_path, state="running", current=accepted,
                       target=args.target, event_sha256=committed_hash,
                       attempt_event_sha256=previous_hash)
        if accepted >= args.target or (args.stop_at_post_peak
                                        and current.get("post_peak_observed")):
            publish_status(
                status_path,
                state=("post_peak" if current.get("post_peak_observed")
                       else "target_reached"),
                current=accepted, target=args.target, event_sha256=committed_hash,
                attempt_event_sha256=previous_hash,
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
            automatic_step_control=args.automatic_step_control,
            minimum_solver_step_divisor=args.minimum_solver_step_divisor,
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
        event["commit_advanced"] = bool(
            result.get("status") == "executed"
            and int(result.get("accepted_points", accepted)) >= requested)
        event_path = events/f"g{requested:06d}.json"
        attempt = 1
        if event_path.exists():
            attempt = 2
            while (events/f"g{requested:06d}_attempt{attempt:03d}.json").exists():
                attempt += 1
            event_path = events/f"g{requested:06d}_attempt{attempt:03d}.json"
        event["attempt_id"] = attempt
        event["event_sha256"] = hashlib.sha256(canonical(event).encode()).hexdigest()
        atomic_json(event_path, event)
        previous_hash = event["event_sha256"]
        if event["commit_advanced"]:
            committed_hash = previous_hash
        if (result.get("status") != "executed"
                or int(result.get("accepted_points", accepted)) < requested):
            publish_status(status_path, state="stopped",
                           current=int(result.get("accepted_points", accepted)),
                           target=args.target, event_sha256=committed_hash,
                           attempt_event_sha256=previous_hash,
                           reason=str(result.get("error")))
            break


if __name__ == "__main__":
    main()
