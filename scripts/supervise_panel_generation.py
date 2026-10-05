#!/usr/bin/env python3
"""Audit a completed generation target and optionally run one safe extension."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tensorfem.panel_generation_readiness import panel_generation_readiness


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--divisions", type=int, required=True)
    parser.add_argument("--normalized-step", type=float, required=True)
    parser.add_argument("--maximum-solver-step", type=float, required=True)
    parser.add_argument("--point-wall-seconds", type=float, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.)
    parser.add_argument("--maximum-extension-seconds", type=float, default=7200.)
    args = parser.parse_args()
    while True:
        status_path = args.cache_dir/"generation-status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get("state") in {"target_reached", "post_peak", "stopped"}:
                break
        time.sleep(args.poll_seconds)
    readiness = panel_generation_readiness(args.cache_dir)
    atomic_json(args.cache_dir/"panel-readiness.json", readiness)
    if readiness["post_peak_observed"] or status.get("state") != "target_reached":
        return
    events = [json.loads(path.read_text()) for path in
              sorted((args.cache_dir/"events").glob("g*.json"))[-10:]]
    seconds = sum(float(event["elapsed_seconds"]) for event in events)/len(events)
    affordable = max(10, min(100, int(args.maximum_extension_seconds/max(seconds, 1.))))
    slope = readiness["recent_mean_force_increment_n_per_generation"]
    # Near-flat or negative unconfirmed paths receive a shorter diagnostic
    # extension; clearly rising paths can safely use the time-budgeted block.
    extension = min(25, affordable) if slope is not None and slope <= 0 else affordable
    target = int(readiness["generation"])+extension
    decision = {
        "schema": "tensorfem.panel-generation-extension/1.0",
        "source_generation": readiness["generation"],
        "source_readiness_sha256": readiness["evidence_sha256"],
        "mean_force_increment_n_per_generation": slope,
        "yielded_fraction": readiness["recent_terminal_yielded_fraction"],
        "mean_point_seconds": seconds,
        "maximum_extension_seconds": args.maximum_extension_seconds,
        "extension_points": extension, "target_generation": target,
        "reason": "monotonic verified prefix without confirmed post-peak",
    }
    decision["decision_sha256"] = hashlib.sha256(canonical(decision).encode()).hexdigest()
    atomic_json(args.cache_dir/"extension-decisions"/
                f"from-g{int(readiness['generation']):06d}.json", decision)
    command = [sys.executable, str(Path(__file__).with_name("run_panel_generation_path.py")),
               "--divisions", str(args.divisions), "--normalized-step",
               str(args.normalized_step), "--maximum-solver-step",
               str(args.maximum_solver_step), "--target", str(target),
               "--cache-dir", str(args.cache_dir), "--point-wall-seconds",
               str(args.point_wall_seconds), "--stop-at-post-peak"]
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    final = panel_generation_readiness(args.cache_dir)
    atomic_json(args.cache_dir/"panel-readiness.json", final)


if __name__ == "__main__":
    main()

