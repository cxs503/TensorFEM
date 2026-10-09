#!/usr/bin/env python3
"""Repair only the published pointer/status after a no-commit timed-out attempt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from tensorfem.marine_panel_execution import _canonical, _solution_state_sha256
from scripts.run_panel_generation_path import latest_event_hashes


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=Path)
    args = parser.parse_args()
    manifests = sorted(args.cache_dir.glob("chunked-*.json"))
    if len(manifests) != 1:
        raise SystemExit("cache must contain one manifest")
    path = manifests[0]
    manifest = json.loads(path.read_text())
    generation = int(manifest["accepted_points"])
    checkpoint = args.cache_dir/f"chunked-{manifest['job_key']}-g{generation:06d}.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    terminal = manifest["point_history"][-1]
    if _solution_state_sha256(saved["displacement"], saved["state"],
                              saved["load_factor"]) != terminal["state_sha256"]:
        raise SystemExit("checkpoint state does not match manifest terminal state")
    manifest["checkpoint_file"] = checkpoint.name
    manifest["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest.pop("evidence_sha256", None)
    manifest["evidence_sha256"] = hashlib.sha256(
        _canonical(manifest).encode()).hexdigest()
    atomic_json(path, manifest)
    attempt, committed = latest_event_hashes(args.cache_dir/"events")
    status_path = args.cache_dir/"generation-status.json"
    old = json.loads(status_path.read_text())
    status = {
        "schema": "tensorfem.panel-generation-status/1.0",
        "state": "stopped", "current_generation": generation,
        "target_generation": int(old["target_generation"]),
        "last_event_sha256": committed,
        "last_attempt_event_sha256": attempt,
        "reason": old.get("reason"), "updated_unix_seconds": time.time(),
    }
    status["status_sha256"] = hashlib.sha256(
        _canonical(status).encode()).hexdigest()
    atomic_json(status_path, status)


if __name__ == "__main__":
    main()
