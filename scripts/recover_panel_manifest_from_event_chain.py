#!/usr/bin/env python3
"""Recover a regressed panel manifest from immutable source/checkpoint/events."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from tensorfem.marine_panel_execution import _canonical, _solution_state_sha256
from scripts.run_panel_generation_path import latest_event_hashes, ordered_event_paths


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    os.replace(temporary, path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--qualification-root", type=Path,
                        default=Path(".qualification"))
    args = parser.parse_args()
    manifest_paths = sorted(args.cache_dir.glob("chunked-*.json"))
    if len(manifest_paths) != 1:
        raise SystemExit("cache must contain exactly one manifest")
    manifest_path = manifest_paths[0]
    damaged_bytes = manifest_path.read_bytes()
    damaged = json.loads(damaged_bytes)
    audit_head, commit_head = latest_event_hashes(args.cache_dir/"events")
    events = []
    for path in ordered_event_paths(args.cache_dir/"events"):
        item = json.loads(path.read_text())
        requested = int(item["generation_requested"])
        generation = int(item["generation_committed"])
        advanced = item.get("commit_advanced",
                            item.get("status") == "executed"
                            and generation == requested)
        if advanced:
            events.append(item)
    if not events or events[-1]["event_sha256"] != commit_head:
        raise SystemExit("committed event head cannot be established")
    generation = int(events[-1]["generation_committed"])
    checkpoint = args.cache_dir/f"chunked-{damaged['job_key']}-g{generation:06d}.pt"
    if not checkpoint.exists():
        raise SystemExit("committed checkpoint is missing")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    strategy = saved.get("strategy_migration") or {}
    source_name = strategy.get("source_manifest")
    source_hash = strategy.get("source_manifest_sha256")
    candidates = [path for path in args.qualification_root.rglob(str(source_name))
                  if digest(path) == source_hash]
    if len(candidates) != 1:
        raise SystemExit("immutable migration source manifest is missing/ambiguous")
    source_path = candidates[0]
    source = json.loads(source_path.read_text())
    source_generation = int(strategy["source_generation"])
    committed_suffix = [item for item in events
                        if int(item["generation_committed"]) > source_generation]
    expected = list(range(source_generation+1, generation+1))
    actual = [int(item["generation_committed"]) for item in committed_suffix]
    if actual != expected:
        raise SystemExit("committed event suffix is not contiguous")
    history = list(source["point_history"])
    chunks = list(source.get("chunks", []))
    for item in committed_suffix:
        history.append(item["terminal_point"])
        chunks.append(item["terminal_chunk"])
    if len(history) != generation:
        raise SystemExit("recovered history length mismatch")
    if _solution_state_sha256(saved["displacement"], saved["state"],
                              saved["load_factor"]) != history[-1]["state_sha256"]:
        raise SystemExit("checkpoint does not match recovered terminal state")
    forces = [float(point["force_n"]) for point in history]
    peak_index = forces.index(max(forces))
    peak_confirmed = bool(len(forces)-peak_index >= 3 and all(
        value < forces[peak_index]*(1-1e-4)
        for value in forces[peak_index+1:peak_index+3]))
    recovered = dict(damaged)
    recovered.update({
        "accepted_points": generation, "status": "executed", "error": None,
        "point_history": history, "chunks": chunks,
        "checkpoint_file": checkpoint.name, "checkpoint_sha256": digest(checkpoint),
        "migration": saved.get("migration"), "strategy_migration": strategy,
        "reference_recoverable_energy_j": saved.get("reference_recoverable_energy"),
        "external_work_j": saved.get("cumulative_external_work"),
        "plastic_dissipation_j": saved.get("cumulative_plastic_dissipation"),
        "energy_prefix_complete": saved.get("energy_prefix_complete", False),
        "peak_force_n": max(forces), "peak_point_index": peak_index,
        "peak_confirmed": peak_confirmed, "post_peak_observed": peak_confirmed,
        "maximum_yielded_fraction": max(float(p["yielded_fraction"]) for p in history),
        # The exact advisory decision objects after migration were not event
        # payloads. Preserve the verified source chain and restart a new
        # advisory suffix; this never changes the mechanical checkpoint.
        "nonlinear_controller_decisions": source.get(
            "nonlinear_controller_decisions", []),
        "nonlinear_controller_latest": source.get("nonlinear_controller_latest"),
    })
    recovered.pop("evidence_sha256", None)
    recovered["evidence_sha256"] = hashlib.sha256(
        _canonical(recovered).encode()).hexdigest()
    recovery = {
        "schema": "tensorfem.panel-manifest-recovery/1.0",
        "damaged_manifest_sha256": hashlib.sha256(damaged_bytes).hexdigest(),
        "source_manifest": str(source_path),
        "source_manifest_sha256": source_hash,
        "source_generation": source_generation,
        "recovered_generation": generation,
        "checkpoint_file": checkpoint.name,
        "checkpoint_sha256": digest(checkpoint),
        "committed_event_sha256": commit_head,
        "attempt_event_sha256": audit_head,
        "mechanical_state_exact": True,
        "event_suffix_contiguous": True,
        "created_unix_seconds": time.time(),
    }
    recovery["recovery_sha256"] = hashlib.sha256(
        _canonical(recovery).encode()).hexdigest()
    atomic_json(args.cache_dir/"manifest-recoveries"/
                f"g{generation:06d}.json", recovery)
    atomic_json(manifest_path, recovered)
    status_path = args.cache_dir/"generation-status.json"
    old = json.loads(status_path.read_text())
    status = {"schema": "tensorfem.panel-generation-status/1.0",
              "state": "stopped", "current_generation": generation,
              "target_generation": int(old["target_generation"]),
              "last_event_sha256": commit_head,
              "last_attempt_event_sha256": audit_head,
              "reason": "recovered committed manifest after no-commit timeout",
              "updated_unix_seconds": time.time()}
    status["status_sha256"] = hashlib.sha256(
        _canonical(status).encode()).hexdigest()
    atomic_json(status_path, status)


if __name__ == "__main__":
    main()
