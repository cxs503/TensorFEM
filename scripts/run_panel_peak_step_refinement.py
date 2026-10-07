#!/usr/bin/env python3
"""Restart a bounded peak window with a smaller arc-step cap.

The source generation is immutable.  This command verifies its scheduler
evidence, copies one pre-peak committed state into a distinct control identity,
and publishes fail-closed sensitivity evidence in the source cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from tensorfem.marine_panel_execution import (
    CHUNKED_SCHEMA, _canonical, _exact_checkpoint_value,
    _solution_state_sha256, execute_panel_chunked_job,
)
from tensorfem.panel_generation_readiness import (
    build_peak_step_sensitivity_evidence, panel_generation_readiness,
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_refined_restart(source_root: Path, destination: Path, *,
                            refined_maximum_step: float,
                            pre_peak_points: int = 5) -> tuple[dict, Path, str]:
    manifests = sorted(source_root.glob("chunked-*.json"))
    if len(manifests) != 1:
        raise ValueError("source cache must contain exactly one manifest")
    source_path = manifests[0]
    source = json.loads(source_path.read_text())
    if source.get("schema") != CHUNKED_SCHEMA or not source.get("peak_confirmed"):
        raise ValueError("source path has no confirmed peak")
    peak_index = int(source["peak_point_index"])
    restart_index = peak_index-pre_peak_points
    if restart_index < 0:
        raise ValueError("peak path is too short for a pre-peak restart")
    point = source["point_history"][restart_index]
    generation = int(point["step"])
    candidates = sorted(source_root.glob(
        f"chunked-{source['job_key']}-g{generation:06d}.pt"))
    if len(candidates) != 1:
        raise ValueError("pre-peak checkpoint is missing or ambiguous")
    source_checkpoint = candidates[0]
    saved = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if _solution_state_sha256(saved["displacement"], saved["state"],
                              saved["load_factor"]) != point["state_sha256"]:
        raise ValueError("pre-peak checkpoint state does not match path history")
    baseline_cap = float(source["solver_maximum_step"])
    if not 0 < refined_maximum_step < baseline_cap:
        raise ValueError("refined maximum step must be smaller than source cap")

    identity = {name: source[name] for name in (
        "schema", "energy_definition", "divisions", "normalized_arc_step",
        "arc_metric", "relative_equilibrium_tolerance")}
    identity["solver_maximum_step"] = float(refined_maximum_step)
    if source.get("automatic_step_control"):
        identity["automatic_step_control"] = True
    if source.get("line_search") is not None:
        identity["line_search"] = source["line_search"]
    key = hashlib.sha256(_canonical(identity).encode()).hexdigest()[:20]
    destination.mkdir(parents=True, exist_ok=True)
    target_checkpoint = destination/f"chunked-{key}-g{generation:06d}.pt"
    target_manifest = destination/f"chunked-{key}.json"
    if target_checkpoint.exists() or target_manifest.exists():
        raise FileExistsError("refinement restart already exists")
    migrated = dict(saved)
    migrated["step_size"] = min(float(saved["step_size"]), refined_maximum_step)
    migration = {
        "schema": "tensorfem.panel-step-refinement-migration/1.0",
        "source_manifest_sha256": sha256(source_path),
        "source_checkpoint_file": source_checkpoint.name,
        "source_checkpoint_sha256": sha256(source_checkpoint),
        "source_generation": generation,
        "source_state_sha256": point["state_sha256"],
        "baseline_maximum_step": baseline_cap,
        "refined_maximum_step": float(refined_maximum_step),
        "mechanical_state_exact": True,
        "energy_ledger_exact": True,
    }
    migration["migration_sha256"] = hashlib.sha256(
        _canonical(migration).encode()).hexdigest()
    migrated["migration"] = migration
    temporary = target_checkpoint.with_suffix(".pt.tmp")
    torch.save(migrated, temporary)
    os.replace(temporary, target_checkpoint)
    reloaded = torch.load(target_checkpoint, map_location="cpu", weights_only=False)
    for name in ("displacement", "state", "previous_increment", "load_factor",
                 "reference_recoverable_energy", "cumulative_external_work",
                 "cumulative_plastic_dissipation", "energy_definition",
                 "energy_prefix_complete"):
        if not _exact_checkpoint_value(saved.get(name), reloaded.get(name)):
            raise RuntimeError("refinement migration changed state or energy ledger")
    controls = dict(source.get("controls", {}))
    controls["solver_maximum_step"] = float(refined_maximum_step)
    history = source["point_history"][:generation]
    chunks = [item for item in source.get("chunks", [])
              if int(item.get("last_step", 0)) <= generation]
    decisions = [item for item in source.get("nonlinear_controller_decisions", [])
                 if int(item.get("observations", {}).get(
                     "accepted_points", generation)) <= generation]
    saved_decision = saved.get("nonlinear_controller_decision_sha256")
    if saved_decision is not None and (
            not decisions or decisions[-1].get("decision_sha256") != saved_decision):
        raise ValueError("pre-peak checkpoint controller identity mismatch")
    target = {
        **source, **identity, "job_key": key, "controls": controls,
        "accepted_points": generation, "point_history": history,
        "chunks": chunks, "nonlinear_controller_decisions": decisions,
        "nonlinear_controller_latest": decisions[-1] if decisions else None,
        "checkpoint_file": target_checkpoint.name,
        "checkpoint_sha256": sha256(target_checkpoint), "migration": migration,
        "status": "executed", "replayed": False,
    }
    for name in ("peak_force_n", "peak_point_index", "peak_confirmed",
                 "post_peak_observed", "evidence_sha256", "error"):
        target.pop(name, None)
    atomic_json(target_manifest, target)
    return source, source_path, point["state_sha256"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--refined-maximum-step", type=float, required=True)
    parser.add_argument("--pre-peak-points", type=int, default=5)
    parser.add_argument("--wall-seconds", type=float, default=7200.)
    args = parser.parse_args()
    readiness = panel_generation_readiness(args.cache_dir)
    if not readiness["post_peak_observed"]:
        raise SystemExit("source path has no verified post-peak evidence")
    destination = args.cache_dir/"peak-step-refinement"
    source, source_path, restart_hash = prepare_refined_restart(
        args.cache_dir, destination,
        refined_maximum_step=args.refined_maximum_step,
        pre_peak_points=args.pre_peak_points)
    refined = execute_panel_chunked_job(
        int(source["divisions"]), float(source["normalized_arc_step"]),
        steps=int(source["accepted_points"]), cache_dir=destination,
        chunk_size=1, resume=True, maximum_wall_seconds=args.wall_seconds,
        maximum_solver_step=args.refined_maximum_step,
        relative_equilibrium_tolerance=float(source["relative_equilibrium_tolerance"]),
        line_search=source.get("line_search"),
        automatic_step_control=bool(source.get("automatic_step_control", False)),
    )
    refined_manifests = sorted(destination.glob("chunked-*.json"))
    refined_path = refined_manifests[0]
    restart_generation = int(refined["migration"]["source_generation"])
    restart_exact = bool(
        refined["point_history"][restart_generation-1]["state_sha256"] == restart_hash
        and refined["migration"]["mechanical_state_exact"]
        and refined["migration"]["energy_ledger_exact"])
    energy_passed = bool(
        refined.get("energy_balance_passed") and refined.get("energy_prefix_complete")
        and all(p.get("controller_incremental_energy_balance_gate", {}).get("passed")
                for p in refined["point_history"][restart_generation:]))
    evidence = build_peak_step_sensitivity_evidence(
        divisions=int(source["divisions"]),
        source_checkpoint_sha256=refined["migration"]["source_checkpoint_sha256"],
        source_checkpoint_file=refined["migration"]["source_checkpoint_file"],
        source_manifest_sha256=sha256(source_path),
        refined_manifest_file=str(refined_path.relative_to(args.cache_dir)),
        refined_manifest_sha256=sha256(refined_path),
        baseline_maximum_step=float(source["solver_maximum_step"]),
        refined_maximum_step=args.refined_maximum_step,
        baseline_peak_force_n=float(source["peak_force_n"]),
        refined_peak_force_n=float(refined["peak_force_n"]),
        energy_balance_passed=energy_passed,
        restart_state_exact=restart_exact,
    )
    if not refined.get("post_peak_observed"):
        evidence["passed"] = False
        evidence["blocking_reason"] = "refined_path_has_no_post_peak"
        evidence.pop("evidence_sha256", None)
        evidence["evidence_sha256"] = hashlib.sha256(
            _canonical(evidence).encode()).hexdigest()
    atomic_json(args.cache_dir/"peak-step-sensitivity.json", evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
