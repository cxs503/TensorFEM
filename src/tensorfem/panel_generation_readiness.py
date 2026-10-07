"""Machine-readable readiness evidence for generation-scheduled panel paths."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .panel_mesh_convergence import evaluate_panel_mesh_convergence


SCHEMA = "tensorfem.panel-generation-readiness/1.0"
AGGREGATE_SCHEMA = "tensorfem.panel-mesh-readiness-aggregate/1.0"
STEP_SENSITIVITY_SCHEMA = "tensorfem.panel-peak-step-sensitivity/1.0"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _ordered_events(root: Path) -> list[Path]:
    pattern = re.compile(r"^g(?P<generation>\d{6})(?:_attempt(?P<attempt>\d{3}))?\.json$")
    values = []
    for path in (root/"events").glob("g*.json"):
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError("panel generation event filename is invalid")
        values.append((int(match.group("generation")),
                       int(match.group("attempt") or 1), path))
    return [item[2] for item in sorted(values)]


def panel_generation_readiness(cache_dir: str | Path) -> dict[str, object]:
    """Build fail-closed prefix/peak evidence without making a v1 claim."""
    root = Path(cache_dir)
    manifests = sorted(root.glob("chunked-*.json"))
    events = _ordered_events(root)
    status_path = root/"generation-status.json"
    if len(manifests) != 1 or not events or not status_path.exists():
        raise ValueError("panel generation evidence is incomplete")
    manifest = json.loads(manifests[0].read_text())
    manifest_sha256 = hashlib.sha256(manifests[0].read_bytes()).hexdigest()
    status = json.loads(status_path.read_text())
    recorded_status = status.pop("status_sha256", None)
    if recorded_status != hashlib.sha256(_canonical(status).encode()).hexdigest():
        raise ValueError("panel generation status integrity mismatch")
    previous = None
    parsed = []
    attempts: dict[int, int] = {}
    for path in events:
        event = json.loads(path.read_text())
        recorded = event.pop("event_sha256", None)
        if (recorded != hashlib.sha256(_canonical(event).encode()).hexdigest()
                or event.get("prior_event_sha256") != previous):
            raise ValueError("panel generation event-chain integrity mismatch")
        requested = int(event.get("generation_requested",
                                  event["generation_committed"]))
        match = re.fullmatch(
            r"g\d{6}(?:_attempt(?P<attempt>\d{3}))?\.json", path.name)
        attempt = int(match.group("attempt") or 1) if match else -1
        if (attempt != attempts.get(requested, 0)+1
                or int(event.get("attempt_id", attempt)) != attempt):
            raise ValueError("panel generation attempt sequence integrity mismatch")
        attempts[requested] = attempt
        event["event_sha256"] = recorded
        previous = recorded
        parsed.append(event)
    checkpoint = root/str(manifest["checkpoint_file"])
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_hash != manifest.get("checkpoint_sha256"):
        raise ValueError("panel generation checkpoint integrity mismatch")
    generation = int(manifest["accepted_points"])
    committed_events = [event for event in parsed if bool(event.get(
        "commit_advanced",
        "generation_requested" not in event or event.get("status") == "executed"
        and int(event.get("generation_committed", -1))
        == int(event.get("generation_requested", -2)),
    ))]
    committed_generations = [int(event["generation_committed"])
                             for event in committed_events]
    if any(right <= left for left, right in zip(
            committed_generations, committed_generations[1:])):
        raise ValueError("panel generation committed generations are not monotonic")
    terminal_event = committed_events[-1] if committed_events else None
    if terminal_event is None or int(terminal_event["generation_committed"]) != generation:
        raise ValueError("panel generation terminal event is not committed manifest")
    if status.get("last_event_sha256") != terminal_event["event_sha256"]:
        raise ValueError("panel generation status is detached from committed event")
    attempt_head = status.get("last_attempt_event_sha256",
                              status.get("last_event_sha256"))
    if attempt_head != previous:
        raise ValueError("panel generation status is detached from attempt chain")
    recent = committed_events[-min(10, len(committed_events)):]
    points = [event["terminal_point"] for event in recent]
    forces = [float(point["force_n"]) for point in points]
    slopes = [right-left for left, right in zip(forces, forces[1:])]
    slope = sum(slopes)/len(slopes) if slopes else None
    energy_passed = all(point["energy_balance_gate"]["passed"]
                        and point["controller_incremental_energy_balance_gate"]["passed"]
                        for point in points)
    equilibrium = max(float(point["equilibrium_relative_norm"]) for point in points)
    post_peak = bool(manifest.get("post_peak_observed"))
    sensitivity_path = root/"peak-step-sensitivity.json"
    sensitivity = None
    sensitivity_error = "missing_peak_step_sensitivity"
    if sensitivity_path.exists():
        candidate = json.loads(sensitivity_path.read_text())
        recorded = candidate.get("evidence_sha256")
        clean_candidate = {key: value for key, value in candidate.items()
                           if key != "evidence_sha256"}
        if recorded != hashlib.sha256(
                _canonical(clean_candidate).encode()).hexdigest():
            sensitivity_error = "peak_step_sensitivity_hash_mismatch"
        elif candidate.get("schema") != STEP_SENSITIVITY_SCHEMA:
            sensitivity_error = "peak_step_sensitivity_schema_mismatch"
        elif int(candidate.get("divisions", 0)) != int(manifest["divisions"]):
            sensitivity_error = "peak_step_sensitivity_mesh_mismatch"
        elif candidate.get("source_manifest_sha256") != manifest_sha256:
            sensitivity_error = "peak_step_sensitivity_source_manifest_mismatch"
        elif not candidate.get("source_checkpoint_file"):
            sensitivity_error = "peak_step_sensitivity_source_checkpoint_missing"
        elif not (root/str(candidate["source_checkpoint_file"])).is_file():
            sensitivity_error = "peak_step_sensitivity_source_checkpoint_missing"
        elif hashlib.sha256((root/str(candidate["source_checkpoint_file"])).read_bytes()).hexdigest() != candidate.get("source_checkpoint_sha256"):
            sensitivity_error = "peak_step_sensitivity_source_checkpoint_hash_mismatch"
        elif not candidate.get("refined_manifest_file"):
            sensitivity_error = "peak_step_sensitivity_refined_manifest_missing"
        elif not (root/str(candidate["refined_manifest_file"])).is_file():
            sensitivity_error = "peak_step_sensitivity_refined_manifest_missing"
        elif hashlib.sha256((root/str(candidate["refined_manifest_file"])).read_bytes()).hexdigest() != candidate.get("refined_manifest_sha256"):
            sensitivity_error = "peak_step_sensitivity_refined_manifest_hash_mismatch"
        elif not candidate.get("energy_balance_passed"):
            sensitivity_error = "peak_step_sensitivity_energy_not_passed"
        elif not candidate.get("restart_state_exact"):
            sensitivity_error = "peak_step_sensitivity_restart_state_mismatch"
        else:
            sensitivity = candidate
            sensitivity_error = None
    evidence = {
        "schema": SCHEMA,
        "integrity_passed": True,
        "source_kind": "generation_scheduler",
        "divisions": int(manifest["divisions"]),
        "source_manifest_sha256": manifest_sha256,
        "arc_metric": manifest.get("arc_metric"),
        "energy_definition": manifest.get("energy_definition"),
        "generation": generation,
        "scheduler_state": status.get("state"),
        "scheduler_target": status.get("target_generation"),
        "checkpoint_sha256": checkpoint_hash,
        "last_event_sha256": terminal_event["event_sha256"],
        "last_attempt_event_sha256": previous,
        "peak_force_n": manifest.get("peak_force_n"),
        "peak_confirmed": bool(manifest.get("peak_confirmed")),
        "post_peak_observed": post_peak,
        "recent_mean_force_increment_n_per_generation": slope,
        "recent_terminal_yielded_fraction": float(points[-1]["yielded_fraction"]),
        "recent_energy_gates_passed": energy_passed,
        "recent_maximum_equilibrium_relative_norm": equilibrium,
        "peak_step_sensitivity": sensitivity,
        "peak_step_sensitivity_error": sensitivity_error,
        "qualification_state": ("post_peak_evidence" if post_peak
                                else "verified_monotonic_prefix"),
        "v1_ready": False,
        "v1_boundary": ("This adapter reports panel-path evidence only; v1 also "
                        "requires cross-mesh convergence and release gates."),
    }
    evidence["evidence_sha256"] = hashlib.sha256(
        _canonical(evidence).encode()).hexdigest()
    return evidence


def legacy_panel_generation_readiness(manifest_path: str | Path) -> dict[str, object]:
    """Adapt a hashed pre-scheduler peak/post-peak manifest without inference."""
    path = Path(manifest_path)
    manifest_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes)
    checkpoint = path.parent/str(manifest["checkpoint_file"])
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_hash != manifest.get("checkpoint_sha256"):
        raise ValueError("legacy panel checkpoint integrity mismatch")
    points = manifest.get("point_history", [])
    if not points:
        raise ValueError("legacy panel path has no accepted-point evidence")
    terminal = points[-1]
    evidence = {
        "schema": SCHEMA, "integrity_passed": True,
        "source_kind": "legacy_hashed_manifest",
        "divisions": int(manifest["divisions"]),
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "arc_metric": manifest.get(
            "arc_metric", manifest.get("controls", {}).get("arc_metric", "legacy_full")),
        "energy_definition": manifest.get("energy_definition"),
        "generation": int(manifest["accepted_points"]),
        "scheduler_state": "legacy_complete", "scheduler_target": None,
        "checkpoint_sha256": checkpoint_hash, "last_event_sha256": None,
        "peak_force_n": manifest.get("peak_force_n"),
        "peak_confirmed": bool(manifest.get("peak_confirmed")),
        "post_peak_observed": bool(manifest.get("post_peak_observed")),
        "recent_mean_force_increment_n_per_generation": None,
        "recent_terminal_yielded_fraction": float(terminal["yielded_fraction"]),
        "recent_energy_gates_passed": bool(
            terminal.get("energy_balance_gate", {}).get("passed")),
        "recent_maximum_equilibrium_relative_norm": float(
            terminal["equilibrium_relative_norm"]),
        "qualification_state": ("post_peak_evidence"
                                if manifest.get("post_peak_observed")
                                else "verified_monotonic_prefix"),
        "v1_ready": False,
        "v1_boundary": ("This adapter reports panel-path evidence only; v1 also "
                        "requires cross-mesh convergence and release gates."),
    }
    evidence["evidence_sha256"] = hashlib.sha256(
        _canonical(evidence).encode()).hexdigest()
    return evidence


def aggregate_panel_mesh_readiness(
    readiness_by_divisions: dict[int, dict[str, object]], *,
    relative_tolerance: float = .03,
) -> dict[str, object]:
    """Verify readiness hashes and fail closed into the existing mesh gate."""
    required = (4, 8, 12)
    accepted = []
    inputs = {}
    rejected = {}
    for divisions in required:
        item = readiness_by_divisions.get(divisions)
        if item is None:
            rejected[divisions] = "missing_readiness"
            continue
        clean = {key: value for key, value in item.items()
                 if key != "evidence_sha256"}
        digest = hashlib.sha256(_canonical(clean).encode()).hexdigest()
        if digest != item.get("evidence_sha256"):
            rejected[divisions] = "readiness_hash_mismatch"
        elif int(item.get("divisions", 0)) != divisions:
            rejected[divisions] = "mesh_identity_mismatch"
        elif not item.get("integrity_passed"):
            rejected[divisions] = "integrity_not_passed"
        elif not item.get("peak_confirmed"):
            rejected[divisions] = "peak_not_confirmed"
        elif not item.get("post_peak_observed"):
            rejected[divisions] = "post_peak_not_observed"
        elif item.get("arc_metric") != "dimensionally_scaled":
            rejected[divisions] = "incompatible_arc_metric"
        elif item.get("peak_step_sensitivity_error") is not None:
            rejected[divisions] = str(item["peak_step_sensitivity_error"])
        elif not item.get("peak_step_sensitivity", {}).get("passed"):
            rejected[divisions] = "peak_step_sensitivity_not_passed"
        else:
            accepted.append({"divisions": divisions,
                             "peak_force_n": item.get("peak_force_n"),
                             "peak_confirmed": True,
                             "post_peak_observed": True})
        inputs[divisions] = {"evidence_sha256": item.get("evidence_sha256"),
                             "source_manifest_sha256": item.get(
                                 "source_manifest_sha256")}
    mesh = evaluate_panel_mesh_convergence(
        accepted, required_divisions=required,
        relative_tolerance=relative_tolerance)
    status = mesh["status"] if not rejected else "blocked"
    clean = {"schema": AGGREGATE_SCHEMA, "status": status,
             "passed": status == "passed", "required_divisions": list(required),
             "inputs": inputs, "rejected": rejected,
             "panel_mesh_convergence": mesh}
    return {**clean, "report_sha256": hashlib.sha256(
        _canonical(clean).encode()).hexdigest()}


def build_peak_step_sensitivity_evidence(
    *, divisions: int, source_checkpoint_sha256: str,
    source_checkpoint_file: str, source_manifest_sha256: str,
    refined_manifest_file: str, refined_manifest_sha256: str,
    baseline_maximum_step: float, refined_maximum_step: float,
    baseline_peak_force_n: float, refined_peak_force_n: float,
    energy_balance_passed: bool, restart_state_exact: bool,
    relative_tolerance: float = .03,
) -> dict[str, object]:
    """Build the local, peak-window-only step sensitivity qualification gate."""
    if refined_maximum_step >= baseline_maximum_step:
        raise ValueError("refined maximum step must be smaller than baseline")
    denominator = max(abs(baseline_peak_force_n), abs(refined_peak_force_n), 1e-30)
    difference = abs(refined_peak_force_n-baseline_peak_force_n)/denominator
    clean = {
        "schema": STEP_SENSITIVITY_SCHEMA,
        "divisions": int(divisions),
        "method": "local_restart_from_pre_peak_checkpoint",
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_checkpoint_file": source_checkpoint_file,
        "source_manifest_sha256": source_manifest_sha256,
        "refined_manifest_file": refined_manifest_file,
        "refined_manifest_sha256": refined_manifest_sha256,
        "baseline_maximum_step": float(baseline_maximum_step),
        "refined_maximum_step": float(refined_maximum_step),
        "baseline_peak_force_n": float(baseline_peak_force_n),
        "refined_peak_force_n": float(refined_peak_force_n),
        "relative_peak_force_difference": difference,
        "relative_tolerance": float(relative_tolerance),
        "energy_balance_passed": bool(energy_balance_passed),
        "restart_state_exact": bool(restart_state_exact),
        "passed": bool(difference < relative_tolerance
                       and energy_balance_passed and restart_state_exact),
    }
    return {**clean, "evidence_sha256": hashlib.sha256(
        _canonical(clean).encode()).hexdigest()}
