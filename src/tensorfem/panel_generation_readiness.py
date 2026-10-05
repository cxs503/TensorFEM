"""Machine-readable readiness evidence for generation-scheduled panel paths."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


SCHEMA = "tensorfem.panel-generation-readiness/1.0"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def panel_generation_readiness(cache_dir: str | Path) -> dict[str, object]:
    """Build fail-closed prefix/peak evidence without making a v1 claim."""
    root = Path(cache_dir)
    manifests = sorted(root.glob("chunked-*.json"))
    events = sorted((root/"events").glob("g*.json"))
    status_path = root/"generation-status.json"
    if len(manifests) != 1 or not events or not status_path.exists():
        raise ValueError("panel generation evidence is incomplete")
    manifest = json.loads(manifests[0].read_text())
    status = json.loads(status_path.read_text())
    recorded_status = status.pop("status_sha256", None)
    if recorded_status != hashlib.sha256(_canonical(status).encode()).hexdigest():
        raise ValueError("panel generation status integrity mismatch")
    previous = None
    parsed = []
    for path in events:
        event = json.loads(path.read_text())
        recorded = event.pop("event_sha256", None)
        if (recorded != hashlib.sha256(_canonical(event).encode()).hexdigest()
                or event.get("prior_event_sha256") != previous):
            raise ValueError("panel generation event-chain integrity mismatch")
        event["event_sha256"] = recorded
        previous = recorded
        parsed.append(event)
    checkpoint = root/str(manifest["checkpoint_file"])
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_hash != manifest.get("checkpoint_sha256"):
        raise ValueError("panel generation checkpoint integrity mismatch")
    generation = int(manifest["accepted_points"])
    terminal_event = parsed[-1]
    if int(terminal_event["generation_committed"]) != generation:
        raise ValueError("panel generation terminal event is not committed manifest")
    if status.get("last_event_sha256") != previous:
        raise ValueError("panel generation status is detached from terminal event")
    recent = parsed[-min(10, len(parsed)):]
    points = [event["terminal_point"] for event in recent]
    forces = [float(point["force_n"]) for point in points]
    slopes = [right-left for left, right in zip(forces, forces[1:])]
    slope = sum(slopes)/len(slopes) if slopes else None
    energy_passed = all(point["energy_balance_gate"]["passed"]
                        and point["controller_incremental_energy_balance_gate"]["passed"]
                        for point in points)
    equilibrium = max(float(point["equilibrium_relative_norm"]) for point in points)
    post_peak = bool(manifest.get("post_peak_observed"))
    evidence = {
        "schema": SCHEMA,
        "integrity_passed": True,
        "generation": generation,
        "scheduler_state": status.get("state"),
        "scheduler_target": status.get("target_generation"),
        "checkpoint_sha256": checkpoint_hash,
        "last_event_sha256": previous,
        "peak_force_n": manifest.get("peak_force_n"),
        "peak_confirmed": bool(manifest.get("peak_confirmed")),
        "post_peak_observed": post_peak,
        "recent_mean_force_increment_n_per_generation": slope,
        "recent_terminal_yielded_fraction": float(points[-1]["yielded_fraction"]),
        "recent_energy_gates_passed": energy_passed,
        "recent_maximum_equilibrium_relative_norm": equilibrium,
        "qualification_state": ("post_peak_evidence" if post_peak
                                else "verified_monotonic_prefix"),
        "v1_ready": False,
        "v1_boundary": ("This adapter reports panel-path evidence only; v1 also "
                        "requires cross-mesh convergence and release gates."),
    }
    evidence["evidence_sha256"] = hashlib.sha256(
        _canonical(evidence).encode()).hexdigest()
    return evidence

