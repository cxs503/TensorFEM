"""Load-window scheduling for resumable marine-panel continuation.

The scheduler deliberately sits above :mod:`marine_panel_execution`: it never
changes an accepted solution or material history.  It only selects the arc
radius used to predict the *next* point and preserves the checkpoint hashes
after that control value is changed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

from .marine_panel_execution import (
    CHUNKED_SCHEMA,
    _atomic_json,
    _canonical,
    execute_panel_chunked_job,
)
from .marine_panel_ultimate_fe import build_panel_case, classical_panel_references
from .panel_path_evidence import ENERGY_DEFINITION


WINDOW_SCHEMA = "tensorfem.panel-peak-window/1.0"


@dataclass(frozen=True)
class PeakWindowPolicy:
    """Dimensionless continuation radii and entry ratios for three zones."""

    coarse_arc_step: float = 0.10
    approach_arc_step: float = 0.05
    peak_arc_step: float = 0.02
    approach_ratio: float = 0.75
    peak_ratio: float = 0.90

    def validate(self) -> None:
        if not (self.coarse_arc_step >= self.approach_arc_step
                >= self.peak_arc_step > 0):
            raise ValueError("arc steps must be positive and non-increasing")
        if not (0 < self.approach_ratio < self.peak_ratio < 1):
            raise ValueError("window ratios must satisfy 0 < approach < peak < 1")


@dataclass(frozen=True)
class PeakTargetEstimate:
    force_n: float
    source: str
    is_confirmed_peak: bool
    source_points: int


def estimate_peak_window_target(
    coarse_manifest: Mapping[str, object] | None,
    *,
    divisions: int = 8,
) -> PeakTargetEstimate:
    """Return a conservative window anchor from the 4x4 path and mechanics.

    A confirmed 4x4 peak is transferable as the best available target.  An
    unfinished pre-yield path is *not* extrapolated as a peak: the classical
    elastic buckling load supplies a conservative entry anchor, bounded below
    by the largest actually accepted 4x4 force.  This makes early refinement
    safe without claiming that the classical value is an ultimate load.
    """
    references = classical_panel_references(build_panel_case(divisions))
    buckling = float(references["classical_elastic_buckling_force"])
    squash = float(references["gross_section_squash_force"])
    manifest = coarse_manifest or {}
    history = manifest.get("point_history") or []
    forces = [float(point["force_n"]) for point in history]
    observed = max(forces, default=0.0)
    confirmed = bool(manifest.get("peak_confirmed"))
    reported_peak = manifest.get("peak_force_n")
    if confirmed and reported_peak is not None and float(reported_peak) > 0:
        return PeakTargetEstimate(float(reported_peak), "confirmed_4x4_peak", True,
                                  len(history))
    # A resumable writer intentionally omits derived peak fields while its
    # manifest is ``running``.  Preserve a physically observed peak when a
    # sufficiently long accepted suffix has already fallen by at least 0.2%,
    # instead of incorrectly reverting the refined mesh to a higher classical
    # buckling anchor during that write window.
    if forces:
        peak_index = forces.index(observed)
        post_peak_points = len(forces) - peak_index - 1
        if post_peak_points >= 5 and forces[-1] <= observed * .998:
            return PeakTargetEstimate(
                observed, "observed_4x4_peak_from_running_history", True,
                len(history),
            )
    anchor = min(squash, max(buckling, observed))
    return PeakTargetEstimate(anchor, "4x4_lower_bound_plus_classical_buckling_anchor",
                              False, len(history))


def select_peak_window(force_n: float, target: PeakTargetEstimate,
                       policy: PeakWindowPolicy = PeakWindowPolicy()) -> tuple[str, float]:
    """Select the next normalized arc radius without changing path direction."""
    policy.validate()
    if target.force_n <= 0 or force_n < 0:
        raise ValueError("forces must define a positive compression target")
    ratio = force_n / target.force_n
    if ratio >= policy.peak_ratio:
        return "peak", policy.peak_arc_step
    if ratio >= policy.approach_ratio:
        return "approach", policy.approach_arc_step
    return "coarse", policy.coarse_arc_step


def _checkpoint_paths(cache_dir: str | Path, normalized_arc_step: float,
                      relative_equilibrium_tolerance: float, *,
                      divisions: int = 8) -> tuple[Path, Path]:
    identity = {"schema": CHUNKED_SCHEMA,
                "energy_definition": ENERGY_DEFINITION,
                "divisions": divisions,
                "normalized_arc_step": normalized_arc_step,
                "relative_equilibrium_tolerance": relative_equilibrium_tolerance}
    key = hashlib.sha256(_canonical(identity).encode()).hexdigest()[:20]
    root = Path(cache_dir)
    return root / f"chunked-{key}.json", root / f"chunked-{key}.pt"


def retune_next_arc_step(manifest_path: str | Path, checkpoint_path: str | Path,
                         *, normalized_arc_step: float,
                         characteristic_displacement_m: float) -> None:
    """Atomically retune a trusted local restart and renew both integrity hashes."""
    if normalized_arc_step <= 0 or characteristic_displacement_m <= 0:
        raise ValueError("arc step and characteristic displacement must be positive")
    manifest_path, checkpoint_path = Path(manifest_path), Path(checkpoint_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binary = checkpoint_path.read_bytes()
    if manifest.get("checkpoint_sha256") != hashlib.sha256(binary).hexdigest():
        raise ValueError("panel checkpoint integrity mismatch before retuning")
    saved = torch.load(checkpoint_path, weights_only=False)
    saved["step_size"] = normalized_arc_step * characteristic_displacement_m
    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(saved, temporary)
    os.replace(temporary, checkpoint_path)
    manifest["checkpoint_sha256"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    manifest["scheduled_next_normalized_arc_step"] = float(normalized_arc_step)
    manifest.pop("evidence_sha256", None)
    manifest["evidence_sha256"] = hashlib.sha256(
        _canonical(manifest).encode()).hexdigest()
    _atomic_json(manifest_path, manifest)


def execute_panel_peak_window(
    *,
    divisions: int,
    cache_dir: str | Path,
    target_steps: int,
    coarse_4x4_manifest: Mapping[str, object] | None = None,
    policy: PeakWindowPolicy = PeakWindowPolicy(),
    maximum_wall_seconds: float | None = None,
    relative_equilibrium_tolerance: float = 1e-6,
) -> dict[str, object]:
    """Advance a hashed refined-mesh path one recoverable point at a time.

    The physical checkpoint identity uses the coarse radius, while the saved
    next-step radius is reset before every one-point chunk.  Hence the solver's
    automatic growth cannot skip the peak window and every accepted point is
    persisted before another expensive 8x8 solve starts.
    """
    policy.validate()
    if divisions < 4 or divisions % 2:
        raise ValueError("divisions must be an even integer >= 4")
    if target_steps < 1:
        raise ValueError("target_steps must be positive")
    target = estimate_peak_window_target(coarse_4x4_manifest, divisions=divisions)
    started = time.monotonic()
    thickness = build_panel_case(divisions).model.thickness
    manifest_path, checkpoint_path = _checkpoint_paths(
        cache_dir, policy.coarse_arc_step, relative_equilibrium_tolerance,
        divisions=divisions)
    schedule: list[dict[str, object]] = []
    last_result: Mapping[str, object] = {}
    while True:
        disk_result = (json.loads(manifest_path.read_text(encoding="utf-8"))
                       if manifest_path.exists() else {})
        # A writer may have returned a newly accepted point before its manifest
        # is visible through a renamed/migrated path.  Never let a stale disk
        # prefix move the scheduler backwards and replay the same target.
        existing = max(
            (disk_result, last_result),
            key=lambda item: int(item.get("accepted_points", 0)),
        )
        accepted = int(existing.get("accepted_points", 0))
        if accepted >= target_steps:
            result = existing
            break
        terminal_force = float((existing.get("point_history") or [{}])[-1].get("force_n", 0.0))
        zone, next_step = select_peak_window(terminal_force, target, policy)
        if checkpoint_path.exists() and manifest_path.exists():
            retune_next_arc_step(
                manifest_path, checkpoint_path,
                normalized_arc_step=next_step,
                characteristic_displacement_m=thickness,
            )
        remaining_wall = (None if maximum_wall_seconds is None else
                          maximum_wall_seconds - (time.monotonic() - started))
        if remaining_wall is not None and remaining_wall <= 0:
            result = existing
            break
        result = execute_panel_chunked_job(
            divisions, policy.coarse_arc_step, steps=accepted + 1,
            cache_dir=cache_dir, chunk_size=1, resume=True,
            maximum_wall_seconds=remaining_wall,
            relative_equilibrium_tolerance=relative_equilibrium_tolerance,
        )
        last_result = result
        schedule.append({"accepted_before": accepted, "zone": zone,
                         "target_force_n": target.force_n,
                         "scheduled_normalized_arc_step": next_step,
                         "accepted_after": int(result.get("accepted_points", accepted))})
        if int(result.get("accepted_points", accepted)) <= accepted:
            break
    return {"schema": WINDOW_SCHEMA, "mesh_divisions": divisions,
            "target": asdict(target), "policy": asdict(policy),
            "schedule": schedule, "path": result}


def execute_8x8_peak_window(
    *,
    cache_dir: str | Path,
    target_steps: int,
    coarse_4x4_manifest: Mapping[str, object] | None = None,
    policy: PeakWindowPolicy = PeakWindowPolicy(),
    maximum_wall_seconds: float | None = None,
    relative_equilibrium_tolerance: float = 1e-6,
) -> dict[str, object]:
    """Backward-compatible 8x8 qualification-window entry point."""
    return execute_panel_peak_window(
        divisions=8,
        cache_dir=cache_dir,
        target_steps=target_steps,
        coarse_4x4_manifest=coarse_4x4_manifest,
        policy=policy,
        maximum_wall_seconds=maximum_wall_seconds,
        relative_equilibrium_tolerance=relative_equilibrium_tolerance,
    )
