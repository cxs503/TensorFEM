"""Fail-closed, resumable execution of the marine-panel qualification matrix.

The public contract uses dimensionless arc increments.  This module converts
them to the dimensional metric used by ``solve_finite_rotation_arc_path`` and
records preflight or solver failures instead of silently replacing the model.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import time
from typing import Iterable

import torch

from .finite_rotation_layered_shell4 import solve_finite_rotation_arc_path
from .layered_shell4_plasticity import layered_shell4_local_frame
from .marine_panel_ultimate_fe import build_panel_case, classical_panel_references


SCHEMA = "tensorfem.marine-panel-execution/1.0"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def panel_geometry_preflight(case) -> dict[str, object]:
    """Measure warpage and exercise the shallow-facet projection guard."""
    worst = 0.0
    worst_element = -1
    first_invalid = -1
    first_error = None
    for number, connectivity in enumerate(case.model.elements):
        xyz = case.model.nodes[connectivity]
        centred = xyz - xyz.mean(0)
        _, singular, vh = torch.linalg.svd(centred, full_matrices=False)
        normal = vh[-1]
        scale = float(singular[0])
        ratio = (float((centred @ normal).abs().max()) / scale
                 if scale > 0 else float("inf"))
        if ratio > worst:
            worst, worst_element = ratio, number
        try:
            layered_shell4_local_frame(xyz)
        except ValueError as error:
            if first_invalid < 0:
                first_invalid, first_error = number, str(error)
    return {
        "passed": first_invalid < 0,
        "geometry_model": "best-fit shallow-facet projection",
        "maximum_allowed_warp_ratio": 0.25,
        "maximum_relative_warpage": worst,
        "maximum_warpage_element": worst_element,
        "first_invalid_element": None if first_invalid < 0 else first_invalid,
        "first_error": first_error,
    }


def dimensional_arc_controls(case, normalized_arc_step: float) -> dict[str, float]:
    """Map a dimensionless contract step to the solver's dimensional metric."""
    if normalized_arc_step <= 0:
        raise ValueError("normalized arc step must be positive")
    force = classical_panel_references(case)["gross_section_squash_force"]
    displacement = case.model.thickness
    return {
        "normalized_arc_step": float(normalized_arc_step),
        "characteristic_displacement_m": displacement,
        "characteristic_force_n": force,
        "solver_step_size": normalized_arc_step * displacement,
        "solver_load_scale_m_per_n": displacement / force,
        "nominal_elastic_load_increment_n": normalized_arc_step * force,
    }


def _job_key(divisions: int, normalized_arc_step: float, steps: int) -> str:
    payload = {"schema": SCHEMA, "divisions": divisions,
               "normalized_arc_step": normalized_arc_step, "steps": steps}
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:20]


def execute_panel_job(divisions: int, normalized_arc_step: float, *, steps: int = 100,
                      cache_dir: str | Path | None = None,
                      resume: bool = True,
                      maximum_wall_seconds: float | None = None) -> dict[str, object]:
    """Execute one matrix cell, or persist an exact fail-closed diagnosis."""
    key = _job_key(divisions, normalized_arc_step, steps)
    output = None if cache_dir is None else Path(cache_dir) / f"{key}.json"
    if resume and output is not None and output.exists():
        cached = json.loads(output.read_text(encoding="utf-8"))
        if (cached.get("job_key") == key and cached.get("schema") == SCHEMA
                and cached.get("status") != "running"):
            cached["replayed"] = True
            return cached

    case = build_panel_case(divisions)
    preflight = panel_geometry_preflight(case)
    controls = dimensional_arc_controls(case, normalized_arc_step)
    result: dict[str, object] = {
        "schema": SCHEMA, "job_key": key, "mesh_divisions": divisions,
        "elements": int(len(case.model.elements)), "requested_steps": steps,
        "controls": controls, "preflight": preflight, "replayed": False,
    }
    started = time.monotonic()
    if output is not None:
        _atomic_json(output, {**result, "status": "running", "passed": False,
                              "started_unix": time.time()})
    if not preflight["passed"]:
        result.update({
            "status": "blocked", "passed": False,
            "blocking_stage": "reference_geometry_preflight",
            "blocking_reason": (
                "at least one stress-free imperfect facet exceeds the guarded shallow-warp "
                "projection capability"
            ),
            "accepted_points": 0,
        })
    else:
        previous_alarm = None
        try:
            if maximum_wall_seconds is not None:
                if maximum_wall_seconds <= 0:
                    raise ValueError("maximum_wall_seconds must be positive")
                def _expired(signum, frame):
                    raise TimeoutError(f"wall limit {maximum_wall_seconds:g} s exceeded")
                previous_alarm = signal.signal(signal.SIGALRM, _expired)
                signal.setitimer(signal.ITIMER_REAL, maximum_wall_seconds)
            path = solve_finite_rotation_arc_path(
                case.model, case.reference_load, case.fixed_dofs, steps=steps,
                step_size=controls["solver_step_size"],
                load_scale=controls["solver_load_scale_m_per_n"],
                minimum_step=controls["solver_step_size"] / 128,
                maximum_step=controls["solver_step_size"],
            )
            forces = [float(point.load_factor) for point in path.points]
            shortenings = [-float(point.displacement[6*case.edge_nodes].mean())
                           for point in path.points]
            deflections = [float(point.displacement[case.probe_dof]) for point in path.points]
            alphas = [float(material.alpha) for point in path.points
                      for element in point.state.points for gauss in element
                      for material in gauss]
            result.update({
                "status": "executed" if path.converged else "incomplete",
                "passed": False,  # matrix-level convergence decides qualification
                "accepted_points": len(path.points), "solver_converged": path.converged,
                "peak_force_n": max(forces) if forces else None,
                "terminal_force_n": forces[-1] if forces else None,
                "terminal_step_size": float(path.step_size),
                "terminal_shortening_m": shortenings[-1] if shortenings else None,
                "terminal_centre_deflection_m": deflections[-1] if deflections else None,
                "maximum_yielded_fraction": (
                    sum(alpha > 1e-12 for alpha in alphas) / len(alphas) if alphas else 0.0
                ),
                "maximum_free_dof_equilibrium_norm": (
                    max(float(point.residual_norm) for point in path.points)
                    if path.points else None
                ),
                "post_peak_observed": bool(
                    len(forces) > 2 and max(forces[:-1]) > forces[-1]
                ),
                "external_work_j": None, "recoverable_energy_j": None,
                "plastic_dissipation_j": None, "energy_residual_j": None,
                "energy_boundary": (
                    "accepted arc points do not yet store conjugate internal-force "
                    "vectors; energy claims remain unavailable"
                ),
            })
        except Exception as error:  # evidence records the exact solver boundary
            result.update({"status": "failed", "passed": False,
                           "blocking_stage": "arc_length_solver",
                           "error_type": type(error).__name__, "error": str(error),
                           "accepted_points": 0})
        finally:
            if previous_alarm is not None:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_alarm)
    result["elapsed_seconds"] = time.monotonic() - started
    clean = dict(result)
    result["evidence_sha256"] = hashlib.sha256(_canonical(clean).encode()).hexdigest()
    if output is not None:
        _atomic_json(output, result)
    return result


def execute_panel_matrix(cache_dir: str | Path, *, divisions: Iterable[int] = (4, 8, 12),
                         normalized_arc_steps: Iterable[float] = (.02, .01),
                         steps: int = 100, resume: bool = True,
                         maximum_wall_seconds: float | None = None) -> dict[str, object]:
    """Run all declared mesh/control cells with per-cell resumable evidence."""
    jobs = [execute_panel_job(n, ds, steps=steps, cache_dir=cache_dir, resume=resume,
                              maximum_wall_seconds=maximum_wall_seconds)
            for n in divisions for ds in normalized_arc_steps]
    complete = all(job["status"] == "executed" for job in jobs)
    report = {"schema": SCHEMA, "status": "executed" if complete else "blocked",
              "passed": False, "jobs": jobs,
              "completed_cells": sum(job["status"] == "executed" for job in jobs),
              "required_cells": len(jobs)}
    report["evidence_sha256"] = hashlib.sha256(_canonical(report).encode()).hexdigest()
    _atomic_json(Path(cache_dir) / "matrix.json", report)
    return report
