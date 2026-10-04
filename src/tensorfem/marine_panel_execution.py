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
from .finite_rotation_layered_shell4 import assemble_finite_rotation_layered_shell4
from .layered_shell4_plasticity import LayeredShell4State, layered_shell4_local_frame
from .marine_panel_ultimate_fe import build_panel_case, classical_panel_references


SCHEMA = "tensorfem.marine-panel-execution/1.0"
PERFORMANCE_GATE_SCHEMA = "tensorfem.marine-panel-performance-gate/1.0"


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


def panel_initial_equilibrium_preflight(case, *, relative_tolerance: float = 1e-6):
    """Verify that imported initial stress is equilibrated on unconstrained DOFs."""
    if relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be positive")
    zero = torch.zeros(case.model.n_dofs, dtype=case.model.nodes.dtype,
                       device=case.model.nodes.device)
    virgin = LayeredShell4State.virgin(case.model)
    response = assemble_finite_rotation_layered_shell4(
        case.model, zero, virgin, tangent=False,
    )
    mask = torch.ones(case.model.n_dofs, dtype=torch.bool, device=case.model.nodes.device)
    mask[case.fixed_dofs.to(mask.device)] = False
    free_norm = float(torch.linalg.vector_norm(response.internal_force[mask]))
    force_scale = classical_panel_references(case)["gross_section_squash_force"]
    relative = free_norm / max(force_scale, 1.0)
    return {
        "passed": relative <= relative_tolerance,
        "free_dof_internal_force_norm_n": free_norm,
        "force_scale_n": force_scale,
        "relative_free_dof_internal_force_norm": relative,
        "relative_tolerance": relative_tolerance,
        "boundary": (
            "An arc-length path must start from an equilibrated committed state; "
            "a globally self-balanced stress formula is insufficient if its "
            "discrete nodal projection is not balanced."
        ),
    }


def _job_key(divisions: int, normalized_arc_step: float, steps: int,
             solver: str = "dense", relative_equilibrium_tolerance: float = 1e-6) -> str:
    payload = {"schema": SCHEMA, "divisions": divisions,
               "normalized_arc_step": normalized_arc_step, "steps": steps,
               "solver": solver,
               "relative_equilibrium_tolerance": relative_equilibrium_tolerance}
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:20]


def _solution_state_sha256(displacement, state, load_factor: float) -> str:
    """Hash a complete solution/material state in a portable form."""
    material = []
    for element in state.points:
        for gauss in element:
            for state in gauss:
                material.append({
                    "plastic_strain": state.plastic_strain.detach().cpu().tolist(),
                    "alpha": float(state.alpha),
                })
    payload = {
        "load_factor": float(load_factor),
        "displacement": displacement.detach().cpu().tolist(),
        "material": material,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


def _accepted_state_sha256(point) -> str:
    return _solution_state_sha256(point.displacement, point.state, point.load_factor)


def execute_panel_job(divisions: int, normalized_arc_step: float, *, steps: int = 100,
                      cache_dir: str | Path | None = None,
                      resume: bool = True,
                      maximum_wall_seconds: float | None = None,
                      solver: str = "dense",
                      relative_equilibrium_tolerance: float = 1e-6) -> dict[str, object]:
    """Execute one matrix cell, or persist an exact fail-closed diagnosis."""
    if solver not in {"dense", "matrix_free"}:
        raise ValueError("solver must be 'dense' or 'matrix_free'")
    if relative_equilibrium_tolerance <= 0:
        raise ValueError("relative_equilibrium_tolerance must be positive")
    key = _job_key(divisions, normalized_arc_step, steps, solver,
                   relative_equilibrium_tolerance)
    output = None if cache_dir is None else Path(cache_dir) / f"{key}.json"
    if resume and output is not None and output.exists():
        cached = json.loads(output.read_text(encoding="utf-8"))
        if (cached.get("job_key") == key and cached.get("schema") == SCHEMA
                and cached.get("status") != "running"):
            cached["replayed"] = True
            return cached

    case = build_panel_case(divisions)
    preflight = panel_geometry_preflight(case)
    initial_equilibrium = (panel_initial_equilibrium_preflight(
        case, relative_tolerance=relative_equilibrium_tolerance,
    ) if preflight["passed"] else None)
    controls = dimensional_arc_controls(case, normalized_arc_step)
    controls["relative_equilibrium_tolerance"] = relative_equilibrium_tolerance
    controls["absolute_equilibrium_tolerance_n"] = (
        relative_equilibrium_tolerance
        * controls["nominal_elastic_load_increment_n"]
        * float(torch.linalg.vector_norm(case.reference_load))
    )
    result: dict[str, object] = {
        "schema": SCHEMA, "job_key": key, "mesh_divisions": divisions,
        "elements": int(len(case.model.elements)), "requested_steps": steps,
        "controls": controls, "preflight": preflight,
        "initial_equilibrium_preflight": initial_equilibrium, "replayed": False,
        "solver": solver,
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
    elif not initial_equilibrium["passed"] and solver != "dense":
        result.update({
            "status": "blocked", "passed": False,
            "blocking_stage": "initial_equilibrium_preflight",
            "blocking_reason": (
                "the residual-stress field is not in discrete equilibrium on "
                "the free DOFs; solve an initial-equilibrium step before continuation"
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
            solve = solve_finite_rotation_arc_path
            solver_controls = {
                # Dense Shell4 uses a relative force and normalized-constraint
                # tolerance.  The dimensional value above is evidence only.
                "tolerance": relative_equilibrium_tolerance,
            }
            diagnostics = []
            solver_controls["diagnostics"] = diagnostics
            if solver == "matrix_free":
                from .matrix_free_arc import solve_matrix_free_finite_rotation_shell4
                solve = solve_matrix_free_finite_rotation_shell4
                solver_controls.update({
                    "relative_equilibrium_tolerance": relative_equilibrium_tolerance,
                    "normalized_constraint_tolerance": relative_equilibrium_tolerance,
                })
                solver_controls.pop("diagnostics")
            path = solve(
                case.model, case.reference_load, case.fixed_dofs, steps=steps,
                step_size=controls["solver_step_size"],
                load_scale=controls["solver_load_scale_m_per_n"],
                minimum_step=controls["solver_step_size"] / 128,
                maximum_step=controls["solver_step_size"],
                **solver_controls,
            )
            forces = [float(point.load_factor) for point in path.points]
            shortenings = [-float(point.displacement[6*case.edge_nodes].mean())
                           for point in path.points]
            deflections = [float(point.displacement[case.probe_dof]) for point in path.points]
            alphas = [float(material.alpha) for point in path.points
                      for element in point.state.points for gauss in element
                      for material in gauss]
            reference_norm = float(torch.linalg.vector_norm(case.reference_load))
            relative_equilibria = [
                float(point.equilibrium_relative_norm)
                if hasattr(point, "equilibrium_relative_norm")
                else float(point.residual_norm)
                     / max(abs(float(point.load_factor)) * reference_norm, 1.0)
                for point in path.points
            ]
            initial_relaxation = None
            if solver == "dense" and diagnostics:
                initial_trace = diagnostics[0]
                iterations = initial_trace.get("iterations", [])
                initial_relaxation = {
                    "passed": bool(
                        initial_trace.get("reason") == "accepted" and iterations
                        and iterations[-1]["residual_relative"]
                            <= relative_equilibrium_tolerance
                    ),
                    "reason": initial_trace.get("reason"),
                    "iterations": len(iterations),
                    "initial_residual_norm_n": (
                        iterations[0]["residual_norm"] if iterations else None
                    ),
                    "terminal_residual_norm_n": (
                        iterations[-1]["residual_norm"] if iterations else None
                    ),
                    "terminal_relative_residual": (
                        iterations[-1]["residual_relative"] if iterations else None
                    ),
                    "relaxed_displacement_norm_m": (
                        float(torch.linalg.vector_norm(path.relaxed_initial_displacement))
                        if path.relaxed_initial_displacement is not None else None
                    ),
                    "relaxed_state_sha256": (
                        _solution_state_sha256(
                            path.relaxed_initial_displacement,
                            path.relaxed_initial_state,
                            0.0,
                        )
                        if (path.relaxed_initial_displacement is not None
                            and path.relaxed_initial_state is not None) else None
                    ),
                }
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
                "maximum_relative_free_dof_equilibrium_norm": (
                    max(relative_equilibria) if relative_equilibria else None
                ),
                "accepted_state_sha256": (
                    _accepted_state_sha256(path.points[-1]) if path.points else None
                ),
                "response_evaluations": getattr(path, "response_evaluations", None),
                "tangent_actions": getattr(path, "tangent_actions", None),
                "krylov_iterations": getattr(path, "krylov_iterations", None),
                "solver_diagnostics": diagnostics if solver == "dense" else None,
                "initial_equilibrium_relaxation": initial_relaxation,
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


def execute_panel_performance_gate(
    cache_dir: str | Path,
    *,
    normalized_arc_step: float = .02,
    first_point_limits: dict[int, float] | None = None,
    equilibrium_limit: float = 1e-6,
    resume: bool = False,
    solver: str = "matrix_free",
) -> dict[str, object]:
    """Run the 2x2 then 4x4 first-point gates without overstating qualification.

    The 2x2 cell is a smoke/performance diagnostic and is not part of the final
    4/8/12 qualification matrix.  The 4x4 cell runs only after 2x2 passes.  A
    gate requires an accepted point, relative free-DOF balance, a complete
    state hash, and completion inside its explicit wall limit.
    """
    limits = {2: 30.0, 4: 120.0} if first_point_limits is None else dict(first_point_limits)
    if set(limits) != {2, 4} or any(value <= 0 for value in limits.values()):
        raise ValueError("first_point_limits must contain positive 2x2 and 4x4 limits")
    stages = []
    for divisions in (2, 4):
        job = execute_panel_job(
            divisions, normalized_arc_step, steps=1, cache_dir=cache_dir,
            resume=resume, maximum_wall_seconds=limits[divisions], solver=solver,
            relative_equilibrium_tolerance=equilibrium_limit,
        )
        balance = job.get("maximum_relative_free_dof_equilibrium_norm")
        raw_initial = job.get("initial_equilibrium_preflight") or {}
        relaxed_initial = job.get("initial_equilibrium_relaxation") or {}
        initial_ok = bool(
            raw_initial.get("passed")
            or (relaxed_initial.get("passed")
                and relaxed_initial.get("relaxed_state_sha256")
                and relaxed_initial.get("relaxed_displacement_norm_m") is not None)
        )
        passed = bool(
            job.get("status") == "executed"
            and job.get("accepted_points") == 1
            and initial_ok
            and balance is not None and balance <= equilibrium_limit
            and job.get("accepted_state_sha256")
            and job.get("elapsed_seconds", float("inf")) <= limits[divisions]
        )
        stages.append({
            "mesh_divisions": divisions,
            "wall_limit_seconds": limits[divisions],
            "passed": passed,
            "job": job,
        })
        if not passed:
            break
    report = {
        "schema": PERFORMANCE_GATE_SCHEMA,
        "status": "passed" if len(stages) == 2 and all(x["passed"] for x in stages)
                  else "blocked",
        "passed": len(stages) == 2 and all(x["passed"] for x in stages),
        "normalized_arc_step": normalized_arc_step,
        "solver": solver,
        "relative_equilibrium_limit": equilibrium_limit,
        "stages": stages,
        "boundary": (
            "This gate establishes first-point scalability only; it does not "
            "establish peak load, post-peak response, or mesh convergence."
        ),
    }
    clean = dict(report)
    report["evidence_sha256"] = hashlib.sha256(_canonical(clean).encode()).hexdigest()
    _atomic_json(Path(cache_dir) / "performance-gate.json", report)
    return report


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
