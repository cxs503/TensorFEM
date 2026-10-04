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
from .panel_path_evidence import (
    energy_balance_gate, evaluate_panel_path, shell_stored_energy,
)


SCHEMA = "tensorfem.marine-panel-execution/1.0"
PERFORMANCE_GATE_SCHEMA = "tensorfem.marine-panel-performance-gate/1.0"
CHUNKED_SCHEMA = "tensorfem.marine-panel-chunked-execution/1.0"


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
    characteristic_energy = force*displacement
    return {
        "normalized_arc_step": float(normalized_arc_step),
        "characteristic_displacement_m": displacement,
        "characteristic_force_n": force,
        "solver_step_size": normalized_arc_step * displacement,
        "solver_load_scale_m_per_n": displacement / force,
        "nominal_elastic_load_increment_n": normalized_arc_step * force,
        "characteristic_energy_j": characteristic_energy,
        "energy_absolute_tolerance_ratio": 1e-6,
        "energy_absolute_tolerance_j": 1e-6*characteristic_energy,
        "energy_relative_tolerance": 1e-4,
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


def _accepted_path_energy_payload(case, path,
                                  point_history: list[dict[str, object]]) -> dict[str, object]:
    """Attach accepted-point energy evidence relative to the relaxed baseline."""
    initial_u = getattr(path, "relaxed_initial_displacement", None)
    initial_state = getattr(path, "relaxed_initial_state", None)
    if initial_u is None or initial_state is None:
        return {
            "energy_evidence_available": False,
            "external_work_j": None,
            "recoverable_energy_j": None,
            "plastic_dissipation_j": None,
            "energy_residual_j": None,
            "maximum_relative_energy_residual": None,
            "failure_mode": "undetermined",
            "energy_boundary": (
                "energy evidence requires the solver's relaxed initial "
                "displacement and committed material state"
            ),
        }
    evidence = evaluate_panel_path(
        case.model, case.reference_load, path.points,
        initial_displacement=initial_u, initial_state=initial_state,
    )
    if len(evidence.points) != len(point_history):
        raise RuntimeError("accepted path/evidence point count mismatch")
    for record, point in zip(point_history, evidence.points):
        gate = energy_balance_gate(
            point.external_work, point.internal_energy,
            characteristic_energy=(
                classical_panel_references(case)["gross_section_squash_force"]
                * case.model.thickness
            ),
        )
        record.update({
            "external_work_j": point.external_work,
            "recoverable_energy_j": point.recoverable_energy,
            "plastic_dissipation_j": point.plastic_dissipation,
            "internal_energy_j": point.internal_energy,
            "energy_residual_j": point.energy_residual,
            "absolute_energy_residual_j": point.absolute_energy_residual,
            "relative_energy_residual": point.relative_energy_residual,
            "mixed_energy_residual": point.mixed_energy_residual,
            "energy_scale_j": point.energy_scale,
            "failure_mode": point.failure_mode,
            "is_peak": point.is_peak,
            "is_post_peak": point.is_post_peak,
            "energy_balance_gate": gate,
        })
    terminal = evidence.points[-1] if evidence.points else None
    return {
        "energy_evidence_available": True,
        "external_work_j": terminal.external_work if terminal else None,
        "recoverable_energy_j": terminal.recoverable_energy if terminal else None,
        "plastic_dissipation_j": terminal.plastic_dissipation if terminal else None,
        "internal_energy_j": terminal.internal_energy if terminal else None,
        "energy_residual_j": terminal.energy_residual if terminal else None,
        "maximum_relative_energy_residual": (
            max(point.relative_energy_residual for point in evidence.points)
            if evidence.points else None
        ),
        "failure_mode": evidence.failure_mode,
        "energy_post_peak_confirmed": evidence.post_peak_confirmed,
        "energy_balance_passed": all(
            bool(record["energy_balance_gate"]["passed"])
            for record in point_history
        ),
        "energy_boundary": (
            "trapezoidal external work, recoverable energy and associative J2 "
            "plastic dissipation use accepted points only and are measured from "
            "the solver's relaxed initial committed state"
        ),
    }


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
    controls["augmented_scaling"] = (
        "normalized" if solver == "dense" else "matrix_free_not_applicable"
    )
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
            if solver == "dense":
                solver_controls["augmented_scaling"] = "normalized"
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
            point_history = []
            for point, force, shortening, deflection, relative in zip(
                    path.points, forces, shortenings, deflections,
                    relative_equilibria):
                material_points = [material for element in point.state.points
                                   for gauss in element for material in gauss]
                yielded = (sum(float(material.alpha) > 1e-12
                               for material in material_points)
                           / len(material_points) if material_points else 0.0)
                point_history.append({
                    "step": int(point.step),
                    "force_n": force,
                    "edge_shortening_m": shortening,
                    "centre_deflection_m": deflection,
                    "yielded_fraction": yielded,
                    "newton_iterations": int(point.iterations),
                    "equilibrium_relative_norm": relative,
                    "state_sha256": _accepted_state_sha256(point),
                })
            attempted_steps = sum(1 for item in diagnostics
                                  if "attempt" in item)
            accepted_attempts = sum(item.get("reason") == "accepted"
                                    for item in diagnostics
                                    if "attempt" in item)
            rejected_attempts = attempted_steps - accepted_attempts
            peak_index = forces.index(max(forces)) if forces else None
            # A single smaller point can be numerical noise.  Require two
            # accepted points below the prior maximum by a visible margin.
            peak_confirmed = bool(
                peak_index is not None and len(forces) - peak_index >= 3
                and all(value < forces[peak_index] * (1.0 - 1e-4)
                        for value in forces[peak_index + 1:peak_index + 3])
            )
            energy_payload = _accepted_path_energy_payload(case, path, point_history)
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
                "attempted_steps": attempted_steps,
                "rejected_steps": rejected_attempts,
                "point_history": point_history,
                "peak_force_n": max(forces) if forces else None,
                "peak_point_index": peak_index,
                "peak_confirmed": peak_confirmed,
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
                "post_peak_observed": peak_confirmed,
                **energy_payload,
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


def execute_panel_chunked_job(
    divisions: int,
    normalized_arc_step: float,
    *,
    steps: int,
    cache_dir: str | Path,
    chunk_size: int = 20,
    resume: bool = True,
    maximum_wall_seconds: float | None = None,
    relative_equilibrium_tolerance: float = 1e-6,
) -> dict[str, object]:
    """Run a long dense path with exact, branch-preserving checkpoints.

    The JSON manifest is portable evidence.  Its companion ``.pt`` file is a
    trusted local restart artifact containing the complete material state and
    previous arc increment.  ``torch.load(weights_only=False)`` is used only
    after its binary hash matches the JSON manifest; untrusted or externally
    supplied checkpoint files must never be loaded.  The hash is recorded after
    every atomic replace, so a stale or partial restart is rejected.
    """
    if steps < 1 or chunk_size < 1:
        raise ValueError("steps and chunk_size must be positive")
    case = build_panel_case(divisions)
    controls = dimensional_arc_controls(case, normalized_arc_step)
    controls["augmented_scaling"] = "normalized"
    identity = {"schema": CHUNKED_SCHEMA, "divisions": divisions,
                "normalized_arc_step": normalized_arc_step,
                "relative_equilibrium_tolerance": relative_equilibrium_tolerance}
    key_payload = {**identity, "steps": steps, "chunk_size": chunk_size}
    # Target length and persistence cadence are deliberately absent: a
    # validated prefix can be extended and checkpointed more frequently near
    # a nonlinear limit point without changing the physical continuation.
    key = hashlib.sha256(_canonical(identity).encode()).hexdigest()[:20]
    root = Path(cache_dir); root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / f"chunked-{key}.json"
    checkpoint_path = root / f"chunked-{key}.pt"
    displacement = None; state = None; previous = None; load_factor = 0.0
    current_step_size = controls["solver_step_size"]
    history: list[dict[str, object]] = []
    chunks: list[dict[str, object]] = []
    reference_recoverable_energy = None
    cumulative_external_work = 0.0
    cumulative_plastic_dissipation = 0.0
    energy_prefix_complete = True
    load_manifest_path, load_checkpoint_path = manifest_path, checkpoint_path
    if resume and not (manifest_path.exists() and checkpoint_path.exists()):
        # One-time migration of v1.0 artifacts whose legacy key included the
        # requested target length.  Select only an exact-control, hashed prefix.
        candidates = []
        for candidate_path in root.glob("chunked-*.json"):
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            if (all(candidate.get(name) == value for name, value in identity.items())
                    and int(candidate.get("accepted_points", 0)) <= steps):
                binary = root / str(candidate.get("checkpoint_file", ""))
                if binary.is_file():
                    candidates.append((int(candidate.get("accepted_points", 0)),
                                       candidate_path, binary))
        if candidates:
            _, load_manifest_path, load_checkpoint_path = max(candidates)
    if resume and load_manifest_path.exists() and load_checkpoint_path.exists():
        manifest = json.loads(load_manifest_path.read_text(encoding="utf-8"))
        binary_hash = hashlib.sha256(load_checkpoint_path.read_bytes()).hexdigest()
        if (manifest.get("schema") != CHUNKED_SCHEMA
                or manifest.get("checkpoint_sha256") != binary_hash):
            raise ValueError("chunked panel checkpoint integrity mismatch")
        if not all(manifest.get(name) == value for name, value in identity.items()):
            raise ValueError("chunked panel checkpoint controls mismatch")
        saved = torch.load(load_checkpoint_path, map_location=case.model.nodes.device,
                           weights_only=False)
        displacement, state = saved["displacement"], saved["state"]
        previous, load_factor = saved["previous_increment"], saved["load_factor"]
        current_step_size = float(saved.get("step_size", current_step_size))
        history = manifest.get("point_history", [])
        chunks = manifest.get("chunks", [])
        required_energy = {"reference_recoverable_energy", "cumulative_external_work",
                           "cumulative_plastic_dissipation"}
        if required_energy <= set(saved):
            reference_recoverable_energy = float(saved["reference_recoverable_energy"])
            cumulative_external_work = float(saved["cumulative_external_work"])
            cumulative_plastic_dissipation = float(saved["cumulative_plastic_dissipation"])
            energy_prefix_complete = bool(saved.get("energy_prefix_complete", True))
        else:
            # Legacy mechanical checkpoints remain valid for branch continuation,
            # but cannot support a whole-path energy claim.  Start a visibly
            # partial ledger at this exact checkpoint rather than inventing zero
            # work for the missing prefix.
            energy_prefix_complete = False
            reference_recoverable_energy = shell_stored_energy(
                case.model, displacement, state,
            ).recoverable
        if len(history) >= steps and manifest.get("status") == "executed":
            manifest["replayed"] = True
            return manifest
    started = time.monotonic()
    initial_offset = len(history)
    status = "running"; error = None
    while len(history) < steps:
        remaining_wall = (None if maximum_wall_seconds is None else
                          maximum_wall_seconds - (time.monotonic() - started))
        if remaining_wall is not None and remaining_wall <= 0:
            status, error = "incomplete", "wall limit exhausted between chunks"
            break
        count = min(chunk_size, steps - len(history))
        chunk_initial_displacement = displacement
        chunk_initial_state = state
        chunk_initial_load_factor = load_factor
        diagnostics: list[dict[str, object]] = []
        previous_alarm = None
        try:
            if remaining_wall is not None:
                def _expired(signum, frame):
                    raise TimeoutError(f"wall limit {maximum_wall_seconds:g} s exceeded")
                previous_alarm = signal.signal(signal.SIGALRM, _expired)
                signal.setitimer(signal.ITIMER_REAL, remaining_wall)
            path = solve_finite_rotation_arc_path(
                case.model, case.reference_load, case.fixed_dofs,
                steps=count, step_size=current_step_size,
                load_scale=controls["solver_load_scale_m_per_n"],
                initial_state=state, initial_displacement=displacement,
                initial_load_factor=load_factor,
                initial_previous_increment=previous,
                tolerance=relative_equilibrium_tolerance,
                minimum_step=controls["solver_step_size"] / 128,
                maximum_step=controls["solver_step_size"], diagnostics=diagnostics,
                augmented_scaling="normalized",
            )
        except TimeoutError as exc:
            status, error = "incomplete", str(exc)
            break
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            break
        finally:
            if previous_alarm is not None:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_alarm)
        if not path.points:
            status, error = "incomplete", "chunk accepted no continuation point"
            break
        offset = len(history)
        if offset == 0:
            chunk_initial_displacement = path.relaxed_initial_displacement
            chunk_initial_state = path.relaxed_initial_state
            chunk_initial_load_factor = 0.0
            reference_recoverable_energy = shell_stored_energy(
                case.model, chunk_initial_displacement, chunk_initial_state,
            ).recoverable
        energy = evaluate_panel_path(
            case.model, case.reference_load, path.points,
            initial_displacement=chunk_initial_displacement,
            initial_state=chunk_initial_state,
            initial_load_factor=chunk_initial_load_factor,
            initial_external_work=cumulative_external_work,
            initial_plastic_dissipation=cumulative_plastic_dissipation,
            reference_recoverable_energy=reference_recoverable_energy,
        )
        reference_norm = float(torch.linalg.vector_norm(case.reference_load))
        for local, (point, point_energy) in enumerate(
                zip(path.points, energy.points), 1):
            energy_gate = energy_balance_gate(
                point_energy.external_work, point_energy.internal_energy,
                characteristic_energy=controls["characteristic_energy_j"],
                absolute_ratio=controls["energy_absolute_tolerance_ratio"],
                relative_tolerance=controls["energy_relative_tolerance"],
            )
            materials = [material for element in point.state.points
                         for gauss in element for material in gauss]
            relative = float(point.residual_norm) / max(
                abs(float(point.load_factor)) * reference_norm, 1.0)
            history.append({
                "step": offset + local,
                "force_n": float(point.load_factor),
                "edge_shortening_m": -float(point.displacement[6*case.edge_nodes].mean()),
                "centre_deflection_m": float(point.displacement[case.probe_dof]),
                "yielded_fraction": (sum(float(x.alpha) > 1e-12 for x in materials)
                                     / len(materials) if materials else 0.0),
                "newton_iterations": int(point.iterations),
                "equilibrium_relative_norm": relative,
                "state_sha256": _accepted_state_sha256(point),
                "external_work_j": point_energy.external_work,
                "recoverable_energy_j": point_energy.recoverable_energy,
                "plastic_dissipation_j": point_energy.plastic_dissipation,
                "internal_energy_j": point_energy.internal_energy,
                "energy_residual_j": point_energy.energy_residual,
                "relative_energy_residual": point_energy.relative_energy_residual,
                "failure_mode": point_energy.failure_mode,
                "energy_balance_gate": energy_gate,
            })
        cumulative_external_work = energy.points[-1].external_work
        cumulative_plastic_dissipation = energy.points[-1].plastic_dissipation
        displacement, state = path.displacement, path.committed_state
        previous, load_factor = path.previous_increment, path.load_factor
        current_step_size = float(path.step_size)
        chunks.append({
            "first_step": offset + 1, "last_step": len(history),
            "accepted": len(path.points),
            "attempted": sum("attempt" in item for item in diagnostics),
            "rejected": sum(item.get("reason") != "accepted" for item in diagnostics
                            if "attempt" in item),
            "solver_converged": path.converged,
            "terminal_external_work_j": cumulative_external_work,
            "terminal_plastic_dissipation_j": cumulative_plastic_dissipation,
            "terminal_energy_residual_j": energy.points[-1].energy_residual,
        })
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save({"displacement": displacement, "state": state,
                    "previous_increment": previous, "load_factor": load_factor,
                    "step_size": path.step_size,
                    "reference_recoverable_energy": reference_recoverable_energy,
                    "cumulative_external_work": cumulative_external_work,
                    "cumulative_plastic_dissipation": cumulative_plastic_dissipation,
                    "energy_prefix_complete": energy_prefix_complete},
                   temporary)
        os.replace(temporary, checkpoint_path)
        checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        status = "executed" if len(history) >= steps else "running"
        manifest = {
            **key_payload, "job_key": key, "status": status, "passed": False,
            "controls": controls,
            "replayed": False, "accepted_points": len(history),
            "point_history": history, "chunks": chunks,
            "reference_recoverable_energy_j": reference_recoverable_energy,
            "external_work_j": cumulative_external_work,
            "plastic_dissipation_j": cumulative_plastic_dissipation,
            "energy_prefix_complete": energy_prefix_complete,
            "checkpoint_file": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_hash,
            "elapsed_seconds_this_run": time.monotonic() - started,
        }
        _atomic_json(manifest_path, manifest)
        if not path.converged:
            status, error = "incomplete", "continuation chunk did not fully converge"
            break
    forces = [float(point["force_n"]) for point in history]
    peak_index = forces.index(max(forces)) if forces else None
    peak_confirmed = bool(peak_index is not None and len(forces)-peak_index >= 3
                          and all(value < forces[peak_index]*(1-1e-4)
                                  for value in forces[peak_index+1:peak_index+3]))
    manifest = {
        **key_payload, "job_key": key, "status": status, "passed": False,
        "controls": controls,
        "replayed": False, "accepted_points": len(history),
        "new_points_this_run": len(history)-initial_offset,
        "point_history": history, "chunks": chunks,
        "peak_force_n": max(forces) if forces else None,
        "peak_point_index": peak_index, "peak_confirmed": peak_confirmed,
        "post_peak_observed": peak_confirmed,
        "maximum_yielded_fraction": max((float(x["yielded_fraction"])
                                         for x in history), default=0.0),
        "reference_recoverable_energy_j": reference_recoverable_energy,
        "energy_prefix_complete": energy_prefix_complete,
        "energy_boundary": (None if energy_prefix_complete else
                            "energy ledger begins at a legacy mechanical checkpoint"),
        "external_work_j": (cumulative_external_work if history and
                             "external_work_j" in history[-1] else None),
        "recoverable_energy_j": history[-1].get("recoverable_energy_j") if history else None,
        "plastic_dissipation_j": (cumulative_plastic_dissipation if history and
                                   "plastic_dissipation_j" in history[-1] else None),
        "internal_energy_j": history[-1].get("internal_energy_j") if history else None,
        "energy_residual_j": history[-1].get("energy_residual_j") if history else None,
        "maximum_relative_energy_residual": max(
            (float(x["relative_energy_residual"]) for x in history
             if x.get("relative_energy_residual") is not None), default=None),
        "energy_balance_passed": bool(
            history and energy_prefix_complete
            and all(x.get("energy_balance_gate", {}).get("passed", False)
                    for x in history)
        ),
        "elapsed_seconds_this_run": time.monotonic()-started,
        "error": error,
    }
    if checkpoint_path.exists():
        manifest["checkpoint_file"] = checkpoint_path.name
        manifest["checkpoint_sha256"] = hashlib.sha256(
            checkpoint_path.read_bytes()).hexdigest()
    clean = dict(manifest)
    manifest["evidence_sha256"] = hashlib.sha256(_canonical(clean).encode()).hexdigest()
    _atomic_json(manifest_path, manifest)
    return manifest


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
