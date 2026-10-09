"""Executable and hashed nonlinear layered-Shell4 robustness evidence."""
from __future__ import annotations

import hashlib
import json
from typing import Mapping

import torch

from .finite_rotation_layered_shell4 import solve_finite_rotation_arc_path
from .layered_shell4_plasticity import LayeredShell4Model, LayeredShell4State
from .panel_path_evidence import shell_stored_energy


SCHEMA = "tensorfem.nonlinear-shell-robustness/1.0"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def run_nonlinear_shell_robustness() -> dict[str, object]:
    dtype = torch.float64
    nodes = torch.tensor(((0.,0.,0.), (1.,0.,0.), (1.,1.,0.), (0.,1.,0.)),
                         dtype=dtype)
    model = LayeredShell4Model(nodes, torch.tensor(((0,1,2,3),)),
                               200000., .3, .08, 250., 1400., layers=3)
    fixed = torch.tensor(tuple(range(6))+tuple(range(18,24)))
    load = torch.zeros(24, dtype=dtype); load[8] = load[14] = -100.
    virgin = LayeredShell4State.virgin(model)
    before = tuple(point.plastic_strain.clone() for element in virgin.points
                   for quadrature in element for point in quadrature)
    controls = dict(steps=1, step_size=.8, maximum_step=.8, minimum_step=.792,
                    load_scale=.001, initial_state=virgin, tolerance=2e-7,
                    max_iterations=15)
    fixed_result = solve_finite_rotation_arc_path(
        model, load, fixed, line_search=None, **controls)
    trace: list[dict[str, object]] = []
    backtracking = solve_finite_rotation_arc_path(
        model, load, fixed, line_search="backtracking", diagnostics=trace,
        **controls)
    reference = solve_finite_rotation_arc_path(
        model, load, fixed, line_search="backtracking",
        **{**controls, "tolerance": 2e-9})
    displacement_error = (float(torch.linalg.vector_norm(
        backtracking.displacement-reference.displacement))
        / max(float(torch.linalg.vector_norm(reference.displacement)), 1e-300))
    load_error = abs(backtracking.load_factor/reference.load_factor-1.)
    energy = shell_stored_energy(
        model, backtracking.displacement, backtracking.committed_state).recoverable
    reference_energy = shell_stored_energy(
        model, reference.displacement, reference.committed_state).recoverable
    energy_error = abs(energy/reference_energy-1.)
    committed_plastic = torch.cat([
        point.plastic_strain.reshape(-1)
        for element in backtracking.committed_state.points
        for quadrature in element for point in quadrature])
    reference_plastic = torch.cat([
        point.plastic_strain.reshape(-1)
        for element in reference.committed_state.points
        for quadrature in element for point in quadrature])
    plastic_history_error = float(torch.linalg.vector_norm(
        committed_plastic-reference_plastic))/max(
            float(torch.linalg.vector_norm(reference_plastic)), 1e-300)
    maximum_plastic_strain = float(torch.max(torch.abs(committed_plastic)))
    after = tuple(point.plastic_strain for element in virgin.points
                  for quadrature in element for point in quadrature)
    rejected = tuple(point.plastic_strain
                     for element in fixed_result.committed_state.points
                     for quadrature in element for point in quadrature)
    rollback = all(torch.equal(a, b) for a, b in zip(before, after)) and all(
        torch.equal(a, b) for a, b in zip(before, rejected))
    reductions = [float(item.get("line_search_alpha", 1.))
                  for item in trace[-1].get("iterations", [])] if trace else []
    maximum_error = max(load_error, displacement_error, energy_error,
                        plastic_history_error)
    clean = {
        "schema": SCHEMA,
        "case": "layered-j2-shell4-difficult-arc-step",
        "fixed_converged": fixed_result.converged,
        "backtracking_converged": backtracking.converged,
        "strict_reference_converged": reference.converged,
        "minimum_line_search_alpha": min(reductions, default=1.),
        "load_relative_error": load_error,
        "displacement_relative_error": displacement_error,
        "recoverable_energy_relative_error": energy_error,
        "plastic_history_relative_error": plastic_history_error,
        "maximum_plastic_strain": maximum_plastic_strain,
        "maximum_relative_error": maximum_error,
        "caller_and_rejected_state_rollback_exact": rollback,
        "difficult_step_improved": (not fixed_result.converged
                                    and backtracking.converged),
    }
    clean["passed"] = bool(
        clean["difficult_step_improved"] and reference.converged
        and minimum_error_gate(maximum_error) and rollback
        and maximum_plastic_strain > 0.
        and clean["minimum_line_search_alpha"] < 1.)
    return {**clean, "evidence_sha256": hashlib.sha256(
        _canonical(clean).encode()).hexdigest()}


def minimum_error_gate(error: float) -> bool:
    return error < .01


def validate_nonlinear_shell_robustness(report: Mapping[str, object]
                                        ) -> Mapping[str, object]:
    if report.get("schema") != SCHEMA:
        raise ValueError("invalid nonlinear Shell4 robustness schema")
    clean = {key: value for key, value in report.items() if key != "evidence_sha256"}
    if report.get("evidence_sha256") != hashlib.sha256(_canonical(clean).encode()).hexdigest():
        raise ValueError("nonlinear Shell4 robustness hash mismatch")
    derived = (not bool(report.get("fixed_converged"))
               and bool(report.get("backtracking_converged"))
               and bool(report.get("strict_reference_converged"))
               and minimum_error_gate(float(report.get("maximum_relative_error", float("inf"))))
               and bool(report.get("caller_and_rejected_state_rollback_exact"))
               and float(report.get("maximum_plastic_strain", 0.)) > 0.
               and float(report.get("minimum_line_search_alpha", 1.)) < 1.)
    if bool(report.get("passed")) != derived:
        raise ValueError("nonlinear Shell4 robustness status is inconsistent")
    return report
