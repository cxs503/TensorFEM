"""Reproducible dense/SuperLU qualification for a yielding Shell4 path."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import resource
from time import perf_counter

import torch

from .finite_rotation_layered_shell4 import solve_finite_rotation_arc_path
from .layered_shell4_plasticity import LayeredShell4Model, LayeredShell4State
from .panel_path_evidence import shell_stored_energy, yielded_fraction


@dataclass(frozen=True)
class PlasticPathQualification:
    backend: str
    converged: bool
    accepted_points: int
    load_factor: float
    displacement_norm: float
    recoverable_energy_j: float
    elastic_energy_j: float
    hardening_energy_j: float
    yielded_fraction: float
    maximum_alpha: float
    alpha_sum: float
    plastic_strain_norm: float
    alpha_monotone: bool
    state_sha256: str
    elapsed_seconds: float
    process_peak_rss_kib: int
    factorizations: int
    maximum_sparse_matrix_bytes: int | None
    maximum_sparse_factor_bytes: int | None
    maximum_linear_relative_residual: float | None


def yielding_shell4_case() -> tuple[LayeredShell4Model, torch.Tensor, torch.Tensor]:
    """One real layered finite-rotation facet driven beyond first yield."""
    dtype = torch.float64
    nodes = torch.tensor(((0., 0., 0.), (1., 0., 0.),
                          (1., 1., 0.), (0., 1., 0.)), dtype=dtype)
    model = LayeredShell4Model(
        nodes, torch.tensor(((0, 1, 2, 3),)), 200000., .3, .08,
        250., 1400., layers=3,
    )
    fixed = torch.tensor((0,1,2,3,4,5, 8,9,10,11, 14,15,16,17,
                          18,19,20,21,22,23))
    load = torch.zeros(model.n_dofs, dtype=dtype)
    load[6] = load[12] = 2.
    return model, load, fixed


def _material_values(state: LayeredShell4State) -> tuple[torch.Tensor, torch.Tensor]:
    points = tuple(point for element in state.points for quadrature in element
                   for point in quadrature)
    alpha = torch.stack(tuple(point.alpha for point in points))
    plastic = torch.stack(tuple(point.plastic_strain for point in points))
    return alpha, plastic


def _state_hash(displacement: torch.Tensor, state: LayeredShell4State,
                load_factor: float) -> str:
    alpha, plastic = _material_values(state)
    digest = hashlib.sha256()
    for value in (displacement, alpha, plastic):
        digest.update(json.dumps(value.detach().cpu().tolist(),
                                 separators=(",", ":")).encode())
    digest.update(float(load_factor).hex().encode())
    return digest.hexdigest()


def run_plastic_path(backend: str, *, steps: int = 80) -> PlasticPathQualification:
    """Run one backend; use separate processes for comparable peak RSS."""
    if backend not in ("dense", "superlu"):
        raise ValueError("backend must be 'dense' or 'superlu'")
    model, load, fixed = yielding_shell4_case()
    diagnostics: list[dict[str, object]] = []
    started = perf_counter()
    result = solve_finite_rotation_arc_path(
        model, load, fixed, steps=steps, step_size=.01, maximum_step=.01,
        load_scale=.1, tolerance=2e-7, augmented_scaling="normalized",
        linear_solver=backend, diagnostics=diagnostics,
    )
    elapsed = perf_counter() - started
    energy = shell_stored_energy(model, result.displacement, result.committed_state)
    alpha, plastic = _material_values(result.committed_state)
    histories = []
    for point in result.points:
        values, _ = _material_values(point.state)
        histories.append(float(values.sum()))
    linear = tuple(row for row in diagnostics if row.get("backend") == "splu")
    return PlasticPathQualification(
        backend, result.converged, len(result.points), result.load_factor,
        float(torch.linalg.vector_norm(result.displacement)), energy.recoverable,
        energy.elastic, energy.hardening,
        yielded_fraction(model, result.committed_state), float(alpha.max()),
        float(alpha.sum()), float(torch.linalg.vector_norm(plastic)),
        all(b >= a-1e-14 for a, b in zip(histories, histories[1:])),
        _state_hash(result.displacement, result.committed_state, result.load_factor),
        elapsed, int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), len(linear),
        max((int(row["estimated_matrix_bytes"]) for row in linear), default=None),
        max((int(row["estimated_factor_bytes"]) for row in linear), default=None),
        max((float(row["last_relative_residual"]) for row in linear), default=None),
    )


def compare_plastic_paths(dense: PlasticPathQualification,
                          sparse: PlasticPathQualification,
                          *, tolerance: float = .01) -> dict[str, object]:
    """Fail-closed qualification gates for physical rather than byte identity."""
    if dense.backend != "dense" or sparse.backend != "superlu":
        raise ValueError("expected dense and superlu reports")
    def relative(a: float, b: float) -> float:
        return abs(a-b)/max(abs(a), 1e-30)
    differences = {
        "load_factor": relative(dense.load_factor, sparse.load_factor),
        "displacement_norm": relative(dense.displacement_norm, sparse.displacement_norm),
        "recoverable_energy": relative(dense.recoverable_energy_j,
                                       sparse.recoverable_energy_j),
        "hardening_energy": relative(dense.hardening_energy_j,
                                     sparse.hardening_energy_j),
        "alpha_sum": relative(dense.alpha_sum, sparse.alpha_sum),
        "plastic_strain_norm": relative(dense.plastic_strain_norm,
                                        sparse.plastic_strain_norm),
    }
    gates = {
        "both_converged": dense.converged and sparse.converged,
        "same_accepted_points": dense.accepted_points == sparse.accepted_points,
        "both_yielded": dense.yielded_fraction > 0 and sparse.yielded_fraction > 0,
        "alpha_monotone": dense.alpha_monotone and sparse.alpha_monotone,
        "physical_differences_below_tolerance": max(differences.values()) < tolerance,
    }
    return {
        "tolerance": tolerance, "differences": differences, "gates": gates,
        "passed": all(gates.values()),
        "byte_identical_state": dense.state_sha256 == sparse.state_sha256,
    }


def report_json(report: PlasticPathQualification) -> str:
    return json.dumps(asdict(report), indent=2, sort_keys=True)
