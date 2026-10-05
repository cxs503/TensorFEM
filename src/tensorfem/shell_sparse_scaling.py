"""Scalable Shell4 tangent benchmark for optional ILU-GMRES qualification."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch

from .finite_rotation_layered_shell4 import assemble_finite_rotation_layered_shell4_sparse
from .layered_shell4_plasticity import LayeredShell4State
from .marine_panel_ultimate_fe import build_panel_case
from .sparse_direct import factorize_sparse


@dataclass(frozen=True)
class ShellSparseScalingPoint:
    mesh: int
    free_dofs: int
    matrix_nnz: int
    assembly_seconds: float
    ilu_factorization_seconds: float
    ilu_solve_seconds: float
    ilu_iterations: int
    ilu_relative_residual: float
    ilu_solution_relative_error: float
    ilu_matrix_bytes: int
    ilu_factor_bytes: int
    superlu_factorization_seconds: float
    superlu_solve_seconds: float
    superlu_relative_residual: float
    superlu_matrix_bytes: int
    superlu_factor_bytes: int
    dense_matrix_bytes: int

    @property
    def accuracy_passed(self) -> bool:
        return self.ilu_solution_relative_error < .01

    @property
    def factor_memory_advantage(self) -> bool:
        return self.ilu_factor_bytes < self.superlu_factor_bytes


def _drilling_stabilized(tangent: torch.Tensor, free: torch.Tensor,
                         relative_penalty: float = 1e-10) -> torch.Tensor:
    tangent = tangent.coalesce()
    gauge = torch.nonzero((free % 6) == 5).flatten()
    if not len(gauge):
        return tangent
    scale = max(float(tangent.values().abs().max()), 1.)
    penalty = torch.sparse_coo_tensor(
        torch.stack((gauge, gauge)),
        tangent.values().new_full((len(gauge),), relative_penalty*scale),
        tangent.shape, check_invariants=True,
    )
    return (tangent+penalty).coalesce()


def benchmark_shell_sparse_scaling(mesh: int, *, drop_tolerance: float = 1e-4,
                                   fill_factor: float = 10.,
                                   rtol: float = 1e-8,
                                   maxiter: int = 300,
                                   tangent_cache: str | Path | None = None
                                   ) -> ShellSparseScalingPoint:
    """Compare ILU-GMRES and SuperLU on the same real assembled panel tangent."""
    if mesh < 2:
        raise ValueError("mesh must be at least 2")
    case = build_panel_case(mesh)
    mask = torch.ones(case.model.n_dofs, dtype=torch.bool)
    mask[case.fixed_dofs] = False
    free = torch.nonzero(mask).flatten()
    cache = None if tangent_cache is None else Path(tangent_cache)
    if cache is not None and cache.exists():
        payload = torch.load(cache, map_location="cpu", weights_only=True)
        if payload.get("mesh") != mesh or not torch.equal(payload.get("free"), free):
            raise RuntimeError("sparse tangent cache identity mismatch")
        tangent = payload["tangent"].coalesce()
        assembly_seconds = float(payload["assembly_seconds"])
    else:
        started = perf_counter()
        response = assemble_finite_rotation_layered_shell4_sparse(
            case.model, torch.zeros(case.model.n_dofs, dtype=case.model.nodes.dtype),
            LayeredShell4State.virgin(case.model), active_dofs=free,
        )
        tangent = _drilling_stabilized(response.tangent, free)
        assembly_seconds = perf_counter()-started
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix(cache.suffix+".tmp")
            torch.save({"mesh": mesh, "free": free, "tangent": tangent,
                        "assembly_seconds": assembly_seconds}, temporary)
            temporary.replace(cache)
    exact = torch.sin(torch.arange(len(free), dtype=case.model.nodes.dtype)*.31)
    rhs = tangent @ exact

    ilu = factorize_sparse(tangent, method="spilu",
                           drop_tolerance=drop_tolerance, fill_factor=fill_factor)
    started = perf_counter()
    iterative = ilu.solve_gmres(rhs, rtol=rtol, maxiter=maxiter, restart=40)
    ilu_solve_seconds = perf_counter()-started
    if not iterative.converged:
        raise RuntimeError(
            f"ILU-GMRES failed after {iterative.iterations} iterations; "
            f"relative residual={iterative.relative_residual:.3e}"
        )
    direct = factorize_sparse(tangent, method="splu")
    started = perf_counter()
    direct_answer = direct.solve(rhs)
    direct_solve_seconds = perf_counter()-started
    ilu_diagnostics = ilu.diagnostics
    direct_diagnostics = direct.diagnostics
    return ShellSparseScalingPoint(
        mesh, len(free), tangent._nnz(), assembly_seconds,
        ilu_diagnostics.factorization_seconds, ilu_solve_seconds,
        iterative.iterations, iterative.relative_residual,
        float(torch.linalg.vector_norm(iterative.x-exact)
              / torch.linalg.vector_norm(exact)),
        ilu_diagnostics.estimated_matrix_bytes,
        ilu_diagnostics.estimated_factor_bytes,
        direct_diagnostics.factorization_seconds, direct_solve_seconds,
        float(torch.linalg.vector_norm(tangent@direct_answer-rhs)
              / torch.linalg.vector_norm(rhs)),
        direct_diagnostics.estimated_matrix_bytes,
        direct_diagnostics.estimated_factor_bytes,
        len(free)*len(free)*case.model.nodes.element_size(),
    )


def scaling_point_dict(point: ShellSparseScalingPoint) -> dict[str, object]:
    result = asdict(point)
    result.update(accuracy_passed=point.accuracy_passed,
                  factor_memory_advantage=point.factor_memory_advantage)
    return result
