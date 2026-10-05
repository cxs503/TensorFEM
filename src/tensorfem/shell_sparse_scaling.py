"""Scalable Shell4 tangent benchmark for optional ILU-GMRES qualification."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import warnings
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


def _write_cache(cache: Path, mesh: int, free: torch.Tensor,
                 tangent: torch.Tensor, assembly_seconds: float) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(cache.suffix+".tmp")
    torch.save({"mesh": mesh, "free": free, "tangent": tangent,
                "assembly_seconds": assembly_seconds}, temporary)
    temporary.replace(cache)
    digest = hashlib.sha256(cache.read_bytes()).hexdigest()
    digest_temporary = cache.with_suffix(cache.suffix+".sha256.tmp")
    digest_temporary.write_text(digest+"\n")
    digest_temporary.replace(cache.with_suffix(cache.suffix+".sha256"))


def _load_sparse_cache(cache: Path) -> dict[str, object]:
    # PyTorch 2.14 emits an informational invariant-scan warning for every
    # weights-only sparse load. TensorFEM requires that scan; suppress only its
    # known notice so release tests may keep all other warnings fatal.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Validating sparse tensor invariants because weights_only=True.*",
            category=UserWarning,
        )
        return torch.load(cache, map_location="cpu", weights_only=True)


def verify_legacy_sparse_tangent_cache(mesh: int, tangent_cache: str | Path,
                                       *, relative_tolerance: float = 1e-10
                                       ) -> dict[str, float | int]:
    """Explicitly reassemble and migrate an unhashed legacy cache.

    Merely hashing an unknown historical file would endorse its contents.
    Migration instead checks identity, reassembles the current qualified
    tangent, compares every stored coefficient and a deterministic action,
    then replaces the cache with the freshly assembled tensor and sidecar.
    """
    cache = Path(tangent_cache)
    if not cache.exists():
        raise FileNotFoundError(cache)
    if cache.with_suffix(cache.suffix+".sha256").exists():
        raise RuntimeError("cache already has an integrity sidecar")
    if relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be positive")
    case = build_panel_case(mesh)
    mask = torch.ones(case.model.n_dofs, dtype=torch.bool)
    mask[case.fixed_dofs] = False
    free = torch.nonzero(mask).flatten()
    legacy = _load_sparse_cache(cache)
    if legacy.get("mesh") != mesh or not torch.equal(legacy.get("free"), free):
        raise RuntimeError("legacy sparse tangent cache identity mismatch")
    started = perf_counter()
    response = assemble_finite_rotation_layered_shell4_sparse(
        case.model, torch.zeros(case.model.n_dofs, dtype=case.model.nodes.dtype),
        LayeredShell4State.virgin(case.model), active_dofs=free,
    )
    current = _drilling_stabilized(response.tangent, free)
    elapsed = perf_counter()-started
    old = legacy.get("tangent")
    if not isinstance(old, torch.Tensor) or old.layout != torch.sparse_coo:
        raise RuntimeError("legacy sparse tangent cache payload is invalid")
    old = old.coalesce()
    same_pattern = (old.shape == current.shape
                    and torch.equal(old.indices(), current.indices()))
    scale = max(float(current.values().abs().max()), 1.)
    coefficient_error = (float((old.values()-current.values()).abs().max())/scale
                         if same_pattern else float("inf"))
    direction = torch.sin(torch.arange(len(free), dtype=current.dtype)*.31)
    action_scale = max(float(torch.linalg.vector_norm(current@direction)), 1.)
    action_error = float(torch.linalg.vector_norm(old@direction-current@direction))/action_scale
    if (not same_pattern or coefficient_error > relative_tolerance
            or action_error > relative_tolerance):
        raise RuntimeError(
            "legacy sparse tangent cache failed reassembly verification: "
            f"coefficient_error={coefficient_error:.3e}, action_error={action_error:.3e}"
        )
    _write_cache(cache, mesh, free, current, elapsed)
    return {"mesh": mesh, "active_dofs": len(free),
            "coefficient_relative_error": coefficient_error,
            "action_relative_error": action_error,
            "reassembly_seconds": elapsed}


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
        sidecar = cache.with_suffix(cache.suffix+".sha256")
        if not sidecar.exists():
            raise RuntimeError("sparse tangent cache integrity sidecar is missing")
        expected = sidecar.read_text().strip()
        actual = hashlib.sha256(cache.read_bytes()).hexdigest()
        if expected != actual:
            raise RuntimeError("sparse tangent cache integrity mismatch")
        payload = _load_sparse_cache(cache)
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
            _write_cache(cache, mesh, free, tangent, assembly_seconds)
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


def sparse_scaling_qualification(
    points: list[ShellSparseScalingPoint] | tuple[ShellSparseScalingPoint, ...],
    *, minimum_active_dofs: int = 10_000, maximum_solution_error: float = .01,
    maximum_true_residual: float = 1e-8,
) -> dict[str, object]:
    """Build a deterministic, fail-closed release gate from measured points."""
    if minimum_active_dofs < 1 or maximum_solution_error <= 0 or maximum_true_residual <= 0:
        raise ValueError("invalid sparse qualification thresholds")
    eligible = [point for point in points if point.free_dofs >= minimum_active_dofs]
    largest = max(eligible, key=lambda point: point.free_dofs) if eligible else None
    gates = {
        "minimum_active_dofs": largest is not None,
        "solution_error": (largest is not None
                           and largest.ilu_solution_relative_error < maximum_solution_error),
        "true_residual": (largest is not None
                          and largest.ilu_relative_residual < maximum_true_residual),
        "memory_advantage_over_superlu": (largest is not None and
            largest.ilu_matrix_bytes+largest.ilu_factor_bytes
            < largest.superlu_matrix_bytes+largest.superlu_factor_bytes),
        "memory_advantage_over_dense": (largest is not None and
            largest.ilu_matrix_bytes+largest.ilu_factor_bytes
            < largest.dense_matrix_bytes),
    }
    if largest is None:
        metrics = None
        performance_claim = "none"
    else:
        ilu_bytes = largest.ilu_matrix_bytes+largest.ilu_factor_bytes
        superlu_bytes = largest.superlu_matrix_bytes+largest.superlu_factor_bytes
        ilu_time = largest.ilu_factorization_seconds+largest.ilu_solve_seconds
        superlu_time = (largest.superlu_factorization_seconds
                        +largest.superlu_solve_seconds)
        metrics = {
            "mesh": largest.mesh, "active_dofs": largest.free_dofs,
            "matrix_nnz": largest.matrix_nnz,
            "solution_relative_error": largest.ilu_solution_relative_error,
            "true_relative_residual": largest.ilu_relative_residual,
            "ilu_total_bytes": ilu_bytes, "superlu_total_bytes": superlu_bytes,
            "dense_matrix_bytes": largest.dense_matrix_bytes,
            "memory_reduction_vs_superlu": 1-ilu_bytes/superlu_bytes,
            "memory_reduction_vs_dense": 1-ilu_bytes/largest.dense_matrix_bytes,
            "assembly_seconds": largest.assembly_seconds,
            "ilu_linear_seconds": ilu_time, "superlu_linear_seconds": superlu_time,
            "linear_time_speedup": superlu_time/ilu_time,
        }
        performance_claim = ("time_and_memory" if ilu_time < superlu_time
                             else "memory_only")
    report = {
        "schema": "tensorfem.shell-sparse-scaling-qualification/1",
        "thresholds": {"minimum_active_dofs": minimum_active_dofs,
                       "maximum_solution_error": maximum_solution_error,
                       "maximum_true_residual": maximum_true_residual},
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__, "device": "cpu"},
        "points": [scaling_point_dict(point) for point in points],
        "gates": gates, "passed": all(gates.values()),
        "performance_claim": performance_claim, "qualified_metrics": metrics,
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["evidence_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return report


def verify_sparse_scaling_report(report: dict[str, object] | str | Path
                                 ) -> dict[str, object]:
    """Verify report schema, evidence hash, and fail-closed gate consistency."""
    if isinstance(report, (str, Path)):
        try:
            payload = json.loads(Path(report).read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"cannot read sparse qualification report: {error}") from error
    else:
        payload = dict(report)
    if payload.get("schema") != "tensorfem.shell-sparse-scaling-qualification/1":
        raise RuntimeError("unsupported sparse qualification report schema")
    expected = payload.pop("evidence_sha256", None)
    if not isinstance(expected, str) or len(expected) != 64:
        raise RuntimeError("sparse qualification report has no valid evidence hash")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    actual = hashlib.sha256(canonical.encode()).hexdigest()
    if actual != expected:
        raise RuntimeError("sparse qualification report evidence hash mismatch")
    gates = payload.get("gates")
    if not isinstance(gates, dict) or not gates:
        raise RuntimeError("sparse qualification report has no gates")
    passed = all(value is True for value in gates.values())
    if payload.get("passed") is not passed:
        raise RuntimeError("sparse qualification report gate result is inconsistent")
    claim = payload.get("performance_claim")
    metrics = payload.get("qualified_metrics")
    if passed and (claim not in ("memory_only", "time_and_memory")
                   or not isinstance(metrics, dict)):
        raise RuntimeError("passed sparse qualification report lacks qualified metrics")
    return {**payload, "evidence_sha256": expected}
