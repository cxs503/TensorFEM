"""Transactional adaptive load stepping for small verified nonlinear problems.

The driver is deliberately independent of element technology.  A problem returns
the free-DOF residual, its consistent *internal-force* tangent, and a trial state.
Material history is committed only after equilibrium, so rejected increments can
be retried without corrupting history.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal
import json
import torch


class NonlinearConvergenceError(RuntimeError):
    """Raised when the minimum increment cannot converge (fail closed)."""


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    residual_norm: float
    correction_norm: float
    line_search_scale: float


@dataclass(frozen=True)
class IncrementRecord:
    load_factor: float
    step_size: float
    iterations: tuple[IterationRecord, ...]
    retries: int


@dataclass
class StepState:
    load_factor: float
    displacement: torch.Tensor
    material_state: Any = None
    history: list[IncrementRecord] = field(default_factory=list)

    def clone(self) -> "StepState":
        return StepState(self.load_factor, self.displacement.clone(),
                         _clone_state(self.material_state), list(self.history))


ResidualTangent = Callable[[torch.Tensor, float, Any], tuple[torch.Tensor, torch.Tensor, Any]]


def _clone_state(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.clone()
    if hasattr(value, "__dataclass_fields__"):
        return type(value)(*(_clone_state(getattr(value, k))
                             for k in value.__dataclass_fields__))
    if isinstance(value, tuple):
        return tuple(_clone_state(x) for x in value)
    if isinstance(value, dict):
        return {k: _clone_state(v) for k, v in value.items()}
    return value


def solve_adaptive(
    residual_tangent: ResidualTangent,
    initial: StepState,
    *,
    target_factor: float = 1.0,
    initial_increment: float = 0.1,
    minimum_increment: float = 1e-5,
    maximum_increment: float = 0.25,
    tolerance: float = 1e-9,
    max_iterations: int = 20,
    method: Literal["newton", "modified_newton"] = "newton",
    line_search: bool = True,
) -> StepState:
    """Advance to ``target_factor`` with rollback and automatic step control."""
    if not (0 < minimum_increment <= initial_increment <= maximum_increment):
        raise ValueError("increments must satisfy 0 < min <= initial <= max")
    if target_factor < initial.load_factor:
        raise ValueError("target_factor must not precede the checkpoint")
    if method not in ("newton", "modified_newton"):
        raise ValueError("unknown Newton method")
    accepted = initial.clone()
    increment = initial_increment
    retries = 0
    while accepted.load_factor < target_factor - 10 * torch.finfo(torch.float64).eps:
        trial_factor = min(target_factor, accepted.load_factor + increment)
        u = accepted.displacement.clone()
        base_state = _clone_state(accepted.material_state)
        records: list[IterationRecord] = []
        frozen_k = None
        converged = False
        scale = None
        trial_state = base_state
        for iteration in range(1, max_iterations + 1):
            r, k, trial_state = residual_tangent(u, trial_factor, base_state)
            rn = float(torch.linalg.vector_norm(r))
            if scale is None:
                scale = max(rn, 1.0)
            if rn <= tolerance * scale:
                converged = True
                break
            if method == "modified_newton":
                frozen_k = k if frozen_k is None else frozen_k
                ksolve = frozen_k
            else:
                ksolve = k
            try:
                du = torch.linalg.solve(ksolve, r)
            except torch.linalg.LinAlgError:
                break
            alpha = 1.0
            if line_search:
                # Armijo backtracking on residual norm; state stays transactional.
                while alpha > 1.0 / 128.0:
                    candidate = u + alpha * du
                    rc, _, _ = residual_tangent(candidate, trial_factor, base_state)
                    if float(torch.linalg.vector_norm(rc)) < rn:
                        break
                    alpha *= 0.5
                if alpha <= 1.0 / 128.0:
                    break
            u = u + alpha * du
            records.append(IterationRecord(iteration, rn,
                           float(torch.linalg.vector_norm(alpha * du)), alpha))
        if converged:
            accepted = StepState(trial_factor, u, _clone_state(trial_state),
                accepted.history + [IncrementRecord(trial_factor,
                    trial_factor - accepted.load_factor, tuple(records), retries)])
            retries = 0
            if len(records) <= 5:
                increment = min(maximum_increment, increment * 1.5)
        else:
            retries += 1
            increment *= 0.5
            if increment < minimum_increment:
                raise NonlinearConvergenceError(
                    f"failed at load factor {trial_factor:g}; minimum increment exhausted")
    return accepted


def save_checkpoint(state: StepState, path: str | Path) -> None:
    """Write a portable JSON checkpoint for tensor/dataclass material states."""
    def encode(x: Any):
        if torch.is_tensor(x):
            return {"tensor": x.detach().cpu().tolist(), "dtype": str(x.dtype)}
        if hasattr(x, "__dataclass_fields__"):
            return {"dataclass": type(x).__name__,
                    "fields": {k: encode(getattr(x, k)) for k in x.__dataclass_fields__}}
        if x is None or isinstance(x, (str, int, float, bool)):
            return x
        if isinstance(x, (list, tuple)):
            return [encode(v) for v in x]
        if isinstance(x, dict):
            return {k: encode(v) for k, v in x.items()}
        raise TypeError(f"unsupported checkpoint value: {type(x).__name__}")
    payload = {"format": "tensorfem.nonlinear-step.v1",
               "load_factor": state.load_factor,
               "displacement": encode(state.displacement),
               "material_state": encode(state.material_state)}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_checkpoint(path: str | Path, *, material_decoder=None) -> StepState:
    """Read a checkpoint; a decoder reconstructs application material state."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "tensorfem.nonlinear-step.v1":
        raise ValueError("unsupported checkpoint format")
    def tensor(obj):
        name = obj["dtype"].split(".")[-1]
        return torch.tensor(obj["tensor"], dtype=getattr(torch, name))
    disp = tensor(payload["displacement"])
    raw = payload["material_state"]
    state = material_decoder(raw, tensor) if material_decoder and raw is not None else raw
    return StepState(float(payload["load_factor"]), disp, state)


def total_lagrangian_truss_problem(model, reference_load: torch.Tensor):
    """Adapt an existing :class:`NonlinearTrussModel` to the unified driver."""
    from .nonlinear_truss import internal_force_and_tangent
    fixed = set(model.fixed_dofs.tolist())
    free = torch.tensor([i for i in range(reference_load.numel()) if i not in fixed],
                        dtype=torch.long, device=reference_load.device)
    def evaluate(u_free, factor, state):
        u = torch.zeros_like(reference_load).index_copy(0, free, u_free)
        fint, kt, _ = internal_force_and_tangent(model, u)
        return factor * reference_load[free] - fint[free], kt[free][:, free], state
    return evaluate, free


def elastoplastic_bar_problem(length: float, area: float, young: float,
                              yield_stress: float, hardening: float,
                              reference_force: float, *, dtype=torch.float64):
    """One-DOF bar adapter proving constitutive state commit/rollback integration."""
    from .plasticity import Plastic1DState, update_bilinear_1d
    zero = torch.zeros((), dtype=dtype)
    initial_state = Plastic1DState(zero.clone(), zero.clone())
    def evaluate(u, factor, committed):
        strain = u[0] / length
        stress, tangent, trial = update_bilinear_1d(
            strain, young, yield_stress, hardening, committed)
        residual = torch.as_tensor([factor * reference_force], dtype=dtype) - area * stress.reshape(1)
        stiffness = (area * tangent / length).reshape(1, 1)
        return residual, stiffness, trial
    return evaluate, StepState(0.0, torch.zeros(1, dtype=dtype), initial_state)
