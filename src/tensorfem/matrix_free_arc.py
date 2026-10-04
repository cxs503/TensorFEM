"""Matrix-free Crisfield continuation for expensive nonlinear FE residuals.

The tangent action is a directional finite difference of the assembled force,
not a column-by-column global tangent.  Every action is evaluated from the
same immutable committed material state.  This is particularly useful for the
finite-rotation layered Shell4, whose current reference tangent is numerical.

The difference quotient is valid on a smooth active-set branch.  At a J2 yield
surface kink GMRES or Newton may fail; the attempted arc step is then reduced
without exposing its trial material state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

import torch

from .sparse_advanced import gmres


State = TypeVar("State")


@dataclass(frozen=True)
class MatrixFreeResponse(Generic[State]):
    internal_force: torch.Tensor
    trial_state: State
    payload: Any = None


@dataclass(frozen=True)
class MatrixFreeArcPoint(Generic[State]):
    step: int
    load_factor: float
    displacement: torch.Tensor
    state: State
    payload: Any
    iterations: int
    residual_norm: float
    krylov_iterations: int
    equilibrium_relative_norm: float
    constraint_relative_error: float


@dataclass(frozen=True)
class MatrixFreeArcResult(Generic[State]):
    points: tuple[MatrixFreeArcPoint[State], ...]
    converged: bool
    step_size: float
    committed_state: State
    displacement: torch.Tensor
    load_factor: float
    response_evaluations: int
    tangent_actions: int
    krylov_iterations: int
    rejected_steps: int
    termination_reason: str
    initial_equilibrium_norm: float
    initial_equilibrium_relative_norm: float


@dataclass(frozen=True)
class MatrixFreeShellArcPoint(Generic[State]):
    step: int
    load_factor: float
    displacement: torch.Tensor
    state: State
    stress: torch.Tensor
    iterations: int
    residual_norm: float
    krylov_iterations: int
    equilibrium_relative_norm: float
    constraint_relative_error: float


@dataclass(frozen=True)
class MatrixFreeShellArcResult(Generic[State]):
    points: tuple[MatrixFreeShellArcPoint[State], ...]
    converged: bool
    step_size: float
    committed_state: State
    displacement: torch.Tensor
    load_factor: float
    response_evaluations: int
    tangent_actions: int
    krylov_iterations: int
    rejected_steps: int
    termination_reason: str
    initial_equilibrium_norm: float
    initial_equilibrium_relative_norm: float


def _directional_step(value: torch.Tensor, direction: torch.Tensor,
                      relative_step: float) -> float:
    norm = float(torch.linalg.vector_norm(direction))
    if norm == 0.0:
        return 0.0
    scale = max(1.0, float(torch.linalg.vector_norm(value)))
    return relative_step * scale / norm


def solve_matrix_free_arc(
    response: Callable[[torch.Tensor, State], MatrixFreeResponse[State]],
    reference_load: torch.Tensor,
    initial: torch.Tensor,
    initial_state: State,
    *,
    steps: int,
    step_size: float,
    load_scale: float,
    tolerance: float = 1e-7,
    relative_equilibrium_tolerance: float | None = None,
    normalized_constraint_tolerance: float | None = None,
    max_iterations: int = 15,
    minimum_step: float = 1e-5,
    maximum_step: float | None = None,
    difference_step: float = 2e-6,
    krylov_rtol: float = 1e-8,
    krylov_atol: float = 1e-12,
    krylov_maxiter: int | None = None,
    krylov_restart: int | None = None,
    inverse_preconditioner: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
    step_tangent: Callable[[torch.Tensor, State], torch.Tensor | Callable[[torch.Tensor], torch.Tensor]] | None = None,
    initial_load_factor: float = 0.0,
    initial_equilibrium_relative_tolerance: float | None = None,
) -> MatrixFreeArcResult[State]:
    """Trace a proportional-load path using JVPs and restarted GMRES.

    ``response(u, committed)`` must be pure with respect to ``committed`` and
    return a trial state.  The solver retains that state only at an accepted
    equilibrium point.  Central directional differences are intentionally
    used for robust agreement with the existing dense numerical tangent.
    """
    if steps < 1 or step_size <= 0 or load_scale <= 0:
        raise ValueError("invalid arc-length controls")
    if difference_step <= 0 or krylov_rtol <= 0 or krylov_atol < 0:
        raise ValueError("invalid matrix-free controls")
    if (relative_equilibrium_tolerance is not None
            and relative_equilibrium_tolerance <= 0):
        raise ValueError("relative equilibrium tolerance must be positive")
    if (normalized_constraint_tolerance is not None
            and normalized_constraint_tolerance <= 0):
        raise ValueError("normalized constraint tolerance must be positive")
    if (initial_equilibrium_relative_tolerance is not None
            and initial_equilibrium_relative_tolerance <= 0):
        raise ValueError("initial equilibrium relative tolerance must be positive")
    if not torch.isfinite(torch.tensor(initial_load_factor)):
        raise ValueError("initial load factor must be finite")
    if initial.ndim != 1 or reference_load.shape != initial.shape:
        raise ValueError("initial and reference load must be equal-length vectors")
    if not bool(torch.all(torch.isfinite(initial))) or not bool(torch.all(torch.isfinite(reference_load))):
        raise ValueError("initial and reference load must be finite")

    maximum_step = step_size if maximum_step is None else maximum_step
    if maximum_step < minimum_step or minimum_step <= 0:
        raise ValueError("invalid minimum/maximum step")
    u = initial.clone()
    load = float(initial_load_factor)
    state = initial_state
    previous = None
    ds = step_size
    points: list[MatrixFreeArcPoint[State]] = []
    evaluations = actions = total_krylov = 0
    rejected_steps = 0
    termination_reason = "completed"

    def evaluate(value: torch.Tensor, committed: State) -> MatrixFreeResponse[State]:
        nonlocal evaluations
        result = response(value, committed)
        evaluations += 1
        if result.internal_force.shape != value.shape:
            raise ValueError("response force has the wrong shape")
        if not bool(torch.all(torch.isfinite(result.internal_force))):
            raise FloatingPointError("response force is not finite")
        return result

    def tangent_action(value: torch.Tensor, committed: State, vector: torch.Tensor) -> torch.Tensor:
        nonlocal actions
        actions += 1
        h = _directional_step(value, vector, difference_step)
        if h == 0.0:
            return torch.zeros_like(vector)
        plus = evaluate(value + h * vector, committed).internal_force
        minus = evaluate(value - h * vector, committed).internal_force
        return (plus - minus) / (2.0 * h)

    def linear(operator, rhs, preconditioner=None):
        nonlocal total_krylov
        answer = gmres(operator, rhs, rtol=krylov_rtol, atol=krylov_atol,
                       maxiter=krylov_maxiter, restart=krylov_restart,
                       inverse_preconditioner=preconditioner)
        total_krylov += answer.iterations
        return answer

    initial_response = evaluate(u, state)
    initial_residual = initial_response.internal_force - load * reference_load
    initial_equilibrium_norm = float(torch.linalg.vector_norm(initial_residual))
    initial_scale = max(
        float(torch.linalg.vector_norm(initial_response.internal_force)),
        abs(load) * float(torch.linalg.vector_norm(reference_load)), 1.0,
    )
    initial_equilibrium_relative_norm = initial_equilibrium_norm / initial_scale
    if (initial_equilibrium_relative_tolerance is not None
            and initial_equilibrium_relative_norm > initial_equilibrium_relative_tolerance):
        return MatrixFreeArcResult(
            (), False, ds, state, u, load, evaluations, actions, total_krylov,
            0, "initial_equilibrium_failed", initial_equilibrium_norm,
            initial_equilibrium_relative_norm,
        )

    accepted = 0
    while accepted < steps:
        committed_u = u.clone()
        committed_load = load
        committed_state = state
        frozen_operator = (None if step_tangent is None
                           else step_tangent(committed_u, committed_state))
        if frozen_operator is None:
            operator = lambda vector: tangent_action(
                committed_u, committed_state, vector
            )
        else:
            operator = lambda vector: (
                frozen_operator(vector) if callable(frozen_operator)
                else frozen_operator @ vector
            )

        predictor = linear(
            operator,
            reference_load,
            inverse_preconditioner,
        )
        if not predictor.converged:
            rejected_steps += 1
            termination_reason = "predictor_gmres_failed"
            ds *= 0.5
            if ds < minimum_step:
                break
            continue
        direction_u = predictor.x
        direction = torch.cat((direction_u, direction_u.new_tensor([1.0])))
        metric = torch.cat((direction_u, direction_u.new_tensor([load_scale])))
        sign = 1.0
        if previous is not None:
            dot = (torch.dot(direction_u, previous[:-1])
                   + load_scale**2 * previous[-1])
            sign = 1.0 if float(dot) >= 0 else -1.0
        dl = sign * ds / float(torch.linalg.vector_norm(metric))
        trial_u = committed_u + dl * direction_u
        trial_load = committed_load + dl
        ok = False
        last = float("inf")
        relative_equilibrium = float("inf")
        relative_constraint = float("inf")
        trial_response = None
        step_krylov_start = total_krylov

        for iteration in range(1, max_iterations + 1):
            trial_response = evaluate(trial_u, committed_state)
            residual = trial_response.internal_force - trial_load * reference_load
            Du = trial_u - committed_u
            Dl = trial_load - committed_load
            constraint = torch.dot(Du, Du) + (load_scale * Dl)**2 - ds**2
            equilibrium_norm = float(torch.linalg.vector_norm(residual))
            # Use the larger of internal and applied loads.  This remains
            # meaningful near a limit point and avoids scaling by cancellation
            # in their difference.  A unit floor supports normalized reference
            # loads while the absolute criterion remains available separately.
            equilibrium_scale = max(
                float(torch.linalg.vector_norm(trial_response.internal_force)),
                abs(trial_load) * float(torch.linalg.vector_norm(reference_load)),
                1.0,
            )
            relative_equilibrium = equilibrium_norm / equilibrium_scale
            constraint_scale = max(ds**2, torch.finfo(initial.dtype).tiny)
            relative_constraint = abs(float(constraint)) / constraint_scale
            if (relative_equilibrium_tolerance is None
                    and normalized_constraint_tolerance is None):
                # Backward-compatible dimensional diagnostic and criteria.
                last = float(torch.linalg.vector_norm(
                    torch.cat((residual, constraint.reshape(1)))
                ))
                equilibrium_ok = equilibrium_norm <= tolerance
                constraint_ok = abs(float(constraint)) <= tolerance * max(ds, 1.0)
            else:
                # Never combine newtons and squared metres.  ``last`` is now a
                # dimensionless worst normalized equation error.
                last = max(relative_equilibrium, relative_constraint)
                equilibrium_ok = (equilibrium_norm <= tolerance
                    if relative_equilibrium_tolerance is None
                    else relative_equilibrium <= relative_equilibrium_tolerance)
                constraint_ok = (abs(float(constraint)) <= tolerance * max(ds, 1.0)
                    if normalized_constraint_tolerance is None
                    else relative_constraint <= normalized_constraint_tolerance)
            if equilibrium_ok and constraint_ok:
                ok = True
                break

            def augmented(vector: torch.Tensor) -> torch.Tensor:
                du, dload = vector[:-1], vector[-1]
                if frozen_operator is None:
                    tangent_du = tangent_action(trial_u, committed_state, du)
                else:
                    tangent_du = (frozen_operator(du) if callable(frozen_operator)
                                  else frozen_operator @ du)
                equilibrium = tangent_du - dload * reference_load
                arc = 2.0 * torch.dot(Du, du) + 2.0 * load_scale**2 * Dl * dload
                return torch.cat((equilibrium, arc.reshape(1)))

            augmented_pre = None
            if callable(inverse_preconditioner):
                # The constraint row has no stable diagonal near the predictor;
                # leave it unscaled while preconditioning equilibrium rows.
                augmented_pre = lambda vector: torch.cat((
                    inverse_preconditioner(vector[:-1]), vector[-1:],
                ))
            elif inverse_preconditioner is not None:
                if inverse_preconditioner.shape != initial.shape:
                    raise ValueError("inverse preconditioner has the wrong shape")
                augmented_pre = torch.cat((inverse_preconditioner, initial.new_ones(1)))
            correction = linear(augmented, -torch.cat((residual, constraint.reshape(1))),
                                augmented_pre)
            if not correction.converged:
                termination_reason = "corrector_gmres_failed"
                break
            trial_u = trial_u + correction.x[:-1]
            trial_load += float(correction.x[-1])

        if ok and trial_response is not None:
            u = trial_u
            load = trial_load
            state = trial_response.trial_state
            accepted += 1
            previous = torch.cat((u - committed_u, u.new_tensor([load - committed_load])))
            points.append(MatrixFreeArcPoint(
                accepted, load, u.clone(), state, trial_response.payload,
                iteration, last, total_krylov - step_krylov_start,
                relative_equilibrium, relative_constraint,
            ))
            if iteration <= 4:
                ds = min(maximum_step, ds * 1.25)
            elif iteration > 8:
                ds = max(minimum_step, ds * 0.75)
        else:
            # ``u``, ``load`` and ``state`` still reference only the last
            # accepted point.  All trial states become unreachable here.
            rejected_steps += 1
            if termination_reason not in ("corrector_gmres_failed",):
                termination_reason = "corrector_iterations_exhausted"
            ds *= 0.5
            if ds < minimum_step:
                break

    return MatrixFreeArcResult(
        tuple(points), accepted == steps, ds, state, u, load,
        evaluations, actions, total_krylov, rejected_steps,
        "completed" if accepted == steps else termination_reason,
        initial_equilibrium_norm, initial_equilibrium_relative_norm,
    )


def solve_matrix_free_finite_rotation_shell4(
    model, reference_load: torch.Tensor, fixed_dofs: torch.Tensor, *,
    steps: int, step_size: float, load_scale: float, initial_state=None,
    initial_displacement: torch.Tensor | None = None, **controls,
):
    """Shell4 adapter that never assembles an element or global tangent."""
    from .finite_rotation_layered_shell4 import assemble_finite_rotation_layered_shell4
    from .layered_shell4_plasticity import LayeredShell4State

    model.validate()
    fixed = torch.tensor(sorted(set(int(i) for i in fixed_dofs)), dtype=torch.long,
                         device=model.nodes.device)
    if bool(torch.any(fixed < 0)) or (len(fixed) and int(fixed.max()) >= model.n_dofs):
        raise ValueError("fixed shell DOF out of range")
    mask = torch.ones(model.n_dofs, dtype=torch.bool, device=model.nodes.device)
    mask[fixed] = False
    free = torch.nonzero(mask).flatten()
    full = torch.zeros(model.n_dofs, dtype=model.nodes.dtype, device=model.nodes.device)
    if initial_displacement is not None:
        if initial_displacement.shape != full.shape:
            raise ValueError("wrong initial displacement length")
        full.copy_(initial_displacement.to(model.nodes))
    state = LayeredShell4State.virgin(model) if initial_state is None else initial_state

    linearization = controls.pop("linearization", "jvp")
    preconditioner = controls.pop("preconditioner", None)
    if linearization not in ("jvp", "frozen_tangent"):
        raise ValueError("linearization must be 'jvp' or 'frozen_tangent'")
    if preconditioner not in (None, "jacobi", "block"):
        raise ValueError("preconditioner must be None, 'jacobi' or 'block'")

    def full_value(reduced):
        value = full.clone()
        value[free] = reduced
        return value

    def shell_response(reduced, committed):
        value = full_value(reduced)
        evaluated = assemble_finite_rotation_layered_shell4(
            model, value, committed, tangent=False,
        )
        return MatrixFreeResponse(
            evaluated.internal_force[free], evaluated.trial_state,
            {"stress": evaluated.stress, "full_displacement": value},
        )

    tangent_cache = {}
    def assembled_tangent(reduced, committed):
        # A step-level cache lets the same assembly provide both the frozen
        # quasi-Newton operator and its preconditioner.
        if (tangent_cache.get("state") is committed
                and torch.equal(tangent_cache["value"], reduced)):
            return tangent_cache["tangent"]
        evaluated = assemble_finite_rotation_layered_shell4(
            model, full_value(reduced), committed, tangent=True,
        )
        tangent_cache.clear()
        tangent_cache.update(state=committed, value=reduced.clone(),
                             tangent=evaluated.tangent[free[:, None], free])
        return tangent_cache["tangent"]

    step_operator = assembled_tangent if linearization == "frozen_tangent" else None
    inverse = controls.pop("inverse_preconditioner", None)
    if preconditioner is not None:
        tangent = assembled_tangent(full[free], state)
        if preconditioner == "jacobi":
            diagonal = torch.diagonal(tangent)
            floor = max(float(diagonal.abs().max()) * 1e-10, 1e-12)
            safe = torch.where(diagonal.abs() > floor, diagonal,
                               torch.full_like(diagonal, floor))
            inverse = safe.reciprocal()
        else:
            # Node blocks preserve translation/rotation coupling while staying
            # cheap.  Pseudoinverses fail closed on constrained/singular modes.
            positions = {int(dof): i for i, dof in enumerate(free.tolist())}
            blocks = []
            for node in range(len(model.nodes)):
                indices = [positions[d] for d in range(6*node, 6*node+6) if d in positions]
                if indices:
                    ids = torch.tensor(indices, dtype=torch.long, device=free.device)
                    blocks.append((ids, torch.linalg.pinv(tangent[ids[:, None], ids], rtol=1e-10)))
            def inverse(vector):
                result = torch.zeros_like(vector)
                for ids, block_inverse in blocks:
                    result[ids] = block_inverse @ vector[ids]
                return result

    reduced = solve_matrix_free_arc(
        shell_response, reference_load.to(model.nodes)[free], full[free], state,
        steps=steps, step_size=step_size, load_scale=load_scale,
        inverse_preconditioner=inverse, step_tangent=step_operator, **controls,
    )
    points = tuple(MatrixFreeShellArcPoint(
        point.step, point.load_factor, point.payload["full_displacement"],
        point.state, point.payload["stress"], point.iterations,
        point.residual_norm, point.krylov_iterations,
        point.equilibrium_relative_norm, point.constraint_relative_error,
    ) for point in reduced.points)
    terminal = full.clone()
    terminal[free] = reduced.displacement
    return MatrixFreeShellArcResult(
        points, reduced.converged, reduced.step_size, reduced.committed_state,
        terminal, reduced.load_factor, reduced.response_evaluations,
        reduced.tangent_actions, reduced.krylov_iterations,
        reduced.rejected_steps, reduced.termination_reason,
        reduced.initial_equilibrium_norm, reduced.initial_equilibrium_relative_norm,
    )
