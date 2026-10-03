"""Corotational finite-rotation wrapper for the layered Shell4 material model.

The element removes a best-fit proper rigid rotation, evaluates the existing
2x2-by-layer plane-stress J2 section in that corotated frame, and maps section
forces back by virtual work.  The global algorithmic tangent differentiates
the complete mapped residual, so it includes frame (geometric) terms.  Total
nodal rotations are exponential-map vectors; strains remain small in the
corotated frame.  This is therefore a finite-rotation/small-strain facet, not a
general finite-membrane-strain shell.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .corotational_shell import _proper_fit_rotation, _rotation_vector
from .layered_shell4_plasticity import (
    LayeredShell4Model,
    LayeredShell4State,
    element_response,
)
from .shell_consistent import rotation_matrix_from_vector


@dataclass(frozen=True)
class FiniteRotationShell4Response:
    internal_force: torch.Tensor
    tangent: torch.Tensor | None
    stress: torch.Tensor
    trial_state: LayeredShell4State


def _element_dofs(conn: torch.Tensor) -> torch.Tensor:
    return torch.stack(tuple(6 * conn + k for k in range(6)), 1).reshape(-1).long()


def _corotated_deformation(reference: torch.Tensor, dofs: torch.Tensor) -> torch.Tensor:
    """Return 24 deformation coordinates after removing rigid motion."""
    q = dofs.reshape(4, 6)
    current = reference + q[:, :3]
    frame = _proper_fit_rotation(reference, current)
    translations = ((current - current.mean(0)) @ frame
                    - (reference - reference.mean(0)))
    rotations = torch.stack([rotation_matrix_from_vector(v) for v in q[:, 3:]])
    relative = frame.T.unsqueeze(0) @ rotations
    rotation_vectors = torch.stack([_rotation_vector(value) for value in relative])
    return torch.cat((translations, rotation_vectors), 1).reshape(-1)


def _jacobian(function, value: torch.Tensor, *, relative_step: float = 2e-6) -> torch.Tensor:
    """Centred numerical Jacobian, deliberately differentiating SVD frames."""
    columns = []
    for j in range(value.numel()):
        h = relative_step * max(1.0, abs(float(value[j])))
        delta = torch.zeros_like(value)
        delta[j] = h
        columns.append((function(value + delta) - function(value - delta)) / (2 * h))
    return torch.stack(columns, 1)


def _element_force(
    model: LayeredShell4Model,
    element: int,
    dofs: torch.Tensor,
    committed,
):
    conn = model.elements[element].long()
    reference = model.nodes[conn]
    deformation = _corotated_deformation(reference, dofs)
    mapping = _jacobian(lambda value: _corotated_deformation(reference, value), dofs)
    local_force, _, stress, trial = element_response(
        model, element, deformation, committed
    )
    return mapping.T @ local_force, stress, trial


def finite_rotation_element_response(
    model: LayeredShell4Model,
    element: int,
    dofs: torch.Tensor,
    committed,
    *,
    tangent: bool = True,
    tangent_step: float = 8e-6,
):
    """Evaluate one finite-rotation facet from immutable committed history.

    All perturbed residual evaluations start from ``committed``.  Consequently
    neither tangent construction nor a rejected Newton iterate can commit a
    plastic material point.
    """
    if dofs.shape != (24,):
        raise ValueError("finite-rotation Shell4 element requires 24 DOFs")
    force, stress, trial = _element_force(model, element, dofs, committed)
    stiffness = None
    if tangent:
        stiffness = _jacobian(
            lambda value: _element_force(model, element, value, committed)[0],
            dofs,
            relative_step=tangent_step,
        )
        # Numerical frame derivatives introduce only round-off skew.  Keep the
        # actual derivative: plastic algorithmic tangents need not be exactly
        # symmetric at an active-set transition.
    return force, stiffness, stress, trial


def assemble_finite_rotation_layered_shell4(
    model: LayeredShell4Model,
    displacement: torch.Tensor,
    committed: LayeredShell4State,
    *,
    tangent: bool = True,
) -> FiniteRotationShell4Response:
    """Assemble a mesh while preserving trial/commit separation."""
    model.validate()
    if displacement.shape != (model.n_dofs,):
        raise ValueError("wrong displacement vector length")
    internal = torch.zeros_like(displacement)
    stiffness = (torch.zeros((model.n_dofs, model.n_dofs), dtype=model.nodes.dtype,
                             device=model.nodes.device) if tangent else None)
    stresses, states = [], []
    for element, conn in enumerate(model.elements):
        ids = _element_dofs(conn)
        force, local_k, stress, state = finite_rotation_element_response(
            model, element, displacement[ids], committed.points[element], tangent=tangent
        )
        internal.index_add_(0, ids, force)
        if tangent:
            stiffness.index_put_((ids[:, None], ids[None, :]), local_k, accumulate=True)
        stresses.append(stress)
        states.append(state)
    return FiniteRotationShell4Response(
        internal, stiffness, torch.stack(stresses), LayeredShell4State(tuple(states))
    )


def model_with_imperfection(
    model: LayeredShell4Model, imperfection: torch.Tensor
) -> LayeredShell4Model:
    """Create a stress-free imperfect reference geometry (never an applied DOF)."""
    if imperfection.shape != model.nodes.shape:
        raise ValueError("imperfection must have the same shape as model.nodes")
    if imperfection.dtype != model.nodes.dtype or imperfection.device != model.nodes.device:
        raise ValueError("imperfection must share model node dtype and device")
    nodes = model.nodes + imperfection
    if not bool(torch.all(torch.isfinite(nodes))):
        raise ValueError("imperfect reference nodes must be finite")
    result = replace(model, nodes=nodes)
    result.validate()
    return result


@dataclass(frozen=True)
class FiniteRotationShell4Step:
    displacement: torch.Tensor
    reaction: torch.Tensor
    stress: torch.Tensor
    state: LayeredShell4State
    iterations: int
    residual_norm: float


@dataclass(frozen=True)
class FiniteRotationArcPoint:
    step: int
    load_factor: float
    displacement: torch.Tensor
    state: LayeredShell4State
    stress: torch.Tensor
    iterations: int
    residual_norm: float


@dataclass(frozen=True)
class FiniteRotationArcResult:
    points: tuple[FiniteRotationArcPoint, ...]
    converged: bool
    step_size: float
    committed_state: LayeredShell4State
    displacement: torch.Tensor
    load_factor: float


def solve_finite_rotation_increment(
    model: LayeredShell4Model,
    external_force: torch.Tensor,
    load_factor: float,
    fixed_dofs: torch.Tensor,
    committed: LayeredShell4State,
    displacement: torch.Tensor,
    *,
    tolerance: float = 1e-7,
    max_iterations: int = 20,
) -> FiniteRotationShell4Step:
    """Full Newton increment; state is returned only after convergence."""
    fixed = fixed_dofs.to(device=model.nodes.device, dtype=torch.long)
    mask = torch.ones(model.n_dofs, dtype=torch.bool, device=model.nodes.device)
    mask[fixed] = False
    free = torch.nonzero(mask).flatten()
    load = external_force.to(model.nodes) * load_factor
    value = displacement.clone()
    scale = max(float(torch.linalg.vector_norm(load[free])), 1.0)
    for iteration in range(1, max_iterations + 1):
        response = assemble_finite_rotation_layered_shell4(model, value, committed)
        residual = load - response.internal_force
        norm = float(torch.linalg.vector_norm(residual[free]))
        if norm <= tolerance * scale:
            return FiniteRotationShell4Step(
                value, response.internal_force - load, response.stress,
                response.trial_state, iteration, norm
            )
        try:
            increment = torch.linalg.solve(
                response.tangent[free[:, None], free], residual[free]
            )
        except torch.linalg.LinAlgError as error:
            raise RuntimeError("finite-rotation layered Shell4 tangent is singular") from error
        value = value.index_add(0, free, increment)
    raise RuntimeError(
        f"finite-rotation layered Shell4 Newton failed after {max_iterations} iterations"
    )


def solve_finite_rotation_arc_path(
    model: LayeredShell4Model,
    reference_load: torch.Tensor,
    fixed_dofs: torch.Tensor,
    *,
    steps: int,
    step_size: float,
    load_scale: float,
    initial_state: LayeredShell4State | None = None,
    initial_displacement: torch.Tensor | None = None,
    initial_load_factor: float = 0.0,
    tolerance: float = 1e-7,
    max_iterations: int = 15,
    minimum_step: float = 1e-5,
    maximum_step: float | None = None,
) -> FiniteRotationArcResult:
    """Path-dependent Crisfield continuation with transactional J2 history.

    The material history is frozen for every predictor/corrector evaluation of
    one attempted step.  Only the trial state evaluated at an accepted
    equilibrium point is committed.  Retried and terminally rejected steps
    therefore cannot leak plastic strain into subsequent evaluations.
    """
    model.validate()
    if steps < 1 or step_size <= 0 or load_scale <= 0:
        raise ValueError("invalid arc-length controls")
    if reference_load.shape != (model.n_dofs,):
        raise ValueError("wrong shell load length")
    fixed = torch.tensor(sorted(set(int(i) for i in fixed_dofs)), dtype=torch.long,
                         device=model.nodes.device)
    if bool(torch.any(fixed < 0)) or (len(fixed) and int(fixed.max()) >= model.n_dofs):
        raise ValueError("fixed shell DOF out of range")
    mask = torch.ones(model.n_dofs, dtype=torch.bool, device=model.nodes.device)
    mask[fixed] = False
    free = torch.nonzero(mask).flatten()
    load_vector = reference_load.to(model.nodes)[free]
    state = LayeredShell4State.virgin(model) if initial_state is None else initial_state
    displacement = torch.zeros(model.n_dofs, dtype=model.nodes.dtype,
                               device=model.nodes.device)
    if initial_displacement is not None:
        if initial_displacement.shape != displacement.shape:
            raise ValueError("wrong initial displacement length")
        displacement = initial_displacement.to(model.nodes).clone()
    load_factor = float(initial_load_factor)
    maximum_step = step_size if maximum_step is None else maximum_step
    ds = step_size
    previous = None
    points = []

    def response(reduced: torch.Tensor, committed: LayeredShell4State):
        value = displacement.clone()
        value[free] = reduced
        evaluated = assemble_finite_rotation_layered_shell4(model, value, committed)
        return evaluated, evaluated.internal_force[free], evaluated.tangent[free[:, None], free]

    accepted = 0
    while accepted < steps:
        committed_u = displacement.clone()
        committed_reduced = committed_u[free].clone()
        committed_load = load_factor
        committed_state = state
        evaluated, internal, stiffness = response(committed_reduced, committed_state)
        try:
            direction_u = torch.linalg.solve(stiffness, load_vector)
        except torch.linalg.LinAlgError:
            return FiniteRotationArcResult(tuple(points), False, ds, state,
                                           displacement, load_factor)
        direction = torch.cat((direction_u, direction_u.new_tensor([1.])))
        metric = torch.cat((direction_u, direction_u.new_tensor([load_scale])))
        sign = 1.0
        if previous is not None:
            dot = (torch.dot(direction[:-1], previous[:-1])
                   + load_scale**2 * direction[-1] * previous[-1])
            sign = 1.0 if float(dot) >= 0 else -1.0
        increment_load = sign * ds / float(torch.linalg.vector_norm(metric))
        trial_u = committed_reduced + increment_load * direction_u
        trial_load = committed_load + increment_load
        ok = False
        last = float("inf")
        for iteration in range(1, max_iterations + 1):
            evaluated, internal, stiffness = response(trial_u, committed_state)
            residual = internal - trial_load * load_vector
            Du = trial_u - committed_reduced
            Dl = trial_load - committed_load
            constraint = torch.dot(Du, Du) + (load_scale * Dl)**2 - ds**2
            last = float(torch.linalg.vector_norm(
                torch.cat((residual, constraint.reshape(1)))
            ))
            if (float(torch.linalg.vector_norm(residual)) <= tolerance
                    and abs(float(constraint)) <= tolerance * max(ds, 1.0)):
                ok = True
                break
            matrix = torch.zeros((len(free)+1, len(free)+1), dtype=model.nodes.dtype,
                                 device=model.nodes.device)
            matrix[:-1, :-1] = stiffness
            matrix[:-1, -1] = -load_vector
            matrix[-1, :-1] = 2 * Du
            matrix[-1, -1] = 2 * load_scale**2 * Dl
            rhs = -torch.cat((residual, constraint.reshape(1)))
            try:
                correction = torch.linalg.solve(matrix, rhs)
            except torch.linalg.LinAlgError:
                break
            trial_u = trial_u + correction[:-1]
            trial_load += float(correction[-1])
        if ok:
            displacement = committed_u.clone()
            displacement[free] = trial_u
            load_factor = trial_load
            # ``evaluated`` was computed from committed_state at the accepted
            # point; this is the sole history commit in an attempted step.
            state = evaluated.trial_state
            accepted += 1
            previous = torch.cat((trial_u-committed_reduced,
                                  trial_u.new_tensor([trial_load-committed_load])))
            points.append(FiniteRotationArcPoint(
                accepted, load_factor, displacement.clone(), state,
                evaluated.stress, iteration, last
            ))
            if iteration <= 4:
                ds = min(maximum_step, ds * 1.25)
            elif iteration > 8:
                ds = max(minimum_step, ds * .75)
        else:
            # displacement, factor and state still identify the last accepted
            # point; no trial material state is reachable outside this branch.
            ds *= .5
            if ds < minimum_step:
                return FiniteRotationArcResult(tuple(points), False, ds, state,
                                               displacement, load_factor)
    return FiniteRotationArcResult(tuple(points), True, ds, state,
                                   displacement, load_factor)
