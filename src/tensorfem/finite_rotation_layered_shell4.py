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

from dataclasses import asdict, dataclass, replace

import torch

from .corotational_shell import _proper_fit_rotation, _rotation_vector
from .layered_shell4_plasticity import (
    LayeredShell4Model,
    LayeredShell4State,
    element_response,
)
from .shell_consistent import rotation_matrix_from_vector


def _arc_metric_weights(model: LayeredShell4Model, free: torch.Tensor,
                        arc_metric: str) -> torch.Tensor:
    """Return squared-length weights for the continuation coordinates."""
    if arc_metric not in ("full", "structural", "dimensionally_scaled"):
        raise ValueError(
            "arc_metric must be 'full', 'structural', or 'dimensionally_scaled'"
        )
    metric = torch.ones(len(free) + 1, dtype=model.nodes.dtype,
                        device=model.nodes.device)
    if arc_metric in ("structural", "dimensionally_scaled"):
        metric[:-1][(free % 6) == 5] = 0.0
    if arc_metric == "dimensionally_scaled":
        # Integrating |u + z x theta|^2 through a symmetric shell thickness
        # gives integral(z^2)/integral(1) = t^2/12.
        metric[:-1][((free % 6) == 3) | ((free % 6) == 4)] = (
            float(model.thickness) ** 2 / 12.0
        )
    return metric


@dataclass(frozen=True)
class FiniteRotationShell4Response:
    internal_force: torch.Tensor
    tangent: torch.Tensor | None
    stress: torch.Tensor
    trial_state: LayeredShell4State


@dataclass(frozen=True)
class SparseFiniteRotationShell4Response:
    """Shell response whose tangent is a reduced sparse COO matrix.

    ``active_dofs`` records the global DOFs represented by the rows and
    columns.  The force and stress remain full-sized/full-mesh quantities so
    callers cannot accidentally lose reactions or material evidence.
    """
    internal_force: torch.Tensor
    tangent: torch.Tensor
    active_dofs: torch.Tensor
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


def _kinematic_mapping(reference: torch.Tensor, dofs: torch.Tensor) -> torch.Tensor:
    """Exact AD Jacobian of the corotated deformation map.

    This replaces 48 SVD/frame evaluations per force evaluation.  Material
    history is not involved in this derivative, and ``create_graph=False``
    ensures that no graph or mutable trial state escapes the call.
    """
    with torch.enable_grad():
        independent = dofs.detach().requires_grad_(True)
        result = torch.autograd.functional.jacobian(
            lambda value: _corotated_deformation(reference, value),
            independent,
            create_graph=False,
            vectorize=True,
        ).detach()
    # A perfectly planar patch makes the covariance rank deficient.  Its polar
    # rotation is well-defined, but the generic SVD backward may return NaNs at
    # the repeated/zero singular value.  Retain the verified centred difference
    # for precisely that exceptional geometry.
    if not bool(torch.all(torch.isfinite(result))):
        return _jacobian(lambda value: _corotated_deformation(reference, value), dofs)
    return result


def _element_force(
    model: LayeredShell4Model,
    element: int,
    dofs: torch.Tensor,
    committed,
):
    conn = model.elements[element].long()
    reference = model.nodes[conn]
    deformation = _corotated_deformation(reference, dofs)
    mapping = _kinematic_mapping(reference, dofs)
    local_force, _, stress, trial = element_response(
        model, element, deformation, committed, compute_tangent=False
    )
    return mapping.T @ local_force, stress, trial


def _mapped_force_without_jacobian(
    reference: torch.Tensor,
    dofs: torch.Tensor,
    local_force: torch.Tensor,
) -> torch.Tensor:
    """Apply ``J(q).T`` without material replay or forming the 24x24 Jacobian.

    A residual-only evaluation is the dominant operation in matrix-free
    continuation.  Forming the complete kinematic Jacobian there computes 24
    columns although only its transpose product with the already integrated
    section force is required.  One reverse-mode scalar gradient evaluates the
    identical virtual-work product.  ``local_force`` is detached deliberately:
    its constitutive derivative belongs to a tangent action, not to the force
    value itself.
    """
    with torch.enable_grad():
        independent = dofs.detach().requires_grad_(True)
        virtual_work = torch.dot(
            local_force.detach(),
            _corotated_deformation(reference, independent),
        )
        result = torch.autograd.grad(
            virtual_work, independent, create_graph=False,
        )[0].detach()
    # As for ``_kinematic_mapping``, an exactly planar facet exposes an
    # undefined generic SVD derivative at its repeated/zero singular value.
    # The centred fallback is finite and already covered by objectivity tests.
    if not bool(torch.all(torch.isfinite(result))):
        return _kinematic_mapping(reference, dofs).T @ local_force
    return result


def _chain_rule_tangent(
    reference: torch.Tensor,
    dofs: torch.Tensor,
    deformation: torch.Tensor,
    mapping: torch.Tensor,
    local_force: torch.Tensor,
    local_tangent: torch.Tensor,
    *,
    relative_step: float,
) -> torch.Tensor:
    """Linearize ``J(q).T f(d(q))`` without reintegrating the material.

    ``local_tangent`` is the condensed plane-stress algorithmic tangent of the
    accepted return-map branch.  A contracted Hessian differentiates only the
    corotational frame.  Thus each element performs one layered material
    integration instead of 48, while retaining both ``J.T K J`` and the frame
    (initial-stress/geometric) contribution ``d(J.T)/dq f``.  A centred,
    kinematics-only fallback covers the rank-deficient planar-SVD derivative.
    """
    # The geometric part is one reverse-over-reverse Hessian contraction,
    # rather than 48 Jacobian constructions.  Detaching the already integrated
    # force is intentional: constitutive variation belongs to J.T K J below.
    with torch.enable_grad():
        independent = dofs.detach().requires_grad_(True)
        fixed_force = local_force.detach()
        geometric = torch.autograd.functional.hessian(
            lambda value: torch.dot(
                fixed_force, _corotated_deformation(reference, value)
            ),
            independent,
            create_graph=False,
            vectorize=True,
        ).detach()
    if bool(torch.all(torch.isfinite(geometric))):
        return mapping.T @ local_tangent @ mapping + geometric

    # A rigorously planar reference has a zero singular value, for which the
    # generic SVD second derivative is undefined although the polar rotation
    # itself is unique.  Keep a fail-safe kinematics-only difference there.
    columns = []
    for j in range(dofs.numel()):
        h = relative_step * max(1.0, abs(float(dofs[j])))
        delta = torch.zeros_like(dofs)
        delta[j] = h
        plus_q = dofs + delta
        minus_q = dofs - delta
        plus_d = _corotated_deformation(reference, plus_q)
        minus_d = _corotated_deformation(reference, minus_q)
        plus_j = _kinematic_mapping(reference, plus_q)
        minus_j = _kinematic_mapping(reference, minus_q)
        plus_force = plus_j.T @ (
            local_force + local_tangent @ (plus_d - deformation)
        )
        minus_force = minus_j.T @ (
            local_force + local_tangent @ (minus_d - deformation)
        )
        columns.append((plus_force - minus_force) / (2 * h))
    return torch.stack(columns, 1)


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
    conn = model.elements[element].long()
    reference = model.nodes[conn]
    deformation = _corotated_deformation(reference, dofs)
    local_force, local_tangent, stress, trial = element_response(
        model, element, deformation, committed, compute_tangent=tangent
    )
    if tangent:
        mapping = _kinematic_mapping(reference, dofs)
        force = mapping.T @ local_force
    else:
        # Matrix-free continuation needs only J.T@f.  Avoid material replay and
        # avoid constructing the full element Jacobian for this force-only path.
        mapping = None
        force = _mapped_force_without_jacobian(reference, dofs, local_force)
    stiffness = None
    if tangent:
        stiffness = _chain_rule_tangent(
            reference, dofs, deformation, mapping, local_force, local_tangent,
            relative_step=tangent_step,
        )
        # Do not symmetrize: active-set transitions and numerical local
        # condensation can legitimately leave a small algorithmic skew.
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


def assemble_finite_rotation_layered_shell4_sparse(
    model: LayeredShell4Model,
    displacement: torch.Tensor,
    committed: LayeredShell4State,
    *,
    active_dofs: torch.Tensor | None = None,
) -> SparseFiniteRotationShell4Response:
    """Assemble the exact element tangent directly into reduced sparse COO.

    This evaluates the same element residual and algorithmic tangent as the
    dense assembler.  Entries attached to prescribed DOFs are discarded before
    allocation, avoiding both the full ``ndof**2`` matrix and dense reduction.
    Duplicate element contributions are summed by ``coalesce``.
    """
    model.validate()
    if displacement.shape != (model.n_dofs,):
        raise ValueError("wrong displacement vector length")
    if active_dofs is None:
        active = torch.arange(model.n_dofs, device=model.nodes.device)
    else:
        active = active_dofs.to(device=model.nodes.device, dtype=torch.long)
        if active.ndim != 1 or len(torch.unique(active)) != len(active):
            raise ValueError("active_dofs must be a one-dimensional unique set")
        if len(active) and (bool(torch.any(active < 0)) or int(active.max()) >= model.n_dofs):
            raise ValueError("active shell DOF out of range")
    global_to_reduced = torch.full(
        (model.n_dofs,), -1, dtype=torch.long, device=model.nodes.device,
    )
    global_to_reduced[active] = torch.arange(len(active), device=model.nodes.device)
    internal = torch.zeros_like(displacement)
    row_parts, column_parts, value_parts = [], [], []
    stresses, states = [], []
    for element, conn in enumerate(model.elements):
        ids = _element_dofs(conn)
        force, local_k, stress, state = finite_rotation_element_response(
            model, element, displacement[ids], committed.points[element], tangent=True
        )
        internal.index_add_(0, ids, force)
        reduced = global_to_reduced[ids]
        retained = reduced >= 0
        local_ids = torch.nonzero(retained).flatten()
        if local_ids.numel():
            mapped = reduced[local_ids]
            row_parts.append(mapped[:, None].expand(-1, len(mapped)).reshape(-1))
            column_parts.append(mapped[None, :].expand(len(mapped), -1).reshape(-1))
            value_parts.append(local_k[local_ids[:, None], local_ids].reshape(-1))
        stresses.append(stress)
        states.append(state)
    if value_parts:
        indices = torch.stack((torch.cat(row_parts), torch.cat(column_parts)))
        values = torch.cat(value_parts)
    else:
        indices = torch.empty((2, 0), dtype=torch.long, device=model.nodes.device)
        values = torch.empty(0, dtype=model.nodes.dtype, device=model.nodes.device)
    tangent = torch.sparse_coo_tensor(
        indices, values, (len(active), len(active)),
        dtype=model.nodes.dtype, device=model.nodes.device,
        check_invariants=True,
    ).coalesce()
    return SparseFiniteRotationShell4Response(
        internal, tangent, active, torch.stack(stresses),
        LayeredShell4State(tuple(states)),
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
    initial_equilibrium_iterations: int = 0
    initial_equilibrium_residual_norm: float = float("inf")
    initial_equilibrium_relative_norm: float = float("inf")
    relaxed_initial_displacement: torch.Tensor | None = None
    relaxed_initial_state: LayeredShell4State | None = None
    previous_increment: torch.Tensor | None = None


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
    linear_solver: str = "dense",
    diagnostics: list[dict[str, object]] | None = None,
) -> FiniteRotationShell4Step:
    """Full Newton increment; state is returned only after convergence."""
    fixed = fixed_dofs.to(device=model.nodes.device, dtype=torch.long)
    mask = torch.ones(model.n_dofs, dtype=torch.bool, device=model.nodes.device)
    mask[fixed] = False
    free = torch.nonzero(mask).flatten()
    load = external_force.to(model.nodes) * load_factor
    value = displacement.clone()
    scale = max(float(torch.linalg.vector_norm(load[free])), 1.0)
    if linear_solver not in ("dense", "superlu"):
        raise ValueError("linear_solver must be 'dense' or 'superlu'")
    if linear_solver == "superlu" and model.nodes.device.type != "cpu":
        raise ValueError("SuperLU Shell4 paths require a CPU model")
    for iteration in range(1, max_iterations + 1):
        response = (assemble_finite_rotation_layered_shell4(model, value, committed)
                    if linear_solver == "dense" else
                    assemble_finite_rotation_layered_shell4_sparse(
                        model, value, committed, active_dofs=free))
        residual = load - response.internal_force
        norm = float(torch.linalg.vector_norm(residual[free]))
        if norm <= tolerance * scale:
            return FiniteRotationShell4Step(
                value, response.internal_force - load, response.stress,
                response.trial_state, iteration, norm
            )
        try:
            if linear_solver == "dense":
                increment = torch.linalg.solve(
                    response.tangent[free[:, None], free], residual[free]
                )
            else:
                from .sparse_direct import SparseLinearSolver
                factor = SparseLinearSolver("superlu").factorize(response.tangent)
                increment = factor.solve(residual[free])
                if diagnostics is not None:
                    diagnostics.append({"phase": "newton_linear_solve", "iteration": iteration,
                                        **asdict(factor.diagnostics)})
        except (torch.linalg.LinAlgError, RuntimeError) as error:
            if error.__class__.__name__ == "SparseBackendUnavailable":
                raise
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
    initial_previous_increment: torch.Tensor | None = None,
    tolerance: float = 1e-7,
    max_iterations: int = 15,
    minimum_step: float = 1e-5,
    maximum_step: float | None = None,
    diagnostics: list[dict[str, object]] | None = None,
    branch_switch: str | None = None,
    branch_mode_fraction: float = 0.25,
    branch_sign: int = 1,
    critical_eigenvalue_ratio: float = 1e-6,
    augmented_scaling: str = "legacy",
    arc_metric: str = "full",
    linear_solver: str = "dense",
    sparse_drilling_regularization: float = 1e-12,
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
    if branch_switch not in (None, "critical_mode"):
        raise ValueError("branch_switch must be None or 'critical_mode'")
    if not 0.0 < branch_mode_fraction <= 1.0 or branch_sign not in (-1, 1):
        raise ValueError("invalid branch-switch controls")
    if critical_eigenvalue_ratio <= 0.0:
        raise ValueError("critical_eigenvalue_ratio must be positive")
    if augmented_scaling not in ("legacy", "normalized"):
        raise ValueError("augmented_scaling must be 'legacy' or 'normalized'")
    if linear_solver not in ("dense", "superlu"):
        raise ValueError("linear_solver must be 'dense' or 'superlu'")
    if linear_solver == "superlu" and model.nodes.device.type != "cpu":
        raise ValueError("SuperLU Shell4 paths require a CPU model")
    if linear_solver == "superlu" and branch_switch is not None:
        raise ValueError("critical-mode branch switching is not available with SuperLU")
    if sparse_drilling_regularization < 0.0:
        raise ValueError("sparse_drilling_regularization must be non-negative")
    fixed = torch.tensor(sorted(set(int(i) for i in fixed_dofs)), dtype=torch.long,
                         device=model.nodes.device)
    if bool(torch.any(fixed < 0)) or (len(fixed) and int(fixed.max()) >= model.n_dofs):
        raise ValueError("fixed shell DOF out of range")
    mask = torch.ones(model.n_dofs, dtype=torch.bool, device=model.nodes.device)
    mask[fixed] = False
    free = torch.nonzero(mask).flatten()
    metric = _arc_metric_weights(model, free, arc_metric)
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
    if initial_previous_increment is not None:
        expected = len(free) + 1
        if initial_previous_increment.shape != (expected,):
            raise ValueError("wrong previous arc increment length")
        previous = initial_previous_increment.to(model.nodes).clone()
    points = []

    def linear_solve(matrix: torch.Tensor, rhs: torch.Tensor,
                     phase: str = "linear_solve") -> torch.Tensor:
        """Minimum-norm solve tolerating physically inactive drilling gauges."""
        if linear_solver == "superlu":
            from .sparse_direct import SparseLinearSolver
            # Free drilling gauges make the shell tangent rank deficient.  A
            # sparse LU has no minimum-norm mode analogous to dense SVD, so
            # add an explicitly reported, scale-relative gauge penalty.  It
            # acts only on theta-z rows and never on the bordered load row.
            matrix = matrix.coalesce()
            gauge = torch.nonzero((free % 6) == 5).flatten()
            regularization = 0.0
            if len(gauge) and sparse_drilling_regularization:
                scale = max(float(matrix.values().abs().max()), 1.0)
                regularization = sparse_drilling_regularization * scale
                indices = torch.stack((gauge, gauge))
                penalty = torch.sparse_coo_tensor(
                    indices, matrix.values().new_full((len(gauge),), regularization),
                    matrix.shape, check_invariants=True,
                )
                matrix = (matrix + penalty).coalesce()
            factor = SparseLinearSolver("superlu").factorize(matrix)
            answer = factor.solve(rhs)
            if diagnostics is not None:
                diagnostics.append({"phase": phase,
                                    "drilling_regularization": regularization,
                                    **asdict(factor.diagnostics)})
            return answer
        # Small benchmark systems retain the explicit rank/conditioning gate:
        # their free drilling gauges can admit a low-residual but non-minimum-
        # norm pivoted solution, which is a poor Newton direction.
        if matrix.shape[0] <= 200:
            try:
                condition = torch.linalg.cond(matrix)
                if bool(torch.isfinite(condition)) and float(condition) < 1e12:
                    return torch.linalg.solve(matrix, rhs)
            except torch.linalg.LinAlgError:
                pass
            return torch.linalg.lstsq(matrix, rhs, driver="gelsd").solution
        try:
            # A full SVD condition estimate costs as much as (and for larger
            # panels materially more than) the solve itself.  Try the pivoted
            # dense solve first and validate its backward error; use the SVD
            # minimum-norm path only for an actually singular/unstable system.
            answer = torch.linalg.solve(matrix, rhs)
            defect = matrix @ answer - rhs
            scale = (float(torch.linalg.vector_norm(matrix))
                     * float(torch.linalg.vector_norm(answer))
                     + float(torch.linalg.vector_norm(rhs)))
            relative = float(torch.linalg.vector_norm(defect)) / max(scale, 1.0)
            if bool(torch.all(torch.isfinite(answer))) and relative <= 1e-10:
                return answer
        except torch.linalg.LinAlgError:
            pass
        return torch.linalg.lstsq(matrix, rhs, driver="gelsd").solution

    def response(reduced: torch.Tensor, committed: LayeredShell4State):
        value = displacement.clone()
        value[free] = reduced
        if linear_solver == "superlu":
            evaluated = assemble_finite_rotation_layered_shell4_sparse(
                model, value, committed, active_dofs=free)
            stiffness = evaluated.tangent
        else:
            evaluated = assemble_finite_rotation_layered_shell4(model, value, committed)
            stiffness = evaluated.tangent[free[:, None], free]
        return evaluated, evaluated.internal_force[free], stiffness

    def augmented_matrix(stiffness: torch.Tensor, Du: torch.Tensor, Dp: float,
                         residual_scale: float, ds: float) -> torch.Tensor:
        """Build the bordered Newton matrix without densifying a sparse tangent."""
        n = len(free)
        if linear_solver == "dense":
            matrix = torch.zeros((n+1, n+1), dtype=model.nodes.dtype,
                                 device=model.nodes.device)
            matrix[:-1, :-1] = stiffness
            matrix[:-1, -1] = -load_vector / load_scale
            matrix[-1, :-1] = 2 * metric[:-1] * Du
            matrix[-1, -1] = 2 * metric[-1] * Dp
            return matrix
        stiffness = stiffness.coalesce()
        rows = [stiffness.indices()[0], torch.arange(n, device=free.device),
                torch.full((n,), n, dtype=torch.long, device=free.device),
                torch.tensor([n], dtype=torch.long, device=free.device)]
        cols = [stiffness.indices()[1],
                torch.full((n,), n, dtype=torch.long, device=free.device),
                torch.arange(n, device=free.device),
                torch.tensor([n], dtype=torch.long, device=free.device)]
        values = [stiffness.values(), -load_vector / load_scale,
                  2 * metric[:-1] * Du, Du.new_tensor([2 * metric[-1] * Dp])]
        matrix = torch.sparse_coo_tensor(
            torch.stack((torch.cat(rows), torch.cat(cols))), torch.cat(values),
            (n+1, n+1), dtype=model.nodes.dtype, device=model.nodes.device,
            check_invariants=True,
        ).coalesce()
        if augmented_scaling == "normalized":
            indices = matrix.indices()
            scales = torch.where(indices[0] < n,
                                 matrix.values().new_tensor(residual_scale),
                                 matrix.values().new_tensor(max(ds**2, torch.finfo(model.nodes.dtype).tiny)))
            matrix = torch.sparse_coo_tensor(indices, matrix.values()/scales,
                                             matrix.shape,
                                             check_invariants=True).coalesce()
        return matrix

    # Imported residual stresses need not be nodally self-equilibrated even if
    # their section resultant is zero.  Arc continuation requires an actual
    # equilibrium base point, so relax it at the prescribed initial load before
    # constructing a predictor.  Every iterate starts from the same committed
    # history and only the converged trial state is committed.
    initial_trace = {"phase": "initial_equilibrium", "iterations": [], "reason": None}
    initial_base = displacement[free].clone()
    initial_state_base = state
    initial_force_scale = max(1.0, maximum_step / load_scale,
                              abs(load_factor) * float(torch.linalg.vector_norm(load_vector)))
    for initial_iteration in range(1, max_iterations + 1):
        initial_eval, initial_internal, initial_stiffness = response(
            initial_base, initial_state_base
        )
        initial_residual = initial_internal - load_factor * load_vector
        initial_norm = float(torch.linalg.vector_norm(initial_residual))
        force_scale = initial_force_scale
        initial_trace["iterations"].append({"iteration": initial_iteration,
                                            "residual_norm": initial_norm,
                                            "residual_relative": initial_norm / force_scale})
        if initial_norm <= tolerance * force_scale:
            displacement[free] = initial_base
            state = initial_eval.trial_state
            initial_trace["reason"] = "accepted"
            break
        try:
            correction = linear_solve(initial_stiffness, -initial_residual,
                                      "initial_equilibrium_linear_solve")
        except RuntimeError as error:
            initial_trace["reason"] = "linear_solver_failed"
            initial_trace["linear_solver_error"] = str(error)
            break
        baseline = initial_norm
        relaxed = False
        for backtrack in range(9):
            fraction = .5**backtrack
            candidate = initial_base + fraction * correction
            candidate_eval, candidate_internal, _ = response(candidate, initial_state_base)
            candidate_norm = float(torch.linalg.vector_norm(
                candidate_internal - load_factor * load_vector
            ))
            if candidate_norm < baseline:
                initial_base = candidate
                relaxed = True
                break
        if not relaxed:
            initial_trace["reason"] = "line_search_failed"
            break
    else:
        initial_trace["reason"] = "maximum_iterations"
    if diagnostics is not None:
        diagnostics.append(initial_trace)
    initial_iterations = len(initial_trace["iterations"])
    initial_residual_norm = (float(initial_trace["iterations"][-1]["residual_norm"])
                             if initial_iterations else float("inf"))
    initial_relative_norm = (float(initial_trace["iterations"][-1]["residual_relative"])
                             if initial_iterations else float("inf"))
    if initial_trace["reason"] != "accepted":
        return FiniteRotationArcResult(
            tuple(points), False, ds, state, displacement, load_factor,
            initial_iterations, initial_residual_norm, initial_relative_norm,
            displacement.clone(), state, previous,
        )
    relaxed_initial_displacement = displacement.clone()
    relaxed_initial_state = state

    def arc_result(converged: bool) -> FiniteRotationArcResult:
        return FiniteRotationArcResult(
            tuple(points), converged, ds, state, displacement, load_factor,
            initial_iterations, initial_residual_norm, initial_relative_norm,
            relaxed_initial_displacement, relaxed_initial_state, previous,
        )

    accepted = 0
    attempt = 0
    while accepted < steps:
        attempt += 1
        committed_u = displacement.clone()
        committed_reduced = committed_u[free].clone()
        committed_load = load_factor
        committed_state = state
        evaluated, internal, stiffness = response(committed_reduced, committed_state)
        try:
            # Continue in the dimensioned arc coordinate p=load_scale*lambda.
            # Solving directly for lambda leaves the bordered column in N and
            # its constraint derivative in m^2/N, which is catastrophically
            # ill-conditioned for industrial MN loads.
            direction_u = linear_solve(stiffness, load_vector / load_scale,
                                       "predictor_linear_solve")
        except (torch.linalg.LinAlgError, RuntimeError) as error:
            if diagnostics is not None:
                diagnostics.append({"attempt": attempt, "step": accepted + 1,
                                    "ds": ds, "reason": "predictor_singular",
                                    "linear_solver_error": str(error)})
            return arc_result(False)
        direction = torch.cat((direction_u, direction_u.new_tensor([1.])))
        sign = 1.0
        if previous is not None:
            dot = torch.dot(metric * direction, previous)
            sign = 1.0 if float(dot) >= 0 else -1.0
        mode_diagnostic = None
        direction_norm = torch.sqrt(torch.dot(metric * direction, direction))
        predictor = sign * direction / direction_norm
        if branch_switch == "critical_mode":
            from .branch_switch import (
                perturbed_arc_predictor, smallest_eligible_symmetric_mode,
            )
            # Shell drilling gauges are intentionally weak/zero and are not
            # structural bifurcation modes. Require translational energy.
            eligible = (free % 6) < 3
            critical = smallest_eligible_symmetric_mode(
                stiffness, eligible=eligible, minimum_eligible_fraction=.5,
            )
            activated = critical.relative_eigenvalue <= critical_eigenvalue_ratio
            mode_diagnostic = {
                "eigenvalue": critical.eigenvalue,
                "relative_eigenvalue": critical.relative_eigenvalue,
                "eligible_energy_fraction": critical.eligible_energy_fraction,
                "activated": activated,
                "sign": branch_sign,
            }
            if activated:
                mode = torch.cat((critical.vector, critical.vector.new_zeros(1)))
                predictor = perturbed_arc_predictor(
                    predictor, mode, mode_fraction=branch_mode_fraction,
                    sign=branch_sign,
                )
        increment = ds * predictor
        trial_u = committed_reduced + increment[:-1]
        trial_p = load_scale * committed_load + float(increment[-1])
        trial_load = trial_p / load_scale
        trace = {"attempt": attempt, "step": accepted + 1, "ds": ds,
                 "predictor_norm": float(torch.linalg.vector_norm(direction_u)),
                 "iterations": [], "reason": None,
                 "critical_mode": mode_diagnostic}
        ok = False
        last = float("inf")
        for iteration in range(1, max_iterations + 1):
            evaluated, internal, stiffness = response(trial_u, committed_state)
            residual = internal - trial_load * load_vector
            Du = trial_u - committed_reduced
            Dp = trial_p - load_scale * committed_load
            full_increment = torch.cat((Du, Du.new_tensor([Dp])))
            constraint = torch.dot(metric * full_increment, full_increment) - ds**2
            residual_norm = float(torch.linalg.vector_norm(residual))
            residual_scale = max(float(torch.linalg.vector_norm(internal)),
                                 abs(trial_load) * float(torch.linalg.vector_norm(load_vector)),
                                 1.0)
            constraint_relative = abs(float(constraint)) / max(ds**2, torch.finfo(model.nodes.dtype).eps)
            last = float(torch.linalg.vector_norm(
                torch.cat((residual / residual_scale,
                           (constraint / max(ds**2, torch.finfo(model.nodes.dtype).eps)).reshape(1)))
            ))
            trace["iterations"].append({"iteration": iteration,
                                        "residual_norm": residual_norm,
                                        "residual_relative": residual_norm / residual_scale,
                                        "constraint": float(constraint),
                                        "constraint_relative": constraint_relative})
            if (residual_norm <= tolerance * residual_scale
                    and constraint_relative <= tolerance):
                ok = True
                trace["reason"] = "accepted"
                break
            matrix = augmented_matrix(stiffness, Du, Dp, residual_scale, ds)
            rhs = -torch.cat((residual, constraint.reshape(1)))
            if augmented_scaling == "normalized":
                # Row equilibration does not change the Newton equations, but
                # avoids mixing O(GN/m) equilibrium rows with an O(m) arc row.
                # This is essential once ds has been reduced near a turning
                # point; otherwise the linear solve can satisfy equilibrium
                # while effectively dropping the constraint equation.
                rhs[:-1] /= residual_scale
                arc_scale = max(ds**2, torch.finfo(model.nodes.dtype).tiny)
                rhs[-1] /= arc_scale
                if linear_solver == "dense":
                    matrix[:-1] /= residual_scale
                    matrix[-1] /= arc_scale
            try:
                correction = linear_solve(matrix, rhs, "corrector_linear_solve")
            except (torch.linalg.LinAlgError, RuntimeError) as error:
                trace["reason"] = "corrector_singular"
                trace["linear_solver_error"] = str(error)
                break
            trial_u = trial_u + correction[:-1]
            trial_p += float(correction[-1])
            trial_load = trial_p / load_scale
        if diagnostics is not None:
            if trace["reason"] is None:
                trace["reason"] = "maximum_iterations"
            diagnostics.append(trace)
        if ok:
            displacement = committed_u.clone()
            displacement[free] = trial_u
            load_factor = trial_load
            # ``evaluated`` was computed from committed_state at the accepted
            # point; this is the sole history commit in an attempted step.
            state = evaluated.trial_state
            accepted += 1
            previous = torch.cat((trial_u-committed_reduced,
                                  trial_u.new_tensor([load_scale*(trial_load-committed_load)])))
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
                return arc_result(False)
    return arc_result(True)
