"""Transactional state path for the corotational multi-facet Shell4.

This is deliberately an elastic-section subset.  It supplies the state and
transaction boundary needed by a later return-mapping material, without
pretending that the present quadratic section law is elastoplastic.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from .arc_length import ArcLengthResult, general_shell_arc_problem, solve_arc_length
from .corotational_shell import _proper_fit_rotation, _rotation_vector
from .general_shell_nonlinear import GeneralShellMesh
from .shell_consistent import rotation_matrix_from_vector
from .spherical_shell import projected_shell4_stiffness


@dataclass(frozen=True)
class Shell4SectionState:
    """Committed one-point generalized elastic state of every facet."""

    generalized_deformation: torch.Tensor
    generalized_resultant: torch.Tensor
    energy: torch.Tensor


@dataclass(frozen=True)
class Shell4PathState:
    """Last accepted global and element state (never a rejected trial)."""

    dofs: torch.Tensor
    load_factor: float
    section: Shell4SectionState


@dataclass(frozen=True)
class Shell4StatePathResult:
    continuation: ArcLengthResult
    committed: Shell4PathState
    free_dofs: torch.Tensor


def shell_with_initial_imperfection(
    mesh: GeneralShellMesh, imperfection: torch.Tensor
) -> GeneralShellMesh:
    """Put a measured nodal imperfection into the reference geometry.

    The field is not applied as an equivalent load: zero displacement is the
    imperfect, stress-free configuration.  Residual stress is consequently
    outside this elastic subset.
    """
    mesh.validate()
    if imperfection.shape != mesh.nodes.shape:
        raise ValueError("imperfection must have the same shape as mesh.nodes")
    if imperfection.dtype != mesh.nodes.dtype or imperfection.device != mesh.nodes.device:
        raise ValueError("imperfection and nodes must share dtype and device")
    if not bool(torch.all(torch.isfinite(imperfection))):
        raise ValueError("imperfection must be finite")
    result = GeneralShellMesh(mesh.nodes + imperfection, mesh.elements, mesh.young,
                              mesh.poisson, mesh.thickness, mesh.drilling_factor)
    result.validate()
    # Force geometry checks now rather than deep in the first Newton iteration.
    for conn in result.elements:
        projected_shell4_stiffness(result.nodes[conn], result.young, result.poisson,
                                   result.thickness,
                                   drilling_factor=result.drilling_factor)
    return result


def _element_deformation(reference: torch.Tensor, local_dofs: torch.Tensor) -> torch.Tensor:
    q = local_dofs.reshape(4, 6)
    current = reference + q[:, :3]
    rotations = torch.stack([rotation_matrix_from_vector(v) for v in q[:, 3:]])
    frame = _proper_fit_rotation(reference, current)
    translations = ((current - current.mean(0)) @ frame
                    - (reference - reference.mean(0)))
    relative = frame.T.unsqueeze(0) @ rotations
    rotation_vectors = torch.stack([_rotation_vector(value) for value in relative])
    return torch.cat((translations, rotation_vectors), dim=1).reshape(-1)


def recover_elastic_section_state(mesh: GeneralShellMesh,
                                  dofs: torch.Tensor) -> Shell4SectionState:
    """Recover the exact generalized state used by the element energy."""
    mesh.validate()
    if dofs.shape != (6 * len(mesh.nodes),) or not bool(torch.all(torch.isfinite(dofs))):
        raise ValueError("invalid global displacement vector")
    deformation, resultant, energy = [], [], []
    for conn in mesh.elements:
        ids = torch.stack(tuple(6 * conn + k for k in range(6)), 1).reshape(-1)
        value = _element_deformation(mesh.nodes[conn], dofs[ids])
        stiffness = projected_shell4_stiffness(
            mesh.nodes[conn], mesh.young, mesh.poisson, mesh.thickness,
            drilling_factor=mesh.drilling_factor)
        force = stiffness @ value
        deformation.append(value)
        resultant.append(force)
        energy.append(.5 * torch.dot(value, force))
    return Shell4SectionState(torch.stack(deformation), torch.stack(resultant),
                              torch.stack(energy))


def solve_shell4_state_path(
    mesh: GeneralShellMesh, reference_load: torch.Tensor, fixed_dofs, *,
    steps: int, step_size: float, load_scale: float, tolerance: float = 1e-8,
    max_iterations: int = 15, minimum_step: float = 1e-6,
) -> Shell4StatePathResult:
    """Run Crisfield continuation and commit only the last accepted state.

    The continuation solver rolls every rejected trial back to its last
    accepted displacement/load pair.  Element state is recovered only from
    that accepted pair, so a failed corrector cannot contaminate it.
    """
    problem, free = general_shell_arc_problem(mesh, reference_load, fixed_dofs)
    result = solve_arc_length(
        problem, torch.zeros(len(free), dtype=reference_load.dtype,
                             device=reference_load.device),
        steps=steps, step_size=step_size, maximum_step=step_size,
        load_scale=load_scale, tolerance=tolerance,
        max_iterations=max_iterations, minimum_step=minimum_step)
    full = torch.zeros_like(reference_load)
    factor = 0.0
    if result.points:
        full[free] = result.points[-1].displacement
        factor = result.points[-1].load_factor
    committed = Shell4PathState(full, factor,
                                recover_elastic_section_state(mesh, full))
    return Shell4StatePathResult(result, committed, free)

