"""Transactional global FE equilibrium with low-order 3-D surface contact.

The implementation couples a linear structural tangent to deformable
node-to-TRI3 contact in the same global residual.  Candidate facets are
searched in the current configuration and their barycentric shape functions
receive equal and opposite reactions.  Coulomb history is trialled from an
immutable committed state at every Newton evaluation.

This is deliberately a node-quadrature surface-contact reference, not a
general mortar, self-contact, or continuous-collision solver.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .contact3d import (
    Contact3DState,
    Contact3DUpdate,
    initial_contact_state,
    project_point_to_facets,
    update_node_facet_contact,
)


@dataclass(frozen=True)
class NodeTrianglePair:
    """A slave quadrature node and its deformable TRI3 search surface."""

    slave: int
    master_faces: tuple[tuple[int, int, int], ...]
    weight: float = 1.0


@dataclass(frozen=True)
class GlobalSurfaceContactModel:
    reference_nodes: torch.Tensor
    stiffness: torch.Tensor
    pairs: tuple[NodeTrianglePair, ...]
    fixed_dofs: tuple[int, ...]
    normal_penalty: float
    tangential_penalty: float
    friction: float

    def __post_init__(self) -> None:
        x, k = self.reference_nodes, self.stiffness
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError("reference_nodes must have shape (n,3)")
        if k.shape != (3 * len(x), 3 * len(x)) or k.dtype != x.dtype:
            raise ValueError("stiffness must have shape (3*n,3*n) and matching dtype")
        if not bool(torch.allclose(k, k.T)):
            raise ValueError("stiffness must be symmetric")
        ndof = 3 * len(x)
        if (len(set(self.fixed_dofs)) != len(self.fixed_dofs)
                or any(d < 0 or d >= ndof for d in self.fixed_dofs)):
            raise ValueError("fixed_dofs must be unique valid indices")
        for pair in self.pairs:
            masters = [node for face in pair.master_faces for node in face]
            if (pair.slave < 0 or pair.slave >= len(x) or not pair.master_faces
                    or pair.slave in masters or any(i < 0 or i >= len(x) for i in masters)):
                raise ValueError("slave and TRI3 master nodes must be valid and disjoint")
            if any(len(set(face)) != 3 for face in pair.master_faces):
                raise ValueError("master facets must contain three distinct nodes")
            if pair.weight <= 0:
                raise ValueError("quadrature weight must be positive")
        if self.normal_penalty <= 0 or self.tangential_penalty <= 0 or self.friction < 0:
            raise ValueError("penalties must be positive and friction non-negative")


@dataclass(frozen=True)
class GlobalSurfaceContactState:
    histories: tuple[Contact3DState, ...]
    displacement: torch.Tensor


@dataclass(frozen=True)
class GlobalSurfaceAssembly:
    residual: torch.Tensor
    tangent: torch.Tensor
    contact_force: torch.Tensor
    trial_state: GlobalSurfaceContactState
    updates: tuple[Contact3DUpdate, ...]
    force_imbalance: torch.Tensor
    moment_imbalance: torch.Tensor
    dissipation_increment: torch.Tensor


@dataclass(frozen=True)
class GlobalSurfaceStep:
    displacement: torch.Tensor
    state: GlobalSurfaceContactState
    iterations: int
    residual_norm: float
    external: torch.Tensor


def _local_surface(pair: NodeTrianglePair, current: torch.Tensor):
    ids = tuple(sorted({node for face in pair.master_faces for node in face}))
    lookup = {node: local for local, node in enumerate(ids)}
    faces = torch.tensor([[lookup[node] for node in face] for face in pair.master_faces],
                         dtype=torch.long, device=current.device)
    return ids, current[list(ids)], faces


def initial_global_surface_state(
    model: GlobalSurfaceContactModel,
    displacement: torch.Tensor | None = None,
) -> GlobalSurfaceContactState:
    ndof = model.stiffness.shape[0]
    u = (torch.zeros(ndof, dtype=model.reference_nodes.dtype, device=model.reference_nodes.device)
         if displacement is None else displacement)
    if u.shape != (ndof,):
        raise ValueError("displacement has the wrong DOF count")
    current = model.reference_nodes + u.reshape(-1, 3)
    histories = []
    for pair in model.pairs:
        _, vertices, faces = _local_surface(pair, current)
        histories.append(initial_contact_state(current[pair.slave], vertices, faces))
    return GlobalSurfaceContactState(tuple(histories), u.clone())


def _contact_vector(
    model: GlobalSurfaceContactModel,
    displacement: torch.Tensor,
    committed: GlobalSurfaceContactState,
) -> tuple[torch.Tensor, GlobalSurfaceContactState, tuple[Contact3DUpdate, ...], torch.Tensor]:
    ndof = model.stiffness.shape[0]
    if displacement.shape != (ndof,) or committed.displacement.shape != (ndof,):
        raise ValueError("displacement has the wrong DOF count")
    if len(committed.histories) != len(model.pairs):
        raise ValueError("contact history count mismatch")
    current = model.reference_nodes + displacement.reshape(-1, 3)
    increment = (displacement - committed.displacement).reshape(-1, 3)
    force = torch.zeros_like(displacement)
    histories, updates = [], []
    dissipation = displacement.new_zeros(())
    for pair, old in zip(model.pairs, committed.histories):
        global_ids, vertices, faces = _local_surface(pair, current)
        projection = project_point_to_facets(current[pair.slave], vertices, faces)
        face = faces[projection.face]
        master_increment = torch.sum(
            projection.weights[:, None] * increment[list(global_ids)][face], dim=0)
        relative_increment = increment[pair.slave] - master_increment
        update = update_node_facet_contact(
            current[pair.slave], vertices, faces, old,
            normal_penalty=model.normal_penalty,
            tangential_penalty=model.tangential_penalty,
            friction=model.friction,
            relative_increment=relative_increment,
        )
        weight = displacement.new_tensor(pair.weight)
        sf = weight * update.slave_force
        force[3 * pair.slave:3 * pair.slave + 3] += sf
        for local, node in enumerate(global_ids):
            force[3 * node:3 * node + 3] += weight * update.master_forces[local]
        histories.append(update.state)
        updates.append(update)
        dissipation += weight * update.dissipation_increment
    trial = GlobalSurfaceContactState(tuple(histories), displacement.clone())
    return force, trial, tuple(updates), dissipation


def assemble_global_surface_contact(
    model: GlobalSurfaceContactModel,
    displacement: torch.Tensor,
    external: torch.Tensor,
    committed: GlobalSurfaceContactState,
    *,
    tangent: bool = True,
) -> GlobalSurfaceAssembly:
    """Assemble structural and contact forces into one FE residual."""
    if external.shape != displacement.shape:
        raise ValueError("external and displacement shapes must match")

    def residual_only(u: torch.Tensor) -> torch.Tensor:
        contact, _, _, _ = _contact_vector(model, u, committed)
        return model.stiffness @ u - external - contact

    contact, trial, updates, dissipation = _contact_vector(model, displacement, committed)
    residual = model.stiffness @ displacement - external - contact
    if tangent:
        u = displacement.detach().requires_grad_(True)
        matrix = torch.autograd.functional.jacobian(residual_only, u, create_graph=False)
    else:
        matrix = torch.empty((0, 0), dtype=displacement.dtype, device=displacement.device)
    nodal = contact.reshape(-1, 3)
    current = model.reference_nodes + displacement.reshape(-1, 3)
    moment = torch.sum(torch.linalg.cross(current, nodal), dim=0)
    return GlobalSurfaceAssembly(
        residual, matrix, contact, trial, updates, torch.sum(nodal, dim=0), moment,
        dissipation,
    )


def solve_global_surface_contact_path(
    model: GlobalSurfaceContactModel,
    loads: Sequence[torch.Tensor],
    *,
    initial_displacement: torch.Tensor | None = None,
    initial_state: GlobalSurfaceContactState | None = None,
    tolerance: float = 1e-9,
    max_iterations: int = 35,
    line_search_steps: int = 10,
) -> tuple[GlobalSurfaceStep, ...]:
    """Solve load increments while committing history only after convergence."""
    ndof = model.stiffness.shape[0]
    u = (torch.zeros(ndof, dtype=model.reference_nodes.dtype, device=model.reference_nodes.device)
         if initial_displacement is None else initial_displacement.clone())
    if u.shape != (ndof,):
        raise ValueError("initial_displacement has the wrong DOF count")
    committed = initial_global_surface_state(model, u) if initial_state is None else initial_state
    fixed = torch.tensor(model.fixed_dofs, dtype=torch.long, device=u.device)
    mask = torch.ones(ndof, dtype=torch.bool, device=u.device)
    mask[fixed] = False
    free = torch.nonzero(mask, as_tuple=False).flatten()
    if len(free) == 0:
        raise ValueError("model has no free degrees of freedom")
    steps: list[GlobalSurfaceStep] = []
    for load in loads:
        if load.shape != (ndof,):
            raise ValueError("every load must have shape (ndof,)")
        trial, converged = u.clone(), False
        norm = float("inf")
        for iteration in range(1, max_iterations + 1):
            assembly = assemble_global_surface_contact(model, trial, load, committed)
            residual = assembly.residual[free]
            norm = float(torch.linalg.vector_norm(residual))
            scale = max(1.0, float(torch.linalg.vector_norm(load[free])))
            if norm <= tolerance * scale:
                converged = True
                break
            delta = torch.linalg.solve(assembly.tangent[free][:, free], -residual)
            accepted = False
            for power in range(line_search_steps + 1):
                candidate = trial.clone()
                candidate[free] += (0.5 ** power) * delta
                candidate[fixed] = 0.0
                candidate_norm = torch.linalg.vector_norm(
                    assemble_global_surface_contact(
                        model, candidate, load, committed, tangent=False).residual[free])
                if float(candidate_norm) < norm:
                    trial, accepted = candidate, True
                    break
            if not accepted:
                break
        if not converged:
            raise RuntimeError(
                "global 3-D surface-contact Newton solve failed; committed state was not changed")
        final = assemble_global_surface_contact(model, trial, load, committed, tangent=False)
        committed, u = final.trial_state, trial
        steps.append(GlobalSurfaceStep(u.clone(), committed, iteration, norm, load.clone()))
    return tuple(steps)
