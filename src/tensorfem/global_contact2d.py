"""Transactional global FE equilibrium with 2-D finite-sliding contact.

This is an intentionally small reference implementation.  Linear structural
forces are combined with deformable node-to-polyline penalty contact in one
global residual.  The closest master segment is searched in the current
configuration at every Newton evaluation and equal/opposite contact forces are
assembled to slave and master degrees of freedom.  It is not a general 3-D
surface contact or mortar solver.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .finite_sliding_contact import (
    ContactUpdate, FrictionState, initial_friction_state,
    project_point_to_polyline, update_node_polyline_contact,
)


@dataclass(frozen=True)
class NodePolylinePair:
    """One slave node and an ordered deformable master polyline."""

    slave: int
    master_nodes: tuple[int, ...]


@dataclass(frozen=True)
class GlobalContactModel:
    reference_nodes: torch.Tensor
    stiffness: torch.Tensor
    pairs: tuple[NodePolylinePair, ...]
    fixed_dofs: tuple[int, ...]
    normal_penalty: float
    tangential_penalty: float
    friction: float

    def __post_init__(self) -> None:
        x, k = self.reference_nodes, self.stiffness
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("reference_nodes must have shape (n,2)")
        if k.shape != (2 * len(x), 2 * len(x)) or k.dtype != x.dtype:
            raise ValueError("stiffness must have shape (2*n,2*n) and matching dtype")
        if not bool(torch.allclose(k, k.T)):
            raise ValueError("stiffness must be symmetric")
        ndof = 2 * len(x)
        if len(set(self.fixed_dofs)) != len(self.fixed_dofs) or any(d < 0 or d >= ndof for d in self.fixed_dofs):
            raise ValueError("fixed_dofs must be unique valid indices")
        for pair in self.pairs:
            if pair.slave < 0 or pair.slave >= len(x) or len(pair.master_nodes) < 2:
                raise ValueError("invalid contact pair")
            if pair.slave in pair.master_nodes or any(i < 0 or i >= len(x) for i in pair.master_nodes):
                raise ValueError("slave and master nodes must be valid and disjoint")
        if self.normal_penalty <= 0 or self.tangential_penalty <= 0 or self.friction < 0:
            raise ValueError("penalties must be positive and friction non-negative")


@dataclass(frozen=True)
class GlobalContactState:
    histories: tuple[FrictionState, ...]
    displacement: torch.Tensor


@dataclass(frozen=True)
class ContactAssembly:
    residual: torch.Tensor
    tangent: torch.Tensor
    contact_force: torch.Tensor
    trial_state: GlobalContactState
    updates: tuple[ContactUpdate, ...]
    force_imbalance: torch.Tensor


@dataclass(frozen=True)
class GlobalContactStep:
    displacement: torch.Tensor
    state: GlobalContactState
    iterations: int
    residual_norm: float
    external: torch.Tensor


def initial_global_contact_state(model: GlobalContactModel,
                                 displacement: torch.Tensor | None = None) -> GlobalContactState:
    u = (torch.zeros(model.stiffness.shape[0], dtype=model.reference_nodes.dtype,
                     device=model.reference_nodes.device)
         if displacement is None else displacement)
    if u.shape != (model.stiffness.shape[0],):
        raise ValueError("displacement has the wrong DOF count")
    current = model.reference_nodes + u.reshape(-1, 2)
    states = []
    for pair in model.pairs:
        states.append(initial_friction_state(current[pair.slave],
                                             current[list(pair.master_nodes)]))
    return GlobalContactState(tuple(states), u.clone())


def _contact_vector(model: GlobalContactModel, displacement: torch.Tensor,
                    committed: GlobalContactState) -> tuple[torch.Tensor, GlobalContactState, tuple[ContactUpdate, ...]]:
    if displacement.shape != (model.stiffness.shape[0],):
        raise ValueError("displacement has the wrong DOF count")
    if len(committed.histories) != len(model.pairs):
        raise ValueError("contact history count mismatch")
    if committed.displacement.shape != displacement.shape:
        raise ValueError("committed displacement has the wrong DOF count")
    current = model.reference_nodes + displacement.reshape(-1, 2)
    increment = (displacement - committed.displacement).reshape(-1, 2)
    force = torch.zeros_like(displacement)
    trial, updates = [], []
    for pair, history in zip(model.pairs, committed.histories):
        master_ids = list(pair.master_nodes)
        # Relative material motion removes rigid translation of both sides.
        # Current interpolation is sufficient for this low-order incremental
        # formulation and remains objective for common rigid motion.
        projection = project_point_to_polyline(current[pair.slave], current[master_ids])
        j, xi = projection.segment, projection.coordinate
        master_increment = ((1.0 - xi) * increment[pair.master_nodes[j]] +
                            xi * increment[pair.master_nodes[j + 1]])
        ds = torch.dot(increment[pair.slave] - master_increment, projection.tangent)
        update = update_node_polyline_contact(
            current[pair.slave], current[master_ids], history,
            normal_penalty=model.normal_penalty,
            tangential_penalty=model.tangential_penalty,
            friction=model.friction,
            relative_tangential_increment=ds,
        )
        sf = update.traction
        force[2 * pair.slave:2 * pair.slave + 2] += sf
        j, xi = update.projection.segment, update.projection.coordinate
        a, b = pair.master_nodes[j], pair.master_nodes[j + 1]
        force[2 * a:2 * a + 2] -= (1.0 - xi) * sf
        force[2 * b:2 * b + 2] -= xi * sf
        trial.append(update.state); updates.append(update)
    return force, GlobalContactState(tuple(trial), displacement.clone()), tuple(updates)


def assemble_global_contact(model: GlobalContactModel, displacement: torch.Tensor,
                            external: torch.Tensor, committed: GlobalContactState,
                            *, tangent: bool = True) -> ContactAssembly:
    """Assemble structural and contact terms in the same global residual.

    ``committed`` is never mutated.  All Newton calls in an increment must pass
    the same committed state and only retain ``trial_state`` after convergence.
    """
    if external.shape != displacement.shape:
        raise ValueError("external and displacement shapes must match")

    def residual_only(u: torch.Tensor) -> torch.Tensor:
        fc, _, _ = _contact_vector(model, u, committed)
        return model.stiffness @ u - external - fc

    contact, trial, updates = _contact_vector(model, displacement, committed)
    residual = model.stiffness @ displacement - external - contact
    if tangent:
        u = displacement.detach().requires_grad_(True)
        matrix = torch.autograd.functional.jacobian(residual_only, u, create_graph=False)
    else:
        matrix = torch.empty((0, 0), dtype=displacement.dtype, device=displacement.device)
    nodal_contact = contact.reshape(-1, 2)
    return ContactAssembly(residual, matrix, contact, trial, updates,
                           torch.sum(nodal_contact, dim=0))


def solve_global_contact_path(model: GlobalContactModel, loads: Sequence[torch.Tensor],
                              *, initial_displacement: torch.Tensor | None = None,
                              initial_state: GlobalContactState | None = None,
                              tolerance: float = 1e-9, max_iterations: int = 35,
                              line_search_steps: int = 10) -> tuple[GlobalContactStep, ...]:
    """Solve arbitrary incremental loads with transactional contact histories."""
    ndof = model.stiffness.shape[0]
    u = (torch.zeros(ndof, dtype=model.reference_nodes.dtype, device=model.reference_nodes.device)
         if initial_displacement is None else initial_displacement.clone())
    committed = initial_global_contact_state(model, u) if initial_state is None else initial_state
    fixed = torch.tensor(model.fixed_dofs, dtype=torch.long, device=u.device)
    mask = torch.ones(ndof, dtype=torch.bool, device=u.device); mask[fixed] = False
    free = torch.nonzero(mask, as_tuple=False).flatten()
    if len(free) == 0:
        raise ValueError("model has no free degrees of freedom")
    steps: list[GlobalContactStep] = []
    for load in loads:
        if load.shape != (ndof,):
            raise ValueError("every load must have shape (ndof,)")
        trial = u.clone(); converged = False
        for iteration in range(1, max_iterations + 1):
            assembly = assemble_global_contact(model, trial, load, committed)
            rf = assembly.residual[free]
            norm = float(torch.linalg.vector_norm(rf))
            scale = max(1.0, float(torch.linalg.vector_norm(load[free])))
            if norm <= tolerance * scale:
                converged = True; break
            delta = torch.linalg.solve(assembly.tangent[free][:, free], -rf)
            base = norm
            accepted = False
            for power in range(line_search_steps + 1):
                candidate = trial.clone(); candidate[free] += (0.5 ** power) * delta
                candidate[fixed] = 0.0
                rn = assemble_global_contact(model, candidate, load, committed, tangent=False).residual[free]
                if float(torch.linalg.vector_norm(rn)) < base:
                    trial = candidate; accepted = True; break
            if not accepted:
                break
        if not converged:
            raise RuntimeError("global finite-sliding contact Newton solve failed; committed state was not changed")
        final = assemble_global_contact(model, trial, load, committed, tangent=False)
        committed = final.trial_state
        u = trial
        steps.append(GlobalContactStep(u.clone(), committed, iteration, norm, load.clone()))
    return tuple(steps)
