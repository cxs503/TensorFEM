"""Reusable global solver for multi-element small-strain TET4 J2 models.

This module is explicitly infinitesimal-strain. It provides global assembly,
transactional integration-point state, automatic increments and restartable
piecewise proportional load paths; it is not a finite-strain formulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .nonlinear_step import IncrementRecord, NonlinearConvergenceError, StepState, solve_adaptive
from .solid_plasticity import (PlasticSolidState, Tet4J2Model,
                               assemble_tet4_j2)

Tensor = torch.Tensor


@dataclass(frozen=True)
class GlobalPlasticResult:
    load_factor: float
    displacement: Tensor
    reaction: Tensor
    stress: Tensor
    equivalent_plastic_strain: Tensor
    material_state: PlasticSolidState
    increments: tuple[IncrementRecord, ...]


def validate_model(model: Tet4J2Model) -> None:
    """Fail closed on topology, constraints, geometry and material inputs."""
    if model.elements.numel() == 0:
        raise ValueError("at least one TET4 element is required")
    if model.elements.dtype != torch.long:
        raise TypeError("elements must use torch.long connectivity")
    if int(model.elements.min()) < 0 or int(model.elements.max()) >= model.nodes.shape[0]:
        raise IndexError("element node outside model")
    fixed = model.fixed_dofs
    if fixed.dtype != torch.long or fixed.ndim != 1:
        raise TypeError("fixed_dofs must be a one-dimensional long tensor")
    if fixed.numel() != torch.unique(fixed).numel():
        raise ValueError("fixed_dofs must be unique")
    if fixed.numel() and (int(fixed.min()) < 0 or int(fixed.max()) >= model.n_dofs):
        raise IndexError("fixed DOF outside model")
    if model.young <= 0 or model.yield_stress <= 0 or model.hardening < 0:
        raise ValueError("invalid J2 material constants")
    # Reuse element kinematics so inverted/degenerate elements are rejected.
    zero = torch.zeros(model.n_dofs, dtype=model.nodes.dtype, device=model.nodes.device)
    assemble_tet4_j2(model, zero, PlasticSolidState.virgin(model))


def _free_dofs(model: Tet4J2Model) -> Tensor:
    mask = torch.ones(model.n_dofs, dtype=torch.bool, device=model.nodes.device)
    mask[model.fixed_dofs.to(model.nodes.device)] = False
    return torch.arange(model.n_dofs, device=model.nodes.device)[mask]


def solve_tet4_j2_path(
    model: Tet4J2Model,
    reference_force: Tensor,
    targets: Sequence[float],
    *,
    initial: GlobalPlasticResult | None = None,
    initial_increment: float = .1,
    minimum_increment: float = 1e-5,
    maximum_increment: float = .25,
    tolerance: float = 1e-9,
    max_iterations: int = 20,
) -> tuple[GlobalPlasticResult, ...]:
    """Solve a piecewise proportional load path with committed-state rollback.

    Each target may increase or decrease. A segment is parameterized internally
    from zero to one, allowing the shared adaptive driver to handle unloading
    without changing its monotone pseudo-time contract.
    """
    validate_model(model)
    force = reference_force.to(model.nodes)
    if force.shape != (model.n_dofs,):
        raise ValueError("reference_force must have one value per global DOF")
    free = _free_dofs(model)
    if free.numel() == 0:
        raise ValueError("model has no free DOFs")
    if initial is None:
        factor = 0.0
        displacement = torch.zeros(model.n_dofs, dtype=model.nodes.dtype,
                                   device=model.nodes.device)
        committed = PlasticSolidState.virgin(model)
    else:
        factor = float(initial.load_factor)
        displacement = initial.displacement.to(model.nodes).clone()
        committed = initial.material_state
        if len(committed.points) != model.elements.shape[0]:
            raise ValueError("restart material state does not match element count")
    outputs: list[GlobalPlasticResult] = []
    for target_value in targets:
        target = float(target_value); start = factor; delta = target-start
        if delta == 0.0:
            internal, _, stress, trial = assemble_tet4_j2(model, displacement, committed)
            outputs.append(_result(model, target, force, displacement, internal,
                                   stress, trial, ()))
            continue

        def evaluate(u_free: Tensor, progress: float, base: PlasticSolidState):
            u = displacement.clone().index_copy(0, free, u_free)
            internal, tangent, _, trial = assemble_tet4_j2(model, u, base)
            physical_factor = start + delta*progress
            residual = physical_factor*force[free]-internal[free]
            return residual, tangent[free][:, free], trial

        state = StepState(0.0, displacement[free].clone(), committed)
        advanced = solve_adaptive(
            evaluate, state, target_factor=1.0,
            initial_increment=initial_increment,
            minimum_increment=minimum_increment,
            maximum_increment=maximum_increment,
            tolerance=tolerance, max_iterations=max_iterations,
        )
        displacement = displacement.clone().index_copy(0, free, advanced.displacement)
        committed = advanced.material_state; factor = target
        internal, _, stress, trial = assemble_tet4_j2(model, displacement, committed)
        # At a converged configuration re-integrating from the newly committed
        # state must be elastic and leave history invariant.
        committed = trial
        outputs.append(_result(model, factor, force, displacement, internal,
                               stress, committed, tuple(advanced.history)))
    return tuple(outputs)


def _result(model, factor, force, displacement, internal, stress, state, history):
    return GlobalPlasticResult(
        factor, displacement.clone(), internal-factor*force, stress,
        torch.stack(tuple(point.alpha for point in state.points)), state, history)
