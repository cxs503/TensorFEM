"""Coupled one-dimensional elastoplasticity and unilateral contact.

This small global problem is intentionally auditable: material yielding and a
rigid stop are assembled into the same residual and consistent tangent, while
``nonlinear_step.solve_adaptive`` owns increment retry and state commit.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from .nonlinear_step import StepState
from .plasticity import Plastic1DState, update_bilinear_1d


@dataclass(frozen=True)
class PlasticBarContact:
    """Axial bar whose loaded end is limited by a penalty rigid stop.

    Positive displacement stretches the bar.  Contact is active for
    ``displacement > clearance`` and opposes the applied tensile force.
    """

    length: float
    area: float
    young: float
    yield_stress: float
    hardening: float
    clearance: float
    contact_penalty: float
    reference_force: float

    def __post_init__(self):
        positive = (self.length, self.area, self.young, self.yield_stress,
                    self.contact_penalty)
        if any(x <= 0 for x in positive):
            raise ValueError("length, area, Young modulus, yield stress and penalty must be positive")
        if self.hardening < 0 or self.clearance < 0:
            raise ValueError("hardening and clearance must be nonnegative")


@dataclass(frozen=True)
class CoupledResponse:
    stress: torch.Tensor
    bar_force: torch.Tensor
    contact_force: torch.Tensor
    gap: torch.Tensor
    active: bool


def coupled_bar_contact_problem(model: PlasticBarContact, *, dtype=torch.float64,
                                device=None):
    """Return a transactional residual adapter and its virgin checkpoint."""
    zero = torch.zeros((), dtype=dtype, device=device)
    initial = StepState(0.0, torch.zeros(1, dtype=dtype, device=device),
                        Plastic1DState(zero.clone(), zero.clone()))

    def evaluate(u: torch.Tensor, factor: float, committed: Plastic1DState):
        stress, material_tangent, trial = update_bilinear_1d(
            u[0] / model.length, model.young, model.yield_stress,
            model.hardening, committed)
        penetration = torch.clamp(u[0] - model.clearance, min=0.0)
        contact = model.contact_penalty * penetration
        residual = u.new_tensor([factor * model.reference_force])
        residual = residual - model.area * stress.reshape(1) - contact.reshape(1)
        active = bool(u[0] > model.clearance)
        tangent = model.area * material_tangent / model.length
        if active:
            tangent = tangent + model.contact_penalty
        return residual, tangent.reshape(1, 1), trial

    return evaluate, initial


def coupled_response(model: PlasticBarContact, displacement: torch.Tensor,
                     state: Plastic1DState) -> CoupledResponse:
    """Recover forces at a converged checkpoint without changing its state."""
    u = displacement.reshape(-1)[0]
    stress = model.young * (u / model.length - state.plastic_strain)
    penetration = torch.clamp(u - model.clearance, min=0.0)
    contact = model.contact_penalty * penetration
    return CoupledResponse(stress, model.area * stress, contact,
                           model.clearance - u, bool(penetration > 0))


def monotonic_closed_form(model: PlasticBarContact, force: float) -> float:
    """Piecewise analytical displacement for monotonic positive loading.

    The formula independently combines the bilinear bar force with the penalty
    stop.  It is a verification reference, not the numerical solve path.
    """
    if force < 0:
        raise ValueError("the monotonic reference requires nonnegative force")
    uy = model.length * model.yield_stress / model.young
    fy = model.area * model.yield_stress
    candidates = sorted({0.0, uy, model.clearance})
    et = (model.young if model.hardening == 0 else
          model.young * model.hardening / (model.young + model.hardening))

    def bar(u: float) -> tuple[float, float]:
        if u <= uy:
            return model.area * model.young * u / model.length, model.area * model.young / model.length
        if model.hardening == 0:
            return fy, 0.0
        intercept = model.area * (model.yield_stress - et * model.yield_stress / model.young)
        return intercept + model.area * et * u / model.length, model.area * et / model.length

    # Each interval is affine. Solve in it, then handle the final half-line.
    bounds = candidates + [float("inf")]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        probe = lo + 1e-12 * max(1.0, lo)
        value, slope = bar(probe)
        if probe > model.clearance:
            value += model.contact_penalty * (probe - model.clearance)
            slope += model.contact_penalty
        root = probe + (force - value) / slope if slope > 0 else float("inf")
        if root >= lo - 1e-12 and root <= hi + 1e-12:
            return root
    raise RuntimeError("closed-form branch selection failed")
