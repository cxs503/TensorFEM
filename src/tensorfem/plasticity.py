"""Material-point return mapping for verification and constitutive prototyping."""
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Plastic1DState:
    plastic_strain: torch.Tensor
    alpha: torch.Tensor


def update_bilinear_1d(strain: torch.Tensor, young: float, yield_stress: float,
                       hardening: float, state: Plastic1DState):
    """Backward-Euler return for 1-D associative plasticity with isotropic hardening."""
    trial = young * (strain - state.plastic_strain)
    f = torch.abs(trial) - (yield_stress + hardening * state.alpha)
    elastic = bool(f <= 0)
    if elastic:
        return trial, torch.as_tensor(young, dtype=strain.dtype, device=strain.device), state
    dgamma = f / (young + hardening)
    direction = torch.sign(trial)
    new_state = Plastic1DState(state.plastic_strain + dgamma * direction,
                               state.alpha + dgamma)
    stress = trial - young * dgamma * direction
    tangent = torch.as_tensor(young * hardening / (young + hardening),
                              dtype=strain.dtype, device=strain.device)
    return stress, tangent, new_state


@dataclass(frozen=True)
class J2State:
    plastic_strain: torch.Tensor
    alpha: torch.Tensor


def update_j2(strain: torch.Tensor, young: float, poisson: float, yield_stress: float,
              hardening: float, state: J2State):
    """3-D small-strain radial return using full symmetric 3x3 tensors."""
    if strain.shape != (3, 3) or state.plastic_strain.shape != (3, 3):
        raise ValueError("strain and plastic_strain must be 3x3 tensors")
    eye = torch.eye(3, dtype=strain.dtype, device=strain.device)
    G = young / (2.0 * (1.0 + poisson))
    K = young / (3.0 * (1.0 - 2.0 * poisson))
    elastic_strain = strain - state.plastic_strain
    volumetric = torch.trace(elastic_strain) / 3.0
    dev = elastic_strain - volumetric * eye
    s_trial = 2.0 * G * dev
    seq = torch.sqrt(1.5 * torch.sum(s_trial * s_trial))
    f = seq - (yield_stress + hardening * state.alpha)
    if bool(f <= 0):
        stress = K * torch.trace(elastic_strain) * eye + s_trial
        return stress, state, torch.zeros((), dtype=strain.dtype, device=strain.device)
    dgamma = f / (3.0 * G + hardening)
    flow = 1.5 * s_trial / seq
    new_state = J2State(state.plastic_strain + dgamma * flow, state.alpha + dgamma)
    s = s_trial - 2.0 * G * dgamma * flow
    stress = K * torch.trace(elastic_strain) * eye + s
    return stress, new_state, dgamma
