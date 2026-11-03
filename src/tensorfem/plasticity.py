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


_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0))


def j2_consistent_tangent(strain: torch.Tensor, young: float, poisson: float,
                          hardening: float, state: J2State, dgamma: torch.Tensor):
    """Exact algorithmic tangent of the radial return returned by ``update_j2``.

    ``dgamma`` is the third value returned by ``update_j2`` for the same strain
    and committed state, so the operator always belongs to the branch that was
    actually integrated (elastic when ``dgamma == 0``).  Voigt order is
    ``[xx, yy, zz, xy, yz, zx]`` and shear columns act on engineering shear
    strains ``gamma = 2*eps``:
    ``C = K (1 x 1) + 2G a I_dev + c s_trial (x) s_trial`` with
    ``a = 1 - 3G dgamma/seq`` and ``c = 3G (da/dseq)/seq``.
    """
    if strain.shape != (3, 3):
        raise ValueError("strain must be 3x3")
    dtype, device = strain.dtype, strain.device
    G = young / (2.0 * (1.0 + poisson))
    K = young / (3.0 * (1.0 - 2.0 * poisson))
    eye3 = torch.eye(3, dtype=dtype, device=device)
    tangent = torch.zeros((6, 6), dtype=dtype, device=device)
    elastic_strain = strain - state.plastic_strain
    s_trial = 2.0 * G * (elastic_strain - torch.trace(elastic_strain) / 3.0 * eye3)
    seq = torch.sqrt(1.5 * torch.sum(s_trial * s_trial))
    hydrostatic = K * torch.ones((3, 3), dtype=dtype, device=device)
    if float(dgamma) == 0.:
        tangent[:3, :3] = hydrostatic + 2.0 * G * (eye3 - torch.ones((3, 3), dtype=dtype, device=device) / 3.0)
        tangent[3, 3] = tangent[4, 4] = tangent[5, 5] = G
        return tangent
    if not bool(seq > 0.):
        raise RuntimeError("plastic return reported zero trial stress")
    a = 1.0 - 3.0 * G * float(dgamma) / float(seq)
    da = -3.0 * G * (1.0 / ((3.0 * G + hardening) * float(seq))
                     - float(dgamma) / (float(seq) ** 2))
    c = 2.0 * G * (1.5 / float(seq)) * da
    tangent[:3, :3] = hydrostatic + 2.0 * G * a * (eye3 - torch.ones((3, 3), dtype=dtype, device=device) / 3.0)
    tangent[3, 3] = tangent[4, 4] = tangent[5, 5] = G * a
    vector = torch.stack([s_trial[i, j] for i, j in _VOIGT_PAIRS])
    return tangent + c * torch.outer(vector, vector)


def condense_plane_stress(tangent: torch.Tensor) -> torch.Tensor:
    """Schur-condense a Voigt 6x6 tangent to ``[xx, yy, xy]`` with ``sigma_zz = 0``."""
    in_plane = torch.tensor([0, 1, 3], dtype=torch.long, device=tangent.device)
    normal = tangent[2, 2]
    if abs(float(normal)) <= torch.finfo(tangent.dtype).eps * max(1.0, float(tangent.abs().max())):
        raise RuntimeError("plane-stress condensation has singular sigma_zz tangent")
    return (tangent[in_plane][:, in_plane]
            - torch.outer(tangent[in_plane, 2], tangent[2, in_plane]) / normal)
