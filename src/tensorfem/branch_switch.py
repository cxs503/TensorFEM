"""Opt-in diagnostics and predictor perturbations at symmetric bifurcations.

The utilities in this module are deliberately independent of material state:
they inspect a tangent but never evaluate or commit a constitutive response.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class CriticalMode:
    eigenvalue: float
    relative_eigenvalue: float
    vector: torch.Tensor
    eligible_energy_fraction: float


def smallest_eligible_symmetric_mode(
    tangent: torch.Tensor,
    *,
    eligible: torch.Tensor | None = None,
    minimum_eligible_fraction: float = 0.5,
) -> CriticalMode:
    """Return the smallest-absolute eligible mode of ``sym(tangent)``.

    ``eligible`` can exclude inactive drilling gauges from shell bifurcation
    detection.  A mode is retained only when the requested fraction of its
    Euclidean energy lies in eligible degrees of freedom.
    """
    if tangent.ndim != 2 or tangent.shape[0] != tangent.shape[1]:
        raise ValueError("tangent must be square")
    if not 0.0 <= minimum_eligible_fraction <= 1.0:
        raise ValueError("minimum_eligible_fraction must lie in [0, 1]")
    n = tangent.shape[0]
    if eligible is None:
        eligible = torch.ones(n, dtype=torch.bool, device=tangent.device)
    else:
        eligible = eligible.to(device=tangent.device, dtype=torch.bool)
        if eligible.shape != (n,):
            raise ValueError("eligible mask has wrong shape")
    values, vectors = torch.linalg.eigh((tangent + tangent.T) * 0.5)
    energy = torch.sum(vectors[eligible] ** 2, dim=0)
    candidates = torch.nonzero(energy >= minimum_eligible_fraction).flatten()
    if candidates.numel() == 0:
        raise ValueError("tangent has no eligible eigenmode")
    local = torch.argmin(torch.abs(values[candidates]))
    index = candidates[local]
    scale = max(float(torch.max(torch.abs(values))), torch.finfo(tangent.dtype).tiny)
    vector = vectors[:, index]
    pivot = torch.argmax(torch.abs(vector))
    # Eigenvectors are sign-indeterminate. Canonicalise them so ``branch_sign``
    # has stable meaning across LAPACK implementations and restart processes.
    if float(vector[pivot]) < 0.0:
        vector = -vector
    return CriticalMode(
        float(values[index]), float(torch.abs(values[index])) / scale,
        vector, float(energy[index]),
    )


def perturbed_arc_predictor(
    tangent_direction: torch.Tensor,
    critical_mode: torch.Tensor,
    *,
    mode_fraction: float,
    sign: int,
) -> torch.Tensor:
    """Blend a unit arc tangent with an orthogonal critical mode.

    Both arguments use dimensioned arc coordinates ``[u, p]``.  The returned
    vector has unit norm, so multiplying it by the arc radius preserves the
    cylindrical/spherical constraint exactly at the predictor.
    """
    if tangent_direction.ndim != 1 or critical_mode.shape != tangent_direction.shape:
        raise ValueError("predictor and mode must be equal-length vectors")
    if not 0.0 < mode_fraction <= 1.0 or sign not in (-1, 1):
        raise ValueError("invalid branch perturbation controls")
    primary = tangent_direction / torch.linalg.vector_norm(tangent_direction)
    mode = critical_mode - torch.dot(critical_mode, primary) * primary
    norm = torch.linalg.vector_norm(mode)
    if float(norm) <= 100 * torch.finfo(mode.dtype).eps:
        raise ValueError("critical mode is parallel to path tangent")
    mode = mode / norm
    retained = max(0.0, 1.0 - mode_fraction**2) ** 0.5
    return retained * primary + float(sign) * mode_fraction * mode
