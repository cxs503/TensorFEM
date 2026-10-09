"""Benzeggagh--Kenane mixed-mode cohesive reference for proportional paths.

This module intentionally limits itself to monotonic proportional opening
paths.  It supplies an exact work-conjugate bilinear law suitable for material
verification; it does not claim arbitrary non-proportional crack growth.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MixedModePath:
    direction: torch.Tensor
    onset_amplitude: torch.Tensor
    failure_amplitude: torch.Tensor
    critical_energy: torch.Tensor
    traction: torch.Tensor


class BenzeggaghKenaneLaw:
    """Bilinear mixed-mode law with the BK propagation criterion.

    ``G_c = G_Ic + (G_IIc-G_Ic) (G_s/(G_n+G_s))**eta``.
    Normal compression is outside the scope of this cohesive reference.
    """
    def __init__(self, normal_stiffness: float, shear_stiffness: float,
                 normal_strength: float, shear_strength: float,
                 mode_i_energy: float, mode_ii_energy: float, eta: float):
        values = (normal_stiffness, shear_stiffness, normal_strength,
                  shear_strength, mode_i_energy, mode_ii_energy, eta)
        if any(v <= 0 for v in values):
            raise ValueError("all BK cohesive parameters must be positive")
        self.kn, self.ks = float(normal_stiffness), float(shear_stiffness)
        self.tn0, self.ts0 = float(normal_strength), float(shear_strength)
        self.gic, self.giic, self.eta = float(mode_i_energy), float(mode_ii_energy), float(eta)

    def critical_energy(self, direction: torch.Tensor) -> torch.Tensor:
        c = self._unit_direction(direction)
        wn, ws = self.kn * c[0] ** 2, self.ks * c[1] ** 2
        fraction = ws / (wn + ws)
        return self.gic + (self.giic - self.gic) * fraction ** self.eta

    @staticmethod
    def _unit_direction(direction: torch.Tensor) -> torch.Tensor:
        if direction.shape != (2,) or float(direction[0]) < 0:
            raise ValueError("direction must be [nonnegative normal, shear]")
        norm = torch.linalg.vector_norm(direction)
        if float(norm) == 0:
            raise ValueError("direction must be nonzero")
        return direction / norm

    def proportional_response(self, amplitudes: torch.Tensor,
                              direction: torch.Tensor) -> MixedModePath:
        """Evaluate traction along ``separation = amplitude * direction``."""
        if amplitudes.ndim == 0:
            amplitudes = amplitudes.reshape(1)
        if torch.any(amplitudes < 0):
            raise ValueError("amplitudes must be nonnegative")
        c = self._unit_direction(direction.to(dtype=amplitudes.dtype, device=amplitudes.device))
        dn0, ds0 = self.tn0 / self.kn, self.ts0 / self.ks
        a0 = 1.0 / torch.sqrt((c[0] / dn0) ** 2 + (c[1] / ds0) ** 2)
        work_stiffness = self.kn * c[0] ** 2 + self.ks * c[1] ** 2
        peak_work_traction = work_stiffness * a0
        gc = self.critical_energy(c)
        af = 2.0 * gc / peak_work_traction
        if float(af) <= float(a0):
            raise ValueError("fracture energy is too small for strengths and stiffnesses")
        elastic_factor = amplitudes
        softening_factor = a0 * (af - amplitudes) / (af - a0)
        factor = torch.where(amplitudes <= a0, elastic_factor,
                             torch.where(amplitudes < af, softening_factor,
                                         torch.zeros_like(amplitudes)))
        traction = factor[:, None] * torch.stack((c[0] * amplitudes.new_tensor(self.kn),
                                                  c[1] * amplitudes.new_tensor(self.ks)))[None, :]
        return MixedModePath(c, a0, af, gc, traction)
