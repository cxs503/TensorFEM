"""Small-strain cohesive-zone constitutive models.

This module deliberately implements a material-point law and a two-node normal
interface only.  It is useful for verification and for prescribed crack paths;
it is not a general crack-growth algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CohesiveState:
    """History for an irreversible cohesive law.

    ``maximum_opening`` is the greatest tensile opening reached so far.
    """

    maximum_opening: torch.Tensor


@dataclass(frozen=True)
class CohesiveResponse:
    traction: torch.Tensor
    tangent: torch.Tensor
    damage: torch.Tensor
    state: CohesiveState


class BilinearCohesiveLaw:
    """Rate-independent mode-I bilinear traction--separation law.

    Parameters are initial penalty stiffness ``stiffness``, peak traction
    ``peak_traction`` and total mode-I fracture energy ``fracture_energy``.
    The failure opening follows exactly from the area of the triangular
    envelope: ``failure_opening = 2*fracture_energy/peak_traction``.

    Tensile damage is irreversible.  Unloading/reloading occurs linearly to
    the origin with the secant stiffness at the historical maximum opening.
    Compression uses the undamaged penalty stiffness and does not alter
    history.
    """

    def __init__(self, stiffness: float, peak_traction: float, fracture_energy: float):
        if stiffness <= 0 or peak_traction <= 0 or fracture_energy <= 0:
            raise ValueError("cohesive parameters must be positive")
        self.stiffness = float(stiffness)
        self.peak_traction = float(peak_traction)
        self.fracture_energy = float(fracture_energy)
        if self.failure_opening <= self.onset_opening:
            raise ValueError("fracture energy is too small for the specified stiffness and strength")

    @property
    def onset_opening(self) -> float:
        return self.peak_traction / self.stiffness

    @property
    def failure_opening(self) -> float:
        return 2.0 * self.fracture_energy / self.peak_traction

    def initial_state(self, *, dtype: torch.dtype = torch.float64, device=None) -> CohesiveState:
        return CohesiveState(torch.zeros((), dtype=dtype, device=device))

    def envelope_traction(self, opening: torch.Tensor) -> torch.Tensor:
        """Monotonic tensile envelope, with zero traction after failure."""
        x = torch.clamp(opening, min=0.0)
        d0, df = self.onset_opening, self.failure_opening
        elastic = self.stiffness * x
        softening = self.peak_traction * (df - x) / (df - d0)
        return torch.where(x <= d0, elastic, torch.where(x < df, softening, torch.zeros_like(x)))

    def update(self, opening: torch.Tensor, state: CohesiveState | None = None) -> CohesiveResponse:
        if not torch.is_tensor(opening):
            opening = torch.as_tensor(opening, dtype=torch.get_default_dtype())
        if state is None:
            state = self.initial_state(dtype=opening.dtype, device=opening.device)
        old_max = state.maximum_opening.to(dtype=opening.dtype, device=opening.device)
        tensile_opening = torch.clamp(opening, min=0.0)
        maximum = torch.maximum(old_max, tensile_opening)

        d0, df = self.onset_opening, self.failure_opening
        tiny = torch.finfo(opening.dtype).tiny
        envelope_at_max = self.envelope_traction(maximum)
        secant = torch.where(maximum > d0, envelope_at_max / torch.clamp(maximum, min=tiny),
                              torch.full_like(maximum, self.stiffness))
        damage = torch.clamp(1.0 - secant / self.stiffness, min=0.0, max=1.0)

        tensile_traction = secant * tensile_opening
        traction = torch.where(opening < 0.0, self.stiffness * opening, tensile_traction)

        loading = (tensile_opening >= old_max) & (tensile_opening > d0) & (tensile_opening < df)
        softening_tangent = torch.full_like(opening, -self.peak_traction / (df - d0))
        tensile_tangent = torch.where(loading, softening_tangent, secant)
        tangent = torch.where(opening < 0.0, torch.full_like(opening, self.stiffness), tensile_tangent)
        return CohesiveResponse(traction, tangent, damage, CohesiveState(maximum))


def two_node_interface_response(
    displacements: torch.Tensor,
    area: float,
    law: BilinearCohesiveLaw,
    state: CohesiveState | None = None,
) -> tuple[torch.Tensor, torch.Tensor, CohesiveState]:
    """Internal force and tangent of a two-node normal cohesive interface.

    The scalar nodal displacement convention is ``opening = u[1] - u[0]``.
    The returned internal forces therefore satisfy exact action--reaction.
    """
    if displacements.shape != (2,):
        raise ValueError("displacements must have shape (2,)")
    if area <= 0:
        raise ValueError("interface area must be positive")
    response = law.update(displacements[1] - displacements[0], state)
    direction = torch.tensor([-1.0, 1.0], dtype=displacements.dtype, device=displacements.device)
    force = area * response.traction * direction
    stiffness = area * response.tangent * torch.outer(direction, direction)
    return force, stiffness, response.state
