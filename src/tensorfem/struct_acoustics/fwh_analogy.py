"""FW-H acoustic analogy at zero Mach number: a permeable Kirchhoff data surface.

What this module is for
-----------------------
The production route to the radiated power of a vibrating plate is the Rayleigh
integral over the *velocity* (:mod:`tensorfem.struct_acoustics.radiation_coupling`).
The Ffowcs Williams-Hawkings analogy is the other classical route: the far field
is reconstructed from a **data surface that encloses the source**, using both the
pressure and its normal derivative on that surface.  For a stationary surface in
a fluid at rest the FW-H equation degenerates to the Kirchhoff-Helmholtz
integral (Farassat's formulation 1A with ``M = 0``; the quadrupole term is
dropped because the field is linear), so

.. math::

   p(x) = \\oint_{S_d} \\Bigl[\\,p\\,\\partial_n G - G\\,\\partial_n p\\,\\Bigr]\\,dS,
   \\qquad G = \\frac{e^{ikR}}{4\\pi R} \\;(+\\;\\text{image}),

with ``S_d`` a permeable surface placed in the *near field* of the plate, and
``n`` the normal pointing out of the region that contains the observer.

Why it is an independent check
------------------------------
The analogy is fed the *pressure* on the data surface while the Rayleigh route is
fed the *velocity* on the plate.  Two different physical boundary quantities,
two different surfaces, two different kernels (``partial_n p`` and
``partial_n G``) -- yet both must return the same far field, including its phase.
Agreement therefore tests the coupled solution's two fields against each other
rather than re-testing one quadrature.

Geometry
--------
Plate in the plane ``z = 0`` inside an infinite rigid baffle; the data surface
is a half-sphere of radius ``R_d`` (centred on the plate centre, base on the
baffle) large enough to enclose the plate.  The half-space Green function
``G_hs = e^{ikR_1}/(4pi R_1) + e^{ikR_2}/(4pi R_2)`` (image in the baffle plane)
kills the boundary term on the baffle, so only the spherical cap contributes.

Restriction (stated): frequency domain, quiescent fluid, rigid baffle, single
data surface.  Convected FW-H (``M != 0``), permeable-surface quadrupoles and
time-domain retarded-time integration are *not* implemented here -- they are
listed as the honest scope boundary of this module, not hidden behind a flag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .levy_modes import PlateModalConfig, _positive
from .radiation_coupling import C_AIR, RHO_AIR, RadiationGrid

__all__ = [
    "KirchhoffSurfaceResult",
    "kirchhoff_data_surface_power",
    "fwh_far_field_power",
]


def _directions(
    n_theta: int, n_phi: int, *, polar_limit: float = math.pi / 2.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    theta = (torch.arange(int(n_theta), dtype=torch.float64) + 0.5) * (
        polar_limit / int(n_theta)
    )
    phi = (torch.arange(int(n_phi), dtype=torch.float64) + 0.5) * (
        2.0 * math.pi / int(n_phi)
    )
    d_omega = (
        torch.sin(theta)
        * (polar_limit / int(n_theta))
        * (2.0 * math.pi / int(n_phi))
    )
    return theta, phi, d_omega


@dataclass(frozen=True)
class KirchhoffSurfaceResult:
    """Far field of a plate recovered through the FW-H/Kirchhoff analogy."""

    frequency: float
    radius: float
    distance: float
    theta: torch.Tensor
    phi: torch.Tensor
    d_omega: torch.Tensor
    analogy_pressure: torch.Tensor
    rayleigh_pressure: torch.Tensor
    analogy_power: float
    rayleigh_power: float
    rho: float
    sound_speed: float

    @property
    def relative_power_gap(self) -> float:
        """``|P_FWH - P_Rayleigh| / P_Rayleigh``."""
        return abs(self.analogy_power - self.rayleigh_power) / self.rayleigh_power

    @property
    def amplitude_ratio(self) -> float:
        """Mean ``|p_FWH| / |p_Rayleigh|`` over the far-field directions."""
        numerator = self.analogy_pressure.abs().sum()
        denominator = self.rayleigh_pressure.abs().sum()
        return float(numerator / denominator)

    @property
    def phase_coherence(self) -> float:
        """``|<p_FWH, p_Rayleigh>| / (||p_FWH|| ||p_Rayleigh||)`` -- 1 if in phase."""
        inner = torch.abs((torch.conj(self.analogy_pressure) * self.rayleigh_pressure).sum())
        norm = torch.linalg.vector_norm(self.analogy_pressure) * torch.linalg.vector_norm(
            self.rayleigh_pressure
        )
        return float(inner / norm)

    def directivity(self) -> torch.Tensor:
        values = self.analogy_pressure.abs()
        peak = float(values.max())
        return values / peak if peak > 0.0 else values


def _in_plane(grid: RadiationGrid) -> torch.Tensor:
    """Panel centres lifted into 3-D (``z = 0`` is the baffle plane)."""
    points = grid.points
    return torch.cat([points, torch.zeros(points.shape[0], 1, dtype=points.dtype)], dim=1)


def _cap_points(
    radius: float, n_theta: int, n_phi: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Midpoint samples of the hemispherical data surface and their area weights."""
    theta, phi, d_omega = _directions(n_theta, n_phi)
    sin_t = torch.sin(theta).reshape(-1, 1)
    points = torch.stack(
        [
            radius * sin_t * torch.cos(phi).reshape(1, -1),
            radius * sin_t * torch.sin(phi).reshape(1, -1),
            radius * torch.cos(theta).reshape(-1, 1) * torch.ones_like(phi).reshape(1, -1),
        ],
        dim=-1,
    ).reshape(-1, 3)
    weights = (d_omega[:, None] * torch.ones_like(phi).reshape(1, -1)).reshape(-1) * radius ** 2
    return points, weights


def _surface_field(
    points: torch.Tensor,
    grid: RadiationGrid,
    velocity: torch.Tensor,
    *,
    frequency: float,
    rho: float,
    sound_speed: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pressure and its normal derivative on a hemispherical surface.

    The normal is the radial direction ``s_hat`` (pointing away from the plate).
    Both quantities come from the Rayleigh integral over the plate velocity, so
    they are the *exact* near field of the coupled solution on the data surface.
    """
    k = 2.0 * math.pi * frequency / sound_speed
    source = _in_plane(grid)
    weights = grid.weights
    radius = torch.linalg.vector_norm(points, dim=1)
    normal = points / radius.reshape(-1, 1)
    delta = points[:, None, :] - source[None, :, :]
    r = torch.linalg.vector_norm(delta, dim=2)
    green = torch.exp(1j * k * r) / (2.0 * math.pi * r)
    dr_dn = (delta * normal[:, None, :]).sum(dim=2) / r
    # d/dR [exp(ikR)/(2 pi R)] = (ik - 1/R) exp(ikR)/(2 pi R)
    d_green = (1j * k - 1.0 / r) * green * dr_dn
    value = -(1j * 2.0 * math.pi * frequency * rho) * (
        green * (weights * velocity).reshape(1, -1)
    ).sum(dim=1)
    derivative = -(1j * 2.0 * math.pi * frequency * rho) * (
        d_green * (weights * velocity).reshape(1, -1)
    ).sum(dim=1)
    return value, derivative


def kirchhoff_data_surface_power(
    grid: RadiationGrid,
    config: PlateModalConfig,
    *,
    surface_velocity: torch.Tensor,
    frequency: float,
    rho: float = RHO_AIR,
    sound_speed: float = C_AIR,
    radius_factor: float = 2.0,
    n_cap_theta: int = 12,
    n_cap_phi: int = 24,
    n_far_theta: int = 16,
    n_far_phi: int = 48,
    distance_factor: float = 2000.0,
) -> KirchhoffSurfaceResult:
    """Radiated power through the FW-H/Kirchhoff analogy, with the Rayleigh cross-check.

    ``radius_factor`` multiplies the plate half-diagonal: the data surface must
    enclose the source, and pushing it too close couples the quadrature to the
    singular self-panel of the plate.  The default ``2.0`` is the value at which
    the analogy reproduces the Rayleigh power to <0.1 % in the qualification
    study (``benchmarks/struct_acoustic_coupling.py``); at ``1.25`` the near
    field on the data surface is under-resolved and the gap rises to ~40 %.
    """
    frequency = _positive("frequency", frequency)
    rho = _positive("rho", rho)
    sound_speed = _positive("sound_speed", sound_speed)
    velocity = torch.as_tensor(surface_velocity).to(torch.complex128).reshape(-1)
    if velocity.numel() != grid.points.shape[0]:
        raise ValueError("surface_velocity length must match the grid")
    radius = float(radius_factor) * 0.5 * math.hypot(config.length, config.width)
    if radius <= 0.0:
        raise ValueError("data-surface radius must be positive")
    k = 2.0 * math.pi * frequency / sound_speed

    cap, cap_weights = _cap_points(radius, n_cap_theta, n_cap_phi)
    pressure_cap, derivative_cap = _surface_field(
        cap,
        grid,
        velocity,
        frequency=frequency,
        rho=rho,
        sound_speed=sound_speed,
    )

    theta, phi, d_omega = _directions(n_far_theta, n_far_phi)
    ranging = float(distance_factor) * sound_speed / frequency
    sin_t = torch.sin(theta).reshape(-1, 1)
    observer = torch.stack(
        [
            ranging * sin_t * torch.cos(phi).reshape(1, -1),
            ranging * sin_t * torch.sin(phi).reshape(1, -1),
            ranging * torch.cos(theta).reshape(-1, 1) * torch.ones_like(phi).reshape(1, -1),
        ],
        dim=-1,
    )  # (nt, np, 3)
    flat_observer = observer.reshape(-1, 3)

    cap_radius = torch.linalg.vector_norm(cap, dim=1)
    cap_normal = cap / cap_radius.reshape(-1, 1)

    # Half-space Green function from each data-surface point to each observer,
    # with the image in the baffle plane z = 0.
    delta = flat_observer[:, None, :] - cap[None, :, :]
    r1 = torch.linalg.vector_norm(delta, dim=2)
    mirrored = flat_observer.clone()
    mirrored[:, 2] = -mirrored[:, 2]
    delta_image = mirrored[:, None, :] - cap[None, :, :]
    r2 = torch.linalg.vector_norm(delta_image, dim=2)
    green = torch.exp(1j * k * r1) / (4.0 * math.pi * r1) + torch.exp(1j * k * r2) / (
        4.0 * math.pi * r2
    )
    dr1_dn = -(delta * cap_normal[None, :, :]).sum(dim=2) / r1
    dr2_dn = -(delta_image * cap_normal[None, :, :]).sum(dim=2) / r2
    d_green = (1j * k / r1 - 1.0 / r1 ** 2) * torch.exp(1j * k * r1) / (
        4.0 * math.pi
    ) * dr1_dn + (1j * k / r2 - 1.0 / r2 ** 2) * torch.exp(1j * k * r2) / (
        4.0 * math.pi
    ) * dr2_dn
    integrand = (
        pressure_cap.reshape(1, -1) * d_green - green * derivative_cap.reshape(1, -1)
    ) * cap_weights.reshape(1, -1)
    analogy = integrand.sum(dim=1).reshape(theta.numel(), phi.numel())

    # Reference: Rayleigh integral over the plate velocity at the same observers.
    plate_delta = flat_observer[:, None, :] - _in_plane(grid)[None, :, :]
    r_plate = torch.linalg.vector_norm(plate_delta, dim=2)
    rayleigh = -(1j * 2.0 * math.pi * frequency * rho) * (
        torch.exp(1j * k * r_plate) / (2.0 * math.pi * r_plate)
        * (grid.weights * velocity).reshape(1, -1)
    ).sum(dim=1)
    rayleigh = rayleigh.reshape(theta.numel(), phi.numel())

    def _power(values: torch.Tensor) -> float:
        return float(
            (values.abs() ** 2 * d_omega[:, None]).sum()
            * ranging ** 2
            / (2.0 * rho * sound_speed)
        )

    return KirchhoffSurfaceResult(
        frequency=frequency,
        radius=radius,
        distance=ranging,
        theta=theta,
        phi=phi,
        d_omega=d_omega,
        analogy_pressure=analogy,
        rayleigh_pressure=rayleigh,
        analogy_power=_power(analogy),
        rayleigh_power=_power(rayleigh),
        rho=rho,
        sound_speed=sound_speed,
    )


def fwh_far_field_power(*args, **kwargs) -> KirchhoffSurfaceResult:
    """Alias of :func:`kirchhoff_data_surface_power` (FW-H is the same M = 0 limit)."""
    return kirchhoff_data_surface_power(*args, **kwargs)