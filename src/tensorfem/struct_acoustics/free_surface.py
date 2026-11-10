"""Underwater radiation below a pressure-release free surface (Lloyd's mirror).

Physics
-------
A plate in an infinite rigid baffle radiates into the *upper* half space; a free
surface (pressure release, ``p = 0``) sits ``d`` above the plate.  Every
elementary source therefore has a mirror image of opposite sign, and the
far-field pressure is the coherent sum of the direct and surface-reflected
paths:

.. math:: p(r,\\theta,\\phi) \\propto A(\\theta,\\phi)\\,\\bigl(e^{i k d\\cos\\theta}
          - e^{-i k d\\cos\\theta}\\bigr)
          = 2i\\,\\sin(k d\\cos\\theta)\\,A(\\theta,\\phi),

with ``theta`` measured from the plate normal.  The interference factor
``2 sin(k d cos theta)`` produces the characteristic Lloyd's-mirror pattern:
nulls at

.. math:: \\cos\\theta_n = \\frac{n\\lambda}{2d}, \\qquad n = 1, 2, ...

and a power ratio ``P_free-surface / P_free-field = <4 sin^2(k d cos theta)>``
that is *not* one: a shallow depth suppresses the radiated power (a pressure
release close to the plate short-circuits it, ratio -> 0 as ``k d -> 0``), while
a deep surface with dense interference fringes gives the *incoherent* limit
``<4 sin^2> = 2`` -- the reflected image adds its energy incoherently, so the
power doubles, not halves.  For a compact source (``k a << 1``) the ratio is the
closed form ``2[1 - sin(2 k d)/(2 k d)]``, which the image sum reproduces to
0.01 % in the tests.  Both are experiment-facing quantities -- free-surface interference is
how an underwater radiator's directivity is distinguished from its free-field
one -- which is why they are the acceptance anchors here.

Two numerically independent routes are implemented:

* :func:`free_surface_directivity` -- the **image sum with exact distances**:
  the plate is replicated (opposite sign) at ``z = 2d`` and both surface
  integrals are evaluated with true ``e^{ikR}/R`` kernels at a large but finite
  range.  No far-field factor is assumed anywhere; the interference has to come
  out of the geometry.
* :func:`free_surface_power_ratio_analytic` -- the **closed-form Lloyd factor**
  applied to the free-field angular spectrum.

Agreement of the two pins the image sum; agreement of the null angles with
``cos theta_n = n lambda / (2d)`` pins the geometry.

Restriction (stated, not hidden): a *single* surface bounce is kept.  The
multiple-bounce series is the shallow-water (Pekeris waveguide) normal-mode
problem -- a different model with a different dispersion -- and is out of scope;
the single-bounce Lloyd model is what ship-acoustic practice uses for
free-surface interference at moderate depth and range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .levy_modes import PlateModalConfig, _positive
from .radiation_coupling import C_WATER, RHO_WATER, RadiationGrid

__all__ = [
    "FreeSurfaceField",
    "lloyd_null_cosines",
    "free_surface_directivity",
    "free_surface_power",
    "free_surface_power_ratio_analytic",
]


def _half_space_directions(
    n_theta: int, n_phi: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Midpoint quadrature of the radiating hemisphere (``theta`` from the normal)."""
    if int(n_theta) < 2 or int(n_phi) < 4:
        raise ValueError("need n_theta >= 2 and n_phi >= 4")
    theta = (torch.arange(int(n_theta), dtype=torch.float64) + 0.5) * (
        math.pi / 2.0 / int(n_theta)
    )
    phi = (torch.arange(int(n_phi), dtype=torch.float64) + 0.5) * (
        2.0 * math.pi / int(n_phi)
    )
    d_omega = (
        torch.sin(theta) * (math.pi / 2.0 / int(n_theta)) * (2.0 * math.pi / int(n_phi))
    )
    return theta, phi, d_omega


def lloyd_null_cosines(
    depth: float, wavelength: float, *, max_order: int = 32
) -> tuple[float, ...]:
    """``cos theta_n = n lambda / (2 d)`` for every null inside the hemisphere."""
    depth = _positive("depth", depth)
    wavelength = _positive("wavelength", wavelength)
    cosines: list[float] = []
    for n in range(1, int(max_order) + 1):
        value = n * wavelength / (2.0 * depth)
        if value > 1.0:
            break
        cosines.append(value)
    return tuple(cosines)


@dataclass(frozen=True)
class FreeSurfaceField:
    """Far-field directivity of a baffled plate below a free surface."""

    depth: float
    frequency: float
    theta: torch.Tensor        # (nt,) polar angle from the plate normal, rad
    phi: torch.Tensor          # (np,) azimuth, rad
    amplitude: torch.Tensor    # (nt, np) complex, ``p * r`` (distance cancels)
    free_amplitude: torch.Tensor
    d_omega: torch.Tensor      # (nt,) solid-angle weights
    power: float
    free_field_power: float
    distance: float

    @property
    def directivity(self) -> torch.Tensor:
        """``|p r|`` normalised to its maximum (linear, not dB)."""
        values = self.amplitude.abs()
        peak = float(values.max())
        return values / peak if peak > 0.0 else values

    @property
    def power_ratio(self) -> float:
        return self.power / self.free_field_power

    def theta_profile(self) -> torch.Tensor:
        """Directivity averaged over the azimuth, shape ``(nt,)``."""
        return self.directivity.abs().mean(dim=1)

    def null_angles(self) -> tuple[float, ...]:
        """Polar angles of the interference minima of the azimuthal mean pattern.

        Each discrete minimum is refined by the vertex of the parabola through
        its two neighbours, so the accuracy is no longer limited by the step of
        the ``theta`` grid (a 24-point grid would otherwise limit the null
        position to ~2 degrees).
        """
        profile = self.theta_profile()
        if profile.numel() < 3:
            return ()
        step = float(self.theta[1] - self.theta[0])
        angles: list[float] = []
        for i in range(1, profile.numel() - 1):
            left, centre, right = profile[i - 1], profile[i], profile[i + 1]
            if centre <= left and centre <= right:
                curvature = float(left - 2.0 * centre + right)
                offset = 0.0
                if curvature > 0.0:
                    offset = 0.5 * float(left - right) / curvature
                    offset = max(-0.5, min(0.5, offset))
                angles.append(float(self.theta[i]) + offset * step)
        return tuple(angles)


def _far_field_amplitudes(
    grid: RadiationGrid,
    velocity: torch.Tensor,
    *,
    frequency: float,
    rho: float,
    sound_speed: float,
    distance: float,
    theta: torch.Tensor,
    phi: torch.Tensor,
    depth: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """``p * distance`` for the image sum and for the free field (exact distances)."""
    points = grid.points
    weights = grid.weights
    k = 2.0 * math.pi * frequency / sound_speed
    sin_t = torch.sin(theta).reshape(-1, 1)
    cos_t = torch.cos(theta).reshape(-1, 1)
    obs_x = distance * sin_t * torch.cos(phi).reshape(1, -1)
    obs_y = distance * sin_t * torch.sin(phi).reshape(1, -1)
    obs_z = distance * cos_t * torch.ones_like(phi).reshape(1, -1)

    dx = obs_x.reshape(1, *obs_x.shape) - points[:, 0].reshape(-1, 1, 1)
    dy = obs_y.reshape(1, *obs_y.shape) - points[:, 1].reshape(-1, 1, 1)
    dz_direct = torch.broadcast_to(
        obs_z.reshape(1, *obs_z.shape), dx.shape
    ).clone()
    r_direct = torch.sqrt(dx * dx + dy * dy + dz_direct * dz_direct)
    direct = torch.exp(1j * k * r_direct) / r_direct
    weighted = weights * velocity
    amplitude_direct = (direct * weighted.reshape(-1, 1, 1)).sum(dim=0)
    if depth is None:
        return amplitude_direct, amplitude_direct
    dz_image = obs_z.reshape(1, *obs_z.shape) - 2.0 * depth
    dz_image = torch.broadcast_to(dz_image, dx.shape).clone()
    r_image = torch.sqrt(dx * dx + dy * dy + dz_image * dz_image)
    image = torch.exp(1j * k * r_image) / r_image
    amplitude_image = (image * weighted.reshape(-1, 1, 1)).sum(dim=0)
    return amplitude_direct - amplitude_image, amplitude_direct


def free_surface_power(
    grid: RadiationGrid,
    config: PlateModalConfig,
    *,
    surface_velocity: torch.Tensor,
    frequency: float,
    depth: Optional[float],
    rho: float = RHO_WATER,
    sound_speed: float = C_WATER,
    n_theta: int = 24,
    n_phi: int = 48,
    distance_factor: float = 2000.0,
) -> FreeSurfaceField:
    """Far-field pattern and radiated power below a free surface (``None``: none).

    ``depth = None`` is the free-field control: the same machinery with the image
    term dropped, which must reproduce the Rayleigh power of
    :mod:`tensorfem.struct_acoustics.radiation_coupling`.
    """
    frequency = _positive("frequency", frequency)
    rho = _positive("rho", rho)
    sound_speed = _positive("sound_speed", sound_speed)
    velocity = torch.as_tensor(surface_velocity).to(torch.complex128).reshape(-1)
    if velocity.numel() != grid.points.shape[0]:
        raise ValueError("surface_velocity length must match the grid")
    wavelength = sound_speed / frequency
    ranging = float(distance_factor) * wavelength
    theta, phi, d_omega = _half_space_directions(n_theta, n_phi)
    depth_value = None if depth is None else _positive("depth", depth)

    amplitude, free_amplitude = _far_field_amplitudes(
        grid,
        velocity,
        frequency=frequency,
        rho=rho,
        sound_speed=sound_speed,
        distance=ranging,
        theta=theta,
        phi=phi,
        depth=depth_value,
    )
    # ``_far_field_amplitudes`` returns sum(w v exp(ikR)/R); the physical far
    # field is p = -(i omega rho / 2 pi) * that sum, and the stored quantity is
    # ``p * distance`` so the range cancels out of the power integral.
    omega = 2.0 * math.pi * frequency
    scale = -1j * omega * rho * ranging / (2.0 * math.pi)
    amplitude = scale * amplitude
    free_amplitude = scale * free_amplitude

    def _power(values: torch.Tensor) -> float:
        return float(
            (values.abs() ** 2 * d_omega[:, None]).sum() / (2.0 * rho * sound_speed)
        )

    return FreeSurfaceField(
        depth=float(depth_value) if depth_value is not None else math.inf,
        frequency=frequency,
        theta=theta,
        phi=phi,
        amplitude=amplitude,
        free_amplitude=free_amplitude,
        d_omega=d_omega,
        power=_power(amplitude),
        free_field_power=_power(free_amplitude),
        distance=ranging,
    )


def free_surface_directivity(*args, **kwargs) -> FreeSurfaceField:
    """Alias of :func:`free_surface_power` (the object carries the pattern too)."""
    return free_surface_power(*args, **kwargs)


def free_surface_power_ratio_analytic(
    grid: RadiationGrid,
    config: PlateModalConfig,
    *,
    surface_velocity: torch.Tensor,
    frequency: float,
    depth: float,
    sound_speed: float = C_WATER,
    n_theta: int = 48,
    n_phi: int = 96,
) -> float:
    """``P_fs / P_free`` from the closed-form Lloyd factor ``4 sin^2(k d cos theta)``.

    This route never builds a kernel: the free-field angular spectrum ``A`` is a
    single spatial Fourier transform of the sampled velocity, and the free
    surface enters only through the analytic interference factor.  It is
    therefore independent of :func:`free_surface_power`, which integrates exact
    distances against a mirrored source.
    """
    frequency = _positive("frequency", frequency)
    depth = _positive("depth", depth)
    sound_speed = _positive("sound_speed", sound_speed)
    velocity = torch.as_tensor(surface_velocity).to(torch.complex128).reshape(-1)
    if velocity.numel() != grid.points.shape[0]:
        raise ValueError("surface_velocity length must match the grid")
    theta, phi, d_omega = _half_space_directions(n_theta, n_phi)
    k = 2.0 * math.pi * frequency / sound_speed
    points = grid.points
    weights = grid.weights
    sin_t = torch.sin(theta).reshape(-1, 1)
    cos_t = torch.cos(theta).reshape(-1, 1)
    kx = k * sin_t * torch.cos(phi).reshape(1, -1)
    ky = k * sin_t * torch.sin(phi).reshape(1, -1)
    kernel = torch.exp(
        -1j * (kx.reshape(*kx.shape, 1) * points[:, 0] + ky.reshape(*ky.shape, 1) * points[:, 1])
    )
    spectrum = (kernel * (weights * velocity).reshape(1, 1, -1)).sum(dim=-1)
    interference = 4.0 * torch.sin(k * depth * cos_t) ** 2 * torch.ones_like(phi).reshape(1, -1)
    numerator = (spectrum.abs() ** 2 * interference * d_omega[:, None]).sum()
    denominator = (spectrum.abs() ** 2 * d_omega[:, None]).sum()
    return float(numerator / denominator)