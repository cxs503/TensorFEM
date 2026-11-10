"""Baffled-plate acoustics: three independent routes to the radiated power.

A thin plate vibrating in one structural mode radiates into the half space above
an infinite rigid baffle.  The radiation efficiency

    sigma = P_rad / (rho * c * S * <v^2>)

measures how effectively the structural vibration couples to the acoustic field.
This module supplies the acoustic half of the structure-acoustic chain; the
structural half (FEM eigenpairs, modal bookkeeping, band averages) lives in
:mod:`tensorfem.plate_radiation_chain`.

Three numerically independent evaluations of ``P_rad`` are provided for the same
sampled surface velocity ``v(x, y)``:

* :func:`rayleigh_power` - the 4-D Rayleigh surface integral
  ``P = rho*omega/(4*pi) * int int v1 v2 sin(k R12)/R12 dS1 dS2``.
* :func:`far_field_power` - far-field boundary-integral extrapolation, i.e. the
  complex pressure directivity ``A(theta, phi) = int v exp(-i k.r_s) dS``
  integrated over the radiating half space.
* :func:`spectral_power_sine_mode` - the closed-form angular spectrum of a
  simply-supported sine mode (no surface quadrature at all).

References
----------
* Rayleigh, J. W. S. (1945). *The Theory of Sound*, vol. 2, Dover.
* Maidanik, G. (1962). J. Acoust. Soc. Am. 34(6), 809-826.
* Leppington, F. G., Broadbent, E. G. & Heron, K. H. (1982).
  Proc. R. Soc. Lond. A 382, 245-271.
* Williams, E. G. (1999). *Fourier Acoustics*, Academic Press.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .acoustic_radiation import _gauss_legendre

__all__ = [
    "AIR_C",
    "AIR_RHO",
    "PlateAcousticConfig",
    "SurfaceVelocityField",
    "critical_frequency",
    "far_field_power",
    "far_field_pressure",
    "rayleigh_power",
    "sine_mode_field",
    "spectral_directivity_sine_mode",
    "spectral_power_sine_mode",
]

AIR_RHO = 1.225
AIR_C = 343.0


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _nonnegative(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


def _count(name: str, value: int, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _poisson(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("poisson must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 0.5:
        raise ValueError("poisson must be finite and in [0, 0.5)")
    return result


# ---------------------------------------------------------------------------
# Plate configuration and coincidence frequency
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlateAcousticConfig:
    """Geometry, structure and fluid of a baffled rectangular plate (SI units).

    The defaults are a 0.5 m x 0.5 m x 2 mm steel panel radiating into air, i.e.
    ``h/a = 0.004`` so that the thin-plate assumptions hold and the coincidence
    frequency sits near 6 kHz.
    """

    length: float = 0.5
    width: float = 0.5
    thickness: float = 0.002
    young: float = 210.0e9
    poisson: float = 0.3
    density_plate: float = 7850.0
    rho_fluid: float = AIR_RHO
    sound_speed: float = AIR_C

    def __post_init__(self) -> None:
        object.__setattr__(self, "length", _positive("length", self.length))
        object.__setattr__(self, "width", _positive("width", self.width))
        object.__setattr__(self, "thickness", _positive("thickness", self.thickness))
        object.__setattr__(self, "young", _positive("young", self.young))
        object.__setattr__(self, "poisson", _poisson(self.poisson))
        object.__setattr__(
            self, "density_plate", _positive("density_plate", self.density_plate)
        )
        object.__setattr__(self, "rho_fluid", _positive("rho_fluid", self.rho_fluid))
        object.__setattr__(
            self, "sound_speed", _positive("sound_speed", self.sound_speed)
        )

    @property
    def area(self) -> float:
        return self.length * self.width

    @property
    def bending_rigidity(self) -> float:
        return self.young * self.thickness ** 3 / (12.0 * (1.0 - self.poisson ** 2))

    @property
    def mass_per_area(self) -> float:
        return self.density_plate * self.thickness

    @property
    def longitudinal_wave_speed(self) -> float:
        """Plate longitudinal (quasi-dilatational) wave speed ``c_L``."""
        return math.sqrt(self.young / (self.density_plate * (1.0 - self.poisson ** 2)))

    @property
    def critical_frequency(self) -> float:
        """Coincidence frequency from the engineering formula ``c^2/(1.8 c_L h)``."""
        return self.sound_speed ** 2 / (
            1.8 * self.longitudinal_wave_speed * self.thickness
        )

    @property
    def coincidence_frequency_exact(self) -> float:
        """Coincidence frequency from the bending dispersion ``c_B(f_c) = c``.

        ``c_B = (D/(rho_s h))^{1/4} sqrt(omega)`` gives ``omega_c = c^2
        sqrt(rho_s h / D)``, i.e. ``c^2/(2 pi) sqrt(rho_s h/D)``; this differs
        from the 1/1.8 engineering factor by about 0.8 %.
        """
        return self.sound_speed ** 2 / (2.0 * math.pi) * math.sqrt(
            self.mass_per_area / self.bending_rigidity
        )

    def bending_wavenumber(self, frequency: float) -> float:
        """Plate bending wavenumber at ``frequency`` (Hz), in rad/m."""
        frequency = _positive("frequency", frequency)
        return math.sqrt(2.0 * math.pi * frequency) * (
            self.mass_per_area / self.bending_rigidity
        ) ** 0.25

    def acoustic_wavenumber(self, frequency: float) -> float:
        return 2.0 * math.pi * _positive("frequency", frequency) / self.sound_speed

    def mode_frequency(self, m: int, n: int) -> float:
        """Simply-supported Kirchhoff eigenvalue ``f_mn`` in Hz."""
        m = _count("m", m)
        n = _count("n", n)
        omega = (
            (m * math.pi / self.length) ** 2 + (n * math.pi / self.width) ** 2
        ) * math.sqrt(self.bending_rigidity / self.mass_per_area)
        return omega / (2.0 * math.pi)


def critical_frequency(
    *,
    sound_speed: float = AIR_C,
    thickness: float,
    young: float = 210.0e9,
    poisson: float = 0.3,
    density_plate: float = 7850.0,
) -> float:
    """Engineering coincidence frequency ``f_c = c^2/(1.8 c_L h)``.

    ``c_L = sqrt(E/(rho_s (1 - nu^2)))`` is the plate longitudinal wave speed.
    """
    c = _positive("sound_speed", sound_speed)
    h = _positive("thickness", thickness)
    young = _positive("young", young)
    poisson = _poisson(poisson)
    density_plate = _positive("density_plate", density_plate)
    c_longitudinal = math.sqrt(young / (density_plate * (1.0 - poisson ** 2)))
    return c ** 2 / (1.8 * c_longitudinal * h)


# ---------------------------------------------------------------------------
# Sampled surface velocity field
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SurfaceVelocityField:
    """Normal velocity of a baffled planar radiator on a tensor grid.

    ``velocity[i, j]`` is ``v(y_i, x_j)``.  The weights are quadrature weights
    for the unweighted surface integral, i.e. ``sum(weight_x) = length`` and
    ``sum(weight_y) = width``, so surface integrals and mean-square averages are
    plain weighted sums.
    """

    x: torch.Tensor
    y: torch.Tensor
    velocity: torch.Tensor
    weight_x: torch.Tensor
    weight_y: torch.Tensor
    label: str = ""

    def __post_init__(self) -> None:
        if self.velocity.ndim != 2:
            raise ValueError("velocity must be a 2-D (ny, nx) tensor")
        ny, nx = self.velocity.shape
        if self.x.numel() != nx or self.y.numel() != ny:
            raise ValueError("velocity shape must be (len(y), len(x))")
        if self.weight_x.numel() != nx or self.weight_y.numel() != ny:
            raise ValueError("quadrature weights do not match the grid")

    @property
    def surface(self) -> float:
        return float(self.weight_x.sum() * self.weight_y.sum())

    @property
    def weight(self) -> torch.Tensor:
        return self.weight_y[:, None] * self.weight_x[None, :]

    @property
    def mean_square_velocity(self) -> float:
        return float((self.velocity ** 2 * self.weight).sum()) / self.surface

    @property
    def volume_velocity(self) -> float:
        return float((self.velocity * self.weight).sum())

    @property
    def rms_velocity(self) -> float:
        return math.sqrt(self.mean_square_velocity)

    def normalized(
        self, rms: float = 1.0, label: Optional[str] = None
    ) -> "SurfaceVelocityField":
        """Return a copy scaled to the requested RMS velocity (default 1 m/s)."""
        rms = _positive("rms", rms)
        current = self.rms_velocity
        if current <= 0.0:
            raise ValueError("cannot normalize a field with zero velocity")
        return SurfaceVelocityField(
            x=self.x,
            y=self.y,
            velocity=self.velocity * (rms / current),
            weight_x=self.weight_x,
            weight_y=self.weight_y,
            label=self.label if label is None else label,
        )


def _gauss_grid(
    length: float, width: float, points: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    points = _count("points", points, minimum=2)
    nodes, weights = _gauss_legendre(points)
    x = 0.5 * length * (nodes + 1.0)
    y = 0.5 * width * (nodes + 1.0)
    wx = 0.5 * length * weights
    wy = 0.5 * width * weights
    return x, y, wx, wy


def sine_mode_field(
    config: PlateAcousticConfig,
    m: int,
    n: int,
    *,
    points: int = 48,
    velocity_amplitude: float = 1.0,
) -> SurfaceVelocityField:
    """Simply-supported sine mode ``sin(m pi x/a) sin(n pi y/b)`` on a Gauss grid."""
    m = _count("m", m)
    n = _count("n", n)
    velocity_amplitude = _nonnegative("velocity_amplitude", velocity_amplitude)
    x, y, wx, wy = _gauss_grid(config.length, config.width, points)
    velocity = velocity_amplitude * torch.sin(
        m * math.pi * x / config.length
    )[None, :] * torch.sin(n * math.pi * y / config.width)[:, None]
    return SurfaceVelocityField(
        x=x, y=y, velocity=velocity, weight_x=wx, weight_y=wy, label=f"sine({m},{n})"
    )


# ---------------------------------------------------------------------------
# Route A: 4-D Rayleigh surface integral
# ---------------------------------------------------------------------------

def _tile_axis(axis: torch.Tensor, repeat: int, *, vary_first: bool) -> torch.Tensor:
    """Tile a 1-D axis over a ``n`` x ``repeat`` grid, C-order flattened.

    ``vary_first`` says whether the axis is the *fast* (last) index of the field,
    i.e. whether its values have to be repeated along the slow index.  Both
    branches used to expand to ``(repeat, axis.numel())``, which only produced a
    conforming shape when ``repeat == axis.numel()``: ``rayleigh_power`` and
    ``far_field_power`` therefore raised on any grid with ``nx != ny`` (see the
    non-square-grid regression in ``tests/test_struct_acoustic_coupling.py``).
    """
    if vary_first:
        tiled = axis[:, None].expand(axis.numel(), repeat)
    else:
        tiled = axis[None, :].expand(repeat, axis.numel())
    return tiled.reshape(-1)


def rayleigh_power(
    surface: SurfaceVelocityField,
    *,
    frequency: float,
    rho: float = AIR_RHO,
    c: float = AIR_C,
    chunk: int = 64,
) -> float:
    """Radiated power of a sampled velocity field by the Rayleigh integral.

    ``P = rho*omega/(4*pi) * int_S int_S v(r1) v(r2) sin(k R12)/R12 dS1 dS2``

    The ``sin(kR)/R`` kernel (not ``cos``) is the real part of ``j*exp(-jkR)``;
    it vanishes on the diagonal so there is no singular self-term.  The 4-D sum
    is evaluated in chunks of source points to bound memory; the cost is
    ``O(N^2)`` in the number of grid points.
    """
    frequency = _positive("frequency", frequency)
    rho = _positive("rho", rho)
    c = _positive("c", c)
    chunk = _count("chunk", chunk)
    k = 2.0 * math.pi * frequency / c
    omega = 2.0 * math.pi * frequency

    ny, nx = surface.velocity.shape
    xs = _tile_axis(surface.x, ny, vary_first=False)
    ys = _tile_axis(surface.y, nx, vary_first=True)
    v = surface.velocity.reshape(-1)
    w = surface.weight.reshape(-1)

    total = torch.zeros((), dtype=torch.float64)
    for start in range(0, v.numel(), chunk):
        stop = min(start + chunk, v.numel())
        dx = xs[start:stop, None] - xs[None, :]
        dy = ys[start:stop, None] - ys[None, :]
        radius = torch.sqrt(dx * dx + dy * dy)
        kernel = k * torch.sinc(k * radius / math.pi)
        weight = (v[start:stop, None] * w[start:stop, None]) * (v[None, :] * w[None, :])
        total = total + (weight * kernel).sum()
    return float(rho * omega / (4.0 * math.pi) * total)


# ---------------------------------------------------------------------------
# Route B: far-field boundary-integral extrapolation
# ---------------------------------------------------------------------------

def _half_space_directions(
    n_theta: int, n_phi: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quadrature for the radiating half space.

    ``theta`` is measured from the plate normal.  Gauss-Legendre nodes in
    ``u = cos(theta)`` on ``[0, 1]`` avoid the grazing endpoint entirely;
    ``phi`` is trapezoidal (periodic, hence spectrally accurate).
    """
    n_theta = _count("n_theta", n_theta, minimum=2)
    n_phi = _count("n_phi", n_phi, minimum=4)
    nodes, weights = _gauss_legendre(n_theta)
    u = 0.5 * (nodes + 1.0)
    wu = 0.5 * weights
    phi = 2.0 * math.pi * torch.arange(n_phi, dtype=torch.float64) / n_phi
    sin_theta = torch.sqrt(torch.clamp(1.0 - u * u, min=0.0))
    d_omega = wu[:, None] * (2.0 * math.pi / n_phi)
    return sin_theta, phi, d_omega


def _wavenumbers(
    frequency: float, c: float, n_theta: int, n_phi: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k = 2.0 * math.pi * _positive("frequency", frequency) / _positive("c", c)
    sin_theta, phi, d_omega = _half_space_directions(n_theta, n_phi)
    kx = k * sin_theta[:, None] * torch.cos(phi)[None, :]
    ky = k * sin_theta[:, None] * torch.sin(phi)[None, :]
    return kx, ky, d_omega


def _directivity(
    surface: SurfaceVelocityField, kx: torch.Tensor, ky: torch.Tensor
) -> torch.Tensor:
    """``A(kx, ky) = sum_ij w_y w_x v_ij exp(-i(kx x_j + ky y_i))`` (complex)."""
    shape = kx.shape
    ndir = kx.numel()
    ex = surface.weight_x[None, :] * torch.exp(
        -1j * kx.reshape(ndir, 1) * surface.x[None, :]
    )
    ey = surface.weight_y[None, :] * torch.exp(
        -1j * ky.reshape(ndir, 1) * surface.y[None, :]
    )
    inner = ex @ surface.velocity.to(ex.dtype).T
    return (ey * inner).sum(dim=1).reshape(shape)


def far_field_power(
    surface: SurfaceVelocityField,
    *,
    frequency: float,
    rho: float = AIR_RHO,
    c: float = AIR_C,
    n_theta: int = 48,
    n_phi: int = 256,
) -> float:
    """Radiated power from the far-field pressure directivity.

    The far-field (``r -> infinity``) pressure of a baffled source is

        ``p(r) = i*omega*rho/(2*pi*r) * exp(i k r) * A(theta, phi)``,
        ``A = int_S v(r_s) exp(-i k rh.r_s) dS``,

    so ``P = rho*omega^2/(8*pi^2*c) * int |A|^2 dOmega``; the distance ``r``
    cancels.  For a planar baffled source this integral reproduces the exact
    Rayleigh power, because only the supersonic part of the angular spectrum
    radiates.
    """
    frequency = _positive("frequency", frequency)
    rho = _positive("rho", rho)
    c = _positive("c", c)
    omega = 2.0 * math.pi * frequency
    kx, ky, d_omega = _wavenumbers(frequency, c, n_theta, n_phi)
    amplitude = _directivity(surface, kx, ky)
    integral = (amplitude.abs() ** 2 * d_omega).sum()
    return float(rho * omega ** 2 / (8.0 * math.pi ** 2 * c) * integral)


def far_field_pressure(
    surface: SurfaceVelocityField,
    *,
    frequency: float,
    distance: float,
    theta: float,
    phi: float,
    rho: float = AIR_RHO,
    c: float = AIR_C,
) -> complex:
    """Far-field pressure at ``(distance, theta, phi)`` for a baffled radiator."""
    frequency = _positive("frequency", frequency)
    distance = _positive("distance", distance)
    rho = _positive("rho", rho)
    c = _positive("c", c)
    if not math.isfinite(theta) or not math.isfinite(phi):
        raise ValueError("theta and phi must be finite")
    k = 2.0 * math.pi * frequency / c
    omega = 2.0 * math.pi * frequency
    theta_t = torch.tensor([[theta]], dtype=torch.float64)
    phi_t = torch.tensor([[phi]], dtype=torch.float64)
    kx = k * torch.sin(theta_t) * torch.cos(phi_t)
    ky = k * torch.sin(theta_t) * torch.sin(phi_t)
    amplitude = _directivity(surface, kx, ky).reshape(())
    return complex(
        1j * omega * rho / (2.0 * math.pi * distance)
        * torch.exp(1j * k * torch.tensor(distance, dtype=torch.float64))
        * amplitude
    )


# ---------------------------------------------------------------------------
# Route C: closed-form angular spectrum of a sine mode
# ---------------------------------------------------------------------------

def _sine_transform(
    wavenumber: torch.Tensor, length: float, half_waves: int
) -> torch.Tensor:
    """``int_0^L sin(m*pi*x/L) exp(-i*kappa*x) dx`` in closed form.

    With ``q = m*pi/L`` the antiderivative is
    ``q*(1-(-1)^m exp(-i kappa L))/(q^2 - kappa^2)``; at the removable
    singularity ``kappa = q`` the limit is ``-i L/2``.
    """
    q = half_waves * math.pi / length
    discriminant = q * q - wavenumber * wavenumber
    numerator = q * (1.0 - (-1) ** half_waves * torch.exp(-1j * wavenumber * length))
    threshold = 1e-9 * q * q
    nonsingular = torch.abs(discriminant) > threshold
    value = numerator / torch.where(
        nonsingular, discriminant, torch.ones_like(discriminant)
    )
    limit = torch.full(
        wavenumber.shape, -0.5j * length, dtype=torch.complex128
    )
    return torch.where(nonsingular, value, limit)


def spectral_directivity_sine_mode(
    config: PlateAcousticConfig,
    m: int,
    n: int,
    kx: torch.Tensor,
    ky: torch.Tensor,
    *,
    velocity_amplitude: float = 1.0,
) -> torch.Tensor:
    """Closed-form angular spectrum ``A(kx, ky)`` of a sine mode."""
    m = _count("m", m)
    n = _count("n", n)
    velocity_amplitude = _nonnegative("velocity_amplitude", velocity_amplitude)
    return velocity_amplitude * _sine_transform(
        kx, config.length, m
    ) * _sine_transform(ky, config.width, n)


def spectral_power_sine_mode(
    config: PlateAcousticConfig,
    m: int,
    n: int,
    *,
    frequency: float,
    n_theta: int = 48,
    n_phi: int = 256,
    velocity_amplitude: float = 1.0,
) -> float:
    """Radiated power of one sine mode from the closed-form angular spectrum.

    No surface quadrature is involved: the angular spectrum is analytic and only
    the half-space angular integral is discretised.  Its resolution has to grow
    with ``k * max(a, b)``; convergence is pinned in
    ``tests/test_plate_radiation_chain.py``.
    """
    frequency = _positive("frequency", frequency)
    velocity_amplitude = _nonnegative("velocity_amplitude", velocity_amplitude)
    omega = 2.0 * math.pi * frequency
    kx, ky, d_omega = _wavenumbers(frequency, config.sound_speed, n_theta, n_phi)
    amplitude = spectral_directivity_sine_mode(
        config, m, n, kx, ky, velocity_amplitude=velocity_amplitude
    )
    integral = (amplitude.abs() ** 2 * d_omega).sum()
    return float(
        config.rho_fluid * omega ** 2 / (8.0 * math.pi ** 2 * config.sound_speed)
        * integral
    )