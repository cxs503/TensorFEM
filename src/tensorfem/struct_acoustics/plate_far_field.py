"""Far field of a baffled circular piston / compact plate: Rayleigh integral vs closed forms.

Physical background
-------------------
A plane circular source of radius ``a`` vibrating with uniform normal velocity
``v0`` (peak, ``e^{-i omega t}``) radiates into a half space.  The campaign node
quotes the far-field pressure of a "baffled plate" as

.. math:: p(r,\\theta) = i\\,\\rho c k v_0 \\frac{a^2}{2r}\\cos\\theta\\,e^{ikr}
          \\qquad\\text{(node specification)}

Two things are wrong with the cosine factor, and this module measures both
instead of asserting them:

1. For a source in an **infinite rigid baffle** the far field follows from the
   Rayleigh integral ``p = i\\omega\\rho/(2\\pi) \\int_S v e^{ikR}/R\\,dS`` and its
   directivity is the Airy pattern

   .. math:: D(\\theta) = \\frac{2J_1(ka\\sin\\theta)}{ka\\sin\\theta},

   i.e. ``D = 1`` on axis, **no** ``cos theta`` factor.  The node formula is
   therefore exact only on axis (where ``cos 0 = D(0) = 1``).

2. The ``cos theta`` lobe is the signature of a **dipole** source (equal and
   opposite faces, no baffle) -- the classic compact-source directivity.  A
   vanishingly thin unbaffled plate radiates a dipole field whose magnitude is
   *proportional to the plate thickness* and therefore collapses in the
   thin-plate limit; this is why the unbaffled/``cos theta`` semiconductor is
   not the right anchor for a wetted hull panel in a baffle-like arrangement.

What is verified here
---------------------
* :func:`rayleigh_far_field` -- direct numerical Rayleigh quadrature (Gauss
  radial x trapezoid azimuthal, *exact* kernel ``e^{ikR}/R``, no far-field
  approximation) for a user-supplied field point.
* :func:`on_axis_exact` -- the exact on-axis closed form
  ``p = rho c v0 (e^{ik sqrt(r^2+a^2)} - e^{ikr})``, obtained by integrating the
  Rayleigh kernel analytically.  This pins the quadrature (amplitude, phase and
  sign) without any far-field approximation.
* :func:`baffled_piston_far_field` -- the closed-form Airy far field above; the
  numerical quadrature must approach it as ``a/r -> 0``.
* :func:`node_cos_theta_far_field` -- the node specification, retained so the
  report can quantify the deviation of ``cos theta`` from ``D(theta)``.
* :func:`piston_radiation_efficiency` -- ``sigma = 1 - J1(2ka)/(ka)``, derived
  by integrating the far-field intensity of ``D(theta)`` over the half space.

References
----------
* Pierce, A. D. (1989). *Acoustics*. Sections 5.3-5.5 (baffled piston).
* Rayleigh, J. W. S. (1896). *Theory of Sound*, Vol. II, Sec. 302.
* Fahy, F. & Gardonio, P. (2007). *Sound and Structural Vibration*, Ch. 4.
* Abramowitz & Stegun (1964). *Handbook of Mathematical Functions*, 9.4.
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import torch

RHO_WATER = 1000.0
C_WATER = 1500.0

_GL_X = (
    -0.9602898564975363, -0.7966664774136267, -0.5255324099163290,
    -0.1834346424956498, 0.1834346424956498, 0.5255324099163290,
    0.7966664774136267, 0.9602898564975363,
)
_GL_W = (
    0.1012285362903763, 0.2223810344533745, 0.3137066458778873,
    0.3626837833783620, 0.3626837833783620, 0.3137066458778873,
    0.2223810344533745, 0.1012285362903763,
)


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return out


def _angle(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    out = float(value)
    if not math.isfinite(out) or out < 0.0 or out > 0.5 * math.pi:
        raise ValueError(f"{name} must lie in [0, pi/2]")
    return out


def wavenumber(frequency: float, c: float = C_WATER) -> float:
    return 2.0 * math.pi * _positive("frequency", frequency) / _positive("c", c)


# ---------------------------------------------------------------------------
# Bessel J1 (power series for |x| < 12, A&S 9.4.6 asymptotic above)
# ---------------------------------------------------------------------------

def bessel_j1(x):
    """First-order Bessel function ``J1`` for real ``x`` (float or tensor).

    Power series ``sum_m (-1)^m (x/2)^(2m+1)/(m! (m+1)!)`` for ``|x| < 12`` and
    the Abramowitz & Stegun 9.4.6 asymptotic form above ``|x| = 12`` (its error
    there is < 1e-10).  The series avoids the ~1e-7 error that the A&S 9.4.4
    polynomial has at ``x -> 3``, which matters for ``J1(2ka)/(ka)`` near
    ``ka ~ 1.5``.
    """
    t = torch.as_tensor(x, dtype=torch.float64)
    ax = torch.abs(t)
    y = 0.25 * t * t
    term = 0.5 * t
    series = term.clone()
    for m in range(1, 40):
        term = -term * y / (m * (m + 1))
        series = series + term
        if m > 20 and bool(torch.all(torch.abs(term) < 1e-18 * torch.clamp(ax, min=1.0))):
            break
    z = 3.0 / torch.clamp(ax, min=1e-12)
    f = (
        0.79788456
        + 0.00000156 * z
        + 0.01659667 * z**2
        + 0.00017105 * z**3
        - 0.00249511 * z**4
        + 0.00113653 * z**5
        - 0.00020033 * z**6
    )
    theta = ax - 2.35619449 + (
        0.12499612 * z
        + 0.00005650 * z**2
        - 0.00637879 * z**3
        + 0.00074348 * z**4
        + 0.00079824 * z**5
        - 0.00020662 * z**6
    )
    asymptotic = torch.sqrt(1.0 / torch.clamp(ax, min=1e-12)) * f * torch.cos(theta)
    out = torch.where(ax < 12.0, series, asymptotic)
    return out if out.ndim else float(out)


# ---------------------------------------------------------------------------
# Directivity
# ---------------------------------------------------------------------------

def piston_directivity(k: float, radius: float, theta: float) -> float:
    """Airy directivity ``D = 2 J1(ka sin theta)/(ka sin theta)`` (``D(0) = 1``)."""
    x = k * _positive("radius", radius) * math.sin(_angle("theta", theta))
    if abs(x) < 1e-9:
        return 1.0 - x * x / 8.0
    return float(2.0 * bessel_j1(x) / x)


def baffled_piston_far_field(
    r, theta, radius: float, frequency: float, v0: float,
    rho: float = RHO_WATER, c: float = C_WATER,
):
    """Closed-form far field of a baffled circular piston:
    ``p = i rho c k v0 a^2/(2r) D(theta) e^{ikr}``."""
    a = _positive("radius", radius)
    k = wavenumber(frequency, c)
    rho = _positive("rho", rho)
    c = _positive("c", c)
    rr = torch.as_tensor(r, dtype=torch.float64)
    if torch.any(rr <= 0.0):
        raise ValueError("r must be positive")
    th = _angle("theta", theta)
    amp = 1.0j * rho * c * k * complex(v0) * a * a / (2.0 * rr)
    field = amp * piston_directivity(k, a, th) * torch.exp(1.0j * k * rr)
    return field if field.ndim else complex(field)


def node_cos_theta_far_field(
    r, theta, radius: float, frequency: float, v0: float,
    rho: float = RHO_WATER, c: float = C_WATER,
):
    """The campaign-node specification ``i rho c k v0 a^2 cos(theta) e^{ikr}/(2r)``."""
    a = _positive("radius", radius)
    k = wavenumber(frequency, c)
    rr = torch.as_tensor(r, dtype=torch.float64)
    if torch.any(rr <= 0.0):
        raise ValueError("r must be positive")
    th = _angle("theta", theta)
    amp = (
        1.0j * _positive("rho", rho) * _positive("c", c) * k * complex(v0) * a * a
        / (2.0 * rr)
    )
    field = amp * math.cos(th) * torch.exp(1.0j * k * rr)
    return field if field.ndim else complex(field)


# ---------------------------------------------------------------------------
# Numerical Rayleigh integral (no far-field approximation)
# ---------------------------------------------------------------------------

def _disc_quadrature(radius: float, radial: int, azimuthal: int) -> Tuple[torch.Tensor, ...]:
    a = _positive("radius", radius)
    if int(radial) < 2 or int(azimuthal) < 8:
        raise ValueError('need radial >= 2 and azimuthal >= 8 quadrature points')
    nphi = int(azimuthal)
    x = torch.tensor(_GL_X, dtype=torch.float64)
    w = torch.tensor(_GL_W, dtype=torch.float64)
    # Map Gauss nodes from [-1, 1] to [0, a]; substitute u = rho^2 to remove the
    # sqrt-free planar measure singularity at the centre.
    rho = 0.5 * a * (x + 1.0)
    weight_rho = 0.5 * a * w * rho  # polar Jacobian rho d(rho): dA = rho drho dphi
    phi = torch.arange(nphi, dtype=torch.float64) * (2.0 * math.pi / nphi)
    weight_phi = torch.full((nphi,), 2.0 * math.pi / nphi, dtype=torch.float64)
    R, P = torch.meshgrid(rho, phi, indexing="ij")
    WR, WP = torch.meshgrid(weight_rho, weight_phi, indexing="ij")
    return R.reshape(-1), P.reshape(-1), (WR * WP).reshape(-1)


def rayleigh_far_field(
    r: float,
    theta: float,
    radius: float,
    frequency: float,
    v0: float = 1.0,
    rho: float = RHO_WATER,
    c: float = C_WATER,
    radial: int = 96,
    azimuthal: int = 256,
) -> complex:
    """Rayleigh-integral pressure at ``(r, theta)`` with the exact kernel.

    ``p = i omega rho v0/(2 pi) \\int_S e^{ikR}/R dS``, ``R`` the exact distance
    from the field point to each source point.  No ``kr >> 1`` approximation is
    made, so the numerical value approaches the closed-form far field only as
    ``a/r -> 0`` -- that decay is itself part of the evidence.
    """
    a = _positive("radius", radius)
    k = wavenumber(frequency, c)
    omega = 2.0 * math.pi * _positive("frequency", frequency)
    rho = _positive("rho", rho)
    rr = _positive("r", r)
    th = _angle("theta", theta)
    rho_s, phi_s, w_s = _disc_quadrature(a, radial, azimuthal)
    xf = rr * math.sin(th)
    zf = rr * math.cos(th)
    xs = rho_s * torch.cos(phi_s)
    dist = torch.sqrt((xf - xs) ** 2 + (rho_s * torch.sin(phi_s)) ** 2 + zf * zf)
    kernel = torch.exp(1.0j * k * dist) / dist
    integral = torch.sum(kernel * w_s)
    return complex(1.0j * omega * rho * complex(v0) / (2.0 * math.pi) * integral)


def on_axis_exact(
    r: float, radius: float, frequency: float, v0: float = 1.0,
    rho: float = RHO_WATER, c: float = C_WATER,
) -> complex:
    """Exact on-axis pressure of a baffled piston:
    ``p = rho c v0 (e^{ik sqrt(r^2+a^2)} - e^{ikr})``."""
    a = _positive("radius", radius)
    k = wavenumber(frequency, c)
    rr = _positive("r", r)
    pref = _positive("rho", rho) * _positive("c", c) * complex(v0)
    return complex(
        pref
        * (
            cmath.exp(1.0j * k * math.sqrt(rr * rr + a * a))
            - cmath.exp(1.0j * k * rr)
        )
    )


def piston_radiation_efficiency(ka: float) -> float:
    """``sigma = 1 - J1(2ka)/(ka)`` (exact for a baffled circular piston)."""
    x = _positive("ka", ka)
    return float(1.0 - bessel_j1(2.0 * x) / x)


def far_field_power_from_directivity(
    radius: float, frequency: float, v0: float,
    rho: float = RHO_WATER, c: float = C_WATER, samples: int = 2001,
) -> float:
    """Integrate the far-field intensity over the half space.

    ``W = 2 pi r^2 \\int_0^{pi/2} |p(r,theta)|^2/(2 rho c) sin theta d theta``;
    the ``r`` cancels, so this is the closed-form power of the Airy field.
    """
    a = _positive("radius", radius)
    k = wavenumber(frequency, c)
    rho = _positive("rho", rho)
    c = _positive("c", c)
    n = max(101, int(samples))
    theta = torch.linspace(0.0, 0.5 * math.pi, n, dtype=torch.float64)
    x = k * a * torch.sin(theta)
    pattern = torch.where(
        x.abs() < 1e-9,
        torch.ones_like(x),
        2.0 * bessel_j1(x) / torch.where(x.abs() < 1e-9, torch.ones_like(x), x),
    )
    # |p|^2 r^2 = (rho c k v0 a^2 / 2)^2 D^2
    scale = (rho * c * k * abs(complex(v0)) * a * a / 2.0) ** 2
    integrand = scale * pattern**2 * torch.sin(theta)
    w = torch.trapezoid(integrand, theta)
    return float(2.0 * math.pi / (2.0 * rho * c) * w)


def angle_sweep_error(
    radius: float,
    frequency: float,
    angles: Iterable[float],
    distance_ratio: float = 200.0,
    v0: float = 1.0,
    rho: float = RHO_WATER,
    c: float = C_WATER,
) -> list[dict]:
    """Compare numerical Rayleigh field to the closed-form Airy far field.

    Returns one row per angle with the magnitude error, the phase error and the
    deviation of the node's ``cos theta`` specification from ``D(theta)``.
    """
    a = _positive("radius", radius)
    r = _positive("distance_ratio", distance_ratio) * a
    k = wavenumber(frequency, c)
    rows = []
    for th_deg in angles:
        th = _angle("theta_deg", math.radians(float(th_deg)))
        num = rayleigh_far_field(r, th, a, frequency, v0, rho, c)
        closed = baffled_piston_far_field(r, th, a, frequency, v0, rho, c)
        node = node_cos_theta_far_field(r, th, a, frequency, v0, rho, c)
        rows.append(
            {
                "theta_deg": float(th_deg),
                "ka": k * a,
                "ka_sin_theta": k * a * math.sin(th),
                "numerical_magnitude": abs(num),
                "closed_form_magnitude": abs(closed),
                "magnitude_relative_error": abs(abs(num) - abs(closed)) / abs(closed),
                "phase_error_deg": float(
                    torch.rad2deg(
                        torch.angle(
                            torch.tensor(num / closed, dtype=torch.complex128)
                        )
                    )
                ),
                "directivity": piston_directivity(k, a, th),
                "node_cos_theta_relative_error": abs(
                    abs(node) - abs(closed)
                ) / abs(closed),
            }
        )
    return rows


@dataclass(frozen=True)
class PistonPowerCheck:
    """Closed-form vs directivity-integrated power of a baffled piston."""

    ka: float
    sigma_closed_form: float
    sigma_from_directivity: float
    power_closed_form: float
    power_from_directivity: float

    @property
    def relative_error(self) -> float:
        return abs(self.power_closed_form - self.power_from_directivity) / abs(
            self.power_closed_form
        )


def piston_power_check(
    radius: float, frequency: float, v0: float = 1.0,
    rho: float = RHO_WATER, c: float = C_WATER,
) -> PistonPowerCheck:
    """Cross-check ``W = rho c S <v^2> sigma`` against the directivity integral."""
    a = _positive("radius", radius)
    k = wavenumber(frequency, c)
    ka = k * a
    sigma = piston_radiation_efficiency(ka)
    area = math.pi * a * a
    mean_square = 0.5 * abs(complex(v0)) ** 2
    power_cf = _positive("rho", rho) * _positive("c", c) * area * mean_square * sigma
    power_d = far_field_power_from_directivity(a, frequency, v0, rho, c)
    sigma_d = power_d / (
        _positive("rho", rho) * _positive("c", c) * area * mean_square
    )
    return PistonPowerCheck(
        ka=ka,
        sigma_closed_form=sigma,
        sigma_from_directivity=sigma_d,
        power_closed_form=power_cf,
        power_from_directivity=power_d,
    )