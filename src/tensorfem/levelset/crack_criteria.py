"""Mixed-mode crack propagation criteria, stress-intensity/energy release rate
conversions and tip kinetic laws.

The module supplies the *driving force* side of a level-set crack-growth loop:

* linear-elastic stress intensity factors of an inclined crack in an infinite
  plate under remote uniaxial stress (:func:`inclined_crack_sifs`) -- the
  analytic benchmark case;
* the two classical deflection criteria, maximum tangential (hoop) stress
  (:func:`kink_angle_max_hoop`) and maximum energy release rate
  (:func:`kink_angle_max_energy`), both returning the kink angle in radians;
* mode I/II stress intensity -> energy release rate conversion
  (:func:`energy_release_rate`) with the thickness correction for the
  plane-stress / plane-strain cases, plus the *plate-bending* counterpart
  (:func:`bending_energy_release_rate`) derived from the Kirchhoff
  plate/plane-stress analogy (see ``docs/levelset-ice-plate-fracture.md``);
* the crack-tip kinetic law ``v = v0 (G/Gc - 1)^m``
  (:func:`growth_velocity`) that turns the energy criterion into a
  level-set advection speed.

Conventions: ``K_I > 0`` opens the crack; ``K_II > 0`` shears the upper bank in
the ``+x`` direction for a crack along ``x``; the kink angle ``theta`` is
measured from the current crack direction, positive counter-clockwise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "KinkAngle",
    "inclined_crack_sifs",
    "hoop_stress_factor",
    "kink_angle_max_hoop",
    "kink_angle_max_energy",
    "energy_release_rate",
    "bending_energy_release_rate",
    "bending_moment_intensity",
    "growth_velocity",
    "propagation_direction",
    "critical_crack_length",
]

_SQRT_2PI = math.sqrt(2.0 * math.pi)


@dataclass(frozen=True)
class KinkAngle:
    """Kink angle (rad) plus the normalised driving force at that angle."""

    theta: float
    value: float
    criterion: str


def inclined_crack_sifs(sigma: float, beta: float, half_length: float) -> tuple[float, float]:
    """``(K_I, K_II)`` of an inclined crack under remote uniaxial tension.

    ``beta`` is the angle between the crack plane and the axis perpendicular to
    the remote stress (i.e. ``beta = 90 deg`` is a crack normal to the load,
    pure mode I).  For an infinite plate

        K_I  = sigma sin^2(beta) sqrt(pi a)
        K_II = sigma sin(beta) cos(beta) sqrt(pi a)
    """
    k0 = sigma * math.sqrt(math.pi * half_length)
    return k0 * math.sin(beta) ** 2, k0 * math.sin(beta) * math.cos(beta)


def hoop_stress_factor(K_I: float, K_II: float, theta: float) -> float:
    """Mode I angular factor of the hoop stress, ``sigma_theta_theta sqrt(2 pi r)``."""
    c = math.cos(0.5 * theta)
    return c * (K_I * c * c - 1.5 * K_II * math.sin(theta))


def kink_angle_max_hoop(K_I: float, K_II: float) -> KinkAngle:
    """Maximum tangential stress criterion (Erdogan & Sih 1963).

    The hoop stress ahead of the tip is ``sigma_tt = hoop_stress_factor/sqrt(2 pi r)``;
    the crack kinks along the direction that maximises it.  For pure mode II
    the result is the classical ``theta = -acos(1/3) = -70.53 deg``.
    """
    if abs(K_I) < 1.0e-30 and abs(K_II) < 1.0e-30:
        raise ValueError("at least one stress intensity factor must be non-zero")
    if abs(K_II) < 1.0e-14 * max(1.0, abs(K_I)):
        return KinkAngle(0.0, abs(K_I), "max_hoop_stress")
    # maximise over a bracket that contains the physical root; the hoop factor
    # is smooth with a single maximum in (-pi, pi) for K_I >= 0
    grid = np.linspace(-math.pi + 1e-6, math.pi - 1e-6, 4001)
    vals = np.array([hoop_stress_factor(K_I, K_II, t) for t in grid])
    k = int(np.argmax(vals))
    lo = grid[max(k - 1, 0)]
    hi = grid[min(k + 1, grid.size - 1)]
    # golden-section refinement of the local maximum
    phi = 0.5 * (math.sqrt(5.0) - 1.0)
    for _ in range(80):
        x1 = hi - phi * (hi - lo)
        x2 = lo + phi * (hi - lo)
        if hoop_stress_factor(K_I, K_II, x1) > hoop_stress_factor(K_I, K_II, x2):
            hi = x2
        else:
            lo = x1
    theta = 0.5 * (lo + hi)
    return KinkAngle(theta, hoop_stress_factor(K_I, K_II, theta), "max_hoop_stress")


def _kinked_sifs(K_I: float, K_II: float, theta: float) -> tuple[float, float]:
    """SIFs of an infinitesimally kinked crack (Erdogan & Sih 1963)."""
    c = math.cos(0.5 * theta)
    s = math.sin(0.5 * theta)
    k1 = c * (K_I * c * c - 3.0 * K_II * s * c)
    k2 = c * (K_I * s * c + K_II * (3.0 * c * c - 2.0))
    return k1, k2


def kink_angle_max_energy(K_I: float, K_II: float) -> KinkAngle:
    """Maximum energy release rate criterion (kinked-crack angular functions)."""
    if abs(K_I) < 1.0e-30 and abs(K_II) < 1.0e-30:
        raise ValueError("at least one stress intensity factor must be non-zero")
    grid = np.linspace(-math.pi + 1e-6, math.pi - 1e-6, 4001)
    vals = np.array([sum(v * v for v in _kinked_sifs(K_I, K_II, t)) for t in grid])
    k = int(np.argmax(vals))
    lo = grid[max(k - 1, 0)]
    hi = grid[min(k + 1, grid.size - 1)]
    phi = 0.5 * (math.sqrt(5.0) - 1.0)
    for _ in range(80):
        x1 = hi - phi * (hi - lo)
        x2 = lo + phi * (hi - lo)
        if sum(v * v for v in _kinked_sifs(K_I, K_II, x1)) > sum(
            v * v for v in _kinked_sifs(K_I, K_II, x2)
        ):
            hi = x2
        else:
            lo = x1
    theta = 0.5 * (lo + hi)
    return KinkAngle(
        theta, math.sqrt(sum(v * v for v in _kinked_sifs(K_I, K_II, theta))),
        "max_energy_release_rate",
    )


def energy_release_rate(K_I: float, K_II: float, young: float, poisson: float,
                        *, plane: str = "stress") -> float:
    """``G = (K_I^2 + K_II^2)/E'`` with ``E' = E`` (stress) or ``E/(1-nu^2)``."""
    if plane == "stress":
        e_eff = young
    elif plane == "strain":
        e_eff = young / (1.0 - poisson ** 2)
    else:
        raise ValueError("plane must be 'stress' or 'strain'")
    return (K_I ** 2 + K_II ** 2) / e_eff


def bending_moment_intensity(moment: float, half_length: float) -> float:
    """Bending moment intensity factor ``k1 = M sqrt(pi a)`` (infinite plate)."""
    return moment * math.sqrt(math.pi * half_length)


#: Far-field bending states of a through crack, expressed through the membrane
#: stress normal to the crack in the Kirchhoff/plane-stress analogy
#: ``sigma = M/(D (1 +- nu))``:
#:
#: ``"biaxial"``          M_nn = M_tt = M      -> sigma = M/(D (1+nu))
#: ``"cylindrical"``      M_tt = nu M_nn       -> sigma = M/D
#: ``"uniaxial_moment"``  M_tt = 0             -> sigma = M/(D (1-nu^2))
_MOMENT_STATE = {
    "biaxial": lambda nu: 1.0 / (1.0 + nu),
    "cylindrical": lambda nu: 1.0,
    "uniaxial_moment": lambda nu: 1.0 / (1.0 - nu ** 2),
}


def bending_energy_release_rate(moment: float, half_length: float, rigidity: float,
                                thickness: float, poisson: float = 0.3,
                                *, moment_state: str = "biaxial") -> float:
    """Energy release rate of a through crack in a Kirchhoff plate.

    With ``M`` the bending moment per unit width normal to the crack, the
    Kirchhoff/plane-stress analogy (``w <-> Airy function``, ``D <-> 1/E``,
    ``nu <-> -nu``) maps the plate crack onto a plane-stress mode I crack under
    ``sigma = c M / D``, so

        G = (1/h) sigma^2 pi a D  =  pi M^2 a c^2 /(D h)

    with ``c = 1/(1+nu)`` for equibiaxial bending, ``c = 1`` for the cylindrical
    state ``M_tt = nu M_nn`` and ``c = 1/(1-nu^2)`` for a pure uniaxial moment
    with a stress-free transverse edge.  Equivalently ``G = k1^2 c^2/(D h)``,
    which is the bending counterpart of the Griffith relation: integrating the
    linear through-thickness stress distribution contributes the ``1/3`` factor
    on the plane-stress result for the surface stress.
    """
    if thickness <= 0.0 or rigidity <= 0.0:
        raise ValueError("thickness and rigidity must be positive")
    try:
        c = _MOMENT_STATE[moment_state](poisson)
    except KeyError:
        raise ValueError(
            "moment_state must be 'biaxial', 'cylindrical' or 'uniaxial_moment'"
        ) from None
    return c ** 2 * bending_moment_intensity(moment, half_length) ** 2 / (rigidity * thickness)


def growth_velocity(G: float, G_c: float, v0: float = 1.0, exponent: float = 1.0) -> float:
    """Tip speed of the kinetic law ``v = v0 (G/G_c - 1)^m`` (zero below threshold)."""
    if G_c <= 0.0:
        raise ValueError("G_c must be positive")
    ratio = G / G_c - 1.0
    if ratio <= 0.0:
        return 0.0
    return v0 * ratio ** exponent


def propagation_direction(crack_direction: float, kink: float) -> np.ndarray:
    """Unit vector of the propagated crack from its direction and kink angle."""
    angle = crack_direction + kink
    return np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)


def critical_crack_length(G_c: float, moment: float, rigidity: float,
                          thickness: float) -> float:
    """Crack half-length at which ``G = G_c`` for a bending-Griffith crack."""
    denom = math.pi * moment ** 2
    if denom <= 0.0:
        raise ValueError("moment must be non-zero")
    return G_c * rigidity * thickness / denom
