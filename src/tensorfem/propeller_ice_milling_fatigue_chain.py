"""Propeller ice-milling high-cycle fatigue chain (campaign node P33-W2-N05).

Chain: ice fragment -> Hertz contact force -> blade stress -> rainflow cycle
count -> S-N / Miner damage, for an ice-class propeller blade in milling mode.

    S1  rigid ellipsoidal ice fragment vs blade leading edge
        F = k_contact delta**1.5 (Hertz) -> F_max, penetration, contact time
    S2  blade modal analysis: clamped-free Hermite beam FE (per-station chord
        and thickness), dry and wet (2-D flat-plate added mass)
    S3  modal superposition of the contact pulse train -> root bending stress
        sigma(t); independent check = direct Newmark-beta on the full FE system
    S4  rainflow cycle counting (ASTM E1049 four-point stack, inherited)
    S5  DNV-RP-C203 curve D in air + Miner  D = sum n_i / N_i
    S6  sweep over ice size L_ice, shaft speed n, advance speed V

Analytic anchors
----------------
A1 Hertz: closed form re-derived numerically from the Boussinesq half-space
   integral of p(r) = p0 sqrt(1-(r/a)^2): quadrature force vs closed form
   < 1e-3 N and uniform surface displacement over the patch (< 1 %).
A2 Blade modes: first five dry cantilever frequencies vs
   f_n = (beta_n L)^2/(2 pi L^2) sqrt(EI/(rho A))  (< 5 % criterion).
   Wet frequencies vs f_dry sqrt(m/(m+m_a)) for uniform added mass.
A3 Stress: modal superposition vs direct Newmark-beta (< 3 % peak/RMS).
A4 Rainflow: sinusoidal history of known cycle count, count error < 1.
A5 S-N: curve D vs DNV-RP-C203 table values (< 5 %).

Deviation (declared): the blade is a bent cantilever beam, not a shell model.
The stage-2 acceptance anchor is the Euler-Bernoulli closed form, which the
beam model can be held to exactly; a twisted aerofoil shell model has no
closed-form reference.  Root stress is a nominal bending stress (no fillet
stress-concentration factor).  The ice is rigid and elastic crushing is not
modelled, so the peak engagement force is stiffness/mass controlled.

References: Johnson (1985) Contact Mechanics ch.4; Blevins (1979) table 8-1;
Newmark (1959); ASTM E1049-85; DNV-RP-C203 (2021) table 2-1; IACS UR I3 for
the design-force scale context only.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .marine_fatigue_advanced import RainflowCycle, rainflow_cycles
from .weld_fatigue_ice import DNVSNCurve

__all__ = [
    "effective_contact_modulus",
    "equivalent_contact_radius",
    "ellipsoid_pole_radius",
    "ice_fragment_geometry",
    "hertz_contact_radius",
    "hertz_peak_pressure",
    "hertz_force",
    "hertz_pressure_quadrature",
    "HertzEngagement",
    "hertz_engagement",
    "hertz_impact_duration",
    "BladeGeometry",
    "blade_section_properties",
    "cantilever_beam_modes",
    "euler_bernoulli_frequency",
    "EULER_BERNOULLI_BETA_L",
    "flat_plate_added_mass_per_length",
    "wet_mode_frequencies",
    "BladeModalBasis",
    "blade_modal_basis",
    "root_curvature_from_dofs",
    "half_sine_pulse",
    "pulse_train",
    "modal_stress_history",
    "newmark_stress_history",
    "rayleigh_damping",
    "sine_stress_history",
    "miner_damage",
    "dnv_d_curve_in_air",
    "DNV_D_TABLE",
    "MillingCondition",
    "milling_engagement",
    "milling_damage_case",
    "milling_bounding_case",
    "milling_parameter_sweep",
]

ICE_DENSITY_KG_M3 = 917.0
ICE_YOUNG_MODULUS_PA = 9.0e9
ICE_POISSON = 0.30
BLADE_YOUNG_MODULUS_PA = 118.0e9  # nickel-aluminium-bronze
BLADE_POISSON = 0.34
BLADE_DENSITY_KG_M3 = 7640.0
SEAWATER_DENSITY_KG_M3 = 1025.0
WAKE_FRACTION = 0.20
DESIGN_LIFE_YEARS = 20.0

EULER_BERNOULLI_BETA_L = (
    1.875104068711961,
    4.694091132974174,
    7.854757438237613,
    10.995540734875467,
    14.137168391046471,
)

DNV_D_TABLE = {
    "log_a1": 12.164,
    "slope_m1": 3.0,
    "log_a2": 15.606,
    "slope_m2": 5.0,
    "knee_cycles": 1.0e7,
    "fat_mpa_at_2e6": 90.0,
    "knee_stress_mpa": 52.63,
}


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return value


# --------------------------------------------------------------------------
# S1: contact geometry and the Hertz law
# --------------------------------------------------------------------------


def effective_contact_modulus(
    e1: float = ICE_YOUNG_MODULUS_PA,
    nu1: float = ICE_POISSON,
    e2: float = BLADE_YOUNG_MODULUS_PA,
    nu2: float = BLADE_POISSON,
) -> float:
    """Hertz equivalent modulus 1/E* = (1-nu1^2)/E1 + (1-nu2^2)/E2."""
    e1 = _positive(e1, "e1")
    e2 = _positive(e2, "e2")
    if not (0.0 <= nu1 < 0.5) or not (0.0 <= nu2 < 0.5):
        raise ValueError("Poisson ratios must lie in [0, 0.5)")
    return 1.0 / ((1.0 - nu1 * nu1) / e1 + (1.0 - nu2 * nu2) / e2)


def equivalent_contact_radius(*radii: float) -> float:
    """Harmonic sum of the principal radii of the two contacting bodies."""
    if not radii:
        raise ValueError("at least one radius is required")
    return 1.0 / sum(1.0 / _positive(r, f"radii[{i}]") for i, r in enumerate(radii))


def ellipsoid_pole_radius(semi_a: float, semi_b: float, semi_c: float) -> float:
    """Geometric-mean principal radius of curvature at the ``c`` pole.

    For x^2/a^2 + y^2/b^2 + z^2/c^2 = 1 the principal radii at (0, 0, c) are
    a^2/c and b^2/c, whose geometric mean a b / c is the equivalent sphere
    radius of the parabolic contact model.
    """
    a = _positive(semi_a, "semi_a")
    b = _positive(semi_b, "semi_b")
    c = _positive(semi_c, "semi_c")
    return a * b / c


def ice_fragment_geometry(length_m: float, *, aspect_ratio: float = 2.0) -> dict[str, float]:
    """Idealise an ice fragment of size L as a flat ellipsoid (L/2, L/2, L/(2k))."""
    if aspect_ratio < 1.0:
        raise ValueError("aspect_ratio must be >= 1")
    length = _positive(length_m, "length_m")
    aspect = _positive(aspect_ratio, "aspect_ratio")
    a = b = 0.5 * length
    c = 0.5 * length / aspect
    volume = 4.0 / 3.0 * math.pi * a * b * c
    return {
        "semi_a_m": a,
        "semi_b_m": b,
        "semi_c_m": c,
        "volume_m3": volume,
        "mass_kg": volume * ICE_DENSITY_KG_M3,
        "pole_radius_m": ellipsoid_pole_radius(a, b, c),
    }


def hertz_contact_radius(radius_m: float, penetration_m: float) -> float:
    """Hertz contact radius a = sqrt(R delta)."""
    return math.sqrt(_positive(radius_m, "radius_m") * _positive(penetration_m, "penetration_m"))


def hertz_peak_pressure(e_star_pa: float, radius_m: float, penetration_m: float) -> float:
    """Peak pressure p0 = 2 E* a / (pi R)."""
    e_star = _positive(e_star_pa, "e_star_pa")
    radius = _positive(radius_m, "radius_m")
    return 2.0 * e_star * hertz_contact_radius(radius, penetration_m) / (math.pi * radius)


def hertz_force(e_star_pa: float, radius_m: float, penetration_m: float) -> float:
    """Hertz force F = (4/3) E* sqrt(R) delta**(3/2)."""
    e_star = _positive(e_star_pa, "e_star_pa")
    radius = _positive(radius_m, "radius_m")
    penetration = _positive(penetration_m, "penetration_m")
    return 4.0 / 3.0 * e_star * math.sqrt(radius) * penetration**1.5


def hertz_pressure_quadrature(
    e_star_pa: float,
    radius_m: float,
    penetration_m: float,
    *,
    n_radial: int = 400,
    n_theta: int = 96,
    probe_ratios: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 0.9),
) -> dict[str, object]:
    """Independent numerical check of the closed-form Hertz solution.

    Integrates p(r) = p0 sqrt(1-(r/a)^2) over the contact disc (force) and
    through the Boussinesq kernel u(r) = 1/(pi E*) int p(s)/|r-s| dA(s)
    (surface displacement).  For this pressure field the half-space solution is
    the parabolic profile u(r) = delta (1 - r^2/(2 a^2)), i.e. the penetration
    minus the indenter shape (Johnson 1985, eq. 4.24), so a wrong prefactor
    breaks either the force, the centre displacement or the parabolic form.
    """
    e_star = _positive(e_star_pa, "e_star_pa")
    radius = _positive(radius_m, "radius_m")
    penetration = _positive(penetration_m, "penetration_m")
    if n_radial < 20 or n_theta < 16:
        raise ValueError("n_radial >= 20 and n_theta >= 16 are required")
    contact = hertz_contact_radius(radius, penetration)
    p0 = hertz_peak_pressure(e_star, radius, penetration)

    nodes, weights = np.polynomial.legendre.leggauss(n_radial)
    s = 0.5 * contact * (nodes + 1.0)
    w = 0.5 * contact * weights
    pressure = p0 * np.sqrt(np.maximum(1.0 - (s / contact) ** 2, 0.0))
    force_quadrature = float(np.sum(2.0 * math.pi * s * pressure * w))

    theta = (np.arange(n_theta, dtype=np.float64) + 0.5) * (2.0 * math.pi / n_theta)
    d_theta = 2.0 * math.pi / n_theta
    probes = np.asarray([float(r) * contact for r in probe_ratios])
    displacements = np.empty(probes.size)
    for index, r in enumerate(probes):
        distance = np.sqrt(
            np.maximum(
                r * r + s[:, None] ** 2 - 2.0 * r * s[:, None] * np.cos(theta)[None, :], 1e-30
            )
        )
        integrand = (pressure * s)[:, None] / distance
        displacements[index] = float(np.sum(integrand * w[:, None]) * d_theta / (math.pi * e_star))
    reference = penetration * (1.0 - (probes / contact) ** 2 / 2.0)
    profile_error = float(np.max(np.abs(displacements - reference)) / penetration)
    return {
        "force_quadrature_n": force_quadrature,
        "force_closed_form_n": hertz_force(e_star, radius, penetration),
        "contact_radius_m": contact,
        "peak_pressure_pa": p0,
        "penetration_m": penetration,
        "probe_radii_m": probes.tolist(),
        "probe_displacements_m": displacements.tolist(),
        "probe_reference_displacements_m": reference.tolist(),
        "centre_displacement_m": float(displacements[0]),
        "centre_displacement_relative_error": float(abs(displacements[0] - penetration) / penetration),
        "displacement_spread_relative": float((displacements.max() - displacements.min()) / penetration),
        "displacement_profile_relative_error": profile_error,
    }


@dataclass(frozen=True)
class HertzEngagement:
    """One ice-blade engagement in the stiffness-controlled Hertz idealisation."""

    force_max_n: float
    penetration_m: float
    duration_s: float
    contact_radius_m: float
    peak_pressure_pa: float
    impact_velocity_m_s: float
    effective_mass_kg: float
    contact_stiffness_n_m15: float
    equivalent_radius_m: float


def hertz_impact_duration(stiffness: float, effective_mass_kg: float, velocity_m_s: float) -> float:
    """Hertz impact contact time t_c = 2.94 (5m/(4k))^(2/5) v^(-1/5) (Johnson 4.32)."""
    stiffness = _positive(stiffness, "stiffness")
    mass = _positive(effective_mass_kg, "effective_mass_kg")
    velocity = _positive(velocity_m_s, "velocity_m_s")
    return 2.94 * (5.0 * mass / (4.0 * stiffness)) ** 0.4 * velocity ** (-0.2)


def hertz_yield_penetration(e_star_pa: float, radius_m: float, crush_pressure_pa: float) -> float:
    """Penetration at which the Hertz peak pressure reaches ``p_crush``.

    p0 = 2 E* a / (pi R) = p_crush with a = sqrt(R delta) gives
    delta_y = (pi p_crush / (2 E*))^2 R.
    """
    e_star = _positive(e_star_pa, "e_star_pa")
    radius = _positive(radius_m, "radius_m")
    crush = _positive(crush_pressure_pa, "crush_pressure_pa")
    return (math.pi * crush / (2.0 * e_star)) ** 2 * radius


def hertz_engagement(
    *,
    e_star_pa: float,
    equivalent_radius_m: float,
    effective_mass_kg: float,
    impact_velocity_m_s: float,
    crush_pressure_pa: float | None = None,
) -> HertzEngagement:
    """One impact from the kinetic-energy balance.

    Elastic (``crush_pressure_pa=None``): (1/2) m v^2 = (4/5) k delta^(5/2)
    gives delta = (5 m v^2 / (8k))^(2/5) and F_max = k delta^(3/2) with
    k = (4/3) E* sqrt(R).

    Strength-limited (``crush_pressure_pa=p_crush``): above the yield
    penetration delta_y the contact pressure cannot exceed the ice crushing
    strength, so the pressure field keeps the Hertz shape with a constant mean
    pressure (2/3) p_crush, i.e. F = (2/3) pi p_crush R delta.  The energy
    balance is closed with that two-branch law (bisection on delta).  The
    contact time follows from the impulse of the resulting pulse,
    t_c = pi m v / (2 F_max) for a half-sine pulse with no rebound.
    """
    e_star = _positive(e_star_pa, "e_star_pa")
    radius = _positive(equivalent_radius_m, "equivalent_radius_m")
    mass = _positive(effective_mass_kg, "effective_mass_kg")
    velocity = _positive(impact_velocity_m_s, "impact_velocity_m_s")
    stiffness = 4.0 / 3.0 * e_star * math.sqrt(radius)
    kinetic_energy = 0.5 * mass * velocity**2

    if crush_pressure_pa is None:
        penetration = (5.0 * mass * velocity**2 / (8.0 * stiffness)) ** 0.4
        force = stiffness * penetration**1.5
        duration = hertz_impact_duration(stiffness, mass, velocity)
    else:
        crush = _positive(crush_pressure_pa, "crush_pressure_pa")
        penetration = _crushing_limited_penetration(
            stiffness, radius, crush, kinetic_energy
        )
        force = min(stiffness * penetration**1.5, 2.0 / 3.0 * math.pi * crush * radius * penetration)
        duration = math.pi * mass * velocity / (2.0 * force)

    return HertzEngagement(
        force_max_n=force,
        penetration_m=penetration,
        duration_s=duration,
        contact_radius_m=hertz_contact_radius(radius, penetration),
        # Below the crushing limit the Hertz peak pressure governs; above it the
        # contact cannot exceed the ice crushing strength.
        peak_pressure_pa=min(
            hertz_peak_pressure(e_star, radius, penetration),
            crush_pressure_pa if crush_pressure_pa is not None else math.inf,
        ),
        impact_velocity_m_s=velocity,
        effective_mass_kg=mass,
        contact_stiffness_n_m15=stiffness,
        equivalent_radius_m=radius,
    )


def _crushing_limited_penetration(
    stiffness: float, radius_m: float, crush_pressure_pa: float, kinetic_energy_j: float
) -> float:
    """Penetration from the two-branch energy balance (exact, closed form).

    Branch 1 (elastic Hertz): E = (4/5) k delta^(5/2).
    Branch 2 (pressure-limited): F = c delta with c = (2/3) pi p_crush R, so
    E = E_y + c (delta^2 - delta_y^2)/2.  The transition delta_y follows from
    continuity of the force law, k delta_y^(3/2) = c delta_y.
    """
    cap_slope = 2.0 / 3.0 * math.pi * crush_pressure_pa * radius_m
    delta_yield = (cap_slope / stiffness) ** 2
    yield_energy = 0.8 * stiffness * delta_yield**2.5
    if kinetic_energy_j <= yield_energy:
        return (kinetic_energy_j / (0.8 * stiffness)) ** 0.4
    return math.sqrt(delta_yield**2 + 2.0 * (kinetic_energy_j - yield_energy) / cap_slope)


# --------------------------------------------------------------------------
# S2: blade modal analysis (clamped-free Hermite beam finite elements)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BladeGeometry:
    """Cantilever blade idealisation; radii measured from the shaft axis (m)."""

    hub_radius_m: float = 0.30
    tip_radius_m: float = 1.35
    root_chord_m: float = 0.60
    tip_chord_m: float = 0.25
    root_thickness_m: float = 0.25
    tip_thickness_m: float = 0.030
    leading_edge_radius_m: float = 0.010
    blade_count: int = 5
    n_elements: int = 24
    contact_span_fraction: float = 0.75

    def __post_init__(self) -> None:
        for name in (
            "hub_radius_m",
            "tip_radius_m",
            "root_chord_m",
            "tip_chord_m",
            "root_thickness_m",
            "tip_thickness_m",
            "leading_edge_radius_m",
        ):
            _positive(getattr(self, name), name)
        if self.tip_radius_m <= self.hub_radius_m:
            raise ValueError("tip_radius_m must exceed hub_radius_m")
        if self.blade_count < 2:
            raise ValueError("blade_count must be >= 2")
        if self.n_elements < 4:
            raise ValueError("n_elements must be >= 4")
        if not (0.0 < self.contact_span_fraction <= 1.0):
            raise ValueError("contact_span_fraction must lie in (0, 1]")

    @property
    def span_m(self) -> float:
        return self.tip_radius_m - self.hub_radius_m

    @property
    def contact_radius_m(self) -> float:
        return self.hub_radius_m + self.contact_span_fraction * self.span_m

    def chord_m(self, x: float) -> float:
        x = min(max(float(x), 0.0), self.span_m)
        return self.root_chord_m + (self.tip_chord_m - self.root_chord_m) * (x / self.span_m)

    def thickness_m(self, x: float) -> float:
        x = min(max(float(x), 0.0), self.span_m)
        return self.root_thickness_m + (self.tip_thickness_m - self.root_thickness_m) * (x / self.span_m)

    def section_modulus_m3(self, x: float = 0.0) -> float:
        """Bending section modulus of the rectangular section idealisation."""
        return self.chord_m(x) * self.thickness_m(x) ** 2 / 6.0


def blade_section_properties(
    geometry: BladeGeometry, *, young_modulus_pa: float = BLADE_YOUNG_MODULUS_PA
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Element-wise EI, rho*A and element mid-span coordinate arrays."""
    e = _positive(young_modulus_pa, "young_modulus_pa")
    n = geometry.n_elements
    edges = np.linspace(0.0, geometry.span_m, n + 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    chord = np.array([geometry.chord_m(x) for x in mid])
    thickness = np.array([geometry.thickness_m(x) for x in mid])
    return e * chord * thickness**3 / 12.0, BLADE_DENSITY_KG_M3 * chord * thickness, mid


def _hermite_beam_matrices(length: float, ei: float, rho_a: float) -> tuple[np.ndarray, np.ndarray]:
    """Consistent stiffness and mass matrices of a 2-node Hermite beam element."""
    le = length
    stiffness = ei / le**3 * np.array(
        [
            [12.0, 6.0 * le, -12.0, 6.0 * le],
            [6.0 * le, 4.0 * le * le, -6.0 * le, 2.0 * le * le],
            [-12.0, -6.0 * le, 12.0, -6.0 * le],
            [6.0 * le, 2.0 * le * le, -6.0 * le, 4.0 * le * le],
        ]
    )
    mass = rho_a * le / 420.0 * np.array(
        [
            [156.0, 22.0 * le, 54.0, -13.0 * le],
            [22.0 * le, 4.0 * le * le, 13.0 * le, -3.0 * le * le],
            [54.0, 13.0 * le, 156.0, -22.0 * le],
            [-13.0 * le, -3.0 * le * le, -22.0 * le, 4.0 * le * le],
        ]
    )
    return stiffness, mass


def _assemble_beam(
    ei: np.ndarray,
    rho_a: np.ndarray,
    length: float,
    added_mass_per_length: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    le = length / ei.size
    ndof = 2 * (ei.size + 1)
    stiffness = np.zeros((ndof, ndof))
    mass = np.zeros((ndof, ndof))
    for element in range(ei.size):
        ke, me = _hermite_beam_matrices(le, ei[element], rho_a[element])
        if added_mass_per_length is not None:
            me = me.copy()
            increment = added_mass_per_length[element] * le
            me[0, 0] += 0.5 * increment
            me[2, 2] += 0.5 * increment
        dofs = np.array([2 * element, 2 * element + 1, 2 * element + 2, 2 * element + 3])
        stiffness[np.ix_(dofs, dofs)] += ke
        mass[np.ix_(dofs, dofs)] += me
    return stiffness, mass


def cantilever_beam_modes(
    ei_elements: Sequence[float],
    rho_a_elements: Sequence[float],
    length_m: float,
    *,
    n_modes: int = 6,
    extra_translation_mass_elements: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clamped-free beam eigenproblem -> (omega, phi, x_dof); phi mass-normalised."""
    ei = np.asarray(ei_elements, dtype=np.float64)
    rho_a = np.asarray(rho_a_elements, dtype=np.float64)
    if ei.shape != rho_a.shape or ei.ndim != 1:
        raise ValueError("ei_elements and rho_a_elements must be 1-D and of equal length")
    if np.any(ei <= 0.0) or np.any(rho_a <= 0.0):
        raise ValueError("element EI and rho*A must be positive")
    length = _positive(length_m, "length_m")
    added = None
    if extra_translation_mass_elements is not None:
        added = np.asarray(extra_translation_mass_elements, dtype=np.float64)
        if added.shape != ei.shape:
            raise ValueError("extra_translation_mass_elements must match the element count")
        if np.any(added < 0.0):
            raise ValueError("extra translational mass must be >= 0")
    ndof = 2 * (ei.size + 1)
    if n_modes < 1 or n_modes > ndof - 2:
        raise ValueError("n_modes out of range")
    stiffness, mass = _assemble_beam(ei, rho_a, length, added)
    free = np.arange(2, ndof)
    k_ff = stiffness[np.ix_(free, free)]
    m_ff = mass[np.ix_(free, free)]
    eig_m, eig_vec = np.linalg.eigh(m_ff)
    if np.any(eig_m <= 0.0):
        raise ValueError("mass matrix is not positive definite")
    inv_sqrt = 1.0 / np.sqrt(eig_m)
    # M^{-1/2} = D^{-1/2} V^T for M = V D V^T; the symmetric reduction
    # B = M^{-1/2} K M^{-1/2} then shares the generalised eigenvalues.
    half_inv = inv_sqrt[:, None] * eig_vec.T
    values, vectors = np.linalg.eigh(half_inv @ k_ff @ half_inv.T)
    omega = np.sqrt(values[:n_modes])
    phi_free = eig_vec @ (inv_sqrt[:, None] * vectors[:, :n_modes])
    gram = phi_free.T @ m_ff @ phi_free
    phi_free = phi_free @ np.linalg.inv(np.linalg.cholesky(gram)).T
    phi = np.zeros((ndof, n_modes))
    phi[free, :] = phi_free
    return omega, phi, np.linspace(0.0, length, ei.size + 1)


def euler_bernoulli_frequency(length_m: float, ei: float, rho_a: float, *, mode: int = 1) -> float:
    """Closed-form clamped-free bending frequency (Hz), Blevins table 8-1."""
    length = _positive(length_m, "length_m")
    ei = _positive(ei, "ei")
    rho_a = _positive(rho_a, "rho_a")
    if mode < 1 or mode > len(EULER_BERNOULLI_BETA_L):
        raise ValueError(f"mode must lie in 1..{len(EULER_BERNOULLI_BETA_L)}")
    beta_l = EULER_BERNOULLI_BETA_L[mode - 1]
    return beta_l**2 / (2.0 * math.pi * length**2) * math.sqrt(ei / rho_a)


def flat_plate_added_mass_per_length(chord_m: float, *, rho: float = SEAWATER_DENSITY_KG_M3) -> float:
    """2-D flat-plate added mass per unit span m_a = pi rho c^2 / 4."""
    return math.pi * _positive(rho, "rho") * _positive(chord_m, "chord_m") ** 2 / 4.0


def wet_mode_frequencies(
    ei_elements: Sequence[float],
    rho_a_elements: Sequence[float],
    added_mass_per_length: Sequence[float],
    length_m: float,
    *,
    n_modes: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Dry and wet angular frequencies of the same finite-element model."""
    dry, _, _ = cantilever_beam_modes(ei_elements, rho_a_elements, length_m, n_modes=n_modes)
    wet, _, _ = cantilever_beam_modes(
        ei_elements,
        rho_a_elements,
        length_m,
        n_modes=n_modes,
        extra_translation_mass_elements=added_mass_per_length,
    )
    return dry, wet


def root_curvature_from_dofs(dofs: Sequence[float], element_length: float) -> float:
    """Curvature v''(0) of the clamped Hermite element from its four DOFs."""
    if len(dofs) != 4:
        raise ValueError("four nodal degrees of freedom are required")
    length = _positive(element_length, "element_length")
    v1, t1, v2, t2 = (float(value) for value in dofs)
    return (-6.0 * v1 - 4.0 * length * t1 + 6.0 * v2 - 2.0 * length * t2) / length**2


@dataclass(frozen=True)
class BladeModalBasis:
    """Modal basis plus the root curvature (stress) transfer of each mode."""

    omega_rad_s: np.ndarray
    phi: np.ndarray
    x_dof_m: np.ndarray
    root_curvature_factors: np.ndarray
    contact_node: int
    contact_phi: np.ndarray
    wet_omega_rad_s: np.ndarray
    effective_mass_kg: float
    blade_mass_kg: float
    frequencies_hz: np.ndarray
    wet_frequencies_hz: np.ndarray

    @property
    def n_modes(self) -> int:
        return int(self.omega_rad_s.size)


@functools.lru_cache(maxsize=8)
def _cached_basis(
    geometry: BladeGeometry, n_modes: int, added_mass_scale: float
) -> BladeModalBasis:
    ei, rho_a, mid = blade_section_properties(geometry)
    omega, phi, x_dof = cantilever_beam_modes(ei, rho_a, geometry.span_m, n_modes=n_modes)
    le = geometry.span_m / geometry.n_elements
    curvature = np.array([root_curvature_from_dofs(phi[:4, m], le) for m in range(n_modes)])
    contact_node = int(np.argmin(np.abs(x_dof - geometry.contact_span_fraction * geometry.span_m)))
    chord_mid = np.array([geometry.chord_m(x) for x in mid])
    added = np.array([flat_plate_added_mass_per_length(c) for c in chord_mid]) * added_mass_scale
    wet_omega, _, _ = cantilever_beam_modes(
        ei, rho_a, geometry.span_m, n_modes=n_modes, extra_translation_mass_elements=added
    )
    contact_phi = phi[2 * contact_node, :]
    effective_mass = 1.0 / float(np.sum(contact_phi**2))
    return BladeModalBasis(
        omega_rad_s=omega,
        phi=phi,
        x_dof_m=x_dof,
        root_curvature_factors=curvature,
        contact_node=contact_node,
        contact_phi=contact_phi,
        wet_omega_rad_s=wet_omega,
        effective_mass_kg=effective_mass,
        blade_mass_kg=float(np.sum(rho_a * geometry.span_m / geometry.n_elements)),
        frequencies_hz=omega / (2.0 * math.pi),
        wet_frequencies_hz=wet_omega / (2.0 * math.pi),
    )


def blade_modal_basis(
    geometry: BladeGeometry = BladeGeometry(),
    *,
    n_modes: int = 5,
    added_mass_scale: float = 1.0,
) -> BladeModalBasis:
    """Cached dry/wet modal basis of the blade."""
    if not isinstance(geometry, BladeGeometry):
        raise TypeError("geometry must be a BladeGeometry")
    return _cached_basis(geometry, n_modes, float(added_mass_scale))


# --------------------------------------------------------------------------
# S3: root stress response
# --------------------------------------------------------------------------


def _fft_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = a.size + b.size - 1
    nfft = 1 << (n - 1).bit_length()
    spectrum = np.fft.rfft(a, nfft) * np.fft.rfft(b, nfft)
    return np.fft.irfft(spectrum, nfft)[:n]


def half_sine_pulse(
    force_max_n: float, duration_s: float, *, n_samples: int = 512
) -> tuple[np.ndarray, np.ndarray, float]:
    """Half-sine contact pulse F(t) = F_max sin(pi t / t_c), 0 <= t <= t_c."""
    force = _positive(force_max_n, "force_max_n")
    duration = _positive(duration_s, "duration_s")
    if n_samples < 16:
        raise ValueError("n_samples must be >= 16")
    dt = duration / n_samples
    phase = math.pi * np.arange(n_samples, dtype=np.float64) / n_samples
    return np.arange(n_samples, dtype=np.float64) * dt, force * np.sin(phase), dt


def pulse_train(
    force_max_n: float,
    duration_s: float,
    impact_frequency_hz: float,
    n_pulses: int,
    *,
    samples_per_period: int = 4096,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Periodic train of half-sine contact pulses (blade pass frequency)."""
    force = _positive(force_max_n, "force_max_n")
    duration = _positive(duration_s, "duration_s")
    frequency = _positive(impact_frequency_hz, "impact_frequency_hz")
    if n_pulses < 1:
        raise ValueError("n_pulses must be >= 1")
    if samples_per_period < 512:
        raise ValueError("samples_per_period must be >= 512")
    period = 1.0 / frequency
    if duration >= period:
        raise ValueError("pulse duration must be shorter than the impact period")
    dt = period / samples_per_period
    n_samples = n_pulses * samples_per_period + int(round(duration / dt)) + 1
    values = np.zeros(n_samples)
    n_pulse_samples = max(int(round(duration / dt)), 8)
    shape = force * np.sin(math.pi * np.arange(n_pulse_samples, dtype=np.float64) / n_pulse_samples)
    for index in range(n_pulses):
        start = index * samples_per_period
        stop = min(start + n_pulse_samples, n_samples)
        values[start:stop] += shape[: stop - start]
    return np.arange(n_samples, dtype=np.float64) * dt, values, dt


def modal_stress_history(
    force_history: Sequence[float],
    dt: float,
    basis: BladeModalBasis,
    *,
    fibre_distance_m: float,
    damping_ratio: float = 0.02,
    young_modulus_pa: float = BLADE_YOUNG_MODULUS_PA,
    wet: bool = True,
    damping_ratios: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Root bending stress from modal superposition (exact SDOF convolution).

    q_n'' + 2 zeta w_n q_n' + w_n^2 q_n = phi_n(x_c) F(t) with mass-normalised
    modes; sigma(t) = E c_root sum_n phi_n''(0) q_n(t).
    """
    force = np.asarray(force_history, dtype=np.float64)
    if force.ndim != 1 or force.size < 2:
        raise ValueError("force_history must be a 1-D array with >= 2 samples")
    dt = _positive(dt, "dt")
    zeta = float(damping_ratio)
    if not (0.0 <= zeta < 1.0):
        raise ValueError("damping_ratio must lie in [0, 1)")
    if damping_ratios is not None:
        zetas = [float(value) for value in damping_ratios]
        if len(zetas) != basis.n_modes or any(not (0.0 <= z < 1.0) for z in zetas):
            raise ValueError("damping_ratios must hold one ratio per mode, in [0, 1)")
    else:
        zetas = [zeta] * basis.n_modes
    if not isinstance(basis, BladeModalBasis):
        raise TypeError("basis must be a BladeModalBasis")
    e = _positive(young_modulus_pa, "young_modulus_pa")
    fibre = _positive(fibre_distance_m, "fibre_distance_m")
    omega = basis.wet_omega_rad_s if wet else basis.omega_rad_s
    n = force.size
    kernel_time = np.arange(n, dtype=np.float64) * dt
    curvature = np.zeros(n)
    for mode in range(basis.n_modes):
        wn = float(omega[mode])
        zeta_mode = zetas[mode]
        wd = wn * math.sqrt(max(1.0 - zeta_mode * zeta_mode, 1e-12))
        kernel = np.exp(-zeta_mode * wn * kernel_time) * np.sin(wd * kernel_time) / wd
        q = _fft_convolve(basis.contact_phi[mode] * force, kernel)[:n] * dt
        curvature += basis.root_curvature_factors[mode] * q
    return np.arange(n, dtype=np.float64) * dt, e * fibre * curvature


def rayleigh_damping(
    stiffness: np.ndarray, mass: np.ndarray, omega1: float, omega2: float, zeta: float
) -> np.ndarray:
    """Rayleigh damping C = a0 M + a1 K matching zeta at two frequencies."""
    if not (omega1 > 0.0 and omega2 > 0.0 and omega1 != omega2):
        raise ValueError("two distinct positive frequencies are required")
    if not (0.0 < zeta < 1.0):
        raise ValueError("zeta must lie in (0, 1)")
    a0 = 2.0 * zeta * omega1 * omega2 / (omega1 + omega2)
    a1 = 2.0 * zeta / (omega1 + omega2)
    return a0 * mass + a1 * stiffness


def newmark_stress_history(
    force_history: Sequence[float],
    dt: float,
    geometry: BladeGeometry = BladeGeometry(),
    *,
    damping_ratio: float = 0.02,
    young_modulus_pa: float = BLADE_YOUNG_MODULUS_PA,
    damping_mode_index: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Direct Newmark-beta (average acceleration) full-order reference.

    Independent of the modal route: the assembled clamped-free beam system is
    integrated in physical coordinates with Rayleigh damping matched at the
    first and the (damping_mode_index+1)-th dry frequency; the root stress
    follows from the first element's nodal degrees of freedom.
    """
    force = np.asarray(force_history, dtype=np.float64)
    if force.ndim != 1 or force.size < 2:
        raise ValueError("force_history must be a 1-D array with >= 2 samples")
    dt = _positive(dt, "dt")
    if not isinstance(geometry, BladeGeometry):
        raise TypeError("geometry must be a BladeGeometry")
    e = _positive(young_modulus_pa, "young_modulus_pa")
    ei, rho_a, _ = blade_section_properties(geometry, young_modulus_pa=e)
    stiffness, mass = _assemble_beam(ei, rho_a, geometry.span_m)
    omega, _, _ = cantilever_beam_modes(ei, rho_a, geometry.span_m, n_modes=8)
    damping = rayleigh_damping(
        stiffness, mass, float(omega[0]), float(omega[damping_mode_index]), damping_ratio
    )
    free = np.arange(2, stiffness.shape[0])
    k = stiffness[np.ix_(free, free)]
    m = mass[np.ix_(free, free)]
    c = damping[np.ix_(free, free)]
    nodes = np.linspace(0.0, geometry.span_m, ei.size + 1)
    contact_node = int(np.argmin(np.abs(nodes - geometry.contact_span_fraction * geometry.span_m)))
    load = np.zeros(free.size)
    load[np.searchsorted(free, 2 * contact_node)] = 1.0

    beta, gamma = 0.25, 0.5
    k_inv = np.linalg.inv(k + gamma / (beta * dt) * c + m / (beta * dt * dt))
    u = np.zeros(free.size)
    v = np.zeros(free.size)
    a = np.zeros(free.size)
    n = force.size
    stress = np.zeros(n)
    le = geometry.span_m / ei.size
    fibre = 0.5 * geometry.thickness_m(0.0)
    for step in range(n):
        p_next = load * force[min(step + 1, n - 1)]
        rhs = (
            p_next
            + m @ (u / (beta * dt * dt) + v / (beta * dt) + (0.5 / beta - 1.0) * a)
            + c
            @ (
                gamma / (beta * dt) * u
                + (gamma / beta - 1.0) * v
                + dt * (gamma / (2.0 * beta) - 1.0) * a
            )
        )
        u_next = k_inv @ rhs
        a_next = (u_next - u) / (beta * dt * dt) - v / (beta * dt) - (0.5 / beta - 1.0) * a
        v_next = v + dt * ((1.0 - gamma) * a + gamma * a_next)
        dofs = (0.0, 0.0, u_next[np.searchsorted(free, 2)], u_next[np.searchsorted(free, 3)])
        stress[step] = e * fibre * root_curvature_from_dofs(dofs, le)
        u, v, a = u_next, v_next, a_next
    return np.arange(n, dtype=np.float64) * dt, stress


# --------------------------------------------------------------------------
# S4/S5: rainflow, S-N curve, Miner accumulation
# --------------------------------------------------------------------------


def sine_stress_history(
    amplitude_pa: float, frequency_hz: float, cycles: float, *, samples_per_cycle: int = 64
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sampled sinusoidal stress history with a known cycle count."""
    amplitude = _positive(amplitude_pa, "amplitude_pa")
    frequency = _positive(frequency_hz, "frequency_hz")
    cycles = _positive(cycles, "cycles")
    if samples_per_cycle < 16:
        raise ValueError("samples_per_cycle must be >= 16")
    dt = 1.0 / (frequency * samples_per_cycle)
    n = int(round(cycles * samples_per_cycle)) + 1
    time = np.arange(n, dtype=np.float64) * dt
    return time, amplitude * np.sin(2.0 * math.pi * frequency * time), dt


def dnv_d_curve_in_air() -> DNVSNCurve:
    """DNV-RP-C203 curve D in air (two slopes, knee at 1e7 cycles)."""
    return DNVSNCurve(
        name="DNV-RP-C203 D air",
        log_a1=DNV_D_TABLE["log_a1"],
        slope_m1=DNV_D_TABLE["slope_m1"],
        log_a2=DNV_D_TABLE["log_a2"],
        slope_m2=DNV_D_TABLE["slope_m2"],
        knee_cycles=DNV_D_TABLE["knee_cycles"],
    )


def miner_damage(cycles: Sequence[RainflowCycle], curve: DNVSNCurve | None = None) -> dict[str, object]:
    """Miner accumulation D = sum n_i / N_i over rainflow cycles."""
    rows = tuple(cycles)
    if not rows:
        raise ValueError("at least one rainflow cycle is required")
    if not all(isinstance(row, RainflowCycle) for row in rows):
        raise TypeError("cycles must contain RainflowCycle values")
    sn = dnv_d_curve_in_air() if curve is None else curve
    if not isinstance(sn, DNVSNCurve):
        raise TypeError("curve must be a DNVSNCurve")
    parts = [row.count / sn.cycles_to_failure(row.range_pa) for row in rows]
    damage = math.fsum(parts)
    return {
        "damage": damage,
        "n_cycles": float(sum(row.count for row in rows)),
        "max_range_pa": float(max(row.range_pa for row in rows)),
        "n_cycle_records": len(rows),
        "passed": damage < 1.0,
    }


# --------------------------------------------------------------------------
# S6: the ice-milling chain
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MillingCondition:
    """Ice-milling operating point of the propeller."""

    ice_length_m: float = 0.15
    rpm: float = 120.0
    advance_speed_m_s: float = 3.0
    pitch_angle_rad: float = 0.35
    wake_fraction: float = WAKE_FRACTION
    ice_aspect_ratio: float = 2.0
    damping_ratio: float = 0.02
    added_mass_scale: float = 1.0
    milling_hours_per_year: float = 120.0
    persistence: float = 1.0
    #: Ice crushing strength limiting the contact pressure (Pa).  ``None``
    #: selects the pure elastic Hertz contact (the campaign specification),
    #: which is reported as the bounding case.
    crush_pressure_pa: float | None = 30.0e6

    def __post_init__(self) -> None:
        for name in (
            "ice_length_m",
            "rpm",
            "advance_speed_m_s",
            "pitch_angle_rad",
            "ice_aspect_ratio",
        ):
            _positive(getattr(self, name), name)
        if not (0.0 < self.pitch_angle_rad < math.pi / 2.0):
            raise ValueError("pitch_angle_rad must lie in (0, pi/2)")
        if not (0.0 <= self.wake_fraction < 1.0):
            raise ValueError("wake_fraction must lie in [0, 1)")
        if not (0.0 <= self.damping_ratio < 1.0):
            raise ValueError("damping_ratio must lie in [0, 1)")
        if self.added_mass_scale < 0.0:
            raise ValueError("added_mass_scale must be >= 0")
        if self.milling_hours_per_year < 0.0:
            raise ValueError("milling_hours_per_year must be >= 0")
        if not (0.0 < self.persistence <= 1.0):
            raise ValueError("persistence must lie in (0, 1]")
        if self.crush_pressure_pa is not None:
            _positive(self.crush_pressure_pa, "crush_pressure_pa")


def _normal_impact_velocity(condition: MillingCondition, geometry: BladeGeometry) -> float:
    """Normal component of the blade-ice relative velocity at the contact radius.

    The section moves tangentially at omega*r while the blade surface normal is
    inclined at the pitch angle, giving omega*r*sin(phi); the ice entering with
    the ship advance adds V*(1-w)*cos(phi).  Both contributions raise the
    approach speed, so the peak force rises with rpm and with advance speed.
    """
    omega = 2.0 * math.pi * condition.rpm / 60.0
    tangential = omega * geometry.contact_radius_m
    advance = condition.advance_speed_m_s * (1.0 - condition.wake_fraction)
    return tangential * math.sin(condition.pitch_angle_rad) + advance * math.cos(
        condition.pitch_angle_rad
    )


def milling_engagement(
    condition: MillingCondition = MillingCondition(),
    geometry: BladeGeometry = BladeGeometry(),
    basis: BladeModalBasis | None = None,
) -> dict[str, object]:
    """Stage 1 + 2 for one operating point: contact pulse and modal inputs."""
    if not isinstance(condition, MillingCondition):
        raise TypeError("condition must be a MillingCondition")
    if not isinstance(geometry, BladeGeometry):
        raise TypeError("geometry must be a BladeGeometry")
    if basis is None:
        basis = blade_modal_basis(geometry, added_mass_scale=condition.added_mass_scale)
    fragment = ice_fragment_geometry(condition.ice_length_m, aspect_ratio=condition.ice_aspect_ratio)
    e_star = effective_contact_modulus()
    radius = equivalent_contact_radius(fragment["pole_radius_m"], geometry.leading_edge_radius_m)
    effective_mass = basis.effective_mass_kg + fragment["mass_kg"]
    velocity = _normal_impact_velocity(condition, geometry)
    engagement = hertz_engagement(
        e_star_pa=e_star,
        equivalent_radius_m=radius,
        effective_mass_kg=effective_mass,
        impact_velocity_m_s=velocity,
        crush_pressure_pa=condition.crush_pressure_pa,
    )
    impact_frequency = geometry.blade_count * condition.rpm / 60.0
    return {
        "fragment": fragment,
        "e_star_pa": e_star,
        "equivalent_radius_m": radius,
        "blade_effective_mass_kg": basis.effective_mass_kg,
        "effective_mass_kg": effective_mass,
        "impact_velocity_m_s": velocity,
        "engagement": engagement,
        "impact_frequency_hz": impact_frequency,
        "blade_mass_kg": basis.blade_mass_kg,
        "dry_frequencies_hz": basis.frequencies_hz.tolist(),
        "wet_frequencies_hz": basis.wet_frequencies_hz.tolist(),
        "contact_radius_m": geometry.contact_radius_m,
        "moment_arm_m": geometry.contact_radius_m - geometry.hub_radius_m,
        "root_section_modulus_m3": geometry.section_modulus_m3(0.0),
        "contact_model": "crushing-limited" if condition.crush_pressure_pa else "elastic-hertz",
        "crush_pressure_pa": condition.crush_pressure_pa,
    }


def milling_damage_case(
    condition: MillingCondition = MillingCondition(),
    geometry: BladeGeometry = BladeGeometry(),
    *,
    n_modes: int = 5,
    n_pulses: int = 10,
    max_samples_per_period: int = 8192,
) -> dict[str, object]:
    """Full chain for one operating point -> stress history, cycles, damage.

    ``condition.persistence`` scales the number of ice strikes (not their
    amplitude): a floe of size L is not present at every blade pass.
    """
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")
    if n_pulses < 2:
        raise ValueError("n_pulses must be >= 2")
    basis = blade_modal_basis(geometry, n_modes=n_modes, added_mass_scale=condition.added_mass_scale)
    state = milling_engagement(condition, geometry, basis)
    engagement: HertzEngagement = state["engagement"]  # type: ignore[assignment]
    frequency = float(state["impact_frequency_hz"])
    period = 1.0 / frequency
    f_max = float(np.max(basis.wet_frequencies_hz))
    dt = min(engagement.duration_s / 48.0, 1.0 / (16.0 * f_max))
    samples_per_period = int(min(max(round(period / dt), 512), max_samples_per_period))
    dt = period / samples_per_period
    time, force, dt = pulse_train(
        engagement.force_max_n,
        engagement.duration_s,
        frequency,
        n_pulses,
        samples_per_period=samples_per_period,
    )
    fibre = 0.5 * geometry.thickness_m(0.0)
    _, stress = modal_stress_history(
        force,
        dt,
        basis,
        fibre_distance_m=fibre,
        damping_ratio=condition.damping_ratio,
        wet=True,
    )
    cycles = rainflow_cycles([float(value) for value in stress])
    sn = dnv_d_curve_in_air()
    fatigue = miner_damage(cycles, sn)
    per_pass_damage_window = float(fatigue["damage"]) / n_pulses
    strikes_per_hour = frequency * 3600.0 * condition.persistence
    damage_per_hour = per_pass_damage_window * strikes_per_hour
    damage_20yr = damage_per_hour * condition.milling_hours_per_year * DESIGN_LIFE_YEARS
    quasi_static_stress = (
        engagement.force_max_n
        * float(state["moment_arm_m"])
        / float(state["root_section_modulus_m3"])
    )
    return {
        "condition": {
            "ice_length_m": condition.ice_length_m,
            "rpm": condition.rpm,
            "advance_speed_m_s": condition.advance_speed_m_s,
            "pitch_angle_rad": condition.pitch_angle_rad,
            "persistence": condition.persistence,
            "milling_hours_per_year": condition.milling_hours_per_year,
        },
        "contact_model": state["contact_model"],
        "crush_pressure_pa": state["crush_pressure_pa"],
        "engagement": {
            "force_max_n": engagement.force_max_n,
            "penetration_m": engagement.penetration_m,
            "duration_s": engagement.duration_s,
            "contact_radius_m": engagement.contact_radius_m,
            "peak_pressure_pa": engagement.peak_pressure_pa,
            "impact_velocity_m_s": engagement.impact_velocity_m_s,
            "effective_mass_kg": engagement.effective_mass_kg,
            "equivalent_radius_m": engagement.equivalent_radius_m,
            "impact_frequency_hz": frequency,
        },
        "stress": {
            "sigma_max_pa": float(np.max(stress)),
            "sigma_min_pa": float(np.min(stress)),
            "sigma_rms_pa": float(np.sqrt(np.mean(stress**2))),
            "sigma_quasi_static_pa": quasi_static_stress,
            "n_samples": int(stress.size),
            "dt_s": dt,
        },
        "cycles": {
            "n_records": int(fatigue["n_cycle_records"]),
            "n_cycles": float(fatigue["n_cycles"]),
            "max_range_pa": float(fatigue["max_range_pa"]),
            "per_pass_damage": per_pass_damage_window,
        },
        "fatigue": {
            "damage_per_hour": damage_per_hour,
            "damage_20yr": damage_20yr,
            "design_life_years": DESIGN_LIFE_YEARS,
            "allowable_hours": math.inf if damage_per_hour <= 0.0 else 1.0 / damage_per_hour,
            "utilization": damage_20yr,
            "passed": damage_20yr < 1.0,
        },
        "history": {
            "time_s": time.tolist(),
            "stress_pa": stress.tolist(),
            "force_n": force.tolist(),
        },
    }


def milling_parameter_sweep(
    *,
    ice_lengths_m: Sequence[float] = (0.10, 0.20, 0.35, 0.50),
    rpms: Sequence[float] = (100.0, 130.0, 160.0, 200.0),
    speeds_m_s: Sequence[float] = (2.0, 3.0, 4.0, 6.0),
    baseline: MillingCondition = MillingCondition(),
    geometry: BladeGeometry = BladeGeometry(),
    n_modes: int = 5,
    n_pulses: int = 8,
) -> dict[str, object]:
    """One-variable-at-a-time sweep of L_ice, n and V with monotonicity checks.

    The damage is expected to increase with all three variables: a larger floe
    enlarges the contact stiffness radius and the participating mass, a higher
    shaft speed raises both the approach velocity and the strike rate, and a
    higher advance speed raises the normal approach velocity.
    """
    def run(**changes) -> dict[str, object]:
        condition = MillingCondition(**{**baseline.__dict__, **changes})
        case = milling_damage_case(condition, geometry, n_modes=n_modes, n_pulses=n_pulses)
        return {
            "ice_length_m": condition.ice_length_m,
            "rpm": condition.rpm,
            "advance_speed_m_s": condition.advance_speed_m_s,
            "force_max_n": case["engagement"]["force_max_n"],
            "sigma_max_pa": case["stress"]["sigma_max_pa"],
            "impact_frequency_hz": case["engagement"]["impact_frequency_hz"],
            "damage_per_hour": case["fatigue"]["damage_per_hour"],
            "damage_20yr": case["fatigue"]["damage_20yr"],
            "allowable_hours": case["fatigue"]["allowable_hours"],
        }

    by_length = [run(ice_length_m=value) for value in ice_lengths_m]
    by_rpm = [run(rpm=value) for value in rpms]
    by_speed = [run(advance_speed_m_s=value) for value in speeds_m_s]

    def monotone(rows: list[dict[str, object]], key: str) -> bool:
        values = [float(row[key]) for row in rows]
        return all(later > earlier for earlier, later in zip(values, values[1:]))

    checks = {
        "damage_monotone_in_ice_length": monotone(by_length, "damage_20yr"),
        "damage_monotone_in_rpm": monotone(by_rpm, "damage_20yr"),
        "damage_monotone_in_speed": monotone(by_speed, "damage_20yr"),
        "force_monotone_in_ice_length": monotone(by_length, "force_max_n"),
        "force_monotone_in_rpm": monotone(by_rpm, "force_max_n"),
        "force_monotone_in_speed": monotone(by_speed, "force_max_n"),
    }
    return {
        "baseline": {
            "ice_length_m": baseline.ice_length_m,
            "rpm": baseline.rpm,
            "advance_speed_m_s": baseline.advance_speed_m_s,
            "milling_hours_per_year": baseline.milling_hours_per_year,
            "persistence": baseline.persistence,
        },
        "by_ice_length": by_length,
        "by_rpm": by_rpm,
        "by_speed": by_speed,
        "checks": checks,
        "all_pass": all(checks.values()),
    }
def milling_bounding_case(
    condition: MillingCondition = MillingCondition(),
    geometry: BladeGeometry = BladeGeometry(),
    **kwargs,
) -> dict[str, object]:
    """Bounding case: pure elastic Hertz contact (the campaign specification).

    The elastic law has no strength limit, so the peak pressure reaches GPa
    levels and the resulting stress/damage is an extreme upper bound; the
    difference between this case and the crushing-limited design case is the
    dominant modelling uncertainty of the chain.
    """
    return milling_damage_case(replace(condition, crush_pressure_pa=None), geometry, **kwargs)
