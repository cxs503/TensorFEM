"""Sound-power / vibration-velocity level scaling law and structural-acoustic energy balance.

The 10 log10 law
----------------
For a surface ``S`` vibrating with time-averaged mean-square normal velocity
``<v^2>`` and radiation efficiency ``sigma`` the radiated sound power is

.. math:: W = \\rho c S \\langle v^2\\rangle \\sigma .

Taking ``Lw = 10 log10(W/W_ref)`` and ``Lv = 10 log10(<v^2>/v_ref^2)`` the law
becomes exactly linear,

.. math:: Lw = Lv + 10\\log_{10}(\\rho c S \\sigma) + 20\\log_{10} v_{ref}
          - 10\\log_{10} W_{ref}
          \\;\\equiv\\; Lv + C ,

so ``C`` is *derived*, not fitted, by :func:`scale_law_offset_db`.  ISO 1683
references are used throughout (``W_ref = 1e-12 W``, ``v_ref = 1e-9 m/s``); with
those numbers ``20 log10 v_ref = -180 dB`` and ``-10 log10 W_ref = +120 dB``,
hence ``C = 10 log10(rho c S sigma) - 60 dB``.

Two failure modes this law is designed to catch, both of which the tests probe
deliberately:

* using **peak** velocity where RMS is required -> a spurious ``+3.01 dB``;
* using ``rho c S`` where the radiating area is one-sided (baffled piston) or
  two-sided (free-standing sphere) -> a spurious ``±3.01 dB``.

Energy balance (two-way coupling)
---------------------------------
For a harmonic structural degree of freedom driven by ``F`` with modal mass
``m``, stiffness ``k`` and viscous damping ``c_d``, and loaded by the acoustic
reaction ``A p(a)`` at its wetted area ``A``,

.. math:: (-\\omega^2 m + k - i\\omega c_d) v = F - A\\,Z(\\omega)\\,v ,
          \\qquad Z = p(a)/v ,

the coupled velocity is ``v = F /(k - \\omega^2 m - i\\omega c_d - A Z)`` and the
power identity

.. math:: \\tfrac12\\Re(F v^*) = \\tfrac12\\Re(A Z)|v|^2 + \\tfrac12 c_d |v|^2

holds identically at the coupled solution -- it holds *because* the acoustic
reaction entered the structural equation with the right sign and the right
area.  :class:`CoupledBreathingSphere` evaluates the radiated power through two
independent acoustic routes (surface flux and outer truncation flux) so the
residual is not a tautology: it measures the acoustic solve as well.

The structural side is a spherical shell breathing mode; the modal parameters
come from the classical membrane solution of a thin sphere
(:func:`spherical_shell_breathing_parameters`), so ``f0`` is a physical
frequency and the sweep crosses the wet resonance where the fluid loading
matters most.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


from .plate_far_field import (
    C_WATER,
    RHO_WATER,
    piston_radiation_efficiency,
    wavenumber,
)
from .pulsating_sphere import (
    radiation_efficiency as monopole_radiation_efficiency,
    solve_pulsating_sphere,
    surface_impedance,
    radiated_power as monopole_power,
)

W_REF = 1.0e-12  # W, ISO 1683
V_REF = 1.0e-9  # m/s, ISO 1683
P_REF = 2.0e-5  # Pa, ISO 1683 (reported for reference only)


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return out


def _nonnegative(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return out


# ---------------------------------------------------------------------------
# Levels
# ---------------------------------------------------------------------------

def sound_power_level(power: float, w_ref: float = W_REF) -> float:
    """``Lw = 10 log10(W/W_ref)`` in dB re 1 pW."""
    return 10.0 * math.log10(_positive("power", power) / _positive("w_ref", w_ref))


def velocity_level(v_rms: float, v_ref: float = V_REF) -> float:
    """``Lv = 10 log10(<v^2>/v_ref^2)`` in dB re 1 nm/s."""
    return 10.0 * math.log10(_positive("v_rms", v_rms) ** 2 / _positive(
        "v_ref", v_ref
    ) ** 2)


def scale_law_offset_db(
    rho: float,
    c: float,
    area: float,
    sigma: float = 1.0,
    v_ref: float = V_REF,
    w_ref: float = W_REF,
) -> float:
    """Derived additive constant ``C`` of ``Lw = Lv + C``.

    ``C = 10 log10(rho c S sigma) + 20 log10 v_ref - 10 log10 w_ref``.
    """
    return (
        10.0 * math.log10(
            _positive("rho", rho) * _positive("c", c) * _positive("area", area)
            * _positive("sigma", sigma)
        )
        + 20.0 * math.log10(_positive("v_ref", v_ref))
        - 10.0 * math.log10(_positive("w_ref", w_ref))
    )


def scale_law_residual_db(
    power: float,
    v_rms: float,
    rho: float,
    c: float,
    area: float,
    sigma: float = 1.0,
    v_ref: float = V_REF,
    w_ref: float = W_REF,
) -> float:
    """``Lw - (Lv + C)`` in dB; zero when the 10 log10 law is satisfied exactly."""
    return (
        sound_power_level(power, w_ref)
        - velocity_level(v_rms, v_ref)
        - scale_law_offset_db(rho, c, area, sigma, v_ref, w_ref)
    )


def mean_square_from_peak(v_peak: float) -> float:
    """Time-averaged mean-square of a harmonic amplitude (``|v|^2/2``)."""
    return 0.5 * abs(complex(v_peak)) ** 2


def rms_from_peak(v_peak: float) -> float:
    """``v_rms = |v_peak|/sqrt(2)``."""
    return abs(complex(v_peak)) / math.sqrt(2.0)


def scaling_slope_db(
    power_of_velocity, v_base: float, decade: float = 10.0
) -> float:
    """``dLw / dLv`` measured over one velocity decade.

    Returns ``(dB power) / (dB velocity)``; a value of 1.0 means the law is
    exactly first order in ``<v^2>``.
    """
    v_base = _positive("v_base", v_base)
    p1 = _positive("power(v)", power_of_velocity(v_base))
    p2 = _positive("power(decade*v)", power_of_velocity(decade * v_base))
    dl_power = 10.0 * math.log10(p2 / p1)
    dl_velocity = 10.0 * math.log10((decade * v_base) ** 2 / v_base**2)
    return dl_power / dl_velocity


# ---------------------------------------------------------------------------
# Structural model: spherical shell breathing mode
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BreathingShell:
    """Membrane breathing mode of a thin spherical shell (modal mass/stiffness)."""

    radius: float
    thickness: float
    rho_s: float
    youngs: float
    poisson: float

    @property
    def mass(self) -> float:
        """Modal mass ``= rho_s h 4 pi a^2`` (lumped membrane mass)."""
        return self.rho_s * self.thickness * 4.0 * math.pi * self.radius**2

    @property
    def stiffness(self) -> float:
        """Modal stiffness from ``omega0^2 = 2 E /(rho_s (1 - nu) a^2)``."""
        omega0 = math.sqrt(
            2.0 * self.youngs / (self.rho_s * (1.0 - self.poisson) * self.radius**2)
        )
        return self.mass * omega0**2

    @property
    def natural_frequency(self) -> float:
        omega0 = math.sqrt(self.stiffness / self.mass)
        return omega0 / (2.0 * math.pi)

    @property
    def wetted_area(self) -> float:
        return 4.0 * math.pi * self.radius**2

    def damping(self, zeta: float) -> float:
        zeta = _positive("zeta", zeta)
        return 2.0 * zeta * math.sqrt(self.mass * self.stiffness)


def spherical_shell_breathing_parameters(
    radius: float = 1.0,
    thickness: float = 0.02,
    rho_s: float = 7850.0,
    youngs: float = 210.0e9,
    poisson: float = 0.3,
) -> BreathingShell:
    """Convenience constructor with mild steel defaults."""
    return BreathingShell(
        radius=_positive("radius", radius),
        thickness=_positive("thickness", thickness),
        rho_s=_positive("rho_s", rho_s),
        youngs=_positive("youngs", youngs),
        poisson=float(poisson),
    )


# ---------------------------------------------------------------------------
# Two-way coupled breathing sphere
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoupledBreathingSphere:
    """Coupled response of a breathing shell loaded by the acoustic field."""

    frequency: float
    zeta: float
    force: float
    velocity: complex
    impedance: complex
    fluid_force: complex
    power_in: float
    power_radiated_surface: float
    power_radiated_outer: float
    power_dissipated: float
    power_radiated_exact: float
    response_exact: complex
    area: float
    rho: float
    c: float

    @property
    def energy_residual(self) -> float:
        """``|P_in - P_rad(surface) - P_diss| / P_in``."""
        return abs(
            self.power_in - self.power_radiated_surface - self.power_dissipated
        ) / abs(self.power_in)

    @property
    def energy_residual_outer(self) -> float:
        """Same residual with the *outer truncation* flux (independent route)."""
        return abs(
            self.power_in - self.power_radiated_outer - self.power_dissipated
        ) / abs(self.power_in)

    @property
    def velocity_relative_error(self) -> float:
        """``|v_fem - v_exact| / |v_exact|`` (exact = analytic impedance)."""
        return abs(self.velocity - self.response_exact) / abs(self.response_exact)

    @property
    def radiation_efficiency(self) -> float:
        """Measured ``sigma = W /(rho c S <v^2>)``."""
        return self.power_radiated_surface / (
            self.rho * self.c * self.area * mean_square_from_peak(self.velocity)
        )


def coupled_breathing_sphere(
    frequency: float,
    shell: BreathingShell,
    force: float = 1.0e4,
    zeta: float = 0.05,
    elements: int = 24,
    truncation_ratio: float = 10.0,
    rho: float = RHO_WATER,
    c: float = C_WATER,
) -> CoupledBreathingSphere:
    """Solve the two-way coupled harmonic problem for a breathing shell.

    The structural equation is solved *with* the acoustic reaction computed by
    the radial Helmholtz FEM (:mod:`tensorfem.struct_acoustics.pulsating_sphere`),
    i.e. ``Z`` is a domain solution, and the radiated power is reported through
    the inner-surface and outer-truncation flux routes independently.
    """
    omega = 2.0 * math.pi * _positive("frequency", frequency)
    zeta = _positive("zeta", zeta)
    a = shell.radius
    area = shell.wetted_area
    # Unit-velocity acoustic solve: response is linear in v0, so one solve gives Z.
    unit = solve_pulsating_sphere(
        frequency, a, 1.0, truncation_ratio, elements, rho, c
    )
    z = unit.impedance
    m = shell.mass
    k = shell.stiffness
    c_d = shell.damping(zeta)
    # Mechanical impedance form (velocity as unknown, e^{-i omega t}):
    #   (c_d + i (k - omega^2 m)/omega + A Z) v = F
    # The stored-energy term i (k - omega^2 m)/omega is reactive and must stay
    # in the operator: dropping it (using the displacement-form stiffness) makes
    # the power identity fail by the reactive share.
    denom = complex(c_d, (k - omega * omega * m) / omega) + area * z
    v = complex(force) / denom
    fluid_force = -area * z * v
    power_in = 0.5 * (complex(force) * v.conjugate()).real
    power_rad_surface = 0.5 * area * z.real * abs(v) ** 2
    power_rad_outer = unit.power_outer_surface * abs(v) ** 2
    power_diss = 0.5 * c_d * abs(v) ** 2
    z_exact = surface_impedance(a, frequency, rho, c)
    v_exact = complex(force) / (
        complex(c_d, (k - omega * omega * m) / omega) + area * z_exact
    )
    power_exact = 0.5 * area * z_exact.real * abs(v_exact) ** 2
    return CoupledBreathingSphere(
        frequency=float(frequency),
        zeta=zeta,
        force=float(force),
        velocity=v,
        impedance=z,
        fluid_force=fluid_force,
        power_in=power_in,
        power_radiated_surface=power_rad_surface,
        power_radiated_outer=power_rad_outer,
        power_dissipated=power_diss,
        power_radiated_exact=power_exact,
        response_exact=v_exact,
        area=area,
        rho=rho,
        c=c,
    )


# ---------------------------------------------------------------------------
# Generic surface source for the scaling law
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SurfaceSourceLaw:
    """A prescribed-velocity radiating surface with its own closed-form sigma."""

    name: str
    area: float
    sigma: float
    power: float
    velocity: complex

    @property
    def v_rms(self) -> float:
        return rms_from_peak(self.velocity)

    def residual_db(self, rho: float = RHO_WATER, c: float = C_WATER) -> float:
        return scale_law_residual_db(
            self.power, self.v_rms, rho, c, self.area, self.sigma
        )


def monopole_surface_source(
    radius: float, frequency: float, v0: float,
    rho: float = RHO_WATER, c: float = C_WATER,
) -> SurfaceSourceLaw:
    """Breathing sphere (monopole): ``W = rho c S <v^2> sigma``, ``sigma = (ka)^2/(1+(ka)^2)``."""
    a = _positive("radius", radius)
    sigma = monopole_radiation_efficiency(a, frequency, c)
    power = monopole_power(a, frequency, v0, rho, c)
    return SurfaceSourceLaw(
        name="pulsating_sphere",
        area=4.0 * math.pi * a * a,
        sigma=sigma,
        power=power,
        velocity=complex(v0),
    )


def piston_surface_source(
    radius: float, frequency: float, v0: float,
    rho: float = RHO_WATER, c: float = C_WATER,
) -> SurfaceSourceLaw:
    """Baffled circular piston (one-sided radiation): ``sigma = 1 - J1(2ka)/(ka)``."""
    a = _positive("radius", radius)
    ka = wavenumber(frequency, c) * a
    sigma = piston_radiation_efficiency(ka)
    area = math.pi * a * a
    power = rho * c * area * mean_square_from_peak(v0) * sigma
    return SurfaceSourceLaw(
        name="baffled_piston",
        area=area,
        sigma=sigma,
        power=power,
        velocity=complex(v0),
    )


def scale_law_sweep(
    source_factory,
    radii: Sequence[float],
    frequencies: Iterable[float],
    v0: float,
    rho: float = RHO_WATER,
    c: float = C_WATER,
) -> list[dict]:
    """Tabulate everything a report needs for a set of ``(radius, frequency)`` cuts."""
    rho = _positive("rho", rho)
    c = _positive("c", c)
    rows = []
    for radius in radii:
        for f in frequencies:
            src = source_factory(radius, f, v0, rho, c)
            offset = scale_law_offset_db(rho, c, src.area, src.sigma)
            rows.append(
                {
                    "source": src.name,
                    "radius": float(radius),
                    "frequency": float(f),
                    "ka": wavenumber(f, c) * float(radius),
                    "sigma": src.sigma,
                    "area": src.area,
                    "v_rms": src.v_rms,
                    "velocity_level_db": velocity_level(src.v_rms),
                    "power_w": src.power,
                    "sound_power_level_db": sound_power_level(src.power),
                    "offset_c_db": offset,
                    "residual_db": src.residual_db(rho, c),
                }
            )
    return rows


def energy_balance_sweep(
    frequencies: Iterable[float],
    damping_ratios: Iterable[float],
    shell: BreathingShell | None = None,
    force: float = 1.0e4,
    elements: int = 32,
    rho: float = RHO_WATER,
    c: float = C_WATER,
) -> list[dict]:
    """Two-way coupled energy balance and level law across f and zeta."""
    shell = shell or spherical_shell_breathing_parameters()
    rows = []
    for f in frequencies:
        for zeta in damping_ratios:
            res = coupled_breathing_sphere(
                f, shell, force, zeta, elements, 10.0, rho, c
            )
            sigma_meas = res.radiation_efficiency
            rows.append(
                {
                    "frequency": float(f),
                    "zeta": float(zeta),
                    "velocity_peak": abs(res.velocity),
                    "velocity_level_db": velocity_level(rms_from_peak(res.velocity)),
                    "power_in_w": res.power_in,
                    "power_radiated_surface_w": res.power_radiated_surface,
                    "power_radiated_outer_w": res.power_radiated_outer,
                    "power_dissipated_w": res.power_dissipated,
                    "sound_power_level_db": sound_power_level(res.power_radiated_surface),
                    "energy_residual": res.energy_residual,
                    "energy_residual_outer": res.energy_residual_outer,
                    "velocity_relative_error_vs_exact": res.velocity_relative_error,
                    "sigma_measured": sigma_meas,
                    "sigma_analytic": monopole_radiation_efficiency(shell.radius, f, c),
                    "scale_law_residual_db": scale_law_residual_db(
                        res.power_radiated_surface,
                        rms_from_peak(res.velocity),
                        rho,
                        c,
                        shell.wetted_area,
                        sigma_meas,
                    ),
                    "scale_law_residual_db_analytic_sigma": scale_law_residual_db(
                        res.power_radiated_surface,
                        rms_from_peak(res.velocity),
                        rho,
                        c,
                        shell.wetted_area,
                        monopole_radiation_efficiency(shell.radius, f, c),
                    ),
                }
            )
    return rows


def impedance_convergence(
    frequency: float, radius: float = 1.0,
    element_counts: Sequence[int] = (2, 4, 8, 16, 32),
    rho: float = RHO_WATER, c: float = C_WATER,
) -> list[dict]:
    """Radiation impedance vs mesh refinement against the exact value."""
    z_exact = surface_impedance(radius, frequency, rho, c)
    rows = []
    for n in element_counts:
        unit = solve_pulsating_sphere(frequency, radius, 1.0, 10.0, n, rho, c)
        z = unit.impedance
        rows.append(
            {
                "elements": int(n),
                "real": z.real,
                "imag": z.imag,
                "real_exact": z_exact.real,
                "imag_exact": z_exact.imag,
                "relative_error": abs(z - z_exact) / abs(z_exact),
            }
        )
    return rows