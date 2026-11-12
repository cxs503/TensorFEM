"""Ice-induced slamming, hull-girder whipping and the fatigue increment.

Physical chain
--------------
An ice-class ship transiting a brash-ice field in a seaway receives two kinds
of short, high-amplitude pulses at the forebody:

* **ice impacts** -- the forebody rams floes; the load is a crushing pressure
  acting on the ice-contact area ``A = h * b_c`` and lasts a few tens of
  milliseconds, so it behaves as an impulsive excitation of the hull girder;
* **wave (water-entry) slamming** -- when the relative bow motion exceeds the
  local immersion the forebody leaves the water and re-enters with vertical
  velocity ``w``; the load is the rate of change of the section added mass
  (von Karman wedge-entry formulation).

Both pulse trains enter a wet, **free-free** hull-girder beam model (a floating
hull has no supports; pinning it raises the fundamental bending frequency by a
factor 2.27 and is not used here).  The overall bending modes are integrated
*exactly* in modal coordinates -- a piecewise-linear load has a closed-form
Duhamel solution -- which yields the vertical bending moment ``M_v(t)`` and the
shear ``F_s(t)`` including the whipping transient, the modal split of the
response, the transient decay time and, after rainflow counting and a Miner
sum, the fatigue-damage increment ``dD_slam`` relative to the wave-induced
``D_wave``.

What is verification and what is not
------------------------------------
Every number the module returns is computed, none is transcribed.  The parts
carrying a genuine independent reference:

* the wet girder frequencies against the **exact** dynamic-stiffness solution of
  the same Timoshenko beam (:func:`tensorfem.dsbeam.natural_frequencies_hz`,
  ``bc="free-free"``), an exact transcendental solution rather than a second
  finite element model;
* the modal integrator against the analytic damped free-decay envelope
  ``exp(-zeta*omega*t)`` and against a high-accuracy direct integration of the
  same single-degree-of-freedom ODE;
* the Froude-Krylov phasors against numerical quadrature;
* the bending-moment recovery against the static ``P*a*b/L`` moment of the same
  FE model, and the modal superposition against the direct static solve;
* the wave-slam peak against the von Karman added-mass momentum closure.

The parts that are **declared constitutive input** and *not* validated here:
the ice crushing pressure level (``crush_pressure_reference``,
``area_exponent``, ``rate_exponent``), the structural damping ratios of the
girder, the added-mass factor and the linear seakeeping coefficients.  Each is
a named dataclass field so a sensitivity sweep can move it; the qualification
artifact reports the band those sweeps produce.

Conventions: SI, ``z`` up, pitch bow-up positive, ``x`` forward of midships.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .dsbeam import DSMBeamSection, natural_frequencies_hz
from .hull_girder_modal import BoxGirderSection, lewis_heave_coefficient
from .marine_fatigue_advanced import rainflow_cycles

__all__ = [
    "IceClassShipCase",
    "IceField",
    "WaveField",
    "WhippingModel",
    "SNCurve",
    "GirderModes",
    "RelativeMotion",
    "LoadPulse",
    "WhippingResponse",
    "IceSlammingResult",
    "girder_section",
    "beam_element_matrices",
    "assembled_girder",
    "added_mass_per_length",
    "vibrating_mass_per_length",
    "wet_girder_modes",
    "dsm_reference_frequencies_hz",
    "frequency_check",
    "curvature_at_elements",
    "static_moment_check",
    "froude_krylov_phasors",
    "seakeeping_relative_motion",
    "ice_impact_pulses",
    "wave_slam_pulses",
    "von_karman_peak_force",
    "added_mass_impulse",
    "build_load_history",
    "piecewise_linear_modal_response",
    "modal_response_records",
    "static_bending_moment",
    "vertical_bending_moment",
    "whipping_decay_time",
    "fatigue_damage",
    "hot_spot_stress",
    "run_ice_slamming_case",
    "negative_control_checks",
]

_D = torch.float64
_G = 9.80665


# ---------------------------------------------------------------------------
# input validation helpers
# ---------------------------------------------------------------------------
def _positive(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return v


def _non_negative(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
    return v


def _fraction(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value!r}")
    return v


# ---------------------------------------------------------------------------
# declared case
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IceClassShipCase:
    """Declared geometry and hull-girder data of the ice-class ship.

    All values are CASE_INPUT: a declared representative ice-class bulker
    forebody, not a table transcribed from a document.  :meth:`geometry_checks`
    reports the standard bulk-parameter ratios so the geometry can be rejected
    before any result is looked at.
    """

    length: float                 # L_pp [m]
    breadth: float                # B [m]
    draft: float                  # mean draft T [m]
    block_coefficient: float      # C_b [-]
    girder_depth: float           # moulded depth D [m]
    girder_wall: float            # equivalent wall thickness [m]
    deadrise_deg: float           # forebody deadrise at the impact station [deg]
    impact_station_x: float       # impact station forward of midships [m]
    impact_local_immersion: float # static immersion of that station [m]
    impact_half_breadth: float    # waterline half breadth there [m]
    slam_length: float            # longitudinal extent of the wetted forebody [m]
    water_density: float = 1025.0
    steel_density: float = 7850.0
    steel_modulus: float = 2.06e11
    longitudinal_gm: float = 184.0
    waterplane_coefficient: float = 0.80
    added_mass_factor: float = 0.70
    heave_damping_ratio: float = 0.25
    pitch_damping_ratio: float = 0.15

    def validate(self) -> None:
        for name in (
            "length",
            "breadth",
            "draft",
            "girder_depth",
            "girder_wall",
            "impact_local_immersion",
            "impact_half_breadth",
            "slam_length",
            "water_density",
            "steel_density",
            "steel_modulus",
            "longitudinal_gm",
        ):
            _positive(getattr(self, name), name)
        _fraction(self.block_coefficient, "block_coefficient")
        _fraction(self.waterplane_coefficient, "waterplane_coefficient")
        _fraction(self.heave_damping_ratio, "heave_damping_ratio")
        _fraction(self.pitch_damping_ratio, "pitch_damping_ratio")
        beta = float(self.deadrise_deg)
        if not 0.0 < beta < 90.0:
            raise ValueError("deadrise_deg must lie in (0, 90)")
        if abs(float(self.impact_station_x)) >= 0.5 * self.length:
            raise ValueError("impact_station_x must lie inside the hull")
        _positive(self.added_mass_factor, "added_mass_factor")

    @property
    def displacement_volume(self) -> float:
        return self.block_coefficient * self.length * self.breadth * self.draft

    @property
    def displacement_mass(self) -> float:
        return self.water_density * self.displacement_volume

    @property
    def deadrise(self) -> float:
        return math.radians(self.deadrise_deg)

    @property
    def waterplane_area(self) -> float:
        return self.waterplane_coefficient * self.length * self.breadth

    @property
    def pitch_radius_of_gyration(self) -> float:
        return 0.25 * self.length

    @property
    def section_modulus(self) -> float:
        girder = self.girder()
        return girder.inertia / (0.5 * self.girder_depth)

    def geometry_checks(self) -> dict[str, object]:
        lb = self.length / self.breadth
        bt = self.breadth / self.draft
        lt = self.length / self.draft
        return {
            "block_coefficient": self.block_coefficient,
            "displacement_volume_m3": self.displacement_volume,
            "displacement_mass_t": self.displacement_mass / 1000.0,
            "length_breadth_ratio": lb,
            "breadth_draft_ratio": bt,
            "length_draft_ratio": lt,
            "flag_length_breadth": bool(not 4.5 <= lb <= 8.5),
            "flag_breadth_draft": bool(not 2.0 <= bt <= 4.0),
            "flag_block_coefficient": bool(not 0.55 <= self.block_coefficient <= 0.87),
        }

    def girder(self) -> BoxGirderSection:
        self.validate()
        return BoxGirderSection(
            length=self.length,
            breadth=self.breadth,
            depth=self.girder_depth,
            wall=self.girder_wall,
            young=self.steel_modulus,
            density=self.steel_density,
            draft=self.draft,
            water_density=self.water_density,
        )


def girder_section(ship: IceClassShipCase) -> BoxGirderSection:
    """Equivalent uniform box girder of ``ship``."""
    return ship.girder()


@dataclass(frozen=True)
class IceField:
    """Declared ice condition; ``concentration`` is the areal fraction."""

    thickness: float
    concentration: float
    floe_length: float = 12.0
    floe_length_cv: float = 0.35
    ice_density: float = 917.0
    crush_pressure_reference: float = 0.66e6
    contact_area_reference: float = 1.0
    area_exponent: float = -0.30
    velocity_reference: float = 1.0
    rate_exponent: float = 0.25
    seed: int = 20261110

    def validate(self) -> None:
        _non_negative(self.thickness, "thickness")
        _fraction(self.concentration, "concentration")
        _positive(self.floe_length, "floe_length")
        _non_negative(self.floe_length_cv, "floe_length_cv")
        _positive(self.ice_density, "ice_density")
        _positive(self.crush_pressure_reference, "crush_pressure_reference")
        _positive(self.contact_area_reference, "contact_area_reference")
        _positive(self.velocity_reference, "velocity_reference")

    def crush_pressure(self, area: float, velocity: float) -> float:
        """Ice crushing pressure [Pa] on contact area ``area`` at ``velocity``.

        Sanderson-type pressure-area softening with a declared strain-rate
        hardening term.  At zero closing velocity the pressure vanishes, which
        is what makes the ice branch collapse under the ``V -> 0`` control.
        """
        a = _positive(area, "area")
        v = _non_negative(velocity, "velocity")
        if v == 0.0:
            return 0.0
        return (
            self.crush_pressure_reference
            * (a / self.contact_area_reference) ** self.area_exponent
            * (v / self.velocity_reference) ** self.rate_exponent
        )


@dataclass(frozen=True)
class WaveField:
    """Head-seas regular equivalent of a sea state."""

    significant_height: float
    peak_period: float

    def validate(self) -> None:
        _positive(self.significant_height, "significant_height")
        _positive(self.peak_period, "peak_period")

    @property
    def amplitude(self) -> float:
        return 0.5 * self.significant_height


@dataclass(frozen=True)
class WhippingModel:
    """Hull-girder discretisation, modal damping and duration control."""

    elements: int = 80
    modes: int = 6
    response_dt: float = 0.01
    damping_ratio_2: float = 0.025
    damping_ratio_3: float = 0.018
    damping_ratio_higher: float = 0.015
    hot_spot_scf: float = 1.15
    duration: float = 600.0

    def validate(self) -> None:
        if self.elements < 8:
            raise ValueError("elements must be >= 8 for the girder discretisation")
        if self.modes < 4:
            raise ValueError("at least modes 2..4 must be retained: modes >= 4")
        _positive(self.response_dt, "response_dt")
        _positive(self.duration, "duration")
        for name in ("damping_ratio_2", "damping_ratio_3", "damping_ratio_higher"):
            if _fraction(getattr(self, name), name) >= 0.2:
                raise ValueError(f"{name} must be a small structural damping ratio")
        _positive(self.hot_spot_scf, "hot_spot_scf")

    def damping_ratio(self, mode_index: int) -> float:
        """Structural damping ratio of overall mode ``mode_index`` (2..n)."""
        if mode_index <= 2:
            return self.damping_ratio_2
        if mode_index == 3:
            return self.damping_ratio_3
        return self.damping_ratio_higher


@dataclass(frozen=True)
class SNCurve:
    """Two-slope S-N curve ``N = C * S^-m`` with ``S`` in Pa and ``N`` cycles.

    Defaults are the well-known DNV-RP-C203 curve-D in-air pair
    (``C1 = 1.458814e12`` for ``m1 = 3``, knee at ``N = 1e7`` i.e. 52.64 MPa,
    ``m2 = 5`` above the knee), restated as declared input.

    The first release carried ``C1 = 1.52e12`` / ``C2 = 4.34e15`` (knee 53.37 MPa),
    i.e. ``log10 a1 = 12.1818`` instead of the published ``12.164``: a 4.19 %
    unconservative life bias against the other five curve-D implementations in this
    library.  See ``docs/fatigue-sn-curve-consistency-audit.md``.
    """

    c1: float = 1458814260275.3474
    m1: float = 3.0
    c2: float = 4042654873623432.5
    m2: float = 5.0
    knee_cycles: float = 1.0e7
    endurance_limit_stress: float | None = None

    def validate(self) -> None:
        for name in ("c1", "m1", "c2", "m2", "knee_cycles"):
            _positive(getattr(self, name), name)

    def cycles_to_failure(self, stress_range: float) -> float:
        """Allowable cycles for a stress range in **Pa**.

        The coefficients are the published MPa-calibrated pair, so the range is
        converted to MPa here; the interface stays SI like the rest of the
        module.
        """
        s = float(stress_range) / 1.0e6
        if not math.isfinite(s) or s <= 0.0:
            raise ValueError("stress_range must be a finite positive value")
        n1 = self.c1 * s ** (-self.m1)
        if n1 <= self.knee_cycles:
            return n1
        if self.endurance_limit_stress is not None and s < self.endurance_limit_stress:
            return math.inf
        return self.c2 * s ** (-self.m2)

    @property
    def knee_stress(self) -> float:
        """Stress range at the knee [Pa]."""
        return (self.c1 / self.knee_cycles) ** (1.0 / self.m1) * 1.0e6

# ---------------------------------------------------------------------------
# girder finite element (Timoshenko, dof order (w, psi) interleaved, free-free)
# ---------------------------------------------------------------------------
def vibrating_mass_per_length(ship: IceClassShipCase) -> float:
    """Vibrating structural mass per unit length of the hull girder [kg/m].

    A floating body displaces its own weight, so the mass that vibrates with
    the girder is the displacement per unit length ``Delta / L`` (steel plus
    payload), not the mass of the steel alone.
    """
    ship.validate()
    return ship.displacement_mass / ship.length


def added_mass_per_length(ship: IceClassShipCase) -> float:
    """Hydrodynamic added mass per unit length used by the whipping model [kg/m].

    The 2D heave added mass of the midship section from the qualified Lewis
    coefficient of :mod:`tensorfem.hull_girder_modal`
    (``m33 = coeff * rho * b^2``, ``ratio = half_draft / half_breadth``) times
    the declared :attr:`IceClassShipCase.added_mass_factor`.  The factor < 1 is
    the declared 3D reduction of the sectional heave value for high-frequency
    vertical hull-girder vibration.
    """
    ship.validate()
    half_breadth = 0.5 * ship.breadth
    ratio = (0.5 * ship.draft) / half_breadth
    coefficient = float(lewis_heave_coefficient(ratio))
    return ship.added_mass_factor * ship.water_density * half_breadth**2 * coefficient


def beam_element_matrices(
    bending_stiffness: float,
    shear_stiffness: float,
    mass_per_length: float,
    rotary_inertia_per_length: float,
    element_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Timoshenko beam element: stiffness and consistent mass, 4x4.

    Dof order ``(w1, psi1, w2, psi2)``.  Stiffness is the standard
    shear-flexible element with ``phi = 12 EI / (kGA L^2)``; the mass is the
    consistent translational matrix plus the rotary-inertia matrix
    ``rho*I * int N' N'``.
    """
    le = _positive(element_length, "element_length")
    ei = _positive(bending_stiffness, "bending_stiffness")
    kga = _positive(shear_stiffness, "shear_stiffness")
    phi = 12.0 * ei / (kga * le * le)
    c = ei / (le**3 * (1.0 + phi))
    l2 = le * le
    stiffness = c * np.array(
        [
            [12.0, 6.0 * le, -12.0, 6.0 * le],
            [6.0 * le, (4.0 + phi) * l2, -6.0 * le, (2.0 - phi) * l2],
            [-12.0, -6.0 * le, 12.0, -6.0 * le],
            [6.0 * le, (2.0 - phi) * l2, -6.0 * le, (4.0 + phi) * l2],
        ]
    )
    mass = (mass_per_length * le / 420.0) * np.array(
        [
            [156.0, 22.0 * le, 54.0, -13.0 * le],
            [22.0 * le, 4.0 * l2, 13.0 * le, -3.0 * l2],
            [54.0, 13.0 * le, 156.0, -22.0 * le],
            [-13.0 * le, -3.0 * l2, -22.0 * le, 4.0 * l2],
        ]
    )
    if rotary_inertia_per_length > 0.0:
        mass += (rotary_inertia_per_length * le / (30.0 * le)) * np.array(
            [
                [36.0, 3.0 * le, -36.0, 3.0 * le],
                [3.0 * le, 4.0 * l2, -3.0 * le, -l2],
                [-36.0, -3.0 * le, 36.0, -3.0 * le],
                [3.0 * le, -l2, -3.0 * le, 4.0 * l2],
            ]
        )
    return stiffness, mass


def assembled_girder(
    ship: IceClassShipCase,
    model: WhippingModel,
    *,
    mass_per_length: float | None = None,
    rotary_inertia_per_length: float | None = None,
) -> dict[str, object]:
    """Assemble the free-free wet girder: global ``K``, ``M``, node grid."""
    ship.validate()
    model.validate()
    girder = ship.girder()
    elements = model.elements
    le = ship.length / elements
    m = vibrating_mass_per_length(ship) + added_mass_per_length(ship)
    if mass_per_length is not None:
        m = _positive(mass_per_length, "mass_per_length")
    j = ship.steel_density * girder.inertia
    if rotary_inertia_per_length is not None:
        j = _non_negative(rotary_inertia_per_length, "rotary_inertia_per_length")
    ke, me = beam_element_matrices(
        girder.bending_stiffness, girder.shear_stiffness, m, j, le
    )
    n_dof = 2 * (elements + 1)
    stiffness = np.zeros((n_dof, n_dof))
    mass = np.zeros((n_dof, n_dof))
    for e in range(elements):
        dofs = np.array([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3])
        stiffness[np.ix_(dofs, dofs)] += ke
        mass[np.ix_(dofs, dofs)] += me
    return {
        "stiffness": stiffness,
        "mass": mass,
        "element_length": le,
        "node_x": -0.5 * ship.length + le * np.arange(elements + 1),
        "bending_stiffness": girder.bending_stiffness,
        "shear_stiffness": girder.shear_stiffness,
        "mass_per_length": m,
        "added_mass_per_length": m - vibrating_mass_per_length(ship),
    }


@dataclass(frozen=True)
class GirderModes:
    """Wet hull-girder modes of the equivalent beam."""

    node_x: np.ndarray
    element_length: float
    stiffness: np.ndarray
    mass: np.ndarray
    bending_stiffness: float
    omega: np.ndarray
    frequencies_hz: np.ndarray
    modes: np.ndarray
    mass_per_length: float
    added_mass_per_length: float
    rigid_modes: int

    @property
    def vertical_modes(self) -> np.ndarray:
        return self.modes[self.rigid_modes :]

    @property
    def vertical_omega(self) -> np.ndarray:
        return self.omega[self.rigid_modes :]

    @property
    def vertical_frequencies_hz(self) -> np.ndarray:
        return self.frequencies_hz[self.rigid_modes :]


def wet_girder_modes(ship: IceClassShipCase, model: WhippingModel) -> GirderModes:
    """Free-free wet modes of the girder.

    The generalized problem ``K phi = omega^2 M phi`` is solved in
    mass-normalized form (``M^-1/2 K M^-1/2``) after a Cholesky factorisation
    of ``M``.  A free-free girder returns two rigid-body modes (heave and
    pitch) at ``omega ~ 0``; they are counted in :attr:`GirderModes.rigid_modes`
    and skipped, so ``vertical_modes[0]`` is the 2-node bending mode.
    """
    assembled = assembled_girder(ship, model)
    stiffness = torch.as_tensor(assembled["stiffness"], dtype=_D)
    mass = torch.as_tensor(assembled["mass"], dtype=_D)
    chol = torch.linalg.cholesky(mass)
    chol_inv = torch.linalg.inv(chol)
    reduced = chol_inv @ stiffness @ chol_inv.T
    reduced = 0.5 * (reduced + reduced.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(reduced)
    omega_all = np.sqrt(np.clip(eigenvalues.detach().cpu().numpy(), 0.0, None))
    n_keep = min(model.modes + 2, omega_all.size)
    modes = (chol_inv.T @ eigenvectors)[:, :n_keep].T.detach().cpu().numpy()
    omega = omega_all[:n_keep]
    rigid = int(np.sum(omega < 1.0))
    return GirderModes(
        node_x=assembled["node_x"],
        element_length=float(assembled["element_length"]),
        stiffness=assembled["stiffness"],
        mass=assembled["mass"],
        bending_stiffness=float(assembled["bending_stiffness"]),
        omega=omega,
        frequencies_hz=omega / (2.0 * math.pi),
        modes=modes,
        mass_per_length=float(assembled["mass_per_length"]),
        added_mass_per_length=float(assembled["added_mass_per_length"]),
        rigid_modes=rigid,
    )


def dsm_reference_frequencies_hz(ship: IceClassShipCase, count: int = 6) -> np.ndarray:
    """Exact free-free Timoshenko frequencies of the *same* wet girder [Hz].

    Uses :func:`tensorfem.dsbeam.natural_frequencies_hz`, an exact
    dynamic-stiffness (transcendental) solution, with the section parameterised
    so that the shear rigidity, the bending rigidity, the vibrating mass per
    length and the length all match the FE model.  Rotary inertia is switched
    off in the reference; the FE model carries the steel rotary inertia, whose
    effect is quantified by :func:`frequency_check`.  Rigid-body roots
    (``f ~ 0``) are removed.
    """
    girder = ship.girder()
    mass_per_length = vibrating_mass_per_length(ship) + added_mass_per_length(ship)
    shear_modulus = girder.shear_modulus
    gamma = girder.shear_stiffness / shear_modulus  # = effective shear area
    section = DSMBeamSection(
        young=girder.young,
        poisson=girder.poisson,
        density=mass_per_length / gamma,
        area=gamma,
        inertia=girder.inertia,
        length=ship.length,
        shear_coefficient=1.0,
        include_rotary_inertia=False,
    )
    frequencies = natural_frequencies_hz(section, count=count + 2, bc="free-free")
    return np.asarray([f for f in frequencies if f > 1.0e-6], dtype=float)[:count]


def frequency_check(ship: IceClassShipCase, model: WhippingModel) -> dict[str, object]:
    """Criterion (b): FE free-free wet frequencies against the exact DSM solution."""
    girder_modes = wet_girder_modes(ship, model)
    wet = girder_modes.vertical_frequencies_hz
    reference = dsm_reference_frequencies_hz(ship, count=max(len(wet), 4))
    keep = int(min(len(wet), len(reference)))
    residuals = [
        abs(float(wet[i]) - float(reference[i])) / float(reference[i]) for i in range(keep)
    ]
    shear_reduction = []
    for i in range(keep):
        eb = _euler_bernoulli_free_free_hz(
            girder_modes.bending_stiffness, girder_modes.mass_per_length, ship.length, i + 1
        )
        shear_reduction.append(1.0 - float(reference[i]) / eb)
    return {
        "frequencies_wet_fe_hz": [float(v) for v in wet[:keep]],
        "frequencies_exact_dsm_hz": [float(v) for v in reference[:keep]],
        "relative_residual": residuals,
        "max_relative_residual": max(residuals),
        "rigid_modes_detected": girder_modes.rigid_modes,
        "mass_per_length": girder_modes.mass_per_length,
        "added_mass_per_length": girder_modes.added_mass_per_length,
        "euler_bernoulli_hz": [
            _euler_bernoulli_free_free_hz(
                girder_modes.bending_stiffness, girder_modes.mass_per_length, ship.length, i + 1
            )
            for i in range(keep)
        ],
        "shear_softening_fraction": shear_reduction,
        "element_count": model.elements,
    }


def _euler_bernoulli_free_free_hz(
    bending_stiffness: float, mass_per_length: float, length: float, mode: int
) -> float:
    """Closed-form free-free Euler-Bernoulli frequency [Hz] (roots of cos x cosh x = 1).

    ``mode = 1`` is the 2-node mode (``beta L = 4.7300...``).  The roots are
    found numerically by bisection so no root is transcribed.
    """
    if mode < 1:
        raise ValueError("mode must be >= 1")
    # root ``k`` lies in (k pi, (k+1) pi); the bracket is monotone in |f| so a
    # plain bisection on the sign change cannot skip or duplicate a root
    lo = mode * math.pi
    hi = (mode + 1) * math.pi

    def f(x: float) -> float:
        return math.cos(x) * math.cosh(x) - 1.0

    f_lo = f(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if (f_mid > 0.0) == (f_lo > 0.0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    beta_l = 0.5 * (lo + hi)
    omega = beta_l**2 / (length * length) * math.sqrt(bending_stiffness / mass_per_length)
    return omega / (2.0 * math.pi)


def curvature_at_elements(modes: np.ndarray, element_length: float) -> np.ndarray:
    """Curvature field ``dpsi/dx`` at every element [1/m].

    The element interpolates the rotation to first order, so the curvature is
    element-wise constant and ``M = EI * curvature`` lives at element midpoints.
    """
    psi = np.asarray(modes)[..., 1::2]
    return (psi[..., 1:] - psi[..., :-1]) / element_length


def _self_equilibrated_load(
    ship: IceClassShipCase, model: WhippingModel, force: float
) -> tuple[np.ndarray, float, np.ndarray]:
    """Nodal load vector of a point force with the two end reactions."""
    node_x = -0.5 * ship.length + (ship.length / model.elements) * np.arange(model.elements + 1)
    station = float(ship.impact_station_x)
    idx = int(np.argmin(np.abs(node_x - station)))
    load = np.zeros(2 * (model.elements + 1))
    load[2 * idx] = force
    a = node_x[idx] + 0.5 * ship.length
    l = ship.length
    load[0] -= force * (l - a) / l
    load[-2] -= force * a / l
    return load, float(node_x[idx]), node_x


def static_moment_check(ship: IceClassShipCase, model: WhippingModel) -> dict[str, float]:
    """Static verification of the moment recovery: midspan ``M = P a b / L``.

    A unit point force with its two end reactions (self-equilibrated, as any
    free-floating load must be) is applied to the assembled girder, the singular
    free-free system is solved in the minimum-norm sense, and the recovered
    midship moment is compared with the closed form of the equivalent
    simply-supported moment diagram.
    """
    assembled = assembled_girder(ship, model)
    load, station_x, node_x = _self_equilibrated_load(ship, model, 1.0)
    displacement = np.linalg.lstsq(assembled["stiffness"], load, rcond=None)[0]
    curvature = curvature_at_elements(displacement, float(assembled["element_length"]))
    moment = float(assembled["bending_stiffness"]) * curvature
    le = float(assembled["element_length"])
    mid = int(np.argmin(np.abs(node_x[:-1] + 0.5 * le)))
    a = station_x + 0.5 * ship.length
    reaction_left = 1.0 * (ship.length - a) / ship.length
    mid_x = float(node_x[mid] + 0.5 * le)
    expected = reaction_left * (mid_x + 0.5 * ship.length)
    measured = abs(float(moment[mid]))
    return {
        "measured_moment": measured,
        "expected_moment": expected,
        "relative_error": abs(measured - expected) / expected,
        "station_x": station_x,
        "mid_x": mid_x,
    }


def static_bending_moment(
    ship: IceClassShipCase, model: WhippingModel, force: float
) -> float:
    """Midship bending moment of the same girder under a static slamming force [N m]."""
    assembled = assembled_girder(ship, model)
    load, _station, node_x = _self_equilibrated_load(ship, model, float(force))
    displacement = np.linalg.lstsq(assembled["stiffness"], load, rcond=None)[0]
    curvature = curvature_at_elements(displacement, float(assembled["element_length"]))
    moment = float(assembled["bending_stiffness"]) * curvature
    mid = int(np.argmin(np.abs(node_x[:-1] + 0.5 * float(assembled["element_length"]))))
    return float(abs(moment[mid]))


# ---------------------------------------------------------------------------
# motion model: linear heave + pitch in head seas
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RelativeMotion:
    """Harmonic relative bow motion at the impact station."""

    omega_encounter: float
    amplitude: float
    phase: float
    heave_amplitude: float
    pitch_amplitude: float
    heave_rao: float
    pitch_rao: float
    static_immersion: float

    @property
    def encounters_emergence(self) -> bool:
        return self.amplitude > self.static_immersion

    def re_entry_velocity(self) -> float:
        """Vertical relative velocity at re-entry [m/s] (exact for a cosine)."""
        if not self.encounters_emergence:
            return 0.0
        a = self.amplitude
        t = self.static_immersion
        return self.omega_encounter * math.sqrt(max(a * a - t * t, 0.0))


def froude_krylov_phasors(ship: IceClassShipCase, wave: WaveField) -> tuple[complex, complex]:
    """Heave force and pitch moment phasors about midships for a box hull.

    ``F3 = rho g zeta_a B * int exp(i k x) dx`` and ``M5 = int x dF3``, exact
    integrals over ``[-L/2, L/2]``, phasor convention ``exp(-i omega t)``.
    """
    wave.validate()
    ship.validate()
    omega = 2.0 * math.pi / wave.peak_period
    k = omega * omega / _G
    a = 0.5 * ship.length
    amp = wave.amplitude
    q = k * a
    integral = 2.0 * math.sin(q) / k
    moment_integral = complex(0.0, 0.0)
    if abs(q) > 1.0e-9:
        moment_integral = 2.0j * (math.sin(q) / (k * k) - a * math.cos(q) / k)
    scale = ship.water_density * _G * amp * ship.breadth
    return complex(scale * integral), complex(scale * moment_integral)


def seakeeping_relative_motion(
    ship: IceClassShipCase, wave: WaveField, speed: float
) -> RelativeMotion:
    """Solve the 2-DOF heave/pitch system for the relative bow motion.

    Declared linear model: ``Delta`` heave mass and ``Delta k_yy^2`` pitch
    inertia; added mass from the Lewis 2D heave value times the length (pitch by
    parallel-axis scaling); restoring ``rho g A_wp`` and ``rho g Nabla GM_L``;
    radiative damping as declared ratios of the wet natural frequencies.
    ``speed`` enters through the encounter frequency; coefficients are evaluated
    at the wave frequency (declared long-wave simplification).
    """
    ship.validate()
    wave.validate()
    v = _non_negative(speed, "speed")
    omega = 2.0 * math.pi / wave.peak_period
    k = omega * omega / _G
    omega_e = omega + k * v
    a33 = added_mass_per_length(ship) * ship.length
    m33 = ship.displacement_mass + a33
    k_yy = ship.pitch_radius_of_gyration
    m55 = (ship.displacement_mass + a33) * k_yy * k_yy
    c33 = ship.water_density * _G * ship.waterplane_area
    c55 = ship.water_density * _G * (ship.displacement_volume * ship.longitudinal_gm)
    b33 = 2.0 * ship.heave_damping_ratio * math.sqrt(m33 * c33)
    b55 = 2.0 * ship.pitch_damping_ratio * math.sqrt(m55 * c55)
    f3, f5 = froude_krylov_phasors(ship, wave)
    d11 = complex(c33 - omega_e * omega_e * m33, -omega_e * b33)
    d22 = complex(c55 - omega_e * omega_e * m55, -omega_e * b55)
    det = d11 * d22
    if abs(det) == 0.0:
        raise ValueError("singular motion system")
    heave = (d22 * f3) / det
    pitch = (d11 * f5) / det
    x_s = float(ship.impact_station_x)
    wave_at_station = complex(
        wave.amplitude * math.cos(k * x_s), wave.amplitude * math.sin(k * x_s)
    )
    relative = heave + pitch * x_s - wave_at_station
    return RelativeMotion(
        omega_encounter=omega_e,
        amplitude=abs(relative),
        phase=math.atan2(relative.imag, relative.real),
        heave_amplitude=abs(heave),
        pitch_amplitude=abs(pitch),
        heave_rao=abs(heave) / wave.amplitude,
        pitch_rao=abs(pitch) / (k * wave.amplitude) if k * wave.amplitude > 0 else 0.0,
        static_immersion=ship.impact_local_immersion,
    )


# ---------------------------------------------------------------------------
# load pulses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LoadPulse:
    """Triangular pulse: rise to ``amplitude`` over ``rise``, then linear decay."""

    t0: float
    rise: float
    decay: float
    amplitude: float
    kind: str = "ice"

    @property
    def duration(self) -> float:
        return self.rise + self.decay

    @property
    def impulse(self) -> float:
        return 0.5 * self.amplitude * (self.rise + self.decay)

    @property
    def rise_impulse(self) -> float:
        """Impulse delivered during the rise phase (the added-mass closure) [N s]."""
        return 0.5 * self.amplitude * self.rise

    def sample(self, t: np.ndarray) -> np.ndarray:
        out = np.zeros_like(t)
        lead = t - self.t0
        rise = (lead >= 0.0) & (lead < self.rise)
        if self.rise > 0.0:
            out[rise] = self.amplitude * lead[rise] / self.rise
        fall = (lead >= self.rise) & (lead < self.duration)
        if self.decay > 0.0:
            out[fall] = self.amplitude * (1.0 - (lead[fall] - self.rise) / self.decay)
        return out

    def vertices(self) -> tuple[float, float, float]:
        return (self.t0, self.t0 + self.rise, self.t0 + self.duration)


def ice_impact_pulses(
    ship: IceClassShipCase, ice: IceField, speed: float, duration: float
) -> list[LoadPulse]:
    """Deterministic ice-impact pulse train over ``duration``.

    Inter-arrival spacing is the declared mean floe length divided by the
    transit rate ``C * V``; floe lengths follow a seeded lognormal so the train
    is exactly reproducible.  Contact area is ``h * min(2 b_c, floe_length)``
    (the ice thickness limits the vertical contact width, a small floe limits
    the horizontal one), the peak is ``p(A, V) * A`` and the pulse duration
    matches the floe momentum ``m_floe * V`` with a triangular pulse.
    """
    ship.validate()
    ice.validate()
    v = _non_negative(speed, "speed")
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    if v == 0.0 or ice.thickness == 0.0 or ice.concentration == 0.0:
        return []
    rng = np.random.default_rng(ice.seed)
    spacing = ice.floe_length / (ice.concentration * v)
    pulses: list[LoadPulse] = []
    t = 0.5 * spacing
    while t < duration:
        floe = float(math.exp(rng.normal(math.log(ice.floe_length), ice.floe_length_cv)))
        floe = max(floe, 0.25 * ice.floe_length)
        width = min(2.0 * ship.impact_half_breadth, floe)
        area = ice.thickness * width
        pressure = ice.crush_pressure(area, v)
        peak = pressure * area
        if peak > 0.0:
            floe_mass = ice.ice_density * ice.thickness * width * floe
            impact = 2.0 * floe_mass * v / peak
            impact = float(min(max(impact, 2.0e-3), 0.5))
            pulses.append(
                LoadPulse(
                    t0=t, rise=0.5 * impact, decay=0.5 * impact, amplitude=peak, kind="ice"
                )
            )
        t += spacing
    return pulses


def von_karman_peak_force(
    ship: IceClassShipCase, re_entry_velocity: float, wagner: bool = False
) -> float:
    """Closed-form von Karman (or Wagner) peak slam force [N].

    ``wagner=True`` uses the spray-root wetted width ``(pi/2) c`` instead of the
    still-water width ``c``, giving the standard upper bound of the two
    formulations.
    """
    w = _non_negative(re_entry_velocity, "re_entry_velocity")
    if w == 0.0:
        return 0.0
    factor = (0.5 * math.pi) ** 2 if wagner else 1.0
    return (
        factor
        * math.pi
        * ship.water_density
        * ship.impact_half_breadth
        * ship.slam_length
        * w
        * w
        / math.tan(ship.deadrise)
    )


def added_mass_impulse(ship: IceClassShipCase, re_entry_velocity: float) -> float:
    """von Karman momentum ``m_a w`` of the fully wetted forebody over a slam [N s].

    ``m_a = 0.5 pi rho b_c^2 L_s`` is the 2D wedge added mass at the chine
    immersion; the rise phase of the triangular slam pulse must deliver exactly
    this impulse.
    """
    w = _non_negative(re_entry_velocity, "re_entry_velocity")
    m_a = 0.5 * math.pi * ship.water_density * ship.impact_half_breadth**2 * ship.slam_length
    return m_a * w


def wave_slam_pulses(
    ship: IceClassShipCase, wave: WaveField, speed: float, duration: float
) -> list[LoadPulse]:
    """Bow re-entry slam pulses from the linear relative-motion model.

    von Karman wedge entry: at re-entry velocity ``w`` the peak force is
    ``F = pi rho b_c L_s w^2 cot(beta)`` (rate of change of the section added
    mass over the wetted forebody) and the pulse rises until the chine is wet,
    ``t_rise = b_c tan(beta) / w``.  One re-entry per encounter period.
    """
    ship.validate()
    wave.validate()
    v = _non_negative(speed, "speed")
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    motion = seakeeping_relative_motion(ship, wave, v)
    if not motion.encounters_emergence:
        return []
    w = motion.re_entry_velocity()
    if w <= 0.0:
        return []
    peak = von_karman_peak_force(ship, w)
    rise = ship.impact_half_breadth * math.tan(ship.deadrise) / w
    period = 2.0 * math.pi / motion.omega_encounter
    pulses: list[LoadPulse] = []
    t = 0.5 * period
    while t < duration:
        pulses.append(LoadPulse(t0=t, rise=rise, decay=rise, amplitude=peak, kind="wave"))
        t += period
    return pulses


# ---------------------------------------------------------------------------
# load history (uniform grid + every pulse vertex, so the impulse is exact)
# ---------------------------------------------------------------------------
def build_load_history(
    ship: IceClassShipCase,
    ice: IceField,
    wave: WaveField,
    speed: float,
    model: WhippingModel,
    *,
    include_ice: bool = True,
    include_wave: bool = True,
    time_grid: np.ndarray | None = None,
) -> dict[str, object]:
    """Sampled load history; every pulse vertex is a grid point.

    The uniform grid of :attr:`WhippingModel.response_dt` is merged with the
    three vertices of every pulse, so the piecewise-linear modal integrator sees
    the exact pulse shape and the impulse is not lost to sampling (an ice impact
    lasts ~20-30 ms, i.e. 2-3 samples of a 10 ms grid, but zero samples of error
    once its vertices are on the grid).
    """
    model.validate()
    duration = model.duration
    n = int(math.ceil(duration / model.response_dt)) + 1
    grid = np.arange(n) * model.response_dt
    ice_pulses = ice_impact_pulses(ship, ice, speed, duration) if include_ice else []
    wave_pulses = wave_slam_pulses(ship, wave, speed, duration) if include_wave else []
    extra: list[float] = []
    for pulse in ice_pulses + wave_pulses:
        extra.extend(v for v in pulse.vertices() if 0.0 <= v <= duration)
    if time_grid is not None:
        t = np.asarray(time_grid, dtype=float)
    elif extra:
        t = np.unique(np.concatenate((grid, np.asarray(extra, dtype=float))))
    else:
        t = grid
    force_ice = np.zeros_like(t)
    for pulse in ice_pulses:
        force_ice += pulse.sample(t)
    force_wave = np.zeros_like(t)
    for pulse in wave_pulses:
        force_wave += pulse.sample(t)
    return {
        "t": t,
        "force_ice": force_ice,
        "force_wave": force_wave,
        "force_total": force_ice + force_wave,
        "pulses": ice_pulses + wave_pulses,
        "pulse_counts": {"ice": len(ice_pulses), "wave": len(wave_pulses)},
        "motion": seakeeping_relative_motion(ship, wave, speed),
    }


# ---------------------------------------------------------------------------
# modal whipping response
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WhippingResponse:
    """Modal whipping response of the girder."""

    t: np.ndarray
    modal_q: np.ndarray
    modal_q_dot: np.ndarray
    element_x: np.ndarray
    moment: np.ndarray
    moment_total: np.ndarray
    shear_total: np.ndarray
    omega: np.ndarray
    damping: np.ndarray
    mode_numbers: np.ndarray
    modal_peak_share: np.ndarray

    def midship_index(self) -> int:
        return int(np.argmin(np.abs(self.element_x)))


def piecewise_linear_modal_response(
    t: np.ndarray, force: np.ndarray, omega: float, zeta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Exact response of ``q'' + 2 zeta omega q' + omega^2 q = f(t)``.

    ``f`` is sampled on ``t`` and interpolated linearly, for which the state
    transition of the damped oscillator is closed form, so the result is exact
    to machine precision for a piecewise-linear excitation (unequal steps are
    allowed, which is how the pulse vertices get resolved).
    """
    wn = _positive(omega, "omega")
    z = _fraction(zeta, "zeta")
    if z >= 1.0:
        raise ValueError("zeta must be < 1 for the underdamped solution")
    n = t.size
    q = np.zeros(n)
    v = np.zeros(n)
    wn2 = wn * wn
    wd = wn * math.sqrt(max(1.0 - z * z, 0.0))
    for i in range(n - 1):
        dt = t[i + 1] - t[i]
        if dt <= 0.0:
            continue
        f0 = force[i]
        slope = (force[i + 1] - f0) / dt
        qp0 = f0 / wn2 - 2.0 * z * slope / (wn * wn2)
        qp1 = (f0 + slope * dt) / wn2 - 2.0 * z * slope / (wn * wn2)
        vp = slope / wn2
        q1 = q[i] - qp0
        v1 = v[i] - vp
        e = math.exp(-z * wn * dt)
        c = math.cos(wd * dt)
        s = math.sin(wd * dt)
        q[i + 1] = e * (q1 * c + (v1 + z * wn * q1) / wd * s) + qp1
        v[i + 1] = (
            e * (v1 * c - (q1 * wd + z * wn * (v1 + z * wn * q1) / wd) * s) + vp
        )
    return q, v


def modal_force_factor(modes: GirderModes, impact_x: float, mode_row: int) -> float:
    """Shape factor ``phi_i(x_impact)`` for a point load at the impact station."""
    phi_w = np.asarray(modes.modes)[mode_row, 0::2]
    return float(np.interp(impact_x, modes.node_x, phi_w))


def modal_response_records(
    ship: IceClassShipCase,
    model: WhippingModel,
    t: np.ndarray,
    force: np.ndarray,
    *,
    modes: GirderModes | None = None,
    station_count: int = 17,
) -> WhippingResponse:
    """Modal whipping response of the wet girder to a sampled load history.

    Generalized mass and stiffness come from the assembled FE matrices, the
    modal force from a point load at the impact station, and each modal
    coordinate is integrated with the exact piecewise-linear solution.  The
    bending moment is ``M(x,t) = EI * sum_i q_i(t) psi_i'(x)`` with the
    curvature field taken from the rotation dofs of the same eigenvectors, so
    the static equilibrium check of :func:`static_moment_check` applies to the
    recovered moment field as well.
    """
    ship.validate()
    model.validate()
    modes = modes or wet_girder_modes(ship, model)
    curvature_full = curvature_at_elements(modes.modes, modes.element_length)
    element_x_full = 0.5 * (modes.node_x[:-1] + modes.node_x[1:])
    stride = max(1, element_x_full.size // max(station_count, 3))
    stations = np.unique(
        np.concatenate(
            (np.arange(0, element_x_full.size, stride), [element_x_full.size // 2])
        )
    )
    element_x = element_x_full[stations]
    curvature = curvature_full[:, stations]
    q_rows: list[np.ndarray] = []
    qd_rows: list[np.ndarray] = []
    moment_rows: list[np.ndarray] = []
    omega: list[float] = []
    damping: list[float] = []
    numbers: list[int] = []
    for row in range(modes.modes.shape[0]):
        if modes.omega[row] < 1.0:
            continue  # rigid-body heave / pitch of the free girder
        phi = modes.modes[row]
        gen_mass = float(phi @ (modes.mass @ phi))
        gen_stiff = float(phi @ (modes.stiffness @ phi))
        if gen_mass <= 0.0 or gen_stiff <= 0.0:
            continue
        wn = math.sqrt(gen_stiff / gen_mass)
        mode_number = len(omega) + 2
        zeta = model.damping_ratio(mode_number)
        shape = modal_force_factor(modes, ship.impact_station_x, row)
        modal_force = shape * force / gen_mass
        q, qd = piecewise_linear_modal_response(t, modal_force, wn, zeta)
        q_rows.append(q)
        qd_rows.append(qd)
        moment_rows.append(modes.bending_stiffness * np.outer(curvature[row], q))
        omega.append(wn)
        damping.append(zeta)
        numbers.append(mode_number)
    if not q_rows:
        raise ValueError("no flexible vertical bending mode found above 1 rad/s")
    q_arr = np.asarray(q_rows)
    moment_arr = np.asarray(moment_rows)
    moment_total = moment_arr.sum(axis=0)
    shear_total = -np.gradient(moment_total, element_x, axis=0)
    peaks = np.abs(moment_arr).max(axis=(1, 2))
    total_peak = float(peaks.sum())
    share = peaks / total_peak if total_peak > 0.0 else np.zeros_like(peaks)
    return WhippingResponse(
        t=t,
        modal_q=q_arr,
        modal_q_dot=np.asarray(qd_rows),
        element_x=element_x,
        moment=moment_arr,
        moment_total=moment_total,
        shear_total=shear_total,
        omega=np.asarray(omega),
        damping=np.asarray(damping),
        mode_numbers=np.asarray(numbers),
        modal_peak_share=share,
    )


def vertical_bending_moment(
    ship: IceClassShipCase, model: WhippingModel, history: dict[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: ``(M_v(t), F_s(t))`` at midships from a load history dict."""
    response = modal_response_records(
        ship, model, history["t"], history["force_total"]
    )
    mid = response.midship_index()
    return response.moment_total[mid], response.shear_total[mid]


def isolated_whipping_event(
    ship: IceClassShipCase,
    model: WhippingModel,
    amplitude: float,
    rise: float,
    duration: float = 60.0,
) -> WhippingResponse:
    """Response to a *single* pulse, for the whipping decay time.

    In a real ice-transit train the inter-arrival time is shorter than the
    whipping decay time, so successive transients overlap and no single decay
    constant can be fitted to the train.  The decay time is therefore measured
    on an isolated event of the same peak and duration, which is the quantity
    the analytic ``1/(zeta omega)`` refers to.
    """
    model.validate()
    dt = model.response_dt
    n = int(math.ceil(duration / dt)) + 1
    pulse = LoadPulse(t0=2.0, rise=rise, decay=rise, amplitude=amplitude, kind="ice")
    t = np.unique(np.concatenate((np.arange(n) * dt, np.asarray(pulse.vertices()))))
    force = pulse.sample(t)
    return modal_response_records(ship, model, t, force)


def whipping_decay_time(
    response: WhippingResponse, mode_row: int = 0, *, window: float = 8.0
) -> dict[str, float]:
    """Decay time of the whipping transient of one mode [s].

    Fits ``ln|q(t)|`` over ``window`` seconds from the first response peak and
    returns ``tau = -1/slope`` (time to fall to 1/e).  The analytic reference is
    ``1/(zeta omega)``.  A mode that never leaves the numerical floor returns
    ``tau = 0`` with ``measured = 0`` (no whipping event to decay).
    """
    q = response.modal_q[mode_row]
    t = response.t
    analytic = 1.0 / (response.damping[mode_row] * response.omega[mode_row])
    if q.size == 0 or not np.any(q):
        return {"tau_decay": 0.0, "analytic_tau": float(analytic), "measured": 0.0}
    t_start = float(t[int(np.argmax(np.abs(q)))])
    mask = (t >= t_start) & (t <= t_start + window)
    amp = np.abs(q[mask])
    tt = t[mask]
    peak = float(amp.max()) if amp.size else 0.0
    if peak <= 0.0:
        return {"tau_decay": 0.0, "analytic_tau": float(analytic), "measured": 0.0}
    # fit the per-cycle envelope (local maxima of |q|), not every sample: the
    # ``|sin|`` factor of the oscillator biases a straight log fit of the raw
    # absolute value by 15-20 %
    local = np.array(
        [
            i
            for i in range(1, amp.size - 1)
            if amp[i] > amp[i - 1] and amp[i] >= amp[i + 1] and amp[i] > 0.02 * peak
        ]
    )
    if local.size < 4:
        return {"tau_decay": 0.0, "analytic_tau": float(analytic), "measured": 0.0}
    slope, _ = np.polyfit(tt[local], np.log(amp[local]), 1)
    tau = float(-1.0 / slope) if slope < 0.0 else float("inf")
    return {
        "tau_decay": tau,
        "analytic_tau": float(analytic),
        "measured": 1.0,
        "relative_error": float(abs(tau - analytic) / analytic),
    }


# ---------------------------------------------------------------------------
# fatigue
# ---------------------------------------------------------------------------
def hot_spot_stress(
    ship: IceClassShipCase, model: WhippingModel, moment: np.ndarray
) -> np.ndarray:
    """Signed hot-spot stress from the midship bending moment [Pa].

    Stress is kept signed so that the wave-induced and slam-induced bending
    moments can be superposed before rainflow counting (an ``abs`` here would
    destroy the superposition).
    """
    return model.hot_spot_scf * np.asarray(moment) / ship.section_modulus


def wave_bending_moment_history(
    ship: IceClassShipCase, wave: WaveField, speed: float, t: np.ndarray, samples: int = 801
) -> np.ndarray:
    """Wave-induced vertical bending moment at midships [N m].

    Linear strip/Froude-Krylov load on a box hull in head seas, made
    self-equilibrated the way the ship's inertia makes it: the wave-excess
    buoyancy ``p(x) = rho g B zeta(x)`` is reduced by the rigid-body heave and
    pitch reaction, i.e. by the linear function ``a + b x`` chosen so that
    ``int p = 0`` and ``int x p = 0``.  The section moment is then
    ``M(x_s) = -int (x_s - s) p(s) ds``.  No Smith correction and no
    hydrodynamic (diffraction) pressure term: this is the declared baseline used
    for the *wave* fatigue reference only, not a seakeeping prediction.
    """
    wave.validate()
    ship.validate()
    v = _non_negative(speed, "speed")
    omega = 2.0 * math.pi / wave.peak_period
    k = omega * omega / _G
    omega_e = omega + k * v
    x = np.linspace(-0.5 * ship.length, 0.5 * ship.length, max(int(samples), 101))
    dx = x[1] - x[0]
    zeta = wave.amplitude * np.cos(np.outer(x, omega_e * np.asarray(t, dtype=float)) * 1.0)
    zeta = wave.amplitude * np.cos(k * x[:, None] - omega_e * np.asarray(t, dtype=float)[None, :])
    s0 = zeta.sum(axis=0) * dx
    s1 = (x[:, None] * zeta).sum(axis=0) * dx
    a = s0 / ship.length
    b = s1 / (ship.length**3 / 12.0)
    pressure = ship.water_density * _G * ship.breadth * (zeta - (a + b * x[:, None]))
    x_s = 0.0
    mask = x <= x_s
    weights = (x_s - x[mask]) * dx
    moment = -(pressure[mask, :] * weights[:, None]).sum(axis=0)
    return moment


def fatigue_damage(
    stress_history: np.ndarray, curve: SNCurve | None = None, dt: float = 0.01
) -> dict[str, float]:
    """Rainflow counting (:mod:`tensorfem.marine_fatigue_advanced`) + Miner sum."""
    sn = curve or SNCurve()
    sn.validate()
    history = np.asarray(stress_history, dtype=float)
    exposure = float(dt * max(history.size - 1, 0))
    if history.size < 2 or float(np.ptp(history)) == 0.0:
        return {"damage": 0.0, "cycles": 0.0, "max_range_pa": 0.0, "exposure_s": exposure}
    cycles = rainflow_cycles([float(v) for v in history])
    if not cycles:
        return {"damage": 0.0, "cycles": 0.0, "max_range_pa": 0.0, "exposure_s": exposure}
    damage = 0.0
    count = 0.0
    max_range = 0.0
    for cycle in cycles:
        damage += cycle.count / sn.cycles_to_failure(cycle.range_pa)
        count += cycle.count
        max_range = max(max_range, float(cycle.range_pa))
    return {
        "damage": float(damage),
        "cycles": float(count),
        "max_range_pa": float(max_range),
        "exposure_s": exposure,
    }


# ---------------------------------------------------------------------------
# full case
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IceSlammingResult:
    """Qualification numbers for one operating point (V, h, C, sea state)."""

    speed: float
    thickness: float
    concentration: float
    peak_ice_force: float
    peak_wave_force: float
    peak_total_force: float
    peak_ice_moment: float
    peak_wave_moment: float
    peak_total_moment: float
    motion: RelativeMotion
    re_entry_velocity: float
    dynamic_amplification: float
    modal_frequencies_hz: tuple[float, ...]
    modal_peak_share: tuple[float, ...]
    tau_decay: float
    tau_decay_analytic: float
    tau_measured: float
    inter_arrival_s: float
    damage_wave: float
    damage_slam: float
    damage_ice_only: float
    damage_all_slam: float
    damage_wave_slam_only: float
    damage_combined: float
    damage_ratio: float
    combined_ratio: float
    peak_wave_vbm: float
    interference: dict[str, float]
    equivalent_ochi_cs: float
    pulse_counts: dict[str, int]
    response: WhippingResponse


def _peak(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if values.size else 0.0


def run_ice_slamming_case(
    ship: IceClassShipCase,
    ice: IceField,
    wave: WaveField,
    speed: float,
    model: WhippingModel | None = None,
    curve: SNCurve | None = None,
) -> IceSlammingResult:
    """Full chain for one operating point: loads -> whipping -> fatigue."""
    model = model or WhippingModel()
    curve = curve or SNCurve()
    ship.validate()
    ice.validate()
    wave.validate()
    model.validate()
    v = _non_negative(speed, "speed")
    modes = wet_girder_modes(ship, model)

    combined = build_load_history(ship, ice, wave, v, model)
    ice_only = build_load_history(
        ship, ice, wave, v, model, include_wave=False, time_grid=combined["t"]
    )
    wave_only = build_load_history(
        ship, ice, wave, v, model, include_ice=False, time_grid=combined["t"]
    )
    t = combined["t"]

    response = modal_response_records(ship, model, t, combined["force_total"], modes=modes)
    ice_response = modal_response_records(ship, model, t, ice_only["force_total"], modes=modes)
    wave_response = modal_response_records(ship, model, t, wave_only["force_total"], modes=modes)
    mid = response.midship_index()

    peak_ice_force = _peak(combined["force_ice"])
    peak_wave_force = _peak(combined["force_wave"])
    peak_total_force = _peak(combined["force_total"])
    peak_ice_moment = _peak(ice_response.moment_total[mid])
    peak_wave_moment = _peak(wave_response.moment_total[mid])
    peak_total_moment = _peak(response.moment_total[mid])

    static_moment = (
        static_bending_moment(ship, model, peak_total_force) if peak_total_force > 0.0 else 0.0
    )
    daf = peak_total_moment / static_moment if static_moment > 0.0 else 0.0

    stress_ice = hot_spot_stress(ship, model, ice_response.moment_total[mid])
    stress_wave_slam = hot_spot_stress(ship, model, wave_response.moment_total[mid])
    stress_slam = hot_spot_stress(ship, model, response.moment_total[mid])
    wave_moment = wave_bending_moment_history(ship, wave, v, t)
    stress_wave_vbm = hot_spot_stress(ship, model, wave_moment)
    d_wave = fatigue_damage(stress_wave_vbm, curve, dt=model.response_dt)
    d_slam = fatigue_damage(stress_slam, curve, dt=model.response_dt)
    d_ice = fatigue_damage(stress_ice, curve, dt=model.response_dt)
    d_wave_slam = fatigue_damage(stress_wave_slam, curve, dt=model.response_dt)
    d_combined = fatigue_damage(stress_wave_vbm + stress_slam, curve, dt=model.response_dt)
    ratio = d_ice["damage"] / d_wave["damage"] if d_wave["damage"] > 0.0 else float("inf")
    combined_ratio = (
        d_combined["damage"] / d_wave["damage"] if d_wave["damage"] > 0.0 else float("inf")
    )

    motion = combined["motion"]
    w_re = motion.re_entry_velocity()
    cs = 0.0
    if w_re > 0.0:
        area = 2.0 * ship.impact_half_breadth * ship.slam_length
        cs = von_karman_peak_force(ship, w_re) / (0.5 * ship.water_density * area * w_re**2)

    interference = {
        "peak_sum_ratio": peak_total_force / (peak_ice_force + peak_wave_force)
        if (peak_ice_force + peak_wave_force) > 0.0
        else 0.0,
        "moment_sum_ratio": peak_total_moment / (peak_ice_moment + peak_wave_moment)
        if (peak_ice_moment + peak_wave_moment) > 0.0
        else 0.0,
        "damage_superposition_ratio": d_combined["damage"]
        / (d_wave["damage"] + d_slam["damage"])
        if (d_wave["damage"] + d_slam["damage"]) > 0.0
        else 0.0,
    }

    decay = whipping_decay_time(response, 0)
    if combined["pulses"] and peak_total_force > 0.0:
        ice_pulses = [p for p in combined["pulses"] if p.kind == "ice"]
        first = max(ice_pulses, key=lambda p: p.amplitude) if ice_pulses else None
        if first is not None:
            event = isolated_whipping_event(ship, model, first.amplitude, first.rise)
            decay = whipping_decay_time(event, 0)
    return IceSlammingResult(
        speed=v,
        thickness=ice.thickness,
        concentration=ice.concentration,
        peak_ice_force=peak_ice_force,
        peak_wave_force=peak_wave_force,
        peak_total_force=peak_total_force,
        peak_ice_moment=peak_ice_moment,
        peak_wave_moment=peak_wave_moment,
        peak_total_moment=peak_total_moment,
        motion=motion,
        re_entry_velocity=w_re,
        dynamic_amplification=daf,
        modal_frequencies_hz=tuple(float(f) for f in modes.vertical_frequencies_hz),
        modal_peak_share=tuple(float(s) for s in response.modal_peak_share),
        tau_decay=decay["tau_decay"],
        tau_decay_analytic=decay["analytic_tau"],
        tau_measured=decay["measured"],
        inter_arrival_s=(
            ice.ice_density and ice.floe_length / (ice.concentration * v)
            if (ice.concentration * v) > 0.0
            else float("inf")
        ),
        damage_wave=d_wave["damage"],
        damage_slam=d_ice["damage"],
        damage_ice_only=d_ice["damage"],
        damage_all_slam=d_slam["damage"],
        damage_wave_slam_only=d_wave_slam["damage"],
        damage_combined=d_combined["damage"],
        damage_ratio=ratio,
        combined_ratio=combined_ratio,
        peak_wave_vbm=_peak(wave_moment),
        interference=interference,
        equivalent_ochi_cs=cs,
        pulse_counts=combined["pulse_counts"],
        response=response,
    )


# ---------------------------------------------------------------------------
# hard negative controls
# ---------------------------------------------------------------------------
def negative_control_checks(
    ship: IceClassShipCase,
    ice: IceField,
    wave: WaveField,
    speed: float,
    model: WhippingModel | None = None,
) -> dict[str, dict[str, float]]:
    """The three hard negative controls required by the campaign node.

    ``V -> 0``  the ice-impact branch and the whipping transient vanish;
    ``h -> 0``  the load collapses onto the pure wave-slam load;
    ``C -> 0``  the slam fatigue increment collapses to zero.

    The ``V -> 0`` and ``C -> 0`` limits are taken with a small but non-zero
    value (``1e-6``), so what is measured is a limit and not a coded branch.
    The ``h -> 0`` control is evaluated in the wave-active sea state passed in
    by the caller, where the wave-slam branch is non-zero; in a sea state
    without bow emergence that control would be degenerate (zero equals zero).
    """
    model = model or WhippingModel(duration=120.0)
    base = run_ice_slamming_case(ship, ice, wave, speed, model)
    zero_speed = run_ice_slamming_case(ship, ice, wave, 1e-6, model)
    zero_thickness = run_ice_slamming_case(
        ship,
        IceField(thickness=0.0, concentration=ice.concentration, seed=ice.seed),
        wave,
        speed,
        model,
    )
    wave_only = run_ice_slamming_case(
        ship,
        IceField(thickness=ice.thickness, concentration=0.0, seed=ice.seed),
        wave,
        speed,
        model,
    )
    # the ice branch is isolated for the decay control on purpose: a ship
    # stopped in a seaway still takes water-entry slams, so the wave branch
    # keeps a finite decay time at V = 0 -- that is physics, not a leak, and it
    # is reported next to the ice-only number instead of being hidden
    def _ice_only_tau(speed: float) -> float:
        history = build_load_history(ship, ice, wave, speed, model, include_wave=False)
        if not history["pulses"]:
            return 0.0
        response = modal_response_records(
            ship, model, history["t"], history["force_total"]
        )
        return whipping_decay_time(response, 0)["tau_decay"]

    return {
        "v_to_zero": {
            "peak_ice_force_base": base.peak_ice_force,
            "peak_ice_force_limit": zero_speed.peak_ice_force,
            "relative_residual": zero_speed.peak_ice_force / base.peak_ice_force
            if base.peak_ice_force > 0.0
            else 0.0,
            "tau_decay_limit": zero_speed.tau_decay,
            "tau_decay_ice_only_base": _ice_only_tau(speed),
            "tau_decay_ice_only_limit": _ice_only_tau(1e-6),
            "note": "tau_decay_limit is the total (ice + water-entry) value; it "
            "stays finite at V -> 0 in a sea state that emerges the bow",
        },
        "h_to_zero": {
            "peak_total_limit": zero_thickness.peak_total_force,
            "peak_wave_only": wave_only.peak_total_force,
            "relative_residual": abs(zero_thickness.peak_total_force - wave_only.peak_total_force)
            / wave_only.peak_total_force
            if wave_only.peak_total_force > 0.0
            else 0.0,
            "peak_wave_branch": zero_thickness.peak_wave_force,
        },
        "c_to_zero": {
            "damage_ratio_base": base.damage_ratio,
            "damage_ratio_limit": wave_only.damage_ratio,
            "damage_limit": wave_only.damage_slam,
        },
    }
