"""ISO 19906 global pressure--area law as the contact load source of a 6-DOF
time-domain integration of a ship advancing into level ice.

What this module is
-------------------
A small, auditable *time-domain* model, not a CFD/DEM ice solver:

* **Contact pressure source** -- the ISO 19906 global pressure--area law

      p_G = C_R * (A / A_0) ** ex,        A = contact area,  ex < 0

  The law is used exactly as a pressure *source*: the contact force on the
  hull is ``N = p_G(A) * A``.  Because ``ex < 0`` the pressure falls and the
  force rises sub-linearly with the contact area -- the whole point of the
  area law -- so the force is dominated by the smallest areas that occur,
  which is why the contact kinematics below matter.

* **Contact kinematics** -- the bow is idealised as a vertical wedge with a
  waterline half-angle ``half_angle_deg``, so the contact width grows with the
  penetration ``d`` measured from the (moving) ice edge, ``w = 2 d tan(alpha)``
  capped by the hull breadth.  The contact area is ``A = h * w`` for ice of
  thickness ``h``.  The ice edge itself advances (i.e. ice is crushed) in
  discrete *spall* increments ``spall_depth_m``: when the penetration reaches
  that increment the contact zone fails, the edge advances and the penetration
  collapses back to ``residual_ratio * spall_depth_m``.  This is the classical
  crushing/spalling saw-tooth idealisation; the saw-tooth *is* the load cycle
  that a design envelope such as ISO 19906 has to cover.

* **Structure** -- a linear (small-motion) 6-DOF rigid body with constant
  rigid + added mass matrix, hydrostatic restoring in heave/roll/pitch and
  linear (plus optional quadratic) damping.  The contact force is applied at
  the bow/ice-line point, so it excites surge, heave and pitch together; the
  contact normal carries an oblique (lift) component through
  ``normal_obliquity_deg``, which is what couples crushing into heave/pitch.
  A thrust governor (resistance feed-forward + proportional speed error)
  holds the commanded open-water speed.

* **Energy accounting** -- every non-conservative generalised force is
  accumulated as discrete work, so the module reports the residual of

      W_thrust + W_ice_normal + W_ice_friction + W_damping = dKE + dPE

  The primary residual uses the trapezoidal work quadrature consistent with
  the velocity-Verlet scheme; a second, deliberately lower-order
  (left-endpoint) quadrature is reported alongside so that a small residual is
  not merely a tautology of the bookkeeping: when the time step is halved the
  trapezoidal residual must fall quadratically while the left-endpoint one
  falls only linearly.

Scope / honesty notes
---------------------
* The ISO 19906 coefficients (``C_R``, ``ex``) are *inputs* of
  :class:`ISO19906PressureArea`.  The defaults are declared reference values
  (``C_R = 1 MPa``, ``A_0 = 1 m2``, ``ex = -0.28``) and are **not** a
  transcription of the standard's table; the normative numbers must come from
  the anchor document (see ``benchmarks/w22_ice_6dof_ensemble.py --anchor``).
  The contact force scales **exactly linearly** in ``C_R`` and only through
  ``1 + ex`` in the area, so energy-balance / convergence / monotonicity
  conclusions are invariant to ``C_R``; the ``ex`` dependence is reported as a
  sensitivity sweep.
* Added mass is constant (frequency-independent), the body is linearised
  (small rotations), ice is level and homogeneous, and no fluid is coupled.
  The crushing/spalling increments are engineering assumptions, not measured
  values.
* Everything is deterministic: no random numbers, so two runs of the same case
  are bit-identical.

All quantities are SI (m, kg, s, N, Pa) unless a field name carries a unit
suffix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import json
import math
from pathlib import Path

import numpy as np

__all__ = [
    "ISO19906PressureArea",
    "IceEdgeContact",
    "Vessel6DOF",
    "CaseConfig",
    "CaseTimeSeries",
    "load_anchor_parameters",
    "contact_area_m2",
    "run_case",
    "case_statistics",
    "energy_balance",
    "windowed_energy_residual",
    "damping_dissipation_crosscheck",
    "count_load_cycles",
    "spearman_rank",
    "natural_periods",
]

GRAVITY = 9.80665


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #
def _finite(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise TypeError(f"{name} must be a real scalar")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _positive(name: str, value) -> float:
    out = _finite(name, value)
    if out <= 0.0:
        raise ValueError(f"{name} must be positive")
    return out


def _nonnegative(name: str, value) -> float:
    out = _finite(name, value)
    if out < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return out


# --------------------------------------------------------------------------- #
# pressure--area law
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ISO19906PressureArea:
    """Global ice pressure from the ISO 19906 pressure--area relation.

    ``p_G = C_R * (A / A_0) ** ex`` with ``ex < 0`` for the *global* pressure
    branch (the local-pressure branch of the same standard uses a different
    exponent and is out of scope here).  The contact force is ``p_G * A``, i.e.
    proportional to ``A ** (1 + ex)``.

    The parameters are deliberately *declared inputs*: ``provenance`` records
    where the numbers came from (anchor transcription, sensitivity assumption,
    or reference normalisation).
    """

    reference_pressure_MPa: float = 1.0
    reference_area_m2: float = 1.0
    exponent: float = -0.28
    scope: str = "global"
    provenance: str = (
        "declared reference normalisation, NOT a transcription of the ISO 19906 "
        "table: C_R = 1 MPa at A_0 = 1 m2 with ex = -0.28 (global-pressure "
        "branch, ex < 0); absolute loads scale exactly linearly in C_R"
    )

    def __post_init__(self) -> None:
        _positive("reference_pressure_MPa", self.reference_pressure_MPa)
        _positive("reference_area_m2", self.reference_area_m2)
        exponent = _finite("exponent", self.exponent)
        if self.scope not in ("global", "local"):
            raise ValueError("scope must be 'global' or 'local'")
        if self.scope == "global" and exponent >= 0.0:
            raise ValueError(
                "the global pressure branch requires a negative exponent "
                f"(global pressure decreases with area); got {exponent}"
            )

    def pressure_MPa(self, area_m2) -> float:
        area = _positive("area_m2", area_m2)
        return self.reference_pressure_MPa * (area / self.reference_area_m2) ** self.exponent

    def pressure_MPa_array(self, area_m2):
        area = np.asarray(area_m2, dtype=float)
        if np.any(area <= 0.0):
            raise ValueError("area_m2 must be positive")
        return self.reference_pressure_MPa * (area / self.reference_area_m2) ** self.exponent

    def force_N(self, area_m2) -> float:
        """Contact force carried by the area law: ``p_G(A) * A``."""
        area = _positive("area_m2", area_m2)
        return self.pressure_MPa(area) * 1.0e6 * area

    def force_N_array(self, area_m2):
        area = np.asarray(area_m2, dtype=float)
        return self.pressure_MPa_array(area) * 1.0e6 * area

    @property
    def force_area_exponent(self) -> float:
        """Exponent of ``N(A)``: ``1 + ex`` (0.72 for ``ex = -0.28``)."""
        return 1.0 + float(self.exponent)


# --------------------------------------------------------------------------- #
# contact geometry / spalling
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IceEdgeContact:
    """Wedge bow, ice-line contact area and the crushing/spalling rule."""

    half_angle_deg: float = 40.0
    breadth_m: float = 20.0
    spall_depth_m: float = 0.30
    residual_ratio: float = 0.20
    friction: float = 0.15
    normal_obliquity_deg: float = 15.0
    contact_height_ratio: float = 1.0
    provenance: str = (
        "wedge waterline half-angle / spall increment / residual ratio / "
        "friction / oblique contact normal are engineering assumptions of this "
        "model, not ISO 19906 values; residual_ratio fixes the saw-tooth "
        "peak-to-mean load ratio"
    )

    def __post_init__(self) -> None:
        _positive("half_angle_deg", self.half_angle_deg)
        if self.half_angle_deg >= 90.0:
            raise ValueError("half_angle_deg must be < 90 deg (wedge bow)")
        _positive("breadth_m", self.breadth_m)
        _positive("spall_depth_m", self.spall_depth_m)
        ratio = _nonnegative("residual_ratio", self.residual_ratio)
        if ratio >= 1.0:
            raise ValueError("residual_ratio must be < 1 (a spall must shed area)")
        _nonnegative("friction", self.friction)
        obliquity = _nonnegative("normal_obliquity_deg", self.normal_obliquity_deg)
        if obliquity >= 90.0:
            raise ValueError("normal_obliquity_deg must be < 90 deg")
        _positive("contact_height_ratio", self.contact_height_ratio)

    def contact_width_m(self, penetration_m) -> float:
        """Waterline contact width of the wedge, capped by the hull breadth."""
        penetration = _nonnegative("penetration_m", penetration_m)
        width = 2.0 * penetration * math.tan(math.radians(self.half_angle_deg))
        return min(self.breadth_m, width)

    def area_m2(self, penetration_m, thickness_m) -> float:
        """Contact area ``h * w(d)`` (zero outside contact)."""
        thickness = _positive("thickness_m", thickness_m)
        if penetration_m <= 0.0:
            return 0.0
        height = min(thickness * self.contact_height_ratio, thickness)
        return self.contact_width_m(penetration_m) * height

    @property
    def residual_penetration_m(self) -> float:
        return self.residual_ratio * self.spall_depth_m

    @property
    def normal_direction(self) -> np.ndarray:
        """Unit contact normal pointing from the ice into the hull."""
        gamma = math.radians(self.normal_obliquity_deg)
        return np.array([-math.cos(gamma), 0.0, math.sin(gamma)], dtype=float)


def contact_area_m2(contact: IceEdgeContact, penetration_m, thickness_m) -> float:
    """Module-level alias, convenient for evidence scripts."""
    return contact.area_m2(penetration_m, thickness_m)


# --------------------------------------------------------------------------- #
# vessel
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Vessel6DOF:
    """Linear 6-DOF rigid body (surge, sway, heave, roll, pitch, yaw).

    Constant added-mass coefficients are ratios of the rigid diagonal; the
    hydrostatic restoring is linear in heave, roll and pitch; damping is linear
    per DOF (tuned from modal damping ratios) plus an optional quadratic term
    in surge and heave.
    """

    length_m: float = 100.0
    breadth_m: float = 20.0
    draught_m: float = 6.0
    block_coefficient: float = 0.70
    water_density: float = 1025.0
    gravity: float = GRAVITY
    kxx_m: float = 7.0
    kyy_m: float = 25.0
    kzz_m: float = 26.0
    added_mass_ratio: tuple = (0.10, 0.80, 0.90, 0.20, 1.00, 1.00)
    waterplane_area_ratio: float = 0.85
    metacentric_height_transverse_m: float = 1.2
    metacentric_height_longitudinal_m: float = 100.0
    damping_ratio: tuple = (0.0, 0.0, 0.20, 0.05, 0.15, 0.0)
    resistance_linear: float = 3.0e4
    resistance_quadratic: float = 4.5e4
    surge_quadratic_damping: float = 0.0
    heave_quadratic_damping: float = 0.0
    x_bow_m: float = 50.0
    z_ice_line_m: float = -5.0
    thrust_gain: float = 2.0e6
    provenance: str = (
        "generic 100 m / 20 m / 6 m / Cb 0.70 single-screw hull with textbook "
        "added-mass ratios and metacentric heights; a representative numerical "
        "test bed, not a particular ship"
    )

    def __post_init__(self) -> None:
        for name in (
            "length_m",
            "breadth_m",
            "draught_m",
            "block_coefficient",
            "water_density",
            "gravity",
            "kxx_m",
            "kyy_m",
            "kzz_m",
            "waterplane_area_ratio",
            "metacentric_height_transverse_m",
            "metacentric_height_longitudinal_m",
            "x_bow_m",
            "thrust_gain",
        ):
            _positive(name, getattr(self, name))
        if self.block_coefficient > 1.0:
            raise ValueError("block_coefficient must be <= 1")
        if self.z_ice_line_m >= 0.0:
            raise ValueError("z_ice_line_m must be below the centre of gravity")
        if len(self.added_mass_ratio) != 6:
            raise ValueError("added_mass_ratio must have 6 entries")
        if len(self.damping_ratio) != 6:
            raise ValueError("damping_ratio must have 6 entries")
        for i, ratio in enumerate(self.added_mass_ratio):
            _nonnegative(f"added_mass_ratio[{i}]", ratio)
        for i, ratio in enumerate(self.damping_ratio):
            _nonnegative(f"damping_ratio[{i}]", ratio)
        _nonnegative("surge_quadratic_damping", self.surge_quadratic_damping)
        _nonnegative("heave_quadratic_damping", self.heave_quadratic_damping)

    # -- mass / stiffness --------------------------------------------------- #
    @property
    def displacement_m3(self) -> float:
        return self.block_coefficient * self.length_m * self.breadth_m * self.draught_m

    @property
    def mass_kg(self) -> float:
        return self.water_density * self.displacement_m3

    @property
    def waterplane_area_m2(self) -> float:
        return self.waterplane_area_ratio * self.length_m * self.breadth_m

    def rigid_inertia(self) -> np.ndarray:
        m = self.mass_kg
        return np.array([m, m, m, m * self.kxx_m**2, m * self.kyy_m**2, m * self.kzz_m**2])

    def mass_matrix(self) -> np.ndarray:
        """Rigid + constant added mass, symmetric positive definite diagonal."""
        rigid = self.rigid_inertia()
        ratios = np.asarray(self.added_mass_ratio, dtype=float)
        return np.diag(rigid * (1.0 + ratios))

    def stiffness(self) -> np.ndarray:
        """Linear hydrostatic restoring in heave, roll and pitch (diagonal)."""
        rho_g = self.water_density * self.gravity
        k = np.zeros(6)
        k[2] = rho_g * self.waterplane_area_m2
        k[3] = rho_g * self.displacement_m3 * self.metacentric_height_transverse_m
        k[4] = rho_g * self.displacement_m3 * self.metacentric_height_longitudinal_m
        return k

    def damping(self) -> np.ndarray:
        """Linear damping, tuned from the modal damping ratios."""
        mass = np.diag(self.mass_matrix())
        stiff = self.stiffness()
        c = np.zeros(6)
        for dof, ratio in enumerate(self.damping_ratio):
            if ratio > 0.0 and stiff[dof] > 0.0:
                omega = math.sqrt(stiff[dof] / mass[dof])
                c[dof] = 2.0 * ratio * omega * mass[dof]
        c[0] = self.resistance_linear
        return c

    def open_water_resistance(self, speed: float) -> float:
        """Feed-forward thrust: quadratic ship resistance in calm water."""
        u = max(0.0, float(speed))
        return self.resistance_linear * u + self.resistance_quadratic * u * u


def natural_periods(vessel: Vessel6DOF) -> dict:
    """Undamped natural periods (s) of heave/roll/pitch; inf if unrestored."""
    mass = np.diag(vessel.mass_matrix())
    stiff = vessel.stiffness()
    out = {}
    for name, dof in (("heave", 2), ("roll", 3), ("pitch", 4)):
        out[name] = (
            math.inf if stiff[dof] <= 0.0 else 2.0 * math.pi / math.sqrt(stiff[dof] / mass[dof])
        )
    return out


# --------------------------------------------------------------------------- #
# configuration and result containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CaseConfig:
    """One ensemble member: ice thickness, commanded speed, law and contact."""

    thickness_m: float
    commanded_speed_mps: float
    law: ISO19906PressureArea = field(default_factory=ISO19906PressureArea)
    contact: IceEdgeContact = field(default_factory=IceEdgeContact)
    vessel: Vessel6DOF = field(default_factory=Vessel6DOF)
    name: str = ""

    def __post_init__(self) -> None:
        _positive("thickness_m", self.thickness_m)
        _positive("commanded_speed_mps", self.commanded_speed_mps)
        if not self.name:
            object.__setattr__(
                self,
                "name",
                f"h{self.thickness_m:.2f}_U{self.commanded_speed_mps:.2f}",
            )


@dataclass
class CaseTimeSeries:
    """Time histories of one case (all SI, arrays of equal length)."""

    config: CaseConfig
    dt_s: float
    time_s: np.ndarray
    displacement: np.ndarray  # (6, N)
    velocity: np.ndarray  # (6, N)
    contact_force_N: np.ndarray  # normal force magnitude
    contact_area_m2: np.ndarray
    contact_pressure_MPa: np.ndarray
    penetration_m: np.ndarray
    resistance_N: np.ndarray  # longitudinal component of the contact force
    spall_events: int
    kinetic_energy_J: np.ndarray
    potential_energy_J: np.ndarray
    work_thrust_J: np.ndarray
    work_ice_normal_J: np.ndarray
    work_ice_friction_J: np.ndarray
    work_damping_J: np.ndarray
    residual_trapezoid_J: np.ndarray
    residual_power_quadrature_J: np.ndarray
    residual_left_endpoint_J: np.ndarray
    stats_window: tuple = (0.0, 0.0)

    def window_slice(self) -> np.ndarray:
        start, stop = self.stats_window
        return (self.time_s >= start) & (self.time_s <= stop)


# --------------------------------------------------------------------------- #
# core integration
# --------------------------------------------------------------------------- #
def run_case(
    config: CaseConfig,
    *,
    dt_s: float = 0.005,
    duration_s: float = 60.0,
    transient_s: float = 10.0,
    diagnostic_work_scale: float = 1.0,
) -> CaseTimeSeries:
    """Integrate one case with velocity-Verlet and accumulate the energy budget.

    Initial condition: the ship at its commanded open-water speed with the bow
    at the residual penetration of the ice edge, so contact is active at
    ``t = 0``.

    ``diagnostic_work_scale`` multiplies the *accumulated* ice work only (never
    the dynamics) and exists so that tests can show the energy-budget metric
    actually detects a mis-accounted force term; reported physics always uses
    the default ``1.0``.
    """
    scale_work = _positive("diagnostic_work_scale", diagnostic_work_scale)
    dt = _positive("dt_s", dt_s)
    duration = _positive("duration_s", duration_s)
    contact = config.contact
    vessel = config.vessel
    law = config.law
    thickness = config.thickness_m
    speed_cmd = config.commanded_speed_mps

    steps = int(round(duration / dt))
    if steps < 10:
        raise ValueError("duration_s / dt_s must give at least 10 steps")

    mass = vessel.mass_matrix()
    mass_inv = np.linalg.inv(mass)
    stiff = vessel.stiffness()
    damping = vessel.damping()
    quad = np.zeros(6)
    quad[0] = vessel.surge_quadratic_damping
    quad[2] = vessel.heave_quadratic_damping
    normal = contact.normal_direction
    r_c = np.array([vessel.x_bow_m, 0.0, vessel.z_ice_line_m])

    q = np.zeros(6)
    q[0] = contact.residual_penetration_m
    v = np.zeros(6)
    v[0] = speed_cmd
    ice_edge = 0.0
    spall_events = 0

    def forces(qq, vv):
        """Generalised force split: thrust, ice normal, ice friction, damping, restoring."""
        penetration = qq[0] + qq[4] * vessel.z_ice_line_m - ice_edge
        if penetration > 0.0:
            area = contact.area_m2(penetration, thickness)
        else:
            area = 0.0
        n_force = law.force_N(area) if area > 0.0 else 0.0

        f_norm_vec = n_force * normal
        gen_norm = np.zeros(6)
        gen_norm[0:3] = f_norm_vec
        gen_norm[3] = r_c[1] * f_norm_vec[2] - r_c[2] * f_norm_vec[1]
        gen_norm[4] = r_c[2] * f_norm_vec[0] - r_c[0] * f_norm_vec[2]
        gen_norm[5] = r_c[0] * f_norm_vec[1] - r_c[1] * f_norm_vec[0]

        gen_fric = np.zeros(6)
        if n_force > 0.0 and contact.friction > 0.0:
            omega = np.array([vv[3], vv[4], vv[5]])
            v_point = vv[0:3] + np.cross(omega, r_c)
            v_tangent = v_point - float(v_point @ normal) * normal
            speed_t = float(np.linalg.norm(v_tangent))
            if speed_t > 1.0e-9:
                f_fric_vec = -contact.friction * n_force * v_tangent / speed_t
                gen_fric[0:3] = f_fric_vec
                gen_fric[3] = r_c[1] * f_fric_vec[2] - r_c[2] * f_fric_vec[1]
                gen_fric[4] = r_c[2] * f_fric_vec[0] - r_c[0] * f_fric_vec[2]
                gen_fric[5] = r_c[0] * f_fric_vec[1] - r_c[1] * f_fric_vec[0]

        thrust = vessel.open_water_resistance(speed_cmd) + vessel.thrust_gain * (speed_cmd - vv[0])
        thrust = max(0.0, thrust)
        f_thrust = np.zeros(6)
        f_thrust[0] = thrust
        f_damp = -(damping * vv) - quad * vv * np.abs(vv)
        f_restore = -stiff * qq
        helper = {
            "normal_force": n_force,
            "penetration": penetration,
            "area": area,
            "resistance": float(-gen_norm[0] - gen_fric[0]),
            "thrust": thrust,
        }
        return f_thrust, gen_norm, gen_fric, f_damp, f_restore, helper

    f_thrust, f_norm, f_fric, f_damp, f_restore, helper = forces(q, v)
    accel = mass_inv @ (f_thrust + f_norm + f_fric + f_damp + f_restore)

    n_samples = steps + 1
    time = np.linspace(0.0, steps * dt, n_samples)
    disp = np.zeros((6, n_samples))
    vel = np.zeros((6, n_samples))
    normal_force = np.zeros(n_samples)
    area_hist = np.zeros(n_samples)
    pressure_hist = np.zeros(n_samples)
    penetration_hist = np.zeros(n_samples)
    resistance_hist = np.zeros(n_samples)
    ke = np.zeros(n_samples)
    pe = np.zeros(n_samples)
    w_thrust = np.zeros(n_samples)
    w_norm = np.zeros(n_samples)
    w_fric = np.zeros(n_samples)
    w_damp = np.zeros(n_samples)
    resid_trap = np.zeros(n_samples)
    resid_power = np.zeros(n_samples)
    resid_left = np.zeros(n_samples)
    w_left = np.zeros(n_samples)
    w_power = np.zeros(n_samples)

    def record(i, qq, vv, h):
        disp[:, i] = qq
        vel[:, i] = vv
        normal_force[i] = h["normal_force"]
        area_hist[i] = h["area"]
        pressure_hist[i] = law.pressure_MPa(h["area"]) if h["area"] > 0.0 else 0.0
        penetration_hist[i] = h["penetration"]
        resistance_hist[i] = h["resistance"]
        ke[i] = 0.5 * float(vv @ (mass @ vv))
        pe[i] = 0.5 * float(np.sum(stiff * qq * qq))

    record(0, q, v, helper)

    for step in range(steps):
        v_half = v + 0.5 * dt * accel
        q_new = q + dt * v_half

        # crushing/spalling: the ice edge advances once the contact zone has
        # been crushed through, which resets the penetration to its residual.
        penetration_new = q_new[0] + q_new[4] * vessel.z_ice_line_m - ice_edge
        if penetration_new >= contact.spall_depth_m:
            ice_edge += penetration_new - contact.residual_penetration_m
            spall_events += 1

        f_thrust_n, f_norm_n, f_fric_n, f_damp_n, f_restore_n, helper_n = forces(q_new, v_half)
        accel_new = mass_inv @ (f_thrust_n + f_norm_n + f_fric_n + f_damp_n + f_restore_n)
        v_new = v_half + 0.5 * dt * accel_new

        # --- discrete work, three quadratures --------------------------- #
        # ice and other contributions are kept separate so that the diagnostic
        # work scale (self-test hook) perturbs the ice work in *all* of them.
        dq = q_new - q
        thrust_trap = 0.5 * (f_thrust[0] + f_thrust_n[0]) * dq[0]
        damp_trap = 0.5 * float((f_damp + f_damp_n) @ dq)
        ice_norm_trap = 0.5 * float((f_norm + f_norm_n) @ dq)
        ice_fric_trap = 0.5 * float((f_fric + f_fric_n) @ dq)
        w_thrust[step + 1] = w_thrust[step] + thrust_trap
        w_damp[step + 1] = w_damp[step] + damp_trap
        w_norm[step + 1] = w_norm[step] + scale_work * ice_norm_trap
        w_fric[step + 1] = w_fric[step] + scale_work * ice_fric_trap

        w_left[step + 1] = w_left[step] + (
            float((f_thrust + f_damp) @ dq) + scale_work * float((f_norm + f_fric) @ dq)
        )
        w_power[step + 1] = w_power[step] + 0.5 * dt * (
            float((f_thrust + f_damp) @ v)
            + float((f_thrust_n + f_damp_n) @ v_new)
            + scale_work * (float((f_norm + f_fric) @ v) + float((f_norm_n + f_fric_n) @ v_new))
        )

        record(step + 1, q_new, v_new, helper_n)

        resid_trap[step + 1] = (
            w_thrust[step + 1]
            + w_norm[step + 1]
            + w_fric[step + 1]
            + w_damp[step + 1]
            - (ke[step + 1] - ke[0])
            - (pe[step + 1] - pe[0])
        )
        energy_change = (ke[step + 1] - ke[0]) + (pe[step + 1] - pe[0])
        resid_left[step + 1] = w_left[step + 1] - energy_change
        resid_power[step + 1] = w_power[step + 1] - energy_change

        q, v, accel = q_new, v_new, accel_new
        f_thrust, f_norm, f_fric, f_damp, f_restore, helper = (
            f_thrust_n,
            f_norm_n,
            f_fric_n,
            f_damp_n,
            f_restore_n,
            helper_n,
        )

    return CaseTimeSeries(
        config=config,
        dt_s=dt,
        time_s=time,
        displacement=disp,
        velocity=vel,
        contact_force_N=normal_force,
        contact_area_m2=area_hist,
        contact_pressure_MPa=pressure_hist,
        penetration_m=penetration_hist,
        resistance_N=resistance_hist,
        spall_events=spall_events,
        kinetic_energy_J=ke,
        potential_energy_J=pe,
        work_thrust_J=w_thrust,
        work_ice_normal_J=w_norm,
        work_ice_friction_J=w_fric,
        work_damping_J=w_damp,
        residual_trapezoid_J=resid_trap,
        residual_power_quadrature_J=resid_power,
        residual_left_endpoint_J=resid_left,
        stats_window=(min(transient_s, duration), duration),
    )


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def count_load_cycles(values, *, prominence_fraction: float = 0.10, min_separation: int = 2):
    """Count load cycles by turning-point (local peak) detection.

    A peak counts when it is a strict local maximum over ``min_separation``
    samples on both sides and its prominence over the lower of the two
    adjacent troughs exceeds ``prominence_fraction * (max - min)`` of the
    series.  Deterministic: no filtering, no interpolation.
    """
    x = np.asarray(values, dtype=float)
    if x.size < 3:
        return 0, np.zeros(0, dtype=int)
    span = float(np.max(x) - np.min(x))
    if span <= 0.0:
        return 0, np.zeros(0, dtype=int)
    threshold = prominence_fraction * span
    peaks = []
    for i in range(min_separation, x.size - min_separation):
        if not (x[i] > x[i - min_separation : i]).all():
            continue
        if not (x[i] >= x[i + 1 : i + 1 + min_separation]).all():
            continue
        prominence = x[i] - max(float(np.min(x[:i])), float(np.min(x[i:])))
        if prominence >= threshold:
            peaks.append(i)
    return len(peaks), np.asarray(peaks, dtype=int)


def case_statistics(result: CaseTimeSeries, *, threshold_factor: float = 1.25) -> dict:
    """Peak / mean / RMS / threshold exceedances / cycles in the stats window."""
    mask = result.window_slice()
    load = result.contact_force_N[mask]
    if load.size == 0:
        raise ValueError("statistics window is empty")
    mean = float(np.mean(load))
    peak = float(np.max(load))
    threshold = threshold_factor * mean
    cycles, _ = count_load_cycles(load)
    speed = result.velocity[0][mask]
    pressure = result.contact_pressure_MPa[mask]
    positive_pressure = pressure[pressure > 0.0]
    return {
        "peak_load_MN": peak / 1.0e6,
        "mean_load_MN": mean / 1.0e6,
        "rms_load_MN": float(np.sqrt(np.mean(load**2))) / 1.0e6,
        "min_load_MN": float(np.min(load)) / 1.0e6,
        "std_load_MN": float(np.std(load)) / 1.0e6,
        "peak_to_mean": peak / mean if mean > 0.0 else math.inf,
        "threshold_MN": threshold / 1.0e6,
        "threshold_factor": threshold_factor,
        "exceedance_count": int(
            np.count_nonzero(
                np.diff(np.concatenate(([False], load > threshold, [False])).astype(int)) == 1
            )
        ),
        "exceedance_samples": int(np.count_nonzero(load > threshold)),
        "exceedance_definition": (
            "number of contiguous excursions above the threshold (one count per "
            "load spike, not per sample)"
        ),
        "cycle_count": int(cycles),
        "cycle_detection": {
            "method": "turning-point peaks",
            "prominence_fraction": 0.10,
            "min_separation_samples": 2,
        },
        "mean_speed_mps": float(np.mean(speed)),
        "min_speed_mps": float(np.min(speed)),
        "max_speed_mps": float(np.max(speed)),
        "speed_loss_fraction": float(1.0 - np.mean(speed) / result.config.commanded_speed_mps),
        "max_abs_heave_m": float(np.max(np.abs(result.displacement[2][mask]))),
        "max_abs_pitch_deg": float(np.max(np.abs(np.degrees(result.displacement[4][mask])))),
        "max_abs_roll_deg": float(np.max(np.abs(np.degrees(result.displacement[3][mask])))),
        "max_pressure_MPa": float(np.max(positive_pressure)) if positive_pressure.size else 0.0,
        "min_pressure_MPa": float(np.min(positive_pressure)) if positive_pressure.size else 0.0,
        "mean_contact_area_m2": float(np.mean(result.contact_area_m2[mask])),
        "spall_events": int(result.spall_events),
        "statistics_window_s": list(result.stats_window),
        "samples": int(load.size),
    }


def _energy_window(result: CaseTimeSeries, i_a: int, i_b: int) -> dict:
    w_thrust = float(result.work_thrust_J[i_b] - result.work_thrust_J[i_a])
    w_norm = float(result.work_ice_normal_J[i_b] - result.work_ice_normal_J[i_a])
    w_fric = float(result.work_ice_friction_J[i_b] - result.work_ice_friction_J[i_a])
    w_damp = float(result.work_damping_J[i_b] - result.work_damping_J[i_a])
    d_ke = float(result.kinetic_energy_J[i_b] - result.kinetic_energy_J[i_a])
    d_pe = float(result.potential_energy_J[i_b] - result.potential_energy_J[i_a])
    residual = w_thrust + w_norm + w_fric + w_damp - d_ke - d_pe
    residual_left = float(
        result.residual_left_endpoint_J[i_b] - result.residual_left_endpoint_J[i_a]
    )
    residual_power = float(
        result.residual_power_quadrature_J[i_b] - result.residual_power_quadrature_J[i_a]
    )
    scale = abs(w_thrust) + abs(w_norm) + abs(w_fric) + 1.0
    return {
        "window_s": [float(result.time_s[i_a]), float(result.time_s[i_b])],
        "work_thrust_J": w_thrust,
        "work_ice_normal_J": w_norm,
        "work_ice_friction_J": w_fric,
        "work_damping_J": w_damp,
        "kinetic_energy_change_J": d_ke,
        "potential_energy_change_J": d_pe,
        "residual_J": residual,
        "residual_fraction_of_nonconservative_work": residual / scale,
        "residual_fraction_of_thrust_work": residual / (abs(w_thrust) + 1.0),
        "residual_fraction_of_ice_work": residual / (abs(w_norm) + abs(w_fric) + 1.0),
        "power_quadrature_residual_J": residual_power,
        "power_quadrature_residual_fraction": residual_power / scale,
        "displacement_quadrature_residual_J": residual,
        "left_endpoint_residual_J": residual_left,
        "left_endpoint_residual_fraction": residual_left / scale,
        "damping_to_thrust_work_ratio": -w_damp / (abs(w_thrust) + 1.0),
    }


def energy_balance(result: CaseTimeSeries) -> dict:
    """Energy budget over the whole run and over the statistics window."""
    idx = np.flatnonzero(result.window_slice())
    i0, i1 = int(idx[0]), int(idx[-1])
    return {
        "case": result.config.name,
        "dt_s": result.dt_s,
        "whole_run": _energy_window(result, 0, result.time_s.size - 1),
        "statistics_window": _energy_window(result, i0, i1),
        "initial_kinetic_energy_J": float(result.kinetic_energy_J[0]),
        "peak_kinetic_energy_J": float(np.max(result.kinetic_energy_J)),
        "peak_potential_energy_J": float(np.max(result.potential_energy_J)),
        "spall_events": int(result.spall_events),
    }


def windowed_energy_residual(result: CaseTimeSeries, *, windows: int = 8, window_s: float = 5.0):
    """Residual of the energy budget in consecutive windows.

    A single end-to-end residual can hide compensating errors; the per-window
    sequence shows whether the budget also holds *locally*.
    """
    duration = float(result.time_s[-1])
    out = []
    t = float(result.stats_window[0])
    while t + window_s <= duration + 1.0e-9 and len(out) < windows:
        i_a = int(round(t / result.dt_s))
        i_b = min(i_a + int(round(window_s / result.dt_s)), result.time_s.size - 1)
        block = _energy_window(result, i_a, i_b)
        out.append(
            {
                "window_s": block["window_s"],
                "residual_J": block["residual_J"],
                "residual_fraction": block["residual_fraction_of_nonconservative_work"],
                "thrust_work_J": block["work_thrust_J"],
                "ice_work_J": block["work_ice_normal_J"] + block["work_ice_friction_J"],
                "damping_work_J": block["work_damping_J"],
            }
        )
        t += window_s
    return out


def _rankdata(values) -> np.ndarray:
    """Average ranks; ties share the mean rank."""
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman_rank(x, y, *, exact_max_n: int = 8) -> dict:
    """Spearman rho with an *exact* permutation p-value for small samples.

    For ``n <= exact_max_n`` the null distribution is enumerated over all
    ``n!`` relabellings, so the two-sided p-value carries no asymptotic
    approximation (which would be indefensible at n = 3 or n = 6).
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.size != b.size or a.size < 3:
        raise ValueError("spearman_rank needs two equal-length vectors of size >= 3")
    ra = _rankdata(a)
    rb = _rankdata(b)
    ra_c = ra - ra.mean()
    rb_c = rb - rb.mean()
    denom = math.sqrt(float(ra_c @ ra_c) * float(rb_c @ rb_c))
    rho = float(ra_c @ rb_c) / denom if denom > 0.0 else float("nan")
    n = a.size
    if n <= exact_max_n:
        hits = 0
        total = 0
        norm_b = math.sqrt(float(rb_c @ rb_c))
        for perm in itertools.permutations(range(n)):
            rp = rb_c[list(perm)]
            denom_p = math.sqrt(float(ra_c @ ra_c)) * norm_b
            rho_p = float(ra_c @ rp) / denom_p if denom_p > 0.0 else float("nan")
            total += 1
            if abs(rho_p) >= abs(rho) - 1.0e-12:
                hits += 1
        return {
            "n": int(n),
            "rho": rho,
            "p_value": hits / total,
            "method": f"exact permutation over all {total} relabellings",
            "x": [float(v) for v in a],
            "y": [float(v) for v in b],
        }
    # Large-n fallback: asymptotic t approximation (not used by this task).
    if abs(rho) >= 1.0:
        p_value = 0.0
    else:
        t = rho * math.sqrt((n - 2) / (1.0 - rho * rho))
        p_value = math.erfc(abs(t) / math.sqrt(2.0))
    return {
        "n": int(n),
        "rho": rho,
        "p_value": float(p_value),
        "method": "asymptotic normal approximation (fallback, n > 8)",
        "x": [float(v) for v in a],
        "y": [float(v) for v in b],
    }


def damping_dissipation_crosscheck(result: CaseTimeSeries, vessel: Vessel6DOF) -> dict:
    """Independent evaluation of the damping dissipation.

    The simulation accumulates the damping work as ``F_damping . dq``.  This
    routine integrates the *dissipative power law* over the recorded velocity
    history instead, ``P = c_i v_i^2 + c2_i |v_i|^3``, so the two numbers come
    from different code paths and a mis-implemented damping force cannot hide.
    """
    c = vessel.damping()
    quad = np.zeros(6)
    quad[0] = vessel.surge_quadratic_damping
    quad[2] = vessel.heave_quadratic_damping
    v = result.velocity
    power = np.sum(c[:, None] * v * v + quad[:, None] * v * v * np.abs(v), axis=0)
    mask = result.window_slice()
    t = result.time_s[mask]
    integrated = float(np.trapezoid(power[mask], t))
    idx = np.flatnonzero(mask)
    i0, i1 = int(idx[0]), int(idx[-1])
    accumulated = -float(result.work_damping_J[i1] - result.work_damping_J[i0])
    return {
        "window_s": [float(t[0]), float(t[-1])],
        "power_law_integral_J": integrated,
        "accumulated_force_displacement_J": accumulated,
        "relative_difference": (
            (accumulated - integrated) / integrated if integrated != 0.0 else None
        ),
    }


# --------------------------------------------------------------------------- #
# anchor handling
# --------------------------------------------------------------------------- #
def load_anchor_parameters(path=None) -> dict:
    """Load the pressure--area coefficients from an anchor transcription.

    The anchor values may only come from a transcribed anchor document, so the
    loader never invents a number: if ``path`` is ``None`` or missing, the
    declared reference normalisation is returned with
    ``"status": "pending_transcription"`` so that downstream artefacts record
    the gap instead of silently passing an unanchored value off as normative.

    Expected anchor JSON (all keys optional)::

        {"anchor_id": "AN-ISO19906",
         "C_R_MPa": ..., "A_0_m2": ..., "exponent": ...,
         "source": "...", "transcribed_by": "...", "transcribed_at": "..."}
    """
    default = {
        "anchor_id": "AN-ISO19906",
        "status": "pending_transcription",
        "C_R_MPa": ISO19906PressureArea().reference_pressure_MPa,
        "A_0_m2": ISO19906PressureArea().reference_area_m2,
        "exponent": ISO19906PressureArea().exponent,
        "source": "not available in this work tree",
        "note": (
            "declared reference normalisation, not a normative transcription; "
            "loads scale exactly linearly in C_R"
        ),
    }
    if path is None:
        return default
    candidate = Path(path)
    if not candidate.is_file():
        default["note"] = f"anchor file {candidate} not found; {default['note']}"
        return default
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    for key in ("C_R_MPa", "A_0_m2", "exponent"):
        if key not in payload:
            raise ValueError(f"anchor file {candidate} is missing {key!r}")
    payload.setdefault("anchor_id", "AN-ISO19906")
    payload["status"] = "transcribed"
    return payload


def law_from_anchor(anchor: dict) -> ISO19906PressureArea:
    """Build the pressure--area law from an anchor payload."""
    status = anchor.get("status", "pending_transcription")
    return ISO19906PressureArea(
        reference_pressure_MPa=float(anchor["C_R_MPa"]),
        reference_area_m2=float(anchor["A_0_m2"]),
        exponent=float(anchor["exponent"]),
        provenance=(
            f"{anchor.get('anchor_id', 'AN-ISO19906')} [{status}]: "
            f"{anchor.get('source', 'no source recorded')}"
        ),
    )
