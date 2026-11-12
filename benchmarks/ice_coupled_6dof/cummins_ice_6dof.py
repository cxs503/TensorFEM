"""6-DOF Cummins (exponential-kernel) model + MMG manoeuvring + random ice field.

This module is the numerical core of the W23 ``ice-coupled-6dof`` study.  It is
deliberately small and auditable: everything it claims is checkable from the
state it carries.

What is modelled
----------------
*Radiation/seakeeping* -- the Cummins (1962) formulation of the linear
radiation problem,

    (M + A_inf) qdd + int_0^t K(t - tau) qd(tau) dtau + B_v qd + C q = F,

with the retardation kernel approximated by a sum of exponentially damped
cosines (``K(t) = sum_j g_j exp(-a_j t) cos(b_j t)``), i.e. a finite-order
state-space realisation (each term is the impulse response of the 2x2 system
``xd = -a x - b y + qd``, ``yd = b x - a y``).  For that kernel the added mass
and damping follow in closed form

    A(w) = A_inf - sum_j g_j/(2w) [ (w+b_j)/(a_j^2+(w+b_j)^2)
                                  + (w-b_j)/(a_j^2+(w-b_j)^2) ]
    B(w) =        sum_j g_j a_j/2 [ 1/(a_j^2+(b_j-w)^2)
                                  + 1/(a_j^2+(b_j+w)^2) ]

so the frequency-domain and time-domain descriptions are *the same model*, and
the zero-ice time-domain response must reproduce the frequency-domain RAO.  That
consistency is the subject of criterion 1 of the W23 node; it is a test of the
integrator and of the kernel realisation, not a validation of the hydrodynamics
(see "Honesty notes").

*Manoeuvring* -- MMG-style surge/sway/yaw: nonlinear resistance, a thrust
governor holding the commanded speed, linear sway/yaw hydrodynamic derivatives.

*Ice* -- a random ice field specified by concentration, floe-diameter and ice
thickness distributions.  Floe encounters form a Poisson process with rate

    lambda = n * (B + E[D]) * V,     n = C / (pi/4 E[D^2]),

i.e. the swept-area argument.  At an encounter a rigid floe (point mass, one
translational DOF along the contact normal) is brought into contact with the bow
through the ISO 19906 global pressure--area law ``p_G = C_R (A/A_0)^ex`` used as
the contact pressure source, ``N = p_G(A) A``.  The contact is a plastic
crushing contact: while loading, ``N`` follows the area law; while unloading the
already-crushed ice carries only ``rebound_factor`` of it.  The force pair on
hull and floe is equal and opposite, so the impact momentum ledger below closes
identically up to quadrature.

*Momentum ledger* -- for every impact the module integrates, per channel, the
hull momentum ``m qd``, the floe momentum, the fluid momentum carried by the
radiation model (``A_inf qd + int ret dt``) and the external impulses; the
per-impact residual is reported (criterion 2).

Honesty notes / scope
---------------------
* The hydrodynamic coefficients are a *declared synthetic* radiation model: the
  gains are chosen so that ``B(w) >= 0`` (passive) on the working band and so
  that ``B(w) -> 0`` at both ends, but the resulting ``A(w)`` *increases* with
  frequency, which is the opposite of the measured trend for ship heave/pitch
  added mass (real retardation kernels have negative lobes, which a
  positive-gain exponential sum cannot represent).  Consequently the RAO used
  here is the RAO of this declared model, not a validated seakeeping RAO.
* The wave excitation is Froude--Kirylov with Smith correction and no
  diffraction; it is used identically in the time and frequency domains, so the
  consistency test is unaffected.
* The ISO 19906 coefficients are inputs (``anchor_status`` records where they
  came from); no normative table value is claimed unless an anchor
  transcription is supplied.
* Level, homogeneous ice; point-mass floes; constant added mass; small
  rotations.  Not a CFD/DEM ice solver.

All quantities SI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Iterable

import numpy as np

GRAVITY = 9.80665

DOFS = ("surge", "sway", "heave", "roll", "pitch", "yaw")


# --------------------------------------------------------------------------- #
# pressure-area law
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PressureAreaLaw:
    """ISO 19906 global pressure--area law ``p_G = C_R (A/A_0)^ex``, ``ex < 0``."""

    reference_pressure_MPa: float = 1.0
    reference_area_m2: float = 1.0
    exponent: float = -0.28
    provenance: str = (
        "declared reference normalisation (C_R = 1 MPa at A_0 = 1 m2, ex = -0.28); "
        "NOT a transcription of a normative ISO 19906 table -- see anchor_status "
        "in the artefacts. Contact force scales exactly linearly in C_R, so every "
        "criterion below is invariant to C_R; the ex sensitivity is reported."
    )

    def pressure_MPa(self, area_m2):
        area = np.asarray(area_m2, dtype=float)
        return self.reference_pressure_MPa * np.power(area / self.reference_area_m2,
                                                      self.exponent)

    def force_N(self, area_m2):
        """Contact force ``p_G(A) * A``."""
        area = np.asarray(area_m2, dtype=float)
        return self.pressure_MPa(area) * 1.0e6 * area


# --------------------------------------------------------------------------- #
# radiation model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KernelTerm:
    gain: float
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if self.alpha <= 0.0:
            raise ValueError("kernel alpha must be positive (causal, decaying)")
        if self.beta < 0.0:
            raise ValueError("kernel beta must be non-negative")


@dataclass(frozen=True)
class RadiationDOF:
    added_mass_inf: float
    terms: tuple

    def added_mass(self, omega):
        w = np.asarray(omega, dtype=float)
        out = np.full(w.shape, self.added_mass_inf, dtype=float)
        for t in self.terms:
            out -= t.gain / (2.0 * w) * (
                (w + t.beta) / (t.alpha**2 + (w + t.beta) ** 2)
                + (w - t.beta) / (t.alpha**2 + (w - t.beta) ** 2)
            )
        return out

    def damping(self, omega):
        w = np.asarray(omega, dtype=float)
        out = np.zeros(w.shape, dtype=float)
        for t in self.terms:
            out += t.gain * t.alpha / 2.0 * (
                1.0 / (t.alpha**2 + (t.beta - w) ** 2)
                + 1.0 / (t.alpha**2 + (t.beta + w) ** 2)
            )
        return out

    def kernel(self, tau):
        tau = np.asarray(tau, dtype=float)
        out = np.zeros(tau.shape, dtype=float)
        for t in self.terms:
            out += t.gain * np.exp(-t.alpha * tau) * np.cos(t.beta * tau)
        return out

    def anomaly(self) -> float:
        """``A(0) - A_inf`` (negative for a positive-gain kernel)."""
        return -sum(t.gain / (t.alpha**2 + t.beta**2) for t in self.terms)


def _synthetic_terms(inf_mass: float, low_freq_mass_ratio: float) -> tuple:
    """Declared synthetic kernel with ``A(0)/inf_mass = low_freq_mass_ratio``.

    ``A(0) = A_inf - sum_j g_j (a_j^2 - b_j^2)/(a_j^2+b_j^2)^2``, so the gains are
    scaled to hit the requested low-frequency added mass.  All gains are positive,
    which keeps ``B(w) >= 0`` (passive) on the working band, at the price of
    ``A(w)`` growing towards ``A_inf`` (declared limitation, see module notes).
    """
    basis = ((0.85, 0.30), (1.80, 0.70), (0.22, 0.10))
    weights = (0.60, 0.30, 0.10)
    need = (1.0 - low_freq_mass_ratio) * inf_mass
    terms = []
    shape = []
    for (alpha, beta), wt in zip(basis, weights):
        d = alpha**2 + beta**2
        shape.append(wt * (alpha**2 - beta**2) / (d * d))
    scale = need / sum(shape)
    for (alpha, beta), wt in zip(basis, weights):
        terms.append(KernelTerm(gain=scale * wt, alpha=alpha, beta=beta))
    return tuple(terms)


# --------------------------------------------------------------------------- #
# ship model
# --------------------------------------------------------------------------- #
@dataclass
class ShipModel:
    length_m: float = 100.0
    breadth_m: float = 20.0
    draught_m: float = 6.0
    block_coefficient: float = 0.70
    water_density: float = 1025.0
    kxx_m: float = 7.0
    kyy_m: float = 25.0
    kzz_m: float = 26.0
    waterplane_area_ratio: float = 0.85
    gmt_m: float = 1.2
    gml_m: float = 100.0
    # declared damping (modal ratios applied on top of the radiation damping)
    zeta_heave: float = 0.03
    zeta_roll: float = 0.08
    zeta_pitch: float = 0.02
    roll_quadratic_damping: float = 0.0
    # MMG
    resistance_linear: float = 3.0e4
    resistance_quadratic: float = 4.5e4
    thrust_gain: float = 5.0e5
    yv: float = -3.0e5
    yr: float = -2.0e6
    nv: float = -1.0e6
    nr: float = -8.0e7
    # ice contact geometry
    half_angle_deg: float = 40.0
    normal_obliquity_deg: float = 15.0
    rebound_factor: float = 0.05
    provenance: str = (
        "hull/main particulars and damping ratios are declared engineering "
        "inputs of this study; the radiation kernel is a synthetic passive model "
        "(see module docstring); MMG derivatives are declared estimates, not "
        "captive-model data"
    )

    # -- derived quantities ---------------------------------------------------
    def __post_init__(self) -> None:
        self.mass_kg = (self.water_density * self.block_coefficient
                        * self.length_m * self.breadth_m * self.draught_m)
        self.displacement_m3 = (self.block_coefficient * self.length_m
                                * self.breadth_m * self.draught_m)
        m = self.mass_kg
        self.mass_rigid = np.array([m, m, m, m * self.kxx_m**2,
                                    m * self.kyy_m**2, m * self.kzz_m**2])
        self.waterplane_area_m2 = self.waterplane_area_ratio * self.length_m * self.breadth_m
        c33 = self.water_density * GRAVITY * self.waterplane_area_m2
        c44 = self.water_density * GRAVITY * self.displacement_m3 * self.gmt_m
        c55 = self.water_density * GRAVITY * self.displacement_m3 * self.gml_m
        self.restoring = np.array([0.0, 0.0, c33, c44, c55, 0.0])
        self.radiation = {
            "surge": RadiationDOF(0.10 * m, _synthetic_terms(0.10 * m, 0.50)),
            "sway": RadiationDOF(0.60 * m, _synthetic_terms(0.60 * m, 0.50)),
            "heave": RadiationDOF(1.00 * m, _synthetic_terms(1.00 * m, 0.30)),
            "roll": RadiationDOF(0.20 * self.mass_rigid[3],
                                 _synthetic_terms(0.20 * self.mass_rigid[3], 0.50)),
            "pitch": RadiationDOF(1.00 * self.mass_rigid[4],
                                  _synthetic_terms(1.00 * self.mass_rigid[4], 0.40)),
            "yaw": RadiationDOF(1.00 * self.mass_rigid[5],
                                _synthetic_terms(1.00 * self.mass_rigid[5], 0.50)),
        }
        self.a_inf = np.array([self.radiation[d].added_mass_inf for d in DOFS])
        self.generalized_mass = self.mass_rigid + self.a_inf
        wn3 = math.sqrt(c33 / self.generalized_mass[2])
        wn4 = math.sqrt(c44 / self.generalized_mass[3])
        wn5 = math.sqrt(c55 / self.generalized_mass[4])
        self.viscous = np.array([
            0.0, 0.0,
            2.0 * self.zeta_heave * wn3 * self.generalized_mass[2],
            2.0 * self.zeta_roll * wn4 * self.generalized_mass[3],
            2.0 * self.zeta_pitch * wn5 * self.generalized_mass[4],
            0.0,
        ])
        self.natural_frequency = {"heave": wn3, "roll": wn4, "pitch": wn5}

    # -- ice contact helpers --------------------------------------------------
    def contact_area_m2(self, penetration_m, thickness_m):
        pen = np.maximum(np.asarray(penetration_m, dtype=float), 0.0)
        width = np.minimum(self.breadth_m,
                           2.0 * pen * math.tan(math.radians(self.half_angle_deg)))
        height = np.minimum(thickness_m, thickness_m)
        return width * height

    def normal_unit(self, azimuth_rad):
        """Unit vector from the hull towards the ice (forward/up-slope)."""
        g = math.radians(self.normal_obliquity_deg)
        return np.array([math.cos(g) * math.cos(azimuth_rad),
                         math.cos(g) * math.sin(azimuth_rad),
                         -math.sin(g)])

    # -- frequency domain -----------------------------------------------------
    def added_mass_vector(self, omega):
        return np.array([self.radiation[d].added_mass(omega) for d in DOFS])

    def damping_vector(self, omega):
        return np.array([self.radiation[d].damping(omega) for d in DOFS])

    def excitation(self, omega, wave_amplitude_m: float = 1.0) -> np.ndarray:
        """Froude--Krylov + Smith excitation (no diffraction), unit wave amplitude."""
        w = float(omega)
        k = w * w / GRAVITY
        smith = math.exp(-k * self.draught_m)
        aw = self.waterplane_area_m2
        iwl = aw * self.length_m**2 / 12.0
        f3 = self.water_density * GRAVITY * aw * wave_amplitude_m * smith
        m5 = self.water_density * GRAVITY * iwl * k * wave_amplitude_m * smith
        f1 = self.water_density * GRAVITY * self.displacement_m3 * k * wave_amplitude_m * smith
        out = np.zeros(6)
        out[0] = f1
        out[2] = f3
        out[4] = m5
        return out

    def rao_frequency_domain(self, omega, wave_amplitude_m: float = 1.0) -> dict:
        """Complex response amplitudes per unit wave amplitude (diagonal model)."""
        w = float(omega)
        force = self.excitation(w, wave_amplitude_m)
        a = self.added_mass_vector(w)
        b = self.damping_vector(w)
        amps = np.zeros(6, dtype=complex)
        for i in range(6):
            if abs(force[i]) == 0.0 and self.restoring[i] == 0.0:
                continue
            denom = (-w * w * (self.mass_rigid[i] + a[i])
                     + 1j * w * (b[i] + self.viscous[i]) + self.restoring[i])
            if denom == 0.0:
                continue
            amps[i] = force[i] / denom
        return {
            "omega_rad_s": w,
            "wave_amplitude_m": wave_amplitude_m,
            "force_amplitude": force.tolist(),
            "added_mass": a.tolist(),
            "damping": b.tolist(),
            "amplitude": amps.real.tolist(),
            "amplitude_imag": amps.imag.tolist(),
            "amplitude_abs": np.abs(amps).tolist(),
        }


# --------------------------------------------------------------------------- #
# ice field / encounter process
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IceField:
    concentration: float
    median_floe_diameter_m: float = 3.0
    lognormal_sigma: float = 0.35
    mean_thickness_m: float = 0.6
    thickness_cv: float = 0.25
    ice_density: float = 900.0
    azimuth_spread_deg: float = 20.0
    floe_velocity_spread: float = 0.30

    def __post_init__(self) -> None:
        if not (0.0 <= self.concentration <= 1.0):
            raise ValueError("concentration must be in [0, 1]")

    @property
    def mean_floe_area_m2(self) -> float:
        d2 = self.median_floe_diameter_m**2 * math.exp(2.0 * self.lognormal_sigma**2)
        return math.pi / 4.0 * d2

    @property
    def mean_floe_diameter(self) -> float:
        return self.median_floe_diameter_m * math.exp(0.5 * self.lognormal_sigma**2)

    def number_density(self) -> float:
        """Floes per square metre implied by the concentration."""
        return self.concentration / self.mean_floe_area_m2

    def encounter_rate(self, ship_speed_mps: float, hull_breadth_m: float) -> float:
        """Poisson rate ``n (B + E[D]) V`` (swept area per unit time)."""
        return (self.number_density()
                * (hull_breadth_m + self.mean_floe_diameter)
                * ship_speed_mps)

    def sample_arrivals(self, duration_s: float, rate_hz: float, rng) -> np.ndarray:
        if rate_hz <= 0.0:
            return np.zeros(0)
        count = rng.poisson(rate_hz * duration_s)
        return np.sort(rng.uniform(0.0, duration_s, count))

    def sample_floes(self, n: int, rng, hull_breadth_m: float) -> list:
        if n == 0:
            return []
        d = rng.lognormal(math.log(self.median_floe_diameter_m), self.lognormal_sigma, n)
        h = np.maximum(0.05, rng.normal(self.mean_thickness_m,
                                        self.thickness_cv * self.mean_thickness_m, n))
        az = np.radians(rng.uniform(-self.azimuth_spread_deg, self.azimuth_spread_deg, n))
        arm = rng.uniform(0.75, 1.0, n) * 0.5 * 0.0  # placeholder, set below
        return [dict(diameter=float(d[i]), thickness=float(h[i]),
                     azimuth=float(az[i]), arm=float(arm[i])) for i in range(n)]


# --------------------------------------------------------------------------- #
# integrator
# --------------------------------------------------------------------------- #
class Cummins6DOF:
    """Velocity-Verlet integrator with exact exponential-kernel filter updates.

    The contact between hull and floes is an equal-and-opposite force pair; a
    *cluster* is a maximal time interval during which at least one floe is in
    contact.  For each cluster the momentum ledger is closed against every
    force channel (hull, fluid carried by the radiation model, floes) so that a
    per-impact momentum residual can be reported without hiding overlapping
    contacts.
    """

    def __init__(self, model: ShipModel):
        self.model = model
        n = len(DOFS)
        terms = []
        for i, d in enumerate(DOFS):
            for t in model.radiation[d].terms:
                terms.append((i, t.gain, t.alpha, t.beta))
        self.n_terms = len(terms)
        self.term_alpha = np.array([t[2] for t in terms], dtype=float)
        self.term_beta = np.array([t[3] for t in terms], dtype=float)
        self.gain_mat = np.zeros((self.n_terms, n))
        for j, t in enumerate(terms):
            self.gain_mat[j, t[0]] += t[1]
        self.commanded_speed = None

    # -- kernel state ---------------------------------------------------------
    def filter_step(self, X, Y, velocity, dt):
        a = self.term_alpha[:, None]
        b = self.term_beta[:, None]
        d = a * a + b * b
        xs = velocity[None, :] * a / d
        ys = velocity[None, :] * b / d
        e = np.exp(-a * dt)
        c = np.cos(b * dt)
        s_ = np.sin(b * dt)
        dx = X - xs
        dy = Y - ys
        return xs + e * (c * dx - s_ * dy), ys + e * (s_ * dx + c * dy)

    def retardation_force(self, X):
        """``-sum_j g_j x_j`` -- the radiation force term of the momentum balance."""
        return -(self.gain_mat * X).sum(axis=0)

    # -- forces ---------------------------------------------------------------
    def hydrodynamic_forces(self, q, v, ret, t, excitation=None):
        """Viscous + restoring + radiation + MMG forces (retardation passed in)."""
        m = self.model
        f = np.array(ret, dtype=float)
        f -= m.viscous * v
        f -= m.restoring * q
        u = v[0]
        f[0] -= m.resistance_linear * u + m.resistance_quadratic * u * abs(u)
        if self.commanded_speed is not None:
            u_cmd = self.commanded_speed
            base = m.resistance_linear * u_cmd + m.resistance_quadratic * u_cmd * abs(u_cmd)
            f[0] += max(0.0, base + m.thrust_gain * (u_cmd - u))
        f[1] += m.yv * v[1] + m.yr * v[5]
        f[5] += m.nv * v[1] + m.nr * v[5]
        if excitation is not None:
            f = f + excitation(t)
        return f

    # -- main driver ----------------------------------------------------------
    def run(self, duration_s, dt_s, *, excitation=None, ice=None, seed=0,
            commanded_speed=None, mutation=None, initial_velocity=None,
            peak_window_s=1.5, contact_law=None):
        """Integrate the coupled system.

        Scheme: velocity-Verlet with the force evaluated at the integer-step
        states (predictor velocity for the velocity-dependent terms).  The
        velocity update is *exactly* the trapezoid rule on the recorded force
        history, which is what makes the momentum ledger below close.

        ``contact_law`` is the pressure source of the ice contact.  It defaults
        to :class:`PressureAreaLaw` (the ISO 19906 global area law), which is
        what every W22/W23 result used.  An alternative object may be injected
        as long as it either exposes ``force_N(area_m2)`` or -- when the local
        contact pressure must also depend on the penetration history, e.g. on
        the indentation strain rate -- ``contact_force(area_m2, penetration_m,
        thickness_m, event)``.  See :mod:`tensorfem.ice_pressure_area` for the
        recording laws used by the W24 pressure--area study; the default path is
        bit-identical to a call without the argument.
        """
        model = self.model
        self.commanded_speed = commanded_speed
        n = len(DOFS)
        nsteps = int(round(duration_s / dt_s))
        q = np.zeros(n)
        v = np.zeros(n)
        if initial_velocity is not None:
            v[0] = float(initial_velocity)
        X = np.zeros((self.n_terms, n))
        Y = np.zeros((self.n_terms, n))
        rng = np.random.default_rng(seed)
        law = PressureAreaLaw() if contact_law is None else contact_law

        floes = []
        rate = 0.0
        expected = 0.0
        scheduled = 0
        if ice is not None and ice.concentration > 0.0:
            speed_ref = commanded_speed if commanded_speed else 5.0
            rate = ice.encounter_rate(speed_ref, model.breadth_m)
            arrivals = ice.sample_arrivals(duration_s, rate, rng)
            scheduled = int(arrivals.size)
            expected = rate * duration_s
            for t_arr in arrivals:
                d = float(rng.lognormal(math.log(ice.median_floe_diameter_m),
                                        ice.lognormal_sigma))
                h = float(max(0.05, rng.normal(ice.mean_thickness_m,
                                               ice.thickness_cv * ice.mean_thickness_m)))
                az = float(np.radians(rng.uniform(-ice.azimuth_spread_deg,
                                                  ice.azimuth_spread_deg)))
                drift = float(rng.uniform(-ice.floe_velocity_spread,
                                          ice.floe_velocity_spread))
                floes.append(dict(
                    t_arrival=float(t_arr), diameter=d, thickness=h, azimuth=az,
                    arm_x=float(rng.uniform(0.30, 0.50) * model.length_m),
                    arm_y=float(rng.uniform(-1.0, 1.0) * 0.10 * model.breadth_m),
                    drift=drift,
                    mass=ice.ice_density * math.pi / 4.0 * d * d * h,
                    n_out=model.normal_unit(az),
                    pen=0.0, u=0.0, started=False, active=False, done=False,
                    unloading=False, peak=np.zeros(2), peak_until=0.0,
                    peak_done=False, J_contact=0.0, mag_start=0.0,
                ))

        rec_t = np.empty(nsteps + 1)
        rec_q = np.empty((nsteps + 1, 3))
        rec_v = np.empty((nsteps + 1, 3))
        rec_fice = np.empty(nsteps + 1)
        events = []
        cluster = None

        def point_velocity(vec, arm_x, arm_y):
            return np.array([vec[0] - vec[5] * arm_y,
                             vec[1] + vec[5] * arm_x,
                             vec[2] + vec[4] * arm_x - vec[3] * arm_y])

        def contact_magnitude(ev):
            pen = ev["pen"]
            if pen <= 0.0:
                return 0.0
            area = float(model.contact_area_m2(pen, ev["thickness"]))
            if area <= 0.0:
                return 0.0
            if hasattr(law, "contact_force"):
                mag = float(law.contact_force(area, pen, ev["thickness"], ev))
            else:
                mag = float(law.force_N(area))
            if ev.get("unloading", False):
                mag *= model.rebound_factor
            return mag

        def assemble(mags, members):
            out = np.zeros(6)
            for ev in members:
                mag = mags[id(ev)]
                if mag <= 0.0:
                    continue
                fvec = -mag * ev["n_out"]
                out[0] += fvec[0]
                out[1] += fvec[1]
                out[2] += fvec[2]
                out[3] += -ev["arm_y"] * fvec[2]
                out[4] += ev["arm_x"] * fvec[2]
                out[5] += ev["arm_x"] * fvec[1] - ev["arm_y"] * fvec[0]
            return out

        def cluster_open(t):
            return dict(t_start=t, v3=v[2], v5=v[4], ext3=[], ext5=[], times=[],
                        ret3=[], ret5=[], ice3=[], ice5=[], members=[],
                        magsum=[],
                        p_floe_start=None)

        ret = self.retardation_force(X)

        for step in range(nsteps + 1):
            t = step * dt_s
            # ---- activate scheduled encounters -----------------------------
            for ev in floes:
                if ev["started"] or ev["done"]:
                    continue
                if t >= ev["t_arrival"] - 1e-12:
                    ev["started"] = True
                    ev["active"] = True
                    ev["t_start"] = t
                    ev["u"] = -ev["drift"]
                    pv = point_velocity(v, ev["arm_x"], ev["arm_y"])
                    ev["v_rel_in"] = float(np.dot(pv, ev["n_out"])) - ev["u"]
                    ev["J_contact"] = 0.0
                    ev["mag_prev"] = 0.0
                    if cluster is None:
                        cluster = cluster_open(t)
                        cluster["p_floe_start"] = []
                    cluster["members"].append(ev)
                    cluster["p_floe_start"].append(
                        (ev["mass"] * ev["u"] * ev["n_out"][2],
                         ev["arm_x"] * ev["mass"] * ev["u"] * ev["n_out"][2]))

            # ---- forces at the start of the step ---------------------------
            mags = {id(ev): contact_magnitude(ev) for ev in floes if ev["active"]}
            for ev in floes:
                if ev["active"]:
                    ev["mag_start"] = mags[id(ev)]
            f_ice = assemble(mags, [ev for ev in floes if ev["active"]])
            f_hydro = self.hydrodynamic_forces(q, v, ret, t, excitation)
            a_now = (f_hydro + f_ice) / model.generalized_mass

            rec_t[step] = t
            rec_q[step] = q[[0, 2, 4]]
            rec_v[step] = v[[0, 2, 4]]
            rec_fice[step] = f_ice[2]

            if cluster is not None:
                cluster["times"].append(t)
                # external (non-radiation) impulse channels
                cluster["ext3"].append(float(f_hydro[2] - ret[2]))
                cluster["ext5"].append(float(f_hydro[4] - ret[4]))
                cluster["ice3"].append(float(f_ice[2]))
                cluster["ice5"].append(float(f_ice[4]))
                cluster["ret3"].append(float(ret[2]))
                cluster["ret5"].append(float(ret[4]))
                cluster["magsum"].append(float(sum(mags.values())))

            if step == nsteps:
                break

            # ---- drift -----------------------------------------------------
            v_mid = v + 0.5 * dt_s * a_now
            X, Y = self.filter_step(X, Y, v_mid, dt_s)
            q_new = q + dt_s * v_mid
            v_pred = v + dt_s * a_now

            # ---- contact kinematics + floe dynamics ------------------------
            for ev in [e for e in floes if e["active"]]:
                pv = point_velocity(v_mid, ev["arm_x"], ev["arm_y"])
                close = float(np.dot(pv, ev["n_out"])) - ev["u"]
                ev["pen"] += close * dt_s
                if close < 0.0:
                    ev["unloading"] = True
                mag_new = contact_magnitude(ev)
                approach_spent = close <= 0.02 * abs(ev["v_rel_in"])
                done_now = (ev["pen"] <= 0.0 and close < 0.0) or approach_spent
                if done_now or (t + dt_s - ev["t_start"]) > 5.0 * max(ev["v_rel_in"], 1e-9) + 5.0:
                    if not done_now:
                        done_now = True
                        mag_new = 0.0
                    # the contact force must vanish on *both* bodies in the same
                    # step; otherwise the floe gains an impulse the hull never sees
                    mag_new = 0.0
                impulse = 0.5 * (ev["mag_start"] + mag_new) * dt_s
                ev["J_contact"] += impulse
                # mutation controls (negative controls for the ledger test)
                factor = 1.0
                if mutation == "floe_reaction_half":
                    factor = 0.5
                elif mutation == "no_floe_reaction":
                    factor = 0.0
                ev["u"] += factor * impulse / ev["mass"]
                if done_now or (t + dt_s - ev["t_start"]) > 5.0 * max(ev["v_rel_in"], 1e-9) + 5.0:
                    if not done_now:
                        done_now = True
                        mag_new = 0.0
                    ev["active"] = False
                    ev["done"] = True
                    ev["t_end"] = t + dt_s
                    ev["v_rel_out"] = close
                    ev["peak_until"] = t + dt_s + peak_window_s

            # ---- forces at the end of the step -----------------------------
            mags_new = {id(ev): contact_magnitude(ev) for ev in floes if ev["active"]}
            f_ice_new = assemble(mags_new, [ev for ev in floes if ev["active"]])
            ret_new = self.retardation_force(X)
            f_hydro_new = self.hydrodynamic_forces(q_new, v_pred, ret_new,
                                                   t + dt_s, excitation)
            a_new = (f_hydro_new + f_ice_new) / model.generalized_mass
            v = v + 0.5 * dt_s * (a_now + a_new)
            q = q_new
            ret = ret_new

            # ---- response peaks -------------------------------------------
            for ev in floes:
                if ev["active"] or (ev["done"] and not ev["peak_done"]):
                    ev["peak"] = np.maximum(ev["peak"], [abs(q[2]), abs(q[4])])
                    if ev["done"] and t + dt_s >= ev["peak_until"]:
                        ev["peak_done"] = True

            # ---- close the cluster ----------------------------------------
            if cluster is not None and not any(ev["active"] for ev in floes):
                cluster["times"].append(t + dt_s)
                cluster["ext3"].append(float(f_hydro_new[2] - ret_new[2]))
                cluster["ext5"].append(float(f_hydro_new[4] - ret_new[4]))
                cluster["ice3"].append(float(f_ice_new[2]))
                cluster["ice5"].append(float(f_ice_new[4]))
                cluster["ret3"].append(float(ret_new[2]))
                cluster["ret5"].append(float(ret_new[4]))
                events.extend(self._finalise_cluster(cluster, t + dt_s, v, model,
                                                     mutation))
                cluster = None

        return {
            "time": rec_t,
            "heave_m": rec_q[:, 1],
            "pitch_rad": rec_q[:, 2],
            "surge_m": rec_q[:, 0],
            "heave_vel": rec_v[:, 1],
            "pitch_vel": rec_v[:, 2],
            "ice_force_N": rec_fice,
            "dt_s": dt_s,
            "duration_s": duration_s,
            "events": events,
            "encounter_count": int(sum(1 for ev in floes if ev["started"])),
            "scheduled_arrivals": scheduled,
            "completed_contacts": len(events),
            "poisson_rate_hz": float(rate),
            "poisson_expected": float(expected),
            "model": model,
        }

    # -- momentum ledger ------------------------------------------------------
    def _finalise_cluster(self, cluster, t_end, v, model, mutation):
        """Close the momentum ledger of one contact cluster (per-impact rows)."""
        members = cluster["members"]
        times = np.array(cluster["times"])
        ext3 = np.array(cluster["ext3"])
        ext5 = np.array(cluster["ext5"])
        ice3 = np.array(cluster["ice3"])
        ice5 = np.array(cluster["ice5"])
        ret3_int = _trapz(np.array(cluster["ret3"]), times)
        ret5_int = _trapz(np.array(cluster["ret5"]), times)
        ext3_int = _trapz(ext3, times)
        ext5_int = _trapz(ext5, times)
        j3_ice = _trapz(ice3, times)
        j5_ice = _trapz(ice5, times)
        m33 = model.mass_rigid[2]
        i55 = model.mass_rigid[4]
        a33 = model.radiation["heave"].added_mass_inf
        a55 = model.radiation["pitch"].added_mass_inf
        dv3 = v[2] - cluster["v3"]
        dv5 = v[4] - cluster["v5"]
        p3f0 = sum(p[0] for p in cluster["p_floe_start"])
        p5f0 = sum(p[1] for p in cluster["p_floe_start"])
        p3f1 = 0.0
        p5f1 = 0.0
        for ev in members:
            vz = ev["u"] * ev["n_out"][2]
            p3f1 += ev["mass"] * vz
            p5f1 += ev["arm_x"] * ev["mass"] * vz
        # hull + fluid momentum: (m + A_inf) dv, and the retardation impulse
        # enters with a minus sign because the radiation force is -d/dt(int K qd)
        dp3 = (m33 + a33) * dv3 - ret3_int + (p3f1 - p3f0)
        dp5 = (i55 + a55) * dv5 - ret5_int + (p5f1 - p5f0)
        denom3 = abs(j3_ice) if abs(j3_ice) > 0.0 else 1.0
        denom5 = abs(j5_ice) if abs(j5_ice) > 0.0 else 1.0
        resid3 = abs(dp3 - ext3_int) / denom3
        resid5 = abs(dp5 - ext5_int) / denom5
        rows = []
        for ev in members:
            m_h = model.generalized_mass[2]
            m_red = m_h * ev["mass"] / (m_h + ev["mass"])
            rows.append(dict(
                t_arrival_s=ev["t_arrival"], t_start_s=ev["t_start"],
                duration_s=ev["t_end"] - ev["t_start"],
                diameter_m=ev["diameter"], thickness_m=ev["thickness"],
                floe_mass_kg=ev["mass"], azimuth_deg=math.degrees(ev["azimuth"]),
                arm_x_m=ev["arm_x"],
                v_rel_in_mps=ev["v_rel_in"],
                v_rel_out_mps=ev.get("v_rel_out", float("nan")),
                restitution=(-ev.get("v_rel_out", 0.0) / ev["v_rel_in"])
                if ev["v_rel_in"] else float("nan"),
                p_in_kgmps=m_red * abs(ev["v_rel_in"]),
                J_contact_Ns=ev["J_contact"],
                dp_hull_kgmps=m33 * dv3,
                dp_floe_kgmps=(p3f1 - p3f0),
                dp_fluid_kgmps=a33 * dv3 - ret3_int,
                ext_impulse_heave_Ns=ext3_int,
                resid_heave=resid3, resid_pitch=resid5,
                resid_max=max(resid3, resid5),
                cluster_size=len(members),
                peak_heave_m=float(ev["peak"][0]),
                peak_pitch_rad=float(ev["peak"][1]),
                mutation=mutation or "none",
            ))
        return rows


def _trapz(y, x):
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


# --------------------------------------------------------------------------- #
# post-processing helpers
# --------------------------------------------------------------------------- #
def fit_rao_time_domain(model: ShipModel, omega, periods: int, dt: float,
                        n_fit_periods: int = 8) -> dict:
    """Run the zero-ice time-domain case for one frequency and fit the amplitude."""
    exc = model.excitation(omega)
    sim = Cummins6DOF(model)
    duration = periods * 2.0 * math.pi / omega
    out = sim.run(duration, dt,
                  excitation=lambda t: exc * math.cos(omega * t),
                  ice=None, commanded_speed=None)
    return {
        "time": out["time"], "heave": out["heave_m"], "pitch": out["pitch_rad"],
        "duration": duration, "excitation": exc,
    }


def harmonic_amplitude(time, series, omega, n_periods=8) -> float:
    """Least-squares amplitude over the last ``n_periods`` whole periods."""
    t = np.asarray(time)
    x = np.asarray(series)
    period = 2.0 * math.pi / omega
    t_end = t[-1]
    t_start = t_end - n_periods * period
    mask = t >= t_start - 1e-12
    tt = t[mask]
    xx = x[mask]
    c = np.cos(omega * tt)
    s = np.sin(omega * tt)
    basis = np.vstack([c, s]).T
    coef, *_ = np.linalg.lstsq(basis, xx, rcond=None)
    return float(math.hypot(coef[0], coef[1]))


def weibull_mle(data) -> dict:
    """2-parameter Weibull MLE (Newton on the shape), with Weibull-plot R^2."""
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x) & (x > 0.0)]
    n = x.size
    if n < 5:
        raise ValueError("need at least 5 positive samples")
    lx = np.log(x)
    mean_lx = float(np.mean(lx))
    # profile likelihood over shape k
    def score(k):
        w = x**k
        sw = float(np.sum(w))
        return 1.0 / k + mean_lx - float(np.sum(w * lx)) / sw

    lo, hi = 0.05, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if score(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    lam = float((np.mean(x**k)) ** (1.0 / k))
    # Weibull probability plot: y = ln(-ln(1-F)) vs ln(x); R^2 of the line
    xs = np.sort(x)
    F = (np.arange(1, n + 1) - 0.3) / (n + 0.4)  # Blom plotting positions
    y = np.log(-np.log(1.0 - F))
    u = np.log(xs)
    slope, intercept = np.polyfit(u, y, 1)
    yhat = slope * u + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "n": int(n),
        "shape_k": float(k),
        "scale_lambda": lam,
        "mean": float(lam * math.gamma(1.0 + 1.0 / k)),
        "weibull_plot_slope": float(slope),
        "weibull_plot_r2": float(r2),
        "max": float(np.max(x)),
        "p95": float(np.percentile(x, 95)),
    }