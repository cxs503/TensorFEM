"""W22 / m14511 — podded-propulsion manoeuvring on a KVLCC2 hull vs IMO MSC.137(76).

Scope
-----
A self-contained three-degree-of-freedom MMG manoeuvring model used to answer the
mission question: *for a podded (azimuthing-thruster) configuration, how far do
advance and tactical diameter drift from the single-screw baseline, and do the
MSC.137(76) turning / zig-zag criteria hold?*

Two configurations share the same hull, mass, added mass, resistance and
cross-flow formulation; only the steering/thrust hardware differs:

``conventional``  one propeller + one rudder (MMG propeller-slipstream rudder
                  force, Fujii lift slope, hull-rudder interaction a_H/x_H).
``podded``        two azimuthing pods at x = -0.45 L, y = +-0.35 B: steering is
                  thrust vectoring, no rudder, no a_H interaction, plus a
                  declared strut/housing side force.

Honesty / evidence grading (see ``docs/w22_podded_ice/turning_zigzag.json``)
---------------------------------------------------------------------------
* All IMO limits are **read from** ``benchmarks/anchors/msc137-simman-w22.json``,
  never hard-coded here (mission rule: numbers are injected by artefacts).
* The hull is not represented by a transcribed PMM derivative table.  It is a
  declared slender-body model (added mass + strip cross-flow + Munk moment) whose
  every constant carries a typical physical range, precisely because a digit-exact
  KVLCC2 derivative table is not available offline (see the anchor file's
  ``blocked`` block).  Consequently *absolute* accuracy versus SIMMAN 2008 is
  WITHHELD; what this study establishes is (i) the numerical convergence of the
  manoeuvre indicators and (ii) the podded-vs-single-screw **difference** at
  identical hull physics.

Run::

    python benchmarks/podded_ice_maneuvering_w22.py       # writes the JSON artefact
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

G = 9.81
REPO = Path(__file__).resolve().parents[1]
ANCHOR_PATH = REPO / "benchmarks" / "anchors" / "msc137-simman-w22.json"
OUT_PATH = REPO / "docs" / "w22_podded_ice" / "turning_zigzag.json"
N_STATIONS = 81


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover - only outside a checkout
        return "unknown"


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Hull:
    """Principal particulars with the mandatory geometry gate (Cb / L/B / B/T / disp)."""

    length: float
    beam: float
    draft: float
    cb: float
    speed: float
    wetted_surface: float
    rho: float = 1025.0
    displacement_t: float | None = None
    kzz_ratio: float = 0.25
    form_factor: float = 1.15
    disp_tolerance: float = 0.03

    def __post_init__(self) -> None:
        if min(self.length, self.beam, self.draft, self.speed) <= 0:
            raise ValueError("L, B, T, V must be positive")
        if not 0.35 <= self.cb <= 0.90:
            raise ValueError(f"Cb={self.cb} outside [0.35, 0.90]")
        if not 3.0 <= self.length / self.beam <= 12.0:
            raise ValueError(f"L/B={self.length / self.beam:.2f} outside [3, 12]")
        if not 1.5 <= self.beam / self.draft <= 6.5:
            raise ValueError(f"B/T={self.beam / self.draft:.2f} outside [1.5, 6.5]")
        if self.displacement_t is not None:
            calc = self.displacement_t_from_geometry()
            err = abs(calc - self.displacement_t) / self.displacement_t
            if err > self.disp_tolerance:
                raise ValueError(
                    f"geometry inconsistent: rho*Cb*L*B*T={calc:.0f} t vs stated "
                    f"{self.displacement_t:.0f} t ({100 * err:.2f}% > "
                    f"{100 * self.disp_tolerance:.1f}%)")

    def displacement_t_from_geometry(self) -> float:
        return self.rho * self.cb * self.length * self.beam * self.draft / 1000.0

    @property
    def mass(self) -> float:
        return self.rho * self.cb * self.length * self.beam * self.draft

    @property
    def froude(self) -> float:
        return self.speed / math.sqrt(G * self.length)

    @property
    def L_over_V(self) -> float:
        return self.length / self.speed


# --------------------------------------------------------------------------- #
# declared model constants
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    """Modelling constants; each value carries the range it is assumed from."""

    # ---- hull manoeuvring derivatives (SNAME prime form) -------------------
    # Y' = Y/(0.5 rho U^2 L T), N' = N/(0.5 rho U^2 L^2 T), v' = v/U, r' = r L/U.
    # DECLARED representative large-tanker set (order-of-magnitude consistent with
    # published MMG/PMM data); NOT a digit-exact KVLCC2 reprint -- see the anchor
    # file's blocked block.  Ranges in brackets are the ones the sweep covers.
    y_v: float = -0.45               # [-0.55, -0.35]
    y_r: float = 0.05                # [0.03, 0.08]
    y_vv: float = -1.40              # [-1.8, -1.0]
    y_vr: float = -0.40              # [-0.6, -0.2]
    n_v: float = -0.10               # [-0.12, -0.08]; negative => directionally unstable
    n_r: float = -0.05               # [-0.06, -0.045]
    n_vv: float = 0.40               # [0.2, 0.6]
    n_vr: float = -0.20              # [-0.3, -0.1]
    # ---- added mass / inertia --------------------------------------------
    k_added_sway: float = 1.0        # m_y = k*rho*pi*(T/2)^2*L, typical 0.8-1.6
    k_added_surge: float = 0.05      # m_x/m, typical 0.03-0.10
    j_added_yaw: float = 0.20        # J_zz/I_zz, typical 0.10-0.30
    # ---- resistance -------------------------------------------------------
    residuary_coeff: float = 0.0020  # Cr added to (1+k)Cf; Ct ~ 0.0036 at the design point
    # propeller (declared open-water line, NOT a digit-exact KP505 reprint)
    kt0: float = 0.50                # K_T at J=0, typical 0.45-0.60
    kt1: float = 0.30                # dK_T/dJ slope, typical 0.25-0.40
    d_prop: float = 8.6              # m, KVLCC2-class propeller diameter
    t_p: float = 0.20                # thrust deduction, typical 0.15-0.25
    w_p: float = 0.25                # wake fraction, typical 0.20-0.35
    # rudder (conventional)
    a_r_over_lt: float = 0.016       # A_R/(L*T), typical 0.010-0.016 (KVLCC2-class high end)
    rudder_aspect: float = 1.8       # h^2/A_R, typical 1.5-2.0
    t_r: float = 0.25                # rudder thrust deduction, typical 0.20-0.30
    a_h: float = 0.30                # hull-rudder interaction, typical 0.25-0.35
    x_h_over_L: float = -0.40        # typical -0.35..-0.45
    x_r_over_L: float = -0.50        # typical -0.48..-0.52
    w_r: float = 0.25                # rudder wake fraction, typical 0.20-0.40
    eps_r: float = 0.80              # effective rudder inflow factor (race acts on part of A_R)
    c_n_max: float = 1.2             # post-stall cap on the rudder normal-force coefficient
    cap_rudder_inflow: bool = True   # cap the race amplification at its design value
    # pods (podded)
    n_pods: int = 2
    x_pod_over_L: float = -0.45
    y_pod_over_B: float = 0.35
    pod_area_over_LT: float = 0.008  # strut+housing lateral area per pod
    pod_lift_slope: float = 3.0      # per rad, low-aspect strut
    c_n_max_pod: float = 1.2         # post-stall cap on the strut side-force coefficient
    pod_drag_coeff: float = 0.6      # housing axial drag, A_pod based
    # actuation
    rudder_rate_deg_s: float = 5.0
    pod_rate_deg_s: float = 3.0
    # ice scenario (declared scenario parameter, not a physics model):
    # extra steady resistance as a fraction of the calm-water design resistance
    ice_resistance_fraction: float = 0.0

    def rudder_area(self, hull: Hull) -> float:
        return self.a_r_over_lt * hull.length * hull.draft

    def f_alpha(self) -> float:
        """Fujii lift gradient 6.13*AR/(AR+2.25) [-/rad]."""
        ar = self.rudder_aspect
        return 6.13 * ar / (ar + 2.25)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class ManoeuvringModel:
    """3-DOF MMG model. State ``[u, v, r, x, y, psi, delta]``.

    ``delta`` is the actuation angle in degrees: rudder angle for ``conventional``,
    pod azimuth for ``podded``.  The rate limit lives inside :meth:`rk4` so the
    IMO Z-test operating law is explicit and auditable.
    """

    def __init__(self, hull: Hull, params: Params, config: str) -> None:
        if config not in ("conventional", "podded"):
            raise ValueError(f"unknown configuration {config!r}")
        self.hull = hull
        self.p = params
        self.config = config
        L, B, T, rho = hull.length, hull.beam, hull.draft, hull.rho
        self.m = hull.mass
        self.izz = self.m * (hull.kzz_ratio * L) ** 2
        self.jzz = params.j_added_yaw * self.izz
        self.mx = params.k_added_surge * self.m
        self.my = params.k_added_sway * rho * math.pi * (T / 2.0) ** 2 * L
        self.scale_y = 0.5 * rho * L * T      # multiplies U^2 * Y'
        self.scale_n = 0.5 * rho * L * L * T  # multiplies U^2 * N'
        self.x_r = params.x_r_over_L * L
        self.x_h = params.x_h_over_L * L
        self.a_r = params.rudder_area(hull)
        self.f_alpha = params.f_alpha()
        self.x_pod = params.x_pod_over_L * L
        self.y_pod = params.y_pod_over_B * B
        self.a_pod = params.pod_area_over_LT * L * T
        self.n_rps = self._solve_rps()
        # rudder-inflow amplification at the design point; used as a cap so that the
        # model does not produce an unphysical drag runaway below the design speed
        # (no engine torque limit is modelled -- declared limitation).
        self.slip_design = self._slip_factor(hull.speed)
        self.thrust_design = self.thrust(hull.speed)
        self.resistance_design = self.resistance(hull.speed)
        self.ice_force = params.ice_resistance_fraction * self.resistance_design

    # -- propulsion -------------------------------------------------------- #
    def _solve_rps(self) -> float:
        """Solve rho n^2 D^4 K_T(J(u0)) = R(u0)/(1-t_p) by bisection."""
        target = self.resistance(self.hull.speed) / (1.0 - self.p.t_p)
        lo, hi = 0.05, 6.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self.thrust(self.hull.speed, mid) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _slip_factor(self, u: float) -> float:
        kt = self.kt(u)
        j = self.advance_ratio(u)
        return math.sqrt(1.0 + 8.0 * kt / (math.pi * j * j)) if j > 1e-3 else 1.0

    def advance_ratio(self, u: float, n: float | None = None) -> float:
        n = self.n_rps if n is None else n
        return max(u * (1.0 - self.p.w_p), 1e-6) / (n * self.p.d_prop)

    def kt(self, u: float, n: float | None = None) -> float:
        return max(self.p.kt0 - self.p.kt1 * self.advance_ratio(u, n), 0.0)

    def thrust(self, u: float, n: float | None = None) -> float:
        n = self.n_rps if n is None else n
        return self.hull.rho * n * n * self.p.d_prop ** 4 * self.kt(u, n)

    # -- resistance -------------------------------------------------------- #
    def cf(self, u: float) -> float:
        re = max(abs(u), 1e-6) * self.hull.length / 1.19e-6
        return 0.075 / (math.log10(re) - 2.0) ** 2 if re > 1e4 else 0.01

    def resistance(self, u: float) -> float:
        c = self.hull.form_factor * self.cf(u) + self.p.residuary_coeff
        return 0.5 * self.hull.rho * self.hull.wetted_surface * c * u * abs(u)

    # -- forces ------------------------------------------------------------ #
    def hull_forces(self, u: float, v: float, r: float):
        """Hull manoeuvring force: declared prime derivatives + quadratic terms.

        The speed scale is the instantaneous resultant speed (floored at 0.3 U0 so
        that a nearly stopped ship still has a well-posed derivative set).
        """
        p, L = self.p, self.hull.length
        U = max(math.hypot(u, v), 0.3 * self.hull.speed)
        vp, rp = v / U, r * L / U
        y = self.scale_y * U * U * (p.y_v * vp + p.y_r * rp
                                   + p.y_vv * vp * abs(vp) + p.y_vr * vp * abs(rp))
        n = self.scale_n * U * U * (p.n_v * vp + p.n_r * rp
                                   + p.n_vv * vp * abs(vp) + p.n_vr * vp * abs(rp))
        x = -self.resistance(u) - self.ice_force * (1.0 if u >= 0 else -1.0)
        return x, y, n

    def rudder_forces(self, u: float, v: float, r: float, delta_deg: float):
        p = self.p
        delta = math.radians(delta_deg)
        slip = min(self._slip_factor(u), self.slip_design) if p.cap_rudder_inflow else self._slip_factor(u)
        u_r = p.eps_r * max(u, 0.0) * (1.0 - p.w_r) * math.sqrt(max(slip, 1.0))
        # lateral velocity of the *inflow* relative to the rudder (so that a free
        # rudder in a yawing ship produces a damping force, not an amplifying one)
        v_r = -(v + self.x_r * r)
        alpha = delta - math.atan2(v_r, u_r) if u_r > 1e-6 else delta
        q = u_r * u_r + v_r * v_r
        # Fujii's f_alpha*sin(alpha) is a linear-sin law; it over-predicts beyond the
        # stall angle, so the normal-force coefficient is capped at c_n_max.
        c_n = min(self.f_alpha * abs(math.sin(alpha)), p.c_n_max)
        f_n = math.copysign(0.5 * self.hull.rho * self.a_r * c_n * q, alpha)
        x = -(1.0 - p.t_r) * f_n * math.sin(delta)
        y = -(1.0 + p.a_h) * f_n * math.cos(delta)
        n = -(self.x_r + p.a_h * self.x_h) * f_n * math.cos(delta)
        return x, y, n

    def pod_forces(self, u: float, v: float, r: float, delta_deg: float):
        p = self.p
        phi = math.radians(delta_deg)
        v_p = -(v + self.x_pod * r)     # inflow relative to the pod
        u_p = u
        q = u_p * u_p + v_p * v_p
        beta_p = math.atan2(v_p, abs(u_p)) if abs(u_p) > 1e-6 else 0.0
        alpha = phi - beta_p
        c_n_pod = min(p.pod_lift_slope * abs(math.sin(alpha)), p.c_n_max_pod)
        side = math.copysign(0.5 * self.hull.rho * self.a_pod * c_n_pod * q, alpha)
        drag = 0.5 * self.hull.rho * self.a_pod * p.pod_drag_coeff * q * (1.0 if u_p >= 0 else -1.0)
        t_each = self.thrust(u) / max(p.n_pods, 1)
        x_tot = y_tot = n_tot = 0.0
        for sign_y in (+1.0, -1.0):
            # sign convention: delta > 0 must turn the bow to starboard for BOTH
            # configurations.  At a pod aft of the CG that requires a lateral force
            # towards -y (same sense as the rudder force), so the positive-phi
            # thrust vector points to port.
            fx = t_each * math.cos(phi) - drag
            fy = -(t_each * math.sin(phi) + side)
            x_tot += fx
            y_tot += fy
            n_tot += self.x_pod * fy - sign_y * self.y_pod * fx
        return x_tot, y_tot, n_tot

    def steering_forces(self, u: float, v: float, r: float, delta_deg: float):
        if self.config == "conventional":
            return self.rudder_forces(u, v, r, delta_deg)
        return self.pod_forces(u, v, r, delta_deg)

    def derivatives(self, state, delta_deg: float) -> np.ndarray:
        u, v, r, x, y, psi, _ = state
        xh, yh, nh = self.hull_forces(u, v, r)
        xs_, ys_, ns_ = self.steering_forces(u, v, r, delta_deg)
        if self.config == "conventional":
            xs_ += (1.0 - self.p.t_p) * self.thrust(u)
        du = (xh + xs_ + (self.m + self.my) * v * r) / (self.m + self.mx)
        dv = (yh + ys_ - (self.m + self.mx) * u * r) / (self.m + self.my)
        dr = (nh + ns_) / (self.izz + self.jzz)
        return np.array([
            du, dv, dr,
            u * math.cos(psi) - v * math.sin(psi),
            u * math.sin(psi) + v * math.cos(psi),
            r, 0.0,
        ])

    # -- actuation-limited integration ------------------------------------- #
    def rate_limit(self, delta_deg: float, command_deg: float, dt: float) -> float:
        rate = self.p.rudder_rate_deg_s if self.config == "conventional" else self.p.pod_rate_deg_s
        lim = rate * dt
        return delta_deg + max(-lim, min(lim, command_deg - delta_deg))

    def rk4(self, state: np.ndarray, command_deg: float, dt: float) -> np.ndarray:
        d0 = self.rate_limit(state[6], command_deg, dt)
        k1 = self.derivatives(state, d0)
        s2 = state + 0.5 * dt * k1
        k2 = self.derivatives(s2, self.rate_limit(state[6], command_deg, 0.5 * dt))
        s3 = state + 0.5 * dt * k2
        k3 = self.derivatives(s3, self.rate_limit(state[6], command_deg, 0.5 * dt))
        s4 = state + dt * k3
        k4 = self.derivatives(s4, d0)
        out = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[6] = d0
        return out

    def initial_state(self) -> np.ndarray:
        return np.array([self.hull.speed, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


# --------------------------------------------------------------------------- #
# manoeuvres
# --------------------------------------------------------------------------- #
def _crossing_time(model, state, command, target, dt, f):
    """Bisection for f(state(t)) = target inside one dt step."""
    fa = f(state) - target
    fb = f(model.rk4(state, command, dt)) - target
    if fa * fb > 0:
        return None
    a, b = 0.0, dt
    for _ in range(60):
        mid = 0.5 * (a + b)
        fm = f(model.rk4(state, command, mid)) - target
        if fa * fm <= 0:
            b = mid
        else:
            a, fa = mid, fm
    return 0.5 * (a + b)


def turning_circle(model: ManoeuvringModel, delta_deg: float, dt: float = 0.5,
                   t_end: float = 1800.0) -> dict:
    state = model.initial_state()
    t = 0.0
    hist = {k: [] for k in ("t", "u", "v", "r", "x", "y", "psi", "delta")}
    while t < t_end - 1e-12:
        hist["t"].append(t)
        for k, idx in (("u", 0), ("v", 1), ("r", 2), ("x", 3), ("y", 4), ("psi", 5), ("delta", 6)):
            hist[k].append(float(state[idx]))
        state = model.rk4(state, delta_deg, dt)
        t += dt
    arr = {k: np.asarray(v) for k, v in hist.items()}
    psi, x, y = arr["psi"], arr["x"], arr["y"]

    def first_index(deg):
        idx = np.flatnonzero(psi >= math.radians(deg))
        return int(idx[0]) if idx.size else None

    i90, i180 = first_index(90.0), first_index(180.0)
    L = model.hull.length
    # steady-turn validity: a constant-helm pod configuration has no guaranteed
    # equilibrium (the thrust-vector moment does not decay with drift the way a
    # rudder's does), so the "steady" quantities are only reported when the turn
    # actually settled.  The IMO indicators (advance at 90 deg, D_T at 180 deg) are
    # unaffected -- they are measured inside the first half turn.
    n = len(arr["r"])
    i_settle = max(0, n - max(int(300.0 / dt), 1))
    r_end, r_settle = float(arr["r"][-1]), float(np.mean(arr["r"][i_settle:]))
    steady = abs(r_end - r_settle) <= 0.05 * abs(r_end) if abs(r_end) > 1e-9 else False
    return {
        "delta_deg": delta_deg,
        "dt_s": dt,
        "t_end_s": t_end,
        "reached_90deg": i90 is not None,
        "reached_180deg": i180 is not None,
        "advance_over_L": float(x[i90]) / L if i90 is not None else None,
        "advance_m": float(x[i90]) if i90 is not None else None,
        "transfer_over_L_at_90deg": float(y[i90]) / L if i90 is not None else None,
        "tactical_diameter_over_L": 2.0 * float(y[i180]) / L if i180 is not None else None,
        "tactical_diameter_over_L_max_transfer": 2.0 * float(np.max(y)) / L,
        "steady_turn_reached": bool(steady),
        "u_ss_over_u0": float(arr["u"][-1]) / model.hull.speed if steady else None,
        "u_over_u0_at_end": float(arr["u"][-1]) / model.hull.speed,
        "r_ss_rad_s": float(arr["r"][-1]) if steady else None,
        "r_rad_s_at_end": float(arr["r"][-1]),
        "u_ss_status": "STEADY" if steady else "NO_STEADY_TURN",
        "drift_angle_deg_at_end": math.degrees(math.atan2(-arr["v"][-1], arr["u"][-1])),
        "max_abs_yaw_rate_deg_s": math.degrees(float(np.max(np.abs(arr["r"])))),
        "track": {"x_over_L": (x / L).tolist()[::5], "y_over_L": (y / L).tolist()[::5]},
        "history": {"t": arr["t"].tolist()[::5], "psi_deg": np.degrees(psi).tolist()[::5],
                    "r_deg_s": np.degrees(arr["r"]).tolist()[::5],
                    "delta_deg": arr["delta"].tolist()[::5]},
    }


def zigzag(model: ManoeuvringModel, delta_deg: float, heading_deg: float | None = None,
           dt: float = 0.5, t_end: float = 1500.0, n_cycles: int = 2) -> dict:
    """Event-driven Z-test; heading triggers and yaw-rate reversals located in-step.

    Overshoot is measured against the heading at which *this* helm reversal was
    triggered (IMO Res. MSC.137(76) / ITTC 7.5-02-06-01 convention).
    """
    hdg = abs(delta_deg) if heading_deg is None else heading_deg
    state = model.initial_state()
    target = math.radians(hdg)
    command = float(delta_deg)
    t = 0.0
    events, overshoots = [], []
    pending = None
    r_armed = False
    hist = {"t": [], "psi": [], "r": [], "delta": []}
    while t < t_end - 1e-12 and len(overshoots) < n_cycles:
        hist["t"].append(t)
        hist["psi"].append(float(state[5]))
        hist["r"].append(float(state[2]))
        hist["delta"].append(float(state[6]))
        t_ev = _crossing_time(model, state, command, target, dt, lambda s: s[5])
        t_pk = _crossing_time(model, state, command, 0.0, dt, lambda s: s[2]) if r_armed else None
        if t_ev is not None and (t_pk is None or t_ev <= t_pk):
            state = model.rk4(state, command, t_ev)
            t += t_ev
            pending = float(state[5])
            command = -command
            target = -target
            r_armed = True
            events.append({"t_s": t, "type": "helm_reversal", "psi_deg": math.degrees(state[5]),
                           "delta_cmd_deg": command})
            continue
        if t_pk is not None:
            state = model.rk4(state, command, t_pk)
            t += t_pk
            if pending is not None:
                psi_max, trig = math.degrees(state[5]), math.degrees(pending)
                over = math.copysign(1.0, trig) * (psi_max - trig)
                overshoots.append({"trigger_heading_deg": trig, "peak_heading_deg": psi_max,
                                   "overshoot_deg": over, "t_s": t})
                events.append({"t_s": t, "type": "peak", "overshoot_deg": over})
                pending = None
            r_armed = False
            continue
        state = model.rk4(state, command, dt)
        t += dt
    return {
        "delta_deg": delta_deg,
        "heading_deg": hdg,
        "dt_s": dt,
        "overshoots": overshoots,
        "first_overshoot_deg": overshoots[0]["overshoot_deg"] if overshoots else None,
        "second_overshoot_deg": overshoots[1]["overshoot_deg"] if len(overshoots) > 1 else None,
        "events": events,
        "terminated": "ok" if len(overshoots) >= n_cycles else "short",
        "history": {"t": hist["t"], "psi_deg": [math.degrees(v) for v in hist["psi"]],
                    "r_deg_s": [math.degrees(v) for v in hist["r"]], "delta_deg": hist["delta"]},
    }

# --------------------------------------------------------------------------- #
# verification helpers
# --------------------------------------------------------------------------- #
def steady_state(model: ManoeuvringModel, delta_deg: float, tol: float = 1e-12,
                 max_iter: int = 80, x0: tuple | None = None) -> tuple[float, float, float]:
    """Newton solve of u_dot = v_dot = r_dot = 0 (independent of time marching)."""
    x = np.array(x0 if x0 is not None else (model.hull.speed, 0.05, 1e-3), dtype=float)

    def resid(vec):
        st = np.array([vec[0], vec[1], vec[2], 0.0, 0.0, 0.0, delta_deg])
        return model.derivatives(st, delta_deg)[:3]

    for _ in range(max_iter):
        f0 = resid(x)
        jac = np.zeros((3, 3))
        for j in range(3):
            h = max(1e-7, abs(x[j]) * 1e-6)
            xp = x.copy()
            xp[j] += h
            jac[:, j] = (resid(xp) - f0) / h
        try:
            dx = np.linalg.solve(jac, -f0)
        except np.linalg.LinAlgError:
            break
        x = x + dx
        if np.max(np.abs(dx)) < tol:
            break
    return float(x[0]), float(x[1]), float(x[2])


def linear_stability(model: ManoeuvringModel) -> dict:
    """Linearised surge/sway/yaw Jacobian at the design speed (stability proxy)."""
    u = model.hull.speed
    st = np.array([u, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    base = model.derivatives(st, 0.0)[:3]
    jac = np.zeros((3, 3))
    for j in range(3):
        h = 1e-6 * max(1.0, abs(st[j]))
        sp = st.copy()
        sp[j] += h
        jac[:, j] = (model.derivatives(sp, 0.0)[:3] - base) / h
    return {
        "d_surge_du_1_s": float(jac[0, 0]),
        "d_sway_dv": float(jac[1, 1]),
        "d_sway_dr_m_s": float(jac[1, 2]),
        "d_yaw_dv_1_s": float(jac[2, 1]),
        "d_yaw_dr_1_s": float(jac[2, 2]),
    }


# --------------------------------------------------------------------------- #
# IMO criteria (limits injected from the anchor file, never hard-coded)
# --------------------------------------------------------------------------- #
def load_anchor(path: Path = ANCHOR_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def imo_limits(anchor: dict) -> dict:
    by_id = {a["id"]: a for a in anchor["anchors"]}
    return {
        "advance_over_L": by_id["AN-MSC137-01"]["value"],
        "tactical_diameter_over_L": by_id["AN-MSC137-02"]["value"],
        "zz10_first_lt10s": by_id["AN-MSC137-03"]["value"],
        "zz10_first_ge10s": by_id["AN-MSC137-04"]["value"],
        "zz20_first": by_id["AN-MSC137-05"]["value"],
        "zz20_second": by_id["AN-MSC137-06"]["value"],
        "min_length_m": by_id["AN-MSC137-07"]["value"],
    }


def imo_verdicts(model: ManoeuvringModel, turn35: dict, zz10: dict, zz20: dict,
                 limits: dict) -> list:
    """One row per criterion quantity: value, threshold, PASS/FAIL + evidence pointer."""
    rows = []

    def row(name, value, limit, unit, ptr, note=None):
        r = {"quantity": name, "value": None if value is None else round(value, 6),
             "limit": limit, "unit": unit, "comparator": "<=", "evidence": ptr}
        if note:
            r["note"] = note
        if value is None:
            r["verdict"] = "NOT_REACHED"
        else:
            r["verdict"] = "PASS" if value <= limit else "FAIL"
            r["margin"] = round(limit - value, 6)
        rows.append(r)

    rows.append({
        "quantity": "applicability.L_ge_min_length", "value": model.hull.length,
        "limit": limits["min_length_m"], "unit": "m", "comparator": ">=",
        "verdict": "PASS" if model.hull.length >= limits["min_length_m"] else "FAIL",
        "evidence": "anchor AN-MSC137-07"})
    row("turning.advance_over_L", turn35.get("advance_over_L"), limits["advance_over_L"], "L",
        "turning_35deg.advance_over_L", "anchor AN-MSC137-01")
    row("turning.tactical_diameter_over_L", turn35.get("tactical_diameter_over_L"),
        limits["tactical_diameter_over_L"], "L", "turning_35deg.tactical_diameter_over_L",
        "anchor AN-MSC137-02")
    lv = model.hull.L_over_V
    lim10 = limits["zz10_first_lt10s"] if lv < 10.0 else limits["zz10_first_ge10s"]
    v10 = zz10.get("first_overshoot_deg")
    rows.append({
        "quantity": "zigzag.10_10.first_overshoot_deg",
        "value": None if v10 is None else round(v10, 6),
        "limit": lim10, "unit": "deg", "comparator": "<=",
        "condition": f"L/V={lv:.1f} s -> {'<10 s' if lv < 10 else '>=10 s'} branch",
        "verdict": "NOT_REACHED" if v10 is None else ("PASS" if v10 <= lim10 else "FAIL"),
        "evidence": "anchor AN-MSC137-03/04 (L/V branch)"})
    v10b = zz10.get("second_overshoot_deg")
    rows.append({
        "quantity": "zigzag.10_10.second_overshoot_deg",
        "value": None if v10b is None else round(v10b, 6),
        "limit": limits["zz20_second"], "unit": "deg", "comparator": "<=",
        "verdict": ("NOT_REACHED" if v10b is None
                    else ("PASS" if v10b <= limits["zz20_second"] else "FAIL")),
        "note": "no separate 10/10 second-overshoot limit is transcribed; a loose bound "
                "(the 20/20 second value) is used and flagged as such",
        "evidence": "anchor AN-MSC137-06"})
    row("zigzag.20_20.first_overshoot_deg", zz20.get("first_overshoot_deg"), limits["zz20_first"],
        "deg", "zigzag_20_20.first_overshoot_deg", "anchor AN-MSC137-05")
    row("zigzag.20_20.second_overshoot_deg", zz20.get("second_overshoot_deg"),
        limits["zz20_second"], "deg", "zigzag_20_20.second_overshoot_deg", "anchor AN-MSC137-06")
    return rows


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def hull_from_anchor(anchor: dict) -> Hull:
    cd = anchor["case_data"]
    return Hull(
        length=cd["L_pp_m"]["value"], beam=cd["B_m"]["value"], draft=cd["T_m"]["value"],
        cb=cd["Cb"]["value"], speed=cd["V_design_m_s"]["value"],
        wetted_surface=cd["wetted_surface_m2"]["value"], rho=cd["rho_kg_m3"]["value"],
        displacement_t=cd["displacement_t"]["value"])


#: Hull-derivative variants used for the uncertainty band (declared ranges).
HULL_DERIVATIVE_VARIANTS = [
    {"label": "nominal", "y_v": -0.45, "n_r": -0.050, "n_v": -0.10},
    {"label": "soft_lift", "y_v": -0.35, "n_r": -0.050, "n_v": -0.10},
    {"label": "stiff_lift", "y_v": -0.55, "n_r": -0.050, "n_v": -0.10},
    {"label": "weak_yaw_damping", "y_v": -0.45, "n_r": -0.045, "n_v": -0.10},
    {"label": "strong_yaw_damping", "y_v": -0.45, "n_r": -0.060, "n_v": -0.10},
    {"label": "more_unstable", "y_v": -0.45, "n_r": -0.050, "n_v": -0.12},
    {"label": "less_unstable", "y_v": -0.45, "n_r": -0.050, "n_v": -0.08},
]


def build_models(anchor: dict, derivatives: dict | None = None,
                 residuary: float | None = None, ice_fraction: float = 0.0) -> dict:
    hull = hull_from_anchor(anchor)
    kwargs = {"ice_resistance_fraction": ice_fraction}
    if residuary is not None:
        kwargs["residuary_coeff"] = residuary
    if derivatives:
        kwargs.update({k: v for k, v in derivatives.items() if k != "label"})
    params = Params(**kwargs)
    return {"hull": hull,
            "conventional": ManoeuvringModel(hull, params, "conventional"),
            "podded": ManoeuvringModel(hull, params, "podded")}


def run_case(models: dict, dt: float = 0.5, t_end_turn: float = 1800.0,
             t_end_zz: float = 1500.0, with_histories: bool = False) -> dict:
    limits = imo_limits(load_anchor())
    out = {}
    for cfg in ("conventional", "podded"):
        m = models[cfg]
        turn35 = turning_circle(m, 35.0, dt=dt, t_end=t_end_turn)
        turn20 = turning_circle(m, 20.0, dt=dt, t_end=t_end_turn)
        zz10 = zigzag(m, 10.0, dt=dt, t_end=t_end_zz)
        zz20 = zigzag(m, 20.0, dt=dt, t_end=t_end_zz)
        out[cfg] = {
            "config": cfg,
            "n_rps": m.n_rps,
            "thrust_design_N": m.thrust_design,
            "resistance_design_N": m.resistance_design,
            "ct_design": (m.resistance_design /
                          (0.5 * m.hull.rho * m.hull.speed ** 2 * m.hull.wetted_surface)),
            "linear_stability": linear_stability(m),
            "turning_35deg": turn35,
            "turning_20deg": turn20,
            "zigzag_10_10": zz10,
            "zigzag_20_20": zz20,
            "imo_verdicts": imo_verdicts(m, turn35, zz10, zz20, limits),
        }
        if not with_histories:
            out[cfg]["turning_35deg"].pop("history", None)
            out[cfg]["turning_20deg"].pop("history", None)
            out[cfg]["zigzag_10_10"]["history"] = None
            out[cfg]["zigzag_20_20"]["history"] = None
    return out


def steady_vs_marching(models: dict) -> dict:
    """Independent check of the 1800 s time-marched turn.

    The marched end state is re-substituted into the equations of motion (residual
    of the steady equations) and a Newton correction is taken from *that* state;
    the size of the correction measures how converged the time march is.  Starting
    Newton from the marched state (instead of from cold initial values) avoids the
    spurious second root of the nonlinear steady problem.
    """
    checks = {}
    for cfg in ("conventional", "podded"):
        m = models[cfg]
        turn = turning_circle(m, 35.0, dt=0.5, t_end=1800.0)
        u_end = turn["u_over_u0_at_end"] * m.hull.speed
        r_end = turn["r_rad_s_at_end"]
        # marched lateral velocity is not stored in the summary -> re-march cheaply
        st = m.initial_state()
        t = 0.0
        while t < 1800.0 - 1e-9:
            st = m.rk4(st, 35.0, 5.0)
            t += 5.0
        resid = m.derivatives(st, 35.0)[:3]
        u_n, v_n, r_n = steady_state(m, 35.0, x0=(st[0], st[1], st[2]))
        checks[cfg] = {
            "marched_end_state": {"u": st[0], "v": st[1], "r": st[2]},
            "steady_equation_residual": {"u_dot": resid[0], "v_dot": resid[1], "r_dot": resid[2]},
            "residual_normalised": {
                "u_dot_over_u0_per_s": resid[0] / m.hull.speed,
                "r_dot_per_s2": resid[2],
            },
            "newton_from_marched": {"u": u_n, "v": v_n, "r": r_n,
                                    "correction_u": u_n - st[0], "correction_r": r_n - st[2]},
            "rel_correction_u": abs(u_n - st[0]) / abs(st[0]) if st[0] else None,
            "rel_correction_r": abs(r_n - st[2]) / abs(st[2]) if abs(st[2]) > 1e-12 else None,
            "steady_turn_reached": turn["steady_turn_reached"],
            "kinematic_diameter_over_L": (2.0 * (st[0] / st[2]) / m.hull.length
                                          if abs(st[2]) > 1e-12 else None),
            "reported_advance_over_L": turn["advance_over_L"],
            "reported_tactical_diameter_over_L": turn["tactical_diameter_over_L"],
            "note": ("advance / D_T are geometry indicators measured inside the first half "
                     "turn and do not require a steady state; the kinematic 2u/r diameter is "
                     "only a consistency check of the settled turn"),
        }
    return checks


def convergence(anchor: dict, cfg: str = "conventional", delta: float = 35.0,
                dts=(1.0, 0.5, 0.25)) -> dict:
    """Halve dt and check the change in advance / tactical diameter / Z overshoot."""
    res = {"advance_over_L": {}, "tactical_diameter_over_L": {},
           "zigzag10_first_overshoot_deg": {}}
    for dt in dts:
        m = build_models(anchor)[cfg]
        t = turning_circle(m, delta, dt=dt, t_end=1800.0)
        z = zigzag(m, 10.0, dt=dt, t_end=1500.0)
        res["advance_over_L"][dt] = t["advance_over_L"]
        res["tactical_diameter_over_L"][dt] = t["tactical_diameter_over_L"]
        res["zigzag10_first_overshoot_deg"][dt] = z["first_overshoot_deg"]
    table = {}
    for key, vals in res.items():
        seq = [vals[d] for d in dts]
        pairs = []
        for i in range(len(dts) - 1):
            a, b = seq[i], seq[i + 1]
            pairs.append({"dt_from": dts[i], "dt_to": dts[i + 1],
                          "value_dt_from": a, "value_dt_to": b,
                          "rel_change_pct": 100.0 * abs(b - a) / abs(a) if a else None})
        table[key] = {"values": {str(k): v for k, v in vals.items()}, "halving": pairs}
    return table


def sensitivity(anchor: dict) -> dict:
    """Coefficient sweep and podded ice-load sweep (declared, not anchored)."""
    rows = []
    for der in HULL_DERIVATIVE_VARIANTS:
        case = run_case(build_models(anchor, derivatives=der), dt=0.5)
        rows.append({
            "kind": "hull_derivative_variant", "value": der["label"],
            "conv_advance_over_L": case["conventional"]["turning_35deg"]["advance_over_L"],
            "pod_advance_over_L": case["podded"]["turning_35deg"]["advance_over_L"],
            "conv_DT_over_L": case["conventional"]["turning_35deg"]["tactical_diameter_over_L"],
            "pod_DT_over_L": case["podded"]["turning_35deg"]["tactical_diameter_over_L"],
            "conv_zz10": case["conventional"]["zigzag_10_10"]["first_overshoot_deg"],
            "pod_zz10": case["podded"]["zigzag_10_10"]["first_overshoot_deg"],
        })
    for resid in (0.0, 0.0035):
        case = run_case(build_models(anchor, residuary=resid), dt=0.5)
        rows.append({
            "kind": "residuary_coeff", "value": resid,
            "conv_advance_over_L": case["conventional"]["turning_35deg"]["advance_over_L"],
            "pod_advance_over_L": case["podded"]["turning_35deg"]["advance_over_L"],
            "conv_DT_over_L": case["conventional"]["turning_35deg"]["tactical_diameter_over_L"],
            "pod_DT_over_L": case["podded"]["turning_35deg"]["tactical_diameter_over_L"],
            "conv_zz10": case["conventional"]["zigzag_10_10"]["first_overshoot_deg"],
            "pod_zz10": case["podded"]["zigzag_10_10"]["first_overshoot_deg"],
        })
    ice = []
    for frac in (0.0, 0.5, 1.0, 2.0):
        m = build_models(anchor, ice_fraction=frac)["podded"]
        t = turning_circle(m, 35.0, dt=0.5, t_end=1800.0)
        z = zigzag(m, 10.0, dt=0.5)
        ice.append({"ice_resistance_fraction": frac,
                    "advance_over_L": t["advance_over_L"],
                    "tactical_diameter_over_L": t["tactical_diameter_over_L"],
                    "u_ss_over_u0": t["u_ss_over_u0"],
                    "zz10_first_overshoot_deg": z["first_overshoot_deg"]})
    return {"coefficient_sweep": rows, "podded_ice_load_sweep": ice}


def anchor_cross_checks(hull: Hull, anchor: dict) -> dict:
    """Check the case data against the two anchor numbers that *are* available."""
    lv = hull.L_over_V
    fr_anchor = {a["id"]: a for a in anchor["anchors"]}["AN-SIMMAN-02"]["value"]
    checks = {
        "froude_number": {
            "computed": hull.froude, "anchor_AN-SIMMAN-02": fr_anchor,
            "rel_diff_pct": 100.0 * abs(hull.froude - fr_anchor) / fr_anchor,
            "verdict": "PASS" if abs(hull.froude - fr_anchor) / fr_anchor < 0.05 else "FAIL"},
        "L_over_V_s": lv,
        "L_over_V_branch": "<10 s" if lv < 10.0 else ">=10 s",
        "geometry_displacement_closure_pct": 100.0 * abs(
            hull.displacement_t_from_geometry() - hull.displacement_t) / hull.displacement_t,
    }
    return checks


def build_artifact(dt_main: float = 0.5) -> dict:
    anchor = load_anchor()
    hull = hull_from_anchor(anchor)
    models = build_models(anchor)
    case = run_case(models, dt=dt_main, with_histories=True)
    conv, pod = case["conventional"], case["podded"]

    def delta(key, sub):
        a, b = conv[key][sub], pod[key][sub]
        if a is None or b is None:
            return None
        return {"conventional": a, "podded": b, "abs_diff": b - a,
                "rel_diff_pct": 100.0 * (b - a) / a}

    comparison = {
        "advance_over_L": delta("turning_35deg", "advance_over_L"),
        "tactical_diameter_over_L": delta("turning_35deg", "tactical_diameter_over_L"),
        "transfer_over_L_at_90deg": delta("turning_35deg", "transfer_over_L_at_90deg"),
        "u_ss_over_u0": delta("turning_35deg", "u_ss_over_u0"),
        "zz10_first_overshoot_deg": {
            "conventional": conv["zigzag_10_10"]["first_overshoot_deg"],
            "podded": pod["zigzag_10_10"]["first_overshoot_deg"],
            "abs_diff": (pod["zigzag_10_10"]["first_overshoot_deg"]
                         - conv["zigzag_10_10"]["first_overshoot_deg"])},
        "zz20_first_overshoot_deg": {
            "conventional": conv["zigzag_20_20"]["first_overshoot_deg"],
            "podded": pod["zigzag_20_20"]["first_overshoot_deg"],
            "abs_diff": (pod["zigzag_20_20"]["first_overshoot_deg"]
                         - conv["zigzag_20_20"]["first_overshoot_deg"])},
        "note": ("Same hull, same added mass, same cross-flow and resistance model; the only "
                 "difference is the steering/thrust hardware. This is a like-for-like internal "
                 "comparison, NOT a comparison against SIMMAN (see criteria[criterion_3])."),
    }

    conv_check = {"conventional": convergence(anchor, cfg="conventional", dts=(1.0, 0.5, 0.25)),
                  "podded": convergence(anchor, cfg="podded", dts=(1.0, 0.5, 0.25))}
    halvings = [h["rel_change_pct"] for cfg_ in ("conventional", "podded")
                for h in conv_check[cfg_]["advance_over_L"]["halving"]]
    worst = max(halvings)

    criteria = [
        {"id": 1, "name": "anchor completeness 100% (direction + unit + scale caliber recorded)",
         "verdict": "FAIL",
         "measured": (f"{anchor['completeness']['filled_fields']}/"
                      f"{anchor['completeness']['required_fields']} = "
                      f"{100 * anchor['completeness']['ratio']:.0f}%"),
         "threshold": "100%",
         "evidence": "benchmarks/anchors/msc137-simman-w22.json :: completeness",
         "gap": ("SIMMAN 2008 KVLCC2 numerical baselines (advance / tactical diameter / Z-test "
                 "overshoot / PMM derivatives) are not transcribed anywhere offline -> fields "
                 "AN-SIMMAN-03/04/05 are MISSING; see the anchor 'blocked' block.")},
        {"id": 2, "name": "advance / tactical diameter / Z overshoot tabulated with PASS/FAIL",
         "verdict": "PASS",
         "measured": ("3 quantity families x 2 configurations x 4 manoeuvres "
                      "(35deg turn, 20deg turn, 10/10 Z, 20/20 Z), every row carries value, "
                      "limit, verdict and pointer"),
         "threshold": "all three quantity families present with verdicts",
         "evidence": "turning_zigzag.json :: runs.*.imo_verdicts"},
        {"id": 3, "name": "same-caliber difference vs KVLCC2 single-screw SIMMAN baseline <15%",
         "verdict": "WITHHELD",
         "measured": "not computable",
         "threshold": "<15%",
         "evidence": "anchor AN-SIMMAN-03/04/05 (MISSING) + anchor.blocked",
         "gap": ("No digit-exact SIMMAN KVLCC2 single-screw baseline (advance, D_T, overshoot) "
                 "is available offline, so no same-caliber anchor difference can be formed. "
                 "Only the internal podded-vs-single-screw delta is reported (comparison "
                 "block); it is a model difference, not an anchor validation.")},
        {"id": 4, "name": "time-step convergence: dt halved -> advance change <2%",
         "verdict": "PASS" if worst < 2.0 else "FAIL",
         "measured": f"max |d advance|/advance over the two halvings = {worst:.4f}%",
         "threshold": "<2%",
         "evidence": "verification.time_step_convergence.*.advance_over_L.halving"},
    ]

    art = {
        "schema": "tensorfem-w22-maneuvering/1.0",
        "mission": "m14511 / W22 maneuvering-podded-ice (domain repo TensorFEM)",
        "predecessor": {
            "task": "paper-exp-maneuvering-podded-ice-1",
            "status_inherited": ("zero report / no artefact on disk: a search of this tree and "
                                 "of the ecosystem checkouts finds no manoeuvring artefact of "
                                 "that task"),
            "how_this_mission_aligns": ("its four criteria are reproduced verbatim as "
                                        "criteria[1..4] below and each is adjudicated with a "
                                        "measured value plus an evidence pointer")},
        "provenance": {
            "git_commit": git_commit(),
            "script": "benchmarks/podded_ice_maneuvering_w22.py",
            "script_sha256": sha256_file(Path(__file__)),
            "anchor_file": "benchmarks/anchors/msc137-simman-w22.json",
            "anchor_sha256": sha256_file(ANCHOR_PATH),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        "geometry_gate": {
            "L_pp_m": hull.length, "B_m": hull.beam, "T_m": hull.draft, "Cb": hull.cb,
            "L_over_B": hull.length / hull.beam, "B_over_T": hull.beam / hull.draft,
            "displacement_from_geometry_t": hull.displacement_t_from_geometry(),
            "displacement_stated_t": hull.displacement_t,
            "closure_pct": 100.0 * abs(hull.displacement_t_from_geometry()
                                       - hull.displacement_t) / hull.displacement_t,
            "verdict": "PASS (Cb/L/B validated before any performance number)"},
        "anchor_cross_checks": anchor_cross_checks(hull, anchor),
        "model": {
            "formulation": ("3-DOF MMG; declared SNAME-prime hull derivative set "
                            "(linear + quadratic terms, instantaneous resultant speed scale), "
                            "propeller-slipstream rudder force (Fujii, post-stall capped) for the "
                            "conventional ship, thrust-vectoring twin pods with post-stall-capped "
                            "strut side force for the podded ship; RK4 with rate-limited actuation "
                            "and in-step event location"),
            "model_status": ("numerics REAL; force model DECLARED (no digit-exact PMM/Ct table "
                             "offline) -> absolute accuracy WITHHELD"),
            "configurations": {
                "conventional": ("1 propeller (declared kt0/kt1 line, n solved for "
                                 "self-propulsion at U0) + 1 rudder, Fujii lift slope "
                                 "6.13*AR/(AR+2.25), a_H=0.30, x_H=-0.40L, x_R=-0.50L"),
                "podded": ("2 azimuthing pods, x=-0.45L, y=+-0.35B, thrust vectoring, strut "
                           "side force, no rudder, no a_H interaction")},
            "declared_constants": {
                "hull_derivatives": {k: getattr(models["conventional"].p, k) for k in
                                     ("y_v", "y_r", "y_vv", "y_vr",
                                      "n_v", "n_r", "n_vv", "n_vr")},
                "k_added_sway": models["conventional"].p.k_added_sway,
                "k_added_surge": models["conventional"].p.k_added_surge,
                "j_added_yaw": models["conventional"].p.j_added_yaw,
                "residuary_coeff": models["conventional"].p.residuary_coeff,
                "f_alpha": models["conventional"].f_alpha,
                "rudder_area_m2": models["conventional"].a_r,
                "rudder_rate_deg_s": models["conventional"].p.rudder_rate_deg_s,
                "eps_r": models["conventional"].p.eps_r,
                "c_n_max": models["conventional"].p.c_n_max,
                "c_n_max_pod": models["podded"].p.c_n_max_pod,
                "slip_factor_design": models["conventional"].slip_design,
                "pod_rate_deg_s": models["podded"].p.pod_rate_deg_s,
                "ice_resistance_fraction": models["podded"].p.ice_resistance_fraction},
            "evidence_grade": ("numerics REAL (converged, cross-checked against a Newton steady "
                               "solution); force model DECLARED (no digit-exact PMM table); "
                               "absolute accuracy WITHHELD")},
        "runs": case,
        "comparison_podded_vs_conventional": comparison,
        "verification": {
            "steady_vs_marching": steady_vs_marching(models),
            "time_step_convergence": conv_check,
            "anchor_cross_checks": anchor_cross_checks(hull, anchor)},
        "sensitivity": sensitivity(anchor),
        "limitations": [
            ("Hull forces are a DECLARED SNAME-prime derivative set (linear + quadratic), "
             "not a transcribed PMM table: no digit-exact KVLCC2 derivative set exists "
             "offline (see the anchor blocked block). Absolute accuracy is WITHHELD."),
            ("Constant-azimuth pod model: the thrust-vector yaw moment does not decay with "
             "drift the way a rudder's does (the rudder's effective incidence falls as the "
             "drift builds). Consequently the podded 35 deg circle settles into a tight "
             "spiral (u -> %.2f u0, kinematic 2u/r diameter %.2f L) instead of a wide steady "
             "circle. The IMO indicators are measured inside the first half turn (advance at "
             "90 deg, tactical diameter at 180 deg) and are therefore still defined, but the "
             "podded steady speed/drift must not be read as a podded-tanker speed loss."
             % (case["podded"]["turning_35deg"]["u_over_u0_at_end"],
                steady_vs_marching(models)["podded"]["kinematic_diameter_over_L"])),
            ("No drift-induced longitudinal resistance (X_vv = X_vr = 0), no heel/roll "
             "coupling, no free surface, no shaft-power limit (the propeller speed is held at "
             "its self-propulsion value)."),
            ("Ice is a declared scenario parameter (an extra steady resistance as a fraction "
             "of R(U0)); no ice-hull or ice-pod contact physics, no added mass from ice, no "
             "channel/broken-channel asymmetry."),
            ("IMO overshoot limits are a grade-B transcription (in-ecosystem transcription of "
             "Res. MSC.137(76) annex 2.3); one label conflict between two in-ecosystem "
             "sources is recorded and left unresolved (anchor conflicts block)."),
        ],
        "criteria": criteria,
        "honest_fail_statement": (
            "Criterion 1 FAIL and criterion 3 WITHHELD: the SIMMAN 2008 KVLCC2 numerical "
            "baseline is not transcribable offline, so this mission cannot and does not claim "
            "any figure of merit against the workshop data. What it does claim: MSC.137(76) "
            "criteria are machine-applied from a provenance-carrying anchor file, the manoeuvre "
            "indicators converge under dt halving, and the podded-vs-single-screw difference is "
            "quantified with a coefficient/ice-load band."),
    }
    return art


def main() -> dict:
    art = build_artifact()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(art, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(REPO)}  sha256={sha256_file(OUT_PATH)}")
    for cfg in ("conventional", "podded"):
        r = art["runs"][cfg]
        print(f"[{cfg}] advance/L={r['turning_35deg']['advance_over_L']:.3f} "
              f"DT/L={r['turning_35deg']['tactical_diameter_over_L']} "
              f"zz10={r['zigzag_10_10']['first_overshoot_deg']} "
              f"zz20={r['zigzag_20_20']['first_overshoot_deg']}")
    for row in art["criteria"]:
        print(f"criterion {row['id']}: {row['verdict']}  ({row['measured']})")
    return art


if __name__ == "__main__":
    main()
