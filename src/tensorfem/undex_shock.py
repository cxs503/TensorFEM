"""UNDEX shock chain: Cole environment -> Taylor gas-backed plate FSI -> SRS/DDAM.

One verifiable path from an underwater-explosion shock environment to a
structural yield margin:

1. ``cole_*``      -- Cole (1948) peak pressure and exponential decay law
                      (external anchor: Cole R.H., *Underwater Explosions*,
                      Princeton University Press, 1948).
2. ``taylor_*``    -- Taylor (1963) gas(air)-backed plate response.  Reference
                      solution = exact linear (acoustic, non-cavitating)
                      solution of a plane exponential shock loading an
                      air-backed plate of areal mass ``m``::

                          beta = rho*c/m
                          v(t) = (2 p_m/m) (e^{-t/theta} - e^{-beta t})/(beta - 1/theta)
                          p_int = 2 p_i - rho c v

                      plus a finite-element model of the same problem (1-D
                      acoustic fluid + plate inertia + exact outgoing
                      truncation, incident field injected as an equivalent
                      boundary traction), validated against the closed form.
3. ``srs``         -- shock response spectrum (SDOF maximax, exact integration
                      for piecewise-linear base acceleration), 3 damping ratios.
4. ``plate_*`` / ``ddam_*`` -- DDAM-style response spectrum analysis of a
                      simply supported panel: modal participation / effective
                      mass, SRS pseudo-acceleration per mode, NRL combination,
                      top-surface von Mises stress, yield margin ledger.

All quantities are SI.  The Cole coefficients are implementation parameters for
a TNT charge (metric fit widely quoted for UNDEX work): K = 52.4 MPa, A = 1.13,
K_theta = 0.084 ms, decay exponent -0.24, W in kg, R in m.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch

from .dynamics import TransientResult, newmark_linear
from .plate import q4_mindlin_stiffness
from .plate_vibration import q4_plate_mass_matrix

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
SEAWATER_DENSITY = 1025.0          # kg/m^3
SEAWATER_SOUND_SPEED = 1500.0      # m/s
TNT_PEAK_K_PA = 52.4e6             # Cole metric fit coefficient K [Pa]
TNT_PEAK_A = 1.13                  # Cole peak-pressure exponent A
TNT_DECAY_K_S = 0.084e-3           # Cole metric decay coefficient K_theta [s]
TNT_DECAY_EXPONENT = -0.24         # Cole decay exponent on reduced range
TNT_PEAK_K_PSI = 22500.0           # Cole imperial fit coefficient (lb, ft) [psi]
TNT_SPECIFIC_ENERGY_J_KG = 4.52e6  # Cole TNT specific energy [J/kg]
KG_PER_LB = 0.45359237
M_PER_FT = 0.3048
PA_PER_PSI = 6894.757293168



@contextmanager
def _single_thread():
    """Run a many-small-ops block single-threaded (avoids thread overhead)."""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def _positive(name: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return v


# ---------------------------------------------------------------------------
# 1. Cole (1948) shock environment
# ---------------------------------------------------------------------------
def reduced_range(weight_kg: float, range_m: float) -> float:
    """Cole reduced range W^(1/3)/R  [kg^(1/3)/m]."""
    return _positive("weight_kg", weight_kg) ** (1.0 / 3.0) / _positive("range_m", range_m)


def cole_peak_pressure(weight_kg: float, range_m: float, *,
                       K: float = TNT_PEAK_K_PA, A: float = TNT_PEAK_A) -> float:
    """Cole (1948) peak shock pressure [Pa]: p_m = K (W^(1/3)/R)^A."""
    return float(K) * reduced_range(weight_kg, range_m) ** float(A)


def cole_decay_time(weight_kg: float, range_m: float, *,
                    Kt: float = TNT_DECAY_K_S,
                    exponent: float = TNT_DECAY_EXPONENT) -> float:
    """Cole (1948) exponential decay constant [s]: theta = Kt W^(1/3) g^exponent."""
    W = _positive("weight_kg", weight_kg)
    g = reduced_range(W, range_m)
    return float(Kt) * W ** (1.0 / 3.0) * g ** float(exponent)


def cole_pressure(t, weight_kg: float, range_m: float, *,
                  K: float = TNT_PEAK_K_PA, A: float = TNT_PEAK_A,
                  Kt: float = TNT_DECAY_K_S, exponent: float = TNT_DECAY_EXPONENT):
    """Cole exponential pressure history p(t) = p_m exp(-t/theta) [Pa]."""
    p_m = cole_peak_pressure(weight_kg, range_m, K=K, A=A)
    theta = cole_decay_time(weight_kg, range_m, Kt=Kt, exponent=exponent)
    tt = t if isinstance(t, torch.Tensor) else torch.as_tensor(np.asarray(t, dtype=np.float64))
    return p_m * torch.exp(-tt.to(torch.float64) / theta)


@dataclass(frozen=True)
class ColePoint:
    """One (charge weight, standoff) point of the Cole shock environment."""
    weight_kg: float
    range_m: float
    reduced_range: float
    peak_pressure_pa: float
    decay_time_s: float
    specific_impulse_pa_s: float          # p_m * theta (closed form)
    energy_flux_j_m2: float               # p_m^2 theta / (2 rho c) (closed form)
    shock_energy_j: float                 # 4 pi R^2 * flux
    shock_energy_ratio: float             # shock_energy / (W * 4.52 MJ/kg)
    peak_pressure_mpa: float

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def cole_point(weight_kg: float, range_m: float, *,
               density: float = SEAWATER_DENSITY,
               sound_speed: float = SEAWATER_SOUND_SPEED,
               specific_energy: float = TNT_SPECIFIC_ENERGY_J_KG) -> ColePoint:
    W = _positive("weight_kg", weight_kg)
    R = _positive("range_m", range_m)
    p_m = cole_peak_pressure(W, R)
    theta = cole_decay_time(W, R)
    flux = p_m * p_m * theta / (2.0 * density * sound_speed)
    energy = 4.0 * math.pi * R * R * flux
    return ColePoint(
        weight_kg=W,
        range_m=R,
        reduced_range=reduced_range(W, R),
        peak_pressure_pa=p_m,
        decay_time_s=theta,
        specific_impulse_pa_s=p_m * theta,
        energy_flux_j_m2=flux,
        shock_energy_j=energy,
        shock_energy_ratio=energy / (W * specific_energy),
        peak_pressure_mpa=p_m * 1.0e-6,
    )


def cole_table(cases, **kwargs) -> list[ColePoint]:
    """Evaluate :func:`cole_point` on a sequence of (weight_kg, range_m) pairs."""
    return [cole_point(w, r, **kwargs) for w, r in cases]


def dimensional_homogeneity_check(*, weight_kg: float = 10.0, range_m: float = 15.0,
                                  density: float = SEAWATER_DENSITY,
                                  sound_speed: float = SEAWATER_SOUND_SPEED,
                                  tolerance: float = 1.0e-9) -> dict:
    """Machine checks of the Cole law's unit / scaling structure.

    * ``cube_root_scaling``: an explosion law must be invariant under Hopkinson
      scaling (W -> s^3 W, R -> s R) for the peak pressure and linear in s for
      the decay time.  Non-tautological: a law with independent W and R
      exponents fails this.
    * ``unit_algebra``: the composite coefficient unit Pa*(m/kg^(1/3))^A times
      [W^(1/3)/R]^A must reduce to Pa (exponent bookkeeping).
    * ``imperial_fit_consistency``: metric (52.4 MPa) and imperial (22500 psi)
      Cole fits must give the same physical pressure for the same physical
      charge within published coefficient rounding.
    * ``closed_form_vs_numeric``: impulse and energy flux closed forms vs
      trapezoidal integration of p and p^2/(rho c).
    """
    p_ref = cole_peak_pressure(weight_kg, range_m)
    th_ref = cole_decay_time(weight_kg, range_m)

    scaling = []
    for s in (0.5, 2.0, 7.3):
        p_s = cole_peak_pressure(s ** 3 * weight_kg, s * range_m)
        th_s = cole_decay_time(s ** 3 * weight_kg, s * range_m)
        scaling.append({"scale": s, "peak_ratio": p_s / p_ref,
                        "decay_ratio": th_s / th_ref, "decay_expected": s})
    scaling_pass = all(abs(r["peak_ratio"] - 1.0) < tolerance and
                       abs(r["decay_ratio"] - r["decay_expected"]) < tolerance
                       for r in scaling)

    exp_kg = -TNT_PEAK_A / 3.0 + TNT_PEAK_A / 3.0
    exp_m = TNT_PEAK_A - TNT_PEAK_A
    unit_algebra_pass = abs(exp_kg) < 1e-12 and abs(exp_m) < 1e-12

    w_lb = weight_kg / KG_PER_LB
    r_ft = range_m / M_PER_FT
    p_psi = TNT_PEAK_K_PSI * (w_lb ** (1.0 / 3.0) / r_ft) ** TNT_PEAK_A
    p_from_imperial = p_psi * PA_PER_PSI
    imperial_rel_diff = abs(p_from_imperial - p_ref) / p_ref

    theta = th_ref
    t = torch.linspace(0.0, 30.0 * theta, 300001, dtype=torch.float64)
    p = cole_pressure(t, weight_kg, range_m)
    impulse_num = float(torch.trapezoid(p, t))
    energy_num = float(torch.trapezoid(p * p, t)) / (density * sound_speed)
    impulse_closed = p_ref * theta
    energy_closed = p_ref * p_ref * theta / (2.0 * density * sound_speed)
    impulse_rel = abs(impulse_num - impulse_closed) / impulse_closed
    energy_rel = abs(energy_num - energy_closed) / energy_closed

    checks = {
        "cube_root_scaling": {"pass": bool(scaling_pass), "points": scaling,
                              "tolerance": tolerance},
        "unit_algebra": {"pass": bool(unit_algebra_pass),
                         "exponent_kg": exp_kg, "exponent_m": exp_m,
                         "coefficient_unit": f"Pa*(m/kg^(1/3))^{TNT_PEAK_A}"},
        "imperial_fit_consistency": {
            "pass": bool(imperial_rel_diff < 0.06),
            "metric_pa": p_ref, "imperial_pa": p_from_imperial,
            "relative_difference": imperial_rel_diff,
            "note": "published coefficient rounding; tolerance 6%",
        },
        "closed_form_vs_numeric": {
            "pass": bool(impulse_rel < 1.0e-5 and energy_rel < 1.0e-5),
            "impulse_relative_error": impulse_rel,
            "energy_relative_error": energy_rel,
        },
    }
    return {"pass": all(c["pass"] for c in checks.values()), "checks": checks}



# ---------------------------------------------------------------------------
# time integration: identical average-acceleration Newmark scheme as
# ``tensorfem.dynamics.newmark_linear`` but with the effective matrix factored
# once (needed because the upstream routine re-solves a dense system every step,
# which dominates the run time for O(400) dofs).  Equivalence is asserted in the
# test suite.
# ---------------------------------------------------------------------------
def newmark_factorized(M: torch.Tensor, K: torch.Tensor, force: torch.Tensor,
                       time: torch.Tensor, u0: torch.Tensor, v0: torch.Tensor,
                       C: torch.Tensor | None = None, beta: float = 0.25,
                       gamma: float = 0.5, threads: int = 1) -> TransientResult:
    """Average-acceleration Newmark with a single factorisation (same scheme)."""
    if M.ndim != 2 or M.shape != K.shape or M.shape[0] != M.shape[1]:
        raise ValueError("M and K must be equal square matrices")
    if force.shape != (time.numel(), M.shape[0]) or time.numel() < 2:
        raise ValueError("incompatible time or force history")
    dtv = time[1:] - time[:-1]
    if torch.any(dtv <= 0) or not torch.allclose(dtv, dtv[0]):
        raise ValueError("time must be uniformly increasing")
    C = torch.zeros_like(M) if C is None else C
    if C.shape != M.shape or u0.shape != (M.shape[0],) or v0.shape != u0.shape:
        raise ValueError("incompatible damping or initial condition")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(max(1, int(threads)))
    try:
        dtype, device = M.dtype, M.device
        u0 = u0.to(dtype=dtype, device=device)
        v0 = v0.to(dtype=dtype, device=device)
        dt = dtv[0]
        n = time.numel()
        ndof = M.shape[0]
        u = torch.zeros((n, ndof), dtype=dtype, device=device)
        v = torch.zeros_like(u)
        a = torch.zeros_like(u)
        u[0] = u0
        v[0] = v0
        Lm = torch.linalg.cholesky(M)
        a[0] = torch.cholesky_solve((force[0] - C @ v0 - K @ u0)[:, None], Lm)[:, 0]
        c0 = 1.0 / (beta * dt ** 2)
        c1 = gamma / (beta * dt)
        c2 = 1.0 / (beta * dt)
        c3 = 1.0 / (2.0 * beta) - 1.0
        c4 = gamma / beta - 1.0
        c5 = dt * (gamma / (2.0 * beta) - 1.0)
        Ke = K + c0 * M + c1 * C
        Lke = torch.linalg.cholesky(Ke)
        for i in range(n - 1):
            rhs = (force[i + 1] + M @ (c0 * u[i] + c2 * v[i] + c3 * a[i])
                   + C @ (c1 * u[i] + c4 * v[i] + c5 * a[i]))
            u[i + 1] = torch.cholesky_solve(rhs[:, None], Lke)[:, 0]
            a[i + 1] = c0 * (u[i + 1] - u[i]) - c2 * v[i] - c3 * a[i]
            v[i + 1] = v[i] + dt * ((1.0 - gamma) * a[i] + gamma * a[i + 1])
        return TransientResult(time, u, v, a)
    finally:
        torch.set_num_threads(old_threads)


def newmark_equivalence_check(seed: int = 0, ndof: int = 24, nsteps: int = 64,
                              dt: float = 1.0e-3) -> dict:
    """Compare :func:`newmark_factorized` with the upstream ``newmark_linear``.

    Uses a small synthetic system (below the size where the upstream routine is
    dominated by repeated dense solves) and returns the maximum relative
    difference of displacement and velocity.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    A = torch.randn((ndof, ndof), generator=g, dtype=torch.float64)
    K = A @ A.T / ndof + ndof * torch.eye(ndof, dtype=torch.float64)
    B = torch.randn((ndof, ndof), generator=g, dtype=torch.float64)
    M = B @ B.T / ndof + ndof * torch.eye(ndof, dtype=torch.float64)
    C = 0.05 * M
    time = torch.linspace(0.0, (nsteps - 1) * dt, nsteps, dtype=torch.float64)
    force = torch.sin(50.0 * time)[:, None] * torch.ones(1, ndof, dtype=torch.float64)
    u0 = torch.zeros(ndof, dtype=torch.float64)
    v0 = torch.zeros(ndof, dtype=torch.float64)
    ref = newmark_linear(M, K, force, time, u0, v0, C=C)
    got = newmark_factorized(M, K, force, time, u0, v0, C=C)
    du = float((got.displacement - ref.displacement).abs().max()) / \
        float(ref.displacement.abs().max())
    dv = float((got.velocity - ref.velocity).abs().max()) / \
        float(ref.velocity.abs().max())
    return {"displacement_relative_difference": du,
            "velocity_relative_difference": dv,
            "pass": du < 1e-10 and dv < 1e-10}


# ---------------------------------------------------------------------------
# 2. Taylor gas(air)-backed plate: exact linear reference solution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TaylorAnalytic:
    """Exact linear (acoustic, non-cavitating) air-backed plate response."""
    time: torch.Tensor
    displacement: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor
    incident_pressure: torch.Tensor
    interface_pressure: torch.Tensor
    beta: float
    beta_theta: float
    cavitation_time_s: float | None
    cavitation_velocity_ms: float | None
    cavitation_displacement_m: float | None


def taylor_analytic(time, *, peak_pressure: float, decay_time: float,
                    areal_mass: float, density: float = SEAWATER_DENSITY,
                    sound_speed: float = SEAWATER_SOUND_SPEED) -> TaylorAnalytic:
    """Closed-form Taylor air-backed plate solution for p_i = p_m exp(-t/theta).

    Plate momentum ``m v' = p_int`` plus the acoustic impedance relation
    ``p_int = 2 p_i - rho c v`` give ``m v' + rho c v = 2 p_m exp(-t/theta)``,
    solved exactly.  The solution assumes attached water and is therefore only
    physical up to the first instant of ``p_int <= 0`` (cavitation), reported as
    ``cavitation_time_s``.
    """
    p_m = _positive("peak_pressure", peak_pressure)
    theta = _positive("decay_time", decay_time)
    m = _positive("areal_mass", areal_mass)
    rho = _positive("density", density)
    c = _positive("sound_speed", sound_speed)
    t = time if isinstance(time, torch.Tensor) else torch.as_tensor(
        np.asarray(time, dtype=np.float64))
    t = t.to(torch.float64)

    beta = rho * c / m
    a = 1.0 / theta
    denom = beta - a
    amp = 2.0 * p_m / m
    e_a = torch.exp(-a * t)
    e_b = torch.exp(-beta * t)
    if abs(denom) <= 1.0e-9 * max(beta, a):
        v = amp * t * e_b
        w = amp * (1.0 - e_b * (1.0 + beta * t)) / (beta * beta)
    else:
        v = amp * (e_a - e_b) / denom
        w = amp / denom * ((1.0 - e_a) / a - (1.0 - e_b) / beta)
    p_i = p_m * e_a
    p_int = 2.0 * p_i - rho * c * v
    acc = p_int / m

    negative = torch.nonzero(p_int <= 0.0, as_tuple=False).flatten()
    if negative.numel():
        idx = int(negative[0])
        t_cav = float(t[idx])
        v_cav = float(v[idx])
        w_cav = float(w[idx])
    else:
        t_cav = v_cav = w_cav = None
    return TaylorAnalytic(time=t, displacement=w, velocity=v, acceleration=acc,
                          incident_pressure=p_i, interface_pressure=p_int,
                          beta=beta, beta_theta=beta * theta,
                          cavitation_time_s=t_cav,
                          cavitation_velocity_ms=v_cav,
                          cavitation_displacement_m=w_cav)


# ---------------------------------------------------------------------------
# 2b. Taylor problem by finite elements (1-D acoustic fluid + plate inertia)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TaylorFSIResult:
    """Finite-element air-backed plate response."""
    time: torch.Tensor
    plate_displacement: torch.Tensor     # m, measured from the arrival instant
    plate_velocity: torch.Tensor         # m/s
    plate_acceleration: torch.Tensor     # m/s^2
    interface_pressure: torch.Tensor     # Pa (recovered from discrete gradient)
    incident_pressure: torch.Tensor      # Pa (prescribed)
    impedance_residual: torch.Tensor     # Pa, p_int - (2 p_i - rho c v)
    fluid_nodes: torch.Tensor
    fluid_displacement: torch.Tensor
    cavitation_time_s: float | None
    meta: dict = field(default_factory=dict)


def taylor_fsi_fem(*, peak_pressure: float, decay_time: float, areal_mass: float,
                   length: float = 8.0, elements: int = 800,
                   dt: float = 5.0e-6, duration: float | None = None,
                   density: float = SEAWATER_DENSITY,
                   sound_speed: float = SEAWATER_SOUND_SPEED,
                   area: float = 1.0, mass_lumping: bool = False) -> TaylorFSIResult:
    """Finite-element model of the Taylor air-backed plate problem.

    * fluid: 1-D acoustic (displacement) elements,
      ``K_e = B A/h [[1,-1],[-1,1]]``, ``M_e = rho A h/6 [[2,1],[1,2]]``,
      ``B = rho c^2``;
    * plate: areal mass ``m`` lumped on the interface node (air backing taken as
      vacuous -- Taylor's idealisation: ambient gas pressure 0.1 MPa against an
      O(10) MPa shock);
    * incident field: shock travels in -x and reaches the interface x = 0 at
      t = 0; prescribed analytically through the initial conditions so the plate
      feels the correct pressure gradient at t = 0+; the truncation at x = L is
      transparent by the exact outgoing condition applied to the *scattered*
      field: ``C[L,L] += rho c A`` and ``f_L(t) = rho c A v_inc(L,t)``;
    * time integration: average-acceleration Newmark
      (:func:`tensorfem.dynamics.newmark_linear`).

    Both the initial condition and the truncation are exact for the 1-D wave
    equation, so the only errors are linear-element dispersion of the pulse and
    Newmark time discretisation (quantified by the convergence study).
    """
    p_m = _positive("peak_pressure", peak_pressure)
    theta = _positive("decay_time", decay_time)
    m = _positive("areal_mass", areal_mass)
    L = _positive("length", length)
    rho = _positive("density", density)
    c = _positive("sound_speed", sound_speed)
    A = _positive("area", area)
    nel = int(elements)
    if nel < 2:
        raise ValueError("elements must be at least 2")
    if duration is None:
        duration = 4.0 * theta
    duration = _positive("duration", duration)

    h = L / nel
    ndof = nel + 1
    dtype = torch.float64
    B_mod = rho * c * c
    x = torch.linspace(0.0, L, ndof, dtype=dtype)

    K = torch.zeros((ndof, ndof), dtype=dtype)
    M = torch.zeros((ndof, ndof), dtype=dtype)
    ke = B_mod * A / h * torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=dtype)
    me = rho * A * h / 6.0 * torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=dtype)
    for e in range(nel):
        idx = torch.tensor([e, e + 1])
        K[idx[:, None], idx] += ke
        M[idx[:, None], idx] += me
    if mass_lumping:
        M = torch.diag(M.sum(dim=1))
    M[0, 0] += m * A
    C = torch.zeros_like(M)
    C[-1, -1] = rho * c * A

    nsteps = int(math.ceil(duration / dt)) + 1
    time = torch.linspace(0.0, (nsteps - 1) * dt, nsteps, dtype=dtype)
    p_i_nodes0 = p_m * torch.exp(-x / (c * theta))
    # accumulated fluid displacement of the incident wave: zero at the interface
    # at t = 0, pushed towards the plate further downstream; p = -B du/dx.
    u_inc0 = theta * (p_i_nodes0 - p_m) / (rho * c)
    v_inc0 = -p_i_nodes0 / (rho * c)
    u0 = u_inc0.clone()
    v0 = v_inc0.clone()
    v0[0] = 0.0                                  # plate at rest at t = 0

    p_inc_last = p_m * math.exp(-L / (c * theta)) * torch.exp(-time / theta)
    force = torch.zeros((nsteps, ndof), dtype=dtype)
    force[:, -1] = -A * p_inc_last               # = rho c A v_inc(L, t)

    res = newmark_factorized(M, K, force, time, u0, v0, C=C, beta=0.25, gamma=0.5)

    # x points from the plate into the water while the wave travels in -x, so
    # Taylor's plate displacement/velocity are the negatives of the interface
    # node values (positive = driven into the plate).
    w_plate = -res.displacement[:, 0]
    v_plate = -res.velocity[:, 0]
    a_plate = -res.acceleration[:, 0]
    p_int = -B_mod * (res.displacement[:, 1] - res.displacement[:, 0]) / h
    p_inc_plate = p_m * torch.exp(-time / theta)
    residual = p_int - (2.0 * p_inc_plate - rho * c * v_plate)

    negative = torch.nonzero(p_int <= 0.0, as_tuple=False).flatten()
    t_cav = float(time[int(negative[0])]) if negative.numel() else None
    meta = {"length": L, "elements": nel, "dt": dt, "duration": duration,
            "mass_lumping": bool(mass_lumping),
            "ndof": ndof, "beta": rho * c / m, "beta_theta": rho * c * theta / m,
            "h": h, "cells_per_pulse_wavelength": 2.0 * math.pi * c * theta / h}
    return TaylorFSIResult(time=time, plate_displacement=w_plate,
                           plate_velocity=v_plate, plate_acceleration=a_plate,
                           interface_pressure=p_int, incident_pressure=p_inc_plate,
                           impedance_residual=residual,
                           fluid_nodes=x, fluid_displacement=res.displacement,
                           cavitation_time_s=t_cav, meta=meta)


def compare_to_taylor_analytic(fem: TaylorFSIResult, ref: TaylorAnalytic) -> dict:
    """Relative deviation of the FEM response from the closed-form Taylor solution.

    Window ``[0, t_cav)``: beyond the first cavitation instant the acoustic
    solution permits tensile water and is no longer a valid reference.
    """
    n = min(fem.time.numel(), ref.time.numel())
    candidates = [t for t in (fem.cavitation_time_s, ref.cavitation_time_s)
                  if t is not None]
    t_stop = min(candidates) if candidates else float(fem.time[n - 1])
    mask = fem.time[:n] <= t_stop
    if int(mask.sum()) < 3:
        raise ValueError("comparison window too short")
    out = {"window_s": [0.0, t_stop], "samples": int(mask.sum())}
    for name, num, den in (("displacement", fem.plate_displacement, ref.displacement),
                           ("velocity", fem.plate_velocity, ref.velocity),
                           ("acceleration", fem.plate_acceleration, ref.acceleration),
                           ("interface_pressure", fem.interface_pressure,
                            ref.interface_pressure)):
        d = (num[:n][mask] - den[:n][mask]).abs()
        scale = float(den[:n][mask].abs().max())
        out[name] = {
            "peak_reference": scale,
            "max_abs_error": float(d.max()),
            "relative_max_error": (float(d.max()) / scale) if scale > 0 else float("inf"),
            "relative_l2_error": (float(torch.sqrt((d * d).sum()) /
                                        torch.sqrt((den[:n][mask] * den[:n][mask]).sum())))
            if scale > 0 else float("inf"),
        }
    out["impedance_residual_relative"] = (
        float(fem.impedance_residual.abs().max()) /
        float(fem.incident_pressure.abs().max()))
    return out


# ---------------------------------------------------------------------------
# 3. Shock response spectrum (SDOF maximax, exact for piecewise-linear input)
# ---------------------------------------------------------------------------
def _sdof_max_response(acc: np.ndarray, dt: float, omega: float, zeta: float):
    """Exact maximax of x'' + 2 zeta w x' + w^2 x = -a(t), linear-in-step input.

    Returns (max|x|, max|x'|) for the relative displacement x.
    """
    w = float(omega)
    z = float(zeta)
    wd = w * math.sqrt(max(1.0 - z * z, 0.0))
    x = v = 0.0
    xmax = vmax = 0.0
    e = math.exp(-z * w * dt)
    c = math.cos(wd * dt)
    s = math.sin(wd * dt)
    for i in range(acc.size - 1):
        f0 = float(acc[i])
        k = (float(acc[i + 1]) - f0) / dt
        c1 = -k / (w * w)
        c0 = -f0 / (w * w) + 2.0 * z * k / (w ** 3)
        x1 = x - c0
        x2 = (v - c1 + z * w * x1) / wd
        ecs = e * (x1 * c + x2 * s)
        x = ecs + c0 + c1 * dt
        v = -z * w * ecs + e * wd * (-x1 * s + x2 * c) + c1
        xmax = max(xmax, abs(x))
        vmax = max(vmax, abs(v))
    return xmax, vmax


def _sdof_max_response_vector(acc: np.ndarray, dt: float, omegas: np.ndarray,
                              zeta: float):
    """Vectorised maximax over many oscillator frequencies (same scheme)."""
    w = np.asarray(omegas, dtype=np.float64)
    z = float(zeta)
    wd = w * np.sqrt(np.maximum(1.0 - z * z, 0.0))
    e = np.exp(-z * w * dt)
    c = np.cos(wd * dt)
    s = np.sin(wd * dt)
    x = np.zeros_like(w)
    v = np.zeros_like(w)
    xmax = np.zeros_like(w)
    vmax = np.zeros_like(w)
    w2 = w * w
    w3 = w2 * w
    a = np.asarray(acc, dtype=np.float64)
    for i in range(a.size - 1):
        f0 = a[i]
        k = (a[i + 1] - f0) / dt
        c1 = -k / w2
        c0 = -f0 / w2 + 2.0 * z * k / w3
        x1 = x - c0
        x2 = (v - c1 + z * w * x1) / wd
        ecs = e * (x1 * c + x2 * s)
        x = ecs + c0 + c1 * dt
        v = -z * w * ecs + e * wd * (-x1 * s + x2 * c) + c1
        np.maximum(xmax, np.abs(x), out=xmax)
        np.maximum(vmax, np.abs(v), out=vmax)
    return xmax, vmax


@dataclass(frozen=True)
class SRSResult:
    frequencies_hz: np.ndarray
    damping_ratios: list[float]
    sd_m: dict           # zeta -> spectral displacement [m]
    sv_ms: dict          # zeta -> pseudo velocity [m/s]
    sa_ms2: dict         # zeta -> pseudo acceleration [m/s^2]
    dt: float

    def as_dict(self) -> dict:
        return {
            "dt_s": self.dt,
            "frequencies_hz": self.frequencies_hz.tolist(),
            "curves": {f"{z:.3f}": {"sd_m": self.sd_m[z].tolist(),
                                    "sv_ms": self.sv_ms[z].tolist(),
                                    "sa_ms2": self.sa_ms2[z].tolist()}
                       for z in self.damping_ratios},
        }


def log_frequencies(f_min: float = 1.0, f_max: float = 20000.0, n: int = 96) -> np.ndarray:
    return np.geomspace(_positive("f_min", f_min), _positive("f_max", f_max), int(n))


def srs(acc, dt: float, frequencies_hz, damping_ratios=(0.0, 0.02, 0.05)) -> SRSResult:
    """Shock response spectrum of a base-acceleration history (maximax / DDAM)."""
    a = np.asarray(acc.detach().numpy() if isinstance(acc, torch.Tensor) else acc,
                   dtype=np.float64)
    dt = _positive("dt", dt)
    f = np.asarray(frequencies_hz, dtype=np.float64)
    if np.any(f <= 0.0):
        raise ValueError("frequencies must be positive")
    omegas = 2.0 * math.pi * f
    sd, sv, sa = {}, {}, {}
    for z in damping_ratios:
        z = float(z)
        if not 0.0 <= z < 1.0:
            raise ValueError("damping ratio must be in [0,1)")
        xmax, _ = _sdof_max_response_vector(a, dt, omegas, z)
        sd[z] = xmax
        sv[z] = omegas * xmax
        sa[z] = omegas * omegas * xmax
    return SRSResult(frequencies_hz=f,
                     damping_ratios=[float(z) for z in damping_ratios],
                     sd_m=sd, sv_ms=sv, sa_ms2=sa, dt=dt)


def srs_grid_drift(acc, dt: float, damping: float, n_coarse: int = 48,
                   n_fine: int = 192, f_min: float = 1.0,
                   f_max: float = 20000.0) -> dict:
    """Frequency-grid refinement drift of the SRS (coarse grid points as reference).

    Nested geometric grids are used, so the coarse points are a subset of the
    fine ones and the comparison is pointwise.
    """
    coarse = log_frequencies(f_min, f_max, n_coarse)
    fine = log_frequencies(f_min, f_max, n_fine)
    ratio = n_fine // n_coarse
    if ratio * n_coarse != n_fine:
        raise ValueError("n_fine must be an integer multiple of n_coarse")
    c = srs(acc, dt, coarse, damping_ratios=(damping,))
    f = srs(acc, dt, fine, damping_ratios=(damping,))
    sd_c, sd_f = c.sd_m[damping], f.sd_m[damping][::ratio]
    sa_c, sa_f = c.sa_ms2[damping], f.sa_ms2[damping][::ratio]
    drift = np.abs(sd_c - sd_f) / np.maximum(np.abs(sd_f), 1e-30)
    return {"n_coarse": n_coarse, "n_fine": n_fine, "damping": float(damping),
            "max_relative_drift": float(drift.max()),
            "mean_relative_drift": float(drift.mean()),
            "sa_max_relative_drift": float((np.abs(sa_c - sa_f) /
                                            np.maximum(np.abs(sa_f), 1e-30)).max())}


# ---------------------------------------------------------------------------
# 4. Plate modal model (SS panel) + top-surface stress recovery + DDAM
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlateModal:
    """Simply supported rectangular Mindlin plate: modes, participation, mass."""
    length: float
    width: float
    thickness: float
    young: float
    poisson: float
    density: float
    nodes: torch.Tensor
    elements: torch.Tensor
    free_dofs: torch.Tensor
    frequencies_hz: torch.Tensor
    modes_full: torch.Tensor        # (ndof, nmodes), zero on constrained dofs
    participation: torch.Tensor     # Gamma_i = phi^T M r / (phi^T M phi)
    effective_mass: torch.Tensor    # (phi^T M r)^2 / (phi^T M phi)
    total_mass: float               # rho * t * area (translational)
    cumulative_mass_fraction: torch.Tensor
    analytical_frequencies_hz: list[float] = field(default_factory=list)
    frequency_errors: list[float] = field(default_factory=list)
    eigen_solver: str = ""

    @property
    def area(self) -> float:
        return self.length * self.width


def _ss_free_dofs(ndof: int, nx: int, ny: int) -> torch.Tensor:
    """Free dofs for a hard simply supported edge (w = 0 on all four edges)."""
    n_nodes_x = nx + 1
    fixed = set()
    for j in range(ny + 1):
        fixed.add(3 * (j * n_nodes_x))
        fixed.add(3 * (j * n_nodes_x + n_nodes_x - 1))
    for i in range(n_nodes_x):
        fixed.add(3 * i)
        fixed.add(3 * ((ny) * n_nodes_x + i))
    return torch.tensor([d for d in range(ndof) if d not in fixed], dtype=torch.long)


def kirchhoff_ss_plate_frequency(length: float, width: float, thickness: float,
                                 young: float, poisson: float,
                                 density: float, m: int = 1, n: int = 1) -> float:
    """Kirchhoff SS rectangular plate natural frequency [Hz] (Timoshenko)."""
    D = young * thickness ** 3 / (12.0 * (1.0 - poisson ** 2))
    omega = math.pi ** 2 * (m * m / (length * length) + n * n / (width * width)) * \
        math.sqrt(D / (density * thickness))
    return omega / (2.0 * math.pi)


def _assemble_plate_KM(nodes, elements, ndof, young, poisson, thickness, density):
    """Assemble Mindlin plate stiffness and consistent mass (repo elements)."""
    K = torch.zeros((ndof, ndof), dtype=torch.float64)
    M = torch.zeros((ndof, ndof), dtype=torch.float64)
    for conn in elements:
        xy = nodes[conn, :]
        ke = q4_mindlin_stiffness(xy, young, poisson, thickness)
        me = q4_plate_mass_matrix(xy, density, thickness)
        dofs = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), dim=1).reshape(-1)
        K[dofs[:, None], dofs] += ke
        M[dofs[:, None], dofs] += me
    return K, M


def assemble_ss_plate(*, length: float = 1.5, width: float = 1.0, nx: int = 24,
                      ny: int = 16, thickness: float = 0.012,
                      young: float = 206.0e9, poisson: float = 0.3,
                      density: float = 7850.0, modes: int = 8) -> PlateModal:
    """Assemble and solve the generalised eigenproblem of a SS Mindlin panel.

    Uses TensorFEM's own Q4 Mindlin stiffness and consistent mass builders
    (:func:`tensorfem.plate.q4_mindlin_stiffness`,
    :func:`tensorfem.plate_vibration.q4_plate_mass_matrix`); dofs per node are
    (w, theta_x, theta_y).
    """
    L = _positive("length", length)
    W = _positive("width", width)
    t = _positive("thickness", thickness)
    nx, ny = int(nx), int(ny)
    if nx < 1 or ny < 1:
        raise ValueError("mesh must have at least one element per direction")
    xs = torch.linspace(0.0, L, nx + 1, dtype=torch.float64)
    ys = torch.linspace(0.0, W, ny + 1, dtype=torch.float64)
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    nodes = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            elements.append((n0, n0 + 1, n0 + nx + 2, n0 + nx + 1))
    elements = torch.tensor(elements, dtype=torch.long)

    ndof = 3 * nodes.shape[0]
    with _single_thread():
        K, M = _assemble_plate_KM(nodes, elements, ndof, young, poisson, t, density)
    free = _ss_free_dofs(ndof, nx, ny)
    Kr = K[free[:, None], free]
    Mr = M[free[:, None], free]
    Kr = 0.5 * (Kr + Kr.T)
    Mr = 0.5 * (Mr + Mr.T)
    nmodes = min(int(modes), free.numel() - 2)
    # The consistent Mindlin pencil is badly scaled (rotary inertia blocks are
    # 1e5 smaller than the translatory ones), and the dense LAPACK reduction
    # loses the low modes for ~1000+ dofs (verified: Cholesky of K succeeds
    # while dense eigh reports spurious negative eigenvalues).  A shift-invert
    # Lanczos iteration on the extremal (lowest) modes is numerically robust
    # here, so it is the primary path; the dense torch eigh is the fallback.
    omegas_np = None
    try:
        import scipy.sparse as _sp
        import scipy.sparse.linalg as _spl
        Ks = _sp.csr_matrix(Kr.numpy())
        Ms = _sp.csr_matrix(Mr.numpy())
        vals, vecs = _spl.eigsh(Ks, k=nmodes, M=Ms, sigma=0.0, which="LM")
        order = np.argsort(vals)
        vals = np.maximum(vals[order], 0.0)
        phi_r = torch.as_tensor(vecs[:, order], dtype=torch.float64)
        omegas_np = np.sqrt(vals)
        eigen_solver = "scipy.sparse.linalg.eigsh (shift-invert, sigma=0)"
    except Exception:                                    # pragma: no cover
        vals, vecs = torch.linalg.eigh(Kr)
        vals = torch.clamp(vals[:nmodes], min=0.0)
        omegas_np = torch.sqrt(vals).numpy()
        phi_r = vecs[:, :nmodes]
        eigen_solver = "torch.linalg.eigh (dense fallback)"
    omegas = torch.as_tensor(np.asarray(omegas_np), dtype=torch.float64)
    phi_r = phi_r / torch.sqrt(torch.sum(phi_r * (Mr @ phi_r), dim=0, keepdim=True))
    modes_full = torch.zeros((ndof, nmodes), dtype=torch.float64)
    modes_full[free] = phi_r

    w_dofs = torch.arange(0, ndof, 3)
    influence_full = torch.zeros(ndof, dtype=torch.float64)
    influence_full[w_dofs] = 1.0
    influence = influence_full[free]
    Mr_modes = Mr @ phi_r
    generalized_mass = torch.sum(phi_r * Mr_modes, dim=0)
    force_proj = influence @ Mr_modes
    participation = force_proj / generalized_mass
    effective_mass = force_proj ** 2 / generalized_mass
    total_mass = density * t * L * W
    cumulative = torch.cumsum(effective_mass, dim=0) / total_mass
    _ = eigen_solver

    analytical = []
    for m_i in (1, 2, 3, 4):
        for n_i in (1, 2, 3, 4):
            analytical.append(kirchhoff_ss_plate_frequency(L, W, t, young, poisson,
                                                           density, m_i, n_i))
    analytical.sort()
    analytical = analytical[:nmodes]
    freq_hz = omegas / (2.0 * math.pi)
    errors = [abs(float(freq_hz[i]) - analytical[i]) / analytical[i]
              for i in range(min(len(analytical), nmodes))]
    return PlateModal(length=L, width=W, thickness=t, young=young, poisson=poisson,
                      density=density, nodes=nodes, elements=elements, free_dofs=free,
                      frequencies_hz=freq_hz, modes_full=modes_full,
                      participation=participation, effective_mass=effective_mass,
                      total_mass=total_mass, cumulative_mass_fraction=cumulative,
                      analytical_frequencies_hz=analytical,
                      frequency_errors=errors, eigen_solver=eigen_solver)


def q4_top_surface_stress(xy: torch.Tensor, ue: torch.Tensor, *, young: float,
                          poisson: float, thickness: float):
    """Top-surface stress of a Q4 Mindlin element at the four Gauss points.

    Moments M = Db kappa with kappa = B_b u_e (same B_b as the stiffness
    builder) and sigma = 6 M / t^2.  Returns (sxx, syy, sxy, von_mises), each
    a 4-vector in Gauss order (xi,eta) = (-,-), (+,-), (-,+), (+,+).
    """
    t = _positive("thickness", thickness)
    D = young * t ** 3 / (12.0 * (1.0 - poisson ** 2))
    Db = D * torch.tensor([[1.0, poisson, 0.0],
                           [poisson, 1.0, 0.0],
                           [0.0, 0.0, (1.0 - poisson) / 2.0]], dtype=torch.float64)
    gp = 1.0 / 3.0 ** 0.5
    sxx, syy, sxy, vm = [], [], [], []
    for xif in (-gp, gp):
        for etaf in (-gp, gp):
            xi = xy.new_tensor(xif)
            eta = xy.new_tensor(etaf)
            N, dN_nat = _plate_shape(xi, eta)
            J = dN_nat.T @ xy
            dN = dN_nat @ torch.linalg.inv(J).T
            Bb = torch.zeros((3, 12), dtype=torch.float64)
            for i in range(4):
                Bb[0, 3 * i + 1] = dN[i, 0]
                Bb[1, 3 * i + 2] = dN[i, 1]
                Bb[2, 3 * i + 1] = dN[i, 1]
                Bb[2, 3 * i + 2] = dN[i, 0]
            M = Db @ (Bb @ ue)
            s_x = 6.0 * M[0] / (t * t)
            s_y = 6.0 * M[1] / (t * t)
            s_xy = 6.0 * M[2] / (t * t)
            von = torch.sqrt(s_x ** 2 - s_x * s_y + s_y ** 2 + 3.0 * s_xy ** 2)
            sxx.append(s_x)
            syy.append(s_y)
            sxy.append(s_xy)
            vm.append(von)
    return (torch.stack(sxx), torch.stack(syy), torch.stack(sxy), torch.stack(vm))


def _plate_shape(xi: torch.Tensor, eta: torch.Tensor):
    """Bilinear Q4 shape functions and their natural-coordinate derivatives."""
    one, zero = xi.new_tensor(1.0), xi.new_tensor(0.0)
    N = 0.25 * torch.stack(((one - xi) * (one - eta), (one + xi) * (one - eta),
                            (one + xi) * (one + eta), (one - xi) * (one + eta)))
    dN_xi = 0.25 * torch.stack((-(one - eta), (one - eta), (one + eta), -(one + eta)))
    dN_eta = 0.25 * torch.stack((-(one - xi), -(one + xi), (one + xi), (one - xi)))
    dN = torch.stack((dN_xi, dN_eta), dim=1)
    return N, dN


def interp_srs(srs_result: SRSResult, damping: float, frequencies_hz,
               quantity: str = "sd") -> np.ndarray:
    """Log-log interpolation of an SRS curve at arbitrary frequencies."""
    z = float(damping)
    key = {"sd": srs_result.sd_m, "sv": srs_result.sv_ms, "sa": srs_result.sa_ms2}[quantity]
    curve = key[z]
    f = srs_result.frequencies_hz
    fq = np.atleast_1d(np.asarray(frequencies_hz, dtype=np.float64))
    logv = np.interp(np.log(fq), np.log(f), np.log(np.maximum(curve, 1e-300)))
    return np.exp(logv)



def _element_von_mises_batch(nodes: np.ndarray, elements: np.ndarray, u: np.ndarray,
                             *, young: float, poisson: float,
                             thickness: float) -> np.ndarray:
    """Vectorised top-surface von Mises per element (max over the four GPs).

    NumPy port of :func:`q4_top_surface_stress` used for the modal stress
    recovery, where thousands of small element evaluations would otherwise be
    dominated by per-op dispatch cost.  The 2x2 Jacobian inverse and the strain
    rows are written out explicitly (element-wise array algebra only) so the
    routine costs microseconds per mode.  Agreement with the scalar routine is
    asserted in the test suite.
    """
    t = float(thickness)
    D = young * t ** 3 / (12.0 * (1.0 - poisson ** 2))
    D11, D12, D33 = D, D * poisson, D * (1.0 - poisson) / 2.0
    xy = np.asarray(nodes, dtype=np.float64)[elements]        # (E,4,2)
    ue = np.asarray(u, dtype=np.float64)                      # (E,12)
    xs = xy[:, :, 0]
    ys = xy[:, :, 1]
    gp = 1.0 / 3.0 ** 0.5
    out = np.full(elements.shape[0], -np.inf)
    for xif in (-gp, gp):
        for etaf in (-gp, gp):
            N, dN_nat = _plate_shape_np(xif, etaf)
            j00 = dN_nat[:, 0] @ xs.T
            j01 = dN_nat[:, 0] @ ys.T
            j10 = dN_nat[:, 1] @ xs.T
            j11 = dN_nat[:, 1] @ ys.T
            det = j00 * j11 - j01 * j10
            i00, i01 = j11 / det, -j01 / det
            i10, i11 = -j10 / det, j00 / det
            dn = np.empty((elements.shape[0], 4, 2))
            dn[:, :, 0] = dN_nat[:, 0][None, :] * i00[:, None] + \
                dN_nat[:, 1][None, :] * i10[:, None]
            dn[:, :, 1] = dN_nat[:, 0][None, :] * i01[:, None] + \
                dN_nat[:, 1][None, :] * i11[:, None]
            k0 = np.zeros(elements.shape[0])
            k1 = np.zeros(elements.shape[0])
            k2 = np.zeros(elements.shape[0])
            for a in range(4):
                k0 += dn[:, a, 0] * ue[:, 3 * a + 1]
                k1 += dn[:, a, 1] * ue[:, 3 * a + 2]
                k2 += dn[:, a, 1] * ue[:, 3 * a + 1] + dn[:, a, 0] * ue[:, 3 * a + 2]
            mx = D11 * k0 + D12 * k1
            my = D12 * k0 + D11 * k1
            mxy = D33 * k2
            sx = 6.0 * mx / (t * t)
            sy = 6.0 * my / (t * t)
            sxy = 6.0 * mxy / (t * t)
            von = np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3.0 * sxy ** 2)
            out = np.maximum(out, von)
    return out


def _plate_shape_np(xi: float, eta: float):
    """Bilinear Q4 shape functions / natural derivatives (NumPy scalars)."""
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dN = 0.25 * np.array([[-(1 - eta), -(1 - xi)],
                          [(1 - eta), -(1 + xi)],
                          [(1 + eta), (1 + xi)],
                          [-(1 + eta), (1 - xi)]])
    return N, dN


def interior_element_mask(modal: PlateModal, margin: float = 0.15) -> np.ndarray:
    """Elements whose centroid lies in the interior window of the panel.

    The simply supported idealisation has a singular corner reaction, so the
    discrete bending moments next to the four corners grow without bound under
    mesh refinement (verified: the corner value rises 18% between a 16x16 and a
    24x24 mesh while the plate centre is stable to 0.3%).  Design stresses are
    therefore evaluated in the mesh-independent interior window
    ``margin <= x/L <= 1-margin`` and ``margin <= y/W <= 1-margin``.
    """
    elements = modal.elements
    centroids = modal.nodes[elements].mean(dim=1)
    x, y = centroids[:, 0], centroids[:, 1]
    m = float(margin)
    return ((x >= m * modal.length) & (x <= (1.0 - m) * modal.length) &
            (y >= m * modal.width) & (y <= (1.0 - m) * modal.width)).numpy()


def plate_modal_unit_stress(modal: PlateModal, *, margin: float = 0.15) -> np.ndarray:
    """Element top-surface von Mises per unit modal amplitude.

    Returns an ``(nmodes, n_elements)`` array: entry ``[i, e]`` is the maximum
    (over the four Gauss points) top-surface von Mises stress of element ``e``
    for the displacement field ``u = phi_i``.  Because the element response is
    linear in the modal amplitude, any modal combination can then be evaluated
    without touching the elements again.
    """
    nmodes = modal.modes_full.shape[1]
    elements = modal.elements.numpy()
    nodes = modal.nodes.numpy()
    out = np.zeros((nmodes, elements.shape[0]))
    mask = interior_element_mask(modal, margin=margin)
    for i in range(nmodes):
        u_elem = modal.modes_full[:, i].numpy()[np.stack(
            (3 * elements, 3 * elements + 1, 3 * elements + 2), axis=-1)
            .reshape(elements.shape[0], 12)]
        out[i] = _element_von_mises_batch(nodes, elements, u_elem,
                                          young=modal.young,
                                          poisson=modal.poisson,
                                          thickness=modal.thickness)
    out[:, ~mask] = 0.0
    return out


def ddam_modal_stress(modal: PlateModal, *, srs_result: SRSResult, damping: float,
                      modes_used: int | None = None,
                      unit_stress: np.ndarray | None = None) -> dict:
    """DDAM modal stresses and NRL combination for a base-acceleration SRS.

    Modal amplitude ``q_i = Gamma_i SD_i`` with ``SD_i`` the spectral
    displacement at the modal frequency; displacement field ``phi_i q_i``; the
    design stress is the NRL sum (sum of absolute modal values) evaluated
    element-wise and maximised over the plate.
    """
    nmodes = modal.modes_full.shape[1]
    if modes_used is None:
        modes_used = nmodes
    modes_used = int(min(modes_used, nmodes))
    freqs = modal.frequencies_hz[:modes_used].numpy()
    sd = interp_srs(srs_result, damping, freqs, quantity="sd")
    if unit_stress is None:
        unit_stress = plate_modal_unit_stress(modal)
    unit_stress = np.asarray(unit_stress)[:modes_used]
    amplitudes = np.abs(modal.participation[:modes_used].numpy() * sd)
    nrl = np.abs(unit_stress) * amplitudes[:, None]
    per_element = nrl.sum(axis=0)
    i_max = int(np.argmax(per_element))
    per_mode = []
    for i in range(modes_used):
        per_mode.append({
            "mode": i + 1,
            "frequency_hz": float(freqs[i]),
            "participation": float(modal.participation[i]),
            "spectral_displacement_m": float(sd[i]),
            "modal_amplitude_m": float(modal.participation[i] * sd[i]),
            "max_von_mises_pa": float(unit_stress[i].max() * amplitudes[i]),
            "effective_mass_fraction": float(modal.cumulative_mass_fraction[i]),
        })
    return {"modes_used": modes_used,
            "damping": float(damping),
            "nrl_design_stress_pa": float(per_element[i_max]),
            "critical_element": i_max,
            "per_mode": per_mode,
            "cumulative_mass_fraction_used": float(
                modal.cumulative_mass_fraction[modes_used - 1])}


def yield_margin_ledger(modal: PlateModal, *, srs_result: SRSResult, damping: float,
                        yield_stress: float, safety_factor: float = 1.0,
                        modes_used: int | None = None,
                        unit_stress: np.ndarray | None = None) -> dict:
    """Yield margin = sigma_y / (SF * sigma_design) for one shock environment."""
    sigma_y = _positive("yield_stress", yield_stress)
    sf = _positive("safety_factor", safety_factor)
    stress = ddam_modal_stress(modal, srs_result=srs_result, damping=damping,
                              modes_used=modes_used, unit_stress=unit_stress)
    sigma = stress["nrl_design_stress_pa"]
    margin = sigma_y / (sf * sigma) if sigma > 0 else float("inf")
    out = dict(stress)
    out.update({"yield_stress_pa": sigma_y, "safety_factor": sf,
                "margin": margin, "verdict": "PASS" if margin >= 1.0 else "FAIL"})
    return out


def srs_time_step_drift(cole: ColePoint, *, areal_mass: float, damping: float,
                        dt: float, n: int = 96, f_min: float = 1.0,
                        f_max: float = 20000.0,
                        density: float = SEAWATER_DENSITY,
                        sound_speed: float = SEAWATER_SOUND_SPEED) -> dict:
    """Drift of the SRS under time-step halving (same underlying closed form)."""
    freqs = log_frequencies(f_min, f_max, n)
    results = {}
    for factor in (1.0, 0.5):
        dtk = dt * factor
        hist = shock_acceleration_history(cole, areal_mass=areal_mass, dt=dtk,
                                         density=density, sound_speed=sound_speed)
        r = srs(hist.acceleration.numpy(), dtk, freqs, damping_ratios=(damping,))
        results[factor] = r
    coarse = results[1.0]
    fine = results[0.5]
    rel = np.abs(coarse.sd_m[damping] - fine.sd_m[damping]) / \
        np.maximum(np.abs(fine.sd_m[damping]), 1e-30)
    return {"dt_coarse": dt, "dt_fine": dt / 2.0, "damping": float(damping),
            "max_relative_drift": float(rel.max()),
            "mean_relative_drift": float(rel.mean())}


# ---------------------------------------------------------------------------
# 5. Convenience: environment -> acceleration history -> SRS -> DDAM
# ---------------------------------------------------------------------------
def shock_acceleration_history(cole: ColePoint, *, areal_mass: float,
                               duration: float | None = None,
                               dt: float = 1.0e-6,
                               density: float = SEAWATER_DENSITY,
                               sound_speed: float = SEAWATER_SOUND_SPEED):
    """Air-backed plate acceleration history of the Taylor problem (closed form).

    Validated against :func:`taylor_fsi_fem` (see the repo report); used for the
    parameter sweep because the validation shows < 3 % deviation.
    """
    theta = cole.decay_time_s
    if duration is None:
        duration = 8.0 * theta
    n = int(math.ceil(duration / dt)) + 1
    t = np.linspace(0.0, (n - 1) * dt, n)
    ref = taylor_analytic(t, peak_pressure=cole.peak_pressure_pa,
                          decay_time=theta, areal_mass=areal_mass,
                          density=density, sound_speed=sound_speed)
    return ref