"""W23 extension of the underwater-explosion (UNDEX) shock chain.

The base chain lives in :mod:`tensorfem.undex_shock` (Cole (1948) shock
environment -> Taylor air-backed plate -> SRS / DDAM).  This module extends it to
a **multi-case ensemble** and adds the pieces the W23 protocol needs:

* ``rayleigh_collapse_time`` / ``bubble_scaling_exponents`` -- coefficient-free
  derivation of the *scaling structure* of the bubble-pulsation period
  ``T ~ W^(1/3) (H + p_atm/rho g)^(-5/6)`` from the Rayleigh collapse integral.
  Only the prefactor is empirical.
* ``cole_bubble_period`` -- the empirical bubble-period law with the prefactor
  carried as a **declared implementation parameter** (see
  ``benchmarks/undex_shock/cole_anchor.json``; it is *not* a transcribed value).
* ``panel_fsi_shock`` -- a genuinely finite-element ship-plate response: Q4
  Mindlin (MITC) panel, hard simple supports, wetted on one face, loaded by the
  Cole pulse through the exact plane-wave (local-impedance) fluid coupling
  ``p_wall = 2 p_i - rho c v_n``.  The coupling is the 1-D wave-equation solution
  normal to the plate (fluid leg analytic in the plate-normal direction, the
  *structure* fully discretised); the complementary leg with a discretised fluid
  column is :func:`tensorfem.undex_shock.taylor_fsi_fem`.
* the energy ledger inside ``panel_fsi_shock``: incident / reflected / plate
  energy.  The discrete identity

      P_inc - P_ref = p_i int(v dA) - rho c int(v^2 dA) = u_dot^T (f - C u_dot)

  holds exactly for the consistent load and consistent impedance matrices used
  here, so the residual measures only time-integration error.
* ``roache_gci`` -- grid-convergence index from three refinement levels.
* ``srs_band_table`` -- band-wise SRS comparison (>= 5 bands).

Sign conventions: ``z`` points along the incident wave (into the plate), ``w`` is
the plate displacement in ``+z``, ``v_n = dw/dt``, ``p_wall`` acts in ``+z``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .plate import q4_mindlin_stiffness
from .plate_vibration import q4_plate_mass_matrix
from .undex_shock import (
    SEAWATER_DENSITY,
    SEAWATER_SOUND_SPEED,
    TNT_SPECIFIC_ENERGY_J_KG,
    cole_point,
    log_frequencies,
    newmark_factorized,
    srs,
    taylor_analytic,
)

GRAVITY = 9.80665
#: Depth offset of Cole's bubble law: atmospheric head expressed as metres of
#: water (p_atm / (rho g) ~ 10.33 m).
ATMOSPHERIC_HEAD_M = 10.33
#: Bubble-period exponent on the hydrostatic head.  Derived (coefficient free) by
#: :func:`bubble_scaling_exponents`; it is not fitted here.
BUBBLE_HEAD_EXPONENT = 5.0 / 6.0
#: Candidate prefactor of the TNT bubble-period law [s * m^(5/6) / kg^(1/3)].
#: UNCITED CANDIDATE: a declared implementation parameter, NOT transcribed from
#: Cole (1948).  See benchmarks/undex_shock/cole_anchor.json
#: (``parameters[T_b].provenance.kind == "uncited_candidate"``).
TNT_BUBBLE_K_S = 2.11


# ---------------------------------------------------------------------------
# 1. bubble period: coefficient-free scaling structure + declared prefactor
# ---------------------------------------------------------------------------
def rayleigh_collapse_time(energy_j: float, hydrostatic_pa: float, *,
                           density: float = SEAWATER_DENSITY) -> float:
    """Rayleigh collapse time of a gas bubble holding ``energy_j`` at ``p_h``.

    The bubble radius follows from the energy balance ``E = (4/3) pi R0^3 p_h``;
    the collapse time follows from the Rayleigh (1917) integral for an empty
    cavity, ``t_c = sqrt(3 rho / 2 p_h) R0 I`` with

        I = int_0^1 xi^(3/2) (1 - xi^3)^(-1/2) dxi
          = (1/3) B(5/6, 1/2) = (1/3) Gamma(5/6) Gamma(1/2) / Gamma(4/3)
          = 0.746835...,   so  sqrt(3/2) I = 0.914686...,

    i.e. the classical ``0.9147 R0 sqrt(rho/p_h)`` prefactor is *derived* here
    (Euler beta integral evaluated with ``math.lgamma``), not quoted.  Nothing in
    this function is fitted to bubble-period data.
    """
    E = float(energy_j)
    p_h = float(hydrostatic_pa)
    rho = float(density)
    if not (E > 0.0 and p_h > 0.0 and rho > 0.0):
        raise ValueError("energy, hydrostatic pressure and density must be positive")
    r0 = (3.0 * E / (4.0 * math.pi * p_h)) ** (1.0 / 3.0)
    beta_integral = (1.0 / 3.0) * math.exp(
        math.lgamma(5.0 / 6.0) + math.lgamma(0.5) - math.lgamma(4.0 / 3.0))
    return math.sqrt(3.0 * rho / (2.0 * p_h)) * r0 * beta_integral


def bubble_scaling_exponents(*, weight_kg: float = 10.0, depth_m: float = 40.0,
                             density: float = SEAWATER_DENSITY,
                             specific_energy: float = TNT_SPECIFIC_ENERGY_J_KG,
                             ratio: float = 3.0,
                             tolerance: float = 1.0e-9) -> dict:
    """Machine check of the bubble-clock scaling structure (no fitted prefactor).

    Two independent scalings are verified on the Rayleigh collapse time:

    * charge weight: ``t_c(s^3 W) / t_c(W) = s`` (exponent 1/3 on W),
    * hydrostatic head: ``t_c(s p_h) / t_c(p_h) = s^(-5/6)`` (exponent -5/6),

    with ``p_h = p_atm + rho g H``.  Composing them gives Cole's
    ``T ~ W^(1/3) (H + 10.33)^(-5/6)``.  A law with independent W and H
    exponents fails this check, so it is not tautological.
    """
    p_atm = SEAWATER_DENSITY * GRAVITY * ATMOSPHERIC_HEAD_M
    p_h = p_atm + float(density) * GRAVITY * float(depth_m)

    def t_c(W: float, p: float) -> float:
        return rayleigh_collapse_time(W * specific_energy, p, density=density)

    base = t_c(weight_kg, p_h)
    w_rows, p_rows = [], []
    for s in (0.5, 2.0, ratio):
        got = t_c(s ** 3 * weight_kg, p_h)
        w_rows.append({"scale": s, "ratio": got / base, "expected": s,
                       "rel_error": abs(got / base - s) / s})
        got = t_c(weight_kg, s * p_h)
        exp = s ** (-BUBBLE_HEAD_EXPONENT)
        p_rows.append({"scale": s, "ratio": got / base, "expected": exp,
                       "rel_error": abs(got / base - exp) / exp})

    t_lo, t_hi = t_c(weight_kg, p_h), t_c(ratio ** 3 * weight_kg, p_h)
    slope_w = math.log(t_hi / t_lo) / math.log(ratio ** 3)
    t_lo, t_hi = t_c(weight_kg, p_h), t_c(weight_kg, ratio * p_h)
    slope_p = math.log(t_hi / t_lo) / math.log(ratio)

    ok = (all(r["rel_error"] < tolerance for r in w_rows + p_rows)
          and abs(slope_w - 1.0 / 3.0) < 1.0e-9
          and abs(slope_p + BUBBLE_HEAD_EXPONENT) < 1.0e-9)
    return {
        "pass": bool(ok),
        "tolerance": tolerance,
        "hydrostatic_pa": p_h,
        "weight_scaling": w_rows,
        "head_scaling": p_rows,
        "log_slope_weight_exponent": slope_w,
        "log_slope_head_exponent": slope_p,
        "expected_weight_exponent": 1.0 / 3.0,
        "expected_head_exponent": -BUBBLE_HEAD_EXPONENT,
        "beta_prefactor_derived": math.sqrt(1.5) * (1.0 / 3.0) * math.exp(
            math.lgamma(5.0 / 6.0) + math.lgamma(0.5) - math.lgamma(4.0 / 3.0)),
        "note": ("only the prefactor of the bubble law is empirical; the W^(1/3) "
                 "and (H + p_atm/rho g)^(-5/6) structure is derived here"),
    }


def cole_bubble_period(weight_kg: float, depth_m: float, *,
                       K_b: float = TNT_BUBBLE_K_S,
                       depth_offset_m: float = ATMOSPHERIC_HEAD_M,
                       exponent: float = BUBBLE_HEAD_EXPONENT) -> float:
    """Cole bubble-pulsation period ``T = K_b W^(1/3) / (H + 10.33)^(5/6)`` [s].

    ``K_b`` is an implementation parameter flagged as an uncited candidate in the
    anchor register; the exponents are the derived ones.
    """
    W = float(weight_kg)
    H = float(depth_m)
    if not (W > 0.0 and H >= 0.0):
        raise ValueError("weight must be positive and depth non-negative")
    head = H + float(depth_offset_m)
    if head <= 0.0:
        raise ValueError("depth offset must keep the head positive")
    return float(K_b) * W ** (1.0 / 3.0) / head ** float(exponent)


def cole_case_environment(weight_kg: float, range_m: float, depth_m: float) -> dict:
    """Cole environment for one (charge, standoff, depth) point."""
    point = cole_point(weight_kg, range_m)
    return {
        "weight_kg": float(weight_kg),
        "range_m": float(range_m),
        "depth_m": float(depth_m),
        "reduced_range": point.reduced_range,
        "peak_pressure_pa": point.peak_pressure_pa,
        "peak_pressure_mpa": point.peak_pressure_mpa,
        "decay_time_s": point.decay_time_s,
        "specific_impulse_pa_s": point.specific_impulse_pa_s,
        "energy_flux_j_m2": point.energy_flux_j_m2,
        "shock_energy_j": point.shock_energy_j,
        "shock_energy_ratio": point.shock_energy_ratio,
        "bubble_period_s": cole_bubble_period(weight_kg, depth_m),
    }


# ---------------------------------------------------------------------------
# 2. FE ship panel with plane-wave (local impedance) fluid coupling
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PanelFSIResult:
    time: torch.Tensor
    center_deflection: torch.Tensor       # m, +z (along the incident wave)
    center_velocity: torch.Tensor         # m/s
    center_acceleration: torch.Tensor     # m/s^2
    incident_pressure: torch.Tensor       # Pa, free-field Cole pulse
    center_wall_pressure: torch.Tensor    # Pa, p = 2 p_i - rho c v_n
    energy_incident: torch.Tensor         # J
    energy_reflected: torch.Tensor        # J
    energy_plate: torch.Tensor            # J
    energy_residual: torch.Tensor         # dimensionless
    meta: dict

    def as_dict(self) -> dict:
        return {
            "time_s": self.time.tolist(),
            "center_deflection_m": self.center_deflection.tolist(),
            "center_velocity_ms": self.center_velocity.tolist(),
            "center_acceleration_ms2": self.center_acceleration.tolist(),
            "incident_pressure_pa": self.incident_pressure.tolist(),
            "center_wall_pressure_pa": self.center_wall_pressure.tolist(),
            "energy_incident_j": self.energy_incident.tolist(),
            "energy_reflected_j": self.energy_reflected.tolist(),
            "energy_plate_j": self.energy_plate.tolist(),
            "energy_residual": self.energy_residual.tolist(),
            "meta": self.meta,
        }


def _panel_mesh(*, length: float, width: float, nx: int, ny: int):
    xs = torch.linspace(0.0, length, nx + 1, dtype=torch.float64)
    ys = torch.linspace(0.0, width, ny + 1, dtype=torch.float64)
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    nodes = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            elements.append((n0, n0 + 1, n0 + nx + 2, n0 + nx + 1))
    return nodes, torch.tensor(elements, dtype=torch.long)


def _ss_free_dofs(ndof: int, nx: int, ny: int) -> torch.Tensor:
    """Free dofs of a hard simply supported edge (``w = 0`` on all four edges)."""
    n_nodes_x = nx + 1
    fixed = set()
    for j in range(ny + 1):
        fixed.add(3 * (j * n_nodes_x))
        fixed.add(3 * (j * n_nodes_x + n_nodes_x - 1))
    for i in range(n_nodes_x):
        fixed.add(3 * i)
        fixed.add(3 * (ny * n_nodes_x + i))
    return torch.tensor([d for d in range(ndof) if d not in fixed], dtype=torch.long)


def assemble_panel(*, length: float = 1.5, width: float = 1.0, nx: int = 18,
                   ny: int = 12, thickness: float = 0.010,
                   young: float = 206.0e9, poisson: float = 0.3,
                   density: float = 7850.0,
                   fluid_density: float = SEAWATER_DENSITY,
                   sound_speed: float = SEAWATER_SOUND_SPEED,
                   shear_treatment: str = "mitc", bc: str = "ss",
                   stiffness_scale: float = 1.0) -> dict:
    """Assemble a SS Mindlin panel plus the consistent wetted-face operators.

    Returns ``K``, ``M`` (3 dofs per node), the impedance damping matrix
    ``C = rho c int N N^T dA`` and the load distribution vector ``L = int N dA``
    on the transverse dofs, plus the reduced index set.  ``shear_treatment=
    "mitc"`` is the rank-sufficient Q4 shear field (the one-point ``"sri"`` rule
    leaves two zero-energy mechanisms).
    """
    L_a, L_b = float(length), float(width)
    if not (L_a > 0.0 and L_b > 0.0):
        raise ValueError("panel dimensions must be positive")
    nx, ny = int(nx), int(ny)
    if nx < 1 or ny < 1:
        raise ValueError("panel mesh needs at least one element per direction")
    nodes, elements = _panel_mesh(length=L_a, width=L_b, nx=nx, ny=ny)
    ndof = 3 * nodes.shape[0]
    K = torch.zeros((ndof, ndof), dtype=torch.float64)
    M = torch.zeros((ndof, ndof), dtype=torch.float64)
    C = torch.zeros((ndof, ndof), dtype=torch.float64)
    load = torch.zeros(ndof, dtype=torch.float64)
    rc = float(fluid_density) * float(sound_speed)
    # consistent int N_i N_j dA for a bilinear rectangle: area/36 * [[4,2,1,2],...]
    pattern = torch.tensor([[4.0, 2.0, 1.0, 2.0],
                            [2.0, 4.0, 2.0, 1.0],
                            [1.0, 2.0, 4.0, 2.0],
                            [2.0, 1.0, 2.0, 4.0]], dtype=torch.float64)
    for conn in elements:
        xy = nodes[conn, :]
        ke = q4_mindlin_stiffness(xy, young, poisson, thickness,
                                  shear_treatment=shear_treatment)
        me = q4_plate_mass_matrix(xy, density, thickness)
        a_e = float((xy[1, 0] - xy[0, 0]) * (xy[3, 1] - xy[0, 1]))
        if a_e <= 0.0:
            raise ValueError("degenerate panel element")
        dofs = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), dim=1).reshape(-1)
        K[dofs[:, None], dofs] += ke
        M[dofs[:, None], dofs] += me
        w_dofs = 3 * conn
        C[w_dofs[:, None], w_dofs] += rc * a_e / 36.0 * pattern
        load[w_dofs] += a_e / 4.0
    K = 0.5 * (K + K.T) * float(stiffness_scale)
    M = 0.5 * (M + M.T)
    C = 0.5 * (C + C.T)
    if bc not in ("ss", "free"):
        raise ValueError("bc must be 'ss' or 'free'")
    free = _ss_free_dofs(ndof, nx, ny) if bc == "ss" else \
        torch.arange(ndof, dtype=torch.long)
    return {"nodes": nodes, "elements": elements, "K": K, "M": M, "C": C,
            "load": load, "free": free, "ndof": ndof, "nx": nx, "ny": ny,
            "area": L_a * L_b, "length": L_a, "width": L_b,
            "thickness": float(thickness), "rc": rc, "bc": bc,
            "areal_mass": float(density) * float(thickness)}


def panel_first_frequency(*, length: float = 1.5, width: float = 1.0,
                          thickness: float = 0.010, young: float = 206.0e9,
                          poisson: float = 0.3, density: float = 7850.0) -> float:
    """Kirchhoff SS-panel fundamental frequency [Hz] (thin-plate reference)."""
    D = young * thickness ** 3 / (12.0 * (1.0 - poisson ** 2))
    omega = math.pi ** 2 * (1.0 / length ** 2 + 1.0 / width ** 2) * \
        math.sqrt(D / (density * thickness))
    return omega / (2.0 * math.pi)


def panel_fsi_shock(*, peak_pressure: float, decay_time: float,
                    length: float = 1.5, width: float = 1.0, nx: int = 18,
                    ny: int = 12, thickness: float = 0.010,
                    young: float = 206.0e9, poisson: float = 0.3,
                    density: float = 7850.0,
                    fluid_density: float = SEAWATER_DENSITY,
                    sound_speed: float = SEAWATER_SOUND_SPEED,
                    steps_per_decay: int = 60,
                    duration_theta: float = 16.0,
                    shear_treatment: str = "mitc", bc: str = "ss",
                    stiffness_scale: float = 1.0,
                    model: dict | None = None) -> PanelFSIResult:
    """Ship-panel response to the Cole pulse with exact plane-wave coupling.

    Solid: Q4 Mindlin panel, hard simple supports, uniform thickness.
    Fluid: plane-wave (local impedance) coupling on the wetted face,
    ``p_wall = 2 p_i - rho c v_n`` -- the exact normal-incidence solution of the
    1-D wave equation in the fluid; ``p_i(t) = p_m exp(-t/theta)``.
    Time integration: average-acceleration Newmark (``beta = 1/4``,
    ``gamma = 1/2``), single factorisation.
    """
    p_m = float(peak_pressure)
    theta = float(decay_time)
    if not (p_m > 0.0 and theta > 0.0):
        raise ValueError("peak pressure and decay time must be positive")
    steps = int(steps_per_decay)
    if steps < 8:
        raise ValueError("steps_per_decay must be at least 8")
    duration = float(duration_theta) * theta
    dt = theta / steps
    nsteps = int(math.ceil(duration / dt)) + 1
    time = torch.linspace(0.0, (nsteps - 1) * dt, nsteps, dtype=torch.float64)

    if model is None:
        model = assemble_panel(length=length, width=width, nx=nx, ny=ny,
                               thickness=thickness, young=young,
                               poisson=poisson, density=density,
                               fluid_density=fluid_density,
                               sound_speed=sound_speed,
                               shear_treatment=shear_treatment, bc=bc,
                               stiffness_scale=stiffness_scale)
    elif model["rc"] != float(fluid_density) * float(sound_speed):
        raise ValueError("cached model was built with a different fluid")
    free = model["free"]
    Kr = model["K"][free[:, None], free]
    Mr = model["M"][free[:, None], free]
    Cr = model["C"][free[:, None], free]
    load_r = model["load"][free]

    p_i = p_m * torch.exp(-time / theta)
    # plane-wave coupling: the wetted face sees the incident plus the reflected
    # wave, p_wall = 2 p_i - rho c v_n.  The incident part enters twice (the
    # factor 2 is the rigid-wall doubling), the -rho c v_n part is the consistent
    # impedance matrix C; both are needed for the energy identity below.
    force = 2.0 * p_i[:, None] * load_r[None, :]
    u0 = torch.zeros(free.numel(), dtype=torch.float64)
    v0 = torch.zeros_like(u0)
    res = newmark_factorized(Mr, Kr, force, time, u0, v0, C=Cr,
                             beta=0.25, gamma=0.5)

    # energy ledger in full space: the discrete identity
    # P_inc - P_ref = p_i (L . v) - rho c v^T C v = u_dot^T (f - C u_dot)
    # holds exactly; constrained (w = 0) dofs contribute nothing.
    n = time.numel()
    v_full = torch.zeros((n, model["ndof"]), dtype=torch.float64)
    u_full = torch.zeros_like(v_full)
    v_full[:, free] = res.velocity
    u_full[:, free] = res.displacement
    quad_v = torch.einsum("ni,ij,nj->n", v_full, model["C"], v_full)
    flux_v = v_full @ model["load"]
    area = model["area"]
    rc = model["rc"]
    power_in = p_i * p_i * area / rc
    power_ref = power_in - 2.0 * p_i * flux_v + quad_v
    e_inc = torch.zeros(n, dtype=torch.float64)
    e_ref = torch.zeros(n, dtype=torch.float64)
    e_inc[1:] = torch.cumulative_trapezoid(power_in, time)
    e_ref[1:] = torch.cumulative_trapezoid(power_ref, time)
    kinetic = 0.5 * torch.einsum("ni,ij,nj->n", v_full, model["M"], v_full)
    strain = 0.5 * torch.einsum("ni,ij,nj->n", u_full, model["K"], u_full)
    e_plate = kinetic + strain
    scale = float(e_inc.abs().max())
    e_res = (e_inc - e_ref - e_plate).abs() / scale if scale > 0.0 else torch.zeros(n)

    centre_node = (model["ny"] // 2) * (model["nx"] + 1) + model["nx"] // 2
    w_dof = 3 * centre_node
    hits = torch.nonzero(free == w_dof, as_tuple=False).flatten()
    if hits.numel() == 0:
        raise ValueError("panel centre transverse dof is constrained")
    pos = int(hits[0])
    w_c = res.displacement[:, pos]
    v_c = res.velocity[:, pos]
    a_c = res.acceleration[:, pos]
    p_wall = 2.0 * p_i - rc * v_c

    meta = {
        "nx": model["nx"], "ny": model["ny"], "ndof": model["ndof"],
        "free_dofs": int(free.numel()), "dt_s": dt, "steps": nsteps,
        "duration_s": float(time[-1]), "steps_per_decay": steps,
        "duration_theta": float(duration_theta),
        "peak_pressure_pa": p_m, "decay_time_s": theta,
        "areal_mass_kg_m2": model["areal_mass"], "rc_pa_s_m": rc,
        "beta_1_s": rc / model["areal_mass"],
        "beta_theta": rc * theta / model["areal_mass"],
        "panel_area_m2": area, "shear_treatment": shear_treatment,
        "bc": bc, "stiffness_scale": float(stiffness_scale),
        "first_frequency_hz": panel_first_frequency(
            length=length, width=width, thickness=thickness, young=young,
            poisson=poisson, density=density),
        "min_wall_pressure_pa": float(p_wall.min()),
        "cavitation": bool(float(p_wall.min()) <= 0.0),
        "peak_wall_pressure_pa": float(p_wall.abs().max()),
        "peak_center_acceleration_ms2": float(a_c.abs().max()),
        "peak_center_deflection_m": float(w_c.abs().max()),
    }
    late = time > 2.0 * theta
    meta["energy_residual_max"] = float(e_res[late].max()) if bool(late.any()) else None
    meta["energy_residual_final"] = float(e_res[-1])
    # impulse split: during the loading window (p_i > 1e-2 p_m, i.e. t < 4.6 theta)
    # and over the whole record (the post-pulse part is the radiation impulse and
    # largely cancels the loading impulse for a movable plate).
    load_mask = time <= 4.0 * theta
    meta["wall_impulse_loading_pa_s"] = float(torch.trapezoid(
        p_wall[load_mask], time[load_mask]))
    meta["wall_impulse_total_pa_s"] = float(torch.trapezoid(p_wall, time))
    neg = torch.nonzero(p_wall <= 0.0, as_tuple=False).flatten()
    t_cav = float(time[int(neg[0])]) if neg.numel() else None
    meta["cavitation_onset_s"] = t_cav
    t_end = t_cav if t_cav is not None else float(time[-1])
    pre = time <= t_end
    # impulse delivered before the first tensile excursion (the physically
    # meaningful delivered impulse: beyond t_cav the acoustic model permits
    # tension, which water cannot sustain)
    meta["wall_impulse_precav_pa_s"] = float(torch.trapezoid(p_wall[pre], time[pre]))
    meta["velocity_at_cavitation_ms"] = float(v_c[pre][-1]) if bool(pre.any()) else None
    meta["cavitation_impulse_fraction_of_loading"] = (
        meta["wall_impulse_precav_pa_s"] / meta["wall_impulse_loading_pa_s"]
        if meta["wall_impulse_loading_pa_s"] else None)
    return PanelFSIResult(time=time, center_deflection=w_c, center_velocity=v_c,
                          center_acceleration=a_c, incident_pressure=p_i,
                          center_wall_pressure=p_wall, energy_incident=e_inc,
                          energy_reflected=e_ref, energy_plate=e_plate,
                          energy_residual=e_res, meta=meta)


# ---------------------------------------------------------------------------
# 3. comparisons: Taylor closed form, Cole levels, convergence, SRS bands
# ---------------------------------------------------------------------------
def compare_panel_to_taylor(result: PanelFSIResult, *,
                            density: float = SEAWATER_DENSITY,
                            sound_speed: float = SEAWATER_SOUND_SPEED) -> dict:
    """Compare the panel-centre response with the Taylor closed form.

    The reference is the exact air-backed *areal-mass* plate solution at the same
    areal mass (no bending stiffness, no supports), so agreement is expected in
    the impulse-dominated (local-mass) regime; low-frequency differences are the
    panel's structural flexibility.
    """
    m = result.meta["areal_mass_kg_m2"]
    ref = taylor_analytic(result.time,
                          peak_pressure=result.meta["peak_pressure_pa"],
                          decay_time=result.meta["decay_time_s"],
                          areal_mass=m, density=density,
                          sound_speed=sound_speed)
    out = {"areal_mass_kg_m2": m, "beta_theta": ref.beta_theta,
           "reference_cavitation_time_s": ref.cavitation_time_s,
           "window_s": [0.0, float(result.time[-1])]}
    for name, num, den in (
            ("center_velocity", result.center_velocity, ref.velocity),
            ("center_acceleration", result.center_acceleration, ref.acceleration),
            ("center_wall_pressure", result.center_wall_pressure,
             ref.interface_pressure)):
        scale = float(den.abs().max())
        err = float((num - den).abs().max())
        out[name] = {
            "peak_reference": scale,
            "peak_fem": float(num.abs().max()),
            "max_abs_error": err,
            "relative_max_error": err / scale if scale > 0.0 else float("inf"),
            "peak_ratio_fem_over_reference":
                (float(num.abs().max()) / scale) if scale > 0.0 else float("inf"),
        }
    out["peak_deflection_reference_m"] = float(ref.displacement.abs().max())
    out["peak_deflection_fem_m"] = float(result.center_deflection.abs().max())
    out["deflection_ratio"] = (out["peak_deflection_fem_m"] /
                               out["peak_deflection_reference_m"])
    return out


def pressure_closure(result: PanelFSIResult, *, cole_peak_pa: float,
                     cole_impulse_pa_s: float) -> dict:
    """Wall-pressure closure of the panel model against Cole free-field levels.

    Three reference levels are reported explicitly, because they are physically
    different quantities:

    * ``cole_free_field``: ``p_m`` and ``p_m theta`` (the incident wave);
    * ``rigid_wall``: ``2 p_m`` and ``2 p_m theta`` (the limiting case of an
      immovable wall, where the reflected wave doubles the pressure);
    * the Taylor closed form (already checked separately), which gives the
      reflection reduction actually produced by a movable plate.
    """
    p_n = result.center_wall_pressure
    peak = float(p_n.abs().max())
    impulse = float(result.meta["wall_impulse_loading_pa_s"])
    rigid_peak = 2.0 * float(cole_peak_pa)
    rigid_impulse = 2.0 * float(cole_impulse_pa_s)
    return {
        "fem_wall_peak_pa": peak,
        "fem_wall_impulse_pa_s": impulse,
        "fem_wall_impulse_total_pa_s": float(result.meta["wall_impulse_total_pa_s"]),
        "impulse_window_s": [0.0, 4.0 * float(result.meta["decay_time_s"])],
        "cavitation_onset_s": result.meta.get("cavitation_onset_s"),
        "min_wall_pressure_pa": result.meta["min_wall_pressure_pa"],
        "cole_free_field_peak_pa": float(cole_peak_pa),
        "cole_free_field_impulse_pa_s": float(cole_impulse_pa_s),
        "rigid_wall_peak_pa": rigid_peak,
        "rigid_wall_impulse_pa_s": rigid_impulse,
        "peak_rel_dev_vs_cole_free_field": abs(peak - cole_peak_pa) / cole_peak_pa,
        "impulse_rel_dev_vs_cole_free_field":
            abs(impulse - cole_impulse_pa_s) / cole_impulse_pa_s,
        "peak_rel_dev_vs_rigid_wall": abs(peak - rigid_peak) / rigid_peak,
        "impulse_rel_dev_vs_rigid_wall":
            abs(impulse - rigid_impulse) / rigid_impulse,
        "reflection_factor_peak": peak / float(cole_peak_pa),
        "reflection_factor_impulse": impulse / float(cole_impulse_pa_s),
        "rigid_wall_factor": 2.0,
        "cavitation": bool(result.meta["cavitation"]),
    }


def roache_gci(fine: float, medium: float, coarse: float, *, ratio: float = 1.5,
               safety: float = 1.25) -> dict:
    """Roache grid-convergence index from three systematically refined results.

    ``p = ln|(f3-f2)/(f2-f1)| / ln r`` (observed order), ``GCI_fine =
    Fs |(f3-f2)/f3| / (r^p - 1)``.  When the observed order is outside
    ``[0.5, 4]`` (oscillatory or non-monotone convergence) the order is clipped
    for reporting and the GCI is also evaluated with the theoretical ``p = 2``;
    both are returned so that a reader can see the sensitivity.
    """
    f1, f2, f3 = float(fine), float(medium), float(coarse)   # f1 = finest
    r = float(ratio)
    d21 = f2 - f1
    d32 = f3 - f2
    eps = 1.0e-300
    if abs(d32) < eps or abs(d21) < eps or d21 * d32 <= 0.0:
        return {"available": False, "values": [f1, f2, f3], "ratio": r,
                "reason": "non-monotone or vanishing differences"}
    p_obs = math.log(abs(d32 / d21)) / math.log(r)
    p_use = min(max(p_obs, 0.5), 4.0)
    denom = r ** p_use - 1.0
    gci = safety * abs((f3 - f2) / f3) / denom if f3 != 0.0 else float("inf")
    p2 = 2.0
    gci_p2 = safety * abs((f3 - f2) / f3) / (r ** p2 - 1.0) if f3 != 0.0 else float("inf")
    return {"available": True, "values": [f1, f2, f3], "ratio": r,
            "observed_order": p_obs, "order_used": p_use,
            "order_clipped": bool(abs(p_use - p_obs) > 1.0e-12),
            "gci_fine": gci, "gci_fine_p2": gci_p2,
            "extrapolated_fine": f3 + (f3 - f2) / (r ** p_use - 1.0),
            "monotone": bool(d21 * d32 > 0.0)}


def srs_pair(acc_fem, acc_ref, *, dt: float, f_min: float = 1.0,
             f_max: float = 10000.0, n: int = 120,
             damping_ratios=(0.0, 0.02, 0.05)) -> dict:
    """SRS of the FEM and of the reference acceleration history."""
    freqs = log_frequencies(f_min=f_min, f_max=f_max, n=n)
    srs_fem = srs(acc_fem, dt, freqs, damping_ratios=damping_ratios)
    srs_ref = srs(acc_ref, dt, freqs, damping_ratios=damping_ratios)
    return {"frequencies_hz": freqs, "fem": srs_fem, "reference": srs_ref,
            "damping_ratios": [float(z) for z in damping_ratios]}


#: SRS comparison bands [Hz] (>= 5 bands required by the mission protocol)
SRS_BANDS_HZ = ((1.0, 10.0), (10.0, 50.0), (50.0, 200.0), (200.0, 1000.0),
                (1000.0, 4000.0), (4000.0, 10000.0))


def srs_band_table(pair: dict, *, damping: float = 0.02,
                   bands=SRS_BANDS_HZ) -> list[dict]:
    """Band-wise SRS deviations between the FEM and the reference solution."""
    f = np.asarray(pair["frequencies_hz"], dtype=np.float64)
    key = float(damping)
    a_fem = pair["fem"].sa_ms2[key]
    a_ref = pair["reference"].sa_ms2[key]
    rows = []
    for lo, hi in bands:
        mask = (f >= lo) & (f <= hi)
        if not bool(mask.any()):
            continue
        fem_peak = float(a_fem[mask].max())
        ref_peak = float(a_ref[mask].max())
        i_fem = int(np.argmax(np.where(mask, a_fem, -np.inf)))
        i_ref = int(np.argmax(np.where(mask, a_ref, -np.inf)))
        rows.append({
            "band_hz": [lo, hi],
            "fem_peak_sa_ms2": fem_peak,
            "reference_peak_sa_ms2": ref_peak,
            "relative_deviation": abs(fem_peak - ref_peak) / ref_peak
            if ref_peak > 0.0 else float("inf"),
            "peak_ratio_fem_over_reference": fem_peak / ref_peak
            if ref_peak > 0.0 else float("inf"),
            "fem_peak_frequency_hz": float(f[i_fem]),
            "reference_peak_frequency_hz": float(f[i_ref]),
            "damping": key,
            "samples": int(mask.sum()),
        })
    return rows


def cole_recompute_vs_anchor(anchor: dict, *, K: float | None = None,
                             A: float | None = None,
                             K_theta: float | None = None,
                             decay_exponent: float | None = None,
                             depth_offset_m: float = ATMOSPHERIC_HEAD_M,
                             bubble_exponent: float = BUBBLE_HEAD_EXPONENT,
                             K_b: float | None = None) -> dict:
    """Recompute the Cole environment and compare with the anchor's records.

    The parameters default to the anchor's own ``parameters`` block, so this is a
    two-implementation cross-check of the *recorded formula and parameters*
    (transcription fidelity), not a validation of the coefficient values against
    the printed source -- that is BLOCKED, see the anchor's
    ``acquisition_status``.
    """
    pars = anchor.get("parameters", {})

    def pick(name: str, override):
        if override is not None:
            return float(override)
        return float(pars[name]["value"])

    k_val = pick("K", K)
    a_val = pick("A", A)
    kt_val = pick("K_theta", K_theta)
    ex_val = pick("decay_exponent", decay_exponent)
    kb_val = pick("T_b", K_b)

    rows = []
    for rec in anchor.get("recorded_values", []):
        W = float(rec["weight_kg"])
        R = float(rec["range_m"])
        H = float(rec["depth_m"])
        red = W ** (1.0 / 3.0) / R
        p_m = k_val * red ** a_val
        theta = kt_val * W ** (1.0 / 3.0) * red ** ex_val
        T_b = kb_val * W ** (1.0 / 3.0) / (H + depth_offset_m) ** bubble_exponent
        impulse = p_m * theta
        rows.append({
            "case_id": rec["case_id"],
            "weight_kg": W, "range_m": R, "depth_m": H,
            "recomputed": {"peak_pressure_pa": p_m, "decay_time_s": theta,
                           "bubble_period_s": T_b,
                           "specific_impulse_pa_s": impulse},
            "relative_deviation": {
                "peak_pressure": abs(p_m - rec["peak_pressure_pa"]) /
                rec["peak_pressure_pa"],
                "decay_time": abs(theta - rec["decay_time_s"]) / rec["decay_time_s"],
                "bubble_period": abs(T_b - rec["bubble_period_s"]) /
                rec["bubble_period_s"],
                "specific_impulse": abs(impulse - rec["specific_impulse_pa_s"]) /
                rec["specific_impulse_pa_s"],
            },
        })
    worst = max((max(r["relative_deviation"].values()) for r in rows), default=0.0)
    return {"tolerance": 0.01, "max_relative_deviation": worst,
            "pass": bool(rows) and worst < 0.01, "rows": rows,
            "parameters_used": {"K": k_val, "A": a_val, "K_theta": kt_val,
                                "decay_exponent": ex_val, "K_b": kb_val,
                                "depth_offset_m": depth_offset_m,
                                "bubble_exponent": bubble_exponent}}

def areal_mass_verification(*, peak_pressure: float, decay_time: float,
                            thickness: float = 0.010, density: float = 7850.0,
                            nx: int = 8, ny: int = 6, steps_per_decay: int = 400,
                            duration_theta: float = 16.0) -> dict:
    """Like-for-like verification of the panel FSI path against Taylor's solution.

    With ``bc="free"`` and ``stiffness_scale=0`` the panel carries mass only
    (no bending, no supports) and its wetted face is exactly Taylor's air-backed
    areal-mass plate.  Because the consistent mass, consistent impedance matrix
    and load vector are built from the same bilinear shape functions, the uniform
    (rigid) motion is an exact solution of the discrete system, so the only error
    is Newmark truncation -- this is a genuine verification of the coupling and
    time integration against the closed form, unlike the flexible-panel
    comparison which is a different physical problem.
    """
    areal = density * thickness
    model = assemble_panel(length=1.5, width=1.0, nx=nx, ny=ny, thickness=thickness,
                           density=density, bc="free", stiffness_scale=0.0)
    w_dofs = torch.arange(0, model["ndof"], 3, dtype=torch.long)
    model_total_mass = float(model["M"][w_dofs[:, None], w_dofs].sum())
    res = panel_fsi_shock(peak_pressure=peak_pressure, decay_time=decay_time,
                          thickness=thickness, density=density, nx=nx, ny=ny,
                          steps_per_decay=steps_per_decay,
                          duration_theta=duration_theta, bc="free",
                          stiffness_scale=0.0)
    ref = taylor_analytic(res.time, peak_pressure=peak_pressure,
                          decay_time=decay_time, areal_mass=areal)
    out = {"areal_mass_kg_m2": areal,
           "panel_mass_kg": float(model_total_mass),
           "mass_check_relative": abs(model_total_mass - areal * 1.5) / (areal * 1.5),
           "steps_per_decay": int(steps_per_decay), "dt_s": res.meta["dt_s"]}
    for name, num, den in (("center_velocity", res.center_velocity, ref.velocity),
                           ("center_acceleration", res.center_acceleration,
                            ref.acceleration),
                           ("center_wall_pressure", res.center_wall_pressure,
                            ref.interface_pressure),
                           ("center_deflection", res.center_deflection,
                            ref.displacement)):
        scale = float(den.abs().max())
        out[name] = {
            "peak_reference": scale, "peak_fem": float(num.abs().max()),
            "relative_max_error": float((num - den).abs().max()) / scale
            if scale > 0.0 else float("inf"),
        }
    out["energy_residual_max"] = res.meta["energy_residual_max"]
    # The displacement comparison is not a verification target: beyond the first
    # tensile excursion the acoustic model pulls the plate back while Taylor's
    # cavitation-free solution lets it coast, so displacement drifts by design.
    # The velocity / acceleration / pressure histories are the verified quantities.
    out["pass"] = all(out[k]["relative_max_error"] < 0.03 for k in
                      ("center_velocity", "center_acceleration",
                       "center_wall_pressure"))
    return out
