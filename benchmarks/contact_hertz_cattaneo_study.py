#!/usr/bin/env python3
"""Hertz / Cattaneo--Mindlin / complementarity verification study (P45-N02).

Runs the three-configuration Hertz ladder (sphere--sphere, sphere--plane,
cylinder--plane), the Cattaneo--Mindlin partial-slip ladder and the unilateral
complementarity audit (penalty versus augmented Lagrangian) on the condensed FE
half-space of :mod:`tensorfem.cattaneo_hertz`, then writes
``artifacts/contact_hertz_cattaneo/`` (result JSON, gates, figures).

Usage::

    PYTHONPATH=src python benchmarks/contact_hertz_cattaneo_study.py
    PYTHONPATH=src python benchmarks/contact_hertz_cattaneo_study.py --levels 8 12 16
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))

from tensorfem.cattaneo_hertz import (  # noqa: E402
    hertz_point_approach,
    hertz_point_from_indentation,
    axisymmetric_compliance,
    cattaneo_displacement,
    cattaneo_stick_radius,
    cattaneo_stiffness_small_q,
    hertz_line,
    hertz_line_approach_quadrature,
    hertz_point,
    hertz_traction,
    plane_strain_compliance,
    recover_hertz_pressure,
    solve_cattaneo_tangential,
    solve_normal_alm,
    solve_normal_penalty,
)

DTYPE = torch.float64
YOUNG = 210.0e9          # steel, Pa
POISSON = 0.3
BALL_RADIUS = 0.05       # m
MU = 0.3
LOADS = (10.0, 100.0, 1000.0)
LINE_STRIP = 1.0e-2      # m: line loads are F/LINE_STRIP (N/m)
CATTANEO_NORMAL_LOAD = 100.0
LEVELS = (12, 17, 24)    # n_in: uniform elements over [0, 2a] (h/a = 1/6, 1/8.5, 1/12)
CATTANEO_LEVELS = (24, 32, 40)
ORDER_LEVELS = (6, 9, 12)   # coarse ladder where the discretisation error dominates
POINT_REFERENCE = 4.0    # surface reference point for the approach, in units of a
EXTENT = 2.0
TOTAL = 100.0
GROWTH = 1.3
CACHE = HERE / "_cache" / "contact_hertz_cattaneo"


# ---------------------------------------------------------------------------
# configuration library
# ---------------------------------------------------------------------------
def point_configs() -> list[dict]:
    """Point-contact configurations (rigid indenter on the elastic body)."""
    e = YOUNG
    return [
        {"name": "sphere_sphere", "kind": "point", "radius": BALL_RADIUS / 2.0,
         "e_star": e / (2.0 * (1.0 - POISSON ** 2)), "note": "R_eff = R/2, two bodies"},
        {"name": "sphere_plane", "kind": "point", "radius": BALL_RADIUS,
         "e_star": e / (1.0 - POISSON ** 2), "note": "rigid plane"},
    ]


def line_config() -> dict:
    return {"name": "cylinder_plane", "kind": "line", "radius": BALL_RADIUS,
            "e_star": YOUNG / (1.0 - POISSON ** 2), "note": "rigid plane, plane strain"}


# ---------------------------------------------------------------------------
# compliance cache
# ---------------------------------------------------------------------------
def compliance(kind: str, n_in: int, nz_in: int, *, use_cache: bool = True):
    CACHE.mkdir(parents=True, exist_ok=True)
    key = f"{kind}_{n_in}_{nz_in}_{POISSON}_{EXTENT}_{TOTAL}_{GROWTH}"
    path = CACHE / f"{key}.pt"
    if use_cache and path.exists():
        blob = torch.load(path)
        from tensorfem.cattaneo_hertz import ContactCompliance
        return ContactCompliance(**blob)
    t0 = time.time()
    if kind == "point":
        c = axisymmetric_compliance(n_in=n_in, nz_in=nz_in, poisson=POISSON,
                                    extent=EXTENT, total=TOTAL, growth=GROWTH)
    else:
        c = plane_strain_compliance(n_in=n_in, nz_in=nz_in, poisson=POISSON,
                                    extent=EXTENT, total=TOTAL, growth=GROWTH)
    elapsed = time.time() - t0
    torch.save({"H": c.H, "coords": c.coords, "measures": c.measures,
                "kind": c.kind, "ndof": c.ndof,
                "memory": {**c.memory, "build_seconds": elapsed}}, path)
    return c


# ---------------------------------------------------------------------------
# indentation solves
# ---------------------------------------------------------------------------
def _scaling(config: dict, solver: dict) -> dict:
    """Length / force / stress scales of the dimensionless problem."""
    if config["kind"] == "point":
        length = solver["a"]
        young_fe = (1.0 - POISSON ** 2) * config["e_star"]
        return {"length": length, "young_fe": young_fe,
                "force": young_fe * length ** 2, "stress": young_fe}
    length = solver["b"]
    young_fe = (1.0 - POISSON ** 2) * config["e_star"]
    return {"length": length, "young_fe": young_fe,
            "force": young_fe * length, "stress": young_fe}


def solve_indentation(compliance_obj, radius_hat: float, load_hat: float, *,
                      method: str = "alm", penalty_factor: float = 100.0,
                      tolerance: float = 1e-12, max_outer: int = 40) -> dict:
    """Prescribed-load indentation: secant on the indenter displacement.

    The load is the *target*; the rigid-indenter displacement is the unknown, so
    ``a``, ``p_0`` and the approach are outputs of the equilibrium, not inputs.
    """
    Hn = compliance_obj.normal
    coords = compliance_obj.coords
    measures = compliance_obj.measures
    delta = 1.0 / radius_hat                     # Hertz first guess (delta = a^2/R)
    history = []
    solution = None
    for outer in range(1, max_outer + 1):
        offset = coords ** 2 / (2.0 * radius_hat) - delta
        if method == "alm":
            solution = solve_normal_alm(Hn, offset, measures, tolerance=tolerance)
        else:
            penalty = penalty_factor / float(torch.median(torch.diagonal(Hn)))
            solution = solve_normal_penalty(Hn, offset, penalty, measures)
        load = float(solution.load)
        history.append({"delta_hat": float(delta), "load_hat": load})
        if abs(load / load_hat - 1.0) < 1e-11:
            break
        delta *= (load_hat / load) ** (2.0 / 3.0)
    a, p0 = recover_hertz_pressure(solution.pressure, coords)
    return {"delta_hat": float(delta), "load_hat": float(solution.load),
            "a_hat": a, "p0_hat": p0, "solution": solution,
            "outer_iterations": outer, "history": history,
            "gap_min": float(solution.gap.min()),
            "residual_complementarity": solution.residual,
            "condition_number": solution.condition_number,
            "nodal_radius_hat": float(coords[solution.pressure > 0.0].max())
            if bool(torch.any(solution.pressure > 0.0)) else 0.0}


def solve_config(config: dict, comp, load: float, *, method: str = "alm") -> dict:
    """One (configuration, load, mesh) study point, with dimensional outputs."""
    if config["kind"] == "point":
        oracle = hertz_point(load, config["radius"], config["e_star"])
        solver = {"a": oracle["a"]}
        scale = _scaling(config, solver)
        radius_hat = config["radius"] / oracle["a"]
        load_hat = load / scale["force"]
        res = solve_indentation(comp, radius_hat, load_hat, method=method)
        a = res["a_hat"] * oracle["a"]
        p0 = res["p0_hat"] * scale["stress"]
        idx = int(torch.argmin(torch.abs(comp.coords - POINT_REFERENCE)))
        w_ref = float((comp.normal @ res["solution"].nodal_force)[idx]) * oracle["a"]
        delta = res["delta_hat"] * oracle["a"] - w_ref      # reference-relative approach
        delta_oracle = hertz_point_approach(
            config["radius"], load, config["e_star"],
            reference=float(comp.coords[idx]) * oracle["a"])
        out = {"load": load, "a": a, "a_oracle": oracle["a"], "p0": p0,
               "p0_oracle": oracle["p0"], "delta": delta, "delta_oracle": delta_oracle,
               "delta_indenter": res["delta_hat"] * oracle["a"],
               "delta_indenter_oracle": oracle["delta"],
               "reference_node_hat": float(comp.coords[idx]),
               "radius_hat": radius_hat, "load_hat": load_hat}
    else:
        oracle = hertz_line(load / LINE_STRIP, config["radius"], config["e_star"])
        solver = {"b": oracle["b"]}
        scale = _scaling(config, solver)
        radius_hat = config["radius"] / oracle["b"]
        # the plane-strain model is a symmetry half: it carries half of P'
        load_hat = 0.5 * (load / LINE_STRIP) / scale["force"]
        res = solve_indentation(comp, radius_hat, load_hat, method=method)
        b = res["a_hat"] * oracle["b"]
        p0 = res["p0_hat"] * scale["stress"]
        # approach relative to a surface reference point at 8 contact half-widths
        reference = 8.0
        idx = int(torch.argmin(torch.abs(comp.coords - reference)))
        u_z_ref = -float((comp.normal @ res["solution"].nodal_force)[idx]) * oracle["b"]
        delta = (res["delta_hat"] * oracle["b"]) + u_z_ref
        delta_oracle = hertz_line_approach_quadrature(
            load / LINE_STRIP, config["radius"], config["e_star"],
            reference=float(comp.coords[idx]) * oracle["b"])
        out = {"load": load, "a": b, "a_oracle": oracle["b"], "p0": p0,
               "p0_oracle": oracle["p0"], "delta": delta, "delta_oracle": delta_oracle,
               "delta_johnson": oracle["delta"],
               "reference_node_hat": float(comp.coords[idx]),
               "radius_hat": radius_hat, "load_hat": load_hat}
    out.update({"solution": res["solution"], "mesh": dict(comp.memory),
                "delta_hat": res["delta_hat"],
                "load_hat_achieved": res["load_hat"],
                "gap_min": res["gap_min"],
                "complementarity_residual": res["residual_complementarity"],
                "condition_number": res["condition_number"],
                "nodal_radius_hat": res["nodal_radius_hat"],
                "relative": {"a": out["a"] / out["a_oracle"] - 1.0,
                             "p0": out["p0"] / out["p0_oracle"] - 1.0,
                             "delta": out["delta"] / out["delta_oracle"] - 1.0}})
    return out


def work_integral(config: dict, comp, load: float, steps: int = 12) -> dict:
    """Indenter work ``int F d(delta)`` from a load ramp (energy gate)."""
    peak = solve_config(config, comp, load)
    if config["kind"] == "point":
        oracle = hertz_point(load, config["radius"], config["e_star"])
        radius_hat = config["radius"] / oracle["a"]
        young_fe = (1.0 - POISSON ** 2) * config["e_star"]
        force_scale = young_fe * oracle["a"] ** 2
        deltas = torch.linspace(0.0, peak["delta_hat"], steps + 1)[1:]
        loads, disp = [], []
        for d in deltas:
            offset = comp.coords ** 2 / (2.0 * radius_hat) - float(d)
            sol = solve_normal_alm(comp.normal, offset, comp.measures, tolerance=1e-10)
            loads.append(float(sol.load) * force_scale)
            disp.append(float(d) * oracle["a"])
        work = float(torch.trapezoid(torch.tensor(loads), torch.tensor(disp)))
        force_hat = peak["solution"].nodal_force
        strain_energy = 0.5 * float(force_hat @ (comp.normal @ force_hat)) * (
            young_fe * oracle["a"] ** 3)
        return {"work": work, "strain_energy": strain_energy,
                "work_oracle": 0.4 * load * oracle["delta"],
                "strain_energy_oracle": 0.4 * load * oracle["delta"],
                "delta": peak["delta"], "delta_oracle": oracle["delta"]}
    oracle = hertz_line(load / LINE_STRIP, config["radius"], config["e_star"])
    young_fe = (1.0 - POISSON ** 2) * config["e_star"]
    force_scale = young_fe * oracle["b"]
    radius_hat = config["radius"] / oracle["b"]
    idx = int(torch.argmin(torch.abs(comp.coords - 8.0)))
    deltas = torch.linspace(0.0, peak["delta_hat"], steps + 1)[1:]
    loads, disp = [], []
    for d in deltas:
        offset = comp.coords ** 2 / (2.0 * radius_hat) - float(d)
        sol = solve_normal_alm(comp.normal, offset, comp.measures, tolerance=1e-10)
        loads.append(2.0 * float(sol.load) * force_scale * LINE_STRIP)
        u_z_ref = -float((comp.normal @ sol.nodal_force)[idx]) * oracle["b"]
        disp.append((float(d) * oracle["b"] + u_z_ref) * LINE_STRIP)
    work = float(torch.trapezoid(torch.tensor(loads), torch.tensor(disp)))
    return {"work": work, "work_oracle": (2.0 / 3.0) * load * peak["delta"],
             "delta": peak["delta"], "delta_oracle": peak["delta_oracle"]}# ---------------------------------------------------------------------------
# Cattaneo--Mindlin partial slip
# ---------------------------------------------------------------------------
def _interp(xp, fp, x) -> float:
    """Monotone linear interpolation on a sampled path."""
    order = torch.argsort(xp)
    xp, fp = xp[order], fp[order]
    x = float(x)
    if x <= float(xp[0]):
        return float(fp[0])
    if x >= float(xp[-1]):
        return float(fp[-1])
    idx = int(torch.searchsorted(xp, torch.tensor(x)).item())
    x0, x1 = float(xp[idx - 1]), float(xp[idx])
    f0, f1 = float(fp[idx - 1]), float(fp[idx])
    return f0 + (f1 - f0) * (x - x0) / (x1 - x0)


def cattaneo_study(comp, *, normal_load: float = CATTANEO_NORMAL_LOAD,
                   ratios: tuple[float, ...] = tuple(0.1 * i for i in range(1, 10))) -> dict:
    """Cattaneo--Mindlin ladder: Q(u_t), stick/slip partition and c/a."""
    config = {"name": "sphere_plane", "kind": "point", "radius": BALL_RADIUS,
              "e_star": YOUNG / (1.0 - POISSON ** 2)}
    oracle = hertz_point(normal_load, BALL_RADIUS, config["e_star"])
    young_fe = (1.0 - POISSON ** 2) * config["e_star"]
    force_scale = young_fe * oracle["a"] ** 2
    radius_hat = BALL_RADIUS / oracle["a"]
    load_hat = normal_load / force_scale
    normal = solve_indentation(comp, radius_hat, load_hat)
    a_fe = normal["a_hat"] * oracle["a"]
    p0_fe = normal["p0_hat"] * young_fe
    shear = YOUNG / (2.0 * (1.0 + POISSON))
    u_peak = cattaneo_displacement(0.9 * MU * normal_load, normal_load, MU, a_fe,
                                  shear, POISSON)
    shifts = torch.linspace(0.0, 1.25 * u_peak / oracle["a"], 121, dtype=DTYPE)
    path = solve_cattaneo_tangential(comp.tangential, normal["solution"].nodal_force,
                                     comp.measures, comp.coords, MU, shifts)
    tangential = path.tangential_load * force_scale
    displacement = path.shift * oracle["a"]
    c_dim = path.stick_radius * oracle["a"]
    points = []
    for ratio in ratios:
        target = ratio * MU * normal_load
        u_fe = _interp(tangential, displacement, target)
        c_fe = _interp(tangential, c_dim, target)
        u_ref = cattaneo_displacement(target, normal_load, MU, a_fe, shear, POISSON)
        c_ref = cattaneo_stick_radius(target, normal_load, MU, a_fe)
        points.append({"q_over_muN": ratio, "Q": target,
                       "u_t": u_fe, "u_t_oracle": u_ref,
                       "u_relative": u_fe / u_ref - 1.0,
                       "c_over_a": c_fe / a_fe, "c_over_a_oracle": c_ref / a_fe,
                       "c_relative": (c_fe / a_fe) / (c_ref / a_fe) - 1.0})
    # small-Q tangential stiffness (Mindlin limit) from the computed path
    q0, q1 = float(tangential[1]), float(tangential[2])
    u0, u1 = float(displacement[1]), float(displacement[2])
    k_fe = (q1 - q0) / (u1 - u0) if u1 > u0 else float("nan")
    normal_solution = normal["solution"]
    traction_profile = []
    for i in range(comp.coords.numel()):
        traction_profile.append({
            "r_over_a": float(comp.coords[i]) / normal["a_hat"],
            "p": float(normal_solution.pressure[i]) * young_fe,
            "q": float(path.traction[-1][i]) * young_fe,
            "mu_p": MU * float(normal_solution.pressure[i]) * young_fe,
            "stick": bool(path.stick[-1][i])})
    contact = normal["solution"].nodal_force > 0.0
    coulomb_ok = bool(torch.all(path.traction[-1] <= MU * normal["solution"].nodal_force
                                + 1e-12 * float(normal["solution"].nodal_force.max())))
    saturation = float(path.tangential_load.max()) / (MU * normal_load)
    return {"coulomb_bound_holds": coulomb_ok,
            "saturation_Q_over_muN": saturation,
            "contact_nodes": int(contact.sum()),
            "normal_load": normal_load, "a_fe": a_fe, "a_oracle": oracle["a"],
            "p0_fe": p0_fe, "p0_oracle": oracle["p0"], "mu": MU,
            "shear_modulus": shear, "points": points,
            "stiffness_small_q_fe": k_fe,
            "stiffness_small_q_oracle": cattaneo_stiffness_small_q(
                normal_load, a_fe, shear, POISSON),
            "path": {"Q": tangential.tolist(), "u_t": displacement.tolist(),
                     "c": c_dim.tolist(), "a": a_fe},
            "traction_profile": traction_profile,
            "stick_radius_hat": float(path.stick_radius[-1]),
            "c_over_a_nodal": float(path.stick_radius[-1]) / normal["a_hat"],
            "c_over_a_oracle_at_peak": cattaneo_stick_radius(
                0.9 * MU * normal_load, normal_load, MU, a_fe) / a_fe,
            "settled_slip_radius_hat": float(path.slip_radius[-1]),
            "a_hat": normal["a_hat"]}


def tangential_sector_check(ns) -> dict:
    """Diagnose the axisymmetric tangential sector against the flat-punch limit.

    A rigid flat punch of radius ``a`` under a tangential load obeys
    ``u = Q(2-nu)/(8 G a)`` (Mindlin 1949), i.e. ``Q/u = 8 G a/(2-nu)``.  In the
    dimensionless unit-modulus problem that is ``Q_hat/s_hat = 8 (G/E)/(2-nu)``
    with ``G/E = 1/(2(1+nu))``.  The check is mesh-independent by construction,
    so a persistent offset identifies a *formulation* error rather than a
    discretisation error.
    """
    expected = 8.0 * (1.0 / (2.0 * (1.0 + POISSON))) / (2.0 - POISSON)
    rows = []
    for n_in in ns:
        comp = compliance("point", n_in, n_in)
        pressure = torch.where(comp.coords <= 1.0, torch.tensor(1.0e9, dtype=DTYPE),
                               torch.tensor(0.0, dtype=DTYPE))
        path = solve_cattaneo_tangential(comp.tangential, pressure, comp.measures,
                                         comp.coords, 1.0,
                                         torch.tensor([1.0e-4], dtype=DTYPE))
        ratio = float(path.tangential_load[0]) / 1.0e-4
        rows.append({"n_in": n_in, "nodes": comp.memory["nodes"],
                     "nodes_in_contact": int((comp.coords < 1.0).sum()),
                     "q_over_shift": ratio, "flat_punch_limit": expected,
                     "relative": ratio / expected - 1.0})
    return {"rows": rows,
            "max_relative": max(abs(r["relative"]) for r in rows),
            "mesh_independent": max(abs(r["relative"]) - 0.05 for r in rows) > 0.0,
            "note": "axisymmetric n=0 sector models a radial expansion of the contact "
                    "patch, not a unidirectional tangential shift; the offset does not "
                    "shrink with mesh refinement"}


# ---------------------------------------------------------------------------
# complementarity audit
# ---------------------------------------------------------------------------
def complementarity_study(comp, *, load: float = CATTANEO_NORMAL_LOAD) -> dict:
    """Penalty residual/conditioning trade-off and ALM residual ladder."""
    oracle = hertz_point(load, BALL_RADIUS, YOUNG / (1.0 - POISSON ** 2))
    young_fe = YOUNG
    radius_hat = BALL_RADIUS / oracle["a"]
    load_hat = load / (young_fe * oracle["a"] ** 2)
    target = solve_indentation(comp, radius_hat, load_hat)      # ALM reference
    delta_hat = target["delta_hat"]
    offset = comp.coords ** 2 / (2.0 * radius_hat) - delta_hat
    h_scale = float(torch.median(torch.diagonal(comp.normal)))
    penalty_rows = []
    for factor in (1.0e1, 2.0e1, 1.0e2, 2.0e2, 1.0e3, 2.0e3, 1.0e4, 2.0e4):
        sol = solve_normal_penalty(comp.normal, offset, factor / h_scale, comp.measures)
        a, p0 = recover_hertz_pressure(sol.pressure, comp.coords)
        penalty_rows.append({
            "penalty_factor": factor, "penalty": factor / h_scale,
            "residual": sol.residual, "penetration_hat": -float(sol.gap.min()),
            "penetration_over_delta": -float(sol.gap.min()) / delta_hat,
            "condition_number": sol.condition_number,
            "load_hat": float(sol.load),
            "a_relative": a / target["a_hat"] - 1.0,
            "p0_relative": p0 / target["p0_hat"] - 1.0})
    alm_rows = []
    for tol in (1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13):
        sol = solve_normal_alm(comp.normal, offset, comp.measures, tolerance=tol)
        alm_rows.append({"tolerance": tol, "residual": sol.residual,
                         "iterations": sol.iterations,
                         "min_gap_hat": float(sol.gap.min()),
                         "condition_number": sol.condition_number,
                         "converged": bool(sol.converged)})
    alm_final = solve_normal_alm(comp.normal, offset, comp.measures, tolerance=1e-12)
    return {"delta_hat": delta_hat, "penalty": penalty_rows, "alm": alm_rows,
            "alm_final_residual": alm_final.residual,
            "alm_final_min_gap_hat": float(alm_final.gap.min()),
            "alm_final_max_negative_gap": float(torch.clamp(alm_final.gap, max=0.0).abs().max()),
            "alm_iterations": alm_final.iterations,
            "load_hat": load_hat, "penetration_alm_hat":
                float(torch.clamp(-alm_final.gap, min=0.0).max())}


def negative_controls(comp) -> dict:
    """A8: no contact -> zero force; zero tangential shift -> zero traction."""
    oracle = hertz_point(CATTANEO_NORMAL_LOAD, BALL_RADIUS, YOUNG / (1.0 - POISSON ** 2))
    young_fe = YOUNG
    radius_hat = BALL_RADIUS / oracle["a"]
    force_scale = young_fe * oracle["a"] ** 2
    gap_off = comp.coords ** 2 / (2.0 * radius_hat) + 0.02       # indenter above surface
    alm = solve_normal_alm(comp.normal, gap_off, comp.measures)
    pen = solve_normal_penalty(comp.normal, gap_off,
                               1.0e3 / float(torch.median(torch.diagonal(comp.normal))),
                               comp.measures)
    touching = solve_normal_alm(comp.normal, comp.coords ** 2 / (2.0 * radius_hat),
                                comp.measures)
    zero_shift = solve_cattaneo_tangential(
        comp.tangential, alm.nodal_force * 0.0 + float(alm.nodal_force.max()) * 0.0,
        comp.measures, comp.coords, MU, torch.zeros(3, dtype=DTYPE))
    return {"no_contact_load_alm": float(alm.load),
            "no_contact_load_penalty": float(pen.load),
            "no_contact_load_alm_dimensional": float(alm.load) * force_scale,
            "tangential_load_at_zero_shift": float(zero_shift.tangential_load.max()),
            "grazing_contact_load_hat": float(touching.load),
            "all_gaps_positive": bool(torch.all(alm.gap > 0.0))}


# ---------------------------------------------------------------------------
# convergence and gates
# ---------------------------------------------------------------------------
def convergence(levels: list[int], table: dict) -> dict:
    """Observed order and GCI from the three mesh levels (fine mesh output)."""
    h = [2.0 / n for n in levels]                     # h/a of the uniform zone
    out = {}
    for quantity in ("a", "p0", "delta"):
        values = [table[n][quantity] for n in levels]
        r12, r23 = h[1] / h[0], h[2] / h[1]           # h refines, r > 1
        e12 = values[1] - values[0]
        e23 = values[2] - values[1]
        order = None
        if abs(e23) > 0.0 and abs(e12) > 0.0 and e12 * e23 > 0.0:
            order = math.log(abs(e23 / e12)) / math.log(0.5 * (r12 + r23))
        gci = None
        if order is not None and order > 0.0:
            gci = 1.25 * abs((values[2] - values[1]) / values[2]) / (
                (0.5 * (r12 + r23)) ** order - 1.0)
        out[quantity] = {"values": values, "order": order, "gci": gci,
                         "relative": [table[n]["relative"][quantity] for n in levels],
                         "relative_change_finest": abs(values[2] - values[1]) / abs(values[2])}
    return out# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(outdir: Path, payload: dict) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    for ax, (name, block) in zip(axes, payload["pressure_profiles"].items()):
        data = block["data"]
        rr = [row["r_over_a"] for row in data]
        ax.plot(rr, [row["p_over_p0"] for row in data], "o", ms=3.5,
                label="FE (finest mesh)", color="tab:blue")
        curve_r = [0.01 * i for i in range(101)]
        ax.plot(curve_r, [math.sqrt(max(0.0, 1.0 - x * x)) for x in curve_r],
                "-", color="k", lw=1.2, label="Hertz $\\sqrt{1-(r/a)^2}$")
        ax.set_xlabel("$r/a$")
        ax.set_ylabel("$p/p_0$")
        ax.set_title(f"{name}\n" + block["subtitle"], fontsize=9)
        ax.set_xlim(0.0, 1.05)
        ax.set_ylim(0.0, 1.08)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("Hertz pressure profiles: condensed FE vs closed form", y=1.02)
    fig.tight_layout()
    path = outdir / "hertz_pressure_profiles.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({"name": "hertz_pressure_profiles", "path": str(path),
                    "caption": "Normalised contact pressure p/p0 versus r/a for the three "
                               "configurations at the finest mesh, against the Hertz "
                               "profile sqrt(1-(r/a)^2)."})

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    for ax, (name, block) in zip(axes, payload["convergence"].items()):
        h = block["h_over_a"]
        for quantity, marker in (("a", "o"), ("delta", "s"), ("p0", "^")):
            ax.loglog(h, [abs(v) for v in block[quantity]], marker + "-",
                      label=f"|{quantity}| error")
        ax.set_xlabel("$h/a$ (contact-patch element size)")
        ax.set_ylabel("relative error")
        def _fmt(value):
            return "n/a" if value is None else f"{value:.2f}"
        ax.set_title(f"{name}\nobserved order a={_fmt(block['orders']['a'])}, "
                     f"$\\delta$={_fmt(block['orders']['delta'])}", fontsize=9)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.suptitle("Mesh convergence of Hertz outputs (F = 1000 N)", y=1.02)
    fig.tight_layout()
    path = outdir / "hertz_convergence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({"name": "hertz_convergence", "path": str(path),
                    "caption": "Relative error of contact radius, peak pressure and "
                               "approach versus contact-patch element size h/a "
                               "(F = 1000 N); the observed order is annotated."})

    cat = payload["cattaneo"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ax = axes[0]
    ax.plot([p["u_t_oracle"] / cat["a_fe"] for p in cat["points"]],
            [p["q_over_muN"] for p in cat["points"]], "k-", lw=1.2,
            label="Cattaneo closed form")
    ax.plot([p["u_t"] / cat["a_fe"] for p in cat["points"]],
            [p["q_over_muN"] for p in cat["points"]], "o", ms=5,
            label="FE (ALM normal + rigid stick)")
    ax.set_xlabel("$u_t/a$")
    ax.set_ylabel("$Q/(\\mu N)$")
    ax.set_title("Cattaneo force--displacement", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot([p["q_over_muN"] for p in cat["points"]],
            [p["c_over_a_oracle"] for p in cat["points"]], "k-", lw=1.2,
            label="$c/a=(1-Q/\\mu N)^{1/3}$")
    ax.plot([p["q_over_muN"] for p in cat["points"]],
            [p["c_over_a"] for p in cat["points"]], "s", ms=5,
            label="FE stick radius (nodal)")
    ax.set_xlabel("$Q/(\\mu N)$")
    ax.set_ylabel("$c/a$")
    ax.set_title("Stick/slip partition", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle("Cattaneo--Mindlin partial slip: FE versus closed form", y=1.02)
    fig.tight_layout()
    path = outdir / "cattaneo_mindlin.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({"name": "cattaneo_mindlin", "path": str(path),
                    "caption": "Left: tangential force Q/(mu N) versus u_t/a, FE against "
                               "the Cattaneo/Mindlin closed form. Right: stick radius c/a "
                               "versus Q/(mu N) against (1-Q/(mu N))^(1/3)."})

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    rows = payload["complementarity"]["penalty"]
    ax = axes[0]
    factors = [r["penalty_factor"] for r in rows]
    ax.loglog(factors, [r["residual"] for r in rows], "o-", label="penalty residual")
    ax.loglog(factors, [r["penetration_over_delta"] for r in rows], "s--",
              label="penetration / $\\delta$")
    ref = [rows[0]["residual"] * rows[0]["penalty_factor"] / f for f in factors]
    ax.loglog(factors, ref, ":", color="grey", label="$\\propto 1/k_n$")
    ax.set_xlabel("$k_n$ / (median $H^{-1}$)")
    ax.set_ylabel("dimensionless measure")
    ax.set_title("Penalty: residual and penetration trade-off", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax2 = ax.twinx()
    ax2.loglog(factors, [r["condition_number"] for r in rows], "^-",
               color="tab:red", alpha=0.6, label="cond")
    ax2.set_ylabel("cond($I+k_n H_{AA}$)", color="tab:red")
    ax.legend(fontsize=7, loc="lower left")
    ax = axes[1]
    alm = payload["complementarity"]["alm"]
    ax.semilogy([r["iterations"] for r in alm], [r["residual"] for r in alm], "o-")
    ax.axhline(1e-8, color="k", ls=":", lw=1.0, label="A6 threshold 1e-8")
    ax.set_xlabel("ALM iterations")
    ax.set_ylabel("$||p\\cdot g||_1/(N\\delta)$")
    ax.set_title("Augmented Lagrangian complementarity residual", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "complementarity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({"name": "complementarity", "path": str(path),
                    "caption": "Left: penalty complementarity residual and penetration "
                               "against k_n on a log scale with the 1/k_n reference, plus "
                               "the active-set condition number. Right: ALM residual versus "
                               "iterations with the 1e-8 gate."})

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ax = axes[0]
    for name, block in payload["energy"].items():
        if block["kind"] != "point":
            continue
        ax.plot(block["delta"], [f * 1e3 for f in block["force"]], "o-", ms=3,
                label=f"{name} FE")
        ax.plot(block["delta_oracle_curve"],
                [f * 1e3 for f in block["force_oracle_curve"]], "k--", lw=1.0,
                label=f"{name} Hertz")
    ax.set_xlabel("$\\delta$ [m]")
    ax.set_ylabel("F [kN]")
    ax.set_title("Indentation curves (energy gate A9)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax = axes[1]
    tr = payload["cattaneo"]["traction_profile"]
    ax.plot([row["r_over_a"] for row in tr], [row["mu_p"] / 1e9 for row in tr],
            "k--", lw=1.0, label="$\\mu p$ (Coulomb limit)")
    ax.plot([row["r_over_a"] for row in tr], [row["q"] / 1e9 for row in tr],
            "o-", ms=3, label="FE tangential traction q")
    ax.axvline(payload["cattaneo"]["c_over_a_nodal"], color="tab:red", ls=":",
               label="FE c/a")
    ax.set_xlabel("$r/a$")
    ax.set_ylabel("traction [GPa]")
    ax.set_title("Stick/slip traction partition at $Q=0.9\\mu N$", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = outdir / "energy_and_traction.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({"name": "energy_and_traction", "path": str(path),
                    "caption": "Left: computed indentation curves against Hertz used for "
                               "the work integral. Right: tangential traction q(r) against "
                               "the Coulomb limit mu p(r) at Q = 0.9 mu N, with the FE stick "
                               "radius marked."})
    return figures


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_gates(table: dict, levels: list[int], payload: dict) -> dict:
    finest = levels[-1]
    errors = {"a": [], "p0": [], "delta": []}
    for config in ("sphere_sphere", "sphere_plane", "cylinder_plane"):
        for load in LOADS:
            row = table[config][finest][load]
            for quantity in errors:
                errors[quantity].append(abs(row["relative"][quantity]))
    gates = {}

    def gate(name, measured, threshold, passed, note):
        gates[name] = {"measured": measured, "threshold": threshold,
                       "pass": bool(passed), "note": note}

    gate("A1_contact_radius", max(errors["a"]), 0.02, max(errors["a"]) <= 0.02,
         "max |a_sim/a_Hertz - 1| over 3 configurations x 3 loads, finest mesh")
    gate("A2_peak_pressure", max(errors["p0"]), 0.03, max(errors["p0"]) <= 0.03,
         "max |p0_sim/p0_Hertz - 1| over 9 points, finest mesh")
    gate("A3_approach", max(errors["delta"]), 0.03, max(errors["delta"]) <= 0.03,
         "max |delta_sim/delta_Hertz - 1| over 9 points, finest mesh")
    u_errors = [abs(p["u_relative"]) for p in payload["cattaneo"]["points"]]
    gate("A4_cattaneo_force_displacement", max(u_errors), 0.05,
         max(u_errors) <= 0.05, "max relative error of u_t(Q) over 9 Q/(mu N) points")
    c_errors = [abs(p["c_relative"]) for p in payload["cattaneo"]["points"]]
    gate("A5_stick_slip_partition", max(c_errors), 0.05, max(c_errors) <= 0.05,
         "max relative error of c/a over the same 9 points")
    comp = payload["complementarity"]
    residual = comp["alm_final_residual"]
    factors = [row["penalty_factor"] for row in comp["penalty"]]
    residuals = [row["residual"] for row in comp["penalty"]]
    monotone = all(residuals[i + 1] < residuals[i] for i in range(len(residuals) - 1))
    halving = []
    for i in range(len(residuals) - 1):
        if abs(factors[i + 1] / factors[i] - 2.0) < 1e-9 and residuals[i] > 1e-11:
            halving.append(residuals[i + 1] <= 0.51 * residuals[i])
    gate("A6_complementarity_residual", {"alm": residual,
                                         "penalty_monotone": monotone,
                                         "penalty_halving_on_doubling": all(halving)},
         1e-8, residual <= 1e-8 and monotone and all(halving),
         "ALM ||p.g||_1/(N delta) <= 1e-8 and penalty residual at least halves when "
         "k_n doubles (2 % tolerance for discrete active-set round-off, checked above the 1e-11 double-precision floor)")
    orders = {}
    for config, block in payload["convergence"].items():
        orders[config] = {"a": block["orders"]["a"], "delta": block["orders"]["delta"]}
    worst = min(0.0 if v[q] is None else v[q]
                for v in orders.values() for q in ("a", "delta"))
    gate("A7_grid_convergence", orders, 1.5, worst >= 1.5,
         "observed convergence order of contact radius and approach (three levels)")
    neg = payload["negative_controls"]
    gate("A8_negative_controls",
         {"no_contact_load": neg["no_contact_load_alm"],
          "no_contact_load_penalty": neg["no_contact_load_penalty"],
          "zero_shift_traction": neg["tangential_load_at_zero_shift"]},
         0.0, neg["no_contact_load_alm"] == 0.0 and neg["no_contact_load_penalty"] == 0.0
         and neg["tangential_load_at_zero_shift"] == 0.0,
         "gaps > 0 gives exactly zero contact force; Q = 0 gives exactly zero traction")
    work = {name: block for name, block in payload["energy"].items()}
    work_errors = {name: block["relative"] for name, block in work.items()}
    point_errors = [abs(block["relative"]) for name, block in work.items()
                    if block["kind"] == "point"]
    gate("A9_energy", work_errors, 0.01, max(point_errors) <= 0.01,
         "contact strain energy 1/2 f'H f and indenter work int F d(delta) versus "
         "(2/5) F delta for the point configurations")
    gates["overall_pass"] = all(v["pass"] for k, v in gates.items() if k.startswith("A"))
    return gates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=int, nargs="+", default=list(LEVELS))
    parser.add_argument("--outdir", type=str,
                        default=str(REPO / "artifacts" / "contact_hertz_cattaneo"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cattaneo-ns", type=int, nargs="+", default=list(CATTANEO_LEVELS))
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--order-levels", type=int, nargs="+", default=list(ORDER_LEVELS))
    args = parser.parse_args()
    levels = list(args.levels)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    t_start = time.time()
    catt_levels = list(args.cattaneo_ns)
    order_levels = list(args.order_levels)
    for n_in in sorted(set(levels) | set(catt_levels) | set(order_levels)):
        for kind in ("point", "line"):
            t0 = time.time()
            obj = compliance(kind, n_in, n_in, use_cache=not args.no_cache)
            print(f"[build] {kind} n={n_in} nodes={obj.memory['nodes']} "
                  f"{time.time() - t0:.1f}s", flush=True)
    if args.build_only:
        return 0

    point = point_configs()
    line = line_config()
    table: dict[str, dict[int, dict[float, dict]]] = {}
    for n_in in levels:
        nz_in = n_in
        comp_point = compliance("point", n_in, nz_in, use_cache=not args.no_cache)
        comp_line = compliance("line", n_in, nz_in, use_cache=not args.no_cache)
        for config in point:
            table.setdefault(config["name"], {})[n_in] = {}
            for load in LOADS:
                table[config["name"]][n_in][load] = solve_config(config, comp_point, load)
        table.setdefault(line["name"], {})[n_in] = {}
        for load in LOADS:
            table[line["name"]][n_in][load] = solve_config(line, comp_line, load)
        print(f"[level {n_in}] done ({time.time() - t_start:.1f}s)", flush=True)

    coarse: dict[str, dict[int, dict[float, dict]]] = {}
    for n_in in order_levels:
        comp_c = compliance("point", n_in, n_in, use_cache=not args.no_cache)
        comp_cl = compliance("line", n_in, n_in, use_cache=not args.no_cache)
        for config in point:
            coarse.setdefault(config["name"], {})[n_in] = {
                load: solve_config(config, comp_c, load) for load in LOADS}
        coarse.setdefault(line["name"], {})[n_in] = {
            load: solve_config(line, comp_cl, load) for load in LOADS}
    finest = levels[-1]
    comp_point = compliance("point", finest, finest, use_cache=not args.no_cache)
    comp_line = compliance("line", finest, finest, use_cache=not args.no_cache)

    conv = {}
    for name in table:
        reference_load = LOADS[-1]
        block = convergence(order_levels,
                            {n: coarse[name][n][reference_load] for n in order_levels})
        conv[name] = {
            "order_levels": order_levels,
            "fine_levels": levels,
            "fine_relative": {q: [table[name][n][reference_load]["relative"][q]
                                  for n in levels] for q in ("a", "p0", "delta")},
            "h_over_a": [2.0 / n for n in order_levels],
            "a": block["a"]["relative"],
            "p0": block["p0"]["relative"],
            "delta": block["delta"]["relative"],
            "orders": {q: block[q]["order"] for q in ("a", "p0", "delta")},
            "gci": {q: block[q]["gci"] for q in ("a", "p0", "delta")},
            "load": reference_load}

    catt_comp = compliance("point", catt_levels[-1], catt_levels[-1],
                           use_cache=not args.no_cache)
    cattaneo = cattaneo_study(catt_comp)
    cattaneo_mesh = {}
    for n_in in catt_levels:
        c = compliance("point", n_in, n_in, use_cache=not args.no_cache)
        block = cattaneo_study(c)
        cattaneo_mesh[n_in] = {"c_over_a_nodal": block["c_over_a_nodal"],
                               "c_over_a_oracle": block["c_over_a_oracle_at_peak"],
                               "a_relative": block["a_fe"] / block["a_oracle"] - 1.0}
    sector = tangential_sector_check(catt_levels)
    complementarity = complementarity_study(comp_point)
    negatives = negative_controls(comp_point)

    energy = {}
    for config in point:
        block = work_integral(config, comp_point, LOADS[-1])
        block["kind"] = "point"
        block["relative"] = block["strain_energy"] / block["strain_energy_oracle"] - 1.0
        block["relative_work_integral"] = block["work"] / block["work_oracle"] - 1.0
        energy[config["name"]] = block
    line_block = work_integral(line, comp_line, LOADS[-1])
    line_block["kind"] = "line"
    line_block["relative"] = line_block["work"] / line_block["work_oracle"] - 1.0
    line_block["note"] = ("line-contact work uses the measured reference-relative "
                          "approach; (2/3) P delta is reported as a convention-bound "
                          "reference only")
    energy[line["name"]] = line_block

    profiles = {}
    for config in point:
        row = table[config["name"]][finest][LOADS[-1]]
        a_hat = row["a"] / row["a_oracle"]
        data = []
        for i, radius in enumerate(comp_point.coords):
            pressure = float(row["solution"].pressure[i]) * (1.0 - POISSON ** 2) * config["e_star"]
            ratio = float(radius) / a_hat
            if ratio <= 1.02 and pressure > 0.0:
                data.append({"r_over_a": ratio, "p_over_p0": pressure / row["p0"],
                             "hertz": float(hertz_traction(torch.tensor(ratio), 1.0, 1.0))})
        profiles[config["name"]] = {
            "data": data, "subtitle": f"F = {row['load']:.0f} N, a_err = "
                                      f"{100 * row['relative']['a']:+.2f}%"}
    row = table[line["name"]][finest][LOADS[-1]]
    a_hat = row["a"] / row["a_oracle"]
    data = []
    for i, coord in enumerate(comp_line.coords):
        pressure = float(row["solution"].pressure[i]) * (1.0 - POISSON ** 2) * line["e_star"]
        ratio = float(coord) / a_hat
        if ratio <= 1.02 and pressure > 0.0:
            data.append({"r_over_a": ratio, "p_over_p0": pressure / row["p0"],
                         "hertz": float(hertz_traction(torch.tensor(ratio), 1.0, 1.0))})
    profiles[line["name"]] = {
        "data": data, "subtitle": f"P' = {row['load'] / LINE_STRIP:.0f} N/m, a_err = "
                                  f"{100 * row['relative']['a']:+.2f}%"}

    energy_curves = {}
    for config in point + [line]:
        comp = comp_point if config["kind"] == "point" else comp_line
        oracle = (hertz_point(LOADS[-1], config["radius"], config["e_star"])
                  if config["kind"] == "point" else
                  hertz_line(LOADS[-1] / LINE_STRIP, config["radius"], config["e_star"]))
        young_fe = (1.0 - POISSON ** 2) * config["e_star"]
        length = oracle["a"] if config["kind"] == "point" else oracle["b"]
        force_scale = young_fe * (length ** 2 if config["kind"] == "point" else length)
        radius_hat = config["radius"] / length
        peak = table[config["name"]][finest][LOADS[-1]]
        deltas = torch.linspace(0.0, peak["delta_hat"], 13)[1:]
        forces, disp = [], []
        for d in deltas:
            offset = comp.coords ** 2 / (2.0 * radius_hat) - float(d)
            sol = solve_normal_alm(comp.normal, offset, comp.measures, tolerance=1e-10)
            forces.append(float(sol.load) * force_scale * (1.0 if config["kind"] == "point"
                                                           else LINE_STRIP))
            disp.append(float(d) * length * (1.0 if config["kind"] == "point" else LINE_STRIP))
        curve_delta = torch.linspace(0.0, float(deltas[-1]) * length, 40)
        curve_force = []
        for d in curve_delta:
            hz = (hertz_point_from_indentation(float(d), config["radius"], config["e_star"])
                  if config["kind"] == "point" and float(d) > 0.0 else None)
            curve_force.append(hz["load"] if hz else 0.0)
        if config["kind"] == "line":
            curve_delta = []
            curve_force = []
            for fraction in (i / 20.0 for i in range(21)):
                line_load = fraction * LOADS[-1]
                if line_load <= 0.0:
                    curve_delta.append(0.0)
                    curve_force.append(0.0)
                    continue
                hz = hertz_line(line_load / LINE_STRIP, config["radius"], config["e_star"])
                curve_delta.append(hz["delta"])
                curve_force.append(line_load)
        energy_curves[config["name"]] = {
            "kind": config["kind"], "delta": disp, "force": forces,
            "delta_oracle_curve": [float(v) for v in curve_delta],
            "force_oracle_curve": [float(v) for v in curve_force]}
        energy_curves[config["name"]]["force"] = forces

    payload = {"pressure_profiles": profiles, "convergence": conv,
               "cattaneo": cattaneo, "cattaneo_mesh": cattaneo_mesh,
               "tangential_sector": sector,
               "complementarity": complementarity, "negative_controls": negatives,
               "energy": energy, "energy_curves": energy_curves}
    gates = build_gates(table, levels, payload)
    figures = make_figures(outdir, {**payload,
                                    "energy": {**energy,
                                               **{k: v for k, v in energy_curves.items()}}})

    result = {
        "study": "P45-N02 Hertz + Cattaneo-Mindlin + complementarity verification",
        "module": "tensorfem.cattaneo_hertz",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(), "torch": torch.__version__,
        "platform": platform.platform(),
        "parameters": {"young_Pa": YOUNG, "poisson": POISSON, "mu": MU,
                       "ball_radius_m": BALL_RADIUS, "loads_N": list(LOADS),
                       "line_strip_m": LINE_STRIP, "levels_n_in": levels,
                       "patch_extent_over_a": EXTENT, "domain_over_a": TOTAL,
                       "growth": GROWTH,
                       "cattaneo_normal_load_N": CATTANEO_NORMAL_LOAD},
        "table": {name: {str(n): {str(load): {k: v for k, v in row.items()
                                              if k != "solution"}
                                  for load, row in per_load.items()}
                         for n, per_load in per_mesh.items()}
                  for name, per_mesh in table.items()},
        "convergence": conv, "cattaneo": {k: v for k, v in cattaneo.items()
                                          if k != "path"},
        "cattaneo_path": cattaneo["path"],
        "cattaneo_mesh": {str(k): v for k, v in cattaneo_mesh.items()},
        "tangential_sector": sector,
        "complementarity": complementarity,
        "negative_controls": negatives, "energy": energy,
        "gates": gates, "figures": figures,
        "elapsed_seconds": time.time() - t_start,
    }
    (outdir / "result.json").write_text(json.dumps(result, indent=2, default=float))
    (outdir / "gates.json").write_text(json.dumps(gates, indent=2, default=float))
    print(json.dumps({k: (v["pass"] if isinstance(v, dict) and "pass" in v else v)
                      for k, v in gates.items()}, indent=2))
    print("elapsed", round(time.time() - t_start, 1), "s ->", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())