#!/usr/bin/env python3
"""Curved-interface convergence statement for the low-order penalty mortar coupling.

Anchor GAP-FEM-MORTAR (acceptance: "mortar contact algorithm + convergence
verification"). `src/tensorfem/mortar_contact3d.py` ships *planar* patch tests;
`docs/mortar-two-block-segment-validation.md` says of them explicitly: "a patch
test, not a convergence statement". This script supplies that convergence
statement for a curved interface, with an error **budget** split into its three
physically distinct sources.

Problem (one undeformed configuration, pure penalty, no friction):

    master: circular arc, radius R, apex at z = delta, faceted into n_master
            chords inscribed in the circle
    slave : planar plate z = phi*x, Gauss-quadratured into n_slave QUAD4 facets
    pen   : penetration measured along the *master facet normal*
            (pen = (z_chord - z_slave)/sqrt(1+m^2), m = chord slope) -- this is
            what the code does: pen = -dot(p - q, n_facet)

Three independent references are compared against the discrete result:

  oracle_radial(tilt)           exact circle, continuum limit. Adaptive
                                quadrature of |p-C| - R with radial normals.
                                This is the *truth* for h -> 0.
  oracle_facet(tilt, n_master)  exact integral of the *faceted* description
                                (piecewise-linear penetration, piecewise
                                constant facet normals), 8-point Gauss per
                                clipped facet. Independent of torch and of the
                                mortar assembly.
  paraboloid closed form        F = (4/3) eps_n W a delta, a = sqrt(2 R delta)
                                (tilt = 0, O(delta/R) limit) -- cross-check of
                                oracle_radial.

Error budget (measured, not assumed):

  geometric faceting   |FE - oracle_radial|   -> must fall as O(h_master^2)
  quadrature           |FE - oracle_facet |   -> slave mesh must resolve the
                                                master kinks; decays with
                                                h_slave, either sign
  normal convention    |oracle_facet - oracle_radial| -> O((a/R)^2) plateau,
                                                *not* a discretisation error:
                                                the code measures a normal gap,
                                                classical formulas use a
                                                vertical gap

Outputs `docs/mortar-curved-convergence/convergence.json` and figures under
`docs/figures/`.

Run:  PYTHONPATH=src python3 benchmarks/mortar_curved_interface_convergence.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from tensorfem.contact3d import project_point_to_facets
from tensorfem.mortar_contact3d import _quadrature, integrate_mortar_contact

torch.set_default_dtype(torch.float64)

OUT_DIR = Path("docs/mortar-curved-convergence")
FIG_DIR = Path("docs/figures")

R = 1.0             # master cylinder radius [m]
DELTA = 0.01        # apex penetration of the undeformed configuration [m]
W = 0.02            # slave plate width in y [m]
L = 0.20            # slave half length in x [m]
EPS_N = 1.0e6       # normal penalty [N/m^3]
MASTER_HALF = 0.35  # master facet band half extent in x [m]
MASTER_W = 0.06     # master facet band width in y [m]
TILT = 0.05         # slope of the tilted slave plate in the asymmetric series

_GAUSS8 = np.polynomial.legendre.leggauss(8)


# ---------------------------------------------------------------- geometry ----
def master_nodes(n_facets: int, *, half: float = MASTER_HALF, radius: float = R,
                 apex: float = DELTA) -> tuple[list[float], list[float]]:
    theta_max = math.asin(half / radius)
    theta = [-theta_max + i * 2.0 * theta_max / n_facets for i in range(n_facets + 1)]
    centre = apex - radius
    return ([radius * math.sin(t) for t in theta],
            [centre + radius * math.cos(t) for t in theta])


def master_cylinder(n_facets: int, *, half: float = MASTER_HALF,
                    width: float = MASTER_W, radius: float = R,
                    apex: float = DELTA) -> tuple[torch.Tensor, torch.Tensor]:
    """Faceted convex master: apex at z=apex, inscribed chords, normals up."""
    xs, zs = master_nodes(n_facets, half=half, radius=radius, apex=apex)
    verts = [[x, 0.0, z] for x, z in zip(xs, zs)]
    verts += [[x, width, z] for x, z in zip(xs, zs)]
    row = n_facets + 1
    faces = [[i, i + 1, row + i + 1, row + i] for i in range(n_facets)]
    return torch.tensor(verts), torch.tensor(faces, dtype=torch.long)


def slave_plate(n_x: int, *, half: float = L, width: float = W,
                tilt: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    xs = [-half + 2.0 * half * i / n_x for i in range(n_x + 1)]
    verts = [[x, 0.0, tilt * x] for x in xs] + [[x, width, tilt * x] for x in xs]
    faces = [[i, i + 1, (n_x + 1) + i + 1, (n_x + 1) + i] for i in range(n_x)]
    return torch.tensor(verts), torch.tensor(faces, dtype=torch.long)


# --------------------------------------------------------------- FE evaluate ---
def equilibrium(tilt: float = 0.0, *, n_slave: int = 40, n_master: int = 64,
                eps_n: float = EPS_N) -> dict:
    """One mortar evaluation of the prescribed (undeformed) configuration."""
    sv, sf = slave_plate(n_slave, tilt=tilt)
    mv, mf = master_cylinder(n_master)
    res = integrate_mortar_contact(sv, sf, mv, mf, normal_penalty=eps_n)
    fs = res.slave_forces
    total = fs.sum(0)
    # moment about the y axis through the origin:  M_y = sum(z F_x - x F_z)
    my = float((sv[:, 2] * fs[:, 0] - sv[:, 0] * fs[:, 2]).sum())
    return {
        "F": float(total[2]),
        "M": my,
        "E": float(res.penalty_energy),
        "active": int(res.active_quadrature_points),
        "pen_max": float(res.maximum_penetration),
        "integrated_area": float(res.integrated_area),
        "balance_roundoff": float(torch.linalg.vector_norm(total + res.master_forces.sum(0))),
    }


def penetration_profile(n_slave: int, n_master: int, tilt: float = 0.0) -> list[dict]:
    """Per-quadrature-point penetration of the discrete model (for figures)."""
    sv, sf = slave_plate(n_slave, tilt=tilt)
    mv, mf = master_cylinder(n_master)
    rows = []
    for face in sf:
        for point, _shape, _da in _quadrature(sv[face]):
            pr = project_point_to_facets(point, mv, mf)
            rows.append({"x": float(point[0]), "pen": float(torch.clamp(-pr.gap, min=0.0))})
    rows.sort(key=lambda r: r["x"])
    return rows


# ---------------------------------------------------------- reference oracles ---
def oracle_facet(tilt: float = 0.0, n_master: int = 64, *, width: float = W,
                 eps_n: float = EPS_N, half: float = MASTER_HALF,
                 radius: float = R, apex: float = DELTA) -> dict:
    """Exact integral of the faceted model (no torch, no mortar assembly).

    Within one chord, penetration is linear in x:
        pen(x) = (z_chord(x) - tilt*x) / sqrt(1+m^2) = a + b x
    so F_z = eps W S pen n_z, M_y = eps W S pen (z n_x - x n_z) and
    E = eps/2 W S pen^2 are exact per clipped facet with 8-point Gauss.
    """
    xs, zs = master_nodes(n_master, half=half, radius=radius, apex=apex)
    t, wt = _GAUSS8
    fz = my = en = 0.0
    pen_max = 0.0
    for x1, x2, z1, z2 in zip(xs[:-1], xs[1:], zs[:-1], zs[1:]):
        m = (z2 - z1) / (x2 - x1)
        s = math.sqrt(1.0 + m * m)
        nz, nx = 1.0 / s, -m / s
        a = (z1 - m * x1) / s
        b = (m - tilt) / s
        cuts = [x1, x2]
        if b != 0.0:
            xr = -a / b
            if x1 < xr < x2:
                cuts = [x1, xr, x2]
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            if hi <= lo:
                continue
            mid, hw = 0.5 * (lo + hi), 0.5 * (hi - lo)
            for tg, wg in zip(t, wt):
                x = mid + hw * tg
                pen = max(0.0, a + b * x)
                wgt = wg * hw
                fz += wgt * pen * nz
                my += wgt * pen * (tilt * x * nx - x * nz)
                en += wgt * pen * pen
                pen_max = max(pen_max, pen)
        pen_max = max(pen_max, max(0.0, a + b * x1), max(0.0, a + b * x2))
    return {"F": eps_n * width * fz, "M": eps_n * width * my,
            "E": 0.5 * eps_n * width * en, "pen_max": pen_max}


def _radial_roots(tilt: float, *, radius: float = R, apex: float = DELTA) -> tuple[float, float]:
    """|p-C| = R with p=(x,0,tilt*x), C=(0,0,apex-radius) -> quadratic in x."""
    cz = apex - radius
    # x^2 + (tilt x - cz)^2 = R^2
    a = 1.0 + tilt * tilt
    b = -2.0 * tilt * cz
    c = cz * cz - radius * radius
    disc = b * b - 4.0 * a * c
    if disc <= 0.0:
        return 0.0, 0.0
    sq = math.sqrt(disc)
    lo, hi = (-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)
    return max(lo, -radius), min(hi, radius)


def oracle_radial(tilt: float = 0.0, *, width: float = W, eps_n: float = EPS_N,
                  radius: float = R, apex: float = DELTA, delta: float | None = None) -> dict:
    """Exact circle / continuum limit: radial normal gap, adaptive quadrature."""
    from scipy.integrate import quad

    if delta is not None:
        apex = delta
    cz = apex - radius
    lo, hi = _radial_roots(tilt, radius=radius, apex=apex)

    def state(x: float) -> tuple[float, float, float]:
        d = math.sqrt(x * x + (tilt * x - cz) ** 2)
        pen = max(0.0, radius - d)
        nx, nz = x / d, (tilt * x - cz) / d
        return pen, nx, nz

    if hi <= lo:
        return {"F": 0.0, "M": 0.0, "E": 0.0, "a": 0.0, "pen_max": 0.0,
                "definition_gap_F": 0.0, "definition_gap_M": 0.0}

    def pen(x: float) -> float:
        return state(x)[0]

    fz = quad(lambda x: pen(x) * state(x)[2], lo, hi, limit=400, epsabs=1e-16, epsrel=1e-13)[0]
    my = quad(lambda x: pen(x) * (tilt * x * state(x)[1] - x * state(x)[2]),
              lo, hi, limit=400, epsabs=1e-16, epsrel=1e-13)[0]
    en = quad(lambda x: pen(x) ** 2, lo, hi, limit=400, epsabs=1e-16, epsrel=1e-13)[0]
    xm = 0.5 * (lo + hi)
    return {"F": eps_n * width * fz, "M": eps_n * width * my,
            "E": 0.5 * eps_n * width * en, "a": 0.5 * (hi - lo),
            "pen_max": max(pen(lo), pen(xm), pen(hi))}


def paraboloid_closed_form(*, width: float = W, eps_n: float = EPS_N,
                           radius: float = R, delta: float = DELTA) -> float:
    """F = (4/3) eps_n W a delta for tilt = 0 (vertical-gap convention)."""
    a = math.sqrt(2.0 * radius * delta - delta * delta)
    return eps_n * width * (4.0 / 3.0) * a * delta


def oracle_selfcheck() -> dict:
    """Cross-check the radial oracle against two independent identities."""
    orc = oracle_radial()
    par = paraboloid_closed_form()
    h = 1e-6
    e_plus = oracle_radial(delta=DELTA + h)["E"]
    e_minus = oracle_radial(delta=DELTA - h)["E"]
    f_from_energy = (e_plus - e_minus) / (2.0 * h)      # F = dE/d(delta)
    # vertical-gap version of the same integral, for the convention study
    from scipy.integrate import quad
    a = math.sqrt(2.0 * R * DELTA - DELTA * DELTA)
    f_vertical = EPS_N * W * quad(lambda x: max(0.0, (DELTA - R) + math.sqrt(R * R - x * x)),
                                 -a, a, epsabs=1e-16, epsrel=1e-13)[0]
    return {
        "F_radial": orc["F"],
        "F_vertical_gap": f_vertical,
        "F_paraboloid_closed_form": par,
        "F_energy_derivative": f_from_energy,
        "rel_gap_paraboloid": abs(f_vertical - par) / par,
        "rel_gap_energy_identity": abs(orc["F"] - f_from_energy) / orc["F"],
        "rel_normal_vs_vertical": abs(orc["F"] - f_vertical) / f_vertical,
    }


# ------------------------------------------------------------------- studies ---
def _rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b != 0.0 else abs(a)


def observed_order(h: list[float], err: list[float]) -> float:
    xs = [math.log(v) for v in h]
    ys = [math.log(max(v, 1e-18)) for v in err]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den) if den else float("nan")


def summarize(rows: list[dict], key_h: str, key_err: str) -> dict:
    rows = [r for r in rows if r.get(key_err, 0.0) > 0.0]
    if len(rows) < 2:
        return {"n_levels": len(rows), "observed_order": None}
    out = {
        "n_levels": len(rows),
        "observed_order": observed_order([r[key_h] for r in rows], [r[key_err] for r in rows]),
        "rel_err_first": rows[0][key_err],
        "rel_err_last": rows[-1][key_err],
    }
    if len(rows) >= 3:   # drop the first (asymptotically invalid) point
        out["observed_order_tail"] = observed_order(
            [r[key_h] for r in rows[1:]], [r[key_err] for r in rows[1:]])
    return out


def series_master(tilt: float = 0.0, *, n_slave: int = 40,
                  levels=(8, 16, 32, 64)) -> list[dict]:
    rad = oracle_radial(tilt)
    rows = []
    for nm in levels:
        d = equilibrium(tilt, n_slave=n_slave, n_master=nm)
        fac = oracle_facet(tilt, nm)
        rows.append({
            "n_master": nm, "h_master": 2.0 * MASTER_HALF / nm,
            "F": d["F"], "F_radial": rad["F"], "rel_err_F": _rel(d["F"], rad["F"]),
            "E": d["E"], "E_radial": rad["E"], "rel_err_E": _rel(d["E"], rad["E"]),
            "M": d["M"], "M_radial": rad["M"], "rel_err_M": _rel(d["M"], rad["M"]),
            "F_facet": fac["F"], "rel_quadrature_F": _rel(d["F"], fac["F"]),
            "E_facet": fac["E"], "rel_quadrature_E": _rel(d["E"], fac["E"]),
            "faceting_err_F": _rel(fac["F"], rad["F"]),
            "faceting_err_E": _rel(fac["E"], rad["E"]),
            "pen_max": d["pen_max"], "pen_radial": rad["pen_max"],
            "active": d["active"], "balance_roundoff": d["balance_roundoff"],
        })
    return rows


def series_slave(tilt: float = 0.0, *, n_master: int = 64,
                 levels=(10, 20, 40, 160)) -> list[dict]:
    rad = oracle_radial(tilt)
    fac = oracle_facet(tilt, n_master)
    rows = []
    for ns in levels:
        d = equilibrium(tilt, n_slave=ns, n_master=n_master)
        rows.append({"n_slave": ns, "h_slave": 2.0 * L / ns, "F": d["F"],
                     "F_radial": rad["F"], "rel_err_F": _rel(d["F"], rad["F"]),
                     "F_facet": fac["F"], "rel_quadrature_F": _rel(d["F"], fac["F"]),
                     "active": d["active"]})
    return rows


def series_joint(tilt: float = 0.0, pairs=((20, 16), (40, 32), (80, 64))) -> list[dict]:
    """Slave refined together with the master, always at least as fine."""
    rad = oracle_radial(tilt)
    rows = []
    for ns, nm in pairs:
        d = equilibrium(tilt, n_slave=ns, n_master=nm)
        fac = oracle_facet(tilt, nm)
        rows.append({"n_slave": ns, "n_master": nm, "h_master": 2.0 * MASTER_HALF / nm,
                     "F": d["F"], "F_radial": rad["F"], "rel_err_F": _rel(d["F"], rad["F"]),
                     "E": d["E"], "E_radial": rad["E"], "rel_err_E": _rel(d["E"], rad["E"]),
                     "rel_quadrature_F": _rel(d["F"], fac["F"]),
                     "faceting_err_F": _rel(fac["F"], rad["F"]), "active": d["active"]})
    return rows


def series_faceting(tilt: float = 0.0, levels=(8, 16, 32, 64, 128, 256, 512)) -> list[dict]:
    """Pure geometric faceting error (numpy oracle only: cheap, many levels)."""
    rad = oracle_radial(tilt)
    rows = []
    for nm in levels:
        fac = oracle_facet(tilt, nm)
        rows.append({"n_master": nm, "h_master": 2.0 * MASTER_HALF / nm,
                     "faceting_err_F": _rel(fac["F"], rad["F"]),
                     "faceting_err_E": _rel(fac["E"], rad["E"]),
                     "faceting_err_M": _rel(fac["M"], rad["M"]) if rad["M"] else None})
    return rows


def series_flat_patch(levels=((10, 8), (20, 16), (40, 32))) -> list[dict]:
    """Control: a *planar* master must be reproduced exactly at every level."""
    rows = []
    for ns, nm in levels:
        sv, sf = slave_plate(ns)
        xs = [-MASTER_HALF + 2.0 * MASTER_HALF * i / nm for i in range(nm + 1)]
        verts = [[x, 0.0, DELTA] for x in xs] + [[x, MASTER_W, DELTA] for x in xs]
        faces = [[i, i + 1, nm + 1 + i + 1, nm + 1 + i] for i in range(nm)]
        mv, mf = torch.tensor(verts), torch.tensor(faces, dtype=torch.long)
        res = integrate_mortar_contact(sv, sf, mv, mf, normal_penalty=EPS_N)
        f_exact = EPS_N * DELTA * (2.0 * L) * W
        e_exact = 0.5 * EPS_N * DELTA ** 2 * (2.0 * L) * W
        f = float(res.slave_forces.sum(0)[2])
        rows.append({"n_slave": ns, "n_master": nm, "F": f, "F_ana": f_exact,
                     "rel_err_F": _rel(f, f_exact), "E": float(res.penalty_energy),
                     "E_ana": e_exact, "rel_err_E": _rel(float(res.penalty_energy), e_exact)})
    return rows


# -------------------------------------------------------------------- figures --
def make_figures(payload: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for key, style, lab in (("master", "o-", "FE vs exact circle (F)"),
                            ("faceting", "s--", "faceted model vs exact circle (F)")):
        rows = payload["series"][key]
        ax.loglog([r["h_master"] for r in rows],
                  [r["rel_err_F"] if key == "master" else r["faceting_err_F"] for r in rows],
                  style, label=lab)
    rows = payload["series"]["master"]
    ax.loglog([r["h_master"] for r in rows], [r["rel_quadrature_F"] for r in rows],
              "^-", label="FE vs faceted model (quadrature error)")
    ref = [rows[-1]["rel_err_F"] * (r["h_master"] / rows[-1]["h_master"]) ** 2 for r in rows]
    ax.loglog([r["h_master"] for r in rows], ref, "k:", label=r"slope $h^2$")
    ax.axhline(payload["oracle_selfcheck"]["rel_normal_vs_vertical"], color="grey", ls="-.",
               label="normal-vs-vertical gap (definition)")
    ax.set_xlabel("master facet size $h_m$ [m]")
    ax.set_ylabel("relative error in normal force")
    ax.set_title("Curved mortar contact: error budget vs master faceting")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    p = FIG_DIR / "mortar_curved_convergence_budget.png"
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    rows = payload["series"]["faceting"]
    ax.loglog([r["h_master"] for r in rows], [r["faceting_err_F"] for r in rows], "o-",
              label="faceted vs exact, F")
    ax.loglog([r["h_master"] for r in rows], [r["faceting_err_E"] for r in rows], "s-",
              label="faceted vs exact, E")
    ref = [rows[-1]["faceting_err_F"] * (r["h_master"] / rows[-1]["h_master"]) ** 2 for r in rows]
    ax.loglog([r["h_master"] for r in rows], ref, "k:", label=r"slope $h^2$")
    ax.set_xlabel("master facet size $h_m$ [m]")
    ax.set_ylabel("relative error")
    ax.set_title("Geometric faceting error of the curved master interface")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    p = FIG_DIR / "mortar_curved_faceting_order.png"
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    xs = [i * 2.0 * L / 2000 - L for i in range(2001)]
    rad = oracle_radial(0.0)
    cz = DELTA - R
    ax.plot(xs, [max(0.0, R - math.hypot(x, cz)) for x in xs], "k-", lw=1.7,
            label="exact circle (radial gap)")
    ax.plot(xs, [max(0.0, (DELTA - R) + math.sqrt(max(R * R - x * x, 0.0))) for x in xs],
            color="grey", ls="--", lw=1.2, label="classical vertical gap")
    for prof, style, lab in ((payload["profiles"]["coarse"], "o-", "n_master=8, n_slave=10"),
                             (payload["profiles"]["fine"], "s--",
                              "n_master=128, n_slave=160")):
        ax.plot([r["x"] for r in prof], [r["pen"] for r in prof], style, ms=3, lw=1.1,
                label=lab)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("normal penetration [m]")
    ax.set_title(f"Discrete penetration field (peak exact {rad['pen_max']:.6f} m)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    p = FIG_DIR / "mortar_curved_convergence_profile.png"
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    rows = payload["series"]["tilt"]
    ax.loglog([r["h_master"] for r in rows], [r["rel_err_F"] for r in rows], "o-",
              label=f"F (tilt={TILT})")
    ax.loglog([r["h_master"] for r in rows], [r["rel_err_M"] for r in rows], "s-",
              label=f"M (tilt={TILT})")
    ref = [rows[-1]["rel_err_F"] * (r["h_master"] / rows[-1]["h_master"]) ** 2 for r in rows]
    ax.loglog([r["h_master"] for r in rows], ref, "k:", label=r"slope $h^2$")
    ax.set_xlabel("master facet size $h_m$ [m]")
    ax.set_ylabel("relative error vs exact circle")
    ax.set_title("Asymmetric contact (tilted slave): force and moment")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    p = FIG_DIR / "mortar_curved_convergence_tilt.png"
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    paths.append(str(p))
    return paths


def main() -> dict:
    print("[oracle] self-check ...", flush=True)
    oracle = oracle_selfcheck()
    print(oracle, flush=True)
    assert oracle["rel_gap_paraboloid"] < 0.02, oracle
    assert oracle["rel_gap_energy_identity"] < 1e-6, oracle

    print("[series 0] pure faceting (numpy oracle only) ...", flush=True)
    faceting = series_faceting()
    print("[series A] FE, master faceting refinement ...", flush=True)
    master = series_master()
    print("[series B] FE, slave refinement (master fixed) ...", flush=True)
    slave = series_slave()
    print("[series C] FE, joint refinement ...", flush=True)
    joint = series_joint()
    print("[series D] FE, planar patch control ...", flush=True)
    flat = series_flat_patch()
    print("[series E] FE, tilted slave (moment) ...", flush=True)
    tilt = series_master(TILT)

    payload = {
        "meta": {
            "module": "tensorfem.mortar_contact3d.integrate_mortar_contact",
            "R": R, "delta": DELTA, "width": W, "half_length": L, "eps_n": EPS_N,
            "master_half": MASTER_HALF, "tilt": TILT,
            "slave_quadrature": "2x2 Gauss per QUAD4, reference-area measure",
            "oracle_radial": "adaptive quadrature of exact circle, radial normal gap",
            "oracle_facet": "exact integral of the faceted description, 8-point Gauss",
        },
        "oracle_selfcheck": oracle,
        "series": {"faceting": faceting, "master": master, "slave": slave,
                   "joint": joint, "flat_patch": flat, "tilt": tilt},
        "orders": {
            "faceting_err_F": summarize(faceting, "h_master", "faceting_err_F"),
            "faceting_err_E": summarize(faceting, "h_master", "faceting_err_E"),
            "master_rel_err_F": summarize(master, "h_master", "rel_err_F"),
            "master_rel_err_E": summarize(master, "h_master", "rel_err_E"),
            "master_quadrature_err_F": summarize(master, "h_master", "rel_quadrature_F"),
            "joint_rel_err_F": summarize(joint, "h_master", "rel_err_F"),
            "joint_rel_err_E": summarize(joint, "h_master", "rel_err_E"),
            "tilt_rel_err_F": summarize(tilt, "h_master", "rel_err_F"),
            "tilt_rel_err_M": summarize(tilt, "h_master", "rel_err_M"),
        },
        "profiles": {"coarse": penetration_profile(10, 8),
                     "fine": penetration_profile(80, 64)},
    }
    payload["slave_quadrature"] = {
        "levels": [r["n_slave"] for r in slave],
        "rel_quadrature_F": [r["rel_quadrature_F"] for r in slave],
        "rel_err_F": [r["rel_err_F"] for r in slave],
        "worst_quadrature_F": max(r["rel_quadrature_F"] for r in slave),
    }
    finest = joint[-1]
    payload["verdict"] = {
        "finest_joint": {k: finest[k] for k in
                         ("n_slave", "n_master", "rel_err_F", "rel_err_E",
                          "rel_quadrature_F")},
        "under_3pct_at": next(({"n_slave": r["n_slave"], "n_master": r["n_master"],
                                "rel_err_F": r["rel_err_F"]}
                               for r in joint if r["rel_err_F"] < 0.03), None),
        "under_1pct_at": next(({"n_slave": r["n_slave"], "n_master": r["n_master"],
                                "rel_err_F": r["rel_err_F"]}
                               for r in joint if r["rel_err_F"] < 0.01), None),
        "flat_patch_max_rel_err_F": max(r["rel_err_F"] for r in flat),
        "flat_patch_max_rel_err_E": max(r["rel_err_E"] for r in flat),
        "faceting_err_at_h_0.01": next((r["faceting_err_F"] for r in faceting
                                        if r["n_master"] == 64), None),
    }
    order = payload["orders"]["faceting_err_F"].get("observed_order_tail") or 2.0
    rows = [r for r in faceting if r["faceting_err_F"] < 0.03 and r["faceting_err_F"] > 0]
    if rows:
        h_ref, e_ref = rows[-1]["h_master"], rows[-1]["faceting_err_F"]
        payload["verdict"]["predicted_h_master_for_3pct"] = h_ref * (0.03 / e_ref) ** (1.0 / order)
        payload["verdict"]["faceting_order_used"] = order

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload["figures"] = make_figures(payload)
    (OUT_DIR / "convergence.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["verdict"], indent=2))
    print(json.dumps(payload["orders"], indent=2))
    print(json.dumps(payload["slave_quadrature"], indent=2))
    print(f"wrote {OUT_DIR/'convergence.json'}")
    return payload


if __name__ == "__main__":
    main()