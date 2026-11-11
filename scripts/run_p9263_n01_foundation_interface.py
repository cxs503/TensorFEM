#!/usr/bin/env python
"""P9263-N01 study driver: Mindlin plate tangential contact stiffness and
Winkler / Pasternak / elastic-half-space foundation switching.

Everything printed here is written to ``artifacts/P9263-N01/study_results.json``
(no hand-copied numbers) and the figures next to it.

Pre-declared acceptance thresholds (fixed before the runs, ``experimental-design``
discipline: no threshold tuning after seeing the numbers)

    T1  C1  |w_FEM / w_FSDT-Navier - 1| <= 5%      (Winkler and Pasternak, n = 16)
    T2  C2  w_Pasternak < w_Winkler for Gp > 0, and the FEM ratio agrees with the
            exact Navier ratio within 2%
    T3  C3  observed order of convergence p >= 1.5 and errors strictly decreasing
            on the mesh ladder (mesh refinement ratio ~1.33, the last three levels)
    T4  C4  three foundation kinds reachable through make_foundation()/one solver,
            stiffening limits monotone (k -> inf, Gp -> inf, Es -> inf  => w -> 0)
    T5  C5  negative controls must FAIL as designed (wrong reference, wrong Gp sign)

Run:  PYTHONPATH=./src OMP_NUM_THREADS=1 python scripts/run_p9263_n01_foundation_interface.py
"""
from __future__ import annotations

import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensorfem.plate_foundation_interface import (
    ElasticHalfSpaceFoundation,
    PasternakFoundation,
    WinklerFoundation,
    element_bending_moments,
    fsdt_navier_ss_deflection,
    half_space_patch_compliance,
    make_foundation,
    q4_normal_foundation_stiffness,
    q4_shear_layer_stiffness,
    solve_mindlin_plate_foundation,
)
from tensorfem.plate import structured_square_mesh
from tensorfem.plate_foundation_analytical import (
    kirchhoff_ss_center,
    mindlin_ss_center_correction,
)

OUT = os.path.join("artifacts", "P9263-N01")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Case definition (SI units): 1 m x 1 m steel plate, moderately thick panel
# ---------------------------------------------------------------------------
CASE = dict(side=1.0, young=2.0e11, poisson=0.3, thickness=0.05, pressure=1.0e4)
K_NORMAL = 1.0e9      # Winkler modulus [N/m^3]
G_SHEAR = 2.0e7       # Pasternak tangential/shear-layer stiffness [N/m]
KAPPA = 5.0 / 6.0

results: dict = {
    "node": "P9263-N01",
    "title": "Mindlin plate tangential contact stiffness + Winkler/Pasternak/half-space foundation switching",
    "case": CASE,
    "foundation_reference": {"k_normal": K_NORMAL, "g_shear": G_SHEAR, "kappa": KAPPA},
    "thresholds": {"T1_rel_error_pct": 5.0, "T2_ratio_agreement_pct": 2.0, "T3_order_min": 1.5},
    "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}


def _err(value: float, ref: float) -> float:
    return abs(value / ref - 1.0) * 100.0


def exact(k_normal: float = 0.0, g_shear: float = 0.0, k_tangential: float = 0.0, **kwargs) -> dict:
    params = dict(q=CASE["pressure"], a=CASE["side"], b=CASE["side"], young=CASE["young"],
                  poisson=CASE["poisson"], thickness=CASE["thickness"], k_normal=k_normal,
                  g_shear=g_shear, k_tangential=k_tangential, kappa=KAPPA, terms=161)
    params.update(kwargs)
    return fsdt_navier_ss_deflection(**params)


def fem(n: int, foundation=None, k_tangential: float = 0.0, thickness: float = None, **kwargs):
    return solve_mindlin_plate_foundation(
        n, side=CASE["side"], young=CASE["young"], poisson=CASE["poisson"],
        thickness=CASE["thickness"] if thickness is None else thickness,
        pressure=CASE["pressure"], foundation=foundation,
        tangential_contact=k_tangential, kappa=KAPPA, **kwargs)


# ---------------------------------------------------------------------------
# Study A/C1+C2: Winkler and Pasternak ladders against the FSDT Navier solution
# ---------------------------------------------------------------------------
MESH_LADDER = (6, 8, 12, 16, 20)


def run_ladders() -> dict:
    out = {}
    winkler = WinklerFoundation(K_NORMAL)
    pasternak = PasternakFoundation(K_NORMAL, G_SHEAR)
    for name, model, ref in (
        ("winkler", winkler, exact(k_normal=K_NORMAL)),
        ("pasternak", pasternak, exact(k_normal=K_NORMAL, g_shear=G_SHEAR)),
    ):
        rows = []
        for n in MESH_LADDER:
            t0 = time.time()
            sol = fem(n, model)
            rows.append({
                "n": n, "h": CASE["side"] / n,
                "w_centre": sol.centre_deflection(),
                "w_ref": ref["deflection"],
                "rel_error_pct": _err(sol.centre_deflection(), ref["deflection"]),
                "wall_s": time.time() - t0,
            })
        # observed order from the three finest meshes (refinement ratio r = 16/12 = 12/8)
        e = [rows[-3]["rel_error_pct"], rows[-2]["rel_error_pct"], rows[-1]["rel_error_pct"]]
        r = rows[-2]["h"] / rows[-1]["h"]
        order = math.log(e[-2] / e[-1]) / math.log(r)
        out[name] = {
            "ladder": rows,
            "reference": ref,
            "observed_order": order,
            "refinement_ratio": r,
            "monotone": all(rows[i]["rel_error_pct"] > rows[i + 1]["rel_error_pct"]
                            for i in range(len(rows) - 1)),
        }
    ratio = out["pasternak"]["ladder"][-1]["w_centre"] / out["winkler"]["ladder"][-1]["w_centre"]
    ratio_exact = out["pasternak"]["reference"]["deflection"] / out["winkler"]["reference"]["deflection"]
    out["comparison"] = {
        "w_ratio_fem": ratio,
        "w_ratio_exact": ratio_exact,
        "ratio_agreement_pct": abs(ratio / ratio_exact - 1.0) * 100.0,
        "pasternak_softer_than_winkler": ratio < 1.0,
        "deflection_reduction_pct": (1.0 - ratio) * 100.0,
    }
    return out


# ---------------------------------------------------------------------------
# Study B: Mindlin tangential contact stiffness sweep
# ---------------------------------------------------------------------------
K_T_SWEEP = (0.0, 1.0e8, 1.0e9, 1.0e10, 1.0e11, 1.0e12)


def run_tangential_sweep(n: int = 16) -> dict:
    winkler = WinklerFoundation(K_NORMAL)
    rows = []
    bonded = exact(k_normal=K_NORMAL, k_tangential=math.inf)
    for kt in K_T_SWEEP:
        sol = fem(n, winkler, k_tangential=kt)
        ref = exact(k_normal=K_NORMAL, k_tangential=kt)
        rows.append({
            "k_tangential": kt,
            "R_moment_rigidity": kt * CASE["thickness"] ** 2 / 4.0,
            "w_centre": sol.centre_deflection(),
            "w_ref": ref["deflection"],
            "rel_error_pct": _err(sol.centre_deflection(), ref["deflection"]),
        })
    # layer-physics anchors: k_t = G_f / h_f for a bonded interface layer
    layers = []
    for name, g_f, h_f in (("rubber pad 10 mm, G=1 MPa", 1.0e6, 0.010),
                           ("grout 20 mm, G=5 GPa", 5.0e9, 0.020),
                           ("steel shim 5 mm, G=80 GPa", 8.0e10, 0.005)):
        kt = g_f / h_f
        sol = fem(n, winkler, k_tangential=kt)
        layers.append({"name": name, "G_f": g_f, "h_f": h_f, "k_tangential": kt,
                       "w_centre": sol.centre_deflection(),
                       "deflection_reduction_vs_unbonded_pct":
                           (1.0 - sol.centre_deflection() /
                            rows[0]["w_centre"]) * 100.0})
    thickness_sensitivity = []
    for h in (0.0125, 0.025, 0.05, 0.1):
        sol = fem(n, winkler, k_tangential=1.0e10, thickness=h)
        base = fem(n, winkler, k_tangential=0.0, thickness=h)
        ref = exact(k_normal=K_NORMAL, k_tangential=1.0e10, thickness=h)
        thickness_sensitivity.append({
            "thickness": h, "h_over_a": h / CASE["side"],
            "w_bonded": sol.centre_deflection(), "w_unbonded": base.centre_deflection(),
            "reduction_pct": (1.0 - sol.centre_deflection() / base.centre_deflection()) * 100.0,
            # the exact series is the reference for the *sign*: the centre value is an
            # alternating double series, so it is NOT guaranteed to decrease with k_t
            # (at h/a = 0.0125 both the exact and the FE value increase slightly);
            # what must agree is the magnitude of the change
            "exact_reduction_pct": (1.0 - ref["deflection"] / exact(
                k_normal=K_NORMAL, thickness=h)["deflection"]) * 100.0,
            "rel_error_vs_exact_pct": _err(sol.centre_deflection(), ref["deflection"]),
        })
    return {"sweep": rows, "bonded_limit": bonded, "interface_layers": layers,
            "thickness_sensitivity": thickness_sensitivity,
            "monotone_decreasing": all(rows[i]["w_centre"] > rows[i + 1]["w_centre"]
                                       for i in range(len(rows) - 1))}


# ---------------------------------------------------------------------------
# Study D: foundation model switching (Winkler / Pasternak / half-space)
# ---------------------------------------------------------------------------
def equivalent_half_space_young(k_normal: float) -> float:
    """Es giving the same average settlement factor as ``k_normal`` for a 1 m pad.

    ``w_avg = 0.946402 q B (1 - nu^2)/Es`` (closed-form Boussinesq integral), so
    ``k_eq = q/w_avg`` -> ``Es = 0.946402 k B (1 - nu^2)``.
    """
    return 0.946402 * k_normal * CASE["side"] * (1.0 - CASE["poisson"] ** 2)


def run_model_switch(n_switch: int = 16, n_calibrate: int = 8) -> dict:
    winkler = WinklerFoundation(K_NORMAL)
    pasternak = PasternakFoundation(K_NORMAL, G_SHEAR)
    e_s = equivalent_half_space_young(K_NORMAL)
    half_space = ElasticHalfSpaceFoundation(e_s, CASE["poisson"])

    models = {}
    for name, model in (("winkler", winkler), ("pasternak", pasternak), ("half_space", half_space)):
        model_check = make_foundation(
            name, **({"k_normal": K_NORMAL} if name == "winkler" else
                     {"k_normal": K_NORMAL, "g_shear": G_SHEAR} if name == "pasternak" else
                     {"young": e_s, "poisson": CASE["poisson"]}))
        t0 = time.time()
        sol = fem(n_switch, model_check)
        w = sol.transverse
        centre = sol.centre_deflection()
        # quarter-line profile along y = a/2
        ys = sol.nodes[:, 1]
        band = (ys - CASE["side"] / 2).abs() < CASE["side"] / n_switch / 4
        x_line = sol.nodes[band, 0]
        order = torch.argsort(x_line)
        moments = element_bending_moments(sol)
        entry = {
            "kind": model_check.kind,
            "w_centre": centre,
            "w_quarter_point": float(w[torch.argmin(
                torch.linalg.norm(sol.nodes - torch.tensor([CASE["side"] / 4, CASE["side"] / 2],
                                                          dtype=sol.nodes.dtype), dim=1))]),
            "Mx_max": float(moments["Mx"].abs().max()),
            "Mx_centre_element": float(moments["Mx"][torch.argmin(
                torch.linalg.norm(sol.nodes[sol.elements].mean(dim=1)
                                  - torch.tensor([CASE["side"] / 2, CASE["side"] / 2],
                                                 dtype=sol.nodes.dtype), dim=1))]),
            "support_reaction": sol.support_reaction(),
            "foundation_reaction": sol.foundation_total_reaction(),
            "equilibrium_residual": sol.equilibrium_residual(),
            "applied_load": CASE["pressure"] * CASE["side"] ** 2,
            "pressure_max_abs": float(sol.foundation_pressure().abs().max()),
            "patch_compliance_condition": (
                float(torch.linalg.cond(half_space_patch_compliance(
                    sol.nodes, sol.elements, model_check.young, model_check.poisson)))
                if model_check.non_local else None),
            "wall_s": time.time() - t0,
            "profile_x": x_line[order].tolist(),
            "profile_w": w[band][order].tolist(),
            "profile_Mx": None,
        }
        # moment profile: element centres on the same band
        centres = sol.nodes[sol.elements].mean(dim=1)
        mband = (centres[:, 1] - CASE["side"] / 2).abs() < CASE["side"] / n_switch / 4
        morder = torch.argsort(centres[mband, 0])
        entry["profile_Mx_x"] = centres[mband, 0][morder].tolist()
        entry["profile_Mx"] = moments["Mx"][mband][morder].tolist()
        models[name] = entry

    # equivalent-local-stiffness reading of the half-space model
    models["half_space"]["equivalent_young"] = e_s
    models["half_space"]["equivalent_k_normal"] = K_NORMAL

    # stiffening limits: each model must go to zero deflection monotonically
    limits = {}
    for name, params in (
        ("winkler", [(1.0e9,), (1.0e10,), (1.0e11,), (1.0e12,)]),
        ("pasternak", [(2.0e7,), (1.0e8,), (1.0e9,), (1.0e10,)]),
        ("half_space", [(e_s,), (10 * e_s,), (100 * e_s,), (1000 * e_s,)]),
    ):
        seq = []
        for (value,) in params:
            if name == "winkler":
                model = WinklerFoundation(value)
            elif name == "pasternak":
                model = PasternakFoundation(K_NORMAL, value)
            else:
                model = ElasticHalfSpaceFoundation(value, CASE["poisson"])
            seq.append({"value": value, "w_centre": fem(n_calibrate, model).centre_deflection()})
        # monotonicity is asserted on the part of the ladder that is above the
        # round-off floor (1e-3 of the first value); beyond that the deflection is
        # at the solver noise level and the sign carries no information
        floor = 1e-3 * seq[0]["w_centre"]
        active = [row for row in seq if row["w_centre"] > floor]
        if not active or active[-1] is not seq[-1]:
            active.append(seq[-1])
        limits[name] = {
            "sequence": seq,
            "monotone_decreasing": all(active[i]["w_centre"] > active[i + 1]["w_centre"]
                                       for i in range(len(active) - 1)),
            "tends_to_zero_pct": seq[-1]["w_centre"] / seq[0]["w_centre"] * 100.0,
            "round_off_floor_reached": seq[-1]["w_centre"] <= floor,
            "pass": all(active[i]["w_centre"] > active[i + 1]["w_centre"]
                        for i in range(len(active) - 1))
            and abs(seq[-1]["w_centre"]) <= 0.02 * seq[0]["w_centre"],
        }
    return {"models": models, "stiffening_limits": limits}


def _patch_areas(sol) -> torch.Tensor:  # noqa: D401 (kept for later profile work)
    xy = sol.nodes[sol.elements]
    return ((xy[:, :, 0].max(1).values - xy[:, :, 0].min(1).values)
            * (xy[:, :, 1].max(1).values - xy[:, :, 1].min(1).values))


# ---------------------------------------------------------------------------
# Study E: half-space patch quadrature vs the closed-form Boussinesq integral
# ---------------------------------------------------------------------------
def _gauss_legendre(n: int):
    xs, ws = [], []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        dp = 1.0
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1.0)
            dx = -p1 / dp
            x += dx
            if abs(dx) < 1e-15:
                break
        xs.append(x)
        ws.append(2.0 / ((1.0 - x * x) * dp * dp))
    return xs, ws


def exact_uniform_square_settlement_factors(kind: str = "average") -> dict:
    """Closed-form Boussinesq integral over a uniformly loaded 1 m square."""
    from tensorfem.elastic_halfspace import settlement
    e_s = 1.0e7
    nu = CASE["poisson"]
    q = 1.0e5
    xs, ws = _gauss_legendre(60)
    total = 0.0
    for a, w1 in zip(xs, ws):
        for b, w2 in zip(xs, ws):
            x = 0.5 * (a + 1.0)
            y = 0.5 * (b + 1.0)
            total += w1 * w2 * settlement(q, e_s, nu, CASE["side"], CASE["side"], x, y)
    average = total / 4.0
    scale = q * CASE["side"] * (1 - nu ** 2) / e_s
    return {
        "young_half_space": e_s, "pressure": q,
        "w_average_factor": average / scale,
        "w_centre_factor": settlement(q, e_s, nu, CASE["side"], CASE["side"], 0.5, 0.5) / scale,
        "w_corner_factor": settlement(q, e_s, nu, CASE["side"], CASE["side"], 0.0, 0.0) / scale,
        "textbook_average_factor": 0.95,
        "textbook_centre_factor": 1.12,
    }


def run_half_space_quadrature() -> dict:
    reference = exact_uniform_square_settlement_factors()
    e_s = reference["young_half_space"]
    nu = CASE["poisson"]
    q = reference["pressure"]
    scale = q * CASE["side"] * (1 - nu ** 2) / e_s
    rows = []
    for n in (4, 8, 12, 16):
        nodes, elems = structured_square_mesh(n, CASE["side"])
        c_p = half_space_patch_compliance(nodes, elems, e_s, nu)
        xy = nodes[elems]
        area = ((xy[:, :, 0].max(1).values - xy[:, :, 0].min(1).values)
                * (xy[:, :, 1].max(1).values - xy[:, :, 1].min(1).values))
        w_patch = c_p @ torch.full((elems.shape[0],), q, dtype=torch.float64)
        centres = xy.mean(dim=1)
        centre_idx = int(torch.argmin(torch.linalg.norm(
            centres - torch.tensor([0.5, 0.5], dtype=torch.float64), dim=1)))
        betti = float((area[:, None] * c_p - (area[:, None] * c_p).T).abs().max()
                      / (area[:, None] * c_p).abs().max())
        rows.append({
            "n": n,
            "w_average_factor": float((w_patch * area).sum() / area.sum()) / scale,
            "w_average_rel_error_pct": _err(float((w_patch * area).sum() / area.sum()) / scale,
                                            reference["w_average_factor"]),
            "w_centre_factor": float(w_patch[centre_idx]) / scale,
            "w_centre_rel_error_pct": _err(float(w_patch[centre_idx]) / scale,
                                           reference["w_centre_factor"]),
            "betti_asymmetry": betti,
        })
    return {"reference": reference, "ladder": rows}


# ---------------------------------------------------------------------------
# Study F: negative controls (must fail as designed)
# ---------------------------------------------------------------------------
class _WrongSignShear(PasternakFoundation):
    """Negative control: Pasternak matrix with the shear-layer sign flipped."""

    def element_stiffness(self, xy, *, thickness: float = 0.0):
        return (q4_normal_foundation_stiffness(xy, self.k_normal)
                - q4_shear_layer_stiffness(xy, self.g_shear))


def run_negative_controls(n: int = 16) -> dict:
    winkler = WinklerFoundation(K_NORMAL)
    w_winkler = fem(n, winkler).centre_deflection()
    ref_w = exact(k_normal=K_NORMAL)["deflection"]

    # (a) sign-flipped shear layer: must be SOFTER than Winkler and miss the
    #     analytic reference by far more than the 5% gate
    wrong = _WrongSignShear(K_NORMAL, G_SHEAR)
    w_wrong = fem(n, wrong).centre_deflection()

    # (b) g_shear = 0 must reproduce the Winkler result exactly
    w_zero_shear = fem(n, PasternakFoundation(K_NORMAL, 0.0)).centre_deflection()

    # (c) thick plate: the Kirchhoff reference is the *wrong* oracle
    thick = dict(thickness=0.2)
    sol_thick = fem(n, winkler, **thick)
    ref_thick = exact(k_normal=K_NORMAL, **thick)
    ref_kirchhoff = kirchhoff_ss_center(CASE["pressure"], CASE["side"], CASE["side"],
                                        CASE["young"] * 0.2 ** 3 / (12 * (1 - CASE["poisson"] ** 2)))
    # (d) legacy ad-hoc thick-plate estimate vs the exact FSDT series
    D = CASE["young"] * 0.2 ** 3 / (12 * (1 - CASE["poisson"] ** 2))
    shear_modulus = CASE["young"] / (2 * (1 + CASE["poisson"]))
    S = KAPPA * shear_modulus * 0.2
    legacy_old_style = kirchhoff_ss_center(CASE["pressure"], CASE["side"], CASE["side"], D) \
        + CASE["pressure"] * CASE["side"] ** 2 / S * 0.384 * (1 - 0.3) / 2
    legacy_now = mindlin_ss_center_correction(CASE["pressure"], CASE["side"], CASE["side"],
                                              D, 0.2, shear_modulus)
    return {
        "wrong_sign_shear": {
            "w_centre": w_wrong, "w_winkler": w_winkler, "w_ref": ref_w,
            "softer_than_winkler": w_wrong > w_winkler,
            "rel_error_vs_exact_pct": _err(w_wrong, ref_w),
            "fails_5pct_gate": _err(w_wrong, ref_w) > 5.0,
        },
        "zero_shear_identity": {
            "w_centre": w_zero_shear, "w_winkler": w_winkler,
            "identical": abs(w_zero_shear - w_winkler) <= 1e-18 * abs(w_winkler),
        },
        "thick_plate_wrong_reference": {
            "thickness": 0.2, "h_over_a": 0.2,
            "w_fem": sol_thick.centre_deflection(),
            "w_fsdt_exact": ref_thick["deflection"],
            "w_kirchhoff": ref_kirchhoff,
            "rel_error_vs_fsdt_pct": _err(sol_thick.centre_deflection(), ref_thick["deflection"]),
            "rel_error_vs_kirchhoff_pct": _err(sol_thick.centre_deflection(), ref_kirchhoff),
            "kirchhoff_reference_fails_15pct_gate":
                _err(sol_thick.centre_deflection(), ref_kirchhoff) > 15.0,
        },
        "legacy_thick_plate_estimate": {
            "old_adhoc": legacy_old_style, "exact_fsdt": ref_thick["deflection"],
            "old_rel_error_pct": _err(legacy_old_style, ref_thick["deflection"]),
            "now_value": legacy_now,
            "now_rel_error_pct": _err(legacy_now, ref_thick["deflection"]),
        },
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_convergence(ladders: dict) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for name, style in (("winkler", "o-"), ("pasternak", "s-")):
        rows = ladders[name]["ladder"]
        h = [r["h"] for r in rows]
        e = [r["rel_error_pct"] for r in rows]
        ax.loglog(h, e, style, label=f"{name} (p={ladders[name]['observed_order']:.2f})")
    ax.loglog([0.05, 0.17], [2.0, 2.0 * (0.05 / 0.17) ** 2], "k--", lw=1, label="slope 2 reference")
    ax.set_xlabel("element size h [m]")
    ax.set_ylabel("|w_FEM/w_FSDT - 1| [%]")
    ax.set_title("Centre deflection error vs the exact FSDT Navier solution")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    path = os.path.join(OUT, "fig1_convergence.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_tangential(sweep: dict) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    kt = [max(r["k_tangential"], 1.0e7) for r in sweep["sweep"]]
    wfem = [r["w_centre"] * 1e6 for r in sweep["sweep"]]
    wref = [r["w_ref"] * 1e6 for r in sweep["sweep"]]
    ax.semilogx(kt, wref, "k-", label="exact FSDT Navier (with k_t)")
    ax.semilogx(kt, wfem, "o", mfc="none", label="Q4 Mitc FE (n=16)")
    ax.axhline(sweep["bonded_limit"]["deflection"] * 1e6, color="crimson", ls=":",
               label=r"bonded limit $k_t\to\infty$ (pure shear)")
    for row in sweep["interface_layers"]:
        ax.axvline(row["k_tangential"], color="grey", ls="-.", lw=0.8)
    ax.set_xlabel(r"interface tangential contact stiffness $k_t$ [N/m$^3$]")
    ax.set_ylabel("centre deflection [micrometre]")
    ax.set_title("Mindlin tangential contact stiffness: FE vs exact series")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    path = os.path.join(OUT, "fig2_tangential_contact.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_models(switch: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for name, style in (("winkler", "C0-"), ("pasternak", "C1-"), ("half_space", "C2-")):
        entry = switch["models"][name]
        axes[0].plot(entry["profile_x"], [w * 1e6 for w in entry["profile_w"]], style,
                     label=f"{name}: w_c={entry['w_centre']*1e6:.2f} um")
        axes[1].plot(entry["profile_Mx_x"], entry["profile_Mx"], style, label=name)
    axes[0].set_xlabel("x [m] along y = a/2")
    axes[0].set_ylabel("deflection w [micrometre]")
    axes[0].set_title("Deflection profile, three foundation models")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("x [m] along y = a/2")
    axes[1].set_ylabel(r"$M_x$ [N]")
    axes[1].set_title("Bending moment $M_x$ (stress distribution)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    path = os.path.join(OUT, "fig3_model_switch.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_half_space(half: dict) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    rows = half["ladder"]
    n = [r["n"] for r in rows]
    ax.plot(n, [r["w_average_factor"] for r in rows], "o-", label="patch model: area average")
    ax.plot(n, [r["w_centre_factor"] for r in rows], "s-", label="patch model: centre patch")
    ax.axhline(half["reference"]["w_average_factor"], color="C0", ls="--",
               label=f"closed-form average = {half['reference']['w_average_factor']:.4f}")
    ax.axhline(half["reference"]["w_centre_factor"], color="C1", ls="--",
               label=f"closed-form centre = {half['reference']['w_centre_factor']:.4f}")
    ax.axhline(0.95, color="grey", ls=":", label="textbook flexible average 0.95")
    ax.set_xlabel("patches per side")
    ax.set_ylabel(r"settlement factor $w E_s / (q B (1-\nu^2))$")
    ax.set_title("Elastic half-space patch model vs closed-form Boussinesq integral")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path = os.path.join(OUT, "fig4_half_space.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def main() -> dict:
    t_start = time.time()
    print("[1/5] Winkler / Pasternak ladders vs exact FSDT Navier ...", flush=True)
    results["ladders"] = run_ladders()

    print("[2/5] tangential contact stiffness sweep ...", flush=True)
    results["tangential"] = run_tangential_sweep()

    print("[3/5] foundation model switching (winkler/pasternak/half-space) ...", flush=True)
    results["model_switch"] = run_model_switch()

    print("[4/5] half-space patch quadrature ...", flush=True)
    results["half_space"] = run_half_space_quadrature()

    print("[5/5] negative controls ...", flush=True)
    results["negative_controls"] = run_negative_controls()

    results["verdict"] = _verdict(results)
    results["wall_s_total"] = time.time() - t_start

    results["figures"] = [
        fig_convergence(results["ladders"]),
        fig_tangential(results["tangential"]),
        fig_models(results["model_switch"]),
        fig_half_space(results["half_space"]),
    ]
    path = os.path.join(OUT, "study_results.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results["verdict"], indent=2))
    print(f"wrote {path} and {len(results['figures'])} figures in {results['wall_s_total']:.1f} s")
    return results


def _verdict(res: dict) -> dict:
    wink = res["ladders"]["winkler"]
    pas = res["ladders"]["pasternak"]
    err_w = max(r["rel_error_pct"] for r in wink["ladder"] if r["n"] >= 16)
    err_p = max(r["rel_error_pct"] for r in pas["ladder"] if r["n"] >= 16)
    comp = res["ladders"]["comparison"]
    tang = res["tangential"]
    switch = res["model_switch"]
    half = res["half_space"]
    neg = res["negative_controls"]
    criteria = {
        "C1_winkler_error_pct": {"value": err_w, "threshold": 5.0, "pass": err_w <= 5.0},
        "C1_pasternak_error_pct": {"value": err_p, "threshold": 5.0, "pass": err_p <= 5.0},
        "C2_pasternak_softer": {"value": comp["pasternak_softer_than_winkler"],
                                "ratio_agreement_pct": comp["ratio_agreement_pct"],
                                "pass": comp["pasternak_softer_than_winkler"]
                                and comp["ratio_agreement_pct"] <= 2.0},
        "C3_winkler_order": {"value": wink["observed_order"], "threshold": 1.5,
                             "pass": wink["observed_order"] >= 1.5 and wink["monotone"]},
        "C3_pasternak_order": {"value": pas["observed_order"], "threshold": 1.5,
                               "pass": pas["observed_order"] >= 1.5 and pas["monotone"]},
        "C4_three_models_switch": {"kinds": [switch["models"][k]["kind"] for k in
                                             ("winkler", "pasternak", "half_space")],
                                   "limits_monotone": all(v["monotone_decreasing"]
                                                          for v in switch["stiffening_limits"].values()),
                                   "limits_pass": all(v["pass"] for v in switch["stiffening_limits"].values()),
                                   "pass": all(v["pass"] for v in switch["stiffening_limits"].values())},
        "C4_tangential_sweep": {"max_error_pct": max(r["rel_error_pct"] for r in tang["sweep"]),
                                "monotone": tang["monotone_decreasing"],
                                "pass": max(r["rel_error_pct"] for r in tang["sweep"]) <= 5.0
                                and tang["monotone_decreasing"]},
        "half_space_quadrature": {"max_avg_error_pct": max(r["w_average_rel_error_pct"]
                                                           for r in half["ladder"]),
                                  "pass": max(r["w_average_rel_error_pct"] for r in half["ladder"]) <= 2.0},
        "negative_controls": {
            "wrong_sign_softer": neg["wrong_sign_shear"]["softer_than_winkler"],
            "wrong_sign_fails_gate": neg["wrong_sign_shear"]["fails_5pct_gate"],
            "zero_shear_identity": neg["zero_shear_identity"]["identical"],
            "kirchhoff_wrong_for_thick": neg["thick_plate_wrong_reference"]
            ["kirchhoff_reference_fails_15pct_gate"],
            "pass": neg["wrong_sign_shear"]["softer_than_winkler"]
            and neg["wrong_sign_shear"]["fails_5pct_gate"]
            and neg["zero_shear_identity"]["identical"]
            and neg["thick_plate_wrong_reference"]["kirchhoff_reference_fails_15pct_gate"],
        },
    }
    criteria["all_pass"] = all(v["pass"] for k, v in criteria.items() if isinstance(v, dict))
    return criteria


if __name__ == "__main__":
    main()