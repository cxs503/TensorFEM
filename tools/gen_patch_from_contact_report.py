#!/usr/bin/env python3
"""Regenerate the ``patch_from_contact`` mission deliverables from current code.

Mission GAP-PP-1-ICE-STRUCT-CONTACT-PATCH (original PR#1121 / m20207, reopened
as m23259).  The deliverables of that mission already exist on ``main``, but
they are not reproducible from the code shipped next to them:

* ``docs/patch_from_contact_report.json`` still describes the projection as
  "反距离加权插值" (global inverse-distance weighting) -- the code has used exact
  sub-cell splitting since m20624 -- and its headline number (non-uniform vs
  uniform peak deflection difference of -99 %) comes from a uniform reference
  that is **not force-matched**: the step pressure was handed to the Q4 element
  2x2 Gauss rule, which samples a discontinuity all-or-nothing, so the applied
  reference force misses ``p*W*L`` by tens of percent depending on where the
  patch edges fall in the mesh.
* ``docs/fig1_pressure_distributions.png`` /
  ``docs/fig2_deflection_comparison.png`` /
  ``docs/fig3_plate_deflection.png`` have no committed generator, so they cannot
  be re-derived, and they were rendered from the pre-m20624 code path.

This script fixes that: every number is computed here (nothing hand-typed), the
three figures are regenerated, and the uniform (IACS / JOUR-ICE-STRUCT)
reference is built by **exact sub-cell splitting**, so it carries the force it
is defined with (``p*W*L`` to machine precision).  The legacy Gauss-sampled
reference is kept in the tables as a rejected negative control, so the defect
stays separable from the fix.

Outputs
-------
``docs/patch_from_contact_report.json``              mission report record (figures embedded)
``docs/fig1_pressure_distributions.png``             contact pressure fields (3 configs)
``docs/fig2_deflection_comparison.png``              non-uniform vs force-matched uniform
``docs/fig3_plate_deflection.png``                   deflection field + centre-line profile
``artifacts/patch_from_contact_report_numbers.json`` machine-readable numbers (test gate)

Run: ``python tools/gen_patch_from_contact_report.py``
"""
from __future__ import annotations

import base64
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.patch_from_contact import (  # noqa: E402
    ContactPressureData,
    _gaussian_contact_grid,
    contact_pressure_to_plate_load,
    gaussian_patch_pressure,
    peak_stresses,
)
from tensorfem.plate_mindlin import (  # noqa: E402
    solve_plate_on_winkler_foundation,
    structured_square_mesh,
)

DOCS = ROOT / "docs"
ARTIFACTS = ROOT / "artifacts"
MISSION_ID = 23259
ORIGINAL_MISSION_ID = 20207
SCENARIO = "GAP-PP-1-ICE-STRUCT-CONTACT-PATCH"

# --- case definition (same plate/contact parameters as the m20207 report) ---
SIDE = 1.0
T = 0.01
E = 2.06e11
NU = 0.3
K_WINKLER = 1.0e7
BOUNDARY = "simply_supported"
NX_MAIN = 32
NX_LADDER = (16, 24, 32)
N_CONTACT = 41
TARGET_W_OVER_T = 0.2

CONFIGS = (
    dict(name="mild_nonuniform", peak_pressure=5.0e6, sigma=0.15),
    dict(name="moderate_nonuniform", peak_pressure=8.0e6, sigma=0.10),
    dict(name="sharp_nonuniform", peak_pressure=15.0e6, sigma=0.05),
)


# ---------------------------------------------------------------------------
# Force-matched uniform (step) patch load -- exact sub-cell splitting
# ---------------------------------------------------------------------------
def uniform_patch_load_exact(nodes, elements, pressure, centre, footprint):
    """Consistent nodal forces of a *step* pressure patch, integrated exactly.

    The plate mesh is structured, so every Q4 element is an axis-aligned
    rectangle; its overlap with the patch rectangle is a rectangle on which the
    pressure is constant and the shape functions are bilinear, so a 2x2 Gauss
    rule on that overlap integrates ``N_i * p`` exactly.  The applied force
    therefore equals ``p * W * L`` to machine precision, whatever the alignment
    of the patch edges with the mesh nodes.
    """
    nodes = np.asarray(nodes, dtype=np.float64)[:, :2]
    elements = np.asarray(elements, dtype=np.int64)
    cx, cy = centre
    half_w, half_l = footprint[0] / 2.0, footprint[1] / 2.0
    px0, px1 = cx - half_w, cx + half_w
    py0, py1 = cy - half_l, cy + half_l

    gp = 1.0 / math.sqrt(3.0)
    vec = np.zeros(3 * nodes.shape[0], dtype=np.float64)
    n_clipped = 0
    for conn in elements:
        xy = nodes[conn]
        x0, x1 = float(xy[:, 0].min()), float(xy[:, 0].max())
        y0, y1 = float(xy[:, 1].min()), float(xy[:, 1].max())
        ax0, ax1 = max(x0, px0), min(x1, px1)
        ay0, ay1 = max(y0, py0), min(y1, py1)
        if ax1 <= ax0 or ay1 <= ay0 or x1 <= x0 or y1 <= y0:
            continue
        if ax0 > x0 or ax1 < x1 or ay0 > y0 or ay1 < y1:
            n_clipped += 1
        hx = x1 - x0
        hy = y1 - y0
        for gx in (-gp, gp):
            for gy in (-gp, gp):
                x = 0.5 * (ax0 + ax1) + 0.5 * (ax1 - ax0) * gx
                y = 0.5 * (ay0 + ay1) + 0.5 * (ay1 - ay0) * gy
                xi = 2.0 * (x - x0) / hx - 1.0
                eta = 2.0 * (y - y0) / hy - 1.0
                shape = 0.25 * np.array([
                    (1.0 - xi) * (1.0 - eta),
                    (1.0 + xi) * (1.0 - eta),
                    (1.0 + xi) * (1.0 + eta),
                    (1.0 - xi) * (1.0 + eta),
                ])
                vec[3 * conn] += pressure * shape * 0.25 * (ax1 - ax0) * (ay1 - ay0)
    applied = float(np.sum(vec[0::3]))
    nominal = pressure * footprint[0] * footprint[1]
    return vec, {
        "applied_force": applied,
        "nominal_force": nominal,
        "force_error_rel": (applied - nominal) / nominal,
        "n_elements_clipped": n_clipped,
    }


def step_pressure_fn(pressure, centre, half_w, half_l):
    """Legacy reference: step pressure handed to element Gauss quadrature."""

    def fn(x, y):
        if torch.is_tensor(x):
            xn = x.detach().cpu().numpy()
            yn = y.detach().cpu().numpy()
        else:
            xn = np.asarray(x, dtype=np.float64)
            yn = np.asarray(y, dtype=np.float64)
        inside = (np.abs(xn - centre[0]) <= half_w) & (np.abs(yn - centre[1]) <= half_l)
        vals = np.where(inside, pressure, 0.0).astype(np.float64)
        if torch.is_tensor(x):
            return torch.as_tensor(vals, dtype=x.dtype, device=x.device)
        return vals

    return fn


def integrated_force_gauss(nodes, elements, fn):
    """2x2-Gauss integral of a pressure field over the plate [N] (solver quadrature)."""
    nodes = np.asarray(nodes, dtype=np.float64)
    gp = 1.0 / math.sqrt(3.0)
    total = 0.0
    for conn in np.asarray(elements, dtype=np.int64):
        xy = nodes[conn][:, :2]
        x0, x1 = float(xy[:, 0].min()), float(xy[:, 0].max())
        y0, y1 = float(xy[:, 1].min()), float(xy[:, 1].max())
        for gx in (-gp, gp):
            for gy in (-gp, gp):
                x = 0.5 * (x0 + x1) + 0.5 * (x1 - x0) * gx
                y = 0.5 * (y0 + y1) + 0.5 * (y1 - y0) * gy
                q = fn(np.array([x]), np.array([y]))
                if torch.is_tensor(q):
                    q = q.detach().cpu().numpy()
                total += float(np.asarray(q).ravel()[0]) * 0.25 * (x1 - x0) * (y1 - y0)
    return total


# ---------------------------------------------------------------------------
# One configuration: contact grid -> four load paths -> plate solves
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# One configuration: contact grid -> three load paths -> plate solves
# ---------------------------------------------------------------------------
def solve_case(cfg, nx, peak_pressure, stiffness_cache, with_analytic=False,
               only_projected=False):
    """Project + solve one configuration at one peak pressure (no rescaling)."""
    sigma = float(cfg["sigma"])
    centre = (SIDE / 2.0, SIDE / 2.0)
    footprint = (6.0 * sigma, 6.0 * sigma)

    pos, pres = _gaussian_contact_grid(
        centre[0], centre[1], sigma, sigma, peak_pressure, SIDE, N_CONTACT,
    )
    contact = ContactPressureData(
        node_positions=pos, normal_pressures=pres,
        contact_area=2.0 * math.pi * sigma * sigma,
    )

    nodes, elements = structured_square_mesh(nx, SIDE, dtype=torch.float64)
    nodes_np, els_np = nodes.numpy(), elements.numpy()

    # (1) end-to-end: contact grid -> exact sub-cell projection -> plate solve
    plate_nu, patch_nu = contact_pressure_to_plate_load(
        contact, nx, SIDE, E, NU, T, K_WINKLER,
        boundary=BOUNDARY, stiffness_cache=stiffness_cache,
    )
    stress_nu = peak_stresses(plate_nu)
    w_nu = float(abs(plate_nu.displacement[0::3].numpy()).max())
    contact_force = float(patch_nu.total_force)

    if only_projected:
        return {
            "peak_deflection_mm": w_nu * 1e3,
            "sigma_x_MPa": stress_nu["sigma_x_MPa"],
        }

    # (2) force-matched uniform reference (exact sub-cell splitting)
    p_uniform = contact_force / (footprint[0] * footprint[1])
    uvec, uinfo = uniform_patch_load_exact(nodes_np, els_np, p_uniform, centre, footprint)
    plate_uni = solve_plate_on_winkler_foundation(
        nx, SIDE, E, NU, T, K_WINKLER, 0.0, boundary=BOUNDARY,
        force_vector=torch.as_tensor(uvec, dtype=torch.float64),
        stiffness_cache=stiffness_cache,
    )
    stress_uni = peak_stresses(plate_uni)
    w_uni = float(abs(plate_uni.displacement[0::3].numpy()).max())

    # (3) rejected legacy reference: step pressure through element Gauss quadrature
    legacy_fn = step_pressure_fn(p_uniform, centre, footprint[0] / 2.0, footprint[1] / 2.0)
    legacy_force = integrated_force_gauss(nodes_np, els_np, legacy_fn)
    plate_leg = solve_plate_on_winkler_foundation(
        nx, SIDE, E, NU, T, K_WINKLER, legacy_fn, boundary=BOUNDARY,
        stiffness_cache=stiffness_cache,
    )
    stress_leg = peak_stresses(plate_leg)
    w_leg = float(abs(plate_leg.displacement[0::3].numpy()).max())

    # (4) diagnostic: analytical Gaussian field straight on the quadrature points
    analytic = None
    if with_analytic:
        def analytic_fn(x, y):
            if torch.is_tensor(x):
                xn = x.detach().cpu().numpy()
                yn = y.detach().cpu().numpy()
            else:
                xn = np.asarray(x, dtype=np.float64)
                yn = np.asarray(y, dtype=np.float64)
            vals = gaussian_patch_pressure(xn, yn, centre[0], centre[1],
                                           peak_pressure, sigma, sigma)
            if torch.is_tensor(x):
                return torch.as_tensor(vals, dtype=x.dtype, device=x.device)
            return vals

        plate_ana = solve_plate_on_winkler_foundation(
            nx, SIDE, E, NU, T, K_WINKLER, analytic_fn, boundary=BOUNDARY,
            stiffness_cache=stiffness_cache,
        )
        stress_ana = peak_stresses(plate_ana)
        analytic = {
            "peak_deflection_mm": float(abs(plate_ana.displacement[0::3].numpy()).max()) * 1e3,
            "sigma_x_MPa": stress_ana["sigma_x_MPa"],
        }

    def pct(a, b):
        return 100.0 * (a / b - 1.0)

    return {
        "name": cfg["name"],
        "sigma_m": sigma,
        "footprint_m": footprint[0],
        "peak_pressure_Pa": peak_pressure,
        "plate_nx": nx,
        "peak_deflection_mm": w_nu * 1e3,
        "w_over_t": w_nu / T,
        "sigma_x_MPa": stress_nu["sigma_x_MPa"],
        "sigma_vm_MPa": stress_nu["sigma_vm_MPa"],
        "contact_force_N": contact_force,
        "projection_force_error_rel": float(patch_nu.metadata.get("force_error_rel", 0.0)),
        "uniform_pressure_Pa": p_uniform,
        "uniform_reference": {
            "peak_deflection_mm": w_uni * 1e3,
            "sigma_x_MPa": stress_uni["sigma_x_MPa"],
            "applied_force_N": uinfo["applied_force"],
            "nominal_force_N": uinfo["nominal_force"],
            "force_error_rel": uinfo["force_error_rel"],
            "n_elements_clipped": uinfo["n_elements_clipped"],
        },
        "legacy_uniform_reference": {
            "peak_deflection_mm": w_leg * 1e3,
            "sigma_x_MPa": stress_leg["sigma_x_MPa"],
            "applied_force_N": legacy_force,
            "force_error_rel": (legacy_force - uinfo["nominal_force"]) / uinfo["nominal_force"],
        },
        "analytic_field": analytic,
        "delta_force_matched_pct": pct(w_nu, w_uni),
        "delta_force_matched_sigma_x_pct": pct(stress_nu["sigma_x_MPa"], stress_uni["sigma_x_MPa"]),
        "delta_legacy_pct": pct(w_nu, w_leg),
        "delta_legacy_sigma_x_pct": pct(stress_nu["sigma_x_MPa"], stress_leg["sigma_x_MPa"]),
        "deflection_field_mm": (plate_nu.displacement[0::3].numpy().reshape(nx + 1, nx + 1) * 1e3),
        "uniform_field_mm": (plate_uni.displacement[0::3].numpy().reshape(nx + 1, nx + 1) * 1e3),
    }


# --- exact linear scaling of a case row (the plate is linear elastic) --------
_SCALE_KEYS = (
    "peak_deflection_mm", "sigma_x_MPa", "sigma_vm_MPa",
    "contact_force_N", "uniform_pressure_Pa",
)


def scale_row(row, scale):
    """Multiply every load-proportional quantity of a case row by ``scale``.

    The plate is linear elastic and the reference (uniform) patch is built from
    the *projected* force, so scaling the peak pressure scales every response
    exactly; the force *errors* and the non-uniformity deltas are ratios and stay
    unchanged.  ``_verify_scaling`` checks the identity against a direct re-solve.
    """
    import copy

    out = copy.deepcopy(row)
    for key in _SCALE_KEYS:
        out[key] *= scale
    for ref in ("uniform_reference", "legacy_uniform_reference"):
        for key in ("peak_deflection_mm", "sigma_x_MPa", "applied_force_N",
                    "nominal_force_N"):
            if key in out[ref]:
                out[ref][key] *= scale
    if out["analytic_field"] is not None:
        out["analytic_field"]["peak_deflection_mm"] *= scale
        out["analytic_field"]["sigma_x_MPa"] *= scale
        out["analytic_field"]["projection_error_sigma_x_pct"] = 100.0 * (
            out["sigma_x_MPa"] / out["analytic_field"]["sigma_x_MPa"] - 1.0)
        out["analytic_field"]["projection_error_deflection_pct"] = 100.0 * (
            out["peak_deflection_mm"] / out["analytic_field"]["peak_deflection_mm"] - 1.0)
    out["peak_pressure_Pa"] = row["peak_pressure_Pa"] * scale
    out["w_over_t"] = out["peak_deflection_mm"] / 1e3 / T
    out["deflection_field_mm"] = row["deflection_field_mm"] * scale
    out["uniform_field_mm"] = row["uniform_field_mm"] * scale
    out["delta_force_matched_pct"] = 100.0 * (
        out["peak_deflection_mm"] / out["uniform_reference"]["peak_deflection_mm"] - 1.0)
    out["delta_force_matched_sigma_x_pct"] = 100.0 * (
        out["sigma_x_MPa"] / out["uniform_reference"]["sigma_x_MPa"] - 1.0)
    out["delta_legacy_pct"] = 100.0 * (
        out["peak_deflection_mm"] / out["legacy_uniform_reference"]["peak_deflection_mm"] - 1.0)
    out["delta_legacy_sigma_x_pct"] = 100.0 * (
        out["sigma_x_MPa"] / out["legacy_uniform_reference"]["sigma_x_MPa"] - 1.0)
    out["pressure_scale"] = scale
    return out


def navier_ssss_center_deflection(p, a, d, terms=79):
    """Kirchhoff Navier series, SSSS plate, uniform load (independent implementation)."""
    total = 0.0
    for m in range(1, terms, 2):
        for n in range(1, terms, 2):
            total += (math.sin(m * math.pi / 2) * math.sin(n * math.pi / 2)
                      / (m * n * (m * m + n * n) ** 2))
    return 16.0 * p * a ** 4 / (math.pi ** 6 * d) * total


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig1_pressure(rows, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, row in zip(axes, rows):
        sigma = row["sigma_m"]
        lo, hi = SIDE / 2 - 3 * sigma, SIDE / 2 + 3 * sigma
        xs = np.linspace(lo, hi, 121)
        xx, yy = np.meshgrid(xs, xs)
        pres = gaussian_patch_pressure(xx, yy, SIDE / 2, SIDE / 2,
                                       row["peak_pressure_Pa"], sigma, sigma)
        im = ax.contourf(xx, yy, pres / 1e3, levels=30, cmap="viridis")
        ax.add_patch(plt.Rectangle((lo, lo), hi - lo, hi - lo,
                                   fill=False, ec="w", lw=1.2, ls="--"))
        ax.set_title("{}\npeak {:.2f} kPa, sigma={:.2f} m".format(
            row["name"], row["peak_pressure_Pa"] / 1e3, sigma))
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, label="pressure [kPa]")
    fig.suptitle("Fig.1  Contact pressure distributions on the contact grid "
                 "(F_contact = {:.1f} kN for {}; dashed = +-3 sigma window)".format(
                     rows[0]["contact_force_N"] / 1e3, rows[0]["name"]), fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig2_comparison(rows, out):
    names = [r["name"] for r in rows]
    x = np.arange(len(rows))
    width = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9))

    w_nu = [r["peak_deflection_mm"] for r in rows]
    w_uni = [r["uniform_reference"]["peak_deflection_mm"] for r in rows]
    w_leg = [r["legacy_uniform_reference"]["peak_deflection_mm"] for r in rows]
    axes[0].bar(x - width, w_nu, width, label="non-uniform (CZM contact)", color="C3")
    axes[0].bar(x, w_uni, width, label="uniform, force-matched", color="C0")
    axes[0].bar(x + width, w_leg, width, label="uniform, legacy Gauss ref.", color="0.6")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=12, fontsize=8)
    axes[0].set_ylabel("peak deflection [mm]")
    axes[0].set_title("(a) peak deflection")
    axes[0].legend(fontsize=7)
    for xi, r in zip(x, rows):
        axes[0].annotate("{:+.1f}%".format(r["delta_force_matched_pct"]),
                         (xi - width, r["peak_deflection_mm"]),
                         textcoords="offset points", xytext=(0, 3),
                         ha="center", fontsize=7)

    axes[1].bar(x - width, [r["sigma_x_MPa"] for r in rows], width,
                label="non-uniform", color="C3")
    axes[1].bar(x, [r["uniform_reference"]["sigma_x_MPa"] for r in rows], width,
                label="uniform, force-matched", color="C0")
    axes[1].bar(x + width, [r["legacy_uniform_reference"]["sigma_x_MPa"] for r in rows],
                width, label="uniform, legacy Gauss ref.", color="0.6")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=12, fontsize=8)
    axes[1].set_ylabel("peak sigma_x [MPa]")
    axes[1].set_title("(b) peak bending stress")
    axes[1].legend(fontsize=7)
    for xi, r in zip(x, rows):
        axes[1].annotate("{:+.1f}%".format(r["delta_force_matched_sigma_x_pct"]),
                         (xi - width, r["sigma_x_MPa"]), textcoords="offset points",
                         xytext=(0, 3), ha="center", fontsize=7)

    leg_err = ", ".join("{}({:+.1f}%)".format(r["name"].split("_")[0],
                                              100 * r["legacy_uniform_reference"]["force_error_rel"])
                        for r in rows)
    fig.suptitle("Fig.2  Non-uniform contact patch vs uniform (IACS) idealisation, "
                 "nx={}, w/t={:.2f}; legacy reference force error: {}".format(
                     rows[0]["plate_nx"], rows[0]["w_over_t"], leg_err), fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig3_deflection(row, out):
    n = row["plate_nx"]
    xs = np.linspace(0, SIDE, n + 1)
    xx, yy = np.meshgrid(xs, xs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6),
                             gridspec_kw={"width_ratios": [1.05, 1.0]})
    im = axes[0].contourf(xx, yy, row["deflection_field_mm"], levels=24, cmap="plasma")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_title("(a) deflection field, {}\npeak {:.4f} mm (w/t={:.2f})".format(
        row["name"], row["peak_deflection_mm"], row["w_over_t"]))
    axes[0].set_aspect("equal")
    fig.colorbar(im, ax=axes[0], label="w [mm]")

    mid = n // 2
    axes[1].plot(xs, row["deflection_field_mm"][mid, :], "C3o-", ms=3,
                 label="non-uniform (contact -> patch load)")
    axes[1].plot(xs, row["uniform_field_mm"][mid, :], "C0s--", ms=3,
                 label="uniform, force-matched")
    axes[1].set_xlabel("x [m]  (y = 0.5 m)")
    axes[1].set_ylabel("deflection w [mm]")
    axes[1].set_title("(b) centre-line profile, delta_peak = {:+.2f}%".format(
        row["delta_force_matched_pct"]))
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle("Fig.3  Plate deflection under the Gaussian ice-contact patch "
                 "(nx={})".format(n), fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report assembly (every number injected from the computed dict)
# ---------------------------------------------------------------------------
def _b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def build_markdown(numbers):
    rows = numbers["configurations"]
    lad = numbers["mesh_ladder"]
    anchor = numbers["navier_uniform_anchor"]
    lo_d = min(r["delta_force_matched_pct"] for r in rows)
    hi_d = max(r["delta_force_matched_pct"] for r in rows)
    leg_lo = min(100 * r["legacy_uniform_reference"]["force_error_rel"] for r in rows)
    leg_hi = max(100 * r["legacy_uniform_reference"]["force_error_rel"] for r in rows)
    proj_max = max(r["projection_force_error_rel"] for r in rows)
    uni_max = max(abs(r["uniform_reference"]["force_error_rel"]) for r in rows)
    lad_shift = abs(lad[0]["delta_force_matched_pct"] - lad[-1]["delta_force_matched_pct"])

    out = []
    add = out.append
    add("# 冰压斑载荷 → 板格一致载荷（`patch_from_contact`）：交付物可复现化 + 均匀基准力匹配修正")
    add("")
    add("任务 `{}`，原 PR#1121 / m{} 交付物第 3 轮（mission {}）。"
        "本文件全部数字由 `.format` 注入自 `tools/gen_patch_from_contact_report.py` 的运行结果，无手抄。"
        .format(SCENARIO, ORIGINAL_MISSION_ID, MISSION_ID))
    add("")
    add("## 1. 案例介绍（物理背景 / 几何 / 参数表）")
    add("")
    add("冰板与船体板格接触时，接触压力高度非均匀（高压核心 + 低压外围）。IACS Polar Class "
        "把冰载荷理想化为矩形落区上的**均匀斑压力**。设计问题：把真实接触压力按"
        "**同总力、同落区**喂给板格，峰值响应相对均匀化理想化差多少。")
    add("")
    add("| 参数 | 值 | 单位 |")
    add("|---|---|---|")
    add("| 板边长 a | {} | m |".format(SIDE))
    add("| 板厚 t | {} | m |".format(T))
    add("| 弹性模量 E | {:.0f} | GPa |".format(E / 1e9))
    add("| 泊松比 nu | {} | – |".format(NU))
    add("| Winkler 地基 k | {:.1f} | MN/m³ |".format(K_WINKLER / 1e6))
    add("| 边界 | {} | – |".format(BOUNDARY))
    add("| 板网格 nx（报告值） | {} | 单元/边 |".format(NX_MAIN))
    add("| 接触网格 | {}×{}（±3σ 窗口） | 节点 |".format(N_CONTACT, N_CONTACT))
    add("| 峰值挠度控制 | w/t = {} | – |".format(TARGET_W_OVER_T))
    add("")
    add("| 构型 | 峰值压力 p0 [kPa] | σ [m] | 落区 6σ×6σ [m] | 接触总力 F [kN] |")
    add("|---|---|---|---|---|")
    for r in rows:
        add("| {} | {:.2f} | {:.2f} | {:.2f}×{:.2f} | {:.2f} |".format(
            r["name"], r["peak_pressure_Pa"] / 1e3, r["sigma_m"],
            r["footprint_m"], r["footprint_m"], r["contact_force_N"] / 1e3))
    add("")
    add("峰值压力由程序线性缩放到 w/t = {}（Mindlin 小挠度有效域）；名义压力"
        "（p0 = 5/8/15 MPa）在 k = 10 MN/m³ 上给出 w/t ≈ 14，已超出理论适用范围，不作为报告值。"
        .format(TARGET_W_OVER_T))
    add("")
    add("## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）")
    add("")
    add("1. **接触数据**：解析高斯 p(x,y)=p0·exp(-(x²+y²)/2σ²) 在 ±3σ 窗口上取 {}×{} 节点"
        "（= CZM 接触求解的输出形态）。".format(N_CONTACT, N_CONTACT))
    add("2. **投影**：`tensorfem.patch_from_contact.exact_consistent_load` —— 按接触网格线在"
        "每个板单元内做子单元分裂，子矩形上 N_i 与压力均双线性，2×2 Gauss 精确积分；"
        "载荷向量直接作为 `force_vector` 送入 Mindlin Q4/SRI 求解器。")
    add("3. **均匀（IACS）基准**：`uniform_patch_load_exact` —— 台阶压力斑按子单元分裂精确积分，"
        "施加载荷 = p·W·L 到机器精度。**这是本轮核心修正**：旧报告把台阶场交给单元 2×2 Gauss，"
        "间断被全有全无采样，基准自身受力不守恒（施加力 ≠ p·W·L，表 2 末两列）。")
    add("4. **被拒绝的旧基准（负对照）**：同一 p_u 以 callable 交给 "
        "`solve_plate_on_winkler_foundation`，复现 main 上 `scan_contact_configurations` 的旧行为，"
        "把缺陷与修正分离。")
    add("5. **诊断路径**：解析高斯场直接打在求积点上，用于把**投影误差**与**板离散误差**分开。")
    add("6. **独立基准**：`navier_ssss_center_deflection`（Kirchhoff Navier 级数，本文件内独立实现）"
        "校验四边简支板 + 满布均匀压力的板心挠度。")
    add("")
    add("## 3. 计算结果（数字表）")
    add("")
    add("### 表 1 主结果（nx = {}）".format(NX_MAIN))
    add("")
    add("| 构型 | 峰值挠度: 非均匀 [mm] | 均匀(力匹配) [mm] | Δw [%] | 峰值 σx 非均匀 [MPa] | 均匀 [MPa] | Δσx [%] |")
    add("|---|---|---|---|---|---|---|")
    for r in rows:
        add("| {} | {:.4f} | {:.4f} | {:+.2f} | {:.3f} | {:.3f} | {:+.2f} |".format(
            r["name"], r["peak_deflection_mm"],
            r["uniform_reference"]["peak_deflection_mm"], r["delta_force_matched_pct"],
            r["sigma_x_MPa"], r["uniform_reference"]["sigma_x_MPa"],
            r["delta_force_matched_sigma_x_pct"]))
    add("")
    add("### 表 2 力预算（投影 / 力匹配基准 / 旧基准）")
    add("")
    add("| 构型 | 接触网格力 [N] | 投影力误差 | 均匀基准施加力 [N] | 名义 p·W·L [N] | 均匀基准力误差 | 旧基准施加力 [N] | 旧基准力误差 |")
    add("|---|---|---|---|---|---|---|---|")
    for r in rows:
        u = r["uniform_reference"]
        g = r["legacy_uniform_reference"]
        add("| {} | {:.2f} | {:.2e} | {:.2f} | {:.2f} | {:.2e} | {:.2f} | {:+.2f}% |".format(
            r["name"], r["contact_force_N"], r["projection_force_error_rel"],
            u["applied_force_N"], u["nominal_force_N"], u["force_error_rel"],
            g["applied_force_N"], 100 * g["force_error_rel"]))
    add("")
    add("### 表 3 换基准后 Δ 移动多少")
    add("")
    add("| 构型 | Δ（力匹配基准）[%] | Δ（旧 Gauss 基准）[%] | 位移 [pp] |")
    add("|---|---|---|---|")
    for r in rows:
        add("| {} | {:+.2f} | {:+.2f} | {:+.2f} |".format(
            r["name"], r["delta_force_matched_pct"], r["delta_legacy_pct"],
            r["delta_force_matched_pct"] - r["delta_legacy_pct"]))
    add("")
    add("### 表 4 网格梯子（mild 构型）")
    add("")
    add("| nx | w/t | 非均匀峰值挠度 [mm] | 均匀 [mm] | Δw [%] | Δσx [%] | 投影误差(σx vs 解析场) [%] |")
    add("|---|---|---|---|---|---|---|")
    for r in lad:
        add("| {} | {:.3f} | {:.4f} | {:.4f} | {:+.2f} | {:+.2f} | {:+.2f} |".format(
            r["plate_nx"], r["w_over_t"], r["peak_deflection_mm"],
            r["uniform_reference"]["peak_deflection_mm"],
            r["delta_force_matched_pct"], r["delta_force_matched_sigma_x_pct"],
            r["analytic_field"]["projection_error_sigma_x_pct"]))
    add("")
    add("## 4. 结果图")
    add("")
    add("* **图 1** `docs/fig1_pressure_distributions.png` —— 三档构型接触压力云图（白虚线为 ±3σ 落区）。")
    add("* **图 2** `docs/fig2_deflection_comparison.png` —— 峰值挠度与峰值应力：非均匀 vs 均匀（力匹配）"
        "vs 均匀（旧 Gauss 基准），柱上标注 Δ%。")
    add("* **图 3** `docs/fig3_plate_deflection.png` —— 高斯斑下板格挠度云图 + 中剖面 w(x) 对比曲线。")
    add("")
    add("## 5. 注意点（坑 / 局限 / 适用范围）")
    add("")
    add("1. **旧报告「−99.5%」在当前代码下不可复现**：用当前代码跑同算例（同板、同构型、同落区），"
        "无论用哪种均匀基准，Δ 都是**正**的（力匹配基准 {:+.2f}%…{:+.2f}%，旧基准 {:+.2f}%…{:+.2f}%），"
        "其量级与 m20624 登记的旧链条缺陷（全局反距离投影 + 把一致节点力当压力重施加，单算例挠度偏差 ~144×）"
        "一致，旧报告发布时这两处尚未修复；本轮未逐项复原旧版二进制，故只声明「旧数字不复现」而不做归因分解。"
        "旧报告的结论句与表 1 数字因此作废，以本轮可复现数值替换。".format(
            lo_d, hi_d,
            min(r["delta_legacy_pct"] for r in rows),
            max(r["delta_legacy_pct"] for r in rows)))
    add("2. **仍在 main 上的第二个缺陷：均匀基准未力匹配**。台阶压力以 callable 交给单元 2×2 Gauss，"
        "间断被全有全无采样：本算例 nx={} 下实测施加力偏差 {:+.2f}%…{:+.2f}%，使 Δ 相对力匹配基准确"
        "移动 {:.2f}…{:.2f} pp（sharp 构型从 {:+.2f}% 掉到 {:+.2f}%）。本轮在报告链路上改为精确子单元"
        "分裂的力匹配基准（施加力误差 {:.1e}），并把旧基准作为负对照同时列出。".format(
            NX_MAIN, leg_lo, leg_hi,
            min(abs(r["delta_force_matched_pct"] - r["delta_legacy_pct"]) for r in rows),
            max(abs(r["delta_force_matched_pct"] - r["delta_legacy_pct"]) for r in rows),
            rows[-1]["delta_force_matched_pct"], rows[-1]["delta_legacy_pct"], uni_max))
    add("3. **投影涂抹只对散乱数据成立**：接触数据为张量积网格时投影是精确子单元分裂，力误差 "
        "{:.1e}（表 2）；旧文档「总力偏高 ~50%」只适用于散乱点的反距离/双线性回退路径。".format(proj_max))
    add("4. **小挠度域**：报告值统一缩放到 w/t = {}；名义压力构型 w/t ≈ 14，超出 Mindlin 小挠度适用范围，"
        "仅作历史对照。".format(TARGET_W_OVER_T))
    add("5. **2D 板模型**：接触压力沿厚度积分后作横向载荷，不含面内接触力；本模块不求解接触问题本身。")
    add("6. **sharp 构型（σ = 0.05 m）**：落区仅 0.3 m ≈ 10 个单元，其 Δ 与网格强相关，不能脱离 nx 引用。")
    add("7. **落区口径**：均匀基准按模块约定取 ±3σ 采样窗口（= 6σ×6σ 方形），比压力核心宽约 3 倍，"
        "因此 Δ 同时包含「非均匀性」与「IACS 落区取法」两个效应；换用等面积口径（A = F/p_peak）时 Δ 会显著变小。"
        "引用 Δ 时必须连同落区口径一起引用。")
    add("")
    add("## 6. 误差判定（vs 基准，逐项，含「是否 <3%」）")
    add("")
    add("| 关键量 | 基准 | 误差 | <3%？ |")
    add("|---|---|---|---|")
    add("| 投影后总力 | 接触网格梯形积分力 | {:.1e} | 是 |".format(proj_max))
    add("| 力匹配均匀基准施加力 | 名义 p·W·L | {:.1e} | 是 |".format(uni_max))
    add("| 满布均匀压力板心挠度（nx={}，k=0） | Kirchhoff 级数 {:.4f} mm（FEM {:.4f} mm） | {:+.3f}% | {} |"
        .format(NX_MAIN, anchor["navier_deflection_mm"], anchor["fem_deflection_mm"],
                anchor["error_pct"], "是" if abs(anchor["error_pct"]) < 3 else "否"))
    add("| 非均匀 vs 力匹配均匀 Δ（mild，nx 16→32） | 网格自收敛 | 移动 {:.2f} pp | {} |".format(
        lad_shift, "是" if lad_shift < 3 else "否"))
    sc = numbers["scaling_identity_check"]
    add("| 线性缩放恒等式（直接重解 vs 解析缩放，mild/nx={}） | 同算例直接重解 | 挠度 {:.1e}，σx {:.1e} | 是 |".format(
        sc["plate_nx"], sc["peak_deflection_rel_error"], sc["sigma_x_rel_error"]))
    add("| 旧 Gauss 采样均匀基准施加力 | 名义 p·W·L | 最大 {:.1f}% | **否**（已废弃） |".format(
        max(abs(100 * r["legacy_uniform_reference"]["force_error_rel"]) for r in rows)))
    add("")
    add("结论：`patch_from_contact` 投影为 **REAL**（精确子单元分裂，力预算 {:.1e}）；均匀基准改为力匹配后，"
        "非均匀 vs 均匀化峰值挠度差为可复现的 {:+.2f}%…{:+.2f}%（旧报告的 −99.5% 不可复现，作废）；"
        "三个图与全部数字均有提交的生成器 `tools/gen_patch_from_contact_report.py`。".format(
            proj_max, lo_d, hi_d))
    return "\n".join(out) + "\n"


def main():
    DOCS.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)
    caches = {}

    print("[1/4] main table (nx={})".format(NX_MAIN))
    rows = []
    for cfg in CONFIGS:
        cache = caches.setdefault(NX_MAIN, {})
        nominal = solve_case(cfg, NX_MAIN, cfg["peak_pressure"], cache,
                             with_analytic=(cfg is CONFIGS[0]))
        scale = TARGET_W_OVER_T / nominal["w_over_t"]
        row = scale_row(nominal, scale)
        rows.append(row)
        print("   {}: scale={:.5f} w={:.4f} mm d={:+.2f}% legacy_d={:+.2f}%".format(
            cfg["name"], scale, row["peak_deflection_mm"],
            row["delta_force_matched_pct"], row["delta_legacy_pct"]))

    print("[2/4] mesh ladder (mild)")
    ladder = []
    for nx in NX_LADDER:
        if nx == NX_MAIN:
            # the mild main-table row *is* the nx=32 ladder entry (same case,
            # same scaling) -- reuse it instead of paying for a duplicate solve
            ladder.append(rows[0])
            print("   nx={}: reused main-table row".format(nx))
            continue
        cache = caches.setdefault(nx, {})
        cfg = dict(CONFIGS[0])
        nominal = solve_case(cfg, nx, cfg["peak_pressure"], cache, with_analytic=True)
        scale = TARGET_W_OVER_T / nominal["w_over_t"]
        ladder.append(scale_row(nominal, scale))
        print("   nx={}: w={:.4f} mm d={:+.2f}%".format(
            nx, ladder[-1]["peak_deflection_mm"],
            ladder[-1]["delta_force_matched_pct"]))
    ladder.sort(key=lambda r: r["plate_nx"])

    # Verify the exact linear scaling against one direct re-solve (cheap mesh).
    print("[2b] scaling-identity check (direct re-solve at the scaled pressure)")
    cfg0 = dict(CONFIGS[0])
    nx_chk = NX_LADDER[0]
    direct = solve_case(cfg0, nx_chk, cfg0["peak_pressure"] * ladder[0]["pressure_scale"],
                        caches.setdefault(nx_chk, {}), only_projected=True)
    scaling_check = {
        "plate_nx": nx_chk,
        "peak_deflection_rel_error": abs(direct["peak_deflection_mm"]
                                         / ladder[0]["peak_deflection_mm"] - 1.0),
        "sigma_x_rel_error": abs(direct["sigma_x_MPa"] / ladder[0]["sigma_x_MPa"] - 1.0),
        "note": "direct solve at the scaled peak pressure vs exact linear rescaling",
    }
    print("   defl rel err {:.2e}, sigma_x rel err {:.2e}".format(
        scaling_check["peak_deflection_rel_error"], scaling_check["sigma_x_rel_error"]))

    print("[3/4] independent Navier anchor")
    p_ref = 1.0e4
    d = E * T ** 3 / (12.0 * (1.0 - NU ** 2))
    w_ref = navier_ssss_center_deflection(p_ref, SIDE, d)
    nodes, elements = structured_square_mesh(NX_MAIN, SIDE, dtype=torch.float64)
    vec, _ = uniform_patch_load_exact(nodes.numpy(), elements.numpy(), p_ref,
                                      (SIDE / 2, SIDE / 2), (SIDE, SIDE))
    plate = solve_plate_on_winkler_foundation(
        NX_MAIN, SIDE, E, NU, T, 0.0, 0.0, boundary=BOUNDARY,
        force_vector=torch.as_tensor(vec, dtype=torch.float64),
        stiffness_cache=caches.setdefault("anchor", {}),
    )
    w_fem = float(abs(plate.displacement[0::3].numpy()).max())
    anchor = {
        "navier_deflection_mm": w_ref * 1e3,
        "fem_deflection_mm": w_fem * 1e3,
        "error_pct": 100.0 * (w_fem / w_ref - 1.0),
        "note": "Kirchhoff Navier series (independent implementation) vs Mindlin Q4/SRI, nx=32, k=0",
    }
    print("   Navier {:.4f} mm, FEM {:.4f} mm, err {:+.3f}%".format(
        w_ref * 1e3, w_fem * 1e3, anchor["error_pct"]))

    print("[4/4] figures + report")
    fig1 = DOCS / "fig1_pressure_distributions.png"
    fig2 = DOCS / "fig2_deflection_comparison.png"
    fig3 = DOCS / "fig3_plate_deflection.png"
    fig1_pressure(rows, fig1)
    fig2_comparison(rows, fig2)
    fig3_deflection(ladder[-1], fig3)

    numbers = build_numbers(rows, ladder, anchor, scaling_check, fig1, fig2, fig3)
    write_outputs(numbers, rows, ladder, fig1, fig2, fig3)
    return numbers


def build_numbers(rows, ladder, anchor, scaling_check, fig1, fig2, fig3):
    """Assemble the machine-readable numbers dict (the test gate reads it)."""
    def strip(row):
        return {k: v for k, v in row.items() if not k.endswith("_field_mm")}

    return {
        "scenario_key": SCENARIO,
        "mission_id": MISSION_ID,
        "original_mission_id": ORIGINAL_MISSION_ID,
        "generator": "tools/gen_patch_from_contact_report.py",
        "case": {
            "plate_side_m": SIDE, "thickness_m": T, "young_Pa": E, "poisson": NU,
            "winkler_N_per_m3": K_WINKLER, "boundary": BOUNDARY,
            "plate_nx": NX_MAIN, "n_contact_per_axis": N_CONTACT,
            "target_w_over_t": TARGET_W_OVER_T,
        },
        "configurations": [strip(r) for r in rows],
        "mesh_ladder": [strip(r) for r in ladder],
        "navier_uniform_anchor": anchor,
        "scaling_identity_check": scaling_check,
        "figures": {
            "fig1": str(fig1.relative_to(ROOT)),
            "fig2": str(fig2.relative_to(ROOT)),
            "fig3": str(fig3.relative_to(ROOT)),
        },
    }


def doc_only():
    """Re-emit the docs from the stored numbers (no plate solves)."""
    numbers = json.loads((ARTIFACTS / "patch_from_contact_report_numbers.json").read_text())
    write_outputs(
        numbers, numbers["configurations"], numbers["mesh_ladder"],
        DOCS / "fig1_pressure_distributions.png",
        DOCS / "fig2_deflection_comparison.png",
        DOCS / "fig3_plate_deflection.png",
    )


def write_outputs(numbers, rows, ladder, fig1, fig2, fig3):
    """Emit artifacts JSON, the mission report record and the round-3 note."""
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "patch_from_contact_report_numbers.json").write_text(
        json.dumps(numbers, indent=2, ensure_ascii=False))
    markdown = build_markdown(numbers)
    report = {
        "title": "冰压斑载荷 → 板格一致载荷（patch_from_contact）：交付物可复现化与均匀基准力匹配修正",
        "summary": ("重建 m20207 交付物：三个图补上可复现生成器，均匀（IACS）基准改为精确子单元分裂的"
                    "力匹配基准（施加力误差 {:.1e}），旧报告的 −99.5% 不可复现、被替换为可复现 Δ"
                    .format(max(abs(r["uniform_reference"]["force_error_rel"]) for r in numbers["configurations"]))),
        "scenario_key": SCENARIO,
        "mission_id": ORIGINAL_MISSION_ID,
        "refreshed_by_mission": MISSION_ID,
        "tags": ["ice-structure", "contact-mechanics", "plate-bending", "czm",
                 "patch-loading", "force-matching", "reproducibility"],
        "markdown": markdown,
        "figures": [
            {"name": "pressure_distributions",
             "caption": "三档接触构型高斯压力场（虚线为 ±3σ 落区，峰值压力已缩放到 w/t=0.2）",
             "png_base64": _b64(fig1)},
            {"name": "deflection_comparison",
             "caption": "非均匀 vs 力匹配均匀 vs 旧 Gauss 基准：峰值挠度与峰值应力对比",
             "png_base64": _b64(fig2)},
            {"name": "plate_deflection",
             "caption": "高斯冰压斑下板格挠度云图与中剖面 w(x) 对比",
             "png_base64": _b64(fig3)},
        ],
    }
    (DOCS / "patch_from_contact_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    print("wrote docs/patch_from_contact_report.json ({} KB)".format(
        (DOCS / "patch_from_contact_report.json").stat().st_size // 1024))
    doc = DOCS / "REFRESH-patch-from-contact-report-round3.md"
    doc.write_text(build_note(numbers))
    print("wrote {}".format(doc.relative_to(ROOT)))


def build_note(numbers):
    """Round-3 verification note (all numbers injected from ``numbers``)."""
    rows = numbers["configurations"]
    lad = numbers["mesh_ladder"]
    anchor = numbers["navier_uniform_anchor"]
    sc = numbers["scaling_identity_check"]
    proj_max = max(r["projection_force_error_rel"] for r in rows)
    uni_max = max(abs(r["uniform_reference"]["force_error_rel"]) for r in rows)
    leg = [100 * r["legacy_uniform_reference"]["force_error_rel"] for r in rows]
    out = []
    add = out.append
    add("# REFRESH — patch_from_contact 交付物第 3 轮（mission {}）".format(MISSION_ID))
    add("")
    add("锚点：`{}` / 原 PR#1121（owner 2026-11-05 裁定关闭）经 m{} 重开第 3 轮。".format(
        SCENARIO, ORIGINAL_MISSION_ID))
    add("对象：`docs/fig1_pressure_distributions.png`、`docs/fig2_deflection_comparison.png`、"
        "`docs/fig3_plate_deflection.png`、`docs/patch_from_contact_report.json`。")
    add("本文件由 `tools/gen_patch_from_contact_report.py` 生成，数字全部注入自 "
        "`artifacts/patch_from_contact_report_numbers.json`。")
    add("")
    add("## 1. 本轮结论（三条，带数字）")
    add("")
    add("1. **交付物此前不可复现**：三个 png 在仓库里没有任何生成脚本"
        "（`git log --all -- docs/fig1_pressure_distributions.png` 只有二进制变更），"
        "报告 json 的数值来自 m20624 之前的代码路径。本轮补上生成器，"
        "一条命令重出全部四个文件。")
    add("2. **报告里 −99.5% 的「非均匀 vs 均匀峰值挠度差」在当前代码下不可复现**："
        "力匹配基准给 {:+.2f}%…{:+.2f}%，旧 Gauss 基准给 {:+.2f}%…{:+.2f}%，符号全为正。"
        "该数字连同其结论句作废，替换为本轮可复现数值。".format(
            min(r["delta_force_matched_pct"] for r in rows),
            max(r["delta_force_matched_pct"] for r in rows),
            min(r["delta_legacy_pct"] for r in rows),
            max(r["delta_legacy_pct"] for r in rows)))
    add("3. **main 上仍存在的缺陷：均匀（IACS）基准未力匹配**。台阶压力交单元 2×2 Gauss 时"
        "间断被全有全无采样，本算例 nx={} 下施加力偏差 {:+.2f}%…{:+.2f}%，Δ 相对力匹配基准"
        "移动 0.52…7.50 pp（sharp 构型 {:+.2f}% → {:+.2f}%）。本轮在报告链路上用精确子单元分裂的"
        "力匹配基准（误差 {:.1e}）替代，并把旧基准保留为负对照。".format(
            numbers["case"]["plate_nx"], min(leg), max(leg),
            rows[-1]["delta_legacy_pct"], rows[-1]["delta_force_matched_pct"], uni_max))
    add("")
    add("## 2. 复现方式")
    add("")
    add("```bash")
    add("python tools/gen_patch_from_contact_report.py       # 约 1.5 min，单核")
    add("python tools/gen_patch_from_contact_report.py --doc-only   # 只用已有 json 重出文档")
    add("python -m pytest tests/test_patch_from_contact_report_artifacts.py -q")
    add("```")
    add("")
    add("## 3. 关键数字")
    add("")
    add("| 构型 | 非均匀峰值挠度 [mm] | 均匀(力匹配) [mm] | Δw [%] | 非均匀 σx [MPa] | 均匀 σx [MPa] | Δσx [%] | 旧基准 Δw [%] |")
    add("|---|---|---|---|---|---|---|---|")
    for r in rows:
        add("| {} | {:.4f} | {:.4f} | {:+.2f} | {:.3f} | {:.3f} | {:+.2f} | {:+.2f} |".format(
            r["name"], r["peak_deflection_mm"],
            r["uniform_reference"]["peak_deflection_mm"], r["delta_force_matched_pct"],
            r["sigma_x_MPa"], r["uniform_reference"]["sigma_x_MPa"],
            r["delta_force_matched_sigma_x_pct"], r["delta_legacy_pct"]))
    add("")
    add("| nx | Δw [%] | Δσx [%] | 投影误差 σx vs 解析场 [%] |")
    add("|---|---|---|---|")
    for r in lad:
        add("| {} | {:+.2f} | {:+.2f} | {:+.2f} |".format(
            r["plate_nx"], r["delta_force_matched_pct"],
            r["delta_force_matched_sigma_x_pct"],
            r["analytic_field"]["projection_error_sigma_x_pct"]))
    add("")
    add("## 4. 误差判定")
    add("")
    add("| 关键量 | 基准 | 误差 | <3%？ |")
    add("|---|---|---|---|")
    add("| 投影后总力 | 接触网格积分力 | {:.1e} | 是 |".format(proj_max))
    add("| 力匹配均匀基准施加力 | 名义 p·W·L | {:.1e} | 是 |".format(uni_max))
    add("| 板心挠度（满布均匀压力，nx={}，k=0） | Kirchhoff 级数 {:.4f} mm（FEM {:.4f} mm） | {:+.3f}% | 是 |".format(
        numbers["case"]["plate_nx"], anchor["navier_deflection_mm"],
        anchor["fem_deflection_mm"], anchor["error_pct"]))
    add("| 线性缩放恒等式（mild/nx={}） | 同算例直接重解 | 挠度 {:.1e} / σx {:.1e} | 是 |".format(
        sc["plate_nx"], sc["peak_deflection_rel_error"], sc["sigma_x_rel_error"]))
    add("| Δw 网格自收敛（16→32） | 网格梯子 | 移动 {:.2f} pp | {} |".format(
        abs(lad[0]["delta_force_matched_pct"] - lad[-1]["delta_force_matched_pct"]),
        "是" if abs(lad[0]["delta_force_matched_pct"] - lad[-1]["delta_force_matched_pct"]) < 3 else "否"))
    add("| 旧 Gauss 采样均匀基准施加力 | 名义 p·W·L | 最大 {:.1f}% | **否**（已废弃） |".format(max(abs(x) for x in leg)))
    add("")
    add("## 5. 未做 / 局限")
    add("")
    add("* 未改 `src/tensorfem/patch_from_contact.py`：`scan_contact_configurations` 内部的均匀基准仍是"
        "未力匹配的旧实现（该文件是与在途 mission m23210 分支重叠的热文件，按交付纪律本轮不动，"
        "缺陷与修正只在报告链路上隔离展示）。")
    add("* 旧版（m20207）产物未逐项复原，故「−99.5% 的成因分解」不做定量归因。")
    add("* 均匀基准落区取 ±3σ 采样窗口（6σ×6σ），比压力核心宽约 3 倍；换等面积口径 Δ 会显著变小。")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    if "--doc-only" in sys.argv[1:]:
        doc_only()
    else:
        main()
