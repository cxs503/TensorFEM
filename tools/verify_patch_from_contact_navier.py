"""Verification of GAP-PP-1 patch_from_contact: unit audit + Navier cross-check.

Runs four independent checks on ``tensorfem.patch_from_contact`` and writes a
machine-readable artifact plus figures:

``A. unit_audit``
    Reproduces the legacy end-to-end chain (consistent nodal *forces* re-applied
    as *pressures*) and compares it with the fixed chain and with a direct
    uniform-pressure solve.  Quantifies the factor that was missing.
``B. force_conservation``
    Projected patch force vs the analytical Gaussian force, for three contact
    configurations and three contact-grid resolutions (contact-data
    discretisation study).
``C. nonuniform_scan``
    ``scan_contact_configurations``: non-uniform (contact-projected) patch vs a
    uniform patch of equal total force and footprint, on **peak face stress**.
``D. navier_crosscheck``
    Peak bending stress against the independent Kirchhoff Navier series
    (``tensorfem.ice_panel_patch_contact.navier_rect_moments``) and against the
    independent finite-element path in ``ice_panel_patch_contact`` for a uniform
    patch; mesh refinement 8 / 16 / 24.

Usage
-----
    PYTHONPATH=src python tools/verify_patch_from_contact_navier.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import tensorfem.patch_from_contact as P
import tensorfem.ice_panel_patch_contact as IC
from tensorfem.plate_mindlin import solve_plate_on_winkler_foundation

DEFAULT_OUT = Path("benchmarks/artifacts/GAP-PP-1-patch-from-contact-verify")

PLATE_SIDE = 1.0
YOUNG = 2.06e11
POISSON = 0.3
THICKNESS = 0.01
WINKLER_K = 1.0e7

CONFIGS = [
    ("mild_nonuniform", 5.0e6, 0.15, 0.30),
    ("moderate_nonuniform", 8.0e6, 0.10, 0.20),
    ("sharp_nonuniform", 15.0e6, 0.05, 0.10),
]


# ---------------------------------------------------------------------------
# A. unit audit of the end-to-end chain
# ---------------------------------------------------------------------------
def unit_audit(nx: int = 12) -> dict:
    p0 = 1.0e6
    direct = solve_plate_on_winkler_foundation(
        nx, PLATE_SIDE, YOUNG, POISSON, THICKNESS, WINKLER_K, p0,
    )
    w_direct = float(np.max(np.abs(direct.displacement[0::3].numpy())))

    nodes, elements = P.structured_square_mesh(nx, PLATE_SIDE, dtype=torch.float64)
    nodes_np, elements_np = nodes.numpy(), elements.numpy()
    contact = P.ContactPressureData(
        node_positions=nodes_np.copy(),
        normal_pressures=np.full(nodes_np.shape[0], p0),
        contact_area=PLATE_SIDE ** 2,
    )
    fixed, patch_fixed = P.contact_pressure_to_plate_load(
        contact, nx, PLATE_SIDE, YOUNG, POISSON, THICKNESS, WINKLER_K,
    )
    w_fixed = float(np.max(np.abs(fixed.displacement[0::3].numpy())))

    # legacy chain: consistent nodal forces (N) re-used as pressures (Pa)
    legacy = P.patch_from_contact(contact, nodes_np, elements_np)
    legacy_fn = P._build_pressure_callable_from_nodes(
        nodes_np, legacy.load_vector[0::3],
    )
    legacy_sol = solve_plate_on_winkler_foundation(
        nx, PLATE_SIDE, YOUNG, POISSON, THICKNESS, WINKLER_K, legacy_fn,
    )
    w_legacy = float(np.max(np.abs(legacy_sol.displacement[0::3].numpy())))
    element_area = (PLATE_SIDE / nx) ** 2

    return {
        "nx": nx,
        "uniform_pressure_Pa": p0,
        "direct_peak_deflection_m": w_direct,
        "fixed_chain_peak_deflection_m": w_fixed,
        "fixed_chain_rel_error": (w_fixed - w_direct) / w_direct,
        "legacy_chain_peak_deflection_m": w_legacy,
        "legacy_over_direct": w_legacy / w_direct,
        "element_area_m2": element_area,
        "element_area_over_legacy_ratio": element_area / (w_legacy / w_direct),
        "load_balance_rel_fixed": patch_fixed.metadata["load_balance_rel"],
        "diagnosis": (
            "legacy ratio equals one element area (to ~1%): the projected nodal "
            "forces [N] were fed back as pressures [Pa], so the plate solve "
            "re-integrated them and scaled the load by element_area"
        ),
    }


# ---------------------------------------------------------------------------
# B. force conservation of the projection
# ---------------------------------------------------------------------------
def force_conservation(contact_grids=(21, 41, 81), nx: int = 12) -> dict:
    nodes, elements = P.structured_square_mesh(nx, PLATE_SIDE, dtype=torch.float64)
    rows = []
    for n_per_axis in contact_grids:
        for name, peak, sigma, _footprint in CONFIGS:
            pos, pres = P._gaussian_contact_grid(
                0.5 * PLATE_SIDE, 0.5 * PLATE_SIDE, sigma, sigma, peak,
                PLATE_SIDE, n_per_axis,
            )
            data = P.ContactPressureData(
                node_positions=pos, normal_pressures=pres,
                contact_area=2.0 * math.pi * sigma * sigma,
            )
            res = P.patch_from_contact(data, nodes.numpy(), elements.numpy())
            analytic = peak * 2.0 * math.pi * sigma * sigma
            rows.append({
                "config": name,
                "contact_grid": n_per_axis,
                "projected_force_N": res.total_force,
                "analytic_force_N": analytic,
                "rel_error_pct": 100.0 * (res.total_force / analytic - 1.0),
                "projection_method": res.metadata["projection_method"],
            })
    return {
        "plate_nx": nx,
        "rows": rows,
        "max_abs_rel_error_pct": max(abs(r["rel_error_pct"]) for r in rows),
        "legacy_reference_max_abs_rel_error_pct": 666.4,
        "legacy_reference_note": (
            "global inverse-distance projection (pre-m20624) inflated the force "
            "by +24 % (mild) to +666 % (sharp); see docs/VERIFY-patch-from-contact-units.md"
        ),
    }


# ---------------------------------------------------------------------------
# D. Navier-series cross-check
# ---------------------------------------------------------------------------
def _contact_patch_from_grid(xs, ys, Pvals, total_force, peak) -> IC.ContactPatch:
    return IC.ContactPatch(
        x=torch.as_tensor(xs, dtype=torch.float64),
        y=torch.as_tensor(ys, dtype=torch.float64),
        pressure=torch.as_tensor(Pvals, dtype=torch.float64),  # (len(x), len(y))
        total_force=float(total_force),
        contact_area=float(abs(xs[-1] - xs[0]) * abs(ys[-1] - ys[0])),
        peak_pressure=float(peak),
        centroid=(0.5 * (xs[0] + xs[-1]), 0.5 * (ys[0] + ys[-1])),
        approach=0.0,
        opening=torch.zeros((len(xs), len(ys)), dtype=torch.float64),
        residual=0.0,
        config={"kind": "verification"},
    )


def navier_crosscheck(meshes=(8, 16, 24), contact_grid: int = 33) -> dict:
    a = b = PLATE_SIDE
    rows = []

    # ---- uniform patch on a free (k=0) simply-supported square plate ----
    half = 0.2
    force = 4.0e4
    p0 = force / (4.0 * half * half)
    xs = np.linspace(0.5 - half, 0.5 + half, 9)
    ys = np.linspace(0.5 - half, 0.5 + half, 9)
    XX, YY = np.meshgrid(xs, ys)
    uni_data = P.ContactPressureData(
        node_positions=np.column_stack([XX.ravel(), YY.ravel()]),
        normal_pressures=np.full(XX.size, p0),
        contact_area=4.0 * half * half,
    )
    uni_patch = IC.uniform_patch_pressure(
        total_force=force, centre=(0.5, 0.5), half_x=half, half_y=half,
        panel=(a, b), grid=contact_grid,
    )

    for nx in meshes:
        mine, patch_res = P.contact_pressure_to_plate_load(
            uni_data, nx, a, YOUNG, POISSON, THICKNESS, 0.0,
        )
        st = P.peak_stresses(mine)
        nodes, elements, f, info = IC.patch_from_contact(uni_patch, nx, nx, a, b)
        sol_ic = IC.solve_rect_panel(
            nx, nx, a, b, young=YOUNG, poisson=POISSON, thickness=THICKNESS, load=f,
        )
        st_ic = sol_ic.peak_stress()
        nav = IC.navier_rect_moments(
            uni_patch, a, b, young=YOUNG, poisson=POISSON, thickness=THICKNESS,
            x=st["sigma_x_at_m"][0], y=st["sigma_x_at_m"][1],
        )
        nav_mx = 6.0 * abs(nav["moment_x"]) / THICKNESS ** 2 / 1e6
        rows.append({
            "patch": "uniform_0.4x0.4",
            "plate_nx": nx,
            "mine_sigma_x_MPa": st["sigma_x_MPa"],
            "ic_fem_sigma_Mx_MPa": st_ic["sigma_Mx_MPa"],
            "navier_sigma_Mx_MPa": nav_mx,
            "mine_vs_navier_pct": 100.0 * (st["sigma_x_MPa"] / nav_mx - 1.0),
            "ic_fem_vs_navier_pct": 100.0 * (st_ic["sigma_Mx_MPa"] / nav_mx - 1.0),
            "force_rel_error": patch_res.metadata["load_balance_rel"],
        })

    # ---- non-uniform Gaussian patch, same plate ----
    peak, sigma_p = 2.0e6, 0.08
    pos, pres = P._gaussian_contact_grid(0.5, 0.5, sigma_p, sigma_p, peak, PLATE_SIDE, contact_grid)
    xg = np.unique(np.round(pos[:, 0], 12))
    yg = np.unique(np.round(pos[:, 1], 12))
    grid_p = pres.reshape(yg.size, xg.size).T  # (len(x), len(y))
    analytic_force = peak * 2.0 * math.pi * sigma_p ** 2
    gauss_patch = _contact_patch_from_grid(xg, yg, grid_p, analytic_force, peak)
    gauss_data = P.ContactPressureData(
        node_positions=pos, normal_pressures=pres,
        contact_area=2.0 * math.pi * sigma_p ** 2,
    )

    for nx in meshes:
        mine, patch_res = P.contact_pressure_to_plate_load(
            gauss_data, nx, a, YOUNG, POISSON, THICKNESS, 0.0,
        )
        st = P.peak_stresses(mine)
        nodes, elements, f, info = IC.patch_from_contact(gauss_patch, nx, nx, a, b)
        sol_ic = IC.solve_rect_panel(
            nx, nx, a, b, young=YOUNG, poisson=POISSON, thickness=THICKNESS, load=f,
        )
        st_ic = sol_ic.peak_stress()
        nav = IC.navier_rect_moments(
            gauss_patch, a, b, young=YOUNG, poisson=POISSON, thickness=THICKNESS,
            x=st["sigma_x_at_m"][0], y=st["sigma_x_at_m"][1],
        )
        nav_mx = 6.0 * abs(nav["moment_x"]) / THICKNESS ** 2 / 1e6
        rows.append({
            "patch": "gaussian_peak2MPa_sigma0.08",
            "plate_nx": nx,
            "mine_sigma_x_MPa": st["sigma_x_MPa"],
            "ic_fem_sigma_Mx_MPa": st_ic["sigma_Mx_MPa"],
            "navier_sigma_Mx_MPa": nav_mx,
            "mine_vs_navier_pct": 100.0 * (st["sigma_x_MPa"] / nav_mx - 1.0),
            "ic_fem_vs_navier_pct": 100.0 * (st_ic["sigma_Mx_MPa"] / nav_mx - 1.0),
            "force_rel_error": patch_res.metadata["load_balance_rel"],
        })

    finest = [r for r in rows if r["plate_nx"] == max(meshes)]
    return {
        "plate": {"side_m": a, "thickness_m": THICKNESS, "E_Pa": YOUNG, "nu": POISSON},
        "contact_grid": contact_grid,
        "rows": rows,
        "max_abs_mine_vs_navier_pct_finest": max(abs(r["mine_vs_navier_pct"]) for r in finest),
        "max_abs_ic_fem_vs_navier_pct_finest": max(abs(r["ic_fem_vs_navier_pct"]) for r in finest),
        "h3_anchor_pct": 0.339,
        "h3_anchor_meaning": (
            "FEM-vs-Kelvin error of the H3-ICE-PLATE-FEM anchor "
            "(benchmarks/artifacts/ice_plate_flexural_limit/evidence.json); the "
            "cross-check above is the equivalent quantity for this chain"
        ),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figures(out: Path, unit: dict, scan: dict, cross: dict) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_out = []

    # fig 1: contact pressure field vs projected plate load
    pos, pres = P._gaussian_contact_grid(0.5, 0.5, 0.10, 0.10, 8.0e6, PLATE_SIDE, 41)
    data = P.ContactPressureData(pos, pres, 2.0 * math.pi * 0.10 ** 2)
    nodes, elements = P.structured_square_mesh(16, PLATE_SIDE, dtype=torch.float64)
    res = P.patch_from_contact(data, nodes.numpy(), elements.numpy())
    nn = nodes.numpy()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    sc = axes[0].scatter(pos[:, 0], pos[:, 1], c=pres / 1e6, s=8, cmap="viridis")
    axes[0].set_title("contact pressure field [MPa]")
    fig.colorbar(sc, ax=axes[0])
    sc2 = axes[1].scatter(nn[:, 0], nn[:, 1], c=res.nodal_pressures / 1e6, s=18, cmap="magma")
    axes[1].set_title("projected nodal pressure [MPa]")
    fig.colorbar(sc2, ax=axes[1])
    for ax in axes:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
    fig.tight_layout()
    path = out / "fig_projection.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figures_out.append({"name": "fig_projection.png",
                        "caption": "Gaussian contact pressure (left) projected onto the plate grid (right)"})

    # fig 2: mesh convergence vs Navier
    fig, ax = plt.subplots(figsize=(6, 4))
    for patch_name, marker in (("uniform_0.4x0.4", "o"), ("gaussian_peak2MPa_sigma0.08", "s")):
        rows = [r for r in cross["rows"] if r["patch"] == patch_name]
        ax.plot([r["plate_nx"] for r in rows], [r["mine_vs_navier_pct"] for r in rows],
                marker=marker, label=f"{patch_name} (this module)")
        nav = rows[0]["navier_sigma_Mx_MPa"]
        ax.axhline(0.0, color="k", lw=0.8)
        ax.annotate(f"Navier = {nav:.1f} MPa", xy=(0.02, 0.02), xycoords="axes fraction",
                    fontsize=8)
    ax.axhline(cross["h3_anchor_pct"], color="r", ls="--", lw=0.9, label="H3 anchor 0.339 %")
    ax.axhline(-cross["h3_anchor_pct"], color="r", ls="--", lw=0.9)
    ax.set_xlabel("plate mesh nx (=ny)")
    ax.set_ylabel("peak bending stress error vs Navier [%]")
    ax.set_title("FEM peak stress vs Kirchhoff Navier series")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = out / "fig_convergence.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figures_out.append({"name": "fig_convergence.png",
                        "caption": "Peak bending stress error vs the Navier series (solid) and the H3 anchor level (dashed)"})

    # fig 3: non-uniform vs uniform peak stress
    cfgs = scan["configurations"]
    labels = [c["name"] for c in cfgs]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].bar(x - 0.2, [c["peak_stress_projected"]["sigma_x_MPa"] for c in cfgs], 0.4,
                label="non-uniform (from contact)")
    axes[0].bar(x + 0.2, [c["peak_stress_uniform"]["sigma_x_MPa"] for c in cfgs], 0.4,
                label="uniform, matched force")
    axes[0].set_xticks(x, labels, rotation=15, fontsize=8)
    axes[0].set_ylabel("peak bending stress [MPa]")
    axes[0].legend(fontsize=7)
    axes[0].set_title("peak stress, non-uniform vs uniform")
    axes[1].bar(x, [c["delta_nonuniform_over_uniform_pct"]["sigma_x"] for c in cfgs], 0.5)
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_xticks(x, labels, rotation=15, fontsize=8)
    axes[1].set_ylabel("non-uniform - uniform [%]")
    axes[1].set_title("non-uniformity effect on peak stress")
    fig.tight_layout()
    path = out / "fig_nonuniform_vs_uniform.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figures_out.append({"name": "fig_nonuniform_vs_uniform.png",
                        "caption": "Peak bending stress for the contact-projected patch vs the matched uniform patch, per contact configuration"})
    return figures_out


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    unit = unit_audit()
    force = force_conservation()
    scan = P.scan_contact_configurations(
        plate_nx=12, plate_side=PLATE_SIDE, young=YOUNG, poisson=POISSON,
        thickness=THICKNESS, winkler_k=WINKLER_K,
    )
    cross = navier_crosscheck()
    figs = figures(out, unit, scan, cross)

    result = {
        "case": "GAP-PP-1-ICE-STRUCT-CONTACT-PATCH verification",
        "mission": 20624,
        "unit_audit": unit,
        "force_conservation": force,
        "nonuniform_scan": scan,
        "navier_crosscheck": cross,
        "figures": figs,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out / "verification.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({
        "unit": unit,
        "force_max_abs_pct": force["max_abs_rel_error_pct"],
        "scan_summary": scan["summary"],
        "cross_finest": cross["max_abs_mine_vs_navier_pct_finest"],
        "wall_s": result["wall_seconds"],
    }, indent=1))
    return result


if __name__ == "__main__":
    main()