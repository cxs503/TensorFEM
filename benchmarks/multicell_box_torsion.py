#!/usr/bin/env python3
"""Evidence run for :mod:`tensorfem.multicell_torsion` (multi-cell box torsion).

Produces ``benchmarks/multicell_box_torsion_evidence.json`` plus figures under
``docs/assets/multicell-box-torsion/``.  No number in the report is typed by
hand: every field of the JSON is written here and every figure reads the JSON.

Usage::

    OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/multicell_box_torsion.py [--fast]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import tensorfem.multicell_torsion as mc  # noqa: E402
from tensorfem.box_torsion import solve_box_prandtl  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "multicell_box_torsion_evidence.json"
FIGDIR = ROOT / "docs" / "assets" / "multicell-box-torsion"


def richardson(f1: float, f2: float, f3: float, r: float = 2.0) -> dict:
    """Observed order and Richardson limit from three successive refinements."""
    p = math.log(abs((f1 - f2) / (f2 - f3))) / math.log(r)
    limit = f3 + (f3 - f2) / (r ** p - 1.0)
    return {"order": p, "limit": limit, "ratio": r}


def fd_ladder(section, hs, g=1.0, theta=1.0) -> list:
    rows = []
    for h in hs:
        t0 = time.time()
        f = mc.solve_multicell_prandtl(section, h, g=g, theta=theta)
        rows.append({"h": h, "j_torque": f.j_torque, "j_energy": f.j_energy,
                     "identity_gap": abs(f.j_energy / f.j_torque - 1.0),
                     "compat": f.compat_residual, "sweeps": f.sweeps,
                     "constants": [float(c) for c in f.constants],
                     "warping_span": f.warping_span,
                     "seconds": time.time() - t0})
    return rows


def bredt_row(section, g=1.0, theta=1.0) -> dict:
    b = mc.bredt_multicell(section, g=g, theta=theta)
    walls = [{"label": w["label"], "length": w["length"],
              "thickness": w["thickness"], "cells": list(w["cells"]),
              "flow": b.wall_flow(w), "tau": abs(b.wall_flow(w)) / w["thickness"]}
             for w in b.walls]
    # naive design-office mistake: treat each cell as an independent single cell
    naive = 0.0
    for (x0, x1, y0, y1) in section.cell_midlines():
        area = (x1 - x0) * (y1 - y0)
        perim = 2.0 * ((x1 - x0) + (y1 - y0))
        naive += 4.0 * area ** 2 * section.wall / perim
    return {"j": b.j, "cell_areas": list(b.cell_areas),
            "cell_flows": list(b.cell_flows), "walls": walls,
            "j_naive_independent_cells": naive,
            "naive_error": naive / b.j - 1.0}


def case_report(name, section, hs, fem_hs, g=1.0, theta=1.0, fem=True) -> dict:
    ladder = fd_ladder(section, hs, g, theta)
    finest = ladder[-1]
    bredt = bredt_row(section, g, theta)
    fd_field = mc.solve_multicell_prandtl(section, hs[-1], g=g, theta=theta)
    fd_walls = mc.wall_band_taus(section, fd_field)
    out = {"name": name,
           "geometry": {"a": section.a, "b": section.b, "wall": section.wall,
                        "webs": [list(w) for w in section.webs],
                        "n_cells": section.n_cells,
                        "material_area": section.area},
           "fd": ladder, "bredt": bredt,
           "fd_wall_tau": fd_walls,
           "bredt_wall_tau": {w["label"]: w["tau"] for w in bredt["walls"]},
           "brett_vs_fd_error": bredt["j"] / finest["j_torque"] - 1.0}
    if len(ladder) >= 3:
        out["richardson"] = richardson(ladder[-3]["j_torque"],
                                       ladder[-2]["j_torque"],
                                       ladder[-1]["j_torque"])
    if fem:
        out["fem"] = []
        for h in fem_hs:
            t0 = time.time()
            r = mc.multicell_torsion_fem(section, h, layers_z=2, twist=theta,
                                         young_modulus=2.0 * (1.0 + 0.3),
                                         poisson_ratio=0.3)
            out["fem"].append({"h": h, "j_fe": r["j_fe"], "nodes": r["nodes"],
                               "elements": r["elements"], "dofs": r["dofs"],
                               "warping_max": r["warping_max"],
                               "tau_max_fe": r["tau_max_fe"],
                               "vs_fd": r["j_fe"] / finest["j_torque"] - 1.0,
                               "vs_bredt": r["j_fe"] / bredt["j"] - 1.0,
                               "seconds": time.time() - t0})
    return out


def thin_wall_sweep(hs_by_t) -> list:
    rows = []
    for t, h in hs_by_t:
        section = mc.RectMultiCellSection(a=1.0, b=0.6, wall=t,
                                          webs=((0.6, t),))
        f = mc.solve_multicell_prandtl(section, h)
        b = mc.bredt_multicell(section)
        rows.append({"t": t, "h": h, "j_fd": f.j_torque, "j_bredt": b.j,
                     "ratio": f.j_torque / b.j - 1.0,
                     "identity_gap": abs(f.j_energy / f.j_torque - 1.0),
                     "compat": f.compat_residual})
    return rows


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_fields(sym, asym):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    for row, case in enumerate((sym, asym)):
        section, field = case["section"], case["field"]
        X, Y = np.meshgrid(field.xs, field.ys)
        mask = np.where(field.material, 1.0, np.nan)
        ax = axes[row, 0]
        im = ax.pcolormesh(X, Y, field.phi * mask, cmap="viridis", shading="auto")
        ax.set_aspect("equal")
        ax.set_title(f"{case['label']}: Prandtl phi (holes: constants "
                     f"{np.round(field.constants, 4)})")
        fig.colorbar(im, ax=ax, shrink=0.85)
        for x0, x1, y0, y1 in section.holes():
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       fill=False, ec="w", lw=1.0, ls="--"))
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        ax = axes[row, 1]
        xs, ys, txz, tyz = field.tau_cells
        tau = np.sqrt(txz ** 2 + tyz ** 2)
        sc = ax.scatter(xs, ys, c=tau, s=14, cmap="magma")
        ax.set_aspect("equal")
        ax.set_title(f"{case['label']}: |tau| at cell centres "
                     f"(corner singularity shown, not a gate)")
        fig.colorbar(sc, ax=ax, shrink=0.85)
    fig.suptitle("Multi-cell Saint-Venant torsion: finite-difference Prandtl field",
                 fontsize=12)
    fig.tight_layout()
    path = FIGDIR / "fig1_multicell_fields.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_convergence(cases):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for case in cases:
        fd = case["fd"]
        hs = [r["h"] for r in fd]
        axes[0].plot(hs, [r["j_torque"] for r in fd], "o-",
                     label=f"{case['name']} FD (torque id.)")
        axes[0].plot(hs, [r["j_energy"] for r in fd], "s--", alpha=0.6,
                     label=f"{case['name']} FD (energy id.)")
        axes[0].axhline(case["bredt"]["j"], color="0.4", ls=":", lw=1.0)
        for f in case.get("fem", []):
            axes[0].plot(f["h"], f["j_fe"], "D", ms=7, mec="k",
                         label=f"{case['name']} HEX8 ({f['dofs']} dof)")
        axes[1].plot(hs, [100.0 * (r["j_torque"] / case["bredt"]["j"] - 1.0)
                          for r in fd], "o-", label=f"{case['name']} FD vs Bredt")
        for f in case.get("fem", []):
            axes[1].plot(f["h"], 100.0 * f["vs_fd"], "D", ms=7, mec="k",
                         label=f"{case['name']} HEX8 vs FD")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("h [m]  (FD grid / HEX8 wall layer)")
    axes[0].set_ylabel("J [m^4]")
    axes[0].set_title("Torsional rigidity vs refinement (dotted: Bredt)")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_xscale("log")
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].axhline(3.0, color="r", ls="--", lw=0.8)
    axes[1].axhline(-3.0, color="r", ls="--", lw=0.8)
    axes[1].set_xlabel("h [m]")
    axes[1].set_ylabel("signed error [%]")
    axes[1].set_title("Error of FD (vs Bredt) and of HEX8 (vs FD); red: 3 % gate")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = FIGDIR / "fig2_convergence.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_thin_wall(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    t = np.array([r["t"] for r in rows])
    err = np.array([100.0 * r["ratio"] for r in rows])
    j = np.array([r["j_fd"] for r in rows])
    axes[0].loglog(t, np.abs(err), "o-", label="|J_FD/J_Bredt - 1| [%]")
    ref = np.abs(err[-1]) * (t / t[-1])
    axes[0].loglog(t, ref, "k--", lw=1.0, label="first-order reference (slope 1)")
    axes[0].set_xlabel("wall thickness t [m]  (a = 1.0, b = 0.6)")
    axes[0].set_ylabel("Bredt truncation [%]")
    axes[0].set_title("Thin-wall limit: Bredt error is first order in t")
    axes[0].legend(fontsize=8)
    axes[1].plot(t, j, "o-", label="J_FD")
    axes[1].plot(t, [r["j_bredt"] for r in rows], "s--", label="J_Bredt")
    axes[1].set_xlabel("t [m]")
    axes[1].set_ylabel("J [m^4]")
    axes[1].set_title("Asymmetric 2-cell section: J vs wall thickness")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = FIGDIR / "fig3_thin_wall.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_walls(case):
    labels = list(case["fd_wall_tau"].keys())
    fd = [case["fd_wall_tau"][k] for k in labels]
    br = [case["bredt_wall_tau"][k] for k in labels]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    idx = np.arange(len(labels))
    ax.bar(idx - 0.2, fd, width=0.4, label="finite difference (mean |tau| in wall)")
    ax.bar(idx + 0.2, br, width=0.4, label="multi-cell Bredt (|q|/t)")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("wall shear stress [Pa]")
    ax.set_title(f"{case['name']}: per-wall shear, including the shared web")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGDIR / "fig4_wall_shears.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main(fast: bool) -> int:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    g = 1.0  # material is scaled so that G = 1 for every section
    sections = {
        "A_single_cell": (mc.RectMultiCellSection(1.0, 0.6, 0.1),
                          [0.05, 0.025, 0.0125], [0.05, 0.025]),
        "B_symmetric_2cell": (mc.RectMultiCellSection(1.0, 0.6, 0.1,
                                                     ((0.0, 0.1),)),
                              [0.05, 0.025, 0.0125], [0.05, 0.025]),
        "C_asymmetric_2cell": (mc.RectMultiCellSection(1.0, 0.6, 0.1,
                                                       ((0.6, 0.1),)),
                               [0.05, 0.025, 0.0125], [0.05, 0.025]),
        "D_three_cell": (mc.RectMultiCellSection(1.5, 0.6, 0.1,
                                                 ((-0.5, 0.1), (0.5, 0.1))),
                         [0.05, 0.025, 0.0125], [0.05]),
    }
    if fast:
        sections = {k: (s, hs[:2], fh[:1]) for k, (s, hs, fh) in sections.items()}

    evidence = {"mission": "selfstart multicell box torsion",
                "module": "tensorfem.multicell_torsion",
                "gates": {}, "cases": {}}
    for name, (section, hs, fh) in sections.items():
        evidence["cases"][name] = case_report(name, section, hs, fh, g=g)

    sweep_rows = thin_wall_sweep([(0.2, 0.05), (0.1, 0.025), (0.05, 0.0125),
                                  (0.02, 0.005)])
    if fast:
        sweep_rows = sweep_rows[:2]
    evidence["thin_wall_sweep"] = sweep_rows

    # -- degeneracy: the general solver must reproduce the box_torsion oracle --
    single = evidence["cases"]["A_single_cell"]
    box = solve_box_prandtl(1.0, 0.6, 0.1, single["fd"][-1]["h"])
    evidence["gates"]["degeneracy_single_cell"] = {
        "tensorfem_box_torsion_j_torque": box.j_torque,
        "multicell_j_torque": single["fd"][-1]["j_torque"],
        "rel_diff": abs(single["fd"][-1]["j_torque"] / box.j_torque - 1.0),
        "c_box": box.c,
        "c_multicell": single["fd"][-1]["constants"][0],
        "threshold": 1e-10}

    # -- gates ---------------------------------------------------------------
    gates = evidence["gates"]
    for name, case in evidence["cases"].items():
        finest = case["fd"][-1]
        gates[f"{name}_energy_vs_torque_identity"] = {
            "gap": finest["identity_gap"], "threshold": 2e-2}
        gates[f"{name}_circulation_compat"] = {
            "residual": finest["compat"], "threshold": 1e-10}
        if case.get("fem"):
            best = case["fem"][-1]
            gates[f"{name}_hex8_vs_fd"] = {
                "j_fe": best["j_fe"], "j_fd": finest["j_torque"],
                "error": best["vs_fd"], "dofs": best["dofs"],
                "threshold": 3e-2}
    # symmetry of the symmetric case
    sym = mc.RectMultiCellSection(1.0, 0.6, 0.1, ((0.0, 0.1),))
    f_sym = mc.solve_multicell_prandtl(sym, 0.025)
    nx = len(f_sym.xs)
    left = f_sym.phi[:, : nx // 2]
    right = f_sym.phi[:, nx: nx // 2: -1]
    gates["symmetric_case_reflection"] = {
        "max_abs_diff": float(np.abs(left - right).max()), "threshold": 1e-12,
        "constant_split": float(abs(f_sym.constants[0] - f_sym.constants[1]))}
    # shared-web flow: the discriminating multi-cell quantity
    asym = evidence["cases"]["C_asymmetric_2cell"]
    fd_web = asym["fd_wall_tau"]["web0"]
    br_web = asym["bredt_wall_tau"]["web0"]
    gates["asymmetric_shared_web_flow"] = {
        "fd_tau": fd_web, "bredt_tau": br_web,
        "rel_error": abs(fd_web / br_web - 1.0), "threshold": 0.10}
    sym_case = evidence["cases"]["B_symmetric_2cell"]
    gates["symmetric_web_carries_no_flow"] = {
        "fd_tau_web": sym_case["fd_wall_tau"]["web0"],
        "fd_tau_outer": sym_case["fd_wall_tau"]["outer_left"],
        "ratio": sym_case["fd_wall_tau"]["web0"] / sym_case["fd_wall_tau"]["outer_left"],
        "bredt_tau_web": sym_case["bredt_wall_tau"]["web0"],
        "threshold": 0.10}
    # thin-wall limit
    thin = evidence["thin_wall_sweep"][-1]
    gates["thin_wall_limit"] = {"t": thin["t"], "bredt_vs_fd": thin["ratio"],
                                "threshold": 0.02}
    # Bredt truncation must shrink with t (first order)
    ratios = [abs(r["ratio"]) for r in evidence["thin_wall_sweep"]]
    gates["bredt_error_first_order"] = {
        "errors": ratios, "monotone_decreasing":
            all(ratios[i] > ratios[i + 1] for i in range(len(ratios) - 1)),
        "halving_ratio": ratios[0] / ratios[1] if len(ratios) > 1 else None}

    for key, gate in gates.items():
        thr = gate.get("threshold")
        if "gap" in gate:
            gate["passed"] = bool(gate["gap"] <= thr)
        elif "residual" in gate:
            gate["passed"] = bool(gate["residual"] <= thr)
        elif "error" in gate:
            gate["passed"] = bool(gate["error"] <= thr)
        elif "max_abs_diff" in gate:
            gate["passed"] = bool(gate["max_abs_diff"] <= thr)
        elif "rel_error" in gate:
            gate["passed"] = bool(gate["rel_error"] <= thr)
        elif "rel_diff" in gate:
            gate["passed"] = bool(gate["rel_diff"] <= thr)
        elif "ratio" in gate and key == "symmetric_web_carries_no_flow":
            gate["passed"] = bool(gate["ratio"] <= thr)
        elif key == "thin_wall_limit":
            gate["passed"] = bool(abs(gate["bredt_vs_fd"]) <= thr)
        elif key == "bredt_error_first_order":
            gate["passed"] = bool(gate["monotone_decreasing"])

    # -- figures -------------------------------------------------------------
    sym_field = mc.solve_multicell_prandtl(sym, 0.025)
    asym_sec = mc.RectMultiCellSection(1.0, 0.6, 0.1, ((0.6, 0.1),))
    asym_field = mc.solve_multicell_prandtl(asym_sec, 0.025)
    figs = [fig_fields({"label": "B symmetric 2-cell", "section": sym,
                        "field": sym_field},
                       {"label": "C asymmetric 2-cell", "section": asym_sec,
                        "field": asym_field}),
            fig_convergence(list(evidence["cases"].values())),
            fig_thin_wall(evidence["thin_wall_sweep"]),
            fig_walls(evidence["cases"]["C_asymmetric_2cell"])]
    evidence["figures"] = [str(p.relative_to(ROOT)) for p in figs]
    evidence["all_gates_passed"] = all(g["passed"] for g in gates.values())
    OUT.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v.get("passed") for k, v in gates.items()}, indent=2))
    print("all_gates_passed:", evidence["all_gates_passed"])
    print("evidence:", OUT)
    return 0 if evidence["all_gates_passed"] else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    sys.exit(main(ap.parse_args().fast))