#!/usr/bin/env python3
"""GAP-PP-3-ICE-STRUCT-PLASTIC-SET — HPC verification of limit load and mesh convergence.

Runs the heavy FE computations that time out on the control node:
1. Plastic limit load verification (80-step arc-length trace)
2. Three-grid mesh convergence study
3. Full load-deflection path with permanent set extraction

Outputs: JSON results + PNG figures for the verification report.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfem.plate_patch_plastic_limit import (
    analytic_first_yield_pressure,
    ilyushin_sine_upper_bound,
    low_temperature_flow_stress,
    patch_plate_mesh,
    plastic_moment,
    run_patch_plastic_path,
    strip_lower_bound,
    yield_line_patch_upper_bound,
)

# ---------------------------------------------------------------------------
# Anchor geometry / material
# ---------------------------------------------------------------------------
A = B = 1.0
T = 0.02
E = 210.0e9
NU = 0.3
SY = 355.0e6          # DH36 @ -20 C nominal
PATCH = (0.5, 0.5)
HARDENING = 1.0e6     # near-perfectly plastic

def main():
    t0 = time.time()
    results = {
        "anchor": "GAP-PP-3-ICE-STRUCT-PLASTIC-SET",
        "geometry": {"a": A, "b": B, "t": T, "patch": PATCH},
        "material": {"E": E, "nu": NU, "sigma_y": SY, "hardening": HARDENING, "grade": "DH36"},
        "analytic_bounds": {},
        "limit_load": {},
        "mesh_convergence": {},
        "permanent_set": {},
        "timing": {},
    }

    # 1. Analytic bounds
    py = analytic_first_yield_pressure(A, B, T, E, NU, SY, PATCH)
    ub = yield_line_patch_upper_bound(A, B, T, SY, PATCH)
    sine = ilyushin_sine_upper_bound(A, B, T, SY, PATCH)
    lb = strip_lower_bound(A, B, T, SY, PATCH)
    results["analytic_bounds"] = {
        "first_yield_pressure_Pa": py["pressure"],
        "yield_line_upper_Pa": ub["pressure"],
        "ilyushin_sine_upper_Pa": sine["pressure"],
        "strip_lower_Pa": lb["pressure"],
    }
    print(f"Analytic bounds: py={py['pressure']/1e6:.3f} MPa, "
          f"lb={lb['pressure']/1e6:.3f} MPa, sine={sine['pressure']/1e6:.3f} MPa, "
          f"yl={ub['pressure']/1e6:.3f} MPa")

    # 2. Limit load verification at three mesh levels
    mesh_levels = [6, 8, 10]
    limit_loads = []
    first_yields = []
    all_paths = {}

    for nx in mesh_levels:
        t_mesh = time.time()
        py_val = py["pressure"]
        # Load ramp: from 0.5*py to 2.0*py in steps
        load_factors = [f * py_val for f in np.linspace(0.5, 2.5, 25)]
        path = run_patch_plastic_path(
            A, B, T, young=E, poisson=NU, yield_stress=SY, hardening=HARDENING,
            patch=PATCH, load_factors=load_factors,
            nx=nx, ny=nx, layers=3,
            inplane_restrained=True, geometric=True,
            unload_steps=3, unload_to=0.0,
        )
        elapsed = time.time() - t_mesh
        results["timing"][f"mesh_{nx}x{nx}"] = elapsed

        fy = path.first_yield_factor
        ll = path.measured_limit_load
        perm = path.permanent_deflection

        first_yields.append(fy)
        limit_loads.append(ll)
        all_paths[nx] = path

        print(f"  {nx}x{nx}: first_yield={fy/1e6:.3f} MPa, "
              f"limit_load={ll/1e6 if ll else 'N/A'} MPa, "
              f"permanent_set={perm*1e3 if perm else 'N/A'} mm, "
              f"time={elapsed:.1f}s")

    # 3. Compute errors
    py_val = py["pressure"]
    fy_errors = [(fy - py_val) / py_val for fy in first_yields if fy is not None]
    ll_errors = []
    for ll in limit_loads:
        if ll is not None:
            # Compare with yield-line upper bound
            err = (ll - ub["pressure"]) / ub["pressure"]
            ll_errors.append(err)

    results["limit_load"] = {
        "first_yield_Pa": first_yields,
        "first_yield_error_pct": [e * 100 for e in fy_errors],
        "limit_load_Pa": limit_loads,
        "limit_load_vs_yield_line_error_pct": [e * 100 for e in ll_errors],
        "mesh_levels": mesh_levels,
    }

    # 4. Mesh convergence (using centre deflection at 0.8*py)
    target_p = 0.8 * py_val
    conv_deflections = []
    for nx in mesh_levels:
        path = run_patch_plastic_path(
            A, B, T, young=E, poisson=NU, yield_stress=SY, hardening=HARDENING,
            patch=PATCH, load_factors=[target_p],
            nx=nx, ny=nx, layers=3,
            inplane_restrained=True, geometric=False,
        )
        w = path.loading[0].centre_deflection if path.loading else 0.0
        conv_deflections.append(w)

    # Relative errors using finest mesh as reference
    ref_w = conv_deflections[-1]
    rel_errors = [abs(w - ref_w) / abs(ref_w) if ref_w != 0 else 0.0 for w in conv_deflections]

    results["mesh_convergence"] = {
        "mesh_levels": mesh_levels,
        "centre_deflection_m": conv_deflections,
        "relative_error_pct": [e * 100 for e in rel_errors],
        "reference_deflection_m": ref_w,
    }

    # 5. Permanent set
    # Use the finest mesh path that had unloading
    finest_path = all_paths.get(max(mesh_levels))
    if finest_path and finest_path.permanent_deflection is not None:
        results["permanent_set"] = {
            "permanent_deflection_m": finest_path.permanent_deflection,
            "permanent_deflection_mm": finest_path.permanent_deflection * 1e3,
            "peak_deflection_m": finest_path.loading[-1].centre_deflection if finest_path.loading else None,
            "recovery_ratio": (finest_path.loading[-1].centre_deflection - finest_path.permanent_deflection)
                              / finest_path.loading[-1].centre_deflection
                              if finest_path.loading and finest_path.loading[-1].centre_deflection != 0 else None,
            "valid": finest_path.permanent_set_valid,
        }

    results["timing"]["total"] = time.time() - t0

    # -----------------------------------------------------------------------
    # Save JSON results
    # -----------------------------------------------------------------------
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "gap_pp3_hpc_results.json")

    # Convert torch tensors to lists for JSON serialization
    def jsonify(obj):
        if isinstance(obj, dict):
            return {k: jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [jsonify(v) for v in obj]
        if hasattr(obj, "item"):
            return obj.item()
        return obj

    with open(json_path, "w") as f:
        json.dump(jsonify(results), f, indent=2)
    print(f"\nResults saved to {json_path}")

    # -----------------------------------------------------------------------
    # Generate figures
    # -----------------------------------------------------------------------
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Figure 1: Load-deflection curves for all mesh levels
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for idx, nx in enumerate(mesh_levels):
        path = all_paths[nx]
        if path.loading:
            loads = [p.load_factor / 1e6 for p in path.loading]
            defs = [p.centre_deflection * 1e3 for p in path.loading]
            ax.plot(defs, loads, "o-", color=colors[idx], label=f"{nx}x{nx}", markersize=4)
            if path.unloading:
                u_loads = [p.load_factor / 1e6 for p in path.unloading]
                u_defs = [p.centre_deflection * 1e3 for p in path.unloading]
                ax.plot(u_defs, u_loads, "--", color=colors[idx], alpha=0.5)

    # Analytic bounds as horizontal lines
    ax.axhline(y=py_val / 1e6, color="red", linestyle=":", alpha=0.7, label=f"First yield ({py_val/1e6:.2f} MPa)")
    ax.axhline(y=ub["pressure"] / 1e6, color="green", linestyle=":", alpha=0.7, label=f"Yield-line UB ({ub['pressure']/1e6:.2f} MPa)")
    ax.axhline(y=lb["pressure"] / 1e6, color="orange", linestyle=":", alpha=0.7, label=f"Strip LB ({lb['pressure']/1e6:.2f} MPa)")

    ax.set_xlabel("Centre deflection w [mm]")
    ax.set_ylabel("Patch pressure p [MPa]")
    ax.set_title("GAP-PP-3: Load-deflection curves (DH36, t=20mm, 1m×1m plate, 50%×50% patch)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig1_path = os.path.join(fig_dir, "gap_pp3_load_deflection.png")
    fig.savefig(fig1_path, dpi=150)
    plt.close(fig)
    print(f"Figure 1 saved: {fig1_path}")

    # Figure 2: Mesh convergence plot
    fig, ax = plt.subplots(figsize=(6, 4))
    n_elements = [nx * nx for nx in mesh_levels]
    ax.semilogx(n_elements, [d * 1e3 for d in conv_deflections], "o-", markersize=8)
    ax.set_xlabel("Number of elements")
    ax.set_ylabel("Centre deflection at 0.8·p_y [mm]")
    ax.set_title("Mesh convergence: centre deflection")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig2_path = os.path.join(fig_dir, "gap_pp3_mesh_convergence.png")
    fig.savefig(fig2_path, dpi=150)
    plt.close(fig)
    print(f"Figure 2 saved: {fig2_path}")

    # Figure 3: First yield error vs mesh
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"{nx}x{nx}" for nx in mesh_levels], [e * 100 for e in fy_errors], color="#1f77b4")
    ax.axhline(y=5.0, color="red", linestyle="--", alpha=0.7, label="5% gate")
    ax.set_xlabel("Mesh resolution")
    ax.set_ylabel("First yield error [%]")
    ax.set_title("Elastic-plastic transition error vs analytic")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig3_path = os.path.join(fig_dir, "gap_pp3_first_yield_error.png")
    fig.savefig(fig3_path, dpi=150)
    plt.close(fig)
    print(f"Figure 3 saved: {fig3_path}")

    # Figure 4: Bounds bracket
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Strip LB", "First Yield", "Ilyushin Sine", "Yield-line UB"]
    values = [lb["pressure"], py_val, sine["pressure"], ub["pressure"]]
    colors_bar = ["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728"]
    ax.barh(labels, [v / 1e6 for v in values], color=colors_bar)
    ax.set_xlabel("Pressure [MPa]")
    ax.set_title("Analytic bounds bracket (DH36, t=20mm)")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig4_path = os.path.join(fig_dir, "gap_pp3_bounds_bracket.png")
    fig.savefig(fig4_path, dpi=150)
    plt.close(fig)
    print(f"Figure 4 saved: {fig4_path}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print("HPC verification complete.")


if __name__ == "__main__":
    main()
