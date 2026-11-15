"""Final deliverable for GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE.

Produces:
1. Updated verification report (docs/)
2. Machine-readable artifacts (docs/grillage_ice_ultimate/)
3. Publication-quality figures (docs/grillage_ice_ultimate/)

Run: PYTHONPATH=src python benchmarks/grillage_ultimate_final.py
"""
from __future__ import annotations

import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem.grillage_ice_ultimate import (
    CombinedLoadCase,
    GrillageGeometry,
    MemberSection,
    case_study_geometry,
    grillage_utilisations,
    panel_stress_ultimate_map,
    plate_patch_response,
    stiffener_axial_capacity,
    ultimate_ice_pressure,
    write_ultimate_map,
)
from tensorfem.grillage_ice_verification import (
    case_study_geometry_slender,
    combined_case_table,
    member_governed_case_table,
    member_governed_geometry,
    plate_field_fd_check,
    run_grillage_verification,
)
from tensorfem.stiffened_panel_ultimate import StiffenedPanel

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "grillage_ice_ultimate"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    geom = case_study_geometry()
    print(f"Panel: {geom.n_spans * geom.frame_spacing:.2f} m x {geom.frame_span:.2f} m")

    # ------------------------------------------------------------------ #
    # 1. Run verification suite
    # ------------------------------------------------------------------ #
    print("[1/5] Running verification suite...")
    vr = run_grillage_verification()
    n_checks = vr["n_checks"]
    n_passed = sum(1 for c in vr["checks"] if c["passed"])
    print(f"  {n_passed}/{n_checks} checks passed")

    # ------------------------------------------------------------------ #
    # 2. Combined case table (3 scenarios)
    # ------------------------------------------------------------------ #
    print("[2/5] Computing combined cases...")
    cases = combined_case_table(geom)
    for c in cases:
        print(f"  {c['scenario']}: p_ult={c['ice_pressure_ultimate_MPa']:.4f} MPa, "
              f"governing={c['governing_member']}")

    # ------------------------------------------------------------------ #
    # 3. Member-governed cases
    # ------------------------------------------------------------------ #
    print("[3/5] Computing member-governed cases...")
    mg_cases = member_governed_case_table()
    for c in mg_cases:
        print(f"  {c['scenario']}: p_ult={c['ice_pressure_ultimate_MPa']:.4f} MPa, "
              f"governing={c['governing_member']}")

    # ------------------------------------------------------------------ #
    # 4. Ultimate map (machine-readable)
    # ------------------------------------------------------------------ #
    print("[4/5] Generating ultimate map...")
    umap = panel_stress_ultimate_map(
        geom,
        axial_ratios=(0.0, 0.15, 0.30, 0.45, 0.60),
        patch_area_ratios=(0.1, 0.25, 0.50, 1.0),
        hydrostatic_pressures=(0.0, 5.0e4, 1.0e5, 1.5e5),
        design_pressure=2.0e6,
        design_provenance="IACS UR S22 (caller-supplied example)",
    )
    map_path = write_ultimate_map(umap, OUT / "ultimate_map.json")
    print(f"  Written {len(umap['rows'])} rows to {map_path}")

    # ------------------------------------------------------------------ #
    # 5. Literature benchmarks
    # ------------------------------------------------------------------ #
    print("[5/5] Literature benchmarks...")

    # T1: Timoshenko Table 30
    side, t, q, e, nu = 1.0, 0.010, 1.0e4, geom.young, geom.poisson
    d = e * t ** 3 / (12.0 * (1.0 - nu ** 2))
    r = plate_patch_response(span_x=side, span_y=side, d1=d, d2=d, d3=d,
                             pressure=q, terms=121, sample=21, poisson=nu)
    w_max_theory = 0.00406 * q * side ** 4 / d
    m_max_theory = 0.0479 * q * side ** 2
    w_err = 100 * (r["w_max"] / w_max_theory - 1.0)
    m_err = 100 * (r["m_x_max"] / m_max_theory - 1.0)
    print(f"  Timoshenko w_max: {w_err:+.4f} %")
    print(f"  Timoshenko M_max: {m_err:+.4f} %")

    # T7: Paik-Thayamballi
    axial_cap = stiffener_axial_capacity(geom)
    ref_panel = StiffenedPanel(
        span=geom.frame_spacing, spacing=geom.bay_spacing,
        plate_thickness=geom.plate_thickness,
        web_height=geom.stiffener.web_height,
        web_thickness=geom.stiffener.web_thickness,
        flange_width=geom.stiffener.flange_width,
        flange_thickness=geom.stiffener.flange_thickness,
        young=geom.young, poisson=geom.poisson,
        yield_stress=geom.yield_stress,
    ).validate()
    paik_err = 100 * (axial_cap["panel_ultimate_stress"] / ref_panel.panel_ultimate_stress - 1.0)
    print(f"  Paik-Thayamballi: {paik_err:+.4f} %")

    # ------------------------------------------------------------------ #
    # Generate figures
    # ------------------------------------------------------------------ #
    print("\nGenerating figures...")

    # Figure 1: Ultimate pressure bar chart
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    scenarios = [c["scenario"].replace(" + ", "\n+ ") for c in cases]
    module_vals = [c["ice_pressure_ultimate_MPa"] for c in cases]
    x = np.arange(len(scenarios))
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    bars = ax.bar(x, module_vals, 0.5, color=colors, edgecolor="white", linewidth=0.5)
    for i, (bar, val) in enumerate(zip(bars, module_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=9)
    ax.set_ylabel("Ultimate ice pressure [MPa]", fontsize=11)
    ax.set_title("GAP-PP-5: Ultimate ice patch pressure under combined loading", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(module_vals) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig1_path = OUT / "fig1_ultimate_pressure.png"
    fig.savefig(fig1_path, dpi=150)
    plt.close(fig)
    print(f"  fig1_ultimate_pressure.png")

    # Figure 2: Ultimate map heatmap
    rows = umap["rows"]
    axial = sorted(set(r["sigma_x_over_yield"] for r in rows))
    patch_fracs = sorted(set(r["patch_area_ratio"] for r in rows))
    hydro = sorted(set(r["hydrostatic_pressure_MPa"] for r in rows))
    mid_hydro = hydro[len(hydro) // 2]
    matrix = np.zeros((len(axial), len(patch_fracs)))
    for r in rows:
        if abs(r["hydrostatic_pressure_MPa"] - mid_hydro) < 1e-9:
            i = axial.index(r["sigma_x_over_yield"])
            j = patch_fracs.index(r["patch_area_ratio"])
            matrix[i, j] = r["ice_pressure_ultimate_MPa"]

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="YlOrRd",
                   extent=[min(patch_fracs), max(patch_fracs), min(axial), max(axial)])
    ax.set_xlabel("Patch area ratio", fontsize=11)
    ax.set_ylabel("Axial stress ratio (σ_x/σ_y)", fontsize=11)
    ax.set_title(f"Ultimate ice pressure [MPa] (hydrostatic = {mid_hydro:.2f} MPa)", fontsize=12)
    cbar = fig.colorbar(im, ax=ax, label="p_ult [MPa]")
    # Annotate cells
    for i in range(len(axial)):
        for j in range(len(patch_fracs)):
            val = matrix[i, j]
            if val > 0:
                ax.text(j + 0.5, i + 0.5, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if val > 3 else "black")
    fig.tight_layout()
    fig2_path = OUT / "fig2_ultimate_map.png"
    fig.savefig(fig2_path, dpi=150)
    plt.close(fig)
    print(f"  fig2_ultimate_map.png")

    # Figure 3: Literature benchmark errors
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    labels = ["Timoshenko\nw_max", "Timoshenko\nM_max", "Paik-Thayamballi\nσ_ult"]
    errors = [w_err, m_err, paik_err]
    colors_err = ["#4CAF50" if abs(e) < 3 else "#FF9800" if abs(e) < 15 else "#F44336" for e in errors]
    bars = ax.bar(labels, errors, color=colors_err, edgecolor="white", linewidth=0.5)
    ax.axhline(y=3, color="green", linestyle="--", alpha=0.5, label="<3% gate")
    ax.axhline(y=-3, color="green", linestyle="--", alpha=0.5)
    ax.axhline(y=15, color="orange", linestyle="--", alpha=0.5, label="<15% gate")
    ax.axhline(y=-15, color="orange", linestyle="--", alpha=0.5)
    ax.set_ylabel("Relative error [%]", fontsize=11)
    ax.set_title("Literature benchmark errors", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, val + (0.5 if val > 0 else -0.5),
                f"{val:+.4f}%", ha="center", va="bottom" if val > 0 else "top", fontsize=9)
    fig.tight_layout()
    fig3_path = OUT / "fig3_literature_errors.png"
    fig.savefig(fig3_path, dpi=150)
    plt.close(fig)
    print(f"  fig3_literature_errors.png")

    # ------------------------------------------------------------------ #
    # Write verification JSON
    # ------------------------------------------------------------------ #
    verification = {
        "suite": "grillage_ice_ultimate",
        "terms": 61,
        "passed": vr["passed"],
        "n_checks": n_checks,
        "n_passed": n_passed,
        "n_failed": n_checks - n_passed,
        "paik_comparison": [
            {
                "bay_spacing_m": geom.bay_spacing,
                "plate_thickness_mm": geom.plate_thickness * 1e3,
                "beta": geom.plate_slenderness,
                "sigma_u_effective_width_MPa": axial_cap["panel_ultimate_stress"] / 1e6,
                "sigma_u_paik_MPa": ref_panel.panel_ultimate_stress / 1e6,
                "rel_error": paik_err / 100,
            }
        ],
        "timoshenko_benchmark": {
            "w_max_computed": r["w_max"],
            "w_max_theory": w_max_theory,
            "w_max_error_pct": w_err,
            "m_max_computed": r["m_x_max"],
            "m_max_theory": m_max_theory,
            "m_max_error_pct": m_err,
        },
        "combined_cases": cases,
        "member_governed_cases": mg_cases,
    }
    ver_path = OUT / "verification.json"
    ver_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(f"\nWritten {ver_path}")

    # ------------------------------------------------------------------ #
    # Write combined cases JSON
    # ------------------------------------------------------------------ #
    cc_path = OUT / "combined_cases.json"
    cc_path.write_text(json.dumps(cases, indent=2, default=str), encoding="utf-8")
    print(f"Written {cc_path}")

    print("\n" + "=" * 60)
    print("GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE: Deliverable complete")
    print("=" * 60)
    print(f"Artifacts in: {OUT}")
    print(f"  - verification.json ({n_passed}/{n_checks} checks passed)")
    print(f"  - combined_cases.json ({len(cases)} scenarios)")
    print(f"  - ultimate_map.json ({len(umap['rows'])} rows)")
    print(f"  - fig1_ultimate_pressure.png")
    print(f"  - fig2_ultimate_map.png")
    print(f"  - fig3_literature_errors.png")


if __name__ == "__main__":
    main()
