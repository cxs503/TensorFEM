"""Comprehensive end-to-end case study for the ice-grillage ultimate strength anchor.

Consolidates rounds 1-5 into a single, publication-quality benchmark that:
1. Runs the full grillage ultimate strength analysis with all member models
2. Compares with literature benchmarks (Timoshenko, Paik-Thayamballi, Neal)
3. Generates publication-quality figures
4. Produces machine-readable JSON artifacts
5. Writes a six-section report suitable for publication

Run: ``PYTHONPATH=src python benchmarks/grillage_ultimate_comprehensive.py``
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem.grillage_ice_ultimate import (
    CombinedLoadCase,
    GrillageGeometry,
    MemberSection,
    beam_column_amplification_exact,
    beam_moment_centre_patch,
    case_study_geometry,
    frame_plastic_collapse_force,
    grillage_utilisations,
    panel_stress_ultimate_map,
    plate_patch_response,
    stiffener_axial_capacity,
    ultimate_ice_pressure,
    write_ultimate_map,
    system_load_shares,
)
from tensorfem.grillage_ice_verification import (
    case_study_geometry_slender,
    combined_case_table,
    member_governed_case_table,
    member_governed_geometry,
    plate_field_fd_check,
    run_grillage_verification,
)
from tensorfem.grillage_plastic_collapse import grillage_plastic_collapse, case_study_load_case
from tensorfem.grillage_member_envelope import member_envelope
from tensorfem.orthotropic_plate_fd import ssss_orthotropic_plate_fd

ROOT = pathlib.Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "grillage_ultimate_comprehensive"
FIG = ROOT / "docs" / "assets" / "grillage_ultimate_comprehensive"
DOC = ROOT / "docs"


def ensure_dirs():
    ART.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Section 1: Case introduction
# --------------------------------------------------------------------------- #
def case_introduction(geom: GrillageGeometry) -> dict:
    return {
        "title": "加筋板架在冰压斑+侧压组合下的极限强度",
        "description": (
            "Ice-class hull structural assessment requires the link from "
            "local panel bending stress to the ultimate strength of the "
            "orthogonally stiffened grillage under a combined ice pressure "
            "patch, hydrostatic lateral pressure, and hull-girder axial "
            "compression."
        ),
        "geometry": {
            "frame_spacing_m": geom.frame_spacing,
            "bay_spacing_m": geom.bay_spacing,
            "n_bays": geom.n_bays,
            "n_spans": geom.n_spans,
            "plate_thickness_mm": geom.plate_thickness * 1e3,
            "panel_length_m": geom.n_spans * geom.frame_spacing,
            "panel_width_m": geom.frame_span,
        },
        "material": {
            "young_GPa": geom.young / 1e9,
            "poisson": geom.poisson,
            "yield_stress_MPa": geom.yield_stress / 1e6,
            "grade": "EH36",
        },
        "stiffener": {
            "web_height_mm": geom.stiffener.web_height * 1e3,
            "web_thickness_mm": geom.stiffener.web_thickness * 1e3,
            "flange_width_mm": geom.stiffener.flange_width * 1e3,
            "flange_thickness_mm": geom.stiffener.flange_thickness * 1e3,
            "type": "Tee",
        },
        "frame": {
            "web_height_mm": geom.frame.web_height * 1e3,
            "web_thickness_mm": geom.frame.web_thickness * 1e3,
            "flange_width_mm": geom.frame.flange_width * 1e3,
            "flange_thickness_mm": geom.frame.flange_thickness * 1e3,
            "type": "Tee",
            "attached_plate_width_m": geom.frame_attached_width,
        },
    }


# --------------------------------------------------------------------------- #
# Section 2: Method
# --------------------------------------------------------------------------- #
def method_summary() -> dict:
    return {
        "plate_response": "Navier double Fourier series for SSSS orthotropic plate (121 terms, 0.5 % convergence gate).",
        "load_sharing": "Elastic modal partition: f_x + f_y + f_t = 1.",
        "member_checks": {
            "plate_local_bending": "Von Mises interaction with Mp = sigma_y * t^2 / 4.",
            "longitudinal_stiffener": "Beam-column with exact secant amplification 2(sec u - 1)/u^2; Faulkner effective width + Perry-Robertson / Johnson-Ostenfeld.",
            "transverse_frame": "Rigid-plastic collapse: 16 Mp/L (uniform), 8 Mp/(L(1-c/2L)) (patch).",
        },
        "plastic_collapse": "Successive plastic hinge formation (Neal 1963). Mp reduced by axial: Mp(1-(P/Py)^2).",
        "member_envelope": "max(B1, B2, C) per member: tributary, panel, discrete beam-grid branches.",
        "ultimate_solve": "Bisection on ice pressure until max utilisation = 1.0 (tol 1e-4).",
        "cross_checks": {
            "T1_T2": "Timoshenko Table 30 (3 % gate)",
            "T7": "Paik-Thayamballi empirical (15 % gate)",
            "T11": "Independent 13-point FD (0.5 % gate)",
            "T12": "Hand algebra stiffener-governed (0.5 % gate)",
            "T13": "Rigid-plastic >= elastic member limit",
        },
    }


# --------------------------------------------------------------------------- #
# Section 3: Results (lightweight - reduced FD grid)
# --------------------------------------------------------------------------- #
def compute_results(geom: GrillageGeometry) -> dict:
    results = {}

    # Combined case table (3 scenarios)
    results["combined_cases"] = combined_case_table(geom)

    # Member-governed cases
    results["member_governed"] = member_governed_case_table()

    # Plate field FD check (reduced grid for speed)
    results["plate_field_fd"] = plate_field_fd_check(geom, nx=120)

    # Ultimate map (reduced grid)
    results["ultimate_map"] = panel_stress_ultimate_map(
        geom,
        axial_ratios=(0.0, 0.30, 0.60),
        patch_area_ratios=(0.2, 0.6, 1.0),
        hydrostatic_pressures=(0.0, 1.0e5),
        design_pressure=2.0e6,
        design_provenance="IACS UR S22 (not obtained; placeholder)",
    )

    # Plastic collapse cross-check
    plastic = grillage_plastic_collapse(geom, case_study_load_case())
    systems = plastic.get("systems", {})
    stiffener_hinges = len(systems.get("longitudinal_stiffener", {}).get("hinges", []))
    frame_hinges = len(systems.get("transverse_frame", {}).get("hinges", []))
    results["plastic_collapse"] = {
        "ice_pressure_ultimate_MPa": plastic["ice_pressure_ultimate"] / 1e6,
        "ice_pressure_grillage_plastic_MPa": plastic["ice_pressure_grillage_plastic"] / 1e6,
        "governing_mode": plastic["governing_mode"],
        "governing_member_line": plastic["governing_member_line"],
        "n_hinges_stiffener": stiffener_hinges,
        "n_hinges_frame": frame_hinges,
        "plastic_reserve_ratio_stiffener": systems.get("longitudinal_stiffener", {}).get("plastic_reserve_ratio", 1.0),
    }

    # T1: Timoshenko Table 30
    side, t, q, e, nu = 1.0, 0.010, 1.0e4, geom.young, geom.poisson
    d = e * t ** 3 / (12.0 * (1.0 - nu ** 2))
    r = plate_patch_response(span_x=side, span_y=side, d1=d, d2=d, d3=d,
                             pressure=q, terms=121, sample=21, poisson=nu)
    w_max_theory = 0.00406 * q * side ** 4 / d
    m_max_theory = 0.0479 * q * side ** 2
    results["timoshenko_benchmark"] = {
        "w_max_computed": r["w_max"],
        "w_max_theory": w_max_theory,
        "w_max_error_pct": 100 * (r["w_max"] / w_max_theory - 1.0),
        "m_max_computed": r["m_x_max"],
        "m_max_theory": m_max_theory,
        "m_max_error_pct": 100 * (r["m_x_max"] / m_max_theory - 1.0),
    }

    # T7: Paik-Thayamballi
    axial_cap = stiffener_axial_capacity(geom)
    from tensorfem.stiffened_panel_ultimate import StiffenedPanel
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
    results["paik_benchmark"] = {
        "panel_ultimate_stress_MPa": axial_cap["panel_ultimate_stress"] / 1e6,
        "paik_panel_ultimate_stress_MPa": ref_panel.panel_ultimate_stress / 1e6,
        "error_pct": 100 * (axial_cap["panel_ultimate_stress"] / ref_panel.panel_ultimate_stress - 1.0),
    }

    return results


# --------------------------------------------------------------------------- #
# Section 4: Figures
# --------------------------------------------------------------------------- #
def generate_figures(geom: GrillageGeometry, results: dict) -> list[dict]:
    figures = []

    # Figure 1: Ultimate pressure comparison
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    scenarios = ["light", "design", "severe"]
    module_vals = [r["ice_pressure_ultimate_MPa"] for r in results["combined_cases"]]
    fd_vals = [r["p_ult_fd_MPa"] for r in results["plate_field_fd"]]
    x = np.arange(len(scenarios))
    ax.bar(x - 0.2, module_vals, 0.4, label="Module (bisection)")
    ax.bar(x + 0.2, fd_vals, 0.4, label="Independent FD")
    for i, (m, f) in enumerate(zip(module_vals, fd_vals)):
        ax.annotate(f"{100*(f/m-1):+.2f}%", (i, max(m, f)), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Light ice\n+ ballast", "Design ice\n+ design head", "Severe ice\n+ tank head"])
    ax.set_ylabel("Ultimate ice pressure [MPa]")
    ax.set_title("Ultimate pressure: module vs independent FD recomputation")
    ax.legend()
    fig.tight_layout()
    path = FIG / "fig1_ultimate_pressure.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append({
        "name": "fig1_ultimate_pressure",
        "caption": "Ultimate ice patch pressure: semi-analytical module vs independent FD recomputation for three combined load cases.",
        "png": str(path),
    })

    # Figure 2: Ultimate map heatmap
    rows = results["ultimate_map"]["rows"]
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
    ax.set_xlabel("Patch area ratio")
    ax.set_ylabel("Axial stress ratio")
    ax.set_title(f"Ultimate ice pressure [MPa] (hydrostatic = {mid_hydro:.2f} MPa)")
    fig.colorbar(im, ax=ax, label="p_ult [MPa]")
    fig.tight_layout()
    path = FIG / "fig2_ultimate_map.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append({
        "name": "fig2_ultimate_map",
        "caption": f"Ultimate ice pressure heatmap over axial stress ratio and patch area ratio (hydrostatic = {mid_hydro/1e5:.1f} bar).",
        "png": str(path),
    })

    # Figure 3: Literature benchmark comparison
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    benchmarks = ["Timoshenko\nw_max", "Timoshenko\nM_max", "Paik-Thayamballi\nσ_ult"]
    errors = [
        results["timoshenko_benchmark"]["w_max_error_pct"],
        results["timoshenko_benchmark"]["m_max_error_pct"],
        results["paik_benchmark"]["error_pct"],
    ]
    colors = ["C0" if abs(e) < 3 else "C3" for e in errors]
    bars = ax.bar(benchmarks, [abs(e) for e in errors], color=colors, width=0.6)
    ax.axhline(3, ls="--", color="gray", label="3 % gate")
    ax.axhline(15, ls=":", color="red", label="15 % gate (literature)")
    for bar, e in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{e:+.3f}%", ha="center", fontsize=9)
    ax.set_ylabel("Relative error [%]")
    ax.set_title("Literature benchmark errors")
    ax.legend()
    ax.set_ylim(0, max(abs(e) for e in errors) * 1.5 + 2)
    fig.tight_layout()
    path = FIG / "fig3_literature_errors.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append({
        "name": "fig3_literature_errors",
        "caption": "Relative errors vs literature benchmarks: Timoshenko plate coefficients and Paik-Thayamballi stiffened-panel strength.",
        "png": str(path),
    })

    return figures


# --------------------------------------------------------------------------- #
# Section 5: Notes
# --------------------------------------------------------------------------- #
def notes_section() -> list[str]:
    return [
        "1. **Small-deflection plate theory**: linear orthotropic plate theory; post-buckling reserve absent.",
        "2. **Stiffener web local buckling / tripping and frame tripping** not modelled.",
        "3. **Load sharing is elastic modal partition**: rigid-plastic collapse module captures redistribution (T13).",
        "4. **CSR/IACS rules not obtained**: literature comparison uses textbook closed-forms (Timoshenko, Neal) and Paik-Thayamballi empirical formula.",
        "5. **Frame end conditions**: discrete beam-grid uses simply supported; parent module uses fixed ends. Conservative simply supported value reported.",
        "6. **Smeared member demand** underestimates frame demand up to 35× (uniform) and 4.2× (patch on frame line). Envelope model recommended.",
    ]


# --------------------------------------------------------------------------- #
# Section 6: Error判定
# --------------------------------------------------------------------------- #
def error_judgment(results: dict) -> list[dict]:
    judgments = []
    e = results["timoshenko_benchmark"]["w_max_error_pct"]
    judgments.append({"quantity": "Plate max deflection (Timoshenko Table 30)", "benchmark": "0.00406 q a^4 / D", "error_pct": round(e, 4), "within_3_pct": abs(e) < 3, "within_15_pct": abs(e) < 15})
    e = results["timoshenko_benchmark"]["m_max_error_pct"]
    judgments.append({"quantity": "Plate max moment (Timoshenko Table 30)", "benchmark": "0.0479 q a^2", "error_pct": round(e, 4), "within_3_pct": abs(e) < 3, "within_15_pct": abs(e) < 15})
    e = results["paik_benchmark"]["error_pct"]
    judgments.append({"quantity": "Stiffened-panel ultimate stress (Paik-Thayamballi)", "benchmark": "Paik & Thayamballi empirical", "error_pct": round(e, 4), "within_3_pct": abs(e) < 3, "within_15_pct": abs(e) < 15})
    for r in results["plate_field_fd"]:
        e = r["p_ult_rel_error"] * 100
        judgments.append({"quantity": f"Ultimate pressure FD vs module ({r['scenario']})", "benchmark": "Independent 13-point FD", "error_pct": round(e, 4), "within_3_pct": abs(e) < 3, "within_15_pct": abs(e) < 15})
    return judgments


# --------------------------------------------------------------------------- #
# Report generation
# --------------------------------------------------------------------------- #
def generate_report(geom: GrillageGeometry, results: dict, figures: list[dict], judgments: list[dict]) -> str:
    intro = case_introduction(geom)
    method = method_summary()
    notes = notes_section()

    lines = [
        f"# {intro['title']}",
        "",
        "## 1. 案例介绍",
        "",
        f"**物理背景**: {intro['description']}",
        "",
        "### 几何参数",
        "",
        "| 参数 | 值 |",
        "| --- | --- |",
    ]
    for k, v in intro["geometry"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "### 材料参数", "", "| 参数 | 值 |", "| --- | --- |"]
    for k, v in intro["material"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "### 加筋参数", "", "| 参数 | 纵骨 | 横框架 |", "| --- | --- | --- |"]
    for k in intro["stiffener"]:
        sv = intro["stiffener"][k]
        fv = intro["frame"].get(k, "—")
        lines.append(f"| {k} | {sv} | {fv} |")

    lines += [
        "",
        "## 2. 方法",
        "",
        f"**板响应**: {method['plate_response']}",
        f"**载荷分配**: {method['load_sharing']}",
        "",
        "**成员校核**:",
        f"- 板局部弯曲: {method['member_checks']['plate_local_bending']}",
        f"- 纵骨梁柱: {method['member_checks']['longitudinal_stiffener']}",
        f"- 横框架塑性倒塌: {method['member_checks']['transverse_frame']}",
        "",
        f"**塑性倒塌**: {method['plastic_collapse']}",
        f"**成员包络**: {method['member_envelope']}",
        f"**极限求解**: {method['ultimate_solve']}",
        "",
        "### 独立交叉校核",
        "",
    ]
    for k, v in method["cross_checks"].items():
        lines.append(f"- **{k}**: {v}")

    lines += [
        "",
        "## 3. 计算结果",
        "",
        "### 3.1 组合工况（三档）",
        "",
        "| 工况 | p_ult [MPa] | 控制成员 | 纵骨利用率 | 框架利用率 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results["combined_cases"]:
        lines.append(
            f"| {r['scenario']} | {r['ice_pressure_ultimate_MPa']:.4f} | "
            f"{r['governing_member']} | {r['utilisation_stiffener']:.3f} | "
            f"{r['utilisation_frame']:.3f} |")

    lines += [
        "",
        "### 3.2 文献基准对照",
        "",
        "| 基准 | 计算值 | 基准值 | 误差 [%] |",
        "| --- | --- | --- | --- |",
        f"| Timoshenko w_max | {results['timoshenko_benchmark']['w_max_computed']:.6g} | {results['timoshenko_benchmark']['w_max_theory']:.6g} | {results['timoshenko_benchmark']['w_max_error_pct']:+.4f} |",
        f"| Timoshenko M_max | {results['timoshenko_benchmark']['m_max_computed']:.6g} | {results['timoshenko_benchmark']['m_max_theory']:.6g} | {results['timoshenko_benchmark']['m_max_error_pct']:+.4f} |",
        f"| Paik-Thayamballi σ_ult | {results['paik_benchmark']['panel_ultimate_stress_MPa']:.4f} MPa | {results['paik_benchmark']['paik_panel_ultimate_stress_MPa']:.4f} MPa | {results['paik_benchmark']['error_pct']:+.4f} |",
    ]
    lines += [
        "",
        "### 3.3 塑性倒塌交叉校核",
        "",
        f"- 弹性极限: {results['plastic_collapse']['ice_pressure_ultimate_MPa']:.4f} MPa",
        f"- 塑性倒塌极限: {results['plastic_collapse']['ice_pressure_grillage_plastic_MPa']:.4f} MPa",
        f"- 控制系统: {results['plastic_collapse']['governing_mode']}",
        f"- 纵骨塑性铰数: {results['plastic_collapse']['n_hinges_stiffener']}",
        f"- 框架塑性铰数: {results['plastic_collapse']['n_hinges_frame']}",
        "",
        "## 4. 结果图",
        "",
    ]
    for fig in figures:
        lines.append(f"![{fig['caption']}](../assets/grillage_ultimate_comprehensive/{fig['name']}.png)")
        lines.append(f"*{fig['caption']}*")
        lines.append("")

    lines += ["## 5. 注意点", ""]
    for note in notes:
        lines.append(note)
        lines.append("")

    lines += [
        "## 6. 误差判定",
        "",
        "| 关键量 | 基准 | 误差 [%] | < 3 % ? | < 15 % ? |",
        "| --- | --- | --- | --- | --- |",
    ]
    for j in judgments:
        lines.append(
            f"| {j['quantity']} | {j['benchmark']} | {j['error_pct']:+.4f} | "
            f"{'是' if j['within_3_pct'] else '否'} | {'是' if j['within_15_pct'] else '否'} |")

    all_within_15 = all(j["within_15_pct"] for j in judgments)
    all_within_3 = all(j["within_3_pct"] for j in judgments)
    lines += [
        "",
        f"**总体判定**: 所有文献/解析基准对照误差均 < 15 %: {'是' if all_within_15 else '否'}。",
        f"所有误差 < 3 %: {'是' if all_within_3 else '否'}。",
        "",
        "---",
        "*Generated by `benchmarks/grillage_ultimate_comprehensive.py`*",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ensure_dirs()
    geom = case_study_geometry()

    print("=" * 60)
    print("GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE: Comprehensive Case Study")
    print("=" * 60)

    print("\n[1/6] Case introduction...")
    intro = case_introduction(geom)
    print(f"  Panel: {intro['geometry']['panel_length_m']:.2f} m x {intro['geometry']['panel_width_m']:.2f} m")

    print("\n[2/6] Method summary...")
    method = method_summary()

    print("\n[3/6] Computing results...")
    results = compute_results(geom)
    print(f"  Combined cases: {len(results['combined_cases'])}")
    print(f"  Ultimate map rows: {len(results['ultimate_map']['rows'])}")
    print(f"  Timoshenko w_max error: {results['timoshenko_benchmark']['w_max_error_pct']:+.4f} %")
    print(f"  Paik-Thayamballi error: {results['paik_benchmark']['error_pct']:+.4f} %")

    print("\n[4/6] Generating figures...")
    figures = generate_figures(geom, results)
    for fig in figures:
        print(f"  {fig['name']}")

    print("\n[5/6] Error judgment...")
    judgments = error_judgment(results)
    for j in judgments:
        status = "✓" if j["within_15_pct"] else "✗"
        print(f"  {status} {j['quantity']}: {j['error_pct']:+.4f} %")

    print("\n[6/6] Writing report...")
    report = generate_report(geom, results, figures, judgments)
    report_path = DOC / "CASE-GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE-round6.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Report: {report_path}")

    evidence = {
        "anchor": "GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE",
        "round": 6,
        "geometry": intro["geometry"],
        "material": intro["material"],
        "combined_cases": results["combined_cases"],
        "member_governed": results["member_governed"],
        "plate_field_fd": results["plate_field_fd"],
        "ultimate_map": results["ultimate_map"],
        "plastic_collapse": results["plastic_collapse"],
        "timoshenko_benchmark": results["timoshenko_benchmark"],
        "paik_benchmark": results["paik_benchmark"],
        "judgments": judgments,
        "figures": [{"name": f["name"], "caption": f["caption"]} for f in figures],
    }
    evidence_path = ART / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, default=str), encoding="utf-8")
    print(f"  Evidence: {evidence_path}")

    map_path = ART / "ultimate_map.json"
    write_ultimate_map(results["ultimate_map"], map_path)
    print(f"  Ultimate map: {map_path}")

    print("\n" + "=" * 60)
    print("DONE: All artifacts written.")
    print("=" * 60)
    return evidence


if __name__ == "__main__":
    main()
