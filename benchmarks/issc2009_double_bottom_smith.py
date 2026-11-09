"""Evidence run for the ISSC 2009 double-bottom Smith progressive collapse.

Produces ``artifacts/issc2009_double_bottom/results.json`` and six figures.
Every number in the JSON comes from the module under test; nothing is typed by
hand.  The experimental anchor for criterion C1 is injected through
``--anchor PATH`` and stays ``NOT_VERIFIED`` when no file is supplied.

Usage
-----
OMP_NUM_THREADS=1 PYTHONPATH=src python benchmarks/issc2009_double_bottom_smith.py \
    [--steps 320] [--pytest-tests 23] [--anchor artifacts/.../anchor.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import replace

from tensorfem.hull_girder_smith_collapse import (
    ISSC2009_DOUBLE_BOTTOM,
    curvature_grid,
    load_anchor,
    run_issc2009_double_bottom,
    solve_moment_curvature,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "issc2009_double_bottom"
FIGURES = OUTPUT / "figures"


def _curves(geometry, *, steps: int, maximum_ratio: float = 8.0):
    props = geometry.elastic_properties()
    eps_yield = geometry.yield_stress / geometry.young
    extreme = max(props["neutral_axis"], geometry.depth - props["neutral_axis"])
    kappa_yield = eps_yield / extreme
    grid = curvature_grid(kappa_yield, steps=steps, maximum_ratio=maximum_ratio)
    fibers = geometry.fibers()
    sagging = solve_moment_curvature(fibers, grid)
    hogging = solve_moment_curvature(fibers, tuple(-value for value in grid))
    return props, kappa_yield, sagging, hogging


def figure_moment_curvature(props, kappa_yield, sagging, hogging, report):
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ultimate = report["ultimate"]
    for axis, result, label, colour in (
            (axes[0], sagging, "sagging (bottom compressed)", "tab:blue"),
            (axes[1], hogging, "hogging (inner bottom compressed)", "tab:red")):
        kappa = [point.curvature / kappa_yield for point in result.points]
        moment = [abs(point.moment) / 1e6 for point in result.points]
        axis.plot(kappa, moment, colour, lw=2, label="Smith M-kappa")
        axis.axhline(props["first_yield_moment"] / 1e6, ls=":", color="grey",
                     label="M_y = %.2f MN m" % (props["first_yield_moment"] / 1e6))
        axis.axhline(report["section_properties"]["plastic_moment"] / 1e6, ls="--",
                     color="black",
                     label="M_p = %.2f MN m" % (report["section_properties"]["plastic_moment"] / 1e6))
        peak = ultimate["sagging_moment"] if colour == "tab:blue" else ultimate["hogging_moment"]
        axis.axhline(peak / 1e6, ls="-.", color=colour, alpha=0.7,
                     label="M_u = %.2f MN m" % (peak / 1e6))
        axis.set_xlabel(r"$\kappa/\kappa_y$ [-]")
        axis.set_ylabel("section moment [MN m]")
        axis.set_title(label)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7.5)
        axis.set_xlim(0, 8)
    figure.suptitle("ISSC 2009 double-bottom: Smith progressive collapse")
    figure.tight_layout()
    path = FIGURES / "moment_curvature.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def figure_component_curves(geometry):
    laws = geometry.component_laws()
    eps_yield = geometry.yield_stress / geometry.young
    strains = np.linspace(-6.0 * eps_yield, 6.0 * eps_yield, 601)
    figure, axis = plt.subplots(figsize=(7.0, 4.6))
    labels = {
        "bottom_plating": "bottom plating field (t=%.0f mm)" % (geometry.bottom_thickness * 1e3),
        "inner_bottom": "inner bottom field (t=%.0f mm)" % (geometry.inner_bottom_thickness * 1e3),
        "side_shell": "side shell field (t=%.0f mm)" % (geometry.side_thickness * 1e3),
        "girder_web": "girder web = min(plate, column)",
    }
    for key, label in labels.items():
        stress = [laws[key].stress(float(value)) / geometry.yield_stress for value in strains]
        axis.plot(strains / eps_yield, stress, lw=2, label=label)
    axis.axhline(0.0, color="grey", lw=0.8)
    axis.set_xlabel(r"$\varepsilon/\varepsilon_y$ [-] (negative = compression)")
    axis.set_ylabel(r"$\sigma/\sigma_y$ [-]")
    axis.set_title("component load-end-shortening curves (w$_0$/t = %.2f)" % geometry.imperfection_ratio)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = FIGURES / "component_curves.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def figure_imperfection(report):
    rows = report["imperfection_study"]
    ratios = [row["w0_over_t"] for row in rows]
    moments = [row["ultimate_moment"] / 1e6 for row in rows]
    drops = [100.0 * row["drop_vs_baseline"] for row in rows]
    plate = [row["plate_ultimate_ratio_bottom"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].plot(ratios, moments, "o-", color="tab:blue", lw=2)
    for ratio, moment, drop in zip(ratios, moments, drops):
        axes[0].annotate("%.1f%%" % drop, (ratio, moment),
                         textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    axes[0].set_xlabel(r"$w_0/t$ [-]")
    axes[0].set_ylabel(r"$M_u$ [MN m]")
    axes[0].set_title("ultimate moment vs initial imperfection")
    axes[0].grid(alpha=0.3)
    axes[1].plot(ratios, plate, "s-", color="tab:green", lw=2)
    axes[1].set_xlabel(r"$w_0/t$ [-]")
    axes[1].set_ylabel(r"$\sigma_{u,plate}/\sigma_y$ [-]")
    axes[1].set_title("bottom plating ultimate ratio (Faulkner + imperfection)")
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    path = FIGURES / "imperfection_sensitivity.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def figure_energy(sagging):
    kappa = [point.curvature for point in sagging.points]
    internal = [point.internal_energy / 1e3 for point in sagging.points]
    external = [point.external_work / 1e3 for point in sagging.points]
    peak = max(internal)
    residual = [abs(a - b) for a, b in zip(internal, external)]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].plot([value * 1e3 for value in kappa], internal, lw=2,
                 label=r"$U=\sum A\int\sigma\,d\varepsilon$")
    axes[0].plot([value * 1e3 for value in kappa], external, "--", lw=2,
                 label=r"$W=\int M\,d\kappa$")
    axes[0].set_xlabel(r"$\kappa$ [1/km]")
    axes[0].set_ylabel("work per unit length [kJ/m]")
    axes[0].set_title("energy ledger (sagging)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].semilogy([value * 1e3 for value in kappa], [max(r, 1e-12) for r in residual],
                     color="tab:purple", lw=2)
    axes[1].axhline(0.05 * peak, color="red", ls="--",
                    label="5% of peak internal energy")
    axes[1].set_xlabel(r"$\kappa$ [1/km]")
    axes[1].set_ylabel(r"$|U-W|$ [kJ/m]")
    axes[1].set_title("ledger residual vs the 5% bound")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    figure.tight_layout()
    path = FIGURES / "energy_ledger.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def figure_mesh_and_state(report, sagging, geometry):
    rows = report["mesh_study"]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].plot([row["subdivisions"] for row in rows],
                 [row["ultimate_moment"] / 1e6 for row in rows], "o-", lw=2,
                 color="tab:orange")
    axes[0].set_xlabel("fibres per deep member [-]")
    axes[0].set_ylabel(r"$M_u$ [MN m]")
    axes[0].set_title("mesh convergence (spread %.2f%%)" % (100 * report["criteria"]["C5"]["spread"]))
    axes[0].grid(alpha=0.3)
    kappa = [point.curvature / report["section_properties"]["yield_curvature"]
             for point in sagging.points]
    axes[1].plot(kappa, [point.neutral_axis for point in sagging.points], lw=2,
                 color="tab:brown", label="neutral axis y$_{na}$ [m]")
    axes[1].set_xlabel(r"$\kappa/\kappa_y$ [-]")
    axes[1].set_ylabel("y$_{na}$ [m]")
    twin = axes[1].twinx()
    twin.plot(kappa, [100 * point.yielded_fraction for point in sagging.points],
              lw=2, ls="--", color="tab:cyan", label="yielded area [%]")
    twin.set_ylabel("yielded area [%]")
    axes[1].set_title("neutral-axis migration and plastification")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="lower left", fontsize=8)
    twin.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    path = FIGURES / "mesh_and_state.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def figure_thickness_sweep(base_geometry, report, *, steps: int):
    thicknesses = (0.008, 0.010, 0.012)
    moments = []
    for value in thicknesses:
        geometry = replace(base_geometry,
                           bottom_thickness=value,
                           inner_bottom_thickness=value,
                           side_thickness=value,
                           girder_thickness=0.008)
        _, _, sagging, _ = _curves(geometry, steps=steps)
        moments.append(sagging.ultimate_moment / 1e6)
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot([value * 1e3 for value in thicknesses], moments, "o-", lw=2,
              color="tab:blue")
    axis.set_xlabel("uniform plating thickness [mm]")
    axis.set_ylabel(r"$M_u$ [MN m]")
    axis.set_title("thickness sweep across the node range t = 8..12 mm")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    path = FIGURES / "thickness_sweep.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path, {"thickness_mm": [value * 1e3 for value in thicknesses],
                  "ultimate_moment": moments}


def render_tables(payload: dict) -> str:
    """Render the result tables as markdown from ``results.json`` (no hand copy)."""
    report = payload["benchmark"]
    lines = []
    push = lines.append
    push("### 3.1 截面弹性/塑性性质（程序化注入）\n")
    push("| 量 | 数值 | 单位 |")
    push("|---|---|---|")
    props = report["section_properties"]
    for name, key, unit, scale in (
            ("截面面积 A", "area", "m^2", 1.0),
            ("中和轴 y_na", "neutral_axis", "m", 1.0),
            ("二次矩 I", "inertia", "m^4", 1.0),
            ("首次屈服弯矩 M_y", "first_yield_moment", "MN m", 1e-6),
            ("塑性弯矩 M_p", "plastic_moment", "MN m", 1e-6),
            ("塑性中和轴", "plastic_neutral_axis", "m", 1.0),
            ("屈服曲率 kappa_y", "yield_curvature", "1/m", 1.0)):
        push("| %s | %.4f | %s |" % (name, props[key] * scale, unit))
    push("| 纤维面积守恒相对误差 | %.2e | - |" % props["fibre_area_relative_error"])
    push("")
    push("### 3.2 极限弯矩\n")
    push("| 工况 | M_u (MN m) | kappa_u (1/m) | M_u/M_p | M_u/M_y | 峰值在扫描区间内部 |")
    push("|---|---|---|---|---|---|")
    ultimate = report["ultimate"]
    push("| 中垂 sagging | %.3f | %.5f | %.3f | %.3f | %s |" % (
        ultimate["sagging_moment"] / 1e6, ultimate["sagging_curvature"],
        ultimate["M_u_over_M_p"], ultimate["M_u_over_M_y"],
        ultimate["peak_is_interior"]))
    push("| 中拱 hogging | %.3f | %.5f | - | - | - |" % (
        ultimate["hogging_moment"] / 1e6, ultimate["hogging_curvature"]))
    envelope = report["cross_method_envelope"]
    push("")
    push("独立方法包络（极限应力模型，无软化）：M_env = %.3f MN m，"
         "M_u/M_env = %.3f。\n" % (envelope["envelope_ultimate_moment"] / 1e6,
                                   envelope["softening_over_envelope"]))
    push("### 3.3 判据 C1-C6\n")
    push("| 判据 | 内容 | 阈值 | 结果 | 状态 |")
    push("|---|---|---|---|---|")
    criteria = report["criteria"]
    push("| C1 | M_u vs ISSC 2009 双底实验锚 | <=15%% | 计算 M_u = %.3f MN m，锚值缺失 | %s |" % (
        criteria["C1"]["detail"]["computed_ultimate_moment"] / 1e6, criteria["C1"]["status"]))
    push("| C2 | 弹性段斜率 vs E*I | <=5%% | %.3f%% | %s |" % (
        100 * criteria["C2"]["relative_error"], criteria["C2"]["status"]))
    push("| C3 | w0/t=0.1 下降 | <=10%% | %.2f%% | %s |" % (
        100 * criteria["C3"]["drop_at_0.1"], criteria["C3"]["status"]))
    push("| C3 | w0/t=0.5 下降 | <=30%% | %.2f%% | %s |" % (
        100 * criteria["C3"]["drop_at_0.5"], criteria["C3"]["status"]))
    push("| C4 | 能量账本残差 | <=5%% | %.2e | %s |" % (
        criteria["C4"]["residual"], criteria["C4"]["status"]))
    push("| C5 | 10/20/40 网格 M_u 极差 | <=5%% | %.3f%% | %s |" % (
        100 * criteria["C5"]["spread"], criteria["C5"]["status"]))
    push("| C6 | pytest 用例数 | >=15 | %s | %s |" % (
        criteria["C6"]["collected_tests"], criteria["C6"]["status"]))
    push("")
    push("### 3.4 初始缺陷灵敏度\n")
    push("| w0/t | M_u (MN m) | 相对 w0/t=0.05 下降 | 底板块极限比 sigma_u/sigma_y |")
    push("|---|---|---|---|")
    for row in report["imperfection_study"]:
        push("| %.2f | %.3f | %.2f%% | %.4f |" % (
            row["w0_over_t"], row["ultimate_moment"] / 1e6,
            100 * row["drop_vs_baseline"], row["plate_ultimate_ratio_bottom"]))
    push("")
    push("### 3.5 网格收敛\n")
    push("| 每个深构件纤维数 | 总纤维数 | M_u (MN m) |")
    push("|---|---|---|")
    for row in report["mesh_study"]:
        push("| %d | %d | %.3f |" % (row["subdivisions"], row["fibres"],
                                      row["ultimate_moment"] / 1e6))
    push("")
    push("### 3.6 板厚扫描（节点给定 t=8..12 mm）\n")
    push("| 均匀板厚 (mm) | M_u (MN m) |")
    push("|---|---|")
    for value, moment in zip(payload["thickness_sweep"]["thickness_mm"],
                             payload["thickness_sweep"]["ultimate_moment"]):
        push("| %.0f | %.3f |" % (value, moment / 1e6))
    push("")
    push("### 3.7 求解器独立验证（与本次双底算例无关的闭式解）\n")
    engine = report["section_engine_verification"]
    push("| 检查 | 相对误差 | 阈值 | 通过 |")
    push("|---|---|---|---|")
    push("| M(κ) vs 理想弹塑性矩形闭式解 | %.2e | 3e-3 | %s |" % (
        engine["max_relative_moment_error"], engine["passed"]))
    push("| 弹性斜率 vs E*I | %.2e | 3e-3 | %s |" % (
        engine["elastic_slope_relative_error"], engine["passed"]))
    push("| 峰值弯矩 vs M_p | %.2e | - | - |" % engine["plastic_moment_relative_error"])
    push("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--pytest-tests", type=int, default=None)
    parser.add_argument("--anchor", type=Path, default=None)
    parser.add_argument("--render-only", action="store_true",
                        help="only re-render the markdown tables from results.json")
    arguments = parser.parse_args()

    if arguments.render_only:
        payload = json.loads((OUTPUT / "results.json").read_text(encoding="utf-8"))
        (OUTPUT / "results_tables.md").write_text(render_tables(payload), encoding="utf-8")
        print("tables ->", OUTPUT / "results_tables.md")
        return 0

    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    anchor = load_anchor(arguments.anchor)
    report = run_issc2009_double_bottom(
        anchor=anchor, curvature_steps=arguments.steps,
        subdivisions=(10, 20, 40),
        imperfection_ratios=(0.05, 0.1, 0.2, 0.5),
        pytest_tests=arguments.pytest_tests)

    geometry = ISSC2009_DOUBLE_BOTTOM
    props, kappa_yield, sagging, hogging = _curves(geometry, steps=arguments.steps)

    figures = {
        "moment_curvature": figure_moment_curvature(props, kappa_yield, sagging,
                                                    hogging, report),
        "component_curves": figure_component_curves(geometry),
        "imperfection_sensitivity": figure_imperfection(report),
        "energy_ledger": figure_energy(sagging),
        "mesh_and_state": figure_mesh_and_state(report, sagging, geometry),
    }
    sweep_path, sweep = figure_thickness_sweep(geometry, report, steps=max(60, arguments.steps // 2))
    figures["thickness_sweep"] = sweep_path

    energy_at_peak = max(point.internal_energy for point in sagging.points)
    residual_curve = [
        {"kappa": point.curvature,
         "internal_energy": point.internal_energy,
         "external_work": point.external_work,
         "neutral_axis": point.neutral_axis,
         "yielded_fraction": point.yielded_fraction,
         "moment": point.moment}
        for point in sagging.points[::max(1, len(sagging.points) // 60)]
    ]
    payload = {
        "schema": "tensorfem.issc2009-double-bottom-smith.evidence/1.0",
        "benchmark": report,
        "thickness_sweep": sweep,
        "ledger_curve": residual_curve,
        "ledger_reference_energy": energy_at_peak,
        "figures": {name: str(path.relative_to(ROOT)) for name, path in figures.items()},
        "commands": {
            "evidence": ("OMP_NUM_THREADS=1 PYTHONPATH=src python "
                         "benchmarks/issc2009_double_bottom_smith.py --steps %d"
                         % arguments.steps),
            "tests": "PYTHONPATH=src python -m pytest tests/test_hull_girder_smith_collapse.py",
        },
        "environment": {
            "python": "3.11",
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (OUTPUT / "results.json").write_text(json.dumps(payload, indent=1),
                                         encoding="utf-8")
    (OUTPUT / "results_tables.md").write_text(render_tables(payload), encoding="utf-8")

    print("--- criteria ---")
    for key, value in report["criteria"].items():
        print(key, value["status"])
    print("M_u sagging  = %.3f MN m" % (report["ultimate"]["sagging_moment"] / 1e6))
    print("M_u hogging  = %.3f MN m" % (report["ultimate"]["hogging_moment"] / 1e6))
    print("M_y / M_p    = %.3f / %.3f MN m"
          % (report["section_properties"]["first_yield_moment"] / 1e6,
             report["section_properties"]["plastic_moment"] / 1e6))
    print("overall      = %s (%d/6 PASS, pending %s)"
          % (report["overall_status"], report["criteria_passed"],
             report["criteria_pending"]))
    print("artifacts    =", OUTPUT / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())