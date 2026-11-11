"""Figures for the W22 ice-coupled-6DOF ensemble report.

Reads *only* the artefacts produced by ``benchmarks/w22_ice_6dof_ensemble.py``
(``docs/w22_ice_6dof/*.json`` and the per-case CSV series) so that no number in
a figure is re-typed by hand.

Usage::

    python benchmarks/w22_ice_6dof_figures.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "w22_ice_6dof"
FIG = OUT / "figures"

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "font.size": 8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }
)


def load(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def read_series(path: pathlib.Path) -> dict:
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {name: data[name] for name in data.dtype.names}


def fig_load_timeseries(summary: dict) -> pathlib.Path:
    cases = [row for row in summary["ensemble"] if row["series_file"]]
    fig, axes = plt.subplots(3, 2, figsize=(6.6, 5.2), sharex=True)
    for ax, case in zip(axes.ravel(), cases):
        series = read_series(ROOT / case["series_file"])
        ax.plot(series["time_s"], series["contact_force_N"] / 1.0e6, lw=0.7, color="#1f4e79")
        ax.axhline(case["threshold_MN"], color="#c00000", lw=0.8, ls="--")
        ax.axvline(summary["provenance"]["settings"]["transient_s"], color="0.4", lw=0.7, ls=":")
        ax.set_title(
            f"h = {case['thickness_m']:.1f} m, U = {case['commanded_speed_mps']:.1f} m/s"
            f"  (mean {case['mean_load_MN']:.3f} MN, peak {case['peak_load_MN']:.3f} MN,"
            f" {case['cycle_count']} cycles)"
        )
        ax.set_ylabel("$N_{ice}$ [MN]")
    for ax in axes[-1]:
        ax.set_xlabel("t [s]")
    fig.suptitle(
        "ISO 19906 area-law contact load, 6-DOF time domain (dashed = 1.25 x mean threshold,"
        " dotted = end of start-up transient)",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = FIG / "fig1_load_timeseries.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_pressure_area(summary: dict, balance: dict) -> pathlib.Path:
    law = summary["provenance"]["law"]
    contact = summary["provenance"]["contact"]
    case = summary["ensemble"][2] if len(summary["ensemble"]) > 2 else summary["ensemble"][0]
    series = read_series(ROOT / case["series_file"])
    areas = np.linspace(0.02, 1.2, 400)
    pressure = law["reference_pressure_MPa"] * (areas / law["reference_area_m2"]) ** law["exponent"]
    force = pressure * areas

    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.8))
    axes[0].loglog(
        areas, pressure, color="#1f4e79", label=f"$p_G=C_R(A/A_0)^{{{law['exponent']:.2f}}}$"
    )
    axes[0].set_xlabel("contact area $A$ [m$^2$]")
    axes[0].set_ylabel("$p_G$ [MPa]")
    axes[0].set_title("global pressure falls with area ($e_x<0$)")
    axes[0].legend()

    axes[1].loglog(areas, force / 1.0e6, color="#7f3f00")
    axes[1].loglog(
        np.clip(series["contact_area_m2"], 1e-6, None),
        series["contact_force_N"] / 1.0e6,
        ".",
        ms=1.0,
        color="#c00000",
        alpha=0.35,
        label="simulated contact",
    )
    axes[1].set_xlabel("contact area $A$ [m$^2$]")
    axes[1].set_ylabel("$N=p_G A$ [MN]")
    axes[1].set_title(f"force $\\propto A^{{{law['force_area_exponent']:.2f}}}$")
    axes[1].legend()

    axes[2].plot(series["time_s"], series["penetration_m"], lw=0.7, color="#1f4e79")
    axes[2].axhline(contact["spall_depth_m"], color="#c00000", lw=0.8, ls="--", label="spall depth")
    axes[2].axhline(
        contact["residual_penetration_m"], color="0.3", lw=0.8, ls=":", label="residual penetration"
    )
    axes[2].set_xlabel("t [s]")
    axes[2].set_ylabel("penetration $d$ [m]")
    axes[2].set_title("crushing / spalling saw-tooth")
    axes[2].legend()
    fig.tight_layout()
    path = FIG / "fig2_area_law_and_contact.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_energy(balance: dict) -> pathlib.Path:
    case = balance["cases"][2] if len(balance["cases"]) > 2 else balance["cases"][0]
    windows = case["windowed_residuals"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    labels = ["thrust", "ice normal", "ice friction", "damping", "$\\Delta$KE", "$\\Delta$PE"]
    keys = [
        "work_thrust_J",
        "work_ice_normal_J",
        "work_ice_friction_J",
        "work_damping_J",
        "kinetic_energy_change_J",
        "potential_energy_change_J",
    ]
    values = [case["statistics_window"][key] / 1.0e6 for key in keys]
    colors = ["#2e7d32" if v > 0 else "#c62828" for v in values]
    axes[0].bar(labels, values, color=colors)
    axes[0].axhline(0.0, color="0.2", lw=0.8)
    axes[0].set_ylabel("energy [MJ]")
    axes[0].set_title(
        f"energy budget, {case['case']} (residual {1e0 * case['statistics_window']['power_quadrature_residual_fraction']:.1e})"
    )
    axes[0].tick_params(axis="x", rotation=35)

    starts = [0.5 * (w["window_s"][0] + w["window_s"][1]) for w in windows]
    resid = [1e2 * w["residual_fraction"] for w in windows]
    axes[1].bar(starts, resid, width=4.0, color="#1f4e79")
    axes[1].set_xlabel("window centre [s]")
    axes[1].set_ylabel("residual [% of work]")
    axes[1].set_title("power-quadrature residual, 5 s windows")
    fig.tight_layout()
    path = FIG / "fig3_energy_budget.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_thickness_trend(summary: dict) -> pathlib.Path:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    speeds = sorted({row["commanded_speed_mps"] for row in summary["ensemble"]})
    markers = {speeds[0]: "o", speeds[-1]: "s"} if len(speeds) > 1 else {speeds[0]: "o"}
    for speed in speeds:
        rows = sorted(
            [row for row in summary["ensemble"] if row["commanded_speed_mps"] == speed],
            key=lambda r: r["thickness_m"],
        )
        h = [row["thickness_m"] for row in rows]
        axes[0].plot(
            h,
            [row["mean_load_MN"] for row in rows],
            marker=markers[speed],
            label=f"U={speed:.1f} m/s",
        )
        axes[1].plot(
            h,
            [row["peak_load_MN"] for row in rows],
            marker=markers[speed],
            label=f"U={speed:.1f} m/s",
        )
        axes[1].plot(
            h,
            [row["mean_load_MN"] for row in rows],
            marker=markers[speed],
            ls="--",
            lw=0.8,
            alpha=0.6,
        )
    axes[0].set_xlabel("ice thickness h [m]")
    axes[0].set_ylabel("mean load [MN]")
    axes[0].set_title("mean load vs thickness")
    axes[0].legend()
    axes[1].set_xlabel("ice thickness h [m]")
    axes[1].set_ylabel("load [MN]")
    axes[1].set_title("peak (solid) and mean (dashed)")
    axes[1].legend()
    pooled = summary["pooled_thickness_trend"]
    fig.suptitle(
        f"thickness trend: pooled Spearman $\\rho$ = {pooled['rho']:.4f}, exact p = {pooled['p_value']:.4f} (tied ranks)",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = FIG / "fig4_thickness_trend.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def quantize(path: pathlib.Path, colors: int = 48, width: int = 900) -> None:
    """Palette-quantise a PNG so the report payload stays small."""
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.width > width:
            height = int(round(image.height * width / image.width))
            image = image.resize((width, height), Image.LANCZOS)
        image = image.quantize(colors=colors, method=Image.MEDIANCUT, dither=Image.NONE)
        image.save(path, optimize=True)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    summary = load("timeseries_summary.json")
    balance = load("energy_balance.json")
    paths = [
        fig_load_timeseries(summary),
        fig_pressure_area(summary, balance),
        fig_energy(balance),
        fig_thickness_trend(summary),
    ]
    for path in paths:
        quantize(path)
        print(f"{path.relative_to(ROOT)}  {path.stat().st_size / 1024:.1f} kB")
    total = sum(path.stat().st_size for path in paths)
    print(f"total {total / 1024:.1f} kB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
