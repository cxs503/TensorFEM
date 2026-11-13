"""Figures for the H1-BROKEN-ICE-DEM study (reads artifacts/.../study.json only).

Run:  PYTHONPATH=src python scripts/broken_ice_floe_field_figures.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "broken_ice_floe_field"


def fig_floatation(payload: dict) -> Path:
    rows = payload["floatation"]
    heave = payload["heave"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    ratios = [r["density_ratio"] for r in rows]
    analytic = [r["freeboard_analytic"] for r in rows]
    measured = [r["freeboard_measured"] for r in rows]
    ax.plot(ratios, analytic, "k-", label="Archimedes $f=h(1-\\rho_i/\\rho_w)$")
    ax.fill_between(ratios, [a * 0.95 for a in analytic], [a * 1.05 for a in analytic],
                    color="0.85", label="5% acceptance band")
    ax.plot(ratios, measured, "o", color="tab:red", ms=7, label="floe-field dynamics")
    ax.set_xlabel("ice / water density ratio $\\rho_i/\\rho_w$")
    ax.set_ylabel("freeboard $f$ (m), h = 0.6 m")
    ax.set_title("(a) floatation equilibrium vs Archimedes")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    thick = [r["thickness"] for r in heave]
    ax.plot(thick, [r["period_analytic"] for r in heave], "k-",
            label="$2\\pi/\\omega$, $\\omega=\\sqrt{\\rho_w g/(\\rho_i h)}$")
    ax.plot(thick, [r["period_measured"] for r in heave], "s", color="tab:blue",
            label="measured zero crossings")
    ax.set_xlabel("ice thickness $h$ (m)")
    ax.set_ylabel("heave period (s)")
    ax.set_title("(b) free heave oscillation period")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT / "fig_floatation_heave.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_field_map(payload: dict) -> Path:
    data = torch.load(OUT / "canonical_field.pt", weights_only=True)
    speed_row = next(r for r in payload["clearing_speed"] if r["speed"] == 1.5)
    half_width = speed_row["half_width"]
    cfg = {"bow_half_angle_deg": 30.0, "bow_length": 8.0, "field_length": 60.0}
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ax.axvspan(-half_width, half_width, color="tab:cyan", alpha=0.12,
               label=f"channel corridor $\\pm${half_width:.2f} m")
    p0 = data["positions0"].numpy()
    p1 = data["positions"].numpy()
    radii = data["radii"].numpy()
    for (x0, y0), r in zip(p0, radii):
        ax.add_patch(plt.Circle((x0, y0), r, facecolor="0.75", edgecolor="0.45", lw=0.5))
    for (x1, y1), r in zip(p1, radii):
        ax.add_patch(plt.Circle((x1, y1), r, facecolor="tab:blue", alpha=0.55,
                                edgecolor="tab:blue", lw=0.5))
    # bow outline at the final apex position
    apex = cfg["field_length"] + cfg["bow_length"] * 0.0 + 2.0
    hw = cfg["bow_length"] * math.tan(math.radians(cfg["bow_half_angle_deg"]))
    ax.plot([0.0, -hw, -hw], [apex, apex - cfg["bow_length"], apex - cfg["bow_length"] - 45.0],
            "k-", lw=1.4, label="bow faces + parallel body")
    ax.plot([0.0, hw, hw], [apex, apex - cfg["bow_length"], apex - cfg["bow_length"] - 45.0],
            "k-", lw=1.4)
    ax.set_xlim(-25, 25)
    ax.set_ylim(-5, 62)
    ax.set_aspect("equal")
    ax.set_xlabel("x across the channel (m)")
    ax.set_ylabel("y along the track (m)")
    ax.set_title("broken-ice field: initial (grey) vs after the sweep (blue)\n"
                 f"V = 1.5 m/s, N = {len(radii)} floes, seed = 20261113")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = OUT / "fig_floe_field_map.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_clearing(payload: dict) -> Path:
    speeds = payload["clearing_speed"]
    angles = payload["clearing_bow_angle"]
    friction = payload["clearing_friction"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    ax = axes[0]
    v = [r["speed"] for r in speeds]
    ax.plot(v, [100 * r["clearing_rate_centroid"] for r in speeds], "o-",
            label="centroid clearing rate")
    ax.plot(v, [100 * r["clearing_rate_coverage"] for r in speeds], "s--",
            label="channel coverage clearing")
    ax.plot(v, [100 * r["dragged_forward_fraction"] for r in speeds], "^:",
            label="dragged forward (bulldozed out of the window)")
    ax.plot(v, [100 * r["residual_fraction"] for r in speeds], "v:",
            label="residual floes left inside the corridor")
    ax.set_xlabel("advance speed V (m/s)")
    ax.set_ylabel("percent of initial corridor floes / coverage")
    ax.set_title("(a) clearing statistics vs advance speed (bow half angle 30$^\\circ$)")
    ax.set_ylim(-2, 105)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    labels = [f"{r['half_angle_deg']:.0f}$^\\circ$" for r in angles] + [
        f"$\\mu$={r['friction']:.1f}" for r in friction
    ]
    values = [100 * r["clearing_rate_centroid"] for r in angles] + [
        100 * r["clearing_rate_centroid"] for r in friction
    ]
    colors = ["tab:blue"] * len(angles) + ["tab:orange"] * len(friction)
    ax.bar(labels, values, color=colors)
    for k, val in enumerate(values):
        ax.text(k, val + 1.0, f"{val:.0f}", ha="center", fontsize=8)
    ax.set_ylabel("centroid clearing rate (%)")
    ax.set_title("(b) sensitivity: bow half angle (blue) and friction (orange)")
    ax.set_ylim(0, 110)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = OUT / "fig_clearing.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    payload = json.loads((OUT / "study.json").read_text())
    for path in (fig_floatation(payload), fig_field_map(payload), fig_clearing(payload)):
        print(path)


if __name__ == "__main__":
    main()