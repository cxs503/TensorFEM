"""Audit: finite-width plate with a central hole - production Q4 path vs Howland.

Produces ``benchmarks/finite_width_hole_audit.json`` and the figures in
``docs/assets/finite_width_hole/``.  Everything printed here is programmatic; no
number is transcribed by hand.

Run:  OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONPATH=src python scripts/audit_finite_width_hole.py
"""
from __future__ import annotations

import json
import math
import os
import pathlib

import matplotlib
import matplotlib.tri

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem.finite_width_hole import (
    HOWLAND_CHART,
    StripHole,
    howland_kt_net,
    solve_strip_case,
    strip_model,
    stress_at_point,
    strip_convergence,
)
from tensorfem.continuum import solve_continuum

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets" / "finite_width_hole"
BENCH = ROOT / "benchmarks"
RHOS = (0.1, 0.2, 0.3, 0.4, 0.5)
LEVELS = (24, 32, 40)
FINEST = 48


def _case(rho: float, length_ratio: float = 6.0) -> StripHole:
    half_width = 1.0 / rho
    return StripHole(radius=1.0, half_width=half_width, half_length=length_ratio * half_width)


def main() -> dict:
    ASSETS.mkdir(parents=True, exist_ok=True)
    BENCH.mkdir(parents=True, exist_ok=True)

    cases = []
    for rho in RHOS:
        strip = _case(rho)
        conv = strip_convergence(strip, levels=LEVELS)
        finest = solve_strip_case(strip, FINEST, int(round(FINEST * 0.75)), 4.0)
        ref = howland_kt_net(rho)
        cases.append(
            {
                "rho": rho,
                "half_width": strip.half_width,
                "half_length": strip.half_length,
                "howland_kt_net": ref,
                "chart_kt_net": min(HOWLAND_CHART, key=lambda kv: abs(kv[0] - rho))[1],
                "levels": conv["levels"],
                "finest": finest,
                "finest_relative_error": abs(finest["kt_net"] - ref) / ref,
                "chart_relative_error": abs(finest["kt_net"] - min(HOWLAND_CHART, key=lambda kv: abs(kv[0] - rho))[1])
                / min(HOWLAND_CHART, key=lambda kv: abs(kv[0] - rho))[1],
            }
        )
        print(
            f"rho={rho:.2f} kt_net={finest['kt_net']:.4f} howland={ref:.4f} "
            f"err={100 * cases[-1]['finest_relative_error']:+.2f}% dofs={finest['dofs']}"
        )

    # small-rho anchor: the finite-width factor must approach Kirsch K_t = 3
    small = _case(0.05)
    small_res = solve_strip_case(small, FINEST, int(round(FINEST * 0.75)), 4.0)

    # length sensitivity: the strip limit needs L/b -> inf
    length_study = []
    for ratio in (3.0, 4.0, 6.0, 8.0):
        strip = _case(0.3, ratio)
        res = solve_strip_case(strip, 40, 30, 4.0)
        length_study.append(
            {
                "half_length_over_half_width": ratio,
                "kt_net": res["kt_net"],
                "relative_error_vs_howland": abs(res["kt_net"] - howland_kt_net(0.3)) / howland_kt_net(0.3),
            }
        )
        print(f"L/b={ratio} kt_net={res['kt_net']:.4f}")

    summary = {
        "description": "Finite-width plate with a central hole: production Q4 plane-stress path vs Howland",
        "criterion": "finest-mesh net-section K_t within 3 % of the Howland literature value",
        "cases": cases,
        "kirsch_limit": {
            "rho": 0.05,
            "kt_net": small_res["kt_net"],
            "relative_error_vs_howland": abs(small_res["kt_net"] - howland_kt_net(0.05))
            / howland_kt_net(0.05),
            "relative_error_vs_3": abs(small_res["kt_net"] - 3.0) / 3.0,
        },
        "length_study": length_study,
        "max_error_all_cases": max(c["finest_relative_error"] for c in cases),
        "max_chart_error_all_cases": max(c["chart_relative_error"] for c in cases),
        "passes_3pct": bool(max(c["finest_relative_error"] for c in cases) < 0.03),
        "field_case": {"rho": 0.3, "n_theta": 56, "n_radial": 42, "peak_over_net": None},
    }
    (BENCH / "finite_width_hole_audit.json").write_text(json.dumps(summary, indent=2))

    # ---- figure 1: K_t vs rho (FE converged, Howland fit, chart points) ----
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    rho_fine = np.linspace(0.02, 0.55, 200)
    ax.plot(rho_fine, [howland_kt_net(float(r)) for r in rho_fine], "-",
            color="C0", lw=1.8, label="Howland/Heywood fit (literature)")
    ax.plot([c[0] for c in HOWLAND_CHART], [c[1] for c in HOWLAND_CHART], "s",
            color="C1", ms=7, label="Peterson chart values (literature)")
    ax.plot([c["rho"] for c in cases], [c["finest"]["kt_net"] for c in cases], "o-",
            color="C3", ms=6, label="TensorFEM Q4 fan mesh (finest, n_theta=48)")
    ax.plot([0.05], [small_res["kt_net"]], "^", color="C4", ms=8,
            label="rho=0.05 anchor (Kirsch limit K_t->3)")
    ax.axhline(3.0, color="k", ls=":", lw=1.0, label="infinite plate K_t = 3")
    ax.set_xlabel(r"$\rho = d/W$")
    ax.set_ylabel(r"$K_t$ (net section, $\sigma_{max}/\sigma_{net}$)")
    ax.set_title("Finite-width plate with a central hole under tension")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(ASSETS / "kt_vs_rho.png", dpi=160)
    plt.close(fig)

    # ---- figure 2: sigma_xx field + hole-edge hoop profile (rho = 0.3) ----
    strip = _case(0.3)
    model, meta = strip_model(strip, 56, 42, 4.0)
    result = solve_continuum(model)
    disp = result.displacement.detach().cpu().numpy()
    nodes = meta["nodes"]
    elements = meta["elements"]
    stored = result.stress.detach().cpu().numpy()
    sxx_nodes = np.zeros(nodes.shape[0])
    counts = np.zeros(nodes.shape[0])
    for e, element in enumerate(elements):
        value = stored[e][0]
        sxx_nodes[element] += value
        counts[element] += 1.0
    sxx_nodes /= np.maximum(counts, 1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    tris = np.vstack([elements[:, [0, 1, 2]], elements[:, [0, 2, 3]]])
    triangulation = matplotlib.tri.Triangulation(nodes[:, 0], nodes[:, 1], tris)
    tri = ax1.tripcolor(triangulation, sxx_nodes / strip.net_stress,
                        shading="gouraud", cmap="viridis")
    ax1.set_aspect("equal")
    ax1.set_xlabel("x/a")
    ax1.set_ylabel("y/a")
    ax1.set_title(r"$\sigma_{xx}/\sigma_{net}$ (rho = 0.3, quarter model)")
    fig.colorbar(tri, ax=ax1, shrink=0.85)

    theta = np.linspace(0.0, math.pi / 2.0, 181)
    hoop = []
    for t in theta:
        # points slightly inside the hole edge, averaged over two radii
        vals = []
        for scale in (1.03, 1.06):
            p = (scale * strip.radius * math.cos(t), scale * strip.radius * math.sin(t))
            sampled = stress_at_point(meta, disp, p, strip)
            sxx, syy, sxy = sampled["stress"]
            c, s = math.cos(t), math.sin(t)
            vals.append(sxx * s * s + syy * c * c - 2.0 * sxy * c * s)
        hoop.append(float(np.mean(vals)))
    ax2.plot(np.degrees(theta), np.asarray(hoop) / strip.net_stress, "-", color="C3",
             label="TensorFEM hoop stress (rho = 0.3)")
    theta_k = np.linspace(0.0, math.pi / 2.0, 181)
    kirsch = (1.0 + 2.0 * np.cos(2.0 * theta_k)) / strip.net_stress
    ax2.plot(np.degrees(theta_k), kirsch, "--", color="C0",
             label="Kirsch (infinite plate, same net stress)")
    ax2.set_xlabel("theta (deg)")
    ax2.set_ylabel(r"$\sigma_{\theta\theta}/\sigma_{net}$")
    ax2.set_title("Hoop stress around the hole (r = 1.03a..1.06a)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "field_and_hoop.png", dpi=160)
    plt.close(fig)
    summary["field_case"]["peak_over_net"] = float(max(hoop) / strip.net_stress)
    (BENCH / "finite_width_hole_audit.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps({k: summary[k] for k in ("max_error_all_cases", "passes_3pct")}, indent=2))
    return summary


if __name__ == "__main__":
    main()