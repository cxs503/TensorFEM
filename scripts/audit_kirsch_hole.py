"""Kirsch circular-hole verification audit: JSON evidence and figures.

Writes ``benchmarks/kirsch_hole_reference.json`` (mesh levels, errors, load
resultants, hole-edge stress concentration, recovery cross-check) and three
figures under ``docs/assets/kirsch/``.

Run: ``OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/audit_kirsch_hole.py``
"""
from __future__ import annotations

import json
import math
import platform as host
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem import __version__
from tensorfem.kirsch_hole import (
    KirschHole,
    annulus_model,
    arc_resultant,
    convergence_study,
    solve_case,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks" / "kirsch_hole_reference.json"
ASSETS = ROOT / "docs" / "assets" / "kirsch"
LEVELS = (16, 24, 32, 48)
OUTER = 10.0
HOLE = KirschHole(radius=1.0, far_field=1.0, young=1.0, poisson=0.3, plane="stress")
FIGURES: list[Path] = []


def _serialisable(case: dict) -> dict:
    return {
        key: case[key]
        for key in (
            "n_theta",
            "n_radial",
            "grading",
            "elements",
            "dofs",
            "displacement_error",
            "stress_rows",
            "max_stress_relative_error",
            "hole_edge",
            "load_resultant",
            "recovery_cross_check",
        )
    }


def _figures(study: dict) -> dict:
    ASSETS.mkdir(parents=True, exist_ok=True)
    levels = study["levels"]

    # (a) hoop stress along the symmetry ray theta = pi/2
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    r = np.linspace(1.0, 3.0, 400)
    exact = HOLE.stress_polar(r, math.pi / 2)[1]
    ax.plot(r, exact, "k-", lw=2.0, label="Kirsch exact")
    for case, marker in zip(study["cases"], ("o", "s", "^", "D")):
        rows = [row for row in case["stress_rows"] if row["theta_degrees"] > 89.0]
        ax.plot(
            [row["r_over_a"] for row in rows],
            [row["hoop_fe"] for row in rows],
            marker=marker,
            ms=5,
            ls="none",
            label=f"Q4 n={case['n_theta']}",
        )
    ax.axhline(3.0, color="tab:red", ls=":", lw=1.2, label=r"$K_t=3$")
    ax.set_xlabel(r"$r/a$")
    ax.set_ylabel(r"$\sigma_{\theta\theta}/T$")
    ax.set_title("Hoop stress on the maximum-stress ray")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path = ASSETS / "kirsch_hoop_profile.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    FIGURES.append(path)

    # (b) field comparison: element-centre hoop stress, FE vs exact contours
    case = study["cases"][-1]
    meta = case["meta"]
    nodes = np.asarray(meta["nodes"])
    elements = np.asarray(meta["elements"])
    centres = nodes[elements].mean(axis=1)
    stress = np.asarray(case["stored_stress"])
    theta_c = np.arctan2(centres[:, 1], centres[:, 0])
    c, s = np.cos(theta_c), np.sin(theta_c)
    hoop = stress[:, 0] * s * s + stress[:, 1] * c * c - 2.0 * stress[:, 2] * c * s
    order = np.argsort(hoop)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3), sharey=True)
    sc = axes[0].tripcolor(
        centres[:, 0], centres[:, 1], hoop, shading="gouraud", cmap="coolwarm"
    )
    axes[0].set_title(f"FE element-centre $\\sigma_{{\\theta\\theta}}/T$ (n={case['n_theta']})")
    fig.colorbar(sc, ax=axes[0], shrink=0.85)
    grid = np.linspace(0.0, 1.05, 220)
    xs, ys = np.meshgrid(grid, grid)
    rr = np.hypot(xs, ys)
    tt = np.arctan2(ys, xs)
    mask = rr >= 1.0
    exact_field = np.where(mask, HOLE.stress_polar(np.where(mask, rr, 1.0), tt)[1], np.nan)
    cs = axes[1].contourf(xs, ys, exact_field, levels=24, cmap="coolwarm")
    axes[1].set_title("Kirsch exact $\\sigma_{\\theta\\theta}/T$")
    fig.colorbar(cs, ax=axes[1], shrink=0.85)
    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xlabel("x / a")
    axes[0].set_ylabel("y / a")
    path = ASSETS / "kirsch_hoop_field.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    FIGURES.append(path)

    # (c) convergence
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    hs = 1.0 / np.asarray(levels)
    axes[0].loglog(
        hs,
        [case["displacement_error"]["l2_relative"] for case in study["cases"]],
        "o-",
        label="nodal displacement $L_2$",
    )
    axes[0].loglog(
        hs,
        [case["max_stress_relative_error"] for case in study["cases"]],
        "s-",
        label="max pointwise hoop stress",
    )
    axes[0].loglog(hs, 0.30 * hs, "k--", lw=1.0, label=r"slope 1")
    axes[0].loglog(hs, 0.30 * hs**2, "k:", lw=1.0, label=r"slope 2")
    axes[0].set_xlabel(r"$h = a/n$ (mesh size)")
    axes[0].set_ylabel("relative error")
    axes[0].set_title("Richardson convergence")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)

    kt = [case["hole_edge"]["hole_edge_at_90_degrees"] for case in study["cases"]]
    axes[1].plot(levels, kt, "o-", label="FE hole-edge $\\sigma_{\\theta\\theta}/T$")
    axes[1].axhline(3.0, color="tab:red", ls=":", label="$K_t=3$ exact")
    axes[1].set_xlabel("elements per edge direction")
    axes[1].set_ylabel("stress concentration")
    axes[1].set_title("Stress concentration at the hole edge")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    path = ASSETS / "kirsch_convergence.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    FIGURES.append(path)
    return {"hoop_profile": path.name, "hoop_field": path.name, "convergence": path.name}


def main() -> dict:
    study = convergence_study(levels=LEVELS, outer_radius=OUTER)
    final = solve_case(HOLE, OUTER, n_theta=LEVELS[-1], sample_points=((1.0, math.pi / 2),))
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tensorfem_version": __version__,
        "python": host.python_version(),
        "numpy": np.__version__,
        "reference": "Kirsch (1898); Timoshenko & Goodier, Theory of Elasticity, 3rd ed., sec. 39",
        "problem": {
            "hole_radius": HOLE.radius,
            "outer_radius": OUTER,
            "far_field_tension": HOLE.far_field,
            "young": HOLE.young,
            "poisson": HOLE.poisson,
            "plane": HOLE.plane,
            "thickness": HOLE.thickness,
            "grading": study["hole"]["grading"],
            "stress_concentration_exact": HOLE.stress_concentration,
        },
        "levels": study["levels"],
        "cases": [_serialisable(case) for case in study["cases"]],
        "observed_orders": study["observed_orders"],
        "max_stress_relative_error": study["max_stress_relative_error"],
        "max_displacement_l2": study["max_displacement_l2"],
        "exact_arc_resultant": arc_resultant(HOLE, OUTER),
        "hole_edge_nodal_value": {
            "theta_90_fe_closest_node": final["hole_edge"]["hole_edge_at_90_degrees"],
            "exact_limit": 3.0,
            "note": "node on r=a, theta=pi/2; nodal average of the adjacent element stresses",
        },
        "figures": _figures(study),
        "verdict": {
            "pointwise_stress_within_3pct": study["max_stress_relative_error"] < 0.03,
            "stress_concentration_within_1pct": abs(
                study["cases"][-1]["hole_edge"]["hole_edge_at_90_degrees"] - 3.0
            )
            / 3.0
            < 0.01,
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    for case in study["cases"]:
        edge = case["hole_edge"]
        print(
            f"n={case['n_theta']:3d} dofs={case['dofs']:5d} "
            f"uL2={case['displacement_error']['l2_relative']:.3e} "
            f"stress_max={case['max_stress_relative_error'] * 100:6.3f}% "
            f"Kt={edge['hole_edge_at_90_degrees']:.4f} "
            f"load_resid={case['load_resultant']['relative_error']:.2e} "
            f"recovery={case['recovery_cross_check']['max_relative_difference']:.1e}"
        )
    print("orders:", [
        (order["levels"], round(order["displacement_order"], 2), round(order["stress_order"], 2))
        for order in study["observed_orders"]
    ])
    print("verdict:", payload["verdict"])
    return payload


if __name__ == "__main__":
    main()