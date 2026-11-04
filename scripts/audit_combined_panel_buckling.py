"""Audit of combined in-plane buckling of simply supported marine panels.

Produces ``benchmarks/combined_panel_buckling.json`` (evidence) and the figures
under ``docs/assets/combined_panel_buckling/``.  Run with::

    OMP_NUM_THREADS=4 PYTHONPATH=src python3 scripts/audit_combined_panel_buckling.py

The script only exercises tensorfem kernels; no external solver is invoked.
"""
from __future__ import annotations

import json
import math
import pathlib

import torch

torch.set_default_dtype(torch.float64)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfem.marine_plate_buckling import (  # noqa: E402
    finite_difference_uniaxial_buckling,
    navier_uniaxial_buckling_load,
)
from tensorfem.marine_plate_combined_buckling import (  # noqa: E402
    classical_shear_buckling_coefficient,
    navier_biaxial_buckling_load,
    panel_buckling_load,
    run_combined_panel_buckling_qualification,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets" / "combined_panel_buckling"
EVIDENCE = ROOT / "benchmarks" / "combined_panel_buckling.json"

E, NU, T = 210.0e9, 0.3, 0.012
D = E * T**3 / (12.0 * (1.0 - NU**2))


def richardson(errors: list[float], ratio: float = 2.0) -> dict[str, float]:
    """Observed order and Richardson limit from the last three of ``errors``."""
    coarse, mid, fine = errors[-3:]
    order = math.log(coarse / mid) / math.log(ratio)
    limit_scale = fine - (mid - fine) / (ratio**order - 1.0)
    return {"observed_order": order, "extrapolated_error": limit_scale}


def biaxial_case(length: float, ratio: float, grids: tuple[int, ...]) -> dict:
    reference = navier_biaxial_buckling_load(
        length=length, width=1.0, rigidity_x=D, rigidity_y=D,
        coupling_rigidity=D, biaxial_ratio=ratio)
    rows = []
    for grid in grids:
        result = panel_buckling_load(
            length=length, width=1.0, thickness=T, rigidity_x=D, rigidity_y=D,
            coupling_rigidity=D, grid_x=grid, grid_y=grid,
            longitudinal_share=1.0, transverse_share=ratio)
        rows.append({
            "grid": list(result.grid),
            "line_load": result.critical_line_load,
            "relative_error": abs(result.critical_line_load / reference.critical_line_load - 1.0),
        })
    finite_difference_uniaxial_buckling(
        length=length, width=1.0, thickness=T, rigidity_x=D, rigidity_y=D,
        coupling_rigidity=D, grid_x=16, grid_y=16)
    return {
        "case": f"biaxial_ratio_{ratio}_aspect_{length:g}",
        "biaxial_ratio": ratio,
        "length": length,
        "width": 1.0,
        "reference_line_load": reference.critical_line_load,
        "reference_mode": [reference.longitudinal_half_waves, reference.transverse_half_waves],
        "grids": rows,
        "richardson": richardson([row["relative_error"] for row in rows]),
    }


def shear_case(aspect: float, grids: tuple[int, ...]) -> dict:
    """Pure shear with ``a/b = aspect``; ``k`` reported against the fit."""
    coefficient_fit = classical_shear_buckling_coefficient(aspect)
    reference = coefficient_fit * math.pi**2 * D
    rows = []
    for grid in grids:
        grid_y = int(round(grid * aspect))
        result = panel_buckling_load(
            length=aspect, width=1.0, thickness=T, rigidity_x=D, rigidity_y=D,
            coupling_rigidity=D, grid_x=grid, grid_y=grid_y,
            longitudinal_share=0.0, shear_share=1.0)
        rows.append({
            "grid": list(result.grid),
            "line_load": result.critical_line_load,
            "coefficient": result.critical_line_load / (math.pi**2 * D),
            "relative_error_vs_fit": abs(result.critical_line_load / reference - 1.0),
        })
    return {
        "case": f"pure_shear_aspect_{aspect:g}",
        "aspect_ratio": aspect,
        "coefficient_fit": coefficient_fit,
        "finest_coefficient": rows[-1]["coefficient"],
        "grids": rows,
        "richardson": richardson([row["relative_error_vs_fit"] for row in rows]),
    }


def interaction_curve(length: float, grids: int) -> dict:
    """Sweep axial+shear combinations and normalise by the pure critical loads."""
    axial_only = panel_buckling_load(
        length=length, width=1.0, thickness=T, rigidity_x=D, rigidity_y=D,
        coupling_rigidity=D, grid_x=grids, grid_y=grids,
        longitudinal_share=1.0).critical_line_load
    shear_only = panel_buckling_load(
        length=length, width=1.0, thickness=T, rigidity_x=D, rigidity_y=D,
        coupling_rigidity=D, grid_x=grids, grid_y=grids,
        longitudinal_share=0.0, shear_share=1.0).critical_line_load
    points = []
    for shear_share in (0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0, 1.4, 2.0, 4.0):
        result = panel_buckling_load(
            length=length, width=1.0, thickness=T, rigidity_x=D, rigidity_y=D,
            coupling_rigidity=D, grid_x=grids, grid_y=grids,
            longitudinal_share=1.0, shear_share=shear_share)
        axial = result.critical_line_load / axial_only
        shear = shear_share * result.critical_line_load / shear_only
        points.append({
            "shear_share": shear_share,
            "normalised_axial": axial,
            "normalised_shear": shear,
            "sum": axial + shear,
            "sum_of_squares": axial**2 + shear**2,
        })
    return {
        "case": f"interaction_aspect_{length:g}",
        "grid": [grids, grids],
        "axial_only_load": axial_only,
        "shear_only_load": shear_only,
        "points": points,
    }


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    qualification = run_combined_panel_buckling_qualification()

    biaxial = [
        biaxial_case(1.0, 1.0, (8, 16, 32, 48)),
        biaxial_case(1.5, 0.5, (8, 16, 32, 48)),
        biaxial_case(2.0, 0.5, (8, 16, 32, 48)),
        # A weak transverse share at a/b = 3 selects a multi-half-wave mode, so
        # the discretisation error no longer reduces to the closed form of the
        # fundamental mode.
        biaxial_case(3.0, 0.1, (8, 16, 32, 48)),
    ]
    # Grids are refined in the short direction ``b``; ``grid_y`` follows the
    # aspect ratio so the element is nearly square.  Aspect 3 stops at 32 nodes
    # across ``b`` (40 x 120 interior nodes would dominate the runtime).
    shear = [
        shear_case(1.0, (24, 32, 40)),
        shear_case(1.5, (24, 32, 40)),
        shear_case(2.0, (24, 32, 40)),
        shear_case(3.0, (16, 24, 32)),
    ]
    interaction = interaction_curve(2.0, 24)

    # Figure 1: grid convergence of the two branches.
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for case, marker in zip(biaxial, ("o", "s", "^", "v")):
        h = [1.0 / (row["grid"][0] + 1) for row in case["grids"]]
        axes[0].loglog(h, [row["relative_error"] for row in case["grids"]],
                       marker=marker, label=case["case"])
    axes[0].set_xlabel("h / a")
    axes[0].set_ylabel("relative error vs Navier")
    axes[0].set_title("Biaxial compression (exact reference)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    for case, marker in zip(shear, ("o", "s", "^", "v")):
        h = [1.0 / (row["grid"][0] + 1) for row in case["grids"]]
        axes[1].loglog(h, [row["relative_error_vs_fit"] for row in case["grids"]],
                       marker=marker, label=case["case"])
    axes[1].axhline(0.03, color="k", linestyle="--", linewidth=1, label="3 % gate")
    axes[1].set_xlabel("h / b")
    axes[1].set_ylabel("|k - k_fit| / k_fit")
    axes[1].set_title("Pure shear (Timoshenko fit anchor)")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "convergence.png", dpi=150)
    plt.close(fig)

    # Figure 2: interaction curve.
    points = interaction["points"]
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.plot([p["normalised_shear"] for p in points], [p["normalised_axial"] for p in points],
            "o-", label="FE interaction (a/b = 2)")
    grid_line = [0.0, 1.0]
    ax.plot(grid_line, grid_line, "k--", linewidth=1, label="linear: sum = 1")
    theta = [i * math.pi / 200 for i in range(101)]
    ax.plot([math.cos(t) for t in theta], [math.sin(t) for t in theta], "k:",
            linewidth=1, label="elliptic: sum of squares = 1")
    ax.axhline(1.0, color="0.6", linewidth=0.8)
    ax.axvline(1.0, color="0.6", linewidth=0.8)
    ax.set_xlabel(r"$N_{xy} / N_{xy,cr}$")
    ax.set_ylabel(r"$N_x / N_{x,cr}$")
    ax.set_title("Axial + shear interaction, simply supported panel")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "interaction.png", dpi=150)
    plt.close(fig)

    # Figure 3: pure shear mode shape on the square panel.
    mode = panel_buckling_load(
        length=1.0, width=1.0, thickness=T, rigidity_x=D, rigidity_y=D,
        coupling_rigidity=D, grid_x=32, grid_y=32,
        longitudinal_share=0.0, shear_share=1.0, keep_mode=True).mode_shape
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    image = ax.imshow(mode.T.numpy(), origin="lower", cmap="RdBu_r",
                      extent=[0.0, 1.0, 0.0, 1.0], aspect="equal")
    fig.colorbar(image, ax=ax, shrink=0.85, label=r"$w / w_{max}$")
    ax.set_xlabel("x / a")
    ax.set_ylabel("y / b")
    ax.set_title("Square panel, pure shear buckling mode")
    fig.tight_layout()
    fig.savefig(ASSETS / "shear_mode.png", dpi=150)
    plt.close(fig)

    payload = {
        "schema": "tensorfem.combined-panel-buckling-audit/1.0",
        "material": {"young": E, "poisson": NU, "thickness": T, "rigidity": D},
        "tolerance": 0.03,
        "qualification": qualification,
        "biaxial": biaxial,
        "shear": shear,
        "interaction": interaction,
        "uniaxial_reference": {
            "navier": navier_uniaxial_buckling_load(
                length=2.0, width=1.0, rigidity_x=D, rigidity_y=D,
                coupling_rigidity=D)[0],
            "checked_by": "tests/test_marine_plate_combined_buckling.py",
        },
        "figures": [
            "docs/assets/combined_panel_buckling/convergence.png",
            "docs/assets/combined_panel_buckling/interaction.png",
            "docs/assets/combined_panel_buckling/shear_mode.png",
        ],
    }
    EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n")
    worst_biaxial = max(row["relative_error"] for case in biaxial for row in case["grids"])
    worst_shear = max(row["relative_error_vs_fit"] for case in shear for row in case["grids"])
    print(f"qualification passed: {qualification['passed']}")
    print(f"worst biaxial error (all grids): {worst_biaxial:.6%}")
    print(f"worst shear error vs fit (all grids/aspects): {worst_shear:.6%}")
    for case in biaxial + shear:
        print(case["case"], "order", round(case["richardson"]["observed_order"], 2),
              "extrapolated error", f"{case['richardson']['extrapolated_error']:.3%}")
    print("evidence ->", EVIDENCE.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())