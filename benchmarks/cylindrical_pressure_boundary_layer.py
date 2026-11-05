"""Evidence generator: pressurised cylinder with clamped ends (bending boundary layer).

Runs the production metric-consistent cylindrical Shell4 element
(:mod:`tensorfem.cylindrical_shell4`) against the exact axisymmetric shell
solution of :mod:`tensorfem.cylindrical_pressure_reference`, writes
``benchmarks/cylindrical_pressure_boundary_layer_evidence.json`` and three
figures under ``docs/assets/``.

Usage::

    OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/cylindrical_pressure_boundary_layer.py
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.cylindrical_pressure_reference import (  # noqa: E402
    cylinder_parameters,
    run_boundary_layer_study,
    self_validation,
    solve_pressure_cylinder,
    work_energy_balance,
)

CASE = dict(
    radius=1.5,
    thickness=0.015,
    young=206.0e9,
    poisson=0.3,
    pressure=1.0e6,
    length=1.2,
)
FACETS_PRODUCTION = 16
AXES_PRODUCTION = 32


def _row(result) -> dict:
    p = result.parameters
    return {
        "axes": result.axes,
        "facets": result.facets,
        "dofs": result.dofs,
        "axial_spacing_times_beta": p.length / result.axes * p.beta,
        "displacement_l2_error": result.displacement_l2_error,
        "peak_displacement_error": result.peak_displacement_error,
        "edge_displacement_error": abs(
            float(result.displacement_fe[1]) / float(result.displacement_exact[1]) - 1.0
        ),
        "plateau_displacement_error": abs(
            float(result.displacement_fe[len(result.x) // 2])
            / float(result.displacement_exact[len(result.x) // 2])
            - 1.0
        ),
        "moment_l2_error": result.moment_l2_error,
        "edge_moment_error": result.edge_moment_error,
        "moment_sign_correlation": result.moment_sign_correlation,
        "axial_force_over_hoop": result.axial_force_error,
        "circumferential_uniformity": result.circumferential_uniformity,
    }


def _orders(rows, key, counts) -> list[float]:
    out = []
    for i in range(len(rows) - 1):
        a, b = rows[i][key], rows[i + 1][key]
        if a > 0 and b > 0 and counts[i + 1] != counts[i]:
            out.append(math.log(a / b) / math.log(counts[i + 1] / counts[i]))
    return out


def main() -> int:
    p = cylinder_parameters(**CASE)
    production = solve_pressure_cylinder(AXES_PRODUCTION, FACETS_PRODUCTION, p)

    facets_rows = [
        _row(solve_pressure_cylinder(16, facets, p)) for facets in (8, 12, 16, 24)
    ]
    axial_counts = (8, 16, 32, 48)
    axial_rows = [
        _row(solve_pressure_cylinder(n, FACETS_PRODUCTION, p)) for n in axial_counts
    ]
    repeat = solve_pressure_cylinder(AXES_PRODUCTION, FACETS_PRODUCTION, p)

    payload = {
        "case": CASE,
        "derived": {
            "beta": p.beta,
            "decay_length": 1.0 / p.beta,
            "boundary_layer_thickness": p.boundary_layer_thickness,
            "beta_length": p.beta * p.length,
            "bending_rigidity": p.bending_rigidity,
            "membrane_displacement": p.membrane_radial_displacement,
            "membrane_hoop_force": p.membrane_hoop_force,
            "membrane_hoop_stress": p.membrane_hoop_stress,
            "edge_moment_long_cylinder": p.edge_moment_long_cylinder,
            "edge_bending_stress": p.edge_bending_stress,
        },
        "self_validation": self_validation(p),
        "work_energy": work_energy_balance(p),
        "reference_study": run_boundary_layer_study(
            p, axis_counts=(8, 16, 32), facets=FACETS_PRODUCTION
        ),
        "facets_study_at_16_axes": facets_rows,
        "axial_study_at_16_facets": axial_rows,
        "observed_orders_axial": {
            key: _orders(axial_rows, key, axial_counts)
            for key in (
                "displacement_l2_error",
                "peak_displacement_error",
                "plateau_displacement_error",
                "moment_l2_error",
            )
        },
        "reproducibility": {
            "production_displacement_l2_error": production.displacement_l2_error,
            "repeat_displacement_l2_error": repeat.displacement_l2_error,
            "bitwise_identical": bool(
                torch.equal(production.displacement_fe, repeat.displacement_fe)
            ),
        },
        "gates": {
            "displacement_l2_below_3pct": production.displacement_l2_error < 0.03,
            "moment_l2_below_3pct": production.moment_l2_error < 0.03,
            "facets_8_displacement_rejected": facets_rows[0]["displacement_l2_error"]
            > 0.03,
        },
        "environment": {
            "torch": torch.__version__,
            "threads": os.environ.get("OMP_NUM_THREADS", "default"),
        },
    }

    out = ROOT / "benchmarks" / "cylindrical_pressure_boundary_layer_evidence.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")

    _figures(p, production, axial_rows, axial_counts, facets_rows)
    print(json.dumps(payload["gates"], indent=1))
    print(json.dumps(axial_rows, indent=1))
    print(json.dumps(facets_rows, indent=1))
    return 0


def _figures(p, production, axial_rows, axial_counts, facets_rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    assets = ROOT / "docs" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    wp = p.membrane_radial_displacement

    x = production.x.detach().numpy() * p.beta
    w_fe = production.displacement_fe.detach().numpy() / wp
    w_ex = production.displacement_exact.detach().numpy() / wp
    m_fe = production.moment_fe.detach().numpy() / p.edge_moment_long_cylinder
    m_ex = production.moment_exact.detach().numpy() / p.edge_moment_long_cylinder

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].plot(x, w_ex, "-", color="black", lw=1.6, label="exact (4-constant shell ODE)")
    axes[0].plot(x, w_fe, "o", ms=3.5, color="tab:blue", label="Shell4 FE")
    axes[0].set_xlabel(r"$\beta x$ [-]")
    axes[0].set_ylabel(r"$w / w_p$ [-]")
    axes[0].set_title("Radial displacement: clamped-edge boundary layer")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].plot(x, m_ex, "-", color="black", lw=1.6, label="exact")
    axes[1].plot(x, m_fe, "o", ms=3.5, color="tab:red", label="Shell4 FE")
    axes[1].set_xlabel(r"$\beta x$ [-]")
    axes[1].set_ylabel(r"$M_x / (p/2\beta^2)$ [-]")
    axes[1].set_title("Axial bending moment")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(assets / "cylindrical_pressure_boundary_layer_profiles.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    counts = np.array(axial_counts, dtype=float)
    for key, label, style in (
        ("displacement_l2_error", r"$w$ L2", "o-"),
        ("plateau_displacement_error", "plateau $w$", "s-"),
        ("moment_l2_error", r"$M_x$ L2", "^-"),
    ):
        values = np.array([max(row[key], 1e-12) for row in axial_rows])
        ax.loglog(counts, 100 * values, style, label=label)
    facets_counts = np.array([row["facets"] for row in facets_rows], dtype=float)
    ax.loglog(
        facets_counts,
        100 * np.array([row["displacement_l2_error"] for row in facets_rows]),
        "d--",
        color="tab:green",
        label=r"$w$ L2 vs facets (16 axes)",
    )
    ax.axhline(3.0, color="black", ls=":", lw=1.2, label="3 % gate")
    ax.set_xlabel("refinement level (elements)")
    ax.set_ylabel("relative error [%]")
    ax.set_title("Convergence of the pressurised-cylinder case")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(assets / "cylindrical_pressure_boundary_layer_convergence.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())