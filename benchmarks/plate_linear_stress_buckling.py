"""Evidence script: buckling of simply supported panels under linear edge stress.

Produces, programmatically (no hand-copied numbers)

1. ``stress_ratio_sweep`` - converged coefficient ``k`` over the degree of
   variation ``gradient`` in [0, 2] (stress ratio ``psi = 1 - gradient``) for
   several aspect ratios, against the classical anchors ``k = 4.0`` (uniform),
   ``7.81`` (triangular) and ``23.9`` / ``25.6`` (pure bending);
2. ``basis_convergence`` - the variational eigenvalue under basis refinement
   (upper bound, monotone decrease);
3. ``strip_convergence`` - the independent finite-difference strip route against
   the Ritz value, showing second-order convergence;
4. ``closed_form_vs_quadrature`` - the closed-form energy matrices against
   Gauss-Legendre quadrature of the same functionals;
5. the qualification report of ``run_linear_stress_qualification``.

Outputs (relative to the repository root)
----------------------------------------
* ``benchmarks/plate_linear_stress_buckling_evidence.json`` - raw numbers
* ``docs/assets/plate_linear_stress_buckling.png`` - sweep, basis and strip convergence

Usage::

    OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/plate_linear_stress_buckling.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorfem.marine_plate_buckling import isotropic_plate_rigidities  # noqa: E402
from tensorfem.plate_linear_stress_buckling import (  # noqa: E402
    LITERATURE_K_PURE_BENDING,
    LITERATURE_K_PURE_BENDING_SQUARE,
    LITERATURE_K_TRIANGULAR,
    LITERATURE_K_UNIFORM,
    ritz_linear_stress_buckling,
    run_linear_stress_qualification,
    sine_stress_matrices,
    stress_energy_quadrature,
    strip_buckling_coefficient,
)

ROOT = Path(__file__).resolve().parents[1]
E, NU, T = 210.0e9, 0.3, 0.012
ASPECTS = (1.0, 2.0, 4.0, 6.0)
GRADIENTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)


def _ritz(aspect: float, gradient: float, basis: int = 16):
    d, dy, h = isotropic_plate_rigidities(E, NU, T)
    return ritz_linear_stress_buckling(
        length=aspect,
        width=1.0,
        rigidity_x=d,
        rigidity_y=dy,
        coupling_rigidity=h,
        stress_gradient=gradient,
        basis_x=basis,
        basis_y=basis,
    )


def main() -> dict:
    d, _, _ = isotropic_plate_rigidities(E, NU, T)
    sweep = []
    for aspect in ASPECTS:
        for gradient in GRADIENTS:
            mode = _ritz(aspect, gradient)
            sweep.append(
                {
                    "aspect_ratio": aspect,
                    "stress_gradient": gradient,
                    "psi": mode.psi,
                    "k": mode.coefficient,
                    "critical_edge_load_n_per_m": mode.multiplier,
                    "critical_edge_stress_pa": mode.multiplier / T,
                    "dominant_mode": list(mode.dominant_mode),
                }
            )

    convergence = []
    for basis in (2, 4, 6, 8, 10, 12, 16, 20):
        mode = _ritz(2.0, 1.0, basis=basis)
        convergence.append({"basis": basis, "k": mode.coefficient})

    strip_convergence = []
    for nodes in (25, 50, 100, 200, 400):
        strip, strip_m = strip_buckling_coefficient(
            length=2.0, width=1.0, rigidity_x=d, rigidity_y=d,
            coupling_rigidity=isotropic_plate_rigidities(E, NU, T)[2],
            stress_gradient=1.0, nodes=nodes,
        )
        reference = _ritz(2.0, 1.0).coefficient
        strip_convergence.append(
            {
                "nodes": nodes,
                "strip_k": strip,
                "ritz_k": reference,
                "relative_difference": abs(strip / reference - 1.0),
                "dominant_half_wave": strip_m,
            }
        )

    quadrature_errors = {}
    for gradient in (0.0, 1.0, 2.0):
        kb, kg, _ = sine_stress_matrices(
            length=2.0, width=1.0, rigidity_x=d, rigidity_y=d,
            coupling_rigidity=isotropic_plate_rigidities(E, NU, T)[2],
            stress_gradient=gradient, basis_x=10, basis_y=10,
        )
        qb, qg, _ = stress_energy_quadrature(
            length=2.0, width=1.0, rigidity_x=d, rigidity_y=d,
            coupling_rigidity=isotropic_plate_rigidities(E, NU, T)[2],
            stress_gradient=gradient, basis_x=10, basis_y=10,
        )
        quadrature_errors[f"{gradient:.2f}"] = {
            "bending": float(np.abs(kb - qb).max() / np.abs(kb).max()),
            "geometric": float(np.abs(kg - qg).max() / np.abs(kg).max()),
        }

    qualification = run_linear_stress_qualification(basis=16)

    evidence = {
        "schema": "tensorfem.plate-linear-stress-buckling-evidence/1.0",
        "parameters": {"young_pa": E, "poisson": NU, "thickness_m": T},
        "literature": {
            "k_uniform": LITERATURE_K_UNIFORM,
            "k_triangular": LITERATURE_K_TRIANGULAR,
            "k_pure_bending_long": LITERATURE_K_PURE_BENDING,
            "k_pure_bending_square": LITERATURE_K_PURE_BENDING_SQUARE,
        },
        "stress_ratio_sweep": sweep,
        "basis_convergence": convergence,
        "strip_convergence": strip_convergence,
        "closed_form_vs_quadrature": quadrature_errors,
        "qualification": qualification,
    }

    figure_path = ROOT / "docs" / "assets" / "plate_linear_stress_buckling.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    colours = {1.0: "#1f77b4", 2.0: "#2ca02c", 4.0: "#ff7f0e", 6.0: "#9467bd"}
    for aspect in ASPECTS:
        rows = [row for row in sweep if row["aspect_ratio"] == aspect]
        axes[0].plot(
            [row["psi"] for row in rows],
            [row["k"] for row in rows],
            "o-",
            ms=3.5,
            color=colours[aspect],
            label=f"a/b = {aspect:g}",
        )
    axes[0].axhline(LITERATURE_K_UNIFORM, ls="--", lw=0.8, color="0.3")
    axes[0].axhline(LITERATURE_K_TRIANGULAR, ls="--", lw=0.8, color="0.3")
    axes[0].axhline(LITERATURE_K_PURE_BENDING, ls=":", lw=0.9, color="#d62728")
    axes[0].axhline(LITERATURE_K_PURE_BENDING_SQUARE, ls=":", lw=0.9, color="#d62728")
    axes[0].annotate("4.0 uniform", (0.05, LITERATURE_K_UNIFORM + 1.0), fontsize=7)
    axes[0].annotate("7.81 triangular", (-0.98, LITERATURE_K_TRIANGULAR + 1.0), fontsize=7)
    axes[0].annotate("23.9 long panel", (-0.98, LITERATURE_K_PURE_BENDING + 1.0), fontsize=7)
    axes[0].annotate("25.6 a/b=1", (0.05, LITERATURE_K_PURE_BENDING_SQUARE + 1.0), fontsize=7)
    axes[0].set_xlabel(r"edge stress ratio $\psi=\sigma_2/\sigma_1$")
    axes[0].set_ylabel(r"$k = N_{0,cr} b^2/(\pi^2 D)$")
    axes[0].set_title("linearly varying compression")
    axes[0].invert_xaxis()
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)

    axes[1].semilogy(
        [row["basis"] for row in convergence],
        [abs(row["k"] / convergence[-1]["k"] - 1.0) for row in convergence],
        "o-",
        color="#1f77b4",
    )
    axes[1].set_xlabel("basis order (x = y = n)")
    axes[1].set_ylabel("relative deviation from basis 20")
    axes[1].set_title(r"variational convergence, $a/b=2$, $\psi=0$")
    axes[1].grid(alpha=0.3, which="both")

    axes[2].loglog(
        [row["nodes"] for row in strip_convergence],
        [row["relative_difference"] for row in strip_convergence],
        "s-",
        color="#2ca02c",
        label="FD strip vs Ritz",
    )
    ref_x = np.array([25.0, 400.0])
    axes[2].loglog(ref_x, 6.0e-4 * (ref_x / 25.0) ** -2.0, "k--", lw=0.8, label="slope -2")
    axes[2].set_xlabel("finite-difference nodes across b")
    axes[2].set_ylabel("relative difference")
    axes[2].set_title("independent strip route")
    axes[2].grid(alpha=0.3, which="both")
    axes[2].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    evidence["figure"] = str(figure_path.relative_to(ROOT))
    out = ROOT / "benchmarks" / "plate_linear_stress_buckling_evidence.json"
    out.write_text(json.dumps(evidence, indent=2) + "\n")

    markdown = ["| psi | gradient | " + " | ".join(f"a/b={a:g}" for a in ASPECTS) + " |"]
    markdown.append("|---|---:|" + "---:|" * len(ASPECTS))
    for gradient in GRADIENTS:
        cells = []
        for aspect in ASPECTS:
            row = next(
                r for r in sweep
                if r["aspect_ratio"] == aspect and r["stress_gradient"] == gradient
            )
            cells.append(f"{row['k']:.4f}")
        markdown.append(
            f"| {1.0 - gradient:.2f} | {gradient:.2f} | " + " | ".join(cells) + " |"
        )

    summary = {
        "qualification_passed": qualification["passed"],
        "literature_errors_percent": {
            row["case"]: round(100.0 * row["relative_error"], 4)
            for row in qualification["cases"]
        },
        "max_strip_vs_ritz_percent": round(
            100.0 * max(row["ritz_vs_strip_relative"] for row in qualification["cases"]), 5
        ),
        "max_quadrature_error": max(
            max(values.values()) for values in quadrature_errors.values()
        ),
        "strip_convergence_ratio_100_to_200": round(
            strip_convergence[2]["relative_difference"]
            / strip_convergence[3]["relative_difference"],
            3,
        ),
        "evidence": str(out.relative_to(ROOT)),
        "figure": evidence["figure"],
    }
    print(json.dumps(summary, indent=2))
    print("\n".join(markdown))
    return evidence


if __name__ == "__main__":
    main()