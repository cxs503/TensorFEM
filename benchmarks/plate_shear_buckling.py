"""Evidence script: shear / combined in-plane buckling of simply supported panels.

Produces, programmatically (no hand-copied numbers)

1. ``aspect_sweep`` - converged shear coefficient ``k_s`` over ``a/b`` in
   [0.5, 10] against the classical ``k_s = 5.34 + 4/(a/b)^2`` and against the
   two literature anchors (9.34 at ``a/b = 1``, 5.34 asymptote);
2. ``basis_convergence`` - the variational eigenvalue under basis refinement,
   showing monotone convergence from above;
3. ``interaction`` - combined compression + shear against the classical
   interaction ``N_x/N_x,cr + (N_xy/N_xy,cr)^2 = 1``;
4. ``closed_form_vs_quadrature`` - the closed-form energy matrices against
   Gauss-Legendre quadrature of the same functionals;
5. the qualification report of ``run_shear_buckling_qualification``.

Outputs (relative to the repository root)
----------------------------------------
* ``benchmarks/plate_shear_buckling_evidence.json`` - raw numbers
* ``docs/assets/plate_shear_buckling.png`` - aspect sweep + interaction curve

Usage::

    OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/plate_shear_buckling.py
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
from tensorfem.plate_shear_buckling import (  # noqa: E402
    LITERATURE_KS_LONG_PANEL,
    LITERATURE_KS_SQUARE,
    classical_shear_coefficient,
    ritz_inplane_buckling_load,
    run_shear_buckling_qualification,
    shear_buckling_coefficient,
    sine_energy_matrices,
    sine_energy_quadrature,
)

ROOT = Path(__file__).resolve().parents[1]
E, NU, T = 210.0e9, 0.3, 0.012


def _coefficient(aspect: float, basis: int = 18) -> dict:
    d, dy, h = isotropic_plate_rigidities(E, NU, T)
    mode = ritz_inplane_buckling_load(
        length=aspect, width=1.0, rigidity_x=d, rigidity_y=dy,
        coupling_rigidity=h, basis_x=basis, basis_y=basis)
    coefficient = shear_buckling_coefficient(
        shear_load=mode.multiplier, rigidity=d, width=1.0)
    classical = classical_shear_coefficient(aspect)
    return {
        "aspect_ratio": aspect,
        "basis": basis,
        "critical_shear_load": mode.multiplier,
        "critical_shear_stress": mode.multiplier / T,
        "coefficient_k_s": coefficient,
        "classical_k_s": classical,
        "relative_error_vs_classical": abs(coefficient / classical - 1.0),
        "dominant_mode": list(mode.dominant_mode),
    }


def main() -> dict:
    aspects = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 10.0]
    sweep = [_coefficient(aspect) for aspect in aspects]

    d, dy, h = isotropic_plate_rigidities(E, NU, T)
    convergence = []
    for basis in (4, 6, 8, 10, 12, 16, 20):
        mode = ritz_inplane_buckling_load(
            length=1.0, width=1.0, rigidity_x=d, rigidity_y=dy,
            coupling_rigidity=h, basis_x=basis, basis_y=basis)
        convergence.append({
            "basis": basis,
            "coefficient_k_s": shear_buckling_coefficient(
                shear_load=mode.multiplier, rigidity=d, width=1.0),
        })
    converged = convergence[-1]["coefficient_k_s"]

    pure_shear = ritz_inplane_buckling_load(
        length=1.0, width=1.0, rigidity_x=d, rigidity_y=dy,
        coupling_rigidity=h, basis_x=16, basis_y=16)
    uniaxial = ritz_inplane_buckling_load(
        length=1.0, width=1.0, rigidity_x=d, rigidity_y=dy,
        coupling_rigidity=h, basis_x=16, basis_y=16, compression_ratio=0.0)
    interaction = []
    for ratio in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        mode = ritz_inplane_buckling_load(
            length=1.0, width=1.0, rigidity_x=d, rigidity_y=dy,
            coupling_rigidity=h, basis_x=16, basis_y=16,
            compression_ratio=ratio)
        axial = mode.compression_x / uniaxial.multiplier
        shear = mode.shear / pure_shear.multiplier
        interaction.append({
            "shear_to_axial_ratio": ratio,
            "axial_utilisation": axial,
            "shear_utilisation": shear,
            "interaction_residual": axial + shear**2 - 1.0,
            "classical_shear_utilisation": math.sqrt(max(0.0, 1.0 - axial)),
        })

    closed = sine_energy_matrices(length=1.0, width=1.0, rigidity_x=d,
                                  rigidity_y=dy, coupling_rigidity=h,
                                  basis_x=4, basis_y=4)
    quadrature = sine_energy_quadrature(length=1.0, width=1.0, rigidity_x=d,
                                        rigidity_y=dy, coupling_rigidity=h,
                                        basis_x=4, basis_y=4)
    quadrature_error = {
        "bending": float(np.abs(closed[0] - quadrature[0]).max()
                         / np.abs(closed[0]).max()),
        "uniaxial": float(np.abs(closed[1] - quadrature[1]).max()
                          / np.abs(closed[1]).max()),
        "shear": float(np.abs(closed[2] - quadrature[2]).max()
                       / np.abs(closed[2]).max()),
    }

    qualification = run_shear_buckling_qualification(basis=16)

    evidence = {
        "schema": "tensorfem.plate-shear-buckling-evidence/1.0",
        "parameters": {"young_pa": E, "poisson": NU, "thickness_m": T},
        "literature": {"k_s_square": LITERATURE_KS_SQUARE,
                       "k_s_long_panel": LITERATURE_KS_LONG_PANEL},
        "aspect_sweep": sweep,
        "basis_convergence": convergence,
        "converged_square_k_s": converged,
        "square_literature_relative_error": abs(converged / LITERATURE_KS_SQUARE - 1.0),
        "interaction": interaction,
        "closed_form_vs_quadrature": quadrature_error,
        "qualification": qualification,
    }

    figure_path = ROOT / "docs" / "assets" / "plate_shear_buckling.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    aspect_values = np.linspace(0.5, 10.0, 400)
    left.plot(aspect_values, [classical_shear_coefficient(a) for a in aspect_values],
              "-", color="0.25", label="classical 5.34 + 4/(a/b)$^2$")
    left.plot([row["aspect_ratio"] for row in sweep],
              [row["coefficient_k_s"] for row in sweep],
              "o", color="#1f77b4", label="TensorFEM sine Galerkin")
    left.axhline(LITERATURE_KS_SQUARE, ls="--", lw=0.8, color="#d62728")
    left.axhline(LITERATURE_KS_LONG_PANEL, ls=":", lw=0.8, color="#d62728")
    left.annotate("9.34 (a/b=1)", (6.4, LITERATURE_KS_SQUARE + 0.15), fontsize=8)
    left.annotate("5.34 (long panel)", (6.4, LITERATURE_KS_LONG_PANEL + 0.15), fontsize=8)
    left.set_xlabel("aspect ratio a/b")
    left.set_ylabel("$k_s = N_{xy,cr} b^2/(\\pi^2 D)$")
    left.set_title("pure shear: converged coefficient")
    left.grid(alpha=0.3)
    left.legend(fontsize=8)

    axial = [row["axial_utilisation"] for row in interaction]
    shear = [row["shear_utilisation"] for row in interaction]
    right.plot(axial, shear, "o-", color="#1f77b4", label="TensorFEM combined load")
    curve = np.linspace(0.0, 1.0, 200)
    right.plot(curve, np.sqrt(np.clip(1.0 - curve, 0.0, None)), "--", color="0.25",
               label="classical interaction")
    right.set_xlabel("$N_x / N_{x,cr}$")
    right.set_ylabel("$N_{xy} / N_{xy,cr}$")
    right.set_title("compression + shear interaction (a/b = 1)")
    right.grid(alpha=0.3)
    right.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    evidence["figure"] = str(figure_path.relative_to(ROOT))
    out = ROOT / "benchmarks" / "plate_shear_buckling_evidence.json"
    out.write_text(json.dumps(evidence, indent=2) + "\n")

    print(json.dumps({
        "converged_square_k_s": converged,
        "literature_relative_error": evidence["square_literature_relative_error"],
        "max_sweep_error_vs_classical": max(
            row["relative_error_vs_classical"] for row in sweep),
        "quadrature_error": quadrature_error,
        "interaction_max_residual": max(
            abs(row["interaction_residual"]) for row in interaction),
        "qualification_passed": qualification["passed"],
        "evidence": str(out.relative_to(ROOT)),
        "figure": evidence["figure"],
    }, indent=2))
    return evidence


if __name__ == "__main__":
    main()