"""Evidence script: buckling of a simply supported panel under a linear load ramp.

Closes the "prebuckling stress gradient" verification debt recorded in
``docs/benchmark-qualification-status.md``.  Produces, programmatically (no
hand-copied numbers)

1. ``gradient_sweep`` - converged coefficient ``k`` for edge stress ratios
   ``psi`` in ``[-1, 1]`` at ``a/b = 1``, with the tabulated anchors;
2. ``aspect_sweep`` - ``k`` against ``a/b`` for ``psi`` in ``{1, 0, -1}``,
   showing that the tabulated ``23.9`` of pure bending is the minimum over
   aspect ratio, not the square-panel value;
3. ``basis_convergence`` - the variational eigenvalue under basis refinement
   (monotone from above, i.e. the Rayleigh quotient cannot overstate ``k``);
4. ``mode_field`` - the critical mode of the pure-bending panel and its Rayleigh
   quotient re-evaluated from the assembled matrices;
5. the four-leg qualification report of ``run_stress_gradient_qualification``.

Outputs (relative to the repository root)
----------------------------------------
* ``benchmarks/plate_stress_gradient_buckling_evidence.json`` - raw numbers
* ``docs/assets/plate_stress_gradient/coefficient_curves.png`` - sweep + aspect
* ``docs/assets/plate_stress_gradient/pure_bending_mode.png`` - mode contour

Usage::

    OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/plate_stress_gradient_buckling.py
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
from tensorfem.plate_stress_gradient_buckling import (  # noqa: E402
    LITERATURE_GRADIENT_COEFFICIENTS,
    REFERENCE_GRADIENT_SWEEP,
    gradient_buckling_mode_field,
    reference_gradient_sweep,
    ritz_gradient_buckling_load,
    run_stress_gradient_qualification,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets" / "plate_stress_gradient"
E, NU, T = 210.0e9, 0.3, 0.012
BASIS = 24


def build_evidence() -> dict[str, object]:
    d, _, h = isotropic_plate_rigidities(E, NU, T)

    def coefficient(psi: float, aspect: float, basis: int = BASIS) -> float:
        return ritz_gradient_buckling_load(
            length=aspect, width=1.0, rigidity_x=d, rigidity_y=d,
            coupling_rigidity=h, stress_gradient=psi,
            basis_x=basis, basis_y=basis,
        ).coefficient

    gradient_sweep = reference_gradient_sweep(
        young=E, poisson=NU, thickness=T, basis=BASIS,
    )
    literature = [
        {
            "stress_gradient": psi,
            "aspect_ratio": aspect,
            "literature_k": value,
            "coefficient_k": coefficient(psi, aspect),
            "relative_error": abs(coefficient(psi, aspect) / value - 1.0),
        }
        for (psi, aspect), value in LITERATURE_GRADIENT_COEFFICIENTS.items()
    ]

    aspect_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0]
    aspect_sweep = [
        {
            "aspect_ratio": aspect,
            "psi_plus_1": coefficient(1.0, aspect),
            "psi_zero": coefficient(0.0, aspect),
            "psi_minus_1": coefficient(-1.0, aspect),
        }
        for aspect in aspect_values
    ]

    convergence = {}
    for label, psi, aspect in (("triangular", 0.0, 1.0), ("pure_bending", -1.0, 2.0)):
        convergence[label] = {
            "stress_gradient": psi,
            "aspect_ratio": aspect,
            "series": [
                {"basis": order, "coefficient_k": coefficient(psi, aspect, order)}
                for order in (4, 6, 8, 10, 12, 16, 20, 24)
            ],
        }

    mode = gradient_buckling_mode_field(
        length=2.0, width=1.0, rigidity_x=d, rigidity_y=d, coupling_rigidity=h,
        stress_gradient=-1.0, basis_x=BASIS, basis_y=BASIS, samples=81,
    )

    qualification = run_stress_gradient_qualification(
        young=E, poisson=NU, thickness=T, basis=BASIS,
    )

    pure_bending_min = min(row["psi_minus_1"] for row in aspect_sweep)
    return {
        "schema": "tensorfem.plate-stress-gradient-buckling-evidence/1.0",
        "parameters": {
            "young_pa": E,
            "poisson": NU,
            "thickness_m": T,
            "basis": BASIS,
            "rigidity_d_nm": d,
            "coupling_rigidity_h_nm": h,
        },
        "literature_anchors": literature,
        "gradient_sweep": gradient_sweep,
        "aspect_sweep": aspect_sweep,
        "pure_bending_minimum_k": pure_bending_min,
        "pure_bending_minimum_aspect": [
            row["aspect_ratio"] for row in aspect_sweep
            if row["psi_minus_1"] == pure_bending_min
        ],
        "basis_convergence": convergence,
        "mode_field": {
            "stress_gradient": mode["stress_gradient"],
            "aspect_ratio": 2.0,
            "critical_load": mode["critical_load"],
            "coefficient_k": mode["coefficient_k"],
            "dominant_mode": mode["dominant_mode"],
            "energy_ratio_load": mode["energy_ratio_load"],
            "energy_ratio_relative_deviation": mode["energy_ratio_relative_deviation"],
        },
        "qualification": qualification,
    }


def render_figures(evidence: dict[str, object]) -> dict[str, str]:
    ASSETS.mkdir(parents=True, exist_ok=True)
    d, _, h = isotropic_plate_rigidities(E, NU, T)

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.4, 4.3))
    sweep = evidence["gradient_sweep"]
    left.plot([row["stress_gradient"] for row in sweep],
              [row["coefficient_k"] for row in sweep],
              "o-", color="#1f77b4", label="TensorFEM sine Galerkin (basis %d)" % BASIS)
    anchors = evidence["literature_anchors"]
    square_anchors = [row for row in anchors if row["aspect_ratio"] == 1.0]
    left.plot([row["stress_gradient"] for row in square_anchors],
              [row["literature_k"] for row in square_anchors],
              "s", mfc="none", color="#d62728", label="tabulated (a/b = 1)")
    left.annotate("k = 4.00 (uniform)", (-0.02, 4.0), fontsize=8,
                  xytext=(0.08, 5.2), textcoords="data")
    left.annotate("k = 7.81 (triangular)", (0.02, 7.81), fontsize=8,
                  xytext=(0.10, 8.9), textcoords="data")
    left.set_xlabel("edge stress ratio $\\psi = N_2/N_1$")
    left.set_ylabel("$k = N_{cr} b^2/(\\pi^2 D)$")
    left.set_title("load ramp: coefficient vs stress gradient")
    left.grid(alpha=0.3)
    left.legend(fontsize=8)

    aspect = evidence["aspect_sweep"]
    ratios = [row["aspect_ratio"] for row in aspect]
    right.plot(ratios, [row["psi_plus_1"] for row in aspect], "o-", lw=1.2,
               color="#1f77b4", label="$\\psi=+1$ uniform")
    right.plot(ratios, [row["psi_zero"] for row in aspect], "s-", lw=1.2,
               color="#2ca02c", label="$\\psi=0$ triangular")
    right.plot(ratios, [row["psi_minus_1"] for row in aspect], "^-", lw=1.2,
               color="#ff7f0e", label="$\\psi=-1$ pure bending")
    right.axhline(23.9, ls="--", lw=0.9, color="#d62728")
    right.annotate("tabulated minimum 23.9", (2.1, 23.95), fontsize=8, color="#d62728")
    right.set_xlabel("aspect ratio a/b")
    right.set_ylabel("$k$")
    right.set_title("coefficient vs aspect ratio (three load ramps)")
    right.grid(alpha=0.3)
    right.legend(fontsize=8)
    figure.tight_layout()
    curves_path = ASSETS / "coefficient_curves.png"
    figure.savefig(curves_path, dpi=160)
    plt.close(figure)

    mode = gradient_buckling_mode_field(
        length=2.0, width=1.0, rigidity_x=d, rigidity_y=d, coupling_rigidity=h,
        stress_gradient=-1.0, basis_x=BASIS, basis_y=BASIS, samples=81,
    )
    figure, axis = plt.subplots(figsize=(6.4, 3.4))
    contour = axis.contourf(mode["x"], mode["y"], mode["shape"].T, levels=20,
                            cmap="RdBu_r")
    figure.colorbar(contour, ax=axis, label="normalised $w$")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]  (compressed edge at y = 0)")
    axis.set_title("critical mode, pure in-plane bending, a/b = 2 (\(k = %.3f\))"
                   % mode["coefficient_k"], fontsize=9)
    axis.set_aspect("equal")
    figure.tight_layout()
    mode_path = ASSETS / "pure_bending_mode.png"
    figure.savefig(mode_path, dpi=160)
    plt.close(figure)
    return {
        "coefficient_curves": str(curves_path.relative_to(ROOT)),
        "pure_bending_mode": str(mode_path.relative_to(ROOT)),
    }


def main() -> dict[str, object]:
    evidence = build_evidence()
    evidence["figures"] = render_figures(evidence)
    out = ROOT / "benchmarks" / "plate_stress_gradient_buckling_evidence.json"
    out.write_text(json.dumps(evidence, indent=2, default=float) + "\n")
    worst = max(row["relative_error"] for row in evidence["literature_anchors"])
    print(json.dumps({
        "worst_literature_error": worst,
        "pure_bending_minimum_k": evidence["pure_bending_minimum_k"],
        "qualification_passed": evidence["qualification"]["passed"],
        "mode_energy_ratio_deviation": evidence["mode_field"]["energy_ratio_relative_deviation"],
        "evidence": str(out.relative_to(ROOT)),
        "figures": evidence["figures"],
    }, indent=2, default=float))
    return evidence


if __name__ == "__main__":
    main()