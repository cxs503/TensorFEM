"""Cross-implementation evidence: two independent discretisations, one answer.

The repository carries two unrelated routes to the critical load of a simply
supported plate under a *linearly varying* in-plane load:

==============================================  ==============================
``tensorfem.plate_stress_gradient_buckling``    double-sine Galerkin, geometric
                                                matrix from the **closed-form**
                                                ramp integral (this branch)
``tensorfem.plate_bending_buckling``            a differently parameterised
                                                Galerkin assembly **plus** a
                                                strong-form finite-difference
                                                solve on a nodal grid (merged
                                                as m3255)
==============================================  ==============================

Neither route imports the other, and the finite-difference path shares no basis,
no energy matrix and no integral with the variational path -- it discretises
``D lap^4 w + N_x(y) w,xx = 0`` in strong form.  Agreement therefore tests both
implementations against a third opinion rather than against themselves, which is
exactly the failure mode a single-formulation verification cannot detect.

The comparison is a *consistency* leg, not a literature leg: the tabulated
anchors (4.00 / 7.81 / 23.9) live in the four-leg qualification of the module
itself.  Here the two code paths are compared against each other at edge stress
ratios with no tabulated value at all (``psi = +-0.5``), where a shared bug in
the load ramp is the realistic risk.

Outputs (relative to the repository root)
----------------------------------------
``benchmarks/plate_stress_gradient_crosscheck_evidence.json``

Usage::

    OMP_NUM_THREADS=2 PYTHONPATH=src python benchmarks/plate_stress_gradient_crosscheck.py

The finite-difference grid keeps the *element* size fixed, so a longer panel
costs more (a/b = 3 takes ~45 s against ~1.3 s at a/b = 1); the sweep below
therefore stops at a/b = 2 while the pytest gate keeps the cheapest subset.
"""
from __future__ import annotations

import json
from pathlib import Path

from tensorfem.marine_plate_buckling import isotropic_plate_rigidities
from tensorfem.plate_bending_buckling import finite_difference_bending_buckling
from tensorfem.plate_stress_gradient_buckling import ritz_gradient_buckling_load

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks" / "plate_stress_gradient_crosscheck_evidence.json"

E, NU, T = 210.0e9, 0.3, 0.012
ASPECTS = (1.0, 2.0)
GRADIENTS = (1.0, 0.5, 0.0, -0.5, -1.0)
BASIS = 24
DIVISIONS = 32
TOLERANCE = 0.01  # 1 %: far looser than the observed 0.33 % worst case


def build_evidence() -> dict[str, object]:
    d, _, h = isotropic_plate_rigidities(E, NU, T)
    rows: list[dict[str, object]] = []
    for aspect in ASPECTS:
        for psi in GRADIENTS:
            mode = ritz_gradient_buckling_load(
                length=aspect, width=1.0, rigidity_x=d, rigidity_y=d,
                coupling_rigidity=h, stress_gradient=psi,
                basis_x=BASIS, basis_y=BASIS,
            )
            difference = finite_difference_bending_buckling(
                length=aspect, width=1.0, rigidity=d,
                edge_ratio=psi, divisions=DIVISIONS,
            )
            error = abs(mode.coefficient / difference["coefficient"] - 1.0)
            rows.append({
                "aspect_ratio": aspect,
                "stress_gradient": psi,
                "galerkin_k": mode.coefficient,
                "finite_difference_k": difference["coefficient"],
                "relative_error": error,
                "passed": bool(error < TOLERANCE),
            })
    worst = max(float(row["relative_error"]) for row in rows)
    return {
        "case": "cross-implementation agreement, linear stress-gradient buckling",
        "description": (
            "Double-sine Galerkin (closed-form ramp integral) against an "
            "independent strong-form finite-difference solve; neither path "
            "shares a basis or an energy matrix with the other."
        ),
        "material": {"young_modulus": E, "poisson_ratio": NU, "thickness": T},
        "settings": {
            "aspect_ratios": list(ASPECTS),
            "stress_gradients": list(GRADIENTS),
            "galerkin_basis": [BASIS, BASIS],
            "finite_difference_divisions": DIVISIONS,
            "tolerance": TOLERANCE,
        },
        "reference": {
            "uniform": 4.0,
            "triangular": 7.81,
            "pure_bending_minimum_over_aspect": 23.9,
            "note": "k = N0 b^2 / (pi^2 D) on the larger edge; consistency leg only",
        },
        "rows": rows,
        "worst_relative_error": worst,
        "worst_relative_error_percent": 100.0 * worst,
        "passed": bool(worst < TOLERANCE),
    }


def main() -> int:
    evidence = build_evidence()
    OUTPUT.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    for row in evidence["rows"]:
        print(f"a/b={row['aspect_ratio']:4.2f} psi={row['stress_gradient']:+.2f} "
              f"galerkin={row['galerkin_k']:9.4f} fd={row['finite_difference_k']:9.4f} "
              f"rel={100.0 * row['relative_error']:6.3f}%", flush=True)
    print(f"worst relative disagreement {evidence['worst_relative_error_percent']:.3f}%"
          f" -> passed={evidence['passed']}")
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())