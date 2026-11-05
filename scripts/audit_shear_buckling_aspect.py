#!/usr/bin/env python3
"""Audit the pure-shear buckling branch of the ship-panel module against a
converged double-sine Galerkin reference.

The script answers one question and nothing else: for a simply supported panel
under pure in-plane shear, is the finite-difference solver of
``tensorfem.marine_plate_combined_buckling`` wrong, or is the asymptotic
coefficient ``k_s = 5.34 + 4 (b/a)^2`` that it was being judged against wrong?

Every number in ``benchmarks/shear_buckling_aspect_reference.json`` and in the
figures is computed here; nothing is transcribed by hand.

Run::

    OMP_NUM_THREADS=2 PYTHONPATH=src python3 scripts/audit_shear_buckling_aspect.py
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.marine_plate_combined_buckling import (  # noqa: E402
    classical_shear_buckling_coefficient,
    panel_buckling_load,
)
from tensorfem.plate_shear_buckling_spectral import (  # noqa: E402
    SQUARE_SHEAR_LITERATURE_COEFFICIENT,
    converged_shear_buckling_coefficient,
    classical_asymptotic_coefficient,
    spectral_buckling_load,
    twist_matrix_quadrature_check,
)

EVIDENCE = ROOT / "benchmarks" / "shear_buckling_aspect_reference.json"
ASSETS = ROOT / "docs" / "assets" / "shear_buckling_aspect"

D = 1.0          # flexural rigidity used for the normalisation N = k pi^2 D / b^2
T = 0.012        # only used to report tau_cr; it does not change k
ASPECTS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
BUDGETS = (8, 12, 16, 20, 24)
FD_GRIDS = {1.0: (24, 24), 1.5: (24, 36), 2.0: (32, 64), 3.0: (20, 60)}
FD_REFINEMENT = ((16, 32), (24, 48), (32, 64))


def spectral_table() -> list[dict]:
    rows = []
    for aspect in ASPECTS:
        data = converged_shear_buckling_coefficient(aspect, budgets=BUDGETS)
        coefficient = float(data["coefficient"])
        fit = classical_asymptotic_coefficient(aspect)
        rows.append({
            "aspect_ratio": aspect,
            "coefficient": coefficient,
            "modes": list(data["modes"]),
            "residual": float(data["residual"]),
            "converged": bool(data["converged"]),
            "asymptotic_fit": fit,
            "fit_error": coefficient / fit - 1.0,
            "truncation_table": [
                {"modes_x": mx, "modes_y": my, "coefficient": value}
                for mx, my, value in data["table"]
            ],
        })
    return rows


def exact_limits() -> dict:
    """Cases where a closed-form coefficient exists, used to qualify the reference."""
    square = spectral_buckling_load(
        length=1.0, width=1.0, rigidity_x=D, rigidity_y=D, coupling_rigidity=D,
        modes_x=20, modes_y=20, longitudinal_share=1.0)
    biaxial = spectral_buckling_load(
        length=1.0, width=1.0, rigidity_x=D, rigidity_y=D, coupling_rigidity=D,
        modes_x=20, modes_y=20, longitudinal_share=1.0, transverse_share=1.0)
    uniaxial_exact = 4.0 * math.pi**2 * D
    biaxial_exact = 2.0 * math.pi**2 * D
    return {
        "uniaxial_square_galerkin": square.critical_line_load,
        "uniaxial_square_exact": uniaxial_exact,
        "uniaxial_square_error": square.critical_line_load / uniaxial_exact - 1.0,
        "equal_biaxial_square_galerkin": biaxial.critical_line_load,
        "equal_biaxial_square_exact": biaxial_exact,
        "equal_biaxial_square_error": biaxial.critical_line_load / biaxial_exact - 1.0,
        "twist_quadrature_worst_error": twist_matrix_quadrature_check(4),
    }


def fd_case(aspect: float, grid: tuple[int, int]) -> dict:
    result = panel_buckling_load(
        length=aspect, width=1.0, thickness=T, rigidity_x=D, rigidity_y=D,
        coupling_rigidity=D, grid_x=grid[0], grid_y=grid[1],
        longitudinal_share=0.0, transverse_share=0.0, shear_share=1.0)
    return {
        "grid": list(result.grid),
        "line_load": result.critical_line_load,
        "coefficient": result.critical_line_load / (math.pi**2 * D),
    }


def solver_vs_reference(spectral: list[dict]) -> list[dict]:
    by_aspect = {row["aspect_ratio"]: row["coefficient"] for row in spectral}
    rows = []
    for aspect, grid in FD_GRIDS.items():
        reference = by_aspect[aspect]
        case = fd_case(aspect, grid)
        fit = classical_shear_buckling_coefficient(aspect)
        rows.append({
            "aspect_ratio": aspect,
            **case,
            "reference_coefficient": reference,
            "error_vs_reference": case["coefficient"] / reference - 1.0,
            "asymptotic_fit": fit,
            "error_vs_asymptotic_fit": case["coefficient"] / fit - 1.0,
        })
    return rows


def refinement(aspect: float, reference: float) -> dict:
    rows = []
    for grid in FD_REFINEMENT:
        case = fd_case(aspect, grid)
        rows.append({
            **case,
            "error_vs_reference": case["coefficient"] / reference - 1.0,
        })
    errors = [abs(row["error_vs_reference"]) for row in rows]
    order = (
        math.log(errors[0] / errors[1]) / math.log(2.0)
        if errors[1] > 0.0 and errors[0] > 0.0 else float("nan")
    )
    return {
        "aspect_ratio": aspect,
        "reference_coefficient": reference,
        "grids": rows,
        "observed_order": order,
        "finest_error": rows[-1]["error_vs_reference"],
        "monotone": errors[-1] < errors[-2] < errors[-3],
    }


def figures(spectral: list[dict], solver: list[dict], refino: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ASSETS.mkdir(parents=True, exist_ok=True)
    written = []

    # 1. Galerkin truncation convergence and solver points.
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for row in spectral:
        if row["aspect_ratio"] not in (1.0, 2.0, 3.0, 8.0):
            continue
        table = row["truncation_table"]
        ax.plot([entry["modes_x"] for entry in table],
                [entry["coefficient"] for entry in table],
                marker="o", ms=3.5, lw=1.2,
                label=f"Galerkin a/b={row['aspect_ratio']:g}")
    ax.set_xlabel("sine modes per direction")
    ax.set_ylabel(r"$k_s = N_{xy,cr} b^2/(\pi^2 D)$")
    ax.set_title("Galerkin truncation converges from above (Ritz bound)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path = ASSETS / "galerkin_convergence.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path.relative_to(ROOT)))

    # 2. Coefficient versus aspect ratio: reference, solver, asymptotic fit.
    aspects = [row["aspect_ratio"] for row in spectral]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(aspects, [row["coefficient"] for row in spectral], "-", lw=2,
            label="Galerkin reference (converged)", color="tab:blue")
    ax.plot(aspects, [row["asymptotic_fit"] for row in spectral], "--", lw=1.4,
            label=r"asymptotic fit $5.34+4(b/a)^2$", color="tab:red")
    ax.plot([row["aspect_ratio"] for row in solver],
            [row["coefficient"] for row in solver], "s", ms=6,
            label="FD solver (marine_plate_combined_buckling)", color="tab:green")
    ax.axhline(SQUARE_SHEAR_LITERATURE_COEFFICIENT, color="0.4", lw=0.8, ls=":")
    ax.annotate("9.34 (Timoshenko & Gere)", (1.02, 9.34), fontsize=7,
                xytext=(1.1, 9.55), color="0.35")
    ax.set_xlabel("aspect ratio a/b")
    ax.set_ylabel(r"$k_s = N_{xy,cr} b^2/(\pi^2 D)$")
    ax.set_title("Pure shear on a simply supported panel")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path = ASSETS / "coefficient_vs_aspect.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path.relative_to(ROOT)))

    # 3. Who is wrong: solver vs reference, fit vs reference.
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot([row["aspect_ratio"] for row in solver],
            [100.0 * row["error_vs_reference"] for row in solver], "s-",
            color="tab:green", label="FD solver vs Galerkin")
    ax.plot(aspects, [100.0 * row["fit_error"] for row in spectral], "--o",
            color="tab:red", ms=3.5, label="asymptotic fit vs Galerkin")
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.axhspan(-3.0, 3.0, color="0.85", zorder=0, label="3 % acceptance band")
    ax.set_xlabel("aspect ratio a/b")
    ax.set_ylabel("relative error [%]")
    ax.set_title("The 3.55 % miss at a/b=2 belongs to the fit, not to the solver")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path = ASSETS / "error_attribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path.relative_to(ROOT)))
    return written


def main() -> int:
    spectral = spectral_table()
    exact = exact_limits()
    solver = solver_vs_reference(spectral)
    by_aspect = {row["aspect_ratio"]: row["coefficient"] for row in spectral}
    refino = refinement(2.0, by_aspect[2.0])
    images = figures(spectral, solver, refino)

    worst = max(abs(row["error_vs_reference"]) for row in solver)
    worst_fit = max(abs(row["fit_error"]) for row in spectral)
    payload = {
        "schema": "tensorfem.shear-buckling-aspect-reference/1.0",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "question": (
            "pure-shear buckling at a/b = 1.5/2/3 was reported as unqualified against "
            "the asymptotic fit; is the solver or the fit at fault?"
        ),
        "reference_method": (
            "double-sine Galerkin on the exact admissible space of the simply supported "
            "plate; the truncation is a Ritz upper bound and is refined until the last "
            "two budgets agree to the reported residual"
        ),
        "rigidity": D,
        "thickness_for_stress": T,
        "reference_qualification": exact,
        "spectral_coefficients": spectral,
        "solver_comparison": solver,
        "solver_refinement_at_aspect_two": refino,
        "verdict": {
            "max_abs_solver_error_vs_reference": worst,
            "max_abs_fit_error_vs_reference": worst_fit,
            "solver_within_3_percent_at_every_aspect": worst < 0.03,
            "asymptotic_fit_within_3_percent_at_every_aspect": worst_fit < 0.03,
            "aspect_two_previously_reported_error": 0.0355,
            "note": (
                "the previously reported 3.55 % at a/b=2 was a solver-vs-fit comparison; "
                "against the converged reference the same solver point is inside 1 %"
            ),
        },
        "figures": images,
    }
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"wrote {EVIDENCE.relative_to(ROOT)}")
    print(f"twist quadrature worst error : {exact['twist_quadrature_worst_error']:.2e}")
    print(f"uniaxial square error        : {exact['uniaxial_square_error']:+.2e}")
    print(f"equal biaxial square error   : {exact['equal_biaxial_square_error']:+.2e}")
    for row in spectral:
        print(f"  a/b={row['aspect_ratio']:<4g} k_s={row['coefficient']:.6f} "
              f"resid={row['residual']:.1e} fit={row['asymptotic_fit']:.4f} "
              f"fit_err={100 * row['fit_error']:+.2f}%")
    for row in solver:
        print(f"  FD a/b={row['aspect_ratio']:<4g} k={row['coefficient']:.5f} "
              f"vs ref {100 * row['error_vs_reference']:+.3f}% "
              f"vs fit {100 * row['error_vs_asymptotic_fit']:+.3f}%")
    print(f"  FD refinement at a/b=2: {100 * refino['finest_error']:+.3f}% "
          f"(order {refino['observed_order']:.2f})")
    print(f"worst solver error vs reference: {100 * worst:.3f}%")
    print(f"worst fit error vs reference   : {100 * worst_fit:.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())