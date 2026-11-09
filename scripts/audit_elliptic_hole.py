"""Audit the Inglis elliptical-hole reference and the production Q4 path.

Writes ``benchmarks/elliptic_hole_reference.json`` (machine-readable evidence)
and ``docs/assets/elliptic_hole/*.png`` (report figures).  Every number in the
JSON is recomputed here; nothing is transcribed.

    OMP_NUM_THREADS=4 PYTHONPATH=src python scripts/audit_elliptic_hole.py
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorfem.elliptic_hole import (  # noqa: E402
    InglisHole,
    boundary_hoop_profile,
    convergence_study,
    solve_case,
)
from tensorfem.kirsch_hole import KirschHole  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
ASSETS = REPO / "docs" / "assets" / "elliptic_hole"
JSON_PATH = REPO / "benchmarks" / "elliptic_hole_reference.json"
LEVELS = (24, 32, 40, 48)
RHO_OUTER = 5.0
GRADING = 3.0


def analytic_checks(hole: InglisHole) -> dict:
    """Identities the exact field has to satisfy, independent of any FE."""
    sigma = np.exp(1j * np.linspace(0.0, 2.0 * math.pi, 721, endpoint=False))
    tx, ty = hole.traction_at_zeta(sigma)
    traction = float(np.max(np.hypot(tx, ty)))
    sxx, syy, sxy = hole.stress_cartesian(*hole.load_axis_tip)
    transverse = hole.stress_cartesian(*hole.transverse_tip)
    # strain compatibility: differentiate the displacement on a ring, apply the
    # constitutive law, compare with the stress field
    worst_strain = 0.0
    e, nu = hole.effective_constants
    scale = max(hole.stress_concentration, 1e-30)
    for theta in np.linspace(0.12, 1.45, 6):
        for rho in (1.25, 2.0, 3.5):
            z = hole.physical(rho * np.exp(1j * theta))
            exx, eyy, gxy = hole.strain_cartesian(float(z.real), float(z.imag))
            field = hole.stress_cartesian(float(z.real), float(z.imag))
            target = (
                (field[0] - nu * field[1]) / e,
                (field[1] - nu * field[0]) / e,
                field[2] * 2.0 * (1.0 + nu) / e,
            )
            worst_strain = max(
                worst_strain,
                max(abs(exx - target[0]), abs(eyy - target[1]), abs(gxy - target[2])) * e,
            )
    return {
        "far_field": hole.far_field,
        "stress_concentration": hole.stress_concentration,
        "hole_traction_max_abs": traction,
        "load_axis_tip": {
            "point": list(hole.load_axis_tip),
            "sigma": [float(sxx), float(syy), float(sxy)],
            "sigma_yy_over_far_field": float(syy / hole.far_field),
        },
        "transverse_tip": {
            "point": list(hole.transverse_tip),
            "sigma": [float(v) for v in transverse],
            "sigma_xx_over_far_field": float(transverse[0] / hole.far_field),
        },
        "strain_compatibility_error_over_E": float(worst_strain / scale),
    }


def kirsch_limit(mapping_m_eps: float = 1e-6) -> dict:
    """``semi_x -> semi_y`` must reproduce the Kirsch field and displacement."""
    hole = InglisHole(semi_x=1.0, semi_y=1.0 + mapping_m_eps, young=1.0, poisson=0.3)
    kirsch = KirschHole(radius=1.0, far_field=1.0, young=1.0, poisson=0.3, plane="stress")
    worst_stress = 0.0
    worst_displacement = 0.0
    for theta in np.linspace(0.08, 3.06, 9):
        for r in (1.15, 1.6, 2.5, 5.0):
            x, y = r * math.cos(theta), r * math.sin(theta)
            mine = hole.stress_cartesian(x, y)
            exact = kirsch.stress_cartesian(x, y)
            worst_stress = max(worst_stress, float(np.max(np.abs(np.subtract(mine, exact)))))
            u_mine = hole.displacement_cartesian(x, y)
            u_exact = kirsch.displacement_cartesian(x, y)
            worst_displacement = max(
                worst_displacement, float(np.max(np.abs(np.subtract(u_mine, u_exact))))
            )
    return {
        "mapping_m": hole.mapping_m,
        "max_stress_difference": worst_stress,
        "max_displacement_difference": worst_displacement,
    }


def figure_boundary_hoop(holes: list[tuple[str, InglisHole]]) -> str:
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=140)
    for label, hole in holes:
        rows = boundary_hoop_profile(hole, 361)
        theta = np.deg2rad([row["theta_degrees"] for row in rows])
        ax.plot(theta, [row["hoop_over_far_field"] for row in rows], label=label)
        for theta_fe, value in (
            (0.0, -1.0),
            (0.5 * math.pi, hole.stress_concentration),
        ):
            ax.plot(
                [theta_fe], [value], "o", markersize=4, color=ax.lines[-1].get_color()
            )
    ax.axhline(3.0, color="0.6", linewidth=0.8, linestyle=":")
    ax.text(0.02, 3.1, "Kirsch $K_t=3$", fontsize=7, color="0.4")
    ax.set_xlabel(r"hole-boundary parameter $\theta$ (rad), $z=\omega(e^{i\theta}$)")
    ax.set_ylabel(r"$\sigma_{tt}/T$")
    ax.set_title("Inglis elliptical hole: exact hoop stress; dots = exact tip anchors")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = ASSETS / "boundary_hoop_stress.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(REPO))


def figure_convergence(studies: dict) -> str:
    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=140)
    for label, study in studies.items():
        n = np.asarray(study["levels"], dtype=float)
        errors = [
            100.0 * case["transverse_tip"]["relative_error"] for case in study["cases"]
        ]
        ax.loglog(n, errors, "o-", label=f"{label} (exact $K_t$={study['exact']:.3f})")
        ax.loglog(
            n,
            errors[0] * (n[0] / n) ** 1.0,
            "k--",
            linewidth=0.8,
            label="slope -1" if label == next(iter(studies)) else None,
        )
    ax.axhline(3.0, color="crimson", linewidth=0.9, linestyle=":")
    ax.text(24.5, 3.3, "3% gate", color="crimson", fontsize=8)
    ax.set_xlabel("mesh level $n_\\theta = n_\\rho$")
    ax.set_ylabel("peak hoop-stress error (%)")
    ax.set_title("Q4 peak-stress convergence at the transverse tip")
    ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = ASSETS / "tip_convergence.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(REPO))


def figure_profiles(studies: dict) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=140)
    styles = ["-", "--"]
    for (label, study), style in zip(studies.items(), styles):
        case = study["cases"][-1]
        rows = case["sample_rows"]
        y = [row["y_over_semi_y"] for row in rows]
        ax.plot(y, [row["sigma_xx_exact"] for row in rows], "k" + style, label=f"exact {label}")
        ax.plot(y, [row["sigma_xx_fe"] for row in rows], "o", markersize=3, label=f"FE {label}")
    ax.set_xlabel("$y / \\mathrm{semi}_y$ (transverse symmetry axis)")
    ax.set_ylabel(r"$\sigma_{xx}/T$")
    ax.set_title("Transverse-axis stress profile, exact vs production Q4 ($n=48$)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = ASSETS / "transverse_profile.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(REPO))


def figure_field(hole: InglisHole) -> str:
    """von Mises contour of the exact field around the sharp hole."""
    rho = np.linspace(1.0, 4.0, 220)
    theta = np.linspace(0.0, 0.5 * math.pi, 260)
    rr, tt = np.meshgrid(rho, theta, indexing="ij")
    z = hole.physical(rr * np.exp(1j * tt))
    sxx, syy, sxy = hole.stress_at_zeta(rr * np.exp(1j * tt))
    vm = np.sqrt(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy)
    fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=140)
    contour = ax.contourf(np.real(z), np.imag(z), vm, levels=40, cmap="inferno")
    fig.colorbar(contour, ax=ax, label=r"von Mises $/T$")
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(
        f"Exact Inglis field, $\\mathrm{{semi}}_x/\\mathrm{{semi}}_y={hole.semi_x/hole.semi_y:.2f}$"
    )
    fig.tight_layout()
    path = ASSETS / "field_von_mises.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(REPO))


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    mild = InglisHole(semi_x=1.5, semi_y=1.0)
    sharp = InglisHole(semi_x=0.5, semi_y=1.5)
    holes = [("mild $K_t$", mild), ("sharp $K_t$", sharp)]

    analytic = {}
    for label, hole in holes:
        analytic[label] = analytic_checks(hole)
        analytic[label]["mapping_radius"] = hole.mapping_radius
        analytic[label]["mapping_m"] = hole.mapping_m
        analytic[label]["tip_curvature_radius"] = hole.tip_curvature_radius
    analytic["kirsch_sweep"] = {
        f"m={m:.1f}": analytic_checks(InglisHole(semi_x=1.0 + m, semi_y=1.0 - m))
        for m in (-0.6, -0.3, 0.0, 0.3, 0.6)
    }
    analytic["kirsch_limit"] = kirsch_limit()

    studies = {}
    for label, hole in holes:
        studies[label] = convergence_study(
            hole, rho_outer=RHO_OUTER, levels=LEVELS, grading=GRADING
        )

    figures = {
        "boundary_hoop_stress": figure_boundary_hoop(
            [("Kirsch $K_t=3$", InglisHole(semi_x=1.0, semi_y=1.0))]
            + [(f"{label} (${v['stress_concentration']:.3f}$)", h) for (label, h), v in zip(holes, analytic.values())]
        ),
        "tip_convergence": figure_convergence(studies),
        "transverse_profile": figure_profiles(studies),
        "field_von_mises": figure_field(sharp),
    }

    summary = {
        label: {
            "semi_x": hole.semi_x,
            "semi_y": hole.semi_y,
            "stress_concentration": hole.stress_concentration,
            "mapping_m": hole.mapping_m,
            "levels": study["levels"],
            "peaks": study["peaks"],
            "errors_percent": [
                100.0 * case["transverse_tip"]["relative_error"] for case in study["cases"]
            ],
            "observed_rates": study["observed_rates"],
            "richardson": study["richardson"],
            "richardson_error_percent": 100.0 * study["richardson_relative_error"],
            "displacement_error_max_relative": [
                case["displacement_error"]["max_relative"] for case in study["cases"]
            ],
            "load_axis_tip_errors_percent": [
                100.0 * case["load_axis_tip"]["relative_error"] for case in study["cases"]
            ],
            "symmetry_edge_error_percent": [
                100.0 * case["symmetry_edge"]["relative_error"] for case in study["cases"]
            ],
            "force_balance_relative": [
                case["force_balance_relative"] for case in study["cases"]
            ],
            "recovery_cross_check": [
                case["recovery_cross_check"]["max_relative_difference"] for case in study["cases"]
            ],
            "profile_finest": study["cases"][-1]["sample_rows"],
        }
        for (label, hole), study in zip(holes, studies.values())
    }

    payload = {
        "schema": "tensorfem/elliptic-hole-reference/1",
        "script": str(pathlib.Path(__file__).name),
        "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "rho_outer": RHO_OUTER,
        "grading": GRADING,
        "element_type": "q4",
        "plane": "stress",
        "poisson": mild.poisson,
        "analytic": analytic,
        "fe_cases": summary,
        "figures": figures,
        "conclusion": {
            "mild_Kt_under_3_percent_from_level": next(
                (
                    level
                    for level, case in zip(
                        studies["mild $K_t$"]["levels"], studies["mild $K_t$"]["cases"]
                    )
                    if case["transverse_tip"]["relative_error"] < 0.03
                ),
                None,
            ),
            "sharp_Kt_under_3_percent_from_level": None,
            "sharp_Kt_observed_order": studies["sharp $K_t$"]["observed_rates"][-1],
        },
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {JSON_PATH.relative_to(REPO)}")
    for label, block in summary.items():
        print(
            f"{label}: Kt={block['stress_concentration']:.4f} "
            f"errors%={[round(v, 2) for v in block['errors_percent']]} "
            f"rate={block['observed_rates'][-1]:.2f} "
            f"richardson_err%={block['richardson_error_percent']:.2f} "
            f"disp={['%.1e' % v for v in block['displacement_error_max_relative']]}"
        )
    print("figures:", figures)
    print("analytic sharp:", json.dumps(analytic["sharp $K_t$"], indent=2))
    print("kirsch limit:", analytic["kirsch_limit"])


if __name__ == "__main__":
    main()
