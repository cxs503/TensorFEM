"""Kirsch circular hole under a general biaxial far field: audit and FE ladder.

The uniaxial Kirsch benchmark (``scripts/audit_kirsch_hole.py``) only covers
``sigma_yy = 0``.  A hull or deck plate with an opening sits in a biaxial
in-plane state, so this script

1. self-checks the *generalised* analytic reference (traction-free hole edge,
   radial equilibrium, displacement/strain consistency, the superposition
   construction, the edge-stress identities and the uniform equibiaxial limit),
2. drives the production plane-stress Q4 path on the quarter-annulus symmetry
   model for a ladder of biaxiality ratios ``l = sigma_yy / sigma_xx`` and mesh
   levels, and grades the hole-edge stress concentration and the pointwise
   stress field against the closed form,
3. writes ``benchmarks/kirsch_biaxial_reference.json`` and four figures under
   ``docs/assets/kirsch_biaxial/``.

Run: ``OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/audit_kirsch_biaxial.py``
"""
from __future__ import annotations

import json
import math
import platform as host
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem import __version__
from tensorfem.kirsch_hole import KirschHole, convergence_study, sample_rows

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks" / "kirsch_biaxial_reference.json"
ASSETS = ROOT / "docs" / "assets" / "kirsch_biaxial"
LEVELS = (16, 24, 32, 48)
OUTER = 10.0
RATIOS = (-1.0, 0.0, 0.25, 0.5, 1.0, 2.0)
BASE = dict(radius=1.0, far_field=1.0, young=1.0, poisson=0.3, plane="stress")
FIGURES: list[str] = []


# ---------------------------------------------------------------------------
# Reference self-checks (prove the benchmark before using it)
# ---------------------------------------------------------------------------

def hole_edge_free(hole: KirschHole, samples: int = 361) -> dict:
    """``sigma_rr(a, theta)`` and ``sigma_rt(a, theta)`` must vanish exactly."""
    theta = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    srr, _, srt = hole.stress_polar(hole.radius, theta)
    scale = max(abs(hole.far_field), abs(hole.far_field_y), 1e-30)
    return {
        "max_abs_sigma_rr_over_scale": float(np.max(np.abs(srr)) / scale),
        "max_abs_sigma_rt_over_scale": float(np.max(np.abs(srt)) / scale),
    }


def radial_equilibrium(hole: KirschHole, samples: int = 9, step: float = 1e-6) -> dict:
    """Residual of ``d sigma_rr/dr + (1/r) d sigma_rt/dtheta + (srr-stt)/r = 0``.

    Central differences in ``r`` and ``theta``; the tolerance is set by the
    finite-difference truncation (``step^2``), not by the physics.
    """
    worst = 0.0
    for r in np.linspace(1.5 * hole.radius, OUTER, samples):
        for theta in np.linspace(0.13, math.pi / 2 - 0.13, samples):
            srr_p, stt_p, srt_p = hole.stress_polar(r + step, theta)
            srr_m, _, srt_m = hole.stress_polar(r - step, theta)
            _, _, srt_tp = hole.stress_polar(r, theta + step)
            _, _, srt_tm = hole.stress_polar(r, theta - step)
            srr, stt, _ = hole.stress_polar(r, theta)
            residual = (
                (srr_p - srr_m) / (2.0 * step)
                + (srt_tp - srt_tm) / (2.0 * step) / r
                + (srr - stt) / r
            )
            worst = max(worst, abs(float(residual)))
    scale = max(abs(hole.far_field), abs(hole.far_field_y), 1e-30)
    return {"max_abs_residual_over_scale": worst / scale, "step": step}


def strain_consistency(hole: KirschHole, step: float = 1e-6) -> dict:
    """Differentiate the closed-form displacement and rebuild the stress.

    The check is material-honest: it uses the *equivalent plane-stress*
    constants of :attr:`KirschHole.effective_constants`, so the plane-strain
    case is graded through the same identity the module documents.
    """
    e, nu = hole.effective_constants

    def displacement(x, y):
        return hole.displacement_cartesian(x, y)

    worst = 0.0
    for x in (1.4, 2.0, 3.5):
        for y in (0.3, -0.8, 1.7):
            ux_x = (displacement(x + step, y)[0] - displacement(x - step, y)[0]) / (2 * step)
            ux_y = (displacement(x, y + step)[0] - displacement(x, y - step)[0]) / (2 * step)
            uy_x = (displacement(x + step, y)[1] - displacement(x - step, y)[1]) / (2 * step)
            uy_y = (displacement(x, y + step)[1] - displacement(x, y - step)[1]) / (2 * step)
            exx, eyy, gxy = ux_x, uy_y, ux_y + uy_x
            rebuilt = np.array(
                [
                    e / (1.0 - nu * nu) * (exx + nu * eyy),
                    e / (1.0 - nu * nu) * (eyy + nu * exx),
                    e / (2.0 * (1.0 + nu)) * gxy,
                ]
            )
            closed = np.array([float(v) for v in hole.stress_cartesian(x, y)])
            scale = max(float(np.max(np.abs(closed))), 1e-30)
            worst = max(worst, float(np.max(np.abs(rebuilt - closed))) / scale)
    return {"max_relative_stress_residual": worst, "step": step}


def superposition_identity(hole: KirschHole, step: float = 1e-6) -> dict:
    """Check ``u(l) = u(0) + l * R90 u(0)(R-90 .)`` against an independent object.

    The rotated field is built from a *uniaxial* :class:`KirschHole` instance
    and rotated back by 90 degrees, so a sign error in the implementation of
    ``_lateral_cartesian`` cannot cancel out.
    """
    uniaxial = KirschHole(**{**BASE, "lateral_ratio": 0.0})
    lam = hole.lateral_ratio
    worst = 0.0
    for x in (1.3, 2.2, 4.0):
        for y in (0.4, -1.1, 2.6):
            ux0, uy0 = uniaxial.displacement_cartesian(y, -x)
            expected = (
                uniaxial.displacement_cartesian(x, y)[0] - lam * uy0,
                uniaxial.displacement_cartesian(x, y)[1] + lam * ux0,
            )
            got = hole.displacement_cartesian(x, y)
            scale = max(math.hypot(*expected), 1e-30)
            worst = max(worst, math.hypot(got[0] - expected[0], got[1] - expected[1]) / scale)
    return {"max_relative_difference": worst}


def edge_identities(hole: KirschHole, samples: int = 181) -> dict:
    """The hole-edge hoop field must equal ``T[(1+l) - 2(1-l) cos 2 theta]``."""
    theta = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    _, stt, _ = hole.stress_polar(hole.radius, theta)
    closed = hole.edge_hoop(theta) * hole.far_field
    scale = max(abs(hole.far_field), abs(hole.far_field_y), 1e-30)
    deviation = float(np.max(np.abs(stt - closed))) / scale
    return {
        "max_relative_deviation": deviation,
        "at_0_degrees": float(stt[0]),
        "at_90_degrees": float(stt[samples // 4]),
        "closed_at_0_degrees": float(hole.edge_hoop(0.0) * hole.far_field),
        "closed_at_90_degrees": float(hole.edge_hoop(math.pi / 2.0) * hole.far_field),
    }


def uniform_equibiaxial(hole: KirschHole, samples: int = 181) -> dict | None:
    """``l = 1``: the edge hoop is exactly ``2T`` everywhere (no ``theta``)."""
    if abs(hole.lateral_ratio - 1.0) > 1e-14:
        return None
    theta = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    _, stt, _ = hole.stress_polar(hole.radius, theta)
    scale = abs(hole.far_field)
    return {
        "max_abs_deviation_from_2T_over_T": float(np.max(np.abs(stt - 2.0 * hole.far_field)) / scale),
        "max_abs_u_theta_over_scale": float(
            np.max(np.abs(hole.displacement_polar(hole.radius, theta)[1]))
            / (hole.far_field * hole.radius / hole.young)
        ),
    }


def reference_checks(hole: KirschHole) -> dict:
    return {
        "lateral_ratio": hole.lateral_ratio,
        "hole_edge_free": hole_edge_free(hole),
        "radial_equilibrium": radial_equilibrium(hole),
        "strain_consistency": strain_consistency(hole),
        "superposition_identity": superposition_identity(hole),
        "edge_identities": edge_identities(hole),
        "uniform_equibiaxial": uniform_equibiaxial(hole),
    }


# ---------------------------------------------------------------------------
# FE grading
# ---------------------------------------------------------------------------

def _element_centres(case: dict) -> np.ndarray:
    nodes = np.asarray(case["meta"]["nodes"], dtype=np.float64)
    elements = np.asarray(case["meta"]["elements"])
    return nodes[elements].mean(axis=1)


def grade(study: dict, hole: KirschHole) -> dict:
    """Reduce one ``convergence_study`` run to the graded quantities."""
    applied_scale = math.hypot(hole.far_field, hole.far_field_y)
    rows = []
    for case in study["cases"]:
        edge = case["hole_edge"]
        applied = case["load_resultant"]["applied"]
        exact = case["load_resultant"]["exact_arc"]
        load_error = math.hypot(applied[0] - exact[0], applied[1] - exact[1]) / applied_scale
        rows.append(
            {
                "n_theta": case["n_theta"],
                "elements": case["elements"],
                "dofs": case["dofs"],
                "kt_at_90_degrees": edge["hole_edge_at_90_degrees"] / hole.far_field,
                "kt_at_90_degrees_exact": float(hole.edge_hoop(math.pi / 2.0)),
                "kt_at_90_degrees_error": abs(
                    edge["hole_edge_at_90_degrees"] / hole.far_field
                    - float(hole.edge_hoop(math.pi / 2.0))
                )
                / max(abs(float(hole.edge_hoop(math.pi / 2.0))), 1e-12),
                "hoop_at_0_degrees_over_T": edge["hole_edge_at_0_degrees"] / hole.far_field,
                "hoop_at_0_degrees_exact": float(hole.edge_hoop(0.0)),
                "peak_hoop_over_T": edge["peak_hoop"] / hole.far_field,
                "peak_theta_degrees": edge["peak_theta_degrees"],
                "max_pointwise_stress_error": case["max_stress_relative_error"],
                "displacement_l2": case["displacement_error"]["l2_relative"],
                "load_resultant_error": float(load_error),
                "recovery_cross_check": case["recovery_cross_check"]["max_relative_difference"],
                "edge_profile": edge["profile"],
                "stress_rows": sample_rows(case),
            }
        )
    peak = hole.peak_edge_stress
    last = rows[-1]
    return {
        "lateral_ratio": hole.lateral_ratio,
        "far_field": hole.far_field,
        "far_field_y": hole.far_field_y,
        "exact_peak_stress_over_T": peak["stress_concentration"],
        "exact_peak_theta_degrees": peak["theta_degrees"],
        "exact_peak_uniform": peak["uniform"],
        "rows": rows,
        "observed_orders": study["observed_orders"],
        "finest": {
            "kt_at_90_degrees_error": last["kt_at_90_degrees_error"],
            "max_pointwise_stress_error": last["max_pointwise_stress_error"],
            "displacement_l2": last["displacement_l2"],
            "peak_hoop_over_T": last["peak_hoop_over_T"],
            "peak_relative_error": abs(last["peak_hoop_over_T"] - peak["stress_concentration"])
            / max(abs(peak["stress_concentration"]), 1e-12),
        },
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save(fig, name: str) -> str:
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    FIGURES.append(str(path.relative_to(ROOT)))
    return str(path)


def figure_edge_profiles(graded: list[dict]) -> str:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    theta = np.linspace(0.0, math.pi / 2.0, 361)
    for entry in graded:
        hole = KirschHole(**{**BASE, "lateral_ratio": entry["lateral_ratio"]})
        ax.plot(
            np.degrees(theta),
            hole.edge_hoop(theta) * hole.far_field,
            lw=1.4,
            label=f"exact l={entry['lateral_ratio']:+.2f}",
        )
        profile = np.asarray(entry["rows"][-1]["edge_profile"], dtype=float)
        ax.plot(
            profile[:, 0],
            profile[:, 1] / hole.far_field,
            "o",
            ms=3.5,
            mfc="none",
            label=f"FE n={entry['rows'][-1]['n_theta']} l={entry['lateral_ratio']:+.2f}",
        )
    ax.set_xlabel("hole-edge angle theta [deg]")
    ax.set_ylabel("sigma_theta_theta(a, theta) / T")
    ax.set_title("Hole-edge hoop stress: exact field vs production Q4 (n=48)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    return _save(fig, "kirsch_biaxial_edge_profile.png")


def figure_concentration(graded: list[dict]) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ratios = np.linspace(-1.5, 3.5, 501)
    exact = [KirschHole(**{**BASE, "lateral_ratio": float(v)}).peak_edge_stress for v in ratios]
    ax.plot(ratios, [e["stress_concentration"] for e in exact], "k-", lw=1.6, label="exact peak K_t")
    ax.plot(
        [g["lateral_ratio"] for g in graded],
        [g["finest"]["peak_hoop_over_T"] for g in graded],
        "s",
        ms=6,
        mfc="none",
        label="FE peak (n=48)",
    )
    ax.axhline(3.0, color="0.6", ls=":", lw=1.0)
    ax.axhline(2.0, color="0.6", ls=":", lw=1.0)
    ax.axhline(4.0, color="0.6", ls=":", lw=1.0)
    for value, text in ((3.0, "uniaxial 3"), (2.0, "equibiaxial 2"), (4.0, "pure shear 4")):
        ax.annotate(text, (3.45, value), fontsize=7, ha="right", va="bottom", color="0.35")
    ax.set_xlabel("biaxiality ratio l = sigma_yy / sigma_xx")
    ax.set_ylabel("peak hole-edge K_t = sigma_theta_theta / T")
    ax.set_title("Stress concentration versus biaxiality")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, "kirsch_biaxial_kt_vs_lambda.png")


def figure_convergence(graded: list[dict]) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for entry in graded:
        n = [row["n_theta"] for row in entry["rows"]]
        ax.loglog(n, [row["max_pointwise_stress_error"] for row in entry["rows"]], "o-",
                  ms=4, lw=1.2, label=f"stress l={entry['lateral_ratio']:+.2f}")
        ax.loglog(n, [row["displacement_l2"] for row in entry["rows"]], "s--",
                  ms=3, lw=0.9, alpha=0.7, label=f"disp L2 l={entry['lateral_ratio']:+.2f}")
    ax.invert_xaxis()
    ax.set_xlabel("n_theta = n_radial per quarter annulus")
    ax.set_ylabel("relative error")
    ax.set_title("Mesh refinement ladder, all biaxiality ratios")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6, ncol=2)
    return _save(fig, "kirsch_biaxial_convergence.png")


def figure_field(graded: list[dict], pick: float = 0.5) -> str:
    hole = KirschHole(**{**BASE, "lateral_ratio": pick})
    # rebuild one solve to keep the raw field (the JSON keeps only the graded rows)
    from tensorfem.kirsch_hole import solve_case

    case = solve_case(hole, OUTER, n_theta=LEVELS[-1])
    centres = _element_centres(case)
    stored = np.asarray(case["stored_stress"], dtype=np.float64)
    r = np.hypot(centres[:, 0], centres[:, 1])
    theta = np.arctan2(centres[:, 1], centres[:, 0])
    c, s = np.cos(theta), np.sin(theta)
    hoop_fe = stored[:, 0] * s * s + stored[:, 1] * c * c - 2.0 * stored[:, 2] * c * s
    _, hoop_exact, _ = hole.stress_polar(r, theta)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    for ax, values, title in (
        (axes[0], hoop_fe, "production Q4 element-centre"),
        (axes[1], hoop_exact, "analytic"),
    ):
        sc = ax.scatter(centres[:, 0], centres[:, 1], c=values / hole.far_field,
                        s=9, cmap="coolwarm")
        ax.set_aspect("equal")
        ax.set_xlabel("x/a")
        ax.set_title(f"sigma_theta_theta/T, l={pick:+.2f}, {title}")
        fig.colorbar(sc, ax=ax, shrink=0.85)
    axes[0].set_ylabel("y/a")
    return _save(fig, "kirsch_biaxial_field.png")


# ---------------------------------------------------------------------------

def main() -> dict:
    graded = []
    checks = []
    for ratio in RATIOS:
        hole = KirschHole(**{**BASE, "lateral_ratio": ratio})
        checks.append(reference_checks(hole))
        study = convergence_study(hole, levels=LEVELS, outer_radius=OUTER)
        graded.append(grade(study, hole))
    payload = {
        "schema": "fem-kirsch-biaxial/1",
        "generated_by": "scripts/audit_kirsch_biaxial.py",
        "environment": {
            "python": host.python_version(),
            "tensorfem": __version__,
            "machine": host.machine(),
        },
        "problem": {
            "radius": BASE["radius"],
            "young": BASE["young"],
            "poisson": BASE["poisson"],
            "plane": BASE["plane"],
            "outer_radius": OUTER,
            "levels": list(LEVELS),
            "ratios": list(RATIOS),
            "model": "quarter-annulus Q4, symmetry on both axes, exact arc tractions",
        },
        "reference_checks": checks,
        "graded": graded,
        "figures": [],
    }
    payload["figures"] = [
        figure_edge_profiles(graded),
        figure_concentration(graded),
        figure_convergence(graded),
        figure_field(graded),
    ]
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    for entry in graded:
        finest = entry["finest"]
        print(
            f"l={entry['lateral_ratio']:+.2f}  Kt(90)err={finest['kt_at_90_degrees_error']*100:.3f}%"
            f"  peak={finest['peak_hoop_over_T']:.4f} (exact {entry['exact_peak_stress_over_T']:.3f})"
            f"  pointwise={finest['max_pointwise_stress_error']*100:.3f}%"
            f"  L2={finest['displacement_l2']:.2e}"
        )
    return payload


if __name__ == "__main__":
    main()