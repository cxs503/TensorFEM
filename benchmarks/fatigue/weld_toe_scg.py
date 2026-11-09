"""Benchmark: weld-toe stress concentration -> hot spot -> S-N fatigue life.

Run with ``PYTHONPATH=src python benchmarks/fatigue/weld_toe_scg.py``.

The benchmark exercises the *detail* fatigue chain of
:mod:`tensorfem.fatigue.weld_toe`:

* transverse fillet T-joint, butt weld with reinforcement and right-angle corner
  joint cross-sections, each with a parametric weld-toe radius ``r`` in the
  physical range 0.5-2 mm (plate thickness ``t = 25 mm``);
* ``K_t`` of the Q8 plane-strain notch model against the independent closed forms
  (Nii for T-joints, Maddox for butt welds), plus the *sensitivity* of the
  comparison to the mesh level, the weld leg, the flank angle, the plane
  assumption and the root-bonding idealisation;
* IIW surface extrapolation at the toe (linear ``0.4t/0.9t`` and quadratic
  ``0.4t/0.9t/1.4t``) and the mesh independence of the extrapolated hot-spot
  stress under 2x refinement;
* DNV-RP-C203 style double-slope S-N curves in air, seawater with cathodic
  protection and seawater free corrosion, with the fatigue life cross-checked
  against an independently coded closed form.

Outputs
-------
``artifacts/fatigue/weld_toe_scg_evidence.json``  machine-readable evidence
``docs/assets/fatigue/*.png``                     figures quoted by the report
"""
from __future__ import annotations

import json
import math
import pathlib
import platform
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.fatigue import (  # noqa: E402
    ENVIRONMENTS,
    QUADRATIC_STATIONS,
    WeldDetail,
    detail_fatigue_assessment,
    fatigue_life,
    hot_spot_stress,
    maddox_stress_concentration,
    mesh_convergence,
    nii_stress_concentration,
    solve_weld_detail,
)

ARTIFACTS = ROOT / "artifacts" / "fatigue"
FIGURES = ROOT / "docs" / "assets" / "fatigue"

PLATE_T = 0.025          # base plate thickness (m)
ATTACH_T = 0.025         # attachment thickness (m)
ATTACH_H = 0.030         # attachment height above the plate (m)  -> h/t = 1.2
WELD_LEG = 0.0175        # fillet weld leg = 0.7 t
REINFORCEMENT = 0.003    # butt-weld cap height (m)
CREST = 0.012            # butt-weld crest width (m)
TOE_RADII = (0.5e-3, 1.0e-3, 2.0e-3)
NOMINAL_RANGE_PA = 100.0e6

#: Acceptance gates of the campaign node (verdicts in the report).
GATES = {
    "t_joint_kt_vs_nii": 0.25,
    "butt_joint_kt_vs_maddox": 0.10,
    "hotspot_mesh_change": 0.02,
    "fatigue_life_vs_analytic": 0.15,
}


def t_joint(radius: float, **kwargs) -> WeldDetail:
    fields = dict(
        joint_type="t_fillet", plate_thickness_m=PLATE_T, toe_radius_m=radius,
        attachment_thickness_m=ATTACH_T, attachment_height_m=ATTACH_H,
        weld_leg_m=WELD_LEG,
    )
    fields.update(kwargs)
    return WeldDetail(**fields)


def butt_joint(radius: float, **kwargs) -> WeldDetail:
    fields = dict(
        joint_type="butt", plate_thickness_m=PLATE_T, toe_radius_m=radius,
        reinforcement_m=REINFORCEMENT, crest_width_m=CREST, flank_angle_deg=30.0,
    )
    fields.update(kwargs)
    return WeldDetail(**fields)


def corner_joint(radius: float, **kwargs) -> WeldDetail:
    fields = dict(
        joint_type="corner", plate_thickness_m=PLATE_T, toe_radius_m=radius,
        attachment_thickness_m=ATTACH_T, attachment_height_m=ATTACH_H,
        weld_leg_m=WELD_LEG, symmetry=False,
    )
    fields.update(kwargs)
    return WeldDetail(**fields)


def analytic_life(stress_range_pa: float, environment: str) -> float:
    """Independent closed-form two-slope S-N evaluation (no module inversion).

    ``N = 10 ** (log a1 - m1 log10 ds)`` below the knee; above it the second branch
    anchored on the knee point.  Written out here so the module's inversion and
    unit handling are checked against a second implementation.
    """
    curve = ENVIRONMENTS[environment].curve
    ds = stress_range_pa / 1.0e6
    log_n1 = curve.log_a1 - curve.slope_m1 * math.log10(ds)
    if 10.0 ** log_n1 <= curve.knee_cycles:
        return 10.0 ** log_n1
    log_knee = math.log10(curve.knee_cycles)
    log_a2 = log_knee + (curve.slope_m2 / curve.slope_m1) * (curve.log_a1 - log_knee)
    return 10.0 ** (log_a2 - curve.slope_m2 * math.log10(ds))


def radius_sweep() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for radius in TOE_RADII:
        for kind, factory, anchor_fn in (
            ("t_fillet", t_joint, "nii"),
            ("butt", butt_joint, "maddox"),
            ("corner", corner_joint, None),
        ):
            detail = factory(radius)
            t0 = time.perf_counter()
            solution = solve_weld_detail(detail, level=1)
            seconds = time.perf_counter() - t0
            lin = hot_spot_stress(solution, detail.plate_thickness_m, method="linear")
            quad = hot_spot_stress(solution, detail.plate_thickness_m, method="quadratic")
            if anchor_fn == "nii":
                anchor = nii_stress_concentration(PLATE_T, radius, ATTACH_H)
            elif anchor_fn == "maddox":
                anchor = maddox_stress_concentration(PLATE_T, radius)
            else:
                anchor = None
            row: dict[str, object] = {
                "joint": kind,
                "toe_radius_m": radius,
                "toe_ratio_t_over_r": PLATE_T / radius,
                "kt_fe": solution.kt_toe,
                "kt_hotspot_linear": lin.stress_ratio,
                "kt_hotspot_quadratic": quad.stress_ratio,
                "attachment_side_kt": solution.attachment_toe_stress_pa / solution.nominal_stress_pa,
                "anchor_kt": anchor,
                "anchor_error": (abs(solution.kt_toe - anchor) / anchor) if anchor else None,
                "hotspot_linear_samples_pa": lin.sampled_stress_pa,
                "hotspot_quadratic_samples_pa": quad.sampled_stress_pa,
                "x_toe_m": solution.x_toe_m,
                "n_elements": solution.n_elements,
                "min_edge_m": solution.min_edge_m,
                "solve_seconds": seconds,
            }
            if kind == "t_fillet" and radius == TOE_RADII[1]:
                row["surface_x_m"] = solution.surface_x_m
                row["surface_sigma_xx_pa"] = solution.surface_sigma_xx_pa
            rows.append(row)
    return rows


def mesh_study() -> dict[str, object]:
    t0 = time.perf_counter()
    study = mesh_convergence(t_joint(TOE_RADII[1]), levels=(1, 2, 3))
    study["wall_seconds"] = time.perf_counter() - t0
    study["butt"] = mesh_convergence(butt_joint(TOE_RADII[1]), levels=(1, 2))
    return study


def sensitivity() -> list[dict[str, object]]:
    """Effect of the geometry conventions the empirical anchors leave open."""
    anchor = nii_stress_concentration(PLATE_T, TOE_RADII[1], ATTACH_H)
    variants = {
        "weld_leg_0.5t": dict(weld_leg_m=0.5 * PLATE_T),
        "weld_leg_0.6t": dict(weld_leg_m=0.6 * PLATE_T),
        "flank_30": dict(flank_angle_deg=30.0),
        "flank_60": dict(flank_angle_deg=60.0),
        "unbonded_root": dict(bonded_root=False),
    }
    rows: list[dict[str, object]] = []
    for name, kwargs in variants.items():
        solution = solve_weld_detail(t_joint(TOE_RADII[1], **kwargs), level=1)
        rows.append({
            "variant": name,
            "kt_fe": solution.kt_toe,
            "anchor_kt": anchor,
            "anchor_error": abs(solution.kt_toe - anchor) / anchor,
            "attachment_side_kt": solution.attachment_toe_stress_pa / solution.nominal_stress_pa,
        })
    plane_stress = solve_weld_detail(t_joint(TOE_RADII[1]), level=1, plane="stress")
    rows.append({
        "variant": "plane_stress",
        "kt_fe": plane_stress.kt_toe,
        "anchor_kt": anchor,
        "anchor_error": abs(plane_stress.kt_toe - anchor) / anchor,
        "attachment_side_kt": plane_stress.attachment_toe_stress_pa / plane_stress.nominal_stress_pa,
    })
    mesh_level_2 = solve_weld_detail(t_joint(TOE_RADII[1]), level=2)
    rows.append({
        "variant": "mesh_level_2",
        "kt_fe": mesh_level_2.kt_toe,
        "anchor_kt": anchor,
        "anchor_error": abs(mesh_level_2.kt_toe - anchor) / anchor,
        "attachment_side_kt": mesh_level_2.attachment_toe_stress_pa / mesh_level_2.nominal_stress_pa,
    })
    return rows


def fatigue_rows() -> dict[str, object]:
    out: dict[str, object] = {"nominal_stress_range_pa": NOMINAL_RANGE_PA, "cases": []}
    for kind, factory in (("t_fillet", t_joint), ("butt", butt_joint), ("corner", corner_joint)):
        for radius in TOE_RADII:
            detail = factory(radius)
            solution = solve_weld_detail(detail, level=1)
            lin = hot_spot_stress(solution, detail.plate_thickness_m, method="linear")
            quad = hot_spot_stress(solution, detail.plate_thickness_m, method="quadratic")
            for environment in ("air", "seawater_cp", "seawater_free"):
                for label, ratio in (
                    ("hotspot_linear", lin.stress_ratio),
                    ("hotspot_quadratic", quad.stress_ratio),
                    ("toe_notch", solution.kt_toe),
                ):
                    stress_range = ratio * NOMINAL_RANGE_PA
                    life = fatigue_life(stress_range, environment=environment)
                    reference = analytic_life(stress_range, environment)
                    out["cases"].append({
                        "joint": kind,
                        "toe_radius_m": radius,
                        "environment": environment,
                        "stress_concept": label,
                        "stress_ratio": ratio,
                        "stress_range_pa": stress_range,
                        "life_cycles": life,
                        "analytic_life_cycles": reference,
                        "life_error": abs(life - reference) / reference,
                    })
    cross = []
    anchor = nii_stress_concentration(PLATE_T, TOE_RADII[1], ATTACH_H)
    fe = solve_weld_detail(t_joint(TOE_RADII[1]), level=1)
    for environment in ("air", "seawater_cp", "seawater_free"):
        life_fe = fatigue_life(fe.kt_toe * NOMINAL_RANGE_PA, environment=environment)
        life_anchor = fatigue_life(anchor * NOMINAL_RANGE_PA, environment=environment)
        cross.append({
            "environment": environment,
            "kt_fe": fe.kt_toe,
            "kt_anchor": anchor,
            "life_fe_cycles": life_fe,
            "life_anchor_cycles": life_anchor,
            "life_ratio": life_fe / life_anchor,
        })
    out["anchor_cross_check"] = cross
    return out


def curve_check() -> dict[str, object]:
    """Continuity and FAT values of the three environments (traceability)."""
    out = {}
    for name, env in ENVIRONMENTS.items():
        curve = env.curve
        knee = curve.knee_cycles
        out[name] = {
            "log_a1": curve.log_a1,
            "slope_m1": curve.slope_m1,
            "knee_cycles": knee,
            "log_a2": curve.log_a2,
            "slope_m2": curve.slope_m2,
            "knee_stress_range_mpa": curve.knee_stress_range_pa / 1.0e6,
            "fat_at_2e6_mpa": curve.stress_range_at_cycles(2.0e6) / 1.0e6,
            "continuity_ratio": (
                curve.cycles_to_failure(curve.knee_stress_range_pa * 1.0000001)
                / curve.cycles_to_failure(curve.knee_stress_range_pa * 0.9999999)
            ),
        }
    return out


def equivalent_radius(sweep: list[dict[str, object]]) -> list[dict[str, object]]:
    """Invert the anchors for the toe radius the FE ``K_t`` implies.

    ``K_t - 1`` scales like ``r ** -0.5`` in both closed forms, so the ratio of the
    measured to the anchored ``K_t - 1`` converts into an *equivalent* toe radius.
    A ratio below one means the empirical fit behaves like a sharper toe than the
    geometric radius of the model.
    """
    rows = []
    for row in sweep:
        if not row.get("anchor_kt"):
            continue
        excess_fe = row["kt_fe"] - 1.0
        excess_anchor = row["anchor_kt"] - 1.0
        if excess_fe <= 0.0:
            continue
        rows.append({
            "joint": row["joint"],
            "toe_radius_m": row["toe_radius_m"],
            "excess_ratio": excess_fe / excess_anchor,
            "equivalent_radius_ratio": (excess_anchor / excess_fe) ** 2,
        })
    return rows


def verdicts(evidence: dict[str, object]) -> dict[str, object]:
    sweep = evidence["radius_sweep"]

    def worst(kind: str) -> float:
        vals = [r["anchor_error"] for r in sweep if r["joint"] == kind and r["anchor_error"]]
        return max(vals) if vals else float("nan")

    study = evidence["mesh_study"]
    hotspot_change = max(study["change"]["hotspot_linear"], study["change"]["hotspot_quadratic"])
    life_error = max(row["life_error"] for row in evidence["fatigue"]["cases"])
    return {
        "t_joint_kt_vs_nii": {"measured": worst("t_fillet"),
                              "limit": GATES["t_joint_kt_vs_nii"],
                              "passed": bool(worst("t_fillet") <= GATES["t_joint_kt_vs_nii"])},
        "butt_joint_kt_vs_maddox": {"measured": worst("butt"),
                                    "limit": GATES["butt_joint_kt_vs_maddox"],
                                    "passed": bool(worst("butt") <= GATES["butt_joint_kt_vs_maddox"])},
        "hotspot_mesh_change": {"measured": hotspot_change,
                                "limit": GATES["hotspot_mesh_change"],
                                "passed": bool(hotspot_change <= GATES["hotspot_mesh_change"])},
        "fatigue_life_vs_analytic": {"measured": life_error,
                                     "limit": GATES["fatigue_life_vs_analytic"],
                                     "passed": bool(life_error <= GATES["fatigue_life_vs_analytic"])},
    }


def figures(evidence: dict[str, object]) -> list[pathlib.Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    paths: list[pathlib.Path] = []
    sweep = evidence["radius_sweep"]

    # figure 1: K_t vs t/r with the independent closed forms -----------------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ratios = np.linspace(8.0, 80.0, 200)
    ax.plot(ratios, [nii_stress_concentration(PLATE_T, PLATE_T / q, ATTACH_H) for q in ratios],
            "-", color="C3", label="Nii closed form (T-joint)")
    ax.plot(ratios, [maddox_stress_concentration(PLATE_T, PLATE_T / q) for q in ratios],
            "--", color="C0", label="Maddox closed form (butt)")
    for kind, style, colour in (("t_fillet", "o", "C3"), ("butt", "s", "C0"), ("corner", "^", "C2")):
        pts = [r for r in sweep if r["joint"] == kind]
        ax.plot([r["toe_ratio_t_over_r"] for r in pts], [r["kt_fe"] for r in pts],
                style, color=colour, label=f"Q8 detail model, {kind}")
    ax.set_xscale("log")
    ax.set_xlabel("t / r")
    ax.set_ylabel("notch stress concentration $K_t$")
    ax.set_title("Weld-toe $K_t$: Q8 detail model vs empirical closed forms")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGURES / "kt_vs_tr.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # figure 2: surface stress path with the extrapolation stations ----------
    row = next(r for r in sweep
               if r["joint"] == "t_fillet" and abs(r["toe_radius_m"] - TOE_RADII[1]) < 1e-12)
    xs = np.asarray(row["surface_x_m"])
    sig = np.asarray(row["surface_sigma_xx_pa"]) / 1.0e6
    toe = row["x_toe_m"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    mask = (xs >= toe - 2.0 * PLATE_T) & (xs <= toe + 2.5 * PLATE_T)
    ax.plot((xs[mask] - toe) * 1e3, sig[mask], "-", color="C0", label="surface $\\sigma_{xx}$ (FE)")
    for station in QUADRATIC_STATIONS:
        ax.axvline(station * PLATE_T * 1e3, color="0.6", ls=":", lw=0.9)
        ax.plot([station * PLATE_T * 1e3], [np.interp(toe + station * PLATE_T, xs, sig)],
                "o", color="C1", ms=4)
    ax.axhline(row["kt_hotspot_linear"], color="C3", ls="--", lw=1.0,
               label=f"linear hot spot {row['kt_hotspot_linear']:.3f}")
    ax.axhline(row["kt_hotspot_quadratic"], color="C4", ls="-.", lw=1.0,
               label=f"quadratic hot spot {row['kt_hotspot_quadratic']:.3f}")
    ax.axhline(row["kt_fe"] / 1.0, color="C2", ls="-", lw=1.0, alpha=0.6,
               label=f"toe notch peak {row['kt_fe']:.3f}")
    ax.set_xlabel("distance from the toe tangent point / mm")
    ax.set_ylabel("$\\sigma_{xx} / \\sigma_{nom}$")
    ax.set_title(f"Surface stress path, T-joint r = {row['toe_radius_m']*1e3:.1f} mm")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGURES / "hotspot_path.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # figure 3: S-N curves with the computed lives ---------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ranges = np.logspace(math.log10(5.0e6), math.log10(400.0e6), 200)
    for name, colour in (("air", "C0"), ("seawater_cp", "C1"), ("seawater_free", "C3")):
        ax.loglog(ranges / 1e6, [ENVIRONMENTS[name].curve.cycles_to_failure(r) for r in ranges],
                  "-", color=colour, label=ENVIRONMENTS[name].curve.name)
    for name, colour in (("air", "C0"), ("seawater_cp", "C1"), ("seawater_free", "C3")):
        pts = [c for c in evidence["fatigue"]["cases"]
               if c["joint"] == "t_fillet" and c["environment"] == name
               and c["stress_concept"] == "hotspot_quadratic"
               and abs(c["toe_radius_m"] - TOE_RADII[1]) < 1e-12]
        if pts:
            ax.plot([pts[0]["stress_range_pa"] / 1e6], [pts[0]["life_cycles"]], "o",
                    color=colour, ms=6, mfc="none")
    ax.set_xlabel("hot-spot stress range / MPa")
    ax.set_ylabel("cycles to failure")
    ax.set_title("DNV-RP-C203 double-slope S-N curves, T-joint r = 1 mm")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = FIGURES / "sn_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # figure 4: mesh convergence ---------------------------------------------
    study = evidence["mesh_study"]
    levels = [row["level"] for row in study["levels"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(levels, [row["kt_toe"] for row in study["levels"]], "o-", label="$K_t$ toe notch")
    ax.plot(levels, [row["hotspot_linear"] for row in study["levels"]], "s--", label="hot spot (linear)")
    ax.plot(levels, [row["hotspot_quadratic"] for row in study["levels"]], "^-.", label="hot spot (quadratic)")
    ax.set_xlabel("mesh level (level 2 doubles the near-toe refinement)")
    ax.set_ylabel("stress ratio")
    ax.set_title(f"Mesh independence, T-joint r = {TOE_RADII[1]*1e3:.1f} mm")
    for row in study["levels"]:
        ax.annotate(f"{row['n_elements']} el.", (row["level"], row["kt_toe"]),
                    textcoords="offset points", xytext=(0, -14), fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGURES / "mesh_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths


def main() -> dict[str, object]:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    evidence: dict[str, object] = {
        "benchmark": "weld_toe_scg",
        "module": "tensorfem.fatigue.weld_toe",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "geometry": {
            "plate_thickness_m": PLATE_T,
            "attachment_thickness_m": ATTACH_T,
            "attachment_height_m": ATTACH_H,
            "weld_leg_m": WELD_LEG,
            "butt_reinforcement_m": REINFORCEMENT,
            "butt_crest_width_m": CREST,
            "toe_radii_m": list(TOE_RADII),
            "plane": "strain",
            "young_pa": 210.0e9,
            "poisson": 0.3,
            "external_toe_angle_deg": 135.0,
            "idealisation": "circular toe arc tangent to plate surface and weld flank; "
                            "bonded attachment/plate root (root-gap sensitivity reported)",
        },
        "gates": GATES,
        "sn_curves": curve_check(),
        "radius_sweep": radius_sweep(),
        "mesh_study": mesh_study(),
        "sensitivity": sensitivity(),
        "fatigue": fatigue_rows(),
        "equivalent_radius": equivalent_radius(radius_sweep()),
        "assessments": [
            detail_fatigue_assessment(factory(TOE_RADII[1]),
                                      nominal_stress_range_pa=NOMINAL_RANGE_PA,
                                      environment="seawater_cp", level=1)
            for factory in (t_joint, butt_joint, corner_joint)
        ],
    }
    evidence["figures"] = [str(p.relative_to(ROOT)) for p in figures(evidence)]
    evidence["verdicts"] = verdicts(evidence)
    evidence["wall_seconds"] = time.perf_counter() - t_start
    path = ARTIFACTS / "weld_toe_scg_evidence.json"
    path.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"verdicts": evidence["verdicts"], "evidence": str(path),
                      "wall_seconds": evidence["wall_seconds"]}, indent=2))
    return evidence


if __name__ == "__main__":
    main()