"""Evidence run for the Smith progressive-collapse panel surface (P70-N01).

Produces ``artifacts/smith_progressive_collapse/evidence.json`` plus four
figures.  Every number in the JSON is rounded to six significant digits
(``round_payload``) and every figure carries its sha256 in the evidence
provenance block, so the artifact cannot drift from the image set.

Run:  python3 benchmarks/smith_progressive_collapse.py
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.smith_progressive_collapse import (  # noqa: E402
    IMPERFECTION_SENSITIVITY,
    PROVENANCE,
    RESIDUAL_SENSITIVITY,
    EXPERIMENT_SCHEMA,
    IACSPanelGeometry,
    PlateStrip,
    compare_experimental_panels,
    csr_format_panel_ultimate,
    equal_biaxial_capacity,
    panel_interaction,
    round_payload,
    sheet_shear_ultimate,
)

OUT = ROOT / "artifacts" / "smith_progressive_collapse"
FIGURES = OUT / "figures"

GEOMETRY = IACSPanelGeometry()
SY = GEOMETRY.yield_stress
E = GEOMETRY.young


def plate_for_beta(beta: float, *, aspect: float = 3.0, **kwargs) -> PlateStrip:
    thickness = GEOMETRY.spacing * math.sqrt(SY / E) / beta
    return PlateStrip(length=aspect * GEOMETRY.spacing, width=GEOMETRY.spacing,
                      thickness=thickness, young=E, poisson=GEOMETRY.poisson,
                      yield_stress=SY, **kwargs).validate()


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def figure_load_end_shortening(panel, plate, stiffener) -> Path:
    grid = [plate.yield_strain * 6.0 * i / 300 for i in range(301)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.plot([eps / plate.yield_strain for eps in grid],
            [plate.stress(eps) / SY for eps in grid], label="plate field", lw=2.0)
    ax.plot([eps / plate.yield_strain for eps in grid],
            [stiffener.stress(eps) / SY for eps in grid], label="tee stiffener", lw=2.0)
    ax.plot([eps / plate.yield_strain for eps in grid],
            [(panel.plate_area * plate.stress(eps) + panel.stiffener_area * stiffener.stress(eps))
             / panel.total_area / SY for eps in grid], label="panel (area weighted)",
            lw=2.4, color="k")
    for value, name in ((plate.ultimate_ratio, "plate"), (stiffener.ultimate_ratio, "stiffener")):
        ax.axhline(value, ls=":", lw=1.0, color="grey")
        ax.annotate(f"{name} ult. {value:.3f}", (0.2, value + 0.01), fontsize=8, color="grey")
    ax.set_xlabel(r"mean strain  $\epsilon/\epsilon_y$")
    ax.set_ylabel(r"average stress  $\sigma/\sigma_y$")
    ax.set_title("Load-end shortening: components and Smith combination")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 6)

    ax = axes[1]
    rows = panel.curve()["rows"]
    ax.plot([row["strain_over_yield"] for row in rows],
            [row["panel_stress_over_yield"] for row in rows], color="k", lw=2.4)
    peak = max(rows, key=lambda row: row["panel_stress_over_yield"])
    ax.plot([peak["strain_over_yield"]], [peak["panel_stress_over_yield"]], "o", ms=7,
            color="crimson")
    ax.annotate(f"panel ultimate {peak['panel_stress_over_yield']:.4f}$\\sigma_y$",
                (peak["strain_over_yield"], peak["panel_stress_over_yield"]),
                textcoords="offset points", xytext=(12, 4), fontsize=9, color="crimson")
    ax.set_xlabel(r"mean strain  $\epsilon/\epsilon_y$")
    ax.set_ylabel(r"panel stress  $\sigma/\sigma_y$")
    ax.set_title("Panel curve: peak and post-ultimate softening")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = FIGURES / "load_end_shortening.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_plate_ultimate_vs_beta(beta_rows) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    betas = [row["beta"] for row in beta_rows]
    ax.plot(betas, [row["faulkner_ratio"] for row in beta_rows], "--", lw=2.0,
            label="Faulkner 1975: $2.25/\\beta-1.25/\\beta^2$")
    ax.plot(betas, [row["ultimate_ratio"] for row in beta_rows], "-", lw=2.4,
            label="plate strip model (reference state)")
    ax.plot(betas, [row["buckling_stress_over_yield"] for row in beta_rows], ":",
            lw=1.6, label="elastic buckling $\\sigma_{cr}/\\sigma_y$")
    ax.set_xlabel(r"plate slenderness $\beta=(b/t)\sqrt{\sigma_y/E}$")
    ax.set_ylabel(r"$\sigma_u/\sigma_y$")
    ax.set_title("Plate ultimate versus slenderness (criterion (a), $\\leq$10 %)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    path = FIGURES / "plate_ultimate_vs_beta.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_sensitivity(imperfection_rows, residual_rows) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    ax = axes[0]
    ax.plot([row["w0_over_thickness"] for row in imperfection_rows],
            [row["ultimate_ratio"] for row in imperfection_rows], "o-", lw=2.0)
    ax.axvline(imperfection_rows[len(imperfection_rows) // 2]["w0_over_thickness"],
               color="grey", ls=":", lw=1.2)
    ax.set_xlabel(r"$w_0/t$")
    ax.set_ylabel(r"$\sigma_u/\sigma_y$")
    ax.set_title("Initial deflection sensitivity (criterion (d))")
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot([row["residual_over_yield"] for row in residual_rows],
            [row["ultimate_ratio"] for row in residual_rows], "s-", lw=2.0, color="darkred")
    ax.axvline(0.2, color="grey", ls=":", lw=1.2)
    ax.set_xlabel(r"$\sigma_{rc}/\sigma_y$")
    ax.set_ylabel(r"$\sigma_u/\sigma_y$")
    ax.set_title("Residual stress sensitivity (criterion (e))")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = FIGURES / "sensitivity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_interaction(envelope, equal_point) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot([row["sigma_x_over_uniaxial"] for row in envelope],
            [row["sigma_y_over_uniaxial"] for row in envelope], lw=2.4,
            label="CSR quadratic surface ($c=0$)")
    values = [i / 100 for i in range(101)]
    ax.plot(values, [math.sqrt(max(0.0, 1.0 - v ** 2)) for v in values], "--", lw=1.4,
            label="von Mises plane stress (yield only)")
    ratio = equal_point["equal_biaxial_stress"] / equal_point["uniaxial_ultimate"]
    ax.plot([ratio], [ratio], "o", ms=8, color="crimson")
    ax.annotate(f"equal biaxial: {ratio:.4f} = 1/$\\sqrt{{2}}$\n(error {100 * equal_point['relative_error']:+.2f} %)",
                (ratio, ratio), textcoords="offset points", xytext=(14, -60),
                fontsize=9, color="crimson")
    ax.set_xlabel(r"$\sigma_x/\sigma_{xu}$")
    ax.set_ylabel(r"$\sigma_y/\sigma_{yu}$")
    ax.set_title("Biaxial interaction envelope (criterion (f))")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    path = FIGURES / "interaction.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    plate = GEOMETRY.plate_strip()
    stiffener = GEOMETRY.stiffener()
    panel = GEOMETRY.smith_panel()

    # (a) plate ultimate vs Faulkner over beta in [1, 3]
    beta_grid = [1.0 + 0.05 * i for i in range(41)]
    beta_rows = []
    max_a = 0.0
    for beta in beta_grid:
        strip = plate_for_beta(beta)
        faulkner = min(1.0, 2.25 / strip.beta - 1.25 / strip.beta ** 2)
        error = strip.ultimate_ratio / faulkner - 1.0
        max_a = max(max_a, abs(error))
        beta_rows.append({
            "beta": strip.beta,
            "buckling_coefficient": strip.buckling_coefficient,
            "buckling_stress_over_yield": strip.buckling_stress / SY,
            "faulkner_ratio": faulkner,
            "ultimate_ratio": strip.ultimate_ratio,
            "relative_error": error,
        })

    # (b) Smith curve peak vs rule-format min-based estimate
    csr = csr_format_panel_ultimate(panel)

    # (c) experimental comparison: harness only (no experimental values available)
    criterion_c = {
        "status": "WITHHELD",
        "reason": ("primary experimental table (Paik & Thayamballi 2003) is not "
                   "available in this environment; no value is quoted from memory"),
        "harness": "compare_experimental_panels",
        "schema_keys": list(EXPERIMENT_SCHEMA),
        "dataset_path": "benchmarks/paik_thayamballi_2003_specimens.json",
        "synthetic_harness_check": compare_experimental_panels([
            {
                "specimen": "SYNTHETIC-FIXTURE",
                "span_m": GEOMETRY.span,
                "spacing_m": GEOMETRY.spacing,
                "plate_thickness_m": GEOMETRY.plate_thickness,
                "web_height_m": GEOMETRY.web_height,
                "web_thickness_m": GEOMETRY.web_thickness,
                "flange_width_m": GEOMETRY.flange_width,
                "flange_thickness_m": GEOMETRY.flange_thickness,
                "young_pa": GEOMETRY.young,
                "poisson": GEOMETRY.poisson,
                "yield_stress_pa": GEOMETRY.yield_stress,
                "measured_ultimate_stress_pa": float(panel.ultimate()["ultimate_stress"]),
                "provenance": "SYNTHETIC FIXTURE - harness verification only, not experimental data",
                "primary_source_verified": False,
            }
        ]),
    }

    # (d) imperfection sensitivity
    reference_w0 = plate.reference_imperfection_ratio
    imperfection_rows = []
    for multiplier in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        strip = GEOMETRY.plate_strip(imperfection_ratio=multiplier * reference_w0)
        imperfection_rows.append({
            "multiplier": multiplier,
            "w0_over_thickness": strip.imperfection_over_thickness,
            "imperfection_factor": strip.imperfection_factor,
            "ultimate_ratio": strip.ultimate_ratio,
        })
    base_ratio = GEOMETRY.plate_strip().ultimate_ratio
    high_ratio = GEOMETRY.plate_strip(imperfection_ratio=3.0 * reference_w0).ultimate_ratio
    drop_d = base_ratio / high_ratio - 1.0

    # (e) residual stress sensitivity
    residual_rows = []
    for residual in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
        strip = GEOMETRY.plate_strip(residual_ratio=residual)
        residual_rows.append({
            "residual_over_yield": residual,
            "residual_factor": strip.residual_factor,
            "post_buckling_fraction": strip.post_buckling_fraction,
            "ultimate_ratio": strip.ultimate_ratio,
        })
    without_residual = GEOMETRY.plate_strip(residual_ratio=0.0).ultimate_ratio
    drop_e = 1.0 - base_ratio / without_residual

    # (f) biaxial interaction
    equal_point = equal_biaxial_capacity(panel)
    equal_point_jockey = equal_biaxial_capacity(panel, jockeying=0.2)
    envelope = []
    for i in range(41):
        fraction = i / 40.0
        state = panel_interaction(panel, sigma_x=fraction * float(csr["smith_ultimate"]),
                                  sigma_y=math.sqrt(max(0.0, 1.0 - fraction ** 2))
                                  * float(csr["smith_ultimate"]))
        envelope.append({
            "sigma_x_over_uniaxial": fraction,
            "sigma_y_over_uniaxial": state["sigma_y"] / float(csr["smith_ultimate"]),
            "csr_quadratic_utilisation": state["csr_quadratic_utilisation"],
        })

    # (g) post-ultimate behaviour
    curve = panel.curve()
    rows = curve["rows"]
    values = [row["panel_stress_over_yield"] for row in rows]
    peak_index = max(range(len(values)), key=values.__getitem__)
    tail = values[peak_index:]
    monotone = all(a >= b - 1e-12 for a, b in zip(tail, tail[1:]))
    pre = values[:peak_index + 1]
    pre_monotone = all(a <= b + 1e-12 for a, b in zip(pre, pre[1:]))
    softening_slope = ((tail[min(len(tail) - 1, 25)] - tail[0])
                       / (rows[peak_index + min(len(tail) - 1, 25)]["strain_over_yield"]
                          - rows[peak_index]["strain_over_yield"]))

    # numerical robustness: peak of the panel curve versus a refined strain grid
    from tensorfem.smith_progressive_collapse import default_strain_grid
    coarse = float(panel.ultimate()["ultimate_stress"])
    fine_grid = default_strain_grid(plate.yield_strain, points=4801)
    fine = float(panel.curve(fine_grid)["ultimate_stress"])
    numerical = {
        "strain_grid_points": 601,
        "refined_grid_points": 4801,
        "coarse_peak": coarse,
        "refined_peak": fine,
        "relative_change": fine / coarse - 1.0,
    }

    # stiffener / shear / interaction details
    shear = sheet_shear_ultimate(plate)

    figures = {
        "load_end_shortening": figure_load_end_shortening(panel, plate, stiffener),
        "plate_ultimate_vs_beta": figure_plate_ultimate_vs_beta(beta_rows),
        "sensitivity": figure_sensitivity(imperfection_rows, residual_rows),
        "interaction": figure_interaction(envelope, equal_point),
    }
    figure_provenance = {}
    for name, path in figures.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        figure_provenance[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": digest,
            "bytes": path.stat().st_size,
        }

    criteria = {
        "a_plate_vs_faulkner": {
            "threshold": 0.10,
            "measured": max_a,
            "passed": max_a <= 0.10,
            "detail": "model reproduces Faulkner exactly at the reference state "
                      "w0/t = 0.1 beta^2, sigma_rc = 0.2 sigma_y; grid beta in [1, 3]",
        },
        "b_panel_vs_csr_format": {
            "threshold": 0.10,
            "measured": abs(float(csr["relative_error"])),
            "signed": float(csr["relative_error"]),
            "passed": abs(float(csr["relative_error"])) <= 0.10,
            "governing_mechanism": csr["governing_mechanism"],
            "smith_ratio": float(csr["smith_ratio"]),
            "csr_format_ratio": float(csr["csr_format_ratio"]),
        },
        "c_experimental_paik_thayamballi_2003": {
            "threshold": 0.15,
            "measured": None,
            "passed": None,
            "status": "WITHHELD",
            "reason": criterion_c["reason"],
        },
        "d_imperfection_sensitivity": {
            "threshold": 0.05,
            "measured": drop_d,
            "passed": drop_d >= 0.05,
            "detail": "w0/t from 0.1 beta^2 to 0.3 beta^2 on the IACS panel",
        },
        "e_residual_stress_effect": {
            "threshold": 0.03,
            "measured": drop_e,
            "passed": drop_e >= 0.03,
            "detail": "sigma_rc/sigma_y 0.2 versus 0.0 on the IACS panel",
        },
        "f_equal_biaxial_ratio": {
            "threshold": 0.10,
            "measured": abs(float(equal_point["relative_error"])),
            "ratio": float(equal_point["ratio_to_uniaxial"]),
            "ideal": 1.0 / math.sqrt(2.0),
            "passed": abs(float(equal_point["relative_error"])) <= 0.10,
            "jockeying_0p2_ratio": float(equal_point_jockey["ratio_to_uniaxial"]),
            "jockeying_0p2_error": abs(float(equal_point_jockey["relative_error"])),
        },
        "g_post_ultimate_softening": {
            "passed": bool(monotone and pre_monotone and softening_slope < 0.0),
            "post_peak_monotone_decreasing": bool(monotone),
            "pre_peak_monotone_increasing": bool(pre_monotone),
            "softening_slope_per_yield_strain": softening_slope,
            "tail_stress_over_peak": tail[-1] / tail[0],
        },
        "h_six_significant_digits": {
            "passed": True,
            "detail": "round_payload applied to the whole evidence payload; every "
                      "figure sha256 recorded in provenance",
            "digits": 6,
        },
    }

    evidence = {
        "node": "P70-N01",
        "repository": "TensorFEM",
        "module": "tensorfem.smith_progressive_collapse",
        "method": "Smith progressive collapse: component load-end shortening curves, "
                  "area-weighted combination, peak of the panel curve",
        "provenance": PROVENANCE,
        "coefficients": {
            "imperfection_sensitivity": IMPERFECTION_SENSITIVITY,
            "residual_sensitivity": RESIDUAL_SENSITIVITY,
        },
        "geometry": {
            "label": "IACS CSR No.47 standard stiffened panel",
            "span_m": GEOMETRY.span,
            "spacing_m": GEOMETRY.spacing,
            "plate_thickness_m": GEOMETRY.plate_thickness,
            "stiffener": "HP 200 x 10, idealised as equivalent tee web 200x9 + flange 55x10",
            "web_height_m": GEOMETRY.web_height,
            "web_thickness_m": GEOMETRY.web_thickness,
            "flange_width_m": GEOMETRY.flange_width,
            "flange_thickness_m": GEOMETRY.flange_thickness,
            "young_pa": GEOMETRY.young,
            "poisson": GEOMETRY.poisson,
            "yield_stress_pa": GEOMETRY.yield_stress,
        },
        "plate": plate.summary(),
        "stiffener": stiffener.summary(),
        "panel": panel.summary(),
        "shear": shear,
        "interaction_equal_biaxial": equal_point,
        "interaction_equal_biaxial_jockeying_0p2": equal_point_jockey,
        "csr_format": csr,
        "criteria": criteria,
        "numerical": numerical,
        "sweeps": {
            "beta": beta_rows,
            "imperfection": imperfection_rows,
            "residual": residual_rows,
            "biaxial_envelope": envelope,
        },
        "curves": {
            "plate": [{"strain_over_yield": row["strain_over_yield"],
                       "stress_over_yield": row["stress_over_yield"]}
                      for row in plate.curve()[::15]],
            "panel": rows[::15],
        },
        "experimental_comparison": criterion_c,
        "figures": figure_provenance,
    }
    evidence = round_payload(evidence)
    out = OUT / "evidence.json"
    out.write_text(json.dumps(evidence, indent=2, sort_keys=False) + "\n")

    summary = {
        "evidence": str(out.relative_to(ROOT)),
        "criteria": {name: {"passed": value.get("passed"),
                            "measured": value.get("measured"),
                            "status": value.get("status", "MEASURED")}
                     for name, value in criteria.items()},
        "figures": [str(path.relative_to(ROOT)) for path in figures.values()],
    }
    print(json.dumps(round_payload(summary), indent=2))
    return evidence


if __name__ == "__main__":
    main()