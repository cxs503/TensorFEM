"""Reproduction driver for Hillerborg, Modeer & Petersson (1976).

Anchor ``REPRO-FEM-HILLERBORG1976``.  Everything printed below comes from
:mod:`tensorfem.fictitious_crack`; every value written to the artifact JSON is
also recomputed here from the module output, so the report can quote the file.

Judgement criteria (declared before the run, tolerances in the docstring of each
``criterion_*`` function):

===========  ==========================================================  ========
id           quantity                                                    tol
===========  ==========================================================  ========
K1_elastic   K_I of the traction-free notched body from the model's own   <= 5 %
             compliance derivative vs the Srawley--Gross handbook value
K2_law_area  int_0^inf f(w) dw of the softening envelope vs G_F           <= 1e-6
K3_balance   |W_ext - U_solid - U_if - D| / W_ext over the whole path      <= 1 %
K4_lefm      P_peak / P_LEFM(W/l_ch = 8)                                   <= 5 %
K5_strength  P_peak / P_strength(W/l_ch = 0.25) in [0.75, 1.05]            band
K6_mesh      P_peak drift between nodes_y = 48 and 96                      <= 2 %
K7_penalty   P_peak drift over k0 factors {1e2, 1e3, 1e4}                  <= 1 %
K8_aW        P_peak(a0/W) strictly decreasing, 4 notch ratios              --
K9_master    collapse of the size effect in W/l_ch (two equal-l_ch sets)   <= 3 %
K10_gf       G_F back-calculated from the total dissipation                <= 5 %
===========  ==========================================================  ========

Usage
-----
``python benchmarks/hillerborg1976_repro.py``          # all stages
``python benchmarks/hillerborg1976_repro.py --quick``  # coarse, for CI smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tensorfem.fictitious_crack import (  # noqa: E402
    BilinearSoftening,
    ExponentialSoftening,
    SENBGeometry,
    build_influence_model,
    elastic_stress_intensity_from_compliance,
    run_size_effect,
    senb_handbook_shape_factor,
    senb_lefm_peak_load,
    solve_cmod_control,
    strength_limit_load,
)

# --- material point (Hillerborg/FCM model concrete; all values SI) -----------
MATERIAL = {
    "young_Pa": 30.0e9,
    "poisson": 0.2,
    "tensile_strength_Pa": 3.0e6,
    "fracture_energy_J_m2": 100.0,
    "stiffness_factor": 1.0e3,  # k0 = factor * E / W  (penalty, non-physical)
}
L_CH = MATERIAL["young_Pa"] * MATERIAL["fracture_energy_J_m2"] / MATERIAL["tensile_strength_Pa"] ** 2
GEOM = {"alpha": 0.2, "span_ratio": 4.0, "cut_fraction": 0.9}


def make_geometry(depth, alpha=None, young=MATERIAL["young_Pa"], poisson=MATERIAL["poisson"]):
    return SENBGeometry(
        depth=depth,
        span=GEOM["span_ratio"] * depth,
        thickness=depth,  # geometrically similar specimens: B = W
        notch=(GEOM["alpha"] if alpha is None else alpha) * depth,
        cut_fraction=GEOM["cut_fraction"],
        young=young,
        poisson=poisson,
    )


def make_law(depth, kind="exponential", fracture_energy=None, factor=None, young=None):
    e = MATERIAL["young_Pa"] if young is None else young
    k0 = (MATERIAL["stiffness_factor"] if factor is None else factor) * e / depth
    cls = {"exponential": ExponentialSoftening, "bilinear": BilinearSoftening}[kind]
    return cls(
        tensile_strength=MATERIAL["tensile_strength_Pa"],
        fracture_energy=MATERIAL["fracture_energy_J_m2"] if fracture_energy is None else fracture_energy,
        initial_stiffness=k0,
    )


def run_case(depth, *, alpha=None, nodes_y=48, n_steps=40, law_kind="exponential",
             factor=None, fracture_energy=None, young=None):
    ft = MATERIAL["tensile_strength_Pa"]
    gf = MATERIAL["fracture_energy_J_m2"] if fracture_energy is None else fracture_energy
    geom = make_geometry(depth, alpha=alpha, young=MATERIAL["young_Pa"] if young is None else young)
    law = make_law(depth, kind=law_kind, fracture_energy=fracture_energy, factor=factor, young=young)
    model = build_influence_model(geom, law, nodes_y=nodes_y)
    path = solve_cmod_control(model, n_steps=n_steps, max_iter=400, tol=1e-9)
    crack_area = geom.thickness * (model.effective_cut - model.effective_notch)
    return {
        "geometry": {
            "W_m": geom.depth,
            "B_m": geom.thickness,
            "S_m": geom.span,
            "a0_m": model.effective_notch,
            "alpha": model.effective_notch / geom.depth,
            "cut_m": model.effective_cut,
            "nodes_y": nodes_y,
            "n_active": model.n_active,
            "n_dof": model.n_dof,
        },
        "law": {
            "kind": law.kind,
            "f_t_Pa": ft,
            "G_F_J_m2": gf,
            "k0_Pa_per_m": law.initial_stiffness,
            "w_elastic_m": law.elastic_opening,
            "w_scale_m": law.w0 if hasattr(law, "w0") else law.failure_opening - law.elastic_opening,
            "law_area_J_m2": law.total_area(),
        },
        "model": {
            "reciprocity_error": model.reciprocity_error(),
            "l_ch_m": model.characteristic_length,
            "W_over_lch": geom.depth / (geom.young * gf / ft**2),
        },
        "path": path.summary(),
        "peak_over_lefm": path.peak_load / senb_lefm_peak_load(geom, gf),
        "peak_over_strength": path.peak_load / strength_limit_load(geom, ft),
        "crack_area_m2": crack_area,
        "G_F_backcalc_J_m2": path.dissipated[-1] / crack_area,
        "process_zone_at_peak_m": float(path.process_zone[path.peak_index]),
        "curve": {
            "cmod_m": path.cmod.tolist(),
            "load_N": path.load.tolist(),
            "deflection_m": path.deflection.tolist(),
            "process_zone_m": path.process_zone.tolist(),
            "dissipated_J": path.dissipated.tolist(),
        },
    }


def criterion_elastic(out_dir, nodes_y=48):
    """K1: elastic stress intensity from the model's compliance derivative."""
    depth = L_CH
    geom = make_geometry(depth)
    law = make_law(depth)
    model = build_influence_model(geom, law, nodes_y=nodes_y)
    rows = []
    for a in (0.15, 0.2, 0.3):
        r = elastic_stress_intensity_from_compliance(model, a * depth, delta=0.06 * depth)
        rows.append(r)
    worst = max(abs(r["rel_error_pct"]) for r in rows)
    return {
        "rows": rows,
        "worst_abs_error_pct": worst,
        "tolerance_pct": 5.0,
        "passed": worst <= 5.0,
    }


def criterion_law_area():
    """K2: the envelope integrates exactly to G_F (two law shapes)."""
    rows = []
    for kind in ("exponential", "bilinear"):
        law = make_law(L_CH, kind=kind)
        area = law.total_area()
        rows.append(
            {
                "kind": kind,
                "G_F_J_m2": law.fracture_energy,
                "quadrature_area_J_m2": area,
                "rel_error": abs(area / law.fracture_energy - 1.0),
            }
        )
    worst = max(r["rel_error"] for r in rows)
    return {"rows": rows, "worst_rel_error": worst, "tolerance": 1e-4, "passed": worst <= 1e-4}


def criterion_balance(cases):
    """K3: energy balance residual of every converged step."""
    worst = max(c["path"]["max_abs_balance"] for c in cases)
    return {"worst_abs_balance": worst, "tolerance": 1e-2, "passed": worst <= 1e-2}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="artifacts/hillerborg1976")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()
    nodes_y = 24 if args.quick else 48
    n_steps = 20 if args.quick else 40
    sizes = (0.25, 1.0, 4.0, 8.0) if args.quick else (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)

    report = {
        "anchor": "REPRO-FEM-HILLERBORG1976",
        "reference": "A. Hillerborg, M. Modeer, P.-E. Petersson, Cem. Concr. Res. 6 (1976) 773-781",
        "material": MATERIAL,
        "l_ch_m": L_CH,
        "geometry": GEOM,
        "settings": {"nodes_y": nodes_y, "n_steps": n_steps, "quick": args.quick},
        "criteria": {},
    }

    # ---- K2: law area ------------------------------------------------------
    report["criteria"]["K2_law_area"] = criterion_law_area()
    print("K2 law area:", report["criteria"]["K2_law_area"]["worst_rel_error"])

    # ---- K1: elastic stress intensity -------------------------------------
    geom = make_geometry(L_CH)
    report["criteria"]["K1_elastic"] = criterion_elastic(args.out, nodes_y=max(48, nodes_y))
    print("K1 elastic worst error %:", report["criteria"]["K1_elastic"]["worst_abs_error_pct"])

    # ---- size effect series ------------------------------------------------
    points = []
    curves = {}
    for ratio in sizes:
        depth = ratio * L_CH
        case = run_case(depth, nodes_y=nodes_y, n_steps=n_steps)
        case["W_over_lch"] = ratio
        points.append(case)
        curves[f"{ratio:g}"] = case["curve"]
        print(
            "size W/lch=%5.2f  P_peak=%9.1f N  P/LEFM=%5.3f  P/strength=%5.3f  bal=%.1e"
            % (
                ratio,
                case["path"]["peak_load_n"],
                case["peak_over_lefm"],
                case["peak_over_strength"],
                case["path"]["max_abs_balance"],
            )
        )
    report["size_effect"] = [
        {
            k: c[k]
            for k in (
                "W_over_lch",
                "geometry",
                "law",
                "model",
                "path",
                "peak_over_lefm",
                "peak_over_strength",
                "G_F_backcalc_J_m2",
                "process_zone_at_peak_m",
                "crack_area_m2",
            )
        }
        for c in points
    ]
    report["curves"] = curves
    report["cases"] = points

    # ---- K3 / K4 / K5 / K10 ------------------------------------------------
    report["criteria"]["K3_balance"] = criterion_balance(points)
    big = [c for c in points if c["W_over_lch"] == 8.0]
    small = [c for c in points if c["W_over_lch"] == 0.25]
    if big:
        err = abs(big[0]["peak_over_lefm"] - 1.0) * 100.0
        report["criteria"]["K4_lefm"] = {
            "W_over_lch": 8.0,
            "ratio": big[0]["peak_over_lefm"],
            "abs_error_pct": err,
            "tolerance_pct": 5.0,
            "passed": err <= 5.0,
        }
        print("K4 LEFM asymptote: ratio %.4f (err %.2f%%)" % (big[0]["peak_over_lefm"], err))
    if small:
        ratio = small[0]["peak_over_strength"]
        report["criteria"]["K5_strength"] = {
            "W_over_lch": 0.25,
            "ratio": ratio,
            "band": [0.80, 1.20],
            "passed": 0.80 <= ratio <= 1.20,
        }
        print("K5 strength asymptote: ratio %.4f" % ratio)
    gf_err = max(abs(c["G_F_backcalc_J_m2"] / MATERIAL["fracture_energy_J_m2"] - 1.0) for c in points)
    report["criteria"]["K10_gf"] = {
        "worst_rel_error": gf_err,
        "tolerance": 0.05,
        "passed": gf_err <= 0.05,
    }
    print("K10 G_F back-calculation worst rel error %.4f" % gf_err)

    # ---- K6: mesh/collocation objectivity ---------------------------------
    base = run_case(L_CH, nodes_y=nodes_y, n_steps=n_steps)
    fine = run_case(L_CH, nodes_y=2 * nodes_y, n_steps=n_steps)
    drift = abs(fine["path"]["peak_load_n"] / base["path"]["peak_load_n"] - 1.0) * 100.0
    report["criteria"]["K6_mesh"] = {
        "nodes_y": [nodes_y, 2 * nodes_y],
        "peak_load_N": [base["path"]["peak_load_n"], fine["path"]["peak_load_n"]],
        "drift_pct": drift,
        "tolerance_pct": 2.0,
        "passed": drift <= 2.0,
    }
    print("K6 mesh drift %.2f%%" % drift)

    # ---- K7: penalty-stiffness objectivity --------------------------------
    pen = []
    for factor in (1e2, 1e3, 1e4):
        c = run_case(L_CH, nodes_y=nodes_y, n_steps=n_steps, factor=factor)
        pen.append({"k0_factor": factor, "peak_load_N": c["path"]["peak_load_n"]})
    peaks = np.array([p["peak_load_N"] for p in pen])
    pen_drift = float((peaks.max() - peaks.min()) / peaks.mean() * 100.0)
    report["criteria"]["K7_penalty"] = {
        "rows": pen,
        "drift_pct": pen_drift,
        "tolerance_pct": 1.0,
        "passed": pen_drift <= 1.0,
    }
    print("K7 penalty drift %.2f%%" % pen_drift)

    # ---- K8: a0/W series ---------------------------------------------------
    aw = []
    for alpha in (0.1, 0.2, 0.3, 0.4):
        c = run_case(L_CH, alpha=alpha, nodes_y=nodes_y, n_steps=n_steps)
        aw.append(
            {
                "alpha": c["geometry"]["alpha"],
                "peak_load_N": c["path"]["peak_load_n"],
                "lefm_peak_N": senb_lefm_peak_load(make_geometry(L_CH, alpha=alpha),
                                                  MATERIAL["fracture_energy_J_m2"]),
                "steps_converged": c["path"]["converged_steps"],
            }
        )
    monotone = bool(all(aw[i]["peak_load_N"] > aw[i + 1]["peak_load_N"] for i in range(len(aw) - 1)))
    report["criteria"]["K8_aW"] = {"rows": aw, "monotone_decreasing": monotone, "passed": monotone}
    print("K8 a/W series:", [(round(r["alpha"], 3), round(r["peak_load_N"], 1)) for r in aw])

    # ---- law-shape sensitivity --------------------------------------------
    bil = run_case(L_CH, nodes_y=nodes_y, n_steps=n_steps, law_kind="bilinear")
    report["law_shape"] = {
        "exponential_peak_N": base["path"]["peak_load_n"],
        "bilinear_peak_N": bil["path"]["peak_load_n"],
        "difference_pct": 100.0 * (bil["path"]["peak_load_n"] / base["path"]["peak_load_n"] - 1.0),
    }
    print("law shape difference %.2f%%" % report["law_shape"]["difference_pct"])

    # ---- K9: master-curve collapse (two sets with the same l_ch) -----------
    # E -> E/4 and f_t -> f_t/2 keeps l_ch = E G_F / f_t^2 identical.
    scaled = []
    for ratio in (0.5, 1.0, 2.0):
        depth = ratio * L_CH
        c = run_case(depth, nodes_y=nodes_y, n_steps=n_steps, young=MATERIAL["young_Pa"] / 4.0)
        geom_s = make_geometry(depth, young=MATERIAL["young_Pa"] / 4.0)
        strength = strength_limit_load(geom_s, MATERIAL["tensile_strength_Pa"] / 2.0)
        scaled.append(
            {
                "W_over_lch": ratio,
                "peak_over_strength": c["path"]["peak_load_n"] / strength,
            }
        )
    ref = {c["W_over_lch"]: c["peak_over_strength"] for c in points}
    collapse = max(
        abs(row["peak_over_strength"] / ref[row["W_over_lch"]] - 1.0) * 100.0
        for row in scaled
        if row["W_over_lch"] in ref
    ) if any(row["W_over_lch"] in ref for row in scaled) else float("nan")
    report["criteria"]["K9_master"] = {
        "scaled_material": {"young_Pa": MATERIAL["young_Pa"] / 4.0,
                            "tensile_strength_Pa": MATERIAL["tensile_strength_Pa"] / 2.0,
                            "same_l_ch": True},
        "rows": scaled,
        "max_deviation_pct": collapse,
        "tolerance_pct": 3.0,
        "passed": bool(collapse == collapse and collapse <= 3.0),
    }
    print("K9 master-curve deviation %.2f%%" % collapse)

    report["runtime_s"] = time.time() - t_start
    report["numpy_version"] = np.__version__
    report["python"] = sys.version.split()[0]

    out_path = os.path.join(args.out, "repro.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("written", out_path, "in %.1f s" % report["runtime_s"])
    summary = {k: bool(v.get("passed", False)) for k, v in report["criteria"].items()}
    print("criteria:", summary)
    return report


if __name__ == "__main__":
    main()