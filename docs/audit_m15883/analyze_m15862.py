"""m15883 board-audit analyser for m15862 / CASE-HOLLOW-SPHERE-3D (TensorFEM).

Reads the delivered artifact only (zero hand-copy) and produces:
  * the two readings of the case metric:
      (A) location-matched sampled reference  -> what the artifact scores
      (B) exact volume-weighted layer integral -> what the artifact declares
          ("reference_functional" + `layer_integrals`) and what the case-pool
          metric text ("volume-weighted probe") literally asks for
  * an independent re-derivation of the Lame field and of the exact layer
    integrals (fresh quadrature, independent of tensorfem)
  * figures: field profiles vs Lame, per-quantity error under both readings,
    convergence ladder.

Usage: python3 docs/audit_m15883/analyze_m15862.py \
           --artifact ../../m15862/artifacts/case_hollow_sphere_3d/case_hollow_sphere_3d.json \
           --out docs/audit_m15883
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROBES = ("radial_displacement_inner", "radial_displacement_outer",
          "radial_stress_inner", "hoop_stress_inner", "von_mises_inner",
          "hoop_stress_outer", "radial_stress_outer")
PRIMARY = ("radial_stress_inner", "hoop_stress_inner", "von_mises_inner",
           "hoop_stress_outer")


def lame(case):
    a, b = case["inner_radius_m"], case["outer_radius_m"]
    E, nu = case["young_modulus_pa"], case["poisson_ratio"]
    p = case["internal_pressure_pa"]
    A = p * a ** 3 / (b ** 3 - a ** 3)
    B = A * b ** 3

    def sr(r):
        return A - B / r ** 3

    def st(r):
        return A + B / (2 * r ** 3)

    def ur(r):
        return ((1 - 2 * nu) * A * r + (1 + nu) * B / (2 * r ** 2)) / E

    return A, B, sr, st, ur


def volume_average(f, ri, ro, n=400000):
    """Independent exact volume-weighted average (midpoint quadrature)."""
    r = ri + (np.arange(n) + 0.5) * (ro - ri) / n
    w = r ** 2
    return float((f(r) * w).sum() / w.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--out", default="docs/audit_m15883")
    args = ap.parse_args()
    art = json.loads(pathlib.Path(args.artifact).read_text())
    case = art["problem"]
    A, B, sr, st, ur = lame(case)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"A={A:.6g} B={B:.6g}  sigma_r(a)={sr(case['inner_radius_m']):.6g} "
          f"(=-p) u_r(a)={ur(case['inner_radius_m']):.6g} u_r(b)={ur(case['outer_radius_m']):.6g}")

    findings = {"independent_reference": {
        "A_Pa": A, "B_Pa_m3": B,
        "sigma_r_inner_Pa": sr(case["inner_radius_m"]),
        "sigma_r_outer_Pa": sr(case["outer_radius_m"]),
        "u_r_inner_m": ur(case["inner_radius_m"]),
        "u_r_outer_m": ur(case["outer_radius_m"]),
        "artifact_u_r_inner_m": art["summary"]["hex20"]["table"][0]["ref"],
        "artifact_u_r_outer_m": art["summary"]["hex20"]["table"][1]["ref"],
    }}

    # --- exact layer integrals, re-derived independently -------------------
    exact = {}
    for order in ("hex8", "hex20"):
        # Element layers are uniform subdivisions of [a, b].  NOTE: the artifact's
        # `reference_layer_integrals.layer_bounds` are NOT used here because they
        # are taken from torch.unique(node radii), which for HEX20 includes the
        # edge-midside radii -> the declared bounds are half-layers.  See finding F1.
        n_layers = art["summary"][order]["finest_mesh"]["radial_layers"]
        a, b = case["inner_radius_m"], case["outer_radius_m"]
        edges = [a + (b - a) * k / n_layers for k in range(n_layers + 1)]
        ri, ro, ri2, ro2 = edges[0], edges[1], edges[-2], edges[-1]
        lt = lambda r1, r2: math.log(r2 / r1) / (r2 ** 3 - r1 ** 3)  # noqa: E731
        exact[order] = {
            "layer_edges": edges,
            "inner": {"radial_stress": A - 3 * B * lt(ri, ro),
                      "hoop_stress": A + 1.5 * B * lt(ri, ro),
                      "von_mises": 4.5 * B * lt(ri, ro)},
            "outer": {"hoop_stress": A + 1.5 * B * lt(ri2, ro2),
                      "radial_stress": A - 3 * B * lt(ri2, ro2)},
        }
        # independent cross-check of the closed forms
        exact[order]["inner"]["radial_stress_quad"] = volume_average(sr, ri, ro)
        exact[order]["inner"]["hoop_stress_quad"] = volume_average(st, ri, ro)
    findings["independent_exact_layer_integrals"] = exact

    # --- two readings of the metric ---------------------------------------
    rows = {}
    for order in ("hex8", "hex20"):
        table = {r["quantity"]: r for r in art["summary"][order]["table"]}
        integrals = art["ladders"][order][-1]["metrics"]["reference_layer_integrals"]
        per = []
        for q in PROBES:
            got = table[q]["got"]
            ref_scored = table[q]["ref"]          # location-matched sampling
            if q.startswith("radial_displacement"):  # surface value, no volume weight
                ref_exact = ref_scored
            elif q.endswith("_inner"):
                ref_exact = exact[order]["inner"][q.replace("_inner", "")]
            elif q.endswith("_outer") and q != "radial_stress_outer":
                ref_exact = exact[order]["outer"][q.replace("_outer", "")]
            elif q == "radial_stress_outer":
                ref_exact = exact[order]["outer"]["radial_stress"]
            else:                                  # surface displacements
                ref_exact = ref_scored
            per.append({"quantity": q, "got": got, "ref_scored": ref_scored,
                        "err_scored": abs(got - ref_scored) / abs(ref_scored),
                        "ref_exact_layer_integral": ref_exact,
                        "err_exact_layer_integral": abs(got - ref_exact) / abs(ref_exact),
                        "artifact_layer_integrals_ref": (
                            integrals["inner"].get(q.replace("_inner", ""))
                            if q.endswith("_inner") else
                            integrals["outer"].get(q.replace("_outer", ""))
                            if q.endswith("_outer") else None)})
        rows[order] = per
        worst_scored = max(per, key=lambda r: r["err_scored"])
        worst_primary_exact = max((r for r in per if r["quantity"] in PRIMARY),
                                  key=lambda r: r["err_exact_layer_integral"])
        findings.setdefault("metric_readings", {})[order] = {
            "worst_scored": [worst_scored["quantity"], worst_scored["err_scored"]],
            "worst_primary_under_exact_layer_integral":
                [worst_primary_exact["quantity"],
                 worst_primary_exact["err_exact_layer_integral"]],
            "artifact_verdict": art["verdict"],
            "tolerance": art["tolerance"],
        }
        print(f"\n{order}: scored worst {worst_scored['quantity']} "
              f"{worst_scored['err_scored']:.4%} | exact-integral worst "
              f"{worst_primary_exact['quantity']} "
              f"{worst_primary_exact['err_exact_layer_integral']:.4%}")

    # declared-vs-scored functional mismatch (machine-checkable)
    mism = []
    for order in ("hex8", "hex20"):
        integrals = art["ladders"][order][-1]["metrics"]["reference_layer_integrals"]
        for side in ("inner", "outer"):
            for key in ("radial_stress", "hoop_stress", "von_mises"):
                declared = integrals[side][key]
                if declared is None:
                    continue
                table = {r["quantity"]: r["ref"] for r in
                         art["summary"][order]["table"]}
                name = f"{key}_{side}"
                if name not in table:
                    continue
                scored = table[name]
                mism.append({"order": order, "probe": f"{key}_{side}",
                             "declared_lame_layer_average": declared,
                             "declared_layer_bounds": bounds if False else None,
                             "scored_ref": scored,
                             "rel_gap": abs(declared - scored) / abs(declared)})
    findings["declared_vs_scored_reference"] = {
        "artifact_reference_functional": art["ladders"]["hex8"][-1]["metrics"].get(
            "reference_functional"),
        "pairs": mism}
    print("\ndeclared reference_functional:", findings["declared_vs_scored_reference"]
          ["artifact_reference_functional"])
    for m in mism:
        print(f"  {m['order']:5s} {m['probe']:20s} declared={m['declared_lame_layer_average']:12.6g} "
              f"scored={m['scored_ref']:12.6g} gap={m['rel_gap']:.2%}")
    (out / "analysis_m15862.json").write_text(json.dumps(
        {"findings": findings, "rows": rows, "artifact": str(args.artifact)}, indent=1))

    # ---------------------------- figures ---------------------------------
    # F1: profiles FE vs Lame, finest mesh of each family
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for order, style in (("hex8", "o--"), ("hex20", "s-")):
        fields = art["ladders"][order][-1]["metrics"]["fields"]
        r = np.asarray(fields["radius"])
        idx = np.argsort(r)
        for ax, key, label in ((axes[0], "radial_stress", r"$\sigma_r$ [Pa]"),
                               (axes[1], "hoop_stress", r"$\sigma_\theta$ [Pa]"),
                               (axes[2], "von_mises", r"$\sigma_{vM}$ [Pa]")):
            ax.plot(r[idx], np.asarray(fields[key])[idx], style,
                    ms=3, label=f"{order} FE")
        rr = np.linspace(case["inner_radius_m"], case["outer_radius_m"], 200)
        for ax, f, key in ((axes[0], sr, "radial"), (axes[1], st, "hoop"),
                           (axes[2], None, "von_mises")):
            if f is None:
                ax.plot(rr, np.abs(st(rr) - sr(rr)), "k-", lw=1.0, label="Lamé")
            else:
                ax.plot(rr, f(rr), "k-", lw=1.0, label="Lamé")
            ax.set_xlabel("r [m]")
            ax.set_ylabel({"radial": r"$\sigma_r$ [Pa]", "hoop": r"$\sigma_\theta$ [Pa]",
                           "von_mises": r"$\sigma_{vM}$ [Pa]"}[key])
    axes[0].legend(fontsize=8)
    fig.suptitle("CASE-HOLLOW-SPHERE-3D: FE fields vs Lamé (element centres, m15862 artifact)")
    fig.tight_layout()
    fig.savefig(out / "fig_fields_vs_lame.png", dpi=130)

    # F2: per-quantity error, both readings
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, order in zip(axes, ("hex8", "hex20")):
        q = [r["quantity"].replace("_", " ").replace("stress", "σ").replace("radial", "r")
             .replace("hoop", "θ").replace("von mises", "vM").replace("displacement", "u")
             for r in rows[order]]
        x = np.arange(len(q))
        ax.bar(x - 0.2, [r["err_scored"] * 100 for r in rows[order]], 0.4,
               label="scored ref (sampled at element centres)")
        ax.bar(x + 0.2, [r["err_exact_layer_integral"] * 100 for r in rows[order]], 0.4,
               label="exact volume-weighted layer integral")
        ax.axhline(art["tolerance"] * 100, color="k", ls=":", lw=1,
                   label="case tolerance 5%")
        ax.set_xticks(x)
        ax.set_xticklabels(q, rotation=35, ha="right", fontsize=7)
        ax.set_title(f"{order} finest mesh")
        ax.set_ylabel("relative error [%]")
    axes[0].legend(fontsize=7)
    fig.suptitle("Two readings of 'volume-weighted probe relative error' (m15883 audit)")
    fig.tight_layout()
    fig.savefig(out / "fig_metric_two_readings.png", dpi=130)

    # F3: convergence ladders
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for order, style in (("hex8", "o--"), ("hex20", "s-")):
        conv = art["summary"][order]["convergence"]
        vals = conv["worst_error_ladder"]
        ax.plot(range(1, len(vals) + 1), [v * 100 for v in vals], style,
                label=f"{order} worst (scored metric)")
    ax.axhline(art["tolerance"] * 100, color="k", ls=":", lw=1, label="5% line")
    ax.set_xlabel("ladder level")
    ax.set_ylabel("worst relative error [%]")
    ax.set_title("Delivered convergence ladders")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_convergence.png", dpi=130)
    print("\nfigures written to", out)


if __name__ == "__main__":
    main()