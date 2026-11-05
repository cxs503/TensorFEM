#!/usr/bin/env python3
"""Does the *merged* B-bar kernel still reproduce both reports' published numbers?

PR #78 merged two independent plane-strain B-bar Q4 cures into one element kernel.
Merging is only safe if the surviving kernel reproduces the accuracy both reports
claimed, so this script re-runs the qualification on the merged tree and compares
against the published anchors:

* plane-strain Lamé cylinder, ``nu = 0.4999``, sector Q4 mesh,
* both formulations: fully integrated Q4 (must stay fail-closed) and B-bar,
* published anchors: radial-stress L2 0.5738 % (formulation-sweep report) and
  0.5999 % / displacement 0.1236 % / mean stress 0.0182 % (plane-strain report).

Outputs ``benchmarks/union_kernel_drift_m3364.json`` and the drift figure, then
prints the error-vs-``nu`` table.

Usage: PYTHONPATH=src python3 scripts/union_kernel_drift.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfem.plane_strain_locking import plane_locking_sweep  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
POISSONS = (0.3, 0.45, 0.49, 0.499, 0.4999)
MESH = {"n_theta": 16, "n_radial": 8}

# published in docs/plane-strain-locking-bbar.md (O) and docs/plane-strain-bbar-locking.md (M)
ANCHORS = {
    "radial_stress_l2_ours": 0.005738,      # O: 0.5738 % at nu = 0.4999, 16x8 (306 DOF)
    "radial_stress_l2_main": 0.005999,      # M: 0.5999 % at nu = 0.4999
    "displacement_l2_main": 0.001236,       # M: 0.1236 %
}


def main() -> int:
    sweep = plane_locking_sweep(poissons=POISSONS, **MESH)
    rows = []
    for formula in ("standard", "bbar"):
        for record in sweep:
            if record["formulation"] != formula:
                continue
            err = record["errors"]
            rows.append({
                "formulation": formula,
                "poisson": record["poisson_ratio"],
                "displacement_l2": err["displacement_l2"],
                "radial_stress_l2": err["radial_stress_l2"],
                "hoop_stress_l2": err["hoop_stress_l2"],
                "bore_displacement_relative": err["bore_displacement_relative"],
                "n_dofs": record["dofs"],
            })

    bbar = {r["poisson"]: r for r in rows if r["formulation"] == "bbar"}
    stiff = {r["poisson"]: r for r in rows if r["formulation"] == "standard"}
    worst = bbar[0.4999]

    drift = {
        "mesh": MESH,
        "anchors": ANCHORS,
        "recomputed_nu_04999": worst,
        "drift_relative": {
            "radial_stress_l2_vs_ours": worst["radial_stress_l2"] / ANCHORS["radial_stress_l2_ours"] - 1.0,
            "radial_stress_l2_vs_main": worst["radial_stress_l2"] / ANCHORS["radial_stress_l2_main"] - 1.0,
            "displacement_l2_vs_main": worst["displacement_l2"] / ANCHORS["displacement_l2_main"] - 1.0,
        },
        "flatness_bbar_radial": worst["radial_stress_l2"] / bbar[0.3]["radial_stress_l2"],
        "locking_ratio_standard_over_bbar": stiff[0.4999]["radial_stress_l2"] / worst["radial_stress_l2"],
        "table": rows,
    }

    out = REPO / "benchmarks/union_kernel_drift_m3364.json"
    out.write_text(json.dumps(drift, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ax = axes[0]
    for formula, style in (("standard", dict(marker="o", color="#b2182b", ls="-")),
                           ("bbar", dict(marker="s", color="#2166ac", ls="-"))):
        pts = [r for r in rows if r["formulation"] == formula]
        ax.semilogy([r["poisson"] for r in pts], [100 * r["radial_stress_l2"] for r in pts],
                    label=f"radial stress, {formula}", **style)
        ax.semilogy([r["poisson"] for r in pts], [100 * r["displacement_l2"] for r in pts],
                    label=f"displacement, {formula}", marker=style["marker"],
                    color=style["color"], ls="--", alpha=0.55)
    ax.axhline(3.0, color="k", ls=":", lw=1.2)
    ax.text(0.302, 3.3, "3 % gate", fontsize=8)
    ax.set_xlabel(r"Poisson ratio $\nu$")
    ax.set_ylabel("relative L2 error [%]")
    ax.set_title("Merged B-bar kernel: error vs $\\nu$ (16x8 sector Q4)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    labels = ["radial\n(ours 0.574 %)", "radial\n(main 0.600 %)", "displ.\n(main 0.124 %)"]
    published = [ANCHORS["radial_stress_l2_ours"], ANCHORS["radial_stress_l2_main"],
                 ANCHORS["displacement_l2_main"]]
    recomputed = [worst["radial_stress_l2"]] * 2 + [worst["displacement_l2"]]
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], [100 * v for v in published], width=0.4,
           label="published anchor", color="#999999")
    ax.bar([i + 0.2 for i in x], [100 * v for v in recomputed], width=0.4,
           label="recomputed on merged kernel", color="#2166ac")
    for i, (p, r) in enumerate(zip(published, recomputed)):
        ax.text(i + 0.2, 100 * r, f"{100 * (r / p - 1):+.1f}%", ha="center",
                va="bottom", fontsize=7.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("relative L2 error [%]")
    ax.set_title(r"Drift check at $\nu = 0.4999$ (labels: recomputed vs anchor)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig_path = REPO / "docs/assets/plane_strain_bbar_locking/union_kernel_drift.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=140)
    print(f"figure -> {fig_path}")

    print(f"\n{'formulation':>12} {'nu':>8} {'displ[%]':>10} {'radial[%]':>10} "
          f"{'hoop[%]':>9} {'bore[%]':>8} {'dof':>6}")
    for r in rows:
        print(f"{r['formulation']:>12} {r['poisson']:>8.4f} {100 * r['displacement_l2']:>10.4f} "
              f"{100 * r['radial_stress_l2']:>10.4f} {100 * r['hoop_stress_l2']:>9.4f} "
              f"{100 * r['bore_displacement_relative']:>8.4f} {r['n_dofs']:>6}")
    print("\ndrift vs anchors:", json.dumps(drift["drift_relative"], indent=2))
    print(f"flatness (0.4999/0.3, B-bar radial) = {drift['flatness_bbar_radial']:.4f}")
    print(f"standard/bbar radial ratio at nu=0.4999 = {drift['locking_ratio_standard_over_bbar']:.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())