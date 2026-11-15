"""Figures for the W30 ``ice-coupled-6dof`` load ensemble vs the IACS design load.

Reads only artefacts (never recomputes physics):

    papers/artifacts/w30/ice-coupled-6dof/ice_load_vs_iso19906.json
    papers/artifacts/w30/ice-coupled-6dof/ensemble_per_impact.csv
    papers/artifacts/w30/ice-coupled-6dof/representative_timeseries.csv

Writes PNGs to ``papers/artifacts/w30/ice-coupled-6dof/figures/``.  Every axis
label carries the source artefact so a reader cannot mistake a fitted curve for
measured data.

    python benchmarks/ice_coupled_6dof/make_w30_figures.py
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ART = ROOT / "papers" / "artifacts" / "w30" / "ice-coupled-6dof"
FIG = ART / "figures"

ARTIFACT = ART / "ice_load_vs_iso19906.json"
PER_IMPACT = ART / "ensemble_per_impact.csv"
SERIES = ART / "representative_timeseries.csv"


def read_csv(path):
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    if not ARTIFACT.is_file():
        print("run benchmarks/ice_coupled_6dof/run_w30_ice_load_vs_code.py first",
              file=sys.stderr)
        return 2
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    groups = data["groups"]
    names = [g["group"]["name"] for g in groups]
    code = data["code_side"]["per_class"]["PC6"]
    f_code = code["design_force_MN"]
    p_code = code["design_pressure_MPa"]
    a_code = code["patch_area_m2"]
    iso = data["anchors"]["iso_19906"]["declared_values"]
    FIG.mkdir(parents=True, exist_ok=True)

    # ---- 1. peak-force distributions with the fits and the code value ----- #
    rows = read_csv(PER_IMPACT)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, g in zip(axes.ravel(), groups):
        gname = g["group"]["name"]
        x = np.array([float(r["peak_force_MN"]) for r in rows if r["group"] == gname])
        ax.hist(x, bins=40, density=True, alpha=0.55, color="tab:blue",
                label=f"per-impact peaks (n={x.size})")
        grid = np.linspace(max(x.min(), 1e-3), max(x.max() * 1.15, f_code * 1.1), 400)
        wf = g["distribution_fits"]["weibull"]["params"]
        gf = g["distribution_fits"]["gumbel"]["params"]
        ax.plot(grid, (wf["k"] / wf["lam"]) * (grid / wf["lam"]) ** (wf["k"] - 1)
                * np.exp(-(grid / wf["lam"]) ** wf["k"]), "b-", lw=1.6,
                label=f"Weibull k={wf['k']:.2f} (KS p={g['distribution_fits']['weibull']['ks_pvalue']:.2f})")
        ax.plot(grid, np.exp(-(grid - gf["loc"]) / gf["scale"]
                             - np.exp(-(grid - gf["loc"]) / gf["scale"])) / gf["scale"],
                "g--", lw=1.4,
                label=f"Gumbel (KS p={g['distribution_fits']['gumbel']['ks_pvalue']:.2f})")
        ax.axvline(f_code, color="red", lw=2,
                   label=f"IACS PC6 design force {f_code:.2f} MN")
        ax.set_xscale("log")
        ax.set_title(f"{gname}  (c={g['ice_field']['concentration']}, "
                     f"h={g['ice_field']['mean_thickness_m']} m, "
                     f"D={g['ice_field']['median_floe_diameter_m']} m)")
        ax.set_xlabel("peak contact force per impact [MN]  (log scale)\n"
                      "source: ensemble_per_impact.csv")
        ax.set_ylabel("probability density")
        ax.legend(fontsize=6.5, loc="upper left")
    fig.suptitle("Random-ice 6-DOF ensemble: per-impact peak ice load vs the IACS UR I2 design load")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_peak_force_distribution.png", dpi=150)
    plt.close(fig)

    # ---- 2. pressure--area plane ----------------------------------------- #
    fig, ax = plt.subplots(figsize=(8.5, 6))
    colours = ["tab:blue", "tab:green", "tab:orange", "tab:red"]
    for c, g in zip(colours, groups):
        gname = g["group"]["name"]
        a = np.array([float(r["peak_area_m2"]) for r in rows if r["group"] == gname])
        p = np.array([float(r["peak_pressure_MPa"]) for r in rows if r["group"] == gname])
        ax.scatter(a, p, s=9, alpha=0.55, color=c,
                   label=f"{gname} (fitted slope {g['statistics']['pressure_area_regression']['slope']:.3f})")
    grid = np.geomspace(min(float(r["peak_area_m2"]) for r in rows) * 0.5,
                        max(a_code * 2, 1.0), 300)
    ax.plot(grid, iso["C_R_MPa"] * (grid / iso["A_0_m2"]) ** iso["ex"], "k-", lw=2,
            label=f"ISO 19906 form, DECLARED ex={iso['ex']} (non-normative, closure arm)")
    ax.scatter([a_code], [p_code], marker="*", s=260, color="red", zorder=5,
               label=f"IACS PC6 design point: Pavg={p_code:.2f} MPa at A={a_code:.2f} m2")
    # the IACS-implied trend, anchored at the PC6 design point
    ex_code = data["code_side"]["code_pressure_area_exponent"]["dlnp_dlnA"]
    ax.plot(grid, p_code * (grid / a_code) ** ex_code, "r-.", lw=1.8,
            label=f"IACS-implied trend dlnP/dlnA=+{ex_code:.3f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("peak contact area per impact [m2]\nsource: ice_load_vs_iso19906.json")
    ax.set_ylabel("peak contact pressure [MPa]")
    ax.set_title("Pressure--area plane: ensemble, declared ISO law, IACS design point")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_pressure_area.png", dpi=150)
    plt.close(fig)

    # ---- 3. seed-axis convergence ---------------------------------------- #
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for c, g in zip(colours, groups):
        run_mean = g["statistics"]["convergence"]["running_mean_force_MN"]
        k = np.arange(1, len(run_mean) + 1)
        final = run_mean[-1]
        ax.plot(k, np.array(run_mean) / final, "-o", ms=3, color=c,
                label=f"{g['group']['name']} (tail diff "
                      f"{g['statistics']['convergence']['mean_force_rel_diff']:+.2%})")
    ax.axhspan(0.95, 1.05, color="grey", alpha=0.2, label="+/-5% band around the full mean")
    ax.set_xlabel("number of seeds in the running mean\nsource: ice_load_vs_iso19906.json")
    ax.set_ylabel("running mean peak force / full-ensemble mean")
    ax.set_title("Ensemble convergence on the seed axis (last-10-seed criterion)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_convergence.png", dpi=150)
    plt.close(fig)

    # ---- 4. representative time histories -------------------------------- #
    series = read_csv(SERIES)
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for ax, g in zip(axes, groups):
        gname = g["group"]["name"]
        t = np.array([float(r["time_s"]) for r in series
                      if r["group"] == gname and r["quantity"] == "ice_force_MN"])
        f = np.array([float(r["value"]) for r in series
                      if r["group"] == gname and r["quantity"] == "ice_force_MN"])
        ax.plot(t, f, lw=0.7, color="tab:blue")
        ax.axhline(f_code, color="red", lw=1.2, ls="--",
                   label=f"IACS PC6 design force {f_code:.2f} MN")
        ax.set_ylabel("ice force [MN]")
        ax.set_title(f"{gname}: representative seed "
                     f"(c={g['ice_field']['concentration']}, "
                     f"h={g['ice_field']['mean_thickness_m']} m)", fontsize=9)
        ax.legend(fontsize=7)
    axes[-1].set_xlabel("time [s]\nsource: representative_timeseries.csv")
    fig.suptitle("Time-domain hull-ice contact force, one representative seed per group")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_timeseries.png", dpi=150)
    plt.close(fig)

    # ---- 5. per-item deviation bars --------------------------------------- #
    items = [(i["item"], i["deviation_pct"])
             for g in groups for i in g["comparison_items"]
             if i.get("deviation_pct") is not None]
    labels = sorted({i[0] for i in items})
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / len(groups)
    for j, (c, g) in enumerate(zip(colours, groups)):
        vals = []
        for lab in labels:
            hit = [i["deviation_pct"] for i in g["comparison_items"] if i["item"] == lab
                   and i.get("deviation_pct") is not None]
            vals.append(hit[0] if hit else np.nan)
        ax.bar(np.arange(len(labels)) + j * width, vals, width, color=c,
               label=g["group"]["name"])
    ax.axhspan(-15, 15, color="grey", alpha=0.25, label="+/-15% gate")
    ax.set_xticks(np.arange(len(labels)) + 0.4 - width / 2)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("deviation from the code value [%]")
    ax.set_title("Per-item deviation from IACS UR I2 design values (negative = ensemble below code)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_deviations.png", dpi=150)
    plt.close(fig)

    print(f"wrote 5 figures to {FIG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())