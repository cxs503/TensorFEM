"""Figure for H4-ICE-BELT-FEM round 1 (pressure-area law x panel cross-validation
x design-pressure zones).

Reads the run artifact and renders three panels; no number is typed in here, all
values come from ``artifacts/H4-ICE-BELT-FEM/pressure_area_zoned_panel.json``.

    PYTHONPATH=src python scripts/fig_h4_ice_belt_pressure_area.py
"""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ART = pathlib.Path("artifacts/H4-ICE-BELT-FEM/pressure_area_zoned_panel.json")
PNG = pathlib.Path("artifacts/H4-ICE-BELT-FEM/fig_h4_pressure_area.png")


def main() -> int:
    data = json.loads(ART.read_text())
    curve = data["curve_table"]
    areas = np.array([r["area_m2"] for r in curve])
    p_direct = np.array([r["pressure_mpa_direct"] for r in curve])
    p_log = np.array([r["pressure_mpa_logspace"] for r in curve])
    p_marg = np.array([r["pressure_mpa_marginal"] for r in curve])
    p_rad = np.array([r["pressure_mpa_radial_mean"] for r in curve])

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # (a) pressure-area curve, log-log, four routes
    ax = axes[0]
    ax.loglog(areas, p_direct, "o-", color="#1f4e79", lw=2,
              label="direct  $p=C_dC_fp_{ref}(A/A_{ref})^{-\\alpha}$")
    ax.loglog(areas, p_log, "s", ms=6, mfc="none", mec="#c00000", mew=1.4,
              label="log-space route")
    ax.loglog(areas, p_marg, "^", ms=6, mfc="none", mec="#2e7d32", mew=1.4,
              label="marginal route  $\\frac{dF/dA}{1-\\alpha}$")
    ax.loglog(areas, p_rad, "x", ms=7, color="#7b1fa2",
              label="radial-mean route (2-D quadrature)")
    ax.set_xlabel("loaded area $A$ [m$^2$]")
    ax.set_ylabel("average ice pressure $p$ [MPa]")
    c1 = data["criterion_1_pressure_area_curve"]
    ax.set_title("(a) pressure-area law, 4 independent routes\n"
                 f"max route divergence {max(c1['max_rel_diff_logspace'], c1['max_rel_diff_marginal'], c1['max_rel_diff_radial_mean']):.1e}"
                 f" | fitted $\\alpha$ err {c1['fitted_exponent_error']:.1e}", fontsize=9)
    ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.7)
    ax.legend(fontsize=7.5, loc="upper right")

    # (b) mesh convergence of the MITC4 panel against the Navier series
    ax = axes[1]
    study = data["criterion_2_panel_cross_validation"]["mesh_study"]
    n = np.array([r["elements_per_side"] for r in study["rows"]])
    err = np.array([r["rel_error"] for r in study["rows"]]) * 100.0
    ax.semilogx(n, err, "o-", color="#1f4e79", lw=2, label="MITC4 layered shell")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(-10.0, color="#c00000", ls="--", lw=1.2, label="10 % acceptance band")
    cv = data["criterion_2_panel_cross_validation"]
    ff, pt = cv["full_field"], cv["centred_patch"]
    ax.annotate(f"full field: MITC4 vs Q4-SRI {ff['pairwise_rel_diff']['mitc4_vs_q4sri']*100:.2f} %\n"
                f"centred patch: MITC4 vs Q4-SRI {pt['pairwise_rel_diff']['mitc4_vs_q4sri']*100:.2f} %",
                xy=(0.03, 0.05), xycoords="axes fraction", fontsize=8,
                bbox=dict(fc="#f2f2f2", ec="#999999"))
    ax.set_xlabel("elements per side (MITC4, square ice-belt panel)")
    ax.set_ylabel("centre deflection error vs Kirchhoff Navier [%]")
    ax.set_title("(b) panel response: three independent discretisations", fontsize=9)
    ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.7)
    ax.legend(fontsize=8)

    # (c) design-pressure zone division
    ax = axes[2]
    rows = data["pressure_zone_table"]
    zones = []
    for r in rows:
        if r["zone"] not in [z["key"] for z in zones]:
            zones.append({"key": r["zone"], "length": r["zone_length"],
                          "midpoint": r["zone_midpoint"]})
    patches = []
    for r in rows:
        if r["patch"] not in patches:
            patches.append(r["patch"])
    colours = {"plate_field": "#1f4e79", "stiffener": "#e07b00",
               "primary_member": "#2e7d32"}
    width = 0.26
    for i, patch in enumerate(patches):
        xs, ys = [], []
        for z in zones:
            row = next(r for r in rows if r["zone"] == z["key"] and r["patch"] == patch)
            xs.append(z["midpoint"] + (i - 1) * width * 25.0)
            ys.append(row["pressure_mpa"])
        ax.bar(xs, ys, width=width * 25.0, color=colours.get(patch, "#777777"),
               label=patch.replace("_", " "))
    ax.set_xlabel("hull-area zone (midpoint along the ship) [m]")
    ax.set_ylabel("design pressure [MPa]")
    c3 = data["criterion_3_design_pressure_zones"]
    zc = c3["zone_checks"]
    ax.set_title("(c) zone division: pressure per zone x load patch\n"
                 f"Z1-Z4 {'PASS' if c3['partition_check']['passed'] else 'FAIL'}, "
                 f"Z5 {'PASS' if c3['belt_check']['passed'] else 'FAIL'}, "
                 f"Z6-Z8 {'PASS' if zc['passed'] else 'FAIL'}, "
                 f"Z9 {c3['class_check'].get('status', 'n/a')}", fontsize=9)
    ax.grid(True, axis="y", ls=":", lw=0.6, alpha=0.7)
    ax.legend(fontsize=8)

    fig.suptitle("H4-ICE-BELT-FEM round 1 — pressure-area law, panel cross-validation, "
                 "design-pressure zones (demonstration anchor: NOT rule data)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG, dpi=170)
    print(f"wrote {PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())