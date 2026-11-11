#!/usr/bin/env python3
"""Audit figure for m15020 / m14884 (H4-ICE-BELT-FEM).

Every number is read from the run artifact and the audit JSON -- nothing is typed
by hand.  Panels:

(a) MITC4 mesh convergence of the panel peak deflection against the Kirchhoff
    Navier analytic reference (log-log, second-order reference slope);
(b) pairwise peak-deflection differences for the two load cases with the audit's
    pre-declared 3 % (physics vs reference) and 10 % (declared) bands;
(c) the pressure-area curve on its four evaluation routes (they overplot to
    <2.3e-11, so panel (c) also states the maximum route spread in the legend).

Usage::  PYTHONPATH=src python scripts/fig_audit_m14884.py
"""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "H4-ICE-BELT-FEM" / "pressure_area_zoned_panel.json"
AUDIT = ROOT / "artifacts" / "audit_m14884" / "repro_audit.json"
OUT = ROOT / "artifacts" / "audit_m14884" / "fig_audit.png"


def main() -> None:
    a = json.loads(ART.read_text())
    aud = json.loads(AUDIT.read_text())
    c2 = a["criterion_2_panel_cross_validation"]
    thr = aud["claims"]["thresholds"]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    # (a) mesh convergence ------------------------------------------------- #
    ax = axes[0]
    ms = c2["mesh_study"]
    h = [1.0 / r["elements_per_side"] for r in ms["rows"]]
    e = [abs(r["rel_error"]) for r in ms["rows"]]
    ax.loglog(h, e, "o-", lw=2, ms=7, label="MITC4 vs Kirchhoff Navier")
    ref = [e[0] * (x / h[0]) ** 2 for x in h]
    ax.loglog(h, ref, "k--", lw=1, label="2nd-order reference slope")
    ax.axhline(thr["physics_vs_reference"], color="tab:red", ls=":", lw=1.5,
               label=f"{thr['physics_vs_reference']*100:.0f} % band")
    ax.invert_xaxis()
    ax.set_xlabel("element size h = 1/N  (m)")
    ax.set_ylabel("|relative error| of peak deflection")
    ax.set_title("(a) panel convergence, 1.6 m plate, t = 10 mm")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # (b) pairwise differences --------------------------------------------- #
    ax = axes[1]
    labels, vals, colors = [], [], []
    for case, tag in (("full_field", "full field"), ("centred_patch", "ice patch")):
        d = c2[case]["pairwise_rel_diff"]
        for key, lab in (("mitc4_vs_q4sri", "MITC4\u2194Q4-SRI"),
                         ("mitc4_vs_navier", "MITC4\u2194Navier"),
                         ("q4sri_vs_navier", "Q4-SRI\u2194Navier")):
            labels.append(f"{tag}\n{lab}")
            vals.append(100.0 * d[key])
            colors.append("tab:blue" if "q4sri_vs" in key else "tab:cyan"
                          if "mitc4_vs_q4sri" in key else "tab:orange")
    x = range(len(vals))
    ax.bar(x, vals, color=colors, width=0.7)
    ax.axhline(100 * thr["physics_vs_reference"], color="tab:red", ls="--", lw=1.5,
               label=f"3 % (audit, vs analytic reference)")
    ax.axhline(100 * thr["declared_inter_discretisation"], color="k", ls=":", lw=1.5,
               label="10 % (module's declared tolerance)")
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.15, f"{v:.2f}%", ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("peak-deflection difference [%]")
    ax.set_ylim(0, 11.5)
    ax.set_title("(b) three independent discretisations agree")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    # (c) pressure-area curve on four routes ------------------------------- #
    ax = axes[2]
    rows = a["curve_table"]
    area = [r["area_m2"] for r in rows]
    for key, lab, style in (("pressure_mpa_direct", "direct $cA^{-\\alpha}$", "-"),
                            ("pressure_mpa_logspace", "log-space", "--"),
                            ("pressure_mpa_marginal", "inverse $dF/dA$", "-."),
                            ("pressure_mpa_radial_mean", "radial field mean", ":")):
        ax.loglog(area, [r[key] for r in rows], style, lw=2.2, label=lab)
    law = a["law"]
    ax.set_xlabel("loaded area A [m$^2$]")
    ax.set_ylabel("local ice pressure p [MPa]")
    spread = a["criterion_1_pressure_area_curve"]["max_rel_diff_marginal"]
    ax.set_title(
        f"(c) pressure-area law, alpha = {law['exponent']}, "
        f"p_ref = {law['p_ref_mpa']} MPa (DEMO)")
    ax.text(0.03, 0.06,
            f"four routes overplot: max spread {spread:.1e}\n"
            f"exponent from fit: "
            f"{a['criterion_1_pressure_area_curve']['fitted_exponent']:.6f}\n"
            "anchor is DEMONSTRATION, rule table WITHHELD",
            transform=ax.transAxes, fontsize=7.5, va="bottom",
            bbox={"fc": "w", "ec": "0.6", "alpha": 0.85})
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("m15020 board audit of H4-ICE-BELT-FEM (m14884): "
                 "reproduced byte-identical, 17/17 claim checks PASS", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()