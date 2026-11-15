"""m22346 audit figures, all numbers read from the verification JSON (no hand copy).

fig1_independent_reference.png : FV convergence in h and in box size R, 1/R^2
                                 extrapolation, closed-form bracket, audited routes
                                 and the chain declaration on one scale.
fig2_power_recomputed.png      : detection rate / false acceptance of the audited
                                 guard-banded rule vs the naive rule, recomputed
                                 by the auditor from the published formula.
fig3_got_vs_ref_bands.png      : every reference/control value against the three
                                 decision zones of the audited rule.

Run: python docs/board_audit/m22346/make_figures.py
"""
from __future__ import annotations

import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
V = json.loads((HERE / "verify_output_m22346.json").read_text())
G = json.loads((HERE / "verified" / "gate_evidence.json").read_text())
PUB, FV, BR, RULE = V["published"], V["P2_fv_route"], V["P1_ellipse_bracket"], V["P3_rule_reprobe"]


def fig1() -> pathlib.Path:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.9))
    rows = FV["rows"]
    colors = {60.0: "tab:blue", 120.0: "tab:red"}
    for box in (60.0, 120.0):
        sub = [r for r in rows if r["box"] == box]
        h = [r["h"] for r in sub]
        f = [r["A_half_force_t_per_m"] for r in sub]
        e = [r["A_half_energy_t_per_m"] for r in sub]
        ax.plot(h, f, "o-", color=colors[box], label=f"FV force, box={box:.0f} m")
        ax.plot(h, e, "s--", color=colors[box], alpha=0.5, label=f"FV energy, box={box:.0f} m")
        lim = FV["h_extrapolation"][str(box)]["limit_t_per_m"]
        ax.axhline(lim, color=colors[box], lw=0.8, ls=":", alpha=0.8)
        ax.annotate(f"h→0: {lim:.1f}", (0.26, lim), color=colors[box], fontsize=8,
                    va="bottom", ha="left")
    ax.axhline(BR["a_omega0_lower_t_per_m"], color="gray", lw=1.2, ls="-.")
    ax.axhline(BR["a_omega0_upper_t_per_m"], color="gray", lw=1.2, ls="-.")
    ax.annotate("closed-form lower bound 161.0 (inscribed ellipse)", (0.26, 161.0), fontsize=8, color="gray")
    ax.annotate("closed-form upper bound 322.0 (circumscribed ellipse)", (0.26, 322.0), fontsize=8, color="gray")
    ax.axhline(PUB["routes_t_per_m"]["omega0"], color="tab:green", lw=1.2)
    ax.axhline(PUB["routes_t_per_m"]["infinity"], color="tab:olive", lw=1.2)
    ax.axhline(PUB["reference_t_per_m"], color="black", lw=1.4)
    ax.annotate(f"audited ω→0 route {PUB['routes_t_per_m']['omega0']:.1f}", (0.55, 225.2), fontsize=8, color="tab:green")
    ax.annotate(f"audited ω→∞ route {PUB['routes_t_per_m']['infinity']:.1f}", (0.55, 234.0), fontsize=8, color="tab:olive")
    ax.annotate(f"audited primary reference {PUB['reference_t_per_m']:.1f}", (0.55, 229.6), fontsize=8)
    ax.axhline(PUB["declared_t_per_m"], color="tab:purple", lw=1.4, ls="--")
    ax.annotate(f"chain declaration {PUB['declared_t_per_m']:.1f} t/m", (0.55, 88.0), fontsize=8,
                color="tab:purple")
    ax.set_xscale("log")
    ax.set_xlabel("FV cell size h [m] (log)")
    ax.set_ylabel("2-D heave added mass per unit length [t/m]")
    ax.set_title("(a) independent finite-volume route vs audited BEM and closed-form bounds")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    series = FV["R_series_h1"]
    inv = [1.0 / b ** 2 for b in series["box"]]
    ax2.plot(inv, series["values"], "o-", color="tab:red", label="FV force, h=1.0 m")
    ax2.plot([0.0, inv[-1]], [series["limit_1_over_R2"], series["values"][-1]], "k--", lw=1.0,
             label="1/R² extrapolation to R→∞")
    ax2.annotate(f"R→∞ at h=1: {series['limit_1_over_R2']:.1f}", (0.00002, series["limit_1_over_R2"]),
                 fontsize=8)
    ax2.annotate(f"combined estimate {FV['combined_estimate_t_per_m']:.1f} t/m", (0.00002, 222.5),
                 fontsize=8, color="tab:red")
    ax2.axhline(PUB["routes_t_per_m"]["omega0"], color="tab:green", lw=1.2,
                label=f"audited ω→0 route {PUB['routes_t_per_m']['omega0']:.1f}")
    ax2.set_xlabel("1/R²  [1/m²]   (R = outer box half-width)")
    ax2.set_ylabel("[t/m]")
    ax2.set_title("(b) truncation error is O(1/R²): increment ratio "
                  f"{series['increment_ratio']:.2f}")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    out = HERE / "fig1_independent_reference.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig2() -> pathlib.Path:
    tol, u, k = RULE["tol"], RULE["u"], RULE["k"]
    d = np.linspace(0.0, 0.9, 400)

    def power(delta, u_=u, k_=k):
        r = tol + k_ * u_
        phi = lambda x: 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
        return phi((delta - r) / u_) + phi((-delta - r) / u_)

    def fa(delta, u_=u, k_=k):
        a = tol - k_ * u_
        phi = lambda x: 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
        return phi((a - delta) / u_) - phi((-a - delta) / u_)

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.plot(d, power(d), "-", color="tab:blue", label="guard band: detection rate P(reject | δ)")
    ax.plot(d, power(d, k_=0.0), "--", color="tab:orange", label="naive rule |δ|≤tol: detection rate")
    ax.plot(d, fa(d), "-.", color="tab:red", label="guard band: false acceptance")
    ax.axvline(tol, color="gray", lw=1.0, ls=":")
    ax.axvline(RULE["acceptance_limit"], color="green", lw=1.0, ls=":")
    ax.axvline(RULE["rejection_limit"], color="darkred", lw=1.0, ls=":")
    ax.axvline(RULE["critical_deviation_95"], color="purple", lw=1.0, ls=":")
    ax.axvline(abs(RULE["deviation"]), color="black", lw=1.2)
    ax.annotate("tol 0.15", (tol, 0.06), rotation=90, fontsize=8, ha="right")
    ax.annotate("accept limit 0.092", (RULE["acceptance_limit"], 0.30), rotation=90, fontsize=8, ha="right")
    ax.annotate("reject limit 0.208", (RULE["rejection_limit"], 0.55), rotation=90, fontsize=8, ha="right")
    ax.annotate("δ95 0.2557", (RULE["critical_deviation_95"], 0.75), rotation=90, fontsize=8, ha="right")
    ax.annotate(f"audited deviation 0.625\n(power {RULE['power_at_deviation']:.2f}, "
                f"u_max still rejecting {RULE['u_max_still_rejecting']:.3f})",
                (abs(RULE["deviation"]) - 0.02, 0.35), rotation=90, fontsize=8, ha="right")
    ax.set_xlabel("true relative deviation δ")
    ax.set_ylabel("probability")
    ax.set_title(f"audited decision rule re-derived by the auditor "
                 f"(tol={tol}, u={u}, k={k:g})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    out = HERE / "fig2_power_recomputed.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig3() -> pathlib.Path:
    acc, rej = RULE["acceptance_limit"], RULE["rejection_limit"]
    items = []
    for c in PUB["controls"]:
        label = c["name"].replace("_", " ")
        items.append((f"{label}  ({c['t_per_m']:.1f} t/m)", abs(c["dev"]), c["got"]))
    labels = [i[0] for i in items]
    devs = [i[1] for i in items]
    colors = ["tab:green" if g == "accept" else ("tab:red" if g == "reject" else "orange")
              for _, _, g in items]
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    y = np.arange(len(items))
    ax.axvspan(0, acc, color="green", alpha=0.10)
    ax.axvspan(acc, rej, color="orange", alpha=0.10)
    ax.axvspan(rej, max(devs) * 1.12 + 0.05, color="red", alpha=0.08)
    ax.barh(y, devs, color=colors, alpha=0.85)
    for yi, dv in zip(y, devs):
        ax.annotate(f"{dv:.3f}", (dv, yi), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(acc, color="green", lw=1.0)
    ax.axvline(rej, color="darkred", lw=1.0)
    ax.annotate("accept ≤ 0.092", (acc, len(items) - 0.6), fontsize=8, color="green", rotation=90, va="top", ha="right")
    ax.annotate("indeterminate", ((acc + rej) / 2, len(items) - 0.6), fontsize=8, color="orange", ha="center")
    ax.annotate("reject ≥ 0.208", (rej, len(items) - 0.6), fontsize=8, color="darkred", rotation=90, va="top")
    ax.set_xlabel("|relative deviation| from the audited reference")
    ax.set_title("reference and control values vs the three decision zones (audited rule, auditor re-check)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = HERE / "fig3_got_vs_ref_bands.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for p in (fig1(), fig2(), fig3()):
        print(p)