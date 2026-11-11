"""Figures for W22 / m14511 podded-ice manoeuvring study.

Every number plotted is read from ``docs/w22_podded_ice/turning_zigzag.json``
(the artefact produced by ``benchmarks/podded_ice_maneuvering_w22.py``); nothing
is typed in by hand.  Run::

    python benchmarks/w22_podded_ice_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "docs" / "w22_podded_ice" / "turning_zigzag.json"
FIGS = REPO / "docs" / "w22_podded_ice" / "figs"
COL = {"conventional": "#1f4e79", "podded": "#c00000"}
LABEL = {"conventional": "single screw + rudder", "podded": "twin azimuthing pods"}


def fig_tracks(art: dict) -> Path:
    run = art["runs"]
    limits = {}
    for cfg in run:
        for row in run[cfg]["imo_verdicts"]:
            limits[row["quantity"]] = row["limit"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    ax = axes[0]
    for cfg in ("conventional", "podded"):
        t = run[cfg]["turning_35deg"]
        ax.plot(t["track"]["y_over_L"], t["track"]["x_over_L"], color=COL[cfg], lw=1.8,
                label=f"{LABEL[cfg]}: adv {t['advance_over_L']:.2f} L, "
                      f"$D_T$ {t['tactical_diameter_over_L']:.2f} L")
        ax.plot([0.0], [0.0], marker="o", color="k", ms=4)
    ax.set_xlabel("transfer $y/L$")
    ax.set_ylabel("advance $x/L$")
    ax.set_title("35$^\\circ$ turning circle, KVLCC2 hull (full scale)")
    ax.axvline(limits["turning.tactical_diameter_over_L"] / 2.0, color="grey", ls="--", lw=1,
               label=r"IMO $\leq$5$L$ $D_T$ ($\pm$2.5$L$ transfer)")
    ax.axhline(limits["turning.advance_over_L"], color="grey", ls=":", lw=1,
               label=r"IMO advance $\leq$4.5$L$")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    ax = axes[1]
    for cfg in ("conventional", "podded"):
        z = run[cfg]["zigzag_10_10"]
        h = z["history"]
        ax.plot(h["t"], h["psi_deg"], color=COL[cfg], lw=1.5,
                label=f"{LABEL[cfg]}: overshoot "
                      f"{z['first_overshoot_deg']:.2f}$^\\circ$ / "
                      f"{z['second_overshoot_deg']:.2f}$^\\circ$")
        ax.plot(h["t"], h["delta_deg"], color=COL[cfg], lw=0.8, ls=":")
    ax.axhline(10.0, color="grey", ls="--", lw=1, label="trigger heading $\\pm$10$^\\circ$")
    ax.axhline(-10.0, color="grey", ls="--", lw=1)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("heading / helm [deg]")
    ax.set_title("10/10 zig-zag: heading (solid) and helm (dotted)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGS / "w22_tracks_zigzag.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_sensitivity(art: dict) -> Path:
    rows = art["sensitivity"]["coefficient_sweep"]
    labels = [str(r["value"]) for r in rows]
    conv_dt = [r["conv_DT_over_L"] for r in rows]
    pod_dt = [r["pod_DT_over_L"] for r in rows]
    conv_ad = [r["conv_advance_over_L"] for r in rows]
    pod_ad = [r["pod_advance_over_L"] for r in rows]
    ice = art["sensitivity"]["podded_ice_load_sweep"]
    limits = {}
    for row in art["runs"]["conventional"]["imo_verdicts"]:
        limits[row["quantity"]] = row["limit"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    x = range(len(labels))
    ax = axes[0]
    ax.bar([i - 0.2 for i in x], conv_ad, width=0.38, color=COL["conventional"],
           label="single screw")
    ax.bar([i + 0.2 for i in x], pod_ad, width=0.38, color=COL["podded"], label="pods")
    ax.axhline(limits["turning.advance_over_L"], color="k", ls="--", lw=1,
               label="IMO 4.5 L")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("advance / L")
    ax.set_title("Advance vs hull-derivative variant")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar([i - 0.2 for i in x], conv_dt, width=0.38, color=COL["conventional"])
    ax.bar([i + 0.2 for i in x], pod_dt, width=0.38, color=COL["podded"])
    ax.axhline(limits["turning.tactical_diameter_over_L"], color="k", ls="--", lw=1,
               label="IMO 5 L")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("tactical diameter / L")
    ax.set_title("Tactical diameter vs hull-derivative variant")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    ax.plot([r["ice_resistance_fraction"] for r in ice],
            [r["advance_over_L"] for r in ice], "o-", color=COL["podded"],
            label="podded advance / L")
    ax.plot([r["ice_resistance_fraction"] for r in ice],
            [r["tactical_diameter_over_L"] for r in ice], "s-", color=COL["conventional"],
            label="podded $D_T$ / L")
    ax2 = ax.twinx()
    ax2.plot([r["ice_resistance_fraction"] for r in ice],
             [r["u_ss_over_u0"] for r in ice], "^:", color="grey",
             label="steady $u/u_0$")
    ax2.set_ylabel("$u/u_0$ [-]")
    ax.set_xlabel("declared ice resistance fraction $\\Delta R_{ice}/R(U_0)$")
    ax.set_ylabel("lengths")
    ax.set_title("Podded configuration: declared ice-load scenario")
    ax.legend(fontsize=8, loc="upper right")
    ax2.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGS / "w22_sensitivity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    art = json.loads(ART.read_text(encoding="utf-8"))
    for path in (fig_tracks(art), fig_sensitivity(art)):
        print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()