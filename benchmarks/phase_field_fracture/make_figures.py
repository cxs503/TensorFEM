"""Figures for the phase-field fracture verification report.

Reads the JSON artifacts written by ``run_suite.py`` and produces PNG files in
``artifacts/phase_field_fracture/figures``.
"""

from __future__ import annotations

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "..", "artifacts", "phase_field_fracture")
FIG = os.path.join(OUT, "figures")


def load(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def fig_damage(kink):
    if not kink:
        return None
    xy = np.array(kink["xy"])
    d = np.array(kink["d_field"])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=d, s=6, cmap="inferno", vmin=0.0, vmax=1.0)
    beta = math.radians(kink["beta_deg"])
    a = 0.35
    ax.plot([0, a * math.cos(beta)], [0.25, 0.25 + a * math.sin(beta)], "w--", lw=1.2,
            label="initial notch")
    th = math.radians(kink["theta_phase_field_deg"]) if not math.isnan(
        kink["theta_phase_field_deg"]) else 0.0
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("phase-field damage, inclined crack (beta = %.0f deg)" % kink["beta_deg"])
    ax.legend(loc="lower right", fontsize=7)
    fig.colorbar(sc, ax=ax, label="d")
    path = os.path.join(FIG, "damage_inclined_crack.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_load_displacement(lconv):
    if not lconv:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for run in lconv["runs"]:
        tr = run["trace"]
        delta = [t["value"] for t in tr]
        load = [abs(t["reaction"]) for t in tr]
        ax.plot(delta, load, "o-", ms=3, label="l = %.3f (h = %.3f)" % (run["l"], run["h"]))
    ax.set_xlabel("prescribed end displacement [mm]")
    ax.set_ylabel("reaction P [N]")
    ax.set_title("SENT load-displacement, compact-band width study")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    path = os.path.join(FIG, "sent_load_displacement.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_l_convergence(lconv):
    if not lconv:
        return None
    ls = [r["l"] for r in lconv["runs"]]
    ratios = [r["G_ratio"] for r in lconv["runs"]]
    dev = [r["deviation_percent"] for r in lconv["runs"]]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(ls, ratios, "s-", color="C0")
    ax.axhline(1.0, ls="--", color="k", lw=0.8, label="G_app = G_c")
    ax.set_xlabel("compact band width l [mm]")
    ax.set_ylabel("G_app / G_c")
    ax2 = ax.twinx()
    ax2.bar(ls, dev, width=min(ls) * 0.25, alpha=0.25, color="C1")
    ax2.set_ylabel("deviation [%]")
    ax.set_title("apparent toughness vs l (monotone l -> 0 convergence)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    path = os.path.join(FIG, "l_convergence.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_newton(lconv):
    if not lconv:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for run in lconv["runs"]:
        hist = run["newton_history"]
        chosen = max(hist, key=len)
        its = np.arange(len(chosen))
        res = np.array(chosen)
        keep = res > 0
        ax.semilogy(its[keep], res[keep], "o-", ms=3, label="l = %.3f" % run["l"])
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("|| R || / || R_0 ||")
    ax.set_title("monolithic Newton residual history (last load step)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    path = os.path.join(FIG, "newton_history.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    os.makedirs(FIG, exist_ok=True)
    def unwrap(obj, key):
        if obj and key in obj:
            return obj[key]
        return obj

    lconv, tpb, kink = load("lconv.json"), load("tpb.json"), load("kink.json")
    lconv = unwrap(lconv, "lconv")
    tpb = unwrap(tpb, "tpb")
    kink = unwrap(kink, "kink")
    made = []
    for func, arg in ((fig_damage, kink), (fig_load_displacement, lconv),
                      (fig_l_convergence, lconv), (fig_newton, lconv)):
        try:
            p = func(arg)
        except Exception as exc:                    # pragma: no cover - diagnostics only
            print("figure failed:", func.__name__, exc)
            p = None
        if p:
            made.append(p)
            print("wrote", p)
    return made


if __name__ == "__main__":
    main()