"""Figures for the ASTM E399 phase-field specimen suite.

Reads ``artifacts/phase_field_fracture/astm_specimens.json`` and writes PNGs into
``artifacts/phase_field_fracture/figures/``:

  fig_load_curves.png   load-displacement with the LEFM peak anchors (CT, SEB)
  fig_sent_conv.png     SENT peak stress vs the regularisation length l
  fig_ct_damage.png     phase-field damage map of the C(T) specimen at the peak
  fig_crack_paths.png   extracted crack paths (CT and SENT)
  fig_staggered.png     alternating-scheme residual history

Numbers are read from the JSON only -- nothing is re-typed by hand.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
ART = os.path.join(ROOT, "artifacts", "phase_field_fracture")
FIG = os.path.join(ART, "figures")


def load():
    with open(os.path.join(ART, "astm_specimens.json")) as fh:
        return json.load(fh)


def fig_load_curves(data):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for ax, key, title in ((axes[0], "ct_peak", "C(T), a/W = 0.5, l = 0.04 W"),
                           (axes[1], "seb_peak", "SE(B), a/W = 0.45, l = 0.04 W")):
        case = data.get(key)
        if not case:
            continue
        recs = case["records"]
        disp = np.array([r["value"] for r in recs])
        force = np.array([abs(r["reaction"]) for r in recs])
        ax.plot(disp, force, "o-", ms=3, lw=1.2, label="phase-field")
        ax.axhline(case["P_pred_astm"], color="crimson", ls="--", lw=1.2,
                   label=f"ASTM E399 prediction {case['P_pred_astm']:.4f}")
        ax.axvline(case["peak"]["value_at_peak"], color="gray", ls=":", lw=0.8)
        ax.set_xlabel("prescribed pin / roller displacement (W)")
        ax.set_ylabel("load P (E W^2)")
        ax.set_title(f"{title}\npeak {case['P_peak']:.4f} vs ASTM {case['P_pred_astm']:.4f}"
                     f" ({case['error_percent']:+.1f} %)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_load_curves.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_sent_conv(data):
    case = data.get("lconv")
    if not case:
        return None
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    l = np.array([r["l"] for r in case["runs"]])
    sig = np.array([r["sigma_peak"] for r in case["runs"]])
    ax.plot(l, sig, "o-", label="phase-field peak stress")
    ax.axhline(case["sigma_griffith"], color="crimson", ls="--",
               label=f"finite-width Griffith {case['sigma_griffith']:.4f}")
    ax.axhline(case["fit"]["limit"], color="green", ls=":",
               label=f"l -> 0 limit {case['fit']['limit']:.4f}")
    for r in case["runs"]:
        ax.annotate(f"{r['error_percent']:+.1f} %", (r["l"], r["sigma_peak"]),
                    textcoords="offset points", xytext=(4, 6), fontsize=7)
    ax.set_xlabel("regularisation length l / W")
    ax.set_ylabel(r"peak stress $\sigma_c$ (E)")
    ax.set_title("SENT, a/W = 0.3: peak stress vs l (monotone shrinking error)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_sent_conv.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_ct_damage(data):
    case = data.get("ct_peak")
    if not case or "field" not in case:
        return None
    nodes = np.array(case["field"]["nodes"])
    d = np.array(case["field"]["d"])
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    sc = ax.scatter(nodes[:, 0], nodes[:, 1], c=d, s=2.0, cmap="inferno", vmin=0.0, vmax=1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x / W")
    ax.set_ylabel("y / W")
    ax.set_title(f"C(T) damage field at the peak load (max d = {d.max():.2f})")
    fig.colorbar(sc, ax=ax, label="phase field d", shrink=0.85)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_ct_damage.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_crack_paths(data):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    ct = data.get("ct_peak")
    if ct and ct.get("crack_path", {}).get("path"):
        p = np.array(ct["crack_path"]["path"])
        axes[0].plot(p[:, 0], p[:, 1], "o-", ms=3)
        axes[0].axhline(0.0, color="gray", ls=":", lw=0.8)
        axes[0].set_xlabel("s (distance from the notch tip, / W)")
        axes[0].set_ylabel("transverse offset t / W")
        axes[0].set_title(f"C(T) crack path, max |t| = {ct['crack_path']['max_abs_t']:.4f} W")
        axes[0].grid(alpha=0.3)
    lc = data.get("lconv")
    if lc:
        finest = min(lc["runs"], key=lambda r: r["l"])
        path = finest.get("crack_path", {}).get("path")
        if path:
            p = np.array(path)
            axes[1].plot(p[:, 0], p[:, 1], "o-", ms=3, color="darkgreen")
            axes[1].set_xlabel("y offset from the notch axis / W")
            axes[1].set_ylabel("transverse offset t / W")
            axes[1].set_title(f"SENT crack path (l = {finest['l']}), "
                              f"max |t| = {finest['crack_path']['max_abs_t']:.4f} W")
            axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_crack_paths.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_staggered(data):
    case = data.get("staggered")
    if not case:
        return None
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for rec in case["alternating"]["records"]:
        hist = rec.get("newton") or []
        if len(hist) > 1:
            ax.semilogy(range(1, len(hist) + 1), hist[1:], lw=0.9)
    ax.axhline(1e-6, color="crimson", ls="--", lw=1.0, label="1e-6 target")
    ax.set_xlabel("alternating iteration")
    ax.set_ylabel(r"$\|R\| / \|R_0\|$")
    ax.set_title("true staggered (u-block / d-block) coupled residual")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_staggered.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main():
    os.makedirs(FIG, exist_ok=True)
    data = load()
    made = {}
    for name, fn in (("load_curves", fig_load_curves), ("sent_conv", fig_sent_conv),
                     ("ct_damage", fig_ct_damage), ("crack_paths", fig_crack_paths),
                     ("staggered", fig_staggered)):
        out = fn(data)
        if out:
            made[name] = out
            print("wrote", out)
    with open(os.path.join(ART, "astm_figures.json"), "w") as fh:
        json.dump(made, fh, indent=1)
    print(json.dumps(made, indent=1))


if __name__ == "__main__":
    main()