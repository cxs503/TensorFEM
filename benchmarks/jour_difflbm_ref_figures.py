#!/usr/bin/env python3
"""Figures for JOUR-DIFFLBM-REF-VERIFY (citation gate + forcing benchmark).

Reads the artifacts written by ``benchmarks/jour_difflbm_ref_verify.py``
(``ref_verify.json``, ``forcing_poiseuille.json``) and produces a four-panel
figure:

(a) steady Poiseuille profile, analytic vs Guo vs uncorrected forcing;
(b) grid convergence of the Guo scheme on log-log axes with a second-order
    reference slope;
(c) the discrete lattice effect as a function of ``tau``: the uncorrected
    scheme's amplitude ratio against ``1 + 1/(2 tau)``, and the corrected
    scheme's ratio (flat at 1);
(d) the citation verification matrix (reference x bibliographic field).

    PYTHONPATH=src python benchmarks/jour_difflbm_ref_figures.py
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
import numpy as np                                 # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
ART = REPO / "artifacts" / "JOUR-DIFFLBM-REF-VERIFY"


def _load(name: str) -> dict:
    path = ART / name
    if not path.exists():
        raise SystemExit(f"missing artifact {path} — run "
                         "benchmarks/jour_difflbm_ref_verify.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def _row(rows: list[dict], scheme: str, solver: str = "PoiseuilleLBM",
         ny: int | None = None) -> dict | None:
    cand = [r for r in rows if r["scheme"] == scheme and r["solver"] == solver
            and (ny is None or r["ny"] == ny)]
    if not cand:
        return None
    return max(cand, key=lambda r: r["ny"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path,
                    default=ART / "fig_ref_verify.png")
    args = ap.parse_args(argv)

    bench = _load("forcing_poiseuille.json")
    cit = _load("ref_verify.json")
    rows = bench["rows"]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))

    # (a) profiles -----------------------------------------------------------
    ax = axes[0][0]
    guo = _row(rows, "guo")
    naive = _row(rows, "naive", ny=guo["ny"] if guo else None)
    if guo:
        ax.plot(guo["u_ref"], guo["y"], "k-", lw=2.0,
                label="analytic $F_x(y-1)(n_y-1-y)/(2\\nu)$")
        ax.plot(guo["u_profile"], guo["y"], "o", ms=4.5, mfc="none",
                label=f"Guo corrected ($n_y$={guo['ny']}, "
                      f"err {100 * guo['rel_err_u_max']:.2f}%)")
    if naive:
        ax.plot(naive["u_profile"], naive["y"], "s", ms=4.0, alpha=0.75,
                label=f"uncorrected ($n_y$={naive['ny']}, "
                      f"err {100 * naive['rel_err_u_max']:.1f}%)")
    ax.set_xlabel("$u_x$ [lattice]")
    ax.set_ylabel("$y$ [lattice]")
    ax.set_title("(a) force-driven Poiseuille: ref vs got")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # (b) grid convergence ---------------------------------------------------
    ax = axes[0][1]
    guo_rows = sorted((r for r in rows if r["scheme"] == "guo"
                       and r["solver"] == "PoiseuilleLBM"), key=lambda r: r["ny"])
    if guo_rows:
        h = np.array([1.0 / (r["ny"] - 2) for r in guo_rows])
        e = np.array([r["rel_l2_profile"] for r in guo_rows])
        ax.loglog(h, e, "o-", label="Guo corrected (L2 profile error)")
        order = bench.get("convergence_order") or {}
        ref = e[0] * (h / h[0]) ** 2
        ax.loglog(h, ref, "k--", lw=1.0, label="slope 2 reference")
        ax.set_xlabel("$h = 1/(n_y-2)$")
        ax.set_ylabel("$\\|u_{num}-u_{exact}\\|_2 / \\|u_{exact}\\|_2$")
        title = "(b) grid convergence"
        if order.get("mean_order"):
            title += f": observed order {order['mean_order']:.2f}"
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    prod = bench.get("production")
    if prod:
        ax.annotate(f"shiping solver (ChannelLBM, $n_y$={prod['ny']}): "
                    f"{100 * prod['rel_err_u_max']:.2f}%",
                    xy=(0.03, 0.80), xycoords="axes fraction", fontsize=8)

    # (c) tau dependence -----------------------------------------------------
    ax = axes[1][0]
    sweep = bench.get("tau_sweep") or []
    if sweep:
        taus = np.array([s["tau"] for s in sweep])
        naive_r = np.array([s["naive_amplitude_ratio"] for s in sweep])
        guo_r = np.array([s["guo_amplitude_ratio"] for s in sweep])
        tt = np.linspace(taus.min() * 0.9, taus.max() * 1.05, 100)
        ax.plot(tt, 1.0 + 0.5 / tt, "k-", lw=1.2,
                label="discrete lattice effect $1+1/(2\\tau)$")
        ax.plot(taus, naive_r, "s", ms=7, label="uncorrected scheme")
        ax.plot(taus, guo_r, "o", ms=7, label="Guo corrected scheme")
        ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
        ax.set_xlabel("$\\tau$")
        ax.set_ylabel("$u_{max,num} / u_{max,exact}$")
        ax.set_title("(c) the correction removes a $\\tau$-dependent "
                     "amplitude error")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # (d) citation matrix ----------------------------------------------------
    ax = axes[1][1]
    keys, fields, mat = [], [], []
    for key, ev in cit["registry_evidence"].items():
        keys.append(key)
        vals = ev["fields"]
        if not fields:
            fields = list(vals)
            mat = [[] for _ in keys[:-1]]
        mat.append([float(vals.get(f, False)) for f in fields])
    if keys:
        data = np.array(mat)
        ax.imshow(data, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(fields)))
        ax.set_xticklabels(fields, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(keys)))
        labels = [f"{k} (HTTP {cit['registry_evidence'][k]['http']})"
                  for k in keys]
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, "OK" if data[i, j] else "X", ha="center",
                        va="center", fontsize=7)
        ax.set_title("(d) citation x field verification vs Crossref")
        withdrawn = ", ".join(cit["withdrawn_dois"])
        ax.annotate(f"withdrawn (unregistered): {withdrawn}",
                    xy=(0.0, -0.28), xycoords="axes fraction", fontsize=7.5,
                    color="firebrick")

    fig.suptitle("JOUR-DIFFLBM-REF-VERIFY — citation gate + Guo (2002) forcing "
                 "benchmark", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())