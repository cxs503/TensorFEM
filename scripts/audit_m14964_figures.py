"""Audit m14964 — figures for the m14685 maturity cross-audit.

All numbers are read from the JSON artifacts (no hand transcription):

* original run  : /root/tp-agent/work/m14685/artifacts/parallel_scaling/{strong,strong_800,strong_1280}.json
* reproduction  : artifacts/audit_m14964/repro_strong_{640,1280}/strong.json

Figure 1  measured parallel efficiency vs the predeclared 0.60@4-rank gate,
          original vs reproduced, three problem sizes.
Figure 2  CG "residual" field of the artifacts vs iteration budget: it does not
          depend on the budget at all -> the field is ||b||, not a residual.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "artifacts", "audit_m14964")
ORIG = "/root/tp-agent/work/m14685/artifacts/parallel_scaling"

SERIES = [
    ("orig 640^2 (20 it)", os.path.join(ORIG, "strong.json"), "tab:blue", "-"),
    ("orig 800^2 (10 it)", os.path.join(ORIG, "strong_800.json"), "tab:blue", "--"),
    ("orig 1280^2 (10 it)", os.path.join(ORIG, "strong_1280.json"), "tab:blue", ":"),
    ("repro 640^2 (20 it)", "artifacts/audit_m14964/repro_strong_640/strong.json", "tab:red", "-"),
    ("repro 1280^2 (10 it)", "artifacts/audit_m14964/repro_strong_1280/strong.json", "tab:red", "--"),
]


def load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def fig_efficiency() -> str:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for label, path, color, style in SERIES:
        d = load(path)
        r = [p["ranks"] for p in d["points"]]
        e = [p["efficiency"] for p in d["points"]]
        ax.plot(r, e, style, marker="o", color=color, label=label, lw=1.4, ms=5)
    ax.axhline(0.60, color="k", ls="--", lw=1.2, label="predeclared gate E(4) >= 0.60")
    ax.set_xlabel("ranks p")
    ax.set_ylabel("parallel efficiency E(p)")
    ax.set_title("m14685 GAP-FEM-PARALLEL: gate metric is not reproducible\n"
                 "same host, same code, same command (20 it / 10 it fixed CG budget)")
    ax.set_xticks([1, 2, 4, 8])
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower left")
    p = os.path.join(OUT, "efficiency_orig_vs_repro.png")
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def fig_residual_bug() -> str:
    """Call the solver directly: reported 'residual' is invariant in the budget."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import torch
    from tensorfem.parallel.poisson import DistPoisson

    nx = 64
    d = DistPoisson(nx=nx)
    b = d.assemble_rhs()
    bnorm = float(torch.sqrt((b * b).sum()))
    budgets = [1, 2, 5, 10, 25, 50, 93]
    reported = [d.cg(b, iters=k)["residual"] for k in budgets]
    l2 = [d.l2_error(d.cg(b, iters=k)["x"]) for k in budgets]

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax[0].plot(budgets, reported, "o-", label="cg()['residual']")
    ax[0].axhline(bnorm, color="k", ls="--", lw=1, label="||b||_2 (pre-solve)")
    ax[0].set_xlabel("CG iteration budget")
    ax[0].set_ylabel("reported residual")
    ax[0].set_title("returned 'residual' == ||b||, never updated")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].semilogy(budgets, l2, "s-", color="tab:green")
    ax[1].set_xlabel("CG iteration budget")
    ax[1].set_ylabel("L2 error vs manufactured solution")
    ax[1].set_title("true error does depend on the budget\n(so the solve itself is fine)")
    ax[1].grid(alpha=0.3)
    p = os.path.join(OUT, "residual_field_bug.png")
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    print(fig_efficiency())
    print(fig_residual_bug())


if __name__ == "__main__":
    main()