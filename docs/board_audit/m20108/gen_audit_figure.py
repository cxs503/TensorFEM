#!/usr/bin/env python3
"""Audit figure for the m20054 (CASE-FEM-HEAT-L2) board verification.

Panel A: L2 error ladder of the sine MMS -- the audited numbers (m20054
evidence.json) next to the board's independent numpy Q4 / shipped-kernel
re-computation, with an O(h^2) reference line.
Panel B: observed L2 order per refinement pair for every discriminating case,
recomputed by the board directly from the raw error arrays.

Output: docs/board_audit/m20108/figures/m20108_heat_l2_crosscheck.png
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
AUDITED = Path("/root/tp-agent/work/m20054/artifacts/CASE-FEM-HEAT-L2/evidence.json")
OUT = HERE / "figures" / "m20108_heat_l2_crosscheck.png"


def orders(hs, errs):
    return [math.log(errs[i] / errs[i + 1]) / math.log(hs[i] / hs[i + 1])
            for i in range(len(errs) - 1)]


def main() -> None:
    ref = json.loads(AUDITED.read_text())
    mine = json.loads((HERE / "independent_heat_l2_check.json").read_text())
    sine = ref["studies"]["sine_generation"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # ---- Panel A
    ax.loglog(sine["h"], sine["l2_relative"], "o-", lw=2.2, ms=7,
              label="m20054 evidence.json (audited)")
    styles = {"own_element_mean_source": ("s", "tab:orange", "board: independent numpy Q4 (element-mean source)"),
              "shipped_kernel": ("^", "tab:green", "board: shipped tensorfem.thermal kernel, board driver"),
              "own_exact_load": ("d", "tab:red", "board: numpy Q4, exactly integrated load")}
    for key, (mk, col, lab) in styles.items():
        v = mine["variants"][key]
        ax.loglog(v["h"], v["l2_relative"], mk + "--", color=col, lw=1.5, ms=6,
                  mfc="none", label=lab)
    h = sine["h"]
    anchor = sine["l2_relative"][0]
    ax.loglog(h, [anchor * (hi / h[0]) ** 2 for hi in h], "k:", lw=1.2, label=r"$O(h^2)$ reference")
    ax.set_xlabel("h = 1/nx  (square domain)")
    ax.set_ylabel(r"relative $L_2$ error  $\|T_h-T\|_0/\|T\|_0$")
    ax.set_title(r"A: sine MMS ladder, audited vs independent ($q=2\pi^2T$)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.4, loc="lower right")

    # ---- Panel B
    names = ["sine_generation", "exp_trig_harmonic", "sine_aspect2", "cubic_harmonic"]
    colors = ["tab:blue", "tab:purple", "tab:brown", "tab:gray"]
    for name, col in zip(names, colors):
        s = ref["studies"][name]
        o = orders(s["h"], s["l2_relative"])
        xs = [2 * hi for hi in s["h"][:-1]]  # coarse-mesh h of each pair
        lab = name + (" (non-discriminating: nodal exact)" if name == "cubic_harmonic" else "")
        ax2.semilogx(xs, o, "o-", color=col, lw=1.8, ms=6, label=lab, alpha=0.9)
    v = mine["variants"]["own_element_mean_source"]
    ax2.semilogx([2 * hi for hi in v["h"][:-1]], v["l2_orders"], "k*", ms=14,
                 label="board independent (element-mean source)", zorder=5)
    ax2.axhline(2.0, color="k", ls="--", lw=1.2, label="theory $p=2$")
    ax2.axhline(1.90, color="tab:red", ls=":", lw=1.4, label="predeclared gate $p\\geq1.90$")
    ax2.set_xlabel("coarse-mesh h of the refinement pair")
    ax2.set_ylabel(r"observed $L_2$ order $p$")
    ax2.set_title("B: per-pair order, recomputed from raw error arrays")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=7.2, loc="lower right")

    fig.suptitle("m20054 / CASE-FEM-HEAT-L2 board audit (m20108): independent reproduction", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(f"figure -> {OUT}")


if __name__ == "__main__":
    main()