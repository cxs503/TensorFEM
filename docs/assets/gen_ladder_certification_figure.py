#!/usr/bin/env python3
"""Figures for docs/CASE-FEM-LADDER-ANCHOR-CERTIFICATION.md.

Two panels, both plotted straight from the shipped JSON artifacts (no number is
typed in by hand):

1. gate margins -- for every gate in the three certificate envelopes, the
   normalised margin  log10(value/threshold) clipped at 0, i.e. how far inside
   the acceptance band the measured quantity sits (negative = failing side).
2. convergence -- the per-level relative error of the fem_kerr_plate branches
   against the exact Navier solution, with the 0.5 % acceptance band drawn.

    PYTHONPATH=src python3 docs/assets/gen_ladder_certification_figure.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFIED = REPO_ROOT / "benchmarks" / "verified"
OUT = Path(__file__).resolve().parent

ANCHORS = ("fem_beam", "fem_plate", "fem_kerr_plate")


def gate_margin(gate: dict) -> float:
    """log10(value/threshold) for a `<=' gate, log10(threshold/value) for a `>=' gate.

    Negative means "outside the acceptance band by a factor of 10**|m|", zero is
    the band edge, positive is inside.
    """
    value, threshold = gate["value"], gate["threshold"]
    if gate["operator"] in ("<=", "<"):
        if value == 0.0:
            return 12.0  # exactly zero residual: arbitrarily deep inside the band
        return -math.log10(value / threshold)
    return math.log10(value / threshold)


def figure_gate_margins(ax) -> None:
    labels, margins, colours = [], [], []
    palette = {"fem_beam": "#1f77b4", "fem_plate": "#2ca02c", "fem_kerr_plate": "#d62728"}
    for anchor in ANCHORS:
        payload = json.loads((VERIFIED / anchor / "gates_verified.json").read_text())
        for gate in payload["gates"]:
            labels.append(f"{anchor}\n{gate['id']}")
            margins.append(gate_margin(gate))
            colours.append(palette[anchor])
    y = range(len(labels))
    ax.barh(list(y), margins, color=colours, height=0.7)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.axvline(0.0, color="k", lw=1.0)
    ax.set_xlabel("margin inside the acceptance band  [decades, >0 = passing]")
    ax.set_title("Every certificate gate recomputed from the artifact", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)


def figure_convergence(ax) -> None:
    payload = json.loads((VERIFIED / "fem_kerr_plate" / "result.json").read_text())
    style = {
        "kerr_full": ("o-", "full Kerr (k=1e8 N/m^3, Gp=5e4 N/m)"),
        "winkler_limit": ("s--", "Winkler limit (Gp = 0)"),
        "free_plate": ("^:", "plain plate (k = Gp = 0)"),
    }
    for name, (fmt, label) in style.items():
        branch = payload["results"][name]
        n = [mesh["n"] for mesh in branch["meshes"]]
        err = [mesh["rel_err"] for mesh in branch["meshes"]]
        ax.loglog(n, err, fmt, label=label, ms=4)
    ax.axhline(5e-3, color="grey", ls="-.", lw=1.0, label="0.5 % acceptance band")
    # second-order reference slope anchored on the n=8 point
    branch = payload["results"]["kerr_full"]
    n8, e8 = branch["meshes"][1]["n"], branch["meshes"][1]["rel_err"]
    ns = [4, 8, 16, 32]
    ax.loglog(ns, [e8 * (n8 / n) ** 2 for n in ns], "k:", lw=0.8, label="slope h^2 reference")
    ax.set_xlabel("elements per side  n")
    ax.set_ylabel("|w_FE / w_exact - 1|")
    ax.set_title("Kerr foundation plate: centre deflection vs Navier closed form", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6)


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8))
    figure_gate_margins(ax1)
    figure_convergence(ax2)
    fig.suptitle(
        "Ladder anchor certification: fem_beam / fem_plate / fem_kerr_plate", fontsize=10
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT / "case-fem-ladder-certification.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()