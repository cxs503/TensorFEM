"""Regenerate quarter-ring curved-cantilever evidence and figures.

Usage::

    OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/curved_beam_ring_evidence.py

Writes ``benchmarks/curved_beam_ring_evidence.json`` and three PNG figures under
``docs/assets/curved-beam-ring/``.  Every number in the published document is
printed by this script, so no value is transcribed by hand.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.curved_beam_benchmark import (  # noqa: E402
    TANGENT_COEFFICIENT,
    quarter_ring_reference,
    quarter_ring_shell,
)

MESHES = (4, 8, 16, 32, 64)
EVIDENCE = ROOT / "benchmarks" / "curved_beam_ring_evidence.json"
FIGDIR = ROOT / "docs" / "assets" / "curved-beam-ring"
ELASTIC = dict(radius=10.0, width=0.2, thickness=0.1, young=2.1e5, poisson=0.3, load=1.0)


def convergence() -> list[dict[str, object]]:
    rows = []
    for element in ("flat", "cylindrical"):
        for n in MESHES:
            summary = quarter_ring_shell(n, 2, element=element, **ELASTIC).summary()
            rows.append(summary)
    return rows


def slenderness() -> list[dict[str, object]]:
    rows = []
    for thickness in (0.02, 0.05, 0.1, 0.2):
        reference = quarter_ring_reference(
            ELASTIC["radius"], ELASTIC["width"], thickness,
            ELASTIC["young"], ELASTIC["poisson"], ELASTIC["load"],
        )
        for element in ("flat", "cylindrical"):
            result = quarter_ring_shell(
                32, 2, element=element, **{**ELASTIC, "thickness": thickness}
            )
            rows.append(
                {
                    "element": element,
                    "thickness": thickness,
                    "radius_over_thickness": ELASTIC["radius"] / thickness,
                    "tip_tangent": result.tip_tangent,
                    "tip_tangent_exact": result.tip_tangent_exact,
                    "tangent_error": result.tangent_error,
                    "coefficient": result.tip_tangent / reference["bending_scale"],
                    "coefficient_exact": TANGENT_COEFFICIENT,
                }
            )
    return rows


def markdown(conv, slen, reference) -> str:
    lines = ["| element | mesh n | delta_t FE | exact | error | delta_r FE | exact | error |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in conv:
        lines.append(
            "| {element} | {n_arc} | {tip_tangent:.5f} | {tip_tangent_exact:.5f} | "
            "{tangent_error:.3%} | {tip_radial:.5f} | {tip_radial_exact:.5f} | {radial_error:.3%} |".format(**row)
        )
    lines.append("")
    lines.append("| element | R/t | coefficient FE | coefficient exact | error |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in slen:
        lines.append(
            "| {element} | {radius_over_thickness:.0f} | {coefficient:.6f} | "
            "{coefficient_exact:.6f} | {tangent_error:.3%} |".format(**row)
        )
    lines.append("")
    lines.append("reference: " + json.dumps(reference, sort_keys=True))
    return "\n".join(lines)


def figures(conv, slen) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGDIR.mkdir(parents=True, exist_ok=True)
    styles = {"flat": ("o-", "flat-facet Shell4"), "cylindrical": ("s--", "metric-consistent Shell4")}

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for element, (style, label) in styles.items():
        rows = [r for r in conv if r["element"] == element]
        ns = [r["n_arc"] for r in rows]
        ax.loglog(ns, [100 * r["tangent_error"] for r in rows], style, label=label + " (tangent)")
        radial_style = "o:" if style.startswith("o") else "s:"
        ax.loglog(ns, [100 * r["radial_error"] for r in rows], radial_style, label=label + " (radial)")
    ax.axhline(3.0, color="crimson", lw=1.0, label="3% acceptance")
    ax.set_xlabel("arc elements n")
    ax.set_ylabel("relative error [%]")
    ax.set_title("Quarter-ring curved cantilever: mesh convergence")
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGDIR / "convergence.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    reference = quarter_ring_reference(
        ELASTIC["radius"], ELASTIC["width"], ELASTIC["thickness"],
        ELASTIC["young"], ELASTIC["poisson"], ELASTIC["load"],
    )
    ax.plot(0.0, 0.0, "k*", ms=15, label="clamped root")
    for element, (style, label) in styles.items():
        rows = [r for r in conv if r["element"] == element]
        ax.plot([r["tip_tangent"] for r in rows], [r["tip_radial"] for r in rows], style, label=label)
    ax.plot(reference["delta_tangent"], reference["delta_radial"], "kX", ms=11, label="closed form")
    ax.set_xlabel("free-end displacement along tangent  $\\delta_t$")
    ax.set_ylabel("free-end radial displacement  $\\delta_r$")
    ax.set_title("Tip displacement path, n = 4 ... 64")
    ax.grid(True, ls=":", lw=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "tip-path.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for element, (style, label) in styles.items():
        rows = [r for r in slen if r["element"] == element]
        ax.semilogx(
            [r["radius_over_thickness"] for r in rows],
            [r["coefficient"] for r in rows],
            style,
            label=label,
        )
    ax.axhline(TANGENT_COEFFICIENT, color="k", lw=1.0, ls="--", label="closed form 3$\\pi$/4-2")
    ax.set_xlabel("slenderness R/t")
    ax.set_ylabel("$\\delta_t E I / (P R^3)$")
    ax.set_title("Thickness sweep at n = 32 (membrane-locking probe)")
    ax.grid(True, ls=":", lw=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "slenderness.png", dpi=160)
    plt.close(fig)


def main() -> dict[str, object]:
    conv = convergence()
    slen = slenderness()
    reference = quarter_ring_reference(
        ELASTIC["radius"], ELASTIC["width"], ELASTIC["thickness"],
        ELASTIC["young"], ELASTIC["poisson"], ELASTIC["load"],
    )
    payload = {
        "benchmark": "quarter_ring_curved_cantilever",
        "parameters": ELASTIC,
        "reference": reference,
        "coefficients": {
            "tangent": TANGENT_COEFFICIENT,
            "radial": 0.5,
            "radial_load": math.pi / 4.0,
        },
        "convergence": conv,
        "slenderness": slen,
        "acceptance": "tangent and radial relative error < 1% at n = 32 for both elements",
        "passed": all(
            row["tangent_error"] < 0.01 and row["radial_error"] < 0.01
            for row in conv
            if row["n_arc"] == 32
        ),
    }
    EVIDENCE.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    figures(conv, slen)
    print(markdown(conv, slen, reference))
    print("\npassed:", payload["passed"])
    return payload


if __name__ == "__main__":
    main()