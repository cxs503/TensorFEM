#!/usr/bin/env python3
"""Asymptotic-accuracy audit of the shell anchors (read-only over stored evidence).

Reads the refinement ladders already stored in ``result.json`` by the two shell
anchors, re-grades them with :mod:`tensorfem.refinement_gate` and renders the
reported-vs-certified figure.  No solve is re-run, so this is safe to execute
anywhere; every number it prints can be traced to a stored artifact.

    PYTHONPATH=src python benchmarks/verified/asymptotic_accuracy_audit.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorfem.refinement_gate import evaluate_reference_ladder, summarise  # noqa: E402

VERIFIED = Path(__file__).parent
ASSETS = REPO / "docs" / "assets"


def pinched_cylinder():
    data = json.loads((VERIFIED / "fem_pinched_cylinder" / "result.json").read_text())
    levels = [(1.0 / math.sqrt(lv["n_elements"]), lv["displacement"]) for lv in data["levels"]]
    return evaluate_reference_ladder(
        levels, reference_value=-1.8248e-05, tolerance=0.03, min_order=1.0,
        coarsest_sanity=0.5, qoi="radial displacement at load point",
        reference_label="MacNeal & Harder (1985)")


def scordelis_lo(element: str):
    data = json.loads((VERIFIED / "fem_shell_scordelis_lo" / "result.json").read_text())
    key = "flat_facet_results" if element == "flat_facet" else "cylindrical_results"
    levels = [(1.0 / lv["nx"], lv["probe_displacement"]) for lv in data[key]]
    return evaluate_reference_ladder(
        levels, reference_value=-0.3024, tolerance=0.03, min_order=1.0,
        coarsest_sanity=0.5, qoi=f"Scordelis-Lo free-edge midpoint ({element})",
        reference_label="Scordelis & Lo (1960)")


def main() -> int:
    cases = {
        "Pinched cylinder (cylindrical Shell4)": (pinched_cylinder(), "o", "C0"),
        "Scordelis-Lo (cylindrical Shell4)": (scordelis_lo("cylindrical"), "s", "C2"),
        "Scordelis-Lo (flat-facet Shell4)": (scordelis_lo("flat_facet"), "^", "C3"),
    }
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for label, (report, marker, colour) in cases.items():
        h = [lv["mesh_size"] for lv in report["levels"]]
        err = [abs(lv["value"] - report["reference"]["value"])
               / abs(report["reference"]["value"]) for lv in report["levels"]]
        ax.loglog(h, [100.0 * e for e in err], marker + "-", color=colour,
                  linewidth=1.8, markersize=7, label=f"{label} — per-level error")
        if report["certified_accuracy"] is not None:
            ax.axhline(100.0 * report["certified_accuracy"], color=colour,
                       linestyle="--", linewidth=1.2,
                       label=f"{label} — certified {100.0 * report['certified_accuracy']:.2f}%")
    ax.axhline(3.0, color="red", linestyle=":", linewidth=1.5, label="3% tolerance")
    ax.set_xlabel("element size h (log scale)")
    ax.set_ylabel("relative error vs published reference (%)")
    ax.set_title("Shell anchors: reported finest-mesh error vs certified (Richardson) error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / "shell_anchor_asymptotic_accuracy.png"
    fig.savefig(out, dpi=160)
    print(f"figure -> {out}")

    for label, (report, _, _) in cases.items():
        s = summarise(report)
        print(f"\n== {label}")
        print(f"   orders: " + ", ".join(
            f"{e['from']}->{e['to']}: "
            + ("n/a" if e['order'] is None else f"p={e['order']:.3f}")
            for e in report["observed_orders"]))
        certified = ("n/a" if report["certified_accuracy"] is None
                     else f"{100 * report['certified_accuracy']:.4f}%")
        print(f"   reported(finest)={100 * report['reported_accuracy_finest']:.4f}%  "
              f"certified={certified}")
        print(f"   accuracy={s['accuracy_verdict']} study={s['study_verdict']} "
              f"overall={s['overall_verdict']} unproven={s['unproven']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())