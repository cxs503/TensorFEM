#!/usr/bin/env python
"""Figure for the ``cohesive3d_verification.json`` evidence gates.

Two panels, every number read from the artifact or measured by this script:

left   per-criterion margin: the committed error metric of each of the eleven
       checks divided by the tolerance stated in that check's *own* criterion
       string (parsed, not transcribed).  A bar < 1 is a pass; the height is the
       headroom.
right  resolution of the numeric reproducibility gate for one representative
       leaf: the detection threshold ``atol + rtol * |value|`` against measured
       perturbations of that leaf (float64 noise, a 1e-6 relative edit and the
       2 % edit used to prove the gate fires).  This is where the honest limit
       lives: perturbations below the threshold are invisible to the numeric
       comparison and are protected by the digest gate only.

Writes ``docs/assets/cohesive3d_artifact_gate.png``.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "cohesive3d_verification.py"
ARTIFACT = ROOT / "benchmarks" / "cohesive3d_verification.json"
OUT = ROOT / "docs" / "assets" / "cohesive3d_artifact_gate.png"

TOLERANCE = re.compile(r"<\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")


def _max(rows, key):
    return max(row[key] for row in rows)


#: reported measure of each criterion -> the scalar that its tolerance bounds
MEASURES = {
    "law_reduces_to_2d_bk": lambda c: c["max_rel_error"],
    "law_energy": lambda c: max(_max(c["pure_modes"], "rel_error"),
                                _max(c["mixed_modes"], "rel_error")),
    "law_initiation": lambda c: _max(c["rows"], "abs_error"),
    "law_tangent_vs_ad": lambda c: c["max_rel_error"],
    "element_patch": lambda c: c["max_rel_error"],
    "element_rigid_rotation": lambda c: _max(c["rows"], "max_rel_error"),
    "element_mesh_objectivity": lambda c: c["max_rel_error"],
    "coupled_block_mode_i": lambda c: c["max_load_rel_error"],
    "coupled_block_mode_ii": lambda c: c["max_load_rel_error"],
    "coupled_block_mesh": lambda c: c["max_rel_error"],
    "coupled_block_energy": lambda c: c["relative_balance_error"],
}


def load_bench():
    spec = importlib.util.spec_from_file_location("cohesive3d_verification", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def gate_resolution(bench, artifact, rtol=1e-8, atol=1e-10):
    """Measure which perturbations of one leaf the numeric gate can see."""
    name, row, leaf = "coupled_block_mesh", 1, "reaction"
    value = artifact["criteria"][name]["rows"][row][leaf]
    threshold = atol + rtol * abs(value)
    measured = []
    for label, delta in (
        ("1e-12 relative\n(float64 noise)", 1e-12 * abs(value)),
        ("1e-6 relative", 1e-6 * abs(value)),
        ("2e-2 relative\n(the tamper probe)", 2e-2 * abs(value)),
    ):
        fresh = {"criteria": json.loads(json.dumps(artifact["criteria"]))}
        fresh["criteria"][name]["rows"][row][leaf] = value + delta
        verdict = bench.compare_payload({"criteria": artifact["criteria"]}, fresh)
        measured.append({"label": label, "delta": delta,
                         "detected": not verdict["ok"]})
    return {"leaf": f"{name}/rows/{row}/{leaf}", "value": value,
            "threshold": threshold, "measured": measured}


def main() -> int:
    artifact = json.loads(ARTIFACT.read_text())
    bench = load_bench()
    criteria = artifact["criteria"]

    names = [name for name in MEASURES if name in criteria]
    ratios, metrics, tolerances = [], [], []
    for name in names:
        tolerance = float(TOLERANCE.search(criteria[name]["criterion"]).group(1))
        metric = MEASURES[name](criteria[name])
        metrics.append(metric)
        tolerances.append(tolerance)
        ratios.append(max(metric, 1e-300) / tolerance)

    resolution = gate_resolution(bench, artifact)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    order = sorted(range(len(names)), key=lambda i: ratios[i])
    ax0.barh([names[i] for i in order], [ratios[i] for i in order],
             color=["#2b7a3d" if ratios[i] < 1.0 else "#b3261e" for i in order])
    ax0.set_xscale("log")
    ax0.axvline(1.0, color="k", ls="--", lw=1.2, label="predeclared tolerance (= 1)")
    ax0.set_xlabel("measured error / tolerance  (log, <1 = pass)")
    ax0.set_title("Committed criteria vs their own tolerances\n"
                  f"all_passed = {artifact['all_passed']}, checks_digest "
                  f"{artifact['checks_digest'][:12]}...")
    for i, index in enumerate(order):
        ax0.text(ratios[index] * 1.15, i,
                 f"{metrics[index]:.1e} / {tolerances[index]:.0e}",
                 va="center", fontsize=7)
    ax0.set_xlim(right=max(ratios) * 40)
    ax0.legend(fontsize=8, loc="lower right")
    ax0.grid(alpha=0.3, which="both", axis="x")

    labels = [m["label"] for m in resolution["measured"]]
    deltas = [m["delta"] for m in resolution["measured"]]
    colors = ["#b3261e" if m["detected"] else "#8a8a8a" for m in resolution["measured"]]
    ax1.bar(range(len(deltas)), deltas, color=colors)
    ax1.set_yscale("log")
    ax1.axhline(resolution["threshold"], color="k", ls="--", lw=1.2,
                label=f"gate threshold atol+rtol*|v| = {resolution['threshold']:.2e}")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel(f"|perturbation| applied to {resolution['leaf']} [N]")
    ax1.set_title("Numeric gate: what it can and cannot see\n"
                  "red = rejected by --check, grey = accepted (digest gate only)")
    for i, m in enumerate(resolution["measured"]):
        ax1.text(i, m["delta"] * 1.4, "rejected" if m["detected"] else "accepted",
                 ha="center", fontsize=8)
    ax1.set_ylim(top=max(deltas) * 30)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both", axis="y")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print("figure written:", OUT)
    print("resolution:", json.dumps(resolution, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())