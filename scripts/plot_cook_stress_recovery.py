"""Figures for the production Q4 stress-recovery certification.

Writes:

* ``docs/assets/cook_recovery_convergence.png`` - worst-site relative error vs mesh,
  recovered field against the stored element-centre field, with the 3 % gate and a
  second-order reference slope;
* ``docs/assets/cook_recovery_field.png`` - the discontinuous element-centre von Mises
  field and the recovered nodal field on the n=32 Cook membrane, with the five frozen
  probe sites marked.

Reproduce::

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src \
        python scripts/plot_cook_stress_recovery.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.additional_benchmark_fields import cook_fields  # noqa: E402

EVIDENCE = ROOT / "benchmarks" / "cook_stress_recovery_evidence.json"
ASSETS = ROOT / "docs" / "assets"
FIELD_LEVEL = 32


def convergence_figure(evidence: dict, path: Path) -> None:
    ladder = evidence["ladder"]
    divisions = [level["divisions"] for level in ladder]
    recovered = [100.0 * level["max_recovered_relative_error"] for level in ladder]
    raw = [100.0 * level["max_raw_relative_error"] for level in ladder]
    gate = 100.0 * evidence["problem"]["gate"]

    figure, axes = plt.subplots(figsize=(6.4, 4.6))
    axes.loglog(divisions, raw, "s--", color="#c0392b", label="stored element-centre field")
    axes.loglog(divisions, recovered, "o-", color="#1f6feb", label="recovered nodal field (production)")
    axes.axhline(gate, color="k", linestyle=":", linewidth=1.2)
    axes.annotate("3 % gate", xy=(divisions[0], gate), xytext=(0, 4),
                  textcoords="offset points", fontsize=9)
    # Second-order reference slope anchored on the finest recovered point.
    anchor = divisions[-1]
    slope = [recovered[-1] * (anchor / n) ** 2 for n in divisions]
    axes.loglog(divisions, slope, "-", color="0.6", linewidth=1.0)
    axes.annotate("O(h$^2$)", xy=(divisions[1], slope[1]), xytext=(4, 6),
                  textcoords="offset points", fontsize=9, color="0.4")
    axes.set_xlabel("mesh divisions n (n x n Q4)")
    axes.set_xticks(divisions)
    axes.set_xticklabels([str(n) for n in divisions])
    axes.set_ylabel("worst-site von Mises error [%]")
    axes.set_title("Cook membrane: stress recovery vs raw element-centre field")
    axes.grid(True, which="both", alpha=0.3)
    axes.legend(fontsize=9, loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def field_figure(evidence: dict, path: Path) -> None:
    case = cook_fields(FIELD_LEVEL)
    nodes = np.asarray(case["nodes"], dtype=float)
    elements = np.asarray(case["elements"], dtype=int)
    raw_vm = np.asarray(case["von_mises"], dtype=float)
    nodal_vm = np.asarray(case["von_mises_nodal"], dtype=float)
    probes = [(np.asarray(p["point"]), p["von_mises"]) for p in evidence["reference"]["probes"]]

    vmin = min(raw_vm.min(), nodal_vm.min())
    vmax = max(raw_vm.max(), nodal_vm.max())
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    patches = nodes[elements]
    collection = PolyCollection(
        patches,
        array=raw_vm,
        cmap="viridis",
        edgecolors="0.75",
        linewidths=0.2,
        clim=(vmin, vmax),
    )
    axes[0].add_collection(collection)
    axes[0].set_xlim(nodes[:, 0].min(), nodes[:, 0].max())
    axes[0].set_ylim(nodes[:, 1].min(), nodes[:, 1].max())
    axes[0].set_title("element-centre field (discontinuous, O(h))")

    triangles = np.vstack((elements[:, [0, 1, 2]], elements[:, [0, 2, 3]]))
    triangulation = Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    filled = axes[1].tricontourf(triangulation, nodal_vm, levels=20, cmap="viridis",
                                 vmin=vmin, vmax=vmax)
    axes[1].triplot(triangulation, color="0.8", linewidth=0.25)
    axes[1].set_title("recovered nodal field (single valued, O(h$^2$))")

    for axis in axes:
        for point, _ in probes:
            axis.plot(point[0], point[1], "o", markerfacecolor="none",
                      markeredgecolor="w", markersize=7, markeredgewidth=1.2)
        axis.set_aspect("equal")
        axis.set_xlabel("x")
    axes[0].set_ylabel("y")
    figure.colorbar(filled, ax=axes, label="von Mises stress", shrink=0.9)
    figure.suptitle(f"Cook membrane n={FIELD_LEVEL}: raw vs recovered von Mises", fontsize=11)
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    evidence = json.loads(EVIDENCE.read_text())
    ASSETS.mkdir(parents=True, exist_ok=True)
    convergence_figure(evidence, ASSETS / "cook_recovery_convergence.png")
    field_figure(evidence, ASSETS / "cook_recovery_field.png")
    for name in ("cook_recovery_convergence.png", "cook_recovery_field.png"):
        size = (ASSETS / name).stat().st_size
        print(f"{name}: {size / 1024:.1f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())