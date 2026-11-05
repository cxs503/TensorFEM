"""Build the Cook membrane stress-recovery qualification evidence and its figures.

Runs the production recovery ladder, writes ``benchmarks/cook_recovery_qualification.json``
and the figures referenced by ``docs/cook-membrane-stress-recovery-qualification.md``.

Usage::

    OMP_NUM_THREADS=2 PYTHONPATH=src python3 scripts/build_cook_recovery_qualification.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri  # noqa: E402
import numpy as np  # noqa: E402

from tensorfem.cook_recovery_qualification import (  # noqa: E402
    GATE,
    cook_recovery_study,
)
from tensorfem.cook_stress_reference import SAMPLE_POINTS  # noqa: E402
from tensorfem.stress_recovery import von_mises  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks" / "cook_recovery_qualification.json"
ASSETS = ROOT / "docs" / "assets" / "cook-recovery"
LEVELS = (8, 16, 32, 64)


def _slim(case: dict) -> dict:
    """Evidence record without the per-case mesh/field payloads (recomputable)."""
    return {key: value for key, value in case.items() if key not in (
        "nodes", "elements", "displacement", "recovered_stress", "recovered_von_mises"
    )}


def figure_convergence(study: dict) -> Path:
    row_mesh = [row["mesh"] for row in study["rows"]]
    recovered = [100.0 * row["max_error"] for row in study["rows"]]
    component = [100.0 * row["max_component_error"] for row in study["rows"]]
    centre = [100.0 * row["max_element_centre_error"] for row in study["rows"]]
    figure, axes = plt.subplots(figsize=(6.4, 4.6))
    axes.loglog(row_mesh, centre, "s--", label="stored element-centre field (merged protocol)")
    axes.loglog(row_mesh, recovered, "o-", label="Gauss-point nodal recovery, von Mises")
    axes.loglog(row_mesh, component, "^:", label="Gauss-point nodal recovery, worst component")
    axes.axhline(100 * GATE, color="crimson", linewidth=1.2, linestyle="-", label="3 % acceptance gate")
    axes.set_xlabel("structured Q4 divisions n (elements per side)")
    axes.set_ylabel("max pointwise relative error over the 5 published points  [%]")
    axes.set_title("Cook membrane: recovery closes the pointwise stress debt")
    axes.set_xticks(row_mesh)
    axes.set_xticklabels([str(n) for n in row_mesh])
    axes.grid(True, which="both", alpha=0.3)
    axes.legend(fontsize=8, loc="lower left")
    figure.tight_layout()
    path = ASSETS / "cook-recovery-convergence.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def figure_cloud(case: dict) -> Path:
    nodes = np.asarray(case["nodes"], dtype=np.float64)
    elements = np.asarray(case["elements"], dtype=np.int64)
    field = np.asarray(case["recovered_von_mises"], dtype=np.float64)
    figure, axes = plt.subplots(figsize=(6.6, 4.8))
    triangles = np.vstack((elements[:, [0, 1, 2]], elements[:, [0, 2, 3]]))
    triangulation = matplotlib.tri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    contour = axes.tricontourf(triangulation, field, levels=24, cmap="viridis")
    figure.colorbar(contour, ax=axes, label="recovered von Mises stress (E=1, unit thickness)")
    probes = np.asarray([point["point"] for point in case["points"]], dtype=np.float64)
    axes.plot(probes[:, 0], probes[:, 1], "wo", markerfacecolor="none", markersize=8,
              markeredgewidth=1.4, label="5 published probe points")
    for point in case["points"]:
        axes.annotate(
            f"{100 * point['error']:.3f}%",
            point["point"],
            textcoords="offset points",
            xytext=(8, 4),
            color="white",
            fontsize=8,
        )
    axes.plot(nodes[elements][:, :, 0].T, nodes[elements][:, :, 1].T, color="white",
              linewidth=0.08, alpha=0.25)
    axes.set_aspect("equal")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_title(f"Recovered von Mises field, n={case['mesh']} (probe errors vs Q8 reference)")
    axes.legend(fontsize=8, loc="upper right")
    figure.tight_layout()
    path = ASSETS / "cook-recovery-field.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def figure_probe_bars(case: dict) -> Path:
    rows = case["points"]
    labels = [f"({row['xi']:.2f}, {row['eta']:.2f})" for row in rows]
    recovered = [100 * row["error"] for row in rows]
    low = [100 * row["element_centre_error_min"] for row in rows]
    high = [100 * row["element_centre_error_max"] for row in rows]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(figsize=(7.2, 4.4))
    axes.bar(x, recovered, width=0.42, label="recovered nodal field", color="#2b7bba")
    axes.errorbar(
        x + 0.46,
        [0.5 * (lo + hi) for lo, hi in zip(low, high)],
        yerr=[[0.5 * (hi - lo) for lo, hi in zip(low, high)],
              [0.5 * (hi - lo) for lo, hi in zip(low, high)]],
        fmt="o",
        color="black",
        capsize=4,
        label="stored element-centre field: one-sided error band",
    )
    axes.axhline(100 * GATE, color="crimson", linewidth=1.2, linestyle="--", label="3 % gate")
    axes.set_xticks(x + 0.23)
    axes.set_xticklabels(labels)
    axes.set_yscale("log")
    axes.set_xlabel("published probe point (xi, eta)")
    axes.set_ylabel("relative error vs frozen Q8 reference  [%]")
    axes.set_title(f"Probe-by-probe: recovered value vs the ill-posed single-point value, n={case['mesh']}")
    axes.legend(fontsize=8)
    axes.grid(True, axis="y", which="both", alpha=0.3)
    figure.tight_layout()
    path = ASSETS / "cook-recovery-probes.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def main() -> dict:
    ASSETS.mkdir(parents=True, exist_ok=True)
    study = cook_recovery_study(LEVELS)
    finest = study["cases"][-1]
    payload = {
        "case": "cook-membrane-stress-recovery",
        "gate": GATE,
        "levels": study["levels"],
        "sample_points": [list(point) for point in SAMPLE_POINTS],
        "rows": study["rows"],
        "finest_max_error": study["finest_max_error"],
        "monotone": study["monotone"],
        "recovery_qualified": study["recovery_qualified"],
        "fail_reasons": study["fail_reasons"],
        "reference": study["reference"],
        "finest_points": finest["points"],
        "stored_element_centre_flag_unchanged": True,
        "figures": [
            "docs/assets/cook-recovery/cook-recovery-convergence.png",
            "docs/assets/cook-recovery/cook-recovery-field.png",
            "docs/assets/cook-recovery/cook-recovery-probes.png",
        ],
        "reproduce": "OMP_NUM_THREADS=2 PYTHONPATH=src python3 scripts/build_cook_recovery_qualification.py",
    }
    EVIDENCE.write_text(json.dumps(payload, indent=1))
    paths = [
        figure_convergence(study),
        figure_cloud(finest),
        figure_probe_bars(finest),
    ]
    print(json.dumps({"evidence": str(EVIDENCE), "figures": [str(p) for p in paths],
                      "rows": study["rows"], "qualified": study["recovery_qualified"],
                      "fail_reasons": study["fail_reasons"]}, indent=1))
    return payload


if __name__ == "__main__":
    main()