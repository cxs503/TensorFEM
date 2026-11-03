"""Figures for the Cook-membrane audit closure (docs/cook-membrane-stress-audit.md).

Panel 1: error vs refinement - the five-point arbitrary-point criterion and the stored
element-centre criterion, with the 3 % gate and the fitted observed order.
Panel 2: loaded-edge vertical displacement profile, Q8 reference vs production Q4.
Panel 3: scale-normalised von Mises error of the stored element-centre field at n = 64,
which localises the residual error at the loaded-edge traction-discontinuity corners.

Run: ``OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/plot_cook_stress_audit_closure.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem import cook_stress_reference as reference
from tensorfem.additional_benchmark_fields import cook_fields

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks" / "cook_membrane_stress_reference.json"
OUTPUT = ROOT / "docs" / "assets"


def main() -> None:
    evidence = json.loads(EVIDENCE.read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axis = axes[0]
    cases = evidence["comparison"]["cases"]
    divisions = np.array([case["divisions"] for case in cases], dtype=float)
    arbitrary = np.array([case["max_von_mises_error"] for case in cases])
    centres = evidence["element_centre_accuracy"]
    centre_levels = np.array([entry["divisions"] for entry in centres], dtype=float)
    centre_p95 = np.array([entry["all_elements"]["p95_relative_error"] for entry in centres])
    axis.loglog(divisions, arbitrary, "o-", label="5 fixed interior points, max")
    axis.loglog(centre_levels, centre_p95, "s-", label="stored element centres, p95")
    axis.axhline(0.03, color="tab:red", ls="--", lw=1.2, label="3 % gate")
    fit = evidence["convergence"]["arbitrary_point_max_von_mises"]
    axis.loglog(divisions, 10 ** np.polyval(
        np.polyfit(np.log10(divisions), np.log10(arbitrary), 1), np.log10(divisions)),
        ":", color="tab:blue", label=f"fit p={fit['observed_order']:.2f}")
    axis.set_xlabel("edge divisions per side")
    axis.set_ylabel("relative von Mises error")
    axis.set_title("Q4 accuracy vs. refinement")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)

    axis = axes[1]
    profiles = evidence["loaded_edge_profiles"]
    reference_profile = profiles["reference_levels"][-1]
    platform_profile = profiles["platform_levels"][-1]
    axis.plot(reference_profile["y"], reference_profile["v"], "-o", ms=3,
              label=f"Q8 reference n={reference_profile['divisions']}")
    axis.plot(platform_profile["y"], platform_profile["v"], "--", lw=2,
              label=f"production Q4 n={platform_profile['divisions']}")
    axis.set_xlabel("y along loaded edge x = 48")
    axis.set_ylabel("vertical displacement v")
    axis.set_title("loaded-edge profile (open item: closed)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)

    axis = axes[2]
    solution = reference.solve_q8(48)
    case = cook_fields(64)
    nodes = np.asarray(case["nodes"], dtype=np.float64)
    elements = np.asarray(case["elements"], dtype=np.int64)
    stored = np.asarray(case["von_mises"], dtype=np.float64)
    centres_xy = nodes[elements].mean(axis=1)
    target = np.array([reference.field_stress_at_point(solution, point)["von_mises"]
                       for point in centres_xy])
    scale = float(np.max(np.abs(target)))
    error = np.abs(stored - target) / scale
    side = int(np.sqrt(len(centres_xy)))
    grid = error.reshape(side, side)
    mesh = axis.pcolormesh(centres_xy.reshape(side, side, 2)[:, :, 0],
                           centres_xy.reshape(side, side, 2)[:, :, 1],
                           grid, shading="auto", cmap="magma")
    figure.colorbar(mesh, ax=axis, label="|dMises| / scale")
    for xi, eta in reference.SAMPLE_POINTS:
        point = reference.map_point(xi, eta)
        axis.plot(point[0], point[1], "o", mfc="none", mec="cyan", ms=9)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title("stored field error, n = 64 (circles = 5 audit points)")

    figure.tight_layout()
    path = OUTPUT / "cook_stress_audit_closure.png"
    figure.savefig(path, dpi=140)
    print("saved", path.relative_to(ROOT))
    print("scale-normalised max %.4f, p95 %.4f" % (error.max(), np.percentile(error, 95)))


if __name__ == "__main__":
    main()
