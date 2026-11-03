"""Figures for docs/cook-membrane-stress-audit.md revision R2.

Left panel: loaded-edge vertical displacement profile, pre-fix (transposed element
Jacobian) versus post-fix Q8 reference and the production Q4 run.  Right panel: pointwise
von Mises error of the production Q4 field against the fixed Q8 reference.

Run: ``OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/plot_cook_stress_recovery.py``
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks" / "cook_membrane_stress_reference.json"
OUTPUT = ROOT / "outputs" / "cook_stress_recovery"


def main() -> None:
    evidence = json.loads(EVIDENCE.read_text())
    old = json.loads(
        subprocess.run(
            ["git", "show", "HEAD:benchmarks/cook_membrane_stress_reference.json"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    profiles = evidence["loaded_edge_profiles"]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axis = axes[0]
    for entry in old["loaded_edge_profiles"]["reference_levels"]:
        axis.plot(entry["y"], entry["v"], "--", color="tab:red", alpha=0.45,
                  label="Q8 pre-fix (transposed Jacobian)" if entry["divisions"] == 12 else None)
    fixed = profiles["reference_levels"][-1]
    axis.plot(fixed["y"], fixed["v"], "-o", ms=3, color="tab:blue",
              label=f"Q8 fixed reference n={fixed['divisions']}")
    platform = profiles["platform_levels"][-1]
    axis.plot(platform["y"], platform["v"], "-", lw=2, color="tab:green",
              label=f"production Q4 n={platform['divisions']}")
    axis.set_xlabel("y along loaded edge x = 48")
    axis.set_ylabel("vertical displacement v")
    axis.set_title("loaded-edge profile: pre-fix vs post-fix")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)

    axis = axes[1]
    cases = evidence["comparison"]["cases"]
    levels = np.array([case["divisions"] for case in cases], dtype=float)
    errors = np.array([[point["von_mises_error"] for point in case["points"]] for case in cases])
    for index in range(errors.shape[1]):
        axis.loglog(levels, 100 * errors[:, index], "-o", ms=3,
                    label=f"point {index + 1}")
    axis.axhline(3.0, color="k", ls=":", lw=1, label="3% platform gate")
    axis.set_xlabel("Q4 divisions per direction")
    axis.set_ylabel("pointwise von Mises error [%]")
    axis.set_title("Q4 pointwise stress error vs Q8 reference")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)

    figure.tight_layout()
    path = OUTPUT / "cook-stress-recovery.png"
    figure.savefig(path, dpi=140)
    print("written", path.relative_to(ROOT))


if __name__ == "__main__":
    main()