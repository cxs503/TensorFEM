"""Evidence generator for non-uniform prebuckling stress (pure bending).

Writes ``benchmarks/plate_bending_buckling_evidence.json`` and two figures into
``docs/assets/``. Every number in the evidence file comes from this script --
nothing is hand-copied into the documentation.

Run: ``PYTHONPATH=src python benchmarks/plate_bending_buckling.py``
"""
from __future__ import annotations

import json
import math
import platform
from pathlib import Path

import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorfem.plate_bending_buckling import (  # noqa: E402
    LITERATURE_KB_LONG_PANEL,
    infinite_plate_bending_coefficient,
    linear_bending_profile,
    ritz_bending_buckling_load,
    run_bending_buckling_qualification,
    uniform_profile,
)
from tensorfem.plate_shear_buckling import (  # noqa: E402
    ritz_inplane_buckling_load,
    shear_buckling_coefficient,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
EVIDENCE = ROOT / "benchmarks" / "plate_bending_buckling_evidence.json"

D = 1.0
B = 1.0


def _ritz(ratio: float, *, basis_x: int, profile=None):
    return ritz_bending_buckling_load(
        length=ratio * B, width=B, rigidity_x=D, rigidity_y=D, coupling_rigidity=D,
        basis_x=basis_x, basis_y=12, profile=profile,
    )


def _mode_field(m: int, n: int, aspect: float, samples: int = 120) -> np.ndarray:
    x = np.linspace(0.0, aspect * B, samples)
    y = np.linspace(0.0, B, samples)
    return np.sin(m * math.pi * x[:, None] / (aspect * B)) * np.sin(n * math.pi * y[None, :] / B)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    qualification = run_bending_buckling_qualification()

    # basis convergence of the long panel
    basis_study = []
    for basis_x in (8, 12, 16, 24, 32):
        mode = _ritz(10.0, basis_x=basis_x)
        basis_study.append(
            {"basis_x": basis_x, "coefficient": mode.coefficient, "dominant_mode": list(mode.dominant_mode)}
        )

    # independent reference convergence (Hermite C1 elements in y)
    reference_study = []
    for elements in (8, 16, 32, 64):
        entry = infinite_plate_bending_coefficient(width=B, rigidity=D, elements=elements)
        uniform_entry = infinite_plate_bending_coefficient(
            width=B, rigidity=D, profile=uniform_profile(1.0), elements=elements
        )
        reference_study.append(
            {
                "elements": elements,
                "bending_coefficient": entry["coefficient"],
                "half_wavelength": entry["half_wavelength"],
                "residual": entry["residual"],
                "uniform_coefficient": uniform_entry["coefficient"],
            }
        )

    # shear-module regression after the shared-pencil fix
    shear = []
    for aspect in (1.0, 6.0):
        mode = ritz_inplane_buckling_load(
            length=aspect * B, width=B, rigidity_x=D, rigidity_y=D, coupling_rigidity=D,
            basis_x=16, basis_y=16,
        )
        shear.append(
            {
                "aspect_ratio": aspect,
                "k_s": shear_buckling_coefficient(shear_load=mode.multiplier, rigidity=D, width=B),
            }
        )

    evidence = {
        "schema": "tensorfem.plate_bending_buckling.evidence/1",
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
        "scope": {
            "verified": "simply supported isotropic panel, linearly varying in-plane stress, k_b vs 23.9",
            "not_verified": "FE shell geometric stiffness under non-uniform prestress; plasticity; stiffened panels",
        },
        "literature": {"k_b_long_panel": LITERATURE_KB_LONG_PANEL, "k_uniaxial": 4.0,
                       "source": "Timoshenko & Gere, Theory of Elastic Stability, 2nd ed., Sec. 9.2"},
        "qualification": qualification,
        "basis_study": basis_study,
        "independent_reference_study": reference_study,
        "shear_regression": shear,
    }
    EVIDENCE.write_text(json.dumps(evidence, indent=2) + "\n")

    # ---------------------------------------------------------------- figures
    sweep = qualification["aspect_sweep"]
    ratios = [entry["aspect_ratio"] for entry in sweep]
    coefficients = [entry["coefficient"] for entry in sweep]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.semilogx(ratios, coefficients, "o-", label="Ritz, non-uniform (pure bending)")
    ax.axhline(LITERATURE_KB_LONG_PANEL, color="crimson", ls="--", label="Timoshenko & Gere, k$_b$=23.9")
    ax.axhline(4.0, color="steelblue", ls=":", label="uniform compression, k=4")
    ax.set_xlabel("aspect ratio a/b")
    ax.set_ylabel("buckling coefficient k$_b$")
    ax.set_title("Simply supported panel, linear prestress")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "plate_bending_buckling_sweep.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    elements = [entry["elements"] for entry in reference_study]
    ax.loglog(elements, [abs(e["bending_coefficient"] - 23.9) / 23.9 for e in reference_study],
              "o-", label="Hermite C1 in y vs 23.9")
    ax.loglog(elements, [abs(e["uniform_coefficient"] - 4.0) / 4.0 for e in reference_study],
              "s--", label="same reference, uniform limit vs 4.0")
    ax.loglog(elements, [abs(e["coefficient"] - 23.9) / 23.9 for e in qualification["aspect_sweep"]
                         if e["aspect_ratio"] == max(ratios)] * len(elements),
              "^:", label="Ritz (a/b=10) vs 23.9")
    ax.set_xlabel("elements across the width")
    ax.set_ylabel("relative error")
    ax.set_title("Independent 1-D reference converges at O(h$^4$)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "plate_bending_buckling_reference.png", dpi=130)
    plt.close(fig)

    mode = _ritz(2.0, basis_x=12)
    field = _mode_field(mode.dominant_mode[0], mode.dominant_mode[1], 2.0)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    image = ax.contourf(np.linspace(0, 2, field.shape[0]), np.linspace(0, 1, field.shape[1]),
                        field.T, levels=20, cmap="RdBu_r")
    ax.contour(np.linspace(0, 2, field.shape[0]), np.linspace(0, 1, field.shape[1]),
               field.T, levels=[0.0], colors="k", linewidths=1.2)
    ax.set_xlabel("x / b (compression at y=0, tension at y=b)")
    ax.set_ylabel("y / b")
    ax.set_title(f"Critical mode a/b=2, (m,n)={mode.dominant_mode}, k$_b$={mode.coefficient:.3f}")
    fig.colorbar(image, ax=ax, label="w / max|w|")
    fig.tight_layout()
    fig.savefig(ASSETS / "plate_bending_buckling_mode.png", dpi=130)
    plt.close(fig)

    print(json.dumps({
        "long_panel": qualification["long_panel"],
        "independent": qualification["independent_reference"],
        "ritz_vs_independent": qualification["ritz_vs_independent"],
        "uniform_limit": qualification["uniform_limit"],
        "shear_regression": shear,
        "evidence": str(EVIDENCE.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
