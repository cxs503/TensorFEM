"""Evidence script: free-edge plate vibration against an independent Rayleigh--Ritz reference.

Runs
----
1. ``ritz_free_plate_convergence`` -- the reference itself is refined, so the
   reader can see that the target values are converged and that the sequence is
   monotone (Rayleigh--Ritz is an upper bound).
2. A finite-element mesh study of the completely free square plate with the SRI
   Q4 Mindlin element of ``tensorfem.plate_vibration`` (``boundary="free"``).
3. The same for a 2:1 free plate.
4. Detection of zero-energy modes: a free plate must have exactly three
   (rigid-body) zero modes; every extra one is a spurious zero-energy
   (hourglass) mode of the element and is reported rather than hidden.

Outputs (relative to the repository root)
----------------------------------------
* ``benchmarks/free_plate_vibration_evidence.json``  raw numbers
* ``docs/assets/free_plate_vibration_convergence.png``
* ``docs/assets/free_plate_vibration_mode_shapes.png``

Usage::

    OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/free_plate_vibration.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensorfem.plate_ritz_reference import (  # noqa: E402
    energy_quadrature_check,
    ritz_free_plate_eigenvalues,
)
from tensorfem.plate_vibration import (  # noqa: E402
    BOUNDARY_FREE,
    plate_rigidity,
    solve_plate_vibration,
)

YOUNG = 200.0e9
POISSON = 0.30
THICKNESS = 0.01
DENSITY = 7850.0
LENGTH = 1.0
RIGIDITY = plate_rigidity(YOUNG, POISSON, THICKNESS)
AREAL = DENSITY * THICKNESS
LAMBDA_SCALE = math.sqrt(AREAL / RIGIDITY)


def dimensionless(result, mode: int) -> float:
    """Convert one FE frequency to ``omega a^2 sqrt(rho h / D)``."""
    omega = 2.0 * math.pi * float(result.frequencies_hz[mode])
    return omega * LENGTH**2 * LAMBDA_SCALE


def fe_study(width: float, meshes, modes: int = 4) -> list[dict]:
    rows = []
    for mesh in meshes:
        result = solve_plate_vibration(
            mesh, mesh, boundary=BOUNDARY_FREE, modes=modes, length=LENGTH, width=width,
            young=YOUNG, poisson=POISSON, thickness=THICKNESS, density=DENSITY,
        )
        rows.append(
            {
                "mesh": mesh,
                "elements": int(result.element_counts[0] * result.element_counts[1]),
                "dofs": int(3 * result.nodes.shape[0]),
                "zero_energy_modes": int(result.rigid_body_modes),
                "lambda": [dimensionless(result, k) for k in range(modes)],
                "first_mode_field": result.modes[:, 0].tolist(),
                "nodes": result.nodes.tolist(),
            }
        )
        print(
            f"  FE {mesh:2d}x{mesh:<2d} zero-energy={result.rigid_body_modes} "
            + " ".join(f"{row:8.4f}" for row in rows[-1]["lambda"]),
            flush=True,
        )
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    assets = root / "docs" / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    print("1) Rayleigh-Ritz reference convergence (upper bounds, monotone)")
    orders = (8, 10, 12, 14, 16, 18)
    ritz_rows = []
    previous = None
    for order in orders:
        result = ritz_free_plate_eigenvalues(
            length=LENGTH, width=LENGTH, poisson=POISSON, order=order, count=4
        )
        values = [float(v) for v in result.dimensionless]
        if previous is not None and values[0] > previous + 1e-12:
            raise SystemExit("Rayleigh-Ritz sequence is not monotone")
        previous = values[0]
        ritz_rows.append(
            {"order": order, "lambda": values, "rigid_body_defect": result.rigid_body_defect}
        )
        print(f"  order {order:2d}: " + " ".join(f"{v:9.5f}" for v in values), flush=True)
    reference = ritz_rows[-1]["lambda"]

    print("2) Assembled-form self-check against direct 2-D quadrature")
    energy_error = energy_quadrature_check(order=10, poisson=POISSON, length=LENGTH)
    print(f"  relative disagreement = {energy_error:.3e}")

    print("3) Free square plate, SRI Q4 Mindlin FE mesh study")
    square = fe_study(LENGTH, (6, 8, 10, 12, 16))

    print("4) Free 2:1 plate (a = 1.0 m, b = 0.5 m)")
    ritz_rect = ritz_free_plate_eigenvalues(
        length=LENGTH, width=0.5, poisson=POISSON, order=18, count=4
    )
    rectangle = fe_study(0.5, (12, 16))
    rectangle_reference = [float(v) for v in ritz_rect.dimensionless]
    print("  Ritz 2:1 reference: " + " ".join(f"{v:9.5f}" for v in rectangle_reference))

    # ------------------------------------------------------------------ errors
    for row in square:
        row["relative_error"] = [
            (row["lambda"][k] - reference[k]) / reference[k] for k in range(len(reference))
        ]
    for row in rectangle:
        row["relative_error"] = [
            (row["lambda"][k] - rectangle_reference[k]) / rectangle_reference[k]
            for k in range(len(rectangle_reference))
        ]

    # ----------------------------------------------------------------- figures
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    colours = ["#1f4e79", "#c00000", "#2e7d32", "#6a1b9a"]
    for k in range(4):
        axes[0].semilogy(
            [row["mesh"] for row in square],
            [abs(row["relative_error"][k]) for row in square],
            "o-", color=colours[k], label=f"mode {k + 1}",
        )
    axes[0].axhline(0.03, color="crimson", ls="--", lw=1.2, label="3 % gate")
    axes[0].set_xlabel("mesh (elements per side)")
    axes[0].set_ylabel("|relative error| vs Rayleigh-Ritz")
    axes[0].set_title("Free square plate: FE convergence")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(
        [row["order"] for row in ritz_rows],
        [row["lambda"][0] for row in ritz_rows],
        "s-", color="#1f4e79", label=r"$\lambda_1$ (Ritz)")
    axes[1].plot(
        [row["order"] for row in ritz_rows],
        [row["lambda"][1] for row in ritz_rows],
        "^-", color="#c00000", label=r"$\lambda_2$ (Ritz)")
    axes[1].axhline(13.468, color="k", ls=":", lw=1.0, label=r"Leissa $\lambda_1=13.468$")
    axes[1].set_xlabel("Ritz polynomial order")
    axes[1].set_ylabel(r"$\lambda = \omega a^2\sqrt{\rho h/D}$")
    axes[1].set_title("Reference convergence (upper bounds)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(assets / "free_plate_vibration_convergence.png", dpi=140)
    plt.close(fig)

    # Mode-shape panels: the spurious zero-energy mode is shown next to physics.
    finest = square[-1]
    nodes = torch.tensor(finest["nodes"])
    field = torch.tensor(finest["first_mode_field"]).reshape(-1, 3)
    mesh = finest["mesh"]
    side = mesh + 1
    grid = nodes.reshape(side, side, 2)
    w = field[:, 0].reshape(side, side).numpy()
    theta_x = field[:, 1].reshape(side, side).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    contour = axes[0].contourf(grid[:, :, 0].numpy(), grid[:, :, 1].numpy(), w, levels=20, cmap="RdBu_r")
    axes[0].set_title(f"Free plate, first elastic mode, w-field ({mesh}x{mesh})")
    axes[0].set_aspect("equal")
    fig.colorbar(contour, ax=axes[0], fraction=0.046)
    contour = axes[1].contourf(
        grid[:, :, 0].numpy(), grid[:, :, 1].numpy(), theta_x, levels=20, cmap="PuOr_r"
    )
    axes[1].set_title(r"Same mode, $\theta_x$ field (rotation, not translation)")
    axes[1].set_aspect("equal")
    fig.colorbar(contour, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(assets / "free_plate_vibration_mode_shapes.png", dpi=140)
    plt.close(fig)

    evidence = {
        "case": "completely free (unsupported) isotropic plate, free vibration",
        "material": {
            "young_pa": YOUNG, "poisson": POISSON, "thickness_m": THICKNESS,
            "density_kg_m3": DENSITY, "flexural_rigidity_n_m": RIGIDITY,
        },
        "geometry": {"length_m": LENGTH, "width_m": LENGTH, "aspect_ratio": 1.0},
        "reference": {
            "method": "global Rayleigh-Ritz, product Legendre basis, rigid-body projection",
            "orders": ritz_rows,
            "lambda": reference,
            "literature": {"source": "Leissa, Vibration of Plates, NASA SP-160 (1969)",
                           "lambda_square": [13.468, 19.596, 24.272]},
            "energy_quadrature_relative_error": energy_error,
        },
        "square_mesh_study": square,
        "rectangle_2to1": {
            "aspect_ratio": 2.0,
            "reference_lambda": rectangle_reference,
            "mesh_study": rectangle,
        },
        "zero_energy_modes": {
            "expected_rigid_body": 3,
            "observed": [row["zero_energy_modes"] for row in square],
            "interpretation": "one extra zero-energy (hourglass) mode per mesh, see docs",
        },
    }
    out = root / "benchmarks" / "free_plate_vibration_evidence.json"
    out.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    print("\nSummary (square, finest mesh):")
    for k in range(4):
        row = square[-1]
        print(
            f"  mode {k + 1}: lambda_FE={row['lambda'][k]:9.4f} "
            f"lambda_Ritz={reference[k]:9.4f} err={100 * row['relative_error'][k]:+.3f} %"
        )
    print("2:1 plate (16x16):")
    for k in range(2):
        row = rectangle[-1]
        print(
            f"  mode {k + 1}: lambda_FE={row['lambda'][k]:9.4f} "
            f"lambda_Ritz={rectangle_reference[k]:9.4f} "
            f"err={100 * row['relative_error'][k]:+.3f} %"
        )
    print(f"\nwrote {out.relative_to(root)} and two figures under docs/assets/")


if __name__ == "__main__":
    main()