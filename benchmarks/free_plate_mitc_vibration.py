"""Free-plate vibration with the MITC4 shear field: hourglass removal evidence.

Runs the three evidence layers quoted by
``docs/free-plate-mitc-vibration-validation.md`` and writes
``benchmarks/free_plate_mitc_vibration_evidence.json`` plus two figures under
``docs/assets/``:

1. stiffness null space of the free plate -- the SRI rule leaves 4 zero-energy
   modes (3 rigid + 1 ``w`` hourglass), the MITC4 rule leaves exactly 3;
2. frequency comparison against the independent Rayleigh--Ritz reference of
   ``tensorfem.plate_ritz_reference`` (which is itself pinned to Leissa SP-160);
3. supported boundaries with the MITC4 rule (parity with the historical SRI
   path, so the new option cannot silently change the qualified cases).

Usage::

    OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/free_plate_mitc_vibration.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from tensorfem.plate import q4_mindlin_stiffness, structured_square_mesh
from tensorfem.plate_ritz_reference import ritz_free_plate_eigenvalues
from tensorfem.plate_vibration import (
    LEISSA_CLAMPED_SQUARE_LAMBDA,
    q4_plate_mass_matrix,
    solve_plate_vibration,
)

LENGTH = 1.0
THICKNESS = 0.01
YOUNG = 200.0e9
POISSON = 0.30
DENSITY = 7850.0
MESHES = (6, 8, 12, 16)
MODES = 4

#: Leissa (NASA SP-160, 1969) for the free square plate: lambda = omega a^2
#: sqrt(rho h / D) of the first three elastic modes.
LEISSA_FREE_SQUARE = (13.468, 19.596, 24.272)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"


def rigidity() -> float:
    return YOUNG * THICKNESS**3 / (12.0 * (1.0 - POISSON**2))


def dimensionless(frequency_hz) -> float:
    return (
        float(frequency_hz)
        * 2.0
        * math.pi
        * LENGTH**2
        * math.sqrt(DENSITY * THICKNESS / rigidity())
    )


def free_plate_stiffness(n: int, shear_treatment: str) -> torch.Tensor:
    nodes, elements = structured_square_mesh(n, LENGTH)
    ndof = 3 * nodes.shape[0]
    stiffness = torch.zeros((ndof, ndof), dtype=torch.float64)
    for connectivity in elements:
        xy = nodes[connectivity]
        element = q4_mindlin_stiffness(
            xy, YOUNG, POISSON, THICKNESS, shear_treatment=shear_treatment
        )
        dofs = torch.stack(
            (3 * connectivity, 3 * connectivity + 1, 3 * connectivity + 2), dim=1
        ).reshape(-1)
        stiffness[dofs[:, None], dofs] += element
    return 0.5 * (stiffness + stiffness.T)


def null_space(stiffness: torch.Tensor, relative_tolerance: float = 1e-9) -> tuple[int, list[float]]:
    eigenvalues = torch.linalg.eigvalsh(stiffness)
    scale = float(eigenvalues.abs().max())
    count = int((eigenvalues.abs() <= relative_tolerance * scale).sum().item())
    return count, [float(x / scale) for x in eigenvalues[: count + 2]]


def rigid_motion_residual(n: int, shear_treatment: str) -> float:
    """Largest ``||K u|| / (||K|| ||u||)`` over the three exact rigid motions."""
    nodes, _ = structured_square_mesh(n, LENGTH)
    stiffness = free_plate_stiffness(n, shear_treatment)
    count = nodes.shape[0]
    scale = float(torch.linalg.matrix_norm(stiffness, ord=2))
    zeros = torch.zeros(count, dtype=torch.float64)
    ones = torch.ones(count, dtype=torch.float64)
    # Global dof layout is interleaved per node: (w, theta_x, theta_y).
    # The element shear strain is gamma = w,x + theta_x, w,y + theta_y, so a
    # rigid motion has theta_x = -w,x and theta_y = -w,y.
    motions = (
        torch.stack((ones, zeros, zeros), dim=1),          # translation w = 1
        torch.stack((nodes[:, 0], -ones, zeros), dim=1),    # rotation about y: w = x
        torch.stack((-nodes[:, 1], zeros, ones), dim=1),    # rotation about x: w = -y
    )
    worst = 0.0
    for vector in motions:
        vector = vector.reshape(-1)
        residual = float(torch.linalg.vector_norm(stiffness @ vector))
        worst = max(worst, residual / (scale * float(torch.linalg.vector_norm(vector))))
    return worst


def extra_zero_mode(n: int, shear_treatment: str = "sri") -> dict | None:
    """Isolate the extra zero-energy direction of the SRI rule and grade it.

    The null space of the free-plate stiffness is 4-dimensional under SRI, which
    makes the eigenvectors returned inside it an arbitrary basis.  The three
    exact rigid motions span a 3-dimensional subspace; the extra direction is
    isolated by projecting a null-space vector onto the Euclidean complement of
    that subspace.  A genuine rigid motion has zero energy under *both* shear
    rules, so the discriminator is the MITC4 energy of the isolated field: if it
    is non-zero the field is a spurious mechanism that only the SRI sampling
    fails to see.
    """
    nodes, _ = structured_square_mesh(n, LENGTH)
    count = nodes.shape[0]
    stiffness = free_plate_stiffness(n, shear_treatment)
    eigenvalues, eigenvectors = torch.linalg.eigh(stiffness)
    scale = float(eigenvalues.abs().max())
    zeros = int((eigenvalues.abs() <= 1e-9 * scale).sum().item())
    if zeros < 4:
        return None
    zeros_ = torch.zeros(count, dtype=torch.float64)
    ones = torch.ones(count, dtype=torch.float64)
    rigid = torch.stack(
        (
            torch.stack((ones, zeros_, zeros_), dim=1),
            torch.stack((nodes[:, 0], -ones, zeros_), dim=1),
            torch.stack((-nodes[:, 1], zeros_, ones), dim=1),
        ),
        dim=2,
    ).reshape(-1, 3)
    candidate = eigenvectors[:, 3]
    coefficients = torch.linalg.lstsq(rigid, candidate).solution
    projected = candidate - rigid @ coefficients
    projected = projected / torch.linalg.vector_norm(projected)
    field = projected.reshape(-1, 3)
    w = field[:, 0]
    rotation = float(torch.linalg.vector_norm(field[:, 1:]) / torch.linalg.vector_norm(w))
    energy_sri = float(0.5 * projected @ (free_plate_stiffness(n, "sri") @ projected))
    energy_mitc = float(0.5 * projected @ (free_plate_stiffness(n, "mitc") @ projected))
    w2 = w.reshape(n + 1, n + 1)
    sign_changes = int((torch.sign(w2[1:, :]) * torch.sign(w2[:-1, :]) < 0).sum().item())
    return {
        "mesh": n,
        "shear_treatment": shear_treatment,
        "zero_energy_modes": zeros,
        "rigid_content_of_null_vector": float(torch.linalg.vector_norm(rigid @ coefficients)),
        "rotation_to_translation_ratio": rotation,
        "mean_w_over_max_w": float(w.mean() / w.abs().max()),
        "sign_changes_along_x": sign_changes,
        "energy_under_sri_j": energy_sri,
        "energy_under_mitc_j": energy_mitc,
        "energy_ratio_sri_over_mitc": energy_sri / energy_mitc,
        "nodal_w": [float(x) for x in w],
    }


def collect_extra_modes() -> list[dict]:
    rows = []
    for n in (6, 8, 12):
        row = extra_zero_mode(n, "sri")
        if row is not None:
            rows.append(row)
    return rows


def write_figures(evidence: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ASSETS.mkdir(parents=True, exist_ok=True)
    written = []

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    reference = evidence["ritz_reference"]["dimensionless"]
    for axis, treatment in zip(axes, ("sri", "mitc")):
        rows = [r for r in evidence["frequencies"] if r["shear_treatment"] == treatment]
        mesh = [r["mesh"] for r in rows]
        for mode in range(MODES):
            axis.loglog(
                [1.0 / m for m in mesh],
                [abs(r["relative_error_percent"][mode]) for r in rows],
                marker="o",
                label=f"mode {mode + 1}",
            )
        axis.axhline(3.0, color="k", linestyle="--", linewidth=1.0, label="3 % gate")
        axis.set_title(f"free plate, shear_treatment={treatment!r}")
        axis.set_xlabel("element size h = a/n")
        axis.set_ylabel("|lambda error| vs Rayleigh-Ritz [%]")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    path = ASSETS / "free_plate_mitc_vibration_convergence.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    written.append(str(path.relative_to(ROOT)))

    extra = evidence["extra_zero_modes"]
    pivot = extra[1] if len(extra) > 1 else extra[0]
    n = pivot["mesh"]
    w = torch.tensor(pivot["nodal_w"], dtype=torch.float64).reshape(n + 1, n + 1)
    result = solve_plate_vibration(
        n, length=LENGTH, young=YOUNG, poisson=POISSON, thickness=THICKNESS,
        density=DENSITY, boundary="free", modes=1, shear_treatment="mitc",
    )
    physical = result.modes[: 3 * (n + 1) * (n + 1) : 3].reshape(n + 1, n + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for axis, field, title in (
        (
            axes[0],
            w,
            f"SRI 4th zero-energy mode (n={n}): w-hourglass\n"
            f"energy under MITC4 / SRI = {pivot['energy_ratio_sri_over_mitc']:.1e}",
        ),
        (axes[1], physical, f"MITC4 first elastic mode (n={n})"),
    ):
        image = axis.imshow(
            field.detach().numpy(), origin="lower", cmap="RdBu_r", extent=(0, 1, 0, 1)
        )
        axis.set_title(title)
        axis.set_xlabel("x / a")
        axis.set_ylabel("y / a")
        figure.colorbar(image, ax=axis, shrink=0.85)
    figure.tight_layout()
    path = ASSETS / "free_plate_mitc_vibration_modes.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    written.append(str(path.relative_to(ROOT)))
    return written


def main() -> dict:
    ritz = ritz_free_plate_eigenvalues(
        length=LENGTH, poisson=POISSON, order=18, count=MODES
    )
    reference = [float(x) for x in ritz.dimensionless]

    evidence: dict = {
        "case": {
            "length_m": LENGTH,
            "thickness_m": THICKNESS,
            "young_pa": YOUNG,
            "poisson": POISSON,
            "density_kg_m3": DENSITY,
            "bending_rigidity_nm": rigidity(),
            "aspect_ratio_a_over_h": LENGTH / THICKNESS,
            "boundary": "free",
            "meshes": list(MESHES),
            "modes": MODES,
        },
        "ritz_reference": {
            "order": ritz.order,
            "dimensionless": reference,
            "rigid_body_defect": ritz.rigid_body_defect,
            "energy_check": ritz.energy_check,
            "leissa_sp160": list(LEISSA_FREE_SQUARE),
            "leissa_relative_error": [
                100.0 * (reference[k] / LEISSA_FREE_SQUARE[k] - 1.0)
                for k in range(len(LEISSA_FREE_SQUARE))
            ],
        },
        "stiffness_null_space": [],
        "rigid_motion": [],
        "frequencies": [],
        "supported_boundary_parity": [],
    }

    for n in MESHES:
        for treatment in ("sri", "mitc"):
            count, smallest = null_space(free_plate_stiffness(n, treatment))
            evidence["stiffness_null_space"].append(
                {
                    "mesh": n,
                    "shear_treatment": treatment,
                    "zero_energy_modes": count,
                    "relative_eigenvalues": smallest,
                }
            )
            evidence["rigid_motion"].append(
                {
                    "mesh": n,
                    "shear_treatment": treatment,
                    "max_rigid_motion_residual": rigid_motion_residual(n, treatment),
                }
            )
            result = solve_plate_vibration(
                n,
                length=LENGTH,
                young=YOUNG,
                poisson=POISSON,
                thickness=THICKNESS,
                density=DENSITY,
                boundary="free",
                modes=MODES,
                shear_treatment=treatment,
            )
            lam = [dimensionless(f) for f in result.frequencies_hz]
            evidence["frequencies"].append(
                {
                    "mesh": n,
                    "shear_treatment": treatment,
                    "rigid_body_modes": result.rigid_body_modes,
                    "dimensionless": lam,
                    "relative_error_percent": [
                        100.0 * (lam[k] / reference[k] - 1.0) for k in range(MODES)
                    ],
                }
            )

    for n in MESHES:
        for treatment in ("sri", "mitc"):
            result = solve_plate_vibration(
                n,
                length=LENGTH,
                young=YOUNG,
                poisson=POISSON,
                thickness=THICKNESS,
                density=DENSITY,
                boundary="clamped",
                modes=2,
                shear_treatment=treatment,
            )
            lam = [dimensionless(f) for f in result.frequencies_hz]
            evidence["supported_boundary_parity"].append(
                {
                    "mesh": n,
                    "shear_treatment": treatment,
                    "clamped_lambda1": lam[0],
                    "leissa_clamped_lambda1": LEISSA_CLAMPED_SQUARE_LAMBDA,
                    "relative_error_percent": 100.0
                    * (lam[0] / LEISSA_CLAMPED_SQUARE_LAMBDA - 1.0),
                }
            )

    evidence["extra_zero_modes"] = collect_extra_modes()
    evidence["figures"] = write_figures(evidence)

    out = ROOT / "benchmarks" / "free_plate_mitc_vibration_evidence.json"
    out.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out}")

    for row in evidence["extra_zero_modes"]:
        print(
            f"mesh {row['mesh']:2d} sri extra zero mode: "
            f"E_sri/E_mitc = {row['energy_ratio_sri_over_mitc']:.2e}, "
            f"|theta|/|w| = {row['rotation_to_translation_ratio']:.2f}, "
            f"mean(w)/max(w) = {row['mean_w_over_max_w']:+.2e}, "
            f"sign changes = {row['sign_changes_along_x']}"
        )
    for row in evidence["stiffness_null_space"]:
        print(
            f"mesh {row['mesh']:2d} {row['shear_treatment']:4s}: "
            f"zero-energy modes = {row['zero_energy_modes']}"
        )
    for row in evidence["frequencies"]:
        print(
            f"mesh {row['mesh']:2d} {row['shear_treatment']:4s}: "
            + " ".join(f"{x:.4f}" for x in row["dimensionless"])
            + "  err% "
            + " ".join(f"{x:+.3f}" for x in row["relative_error_percent"])
        )
    return evidence


if __name__ == "__main__":
    main()