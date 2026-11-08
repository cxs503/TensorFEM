"""Evidence script: free-edge plate on a Winkler foundation against exact references.

Closes the debt recorded in ``docs/free-plate-vibration-validation.md`` §5.5
("free edge + elastic foundation not verified because the Winkler term has no
exact characteristic function for a free edge") and in the module docstring of
``tensorfem.plate_ritz_reference``.

Three independent references are used, none of them produced by the FE solver:

1. **Vertical translation anchor** -- a plate held only by a uniform Winkler bed
   has the exact eigenpair ``w = const``, ``omega^2 = k_w / (rho h)`` for *any*
   boundary condition.  The consistent Q4 foundation matrix and the consistent
   translatory mass share the same shape-function products, so the discrete
   Rayleigh quotient must reproduce that value at machine precision.  Measured
   here on the assembled global matrices (element level, no eigen-solve).
2. **Foundation row-sum identity** -- the row sum of the assembled foundation
   matrix at a ``w`` degree of freedom equals ``k_w`` times the *exact* tributary
   area of that node (``a b / 4`` interior, half at an edge, quarter at a corner
   for a uniform mesh).  This is an analytic identity about the element matrix,
   not a comparison with this repository's earlier output.
3. **Shifted free-plate Rayleigh--Ritz spectrum** -- in the Kirchhoff model the
   Winkler potential is ``k_w/2 int w^2``, i.e. exactly proportional to the mass
   operator, so *every* eigenpair of the free plate satisfies
   ``omega^2(k_w) = omega^2(0) + k_w/(rho h)`` with unchanged mode shapes.  The
   independent Rayleigh--Ritz reference of ``tensorfem.plate_ritz_reference``
   (Legendre basis, exact Gauss quadrature, no element, no shared quadrature)
   therefore supplies an exact free-edge Winkler reference after all.

Also recorded (not hidden):

* the four-dimensional near-mechanism subspace of the SRI Q4 element (three
  rigid modes + one hourglass mode at ``k_w = 0``) is lifted to ``k_w/(rho h)``
  by the foundation; two of the four sit below that value by the rotary-inertia
  ratio (mesh independent), the other two match to round-off;
* a large ``k_w`` *dilutes* the relative error of the bending branches
  (``err = err(k_w=0) * lambda_0^2 / (lambda_0^2 + kappa)``), so the de-shifted
  comparison is the meaningful one and is reported as well.

Outputs (relative to the repository root)
----------------------------------------
* ``benchmarks/free_plate_winkler_foundation_evidence.json``
* ``docs/assets/free_plate_winkler_foundation/foundation_branches.png``
* ``docs/assets/free_plate_winkler_foundation/mesh_convergence.png``

Usage::

    OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/free_plate_winkler_foundation.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensorfem.plate_mindlin import (  # noqa: E402
    q4_mindlin_stiffness,
    q4_winkler_foundation_stiffness,
)
from tensorfem.plate_ritz_reference import ritz_free_plate_eigenvalues  # noqa: E402
from tensorfem.plate_vibration import (  # noqa: E402
    BOUNDARY_FREE,
    plate_rigidity,
    q4_plate_mass_matrix,
    solve_plate_vibration,
    structured_rectangular_mesh,
    winkler_translation_angular_frequency,
)

YOUNG = 200.0e9
POISSON = 0.30
THICKNESS = 0.01
DENSITY = 7850.0
LENGTH = 1.0
RIGIDITY = plate_rigidity(YOUNG, POISSON, THICKNESS)
AREAL = DENSITY * THICKNESS
DTYPE = torch.float64

# ``(label, length, width, [(elements_x, elements_y), ...])``
GEOMETRIES = (
    ("square", 1.0, 1.0, ((6, 6), (8, 8), (12, 12))),
    ("rectangle_2to1", 1.0, 0.5, ((8, 4), (12, 6), (16, 8))),
)
WINKLER_K = (0.0, 1.0e5, 1.0e6, 1.0e8)
SOFT_MODES = 4


def assemble(mesh: tuple[int, int], length: float, width: float, winkler_k: float):
    """Assemble the global plate/foundation/mass matrices with the public builders."""
    nodes, elements = structured_rectangular_mesh(
        mesh[0], mesh[1], length, width, dtype=DTYPE
    )
    ndof = 3 * nodes.shape[0]
    stiffness = torch.zeros((ndof, ndof), dtype=DTYPE)
    foundation = torch.zeros_like(stiffness)
    mass = torch.zeros_like(stiffness)
    for connectivity in elements:
        xy = nodes[connectivity]
        dofs = torch.stack(
            (3 * connectivity, 3 * connectivity + 1, 3 * connectivity + 2), dim=1
        ).reshape(-1)
        stiffness[dofs[:, None], dofs] += q4_mindlin_stiffness(xy, YOUNG, POISSON, THICKNESS)
        mass[dofs[:, None], dofs] += q4_plate_mass_matrix(xy, DENSITY, THICKNESS)
        if winkler_k > 0.0:
            foundation[dofs[:, None], dofs] += q4_winkler_foundation_stiffness(xy, winkler_k)
    return nodes, stiffness, foundation, mass


def translation_anchor(mesh: tuple[int, int], length: float, width: float, winkler_k: float):
    """Rayleigh quotient of the exact ``w = const, theta = 0`` translation vector."""
    nodes, _, foundation, mass = assemble(mesh, length, width, winkler_k)
    ndof = 3 * nodes.shape[0]
    vector = torch.zeros(ndof, dtype=DTYPE)
    vector[0::3] = 1.0
    numerator = float(vector @ (foundation @ vector))
    denominator = float(vector @ (mass @ vector))
    measured = math.sqrt(numerator / denominator)
    exact = winkler_translation_angular_frequency(
        density=DENSITY, thickness=THICKNESS, winkler_k=winkler_k
    )
    return {
        "mesh": list(mesh),
        "winkler_k": winkler_k,
        "measured_omega": measured,
        "exact_omega": exact,
        "relative_error": measured / exact - 1.0,
    }


def foundation_row_sums(mesh: tuple[int, int], length: float, width: float, winkler_k: float):
    """Compare assembled ``w``-row sums with the exact tributary areas."""
    nodes, _, foundation, _ = assemble(mesh, length, width, winkler_k)
    areas = foundation.sum(dim=1)[0::3].numpy() / winkler_k
    side_x = length / mesh[0]
    side_y = width / mesh[1]
    exact = []
    for index in range(areas.size):
        # structured_rectangular_mesh numbers nodes with x fastest
        node_y, node_x = divmod(index, mesh[0] + 1)
        share_x = 1.0 if 0 < node_x < mesh[0] else 0.5
        share_y = 1.0 if 0 < node_y < mesh[1] else 0.5
        exact.append(share_x * side_x * share_y * side_y)
    exact_array = torch.tensor(exact, dtype=DTYPE).numpy()
    return {
        "mesh": list(mesh),
        "winkler_k": winkler_k,
        "max_relative_error": float(abs(areas - exact_array).max() / exact_array.max()),
        "total_area_relative_error": float(
            abs(areas.sum() - length * width) / (length * width)
        ),
    }


def ritz_reference(length: float, width: float, kappa: float, count: int = 8):
    result = ritz_free_plate_eigenvalues(
        length=length, width=width, poisson=POISSON, order=14, count=count,
        winkler_parameter=kappa,
    )
    return {
        "winkler_parameter": kappa,
        "dimensionless": [float(value) for value in result.dimensionless],
        "foundation_check": float(result.foundation_check),
        "rigid_body_defect": float(result.rigid_body_defect),
    }


def fe_study(mesh: tuple[int, int], length: float, width: float, winkler_k: float,
             reference, rigid_body_relative_tolerance: float = 1.0e-9) -> dict:
    """One FE solve and its comparison with the shifted Rayleigh--Ritz reference."""
    modes = 10
    result = solve_plate_vibration(
        mesh[0], mesh[1], length=length, width=width, young=YOUNG, poisson=POISSON,
        thickness=THICKNESS, density=DENSITY, boundary=BOUNDARY_FREE, modes=modes,
        winkler_k=winkler_k,
        rigid_body_relative_tolerance=rigid_body_relative_tolerance,
    )
    angular_squared = (result.angular_frequencies**2).numpy()
    reference_angular_squared = torch.tensor(reference["dimensionless"], dtype=DTYPE).numpy() ** 2
    reference_angular_squared *= RIGIDITY / (AREAL * length**4)
    shift = winkler_k / AREAL
    offset = SOFT_MODES if winkler_k > 0.0 and result.rigid_body_modes == 0 else 0
    targets = reference_angular_squared[:3] + shift
    measured = angular_squared[offset : offset + 3]
    return {
        "mesh": list(mesh),
        "winkler_k": winkler_k,
        "shift_omega_squared": shift,
        "rigid_body_modes": int(result.rigid_body_modes),
        "rigid_body_relative_tolerance": rigid_body_relative_tolerance,
        "angular_squared": [float(value) for value in angular_squared],
        "soft_mode_ratio_to_shift": (
            [float(value / shift) for value in angular_squared[:SOFT_MODES]]
            if winkler_k > 0.0 else []
        ),
        "bending_omega_squared": [float(value) for value in measured],
        "reference_omega_squared": [float(value) for value in targets],
        "relative_error": [float(value) for value in (measured - targets) / targets],
        "deshifted_relative_error": [
            float(value) for value in
            ((measured - shift) - reference_angular_squared[:3]) / reference_angular_squared[:3]
        ],
    }


def kappa_from_k(winkler_k: float, length: float) -> float:
    return winkler_k * length**4 / RIGIDITY


def plot_branches(path: Path, length: float, width: float, sweep: list[dict], reference) -> None:
    """omega^2 versus foundation modulus: exact branch lines against FE points."""
    base = torch.tensor(reference["dimensionless"], dtype=DTYPE).numpy() ** 2
    base *= RIGIDITY / (AREAL * length**4)
    values = [row["winkler_k"] for row in sweep]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].plot(values, [row["winkler_k"] / AREAL for row in sweep], "k:", lw=1.2,
                 label=r"exact $k_w/(\rho h)$ (translation)")
    for mode in range(SOFT_MODES):
        axes[0].semilogy(values, [row["angular_squared"][mode] for row in sweep], "o",
                         ms=3.5, label=f"FE soft mode {mode + 1}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"Winkler modulus $k_w$ [N/m$^3$]")
    axes[0].set_ylabel(r"$\omega^2$ [rad$^2$/s$^2$]")
    axes[0].set_title("Foundation branch saturates at the exact translation value")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)

    for mode in range(3):
        axes[1].plot(values, [row["bending_omega_squared"][mode] for row in sweep], "o",
                     ms=3.5, label=f"FE bending mode {mode + 1}")
        axes[1].plot(values, [base[mode] + row["winkler_k"] / AREAL for row in sweep],
                     "-", lw=1.0, label=f"Ritz $\\lambda_{mode + 1}^2+\\kappa$")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"Winkler modulus $k_w$ [N/m$^3$]")
    axes[1].set_ylabel(r"$\omega^2$ [rad$^2$/s$^2$]")
    axes[1].set_title("Bending branches shift by exactly $k_w/(\\rho h)$")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_convergence(path: Path, rows: list[dict]) -> None:
    """Relative error of the first two bending modes versus mesh refinement."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for position, label in enumerate(("square", "rectangle_2to1")):
        axes[position].set_title(label)
        for winkler_k in (0.0, 1.0e6):
            selected = [row for row in rows if row["label"] == label and row["winkler_k"] == winkler_k]
            selected.sort(key=lambda row: row["mesh"][0] * row["mesh"][1])
            elements = [row["mesh"][0] * row["mesh"][1] for row in selected]
            for mode in range(2):
                axes[position].semilogy(
                    elements, [abs(row["relative_error"][mode]) for row in selected],
                    "o-" if mode == 0 else "s--", lw=1.0, ms=4,
                    label=("mode %d, " % (mode + 1))
                    + (r"$k_w=0$" if winkler_k == 0.0 else r"$k_w=10^6$"),
                )
        axes[position].axhline(0.03, color="k", ls=":", lw=1.0, label="3 % gate")
        axes[position].set_xlabel("elements")
        axes[position].set_ylabel("relative error in $\\omega^2$")
        axes[position].grid(True, which="both", alpha=0.3)
        axes[position].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    assets = root / "docs" / "assets" / "free_plate_winkler_foundation"
    assets.mkdir(parents=True, exist_ok=True)

    anchors = []
    row_sum_checks = []
    references = {}
    studies = []
    sweep = []

    for label, length, width, meshes in GEOMETRIES:
        reference = ritz_reference(length, width, 0.0)
        shifted = ritz_reference(length, width, kappa_from_k(1.0e6, length))
        # The Ritz reference has three lifted rigid modes (no hourglass), so the
        # elastic spectrum starts at index 3; the FE spectrum starts at 4.
        shift_residual = [
            float((value**2 - base**2) - shifted["winkler_parameter"])
            for value, base in zip(shifted["dimensionless"][3:], reference["dimensionless"])
        ]
        references[label] = {
            "length": length,
            "width": width,
            "kappa_zero": reference,
            "kappa_unit": shifted,
            "shift_identity_residual": shift_residual,
        }
        for winkler_k in (5.0e6, 1.0e9):
            anchors.append(translation_anchor(meshes[0], length, width, winkler_k))
            row_sum_checks.append(foundation_row_sums(meshes[0], length, width, winkler_k))
        for parents in meshes:
            for winkler_k in WINKLER_K:
                row = fe_study(parents, length, width, winkler_k, reference)
                row["label"] = label
                studies.append(row)

    # Branch sweep at a fixed mesh for the figure.
    length, width = LENGTH, LENGTH
    fixed = (10, 10)
    reference = ritz_reference(length, width, 0.0)
    for exponent in (4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0):
        winkler_k = 10.0**exponent
        sweep.append(fe_study(fixed, length, width, winkler_k, reference))

    # Usability boundary: the rigid-mode counter uses a threshold relative to the
    # *largest* eigenvalue, so a foundation soft enough that k_w/(rho h) falls
    # below that threshold has its lifted translation modes silently counted as
    # mechanisms.  Record where that happens and that the caller can tighten the
    # tolerance (``rigid_body_relative_tolerance``).
    tolerance_floor = []
    for exponent in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        winkler_k = 10.0**exponent
        row = {"winkler_k": winkler_k, "shift_omega_squared": winkler_k / AREAL}
        for tolerance in (1.0e-9, 1.0e-12):
            probe = fe_study((8, 8), LENGTH, LENGTH, winkler_k, reference,
                             rigid_body_relative_tolerance=tolerance)
            row["rigid_body_modes_at_1e%d" % round(-math.log10(tolerance))] = probe["rigid_body_modes"]
        tolerance_floor.append(row)

    plot_branches(assets / "foundation_branches.png", length, width, sweep, reference)
    plot_convergence(assets / "mesh_convergence.png", studies)

    evidence = {
        "case": "completely free isotropic plate on a uniform Winkler foundation, free vibration",
        "schema": "tensorfem.free-plate-winkler-foundation/1",
        "material": {
            "young_pa": YOUNG, "poisson": POISSON, "thickness_m": THICKNESS,
            "density_kg_m3": DENSITY, "areal_density_kg_m2": AREAL,
            "flexural_rigidity_n_m": RIGIDITY,
        },
        "method": {
            "fe": "SRI Q4 Mindlin plate, consistent mass with rotary inertia, boundary='free'",
            "reference": "global Rayleigh-Ritz Kirchhoff plate + exact Winkler shift",
            "exact_anchor": "vertical translation omega^2 = k_w/(rho h), any boundary",
        },
        "translation_anchor": anchors,
        "foundation_row_sums": row_sum_checks,
        "ritz_reference": references,
        "mesh_study": studies,
        "branch_sweep": sweep,
        "tolerance_floor": tolerance_floor,
        "figures": [
            {"name": "foundation_branches.png",
             "caption": "omega^2 versus Winkler modulus: the four-element near-mechanism "
                        "subspace saturates at the exact k_w/(rho h) line and the bending "
                        "branches follow Ritz lambda^2 + kappa."},
            {"name": "mesh_convergence.png",
             "caption": "Relative error of the first two bending modes against the shifted "
                        "Rayleigh-Ritz reference for a free edge, k_w = 0 and k_w = 1e6."},
        ],
    }
    out = root / "benchmarks" / "free_plate_winkler_foundation_evidence.json"
    out.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    worst_anchor = max(abs(row["relative_error"]) for row in anchors)
    worst_row_sum = max(row["max_relative_error"] for row in row_sum_checks)
    worst_shift = max(abs(value) for ref in references.values() for value in ref["shift_identity_residual"])
    print("translation anchor worst |rel| =", f"{worst_anchor:.3e}")
    print("foundation row-sum worst |rel| =", f"{worst_row_sum:.3e}")
    print("Ritz shift-identity worst |residual| =", f"{worst_shift:.3e}")
    for label in references:
        for row in studies:
            if row["label"] != label:
                continue
            errors = ", ".join(f"{100.0 * value:8.4f}%" for value in row["relative_error"])
            print(f"{label:16s} mesh={row['mesh']} k_w={row['winkler_k']:8.1e} "
                  f"rigid={row['rigid_body_modes']} err=[{errors}]")
    print("tolerance floor (rigid-mode counter vs foundation shift):")
    for row in tolerance_floor:
        print("  k_w=%.0e shift=%.3e rigid@1e-9=%d rigid@1e-12=%d"
              % (row["winkler_k"], row["shift_omega_squared"],
                 row["rigid_body_modes_at_1e9"], row["rigid_body_modes_at_1e12"]))
    print("wrote", out.relative_to(root))


if __name__ == "__main__":
    main()