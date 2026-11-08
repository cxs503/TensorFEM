"""Generate the TET10 integration-accuracy evidence (JSON + figures).

Run from the repository root::

    OMP_NUM_THREADS=1 PYTHONPATH=src python3 scripts/audit_tet10_integration.py

Outputs
-------
``docs/benchmarks/tutorials/tet10-integration-accuracy.json``
    Every scalar quoted in ``docs/tet10-integration-accuracy.md``: the Duffy oracle
    self-verification table, the Keast error versus mid-edge curvature (hand-picked and
    pseudo-random warps), the strain-energy errors, the consistent-mass numbers and the
    production-versus-oracle cross-implementation agreements.
``docs/assets/tet10_integration_error.png``
    Keast error versus curvature with the 3 % acceptance gate drawn in.
``docs/assets/tet10_integration_oracle.png``
    Self-verification of the Duffy reference (order convergence).
``docs/assets/tet10_integration_mass.png``
    Consistent-mass error, which the shipped rule cannot represent at all.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)  # shared host: thread oversubscription dominates runtime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfem.quadratic_solid import tet10_quadrature, tet10_stiffness  # noqa: E402
from tensorfem.solid3d import elasticity_matrix_3d  # noqa: E402
from tensorfem.tet10_integration_audit import (  # noqa: E402
    KEAST_POINTS,
    KEAST_WEIGHTS,
    duffy_rule,
    duffy_self_verification,
    elasticity_matrix,
    element_stiffness,
    integration_sweep,
    mass_matrix_audit,
    strain_energy_audit,
    tet10_geometry,
)

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "docs" / "benchmarks" / "tutorials" / "tet10-integration-accuracy.json"
ASSET_DIR = ROOT / "docs" / "assets"
YOUNG = 200.0e9
POISSON = 0.3
GATE = 0.03  # the platform's 3 % acceptance gate


def _production_keast_matches_oracle() -> list[dict]:
    """Cross-implementation agreement: torch production vs independent NumPy oracle."""
    d3 = elasticity_matrix_3d(torch.tensor([YOUNG]), torch.tensor([POISSON]))
    d = elasticity_matrix(YOUNG, POISSON)
    rows = []
    for kind, curvature in (("regular", 0.0), ("distorted", 0.0), ("curved", 0.05),
                            ("curved", 0.11), ("warped", 0.02)):
        nodes = tet10_geometry(kind, curvature)
        k_prod, _ = tet10_stiffness(torch.tensor(nodes[None]), d3)
        k_oracle = element_stiffness(nodes, d, KEAST_POINTS, KEAST_WEIGHTS)
        rows.append(
            {
                "geometry": f"{kind} curvature={curvature}",
                "production_vs_oracle_relative_frobenius": float(
                    np.linalg.norm(k_prod[0].numpy() - k_oracle) / np.linalg.norm(k_oracle)
                ),
            }
        )
    return rows


def _duffy_rule_matches_production_convention() -> dict:
    points, weights = tet10_quadrature()
    return {
        "keast_points_max_abs_diff": float(np.max(np.abs(points.numpy() - KEAST_POINTS))),
        "keast_weights_max_abs_diff": float(np.max(np.abs(weights.numpy() - KEAST_WEIGHTS))),
        "keast_weight_sum": float(weights.numpy().sum()),
        "reference_volume": 1.0 / 6.0,
    }


def _opt_in_rule_recovers_reference() -> list[dict]:
    """The documented ``rule=`` hook must land on the high-order reference value."""
    d3 = elasticity_matrix_3d(torch.tensor([YOUNG]), torch.tensor([POISSON]))
    ref_points, ref_weights = duffy_rule(24)
    rows = []
    for curvature in (0.05, 0.11):
        nodes = tet10_geometry("curved", curvature)
        x = torch.tensor(nodes[None])
        k_default, _ = tet10_stiffness(x, d3)
        points, weights = duffy_rule(6)
        k_opt, _ = tet10_stiffness(x, d3, rule=(points, weights))
        k_ref = element_stiffness(nodes, elasticity_matrix(YOUNG, POISSON), ref_points, ref_weights)
        norm = float(np.linalg.norm(k_ref))
        rows.append(
            {
                "curvature": curvature,
                "default_vs_reference": float(np.linalg.norm(k_default[0].numpy() - k_ref) / norm),
                "opt_in_6th_order_vs_reference": float(
                    np.linalg.norm(k_opt[0].numpy() - k_ref) / norm
                ),
            }
        )
    return rows


def _benchmark_mesh_elements() -> list[dict]:
    """How much integration error does the shipped TET10 benchmark actually see?

    ``cantilever_tet10_benchmark`` upgrades a structured HEX8 mesh, so every mid-edge
    node sits at the exact midpoint: the mesh is piecewise affine and the shipped rule
    is exact there.  This block measures that claim instead of asserting it.
    """
    d = elasticity_matrix(YOUNG, POISSON)
    from tensorfem.quadratic_solid import cantilever_tet10_benchmark, tet4_to_tet10
    from tensorfem.solid3d import hex_to_tet_mesh, structured_hex_mesh

    nodes, hexes = structured_hex_mesh(10.0, 1.0, 1.0, 5, 1, 1, dtype=torch.float64)
    _, tet4 = hex_to_tet_mesh(nodes, hexes)
    nodes10, cells = tet4_to_tet10(nodes, tet4)
    ref_points, ref_weights = duffy_rule(16)
    worst = 0.0
    for cell in cells[:8].tolist():
        element = nodes10[cell].numpy()
        k_keast = element_stiffness(element, d, KEAST_POINTS, KEAST_WEIGHTS)
        k_ref = element_stiffness(element, d, ref_points, ref_weights)
        worst = max(worst, float(np.linalg.norm(k_keast - k_ref) / np.linalg.norm(k_ref)))
    computed, reference = cantilever_tet10_benchmark(10)
    return [
        {
            "benchmark_elements_checked": min(8, len(cells)),
            "worst_stiffness_relative_error": worst,
            "cantilever_tip_deflection_computed": float(computed),
            "cantilever_tip_deflection_reference": float(reference),
            "cantilever_relative_error": abs(float(computed) - float(reference)) / float(reference),
        }
    ]


def _plot_error_sweep(curved, warped, energy, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    for rows, label, style in ((curved, "curved (alternating)", "-o"), (warped, "warped (random)", "--s")):
        alpha = [r["curvature"] for r in rows]
        err = [100.0 * r["stiffness_keast_vs_reference"] for r in rows]
        ax.plot(alpha, err, style, label=f"stiffness, {label}", markersize=4, lw=1.4)
    alpha = [r["curvature"] for r in energy]
    bend = [100.0 * r["bending_energy_relative_error"] for r in energy]
    ax.plot(alpha, bend, "-^", color="tab:green", lw=1.4, markersize=4,
            label="bending strain energy, curved")
    ax.axhline(100 * GATE, color="crimson", ls=":", lw=1.2, label="3 % gate")
    ax.set_xlabel("mid-edge offset / element edge length")
    ax.set_ylabel("Keast error vs high-order reference [%]")
    ax.set_yscale("log")
    ax.set_title("Shipped 5-point Keast rule on quadratic tets")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5)

    ax = axes[1]
    alpha = [r["curvature"] for r in energy]
    ax.plot(alpha, [100 * r["affine_energy_relative_error"] for r in energy], "-o",
            lw=1.4, markersize=4, label="affine field (detJ only)")
    ax.plot(alpha, [100 * r["bending_energy_relative_error"] for r in energy], "-^",
            lw=1.4, markersize=4, label="bending field (B variation)")
    ax.axhline(100 * GATE, color="crimson", ls=":", lw=1.2, label="3 % gate")
    ax.set_xlabel("mid-edge offset / element edge length")
    ax.set_ylabel("strain-energy error [%]")
    ax.set_yscale("log")
    ax.set_title("Where the error comes from")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_oracle(verification, curved_tail, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    orders = [r["order"] for r in verification]
    ax = axes[0]
    ax.semilogy(orders, [max(r["worst_monomial_relative_error"], 1e-17) for r in verification],
                "-o", lw=1.4, markersize=4, label="worst monomial error")
    ax.semilogy(orders, [max(r["volume_relative_error"], 1e-17) for r in verification],
                "-s", lw=1.4, markersize=4, label="reference volume error")
    ax.axhline(1e-14, color="crimson", ls=":", lw=1.2, label="1e-14 acceptance")
    ax.set_xlabel("Gauss-Legendre points per collapsed-cube direction")
    ax.set_ylabel("relative error")
    ax.set_title("Duffy oracle self-verification (closed-form monomials)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.semilogy([r["curvature"] for r in curved_tail],
                [max(r["reference_half_vs_full"], 1e-18) for r in curved_tail],
                "-o", lw=1.4, markersize=4)
    ax.axhline(1e-10, color="crimson", ls=":", lw=1.2, label="1e-10 reference stability")
    ax.set_xlabel("mid-edge offset / element edge length")
    ax.set_ylabel("order-12 vs order-24 relative difference")
    ax.set_title("Reference convergence on curved geometry")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_mass(mass_rows, path: Path) -> None:
    labels = [f"offset {r['curvature']:.2f}" for r in mass_rows]
    xpos = np.arange(len(mass_rows))
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.bar(xpos - 0.2, [100 * r["relative_frobenius_error"] for r in mass_rows], width=0.4,
           label="Frobenius")
    ax.bar(xpos + 0.2, [100 * r["diagonal_relative_error_max"] for r in mass_rows], width=0.4,
           label="max diagonal")
    ax.axhline(100 * GATE, color="crimson", ls=":", lw=1.2, label="3 % gate")
    ax.set_xticks(xpos, labels)
    ax.set_ylabel("consistent-mass error [%]")
    ax.set_title("Shipped Keast rule cannot integrate the quadratic mass matrix")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _at(rows, curvature: float, key: str = "stiffness_keast_vs_reference") -> float:
    return next(r[key] for r in rows if abs(r["curvature"] - curvature) < 1e-12)


def main() -> dict:
    verification = duffy_self_verification()
    curved = integration_sweep(kind="curved",
                               curvatures=(0.0, 0.02, 0.05, 0.08, 0.11, 0.12, 0.15))
    warped = integration_sweep(kind="warped", curvatures=(0.0, 0.02, 0.05))
    energy = strain_energy_audit()
    mass_rows = mass_matrix_audit(curvatures=(0.0, 0.05, 0.11))
    payload = {
        "case": "TET10 (10-node quadratic tetrahedron) integration accuracy",
        "acceptance_gate": GATE,
        "material": {"young_pa": YOUNG, "poisson": POISSON},
        "reference_rule": {
            "name": "Duffy collapsed-cube tensor-product Gauss-Legendre",
            "order": 24,
            "points": 24 ** 3,
            "self_verification": verification,
        },
        "oracle_self_verification": {
            "closed_forms": verification,
            "keast_convention": _duffy_rule_matches_production_convention(),
            "production_vs_oracle": _production_keast_matches_oracle(),
        },
        "curved_sweep": curved,
        "warped_sweep": warped,
        "strain_energy": energy,
        "consistent_mass": mass_rows,
        "opt_in_rule": _opt_in_rule_recovers_reference(),
        "benchmark_mesh": _benchmark_mesh_elements(),
        "verdict": {
            "straight_edges_exact": bool(curved[0]["stiffness_keast_vs_reference"] < 1e-12),
            "curvature_0.11_stiffness_error": _at(curved, 0.11),
            "curvature_0.12_stiffness_error": _at(curved, 0.12),
            "curvature_0.15_stiffness_error": _at(curved, 0.15),
            "curvature_0.12_jacobian_min": _at(curved, 0.12, key="det_jacobian_min"),
            "reference_tail_at_0.11": _at(curved, 0.11, key="reference_half_vs_full"),
            "reference_tail_at_0.15": _at(curved, 0.15, key="reference_half_vs_full"),
            "first_failing_curvature": next(
                (r["curvature"] for r in curved if r["stiffness_keast_vs_reference"] > GATE), None
            ),
            "mass_matrix_error_straight": mass_rows[0]["relative_frobenius_error"],
        },
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    _plot_error_sweep(curved, warped, energy, ASSET_DIR / "tet10_integration_error.png")
    _plot_oracle(verification, curved, ASSET_DIR / "tet10_integration_oracle.png")
    _plot_mass(mass_rows, ASSET_DIR / "tet10_integration_mass.png")
    print(json.dumps(payload["verdict"], indent=2))
    print(f"wrote {JSON_PATH.relative_to(ROOT)}")
    return payload


if __name__ == "__main__":
    main()