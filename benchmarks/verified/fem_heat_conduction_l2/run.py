#!/usr/bin/env python3
"""CASE-FEM-HEAT-L2: Steady-state heat conduction convergence ≥ 2nd order.

Manufactured-solution verification of the TensorFEM Q4 heat-conduction kernel.

Physics
-------
Steady heat conduction on the unit square with a manufactured solution:

    T(x,y) = sin(πx) · sinh(πy) / sinh(π)

This field is exactly harmonic (∇²T = 0), so it satisfies the steady
heat equation with zero source.  Dirichlet boundary conditions are applied
from the exact field on all four edges.  The Q4 element with 2×2 Gauss
quadrature is expected to converge at O(h²) in the L₂ norm.

Reference
---------
The manufactured solution is the classical harmonic field used in FEM
verification (e.g. Zienkiewicz & Taylor, *The Finite Element Method*,
vol. 1, 5th ed., Butterworth-Heinemann, 2000, ch. 3-4).  The exact
expression is evaluated here — no hand-copied numbers.

Acceptance gates (predeclared)
------------------------------
* Gate 1: finest-mesh relative L₂ error < 3 %
* Gate 2: observed h-refinement order ≥ 1.8 (target 2.0, tolerance 0.2)
* Gate 3: heat-balance residual < 1e-10 (global conservation)

Outputs
-------
* ``result.json``        — raw numbers at every mesh level
* ``gates_verified.json`` — pass/fail verdicts
* ``result.json``        — raw numbers + the sha256 of every figure it produced
* ``docs/assets/verified/fem_heat_conduction_l2/fig_convergence.png``
                          — log-log error vs element size
* ``docs/assets/verified/fem_heat_conduction_l2/fig_field.png``
                          — FEM temperature field + error map
* ``README.md``           — this document

The figures render under ``docs/`` on purpose: the platform push gate carries
``docs/**`` binaries and strips every other ``*.png``
(``tests/test_anchor_figure_gate_safety.py``), so a certificate that names a
figure next to its anchor names evidence that never reaches GitHub.

Usage::

    OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/verified/fem_heat_conduction_l2/run.py
"""
from __future__ import annotations

import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))

from tensorfem.thermal import ThermalModel, assemble_thermal, solve_steady_thermal

FLOAT = torch.float64
DEV = torch.device("cpu")

REPO_ROOT = HERE.parents[2]
#: Certificate figures live under ``docs/`` so the push gate carries them.
FIGURE_DIR = REPO_ROOT / "docs" / "assets" / "verified" / "fem_heat_conduction_l2"

# ---------------------------------------------------------------------------
# manufactured solution
# ---------------------------------------------------------------------------
def exact_temperature(nodes: torch.Tensor) -> torch.Tensor:
    """T(x,y) = sin(πx)·sinh(πy)/sinh(π)."""
    x, y = nodes[:, 0], nodes[:, 1]
    return torch.sin(math.pi * x) * torch.sinh(math.pi * y) / math.sinh(math.pi)


def exact_gradient(nodes: torch.Tensor) -> torch.Tensor:
    """∇T = (π·cos(πx)·sinh(πy)/sinh(π),  π·sin(πx)·cosh(πy)/sinh(π))."""
    x, y = nodes[:, 0], nodes[:, 1]
    s = math.sinh(math.pi)
    dx = math.pi * torch.cos(math.pi * x) * torch.sinh(math.pi * y) / s
    dy = math.pi * torch.sin(math.pi * x) * torch.cosh(math.pi * y) / s
    return torch.stack((dx, dy), dim=1)


# ---------------------------------------------------------------------------
# mesh generation
# ---------------------------------------------------------------------------
def structured_quad_mesh(nx: int, ny: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Structured Q4 mesh on [0,1]×[0,1] with nx×ny elements.

    Node (i,j) has index i*(ny+1) + j.
    Elements are ordered counter-clockwise: BL, BR, TR, TL.
    """
    xs = torch.linspace(0.0, 1.0, nx + 1, dtype=FLOAT, device=DEV)
    ys = torch.linspace(0.0, 1.0, ny + 1, dtype=FLOAT, device=DEV)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    nodes = torch.stack([gx.flatten(), gy.flatten()], dim=1)
    conn = []
    for i in range(nx):
        for j in range(ny):
            n0 = i * (ny + 1) + j
            conn.append([n0, n0 + ny + 1, n0 + ny + 2, n0 + 1])
    elems = torch.tensor(conn, dtype=torch.long, device=DEV)
    return nodes, elems


def edge_nodes(nx: int, ny: int) -> list[int]:
    """Return sorted list of node indices on the boundary of an nx×ny mesh."""
    boundary = set()
    for i in range(nx + 1):
        boundary.add(i * (ny + 1))          # bottom
        boundary.add(i * (ny + 1) + ny)     # top
    for j in range(ny + 1):
        boundary.add(j)                      # left
        boundary.add(nx * (ny + 1) + j)     # right
    return sorted(boundary)


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------
def solve_heat_conduction(nx: int, ny: int) -> dict:
    """Run one mesh level and return errors."""
    nodes, elems = structured_quad_mesh(nx, ny)
    n_nodes = len(nodes)

    # Dirichlet BC from exact solution
    bdry = edge_nodes(nx, ny)
    fixed_nodes = torch.tensor(bdry, dtype=torch.long, device=DEV)
    fixed_temps = exact_temperature(nodes[fixed_nodes])

    model = ThermalModel(
        nodes=nodes,
        elements=elems,
        conductivity=torch.tensor(1.0, dtype=FLOAT),
        density=torch.tensor(1.0, dtype=FLOAT),
        specific_heat=torch.tensor(1.0, dtype=FLOAT),
        thickness=torch.tensor(1.0, dtype=FLOAT),
        fixed_nodes=fixed_nodes,
        fixed_temperatures=fixed_temps,
        element_type="q4",
    )

    T_fem = solve_steady_thermal(model)
    T_exact = exact_temperature(nodes)

    # L2 relative error (area-weighted)
    err = T_fem - T_exact
    l2_err = float(torch.sqrt(torch.mean(err ** 2)))
    l2_rel = float(l2_err / torch.sqrt(torch.mean(T_exact ** 2)))

    # Max pointwise relative error
    max_abs = float(torch.max(torch.abs(err)))
    max_rel = float(torch.max(torch.abs(err / torch.where(T_exact != 0, T_exact, torch.ones_like(T_exact)))))

    # Heat balance: residual at free nodes should be ~0 (machine precision)
    matrices = assemble_thermal(model)
    free_mask = torch.ones(n_nodes, dtype=torch.bool, device=DEV)
    free_mask[fixed_nodes] = False
    residual_free = matrices.conductivity[free_mask] @ T_fem - matrices.heat[free_mask]
    heat_balance = float(torch.max(torch.abs(residual_free)))

    n_elem = nx * ny
    h = 1.0 / max(nx, ny)

    return {
        "nx": nx,
        "ny": ny,
        "n_elements": n_elem,
        "n_nodes": n_nodes,
        "n_dof": n_nodes - len(bdry),
        "h": h,
        "l2_absolute": l2_err,
        "l2_relative": l2_rel,
        "max_absolute": max_abs,
        "max_relative": max_rel,
        "heat_balance_residual": heat_balance,
    }


# ---------------------------------------------------------------------------
# convergence study
# ---------------------------------------------------------------------------
MESH_LEVELS = [(4, 4), (8, 8), (16, 16), (32, 32)]


def convergence_study() -> list[dict]:
    results = []
    for nx, ny in MESH_LEVELS:
        r = solve_heat_conduction(nx, ny)
        results.append(r)
        print(f"  {nx}x{ny}: n_elem={r['n_elements']}, L2_rel={r['l2_relative']:.6e}, max_rel={r['max_relative']:.6e}")
    return results


def observed_orders(results: list[dict]) -> list[float]:
    """Compute observed convergence order between successive mesh levels."""
    orders = []
    for i in range(1, len(results)):
        h_ratio = results[i - 1]["h"] / results[i]["h"]
        err_ratio = results[i - 1]["l2_relative"] / results[i]["l2_relative"]
        if err_ratio > 0 and h_ratio > 1:
            order = math.log(err_ratio) / math.log(h_ratio)
        else:
            order = 0.0
        orders.append(order)
    return orders


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_convergence_figure(results: list[dict], orders: list[float], outpath: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    hs = [r["h"] for r in results]
    errs = [r["l2_relative"] for r in results]

    ax.loglog(hs, errs, "o-", label="Q4 L₂ relative error", linewidth=2, markersize=8)

    # Reference slope lines
    h_ref = np.array([hs[0], hs[0] / 4])
    for slope, label in [(1.0, "O(h¹)"), (2.0, "O(h²)")]:
        e_ref = errs[0] * (h_ref / hs[0]) ** slope
        ax.loglog(h_ref, e_ref, "--", alpha=0.5, label=label)

    ax.set_xlabel("Element size h")
    ax.set_ylabel("Relative L₂ error")
    ax.set_title("CASE-FEM-HEAT-L2: Convergence of Q4 heat conduction\n"
                 f"Observed orders: {', '.join(f'{o:.3f}' for o in orders)}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def make_field_figure(nx: int, ny: int, outpath: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nodes, elems = structured_quad_mesh(nx, ny)
    bdry = edge_nodes(nx, ny)
    fixed_nodes = torch.tensor(bdry, dtype=torch.long, device=DEV)
    fixed_temps = exact_temperature(nodes[fixed_nodes])

    model = ThermalModel(
        nodes=nodes, elements=elems,
        conductivity=torch.tensor(1.0, dtype=FLOAT),
        density=torch.tensor(1.0, dtype=FLOAT),
        specific_heat=torch.tensor(1.0, dtype=FLOAT),
        thickness=torch.tensor(1.0, dtype=FLOAT),
        fixed_nodes=fixed_nodes,
        fixed_temperatures=fixed_temps,
        element_type="q4",
    )
    T_fem = solve_steady_thermal(model)
    T_exact = exact_temperature(nodes)
    err = (T_fem - T_exact).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # FEM field
    im0 = axes[0].tricontourf(nodes[:, 0].numpy(), nodes[:, 1].numpy(), T_fem.numpy(),
                               levels=30, cmap="viridis")
    axes[0].set_title("FEM Temperature")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    # Exact field
    im1 = axes[1].tricontourf(nodes[:, 0].numpy(), nodes[:, 1].numpy(), T_exact.numpy(),
                               levels=30, cmap="viridis")
    axes[1].set_title("Exact Temperature")
    axes[1].set_aspect("equal")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    # Error
    im2 = axes[2].tricontourf(nodes[:, 0].numpy(), nodes[:, 1].numpy(), err,
                               levels=30, cmap="RdBu_r")
    axes[2].set_title("Error (FEM − Exact)")
    axes[2].set_aspect("equal")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(f"CASE-FEM-HEAT-L2: Temperature field ({nx}×{ny} mesh, {nx*ny} elements)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def main():
    t0 = time.time()
    print("=" * 60)
    print("CASE-FEM-HEAT-L2: Steady heat conduction convergence study")
    print("=" * 60)

    results = convergence_study()
    orders = observed_orders(results)

    # Finest mesh field figure
    nx_f, ny_f = MESH_LEVELS[-1]
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig_field = FIGURE_DIR / "fig_field.png"
    fig_convergence = FIGURE_DIR / "fig_convergence.png"
    make_field_figure(nx_f, ny_f, fig_field)
    make_convergence_figure(results, orders, fig_convergence)

    elapsed = time.time() - t0

    # Build result.json
    result = {
        "benchmark": "fem_heat_conduction_l2",
        "case_key": "CASE-FEM-HEAT-L2",
        "description": "Steady-state heat conduction: manufactured harmonic solution on unit square, Q4 elements",
        "generated_at": utc_now(),
        "elapsed_seconds": round(elapsed, 2),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "machine": platform.machine(),
            "dtype": "torch.float64",
        },
        "reference": {
            "solution": "T(x,y) = sin(πx)·sinh(πy)/sinh(π)",
            "equation": "∇²T = 0 on [0,1]², Dirichlet BC from exact field",
            "source": "Manufactured harmonic field (Zienkiewicz & Taylor, FEM vol.1, ch.3-4)",
        },
        "figures": [
            {"path": _repo_relative(path), "sha256": _file_sha256(path)}
            for path in (fig_convergence, fig_field)
        ],
        "mesh_levels": results,
        "observed_orders_l2": orders,
        "mean_order": float(np.mean(orders)) if orders else 0.0,
    }

    write_json(result, HERE / "result.json")

    # Gates
    gates = [
        Gate(
            id="finest_l2_below_3pct",
            quantity="Relative L₂ error at finest mesh (32×32)",
            reference="T(x,y) = sin(πx)·sinh(πy)/sinh(π)",
            value=results[-1]["l2_relative"],
            threshold=0.03,
            operator="<=",
            passed=results[-1]["l2_relative"] <= 0.03,
        ),
        Gate(
            id="convergence_order_ge_1p8",
            quantity="Mean observed h-refinement order (L₂ norm)",
            reference="Q4 element: theoretical order 2.0 in L₂",
            value=float(np.mean(orders)) if orders else 0.0,
            threshold=1.8,
            operator=">=",
            passed=(float(np.mean(orders)) >= 1.8) if orders else False,
        ),
        Gate(
            id="heat_balance_residual",
            quantity="Max |K·T − f| (global conservation)",
            reference="0 (harmonic field, no source)",
            value=results[-1]["heat_balance_residual"],
            threshold=1e-10,
            operator="<=",
            passed=results[-1]["heat_balance_residual"] <= 1e-10,
        ),
    ]

    manifest = gate_manifest(gates)
    write_json(manifest, HERE / "gates_verified.json")

    # Summary
    all_pass = all(g.passed for g in gates)
    print(f"\n{'=' * 60}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Gates: {sum(g.passed for g in gates)}/{len(gates)} passed")
    for g in gates:
        status = "✅ PASS" if g.passed else "❌ FAIL"
        print(f"  {status}  {g.id}: {g.value:.6e} (threshold {g.operator} {g.threshold})")
    print(f"Figures: {fig_field.name}, {fig_convergence.name} -> {FIGURE_DIR}")
    print(f"Overall: {'✅ ALL PASS' if all_pass else '❌ SOME FAIL'}")
    print(f"{'=' * 60}")

    return 0 if all_pass else 1


# ---------------------------------------------------------------------------
# harness compatibility
# ---------------------------------------------------------------------------
class Gate:
    def __init__(self, id, quantity, reference, value, threshold, operator, passed):
        self.id = id
        self.quantity = quantity
        self.reference = reference
        self.value = value
        self.threshold = threshold
        self.operator = operator
        self.passed = passed

    def to_dict(self):
        return {
            "id": self.id,
            "quantity": self.quantity,
            "reference": self.reference,
            "value": self.value,
            "threshold": self.threshold,
            "operator": self.operator,
            "passed": self.passed,
        }


def gate_manifest(gates: list[Gate]) -> dict:
    return {
        "schema": "tensorfem.verified-gates.v1",
        "anchor": "fem_heat_conduction_l2",
        "title": "Steady heat conduction: manufactured harmonic solution, Q4 convergence ≥ 2nd order",
        "mission": "CASE-FEM-HEAT-L2",
        "generated_at": utc_now(),
        "criteria": {
            "finest_l2_below_3pct": 0.03,
            "convergence_order_ge_1p8": 1.8,
            "heat_balance_residual": 1e-10,
        },
        "gates": [g.to_dict() for g in gates],
        "passed": all(g.passed for g in gates),
        "failed_gates": [g.id for g in gates if not g.passed],
        "artifacts": {
            "result": "result.json",
            "gates": "gates_verified.json",
            "figures": [
                _repo_relative(FIGURE_DIR / "fig_convergence.png"),
                _repo_relative(FIGURE_DIR / "fig_field.png"),
            ],
        },
    }


def _repo_relative(path: Path) -> str:
    """How the certificate names a path: repo-relative POSIX, resolvable by the auditor."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:  # rendered outside the checkout (ad-hoc ``--out-dir``)
        return path.as_posix()


def _file_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(data: dict, path: Path):
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


if __name__ == "__main__":
    sys.exit(main())
