#!/usr/bin/env python3
"""CASE-FEM-HEAT-L2: Steady-state heat conduction convergence ≥ 2nd order.

Manufactured-solution verification of the TensorFEM Q4 heat-conduction kernel.

Physics
-------
Steady heat conduction on the unit square with a manufactured solution:

    T(x,y) = sin(πx) · sinh(πy) / sinh(π)

This field is exactly harmonic (∇²T = 0), so it satisfies the steady heat
equation with zero source.  Dirichlet boundary conditions are applied from the
exact field on all four edges, so the FE error is pure discretisation error --
a code-verification configuration, not a validation one.  The Q4 element with
2×2 Gauss quadrature is expected to converge at O(h²) in the L2 norm.

Reference
---------
The manufactured harmonic field is the classical FEM verification construction
(e.g. Zienkiewicz & Taylor, *The Finite Element Method*, vol. 1, 5th ed.,
Butterworth-Heinemann, 2000, ch. 3-4).  The exact expression is evaluated here;
no reference number is hand-copied.  The target order 2 is the formal order of
the bilinear element, stated in this repository's own versioned a-priori table
``src/tensorfem/adaptive_poisson_2d.py`` (``L2 norm  O(h^2)``).

Measured norm (m20995 correction -- read before trusting an order)
-------------------------------------------------------------------
The acceptance criterion is a *rate*, so it is carried by the norm the rate
theory is stated in: the domain integral

    ||e||_0 = sqrt( ∫(T_h - T)²dΩ / ∫T²dΩ ),   |e|_1 = sqrt( ∫|∇T_h-∇T|²/∫|∇T|² )

evaluated at 6×6 Gauss points by ``tensorfem.heat_conduction_l2.integral_error_norms``
(a post-processor that owns its shape functions and never touches the assembly
kernel), with the 4-point and 8-point values also reported so the quadrature
independence of the order is checkable.

Up to m20372 this file reported ``sqrt(mean(e²))`` over the *nodes* -- a
nodal-RMS sampling pseudo-norm -- and shipped the observed orders 1.810 / 1.884 /
1.935 as "approaching 2".  Those numbers measure the *metric*, not the solver:
the Dirichlet nodes contribute exactly zero error, so the sample mixes an
interior error with an exactly-zero boundary layer and the ratio
pseudo-norm/true-norm drifts as O(h^-0.05); the measured order is biased low by
0.03-0.19.  Under the true integral norm the *unchanged* kernel gives
1.99225 / 1.99776 / 1.99942 / 1.99985, and an independently written numpy/scipy
Q4 solver (``independent_sparse.py``, no ``tensorfem`` import) reproduces every
L2 value to 10 significant digits.  The pseudo-norm is kept as a diagnostic
(``l2_nodal_rms_relative``, ``nodal_rms_orders``) with no gate attached; the
evidence and the review are in
``docs/CASE-FEM-HEAT-L2-norm-measurement-defect-m20995.md``.

Acceptance gates (predeclared, see the table of the same name in ``README.md``)
------------------------------------------------------------------------------
``finest_l2_below_3pct``            integral L2 at the finest mesh (64×64) ≤ 3 %
``l2_pair_order_ge_1p95``           every consecutive pair whose coarse mesh has
                                    nx ≥ 8: observed L2 order ≥ 1.95 (theory 2.0,
                                    a 2.5 % band)
``l2_finest_order_in_band``         finest pair order inside [1.95, 2.10] (no
                                    runaway super-convergence either)
``power_law_residual_le_1pct``      Roache extrapolation of the two finest levels
                                    with p_ref = 2 leaves |e0|/e_f ≤ 1 %: the
                                    ladder really is the theoretical power law
``h1_pair_order_ge_0p9``            H1 seminorm order ≥ 0.9 (theory 1.0)
``quadrature_spread_le_1e-6``       reported norms move ≤ 1e-6 (relative) from
                                    4-point to 8-point post-processing on every
                                    mesh with nx ≥ 8
``independent_code_l2_agreement``   ``independent_sparse.py`` reproduces the L2
                                    norm of every shared level to ≤ 1e-9 relative
                                    and the nodal field to round-off
``heat_balance_residual``           max |K·T − f| over the free nodes ≤ 1e-10

Outputs
-------
* ``result.json``        — raw numbers at every mesh level
* ``gates_verified.json`` — pass/fail verdicts (schema ``tensorfem.verified-gates.v1``)
* ``docs/assets/verified/fem_heat_conduction_l2/fig_convergence.png``
                          — log-log error vs element size (integral norm, the
                            pseudo-norm, and the p = 2 / p = 1 slopes)
* ``docs/assets/verified/fem_heat_conduction_l2/fig_field.png``
                          — FEM temperature field, exact field, pointwise error

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

from tensorfem.heat_conduction_l2 import integral_error_norms  # noqa: E402

FLOAT = torch.float64
DEV = torch.device("cpu")

REPO_ROOT = HERE.parents[2]
#: Certificate figures live under ``docs/`` so the push gate carries them.
FIGURE_DIR = REPO_ROOT / "docs" / "assets" / "verified" / "fem_heat_conduction_l2"

#: Predeclared gate thresholds.
L2_GATE = 0.03
ORDER_GATE = 1.95
ORDER_BAND = (1.95, 2.10)
POWER_LAW_GATE = 1.0e-2
H1_ORDER_GATE = 0.90
QUADRATURE_GATE = 1.0e-6
CROSS_CHECK_GATE = 1.0e-9
HEAT_BALANCE_GATE = 1.0e-10
P_REF = 2.0
#: Levels shared with the independent solver for the cross-check.
CROSS_CHECK_LEVELS = (8, 16, 32, 64)


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
        boundary.add(j)                     # left
        boundary.add(nx * (ny + 1) + j)     # right
    return sorted(boundary)


# ---------------------------------------------------------------------------
# one mesh level
# ---------------------------------------------------------------------------
def solve_heat_conduction(nx: int, ny: int) -> dict:
    """Solve one level and return every error metric.

    ``l2_relative`` / ``h1_relative`` are the domain-integral norms (gate
    metrics).  ``l2_nodal_rms_relative`` is the former sampling pseudo-norm,
    kept as a diagnostic only.
    """
    from tensorfem.thermal import ThermalModel, assemble_thermal, solve_steady_thermal

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
    err = T_fem - T_exact

    norms = {q: integral_error_norms(nodes, elems, T_fem, q) for q in (4, 6, 8)}
    l2_by_quad = {str(q): norms[q]["l2_relative"] for q in (4, 6, 8)}
    quadrature_spread = max(
        abs(norms[4]["l2_relative"] - norms[8]["l2_relative"]) / norms[6]["l2_relative"],
        abs(norms[4]["h1_relative"] - norms[8]["h1_relative"]) / norms[6]["h1_relative"],
    )

    # legacy sampling pseudo-norm -- NOT the L2 norm, diagnostic only
    nodal_rms = float(torch.sqrt(torch.mean(err ** 2)) / torch.sqrt(torch.mean(T_exact ** 2)))

    # Max pointwise relative error
    max_abs = float(torch.max(torch.abs(err)))
    max_rel = float(torch.max(torch.abs(err / torch.where(T_exact != 0, T_exact, torch.ones_like(T_exact)))))

    # Heat balance: residual at free nodes should be ~machine precision
    matrices = assemble_thermal(model)
    free_mask = torch.ones(n_nodes, dtype=torch.bool, device=DEV)
    free_mask[fixed_nodes] = False
    residual_free = matrices.conductivity[free_mask] @ T_fem - matrices.heat[free_mask]
    heat_balance = float(torch.max(torch.abs(residual_free)))

    return {
        "nx": nx,
        "ny": ny,
        "n_elements": nx * ny,
        "n_nodes": n_nodes,
        "n_dof": n_nodes - len(bdry),
        "h": 1.0 / max(nx, ny),
        "l2_relative": norms[6]["l2_relative"],
        "l2_by_quadrature": l2_by_quad,
        "h1_relative": norms[6]["h1_relative"],
        "l2_nodal_rms_relative": nodal_rms,
        "quadrature_spread": quadrature_spread,
        "max_absolute": max_abs,
        "max_relative": max_rel,
        "heat_balance_residual": heat_balance,
        "_temperature": T_fem,
    }


# ---------------------------------------------------------------------------
# convergence study
# ---------------------------------------------------------------------------
MESH_LEVELS = [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]


def convergence_study() -> list[dict]:
    results = []
    for nx, ny in MESH_LEVELS:
        r = solve_heat_conduction(nx, ny)
        results.append(r)
        print(f"  {nx}x{ny}: n_elem={r['n_elements']}, "
              f"L2_int={r['l2_relative']:.6e}, L2_nodal={r['l2_nodal_rms_relative']:.6e}")
    return results


def observed_orders(results: list[dict], key: str = "l2_relative") -> list[float]:
    """Observed h-refinement order between successive levels for ``key``."""
    orders = []
    for i in range(1, len(results)):
        h_ratio = results[i - 1]["h"] / results[i]["h"]
        err_ratio = results[i - 1][key] / results[i][key]
        if err_ratio > 0 and h_ratio > 1:
            orders.append(math.log(err_ratio) / math.log(h_ratio))
        else:
            orders.append(float("nan"))
    return orders


def power_law_residual(results: list[dict], p_ref: float = P_REF) -> float:
    """Roache extrapolation residual of the two finest levels at the target order.

    ``e0 = e_f + (e_f - e_c) / (r**p_ref - 1)`` vanishes when the ladder is
    exactly the theoretical power law; the returned value is ``|e0| / e_f``.
    """
    e_c = results[-2]["l2_relative"]
    e_f = results[-1]["l2_relative"]
    r = results[-2]["h"] / results[-1]["h"]          # refinement ratio h_coarse/h_fine
    e0 = e_f + (e_f - e_c) / (r ** p_ref - 1.0)
    return abs(e0) / e_f


def load_independent_solver():
    """Import ``independent_sparse`` by path, independent of ``sys.path``.

    The cross-check solver sits next to this file and imports no ``tensorfem``
    module.  ``python benchmarks/verified/fem_heat_conduction_l2/run.py`` puts
    that directory on ``sys.path``, so a bare ``import independent_sparse`` used
    to work -- but only in that one invocation mode.  Reached as a library, from
    the test suite or from ``pytest benchmarks/verified/.../run.py``, the same
    import raised ``ModuleNotFoundError`` and the reproduction gate
    (``recompute_without_writing`` -> ``independent_cross_check``) could not run
    at all (m22937).  Loading by path makes the cross-check reachable from every
    entry point; the module is cached under its own name so repeated calls and an
    explicit ``import independent_sparse`` share one instance.
    """
    import importlib.util

    cached = sys.modules.get("independent_sparse")
    if cached is not None:
        return cached
    path = Path(__file__).resolve().parent / "independent_sparse.py"
    spec = importlib.util.spec_from_file_location("independent_sparse", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["independent_sparse"] = module
    spec.loader.exec_module(module)
    return module


def independent_cross_check(results: list[dict]) -> dict:
    """Reproduce the ladder with the from-scratch numpy/scipy Q4 solver."""
    isp = load_independent_solver()  # same directory, no tensorfem import

    entries = []
    worst_norm = 0.0
    worst_field = 0.0
    for level in results:
        nx, ny = level["nx"], level["ny"]
        if nx not in CROSS_CHECK_LEVELS:
            continue
        ref = isp.solve_level(nx, ny)
        norm_diff = abs(ref["l2_relative"] - level["l2_by_quadrature"]["8"]) / ref["l2_relative"]
        field_diff = float(np.max(np.abs(
            ref["temperature"] - level["_temperature"].numpy()))) / float(np.max(np.abs(ref["temperature"])))
        worst_norm = max(worst_norm, norm_diff)
        worst_field = max(worst_field, field_diff)
        entries.append({"nx": nx, "n_dof": ref["n_dof"],
                        "l2_relative_independent": ref["l2_relative"],
                        "l2_relative_kernel_8pt": level["l2_by_quadrature"]["8"],
                        "norm_relative_difference": norm_diff,
                        "field_max_relative_difference": field_diff})
    return {"entries": entries,
            "worst_norm_relative_difference": worst_norm,
            "worst_field_relative_difference": worst_field}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_convergence_figure(results: list[dict], orders: list[float],
                            nodal_orders: list[float], outpath: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    hs = [r["h"] for r in results]
    errs = [r["l2_relative"] for r in results]
    nodal = [r["l2_nodal_rms_relative"] for r in results]

    ax.loglog(hs, errs, "o-", color="C0", linewidth=2, markersize=8,
              label="Q4 L₂ (domain integral, gate metric)")
    ax.loglog(hs, nodal, "s--", color="C3", linewidth=1.4, markersize=6,
              label="nodal-RMS pseudo-norm (biased order, diagnostic)")

    # Reference slope lines anchored at the coarsest mesh
    h_ref = np.array([hs[0], hs[-1]])
    for slope, label in [(1.0, "O(h¹)"), (2.0, "O(h²)")]:
        e_ref = errs[0] * (h_ref / hs[0]) ** slope
        ax.loglog(h_ref, e_ref, ":", color="0.45", alpha=0.8, label=label)

    ax.set_xlabel("Element size h")
    ax.set_ylabel("Relative error")
    ax.set_title("CASE-FEM-HEAT-L2: Q4 steady conduction, harmonic MMS\n"
                 f"integral-norm orders: {', '.join(f'{o:.4f}' for o in orders)}"
                 f"  |  pseudo-norm: {', '.join(f'{o:.3f}' for o in nodal_orders)}")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def make_field_figure(nx: int, ny: int, outpath: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    level = solve_heat_conduction(nx, ny)
    nodes, _ = structured_quad_mesh(nx, ny)
    T_fem = level["_temperature"]
    T_exact = exact_temperature(nodes)
    err = (T_fem - T_exact).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    im0 = axes[0].tricontourf(nodes[:, 0].numpy(), nodes[:, 1].numpy(), T_fem.numpy(),
                               levels=30, cmap="viridis")
    axes[0].set_title("FEM temperature")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].tricontourf(nodes[:, 0].numpy(), nodes[:, 1].numpy(), T_exact.numpy(),
                               levels=30, cmap="viridis")
    axes[1].set_title("Exact temperature")
    axes[1].set_aspect("equal")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].tricontourf(nodes[:, 0].numpy(), nodes[:, 1].numpy(), err,
                               levels=30, cmap="RdBu_r")
    axes[2].set_title("Error (FEM − exact)")
    axes[2].set_aspect("equal")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(f"CASE-FEM-HEAT-L2: temperature field ({nx}×{ny} mesh, {nx*ny} elements)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# gates: built from the numbers, never typed
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


def build_gates(levels: list[dict], orders: list[float], h1_orders: list[float],
                residual: float, cross: dict) -> list[Gate]:
    """The eight acceptance gates, each one a scalar ``value <op> threshold``.

    Every verdict must be recomputable by a reader who has only the JSON, so no
    operator may be prose.  The band [1.95, 2.10] is therefore carried by two
    scalar gates rather than one ``"<= and >= 1.95 2.1"`` string: the upper bound
    is ``l2_finest_order_le_2p1`` and the lower bound is
    ``l2_pair_order_ge_1p95``, whose ``min`` runs over *all* asymptotic pairs and
    therefore bounds the finest pair from below too.  (The prose operator that
    used to ship here made the platform auditor record the gate as
    unverifiable; see ``docs/CASE-FEM-HEAT-L2-certificate-derivation-m22874.md``.)
    """
    asymptotic = [p for p, lv in zip(orders, levels[:-1]) if lv["nx"] >= 8]
    h1_asymptotic = [p for p, lv in zip(h1_orders, levels[:-1]) if lv["nx"] >= 8]
    return [
        Gate(
            id="finest_l2_below_3pct",
            quantity="Relative L₂ error at the finest mesh (64×64), domain integral",
            reference="T(x,y) = sin(πx)·sinh(πy)/sinh(π)",
            value=levels[-1]["l2_relative"],
            threshold=L2_GATE,
            operator="<=",
            passed=levels[-1]["l2_relative"] <= L2_GATE,
        ),
        Gate(
            id="l2_pair_order_ge_1p95",
            quantity="Worst observed h-refinement order over all pairs with coarse nx ≥ 8",
            reference="Q4 element: theoretical order 2.0 in L₂",
            value=min(asymptotic) if asymptotic else float("nan"),
            threshold=ORDER_GATE,
            operator=">=",
            passed=bool(asymptotic) and min(asymptotic) >= ORDER_GATE,
        ),
        Gate(
            id="l2_finest_order_le_2p1",
            quantity="Observed L₂ order of the finest pair (32×32 → 64×64), upper bound of the band [1.95, 2.10]",
            reference="theoretical order 2.0; no runaway super-convergence above 2.10",
            value=orders[-1],
            threshold=ORDER_BAND[1],
            operator="<=",
            passed=orders[-1] <= ORDER_BAND[1],
        ),
        Gate(
            id="power_law_residual_le_1pct",
            quantity="Roache extrapolation residual |e0|/e_f at p_ref = 2 (two finest levels)",
            reference="0 for an exact O(h²) power law",
            value=residual,
            threshold=POWER_LAW_GATE,
            operator="<=",
            passed=residual <= POWER_LAW_GATE,
        ),
        Gate(
            id="h1_pair_order_ge_0p9",
            quantity="Worst observed H1-seminorm order over all pairs with coarse nx ≥ 8",
            reference="Q4 element: theoretical order 1.0 in H1",
            value=min(h1_asymptotic) if h1_asymptotic else float("nan"),
            threshold=H1_ORDER_GATE,
            operator=">=",
            passed=bool(h1_asymptotic) and min(h1_asymptotic) >= H1_ORDER_GATE,
        ),
        Gate(
            id="quadrature_spread_le_1e-6",
            quantity="Worst relative norm spread between 4-point and 8-point post-processing (nx ≥ 8)",
            reference="the reported norms must not depend on the quadrature rule",
            value=max(lv["quadrature_spread"] for lv in levels if lv["nx"] >= 8),
            threshold=QUADRATURE_GATE,
            operator="<=",
            passed=all(lv["quadrature_spread"] <= QUADRATURE_GATE for lv in levels if lv["nx"] >= 8),
        ),
        Gate(
            id="independent_code_l2_agreement",
            quantity="Worst relative L₂ difference: from-scratch numpy/scipy solver vs kernel",
            reference="same discrete problem -> same norms to round-off",
            value=cross["worst_norm_relative_difference"],
            threshold=CROSS_CHECK_GATE,
            operator="<=",
            passed=cross["worst_norm_relative_difference"] <= CROSS_CHECK_GATE,
        ),
        Gate(
            id="heat_balance_residual",
            quantity="Max |K·T − f| (global conservation), finest mesh",
            reference="0 (harmonic field, no source)",
            value=levels[-1]["heat_balance_residual"],
            threshold=HEAT_BALANCE_GATE,
            operator="<=",
            passed=levels[-1]["heat_balance_residual"] <= HEAT_BALANCE_GATE,
        ),
    ]


def gate_manifest(gates: list[Gate]) -> dict:
    return {
        "schema": "tensorfem.verified-gates.v1",
        "anchor": "fem_heat_conduction_l2",
        "title": "Steady heat conduction: manufactured harmonic solution, Q4 convergence ≥ 2nd order",
        "mission": "CASE-FEM-HEAT-L2",
        "generated_at": utc_now(),
        "criteria": {
            "finest_l2_below_3pct": L2_GATE,
            "l2_pair_order_ge_1p95": ORDER_GATE,
            "l2_finest_order_le_2p1": ORDER_BAND[1],
            "power_law_residual_le_1pct": POWER_LAW_GATE,
            "h1_pair_order_ge_0p9": H1_ORDER_GATE,
            "quadrature_spread_le_1e-6": QUADRATURE_GATE,
            "independent_code_l2_agreement": CROSS_CHECK_GATE,
            "heat_balance_residual": HEAT_BALANCE_GATE,
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


def build_result_payload(*, levels, orders, h1_orders, nodal_orders, residual,
                         cross, elapsed, figures) -> dict:
    return {
        "benchmark": "fem_heat_conduction_l2",
        "case_key": "CASE-FEM-HEAT-L2",
        "description": "Steady-state heat conduction: manufactured harmonic solution on the unit square, Q4 elements",
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
            "target_order": P_REF,
            "target_order_source": "src/tensorfem/adaptive_poisson_2d.py (L2 norm O(h^2), bilinear Q4)",
            "source": "Manufactured harmonic field (Zienkiewicz & Taylor, FEM vol.1, ch.3-4)",
        },
        "norm_definition": {
            "gate_metric": "relative L2 domain integral at 6x6 Gauss (tensorfem.heat_conduction_l2.integral_error_norms)",
            "diagnostic": "l2_nodal_rms_relative = sqrt(mean_nodal(e^2)/mean_nodal(T^2)) -- sampling pseudo-norm, no gate",
            "quadrature_orders": [4, 6, 8],
        },
        "figures": figures,
        "mesh_levels": levels,
        "observed_orders_l2": orders,
        "observed_orders_h1": h1_orders,
        "nodal_rms_orders": nodal_orders,
        "mean_order": float(np.mean(orders)) if orders else 0.0,
        "mean_nodal_rms_order": float(np.mean(nodal_orders)) if nodal_orders else 0.0,
        "power_law_residual": residual,
        "independent_cross_check": cross,
    }


# ---------------------------------------------------------------------------
# README.md is a *derived* artifact  (m22874)
# ---------------------------------------------------------------------------
#: The README states "no number here is typed by hand".  That claim is only
#: worth something if the file is machine-produced, so it is rendered from the
#: two JSON artifacts and the reproduction gate compares the two byte for byte.
#: (Before m22874 the README was maintained by hand and had drifted: its
#: ``generated`` stamp named a run ~1.3 days older than the result.json it sits
#: next to, and nothing could notice.)
def render_readme(result: dict, manifest: dict) -> str:
    """Render ``README.md`` from ``result.json`` + ``gates_verified.json``."""
    levels = result["mesh_levels"]
    orders = result["observed_orders_l2"]
    h1_orders = result["observed_orders_h1"]
    nodal_orders = result["nodal_rms_orders"]
    gates = manifest["gates"]

    ladder = ["| Mesh | dof | h | L₂ (integral, gate) | L₂ nodal RMS (diagnostic) | observed order |",
              "|---|---|---|---|---|---|"]
    for i, lv in enumerate(levels):
        order = "—" if i == 0 else f"{orders[i - 1]:.5f}"
        ladder.append(
            f"| {lv['nx']}×{lv['ny']} | {lv['n_dof']} | {lv['h']:.4f} | "
            f"{lv['l2_relative']:.6e} | {lv['l2_nodal_rms_relative']:.6e} | {order} |"
        )

    gate_rows = ["| Gate | Quantity | Value | Threshold | Verdict |",
                 "|---|---|---|---|---|"]
    for g in gates:
        gate_rows.append(
            f"| `{g['id']}` | {g['quantity']} | {g['value']:.6e} | "
            f"{g['operator']} {g['threshold']!r} | {'PASS' if g['passed'] else 'FAIL'} |"
        )

    verdict = ("**ALL PASS**" if manifest["passed"]
               else "**SOME FAIL** (" + ", ".join(manifest["failed_gates"]) + ")")
    h1_text = ", ".join(f"{o:.4f}" for o in h1_orders)

    return "\n".join([
        "# CASE-FEM-HEAT-L2 — Q4 steady heat conduction, L₂ convergence ≥ 2nd order",
        "",
        "Manufactured harmonic field `T = sin(πx)·sinh(πy)/sinh(π)` on the unit square,",
        "Dirichlet conditions from the exact field, Q4 elements with 2×2 Gauss.",
        "Everything below is regenerated by `run.py`; no number here is typed by hand.",
        "",
        "```",
        "OMP_NUM_THREADS=8 PYTHONPATH=src python3 benchmarks/verified/fem_heat_conduction_l2/run.py",
        "```",
        "",
        "## Gate metric",
        "",
        "The acceptance is a *rate*, so the gate metric is the **domain integral**",
        "`||e||_0 = sqrt(∫(T_h-T)²dΩ/∫T²dΩ)` at 6×6 Gauss, together with the 4-point and",
        "8-point values so quadrature independence is checkable.  The former",
        "`sqrt(mean_nodal(e²))` sampling pseudo-norm is kept as a diagnostic only",
        "(`l2_nodal_rms_relative`): its Dirichlet nodes carry exactly zero error, which",
        f"biases the measured order low ({nodal_orders[0]:.2f} → {nodal_orders[-1]:.2f} "
        f"instead of {orders[0]:.2f} → {orders[-1]:.2f}).  See",
        "`docs/CASE-FEM-HEAT-L2-norm-measurement-defect-m20995.md`.",
        "",
        "## Ladder",
        "",
        *ladder,
        "",
        f"Mean integral order **{result['mean_order']:.5f}** (theory 2.0), H₁ orders",
        f"{h1_text} (theory 1.0),",
        f"Roache power-law residual {result['power_law_residual']:.2e} at p_ref = 2.",
        "The ladder is additionally reproduced by `independent_sparse.py`, a",
        "numpy/scipy Q4 solver that imports no `tensorfem` module (worst L₂ difference",
        f"{result['independent_cross_check']['worst_norm_relative_difference']:.2e}).",
        "",
        "## Gates",
        "",
        "Every gate is a scalar `value <operator> threshold` so a reader can recompute the",
        "verdict from this table alone; the band [1.95, 2.10] on the finest-pair order is",
        "carried jointly by `l2_pair_order_ge_1p95` (lower bound) and",
        "`l2_finest_order_le_2p1` (upper bound), because a prose operator is not",
        "machine-checkable.",
        "",
        *gate_rows,
        "",
        f"Overall: {verdict} ({len(gates)} gates,",
        f"generated {manifest['generated_at']}).",
        "",
        "## Reproduction",
        "",
        "`run.py --check` recomputes the ladder and verifies the committed artifacts",
        "without writing anything: the numbers must agree to 1e-12 relative, the gate",
        "verdicts exactly, the figure hashes with `result.json`, and this file byte for",
        "byte with `render_readme(result.json, gates_verified.json)`.  See",
        "`docs/CASE-FEM-HEAT-L2-certificate-derivation-m22874.md`.",
        "",
    ])


# ---------------------------------------------------------------------------
# reproduction gate
# ---------------------------------------------------------------------------
#: Fields that record *when* and *where* a run happened, not *what* it computed.
VOLATILE_KEYS = frozenset({"generated_at", "elapsed_seconds", "sha256", "environment"})

#: Predeclared relative tolerance for the reproduction gate.  Deliberately not
#: zero: a bitwise claim would only hold on one BLAS build, and "reproducible"
#: means the physics, not the reduction order.  1e-12 sits two orders of
#: magnitude below the sixth printed digit, so any drift that could change a
#: published number is still caught.
DRIFT_RTOL = 1.0e-12


def numeric_drift(committed, fresh, path="") -> tuple[float, list[str]]:
    """Walk two JSON trees; return (worst relative drift, list of messages)."""
    msgs: list[str] = []
    worst = 0.0

    def walk(a, b, here):
        nonlocal worst
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) - set(b)):
                msgs.append(f"{here}/{key}: absent from the fresh run")
            for key in sorted(set(b) - set(a)):
                msgs.append(f"{here}/{key}: absent from the committed artifact")
            for key in sorted(set(a) & set(b)):
                if key in VOLATILE_KEYS:
                    continue
                walk(a[key], b[key], f"{here}/{key}")
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                msgs.append(f"{here}: length {len(a)} committed vs {len(b)} fresh")
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{here}[{i}]")
        elif isinstance(a, bool) or isinstance(b, bool):
            if a is not b:
                msgs.append(f"{here}: {a!r} committed vs {b!r} fresh")
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if isinstance(a, int) and isinstance(b, int):
                if a != b:
                    msgs.append(f"{here}: {a} committed vs {b} fresh")
                return
            denom = max(abs(float(a)), abs(float(b)), 1e-300)
            drift = abs(float(a) - float(b)) / denom
            worst = max(worst, drift)
            if drift > DRIFT_RTOL:
                msgs.append(f"{here}: {a!r} committed vs {b!r} fresh (rel drift {drift:.3e})")
        else:
            if a != b:
                msgs.append(f"{here}: {a!r} committed vs {b!r} fresh")

    walk(committed, fresh, path or "root")
    return worst, msgs


def recompute_without_writing() -> tuple[list[dict], dict, tuple]:
    """Re-run the study and build the payloads -- nothing is written to disk."""
    results = convergence_study()
    orders = observed_orders(results)
    h1_orders = observed_orders(results, "h1_relative")
    nodal_orders = observed_orders(results, "l2_nodal_rms_relative")
    residual = power_law_residual(results)
    cross = independent_cross_check(results)
    levels = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    manifest = gate_manifest(build_gates(levels, orders, h1_orders, residual, cross))
    return levels, manifest, (orders, h1_orders, nodal_orders, residual, cross)


def run_check() -> int:
    """Verify the committed artifact set against a fresh run. Read-only."""
    committed_result = json.loads((HERE / "result.json").read_text())
    committed_manifest = json.loads((HERE / "gates_verified.json").read_text())
    readme_committed = (HERE / "README.md").read_text()

    print("=" * 60)
    print("CASE-FEM-HEAT-L2 reproduction gate (read-only)")
    print("=" * 60)

    failures: list[str] = []

    # 1. the README is the rendering of the committed numbers
    expected_readme = render_readme(committed_result, committed_manifest)
    if readme_committed == expected_readme:
        print(f"  [OK]   README.md == render_readme(result.json, gates_verified.json) "
              f"({len(readme_committed.encode())} bytes)")
    else:
        failures.append("README.md is not the rendering of the committed JSON")
        import difflib
        diff = list(difflib.unified_diff(
            expected_readme.splitlines(), readme_committed.splitlines(),
            "rendered-from-JSON", "README.md", lineterm="", n=1))
        print("  [FAIL] README.md drifted from its own numbers:")
        for line in diff[:24]:
            print(f"         {line}")

    # 2. the committed figures are the ones those numbers were drawn from
    recorded = {e["path"]: e["sha256"] for e in committed_result["figures"]}
    declared = set(committed_manifest["artifacts"]["figures"])
    if set(recorded) == declared and all(
            _file_sha256(REPO_ROOT / rel) == digest for rel, digest in recorded.items()):
        print(f"  [OK]   figures: {len(recorded)} declared, sha256 match result.json")
    else:
        failures.append("figure hashes do not match result.json")
        print("  [FAIL] figures: hash mismatch or declaration disagreement")

    # 3. re-run the study: numbers to 1e-12, verdicts exactly
    levels, manifest, (orders, h1_orders, nodal_orders, residual, cross) = recompute_without_writing()
    fresh_result = build_result_payload(
        levels=levels, orders=orders, h1_orders=h1_orders, nodal_orders=nodal_orders,
        residual=residual, cross=cross, elapsed=0.0,
        figures=committed_result["figures"],
    )
    worst, msgs = numeric_drift(committed_result, fresh_result)
    if not msgs:
        print(f"  [OK]   result.json: {len(levels)} levels reproduce "
              f"(worst relative drift {worst:.1e} <= {DRIFT_RTOL:.0e})")
    else:
        failures.append("result.json numbers drifted")
        print(f"  [FAIL] result.json: {len(msgs)} field(s) drifted (worst {worst:.1e}):")
        for m in msgs[:10]:
            print(f"         {m}")

    worst_g, msgs_g = numeric_drift(committed_manifest, manifest)
    if not msgs_g:
        print(f"  [OK]   gates_verified.json: {len(manifest['gates'])} verdicts reproduce "
              f"(worst relative drift {worst_g:.1e})")
    else:
        failures.append("gate verdicts drifted")
        print(f"  [FAIL] gates_verified.json: {len(msgs_g)} field(s) drifted:")
        for m in msgs_g[:10]:
            print(f"         {m}")

    print("-" * 60)
    if failures:
        print(f"NOT REPRODUCIBLE: {'; '.join(failures)}")
        return 1
    print("REPRODUCIBLE: the committed artifact set is a derived function of "
          "run.py on this checkout")
    return 0


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


def main() -> int:
    t0 = time.time()
    print("=" * 60)
    print("CASE-FEM-HEAT-L2: Steady heat conduction convergence study")
    print("=" * 60)

    results = convergence_study()
    orders = observed_orders(results)
    h1_orders = observed_orders(results, "h1_relative")
    nodal_orders = observed_orders(results, "l2_nodal_rms_relative")
    residual = power_law_residual(results)
    cross = independent_cross_check(results)

    # Finest mesh field figure
    nx_f, ny_f = MESH_LEVELS[-1]
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig_field = FIGURE_DIR / "fig_field.png"
    fig_convergence = FIGURE_DIR / "fig_convergence.png"
    make_field_figure(nx_f, ny_f, fig_field)
    make_convergence_figure(results, orders, nodal_orders, fig_convergence)

    elapsed = time.time() - t0

    # separable diagnostics: the temperature fields are not JSON material
    levels = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]

    figures = [
        {"path": _repo_relative(path), "sha256": _file_sha256(path)}
        for path in (fig_convergence, fig_field)
    ]
    result = build_result_payload(
        levels=levels, orders=orders, h1_orders=h1_orders, nodal_orders=nodal_orders,
        residual=residual, cross=cross, elapsed=elapsed, figures=figures,
    )
    write_json(result, HERE / "result.json")

    gates = build_gates(levels, orders, h1_orders, residual, cross)
    manifest = gate_manifest(gates)
    write_json(manifest, HERE / "gates_verified.json")

    # README.md is rendered from the two JSON files just written: the "no number
    # is typed by hand" claim is enforced by ``run.py --check``, which re-renders
    # it from the committed JSON and compares byte for byte.
    (HERE / "README.md").write_text(render_readme(result, manifest))

    # Summary
    all_pass = all(g.passed for g in gates)
    print(f"\n{'=' * 60}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Observed L2 orders (integral): {[round(o, 5) for o in orders]}")
    print(f"Observed L2 orders (nodal pseudo-norm, diagnostic): {[round(o, 4) for o in nodal_orders]}")
    print(f"Gates: {sum(g.passed for g in gates)}/{len(gates)} passed")
    for g in gates:
        status = "PASS" if g.passed else "FAIL"
        print(f"  [{status}] {g.id}: {g.value:.6e} (threshold {g.operator} {g.threshold})")
    print(f"Figures: {fig_field.name}, {fig_convergence.name} -> {FIGURE_DIR}")
    print("README.md rendered from result.json + gates_verified.json")
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    if "--check" in sys.argv[1:]:
        sys.exit(run_check())
    sys.exit(main())
