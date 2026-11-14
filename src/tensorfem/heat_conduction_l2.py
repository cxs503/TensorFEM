"""Manufactured-solution convergence study for steady heat conduction.

Callable entry point for the CASE-FEM-HEAT-L2 benchmark, usable both as a
library call from the platform / main solver and as the engine under the
``benchmarks/verified/fem_heat_conduction_l2`` certificate.

Usage::

    from tensorfem.heat_conduction_l2 import run_heat_conduction_l2
    result = run_heat_conduction_l2()
    print(f"Mean order: {result.mean_order:.3f}")

Measured norm (read this before trusting an observed order)
-----------------------------------------------------------
The acceptance criterion of CASE-FEM-HEAT-L2 is a *convergence rate*, so the
quantity that carries it must be the norm the rate theory is stated in: the
domain integral

    ||e||_0 = sqrt( int (T_h - T)^2 dOmega / int T^2 dOmega )      (L2)
    |e|_1   = sqrt( int |grad T_h - grad T|^2 / int |grad T|^2 )   (H1)

evaluated by :func:`integral_error_norms`, a Gauss post-processor that rebuilds
the bilinear field from the nodal values with its own shape functions and never
touches the assembly kernel, so an assembly bug cannot hide inside the norm.

Earlier revisions of this module -- and the ``fem_heat_conduction_l2``
certificate built on it -- reported ``sqrt(mean(e**2))`` over the *nodes*, a
nodal-RMS sampling pseudo-norm, and obtained observed orders 1.81 / 1.88 / 1.94
against the theoretical 2.0.  That deficit belongs to the *measurement*, not to
the solver: the Dirichlet boundary nodes carry exactly zero error by
construction, so the sample mixes an interior error with an exactly-zero
boundary layer and the ratio pseudo-norm / true norm drifts as ``O(h**-0.05)``
along the ladder, costing 0.03-0.19 of measured order.  The same kernel under
the true integral norm gives 1.9923 / 1.9978 / 1.9994 on the same meshes, and an
independently written sparse Q4 code reproduces the integral value at 32x32 to
every printed digit (``docs/CASE-FEM-HEAT-L2-norm-measurement-defect-m20995.md``).
The nodal-RMS number is still reported as ``l2_nodal_rms_relative`` so the older
records stay comparable, but it carries **no** gate.

Predeclared acceptance (fixed here before the ladder is run)
------------------------------------------------------------
``ORDER_GATE = 1.95``          the mean L2 order, the finest-pair L2 order and
                               every consecutive pair whose coarse mesh has
                               ``nx >= 8`` must satisfy ``p >= 1.95`` (theory
                               2.0, i.e. a 2.5% band);
``L2_GATE = 0.03``             finest-mesh integral L2 error below 3%;
``H1_ORDER_GATE = 0.90``       supplementary H1-seminorm order (theory 1.0);
``QUADRATURE_GATE = 1.0e-6``   for every level with ``nx >= 8`` the reported L2
                               and H1 norms must not move by more than this
                               relative amount when the post-processing rule
                               goes 4 -> 6 -> 8 points (no quadrature
                               contamination of the measured order -- the trap
                               that once invalidated a thermal "order 2.0"
                               claim in this repository);
``QUADRATURE_ORDER_GATE=1e-3`` the observed order must not move between the
                               6-point and 8-point post-processing;
``HEAT_BALANCE_GATE = 1e-10``  ``max |K T - f|`` over the free nodes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .heat_l2_convergence import gauss_legendre_1d, q4_shape, q4_shape_grad
from .thermal import ThermalModel, assemble_thermal, solve_steady_thermal

FLOAT = torch.float64
DEV = torch.device("cpu")

#: Predeclared gate constants (see the module docstring).
ORDER_GATE = 1.95
L2_GATE = 0.03
H1_ORDER_GATE = 0.90
QUADRATURE_GATE = 1.0e-6
QUADRATURE_ORDER_GATE = 1.0e-3
HEAT_BALANCE_GATE = 1.0e-10
DEFAULT_QUAD_ORDER = 6
#: Post-processing rules whose spread is gated (4 -> 6 -> 8 points per axis).
QUAD_ORDERS = (4, 6, 8)


# ---------------------------------------------------------------------------
# manufactured solution
# ---------------------------------------------------------------------------
def _exact_temperature(nodes: torch.Tensor) -> torch.Tensor:
    """T(x,y) = sin(πx)·sinh(πy)/sinh(π)."""
    x, y = nodes[:, 0], nodes[:, 1]
    return torch.sin(math.pi * x) * torch.sinh(math.pi * y) / math.sinh(math.pi)


def _exact_gradient(nodes: torch.Tensor) -> torch.Tensor:
    """∇T of the manufactured field (needed for the H1 seminorm)."""
    x, y = nodes[:, 0], nodes[:, 1]
    s = math.sinh(math.pi)
    return torch.stack((math.pi * torch.cos(math.pi * x) * torch.sinh(math.pi * y) / s,
                        math.pi * torch.sin(math.pi * x) * torch.cosh(math.pi * y) / s),
                       dim=1)


# ---------------------------------------------------------------------------
# mesh
# ---------------------------------------------------------------------------
def _structured_quad_mesh(nx: int, ny: int) -> tuple[torch.Tensor, torch.Tensor]:
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


def _edge_nodes(nx: int, ny: int) -> list[int]:
    boundary = set()
    for i in range(nx + 1):
        boundary.add(i * (ny + 1))
        boundary.add(i * (ny + 1) + ny)
    for j in range(ny + 1):
        boundary.add(j)
        boundary.add(nx * (ny + 1) + j)
    return sorted(boundary)


# ---------------------------------------------------------------------------
# error norms (independent post-processor)
# ---------------------------------------------------------------------------
def integral_error_norms(nodes: torch.Tensor, elements: torch.Tensor,
                         temperature: torch.Tensor,
                         quad_order: int = DEFAULT_QUAD_ORDER) -> dict[str, float]:
    """Domain-integral relative L2 and H1 errors of a bilinear (Q4) field.

    Independent of :mod:`tensorfem.thermal`: shape functions, Jacobian and
    quadrature are rebuilt here, so the norm cannot inherit an assembly bug.
    Both norms are normalised by the exact field and are therefore
    dimensionless.
    """
    coords = nodes[elements].to(FLOAT)                      # (ne, 4, 2)
    values = temperature[elements].to(FLOAT)                # (ne, 4)
    xi, w1 = gauss_legendre_1d(quad_order)
    num_l2 = den_l2 = num_h1 = den_h1 = 0.0
    for a in range(quad_order):
        for b in range(quad_order):
            weight = float(w1[a] * w1[b])
            shape = q4_shape(float(xi[a]), float(xi[b]))
            dshape = q4_shape_grad(float(xi[a]), float(xi[b]))    # (4, 2) = dN_i/dxi_c
            jac = torch.einsum("ik,eil->ekl", dshape, coords)     # J[e,k,l] = dX_k/dxi_l
            det = torch.linalg.det(jac)
            inv = torch.linalg.inv(jac)
            dshape_x = dshape.unsqueeze(0) @ inv.transpose(1, 2)   # (ne, 4, 2) = dN_i/dx_c
            points = torch.einsum("i,eic->ec", shape, coords)
            t_h = torch.einsum("i,ei->e", shape, values)
            t_e = _exact_temperature(points)
            g_h = torch.einsum("eic,ei->ec", dshape_x, values)
            g_e = _exact_gradient(points)
            measure = det * weight
            num_l2 = num_l2 + float((measure * (t_h - t_e) ** 2).sum())
            den_l2 = den_l2 + float((measure * t_e ** 2).sum())
            num_h1 = num_h1 + float((measure * ((g_h - g_e) ** 2).sum(dim=1)).sum())
            den_h1 = den_h1 + float((measure * (g_e ** 2).sum(dim=1)).sum())
    return {
        "l2_relative": math.sqrt(num_l2 / den_l2),
        "h1_relative": math.sqrt(num_h1 / den_h1),
    }


# ---------------------------------------------------------------------------
# single level
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HeatConductionLevel:
    nx: int
    ny: int
    n_elements: int
    n_nodes: int
    n_dof: int
    h: float
    l2_relative: float            # domain-integral relative L2 (gate metric)
    max_relative: float
    heat_balance_residual: float
    h1_relative: float = float("nan")            # domain-integral relative H1
    l2_nodal_rms_relative: float = float("nan")  # sampling pseudo-norm (no gate)
    quadrature_spread: float = 0.0               # max rel. spread of the norms, 4 vs 8 pt
    l2_by_quadrature: dict[int, float] = field(default_factory=dict)
    h1_by_quadrature: dict[int, float] = field(default_factory=dict)


def _solve_level(nx: int, ny: int) -> HeatConductionLevel:
    nodes, elems = _structured_quad_mesh(nx, ny)
    n_nodes = len(nodes)
    bdry = _edge_nodes(nx, ny)
    fixed_nodes = torch.tensor(bdry, dtype=torch.long, device=DEV)
    fixed_temps = _exact_temperature(nodes[fixed_nodes])

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
    T_exact = _exact_temperature(nodes)
    err = T_fem - T_exact

    norms = {q: integral_error_norms(nodes, elems, T_fem, q) for q in QUAD_ORDERS}
    l2_by_quad = {q: norms[q]["l2_relative"] for q in QUAD_ORDERS}
    h1_by_quad = {q: norms[q]["h1_relative"] for q in QUAD_ORDERS}
    reference = norms[DEFAULT_QUAD_ORDER]
    spread = max(abs(l2_by_quad[4] - l2_by_quad[8]) / reference["l2_relative"],
                 abs(h1_by_quad[4] - h1_by_quad[8]) / reference["h1_relative"])

    # kept for continuity with the pre-m20995 records; NOT the L2 norm
    nodal_rms = float(torch.sqrt(torch.mean(err ** 2)) / torch.sqrt(torch.mean(T_exact ** 2)))
    max_rel = float(torch.max(torch.abs(err / torch.where(T_exact != 0, T_exact, torch.ones_like(T_exact)))))

    matrices = assemble_thermal(model)
    free_mask = torch.ones(n_nodes, dtype=torch.bool, device=DEV)
    free_mask[fixed_nodes] = False
    residual_free = matrices.conductivity[free_mask] @ T_fem - matrices.heat[free_mask]
    heat_balance = float(torch.max(torch.abs(residual_free)))

    return HeatConductionLevel(
        nx=nx, ny=ny,
        n_elements=nx * ny,
        n_nodes=n_nodes,
        n_dof=n_nodes - len(bdry),
        h=1.0 / max(nx, ny),
        l2_relative=reference["l2_relative"],
        max_relative=max_rel,
        heat_balance_residual=heat_balance,
        h1_relative=reference["h1_relative"],
        l2_nodal_rms_relative=nodal_rms,
        quadrature_spread=spread,
        l2_by_quadrature=l2_by_quad,
        h1_by_quadrature=h1_by_quad,
    )


# ---------------------------------------------------------------------------
# convergence study
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HeatConductionL2Result:
    levels: list[HeatConductionLevel]
    observed_orders: list[float]
    mean_order: float
    all_gates_pass: bool
    h1_orders: list[float] = field(default_factory=list)
    nodal_rms_orders: list[float] = field(default_factory=list)
    order_shift_quadrature: float = 0.0
    gates: dict[str, bool] = field(default_factory=dict)


def _orders_from(values: list[float], levels: list[HeatConductionLevel]) -> list[float]:
    orders: list[float] = []
    for i in range(1, len(levels)):
        h_ratio = levels[i - 1].h / levels[i].h
        err_ratio = values[i - 1] / values[i]
        if err_ratio > 0 and h_ratio > 1:
            orders.append(math.log(err_ratio) / math.log(h_ratio))
        else:
            orders.append(0.0)
    return orders


def run_heat_conduction_l2(
    mesh_levels: list[tuple[int, int]] | None = None,
    order_threshold: float = ORDER_GATE,
    l2_threshold: float = L2_GATE,
) -> HeatConductionL2Result:
    """Run the CASE-FEM-HEAT-L2 convergence study.

    Args:
        mesh_levels: List of (nx, ny) tuples. Default: [(4,4),(8,8),(16,16),(32,32)].
        order_threshold: Minimum observed integral L2 order -- applied to the
            mean, to the finest pair, and to every pair whose coarse mesh has
            ``nx >= 8`` (default 1.95, a 2.5% band under the theoretical 2.0).
        l2_threshold: Maximum domain-integral relative L2 error at the finest
            mesh (default 0.03).

    Returns:
        HeatConductionL2Result with every level, the L2/H1/pseudo-norm order
        sequences and the gate verdicts.
    """
    if mesh_levels is None:
        mesh_levels = [(4, 4), (8, 8), (16, 16), (32, 32)]

    levels = [_solve_level(nx, ny) for nx, ny in mesh_levels]

    l2_values = [lv.l2_relative for lv in levels]
    orders = _orders_from(l2_values, levels)
    h1_orders = _orders_from([lv.h1_relative for lv in levels], levels)
    nodal_orders = _orders_from([lv.l2_nodal_rms_relative for lv in levels], levels)
    mean_order = sum(orders) / len(orders) if orders else 0.0

    # how far the observed order moves if the post-processing rule is changed
    orders_8pt = _orders_from([lv.l2_by_quadrature[8] for lv in levels], levels)
    order_shift = max((abs(a - b) for a, b in zip(orders, orders_8pt)), default=0.0)

    asymptotic = [p for p, lv in zip(orders, levels[:-1]) if lv.nx >= 8]
    finest = levels[-1]
    gates = {
        "finest_l2_below_3pct": finest.l2_relative <= l2_threshold,
        "mean_order_ge_threshold": mean_order >= order_threshold,
        "finest_order_ge_threshold": bool(orders) and orders[-1] >= order_threshold,
        "pair_orders_ge_threshold": all(p >= order_threshold for p in asymptotic),
        "h1_order_ge_threshold": all(p >= H1_ORDER_GATE
                                     for p, lv in zip(h1_orders, levels[:-1]) if lv.nx >= 8),
        "quadrature_norm_stability": all(lv.quadrature_spread <= QUADRATURE_GATE
                                         for lv in levels if lv.nx >= 8),
        "quadrature_order_stability": order_shift <= QUADRATURE_ORDER_GATE,
        "heat_balance_residual": finest.heat_balance_residual <= HEAT_BALANCE_GATE,
    }

    return HeatConductionL2Result(
        levels=levels,
        observed_orders=orders,
        mean_order=mean_order,
        all_gates_pass=all(gates.values()),
        h1_orders=h1_orders,
        nodal_rms_orders=nodal_orders,
        order_shift_quadrature=order_shift,
        gates=gates,
    )