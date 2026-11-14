"""Manufactured-solution convergence study for steady heat conduction.

Provides a callable entry point for the CASE-FEM-HEAT-L2 benchmark,
usable both as a standalone script and as an integration caller from
the platform / main solver.

Usage::

    from tensorfem.heat_conduction_l2 import run_heat_conduction_l2
    result = run_heat_conduction_l2()
    print(f"Mean order: {result['mean_order']:.3f}")
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .thermal import ThermalModel, assemble_thermal, solve_steady_thermal

FLOAT = torch.float64
DEV = torch.device("cpu")


# ---------------------------------------------------------------------------
# manufactured solution
# ---------------------------------------------------------------------------
def _exact_temperature(nodes: torch.Tensor) -> torch.Tensor:
    """T(x,y) = sin(πx)·sinh(πy)/sinh(π)."""
    x, y = nodes[:, 0], nodes[:, 1]
    return torch.sin(math.pi * x) * torch.sinh(math.pi * y) / math.sinh(math.pi)


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
    l2_relative: float
    max_relative: float
    heat_balance_residual: float


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

    l2_rel = float(torch.sqrt(torch.mean(err ** 2)) / torch.sqrt(torch.mean(T_exact ** 2)))
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
        l2_relative=l2_rel,
        max_relative=max_rel,
        heat_balance_residual=heat_balance,
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


def run_heat_conduction_l2(
    mesh_levels: list[tuple[int, int]] | None = None,
    order_threshold: float = 1.8,
    l2_threshold: float = 0.03,
) -> HeatConductionL2Result:
    """Run the CASE-FEM-HEAT-L2 convergence study.

    Args:
        mesh_levels: List of (nx, ny) tuples. Default: [(4,4),(8,8),(16,16),(32,32)].
        order_threshold: Minimum mean observed order to pass (default 1.8).
        l2_threshold: Maximum relative L₂ error at finest mesh (default 0.03).

    Returns:
        HeatConductionL2Result with all levels, orders, and gate verdicts.
    """
    if mesh_levels is None:
        mesh_levels = [(4, 4), (8, 8), (16, 16), (32, 32)]

    levels = [_solve_level(nx, ny) for nx, ny in mesh_levels]

    orders: list[float] = []
    for i in range(1, len(levels)):
        h_ratio = levels[i - 1].h / levels[i].h
        err_ratio = levels[i - 1].l2_relative / levels[i].l2_relative
        if err_ratio > 0 and h_ratio > 1:
            orders.append(math.log(err_ratio) / math.log(h_ratio))
        else:
            orders.append(0.0)

    mean_order = sum(orders) / len(orders) if orders else 0.0

    finest = levels[-1]
    gates_pass = (
        finest.l2_relative <= l2_threshold
        and mean_order >= order_threshold
        and finest.heat_balance_residual <= 1e-10
    )

    return HeatConductionL2Result(
        levels=levels,
        observed_orders=orders,
        mean_order=mean_order,
        all_gates_pass=gates_pass,
    )
