"""Classical curved-shell benchmark meshes built from flat Shell4 facets.

The geometry is faceted deliberately: adjacent element local frames differ,
so membrane and bending actions transmit curvature through their common global
degrees of freedom.  This module contains no special benchmark correction.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from .shell4 import shell4_stiffness


@dataclass(frozen=True)
class ShellBenchmarkResult:
    displacement: torch.Tensor
    reaction: torch.Tensor
    nodes: torch.Tensor
    elements: torch.Tensor
    probe_dof: int

    @property
    def probe_displacement(self) -> float:
        return float(self.displacement[self.probe_dof])


def scordelis_lo_roof(
    nx: int,
    nt: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> ShellBenchmarkResult:
    """Solve the full Scordelis--Lo cylindrical-roof benchmark.

    Parameters follow the classical problem: length 50, radius 25, half-angle
    40 degrees, thickness .25, E=4.32e8, nu=0 and vertical surface load 90.
    The curved ends are supported by rigid diaphragms (global transverse and
    vertical translations fixed); one axial DOF removes the remaining rigid
    translation.  The reported response is downward displacement at the
    midspan of the positive-angle free edge.
    """
    if nx < 1 or nt < 1 or nx % 2 or nt % 2:
        raise ValueError("nx and nt must be positive even integers")
    dev = torch.device(device or "cpu")
    L, R, angle = 50.0, 25.0, math.radians(40.0)
    E, nu, thickness, pressure = 4.32e8, 0.0, .25, 90.0
    xs = torch.linspace(-L/2, L/2, nx+1, dtype=dtype, device=dev)
    ts = torch.linspace(-angle, angle, nt+1, dtype=dtype, device=dev)
    grid_x, grid_t = torch.meshgrid(xs, ts, indexing="ij")
    nodes = torch.stack((grid_x, R*torch.sin(grid_t), R*torch.cos(grid_t)), -1).reshape(-1, 3)
    conn = []
    for i in range(nx):
        for j in range(nt):
            a = i*(nt+1)+j
            conn.append((a, a+nt+1, a+nt+2, a+1))
    elements = torch.tensor(conn, dtype=torch.long, device=dev)
    ndof = 6*nodes.shape[0]
    K = torch.zeros((ndof, ndof), dtype=dtype, device=dev)
    f = torch.zeros(ndof, dtype=dtype, device=dev)
    # Consistent resultant of uniform vertical load on each planar facet.
    for element in elements:
        xyz = nodes[element]
        ke = shell4_stiffness(xyz, E, nu, thickness)
        dofs = torch.stack(tuple(6*element+k for k in range(6)), 1).reshape(-1)
        K[dofs[:, None], dofs] += ke
        area = .5*torch.linalg.vector_norm(torch.linalg.cross(xyz[1]-xyz[0], xyz[2]-xyz[0]))
        area += .5*torch.linalg.vector_norm(torch.linalg.cross(xyz[2]-xyz[0], xyz[3]-xyz[0]))
        f[6*element+2] -= pressure*area/4
    constrained = []
    for i in (0, nx):
        row = torch.arange(i*(nt+1), (i+1)*(nt+1), device=dev)
        constrained.extend((6*row+1).tolist())
        constrained.extend((6*row+2).tolist())
    constrained.append(0)  # eliminate free axial rigid translation
    fixed = torch.tensor(sorted(set(constrained)), dtype=torch.long, device=dev)
    free_mask = torch.ones(ndof, dtype=torch.bool, device=dev)
    free_mask[fixed] = False
    free = torch.nonzero(free_mask).flatten()
    u = torch.zeros(ndof, dtype=dtype, device=dev)
    u[free] = torch.linalg.solve(K[free[:, None], free], f[free])
    reaction = K@u-f
    probe_node = (nx//2)*(nt+1)+nt
    return ShellBenchmarkResult(u, reaction, nodes, elements, 6*probe_node+2)
