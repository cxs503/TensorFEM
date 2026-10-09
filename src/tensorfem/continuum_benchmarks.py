"""Reproducible two-dimensional continuum benchmark builders.

The Cook membrane follows the conventional tapered panel with corners
``(0, 0), (48, 44), (48, 60), (0, 44)``.  A unit resultant vertical shear
traction is applied on the right edge.  The reported response is the vertical
displacement at its loaded-edge midpoint.
"""

from dataclasses import dataclass

import torch

from .continuum import ContinuumModel, solve_continuum


@dataclass(frozen=True)
class ContinuumBenchmarkResult:
    computed: float
    reference: float
    relative_error: float
    quantity: str


def mapped_q4_mesh(corners: torch.Tensor, nx: int, ny: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a structured unit-square grid to a bilinear quadrilateral."""
    if corners.shape != (4, 2) or nx < 1 or ny < 1:
        raise ValueError("corners must be [4,2] and nx, ny must be positive")
    xi = torch.linspace(0.0, 1.0, nx + 1, dtype=corners.dtype, device=corners.device)
    eta = torch.linspace(0.0, 1.0, ny + 1, dtype=corners.dtype, device=corners.device)
    ee, xx = torch.meshgrid(eta, xi, indexing="ij")
    shape = torch.stack(((1-xx)*(1-ee), xx*(1-ee), xx*ee, (1-xx)*ee), -1)
    nodes = (shape[..., None] * corners).sum(-2).reshape(-1, 2)
    cells = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            cells.append((n0, n0+1, n0+nx+2, n0+nx+1))
    return nodes, torch.tensor(cells, dtype=torch.long, device=corners.device)


def cook_membrane(nx: int, ny: int | None = None, *, dtype=torch.float64) -> tuple[ContinuumModel, int]:
    """Build the classical Cook membrane model and return its response DOF."""
    ny = nx if ny is None else ny
    if ny % 2:
        raise ValueError("Cook reference requires an even edge subdivision count")
    corners = torch.tensor(((0., 0.), (48., 44.), (48., 60.), (0., 44.)), dtype=dtype)
    nodes, elements = mapped_q4_mesh(corners, nx, ny)
    forces = torch.zeros(2 * len(nodes), dtype=dtype)
    right = torch.arange(nx, (ny + 1) * (nx + 1), nx + 1)
    # A constant boundary traction whose resultant is one. Consistent Q4 edge
    # nodal loads reduce to a trapezoidal distribution on a uniform edge mesh.
    forces[2 * right + 1] = 1.0 / ny
    forces[2 * right[[0, -1]] + 1] *= 0.5
    left = torch.arange(0, (ny + 1) * (nx + 1), nx + 1)
    fixed = torch.stack((2 * left, 2 * left + 1), 1).reshape(-1)
    model = ContinuumModel(
        nodes, elements, torch.tensor(1.0, dtype=dtype),
        torch.tensor(1.0/3.0, dtype=dtype), torch.tensor(1.0, dtype=dtype),
        forces, fixed, element_type="q4", plane="stress",
    )
    response_dof = 2 * int(right[ny // 2]) + 1
    return model, response_dof


def run_cook_membrane(nx: int = 16, *, reference: float = 23.96) -> ContinuumBenchmarkResult:
    model, dof = cook_membrane(nx)
    value = float(solve_continuum(model).displacement[dof])
    error = abs(value-reference) / abs(reference)
    return ContinuumBenchmarkResult(value, reference, error, "loaded-edge midpoint vertical displacement")
