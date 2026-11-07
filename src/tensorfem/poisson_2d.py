"""2D Poisson problem on a structured quadrilateral mesh with analytical solution.

Provides a manufactured-solution benchmark for domain-decomposition verification:
    -∇²u = f  on Ω = [0,1]²,   u = g on ∂Ω

with u_exact(x,y) = sin(πx) sin(πy),  f = 2π² sin(πx) sin(πy).

The mesh is a regular Nx×Ny grid of bilinear Q4 elements.  Assembly produces
a sparse SPD stiffness matrix K and load vector F suitable for CG / Schwarz solvers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

Tensor = torch.Tensor
D = torch.float64


@dataclass(frozen=True)
class Poisson2DProblem:
    """Assembled 2D Poisson system on a structured Q4 mesh."""
    K: Tensor          # (ndof, ndof) sparse COO stiffness
    F: Tensor          # (ndof,) load vector
    u_exact: Tensor    # (ndof,) exact solution at all DOFs
    node_coords: Tensor  # (nnodes, 2) node coordinates
    nx: int            # number of elements in x
    ny: int            # number of elements in y
    element_dofs: Tensor  # (ne, 4) element-to-DOF mapping
    h: float           # characteristic element size

    @property
    def ndof(self) -> int:
        return self.K.shape[0]

    @property
    def nnodes(self) -> int:
        return self.node_coords.shape[0]

    @property
    def ne(self) -> int:
        return self.element_dofs.shape[0]


def _q4_shape_grad(xi: float, eta: float) -> list[tuple[float, float]]:
    """∇_ref N_i at (xi, eta) for bilinear Q4, ordered CCW from (-1,-1)."""
    dN = [
        (-0.25 * (1 - eta), -0.25 * (1 - xi)),
        ( 0.25 * (1 - eta), -0.25 * (1 + xi)),
        ( 0.25 * (1 + eta),  0.25 * (1 + xi)),
        (-0.25 * (1 + eta),  0.25 * (1 - xi)),
    ]
    return dN


def _q4_shape(xi: float, eta: float) -> list[float]:
    """N_i at (xi, eta) for bilinear Q4."""
    return [
        0.25 * (1 - xi) * (1 - eta),
        0.25 * (1 + xi) * (1 - eta),
        0.25 * (1 + xi) * (1 + eta),
        0.25 * (1 - xi) * (1 + eta),
    ]


def _element_stiffness(hx: float, hy: float) -> Tensor:
    """4×4 element stiffness for a rectangular Q4 element (hx, hy)."""
    # Jacobian: J = diag(hx/2, hy/2), detJ = hx*hy/4
    detJ = hx * hy / 4.0
    Jinv_x = 2.0 / hx
    Jinv_y = 2.0 / hy
    Ke = torch.zeros((4, 4), dtype=D)
    # 2×2 Gauss quadrature
    gp = 1.0 / torch.sqrt(torch.tensor(3.0, dtype=D))
    points = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    for xi, eta in points:
        dN_ref = _q4_shape_grad(xi, eta)
        # Map to physical: ∇_phys = J^{-T} ∇_ref
        dN_phys = [(dN_ref[i][0] * Jinv_x, dN_ref[i][1] * Jinv_y) for i in range(4)]
        for i in range(4):
            for j in range(4):
                Ke[i, j] += (dN_phys[i][0] * dN_phys[j][0] +
                             dN_phys[i][1] * dN_phys[j][1]) * detJ
    return Ke


def _element_load(hx: float, hy: float, cx: float, cy: float) -> Tensor:
    """4×1 element load vector for f = 2π² sin(πx) sin(πy)."""
    fe = torch.zeros(4, dtype=D)
    gp = 1.0 / torch.sqrt(torch.tensor(3.0, dtype=D))
    points = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    detJ = hx * hy / 4.0
    for xi, eta in points:
        N = _q4_shape(xi, eta)
        # Physical coordinates at Gauss point
        x = cx + hx / 2 * xi
        y = cy + hy / 2 * eta
        f_val = 2.0 * (torch.pi ** 2) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
        for i in range(4):
            fe[i] += f_val * N[i] * detJ
    return fe


def build_poisson_2d(nx: int, ny: int, *, device=None) -> Poisson2DProblem:
    """Build the 2D Poisson problem on an nx×ny Q4 mesh with Dirichlet BCs.

    All boundary DOFs are prescribed to the exact solution; interior DOFs
    are free.  The returned K and F are for the full system (before BC
    elimination), with boundary rows/columns identity-penalised.
    """
    dev = device or torch.device("cpu")
    nnx, nny = nx + 1, ny + 1  # nodes per direction
    nnodes = nnx * nny
    ndof = nnodes  # one DOF per node (scalar Poisson)

    # Node coordinates
    coords = torch.zeros((nnodes, 2), dtype=D, device=dev)
    for j in range(nny):
        for i in range(nnx):
            idx = j * nnx + i
            coords[idx, 0] = i / nx
            coords[idx, 1] = j / ny

    # Element-to-DOF mapping
    ne = nx * ny
    elem_dofs = torch.zeros((ne, 4), dtype=torch.long, device=dev)
    for ey in range(ny):
        for ex in range(nx):
            e = ey * nx + ex
            n0 = ey * nnx + ex
            n1 = n0 + 1
            n2 = n1 + nnx
            n3 = n0 + nnx
            elem_dofs[e] = torch.tensor([n0, n1, n2, n3], dtype=torch.long)

    hx, hy = 1.0 / nx, 1.0 / ny
    Ke = _element_stiffness(hx, hy)

    # Assemble global K and F
    rows_list, cols_list, vals_list = [], [], []
    F = torch.zeros(ndof, dtype=D, device=dev)

    for e in range(ne):
        dofs = elem_dofs[e]
        for i in range(4):
            for j in range(4):
                rows_list.append(int(dofs[i]))
                cols_list.append(int(dofs[j]))
                vals_list.append(float(Ke[i, j]))
        # Element load
        cx = (dofs[0] % nnx + 0.5) / nx
        cy = (dofs[0] // nnx + 0.5) / ny
        fe = _element_load(hx, hy, cx, cy)
        for i in range(4):
            F[dofs[i]] += fe[i]

    rows = torch.tensor(rows_list, dtype=torch.long, device=dev)
    cols = torch.tensor(cols_list, dtype=torch.long, device=dev)
    vals = torch.tensor(vals_list, dtype=D, device=dev)

    with torch.sparse.check_sparse_tensor_invariants(enable=False):
        K = torch.sparse_coo_tensor(
            torch.stack([rows, cols]), vals, (ndof, ndof),
            check_invariants=False).coalesce()

    # Apply Dirichlet BCs on boundary: identity penalty
    boundary_nodes = set()
    for i in range(nnx):
        boundary_nodes.add(i)                    # bottom
        boundary_nodes.add((nny - 1) * nnx + i)  # top
    for j in range(nny):
        boundary_nodes.add(j * nnx)              # left
        boundary_nodes.add(j * nnx + nnx - 1)    # right

    # Exact solution at all nodes
    u_exact = torch.sin(torch.pi * coords[:, 0]) * torch.sin(torch.pi * coords[:, 1])

    # Modify K and F for Dirichlet BCs
    bc_rows, bc_cols, bc_vals = [], [], []
    for node in boundary_nodes:
        bc_rows.append(node); bc_cols.append(node); bc_vals.append(1e12)
        F[node] = 1e12 * u_exact[node]

    if bc_rows:
        br = torch.tensor(bc_rows, dtype=torch.long, device=dev)
        bcl = torch.tensor(bc_cols, dtype=torch.long, device=dev)
        bv = torch.tensor(bc_vals, dtype=D, device=dev)
        K_bc = torch.sparse_coo_tensor(
            torch.stack([br, bcl]), bv, (ndof, ndof),
            check_invariants=False).coalesce()
        K = K + K_bc

    h_char = max(hx, hy)
    return Poisson2DProblem(K, F, u_exact, coords, nx, ny, elem_dofs, h_char)


def poisson_analytical(x: float, y: float) -> float:
    """Exact solution u(x,y) = sin(πx) sin(πy)."""
    return float(torch.sin(torch.pi * torch.tensor(x)) *
                 torch.sin(torch.pi * torch.tensor(y)))


def poisson_source(x: float, y: float) -> float:
    """Source term f(x,y) = 2π² sin(πx) sin(πy)."""
    return float(2.0 * (torch.pi ** 2) *
                 torch.sin(torch.pi * torch.tensor(x)) *
                 torch.sin(torch.pi * torch.tensor(y)))
