"""Independent Q4 steady-conduction solver used to cross-check the kernel.

Deliberately written from scratch with **numpy + scipy only** -- it imports no
``tensorfem`` module -- so that the L2 ladder of
``benchmarks/verified/fem_heat_conduction_l2/run.py`` is confirmed by a second,
independently written implementation of the same discretisation:

* its own structured mesh and connectivity ordering (node ``(i,j) -> i*(ny+1)+j``,
  elements counter-clockwise BL, BR, TR, TL -- the layout of ``run.py``),
* its own bilinear shape functions, Jacobian and 2x2 Gauss assembly,
* its own Dirichlet elimination and ``scipy.sparse.linalg.spsolve``,
* its own 8x8 Gauss error post-processor.

The manufactured field is repeated here (two lines of trigonometry) rather than
imported, for the same reason.

Agreement of the two codes on the *same* discrete problem is a solver/assembly
check: both solve the identical linear system, so the nodal fields must agree to
round-off and the error norms to the level of the shared quadrature rule.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

FLOAT = np.float64
QUAD_ORDER = 8


def exact_temperature(xy: np.ndarray) -> np.ndarray:
    """T(x,y) = sin(px)*sinh(py)/sinh(p), p = pi."""
    x, y = xy[:, 0], xy[:, 1]
    return np.sin(math.pi * x) * np.sinh(math.pi * y) / math.sinh(math.pi)


def exact_gradient(xy: np.ndarray) -> np.ndarray:
    x, y = xy[:, 0], xy[:, 1]
    s = math.sinh(math.pi)
    return np.stack((math.pi * np.cos(math.pi * x) * np.sinh(math.pi * y) / s,
                     math.pi * np.sin(math.pi * x) * np.cosh(math.pi * y) / s), axis=1)


def _shape(xi: float, eta: float) -> np.ndarray:
    return np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                     (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)], dtype=FLOAT) / 4.0


def _shape_grad(xi: float, eta: float) -> np.ndarray:
    """(4, 2): dN_i/dxi_c in the reference square."""
    return np.array([[-(1 - eta), -(1 - xi)],
                     [(1 - eta), -(1 + xi)],
                     [(1 + eta), (1 + xi)],
                     [-(1 + eta), (1 - xi)]], dtype=FLOAT) / 4.0


def structured_mesh(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    xs = np.linspace(0.0, 1.0, nx + 1)
    ys = np.linspace(0.0, 1.0, ny + 1)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.stack([gx.ravel(), gy.ravel()], axis=1)
    index = np.arange(nodes.shape[0]).reshape(nx + 1, ny + 1)
    elements = np.stack([index[:-1, :-1].ravel(), index[1:, :-1].ravel(),
                         index[1:, 1:].ravel(), index[:-1, 1:].ravel()], axis=1)
    boundary = sorted(set(index[0, :]) | set(index[-1, :])
                      | set(index[:, 0]) | set(index[:, -1]))
    return nodes, elements, boundary


def assemble(nodes: np.ndarray, elements: np.ndarray) -> sp.csr_matrix:
    """Conductivity matrix for k = 1 and unit thickness, 2x2 Gauss."""
    n = nodes.shape[0]
    rows, cols, vals = [], [], []
    g = 1.0 / math.sqrt(3.0)
    coords = nodes[elements]
    for xi, eta in ((-g, -g), (g, -g), (g, g), (-g, g)):
        dshape = _shape_grad(xi, eta)                              # (4, 2)
        jac = np.einsum("ik,eil->ekl", dshape, coords)             # (ne, 2, 2)
        det = np.linalg.det(jac)
        inv = np.linalg.inv(jac)
        dshape_x = dshape[None, :, :] @ np.transpose(inv, (0, 2, 1))   # (ne, 4, 2)
        ke = np.einsum("eic,ejc->eij", dshape_x, dshape_x) * det[:, None, None]
        for a in range(4):
            for b in range(4):
                rows.append(elements[:, a])
                cols.append(elements[:, b])
                vals.append(ke[:, a, b])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def error_norms(nodes: np.ndarray, elements: np.ndarray, temperature: np.ndarray,
                quad_order: int = QUAD_ORDER) -> dict[str, float]:
    """8x8 Gauss domain integrals of the L2 and H1 errors (own post-processor)."""
    xi, w1 = np.polynomial.legendre.leggauss(quad_order)
    coords = nodes[elements]
    values = temperature[elements]
    num_l2 = den_l2 = num_h1 = den_h1 = 0.0
    for a in range(quad_order):
        for b in range(quad_order):
            weight = w1[a] * w1[b]
            shape = _shape(xi[a], xi[b])
            dshape = _shape_grad(xi[a], xi[b])
            jac = np.einsum("ik,eil->ekl", dshape, coords)
            det = np.linalg.det(jac)
            inv = np.linalg.inv(jac)
            dshape_x = dshape[None, :, :] @ np.transpose(inv, (0, 2, 1))
            points = np.einsum("i,eic->ec", shape, coords)
            t_h = np.einsum("i,ei->e", shape, values)
            t_e = exact_temperature(points)
            g_h = np.einsum("eic,ei->ec", dshape_x, values)
            g_e = exact_gradient(points)
            measure = det * weight
            num_l2 += float(np.sum(measure * (t_h - t_e) ** 2))
            den_l2 += float(np.sum(measure * t_e ** 2))
            num_h1 += float(np.sum(measure * np.sum((g_h - g_e) ** 2, axis=1)))
            den_h1 += float(np.sum(measure * np.sum(g_e ** 2, axis=1)))
    return {"l2_relative": math.sqrt(num_l2 / den_l2),
            "h1_relative": math.sqrt(num_h1 / den_h1)}


def solve_level(nx: int, ny: int) -> dict:
    nodes, elements, boundary = structured_mesh(nx, ny)
    matrix = assemble(nodes, elements)
    fixed = np.asarray(boundary, dtype=int)
    free = np.setdiff1d(np.arange(nodes.shape[0]), fixed)
    temperature = np.zeros(nodes.shape[0], dtype=FLOAT)
    temperature[fixed] = exact_temperature(nodes[fixed])
    rhs = -matrix[free][:, fixed] @ temperature[fixed]
    temperature[free] = spla.spsolve(matrix[free][:, free].tocsc(), rhs)
    norms = error_norms(nodes, elements, temperature)
    return {
        "nx": nx,
        "ny": ny,
        "n_nodes": int(nodes.shape[0]),
        "n_dof": int(free.size),
        "temperature": temperature,
        **norms,
    }