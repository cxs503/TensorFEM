"""HEX20 stress recovery: Gauss-point (Barlow) sampling, nodal least squares, pointwise.

Why this module exists
----------------------
``docs/plate-3d-exact-anchor.md`` §8 closed the *analytic* side of the shear
profile of the three-dimensional plate solution but left one item open
(§8.1, third bullet): the **finite-element recovery error** of ``tau_xz`` had no
independent reference, because the previous evidence bundle only read the
displacement field.  Answering that needs three things this module provides:

1. ``hex20_gauss_stresses`` — ``sigma = D B u`` at the ``3 x 3 x 3`` Gauss
   points, which are the superconvergent (Barlow) sampling locations of the
   quadratic brick;
2. ``hex20_nodal_stresses`` — per-element least-squares fit of the ``27`` Gauss
   tensors with the ``20`` serendipity shape functions, then element-averaged at
   shared nodes.  This is what a post-processor shows, and it is the field the
   previous note blamed for losing accuracy near the mid-surface;
3. ``pointwise_stress`` / ``line_profile`` — the exact derivative of the FE
   displacement field at an *arbitrary* point (Newton inversion of the
   isoparametric map), i.e. the best local recovery the discretisation can give
   without any extrapolation at all.

Keeping the three apart matters: they differ by ``O(h)``/``O(h^2)`` in the wrong
circumstances, and a verification that silently mixes them measures the
post-processor instead of the element.

Conventions
-----------
* Voigt order is the repository standard ``[xx, yy, zz, xy, yz, xz]``
  (``tensorfem.solid3d._b``), i.e. **``txz`` is component 5 and ``tyz`` is 4**.
  Anything comparing against ``plate_3d_exact.SSPlateExact3D.stress`` (which
  returns ``sxx, syy, szz, txy, txz, tyz``) must swap the last two.
* ``u`` is the assembled displacement vector of length ``3 * n_nodes`` with
  ``[ux0, uy0, uz0, ux1, ...]`` interleaving (the ``hex20_stiffness``
  convention).
"""
from __future__ import annotations

import math

import torch

from .hex20 import _gradient, hex20_shape_gradients
from .solid3d import _b, elasticity_matrix_3d

GAUSS_1D = (-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0))
WEIGHT_1D = (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)
GAUSS_3D = tuple((r, s, t) for r in GAUSS_1D for s in GAUSS_1D for t in GAUSS_1D)


def _dof_indices(cells: torch.Tensor) -> torch.Tensor:
    """``[ne, 60]`` interleaved dof indices, matching ``hex20_stiffness``."""
    return torch.stack(tuple(3 * cells + i for i in range(3)), 2).reshape(len(cells), 60)


def _material(d: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Broadcast a 6x6 (or [ne,6,6]) elasticity matrix to all elements."""
    d = torch.as_tensor(d, dtype=like.dtype, device=like.device)
    return d if d.ndim == 3 else d.unsqueeze(0)


def _element_stress(x: torch.Tensor, ue: torch.Tensor, d: torch.Tensor,
                    dn: torch.Tensor) -> torch.Tensor:
    """``sigma = D B u`` for a batch of elements at fixed natural coordinates."""
    g, _det = _gradient(x, dn)
    b = _b(g)
    strain = (b @ ue.unsqueeze(-1)).squeeze(-1)
    return (d @ strain.unsqueeze(-1)).squeeze(-1)


def hex20_stresses_at_natural(nodes: torch.Tensor, cells: torch.Tensor, u: torch.Tensor,
                              young, poisson, natural) -> torch.Tensor:
    """Stress ``[ne, npts, 6]`` at one or more natural coordinates per element.

    ``natural`` is either a single ``(r, s, t)`` triple or a sequence of triples.
    """
    pts = [natural] if len(natural) == 3 and not isinstance(natural[0], (tuple, list)) else list(natural)
    d = _material(elasticity_matrix_3d(
        nodes.new_tensor(float(young)).reshape(1), nodes.new_tensor(float(poisson)).reshape(1)), nodes)
    d = d.expand(len(cells), 6, 6)
    x = nodes[cells]
    ue = u[_dof_indices(cells)]
    out = []
    for r, s, t in pts:
        _n, dn = hex20_shape_gradients(r, s, t, like=nodes)
        out.append(_element_stress(x, ue, d, dn.detach()))
    return torch.stack(out, 1)


def hex20_point_coordinates(nodes: torch.Tensor, cells: torch.Tensor, natural) -> torch.Tensor:
    """Physical coordinates ``[ne, npts, 3]`` of natural points (shape-value map)."""
    pts = [natural] if len(natural) == 3 and not isinstance(natural[0], (tuple, list)) else list(natural)
    x = nodes[cells]
    rows = []
    for r, s, t in pts:
        n, _dn = hex20_shape_gradients(r, s, t, like=nodes)
        rows.append(torch.einsum("a,eai->ei", n.detach(), x))
    return torch.stack(rows, 1)


def hex20_gauss_stresses(nodes: torch.Tensor, cells: torch.Tensor, u: torch.Tensor,
                         young, poisson) -> dict:
    """Superconvergent sampling: stress at the 27 Gauss points of every element."""
    coords = hex20_point_coordinates(nodes, cells, GAUSS_3D)
    stress = hex20_stresses_at_natural(nodes, cells, u, young, poisson, GAUSS_3D)
    return {"coords": coords, "stress": stress, "n_points": len(GAUSS_3D)}


def hex20_nodal_stresses(nodes: torch.Tensor, cells: torch.Tensor, u: torch.Tensor,
                         young, poisson) -> dict:
    """Least-squares nodal recovery from the 27 Gauss tensors + element averaging.

    The serendipity shape functions are used as the fit basis on the Gauss
    points, so the fitted coefficients *are* the nodal values (``N_a(node_b) =
    delta_ab``).  ``27 > 20`` makes the system over-determined; it is solved in a
    least-squares sense.
    """
    g = hex20_gauss_stresses(nodes, cells, u, young, poisson)
    like = nodes
    amat = torch.stack([
        hex20_shape_gradients(r, s, t, like=like)[0].detach() for r, s, t in GAUSS_3D
    ])  # [27, 20]
    sol = torch.linalg.lstsq(amat.unsqueeze(0).expand(len(cells), 27, 20),
                             g["stress"])
    weights = sol.solution  # [ne, 20, 6]
    # element-averaged nodal field
    flat = cells.reshape(-1)
    val = weights.reshape(-1, 6)
    total = torch.zeros((len(nodes), 6), dtype=nodes.dtype)
    total.index_add_(0, flat, val)
    count = torch.zeros(len(nodes), dtype=nodes.dtype)
    count.index_add_(0, flat, torch.ones_like(flat, dtype=nodes.dtype))
    return {
        "nodal": total / count.clamp_min(1.0).unsqueeze(-1),
        "counts": count,
        "element_nodal": weights,
        "gauss_residual": float((amat.unsqueeze(0) @ weights - g["stress"]).abs().max()),
    }


def locate_hex20_element(nodes: torch.Tensor, cells: torch.Tensor, point,
                         tol: float = 1e-11, max_iter: int = 40) -> tuple[int, tuple]:
    """Find the element holding ``point`` and its natural coordinates (Newton).

    Raises ``ValueError`` when the point is outside the mesh or the inversion
    fails, so a caller can never silently probe the wrong element.
    """
    p = torch.as_tensor(point, dtype=nodes.dtype)
    corner_box = nodes[cells[:, :8]]
    lo = corner_box.min(dim=1).values
    hi = corner_box.max(dim=1).values
    inside = torch.all((p >= lo - tol) & (p <= hi + tol), dim=1)
    cand = torch.nonzero(inside).flatten()
    if len(cand) == 0:
        raise ValueError(f"point {tuple(round(float(v), 6) for v in p)} outside the mesh")
    for idx in cand.tolist():
        r = s = t = 0.0
        ok = False
        for _ in range(max_iter):
            n, dn = hex20_shape_gradients(r, s, t, like=nodes)
            x = nodes[cells[idx]]
            n, dn = n.detach(), dn.detach()
            cur = torch.einsum("a,ai->i", n, x)
            resid = cur - p
            jac = torch.einsum("ia,ib->ab", x, dn)
            step = torch.linalg.solve(jac, resid)
            r, s, t = r - float(step[0]), s - float(step[1]), t - float(step[2])
            if float(step.abs().max()) < tol:
                ok = True
                break
        if ok and max(abs(r), abs(s), abs(t)) <= 1.0 + 1e-8:
            return idx, (r, s, t)
    raise ValueError(f"point {tuple(round(float(v), 6) for v in p)} not inverted inside any element")


def pointwise_stress(nodes: torch.Tensor, cells: torch.Tensor, u: torch.Tensor,
                     young, poisson, point) -> dict:
    """Stress at an arbitrary physical point, from the element that contains it."""
    idx, nat = locate_hex20_element(nodes, cells, point)
    d = _material(elasticity_matrix_3d(
        nodes.new_tensor(float(young)).reshape(1), nodes.new_tensor(float(poisson)).reshape(1)), nodes)
    x = nodes[cells[idx]].unsqueeze(0)
    ue = u[_dof_indices(cells[idx:idx + 1])]
    _n, dn = hex20_shape_gradients(*nat, like=nodes)
    sig = _element_stress(x, ue, d.expand(1, 6, 6), dn.detach())
    return {"stress": sig[0], "element": idx, "natural": nat}


def line_profile(nodes: torch.Tensor, cells: torch.Tensor, u: torch.Tensor,
                 young, poisson, points, nodal: dict | None = None) -> torch.Tensor:
    """Pointwise recovered stress ``[npts, 6]`` along a list of physical points."""
    return torch.stack([pointwise_stress(nodes, cells, u, young, poisson, p)["stress"]
                        for p in points])


def interpolate_nodal(nodes: torch.Tensor, cells: torch.Tensor, nodal: torch.Tensor,
                      point) -> torch.Tensor:
    """Sample a nodal stress field at ``point`` with the containing element's shape functions.

    This is the "post-processor" reading of the same discretisation; comparing it
    with :func:`pointwise_stress` isolates recovery error from discretisation
    error.
    """
    idx, nat = locate_hex20_element(nodes, cells, point)
    n, _dn = hex20_shape_gradients(*nat, like=nodes)
    return torch.einsum("a,ai->i", n.detach(), nodal[cells[idx]])


def tau_xz(stress: torch.Tensor) -> torch.Tensor:
    """Shear component ``tau_xz`` under the repository's Voigt order."""
    return stress[..., 5]


def tau_yz(stress: torch.Tensor) -> torch.Tensor:
    """Shear component ``tau_yz`` under the repository's Voigt order."""
    return stress[..., 4]