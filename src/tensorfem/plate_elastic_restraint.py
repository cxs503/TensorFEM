"""Elastic rotational restraint boundary conditions for Mindlin-Reissner plates.

Problem statement
-----------------
Real ice-belt panels are bounded by longitudinal stiffeners and transverse
frames that provide *finite* rotational stiffness — neither simply supported
(SSSS, zero edge moment) nor fully clamped (CCCC, zero edge rotation).  The
existing plate solver only supports these two extremes, so its SSSS conclusion
can only be stated as an upper bound on deflection.  This module closes the gap
by introducing Winkler-type rotational springs at the plate edges, parameterised
by an embedding coefficient ``beta`` in [0, 1]:

* ``beta = 0``  →  simply supported (free normal rotation, K_rot = 0)
* ``beta = 1``  →  clamped (zero normal rotation, K_rot → ∞)
* ``0 < beta < 1``  →  partial fixity, M_edge = K_rot · theta_n_edge

Theory
------
For a Kirchhoff/Mindlin plate the edge moment–rotation relation is

    M_n = K_rot · theta_n

where ``M_n`` is the bending moment normal to the edge and ``theta_n`` is the
rotation about the edge normal.  The rotational spring stiffness is
parameterised as

    K_rot = zeta · D / a

with ``D = E t³ / (12(1-ν²))`` the flexural rigidity, ``a`` the plate dimension
normal to the edge, and ``zeta`` a dimensionless restraint parameter.  The
mapping from ``beta`` to ``zeta`` is

    zeta = beta / (1 - beta + eps)

so that ``beta = 0 → zeta = 0`` (free) and ``beta → 1 → zeta → 1/eps``
(very large, approximating clamped).  We use ``eps = 1e-6``.

This parameterisation follows Timoshenko & Woinowsky-Krieger, *Theory of Plates
and Shells*, 2nd ed., §33 (elastically restrained edges) and is consistent with
the non-dimensional restraint parameter used in naval architecture panel
collapse assessments (e.g. Paik & Thayamballi, *Ultimate Limit State Design
of Steel-Plated Structures*, §3.2).

Boundary condition convention (SS2 / hard simply supported)
-----------------------------------------------------------
Following the existing solver's convention:

* **w = 0** at all edge nodes (transverse displacement fixed)
* **theta_t = 0** at all edge nodes (tangential rotation fixed)
  - Edges x=0, x=a: theta_y is tangential (rotation about edge tangent) → fixed
  - Edges y=0, y=b: theta_x is tangential (rotation about edge tangent) → fixed
* **theta_n** controlled by rotational spring:
  - Edges x=0, x=a: theta_x is normal (rotation about edge normal) → spring
  - Edges y=0, y=b: theta_y is normal (rotation about edge normal) → spring

When ``beta = 0`` (K_rot = 0), this reduces exactly to SS2 (simply supported).
When ``beta → 1`` (K_rot → ∞), this approaches CCCC (clamped).

Implementation
--------------
The rotational spring contributes an additional diagonal stiffness term to the
global system.  For each edge node, the spring stiffness is added to the
rotational DOF that corresponds to rotation about the edge normal:

* Edges x = 0 and x = a: spring on theta_x (DOF index 3*i + 1)
* Edges y = 0 and y = b: spring on theta_y (DOF index 3*i + 2)

The tangential rotations are fixed directly (eliminated from the system),
matching the SS2 convention.

Validation
----------
* ``beta = 0`` must reproduce the simply supported (SS2) solution to machine
  precision.
* ``beta = 1`` (via ``eps``) must reproduce the clamped solution within 1%.
* Intermediate ``beta`` values must produce deflections monotonically between
  the two extremes.
* The non-dimensional deflection ``w·D/(q·a⁴)`` should match published values
  for elastically restrained plates (Timoshenko & Woinowsky-Krieger Table 8,
  or equivalent).

Reference
---------
Timoshenko, S. & Woinowsky-Krieger, S. (1959). *Theory of Plates and Shells*,
2nd ed., McGraw-Hill.  §33: "Rectangular plates with elastically restrained
edges", pp. 198-202.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .plate import q4_mindlin_stiffness, _shape, structured_square_mesh
from .winkler_foundation import q4_winkler_foundation_stiffness


# ---------------------------------------------------------------------------
# Restraint parameter mapping
# ---------------------------------------------------------------------------

def beta_to_zeta(beta: float, eps: float = 1e-6) -> float:
    """Map embedding coefficient ``beta`` in [0, 1] to dimensionless restraint ``zeta``.

    ``zeta = beta / (1 - beta + eps)`` so that:
    - ``beta = 0`` → ``zeta = 0`` (simply supported)
    - ``beta → 1`` → ``zeta → 1/eps`` (≈ clamped)
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    return beta / (1.0 - beta + eps)


def rotational_stiffness(
    beta: float,
    young: float,
    poisson: float,
    thickness: float,
    characteristic_length: float,
    eps: float = 1e-6,
) -> float:
    """Compute rotational spring stiffness K_rot = zeta · D / a.

    Parameters
    ----------
    beta : float
        Embedding coefficient in [0, 1].
    young : float
        Elastic modulus [Pa].
    poisson : float
        Poisson's ratio.
    thickness : float
        Plate thickness [m].
    characteristic_length : float
        Plate dimension normal to the edge [m].
    eps : float
        Regularisation for beta → 1.

    Returns
    -------
    K_rot : float
        Rotational spring stiffness [N·m/rad].
    """
    D = young * thickness**3 / (12.0 * (1.0 - poisson**2))
    zeta = beta_to_zeta(beta, eps)
    return zeta * D / characteristic_length


# ---------------------------------------------------------------------------
# Edge identification
# ---------------------------------------------------------------------------

def identify_edge_nodes(
    nodes: torch.Tensor,
    side: float,
    tol: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Identify nodes on each edge of a square plate [0, side] × [0, side].

    Returns
    -------
    edges : dict
        Keys: 'x0', 'xa', 'y0', 'ya' (edges at x=0, x=side, y=0, y=side).
        Values: 1-D tensors of node indices.
    """
    tol_abs = side * tol
    x0 = torch.nonzero(nodes[:, 0] < tol_abs, as_tuple=False).flatten()
    xa = torch.nonzero((nodes[:, 0] - side).abs() < tol_abs, as_tuple=False).flatten()
    y0 = torch.nonzero(nodes[:, 1] < tol_abs, as_tuple=False).flatten()
    ya = torch.nonzero((nodes[:, 1] - side).abs() < tol_abs, as_tuple=False).flatten()
    return {'x0': x0, 'xa': xa, 'y0': y0, 'ya': ya}


def assemble_rotational_spring_stiffness(
    nodes: torch.Tensor,
    side: float,
    K_rot: float,
    edges: Optional[dict[str, torch.Tensor]] = None,
    ndof: Optional[int] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Assemble global rotational spring stiffness matrix.

    For each edge node, add K_rot to the rotational DOF about the edge normal:
    - Edges x=0, x=a: spring on theta_x (DOF 3*i+1) — normal to x-edges
    - Edges y=0, y=b: spring on theta_y (DOF 3*i+2) — normal to y-edges

    The tangential rotations (theta_y on x-edges, theta_x on y-edges) are
    handled by the boundary condition elimination, not by springs.

    Parameters
    ----------
    nodes : (N, 2) tensor
        Nodal coordinates.
    side : float
        Plate side length.
    K_rot : float
        Rotational spring stiffness [N·m/rad].
    edges : dict, optional
        Pre-computed edge node indices (from :func:`identify_edge_nodes`).
    ndof : int, optional
        Total number of DOFs (3 * N).  Computed if not given.
    dtype : torch.dtype
        Data type.

    Returns
    -------
    K_spring : (ndof, ndof) tensor
        Diagonal rotational spring stiffness matrix.
    """
    if edges is None:
        edges = identify_edge_nodes(nodes, side)
    if ndof is None:
        ndof = 3 * nodes.shape[0]

    device = nodes.device
    K_spring = torch.zeros((ndof, ndof), dtype=dtype, device=device)

    if K_rot == 0.0:
        return K_spring

    # Edges x=0 and x=a: spring on theta_x (DOF 3*i+1) — normal rotation
    for edge_key in ('x0', 'xa'):
        for node_idx in edges[edge_key]:
            dof = 3 * int(node_idx) + 1
            K_spring[dof, dof] += K_rot

    # Edges y=0 and y=b: spring on theta_y (DOF 3*i+2) — normal rotation
    for edge_key in ('y0', 'ya'):
        for node_idx in edges[edge_key]:
            dof = 3 * int(node_idx) + 2
            K_spring[dof, dof] += K_rot

    return K_spring


# ---------------------------------------------------------------------------
# Boundary condition application
# ---------------------------------------------------------------------------

def _apply_bc_ss2(
    nodes: torch.Tensor,
    side: float,
    ndof: int,
) -> torch.Tensor:
    """Return fixed DOFs for SS2 (hard simply supported): w + tangential rotation fixed.

    - w = 0 at all edge nodes
    - theta_y = 0 at x-edges (tangential to x-edges)
    - theta_x = 0 at y-edges (tangential to y-edges)
    """
    tol = side * 1e-12
    xedge = (nodes[:, 0].abs() < tol) | ((nodes[:, 0] - side).abs() < tol)
    yedge = (nodes[:, 1].abs() < tol) | ((nodes[:, 1] - side).abs() < tol)
    edge = xedge | yedge

    fixed = torch.cat((
        # w fixed at all edge nodes
        3 * torch.nonzero(edge, as_tuple=False).flatten(),
        # theta_y fixed at x-edges (tangential to x-edges)
        3 * torch.nonzero(xedge, as_tuple=False).flatten() + 2,
        # theta_x fixed at y-edges (tangential to y-edges)
        3 * torch.nonzero(yedge, as_tuple=False).flatten() + 1,
    )).unique()

    return fixed


def _apply_bc_clamped(
    nodes: torch.Tensor,
    side: float,
    ndof: int,
) -> torch.Tensor:
    """Return fixed DOFs for clamped: w + both rotations fixed."""
    tol = side * 1e-12
    edge = (nodes[:, 0].abs() < tol) | ((nodes[:, 0] - side).abs() < tol) | \
           (nodes[:, 1].abs() < tol) | ((nodes[:, 1] - side).abs() < tol)

    fixed = torch.cat((
        3 * torch.nonzero(edge, as_tuple=False).flatten(),
        3 * torch.nonzero(edge, as_tuple=False).flatten() + 1,
        3 * torch.nonzero(edge, as_tuple=False).flatten() + 2,
    )).unique()

    return fixed


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ElasticRestraintPlateResult:
    """Result of plate solve with elastic rotational restraint."""
    displacement: torch.Tensor
    nodes: torch.Tensor
    elements: torch.Tensor
    young: float
    poisson: float
    thickness: float
    winkler_k: float
    beta: float
    K_rot: float
    boundary: str


def solve_plate_elastic_restraint(
    nx: int,
    side: float,
    young: float,
    poisson: float,
    thickness: float,
    winkler_k: float,
    pressure: float | callable,
    beta: float = 0.0,
    boundary: str = "elastic_restraint",
    dtype: torch.dtype = torch.float64,
    eps: float = 1e-6,
) -> ElasticRestraintPlateResult:
    """Solve a Mindlin plate with elastic rotational restraint at the edges.

    Parameters
    ----------
    nx : int
        Elements per side.
    side : float
        Plate side length [m].
    young : float
        Elastic modulus [Pa].
    poisson : float
        Poisson's ratio.
    thickness : float
        Plate thickness [m].
    winkler_k : float
        Winkler foundation coefficient [N/m³].
    pressure : float or callable
        Uniform pressure or pressure(x, y) function.
    beta : float
        Embedding coefficient in [0, 1].  0 = simply supported, 1 = clamped.
    boundary : str
        'simply_supported', 'clamped', or 'elastic_restraint'.
    dtype : torch.dtype
        Data type.
    eps : float
        Regularisation for beta → 1.

    Returns
    -------
    result : ElasticRestraintPlateResult
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")

    nodes, elements = structured_square_mesh(nx, side, dtype=dtype)
    n_nodes = nodes.shape[0]
    ndof = 3 * n_nodes

    # Compute rotational spring stiffness
    K_rot = rotational_stiffness(beta, young, poisson, thickness, side, eps)

    # Assemble plate stiffness (bending + Winkler)
    K = torch.zeros((ndof, ndof), dtype=dtype)
    for conn in elements:
        xy = nodes[conn]
        ke = q4_mindlin_stiffness(xy, young, poisson, thickness)
        ke += q4_winkler_foundation_stiffness(xy, winkler_k)
        dofs = torch.stack((3*conn, 3*conn+1, 3*conn+2), dim=1).reshape(-1)
        K[dofs[:, None], dofs] += ke

    # Add rotational spring stiffness (only for elastic_restraint mode)
    if boundary == "elastic_restraint" and K_rot > 0.0:
        edges = identify_edge_nodes(nodes, side)
        K_spring = assemble_rotational_spring_stiffness(nodes, side, K_rot, edges, ndof, dtype)
        K = K + K_spring

    # Assemble load vector
    f = torch.zeros(ndof, dtype=dtype)
    for conn in elements:
        xy = nodes[conn]
        fe = _consistent_load_vector(xy, pressure, dtype)
        dofs = torch.stack((3*conn, 3*conn+1, 3*conn+2), dim=1).reshape(-1)
        f[dofs] += fe

    # Apply boundary conditions
    if boundary == "elastic_restraint":
        # SS2-style: w + tangential rotation fixed; normal rotation via spring
        fixed = _apply_bc_ss2(nodes, side, ndof)
    elif boundary == "simply_supported":
        fixed = _apply_bc_ss2(nodes, side, ndof)
    elif boundary == "clamped":
        fixed = _apply_bc_clamped(nodes, side, ndof)
    else:
        raise ValueError(f"Unknown boundary: {boundary}")

    all_dofs = torch.arange(ndof)
    free = all_dofs[~torch.isin(all_dofs, fixed)]

    K_ff = K[free][:, free]
    f_f = f[free]

    u_f = torch.linalg.solve(K_ff, f_f)

    u = torch.zeros(ndof, dtype=dtype)
    u[free] = u_f

    return ElasticRestraintPlateResult(
        displacement=u,
        nodes=nodes,
        elements=elements,
        young=young,
        poisson=poisson,
        thickness=thickness,
        winkler_k=winkler_k,
        beta=beta,
        K_rot=K_rot,
        boundary=boundary,
    )


def _consistent_load_vector(
    xy: torch.Tensor,
    pressure: torch.Tensor | float | callable,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Consistent load vector for a Q4 element (w DOFs only)."""
    device = xy.device
    fe = torch.zeros(12, dtype=dtype, device=device)

    gp = 1.0 / 3.0**0.5
    for xif in (-gp, gp):
        for etaf in (-gp, gp):
            xi = xy.new_tensor(xif)
            eta = xy.new_tensor(etaf)
            N, dN_nat = _shape(xi, eta)
            J = dN_nat.T @ xy
            detJ = torch.linalg.det(J)

            if callable(pressure):
                p = N @ xy
                q = pressure(p[0], p[1])
                if not torch.is_tensor(q):
                    q = xy.new_tensor(q)
            else:
                q = torch.as_tensor(pressure, dtype=dtype, device=device)

            for i in range(4):
                fe[3*i] += N[i] * q * detJ

    return fe


# ---------------------------------------------------------------------------
# Beta sweep utility
# ---------------------------------------------------------------------------

def beta_sweep(
    nx: int,
    side: float,
    young: float,
    poisson: float,
    thickness: float,
    winkler_k: float,
    pressure: float,
    beta_values: Optional[list[float]] = None,
    dtype: torch.dtype = torch.float64,
) -> list[dict]:
    """Solve plate for a range of beta values and collect key results.

    Returns
    -------
    results : list of dict
        Each dict contains: beta, K_rot, w_center, w_center_normalized, zeta.
    """
    if beta_values is None:
        beta_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]

    D = young * thickness**3 / (12.0 * (1.0 - poisson**2))
    center_x, center_y = side / 2, side / 2

    out = []
    for beta in beta_values:
        res = solve_plate_elastic_restraint(
            nx=nx, side=side, young=young, poisson=poisson,
            thickness=thickness, winkler_k=winkler_k, pressure=pressure,
            beta=beta, dtype=dtype,
        )
        # Find center node
        dist = (res.nodes[:, 0] - center_x)**2 + (res.nodes[:, 1] - center_y)**2
        center_idx = torch.argmin(dist).item()
        w_center = res.displacement[3 * center_idx].item()
        w_norm = w_center * D / (pressure * side**4)

        out.append({
            'beta': beta,
            'zeta': beta_to_zeta(beta),
            'K_rot': res.K_rot,
            'w_center': w_center,
            'w_center_normalized': w_norm,
        })

    return out
