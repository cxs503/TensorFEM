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


# ===========================================================================
# Rectangular panels with per-edge elastic restraint, sparse assembly and
# bending-moment / stress recovery.
#
# Why this exists: the paper anchor (JOUR-ICE-STRUCT Sec. 4.2) states its SSSS
# result as *an upper bound only*, because a real ice-belt plate field is
# bounded by longitudinals on two sides and transverse frames on the other two
# -- i.e. two different finite rotational restraints on orthogonal edges, on a
# rectangular (0.8 m x 0.7 m) plate field.  The square/single-beta solver above
# cannot express that, and it returns displacement only, so the quantity the
# paper reports (peak bending stress) cannot be cross-checked at all.
#
# Everything below is additive: `solve_plate_elastic_restraint` and
# `beta_sweep` keep their behaviour bit-for-bit (the vectorised assembly here is
# asserted against the historical loop assembly by
# tests/test_plate_restraint_panel.py::test_sparse_path_matches_dense_path).
# ---------------------------------------------------------------------------

def structured_rect_mesh(
    nx: int,
    ny: int,
    side_x: float,
    side_y: float,
    *,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Structured Q4 mesh over ``[0, side_x] x [0, side_y]``.

    Node ``j*(nx+1)+i`` sits at ``(i*side_x/nx, j*side_y/ny)``; element
    ``(i,j)`` is ``(n0, n0+1, n0+nx+2, n0+nx+1)`` i.e. counter-clockwise in
    natural coordinates ``(-1,-1), (1,-1), (1,1), (-1,1)``.  With
    ``nx == ny`` and ``side_x == side_y`` the connectivity is the same as
    :func:`tensorfem.plate.structured_square_mesh`.
    """
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive")
    x = torch.linspace(0.0, side_x, nx + 1, dtype=dtype)
    y = torch.linspace(0.0, side_y, ny + 1, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    nodes = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    jj, ii = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing="ij")
    n0 = (jj * (nx + 1) + ii).reshape(-1)
    elements = torch.stack((n0, n0 + 1, n0 + nx + 2, n0 + nx + 1), dim=1)
    return nodes, elements.long()


def _rect_edges(
    nodes: torch.Tensor,
    side_x: float,
    side_y: float,
    tol: float = 1e-9,
) -> dict[str, torch.Tensor]:
    """Edge node indices of a rectangular panel (``x0, xa, y0, yb``)."""
    tx, ty = side_x * tol, side_y * tol
    return {
        'x0': torch.nonzero(nodes[:, 0].abs() < tx, as_tuple=False).flatten(),
        'xa': torch.nonzero((nodes[:, 0] - side_x).abs() < tx, as_tuple=False).flatten(),
        'y0': torch.nonzero(nodes[:, 1].abs() < ty, as_tuple=False).flatten(),
        'yb': torch.nonzero((nodes[:, 1] - side_y).abs() < ty, as_tuple=False).flatten(),
    }


def _rigidities(young: float, poisson: float, thickness: float, dtype):
    """Bending ``Db`` (3x3) and transverse-shear ``Ds`` (2x2) matrices."""
    E = torch.as_tensor(young, dtype=dtype)
    nu = torch.as_tensor(poisson, dtype=dtype)
    t = torch.as_tensor(thickness, dtype=dtype)
    one, zero = torch.ones_like(nu), torch.zeros_like(nu)
    Db = E * t**3 / (12 * (1 - nu**2)) * torch.stack((
        torch.stack((one, nu, zero)),
        torch.stack((nu, one, zero)),
        torch.stack((zero, zero, (1 - nu) / 2)),
    ))
    Ds = (5.0 / 6.0) * E / (2 * (1 + nu)) * t * torch.eye(2, dtype=dtype)
    return Db, Ds


def _batched_mindlin_stiffness(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    young: float,
    poisson: float,
    thickness: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorised Q4 Mindlin stiffness for every element at once.

    Bending block: 2x2 Gauss.  Shear block: one-point (SRI), the same
    ``shear_treatment="sri"`` rule :func:`tensorfem.plate.q4_mindlin_stiffness`
    uses by default, so the two paths describe the same discrete operator.

    Returns
    -------
    ke : (ne, 12, 12)
    Bb : (ne, 4, 3, 12)  bending strain-displacement at the four Gauss points,
                          ordered ``(-g,-g), (g,-g), (g,g), (-g,g)``
    detJ : (ne, 4)
    """
    dtype = nodes.dtype
    xy = nodes[elements]                                    # (ne, 4, 2)
    Db, Ds = _rigidities(young, poisson, thickness, dtype)
    ne = xy.shape[0]
    gauss = 1.0 / 3.0**0.5
    knots = ((-gauss, -gauss), (gauss, -gauss), (gauss, gauss), (-gauss, gauss))
    lidx = torch.arange(4)

    Bb_all, detJ_all = [], []
    for xi_v, eta_v in knots:
        xi, eta = xy.new_tensor(xi_v), xy.new_tensor(eta_v)
        _, dN_nat = _shape(xi, eta)                         # (4, 2)
        J = torch.einsum('ai,naj->nij', dN_nat, xy)          # (ne, 2, 2)
        detJ = torch.linalg.det(J)
        if bool((detJ <= 0).any().item()):
            raise ValueError("element with non-positive Jacobian")
        dN = torch.einsum('ja,nca->njc', dN_nat,
                          torch.linalg.inv(J).transpose(1, 2))
        Bb = torch.zeros((ne, 3, 12), dtype=dtype)
        Bb[:, 0, 3 * lidx + 1] = dN[:, :, 0]
        Bb[:, 1, 3 * lidx + 2] = dN[:, :, 1]
        Bb[:, 2, 3 * lidx + 1] = dN[:, :, 1]
        Bb[:, 2, 3 * lidx + 2] = dN[:, :, 0]
        Bb_all.append(Bb)
        detJ_all.append(detJ)

    ke = torch.zeros((ne, 12, 12), dtype=dtype)
    for Bb, detJ in zip(Bb_all, detJ_all):
        ke += torch.einsum('nka,kj,njb,n->nab', Bb, Db, Bb, detJ)

    N0, dN_nat0 = _shape(xy.new_tensor(0.0), xy.new_tensor(0.0))
    J0 = torch.einsum('ai,naj->nij', dN_nat0, xy)
    detJ0 = torch.linalg.det(J0)
    dN0 = torch.einsum('ja,nca->njc', dN_nat0,
                       torch.linalg.inv(J0).transpose(1, 2))
    Bs = torch.zeros((ne, 2, 12), dtype=dtype)
    Bs[:, 0, 3 * lidx] = dN0[:, :, 0]
    Bs[:, 0, 3 * lidx + 1] = N0
    Bs[:, 1, 3 * lidx] = dN0[:, :, 1]
    Bs[:, 1, 3 * lidx + 2] = N0
    ke += torch.einsum('nka,kj,njb,n->nab', Bs, Ds, Bs, detJ0 * 4.0)

    return ke, torch.stack(Bb_all, dim=1), torch.stack(detJ_all, dim=1)



def _eval_pressure(
    pressure,
    xq: torch.Tensor,
    yq: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Evaluate the transverse pressure at a batch of quadrature points.

    A callable is first offered the (vectorised) coordinate tensors -- pressure
    patches are then evaluated element-wise in one shot, which keeps fine-mesh
    patch-load runs cheap.  Scalar-only callables raise on tensor input, so we
    fall back to evaluating them point by point.
    """
    if not callable(pressure):
        return xq.new_tensor(float(pressure)).expand(xq.shape[0])
    try:
        q = torch.as_tensor(pressure(xq, yq), dtype=dtype)
        return q.expand(xq.shape[0])
    except Exception:
        vals = [float(pressure(float(a), float(b))) for a, b in zip(xq, yq)]
        return torch.tensor(vals, dtype=dtype)


def rect_patch_pressure(
    intensity: float,
    x1: float,
    x2: float,
    y1: float,
    y2: float,
):
    """Vectorised uniform pressure over the rectangle ``[x1,x2] x [y1,y2]``.

    Returns a callable ``p(x, y)`` that accepts both scalars and 1-D tensors,
    so a pressure patch can be applied without a per-element Python loop.
    """
    def p(x, y):
        if torch.is_tensor(x):
            return torch.where(
                (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2),
                x.new_tensor(float(intensity)),
                x.new_tensor(0.0),
            )
        return intensity if (x1 <= x <= x2 and y1 <= y <= y2) else 0.0
    return p


@dataclass(frozen=True)
class RestrainedPanelResult:
    """Solution of a rectangular panel with per-edge elastic restraint."""
    displacement: torch.Tensor
    nodes: torch.Tensor
    elements: torch.Tensor
    young: float
    poisson: float
    thickness: float
    winkler_k: float
    beta_x: float
    beta_y: float
    K_rot_x: float
    K_rot_y: float
    side_x: float
    side_y: float
    nx: int
    ny: int

    @property
    def ndof(self) -> int:
        return 3 * int(self.nodes.shape[0])

    def w(self) -> torch.Tensor:
        """Transverse displacement at every node."""
        return self.displacement[0::3]


def solve_restrained_panel(
    nx: int,
    ny: int,
    side_x: float,
    side_y: float,
    young: float,
    poisson: float,
    thickness: float,
    pressure: float | callable,
    beta_x: float = 0.0,
    beta_y: Optional[float] = None,
    winkler_k: float = 0.0,
    eps: float = 1e-6,
    dtype: torch.dtype = torch.float64,
    *,
    boundary: str = "ss2",
) -> RestrainedPanelResult:
    """Solve a rectangular Mindlin panel with elastic rotational restraint.

    The panel lies on ``[0, side_x] x [0, side_y]``.  Edges ``x = 0, side_x``
    carry a rotational spring of stiffness ``K_rot_x = zeta(beta_x) D / side_x``
    acting on the *normal* rotation ``theta_x``; edges ``y = 0, side_y`` carry
    ``K_rot_y = zeta(beta_y) D / side_y`` acting on ``theta_y``.  Because the
    two spans are restrained by different structural members in a hull (the
    longitudinals restrain one pair of edges, the transverse frames the other),
    ``beta_x`` and ``beta_y`` are independent.  Transverse displacement and the
    tangential rotation stay hard-fixed (the SS2 convention of this module), so
    ``beta_x = beta_y = 0`` is exactly the SSSS plate and
    ``beta_x = beta_y = 1`` is clamped.

    Parameters
    ----------
    boundary : {"ss2", "clamped"}
        ``"ss2"`` uses the rotational springs; ``"clamped"`` fixes both
        rotations and ignores the springs (kept as an independent check).

    Raises
    ------
    RuntimeError
        If scipy is unavailable (sparse direct solve is required here).

    Returns
    -------
    RestrainedPanelResult
    """
    for name, value in (("beta_x", beta_x), ("beta_y", beta_y)):
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if boundary not in ("ss2", "clamped"):
        raise ValueError("boundary must be 'ss2' or 'clamped'")
    if beta_y is None:
        beta_y = beta_x
    if side_x <= 0 or side_y <= 0 or nx < 1 or ny < 1:
        raise ValueError("need positive spans and mesh sizes")

    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
    except ImportError as exc:  # pragma: no cover - scipy is a dev dependency
        raise RuntimeError(
            "solve_restrained_panel needs scipy for the sparse direct solve"
        ) from exc

    nodes, elements = structured_rect_mesh(nx, ny, side_x, side_y, dtype=dtype)
    n_nodes = int(nodes.shape[0])
    ndof = 3 * n_nodes
    ne = int(elements.shape[0])

    ke, _, _ = _batched_mindlin_stiffness(nodes, elements, young, poisson, thickness)

    # Winkler foundation: consistent diagonal/consistent matrix per element.
    if winkler_k:
        from .winkler_foundation import q4_winkler_foundation_stiffness
        xy = nodes[elements]
        kw = torch.stack([
            q4_winkler_foundation_stiffness(xy[e], winkler_k) for e in range(ne)
        ])
        ke = ke + kw

    dofs = torch.stack((3 * elements, 3 * elements + 1, 3 * elements + 2), dim=2)
    dofs = dofs.reshape(ne, 12)
    rows = torch.repeat_interleave(dofs, 12, dim=1).reshape(-1)
    cols = dofs.repeat(1, 12).reshape(-1)
    vals = ke.reshape(-1)
    K = sp.coo_matrix(
        (vals.detach().numpy(), (rows.numpy(), cols.numpy())),
        shape=(ndof, ndof),
    ).tocsr()

    # Consistent load vector on the w DOFs (2x2 Gauss).
    f = torch.zeros(ndof, dtype=dtype)
    gauss = 1.0 / 3.0**0.5
    fe = torch.zeros((ne, 12), dtype=dtype)
    lidx = torch.arange(4)
    for xi_v, eta_v in ((-gauss, -gauss), (gauss, -gauss),
                        (gauss, gauss), (-gauss, gauss)):
        xi, eta = nodes.new_tensor(xi_v), nodes.new_tensor(eta_v)
        N, dN_nat = _shape(xi, eta)
        xy = nodes[elements]
        J = torch.einsum('ai,naj->nij', dN_nat, xy)
        detJ = torch.linalg.det(J)
        xq = torch.einsum('i,ni->n', N, xy[:, :, 0])
        yq = torch.einsum('i,ni->n', N, xy[:, :, 1])
        q = _eval_pressure(pressure, xq, yq, dtype)
        fe[:, 3 * lidx] += (N[None, :] * (q * detJ)[:, None])
    f.index_add_(0, (3 * elements).reshape(-1), fe[:, 3 * lidx].reshape(-1))

    edges = _rect_edges(nodes, side_x, side_y)
    D = young * thickness**3 / (12.0 * (1.0 - poisson**2))
    K_rot_x = (0.0 if boundary == "clamped" else
               beta_to_zeta(beta_x, eps) * D / side_x)
    K_rot_y = (0.0 if boundary == "clamped" else
               beta_to_zeta(beta_y, eps) * D / side_y)

    # SS2 hard constraints: w everywhere on the edge, tangential rotation.
    edge_mask = torch.zeros(n_nodes, dtype=torch.bool)
    for key in edges:
        edge_mask[edges[key]] = True
    xmask = torch.zeros(n_nodes, dtype=torch.bool)
    xmask[edges['x0']] = True
    xmask[edges['xa']] = True
    ymask = torch.zeros(n_nodes, dtype=torch.bool)
    ymask[edges['y0']] = True
    ymask[edges['yb']] = True

    fixed = [3 * torch.nonzero(edge_mask, as_tuple=False).flatten()]
    if boundary == "clamped":
        fixed.append(3 * torch.nonzero(edge_mask, as_tuple=False).flatten() + 1)
        fixed.append(3 * torch.nonzero(edge_mask, as_tuple=False).flatten() + 2)
        spring = torch.zeros(ndof, dtype=dtype)
    else:
        fixed.append(3 * torch.nonzero(xmask, as_tuple=False).flatten() + 2)
        fixed.append(3 * torch.nonzero(ymask, as_tuple=False).flatten() + 1)
        spring = torch.zeros(ndof, dtype=dtype)
        if K_rot_x:
            spring[3 * edges['x0'] + 1] += K_rot_x
            spring[3 * edges['xa'] + 1] += K_rot_x
        if K_rot_y:
            spring[3 * edges['y0'] + 2] += K_rot_y
            spring[3 * edges['yb'] + 2] += K_rot_y
    fixed_dofs = torch.unique(torch.cat(fixed))
    free = torch.ones(ndof, dtype=torch.bool)
    free[fixed_dofs] = False
    free_idx = torch.nonzero(free, as_tuple=False).flatten().numpy()

    Kff = K[free_idx][:, free_idx].tocsc()
    if bool(spring.any().item()):
        Kff = Kff + sp.diags(spring.numpy()[free_idx])
    u = torch.zeros(ndof, dtype=dtype)
    u[torch.as_tensor(free_idx)] = torch.as_tensor(
        spla.spsolve(Kff, f.numpy()[free_idx]), dtype=dtype
    )

    return RestrainedPanelResult(
        displacement=u,
        nodes=nodes,
        elements=elements,
        young=young,
        poisson=poisson,
        thickness=thickness,
        winkler_k=winkler_k,
        beta_x=beta_x,
        beta_y=beta_y,
        K_rot_x=K_rot_x,
        K_rot_y=K_rot_y,
        side_x=side_x,
        side_y=side_y,
        nx=nx,
        ny=ny,
    )


# Gauss-point -> corner-node extrapolation for the bilinear Q4 element:
# nodes sit at +/-1, Gauss points at +/-1/sqrt(3), so each corner value is
# 1.36603 * (nearest Gauss) - 0.36603 * (far Gauss).  Row l corresponds to
# local node l (natural (-1,-1), (1,-1), (1,1), (-1,1)); column g to Gauss
# point g in the same ordering.
_EXTRAP_Q4 = torch.tensor([
    [1.3660254037844386, -0.3660254037844386],
    [-0.3660254037844386, 1.3660254037844386],
], dtype=torch.float64)


def _extrapolation_operator(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    W = _EXTRAP_Q4.to(dtype)
    X = torch.zeros((4, 4), dtype=dtype)
    for corner, (ix, iy) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
        for gx in (0, 1):
            for gy in (0, 1):
                X[corner, 2 * gy + gx] = W[iy, gy] * W[ix, gx]
    return X


def moments_at_nodes(result: RestrainedPanelResult) -> torch.Tensor:
    """Bending moments ``[Mx, My, Mxy]`` per node [N] (or N*m/m).

    Computed at the four Gauss points of every element, extrapolated to the
    element corners and averaged over the elements sharing a node (the standard
    ``B * D * B * u`` recovery).  Averages are unweighted, which is exact for a
    uniform mesh and slightly biased on graded meshes -- acceptable here because
    only uniform structured meshes are produced by this module.
    """
    nodes, elements = result.nodes, result.elements
    dtype = nodes.dtype
    ne = int(elements.shape[0])
    ke, Bb, _ = _batched_mindlin_stiffness(
        nodes, elements, result.young, result.poisson, result.thickness)
    del ke
    Db, _ = _rigidities(result.young, result.poisson, result.thickness, dtype)

    dofs = torch.stack((3 * elements, 3 * elements + 1, 3 * elements + 2), dim=2)
    ue = result.displacement[dofs.reshape(ne, 12)]           # (ne, 12)
    kappa = torch.einsum('ngki,ni->ngk', Bb, ue)             # (ne, 4, 3)
    M_gauss = torch.einsum('kj,ngj->ngk', Db, kappa)         # (ne, 4, 3)

    X = _extrapolation_operator(dtype)
    M_nodes_el = torch.einsum('lg,ngk->nlk', X, M_gauss)     # (ne, 4, 3)

    n_nodes = int(nodes.shape[0])
    acc = torch.zeros((n_nodes, 3), dtype=dtype)
    cnt = torch.zeros(n_nodes, dtype=dtype)
    acc.index_add_(0, elements.reshape(-1), M_nodes_el.reshape(-1, 3))
    cnt.index_add_(0, elements.reshape(-1), torch.ones(ne * 4, dtype=dtype))
    return acc / cnt[:, None]


def bending_stress_at_nodes(result: RestrainedPanelResult) -> torch.Tensor:
    """Extreme-fibre bending stress ``[sigma_x, sigma_y, tau_xy]`` [Pa] at z=+t/2.

    ``sigma = 6 M / t^2`` (Kirchhoff/Mindlin linear stress distribution); the
    sign follows the moment sign convention ``M = Db * kappa`` of
    :func:`tensorfem.plate.q4_mindlin_stiffness`.
    """
    M = moments_at_nodes(result)
    return 6.0 * M / result.thickness**2


def von_mises_bending_at_nodes(result: RestrainedPanelResult) -> torch.Tensor:
    """Von Mises bending stress [Pa] built from the plate moments.

    ``sigma_vm = 6/t^2 * sqrt(Mx^2 - Mx My + My^2 + 3 Mxy^2)``, the same
    definition the JOUR-ICE-STRUCT panel benchmark uses.
    """
    M = moments_at_nodes(result)
    Mx, My, Mxy = M[:, 0], M[:, 1], M[:, 2]
    return 6.0 / result.thickness**2 * torch.sqrt(
        Mx**2 - Mx * My + My**2 + 3.0 * Mxy**2)


def peak_bending_stress(result: RestrainedPanelResult) -> dict:
    """Centre value and global maximum of the von Mises bending stress.

    The paper's ``sigma_plate`` is the *centre* of the panel, which is not the
    same as the global maximum when the load is a pressure patch: the patch
    edge carries a moment peak that is mesh-sensitive (the pressure jump is a
    load discontinuity).  Both numbers are reported so callers cannot confuse
    them.
    """
    nodes = result.nodes
    vm = von_mises_bending_at_nodes(result)
    M = moments_at_nodes(result)
    dist = (nodes[:, 0] - result.side_x / 2) ** 2 + \
           (nodes[:, 1] - result.side_y / 2) ** 2
    c = int(torch.argmin(dist))
    g = int(torch.argmax(vm))
    return {
        'centre_index': c,
        'centre_von_mises': float(vm[c]),
        'centre_Mx': float(M[c, 0]),
        'centre_My': float(M[c, 1]),
        'centre_sigma_x': float(6.0 * M[c, 0] / result.thickness**2),
        'max_von_mises': float(vm[g]),
        'max_index': g,
        'max_x': float(nodes[g, 0]),
        'max_y': float(nodes[g, 1]),
        'w_centre': float(result.displacement[3 * c]),
    }
