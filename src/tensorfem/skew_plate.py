"""Skew (parallelogram) plate benchmarks with the Q4 Mindlin element.

Why this module exists
----------------------
Every plate benchmark already in the tree sits on a *rectangular* footprint:
``plate.structured_square_mesh`` and ``plate_bfs.assemble_bfs_plate`` only
build axis-aligned meshes, so no in-tree element has ever been exercised on a
non-orthogonal planform.  Skew plates are the standard probe for that gap
because their obtuse corners carry a singular bending moment -- an element
that quietly relies on orthogonal geometry (or on rectangular special-casing
of the boundary conditions) will not converge to the published series
solution.  Morley's rhombic plate (acute angle 30 deg, uniformly loaded,
simply supported) is the canonical case; see
``benchmarks/verified/fem_morley_skew_plate/run.py`` for the graded run and
``docs/morley-skew-plate-validation.md`` for the evidence table.

What is provided
----------------
* :func:`rhombus_mesh` -- structured ``n x n`` quad mesh of a rhombus, built
  as the affine image of the unit square::

      x = L (xi + cos(alpha) eta),   y = L sin(alpha) eta,   xi, eta in [0,1]

  so the side length is ``L`` and the acute angle is ``alpha``.
* :func:`assemble_skew_plate` -- sparse assembly of the Q4 Mindlin bending
  stiffness (reused from :func:`tensorfem.plate.q4_mindlin_stiffness`, whose
  ``sri``/``mitc``/``full`` shear treatments are exposed unchanged) plus the
  consistent uniform-pressure load.
* :func:`plate_dofs` / :func:`solve_skew_plate` -- boundary conditions and the
  sparse direct solve.  Both simply-supported variants are supported because
  the two differ on a skew planform and the difference is *measurable*:

  ``soft``   ``w = 0`` only.  This is the Mindlin transcription of the
             Kirchhoff condition (``w = 0`` essential, ``M_n = 0`` natural)
             and is the one the published skew-plate reference values use.
  ``hard``   ``w = 0`` and the *tangential* rotation ``theta_t = 0``.  The
             over-constrained variant, kept as a control: it kills the corner
             singularity and must therefore miss the published number.
* :func:`centre_moments` -- bending moments at the plate centre, averaged over
  the elements incident on the centre node (the centre of the bending field is
  smooth; the singularity sits at the obtuse corners).
* :func:`ritz_sine_solution` -- an *independent* Rayleigh-Ritz oracle on the
  same parallelogram: trial space ``sin(m pi xi) sin(n pi eta)``, which is
  complete on the parallelogram and vanishes on all four edges.  Its
  stiffness/load integrals are evaluated in closed form (separability of the
  sine basis, see :func:`_one_dimensional_tables`), so the oracle has no
  quadrature error at all; its only error is the truncation of the series,
  which is *measured* by the caller.  The oracle shares no code with the FE
  path -- that independence is what makes it usable as a check on both the
  element and the published value.

Conventions
-----------
``w`` positive down, uniform load ``q`` positive down, ``D = E t^3 /
(12 (1 - nu^2))``, moments ``M = Db kappa`` with
``kappa = (theta_x,x, theta_y,y, theta_x,y + theta_y,x)`` (same ``Db`` and
``Bb`` convention as :mod:`tensorfem.plate`), so the third moment component is
the tensor twisting moment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .plate import _shape, q4_mindlin_stiffness

__all__ = [
    "SkewPlateMesh",
    "rhombus_mesh",
    "plate_rigidity",
    "assemble_skew_plate",
    "solve_skew_plate",
    "centre_moments",
    "principal_moments",
    "ritz_sine_solution",
]


# ---------------------------------------------------------------------------
# mesh
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SkewPlateMesh:
    """Structured rhombus mesh.

    ``nodes`` is ``((n+1)^2, 2)`` in physical coordinates, ``elements`` is
    ``(n^2, 4)`` counter-clockwise connectivity, and ``boundary_nodes`` holds
    the four edges (in the order bottom, right, top, left).
    """

    nodes: np.ndarray
    elements: np.ndarray
    n: int
    side: float
    skew_deg: float

    @property
    def acute_angle_rad(self) -> float:
        return math.radians(self.skew_deg)

    @property
    def centre(self) -> tuple[float, float]:
        return self._centre

    @property
    def centre_node(self) -> int:
        return self._centre_node

    def boundary_edge(self, name: str) -> np.ndarray:
        return self._edges[name]

    # filled by rhombus_mesh via object.__setattr__ (frozen dataclass)
    _centre: tuple[float, float] = (0.0, 0.0)
    _centre_node: int = -1
    _edges: dict = None  # type: ignore[assignment]


def rhombus_mesh(
    n: int, side: float = 1.0, skew_deg: float = 30.0,
) -> SkewPlateMesh:
    """Structured ``n x n`` quad mesh of a rhombus with side ``side``.

    The acute angle is ``skew_deg``; nodes are ordered row-major with ``eta``
    the outer (axis ``j``) index and ``xi`` the inner (axis ``i``) index, so
    node ``(i, j)`` has index ``j*(n+1) + i``.
    """
    if n < 1:
        raise ValueError("n must be positive")
    if not (0.0 < skew_deg <= 90.0):
        raise ValueError("skew_deg must lie in (0, 90]; 90 deg is the square/rectangle control")
    alpha = math.radians(skew_deg)
    xi = np.linspace(0.0, 1.0, n + 1)
    eta = np.linspace(0.0, 1.0, n + 1)
    XI, ETA = np.meshgrid(xi, eta, indexing="xy")
    x = side * (XI + math.cos(alpha) * ETA)
    y = side * (math.sin(alpha) * ETA)
    nodes = np.stack((x.reshape(-1), y.reshape(-1)), axis=1)

    elements = []
    for j in range(n):
        for i in range(n):
            n0 = j * (n + 1) + i
            elements.append((n0, n0 + 1, n0 + n + 2, n0 + n + 1))
    elements = np.asarray(elements, dtype=np.int64)

    top = np.arange(n + 1)                 # eta = 0  (j = 0)
    bottom = top + n * (n + 1)             # eta = 1  (j = n)
    left = np.arange(n + 1) * (n + 1)      # xi = 0   (i = 0)
    right = left + n                       # xi = 1   (i = n)
    edges = {
        "xi0": left,           # edge along the eta direction at xi = 0
        "xi1": right,
        "eta0": top,           # edge parallel to x at eta = 0
        "eta1": bottom,
    }

    if n % 2 == 0:
        centre_index = (n // 2) * (n + 1) + n // 2
    else:  # pragma: no cover - callers use even n
        centre_index = -1

    mesh = SkewPlateMesh(
        nodes=nodes, elements=elements, n=n, side=side, skew_deg=skew_deg,
    )
    object.__setattr__(mesh, "_centre", (float(side * (0.5 + 0.5 * math.cos(alpha))),
                                         float(0.5 * side * math.sin(alpha))))
    object.__setattr__(mesh, "_centre_node", int(centre_index))
    object.__setattr__(mesh, "_edges", edges)
    return mesh


# ---------------------------------------------------------------------------
# material
# ---------------------------------------------------------------------------
def plate_rigidity(young: float, poisson: float, thickness: float) -> float:
    """``D = E t^3 / (12 (1 - nu^2))``."""
    return young * thickness ** 3 / (12.0 * (1.0 - poisson ** 2))


def _bending_matrix(young: float, poisson: float, thickness: float) -> np.ndarray:
    d = plate_rigidity(young, poisson, thickness)
    return d * np.array(
        [[1.0, poisson, 0.0], [poisson, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - poisson)]]
    )


# ---------------------------------------------------------------------------
# degrees of freedom
# ---------------------------------------------------------------------------
def plate_dofs(mesh: SkewPlateMesh, support: str = "soft") -> tuple[np.ndarray, list]:
    """Return ``(fixed_dofs, constraints)`` for a simply-supported rhombus.

    ``support="soft"``: ``w = 0`` on all four edges, rotations free.
    ``support="hard"``: additionally ``theta_t = 0`` (rotation about the edge
    tangent) on all four edges.  The rhombus has only two edge directions, so
    the tangential constraint is a two-term linear relation:
    edges ``eta0``/``eta1`` are parallel to ``x`` (``theta_x = 0``) and edges
    ``xi0``/``xi1`` are parallel to ``(cos a, sin a)``
    (``theta_x cos a + theta_y sin a = 0``).

    ``constraints`` is a list of ``(dependent_dof, [(master_dof, factor), ...])``
    used by :func:`solve_skew_plate` to eliminate the dependent dofs exactly.
    """
    if support not in ("soft", "hard"):
        raise ValueError("support must be 'soft' or 'hard'")
    alpha = mesh.acute_angle_rad
    boundary = np.unique(np.concatenate(
        [mesh.boundary_edge(k) for k in ("xi0", "xi1", "eta0", "eta1")]
    ))
    fixed = [3 * int(node) for node in boundary]
    constraints: list[tuple[int, list[tuple[int, float]]]] = []
    if support == "hard":
        for key, along_x in (("eta0", True), ("eta1", True),
                             ("xi0", False), ("xi1", False)):
            for node in mesh.boundary_edge(key):
                node = int(node)
                if along_x:
                    # theta_x = 0 : eliminate the theta_x dof.
                    constraints.append((3 * node + 1, []))
                else:
                    # theta_x cos a + theta_y sin a = 0 -> theta_y = -cot(a) theta_x
                    constraints.append(
                        (3 * node + 2, [(3 * node + 1, -math.cos(alpha) / math.sin(alpha))])
                    )
    return np.unique(np.asarray(fixed, dtype=np.int64)), constraints


# ---------------------------------------------------------------------------
# assembly / solve
# ---------------------------------------------------------------------------
def assemble_skew_plate(
    mesh: SkewPlateMesh,
    young: float,
    poisson: float,
    thickness: float,
    *,
    pressure: float = 1.0,
    shear_correction: float = 5.0 / 6.0,
    shear_treatment: str = "mitc",
):
    """Sparse ``(K, f)`` of the rhombus, Q4 Mindlin with a uniform pressure."""
    from scipy.sparse import coo_matrix

    n_nodes = mesh.nodes.shape[0]
    ndof = 3 * n_nodes
    rows, cols, vals = [], [], []
    f = np.zeros(ndof)
    gp = 1.0 / math.sqrt(3.0)
    for conn in mesh.elements:
        xy = torch.as_tensor(mesh.nodes[conn], dtype=torch.float64)
        ke = q4_mindlin_stiffness(
            xy, young, poisson, thickness,
            shear_correction=shear_correction, shear_treatment=shear_treatment,
        ).numpy()
        dofs = np.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), axis=1).reshape(-1)
        rows.append(np.repeat(dofs, len(dofs)))
        cols.append(np.tile(dofs, len(dofs)))
        vals.append(ke.reshape(-1))
        # consistent uniform load (2x2 Gauss is exact for bilinear N, const detJ)
        fe = np.zeros(12)
        for xif in (-gp, gp):
            for etaf in (-gp, gp):
                xi = torch.tensor(xif, dtype=torch.float64)
                eta = torch.tensor(etaf, dtype=torch.float64)
                N, dN_nat = _shape(xi, eta)
                J = dN_nat.T @ xy
                detJ = float(torch.linalg.det(J))
                fe[0::3] += float(pressure) * N.numpy() * detJ
        f[dofs] += fe
    K = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(ndof, ndof),
    ).tocsr()
    return K, f


def _constraint_operator(
    ndof: int, fixed: np.ndarray, constraints: list, *, sparse=True,
):
    """Map ``u = P u_hat`` with the constrained dofs removed from ``u_hat``."""
    from scipy.sparse import coo_matrix

    dependent = np.array([c[0] for c in constraints], dtype=np.int64)
    all_fixed = np.unique(np.concatenate([fixed, dependent])) if len(dependent) else fixed
    free = np.setdiff1d(np.arange(ndof), all_fixed)
    column = {int(dof): k for k, dof in enumerate(free)}
    rows, cols, vals = [], [], []
    for k, dof in enumerate(free):
        rows.append(int(dof))
        cols.append(k)
        vals.append(1.0)
    for dep, masters in constraints:
        for master, factor in masters:
            rows.append(int(dep))
            cols.append(column[int(master)])
            vals.append(float(factor))
    P = coo_matrix((vals, (rows, cols)), shape=(ndof, len(free))).tocsr()
    return P, free


def solve_skew_plate(
    mesh: SkewPlateMesh,
    young: float,
    poisson: float,
    thickness: float,
    *,
    pressure: float = 1.0,
    support: str = "soft",
    shear_correction: float = 5.0 / 6.0,
    shear_treatment: str = "mitc",
) -> dict:
    """Assemble, apply the simply-supported conditions and solve.

    Returns the full displacement vector plus bookkeeping (dof counts, the
    sparse operator used) so a caller can post-process fields.
    """
    from scipy.sparse.linalg import spsolve

    K, f = assemble_skew_plate(
        mesh, young, poisson, thickness, pressure=pressure,
        shear_correction=shear_correction, shear_treatment=shear_treatment,
    )
    fixed, constraints = plate_dofs(mesh, support)
    P, free = _constraint_operator(K.shape[0], fixed, constraints)
    K_hat = (P.T @ K @ P).tocsc()
    f_hat = P.T @ f
    u_hat = spsolve(K_hat, f_hat)
    u = np.asarray(P @ u_hat, dtype=float)
    return {
        "u": u,
        "free_dofs": free,
        "n_dof": int(K.shape[0]),
        "n_free_dof": int(len(free)),
        "K_nnz": int(K.nnz),
        "support": support,
        "shear_treatment": shear_treatment,
    }


# ---------------------------------------------------------------------------
# post-processing
# ---------------------------------------------------------------------------
def _element_moments(
    xy: np.ndarray, dofs: np.ndarray, u: np.ndarray,
    young: float, poisson: float, thickness: float,
    at: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Bending/twisting moments ``(M_x, M_y, M_xy)`` at natural point ``at``."""
    Db = _bending_matrix(young, poisson, thickness)
    xy_t = torch.as_tensor(xy, dtype=torch.float64)
    xi = torch.tensor(at[0], dtype=torch.float64)
    eta = torch.tensor(at[1], dtype=torch.float64)
    _, dN_nat = _shape(xi, eta)
    J = dN_nat.T @ xy_t
    dN = (dN_nat @ torch.linalg.inv(J).T).numpy()
    Bb = np.zeros((3, 12))
    for i in range(4):
        Bb[0, 3 * i + 1] = dN[i, 0]
        Bb[1, 3 * i + 2] = dN[i, 1]
        Bb[2, 3 * i + 1] = dN[i, 1]
        Bb[2, 3 * i + 2] = dN[i, 0]
    kappa = Bb @ u[dofs]
    return Db @ kappa


def centre_moments(
    mesh: SkewPlateMesh,
    u: np.ndarray,
    young: float,
    poisson: float,
    thickness: float,
) -> dict:
    """Moments at the centre node, averaged over the incident elements.

    Each incident element contributes its moment at the element centroid
    (``xi = eta = 0``); the spread across the incident elements is returned as
    a discretisation diagnostic.  The centre of the rhombus is a node whenever
    ``n`` is even (the meshes used here), otherwise the element centroid
    closest to the geometric centre is used.
    """
    centre = np.asarray(mesh.centre)
    if mesh.centre_node >= 0:
        incident = [e for e in mesh.elements if mesh.centre_node in e]
        reference = mesh.nodes[mesh.centre_node]
    else:  # pragma: no cover
        centroids = mesh.nodes[mesh.elements].mean(axis=1)
        k = int(np.argmin(np.linalg.norm(centroids - centre, axis=1)))
        incident = [mesh.elements[k]]
        reference = centroids[k]
    values = []
    for conn in incident:
        dofs = np.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), axis=1).reshape(-1)
        values.append(_element_moments(
            mesh.nodes[conn], dofs, u, young, poisson, thickness,
        ))
    values = np.asarray(values)
    mean = values.mean(axis=0)
    return {
        "M_x": float(mean[0]),
        "M_y": float(mean[1]),
        "M_xy": float(mean[2]),
        "spread": float(values.max(axis=0).max() - values.min(axis=0).min()),
        "n_incident": int(len(values)),
        "distance_to_centre": float(np.linalg.norm(reference - centre)),
        "M_at_node": mesh.nodes[mesh.centre_node].tolist() if mesh.centre_node >= 0 else None,
    }


def principal_moments(M_x: float, M_y: float, M_xy: float) -> tuple[float, float]:
    """Principal moments ``(M1, M2)`` with ``M1 >= M2``."""
    mean = 0.5 * (M_x + M_y)
    radius = math.hypot(0.5 * (M_x - M_y), M_xy)
    return mean + radius, mean - radius


# ---------------------------------------------------------------------------
# independent Rayleigh-Ritz oracle (thin plate, sine basis, closed-form integrals)
# ---------------------------------------------------------------------------
def _one_dimensional_tables(n_terms: int) -> tuple[np.ndarray, np.ndarray]:
    """``S[a,b] = int_0^1 sin(a pi x) sin(b pi x) dx`` and ``X[a,b] = ...cos...``.

    ``S`` is ``delta_ab / 2``; the mixed table
    ``X[a,b] = int_0^1 sin(a pi x) cos(b pi x) dx`` equals
    ``1/((a+b)pi) + 1/((a-b)pi)`` for ``a+b`` odd and zero for ``a+b`` even.
    """
    a = np.arange(1, n_terms + 1)
    S = np.where(a[:, None] == a[None, :], 0.5, 0.0)
    denom = (a[:, None] + a[None, :]) * (a[:, None] - a[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        X = np.where(((a[:, None] + a[None, :]) % 2) == 1, 2.0 * a[:, None] / (math.pi * denom), 0.0)
    return S, X


def ritz_sine_solution(
    side: float,
    skew_deg: float,
    n_terms: int,
    young: float,
    poisson: float,
    thickness: float,
    pressure: float = 1.0,
) -> dict:
    """Kirchhoff Rayleigh-Ritz solution of the uniformly loaded rhombus.

    Trial space ``w = sum c_mn sin(m pi xi) sin(n pi eta)`` with
    ``m, n = 1 .. n_terms``, which is complete on the parallelogram and
    satisfies ``w = 0`` on all four edges.  The strain-energy and load
    integrals are evaluated in closed form in the separable basis, so the only
    error is the truncation ``n_terms``; the natural boundary condition
    ``M_n = 0`` is the one the discretised energy minimises, i.e. the same
    simply-supported model as the "soft" FE variant.

    Returns the centre deflection, the centre moments and the principal
    moments, together with the number of unknowns actually solved.
    """
    if n_terms < 1:
        raise ValueError("n_terms must be positive")
    alpha = math.radians(skew_deg)
    sin_a, cos_a = math.sin(alpha), math.cos(alpha)
    L = float(side)
    d_flex = plate_rigidity(young, poisson, thickness)

    # physical-to-natural derivative map (xi_x, xi_y, eta_x, eta_y)
    p1 = 1.0 / L                 # xi,x
    p2 = -cos_a / (L * sin_a)    # xi,y
    p3 = 1.0 / (L * sin_a)       # eta,y
    # second-derivative map: (w_xx, w_yy, w_xy) = G (w_xixi, w_xieta, w_etaeta)
    G = np.array([[p1 ** 2, 0.0, 0.0],
                  [p2 ** 2, 2.0 * p2 * p3, p3 ** 2],
                  [p1 * p2, p1 * p3, 0.0]])
    C = np.array([[1.0, poisson, 0.0],
                  [poisson, 1.0, 0.0],
                  [0.0, 0.0, 2.0 * (1.0 - poisson)]]) * d_flex
    H = G.T @ C @ G              # energy weight of the natural 2nd-derivative Gram

    S, X = _one_dimensional_tables(n_terms)
    ak = np.arange(1, n_terms + 1)          # mode numbers m, n
    a = ak * math.pi                        # circular wavenumbers
    n_basis = n_terms * n_terms
    idx_m, idx_n = np.meshgrid(np.arange(n_terms), np.arange(n_terms), indexing="ij")
    idx_m = idx_m.reshape(-1)
    idx_n = idx_n.reshape(-1)

    area = L ** 2 * sin_a

    # Natural second-derivative Gram for basis p = (m, n) and q = (r, s):
    #   n1 = -a_m^2 sin_m sin_n,  n2 = a_m a_n cos_m cos_n,
    #   n3 = -a_n^2 sin_m sin_n
    # Products separate into the 1-D tables S (sin-sin), C (cos-cos) and
    # X[a,b] = int sin_a cos_b; Q[a,b] = int cos_a sin_b = X[b,a].
    def ix1(table):
        return table[np.ix_(idx_m, idx_m)]

    def ix2(table):
        return table[np.ix_(idx_n, idx_n)]

    SS, SS2 = ix1(S), ix2(S)
    CC, CC2 = SS, SS2          # cos-cos has the same delta_ab/2 table as sin-sin
    PX, PX2 = ix1(X), ix2(X)                       # int sin_m cos_r
    QX, QX2 = ix1(X.T), ix2(X.T)                   # int cos_m sin_r

    am = a[idx_m][:, None]
    an = a[idx_n][:, None]
    ar = a[idx_m][None, :]
    ass = a[idx_n][None, :]

    N11 = am ** 2 * ar ** 2 * SS * SS2
    N12 = -am ** 2 * ar * ass * PX * PX2
    N13 = am ** 2 * ass ** 2 * SS * SS2
    N22 = am * an * ar * ass * CC * CC2
    N23 = -am * an * ass ** 2 * QX * QX2
    N33 = an ** 2 * ass ** 2 * SS * SS2
    K = area * (H[0, 0] * N11 + 2.0 * H[0, 1] * N12 + 2.0 * H[0, 2] * N13
                + H[1, 1] * N22 + 2.0 * H[1, 2] * N23 + H[2, 2] * N33)

    # consistent load of a uniform pressure: int sin(m pi xi) d xi
    s_m = (1.0 - np.cos(ak * np.pi)) / (ak * np.pi)
    f = pressure * area * s_m[idx_m] * s_m[idx_n]

    c = np.linalg.solve(K, f)

    sin_half_m = np.sin(ak * np.pi / 2.0)         # sin(m pi / 2)
    cos_half_m = np.cos(ak * np.pi / 2.0)
    w_centre = float(np.sum(c * sin_half_m[idx_m] * sin_half_m[idx_n]))

    # second derivatives at the centre: (w_xx, w_yy, w_xy)
    w_xixi = -float(np.sum(c * a[idx_m] ** 2 * sin_half_m[idx_m] * sin_half_m[idx_n]))
    w_etaeta = -float(np.sum(c * a[idx_n] ** 2 * sin_half_m[idx_m] * sin_half_m[idx_n]))
    w_xieta = float(np.sum(
        c * (a[idx_m] * a[idx_n]) * cos_half_m[idx_m] * cos_half_m[idx_n]
    ))
    curvatures = G @ np.array([w_xixi, w_xieta, w_etaeta])
    moments = -d_flex * np.array(
        [[1.0, poisson, 0.0],
         [poisson, 1.0, 0.0],
         [0.0, 0.0, 1.0 - poisson]]
    ) @ curvatures
    M1, M2 = principal_moments(moments[0], moments[1], moments[2])
    return {
        "n_terms": n_terms,
        "n_basis": n_basis,
        "centre_deflection": w_centre,
        "M_x": float(moments[0]),
        "M_y": float(moments[1]),
        "M_xy": float(moments[2]),
        "M1": float(M1),
        "M2": float(M2),
        "rigidity": d_flex,
        "area": area,
    }