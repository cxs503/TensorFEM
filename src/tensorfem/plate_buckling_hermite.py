"""Independent C1 conforming check for shear buckling of a simply supported plate.

Why this module exists
----------------------
``tensorfem.plate_shear_buckling`` solves the simply supported shear buckling
problem with a Rayleigh--Ritz ansatz built from products of sine modes.  That
ansatz is variationally consistent, but it shares its approximation space with
the plate-vibration and uniaxial-buckling code paths, so a systematic error in
the space (mode ordering, Jacobian, aspect handling) would not show up in any
self-consistency test of the same module.

This module re-solves the *same* eigenproblem with a discretisation that shares
no code and no approximation space: a Bogner--Fox--Schmit bicubic Hermite
rectangle (16 dof per element: ``w``, ``w_,x``, ``w_,y``, ``w_,xy``), i.e. a
conforming C1 finite element assembled from element-level Gauss quadrature.  The
comparison of the two independent families is what turns a coefficient value
into evidence.

Formulation
-----------
For a plate of flexural rigidity ``D`` under uniform in-plane shear ``N_xy`` the
buckled state satisfies

    D * lap(lap(w)) + 2 * N_xy * w_,xy = 0,

whose weak form splits into the bending operator and the shear (geometric)
operator

    K_b[i, j] = 1/2 * integral (E : eps_i : eps_j) dA   (curvatures, see below)
    G  [i, j] = 2 * integral w_,x^i * w_,y^j dA         (unit N_xy),

with the buckling condition ``K_b + N_xy G = 0``, i.e. the smallest positive
eigenvalue of ``K_b phi = lambda (-G) phi``.  Eigenvalues are reported in the
tabulated non-dimensional form

    k_s = N_xy * b^2 / (pi^2 * D),     b = panel width,

which is the same normalisation as
``plate_shear_buckling.shear_buckling_coefficient``.

The duality identity
--------------------
Rotating the panel by 90 degrees maps a uniform shear state onto a uniform shear
state, so the *exact* coefficient obeys ``k_s(a/b) * (a/b)^2 = k_s(b/a)``.  Both
the Ritz path and this FE path are checked against that identity, which needs no
literature value at all.

Nothing in this module mutates production solver state; it only assembles and
diagonalises small matrices.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import eigh

__all__ = [
    "YOUNG",
    "POISSON",
    "THICKNESS",
    "hermite_shear_coefficient",
    "hermite_shear_coefficient_ladder",
    "hermite_duality_residual",
]

YOUNG, POISSON, THICKNESS = 210.0e9, 0.3, 0.012

_GAUSS = (-math.sqrt(0.6), 0.0, math.sqrt(0.6))
_GAUSS_W = (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)


def _hermite_1d(s: float, length: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Values, first and second derivatives of the four Hermite cubics on ``[0, 1]``.

    Ordering is ``(value@0, slope@0, value@1, slope@1)``; slope dofs are scaled
    by the element length so a nodal dof holds the true derivative in ``1/m``.
    """
    value = np.array(
        [
            1.0 - 3.0 * s**2 + 2.0 * s**3,
            length * (s - 2.0 * s**2 + s**3),
            3.0 * s**2 - 2.0 * s**3,
            length * (-(s**2) + s**3),
        ]
    )
    first = np.array(
        [
            (-6.0 * s + 6.0 * s**2) / length,
            1.0 - 4.0 * s + 3.0 * s**2,
            (6.0 * s - 6.0 * s**2) / length,
            -2.0 * s + 3.0 * s**2,
        ]
    )
    second = np.array(
        [
            (-6.0 + 12.0 * s) / length**2,
            (-4.0 + 6.0 * s) / length,
            (6.0 - 12.0 * s) / length**2,
            (-2.0 + 6.0 * s) / length,
        ]
    )
    return value, first, second


def _element_matrices(
    a: float, b: float, young: float, poisson: float, thickness: float
) -> tuple[np.ndarray, np.ndarray]:
    """Bending and unit-shear geometric ``16x16`` element matrices.

    Local dof order matches ``_mesh_dofs``: ``4 * (2*ny + nx) + k`` for element
    node ``(nx, ny)`` with ``nx, ny in {0, 1}`` and ``k`` in
    ``(0: w, 1: w_,x, 2: w_,y, 3: w_,xy)``.
    """
    rigidity = young * thickness**3 / (12.0 * (1.0 - poisson**2))
    elasticity = (
        np.array([[1.0, poisson, 0.0], [poisson, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - poisson)]])
        * rigidity
    )
    kb = np.zeros((16, 16))
    kg = np.zeros((16, 16))
    for gi, s in enumerate(_GAUSS):
        xi = 0.5 * (1.0 + s)
        hx, dx, ddx = _hermite_1d(xi, a)
        for gj, t in enumerate(_GAUSS):
            eta = 0.5 * (1.0 + t)
            hy, dy, ddy = _hermite_1d(eta, b)
            weight = _GAUSS_W[gi] * _GAUSS_W[gj] * (a / 2.0) * (b / 2.0)
            values = np.zeros((16, 2))  # (w_,x, w_,y) per dof
            curvature = np.zeros((16, 3))  # (w_,xx, w_,yy, 2*w_,xy) per dof
            for ny in (0, 1):
                for nx in (0, 1):
                    base = 4 * (2 * ny + nx)
                    hi_x, dj_x = (0, 1) if nx == 0 else (2, 3)
                    hi_y, dj_y = (0, 1) if ny == 0 else (2, 3)
                    # w
                    values[base, 0] = dx[hi_x] * hy[hi_y]
                    values[base, 1] = hx[hi_x] * dy[hi_y]
                    curvature[base] = (
                        ddx[hi_x] * hy[hi_y],
                        hx[hi_x] * ddy[hi_y],
                        2.0 * dx[hi_x] * dy[hi_y],
                    )
                    # w_,x
                    values[base + 1, 0] = dx[dj_x] * hy[hi_y]
                    values[base + 1, 1] = hx[dj_x] * dy[hi_y]
                    curvature[base + 1] = (
                        ddx[dj_x] * hy[hi_y],
                        hx[dj_x] * ddy[hi_y],
                        2.0 * dx[dj_x] * dy[hi_y],
                    )
                    # w_,y
                    values[base + 2, 0] = dx[hi_x] * hy[dj_y]
                    values[base + 2, 1] = hx[hi_x] * dy[dj_y]
                    curvature[base + 2] = (
                        ddx[hi_x] * hy[dj_y],
                        hx[hi_x] * ddy[dj_y],
                        2.0 * dx[hi_x] * dy[dj_y],
                    )
                    # w_,xy
                    values[base + 3, 0] = dx[dj_x] * hy[dj_y]
                    values[base + 3, 1] = hx[dj_x] * dy[dj_y]
                    curvature[base + 3] = (
                        ddx[dj_x] * hy[dj_y],
                        hx[dj_x] * ddy[dj_y],
                        2.0 * dx[dj_x] * dy[dj_y],
                    )
            kb += weight * (curvature @ elasticity @ curvature.T)
            # weak form 2*N_xy*w_,xy -> bilinear 2*N_xy*integral(w_,x w_,y)
            kg += 2.0 * weight * np.outer(values[:, 0], values[:, 1])
    return kb, kg


_CACHE: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}


def _element(a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    key = (round(a, 12), round(b, 12))
    if key not in _CACHE:
        _CACHE[key] = _element_matrices(a, b, YOUNG, POISSON, THICKNESS)
    return _CACHE[key]


def _nodal_dof(nodes_x: int, i: int, j: int, k: int) -> int:
    return 4 * (j * nodes_x + i) + k


def _mesh_matrices(n_x: int, n_y: int, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    nodes_x = n_x + 1
    ndof = 4 * nodes_x * (n_y + 1)
    kb = np.zeros((ndof, ndof))
    kg = np.zeros((ndof, ndof))
    ke_b, ke_g = _element(a / n_x, b / n_y)
    for j in range(n_y):
        for i in range(n_x):
            dofs = np.array(
                [
                    _nodal_dof(nodes_x, ii, jj, k)
                    for jj in (j, j + 1)
                    for ii in (i, i + 1)
                    for k in range(4)
                ]
            )
            kb[np.ix_(dofs, dofs)] += ke_b
            kg[np.ix_(dofs, dofs)] += ke_g
    return kb, kg


def _simply_supported_free_dofs(n_x: int, n_y: int) -> np.ndarray:
    """Free dofs for simply supported edges: ``w = 0`` and tangential slope ``= 0``."""
    nodes_x, nodes_y = n_x + 1, n_y + 1
    fixed: set[int] = set()
    for i in range(nodes_x):  # y = 0 and y = b edges
        for j in (0, n_y):
            fixed.add(_nodal_dof(nodes_x, i, j, 0))  # w
            fixed.add(_nodal_dof(nodes_x, i, j, 1))  # w_,x (tangential)
    for j in range(nodes_y):  # x = 0 and x = a edges
        for i in (0, n_x):
            fixed.add(_nodal_dof(nodes_x, i, j, 0))
            fixed.add(_nodal_dof(nodes_x, i, j, 2))  # w_,y (tangential)
    return np.array([d for d in range(4 * nodes_x * nodes_y) if d not in fixed])


def hermite_shear_coefficient(
    *, aspect_ratio: float, n_y: int = 12, n_x: int | None = None
) -> float:
    """Critical shear coefficient ``k_s`` from the C1 Hermite FE discretisation.

    ``aspect_ratio`` is ``a/b`` with the width ``b`` fixed at 1 m; ``n_y`` counts
    elements across the width and ``n_x`` defaults to ``round(aspect*n_y)`` so the
    mesh stays uniform.
    """
    if not math.isfinite(float(aspect_ratio)) or float(aspect_ratio) <= 0.0:
        raise ValueError("aspect_ratio must be positive and finite")
    if n_y < 2:
        raise ValueError("n_y must be at least 2")
    ratio = float(aspect_ratio)
    n_x = int(round(ratio * n_y)) if n_x is None else int(n_x)
    if n_x < 4:
        # Below four elements along the span the assembled bending operator is not
        # positive definite for the simply supported constraint set, so the
        # generalised problem has no meaning; refuse instead of returning noise.
        raise ValueError(f"mesh too coarse along the span: n_x = {n_x} < 4 (increase n_y)")
    kb, kg = _mesh_matrices(n_x, n_y, ratio, 1.0)
    free = _simply_supported_free_dofs(n_x, n_y)
    kb_f = kb[np.ix_(free, free)]
    shear_f = -kg[np.ix_(free, free)]  # unit-shear geometric matrix
    # K_b is symmetric positive definite and the shear operator is indefinite, so
    # the generalised problem is posed as shear * phi = mu * K_b * phi.  It is
    # reduced through the orthogonal eigenbasis of K_b instead of calling a
    # generalised symmetric LAPACK driver: the bending operator is badly scaled on
    # thin meshes (smallest/largest eigenvalue ~1e-9), and the generalised driver
    # aborts on that pencil for some coarse meshes.
    kb_eval, kb_evec = eigh(kb_f)
    if kb_eval[0] <= 0.0:
        if -kb_eval[0] <= 1e-11 * kb_eval[-1]:
            kb_eval = np.maximum(kb_eval, kb_eval[-1] * 1e-14)  # round-off only
        else:
            raise AssertionError("bending operator is not positive definite: BC defect")
    scale = 1.0 / np.sqrt(kb_eval)
    reduced = (kb_evec * scale).T @ shear_f @ (kb_evec * scale)
    reduced = 0.5 * (reduced + reduced.T)
    del kb_f, shear_f
    mu = np.linalg.eigvalsh(reduced)
    positive = mu[mu > 0.0]
    if positive.size == 0:
        raise AssertionError("no positive shear eigenvalue: mesh or BC defect")
    multiplier = 1.0 / positive.max()  # smallest positive buckling multiplier
    rigidity = YOUNG * THICKNESS**3 / (12.0 * (1.0 - POISSON**2))
    return multiplier * 1.0**2 / (math.pi**2 * rigidity)


def hermite_shear_coefficient_ladder(
    *,
    aspect_ratio: float,
    widths: tuple[int, ...] = (6, 8, 10, 12),
) -> list[dict[str, float]]:
    """``k_s`` on a mesh sequence plus the observed order against the finest mesh."""
    rows: list[dict[str, float]] = []
    rejected: list[dict[str, object]] = []
    for n_y in widths:
        try:
            value = hermite_shear_coefficient(aspect_ratio=aspect_ratio, n_y=n_y)
        except (AssertionError, ValueError) as error:
            # Coarse meshes can leave a zero-energy boundary mode in the bending
            # operator; they are reported, never silently used.
            rejected.append({"n_y": float(n_y), "reason": str(error)})
            continue
        rows.append(
            {
                "n_y": float(n_y),
                "n_x": float(round(float(aspect_ratio) * n_y)),
                "k_s": value,
            }
        )
    if len(rows) < 2:
        raise AssertionError(f"fewer than two valid meshes for a/b = {aspect_ratio}: {rejected}")
    finest = rows[-1]["k_s"]
    for row in rows[:-1]:
        row["gap_to_finest"] = abs(row["k_s"] - finest)
    rows[-1]["gap_to_finest"] = 0.0
    for previous, current in zip(rows, rows[1:]):
        ratio = current["n_y"] / previous["n_y"]
        if previous["gap_to_finest"] > 0.0 and current["gap_to_finest"] > 0.0:
            current["observed_order"] = math.log(
                previous["gap_to_finest"] / current["gap_to_finest"]
            ) / math.log(ratio)
        else:
            current["observed_order"] = float("nan")
    return rows


def hermite_duality_residual(*, aspect_ratio: float, n_y: int = 12) -> float:
    """Relative violation of the exact identity ``k(a) * a^2 = k(1/a)``.

    The identity follows from rotating the same physical panel by 90 degrees;
    with the width normalisation ``k_s = N_xy b^2/(pi^2 D)`` the rotated panel
    reports ``k(1/a)`` computed with ``b = a``, so ``k(a) * a^2`` must reproduce
    it.  Discretisation error is the only legitimate source of residual.
    """
    ratio = float(aspect_ratio)
    direct = hermite_shear_coefficient(aspect_ratio=ratio, n_y=n_y)
    inverse = hermite_shear_coefficient(aspect_ratio=1.0 / ratio, n_y=n_y)
    return abs(direct * ratio**2 / inverse - 1.0)
