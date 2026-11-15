"""Independent 2-D potential-flow added mass by a boundary-element method.

Why this module exists
----------------------
``docs/xval_implicit_ibm_aitken_v7.md`` attributed the x3.6 over-prediction of the
implicit-IBM added mass to *geometry*: "the IBM body is a bluff body of thickness
~2 kernel supports", with the effective half-width fitted as ``c + 3.5 dx``.  That
fit was made by inflating the *chord* of a thin-plate reference, which is not the
added mass of a thick body.  To test the attribution properly one needs the
potential-flow added mass of a body of the *claimed* thickness, in unbounded
fluid, to a stated accuracy -- i.e. a reference that is independent of both the
lattice-Boltzmann solver and of the IBM coupling.

This module provides exactly that, by a classical constant-strength source-panel
(single-layer) BEM for the Neumann problem

    ``laplace(phi) = 0``,   ``dphi/dn = U . n``  on the body surface,

whose solution yields the added-mass tensor

    ``m_ij = -rho * contour_integral(phi_i n_j) dl``

(``phi_i`` = potential for unit translational velocity in direction ``i``).  The
convention is fixed by the analytic benchmarks in :func:`benchmarks`:

* circular cylinder of radius ``a`` -> ``m_xx = m_yy = rho pi a^2`` (Lamb art. 73),
* ellipse with semi-axis ``a`` along ``x`` (motion along ``x``) -> ``rho pi b^2``
  and (motion along ``y``) -> ``rho pi a^2`` (Lamb art. 71 limit ``b -> 0`` gives
  the thin plate ``rho pi c^2``),
* rectangle ``2c x 2h`` -> the tabulated "thick plate" coefficient
  ``C = m/(rho pi c^2)``, which grows from 1 (``h/c -> 0``) as the section thickens.

Discretisation
--------------
Constant panel source strength; the potential influence ``A`` uses the analytic
self term ``l/(2 pi) (ln(l/2) - 1)`` and 4-point Gauss quadrature elsewhere; the
normal-derivative operator uses the principal value ``-sigma/2`` on the diagonal.
No corner special-casing: corner panels are resolved by refinement of the panel
count on each edge, so the reported error is a genuine panel-count convergence,
not a smoothing of the geometry.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "body_panels",
    "circle_panels",
    "ellipse_panels",
    "rectangle_panels",
    "influence_matrices",
    "added_mass_matrix",
    "rectangle_added_mass_coefficient",
    "benchmarks",
]

_GAUSS_X = np.array([-0.8611363115940526, -0.3399810435848563,
                     0.3399810435848563, 0.8611363115940526])
_GAUSS_W = np.array([0.3478548451374538, 0.6521451548625461,
                     0.6521451548625461, 0.3478548451374538])


# ---------------------------------------------------------------------------
# geometries: closed curves, counter-clockwise, returned as panel data
# ---------------------------------------------------------------------------
def body_panels(midpoints, normals, lengths):
    """Bundle an explicit panel description (used by the callers below)."""
    return {
        "midpoints": np.asarray(midpoints, dtype=float),
        "normals": np.asarray(normals, dtype=float),
        "lengths": np.asarray(lengths, dtype=float),
    }


def _curve_panels(x, y):
    """Panels on the closed polyline ``(x, y)`` (counter-clockwise)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if (x[0], y[0]) != (x[-1], y[-1]):
        x = np.append(x, x[0])
        y = np.append(y, y[0])
    dx, dy = np.diff(x), np.diff(y)
    length = np.hypot(dx, dy)
    mid = np.stack((0.5 * (x[:-1] + x[1:]), 0.5 * (y[:-1] + y[1:])), axis=1)
    # outward normal for a counter-clockwise curve is (dy, -dx) normalised
    normals = np.stack((dy, -dx), axis=1) / length[:, None]
    return body_panels(mid, normals, length)


def circle_panels(radius, n=360):
    theta = np.linspace(0.0, 2.0 * math.pi, int(n) + 1)
    return _curve_panels(radius * np.cos(theta), radius * np.sin(theta))


def ellipse_panels(a, b, n=360):
    theta = np.linspace(0.0, 2.0 * math.pi, int(n) + 1)
    return _curve_panels(a * np.cos(theta), b * np.sin(theta))


def rectangle_panels(c, h, per_edge=200, cluster=0.6):
    """Rectangle ``[-c, c] x [-h, h]``, counter-clockwise, outward normals.

    Each side gets ``per_edge`` panels with cosine clustering towards both ends
    (``cluster`` in [0, 1) sets how strongly the panel density grows near the 90
    degree corners, where the potential gradient is largest).
    """
    def side(p0, p1, n_panels):
        u = np.linspace(0.0, 1.0, n_panels + 1)
        s = u - cluster / (2.0 * math.pi) * np.sin(2.0 * math.pi * u)
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        return p0[None, :] + s[:, None] * (p1 - p0)[None, :]

    pts = np.concatenate([
        side((-c, -h), (c, -h), per_edge)[:-1],   # bottom, left -> right
        side((c, -h), (c, h), per_edge)[:-1],     # right, bottom -> top
        side((c, h), (-c, h), per_edge)[:-1],     # top, right -> left
        side((-c, h), (-c, -h), per_edge),        # left, top -> bottom (closes)
    ])
    return _curve_panels(pts[:, 0], pts[:, 1])


# ---------------------------------------------------------------------------
# BEM operators
# ---------------------------------------------------------------------------
def influence_matrices(panels):
    """Potential influence ``A`` and normal-derivative ``D`` (dense)."""
    mid = panels["midpoints"]
    nrm = panels["normals"]
    ln = panels["lengths"]
    n = len(ln)
    A = np.empty((n, n))
    D = np.empty((n, n))
    for k in range(n):
        # Gauss points and weights on every panel j
        tang = np.stack((-nrm[:, 1], nrm[:, 0]), axis=1)          # unit tangent
        gp = (mid[:, None, :] + 0.5 * ln[:, None, None]
              * _GAUSS_X[None, :, None] * tang[:, None, :])
        r = mid[k][None, None, :] - gp
        dist = np.hypot(r[..., 0], r[..., 1])
        logr = np.log(np.maximum(dist, 1e-12))
        # kernel weights
        wj = (0.5 * ln[:, None] * _GAUSS_W[None, :])
        A[k] = (logr * wj).sum(axis=1) / (2.0 * math.pi)
        dgdn = (r[..., 0] * nrm[k][0] + r[..., 1] * nrm[k][1]) / np.maximum(dist ** 2, 1e-24)
        D[k] = (dgdn * wj).sum(axis=1) / (2.0 * math.pi)
        # analytic self terms
        A[k, k] = ln[k] * (math.log(0.5 * ln[k]) - 1.0) / (2.0 * math.pi)
        D[k, k] = 0.0
    D += 0.5 * np.eye(n)
    return A, D


def added_mass_matrix(panels, rho=1.0, matrices=None):
    """``m_ij`` for unit velocity in ``j`` and force in ``i`` (per unit length)."""
    A, D = matrices if matrices is not None else influence_matrices(panels)
    nrm = panels["normals"]
    ln = panels["lengths"]
    phi = np.empty((2, len(ln)))
    for j, direction in enumerate(((1.0, 0.0), (0.0, 1.0))):
        g = nrm @ np.asarray(direction, dtype=float)
        sigma = np.linalg.solve(D, g)
        phi[j] = A @ sigma
    m = np.empty((2, 2))
    for i in range(2):
        for j in range(2):
            m[i, j] = -rho * float(np.sum(phi[j] * nrm[:, i] * ln))
    return m


def rectangle_added_mass_coefficient(h_over_c, rho=1.0, per_edge=160, c=1.0):
    """``C_a = m_yy / (rho pi c^2)`` for a rectangle in broadside (``y``) motion."""
    panels = rectangle_panels(c, h_over_c * c, per_edge=per_edge)
    m = added_mass_matrix(panels, rho=rho)
    return float(m[1, 1] / (rho * math.pi * c ** 2))


# ---------------------------------------------------------------------------
# self-validation
# ---------------------------------------------------------------------------
def benchmarks(n_circle=360, n_ellipse=360, rho=1.0):
    """Added mass of the analytic shapes vs their closed-form values."""
    a = 1.0
    rows = []
    mc = added_mass_matrix(circle_panels(a, n_circle), rho=rho)
    rows.append({
        "shape": "circle", "a": a,
        "m_xx": float(mc[0, 0]), "m_yy": float(mc[1, 1]),
        "ref": rho * math.pi * a ** 2,
        "err_xx": float(mc[0, 0] / (rho * math.pi * a ** 2) - 1.0),
        "err_yy": float(mc[1, 1] / (rho * math.pi * a ** 2) - 1.0),
    })
    for (ea, eb) in ((1.0, 0.5), (1.0, 0.25), (1.0, 0.1)):
        me = added_mass_matrix(ellipse_panels(ea, eb, n_ellipse), rho=rho)
        # motion along x -> rho pi b^2 ; along y -> rho pi a^2
        rows.append({
            "shape": f"ellipse(a={ea},b={eb})", "a": ea, "b": eb,
            "m_xx": float(me[0, 0]), "m_yy": float(me[1, 1]),
            "ref_xx": rho * math.pi * eb ** 2, "ref_yy": rho * math.pi * ea ** 2,
            "err_xx": float(me[0, 0] / (rho * math.pi * eb ** 2) - 1.0),
            "err_yy": float(me[1, 1] / (rho * math.pi * ea ** 2) - 1.0),
        })
    return rows