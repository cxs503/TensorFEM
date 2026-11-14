"""Independent finite-difference reference for a simply supported orthotropic
plate under an eccentric rectangular patch load.

Purpose: a verification reference that shares *no* code and *no* method with the
Navier (double sine series) solution used by
``grillage_ice_ultimate.plate_patch_response``.  The series solution forces the
user to choose a mode set; getting that choice wrong for an off-centre patch
(odd modes only) is invisible from inside the series itself, so a discretisation
in physical space is needed to check it.

Governing equation for the smeared orthotropic hull panel::

    D1 w,xxxx + 2 D3 w,xxyy + D2 w,yyyy = q(x, y)

with simple supports on all four edges (``w = 0`` and ``w,nn = 0``), which is the
convention of ``plate_patch_response`` (there ``D3 = nu sqrt(D1 D2) + 2 Dxy``
already contains the Poisson term, so the mixed derivative carries the factor
``2 D3``).

Discretisation: the standard 13-point fourth-order stencil for the biharmonic
operator (central differences, 2nd-order accurate) on a uniform grid with equal
spacing in each direction, simply supported edges imposed with antisymmetric
ghost nodes (``w_-1 = -w_1``), and the patch load integrated *exactly* over each
cell so the staircase representation of the load does not add first-order error.
Sparse direct solve, so the result is a genuine independent reference; it
converges to the series solution from below as the grid is refined.

Run a self-check::

    PYTHONPATH=src python -m tensorfem.orthotropic_plate_fd
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["ssss_orthotropic_plate_fd"]

# (di, dj, coefficient of the biharmonic stencil), scaled per direction below
_FOURTH = ((0, 0, 6.0), (1, 0, -4.0), (2, 0, 1.0), (-1, 0, -4.0), (-2, 0, 1.0))
_MIXED = ((1, 1, 1.0), (1, -1, 1.0), (-1, 1, 1.0), (-1, -1, 1.0),
          (1, 0, -2.0), (-1, 0, -2.0), (0, 1, -2.0), (0, -1, -2.0),
          (0, 0, 4.0))


def ssss_orthotropic_plate_fd(*, span_x: float, span_y: float, d1: float,
                             d2: float, d3: float, pressure: float,
                             patch: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
                             nx: int = 120, ny: int | None = None,
                             poisson: float = 0.3,
                             return_field: bool = False) -> dict[str, object]:
    """Simply supported orthotropic plate under a rectangular patch load.

    ``patch = (fx1, fx2, fy1, fy2)`` are fractional extents of the plate spans
    (same convention as ``plate_patch_response``) and need not be centred.
    Returns the peak deflection, the peak bending moments, their locations, and
    the grid used, all in SI units.  With ``return_field`` the deflection array
    and its coordinates are added (plotting / slice comparison).
    """
    for name, value in (("span_x", span_x), ("span_y", span_y), ("d1", d1),
                        ("d2", d2), ("d3", d3)):
        v = float(value)
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    if not math.isfinite(float(pressure)) or abs(float(pressure)) < 1e-30:
        raise ValueError("pressure must be a finite non-zero number")
    fx1, fx2, fy1, fy2 = (float(v) for v in patch)
    for name, v in (("fx1", fx1), ("fx2", fx2), ("fy1", fy1), ("fy2", fy2)):
        if not (-1e-12 <= v <= 1.0 + 1e-12):
            raise ValueError(f"{name} must be a fraction in [0, 1], got {v!r}")
    if not (fx2 > fx1 and fy2 > fy1):
        raise ValueError("patch extents must satisfy fx2 > fx1 and fy2 > fy1")
    nx = int(nx)
    if nx < 8:
        raise ValueError("nx must be >= 8")
    ny = int(round(nx * span_y / span_x)) if ny is None else int(ny)
    if ny < 8:
        raise ValueError("ny must be >= 8")

    hx, hy = span_x / nx, span_y / ny
    n = (nx - 1) * (ny - 1)

    def index(i: int, j: int) -> int:
        return (i - 1) * (ny - 1) + (j - 1)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    def add(row: int, i: int, j: int, coef: float) -> None:
        """Add ``coef * w(i, j)`` to ``row``, applying the SS boundary rules."""
        sign = 1.0
        if i < 0:
            i, sign = -i, -sign
        elif i > nx:
            i, sign = 2 * nx - i, -sign
        if j < 0:
            j, sign = -j, -sign
        elif j > ny:
            j, sign = 2 * ny - j, -sign
        if i == 0 or i == nx or j == 0 or j == ny:  # support: w = 0, no unknown
            return
        rows.append(row)
        cols.append(index(i, j))
        vals.append(sign * coef)

    for i in range(1, nx):
        for j in range(1, ny):
            row = index(i, j)
            for di, _dj, c in _FOURTH:  # 1-D stencil, applied to x then to y
                add(row, i + di, j, c * d1 / hx ** 4)
                add(row, i, j + di, c * d2 / hy ** 4)
            for di, dj, c in _MIXED:
                add(row, i + di, j + dj, c * 2.0 * d3 / (hx * hx * hy * hy))

    A = sp.csr_matrix(sp.coo_matrix((vals, (rows, cols)), shape=(n, n)))

    # patch load integrated exactly over each cell (no staircase error)
    x1, x2 = fx1 * span_x, fx2 * span_x
    y1, y2 = fy1 * span_y, fy2 * span_y
    xe = np.linspace(0.0, span_x, nx + 1)
    ye = np.linspace(0.0, span_y, ny + 1)
    ox = np.clip(np.minimum(xe[1:], x2) - np.maximum(xe[:-1], x1), 0.0, None)
    oy = np.clip(np.minimum(ye[1:], y2) - np.maximum(ye[:-1], y1), 0.0, None)
    q_cell = float(pressure) * np.outer(ox, oy) / (hx * hy)  # per unit area
    rhs = q_cell[1:nx, 1:ny].reshape(-1).copy()

    w_int = spla.spsolve(A.tocsc(), rhs)
    w = np.zeros((nx + 1, ny + 1))
    w[1:nx, 1:ny] = w_int.reshape(nx - 1, ny - 1)

    # curvatures / moments by central differences (w = 0 on the boundary)
    wxx = np.zeros_like(w)
    wyy = np.zeros_like(w)
    wxx[1:nx, 1:ny] = (w[0:nx - 1, 1:ny] - 2.0 * w[1:nx, 1:ny]
                       + w[2:nx + 1, 1:ny]) / hx ** 2
    wyy[1:nx, 1:ny] = (w[1:nx, 0:ny - 1] - 2.0 * w[1:nx, 1:ny]
                       + w[1:nx, 2:ny + 1]) / hy ** 2
    d12 = poisson * math.sqrt(d1 * d2)
    m_x = -(d1 * wxx + d12 * wyy)
    m_y = -(d2 * wyy + d12 * wxx)

    i_peak, j_peak = np.unravel_index(np.argmax(np.abs(w)), w.shape)
    mx_abs = np.abs(m_x[1:nx, 1:ny])
    im, jm = np.unravel_index(np.argmax(mx_abs), mx_abs.shape)
    my_abs = np.abs(m_y[1:nx, 1:ny])
    imy, jmy = np.unravel_index(np.argmax(my_abs), my_abs.shape)
    out: dict[str, object] = {
        "w_max": float(np.max(np.abs(w))),
        "w_peak_x": float(i_peak * hx),
        "w_peak_y": float(j_peak * hy),
        "m_x_max": float(mx_abs[im, jm]),
        "m_x_peak_x": float((im + 1) * hx),
        "m_x_peak_y": float((jm + 1) * hy),
        "m_y_max": float(my_abs[imy, jmy]),
        "nx": float(nx),
        "ny": float(ny),
        "hx": hx,
        "hy": hy,
        "total_force": float(pressure) * (x2 - x1) * (y2 - y1),
        "load_area": (x2 - x1) * (y2 - y1),
    }
    if return_field:
        out["w"] = w
        out["x"] = np.linspace(0.0, span_x, nx + 1)
        out["y"] = np.linspace(0.0, span_y, ny + 1)
    return out


def _self_check() -> None:  # pragma: no cover - manual smoke test
    """Grid convergence for a centred patch on an isotropic square plate."""
    from .grillage_ice_ultimate import plate_patch_response

    side, d, p = 2.0, 1.0e6, 1.0e4
    patch = (0.4, 0.6, 0.4, 0.6)
    for nx in (24, 48, 96):
        fd = ssss_orthotropic_plate_fd(span_x=side, span_y=side, d1=d, d2=d,
                                       d3=d, pressure=p, patch=patch, nx=nx)
        print(f"nx={nx:4d} w_max={fd['w_max']:.6e}")
    nav = plate_patch_response(span_x=side, span_y=side, d1=d, d2=d, d3=d,
                               pressure=p, patch=patch)
    print(f"navier   w_max={nav['w_max']:.6e}  m_x_max={nav['m_x_max']:.6e}")


if __name__ == "__main__":  # pragma: no cover
    _self_check()