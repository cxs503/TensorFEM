"""Level-set front tracking for crack propagation (2-D, narrow band).

The crack is represented implicitly as the zero contour of a scalar field
``phi(x, t)``: the crack surface is ``phi = 0``, its two banks are ``phi < 0``
and ``phi > 0``.  The field is kept a *signed distance function*
(``|grad phi| = 1``) so that the crack path length is the integral of the field,
the normal at any point is ``grad phi``, and a tip advance is a rigid
translation of the front.

Numerical ingredients (each carries its own machine-checkable quality metric):

``LevelSetField.reinitialize``
    Signed-distance re-initialisation by the Sussman PDE iteration

        dphi/dtau + S(phi0) (|grad phi| - 1) = 0,  S(phi0) = phi0/sqrt(phi0^2+h^2)

    discretised with the Godunov upwind Hamiltonian (2nd-order one-sided
    differences in the interior, 1st order at the boundary) and explicit Euler
    steps ``dtau = cfl*h/max|S|``.  :meth:`gradient_error` reports
    ``max ||grad phi| - 1|`` and the RMS value over the band.

``LevelSetField.advect``
    Transport of the front with the crack-tip velocity field,
    ``dphi/dt + V . grad phi = 0``.  Two schemes: ``"upwind"`` (1st-order
    Eulerian, CFL limited, sub-stepped) and ``"sl2"`` (semi-Lagrangian RK2
    backtrace with bilinear sampling, unconditionally stable -- the scheme used
    by the crack-growth driver).

``LevelSetField.extend_velocity``
    Velocity extension off the front, ``grad V . grad phi = 0``
    (Adalsteinsson & Sethian ordered sweeping), so that a speed known only on
    the crack set transports the whole band without destroying the distance
    property.

Geometry: :meth:`LevelSetField.zero_contour` (marching squares with exact
edge-key topology), :func:`resample_polyline`, :func:`polyline_deviation`
(one-way Hausdorff / RMS deviation used to compare a computed crack path with a
reference path, normalised by the reference path length).

Index convention: ``phi[j, i]`` at ``x = x0 + i*h``, ``y = y0 + j*h``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "LevelSetField",
    "circle_level_set",
    "plane_level_set",
    "crack_level_set",
    "resample_polyline",
    "polyline_deviation",
    "path_length",
]

_EPS = 1.0e-14


@dataclass
class LevelSetField:
    """Signed distance field on a uniform Cartesian grid."""

    phi: np.ndarray
    h: float
    x0: float = 0.0
    y0: float = 0.0

    def __post_init__(self) -> None:
        phi = np.ascontiguousarray(self.phi, dtype=np.float64)
        if phi.ndim != 2:
            raise ValueError("phi must be a 2-D array")
        if not (self.h > 0.0):
            raise ValueError("h must be positive")
        if not np.all(np.isfinite(phi)):
            raise ValueError("phi must be finite")
        self.phi = phi

    @property
    def shape(self) -> tuple[int, int]:
        return self.phi.shape

    @property
    def ny(self) -> int:
        return self.phi.shape[0]

    @property
    def nx(self) -> int:
        return self.phi.shape[1]

    def copy(self) -> "LevelSetField":
        return LevelSetField(self.phi.copy(), self.h, self.x0, self.y0)

    def coords(self) -> tuple[np.ndarray, np.ndarray]:
        jj, ii = np.meshgrid(np.arange(self.ny), np.arange(self.nx), indexing="ij")
        return self.x0 + ii * self.h, self.y0 + jj * self.h

    def values_at(self, points: np.ndarray) -> np.ndarray:
        """Bilinear interpolation of the field at ``points`` of shape (N, 2)."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        fx = (pts[:, 0] - self.x0) / self.h
        fy = (pts[:, 1] - self.y0) / self.h
        fx = np.clip(fx, 0.0, self.nx - 1.0 - 1e-12)
        fy = np.clip(fy, 0.0, self.ny - 1.0 - 1e-12)
        i0 = np.floor(fx).astype(int)
        j0 = np.floor(fy).astype(int)
        tx = fx - i0
        ty = fy - j0
        p = self.phi
        return (
            p[j0, i0] * (1 - tx) * (1 - ty)
            + p[j0, i0 + 1] * tx * (1 - ty)
            + p[j0 + 1, i0] * (1 - tx) * ty
            + p[j0 + 1, i0 + 1] * tx * ty
        )

    # -------------------------------------------------------------- gradients
    def gradient(self) -> tuple[np.ndarray, np.ndarray]:
        """Central-difference gradient (2nd order interior, 1st order at edges)."""
        p, h = self.phi, self.h
        gx = np.empty_like(p)
        gy = np.empty_like(p)
        gx[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2.0 * h)
        gx[:, 0] = (p[:, 1] - p[:, 0]) / h
        gx[:, -1] = (p[:, -1] - p[:, -2]) / h
        gy[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2.0 * h)
        gy[0, :] = (p[1, :] - p[0, :]) / h
        gy[-1, :] = (p[-1, :] - p[-2, :]) / h
        return gx, gy

    def gradient_norm(self) -> np.ndarray:
        gx, gy = self.gradient()
        return np.hypot(gx, gy)

    def gradient_error(self, band: float | None = None) -> dict[str, float]:
        """Deviation of ``|grad phi|`` from unity inside ``|phi| <= band``."""
        g = self.gradient_norm()
        if band is None:
            band = float(np.max(np.abs(self.phi)))
        mask = np.abs(self.phi) <= band
        if not np.any(mask):
            raise ValueError("band contains no grid points")
        err = np.abs(g[mask] - 1.0)
        return {
            "band": float(band),
            "max_abs": float(err.max()),
            "rms": float(np.sqrt(np.mean(err ** 2))),
            "rel_max": float(err.max()),
            "rel_rms": float(np.sqrt(np.mean(err ** 2))),
            "points": int(mask.sum()),
        }

    def error_vs_exact(
        self, exact: Callable[[np.ndarray, np.ndarray], np.ndarray],
        band: float | None = None,
    ) -> dict[str, float]:
        """``L-inf``/RMS error of ``phi`` against an analytic signed distance."""
        X, Y = self.coords()
        ref = np.asarray(exact(X, Y), dtype=np.float64)
        if band is None:
            band = float(np.max(np.abs(self.phi)))
        mask = np.abs(ref) <= band
        d = self.phi[mask] - ref[mask]
        return {
            "band": float(band),
            "max_abs": float(np.max(np.abs(d))),
            "rms": float(np.sqrt(np.mean(d ** 2))),
            "max_rel_to_h": float(np.max(np.abs(d)) / self.h),
            "points": int(mask.sum()),
        }

    # ---------------------------------------------------------- reinit: Sussman
    def reinitialize(
        self,
        *,
        band: float | None = None,
        n_iter: int | None = None,
        cfl: float = 0.3,
        order: int = 1,
    ) -> "LevelSetField":
        """Signed-distance re-initialisation (Sussman PDE, Godunov upwind).

        A frozen copy ``phi0`` holds the sign function, so the zero contour does
        not move; the iteration only repairs ``|grad phi| = 1``.  Only the band
        ``|phi0| <= band + 2h`` is updated (narrow-band re-initialisation);
        outside it the field is left untouched.

        ``order=1`` (default) is the monotone first-order Godunov Hamiltonian:
        robust, converges to a monotone distance function, truncation error
        ``O(h)``.  ``order=2`` uses second-order one-sided differences in the
        interior of the band and falls back to first order within two cells of
        the band edge or the domain boundary (without that fallback the
        second-order stencil across the band-edge jump is unstable -- observed
        and documented in ``docs/levelset-ice-plate-fracture.md``).
        """
        h = self.h
        phi0 = self.phi.copy()
        phi = phi0.copy()
        if band is None:
            band = float(np.max(np.abs(phi0)))
        if n_iter is None:
            n_iter = int(math.ceil(4.0 * band / (cfl * h))) + 10
        dtau = cfl * h
        s = phi0 / np.sqrt(phi0 ** 2 + h ** 2)
        seed = np.abs(phi0) <= band + 2.0 * h
        for _ in range(int(n_iter)):
            # the update region tracks the *current* field as well, so a badly
            # distorted initial field cannot leave a jump at the band edge
            active = seed | (np.abs(phi) <= band + 2.0 * h)
            safe = _second_order_safe(active)
            if order >= 2:
                dmx = _pick(_backward(phi, h, 2), _backward(phi, h, 1), safe)
                dpx = _pick(_forward(phi, h, 2), _forward(phi, h, 1), safe)
                dmy = _pick(_backward(phi.T, h, 2).T, _backward(phi.T, h, 1).T, safe)
                dpy = _pick(_forward(phi.T, h, 2).T, _forward(phi.T, h, 1).T, safe)
            else:
                dmx = _backward(phi, h, 1)
                dpx = _forward(phi, h, 1)
                dmy = _backward(phi.T, h, 1).T
                dpy = _forward(phi.T, h, 1).T
            pos = s > 0.0
            grad2 = (
                np.where(pos, np.maximum(dmx, 0.0), np.maximum(dpx, 0.0)) ** 2
                + np.where(pos, np.minimum(dpx, 0.0), np.minimum(dmx, 0.0)) ** 2
                + np.where(pos, np.maximum(dmy, 0.0), np.maximum(dpy, 0.0)) ** 2
                + np.where(pos, np.minimum(dpy, 0.0), np.minimum(dmy, 0.0)) ** 2
            )
            upd = s * (np.sqrt(grad2) - 1.0)
            phi = np.where(active, phi - dtau * upd, phi)
        return LevelSetField(phi, h, self.x0, self.y0)

    # --------------------------------------------------------------- advection
    def advect(
        self,
        vx: np.ndarray | float,
        vy: np.ndarray | float,
        dt: float,
        *,
        scheme: str = "sl2",
        substeps: int | None = None,
    ) -> "LevelSetField":
        """Advance ``dphi/dt + V . grad phi = 0`` by ``dt``."""
        vx_arr = np.asarray(vx, dtype=np.float64)
        vy_arr = np.asarray(vy, dtype=np.float64)
        if vx_arr.ndim == 0:
            vx_arr = np.full(self.shape, float(vx_arr))
        if vy_arr.ndim == 0:
            vy_arr = np.full(self.shape, float(vy_arr))
        if vx_arr.shape != self.shape or vy_arr.shape != self.shape:
            raise ValueError("velocity fields must broadcast to the grid shape")
        if scheme == "sl2":
            return LevelSetField(
                _semi_lagrangian(self, vx_arr, vy_arr, dt), self.h, self.x0, self.y0
            )
        if scheme == "upwind":
            speed = max(float(np.max(np.hypot(vx_arr, vy_arr))), _EPS)
            n = substeps or max(1, int(math.ceil(dt * speed / (0.5 * self.h))))
            out = self.phi.copy()
            dt_sub = dt / n
            for _ in range(n):
                dmx = _backward(out, self.h, 1)
                dpx = _forward(out, self.h, 1)
                dmy = _backward(out.T, self.h, 1).T
                dpy = _forward(out.T, self.h, 1).T
                flux_x = np.where(vx_arr > 0.0, vx_arr * dmx, vx_arr * dpx)
                flux_y = np.where(vy_arr > 0.0, vy_arr * dmy, vy_arr * dpy)
                out = out - dt_sub * (flux_x + flux_y)
            return LevelSetField(out, self.h, self.x0, self.y0)
        raise ValueError("scheme must be 'sl2' or 'upwind'")

    def advect_tip_speed(
        self, speed: float, direction: Sequence[float], dt: float, *,
        scheme: str = "sl2",
    ) -> "LevelSetField":
        """Constant-extension growth: rigid translation of the front by ``speed*dt``."""
        n = np.asarray(direction, dtype=np.float64)
        norm = float(np.hypot(*n))
        if norm <= 0.0:
            raise ValueError("direction must be non-zero")
        n = n / norm
        return self.advect(speed * n[0], speed * n[1], dt, scheme=scheme)

    # -------------------------------------------------------- velocity extension
    def extend_velocity(self, front_velocity: float) -> np.ndarray:
        """Extend a front value off the front along the field normals.

        Solves ``grad V . grad phi = 0`` by ordered sweeping: nodes are visited
        in order of increasing ``|phi|`` and each unknown takes the
        distance-weighted average of the already-known neighbours on its own
        side of the front (Adalsteinsson & Sethian 1995).
        """
        phi = self.phi
        out = np.full(phi.shape, np.nan)
        out[np.abs(phi) < self.h] = float(front_velocity)
        order = np.argsort(phi.ravel(), kind="stable")
        flat = phi.ravel()
        plus = order[flat[order] > 0.0]
        minus = order[flat[order] < 0.0][::-1]
        off = out.ravel()
        ny, nx = phi.shape
        for seq in (plus, minus):
            for k in seq:
                if not np.isnan(off[k]):
                    continue
                j, i = divmod(int(k), nx)
                ph = abs(phi[j, i])
                num = 0.0
                den = 0.0
                for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    jj, ii = j + dj, i + di
                    if 0 <= jj < ny and 0 <= ii < nx:
                        v = off[jj * nx + ii]
                        if not np.isnan(v):
                            w = ph - abs(phi[jj, ii])
                            if w > 0.0:
                                num += w * v
                                den += w
                if den > 0.0:
                    off[k] = num / den
        if np.isnan(off).any():
            off[np.isnan(off)] = float(front_velocity)
        return off.reshape(phi.shape)

    # ------------------------------------------------------------------ geometry
    def zero_contour(self, level: float = 0.0) -> list[np.ndarray]:
        """Extract the ``phi = level`` isoline as ordered polylines ``(N, 2)``."""
        return _marching_squares(self, level)


# ---------------------------------------------------------------------------
# finite-difference stencils
# ---------------------------------------------------------------------------
def _backward(p: np.ndarray, h: float, order: int) -> np.ndarray:
    d = np.empty_like(p)
    if order >= 2:
        d[:, 2:] = (3.0 * p[:, 2:] - 4.0 * p[:, 1:-1] + p[:, :-2]) / (2.0 * h)
    d[:, 0] = (p[:, 0] - p[:, 1]) / h
    if order >= 2:
        d[:, 1] = (p[:, 1] - p[:, 0]) / h
    else:
        d[:, 1:] = (p[:, 1:] - p[:, :-1]) / h
    return d


def _forward(p: np.ndarray, h: float, order: int) -> np.ndarray:
    d = np.empty_like(p)
    if order >= 2:
        d[:, :-2] = (-3.0 * p[:, :-2] + 4.0 * p[:, 1:-1] - p[:, 2:]) / (2.0 * h)
    d[:, -1] = (p[:, -1] - p[:, -2]) / h
    if order >= 2:
        d[:, -2] = (p[:, -1] - p[:, -2]) / h
    else:
        d[:, :-1] = (p[:, 1:] - p[:, :-1]) / h
    return d


def _pick(second: np.ndarray, first: np.ndarray, safe: np.ndarray) -> np.ndarray:
    return np.where(safe, second, first)


def _second_order_safe(active: np.ndarray, margin: int = 2) -> np.ndarray:
    """Cells where the second-order one-sided stencil stays inside the band."""
    ny, nx = active.shape
    safe = active.copy()
    if ny <= 2 * margin or nx <= 2 * margin:
        return np.zeros_like(active)
    safe[:, :margin] = False
    safe[:, -margin:] = False
    safe[:margin, :] = False
    safe[-margin:, :] = False
    for dj, di in ((margin, 0), (-margin, 0), (0, margin), (0, -margin)):
        shifted = np.zeros_like(active)
        js = slice(max(dj, 0), ny + min(dj, 0))
        jt = slice(max(-dj, 0), ny + min(-dj, 0))
        isl = slice(max(di, 0), nx + min(di, 0))
        itl = slice(max(-di, 0), nx + min(-di, 0))
        shifted[js, isl] = active[jt, itl]
        safe &= shifted
    return safe


def _semi_lagrangian(field: LevelSetField, vx: np.ndarray, vy: np.ndarray,
                     dt: float) -> np.ndarray:
    """RK2 backtrace + bilinear sampling of the level-set equation."""
    ny, nx = field.shape
    h = field.h
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    x = field.x0 + ii * h
    y = field.y0 + jj * h
    # midpoint of the characteristic
    xm = np.clip(x - 0.5 * dt * vx, field.x0, field.x0 + (nx - 1) * h)
    ym = np.clip(y - 0.5 * dt * vy, field.y0, field.y0 + (ny - 1) * h)
    mid = np.stack((xm, ym), axis=-1).reshape(-1, 2)
    vxm = field.values_at_field(vx, mid).reshape(ny, nx)
    vym = field.values_at_field(vy, mid).reshape(ny, nx)
    xd = np.clip(x - dt * vxm, field.x0, field.x0 + (nx - 1) * h)
    yd = np.clip(y - dt * vym, field.y0, field.y0 + (ny - 1) * h)
    dep = np.stack((xd, yd), axis=-1).reshape(-1, 2)
    return field.values_at(dep).reshape(ny, nx)


def _values_at_array(arr: np.ndarray, h: float, x0: float, y0: float,
                     points: np.ndarray) -> np.ndarray:
    ny, nx = arr.shape
    pts = np.atleast_2d(points)
    fx = np.clip((pts[:, 0] - x0) / h, 0.0, nx - 1.0 - 1e-12)
    fy = np.clip((pts[:, 1] - y0) / h, 0.0, ny - 1.0 - 1e-12)
    i0 = np.floor(fx).astype(int)
    j0 = np.floor(fy).astype(int)
    tx = fx - i0
    ty = fy - j0
    return (
        arr[j0, i0] * (1 - tx) * (1 - ty)
        + arr[j0, i0 + 1] * tx * (1 - ty)
        + arr[j0 + 1, i0] * (1 - tx) * ty
        + arr[j0 + 1, i0 + 1] * tx * ty
    )


def _values_at_field(self: LevelSetField, arr: np.ndarray, points: np.ndarray) -> np.ndarray:
    return _values_at_array(arr, self.h, self.x0, self.y0, points)


LevelSetField.values_at_field = _values_at_field  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# marching squares
# ---------------------------------------------------------------------------
def _marching_squares(field: LevelSetField, level: float) -> list[np.ndarray]:
    """Isoline extraction with exact edge-key topology so segments stitch exactly."""
    p = field.phi - float(level)
    ny, nx = p.shape
    h = field.h
    nodes: dict[tuple, tuple[float, float]] = {}

    def vcoord(key: tuple) -> tuple[float, float]:
        if key in nodes:
            return nodes[key]
        kind, i, j = key
        if kind == "h":  # between (i, j) and (i+1, j)
            a, b = p[j, i], p[j, i + 1]
            t = a / (a - b)
            xy = (field.x0 + (i + t) * h, field.y0 + j * h)
        else:  # "v": between (i, j) and (i, j+1)
            a, b = p[j, i], p[j + 1, i]
            t = a / (a - b)
            xy = (field.x0 + i * h, field.y0 + (j + t) * h)
        nodes[key] = xy
        return xy

    segs: list[tuple[tuple, tuple]] = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v00 = p[j, i]
            v10 = p[j, i + 1]
            v11 = p[j + 1, i + 1]
            v01 = p[j + 1, i]
            inside = (v00 > 0, v10 > 0, v11 > 0, v01 > 0)
            n_in = sum(inside)
            if n_in == 0 or n_in == 4:
                continue
            e = []  # crossing edge keys in CCW order: bottom, right, top, left
            if inside[0] != inside[1]:
                e.append(("h", i, j))
            if inside[1] != inside[2]:
                e.append(("v", i + 1, j))
            if inside[3] != inside[2]:
                e.append(("h", i, j + 1))
            if inside[0] != inside[3]:
                e.append(("v", i, j))
            if len(e) == 2:
                segs.append((e[0], e[1]))
            elif len(e) == 4:
                centre = 0.25 * (v00 + v10 + v11 + v01)
                if (centre > 0) == inside[0]:
                    for a, b in ((0, 1), (2, 3)):
                        segs.append((e[a], e[b]))
                else:
                    for a, b in ((0, 3), (1, 2)):
                        segs.append((e[a], e[b]))
    # stitch segments into polylines
    adj: dict[tuple, list[tuple]] = {}
    for a, b in segs:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    paths: list[np.ndarray] = []
    visited: set[frozenset] = set()
    for a, b in segs:
        if frozenset((a, b)) in visited:
            continue
        chain = [a, b]
        visited.add(frozenset((a, b)))
        while True:
            tail = chain[-1]
            nxt = None
            for cand in adj[tail]:
                if frozenset((tail, cand)) not in visited:
                    nxt = cand
                    break
            if nxt is None:
                break
            visited.add(frozenset((tail, nxt)))
            chain.append(nxt)
        # try to extend at the head
        while True:
            head = chain[0]
            nxt = None
            for cand in adj[head]:
                if frozenset((head, cand)) not in visited:
                    nxt = cand
                    break
            if nxt is None:
                break
            visited.add(frozenset((head, nxt)))
            chain.insert(0, nxt)
        paths.append(np.array([vcoord(k) for k in chain], dtype=np.float64))
    paths.sort(key=lambda arr: -path_length(arr))
    return paths


# ---------------------------------------------------------------------------
# path utilities
# ---------------------------------------------------------------------------
def path_length(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return 0.0
    return float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))


def resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to ``n`` equally spaced arc-length points."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return np.repeat(pts[:1], n, axis=0)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    total = s[-1]
    if total <= 0.0:
        return np.repeat(pts[:1], n, axis=0)
    targets = np.linspace(0.0, total, n)
    return np.stack(
        (np.interp(targets, s, pts[:, 0]), np.interp(targets, s, pts[:, 1])), axis=1
    )


def _point_to_polyline_distance(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Shortest distance from each ``points`` row to the polyline ``poly``."""
    a = poly[:-1]
    b = poly[1:]
    ab = b - a
    ab2 = np.einsum("ij,ij->i", ab, ab)
    ab2 = np.where(ab2 <= 0.0, _EPS, ab2)
    out = np.empty(points.shape[0], dtype=np.float64)
    for k, p in enumerate(points):
        ap = p - a
        t = np.clip(np.einsum("ij,ij->i", ap, ab) / ab2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        out[k] = float(np.min(np.hypot(*(proj - p).T)))
    return out


def polyline_deviation(candidate: np.ndarray, reference: np.ndarray,
                       *, samples: int = 200) -> dict[str, float]:
    """Normalised deviation of ``candidate`` from ``reference``.

    Both paths are re-sampled by arc length and the one-way distances are
    reported as a fraction of the reference path length: ``max_rel``
    (Hausdorff-style) and ``rms_rel``.  These are the numbers used against the
    ``< 15 %`` crack-path acceptance gate.
    """
    cand = resample_polyline(candidate, samples)
    ref = resample_polyline(reference, samples)
    ref_len = path_length(ref)
    if ref_len <= 0.0:
        raise ValueError("reference path has zero length")
    d = _point_to_polyline_distance(cand, ref)
    return {
        "max_abs": float(d.max()),
        "rms_abs": float(np.sqrt(np.mean(d ** 2))),
        "max_rel": float(d.max() / ref_len),
        "rms_rel": float(np.sqrt(np.mean(d ** 2)) / ref_len),
        "reference_length": float(ref_len),
        "candidate_length": float(path_length(cand)),
    }


# ---------------------------------------------------------------------------
# constructors
# ---------------------------------------------------------------------------
def _grid(nx: int, ny: int, h: float, x0: float, y0: float):
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    return x0 + ii * h, y0 + jj * h


def circle_level_set(radius: float, *, nx: int = 129, ny: int | None = None,
                     h: float | None = None, x0: float | None = None,
                     y0: float | None = None) -> LevelSetField:
    """Exact signed distance to a circle centred on the domain centre."""
    ny = nx if ny is None else ny
    if h is None:
        h = 2.2 * radius / (nx - 1)
    x0 = -0.5 * (nx - 1) * h if x0 is None else x0
    y0 = -0.5 * (ny - 1) * h if y0 is None else y0
    X, Y = _grid(nx, ny, h, x0, y0)
    return LevelSetField(np.hypot(X, Y) - radius, h, x0, y0)


def plane_level_set(normal: Sequence[float], offset: float = 0.0, *, nx: int = 65,
                    ny: int = 65, h: float = 1.0, x0: float | None = None,
                    y0: float | None = None) -> LevelSetField:
    """Exact signed distance to a plane ``n . x - offset = 0``."""
    n = np.asarray(normal, dtype=np.float64)
    norm = float(np.hypot(*n))
    if norm <= 0.0:
        raise ValueError("normal must be non-zero")
    n = n / norm
    x0 = -0.5 * (nx - 1) * h if x0 is None else x0
    y0 = -0.5 * (ny - 1) * h if y0 is None else y0
    X, Y = _grid(nx, ny, h, x0, y0)
    return LevelSetField(n[0] * X + n[1] * Y - offset, h, x0, y0)


def crack_level_set(length: float, *, centre: Sequence[float] = (0.0, 0.0),
                    angle: float = 0.0, nx: int = 129, ny: int = 129,
                    h: float = 0.05) -> LevelSetField:
    """Exact signed distance to a straight crack segment of ``length``.

    ``phi < 0`` on one bank, ``phi > 0`` on the other; the field is
    differentiable everywhere except on the two tip normal rays, where no
    distance field can be differentiable.
    """
    cx, cy = centre
    x0 = cx - 0.5 * (nx - 1) * h
    y0 = cy - 0.5 * (ny - 1) * h
    X, Y = _grid(nx, ny, h, x0, y0)
    dx, dy = math.cos(angle), math.sin(angle)
    s = (X - cx) * dx + (Y - cy) * dy
    r = -(X - cx) * dy + (Y - cy) * dx
    half = 0.5 * length
    along = np.clip(s, -half, half)
    inside = np.abs(s) <= half * (1.0 + 1e-9)  # keep the tips on grid nodes
    # one-sided "notch" representation: the signed perpendicular distance on the
    # crack itself, positive everywhere ahead of the tips.  Keeping the bank sign
    # beyond the tips (a signed distance to the whole segment) would put a
    # spurious zero contour along the tangent line there, which would be picked
    # up by the front extraction; the price is a jump across the two tip rays.
    phi = np.where(inside, r, np.hypot(s - along, r))
    return LevelSetField(phi, h, x0, y0)
