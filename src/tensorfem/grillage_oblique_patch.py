"""Oblique (rotated) ice pressure patch on the ice-belt grillage.

Why this module exists
----------------------
``grillage_ice_ultimate`` treats the ice pressure patch as an **axis-aligned
rectangle** whose fractional extents are clamped to the representative panel
(``CombinedLoadCase.patch_extents``), always centred on the panel.  In reality an
ice edge is rarely parallel to the framing: a patch inclined to the longitudinal
stiffeners spills over a *different* set of members, changes the modal load
share, and moves the peak plate moment off the plate centre.  The independent
audit of the anchor ``GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE`` listed "oblique
patch" as ``not_run``.  This module removes that restriction for the plate
response and for the member load path, without touching any of the verified
axis-aligned numbers.

What is added
-------------
1. ``ObliqueIcePatch`` -- a rectangular patch of arbitrary centre, length,
   width and orientation ``theta`` about its own centre, clipped exactly to the
   representative panel.
2. ``patch_fourier_coefficients`` -- the Navier load coefficients
   ``q_mn = (4/ab) * integral q sin(m pi x/a) sin(n pi y/b) dA`` for an arbitrary
   load polygon, evaluated by exact polygon clipping on a structured cell grid
   plus a degree-5 triangle Gauss rule.  For an axis-aligned patch this
   reproduces the closed-form coefficients of
   ``grillage_ice_ultimate.plate_patch_response`` (gate ``O1``), so the general
   path is anchored to the verified special case instead of replacing it.
3. ``oblique_patch_response`` -- the same orthotropic SSSS Navier field (peak
   deflection and bending moments, modal load share ``f_x + f_y + f_t = 1``)
   driven by the oblique load.
4. ``tributary_areas`` -- exact polygon/strip clipping of the patch onto the
   longitudinal-stiffener and transverse-frame tributary strips: the
   generalisation of the axis-aligned layer's "the ``n_overlap`` touched
   stiffeners share the patch load equally" rule, which it reproduces exactly for
   an axis-aligned centred patch (gate ``O5``).
5. ``oblique_utilisations`` / ``oblique_ultimate_pressure`` -- the member
   utilisation set of ``grillage_ice_ultimate`` re-evaluated for the oblique load
   path.  **The capacity laws are imported, not re-implemented** (plate strip
   plastic moment, effective-width beam-column, frame rigid-plastic collapse):
   only the load entering them changes, so the two modules cannot drift apart in
   their physics.
6. ``orientation_sensitivity`` -- the machine-readable "patch orientation ->
   ultimate ice pressure" table that answers the reviewer question "what does the
   assumption that the ice patch is parallel to the framing cost?".

Method summary
--------------
* Plate: classical Navier double series on the representative orthotropic panel
  (``span_x = n_spans * frame_spacing``, ``span_y = frame_span``), mode set
  ``1 .. 2*terms`` -- the same set as the audited axis-aligned layer, so that
  eccentric and oblique patches are representable.
* Load path: modal stiffness partition (``system_load_shares``) times exact
  tributary share; per-member force ``= p * A_patch * f_system * share``.
* Limit states: plate local bending, longitudinal stiffener beam-column, frame
  rigid-plastic collapse.  Each is **linear in the ice pressure** with the
  coexisting axial stress and hydrostatic pressure carried as the offset, so the
  ultimate pressure has a closed form per branch; bisection on the exact
  utilisation function is kept as an internal cross-check (gate ``O6b``).

Scope and limitations (explicit)
--------------------------------
* The patch is a rotated rectangle with a **uniform** pressure; non-uniform
  (pressure-area decay) patches are out of scope.
* Only the *loaded* area counts: a patch hanging over the panel edge is
  truncated, exactly as in the axis-aligned layer.
* Member load eccentricity along the member span is not amplified: each loaded
  member receives its tributary force and is checked as a central patch of the
  corresponding projected length (the same conservative convention as the
  axis-aligned layer).  Frame-line eccentricity is treated separately in
  ``grillage_plastic_collapse``.
* Small-deflection plate theory, elastic-perfectly-plastic steel, no
  post-buckling reserve, no stiffener tripping -- the limitations of the
  axis-aligned layer are inherited unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .grillage_ice_ultimate import (
    GrillageGeometry,
    beam_column_amplification_exact,
    beam_moment_centre_patch,
    frame_plastic_collapse_force,
    plate_patch_response,
    stiffener_axial_capacity,
    system_load_shares,
)

__all__ = [
    "ObliqueIcePatch",
    "clip_polygon_to_rect",
    "polygon_area",
    "polygon_centroid",
    "patch_fourier_coefficients",
    "oblique_patch_response",
    "tributary_areas",
    "oblique_utilisations",
    "oblique_ultimate_pressure",
    "orientation_sensitivity",
    "write_orientation_table",
]

Vertex = tuple[float, float]


# --------------------------------------------------------------------------- #
# polygon utilities (exact, dependency free)
# --------------------------------------------------------------------------- #
def polygon_area(poly: Sequence[Vertex]) -> float:
    """Signed area of a polygon (positive for counter-clockwise ordering)."""
    if len(poly) < 3:
        return 0.0
    acc = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        acc += x1 * y2 - x2 * y1
    return 0.5 * acc


def polygon_centroid(poly: Sequence[Vertex]) -> Vertex:
    """Centroid of a polygon (vertex mean for degenerate ones)."""
    a = polygon_area(poly)
    if abs(a) < 1e-30:
        xs = [p[0] for p in poly] or [0.0]
        ys = [p[1] for p in poly] or [0.0]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    cx = cy = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    return (cx / (6.0 * a), cy / (6.0 * a))


def _clip_half_plane(poly: Sequence[Vertex], axis: int, bound: float,
                     keep_greater: bool) -> list[Vertex]:
    """Sutherland-Hodgman clip of a polygon against one axis-aligned half-plane."""
    if not poly:
        return []
    out: list[Vertex] = []

    def inside(p: Vertex) -> bool:
        return p[axis] >= bound - 1e-15 if keep_greater else p[axis] <= bound + 1e-15

    n = len(poly)
    for i in range(n):
        cur = poly[i]
        nxt = poly[(i + 1) % n]
        cin = inside(cur)
        nin = inside(nxt)
        if cin:
            out.append(cur)
        if cin != nin:
            denom = nxt[axis] - cur[axis]
            if abs(denom) > 1e-30:
                t = (bound - cur[axis]) / denom
                if axis == 0:
                    out.append((bound, cur[1] + t * (nxt[1] - cur[1])))
                else:
                    out.append((cur[0] + t * (nxt[0] - cur[0]), bound))
    return out


def clip_polygon_to_rect(poly: Sequence[Vertex], x0: float, x1: float,
                         y0: float, y1: float) -> list[Vertex]:
    """Clip a polygon to ``[x0, x1] x [y0, y1]`` (``[]`` when disjoint)."""
    if x1 < x0 or y1 < y0:
        raise ValueError("clip rectangle must satisfy x1 >= x0 and y1 >= y0")
    out = list(poly)
    for axis, bound, keep in ((0, x0, True), (0, x1, False),
                              (1, y0, True), (1, y1, False)):
        out = _clip_half_plane(out, axis, bound, keep)
        if not out:
            return []
    return out


# 7-point degree-5 rule on the unit triangle (Dunavant): exact for quintics, so
# the cos*cos integrand of the Navier coefficients is integrated with a high
# order rule and the residual error is the cell subdivision alone.
_GAUSS7: tuple[tuple[float, float, float], ...] = (
    (1.0 / 3.0, 1.0 / 3.0, 0.225),
    (0.059715871789770, 0.470142064105115, 0.132394152788506),
    (0.470142064105115, 0.059715871789770, 0.132394152788506),
    (0.470142064105115, 0.470142064105115, 0.132394152788506),
    (0.797426985353087, 0.101286507323456, 0.125939180544827),
    (0.101286507323456, 0.797426985353087, 0.125939180544827),
    (0.101286507323456, 0.101286507323456, 0.125939180544827),
)


def _quadrature_points(poly: Sequence[Vertex], cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss points/weights covering ``poly`` by clipping a ``cells x cells`` grid.

    Cells fully inside the polygon are integrated with the same degree-5 rule
    without any clipping work (the vectorised fast path), so the cost of the
    exact Sutherland-Hodgman path is paid only on the O(cells) boundary cells.
    Weights sum to the polygon area to machine precision because each piece is
    triangulated exactly from its centroid.
    """
    if len(poly) < 3:
        return np.zeros((0, 2)), np.zeros(0)
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    nx = ny = max(1, int(cells))
    hx = max(x1 - x0, 1e-12) / nx
    hy = max(y1 - y0, 1e-12) / ny

    # reference rule on the unit square: 3 x 3 tensor Gauss, exact for quintics
    # in each direction, so a fully interior cell needs 9 points (not 42).
    g3 = (0.5 - 0.5 * math.sqrt(0.6), 0.5, 0.5 + 0.5 * math.sqrt(0.6))
    w3 = (5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0)
    ref_arr = np.array([(gx, gy, wx * wy)
                        for gx, wx in zip(g3, w3)
                        for gy, wy in zip(g3, w3)], dtype=float)

    pts: list[np.ndarray] = []
    wts: list[np.ndarray] = []
    partial: list[tuple[float, float, float, float]] = []
    for i in range(nx):
        cx0 = x0 + i * hx
        cx1 = cx0 + hx
        for j in range(ny):
            cy0 = y0 + j * hy
            cy1 = cy0 + hy
            corners = ((cx0, cy0), (cx1, cy0), (cx1, cy1), (cx0, cy1))
            if _all_inside(poly, corners):
                px = cx0 + ref_arr[:, 0] * hx
                py = cy0 + ref_arr[:, 1] * hy
                pts.append(np.column_stack((px, py)))
                wts.append(ref_arr[:, 2] * hx * hy)
            else:
                partial.append((cx0, cx1, cy0, cy1))
    for cx0, cx1, cy0, cy1 in partial:
        piece = clip_polygon_to_rect(poly, cx0, cx1, cy0, cy1)
        if len(piece) < 3:
            continue
        gx, gy = polygon_centroid(piece)
        m = len(piece)
        for k in range(m):
            a = piece[k]
            b = piece[(k + 1) % m]
            area = 0.5 * abs((a[0] - gx) * (b[1] - gy) - (b[0] - gx) * (a[1] - gy))
            if area <= 1e-18:
                continue
            for l1, l2, w in _GAUSS7:
                pts.append(np.array([[gx + l1 * (a[0] - gx) + l2 * (b[0] - gx),
                                      gy + l1 * (a[1] - gy) + l2 * (b[1] - gy)]]))
                wts.append(np.array([w * area]))
    if not pts:
        return np.zeros((0, 2)), np.zeros(0)
    return np.vstack(pts), np.concatenate(wts)


def _all_inside(poly: Sequence[Vertex], points: Sequence[Vertex]) -> bool:
    """True when every point is inside the convex CCW polygon ``poly``."""
    n = len(poly)
    for px, py in points:
        for i in range(n):
            ax, ay = poly[i]
            bx, by = poly[(i + 1) % n]
            if (bx - ax) * (py - ay) - (by - ay) * (px - ax) < -1e-12:
                return False
    return True


# --------------------------------------------------------------------------- #
# the patch
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ObliqueIcePatch:
    """Rectangular ice pressure patch of arbitrary centre and orientation.

    ``length`` runs along the patch's own ``u`` axis (``angle_deg = 0`` makes it
    parallel to the longitudinal stiffeners, i.e. the axis-aligned convention),
    ``width`` along ``v``.  ``pressure`` is the uniform patch pressure (Pa).
    """

    centre_x: float
    centre_y: float
    length: float
    width: float
    angle_deg: float = 0.0
    pressure: float = 0.0

    def validate(self) -> "ObliqueIcePatch":
        for name in ("centre_x", "centre_y", "length", "width",
                     "angle_deg", "pressure"):
            v = float(getattr(self, name))
            if not math.isfinite(v):
                raise ValueError(f"{name} must be finite, got {v!r}")
        if self.length <= 0.0 or self.width <= 0.0:
            raise ValueError("length and width must be positive")
        if self.pressure < 0.0:
            raise ValueError("pressure must be non-negative")
        return self

    @property
    def angle_rad(self) -> float:
        return math.radians(self.angle_deg)

    def polygon(self) -> list[Vertex]:
        """CCW vertices of the patch."""
        th = self.angle_rad
        ux, uy = math.cos(th), math.sin(th)
        vx, vy = -math.sin(th), math.cos(th)
        hu = 0.5 * self.length
        hv = 0.5 * self.width
        cx, cy = self.centre_x, self.centre_y
        return [
            (cx - hu * ux - hv * vx, cy - hu * uy - hv * vy),
            (cx + hu * ux - hv * vx, cy + hu * uy - hv * vy),
            (cx + hu * ux + hv * vx, cy + hu * uy + hv * vy),
            (cx - hu * ux + hv * vx, cy - hu * uy + hv * vy),
        ]

    def clipped(self, span_x: float, span_y: float) -> list[Vertex]:
        """The loaded polygon: the patch clipped to the representative panel."""
        return clip_polygon_to_rect(self.polygon(), 0.0, span_x, 0.0, span_y)

    def bbox(self) -> tuple[float, float, float, float]:
        poly = self.polygon()
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return (min(xs), max(xs), min(ys), max(ys))


# --------------------------------------------------------------------------- #
# Navier coefficients for a general load region
# --------------------------------------------------------------------------- #
def patch_fourier_coefficients(poly: Sequence[Vertex], span_x: float, span_y: float,
                               terms: int = 61, pressure: float = 1.0,
                               cells: int | None = None) -> np.ndarray:
    """Navier load coefficients ``q_mn`` for a uniform pressure on ``poly``.

    ``q_mn = (4/(a b)) * p * int_poly sin(m pi x/a) sin(n pi y/b) dA`` with
    ``m, n = 1 .. 2*terms`` -- the mode set of the axis-aligned layer.
    """
    if terms < 1:
        raise ValueError("terms must be >= 1")
    if not (span_x > 0.0 and span_y > 0.0):
        raise ValueError("spans must be positive")
    n_mode = 2 * int(terms)
    if len(poly) < 3:
        return np.zeros((n_mode, n_mode))
    if cells is None:
        ex = max(p[0] for p in poly) - min(p[0] for p in poly)
        ey = max(p[1] for p in poly) - min(p[1] for p in poly)
        # resolve the shortest Fourier wavelength (2*span/n_mode) with ~4 cells
        n_cells = int(max(8, min(64, math.ceil(4.0 * n_mode * max(ex / span_x,
                                                                 ey / span_y)))))
    else:
        n_cells = int(cells)
    pts, wts = _quadrature_points(poly, n_cells)
    if pts.shape[0] == 0:
        return np.zeros((n_mode, n_mode))
    m_vec = np.arange(1.0, n_mode + 1.0)
    sx = np.sin(np.pi * np.outer(m_vec, pts[:, 0]) / span_x)   # (n_mode, P)
    sy = np.sin(np.pi * np.outer(m_vec, pts[:, 1]) / span_y)
    weighted = sx * wts[None, :]
    return (4.0 * float(pressure) / (span_x * span_y)) * (weighted @ sy.T)


def _peak_moments(q_mn: np.ndarray, span_x: float, span_y: float,
                  d1: float, d2: float, d3: float, poisson: float,
                  sample: int) -> tuple[float, float, float, dict[str, np.ndarray] | None]:
    """Sampled peak ``(m_x, m_y, w)`` of the Navier field for load ``q_mn``.

    Splitting this out lets the local-bay branch reuse the exact same field
    evaluation as the global branch instead of a second implementation.
    """
    n_mode = q_mn.shape[0]
    m_vec = np.arange(1.0, n_mode + 1.0)
    m = m_vec[:, None]
    n = m_vec[None, :]
    kx = d1 * m ** 4 / span_x ** 4
    ky = d2 * n ** 4 / span_y ** 4
    kt = 2.0 * d3 * m ** 2 * n ** 2 / (span_x ** 2 * span_y ** 2)
    k_sum = kx + ky + kt
    amp = q_mn / (np.pi ** 4 * k_sum)
    xs = np.linspace(0.0, span_x, sample)
    ys = np.linspace(0.0, span_y, sample)
    sx = np.sin(np.pi * m_vec[:, None] * xs[None, :] / span_x)
    sy = np.sin(np.pi * m_vec[:, None] * ys[None, :] / span_y)
    ax = (m * math.pi / span_x) ** 2
    ay = (n * math.pi / span_y) ** 2
    d12 = poisson * math.sqrt(d1 * d2)
    w_field = np.einsum("mn,mi,nj->ij", amp, sx, sy)
    m_x = np.einsum("mn,mi,nj->ij", (d1 * ax + d12 * ay) * amp, sx, sy)
    m_y = np.einsum("mn,mi,nj->ij", (d2 * ay + d12 * ax) * amp, sx, sy)
    return (float(np.max(np.abs(m_x))), float(np.max(np.abs(m_y))),
            float(np.max(np.abs(w_field))),
            {"x": xs, "y": ys, "w": w_field, "m_x": m_x, "m_y": m_y})


def oblique_patch_response(*, span_x: float, span_y: float, d1: float, d2: float,
                           d3: float, patch: ObliqueIcePatch,
                           pressure: float | None = None,
                           terms: int = 61, sample: int = 61,
                           poisson: float = 0.3,
                           cells: int | None = None,
                           return_fields: bool = False) -> dict[str, object]:
    """Navier response of an SSSS orthotropic plate under an oblique patch load.

    Returns the same quantities as
    ``grillage_ice_ultimate.plate_patch_response`` (``w_max``, ``m_x_max``,
    ``m_y_max``, the three modal load shares, ``load_area``, ``total_force``)
    plus ``clipped_area_fraction``.  Peaks are sampled on a ``sample x sample``
    grid over the whole panel: for an oblique patch the peak is not at the plate
    centre, so ``sample`` convergence is gated (``O4b``) rather than assumed.
    """
    for name, value in (("span_x", span_x), ("span_y", span_y),
                        ("d1", d1), ("d2", d2), ("d3", d3)):
        if not (float(value) > 0.0 and math.isfinite(float(value))):
            raise ValueError(f"{name} must be a finite positive number")
    if sample < 3:
        raise ValueError("sample must be >= 3")
    patch.validate()
    p = float(patch.pressure if pressure is None else pressure)
    if not math.isfinite(p):
        raise ValueError("pressure must be finite")

    clipped = patch.clipped(span_x, span_y)
    area = abs(polygon_area(clipped))
    full_area = patch.length * patch.width
    empty: dict[str, object] = {
        "w_max": 0.0, "m_x_max": 0.0, "m_y_max": 0.0,
        "load_share_x": 0.0, "load_share_y": 0.0, "load_share_torsion": 0.0,
        "load_area": area, "total_force": p * area,
        "clipped_area_fraction": (area / full_area) if full_area > 0.0 else 0.0,
    }
    if area <= 1e-15 or p == 0.0:
        return empty

    q_mn = patch_fourier_coefficients(clipped, span_x, span_y, terms=terms,
                                      pressure=p, cells=cells)
    n_mode = q_mn.shape[0]
    m_vec = np.arange(1.0, n_mode + 1.0)
    m = m_vec[:, None]
    n = m_vec[None, :]
    kx = d1 * m ** 4 / span_x ** 4
    ky = d2 * n ** 4 / span_y ** 4
    kt = 2.0 * d3 * m ** 2 * n ** 2 / (span_x ** 2 * span_y ** 2)
    k_sum = kx + ky + kt
    amp = q_mn / (np.pi ** 4 * k_sum)

    w_modal = q_mn ** 2 / k_sum
    w_total = float(np.sum(w_modal))
    if w_total <= 0.0:  # pragma: no cover - defensive
        return empty
    f_x = float(np.sum(w_modal * (kx / k_sum))) / w_total
    f_y = float(np.sum(w_modal * (ky / k_sum))) / w_total
    f_t = float(np.sum(w_modal * (kt / k_sum))) / w_total

    xs = np.linspace(0.0, span_x, sample)
    ys = np.linspace(0.0, span_y, sample)
    sx = np.sin(np.pi * m_vec[:, None] * xs[None, :] / span_x)
    sy = np.sin(np.pi * m_vec[:, None] * ys[None, :] / span_y)
    ax = (m * math.pi / span_x) ** 2
    ay = (n * math.pi / span_y) ** 2
    d12 = poisson * math.sqrt(d1 * d2)
    w_field = np.einsum("mn,mi,nj->ij", amp, sx, sy)
    m_x = np.einsum("mn,mi,nj->ij", (d1 * ax + d12 * ay) * amp, sx, sy)
    m_y = np.einsum("mn,mi,nj->ij", (d2 * ay + d12 * ax) * amp, sx, sy)

    out: dict[str, object] = {
        "w_max": float(np.max(np.abs(w_field))),
        "m_x_max": float(np.max(np.abs(m_x))),
        "m_y_max": float(np.max(np.abs(m_y))),
        "load_share_x": f_x,
        "load_share_y": f_y,
        "load_share_torsion": f_t,
        "load_area": area,
        "total_force": p * area,
        "clipped_area_fraction": (area / full_area) if full_area > 0.0 else 0.0,
    }
    if return_fields:
        out["fields"] = {"x": xs, "y": ys, "w": w_field, "m_x": m_x, "m_y": m_y}
    return out


# --------------------------------------------------------------------------- #
# member load path: exact tributary clipping
# --------------------------------------------------------------------------- #
def _partition_edges(n_members: int, spacing: float, span: float) -> list[tuple[float, float]]:
    """Tributary strips ``[lo, hi]`` partitioning ``[0, span]`` over the members."""
    half = 0.5 * spacing
    edges: list[tuple[float, float]] = []
    for i in range(n_members):
        centre = i * spacing
        edges.append((max(0.0, centre - half), min(span, centre + half)))
    return edges


def tributary_areas(geom: GrillageGeometry, patch: ObliqueIcePatch,
                    span_x: float, span_y: float) -> dict[str, object]:
    """Exact patch area falling in each stiffener / frame tributary strip.

    Returns the clipped areas, the normalised shares (summing to 1 over the
    touched members), and the patch extent along each member inside its own strip
    -- the "central patch length" used by the member moment and collapse
    formulas.
    """
    geom.validate()
    clipped = patch.clipped(span_x, span_y)
    total = abs(polygon_area(clipped))
    stiff_edges = _partition_edges(geom.n_stiffeners, geom.bay_spacing, span_y)
    frame_edges = _partition_edges(geom.n_frames, geom.frame_spacing, span_x)

    stiff_rows: list[dict[str, float]] = []
    for i, (lo, hi) in enumerate(stiff_edges):
        piece = clip_polygon_to_rect(clipped, 0.0, span_x, lo, hi)
        area = abs(polygon_area(piece))
        extent = (max(p[0] for p in piece) - min(p[0] for p in piece)
                  if len(piece) >= 3 else 0.0)
        stiff_rows.append({"index": float(i), "lo": lo, "hi": hi, "area": area,
                           "share": (area / total) if total > 0.0 else 0.0,
                           "extent_along_member": extent})

    frame_rows: list[dict[str, float]] = []
    for j, (lo, hi) in enumerate(frame_edges):
        piece = clip_polygon_to_rect(clipped, lo, hi, 0.0, span_y)
        area = abs(polygon_area(piece))
        extent = (max(p[1] for p in piece) - min(p[1] for p in piece)
                  if len(piece) >= 3 else 0.0)
        frame_rows.append({"index": float(j), "lo": lo, "hi": hi, "area": area,
                           "share": (area / total) if total > 0.0 else 0.0,
                           "extent_along_member": extent})

    return {
        "patch_area": total,
        "stiffener": stiff_rows,
        "frame": frame_rows,
        "stiffener_touched": int(sum(1 for r in stiff_rows if r["area"] > 0.0)),
        "frame_touched": int(sum(1 for r in frame_rows if r["area"] > 0.0)),
        "share_sum_stiffener": float(sum(r["share"] for r in stiff_rows)),
        "share_sum_frame": float(sum(r["share"] for r in frame_rows)),
    }


# --------------------------------------------------------------------------- #
# bay-level plate branch and member load path
# --------------------------------------------------------------------------- #
def _plate_capacity(geom: GrillageGeometry, axial_stress: float) -> float:
    """Plate strip plastic moment reduced by the coexisting membrane compression."""
    mp = geom.yield_stress * geom.plate_thickness ** 2 / 4.0
    ratio = min(axial_stress / geom.yield_stress, 1.0)
    return mp * math.sqrt(max(1.0 - ratio ** 2, 0.0))


def bay_load_polygon(patch: ObliqueIcePatch, geom: GrillageGeometry,
                     span_x: float, span_y: float,
                     mode: str = "centred") -> tuple[list[Vertex], float, float]:
    """Patch polygon as seen by the most loaded plate bay, in bay coordinates.

    The plate limit state is a *local* quantity: the axis-aligned layer evaluates
    it on one bay (``frame_spacing x bay_spacing``) with the patch clamped to the
    bay and placed centrally (``grillage_ice_ultimate`` "local plate bending"
    branch).  Reproduced here for a general polygon:

    ``mode="centred"`` (default, the audited convention) -- a bay-sized window is
    placed on the loaded centroid of the patch, the patch is clipped to it, and
    the window is then mapped onto the bay: a patch that fits inside one bay
    therefore sits centrally in the bay, exactly as in the axis-aligned layer.

    ``mode="clip"`` -- the patch is clipped to the *actual* bay that contains its
    loaded centroid, so intra-bay eccentricity is retained.

    Returns ``(polygon_in_bay_coordinates, bay_span_x, bay_span_y)``.
    """
    if mode not in ("centred", "clip"):
        raise ValueError("plate_bay_mode must be 'centred' or 'clip'")
    a, b = geom.frame_spacing, geom.bay_spacing
    clipped = patch.clipped(span_x, span_y)
    if len(clipped) < 3:
        return [], a, b
    cx, cy = polygon_centroid(clipped)
    if mode == "centred":
        piece = clip_polygon_to_rect(clipped, cx - 0.5 * a, cx + 0.5 * a,
                                     cy - 0.5 * b, cy + 0.5 * b)
        dx, dy = 0.5 * a - cx, 0.5 * b - cy
    else:
        i = min(max(int(cx // a), 0), max(geom.n_spans - 1, 0))
        j = min(max(int(cy // b), 0), max(geom.n_bays - 1, 0))
        piece = clip_polygon_to_rect(clipped, i * a, (i + 1) * a, j * b, (j + 1) * b)
        dx, dy = -i * a, -j * b
    moved = [(x + dx, y + dy) for x, y in piece]
    return moved, a, b


def _partition_edges(n_members: int, spacing: float, span: float) -> list[tuple[float, float]]:
    """Tributary strips ``[lo, hi]`` partitioning ``[0, span]`` over the members."""
    half = 0.5 * spacing
    edges: list[tuple[float, float]] = []
    for i in range(n_members):
        centre = i * spacing
        edges.append((max(0.0, centre - half), min(span, centre + half)))
    return edges


def tributary_areas(geom: GrillageGeometry, patch: ObliqueIcePatch,
                    span_x: float, span_y: float) -> dict[str, object]:
    """Exact patch area falling in each stiffener / frame tributary strip.

    Returns the clipped areas, the normalised shares (summing to 1 over the
    touched members), and the patch extent along each member inside its own strip
    -- the "central patch length" used by the member moment and collapse formulas.
    """
    geom.validate()
    clipped = patch.clipped(span_x, span_y)
    total = abs(polygon_area(clipped))
    stiff_edges = _partition_edges(geom.n_stiffeners, geom.bay_spacing, span_y)
    frame_edges = _partition_edges(geom.n_frames, geom.frame_spacing, span_x)

    stiff_rows: list[dict[str, float]] = []
    for i, (lo, hi) in enumerate(stiff_edges):
        piece = clip_polygon_to_rect(clipped, 0.0, span_x, lo, hi)
        area = abs(polygon_area(piece))
        extent = (max(p[0] for p in piece) - min(p[0] for p in piece)
                  if len(piece) >= 3 else 0.0)
        stiff_rows.append({"index": float(i), "lo": lo, "hi": hi, "area": area,
                           "share": (area / total) if total > 0.0 else 0.0,
                           "extent_along_member": extent})

    frame_rows: list[dict[str, float]] = []
    for j, (lo, hi) in enumerate(frame_edges):
        piece = clip_polygon_to_rect(clipped, lo, hi, 0.0, span_y)
        area = abs(polygon_area(piece))
        extent = (max(p[1] for p in piece) - min(p[1] for p in piece)
                  if len(piece) >= 3 else 0.0)
        frame_rows.append({"index": float(j), "lo": lo, "hi": hi, "area": area,
                           "share": (area / total) if total > 0.0 else 0.0,
                           "extent_along_member": extent})

    return {
        "patch_area": total,
        "stiffener": stiff_rows,
        "frame": frame_rows,
        "stiffener_touched": int(sum(1 for r in stiff_rows if r["area"] > 0.0)),
        "frame_touched": int(sum(1 for r in frame_rows if r["area"] > 0.0)),
        "share_sum_stiffener": float(sum(r["share"] for r in stiff_rows)),
        "share_sum_frame": float(sum(r["share"] for r in frame_rows)),
    }


def _patch_projected_extents(patch: ObliqueIcePatch, span_x: float,
                             span_y: float) -> tuple[float, float, float, float]:
    """Axis-aligned extents of the clipped patch: ``(x_extent, y_extent, cx, cy)``."""
    clipped = patch.clipped(span_x, span_y)
    if len(clipped) < 3:
        return 0.0, 0.0, 0.0, 0.0
    xs = [p[0] for p in clipped]
    ys = [p[1] for p in clipped]
    return (max(xs) - min(xs), max(ys) - min(ys),
            0.5 * (max(xs) + min(xs)), 0.5 * (max(ys) + min(ys)))


def oblique_utilisations(geom: GrillageGeometry, patch: ObliqueIcePatch, *,
                         axial_stress: float = 0.0,
                         hydrostatic_pressure: float = 0.0,
                         ice_pressure: float | None = None,
                         terms: int = 61, sample: int = 61,
                         poisson: float | None = None,
                         plate_bay_mode: str = "centred",
                         member_allocation: str = "tributary") -> dict[str, object]:
    """Utilisations of the three limit states for an oblique patch load case.

    The capacity laws and the local-bay plate idealisation are taken from
    ``grillage_ice_ultimate`` (single source of truth); only the *load path* is
    generalised.  ``ice_pressure = 0`` is allowed and returns the coexisting-load
    utilisations, which is what the linear decomposition of the ultimate pressure
    needs.

    ``member_allocation``
        ``"tributary"`` (default) -- each touched stiffener/frame receives the
        share of the patch force given by its clipped tributary area.
        ``"base_convention"`` -- the audited axis-aligned rule: the load is spread
        over ``max(2, round(ly/bay)+1)`` stiffeners and over exactly two frames at
        50 % each, with the patch extents clamped to one bay/frame space.  It
        exists so that a theta = 0 case can be checked against the audited
        numbers (gate O6) instead of being quietly re-based.
    """
    geom.validate()
    patch.validate()
    if not math.isfinite(axial_stress) or axial_stress < 0.0:
        raise ValueError("axial_stress must be finite and >= 0")
    if axial_stress >= geom.yield_stress:
        raise ValueError("axial_stress must stay below the yield stress")
    if not math.isfinite(hydrostatic_pressure) or hydrostatic_pressure < 0.0:
        raise ValueError("hydrostatic_pressure must be finite and >= 0")
    p_ice = float(patch.pressure if ice_pressure is None else ice_pressure)
    if not math.isfinite(p_ice) or p_ice < 0.0:
        raise ValueError("ice_pressure must be finite and >= 0")
    if member_allocation not in ("tributary", "base_convention"):
        raise ValueError("member_allocation must be 'tributary' or 'base_convention'")
    if poisson is None:
        poisson = geom.poisson

    span_x = geom.n_spans * geom.frame_spacing
    span_y = geom.frame_span
    dx, dy, d3 = geom.plate_rigidities()
    d_plate = geom.plate_rigidity

    if member_allocation == "base_convention":
        # The audited axis-aligned layer clamps the patch to one frame space by
        # one bay and centres it there -- including for the modal load share and
        # the total patch force.  Reproduce that geometry exactly (the audited
        # layer has no orientation, so the projected extents are used).
        clipped = patch.clipped(span_x, span_y)
        if len(clipped) >= 3:
            xs = [q[0] for q in clipped]
            ys = [q[1] for q in clipped]
            lx_c = min(max(xs) - min(xs), geom.frame_spacing)
            ly_c = min(max(ys) - min(ys), geom.bay_spacing)
        else:
            lx_c = ly_c = 0.0
        eff = ObliqueIcePatch(centre_x=0.5 * span_x, centre_y=0.5 * span_y,
                              length=max(lx_c, 1e-9), width=max(ly_c, 1e-9),
                              angle_deg=0.0, pressure=patch.pressure).validate()
    else:
        eff = patch

    # global (grillage) response: modal load share and total patch force
    patch_resp = oblique_patch_response(
        span_x=span_x, span_y=span_y, d1=dx, d2=dy, d3=d3, patch=eff,
        pressure=p_ice, terms=terms, sample=sample, poisson=poisson)
    force_ice = float(patch_resp["total_force"])
    if p_ice > 0.0 and force_ice > 0.0:
        f_x_ice, f_y_ice = system_load_shares(patch_resp)
    else:
        f_x_ice = f_y_ice = 0.0

    # (a) plate local bending in the most loaded bay (isotropic plate rigidity,
    #     patch clamped to the bay -- the audited "local plate bending" branch)
    plate_cap = _plate_capacity(geom, axial_stress)
    m_plate = 0.0
    if p_ice > 0.0 and len(eff.clipped(span_x, span_y)) >= 3:
        bay_poly, a, b = bay_load_polygon(eff, geom, span_x, span_y,
                                          mode=plate_bay_mode)
        if len(bay_poly) >= 3:
            q_bay = patch_fourier_coefficients(bay_poly, a, b, terms=terms,
                                               pressure=p_ice)
            m_bay = _peak_moments(q_bay, a, b, d_plate, d_plate, d_plate,
                                  poisson, max(sample, 41))
            m_plate += max(m_bay[0], m_bay[1])
    f_x_h = f_y_h = 0.0
    force_h = 0.0
    if hydrostatic_pressure > 0.0:
        a, b = geom.frame_spacing, geom.bay_spacing
        # local bay field -> the plate bending moment (audited "local plate
        # bending" branch: isotropic rigidity, one bay)
        bay_hydro = plate_patch_response(
            span_x=a, span_y=b, d1=d_plate, d2=d_plate, d3=d_plate,
            pressure=hydrostatic_pressure, patch=(0.0, 1.0, 0.0, 1.0),
            terms=terms, poisson=poisson, sample=max(sample, 41))
        m_plate += max(float(bay_hydro["m_x_max"]), float(bay_hydro["m_y_max"]))
        # global panel field -> the modal load share of the uniform pressure
        global_hydro = plate_patch_response(
            span_x=span_x, span_y=span_y, d1=dx, d2=dy, d3=d3,
            pressure=hydrostatic_pressure, patch=(0.0, 1.0, 0.0, 1.0),
            terms=terms, poisson=poisson)
        f_x_h, f_y_h = system_load_shares(global_hydro)
        force_h = hydrostatic_pressure * geom.panel_area
    u_plate = m_plate / plate_cap if plate_cap > 0.0 else math.inf

    # (b) longitudinal stiffener beam-column
    cap = stiffener_axial_capacity(geom)
    sigma_l = float(cap["limiting_stress"])
    area_e = float(cap["area_effective"])
    section = geom.with_plate(geom.stiffener, float(cap["effective_width"]))
    axial_force = axial_stress * area_e
    pe = math.pi ** 2 * geom.young * section.inertia / geom.frame_spacing ** 2
    if 0.0 < axial_force < pe:
        beta_m = beam_column_amplification_exact(axial_force, pe)
    elif axial_force >= pe:
        beta_m = 10.0
    else:
        beta_m = 1.0

    trib = tributary_areas(geom, eff, span_x, span_y)
    hydro_per_stiff = force_h * f_x_h / geom.n_stiffeners
    m_h_stiff = (beam_moment_centre_patch(hydro_per_stiff, geom.frame_spacing,
                                          geom.frame_spacing)
                 if hydro_per_stiff > 0.0 else 0.0)
    u_stiff = 0.0
    worst_stiff = -1
    if member_allocation == "base_convention":
        x_ext, y_ext, cx_p, cy_p = _patch_projected_extents(eff, span_x, span_y)
        lx_c = min(x_ext, geom.frame_spacing)
        ly_c = min(y_ext, geom.bay_spacing)
        n_overlap = min(max(2, int(round(ly_c / geom.bay_spacing)) + 1),
                        geom.n_stiffeners)
        if force_ice > 0.0:
            per_ice = force_ice * f_x_ice / n_overlap
            m_ice = beam_moment_centre_patch(per_ice, geom.frame_spacing, lx_c)
            u_stiff = ((axial_stress
                        + beta_m * (m_ice + m_h_stiff) / section.elastic_modulus_section)
                       / sigma_l)
        else:
            u_stiff = ((axial_stress
                        + beta_m * m_h_stiff / section.elastic_modulus_section)
                       / sigma_l) if m_h_stiff > 0.0 else axial_stress / sigma_l
    else:
        for row in trib["stiffener"]:
            share = float(row["share"])
            if share <= 0.0:
                continue
            per_ice = force_ice * f_x_ice * share
            m_ice = beam_moment_centre_patch(per_ice, geom.frame_spacing,
                                             float(row["extent_along_member"]))
            u = ((axial_stress
                  + beta_m * (m_ice + m_h_stiff) / section.elastic_modulus_section)
                 / sigma_l)
            if u > u_stiff:
                u_stiff, worst_stiff = u, int(row["index"])

    # (c) transverse frame rigid-plastic collapse
    mp = geom.yield_stress * geom.frame_section.plastic_modulus
    # hydrostatic pressure covers the whole panel: the tributary rule spreads it
    # over every frame, the audited convention over the two bounding frames
    n_frame_share = 2.0 if member_allocation == "base_convention" else float(geom.n_frames)
    per_frame_h = force_h * f_y_h / n_frame_share
    f_u_uniform = frame_plastic_collapse_force(mp, span_y, span_y)
    u_frame = 0.0
    worst_frame = -1
    if member_allocation == "base_convention":
        x_ext, y_ext, cx_p, cy_p = _patch_projected_extents(eff, span_x, span_y)
        ly_c = min(y_ext, geom.bay_spacing)
        if force_ice > 0.0:
            f_u_patch = frame_plastic_collapse_force(mp, span_y, ly_c)
            u_frame = 0.5 * force_ice * f_y_ice / f_u_patch
        if per_frame_h > 0.0:
            u_frame += per_frame_h / f_u_uniform
    else:
        for row in trib["frame"]:
            share = float(row["share"])
            if share <= 0.0:
                continue
            per_ice = force_ice * f_y_ice * share
            f_u_patch = frame_plastic_collapse_force(
                mp, span_y, float(row["extent_along_member"]))
            u = per_ice / f_u_patch
            if per_frame_h > 0.0:
                u += per_frame_h / f_u_uniform
            if u > u_frame:
                u_frame, worst_frame = u, int(row["index"])

    return {
        "plate_local_bending": u_plate,
        "longitudinal_stiffener": u_stiff,
        "transverse_frame": u_frame,
        "m_plate": m_plate,
        "m_plate_capacity": plate_cap,
        "stiffener_limiting_stress": sigma_l,
        "moment_amplification": beta_m,
        "frame_plastic_moment": mp,
        "patch_area": float(trib["patch_area"]),
        "patch_area_fraction": float(patch_resp["clipped_area_fraction"]),
        "force_ice": force_ice,
        "force_hydrostatic": force_h,
        "load_share_x_system": f_x_ice,
        "load_share_y_system": f_y_ice,
        "stiffener_touched": float(trib["stiffener_touched"]),
        "frame_touched": float(trib["frame_touched"]),
        "worst_stiffener": float(worst_stiff),
        "worst_frame": float(worst_frame),
        "plate_bay_mode": plate_bay_mode,
        "member_allocation": member_allocation,
        "tributary": trib,
    }


_MEMBER_KEYS = ("plate_local_bending", "longitudinal_stiffener", "transverse_frame")


def _governing(u: Mapping[str, float]) -> str:
    return max(_MEMBER_KEYS, key=lambda k: (float(u[k]), k))


def oblique_ultimate_pressure(geom: GrillageGeometry, patch: ObliqueIcePatch, *,
                              axial_stress: float = 0.0,
                              hydrostatic_pressure: float = 0.0,
                              design_pressure: float | None = None,
                              terms: int = 61, sample: int = 61,
                              verify_bisection: bool = False,
                              bisection_tol: float = 1e-6,
                              poisson: float | None = None,
                              plate_bay_mode: str = "centred",
                              member_allocation: str = "tributary") -> dict[str, object]:
    """Largest oblique patch pressure keeping every member utilisation ``<= 1``.

    Every utilisation branch is linear in the ice patch pressure (the coexisting
    axial stress and hydrostatic pressure form the offset), so the ultimate
    pressure is the minimum over the three limit states of the closed-form root
    ``(1 - u0) / du``.  ``verify_bisection`` re-solves the exact utilisation
    function and records the difference (gate ``O6b``).
    """
    patch.validate()
    base = oblique_utilisations(geom, patch, axial_stress=axial_stress,
                                hydrostatic_pressure=hydrostatic_pressure,
                                ice_pressure=0.0, terms=terms, sample=sample,
                                poisson=poisson, plate_bay_mode=plate_bay_mode,
                                member_allocation=member_allocation)
    u0 = {k: float(base[k]) for k in _MEMBER_KEYS}
    if max(u0.values()) >= 1.0:
        return {
            "ice_pressure_ultimate": 0.0,
            "governing_member": _governing(u0),
            "utilisation_without_ice": max(u0.values()),
            "base_utilisations": u0,
            "utilisations_at_limit": u0,
            "closed_form_roots": {k: 0.0 for k in _MEMBER_KEYS},
            "margin_vs_design": 0.0 if design_pressure else None,
            "patch_area": float(base["patch_area"]),
            "patch_area_fraction": float(base["patch_area_fraction"]),
            "converged": True,
        }

    p_ref = 1.0e6
    ref = oblique_utilisations(geom, patch, axial_stress=axial_stress,
                               hydrostatic_pressure=hydrostatic_pressure,
                               ice_pressure=p_ref, terms=terms, sample=sample,
                               poisson=poisson, plate_bay_mode=plate_bay_mode,
                               member_allocation=member_allocation)
    slopes = {k: (float(ref[k]) - u0[k]) / p_ref for k in _MEMBER_KEYS}
    roots = {k: ((1.0 - u0[k]) / slopes[k] if slopes[k] > 0.0 else math.inf)
             for k in _MEMBER_KEYS}
    governing = min(_MEMBER_KEYS, key=lambda k: (roots[k], k))
    p_ult = roots[governing]

    result: dict[str, object] = {
        "ice_pressure_ultimate": p_ult,
        "governing_member": governing,
        "utilisation_without_ice": max(u0.values()),
        "base_utilisations": u0,
        "slopes_per_Pa": slopes,
        "closed_form_roots": roots,
        "patch_area": float(base["patch_area"]),
        "patch_area_fraction": float(base["patch_area_fraction"]),
        "converged": math.isfinite(p_ult),
        "margin_vs_design": (p_ult / design_pressure) if design_pressure else None,
    }
    if math.isfinite(p_ult):
        at_limit = oblique_utilisations(geom, patch, axial_stress=axial_stress,
                                        hydrostatic_pressure=hydrostatic_pressure,
                                        ice_pressure=p_ult, terms=terms,
                                        sample=sample, poisson=poisson,
                                        plate_bay_mode=plate_bay_mode,
                                        member_allocation=member_allocation)
        result["utilisations_at_limit"] = {k: float(at_limit[k]) for k in _MEMBER_KEYS}
    else:  # pragma: no cover - no branch responds to the ice pressure
        result["utilisations_at_limit"] = u0

    if verify_bisection and math.isfinite(p_ult):
        lo, hi = 0.0, max(p_ult, 1.0) * 4.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            u_mid = oblique_utilisations(geom, patch, axial_stress=axial_stress,
                                         hydrostatic_pressure=hydrostatic_pressure,
                                         ice_pressure=mid, terms=terms,
                                         sample=sample, poisson=poisson,
                                         plate_bay_mode=plate_bay_mode,
                                         member_allocation=member_allocation)
            if max(float(u_mid[k]) for k in _MEMBER_KEYS) < 1.0:
                lo = mid
            else:
                hi = mid
            if hi - lo <= bisection_tol * max(hi, 1.0):
                break
        bisect = 0.5 * (lo + hi)
        result["bisection_pressure"] = bisect
        result["bisection_relative_difference"] = (abs(bisect - p_ult) / p_ult
                                                   if p_ult > 0.0 else 0.0)
    return result


# --------------------------------------------------------------------------- #
# orientation sensitivity: the review question
# --------------------------------------------------------------------------- #
def orientation_sensitivity(geom: GrillageGeometry, *,
                            patch_length: float,
                            patch_width: float,
                            angles_deg: Sequence[float] = (0.0, 15.0, 30.0, 45.0,
                                                           60.0, 75.0, 90.0),
                            axial_stress: float = 0.0,
                            hydrostatic_pressure: float = 0.0,
                            design_pressure: float | None = None,
                            centre: tuple[float, float] | None = None,
                            terms: int = 61, sample: int = 61,
                            plate_bay_mode: str = "centred",
                            member_allocation: str = "tributary") -> dict[str, object]:
    """Ultimate ice pressure versus patch orientation at fixed patch size.

    The patch centre defaults to the centre of the representative panel.  The
    clipped area is recorded per row, so a patch that overhangs the panel edge
    cannot be mistaken for an equal-area comparison.
    """
    span_x = geom.n_spans * geom.frame_spacing
    span_y = geom.frame_span
    cx, cy = centre if centre is not None else (0.5 * span_x, 0.5 * span_y)
    rows: list[dict[str, object]] = []
    for angle in angles_deg:
        patch = ObliqueIcePatch(centre_x=cx, centre_y=cy, length=patch_length,
                                width=patch_width, angle_deg=float(angle),
                                pressure=0.0).validate()
        res = oblique_ultimate_pressure(
            geom, patch, axial_stress=axial_stress,
            hydrostatic_pressure=hydrostatic_pressure,
            design_pressure=design_pressure, terms=terms, sample=sample,
            plate_bay_mode=plate_bay_mode,
            member_allocation=member_allocation)
        rows.append({
            "angle_deg": float(angle),
            "patch_length_m": patch_length,
            "patch_width_m": patch_width,
            "patch_area_m2": float(res["patch_area"]),
            "patch_area_fraction": float(res["patch_area_fraction"]),
            "ice_pressure_ultimate_MPa": (float(res["ice_pressure_ultimate"]) / 1e6
                                          if math.isfinite(float(res["ice_pressure_ultimate"]))
                                          else None),
            "governing_member": res["governing_member"],
            "margin_vs_design": res.get("margin_vs_design"),
            "utilisation_plate": float(res["utilisations_at_limit"]["plate_local_bending"]),
            "utilisation_stiffener": float(res["utilisations_at_limit"]["longitudinal_stiffener"]),
            "utilisation_frame": float(res["utilisations_at_limit"]["transverse_frame"]),
            "converged": bool(res["converged"]),
        })
    finite = [r["ice_pressure_ultimate_MPa"] for r in rows
              if r["ice_pressure_ultimate_MPa"] is not None]
    reference = rows[0]["ice_pressure_ultimate_MPa"] if rows else None
    spread = ((max(finite) - min(finite)) / reference
              if finite and reference else None)
    worst = (min(rows, key=lambda r: r["ice_pressure_ultimate_MPa"])
             if finite else None)
    return {
        "geometry": {
            "frame_spacing_m": geom.frame_spacing,
            "bay_spacing_m": geom.bay_spacing,
            "n_bays": int(geom.n_bays),
            "n_spans": int(geom.n_spans),
            "plate_thickness_mm": geom.plate_thickness * 1e3,
            "yield_stress_MPa": geom.yield_stress / 1e6,
            "panel_span_x_m": span_x,
            "panel_span_y_m": span_y,
        },
        "axial_stress_MPa": axial_stress / 1e6,
        "hydrostatic_pressure_MPa": hydrostatic_pressure / 1e6,
        "design_pressure_MPa": (design_pressure / 1e6) if design_pressure else None,
        "plate_bay_mode": plate_bay_mode,
        "member_allocation": member_allocation,
        "rows": rows,
        "spread_relative_to_angle_0": spread,
        "worst_angle_deg": (float(worst["angle_deg"]) if worst else None),
        "worst_pressure_MPa": (worst["ice_pressure_ultimate_MPa"] if worst else None),
    }


def write_orientation_table(table: Mapping[str, object], path: str | Path) -> Path:
    """Write the orientation table as JSON (machine-readable acceptance artifact)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "tensorfem.grillage_oblique_patch.orientation/v1",
               **dict(table)}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out