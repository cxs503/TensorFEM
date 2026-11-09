"""Weld-toe stress concentration and detail fatigue assessment (SI units).

This module owns the TensorFEM-side *detail* fatigue chain for welded joints:

1. :class:`WeldDetail` -- parametric welded joint (transverse fillet T-joint,
   butt weld with reinforcement, or right-angle corner / L joint) with an explicit
   weld-toe radius ``r`` in the physical range 0.5-2 mm, weld leg (or
   reinforcement), flank angle and attachment height.
2. :func:`nii_stress_concentration` / :func:`maddox_stress_concentration` --
   independent closed-form ``K_t`` anchors against which the finite-element value
   is judged (Nii: ``K_t = 1 + 0.27 sqrt(t/r) sqrt(h/t)`` for transverse fillet
   T-joints; Maddox: ``K_t = 1 + 0.15 sqrt(t/r)`` for butt joints).
3. :func:`solve_weld_detail` -- a body-fitted quadratic (Q8) plane model of the
   joint cross-section.  The mesh is built from structured blocks whose rows
   follow the *free surface* (constant depth below the weld profile), so the toe
   arc of radius ``r`` is resolved without sliver elements.  The model returns the
   peak toe stress, the axial surface stress path along the plate and both
   hot-spot extrapolations.
4. :func:`hot_spot_stress` -- IIW-style surface extrapolation (linear
   ``0.4t / 0.9t``, quadratic ``0.4t / 0.9t / 1.4t``) of the structural stress.
5. :class:`TwoSlopeSNCurve` / :class:`SNEnvironment` -- DNV-RP-C203 style
   double-slope S-N curves ``log10 N = log10 a - m log10 ds`` for air, seawater
   with cathodic protection and seawater free corrosion.
6. :func:`fatigue_life` / :func:`detail_fatigue_assessment` -- stress range to
   cycles conversion plus the mesh-convergence ledger used by the benchmarks.

Scope and limitations.  Everything here is a 2-D plane idealisation of a weld
cross-section: no 3-D weld start/stop ends, no weld profile scatter, no residual
stress, no misalignment, no multiple-crack interaction.  The finite-element ``K_t``
is a *notch* stress concentration relative to the nominal membrane stress in the
plate, and it is compared with the empirical anchors only inside the geometry
family for which those anchors were fitted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from ..quadratic_plane import (
    GAUSS_POINTS_3X3,
    GAUSS_WEIGHTS_3X3,
    elasticity_matrix,
    q8_element_dofs,
    q8_strain_matrix,
)
from ..sparse_core import assemble_coo
from ..sparse_direct import factorize_sparse

__all__ = [
    "WeldDetail",
    "WeldToeSolution",
    "HotSpotResult",
    "nii_stress_concentration",
    "maddox_stress_concentration",
    "anchor_stress_concentration",
    "solve_weld_detail",
    "hot_spot_stress",
    "mesh_convergence",
    "TwoSlopeSNCurve",
    "SNEnvironment",
    "ENVIRONMENTS",
    "fatigue_life",
    "detail_fatigue_assessment",
    "YOUNG_STEEL_PA",
    "POISSON_STEEL",
    "LINEAR_STATIONS",
    "QUADRATIC_STATIONS",
]

#: Supported load cases of a :class:`WeldDetail` (see the class docstring).
ALLOWED_LOAD_CASES: frozenset[str] = frozenset(
    {"carried", "plate_axial", "attachment_supported"}
)

#: Structural steel elastic constants of the joint models.
YOUNG_STEEL_PA = 210.0e9
POISSON_STEEL = 0.3

#: IIW surface extrapolation stations, in units of the plate thickness.
LINEAR_STATIONS: tuple[float, ...] = (0.4, 0.9)
QUADRATIC_STATIONS: tuple[float, ...] = (0.4, 0.9, 1.4)


# --------------------------------------------------------------------------- #
# 1. Geometry and independent closed-form anchors
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WeldDetail:
    """Parametric planar weld detail (lengths in m, angles in degrees).

    Parameters
    ----------
    joint_type:
        ``"t_fillet"`` (transverse fillet T-joint, symmetric half model),
        ``"butt"`` (butt weld with reinforcement, symmetric half model) or
        ``"corner"`` (right-angle L joint welded on the inside, full section).
    plate_thickness_m:
        Thickness ``t`` of the plate that carries the nominal stress.
    toe_radius_m:
        Weld-toe radius ``r``; the physical range of interest is 0.5-2 mm.
    attachment_thickness_m, attachment_height_m:
        Attachment thickness and height above the plate surface (T / corner).
    weld_leg_m:
        Fillet weld leg (T / corner joints).  Also the depth of the
        surface-following mesh band below the weld profile.
    flank_angle_deg:
        Angle between the weld flank and the plate surface (45 deg for a standard
        fillet, 30 deg for a typical butt-weld reinforcement slope).
    reinforcement_m, crest_width_m:
        Butt-weld reinforcement height and crest width.
    symmetry:
        ``True`` for the symmetric half model, ``False`` for the full L section.
    patch_length_m:
        Half length of the modelled plate; defaults to ``8 t``.
    load_case:
        Loading of the joint family.  ``"carried"`` (default, backward compatible)
        applies the plate axial traction *and* the attachment traction to opposite
        far edges; ``"plate_axial"`` keeps only the plate axial traction, i.e. the
        non-load-carrying transverse attachment case; ``"attachment_supported"``
        pulls the attachment axially while the far plate edge is supported
        vertically, i.e. the load-carrying joint whose base plate carries the
        attachment load in bending, as in the joint embedded in a stiffened panel.  The nominal stress normaliser is the same plate membrane stress
        in every case, so the three cases are directly comparable.
    """

    joint_type: str
    plate_thickness_m: float
    toe_radius_m: float
    attachment_thickness_m: float = 0.0
    attachment_height_m: float = 0.0
    weld_leg_m: float = 0.0
    flank_angle_deg: float = 45.0
    reinforcement_m: float = 0.012
    crest_width_m: float = 0.010
    symmetry: bool = True
    patch_length_m: float = 0.0
    bonded_root: bool = True
    load_case: str = "carried"

    def __post_init__(self) -> None:
        allowed = {"t_fillet", "butt", "corner"}
        if self.joint_type not in allowed:
            raise ValueError(f"joint_type must be one of {sorted(allowed)}")
        for name in ("plate_thickness_m", "toe_radius_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real scalar")
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be positive and finite")
        if self.load_case not in ALLOWED_LOAD_CASES:
            raise ValueError(f"load_case must be one of {sorted(ALLOWED_LOAD_CASES)}")
        if not 0.0 < self.flank_angle_deg < 90.0:
            raise ValueError("flank_angle_deg must lie in (0, 90)")
        if self.joint_type in {"t_fillet", "corner"}:
            if self.attachment_thickness_m <= 0.0 or self.attachment_height_m <= 0.0:
                raise ValueError("T/corner joints need attachment thickness and height")
            if self.weld_leg_m <= self.toe_radius_m:
                raise ValueError("weld leg must exceed the toe radius")
            if self.attachment_height_m <= self.weld_leg_m:
                raise ValueError("attachment height must exceed the weld leg")
        else:
            if self.reinforcement_m <= self.toe_radius_m:
                raise ValueError("reinforcement must exceed the toe radius")
        if self.joint_type == "butt" and self.load_case == "attachment_supported":
            raise ValueError("butt welds have no attachment load path")
        if self.joint_type == "corner" and self.symmetry:
            raise ValueError("corner joints use the full L section (symmetry=False)")

    # -- derived geometry ---------------------------------------------------
    @property
    def flank_angle_rad(self) -> float:
        return math.radians(self.flank_angle_deg)

    @property
    def toe_ratio(self) -> float:
        """``t / r`` of the detail; drives the empirical anchors."""
        return self.plate_thickness_m / self.toe_radius_m

    @property
    def attachment_ratio(self) -> float:
        """``h / t`` of the detail."""
        return self.attachment_height_m / self.plate_thickness_m

    @property
    def half_width_m(self) -> float:
        """Attachment width of the modelled section (half width when symmetric)."""
        if self.joint_type == "corner":
            return self.attachment_thickness_m
        return 0.5 * self.attachment_thickness_m

    @property
    def patch_m(self) -> float:
        if self.patch_length_m > 0.0:
            return float(self.patch_length_m)
        return 8.0 * self.plate_thickness_m

    @property
    def plate_depth_m(self) -> float:
        """Modelled depth below the plate surface (half thickness for a butt weld)."""
        if self.joint_type == "butt":
            return 0.5 * self.plate_thickness_m
        return self.plate_thickness_m

    @property
    def surface_band_m(self) -> float:
        """Depth of the surface-following mesh band below the weld profile.

        The band must stay well inside the modelled depth, otherwise the graded
        tail rows below it collapse onto the far row and the Q8 elements lose
        their area (this happened for ``band >= plate_depth_m``: butt welds with
        a thick reinforcement and T-joints with a weld leg above the plate depth).
        """
        if self.joint_type == "butt":
            band = max(2.5 * self.reinforcement_m, 4.0 * self.toe_radius_m)
        else:
            band = float(self.weld_leg_m)
        return min(band, 0.85 * self.plate_depth_m)


def nii_stress_concentration(
    plate_thickness_m: float, toe_radius_m: float, attachment_height_m: float
) -> float:
    """Nii closed form for the transverse fillet T-joint toe.

    ``K_t = 1 + 0.27 sqrt(t/r) sqrt(h/t)`` in terms of the plate thickness ``t``,
    the toe radius ``r`` and the attachment height ``h``.  It is an empirical fit,
    not a solution of the elasticity problem, and is used here only as an
    independent anchor for the finite-element value.
    """
    t, r, h = float(plate_thickness_m), float(toe_radius_m), float(attachment_height_m)
    if min(t, r, h) <= 0.0:
        raise ValueError("t, r and h must be positive")
    return 1.0 + 0.27 * math.sqrt(t / r) * math.sqrt(h / t)


def maddox_stress_concentration(plate_thickness_m: float, toe_radius_m: float) -> float:
    """Maddox closed form for the butt-weld toe, ``K_t = 1 + 0.15 sqrt(t/r)``."""
    t, r = float(plate_thickness_m), float(toe_radius_m)
    if min(t, r) <= 0.0:
        raise ValueError("t and r must be positive")
    return 1.0 + 0.15 * math.sqrt(t / r)


def anchor_stress_concentration(detail: WeldDetail) -> float | None:
    """Closed-form anchor of the detail, or ``None`` when none is claimed."""
    if detail.joint_type == "t_fillet":
        return nii_stress_concentration(
            detail.plate_thickness_m, detail.toe_radius_m, detail.attachment_height_m
        )
    if detail.joint_type == "butt":
        return maddox_stress_concentration(detail.plate_thickness_m, detail.toe_radius_m)
    return None


# --------------------------------------------------------------------------- #
# 2. Free-surface profile of the weld detail
# --------------------------------------------------------------------------- #
class _JointProfile:
    """X-monotonic, convex free surface ``y = f(x)`` of a weld detail.

    The toe is modelled as a reentrant corner -- weld flank meeting the plate
    surface -- rounded by a circular arc of radius ``r`` tangent to both walls.
    The void wedge angle is ``psi = pi - flank``, so the arc centre sits ``r``
    above the plate surface and ``r / tan(psi / 2)`` from the theoretical corner
    along each wall.
    """

    def __init__(self, detail: WeldDetail):
        self.detail = detail
        r = detail.toe_radius_m
        slope = math.tan(detail.flank_angle_rad)
        self.slope = slope
        self.crest = detail.reinforcement_m if detail.joint_type == "butt" else detail.weld_leg_m
        if detail.joint_type == "butt":
            self.flank_origin = 0.5 * detail.crest_width_m
            self.surface_start = 0.0
            self.has_crest = True
        else:
            self.flank_origin = detail.half_width_m
            self.surface_start = detail.half_width_m
            self.has_crest = False
        self.corner = self.flank_origin + self.crest / slope
        self.tangent = r * math.tan(0.5 * detail.flank_angle_rad)
        self.toe = self.corner + self.tangent
        self.arc_start = self.corner - self.tangent * math.cos(detail.flank_angle_rad)
        self.arc_centre_y = r

    def height(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        r = self.detail.toe_radius_m
        y = np.zeros_like(x)
        crest = x <= self.flank_origin if self.has_crest else np.zeros(x.shape, dtype=bool)
        if self.has_crest:
            y[crest] = self.crest
        flank = (~crest) & (x <= self.arc_start)
        y[flank] = self.crest - (x[flank] - self.flank_origin) * self.slope
        arc = (x > self.arc_start) & (x <= self.toe)
        dx = x[arc] - self.toe
        y[arc] = self.arc_centre_y - np.sqrt(np.maximum(r * r - dx * dx, 0.0))
        return y

    def height_scalar(self, x: float) -> float:
        return float(self.height(np.array([x]))[0])


def _graded(a: float, b: float, n: int, power: float = 1.8) -> np.ndarray:
    """``n + 1`` points from ``a`` to ``b``, clustered towards ``a``."""
    s = np.linspace(0.0, 1.0, n + 1)
    return a + (b - a) * s**power


def _weld_columns(profile: _JointProfile, detail: WeldDetail, level: int) -> np.ndarray:
    """Column abscissae of the weld block: fine across the toe arc, graded away.

    The uniform ``fine`` band brackets the rounded toe (``arc_start`` to ``toe``)
    with a margin of a quarter radius on the flank side and half a radius on the
    plate side, so the curvature of the toe arc is sampled by ``16 * level``
    elements.  Outside the band the columns are geometrically graded towards it.
    """
    r = detail.toe_radius_m
    n_fine = 16 * level
    band_lo = max(profile.arc_start - 0.25 * r, profile.surface_start)
    band_hi = min(profile.toe + 0.5 * r, detail.patch_m)
    fine = np.linspace(band_lo, band_hi, n_fine + 1)
    left = _graded(band_lo, profile.surface_start, max(6, 8 * level), 2.2)[::-1]
    right = _graded(band_hi, detail.patch_m, max(8, 12 * level), 2.2)
    cols = np.concatenate((left, fine[1:], right[1:]))
    # make the toe tangent point, the arc start and the theoretical corner mesh
    # nodes: the notch peak sits exactly on the toe tangent point
    key_points = np.array(
        [
            p
            for p in (
                profile.flank_origin,
                profile.arc_start,
                profile.corner,
                profile.toe,
            )
            if profile.surface_start < p < detail.patch_m
        ]
    )
    if key_points.size:
        cols = np.concatenate((cols, key_points))
    key = np.round(cols, 12)
    _, idx = np.unique(key, return_index=True)
    return np.sort(cols[idx])


def _depth_levels(detail: WeldDetail, level: int, sagitta_m: float = 0.0) -> np.ndarray:
    """Depth levels below the free surface (geometric, fine at the toe).

    The first level must exceed the sagitta ``sagitta_m`` of the projected surface
    chords by a safe factor, otherwise the projected mid-side nodes fold the
    surface elements; the projected-toe geometry is measured in :func:`_build_mesh`.
    """
    band = detail.surface_band_m
    n = 18 + 4 * (level - 1)
    first = max(0.05 * detail.toe_radius_m / level, 3.0 * sagitta_m)
    first = min(first, 0.2 * band)
    ratio = (band / first) ** (1.0 / n)
    depths = first * ratio ** np.arange(n + 1)
    depths[-1] = band
    depths[0] = 0.0
    return depths


def _column_rows(detail: WeldDetail, profile: _JointProfile, x: float, depths: np.ndarray):
    """Ascending ``y`` values of one weld-block column at abscissa ``x``."""
    top = profile.height_scalar(x)
    below = top - depths                      # descending
    lower = float(below[-1])                  # = top - band
    anchor = -detail.plate_depth_m
    span = anchor - lower
    tail = np.empty(0)
    if span < -1.0e-12:                       # tail rows (graded towards the anchor)
        u = np.linspace(0.0, 1.0, 11)[1:]
        tail = lower + span * u**1.6
    rows = np.concatenate((below, tail))[::-1]
    # a repeated abscissa collapses a Q8 element; keep the first occurrence
    keep = np.ones(rows.shape[0], dtype=bool)
    keep[1:] = np.abs(np.diff(rows)) > 1.0e-15
    return rows[keep]


class _NodeRegistry:
    """Coordinate-keyed node registry; bitwise-identical duplicates are merged."""

    def __init__(self) -> None:
        self._index: dict[tuple[str, float, float], int] = {}
        self.coords: list[tuple[float, float]] = []

    def add(self, x: float, y: float, group: str = "") -> int:
        """Register (or reuse) the node at ``(x, y)``.

        ``group`` keeps coincident nodes of *different* groups apart; it is used for
        the unbonded attachment/plate interface of a T-joint, whose material is only
        continuous through the weld metal.
        """
        key = (group or "", round(float(x), 12), round(float(y), 12))
        idx = self._index.get(key)
        if idx is None:
            idx = len(self.coords)
            self._index[key] = idx
            self.coords.append((float(x), float(y)))
        return idx

    def array(self) -> np.ndarray:
        return np.asarray(self.coords, dtype=float)


def _structured_block(
    registry: _NodeRegistry,
    column_x,
    row_y,
    *,
    profile: _JointProfile | None = None,
    project_top: bool = False,
    group_of=None,
):
    """Structured Q8 block over ``(i, j) -> (x_i, row_y(x_i)[j])``.

    Returns ``(elements, top_elements)``; the last row is the free surface for the
    weld / butt blocks.  Connectivity is counter-clockwise:
    ``(i,j), (i+1,j), (i+1,j+1), (i,j+1)`` plus edge mid-nodes, so the free surface
    is the ``eta = +1`` edge (local nodes 2-3, mid-node 6).
    """
    nx = len(column_x) - 1
    rows = [np.asarray(row_y(x), dtype=float) for x in column_x]
    ny = len(rows[0]) - 1
    if any(len(v) - 1 != ny for v in rows):
        raise ValueError("all columns must have the same number of rows")
    def tag(i, j):
        return (group_of(i, j) or "") if group_of is not None else ""

    corner = np.empty((nx + 1, ny + 1), dtype=np.int64)
    for i, x in enumerate(column_x):
        for j, y in enumerate(rows[i]):
            corner[i, j] = registry.add(float(x), float(y), tag(i, j))
    elements = np.empty((nx * ny, 8), dtype=np.int64)
    e = 0
    for i in range(nx):
        for j in range(ny):
            nodes = (corner[i, j], corner[i + 1, j], corner[i + 1, j + 1], corner[i, j + 1])
            curved = project_top and (j == ny - 1)
            corners_idx = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
            mids = []
            for k in range(4):
                a, b = nodes[k], nodes[(k + 1) % 4]
                pa, pb = registry.coords[a], registry.coords[b]
                xm = 0.5 * (pa[0] + pb[0])
                ym = 0.5 * (pa[1] + pb[1])
                if curved and k == 2 and profile is not None:
                    ym = profile.height_scalar(xm)
                mids.append(registry.add(xm, ym, tag(*corners_idx[k])))
            elements[e] = (*nodes, *mids)
            e += 1
    ids = np.arange(nx * ny).reshape(nx, ny)
    return elements, [int(ids[i, ny - 1]) for i in range(nx)]

# --------------------------------------------------------------------------- #
# 3. Finite-element model of the joint cross-section
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WeldToeSolution:
    """Result of the plane weld-detail model (stresses in Pa, lengths in m)."""

    joint_type: str
    toe_radius_m: float
    nominal_stress_pa: float
    peak_toe_stress_pa: float
    kt_toe: float
    attachment_toe_stress_pa: float
    surface_x_m: tuple[float, ...]
    surface_sigma_xx_pa: tuple[float, ...]
    hotspot_linear_pa: float
    hotspot_quadratic_pa: float
    kt_hotspot_linear: float
    kt_hotspot_quadratic: float
    x_toe_m: float
    min_edge_m: float
    n_nodes: int
    n_elements: int
    level: int

    def hotspot_stress(self, method: str = "quadratic") -> float:
        if method == "linear":
            return self.hotspot_linear_pa
        if method == "quadratic":
            return self.hotspot_quadratic_pa
        raise ValueError("method must be 'linear' or 'quadratic'")



# --- local vectorised Q8 kernels (analytic 2x2 Jacobian) -------------------- #
# ``quadratic_plane.q8_stiffness`` calls ``torch.linalg.det``/``inv`` on a batch of
# 2x2 matrices, which is pathologically slow on the target (loong64) build for the
# ~1e3-element multi-scale meshes used here.  The kernels below reproduce the same
# 3x3 Gauss rule, the same shape functions and the same strain-displacement layout,
# with the 2x2 inverse written out analytically.

def _q8_dN(xi: float, eta: float) -> np.ndarray:
    """Derivatives of the eight Q8 shape functions, shape ``(8, 2)``."""
    x, y = xi, eta
    dxi = np.array([
        0.25 * (1 - y) * (2 * x + y), 0.25 * (1 - y) * (2 * x - y),
        0.25 * (1 + y) * (2 * x + y), 0.25 * (1 + y) * (2 * x - y),
        -x * (1 - y), 0.5 * (1 - y * y), -x * (1 + y), -0.5 * (1 - y * y),
    ])
    deta = np.array([
        0.25 * (1 - x) * (x + 2 * y), 0.25 * (1 + x) * (-x + 2 * y),
        0.25 * (1 + x) * (x + 2 * y), 0.25 * (1 - x) * (-x + 2 * y),
        -0.5 * (1 - x * x), -(1 + x) * y, 0.5 * (1 - x * x), -(1 - x) * y,
    ])
    return np.stack((dxi, deta), axis=1)


def _b_matrix_np(derivative: np.ndarray) -> np.ndarray:
    """Engineering strain-displacement matrix ``(ne, 3, 16)`` from ``(ne, 8, 2)``."""
    ne = derivative.shape[0]
    matrix = np.zeros((ne, 3, 16))
    matrix[:, 0, 0::2] = derivative[:, :, 0]
    matrix[:, 1, 1::2] = derivative[:, :, 1]
    matrix[:, 2, 0::2] = derivative[:, :, 1]
    matrix[:, 2, 1::2] = derivative[:, :, 0]
    return matrix


def _jacobian_np(coords: np.ndarray, derivative: np.ndarray):
    """Jacobian ``(ne, 2, 2)``, its determinant and inverse (analytic 2x2)."""
    jac = np.einsum("eia,ib->eab", coords, derivative)
    det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
    inverse = np.empty_like(jac)
    inverse[:, 0, 0] = jac[:, 1, 1]
    inverse[:, 1, 1] = jac[:, 0, 0]
    inverse[:, 0, 1] = -jac[:, 0, 1]
    inverse[:, 1, 0] = -jac[:, 1, 0]
    inverse /= det[:, None, None]
    return jac, det, inverse


def _q8_stiffness_fast(coords: np.ndarray, dmat: np.ndarray, thickness: float = 1.0):
    """Q8 element stiffness ``(ne, 16, 16)`` by the 3x3 Gauss rule (numpy)."""
    ne = coords.shape[0]
    stiffness = np.zeros((ne, 16, 16))
    for (xi, eta), (wa, wb) in zip(GAUSS_POINTS_3X3, GAUSS_WEIGHTS_3X3):
        derivative = _q8_dN(float(xi), float(eta))
        _, det, inverse = _jacobian_np(coords, derivative)
        if not np.all(det > 0.0):
            bad = int(np.argmin(det))
            raise ValueError(
                "Q8 element has a non-positive Jacobian "
                f"(element {bad}, min det {float(det.min()):.3e}); "
                "the toe mesh is degenerate for this geometry"
            )
        dNdx = np.einsum("ib,ebc->eic", derivative, inverse)
        b = _b_matrix_np(dNdx)
        stiffness += np.einsum("eki,kl,elj->eij", b, dmat, b) * (
            det * thickness * float(wa) * float(wb)
        )[:, None, None]
    return stiffness


def _stress_at_surface(coords: np.ndarray, local_u: np.ndarray, dmat: np.ndarray, xi: float):
    """Stress ``(ne, 3)`` and abscissa ``(ne,)`` on the ``eta = +1`` edge."""
    derivative = _q8_dN(float(xi), 1.0)
    _, _, inverse = _jacobian_np(coords, derivative)
    dNdx = np.einsum("ib,ebc->eic", derivative, inverse)
    b = _b_matrix_np(dNdx)
    stress = np.einsum("kl,eli,ei->ek", dmat, b, local_u)
    shape = _q8_shape_row(float(xi), 1.0)
    x = shape @ coords[:, :, 0].T
    return stress, x

def _consistent_edge_loads(coords, edge, traction_pa):
    """Consistent nodal loads of a uniform traction on a straight Q8 edge (1/m)."""
    a, m, b = edge
    pa = np.asarray(coords[a], dtype=float)
    pb = np.asarray(coords[b], dtype=float)
    length = float(np.linalg.norm(pb - pa))
    share = traction_pa * length
    out: dict[int, float] = {}
    for node, weight in ((a, 1.0 / 6.0), (m, 4.0 / 6.0), (b, 1.0 / 6.0)):
        out[node] = out.get(node, 0.0) + share * weight
    return out


def _min_edge(coords: np.ndarray, elements: np.ndarray) -> float:
    best = math.inf
    for elem in elements:
        xy = coords[elem]
        for k in range(4):
            best = min(best, float(np.linalg.norm(xy[(k + 1) % 4] - xy[k])))
    return best


def _q8_shape_row(xi: float, eta: float) -> np.ndarray:
    """Row vector of the eight Q8 shape functions at ``(xi, eta)``."""
    x, y = xi, eta
    return np.array(
        [
            0.25 * (1 - x) * (1 - y) * (-x - y - 1),
            0.25 * (1 + x) * (1 - y) * (x - y - 1),
            0.25 * (1 + x) * (1 + y) * (x + y - 1),
            0.25 * (1 - x) * (1 + y) * (-x + y - 1),
            0.5 * (1 - x * x) * (1 - y),
            0.5 * (1 + x) * (1 - y * y),
            0.5 * (1 - x * x) * (1 + y),
            0.5 * (1 - x) * (1 - y * y),
        ]
    )


def _surface_path(coords, elements, displacement, surface_elems, dmat, arc_start, toe):
    """Axial (``sigma_xx``) stress path along the free surface of the weld block."""
    conn = np.asarray(elements)[np.asarray(surface_elems, dtype=int)]
    xy = np.asarray(coords, dtype=float)[conn]                  # (ne, 8, 2)
    local = np.zeros((conn.shape[0], 16))
    local[:, 0::2] = np.asarray(displacement)[2 * conn]
    local[:, 1::2] = np.asarray(displacement)[2 * conn + 1]
    xs: list[np.ndarray] = []
    ss: list[np.ndarray] = []
    for xi in np.linspace(-1.0, 1.0, 7):
        stress, x = _stress_at_surface(xy, local, np.asarray(dmat), float(xi))
        xs.append(x)
        ss.append(stress[:, 0])
    x_all = np.concatenate(xs)
    s_all = np.concatenate(ss)
    order = np.argsort(x_all)
    x_all, s_all = x_all[order], s_all[order]
    _, idx = np.unique(np.round(x_all, 12), return_index=True)
    xs_u, ss_u = x_all[idx], s_all[idx]
    attach = s_all[x_all < arc_start]
    attach_peak = float(attach.max()) if attach.size else float("nan")
    return xs_u, ss_u, attach_peak


def _lagrange_extrapolate(stations, values, target: float = 0.0) -> float:
    """Lagrange extrapolation of ``values`` sampled at ``stations`` to ``target``."""
    total = 0.0
    for i, xi in enumerate(stations):
        weight = 1.0
        for j, xj in enumerate(stations):
            if i != j:
                weight *= (target - xj) / (xi - xj)
        total += weight * values[i]
    return total


@dataclass(frozen=True)
class HotSpotResult:
    """Hot-spot (structural) stress of a weld detail (Pa)."""

    method: str
    stations: tuple[float, ...]
    sampled_stress_pa: tuple[float, ...]
    stress_pa: float
    stress_ratio: float
    toe_x_m: float


def hot_spot_stress(
    solution: WeldToeSolution, plate_thickness_m: float, *, method: str = "quadratic"
) -> HotSpotResult:
    """IIW-style surface extrapolation of the structural hot-spot stress.

    ``method="linear"`` samples ``0.4t`` and ``0.9t`` from the toe tangent point
    (weights ``1.8, -0.8``); ``method="quadratic"`` samples ``0.4t``, ``0.9t`` and
    ``1.4t`` (weights ``2.52, -2.24, 0.72``).
    """
    if method == "linear":
        stations = LINEAR_STATIONS
    elif method == "quadratic":
        stations = QUADRATIC_STATIONS
    else:
        raise ValueError("method must be 'linear' or 'quadratic'")
    t = float(plate_thickness_m)
    if t <= 0.0:
        raise ValueError("plate_thickness_m must be positive")
    x = np.asarray(solution.surface_x_m, dtype=float)
    s = np.asarray(solution.surface_sigma_xx_pa, dtype=float)
    if x.size < 2:
        raise ValueError("surface path is too short for extrapolation")
    samples = tuple(float(np.interp(solution.x_toe_m + f * t, x, s)) for f in stations)
    value = _lagrange_extrapolate(list(stations), list(samples))
    return HotSpotResult(
        method=method,
        stations=tuple(stations),
        sampled_stress_pa=samples,
        stress_pa=value,
        stress_ratio=value / solution.nominal_stress_pa,
        toe_x_m=solution.x_toe_m,
    )


def mesh_convergence(detail: WeldDetail, *, levels=(1, 2)) -> dict[str, object]:
    """Toe and hot-spot ``K_t`` over successive mesh levels.

    The returned ``change`` dictionary holds the relative change of each quantity
    between the two finest levels -- the mesh-independence measure used by the
    acceptance gate (2 % on the hot-spot stress).
    """
    rows = []
    for level in levels:
        sol = solve_weld_detail(detail, level=level)
        lin = hot_spot_stress(sol, detail.plate_thickness_m, method="linear")
        quad = hot_spot_stress(sol, detail.plate_thickness_m, method="quadratic")
        rows.append(
            dict(
                level=level,
                kt_toe=sol.kt_toe,
                hotspot_linear=lin.stress_ratio,
                hotspot_quadratic=quad.stress_ratio,
                min_edge_m=sol.min_edge_m,
                n_nodes=sol.n_nodes,
                n_elements=sol.n_elements,
            )
        )
    finest = rows[-1]
    coarse = rows[-2] if len(rows) > 1 else rows[-1]
    change = {}
    for key in ("kt_toe", "hotspot_linear", "hotspot_quadratic"):
        prev, cur = float(coarse[key]), float(finest[key])
        change[key] = abs(cur - prev) / abs(cur) if cur else float("nan")
    return {"levels": rows, "change": change}


def _build_mesh(detail: WeldDetail, level: int) -> dict[str, object]:
    """Build the plane mesh, the consistent load vector and the Dirichlet set."""
    profile = _JointProfile(detail)
    registry = _NodeRegistry()
    columns = _weld_columns(profile, detail, level)
    sagitta = max(
        (
            abs(
                profile.height_scalar(0.5 * (a + b))
                - 0.5 * (profile.height_scalar(a) + profile.height_scalar(b))
            )
            for a, b in zip(columns[:-1], columns[1:])
        ),
        default=0.0,
    )
    depths = _depth_levels(detail, level, sagitta)
    blocks: list[np.ndarray] = []
    loaded: list[tuple[tuple[int, int, int], float, tuple[float, float]]] = []
    symmetry: list[int] = []
    clamped: list[int] = []

    if detail.joint_type == "butt":
        cols = np.concatenate(([0.0], columns[columns > 0.0]))
        elements, top_local = _structured_block(
            registry,
            cols,
            lambda x: _column_rows(detail, profile, x, depths),
            profile=profile,
            project_top=True,
        )
        blocks.append(elements)
        top = top_local  # single block: local ids are already global
        first = _column_rows(detail, profile, cols[0], depths)
        symmetry = [registry.add(cols[0], y) for y in first]
        last = _column_rows(detail, profile, cols[-1], depths)
        far = [registry.add(cols[-1], y) for y in last]
        for k in range(len(far) - 1):
            mid = registry.add(cols[-1], 0.5 * (last[k] + last[k + 1]))
            loaded.append(((far[k], mid, far[k + 1]), 1.0, (1.0, 0.0)))
        anchor_node = registry.add(cols[0], -detail.plate_depth_m)
    else:
        seam = detail.half_width_m
        xa = _graded(0.0, seam, max(6, 5 * level), 1.6)
        band = detail.surface_band_m
        band_rows = (band - depths)[::-1]                 # ascending 0 .. band
        above = np.linspace(band, detail.attachment_height_m, max(4, 3 * level) + 1)[1:]
        att_rows = np.concatenate((band_rows, above))
        interface_tag = None
        if not detail.bonded_root:
            interface_tag = lambda i, j: "attachment_interface" if j == 0 else None  # noqa: E731
        attachment, _ = _structured_block(
            registry, xa, lambda x: att_rows, group_of=interface_tag
        )
        blocks.append(attachment)

        seam_rows = _column_rows(detail, profile, seam, depths)
        plate_rows = seam_rows[seam_rows <= 0.0]
        plate, _ = _structured_block(registry, xa, lambda x: plate_rows)
        blocks.append(plate)

        cols = np.concatenate(([seam], columns[columns > seam]))
        weld, top_local = _structured_block(
            registry,
            cols,
            lambda x: _column_rows(detail, profile, x, depths),
            profile=profile,
            project_top=True,
        )
        offset = attachment.shape[0] + plate.shape[0]
        top = [t + offset for t in top_local]
        blocks.append(weld)

        if detail.joint_type != "corner":
            symmetry = [registry.add(0.0, y) for y in np.concatenate((plate_rows, att_rows))]
        att_top = [registry.add(x, detail.attachment_height_m) for x in xa]
        traction_att = detail.plate_thickness_m / seam
        if detail.load_case != "plate_axial":
            for k in range(len(att_top) - 1):
                mid = registry.add(0.5 * (xa[k] + xa[k + 1]), detail.attachment_height_m)
                loaded.append(((att_top[k], mid, att_top[k + 1]), traction_att, (0.0, 1.0)))
        last = _column_rows(detail, profile, cols[-1], depths)
        far = [registry.add(cols[-1], y) for y in last]
        if detail.load_case == "attachment_supported":
            # the surrounding structure supports the base plate vertically: the
            # attachment pull is carried by the plate in bending, not by the pin
            clamped.extend(far)
        else:
            for k in range(len(far) - 1):
                mid = registry.add(cols[-1], 0.5 * (last[k] + last[k + 1]))
                loaded.append(((far[k], mid, far[k + 1]), 1.0, (1.0, 0.0)))
        anchor_node = registry.add(0.0, -detail.plate_depth_m)

    elements = np.concatenate(blocks, axis=0)
    coords = registry.array()
    forces = np.zeros(2 * coords.shape[0], dtype=float)
    for edge, traction, direction in loaded:
        for node, value in _consistent_edge_loads(coords, edge, traction).items():
            forces[2 * node] += value * direction[0]
            forces[2 * node + 1] += value * direction[1]
    dirichlet: dict[int, float] = {}
    for node in symmetry:
        dirichlet[2 * node] = 0.0
    if detail.joint_type == "corner":
        dirichlet[2 * anchor_node] = 0.0
    for node in clamped:
        # vertical support of the far plate edge only: the plate stays free to
        # slide axially so the case carries no spurious axial restraint
        dirichlet[2 * node + 1] = 0.0
    if not clamped:
        dirichlet[2 * anchor_node + 1] = 0.0
    return {
        "coords": coords,
        "elements": elements,
        "forces": forces,
        "dirichlet": dirichlet,
        "top": top,
        "profile": profile,
        "blocks": blocks,
    }


def _jacobian_report(coords: np.ndarray, elements: np.ndarray, samples=3) -> list[int]:
    """Element ids whose Jacobian is non-positive at any sample point."""
    bad = []
    xs = np.linspace(-1.0, 1.0, samples)
    for e, conn in enumerate(elements):
        xy = torch.as_tensor(coords[conn], dtype=torch.float64).unsqueeze(0)
        for xi in xs:
            for eta in xs:
                try:
                    _, det = q8_strain_matrix(float(xi), float(eta), xy)
                except ValueError:
                    bad.append(int(e))
                    break
                if float(det.min()) <= 0.0:
                    bad.append(int(e))
                    break
            if bad and bad[-1] == e:
                break
    return bad


def _solve_plane(coords: np.ndarray, elements: np.ndarray, forces: np.ndarray,
                 dirichlet: dict[int, float], plane: str):
    """Exact sparse (SuperLU) solve of the constrained Q8 plane problem.

    The multi-scale notch mesh makes unpreconditioned CG impractical, so the
    reduced system is factorised directly.  All Dirichlet values of this model are
    zero (symmetry planes and the anchor point), which keeps the elimination to a
    row/column filter.
    """
    elements_t = torch.as_tensor(elements, dtype=torch.long)
    dmat = (
        elasticity_matrix(
            torch.tensor([YOUNG_STEEL_PA], dtype=torch.float64),
            torch.tensor([POISSON_STEEL], dtype=torch.float64),
            plane,
        )[0]
        .numpy()
    )
    stiffness = torch.as_tensor(_q8_stiffness_fast(coords[elements], dmat), dtype=torch.float64)
    dofs = q8_element_dofs(elements_t)
    ndof = 2 * coords.shape[0]
    K = assemble_coo(stiffness, dofs, ndof).coalesce()
    fixed = np.zeros(ndof, dtype=bool)
    for dof, value in dirichlet.items():
        if value != 0.0:
            raise ValueError("only homogeneous Dirichlet data are supported")
        fixed[dof] = True
    free = np.where(~fixed)[0]
    remap = -np.ones(ndof, dtype=np.int64)
    remap[free] = np.arange(free.size)
    index = K.indices().numpy()
    keep = ~fixed[index[0]] & ~fixed[index[1]]
    rows = remap[index[0][keep]]
    cols = remap[index[1][keep]]
    values = K.values().numpy()[keep]
    reduced = torch.sparse_coo_tensor(
        torch.as_tensor(np.stack((rows, cols))),
        torch.as_tensor(values, dtype=torch.float64),
        (free.size, free.size),
    ).coalesce()
    factor = factorize_sparse(reduced, method="splu")
    rhs = torch.as_tensor(np.asarray(forces, dtype=float)[free], dtype=torch.float64)
    displacement = np.zeros(ndof, dtype=float)
    displacement[free] = factor.solve(rhs).detach().numpy()
    info = {
        "n_dofs": int(ndof),
        "n_free": int(free.size),
        "nnz": int(K._nnz()),
        "factor_nnz": int(factor.factor_nnz),
        "factor_seconds": float(factor.factorization_seconds),
        "relative_residual": float(factor.last_relative_residual),
    }
    return displacement, info


def _assemble(
    detail: WeldDetail,
    level: int,
    *,
    plane: str,
    nominal_stress_pa: float,
) -> tuple[WeldToeSolution, dict[str, object], dict[str, object]]:
    """Build the mesh, solve it and return (solution, mesh, info)."""
    mesh = _build_mesh(detail, level)
    coords = mesh["coords"]
    elements = mesh["elements"]
    forces = np.asarray(mesh["forces"], dtype=float) * nominal_stress_pa
    displacement, solve_info = _solve_plane(
        coords, elements, forces, dict(mesh["dirichlet"]), plane
    )
    dmat = (
        elasticity_matrix(
            torch.tensor([YOUNG_STEEL_PA], dtype=torch.float64),
            torch.tensor([POISSON_STEEL], dtype=torch.float64),
            plane,
        )[0]
        .numpy()
    )
    profile = mesh["profile"]
    xs, sigmas, attach_peak = _surface_path(
        coords, elements, displacement, mesh["top"], dmat, profile.arc_start, profile.toe
    )
    near = xs >= profile.toe - 3.0 * detail.toe_radius_m
    peak = float(sigmas[near].max()) if near.any() else float(sigmas.max())
    solution = WeldToeSolution(
        joint_type=detail.joint_type,
        toe_radius_m=detail.toe_radius_m,
        nominal_stress_pa=nominal_stress_pa,
        peak_toe_stress_pa=peak,
        kt_toe=peak / nominal_stress_pa,
        attachment_toe_stress_pa=attach_peak,
        surface_x_m=tuple(float(v) for v in xs),
        surface_sigma_xx_pa=tuple(float(v) for v in sigmas),
        hotspot_linear_pa=float("nan"),
        hotspot_quadratic_pa=float("nan"),
        kt_hotspot_linear=float("nan"),
        kt_hotspot_quadratic=float("nan"),
        x_toe_m=profile.toe,
        min_edge_m=_min_edge(coords, elements),
        n_nodes=int(coords.shape[0]),
        n_elements=int(elements.shape[0]),
        level=level,
    )
    lin = hot_spot_stress(solution, detail.plate_thickness_m, method="linear")
    quad = hot_spot_stress(solution, detail.plate_thickness_m, method="quadratic")
    filled = {field: getattr(solution, field) for field in solution.__dataclass_fields__}
    filled.update(
        hotspot_linear_pa=lin.stress_pa,
        hotspot_quadratic_pa=quad.stress_pa,
        kt_hotspot_linear=lin.stress_ratio,
        kt_hotspot_quadratic=quad.stress_ratio,
    )
    info = {
        "solve": solve_info,
        "hotspot_linear": lin,
        "hotspot_quadratic": quad,
    }
    return WeldToeSolution(**filled), mesh, info


@lru_cache(maxsize=64)
def _solve_cached(detail: WeldDetail, level: int) -> WeldToeSolution:
    solution, _, _ = _assemble(detail, level, plane="strain", nominal_stress_pa=1.0e6)
    return solution


def solve_weld_detail(
    detail: WeldDetail,
    *,
    level: int = 1,
    plane: str = "strain",
    nominal_stress_pa: float = 1.0e6,
) -> WeldToeSolution:
    """Solve the plane weld-detail model and return the toe stress state.

    ``level`` scales the near-toe refinement (``level=2`` doubles it) and drives the
    mesh-convergence check of the hot-spot extrapolation.  The default plane is
    ``"strain"`` because the joint cross-section is far from a free surface in the
    weld direction.
    """
    if not isinstance(detail, WeldDetail):
        raise TypeError("detail must be a WeldDetail")
    if level not in (1, 2, 3):
        raise ValueError("level must be 1, 2 or 3")
    if plane not in ("stress", "strain"):
        raise ValueError("plane must be 'stress' or 'strain'")
    stress = float(nominal_stress_pa)
    if isinstance(nominal_stress_pa, bool) or not (stress > 0.0 and math.isfinite(stress)):
        raise ValueError("nominal_stress_pa must be positive and finite")
    if plane == "strain" and stress == 1.0e6:
        return _solve_cached(detail, level)
    solution, _, _ = _assemble(detail, level, plane=plane, nominal_stress_pa=stress)
    return solution


# --------------------------------------------------------------------------- #
# 4. Double-slope S-N curves (DNV-RP-C203 form)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TwoSlopeSNCurve:
    """Double-slope S-N curve ``log10 N = log10 a - m log10 ds`` (ds in MPa).

    The second branch is anchored on the knee point, so the curve is continuous by
    construction: ``log10 a2 = log10 N_knee + (m2 / m1) (log10 a1 - log10 N_knee)``.
    """

    name: str
    log_a1: float
    slope_m1: float
    knee_cycles: float
    slope_m2: float

    def __post_init__(self) -> None:
        if self.knee_cycles <= 0.0:
            raise ValueError("knee_cycles must be positive")
        if self.slope_m1 <= 0.0 or self.slope_m2 <= 0.0:
            raise ValueError("slopes must be positive")

    @property
    def log_a2(self) -> float:
        log_knee = math.log10(self.knee_cycles)
        return log_knee + (self.slope_m2 / self.slope_m1) * (self.log_a1 - log_knee)

    def cycles_to_failure(self, stress_range_pa: float) -> float:
        stress = float(stress_range_pa)
        if not (stress > 0.0 and math.isfinite(stress)):
            raise ValueError("stress_range_pa must be positive and finite")
        s_mpa = stress / 1.0e6
        n1 = 10.0 ** self.log_a1 * s_mpa ** (-self.slope_m1)
        if n1 <= self.knee_cycles:
            return n1
        return 10.0 ** self.log_a2 * s_mpa ** (-self.slope_m2)

    def stress_range_at_cycles(self, cycles: float) -> float:
        n = float(cycles)
        if not (n > 0.0 and math.isfinite(n)):
            raise ValueError("cycles must be positive and finite")
        if n <= self.knee_cycles:
            return 1.0e6 * (10.0 ** self.log_a1 / n) ** (1.0 / self.slope_m1)
        return 1.0e6 * (10.0 ** self.log_a2 / n) ** (1.0 / self.slope_m2)

    @property
    def knee_stress_range_pa(self) -> float:
        return self.stress_range_at_cycles(self.knee_cycles)


@dataclass(frozen=True)
class SNEnvironment:
    """Named S-N environment: DNV-RP-C203 curve D with an environment knee."""

    name: str
    description: str
    curve: TwoSlopeSNCurve

    def cycles_to_failure(self, stress_range_pa: float) -> float:
        return self.curve.cycles_to_failure(stress_range_pa)


#: DNV-RP-C203 curve D style environments.
#:
#: ``air``: curve D in air -- FAT90 at 2e6 cycles, knee at 1e7 cycles, slopes 3/5
#: (``log10 a1 = 12.164``).  ``seawater_cp``: same slope-3 branch and the same
#: ``log10 a1`` with the knee moved to 1e6 cycles, which is the DNV treatment of
#: cathodically protected joints; the second branch follows from continuity.
#: ``seawater_free``: free corrosion -- the same knee at 1e6 cycles with the
#: corrosion knock-down ``log10 a1 = 11.764`` (a factor of about 0.4 in life at a
#: given stress range).  These are *encoded table inputs*: see the benchmark report
#: for the traceability note and the sensitivity of the fatigue life to them.
ENVIRONMENTS: dict[str, SNEnvironment] = {
    "air": SNEnvironment(
        name="air",
        description="DNV-RP-C203 curve D in air (FAT90, knee at 1e7 cycles)",
        curve=TwoSlopeSNCurve(
            name="DNV-RP-C203 D air", log_a1=12.164, slope_m1=3.0,
            knee_cycles=1.0e7, slope_m2=5.0,
        ),
    ),
    "seawater_cp": SNEnvironment(
        name="seawater_cp",
        description="curve D, cathodic protection (knee at 1e6 cycles)",
        curve=TwoSlopeSNCurve(
            name="DNV-RP-C203 D CP", log_a1=12.164, slope_m1=3.0,
            knee_cycles=1.0e6, slope_m2=5.0,
        ),
    ),
    "seawater_free": SNEnvironment(
        name="seawater_free",
        description="curve D, seawater free corrosion (corrosion knock-down)",
        curve=TwoSlopeSNCurve(
            name="DNV-RP-C203 D free corrosion", log_a1=11.764, slope_m1=3.0,
            knee_cycles=1.0e6, slope_m2=5.0,
        ),
    ),
}


def fatigue_life(
    stress_range_pa: float, *, environment: str = "air", thickness_m: float | None = None
) -> float:
    """Cycles to failure of a stress range on the named environment's S-N curve.

    ``thickness_m`` applies the DNV thickness correction ``(t / 25 mm)^0.2`` for
    plates thicker than 25 mm by lowering the curve (raising the effective range).
    """
    if environment not in ENVIRONMENTS:
        raise ValueError(f"environment must be one of {sorted(ENVIRONMENTS)}")
    curve = ENVIRONMENTS[environment].curve
    stress = float(stress_range_pa)
    if thickness_m is not None and thickness_m > 0.025:
        stress *= (float(thickness_m) / 0.025) ** 0.2
    return curve.cycles_to_failure(stress)


def detail_fatigue_assessment(
    detail: WeldDetail,
    *,
    nominal_stress_range_pa: float = 100.0e6,
    environment: str = "air",
    level: int = 1,
) -> dict[str, object]:
    """Full detail chain: FE toe stress -> hot spot -> S-N life.

    Returns the finite-element ``K_t`` (toe and hot spot, both extrapolations), the
    stress ranges, the fatigue life for each hot-spot variant, and the empirical
    anchor of the detail (``None`` when the geometry has no accepted closed form).
    """
    solution = solve_weld_detail(detail, level=level)
    ds_nom = float(nominal_stress_range_pa)
    if not (ds_nom > 0.0 and math.isfinite(ds_nom)):
        raise ValueError("nominal_stress_range_pa must be positive and finite")
    lin = hot_spot_stress(solution, detail.plate_thickness_m, method="linear")
    quad = hot_spot_stress(solution, detail.plate_thickness_m, method="quadratic")
    anchor = anchor_stress_concentration(detail)
    out: dict[str, object] = {
        "joint_type": detail.joint_type,
        "toe_radius_m": detail.toe_radius_m,
        "toe_ratio": detail.toe_ratio,
        "kt_toe_fe": solution.kt_toe,
        "kt_hotspot_linear": lin.stress_ratio,
        "kt_hotspot_quadratic": quad.stress_ratio,
        "kt_anchor": anchor,
        "anchor_error": (
            abs(solution.kt_toe - anchor) / anchor if anchor else None
        ),
        "hotspot_stress_range_linear_pa": lin.stress_ratio * ds_nom,
        "hotspot_stress_range_quadratic_pa": quad.stress_ratio * ds_nom,
        "life_linear_cycles": fatigue_life(
            lin.stress_ratio * ds_nom, environment=environment,
            thickness_m=detail.plate_thickness_m,
        ),
        "life_quadratic_cycles": fatigue_life(
            quad.stress_ratio * ds_nom, environment=environment,
            thickness_m=detail.plate_thickness_m,
        ),
        "environment": environment,
        "nominal_stress_range_pa": ds_nom,
        "mesh": {
            "level": solution.level,
            "n_nodes": solution.n_nodes,
            "n_elements": solution.n_elements,
            "min_edge_m": solution.min_edge_m,
        },
    }
    if anchor:
        out["life_anchor_cycles"] = fatigue_life(
            anchor * ds_nom, environment=environment, thickness_m=detail.plate_thickness_m
        )
    return out
