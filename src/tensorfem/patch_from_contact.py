"""Ice-pressure patch loading from CZM contact solver → plate-grid load projection.

Why this module exists
----------------------
The ice-structure contact solver (CZM / penalty / augmented-Lagrangian) produces
a *contact pressure field* on the interface surface.  Structural assessment of
the ship hull plate requires that pressure to be mapped onto the plate-grid
(FE mesh) as consistent nodal forces.  Before this module the chain was broken:
contact pressure existed but had no caller to feed it into the plate/shell
solver.  This module provides the missing link.

Physical background
-------------------
When an ice sheet contacts a ship hull, the contact pressure is highly
non-uniform: a high-pressure core surrounded by a lower-pressure periphery.
The standard idealisation (used in IACS Polar Class rules) is a *uniform*
patch pressure over a rectangular area.  This module enables both:

1. **Uniform patch**: the classical idealisation, for baseline comparison.
2. **Non-uniform patch from contact**: the actual pressure distribution from
   a CZM contact solve, projected onto the plate grid.

The key question for structural assessment is: *how much does the non-uniformity
of the real contact pressure affect the peak plate stress compared to the
uniform-patch idealisation?*  This module provides the tools to answer that
question quantitatively.

Interface
---------
* :func:`patch_from_contact` -- takes contact pressure data (node positions +
  normal pressures) and projects them as consistent nodal forces onto a plate
  mesh.  Supports both uniform and non-uniform pressure distributions.
* :func:`gaussian_patch_pressure` -- analytical Gaussian pressure distribution
  model for parametric studies (high-pressure core + exponential decay).
* :func:`contact_pressure_to_plate_load` -- end-to-end: contact pressure →
  plate mesh consistent load vector.
* :func:`scan_contact_configurations` -- parametric scan over >= 3 contact
  configurations, returning peak stress comparison (non-uniform vs uniform).

Verification anchors
--------------------
* ``V1``: uniform patch load from :func:`patch_from_contact` matches the
  analytical total force to < 0.1 %.
* ``V2``: Gaussian patch total force matches analytical integral to < 0.5 %.
* ``V3``: non-uniform patch peak stress vs uniform patch peak stress difference
  is consistent with the H3-ICE-PLATE-FEM anchor (≈ 0.339 % level for mild
  non-uniformity).
* ``V4``: end-to-end contact → patch load → plate stress is reproducible.

Scope and limitations
---------------------
* Small-deflection plate theory (Mindlin-Reissner Q4 SRI).
* Contact pressure is assumed known (from a prior CZM / contact solve); this
  module does not solve the contact problem itself.
* The projection uses the plate mesh nodes as target; if the contact mesh and
  plate mesh differ, nearest-node interpolation is used (documented error).
* 2D plate model only (the contact pressure is integrated through the thickness
  direction and applied as transverse load).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from .plate_mindlin import (
    MindlinPlateResult,
    compute_face_stresses,
    solve_plate_on_winkler_foundation,
    structured_square_mesh,
)

__all__ = [
    "ContactPressureData",
    "PatchLoadResult",
    "patch_from_contact",
    "gaussian_patch_pressure",
    "contact_pressure_to_plate_load",
    "scan_contact_configurations",
    "uniform_patch_total_force",
    "project_pressure_to_nodes",
    "peak_stresses",
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContactPressureData:
    """Contact pressure data from a CZM / contact solver.

    Attributes
    ----------
    node_positions : (N, 2) or (N, 3) array
        Contact node positions (x, y) or (x, y, z).
    normal_pressures : (N,) array
        Normal contact pressure at each node [Pa].
    contact_area : float
        Total contact area [m²].
    """
    node_positions: np.ndarray
    normal_pressures: np.ndarray
    contact_area: float

    def __post_init__(self):
        pos = np.asarray(self.node_positions)
        pres = np.asarray(self.normal_pressures)
        if pos.ndim != 2 or pos.shape[1] not in (2, 3):
            raise ValueError("node_positions must be (N, 2) or (N, 3)")
        if pres.shape != (pos.shape[0],):
            raise ValueError("normal_pressures must have one entry per node")
        if self.contact_area <= 0:
            raise ValueError("contact_area must be positive")
        if np.any(pres < 0):
            raise ValueError("normal_pressures must be non-negative")


@dataclass
class PatchLoadResult:
    """Result of projecting contact pressure onto a plate mesh.

    Attributes
    ----------
    load_vector : (ndof,) array
        Consistent nodal force vector (w DOFs only, 3*i entries).
    total_force : float
        Total applied force [N].
    peak_pressure : float
        Maximum pressure in the patch [Pa].
    mean_pressure : float
        Mean pressure over the contact area [Pa].
    pressure_nonuniformity : float
        Coefficient of variation of pressure (std/mean).
    patch_centre : tuple[float, float]
        Centroid of the loaded region.
    metadata : dict
        Additional metadata.
    nodal_pressures : (nn,) array or None
        The **pressure** field [Pa] on the plate nodes that the load vector was
        integrated from.  This is what a downstream plate solve must consume;
        the load vector entries are forces [N] and must never be re-applied as
        pressures.
    """
    load_vector: np.ndarray
    total_force: float
    peak_pressure: float
    mean_pressure: float
    pressure_nonuniformity: float
    patch_centre: tuple[float, float]
    metadata: dict = field(default_factory=dict)
    nodal_pressures: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Core projection
# ---------------------------------------------------------------------------

def _bilinear_field(xs: np.ndarray, ys: np.ndarray, P: np.ndarray, *,
                    outside_zero: bool = True) -> Callable:
    """Bilinear interpolant of a tensor-product pressure grid.

    With ``outside_zero`` the field is zero outside ``[xs[0], xs[-1]] x [ys[0], ys[-1]]``
    (a contact patch does not load the plate outside its footprint); otherwise it
    is clamped at the boundary.
    """
    def field(x, y):
        xa, x_scalar = _as_1d(x)
        ya, y_scalar = _as_1d(y)
        if outside_zero:
            outside = (
                (xa < xs[0]) | (xa > xs[-1]) | (ya < ys[0]) | (ya > ys[-1])
            )
        else:
            outside = np.zeros(xa.shape, dtype=bool)
        i = np.clip(np.searchsorted(xs, xa) - 1, 0, xs.size - 2)
        j = np.clip(np.searchsorted(ys, ya) - 1, 0, ys.size - 2)
        tx = np.clip((xa - xs[i]) / (xs[i + 1] - xs[i]), 0.0, 1.0)
        ty = np.clip((ya - ys[j]) / (ys[j + 1] - ys[j]), 0.0, 1.0)
        val = (
            (1.0 - tx) * (1.0 - ty) * P[j, i]
            + tx * (1.0 - ty) * P[j, i + 1]
            + (1.0 - tx) * ty * P[j + 1, i]
            + tx * ty * P[j + 1, i + 1]
        )
        if outside_zero:
            val = np.where(outside, 0.0, val)
        if x_scalar and y_scalar:
            val = float(np.asarray(val).reshape(-1)[0])
        return val
    return field


def project_pressure_to_nodes(
    contact_data: ContactPressureData,
    plate_nodes: np.ndarray,
    plate_elements: np.ndarray,
    *,
    interpolation_radius: Optional[float] = None,
    method: str = "auto",
) -> np.ndarray:
    """Project the contact pressure field onto the plate nodes [Pa].

    ``method="auto"`` (default) uses a **bilinear interpolant of the contact
    grid** whenever the contact nodes form a tensor-product grid -- which is how
    a CZM / contact solve delivers its interface pressures.  The field is zero
    outside the contact footprint, so the projected total force stays equal to
    the contact force (trapezoid-level accuracy) instead of smearing the patch
    over the whole plate.

    Since m20624 the legacy global inverse-distance weighting is only a fallback
    for genuinely scattered contact nodes, and its averaging radius defaults to
    the local node spacing rather than the largest inter-node distance (the old
    default inflated the projected force by 24 %-666 % on Gaussian patches; see
    ``docs/VERIFY-patch-from-contact-units.md``).

    Parameters
    ----------
    contact_data : ContactPressureData
    plate_nodes : (nn, 2) array
    plate_elements : (ne, 4) array
    interpolation_radius : float, optional
        Averaging radius for the scattered-data fallback.
    method : {"auto", "bilinear", "idw"}
        Force a projection method (``"bilinear"`` raises for scattered data).

    Returns
    -------
    nodal_pressures : (nn,) array
        Pressure at each plate node [Pa].
    """
    pos = np.asarray(contact_data.node_positions, dtype=np.float64)
    pres = np.asarray(contact_data.normal_pressures, dtype=np.float64)
    nodes = np.asarray(plate_nodes, dtype=np.float64)
    if pos.shape[1] == 3:
        pos = pos[:, :2]
    if nodes.shape[1] == 3:
        nodes = nodes[:, :2]

    grid = _structured_grid_index(pos) if pos.shape[0] >= 4 else None
    if method == "auto":
        use_bilinear = grid is not None
    elif method == "bilinear":
        if grid is None:
            raise ValueError("contact nodes are not a tensor-product grid")
        use_bilinear = True
    elif method == "idw":
        use_bilinear = False
    else:
        raise ValueError("method must be 'auto', 'bilinear' or 'idw'")

    if use_bilinear:
        xs, ys, index = grid
        P = pres[index]
        field = _bilinear_field(xs, ys, P, outside_zero=True)
        return np.asarray(field(nodes[:, 0], nodes[:, 1]), dtype=np.float64)

    # ---- scattered-data fallback: local inverse-distance weighting ----
    if pos.shape[0] == 1:
        nodal = np.full(nodes.shape[0], float(pres[0]))
        return nodal
    pair = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(pair, np.inf)
    spacing = float(np.median(np.min(pair, axis=1)))
    radius = interpolation_radius if interpolation_radius is not None else 2.0 * spacing

    nn = nodes.shape[0]
    nodal = np.zeros(nn, dtype=np.float64)
    for i in range(nn):
        dists = np.linalg.norm(nodes[i, :] - pos, axis=1)
        within = dists <= radius
        if not np.any(within):
            continue
        d = dists[within]
        weights = 1.0 / (d + 1e-12)
        weights /= weights.sum()
        nodal[i] = float(np.sum(weights * pres[within]))
    return nodal


def uniform_patch_total_force(
    pressure: float,
    patch_width: float,
    patch_length: float,
) -> float:
    """Total force of a uniform rectangular patch.

    Parameters
    ----------
    pressure : float
        Uniform pressure [Pa].
    patch_width : float
        Patch width (y-direction) [m].
    patch_length : float
        Patch length (x-direction) [m].

    Returns
    -------
    total_force : float
        Total force [N].
    """
    return pressure * patch_width * patch_length


def gaussian_patch_pressure(
    x: np.ndarray,
    y: np.ndarray,
    centre_x: float,
    centre_y: float,
    peak_pressure: float,
    sigma_x: float,
    sigma_y: float,
) -> np.ndarray:
    """Gaussian pressure distribution model.

    Models a high-pressure core with exponential decay, typical of ice-structure
    contact.  The total force is ``peak_pressure * 2*pi*sigma_x*sigma_y``.

    Parameters
    ----------
    x, y : array
        Coordinates at which to evaluate pressure.
    centre_x, centre_y : float
        Centre of the Gaussian patch.
    peak_pressure : float
        Peak pressure at the centre [Pa].
    sigma_x, sigma_y : float
        Standard deviations in x and y [m].

    Returns
    -------
    pressure : array
        Pressure at each point [Pa].
    """
    dx = (x - centre_x) / sigma_x
    dy = (y - centre_y) / sigma_y
    return peak_pressure * np.exp(-0.5 * (dx**2 + dy**2))


def _structured_grid_index(plate_nodes_np: np.ndarray):
    """Return ``(xs, ys, index)`` for a tensor-product node grid, else ``None``.

    ``index[j, i]`` is the node number sitting at ``(xs[i], ys[j])``.
    """
    nodes = np.asarray(plate_nodes_np, dtype=np.float64)
    xs = np.unique(np.round(nodes[:, 0], 12))
    ys = np.unique(np.round(nodes[:, 1], 12))
    if xs.size < 2 or ys.size < 2 or xs.size * ys.size != nodes.shape[0]:
        return None
    ix = np.searchsorted(xs, np.round(nodes[:, 0], 12))
    iy = np.searchsorted(ys, np.round(nodes[:, 1], 12))
    index = np.full((ys.size, xs.size), -1, dtype=np.int64)
    index[iy, ix] = np.arange(nodes.shape[0])
    if np.any(index < 0):
        return None
    return xs, ys, index


def _as_1d(values) -> tuple[np.ndarray, bool]:
    """Normalise a scalar/array/tensor argument to ``(1-D float array, was_scalar)``."""
    is_tensor = torch.is_tensor(values)
    if is_tensor:
        arr = values.detach().cpu().numpy()
    else:
        arr = np.asarray(values)
    arr = np.atleast_1d(arr).astype(np.float64)
    return arr, bool(arr.size == 1 and (is_tensor or np.ndim(values) == 0))


def _build_pressure_callable_from_nodes(
    plate_nodes_np: np.ndarray,
    nodal_pressures: np.ndarray,
) -> Callable:
    """Rebuild a pressure callable [Pa] from the *nodal pressure field*.

    The plate grid is a tensor-product grid, so the field is interpolated
    bilinearly (clamped to the domain); non tensor-product meshes fall back to
    nearest-node lookup.

    Notes
    -----
    The argument must be a pressure field [Pa].  Passing the consistent nodal
    *forces* (``load_vector[0::3]``, [N]) re-integrates them as a pressure and
    scales the applied load by one element area -- that defect (a factor
    ``element_area`` on the deflection, i.e. ~144x at nx=12) is fixed in
    mission m20624; see ``docs/VERIFY-patch-from-contact-units.md``.
    """
    nodes_np = np.asarray(plate_nodes_np, dtype=np.float64)
    p_nodes = np.asarray(nodal_pressures, dtype=np.float64)
    if p_nodes.shape != (nodes_np.shape[0],):
        raise ValueError("nodal_pressures must hold one value per plate node")
    grid = _structured_grid_index(nodes_np)

    if grid is not None:
        xs, ys, index = grid
        P = p_nodes[index]  # (ny, nx)

        def pressure_fn(x, y):
            xa, x_scalar = _as_1d(x)
            ya, y_scalar = _as_1d(y)
            i = np.clip(np.searchsorted(xs, xa) - 1, 0, xs.size - 2)
            j = np.clip(np.searchsorted(ys, ya) - 1, 0, ys.size - 2)
            tx = np.clip((xa - xs[i]) / (xs[i + 1] - xs[i]), 0.0, 1.0)
            ty = np.clip((ya - ys[j]) / (ys[j + 1] - ys[j]), 0.0, 1.0)
            val = (
                (1.0 - tx) * (1.0 - ty) * P[j, i]
                + tx * (1.0 - ty) * P[j, i + 1]
                + (1.0 - tx) * ty * P[j + 1, i]
                + tx * ty * P[j + 1, i + 1]
            )
            if x_scalar and y_scalar:
                val = float(np.asarray(val).reshape(-1)[0])
            if torch.is_tensor(x):
                return torch.as_tensor(val, dtype=x.dtype, device=x.device)
            return val

        return pressure_fn

    def pressure_fn_nearest(x, y):
        xa, x_scalar = _as_1d(x)
        ya, y_scalar = _as_1d(y)
        d2 = (nodes_np[None, :, 0] - xa[:, None]) ** 2 + (
            nodes_np[None, :, 1] - ya[:, None]
        ) ** 2
        val = p_nodes[np.argmin(d2, axis=1)]
        if x_scalar and y_scalar:
            val = float(np.asarray(val).reshape(-1)[0])
        if torch.is_tensor(x):
            return torch.as_tensor(val, dtype=x.dtype, device=x.device)
        return val

    return pressure_fn_nearest


def _integrated_load(
    plate_nodes_np: np.ndarray,
    plate_elements: np.ndarray,
    pressure_fn: Callable,
) -> float:
    """Integral of a pressure field over the plate [N], 2x2 Gauss per Q4 element."""
    nodes = np.asarray(plate_nodes_np, dtype=np.float64)
    gp = 1.0 / math.sqrt(3.0)
    total = 0.0
    for conn in np.asarray(plate_elements, dtype=np.int64):
        xy = nodes[conn]
        for xi in (-gp, gp):
            for eta in (-gp, gp):
                N = 0.25 * np.array([
                    (1.0 - xi) * (1.0 - eta),
                    (1.0 + xi) * (1.0 - eta),
                    (1.0 + xi) * (1.0 + eta),
                    (1.0 - xi) * (1.0 + eta),
                ])
                dN_nat = 0.25 * np.array([
                    [-(1.0 - eta), -(1.0 - xi)],
                    [(1.0 - eta), -(1.0 + xi)],
                    [(1.0 + eta), (1.0 + xi)],
                    [-(1.0 + eta), (1.0 - xi)],
                ])
                J = dN_nat.T @ xy
                detJ = abs(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
                q = pressure_fn(float(N @ xy[:, 0]), float(N @ xy[:, 1]))
                if torch.is_tensor(q):
                    q = float(q.detach().cpu())
                total += float(q) * detJ
    return total


def peak_stresses(result: MindlinPlateResult) -> dict:
    """Peak plate face stresses of a Mindlin solve [MPa].

    Uses :func:`tensorfem.plate_mindlin.compute_face_stresses` (per-element
    centroid curvature -> top/bottom face stress) and reports the peak von Mises
    and peak bending stress magnitudes, so that "peak stress" is a real stress
    measure and not a deflection proxy.
    """
    faces = compute_face_stresses(result)
    if not faces:
        raise ValueError("compute_face_stresses returned no faces")
    vm_best = None
    mx_best = None
    for face in faces:
        s = face["stress"]
        sx, sy, txy = s["sigma_x"], s["sigma_y"], s["tau_xy"]
        sigma_vm = math.sqrt(max(sx * sx + sy * sy - sx * sy + 3.0 * txy * txy, 0.0))
        if vm_best is None or sigma_vm > vm_best["sigma_vm"]:
            vm_best = {"sigma_vm": sigma_vm, "centroid": face["centroid"]}
        if mx_best is None or abs(sx) > mx_best["abs_sigma_x"]:
            mx_best = {"abs_sigma_x": abs(sx), "centroid": face["centroid"]}
    return {
        "sigma_vm_MPa": vm_best["sigma_vm"] / 1e6,
        "sigma_vm_at_m": [vm_best["centroid"][0], vm_best["centroid"][1]],
        "sigma_x_MPa": mx_best["abs_sigma_x"] / 1e6,
        "sigma_x_at_m": [mx_best["centroid"][0], mx_best["centroid"][1]],
    }


def _assemble_gauss_load(
    nodes: np.ndarray,
    elements: np.ndarray,
    nodal_pressures: np.ndarray,
) -> np.ndarray:
    """Fallback consistent load vector from nodal pressures (2x2 Gauss per element)."""
    nn = nodes.shape[0]
    load_vector = np.zeros(3 * nn, dtype=np.float64)
    gp = 1.0 / math.sqrt(3.0)

    for conn in np.asarray(elements, dtype=np.int64):
        xy = nodes[conn]
        for xi in (-gp, gp):
            for eta in (-gp, gp):
                N = 0.25 * np.array([
                    (1.0 - xi) * (1.0 - eta),
                    (1.0 + xi) * (1.0 - eta),
                    (1.0 + xi) * (1.0 + eta),
                    (1.0 - xi) * (1.0 + eta),
                ])
                dN_nat = 0.25 * np.array([
                    [-(1.0 - eta), -(1.0 - xi)],
                    [(1.0 - eta), -(1.0 + xi)],
                    [(1.0 + eta), (1.0 + xi)],
                    [-(1.0 + eta), (1.0 - xi)],
                ])
                J = dN_nat.T @ xy
                detJ = abs(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
                p_gp = float(N @ nodal_pressures[conn])
                for i in range(4):
                    load_vector[3 * conn[i]] += N[i] * p_gp * detJ
    return load_vector


def patch_from_contact(
    contact_data: ContactPressureData,
    plate_nodes: np.ndarray,
    plate_elements: np.ndarray,
    *,
    pressure_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    interpolation_radius: Optional[float] = None,
) -> PatchLoadResult:
    """Project contact pressure onto plate mesh as consistent nodal forces.

    This is the core interface: it takes contact pressure data and produces
    a load vector suitable for the plate solver.

    Parameters
    ----------
    contact_data : ContactPressureData
        Contact pressure data from CZM / contact solver.
    plate_nodes : (nn, 2) array
        Plate mesh node coordinates.
    plate_elements : (ne, 4) array
        Plate mesh element connectivity.
    pressure_fn : callable, optional
        If provided, overrides the contact data pressure with a custom
        pressure function ``pressure_fn(x, y) -> pressure``.
    interpolation_radius : float, optional
        Maximum distance for interpolation.

    Returns
    -------
    result : PatchLoadResult
        Projected load vector and metadata.
    """
    pos = np.asarray(contact_data.node_positions, dtype=np.float64)
    pres = np.asarray(contact_data.normal_pressures, dtype=np.float64)
    nodes = np.asarray(plate_nodes, dtype=np.float64)
    elements = np.asarray(plate_elements, dtype=np.int64)

    if pos.shape[1] == 3:
        pos = pos[:, :2]
    if nodes.shape[1] == 3:
        nodes = nodes[:, :2]

    # Get the pressure field on the plate nodes [Pa] and, when the contact data
    # is a tensor-product grid, the field itself (integrating the field directly
    # avoids a second interpolation of an already interpolated field).
    field_fn: Optional[Callable] = None
    contact_grid = _structured_grid_index(pos) if pos.shape[0] >= 4 else None
    if pressure_fn is not None:
        nodal_pressures = pressure_fn(nodes[:, 0], nodes[:, 1])
        if torch.is_tensor(nodal_pressures):
            nodal_pressures = nodal_pressures.detach().cpu().numpy()
        nodal_pressures = np.asarray(nodal_pressures, dtype=np.float64)
        field_fn = pressure_fn
    else:
        nodal_pressures = project_pressure_to_nodes(
            contact_data, nodes, elements,
            interpolation_radius=interpolation_radius,
        )
        if contact_grid is not None:
            xs_c, ys_c, index_c = contact_grid
            field_fn = _bilinear_field(xs_c, ys_c, pres[index_c], outside_zero=True)

    # Build consistent load vector
    nn = nodes.shape[0]
    ndof = 3 * nn

    # The plate solver re-integrates a pressure callable element by element
    # (2x2 Gauss), so the load vector is built from the same element bilinear
    # interpolant of the nodal pressures -- otherwise the two disagree at patch
    # edges, where the contact field is discontinuous.
    load_vector = _assemble_gauss_load(nodes, elements, nodal_pressures)

    # Compute metadata
    total_force = float(np.sum(load_vector[0::3]))
    peak_pressure = float(np.max(pres))
    mean_pressure = float(np.mean(pres))
    std_pressure = float(np.std(pres))
    nonuniformity = std_pressure / mean_pressure if mean_pressure > 0 else 0.0

    # Patch centroid (pressure-weighted)
    if pos.shape[0] > 0:
        weights = pres / pres.sum() if pres.sum() > 0 else np.ones(len(pres)) / len(pres)
        cx = float(np.sum(weights * pos[:, 0]))
        cy = float(np.sum(weights * pos[:, 1]))
    else:
        cx, cy = 0.0, 0.0

    return PatchLoadResult(
        load_vector=load_vector,
        total_force=total_force,
        peak_pressure=peak_pressure,
        mean_pressure=mean_pressure,
        pressure_nonuniformity=nonuniformity,
        patch_centre=(cx, cy),
        metadata={
            "n_contact_nodes": len(pres),
            "n_plate_nodes": nn,
            "n_plate_elements": len(elements),
            # the pressure field [Pa] the load vector was integrated from, so a
            # downstream solve can re-use it instead of re-interpolating
            "field_fn": field_fn,
            "projection_method": (
                "exact_contact_grid_bilinear" if field_fn is not None
                else "plate_node_interpolation"
            ),
        },
        nodal_pressures=np.asarray(nodal_pressures, dtype=np.float64),
    )

def contact_pressure_to_plate_load(
    contact_data: ContactPressureData,
    plate_nx: int,
    plate_side: float,
    young: float,
    poisson: float,
    thickness: float,
    winkler_k: float,
    *,
    boundary: str = "simply_supported",
    pressure_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    dtype=torch.float64,
) -> tuple[MindlinPlateResult, PatchLoadResult]:
    """End-to-end: contact pressure -> projected plate load -> plate solve.

    The contact pressure is projected onto the plate nodes (:func:`patch_from_contact`),
    the resulting **nodal pressure field** [Pa] is interpolated bilinearly inside
    each element, and the plate is solved with the merged Mindlin Q4 solver.

    The total force actually applied by this chain is integrated independently
    and stored in ``PatchLoadResult.metadata`` as ``applied_force`` together with
    ``load_balance_rel`` (relative force imbalance against the projected nodal
    loads), so the unit consistency of the chain is auditable from every result.

    Parameters
    ----------
    contact_data : ContactPressureData
        Contact pressure data from a CZM / contact solve.
    plate_nx : int
        Number of elements per side.
    plate_side : float
        Plate side length [m].
    young, poisson, thickness : float
        Plate material / thickness.
    winkler_k : float
        Winkler foundation coefficient [N/m^3] (0 for a free plate).
    boundary : str
        "simply_supported" or "clamped".
    pressure_fn : callable, optional
        Custom pressure field [Pa] overriding the contact data (diagnostics).
    dtype : torch.dtype
        Solver dtype.

    Returns
    -------
    plate_result : MindlinPlateResult
    patch_result : PatchLoadResult
        Carries ``nodal_pressures`` [Pa] and the force-balance audit.
    """
    nodes, elements = structured_square_mesh(plate_nx, plate_side, dtype=dtype)
    nodes_np = nodes.numpy()
    elements_np = elements.numpy()

    patch_result = patch_from_contact(
        contact_data, nodes_np, elements_np, pressure_fn=pressure_fn,
    )
    if patch_result.nodal_pressures is None:  # pragma: no cover - defensive
        raise RuntimeError("patch_from_contact did not return a nodal pressure field")

    # Solve with the same interpolant the load vector was built from, so the
    # applied load and the projected nodal loads use identical quadrature.
    pressure_callable = _build_pressure_callable_from_nodes(
        nodes_np, patch_result.nodal_pressures,
    )

    # Independent audit of the applied force (units are Pa * m^2 = N).
    applied_force = _integrated_load(nodes_np, elements_np, pressure_callable)
    reference_force = patch_result.total_force
    patch_result.metadata["applied_force"] = applied_force
    patch_result.metadata["load_balance_rel"] = (
        applied_force - reference_force
    ) / reference_force if reference_force > 0.0 else 0.0

    plate_result = solve_plate_on_winkler_foundation(
        plate_nx, plate_side, young, poisson, thickness,
        winkler_k, pressure_callable, boundary=boundary, dtype=dtype,
    )
    return plate_result, patch_result


def _gaussian_contact_grid(
    centre_x: float,
    centre_y: float,
    sigma_x: float,
    sigma_y: float,
    peak_pressure: float,
    plate_side: float,
    n_per_axis: int,
):
    """Deterministic contact-node cloud for a Gaussian pressure patch.

    Sample the analytical field on a ``n_per_axis^2`` grid spanning
    ``centre +- 3 sigma`` (clipped to the plate), i.e. 99.7 % of the force is
    inside the sampled window.  The nodal values are the *exact* analytical
    pressures, so the residual projection error is the discretisation of the
    contact data itself -- reported per configuration, not hidden.
    """
    lo_x = max(0.0, centre_x - 3.0 * sigma_x)
    hi_x = min(plate_side, centre_x + 3.0 * sigma_x)
    lo_y = max(0.0, centre_y - 3.0 * sigma_y)
    hi_y = min(plate_side, centre_y + 3.0 * sigma_y)
    xs = np.linspace(lo_x, hi_x, n_per_axis)
    ys = np.linspace(lo_y, hi_y, n_per_axis)
    xx, yy = np.meshgrid(xs, ys)
    pos = np.column_stack([xx.ravel(), yy.ravel()])
    pres = gaussian_patch_pressure(
        pos[:, 0], pos[:, 1], centre_x, centre_y, peak_pressure, sigma_x, sigma_y,
    )
    return pos, pres


def scan_contact_configurations(
    plate_nx: int,
    plate_side: float,
    young: float,
    poisson: float,
    thickness: float,
    winkler_k: float,
    *,
    configurations: Optional[Sequence[dict]] = None,
    boundary: str = "simply_supported",
    dtype=torch.float64,
    n_contact_per_axis: int = 21,
) -> dict:
    """Scan >= 3 contact configurations: non-uniform patch vs uniform patch.

    For every configuration three solves are run on the same plate mesh:

    ``projection``
        contact data (CZM-style node pressures) -> :func:`patch_from_contact` ->
        :func:`contact_pressure_to_plate_load`.  **This is the end-to-end chain**
        (contact -> patch load -> plate stress); no analytical shortcut.
    ``analytic_field``
        the same Gaussian pressure evaluated exactly at the quadrature points.
        Used only to separate the projection error from the plate-solver error.
    ``uniform``
        a uniform patch with the *same total force and the same footprint* as the
        projected patch (the IACS/JOUR-ICE-STRUCT idealisation), so the delta
        isolates pressure non-uniformity.

    Reported quantities are **peak face stresses** (:func:`peak_stresses`:
    von Mises and bending stress from the curvature field) as well as peak
    deflections; the machine-readable summary carries the non-uniform vs uniform
    stress deltas, the projection error, and the force-balance audit.

    Parameters
    ----------
    plate_nx : int
        Elements per side.
    plate_side : float
        Plate side length [m].
    young, poisson, thickness : float
        Plate material / thickness.
    winkler_k : float
        Winkler foundation coefficient [N/m^3] (0 for a free plate).
    configurations : sequence of dict, optional
        Keys: ``name``, ``peak_pressure`` [Pa], ``sigma_x``, ``sigma_y`` [m],
        ``patch_width``, ``patch_length`` [m] (uniform reference footprint).
    boundary : str
        Plate boundary condition.
    dtype : torch.dtype
        Solver dtype.
    n_contact_per_axis : int
        Contact-node resolution per axis of the Gaussian sampling window.

    Returns
    -------
    results : dict
        ``{"configurations": [...], "summary": {...}}``, JSON-serialisable.
    """
    if configurations is None:
        # Default: 3 configurations with increasing non-uniformity.
        configurations = [
            {
                "name": "mild_nonuniform",
                "peak_pressure": 5.0e6,
                "sigma_x": 0.15 * plate_side,
                "sigma_y": 0.15 * plate_side,
                "patch_width": 0.3 * plate_side,
                "patch_length": 0.3 * plate_side,
            },
            {
                "name": "moderate_nonuniform",
                "peak_pressure": 8.0e6,
                "sigma_x": 0.10 * plate_side,
                "sigma_y": 0.10 * plate_side,
                "patch_width": 0.2 * plate_side,
                "patch_length": 0.2 * plate_side,
            },
            {
                "name": "sharp_nonuniform",
                "peak_pressure": 15.0e6,
                "sigma_x": 0.05 * plate_side,
                "sigma_y": 0.05 * plate_side,
                "patch_width": 0.1 * plate_side,
                "patch_length": 0.1 * plate_side,
            },
        ]

    def gaussian_fn_for(peak, sx, sy, cx, cy):
        def gaussian_fn(x, y):
            if torch.is_tensor(x):
                x = x.detach().cpu().numpy()
                y = y.detach().cpu().numpy()
            return gaussian_patch_pressure(x, y, cx, cy, peak, sx, sy)
        return gaussian_fn

    def uniform_fn_for(pressure, cx, cy, half_w, half_l):
        def uniform_fn(x, y):
            if torch.is_tensor(x):
                x_np = x.detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()
            else:
                x_np = np.asarray(x, dtype=np.float64)
                y_np = np.asarray(y, dtype=np.float64)
            inside = (np.abs(x_np - cx) <= half_l) & (np.abs(y_np - cy) <= half_w)
            values = np.where(inside, pressure, 0.0).astype(np.float64)
            if np.ndim(values) == 0:
                values = float(values)
            if torch.is_tensor(x):
                return torch.as_tensor(values, dtype=x.dtype, device=x.device)
            return values
        return uniform_fn

    centre = plate_side / 2.0
    rows: list[dict] = []

    for cfg in configurations:
        peak_p = float(cfg["peak_pressure"])
        sx = float(cfg["sigma_x"])
        sy = float(cfg["sigma_y"])
        patch_width = float(cfg["patch_width"])
        patch_length = float(cfg["patch_length"])

        analytic_total_force = peak_p * 2.0 * math.pi * sx * sy

        # --- contact data as a CZM solve would deliver it ---
        contact_pos, contact_pres = _gaussian_contact_grid(
            centre, centre, sx, sy, peak_p, plate_side, n_contact_per_axis,
        )
        contact_data = ContactPressureData(
            node_positions=contact_pos,
            normal_pressures=contact_pres,
            contact_area=2.0 * math.pi * sx * sy,
        )

        # --- (1) end-to-end: contact -> projected patch load -> plate ---
        plate_proj, patch_proj = contact_pressure_to_plate_load(
            contact_data, plate_nx, plate_side, young, poisson, thickness,
            winkler_k, boundary=boundary, dtype=dtype,
        )
        stress_proj = peak_stresses(plate_proj)
        w_proj = float(np.max(np.abs(plate_proj.displacement[0::3].numpy())))

        # --- (2) diagnostic: exact analytic pressure field on the same mesh ---
        plate_ana, _ = contact_pressure_to_plate_load(
            contact_data, plate_nx, plate_side, young, poisson, thickness,
            winkler_k, boundary=boundary, dtype=dtype,
            pressure_fn=gaussian_fn_for(peak_p, sx, sy, centre, centre),
        )
        stress_ana = peak_stresses(plate_ana)
        w_ana = float(np.max(np.abs(plate_ana.displacement[0::3].numpy())))

        # --- (3) uniform reference: same total force, same footprint ---
        total_force = patch_proj.total_force
        uniform_p = total_force / (patch_width * patch_length)
        uniform_fn = uniform_fn_for(
            uniform_p, centre, centre, patch_width / 2.0, patch_length / 2.0,
        )
        plate_uni = solve_plate_on_winkler_foundation(
            plate_nx, plate_side, young, poisson, thickness,
            winkler_k, uniform_fn, boundary=boundary, dtype=dtype,
        )
        stress_uni = peak_stresses(plate_uni)
        w_uni = float(np.max(np.abs(plate_uni.displacement[0::3].numpy())))

        def pct(a, b):
            return 100.0 * (a / b - 1.0) if b else float("nan")

        rows.append({
            "name": cfg["name"],
            "peak_pressure_MPa": peak_p / 1e6,
            "sigma_x_m": sx,
            "sigma_y_m": sy,
            "contact_grid": [n_contact_per_axis, n_contact_per_axis],
            "analytic_total_force_N": analytic_total_force,
            "total_force_nonuniform_N": total_force,
            "uniform_pressure_MPa": uniform_p / 1e6,
            "uniform_patch_width_m": patch_width,
            "uniform_patch_length_m": patch_length,
            "total_force_uniform_N": uniform_p * patch_width * patch_length,
            "projection_total_force_error_pct": pct(total_force, analytic_total_force),
            "load_balance_rel": patch_proj.metadata.get("load_balance_rel", float("nan")),
            "pressure_nonuniformity": patch_proj.pressure_nonuniformity,
            # peak deflection (kept for continuity with the earlier artifact)
            "max_deflection_nonuniform_m": w_proj,
            "max_deflection_uniform_m": w_uni,
            "deflection_diff_pct": pct(w_proj, w_uni),
            # peak stress: the acceptance quantity
            "peak_stress_projected": stress_proj,
            "peak_stress_uniform": stress_uni,
            "peak_stress_analytic_field": stress_ana,
            "delta_nonuniform_over_uniform_pct": {
                "sigma_vm": pct(stress_proj["sigma_vm_MPa"], stress_uni["sigma_vm_MPa"]),
                "sigma_x": pct(stress_proj["sigma_x_MPa"], stress_uni["sigma_x_MPa"]),
                "deflection": pct(w_proj, w_uni),
            },
            "projection_error_pct": {
                "sigma_vm": pct(stress_proj["sigma_vm_MPa"], stress_ana["sigma_vm_MPa"]),
                "sigma_x": pct(stress_proj["sigma_x_MPa"], stress_ana["sigma_x_MPa"]),
                "deflection": pct(w_proj, w_ana),
            },
        })

    delta_vm = [abs(r["delta_nonuniform_over_uniform_pct"]["sigma_vm"]) for r in rows]
    delta_mx = [abs(r["delta_nonuniform_over_uniform_pct"]["sigma_x"]) for r in rows]
    proj_vm = [abs(r["projection_error_pct"]["sigma_vm"]) for r in rows]
    proj_mx = [abs(r["projection_error_pct"]["sigma_x"]) for r in rows]
    balance = [abs(r["load_balance_rel"]) for r in rows]

    h3_anchor_pct = 0.339
    max_proj_mx = max(proj_mx)
    results = {
        "configurations": rows,
        "summary": {
            "n_configurations": len(rows),
            "max_abs_sigma_vm_diff_pct": max(delta_vm),
            "max_abs_sigma_x_diff_pct": max(delta_mx),
            "min_abs_sigma_x_diff_pct": min(delta_mx),
            "max_abs_projection_error_sigma_vm_pct": max(proj_vm),
            "max_abs_projection_error_sigma_x_pct": max_proj_mx,
            "max_abs_load_balance_rel": max(balance),
            "max_deflection_diff_pct": max(
                abs(r["deflection_diff_pct"]) for r in rows
            ),
            # The H3-ICE-PLATE-FEM anchor (0.339 % = FEM vs Kelvin series,
            # benchmarks/artifacts/ice_plate_flexural_limit/evidence.json) is an
            # *accuracy* scale for the solution chain, NOT a bound on the
            # non-uniformity effect.  It is compared here against the projection
            # error (the only error of this module's own chain).
            "h3_anchor_level_pct": h3_anchor_pct,
            "h3_anchor_meaning": (
                "FEM-vs-Kelvin analytical error of the H3-ICE-PLATE-FEM anchor; "
                "used as the accuracy scale for this chain, not as a tolerance "
                "on the non-uniformity effect"
            ),
            "h3_consistency": "within_anchor" if max_proj_mx <= h3_anchor_pct else "exceeds_anchor",
            "h3_consistency_basis": (
                "max |projection-path peak bending stress vs analytic-pressure-field "
                "peak bending stress| on the same mesh"
            ),
            "all_configurations_valid": all(
                r["peak_stress_projected"]["sigma_x_MPa"] > 0.0 for r in rows
            ),
        },
    }
    return results
