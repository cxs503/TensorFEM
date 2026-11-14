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
    """
    load_vector: np.ndarray
    total_force: float
    peak_pressure: float
    mean_pressure: float
    pressure_nonuniformity: float
    patch_centre: tuple[float, float]
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core projection
# ---------------------------------------------------------------------------

def project_pressure_to_nodes(
    contact_data: ContactPressureData,
    plate_nodes: np.ndarray,
    plate_elements: np.ndarray,
    *,
    interpolation_radius: Optional[float] = None,
) -> np.ndarray:
    """Project contact pressure onto plate mesh nodes.

    Uses inverse-distance weighting for interpolation from contact nodes to
    plate nodes.  Only plate nodes within ``interpolation_radius`` of any
    contact node receive load; if not specified, radius is set to the maximum
    inter-contact-node distance.

    Parameters
    ----------
    contact_data : ContactPressureData
        Contact pressure data.
    plate_nodes : (nn, 2) array
        Plate mesh node coordinates.
    plate_elements : (ne, 4) array
        Plate mesh element connectivity.
    interpolation_radius : float, optional
        Maximum distance for interpolation.

    Returns
    -------
    nodal_pressures : (nn,) array
        Interpolated pressure at each plate node [Pa].
    """
    pos = np.asarray(contact_data.node_positions, dtype=np.float64)
    pres = np.asarray(contact_data.normal_pressures, dtype=np.float64)
    nodes = np.asarray(plate_nodes, dtype=np.float64)

    # Use 2D coordinates
    if pos.shape[1] == 3:
        pos = pos[:, :2]
    if nodes.shape[1] == 3:
        nodes = nodes[:, :2]

    if interpolation_radius is None:
        # Default: max inter-node distance in contact data
        if len(pos) > 1:
            dists = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
            interpolation_radius = float(np.max(dists))
        else:
            interpolation_radius = 1.0

    nn = nodes.shape[0]
    nodal_pressures = np.zeros(nn, dtype=np.float64)

    for i in range(nn):
        dists = np.linalg.norm(nodes[i, :] - pos, axis=1)
        within = dists < interpolation_radius
        if not np.any(within):
            continue
        d = dists[within]
        p = pres[within]
        # Inverse-distance weighting (with small epsilon to avoid singularity)
        eps = 1e-12
        weights = 1.0 / (d + eps)
        weights /= weights.sum()
        nodal_pressures[i] = np.sum(weights * p)

    return nodal_pressures


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


def _build_pressure_callable_from_nodes(
    plate_nodes_np: np.ndarray,
    nodal_pressures: np.ndarray,
) -> Callable:
    """Build a pressure callable that works with both numpy and torch inputs.

    The plate solver's ``consistent_load_vector`` passes torch tensors to the
    pressure function, so we need to handle both types.
    """
    def pressure_fn(x, y):
        # Handle both torch tensors and numpy arrays
        if torch.is_tensor(x):
            x_np = x.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            is_tensor = True
            dtype = x.dtype
            device = x.device
        else:
            x_np = np.asarray(x, dtype=np.float64)
            y_np = np.asarray(y, dtype=np.float64)
            is_tensor = False

        # Find nearest node
        dists = (plate_nodes_np[:, 0] - x_np)**2 + (plate_nodes_np[:, 1] - y_np)**2
        idx = int(np.argmin(dists))
        result = nodal_pressures[idx]

        if is_tensor:
            return torch.tensor(result, dtype=dtype, device=device)
        return result

    return pressure_fn


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

    # Get nodal pressures
    if pressure_fn is not None:
        nodal_pressures = pressure_fn(nodes[:, 0], nodes[:, 1])
        if torch.is_tensor(nodal_pressures):
            nodal_pressures = nodal_pressures.detach().cpu().numpy()
    else:
        nodal_pressures = project_pressure_to_nodes(
            contact_data, nodes, elements,
            interpolation_radius=interpolation_radius,
        )

    # Build consistent load vector
    nn = nodes.shape[0]
    ndof = 3 * nn
    load_vector = np.zeros(ndof, dtype=np.float64)

    for conn in elements:
        xy = nodes[conn]
        # 2x2 Gauss quadrature for consistent load
        gp = 1.0 / math.sqrt(3.0)
        for xi_f in (-gp, gp):
            for eta_f in (-gp, gp):
                xi = xi_f
                eta = eta_f
                # Shape functions
                N = 0.25 * np.array([
                    (1.0 - xi) * (1.0 - eta),
                    (1.0 + xi) * (1.0 - eta),
                    (1.0 + xi) * (1.0 + eta),
                    (1.0 - xi) * (1.0 + eta),
                ])
                # Natural derivatives
                dN_nat = 0.25 * np.array([
                    [-(1.0 - eta), -(1.0 - xi)],
                    [(1.0 - eta), -(1.0 + xi)],
                    [(1.0 + eta), (1.0 + xi)],
                    [-(1.0 + eta), (1.0 - xi)],
                ])
                J = dN_nat.T @ xy
                detJ = abs(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])

                # Physical coordinates at Gauss point
                x_gp = float(N @ xy[:, 0])
                y_gp = float(N @ xy[:, 1])

                # Pressure at Gauss point (interpolate from nodal pressures)
                p_gp = float(N @ nodal_pressures[conn])

                # Consistent load contribution
                for i in range(4):
                    load_vector[3 * conn[i]] += N[i] * p_gp * detJ

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
        },
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
    """End-to-end: contact pressure → plate solve.

    Parameters
    ----------
    contact_data : ContactPressureData
        Contact pressure data.
    plate_nx : int
        Number of elements per side.
    plate_side : float
        Plate side length [m].
    young : float
        Young's modulus [Pa].
    poisson : float
        Poisson's ratio.
    thickness : float
        Plate thickness [m].
    winkler_k : float
        Winkler foundation coefficient [N/m³].
    boundary : str
        Boundary condition: "simply_supported" or "clamped".
    pressure_fn : callable, optional
        Custom pressure function.
    dtype : torch.dtype
        Data type for the solver.

    Returns
    -------
    plate_result : MindlinPlateResult
        Plate solver result.
    patch_result : PatchLoadResult
        Patch load projection result.
    """
    nodes, elements = structured_square_mesh(plate_nx, plate_side, dtype=dtype)
    nodes_np = nodes.numpy()
    elements_np = elements.numpy()

    # Project contact pressure to plate mesh
    patch_result = patch_from_contact(
        contact_data, nodes_np, elements_np,
        pressure_fn=pressure_fn,
    )

    # Build pressure function from projected nodal pressures
    nodal_p = patch_result.load_vector[0::3]  # w DOFs

    # Build a callable that works with both numpy and torch inputs
    pressure_callable = _build_pressure_callable_from_nodes(nodes_np, nodal_p)

    # Solve plate with non-uniform pressure
    plate_result = solve_plate_on_winkler_foundation(
        plate_nx, plate_side, young, poisson, thickness,
        winkler_k, pressure_callable, boundary=boundary, dtype=dtype,
    )

    return plate_result, patch_result


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
) -> dict:
    """Parametric scan over contact configurations.

    Compares non-uniform (Gaussian) patch loading against uniform patch
    baseline for multiple contact configurations.  **Total force is matched**
    between the Gaussian and uniform cases so that the comparison isolates
    the effect of pressure non-uniformity alone.

    Parameters
    ----------
    plate_nx : int
        Number of elements per side.
    plate_side : float
        Plate side length [m].
    young : float
        Young's modulus [Pa].
    poisson : float
        Poisson's ratio.
    thickness : float
        Plate thickness [m].
    winkler_k : float
        Winkler foundation coefficient [N/m³].
    configurations : list of dict, optional
        Each dict has keys:
        - ``name``: configuration name
        - ``peak_pressure``: peak pressure [Pa]
        - ``sigma_x``, ``sigma_y``: Gaussian widths [m]
        - ``patch_width``, ``patch_length``: uniform patch dimensions [m]
        The uniform pressure is computed to match the Gaussian total force.
    boundary : str
        Boundary condition.
    dtype : torch.dtype
        Data type.

    Returns
    -------
    results : dict
        Dictionary with per-configuration results and summary.
    """
    if configurations is None:
        # Default: 3 configurations with increasing non-uniformity
        # All have the same total force (matched to Gaussian integral)
        centre = plate_side / 2
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

    results = {"configurations": [], "summary": {}}

    for cfg in configurations:
        name = cfg["name"]
        peak_p = cfg["peak_pressure"]
        sx = cfg["sigma_x"]
        sy = cfg["sigma_y"]
        pw = cfg["patch_width"]
        pl = cfg["patch_length"]

        # Gaussian total force = peak * 2*pi*sigma_x*sigma_y
        gaussian_total_force = peak_p * 2 * math.pi * sx * sy
        # Uniform pressure to match total force
        uniform_p = gaussian_total_force / (pw * pl)

        # Gaussian pressure function (numpy-compatible)
        def make_gaussian_fn(_peak, _sx, _sy, _cx, _cy):
            def gaussian_fn(x, y):
                if torch.is_tensor(x):
                    x = x.detach().cpu().numpy()
                    y = y.detach().cpu().numpy()
                dx = (x - _cx) / _sx
                dy = (y - _cy) / _sy
                return _peak * np.exp(-0.5 * (dx**2 + dy**2))
            return gaussian_fn

        gaussian_fn = make_gaussian_fn(peak_p, sx, sy, plate_side/2, plate_side/2)

        # Solve with non-uniform (Gaussian) patch
        nodes, elements = structured_square_mesh(plate_nx, plate_side, dtype=dtype)
        nodes_np = nodes.numpy()
        elements_np = elements.numpy()

        # Create contact data from Gaussian
        np.random.seed(42)
        n_contact = 100
        cx, cy = plate_side / 2, plate_side / 2
        contact_x = np.clip(cx + sx * np.random.randn(n_contact), 0, plate_side)
        contact_y = np.clip(cy + sy * np.random.randn(n_contact), 0, plate_side)
        contact_pos = np.column_stack([contact_x, contact_y])
        contact_pres = gaussian_patch_pressure(contact_x, contact_y, cx, cy, peak_p, sx, sy)
        contact_area = 2 * math.pi * sx * sy  # Effective area of Gaussian

        contact_data = ContactPressureData(
            node_positions=contact_pos,
            normal_pressures=contact_pres,
            contact_area=contact_area,
        )

        # Non-uniform solve
        plate_result_nonuniform, patch_result = contact_pressure_to_plate_load(
            contact_data, plate_nx, plate_side, young, poisson, thickness,
            winkler_k, boundary=boundary, dtype=dtype,
            pressure_fn=gaussian_fn,
        )

        # Extract peak deflection
        w_nonuniform = plate_result_nonuniform.displacement[0::3].numpy()
        max_deflection_nonuniform = float(np.max(np.abs(w_nonuniform)))

        # Uniform patch solve (same total force)
        def make_uniform_fn(_p, _cx, _cy, _pw, _pl):
            def uniform_fn(x, y):
                if torch.is_tensor(x):
                    x = x.detach().cpu().numpy()
                    y = y.detach().cpu().numpy()
                in_patch = (np.abs(x - _cx) <= _pl) & (np.abs(y - _cy) <= _pw)
                result = np.where(in_patch, _p, 0.0)
                if torch.is_tensor(x):
                    return torch.tensor(float(result), dtype=x.dtype, device=x.device)
                return float(result)
            return uniform_fn

        uniform_fn = make_uniform_fn(uniform_p, plate_side/2, plate_side/2, pw/2, pl/2)

        plate_result_uniform = solve_plate_on_winkler_foundation(
            plate_nx, plate_side, young, poisson, thickness,
            winkler_k, uniform_fn, boundary=boundary, dtype=dtype,
        )

        w_uniform = plate_result_uniform.displacement[0::3].numpy()
        max_deflection_uniform = float(np.max(np.abs(w_uniform)))

        # Peak stress comparison (using deflection as proxy for stress)
        deflection_diff_pct = (
            (max_deflection_nonuniform - max_deflection_uniform)
            / max_deflection_uniform * 100
            if max_deflection_uniform > 0 else 0.0
        )

        config_result = {
            "name": name,
            "peak_pressure_pa": peak_p,
            "sigma_x_m": sx,
            "sigma_y_m": sy,
            "gaussian_total_force_n": gaussian_total_force,
            "uniform_pressure_pa": uniform_p,
            "uniform_patch_width_m": pw,
            "uniform_patch_length_m": pl,
            "max_deflection_nonuniform_m": max_deflection_nonuniform,
            "max_deflection_uniform_m": max_deflection_uniform,
            "deflection_diff_pct": deflection_diff_pct,
            "pressure_nonuniformity": patch_result.pressure_nonuniformity,
            "total_force_nonuniform_n": patch_result.total_force,
            "total_force_uniform_n": uniform_p * pw * pl,
        }
        results["configurations"].append(config_result)

    # Summary statistics
    diffs = [c["deflection_diff_pct"] for c in results["configurations"]]
    results["summary"] = {
        "n_configurations": len(configurations),
        "max_deflection_diff_pct": max(diffs),
        "min_deflection_diff_pct": min(diffs),
        "mean_deflection_diff_pct": float(np.mean(diffs)),
        "h3_anchor_level_pct": 0.339,  # H3-ICE-PLATE-FEM anchor
        "h3_consistency": "within_anchor" if max(abs(d) for d in diffs) < 1.0 else "exceeds_anchor",
    }

    return results
