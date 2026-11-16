"""Wet modal analysis of a baffled panel with a *nonlocal* fluid operator.

Why this module exists
----------------------
:mod:`tensorfem.fsi_wet_modal` predicts the wet/dry frequency ratio of a
simply supported panel from a **single-mode** added-mass ratio,

    f_wet / f_dry = 1 / sqrt(1 + mu),      mu = M_a / M_s,

where ``M_a`` is either the exact baffled-panel Rayleigh integral, its
``rho_f / k_mn`` asymptote, or a panel-discretised operator evaluated *on the
analytic sine mode*.  All three share one assumption: the wet mode shape is the
dry mode shape and only its diagonal added mass matters.  That assumption is
never checked against a structural discretisation, so the "wet modal" number is
partly an input rather than a result.

This module closes that gap.  The dry Mindlin plate eigenproblem is solved with
:func:`tensorfem.plate_vibration.solve_plate_vibration`; the *same* panel mesh
carries the numerical fluid operator
:func:`tensorfem.hydro_added_mass.dipole_added_mass_matrix` (low-frequency
baffled panel, exact equal-area self term, ``rho/(2 pi |x - x'|)`` kernel).
The dry eigenvectors are sampled at element centroids and the full (non-diagonal)
added-mass coupling matrix is formed in the dry modal basis,

    A_ij = phi_i^T M_hydro phi_j ,
    (diag(w_i^2 g_i) - w^2 [diag(g_i) + A]) x = 0 ,   g_i = phi_i^T M phi_i ,

and the reduced generalised eigenproblem is solved for the wet frequencies.  No
analytic mode shape and no analytic added mass enter the wet result, so the
wet/dry ratio becomes a *predictable* quantity that can be compared with the
closed forms on equal terms.

Approximations kept (and their measured validity, see the W32 artifact)
-----------------------------------------------------------------------
* The fluid is incompressible and the panel sits in an infinite rigid baffle
  (low-frequency, ``k_0 a << 1``), matching the analytic kernels it is
  compared against.
* The modal truncation is the ``modes`` argument.  Convergence in the number of
  retained dry modes is reported by :func:`wet_plate_modal_convergence`.
* Element centroids carry the fluid panels; centroid sampling of a bilinear
  Mindlin mode is exact at the centroid (shape functions equal ``1/4``).
* Diagonalisation of the coupled problem is *not* assumed: the returned
  ``mu_modal_diag`` and ``coupling_ratio`` quantify how far the coupled result
  is from the single-mode ``1/sqrt(1+mu)`` prediction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.linalg
import torch

from .hydro_added_mass import dipole_added_mass_matrix
from .plate_vibration import (
    BOUNDARY_SIMPLE,
    q4_plate_mass_matrix,
    solve_plate_vibration,
)


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _count(name: str, value: int) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


# ---------------------------------------------------------------------------
# Crude added-mass approximations used as comparison kernels
# ---------------------------------------------------------------------------


def piston_added_mass_per_area(area: float, rho_fluid: float) -> float:
    """"Flat-plate ellipsoid" (equivalent piston) added mass per unit area.

    A simply supported panel mode is replaced by a rigid circular piston of the
    same area, ``a_eq = sqrt(A/pi)``.  The low-frequency radiation mass of a
    baffled piston is ``8 rho a_eq^3 / 3`` (Rayleigh), so the smeared value is

        m_a = 8 rho_f a_eq / (3 pi) = 8 rho_f sqrt(A/pi) / (3 pi) .

    This is the classical "the plate pushes a slug of water as a rigid body'
    engineering estimate; it ignores the modal wave number and therefore
    overestimates the added mass of the low-order panel modes.  It is kept here
    precisely so the size of that error can be measured instead of assumed.
    """
    area = _positive("area", area)
    rho_fluid = _positive("rho_fluid", rho_fluid)
    a_eq = math.sqrt(area / math.pi)
    return 8.0 * rho_fluid * a_eq / (3.0 * math.pi)


def prolate_spheroid_transverse_added_mass(
    radius: float, length: float, rho_fluid: float
) -> dict:
    """Transverse added mass of a prolate spheroid enveloping a cylinder.

    Semi-axes ``a = length/2`` (axis), ``b = radius``; eccentricity
    ``e = sqrt(1 - (b/a)^2)``.  Motion perpendicular to the axis (Lamb 1932,
    §114; Newman 1977):

        beta0 = 1/e^2 - ((1 - e^2)/(2 e^3)) ln((1+e)/(1-e)),
        k2    = beta0 / (2 - beta0),
        m_a   = k2 * rho_f * (4/3) pi a b^2 .

    Used as the "ellipsoid approximation" of the cylinder added mass: as
    ``e -> 1`` it gives ``m_a -> rho_f (4/3) pi a b^2 = (2/3) rho_f pi R^2 L``,
    i.e. two thirds of the exact two-dimensional ``rho_f pi R^2 L`` because the
    spheroid has pointed ends and no end-effect correction.  For a sphere
    (``e -> 0``) it reproduces the exact ``0.5 m_displaced``.
    """
    radius = _positive("radius", radius)
    length = _positive("length", length)
    rho_fluid = _positive("rho_fluid", rho_fluid)
    semi_axis = 0.5 * length
    if radius >= semi_axis:
        raise ValueError("a prolate spheroid needs length/2 > radius")
    b = radius
    a = semi_axis
    e = math.sqrt(1.0 - (b / a) ** 2)
    log_term = math.log((1.0 + e) / (1.0 - e))
    beta0 = 1.0 / e ** 2 - (1.0 - e ** 2) / (2.0 * e ** 3) * log_term
    k2 = beta0 / (2.0 - beta0)
    displaced = rho_fluid * (4.0 / 3.0) * math.pi * a * b ** 2
    return {
        "eccentricity": e,
        "beta0": beta0,
        "k2": k2,
        "displaced_mass": displaced,
        "added_mass": k2 * displaced,
        "ratio_to_2d_cylinder": k2 * displaced / (rho_fluid * math.pi * b ** 2 * length),
    }


# ---------------------------------------------------------------------------
# Wet FE-modal solve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WetPlateFEModalResult:
    """Coupled wet modal result of a baffled, simply supported panel."""

    mode_numbers: tuple[tuple[int, int], ...]
    dry_hz_fe: tuple[float, ...]
    wet_hz_fe: tuple[float, ...]
    ratio_fe: tuple[float, ...]
    mu_modal_diag: tuple[float, ...]
    coupling_ratio: tuple[float, ...]
    dry_hz_navier: tuple[float, ...]
    length: float
    width: float
    thickness: float
    rho_fluid: float
    rho_struct: float
    fe_mesh: tuple[int, int]
    collapsed_modes: int
    # Field data for post-processing (not part of summary()/JSON).
    nodes: np.ndarray | None = None
    elements: np.ndarray | None = None
    dry_mode_shapes: np.ndarray | None = None      # (n_nodes, modes) transverse w
    wet_mode_shapes: np.ndarray | None = None      # (n_nodes, modes) transverse w
    panel_centres: np.ndarray | None = None        # (n_panels, 2)
    panel_pressure: np.ndarray | None = None       # (n_panels, modes) modal fluid reaction

    def summary(self) -> dict:
        return {
            "mode_numbers": [list(m) for m in self.mode_numbers],
            "dry_hz_fe": list(self.dry_hz_fe),
            "wet_hz_fe": list(self.wet_hz_fe),
            "ratio_fe": list(self.ratio_fe),
            "mu_modal_diag": list(self.mu_modal_diag),
            "coupling_ratio": list(self.coupling_ratio),
            "dry_hz_navier": list(self.dry_hz_navier),
            "length": self.length,
            "width": self.width,
            "thickness": self.thickness,
            "rho_fluid": self.rho_fluid,
            "rho_struct": self.rho_struct,
            "fe_mesh": list(self.fe_mesh),
            "collapsed_modes": self.collapsed_modes,
        }


def solve_wet_plate_fe_modal(
    *,
    length: float = 10.0,
    width: float = 4.0,
    thickness: float = 0.02,
    young: float = 210.0e9,
    poisson: float = 0.3,
    rho_struct: float = 7850.0,
    rho_fluid: float = 1000.0,
    modes: int = 6,
    fe_mesh: tuple[int, int] = (12, 12),
    panel_mesh: tuple[int, int] | None = None,
    dry_mode_numbers: Sequence[tuple[int, int]] = ((1, 1), (1, 2), (2, 1), (2, 2)),
) -> WetPlateFEModalResult:
    """Wet frequencies of a simply supported panel with a nonlocal fluid operator.

    ``modes`` dry Mindlin modes are retained in the coupling basis;
    ``panel_mesh`` (default: equal to ``fe_mesh``) sets the fluid panel
    discretisation, which is *independent* of the structural mesh so that the
    fluid convergence can be studied without touching the structure.

    ``dry_mode_numbers`` only labels the reported modes: for each analytic
    ``(m, n)`` the dry FE mode closest in frequency is selected, and its wet
    counterpart is taken as the coupled eigenvalue of the same mode index.  The
    ordering of the coupled eigenvalues is checked (``collapsed_modes`` counts
    eigenvalue crossings, which invalidate a 1:1 index mapping).
    """
    from .fsi_wet_modal import navier_plate_frequency_hz

    length = _positive("length", length)
    width = _positive("width", width)
    thickness = _positive("thickness", thickness)
    young = _positive("young", young)
    rho_struct = _positive("rho_struct", rho_struct)
    rho_fluid = _positive("rho_fluid", rho_fluid)
    modes = _count("modes", modes)
    fe_mesh = (_count("fe_mesh[0]", fe_mesh[0]), _count("fe_mesh[1]", fe_mesh[1]))
    panel_mesh = fe_mesh if panel_mesh is None else (
        _count("panel_mesh[0]", panel_mesh[0]),
        _count("panel_mesh[1]", panel_mesh[1]),
    )

    dry = solve_plate_vibration(
        fe_mesh[0],
        fe_mesh[1],
        length=length,
        width=width,
        young=young,
        poisson=poisson,
        thickness=thickness,
        density=rho_struct,
        boundary=BOUNDARY_SIMPLE,
        modes=modes,
    )
    nodes = dry.nodes
    elements = dry.elements
    ndof = 3 * nodes.shape[0]
    phi = dry.modes
    if phi.shape[1] < modes:
        raise RuntimeError("dry eigen-solve returned fewer modes than requested")

    # Consistent structural mass matrix (same element as the eigen-solve).
    mass = torch.zeros((ndof, ndof), dtype=nodes.dtype, device=nodes.device)
    for elem in elements:
        xy = nodes[elem]
        ke = q4_plate_mass_matrix(xy, rho_struct, thickness)
        idx = torch.stack([3 * elem + k for k in range(3)], dim=1).reshape(-1)
        mass[idx[:, None], idx[None, :]] += ke

    # Nodal transverse DOF values (dof 0 of every node is w).
    phi_w = phi[0::3, :]

    # Fluid panels at the element centroids of the fluid mesh.
    centres, areas = _centroid_panels(length, width, panel_mesh)
    hydro = dipole_added_mass_matrix(
        torch.as_tensor(centres, dtype=torch.float64),
        torch.as_tensor(areas, dtype=torch.float64),
        rho_fluid,
    ).numpy()

    # Bilinear sampling of the dry modes at the fluid-panel centroids.  The two
    # meshes are independent, so the sampling cannot assume coincident centroids.
    sampled = _sample_nodal_field(
        nodes.numpy(),
        elements.numpy(),
        phi_w.numpy(),
        centres,
        fe_mesh,
        length,
        width,
    )
    coupling = sampled.T @ (hydro @ sampled)                      # (modes, modes)
    gen_mass = np.einsum("ik,ij,jk->k", phi.numpy(), mass.numpy(), phi.numpy())

    omega_dry = 2.0 * math.pi * dry.frequencies_hz.numpy()[:modes]
    stiffness = np.diag(omega_dry ** 2 * gen_mass)
    total_mass = np.diag(gen_mass) + coupling
    eigenvalues, eigenvectors = scipy.linalg.eigh(stiffness, total_mass, subset_by_index=(0, modes - 1))
    if np.any(eigenvalues <= 0.0):
        raise RuntimeError("coupled eigenproblem produced non-positive eigenvalues")
    wet_hz = np.sqrt(eigenvalues) / (2.0 * math.pi)
    wet_nodal = phi_w.numpy() @ eigenvectors
    panel_pressure = hydro @ (sampled @ eigenvectors)

    dry_hz = dry.frequencies_hz.numpy()[:modes]
    ratio = wet_hz / dry_hz
    mu_diag = np.diag(coupling) / gen_mass
    # Off-diagonal coupling magnitude relative to the diagonal (0 = single-mode exact).
    off_diag = np.linalg.norm(coupling - np.diag(np.diag(coupling))) / np.linalg.norm(coupling)
    coupling_ratio = [float(off_diag)] * modes

    # Pair each analytic (m, n) with the closest dry FE frequency; the wet value
    # is the coupled eigenvalue of the same index.
    order = np.argsort(dry_hz)
    dry_sorted = dry_hz[order]
    wet_sorted = wet_hz[order]
    crossings = int(np.sum(np.diff(np.argsort(np.argsort(wet_hz))) < 0))
    labels: list[tuple[int, int]] = []
    dry_sel: list[float] = []
    wet_sel: list[float] = []
    ratio_sel: list[float] = []
    mu_sel: list[float] = []
    coupling_sel: list[float] = []
    used: set[int] = set()
    for m, n in dry_mode_numbers:
        f_navier = navier_plate_frequency_hz(
            length, width, thickness, young, poisson, rho_struct, m, n
        )
        candidates = [i for i in range(len(dry_sorted)) if i not in used]
        if not candidates:
            break
        pick = min(candidates, key=lambda i: abs(dry_sorted[i] - f_navier))
        used.add(pick)
        original = int(order[pick])
        labels.append((int(m), int(n)))
        dry_sel.append(float(dry_hz[original]))
        wet_sel.append(float(wet_hz[original]))
        ratio_sel.append(float(ratio[original]))
        mu_sel.append(float(mu_diag[original]))
        coupling_sel.append(float(coupling_ratio[original]))

    return WetPlateFEModalResult(
        mode_numbers=tuple(labels),
        dry_hz_fe=tuple(dry_sel),
        wet_hz_fe=tuple(wet_sel),
        ratio_fe=tuple(ratio_sel),
        mu_modal_diag=tuple(mu_sel),
        coupling_ratio=tuple(coupling_sel),
        dry_hz_navier=tuple(
            navier_plate_frequency_hz(length, width, thickness, young, poisson, rho_struct, m, n)
            for m, n in labels
        ),
        length=length,
        width=width,
        thickness=thickness,
        rho_fluid=rho_fluid,
        rho_struct=rho_struct,
        fe_mesh=fe_mesh,
        collapsed_modes=crossings,
        nodes=nodes.numpy(),
        elements=elements.numpy(),
        dry_mode_shapes=phi_w.numpy(),
        wet_mode_shapes=wet_nodal,
        panel_centres=centres,
        panel_pressure=panel_pressure,
    )


def _sample_nodal_field(
    nodes: np.ndarray,
    elements: np.ndarray,
    nodal: np.ndarray,
    centres: np.ndarray,
    fe_mesh: tuple[int, int],
    length: float,
    width: float,
) -> np.ndarray:
    """Bilinear interpolation of a nodal field at arbitrary points.

    ``nodal`` is ``(n_nodes, k)``; the result is ``(len(centres), k)``.  The
    structural mesh is the uniform rectangle produced by
    :func:`tensorfem.plate_vibration.structured_rectangular_mesh`, so each
    element is located by its grid index instead of a global search.
    """
    nx, ny = fe_mesh
    dx, dy = length / nx, width / ny
    element_centres = nodes[elements].mean(axis=1)
    grid: dict[tuple[int, int], int] = {}
    for e, (cx, cy) in enumerate(element_centres):
        grid[(int(min(nx - 1, cx / dx)), int(min(ny - 1, cy / dy)))] = e

    out = np.zeros((centres.shape[0], nodal.shape[1]))
    for p, (px, py) in enumerate(centres):
        i = int(min(nx - 1, max(0, px / dx)))
        j = int(min(ny - 1, max(0, py / dy)))
        e = grid[(i, j)]
        xy = nodes[elements[e]]
        x0, x1 = xy[:, 0].min(), xy[:, 0].max()
        y0, y1 = xy[:, 1].min(), xy[:, 1].max()
        xi = 2.0 * (px - x0) / (x1 - x0) - 1.0
        eta = 2.0 * (py - y0) / (y1 - y0) - 1.0
        shape = np.array(
            [
                0.25 * (1 - xi) * (1 - eta),
                0.25 * (1 + xi) * (1 - eta),
                0.25 * (1 + xi) * (1 + eta),
                0.25 * (1 - xi) * (1 + eta),
            ]
        )
        out[p, :] = shape @ nodal[elements[e], :]
    return out


def _centroid_panels(length: float, width: float, mesh: tuple[int, int]):
    """Uniform panel grid: element centroids and areas (fluid discretisation)."""
    nx, ny = _count("nx", mesh[0]), _count("ny", mesh[1])
    dx, dy = length / nx, width / ny
    xs = (np.arange(nx) + 0.5) * dx
    ys = (np.arange(ny) + 0.5) * dy
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    centres = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    return centres, np.full(nx * ny, dx * dy)


def wet_plate_modal_convergence(
    *,
    length: float = 10.0,
    width: float = 4.0,
    thickness: float = 0.02,
    young: float = 210.0e9,
    poisson: float = 0.3,
    rho_struct: float = 7850.0,
    rho_fluid: float = 1000.0,
    fe_meshes: Sequence[tuple[int, int]] = ((8, 8), (12, 12), (16, 16), (20, 20)),
    panel_mesh: tuple[int, int] = (24, 24),
    modes: int = 6,
) -> list[dict]:
    """Wet first-mode convergence in the structural mesh (fluid mesh fixed).

    Returns one row per structural mesh with the dry/wet first frequency and
    the wet/dry ratio; the panel (fluid) mesh is held at ``panel_mesh`` so the
    two discretisations are separated.
    """
    rows = []
    for mesh in fe_meshes:
        result = solve_wet_plate_fe_modal(
            length=length,
            width=width,
            thickness=thickness,
            young=young,
            poisson=poisson,
            rho_struct=rho_struct,
            rho_fluid=rho_fluid,
            modes=modes,
            fe_mesh=(int(mesh[0]), int(mesh[1])),
            panel_mesh=(int(panel_mesh[0]), int(panel_mesh[1])),
        )
        rows.append(
            {
                "fe_mesh": [int(mesh[0]), int(mesh[1])],
                "dry_hz_fe": result.dry_hz_fe[0],
                "wet_hz_fe": result.wet_hz_fe[0],
                "ratio_fe": result.ratio_fe[0],
            }
        )
    return rows


def plate_fluid_mesh_convergence(
    *,
    length: float = 10.0,
    width: float = 4.0,
    thickness: float = 0.02,
    young: float = 210.0e9,
    poisson: float = 0.3,
    rho_struct: float = 7850.0,
    rho_fluid: float = 1000.0,
    fe_mesh: tuple[int, int] = (16, 16),
    panel_meshes: Sequence[tuple[int, int]] = ((8, 8), (12, 12), (16, 16), (20, 20), (24, 24)),
    modes: int = 6,
) -> list[dict]:
    """Wet first-mode convergence in the fluid (panel) mesh, structure fixed."""
    rows = []
    for panel in panel_meshes:
        result = solve_wet_plate_fe_modal(
            length=length,
            width=width,
            thickness=thickness,
            young=young,
            poisson=poisson,
            rho_struct=rho_struct,
            rho_fluid=rho_fluid,
            modes=modes,
            fe_mesh=(int(fe_mesh[0]), int(fe_mesh[1])),
            panel_mesh=(int(panel[0]), int(panel[1])),
        )
        rows.append(
            {
                "panel_mesh": [int(panel[0]), int(panel[1])],
                "dry_hz_fe": result.dry_hz_fe[0],
                "wet_hz_fe": result.wet_hz_fe[0],
                "ratio_fe": result.ratio_fe[0],
            }
        )
    return rows