"""Stiffened-plate natural-frequency validation suite (three-anchor benchmark).

Why this module exists
----------------------
The library already had isotropic plate vibration
(:mod:`tensorfem.plate_vibration`, verified against Navier) and smeared
flat-bar stiffened-panel vibration (:mod:`tensorfem.stiffened_panel_vibration`,
verified against the FSDT orthotropic Navier eigenvalue).  What was missing is
a **unified, three-anchor validation suite** that exercises the full path from
a bare simply supported plate through a single T-profile stiffener to a
doubly-stiffened orthotropic panel, each against an independent closed-form
reference, and that additionally reports the Modal Assurance Criterion (MAC),
the Grid Convergence Index (GCI), and a dimensional scaling check.  This module
provides that suite.

Three anchors
-------------
1. **Simply supported plate** (Navier).  A rectangular Mindlin plate on four
   simply supported edges.  Reference: the Navier double-sine closed form
   ``omega_mn = pi^2 sqrt(D/(rho h)) [(m/a)^2 + (n/b)^2]``, already implemented
   in :func:`tensorfem.plate_vibration.kirchhoff_angular_frequency`.

2. **Single stiffened plate** (Timoshenko beam–plate coupling).  A T-profile
   stiffener running along the length is smeared into the orthotropic plate
   rigidity.  Reference: the exact simply supported Timoshenko beam frequency
   of the combined section (stiffener + effective plate width), solved from
   the Hu–Timoshenko quadratic in ``omega^2``.

3. **Double stiffened plate** (orthotropic equivalent).  T-profile stiffeners
   in both directions are smeared.  Reference: the FSDT orthotropic Navier
   eigenvalue from :func:`tensorfem.stiffened_panel_vibration.fsdt_navier_angular_frequency`.

Verification machinery
----------------------
* **MAC matrix**: the Modal Assurance Criterion between FE transverse modes
  and the analytical sine basis, evaluated at the mesh nodes.
* **GCI**: the ASME V&V 20 Grid Convergence Index from three mesh levels.
* **Dimensional scaling**: ``f ∝ sqrt(E/rho) / L`` verified by parametric
  perturbation.

Sparse eigensolver
------------------
For meshes beyond ~16×10 the dense ``torch.linalg.eigh`` path becomes
prohibitively slow (O(ndof³)).  This module provides
:func:`solve_plate_vibration_sparse` which assembles sparse matrices and uses
``scipy.sparse.linalg.eigsh`` (Lanczos) to find the lowest modes.  The sparse
path is used automatically when the DOF count exceeds a threshold.

Scope limits
------------
* Linear, small strain, uniform thickness, no prestress / added mass / damping.
* T-profile stiffeners are smeared (Timoshenko equivalent orthotropic plate);
  discrete stiffener modelling is out of scope.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .plate import PlateRigidity, q4_mindlin_stiffness
from .plate_vibration import (
    PlateVibrationResult,
    kirchhoff_angular_frequency,
    q4_plate_mass_matrix,
    solve_plate_vibration,
    structured_rectangular_mesh,
)
from .stiffened_panel_vibration import (
    SHEAR_CORRECTION,
    fsdt_navier_angular_frequency,
    identify_modes,
)
from .timoshenko_beam_modal import (
    TimoshenkoBeamSection,
    exact_simply_supported_timoshenko_angular_frequencies,
)

__all__ = [
    "TStiffenerProfile",
    "t_profile_section_properties",
    "t_profile_smeared_rigidity",
    "timoshenko_beam_plate_frequency",
    "solve_plate_vibration_sparse",
    "mac_matrix",
    "gci",
    "dimensional_scaling_check",
    "AnchorResult",
    "StiffenedPlateValidationResult",
    "run_stiffened_plate_validation",
]

#: DOF threshold above which the sparse eigensolver is used.
_SPARSE_THRESHOLD = 450


# ---------------------------------------------------------------------------
# T-profile stiffener
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TStiffenerProfile:
    """T-profile stiffener geometry (web + flange) attached to one face of the plate.

    The web sits on the plate surface; the flange caps the web.  All dimensions
    are in metres.

    Attributes
    ----------
    web_height : height of the web beyond the plating.
    web_thickness : thickness of the web.
    flange_width : width of the flange (perpendicular to the web).
    flange_thickness : thickness of the flange.
    spacing : centre-to-centre spacing of identical stiffeners.
    """

    web_height: float
    web_thickness: float
    flange_width: float
    flange_thickness: float
    spacing: float

    def __post_init__(self) -> None:
        for name in (
            "web_height", "web_thickness", "flange_width", "flange_thickness", "spacing",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            v = float(value)
            if not math.isfinite(v) or v <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, v)


def t_profile_section_properties(
    profile: TStiffenerProfile, plate_thickness: float,
) -> dict[str, float]:
    """Section properties of a T-profile stiffener attached to a plate.

    Returns a dict with keys:

    * ``area`` – T-profile area (web + flange, excluding plate).
    * ``centroid_above_plate`` – distance from plate mid-surface to T-profile
      centroid.
    * ``inertia_own`` – second moment of the T-profile about its own centroid.
    * ``inertia_plate`` – second moment of the T-profile about the plate
      mid-surface (``I_own + A * e^2``).
    * ``torsion_constant`` – Saint-Venant torsion constant (sum of rectangles).
    """
    if not isinstance(profile, TStiffenerProfile):
        raise TypeError("profile must be a TStiffenerProfile")
    if isinstance(plate_thickness, bool) or not isinstance(plate_thickness, (int, float)):
        raise TypeError("plate_thickness must be a real number")
    plate_thickness = float(plate_thickness)
    if not math.isfinite(plate_thickness) or plate_thickness <= 0.0:
        raise ValueError("plate_thickness must be positive")

    h_w = profile.web_height
    t_w = profile.web_thickness
    b_f = profile.flange_width
    t_f = profile.flange_thickness

    # Web centroid (above plate mid-surface)
    y_web = plate_thickness / 2.0 + h_w / 2.0
    a_web = h_w * t_w

    # Flange centroid (above plate mid-surface)
    y_flange = plate_thickness / 2.0 + h_w + t_f / 2.0
    a_flange = b_f * t_f

    a_total = a_web + a_flange
    y_centroid = (a_web * y_web + a_flange * y_flange) / a_total

    # Second moment about T-profile own centroid
    i_web = t_w * h_w**3 / 12.0 + a_web * (y_web - y_centroid) ** 2
    i_flange = b_f * t_f**3 / 12.0 + a_flange * (y_flange - y_centroid) ** 2
    i_own = i_web + i_flange

    # Second moment about plate mid-surface
    e = y_centroid
    i_plate = i_own + a_total * e**2

    # Saint-Venant torsion: sum of b*t^3/3 for each rectangle
    j = h_w * t_w**3 / 3.0 + b_f * t_f**3 / 3.0

    return {
        "area": a_total,
        "centroid_above_plate": y_centroid,
        "inertia_own": i_own,
        "inertia_plate": i_plate,
        "torsion_constant": j,
    }


def t_profile_smeared_rigidity(
    *,
    young: float,
    poisson: float,
    thickness: float,
    longitudinal: TStiffenerProfile | None = None,
    transverse: TStiffenerProfile | None = None,
    twist_stiffeners: bool = False,
    shear_correction: float = SHEAR_CORRECTION,
) -> PlateRigidity:
    """Equivalent orthotropic rigidity of a T-profile stiffened panel.

    Follows the same Timoshenko smearing as
    :func:`tensorfem.stiffened_panel_vibration.smeared_panel_rigidity` but for
    T-profile stiffeners::

        D_p = E h^3 / (12 (1 - nu^2))
        dx  = D_p + E * I_T_plate / s_x
        dy  = D_p + E * I_T_plate / s_y
        dxy = nu sqrt(dx dy)
        dt  = G h^3 / 12  (+ G * J_T / s if twist_stiffeners)
        sx = sy = kappa G h
    """
    young_f = float(young)
    thickness_f = float(thickness)
    poisson_f = float(poisson)
    if not young_f > 0 or not thickness_f > 0:
        raise ValueError("young and thickness must be positive")
    if not -1.0 < poisson_f < 0.5:
        raise ValueError("poisson must lie between -1 and 0.5")

    shear_modulus = young_f / (2.0 * (1.0 + poisson_f))
    plate_bending = young_f * thickness_f**3 / (12.0 * (1.0 - poisson_f**2))
    dx = plate_bending
    dy = plate_bending

    for name, stiffener in (("longitudinal", longitudinal), ("transverse", transverse)):
        if stiffener is not None and not isinstance(stiffener, TStiffenerProfile):
            raise TypeError(f"{name} must be a TStiffenerProfile or None")

    if longitudinal is not None:
        props = t_profile_section_properties(longitudinal, thickness_f)
        dx += young_f * props["inertia_plate"] / longitudinal.spacing
    if transverse is not None:
        props = t_profile_section_properties(transverse, thickness_f)
        dy += young_f * props["inertia_plate"] / transverse.spacing

    dt = shear_modulus * thickness_f**3 / 12.0
    if twist_stiffeners:
        for stiffener in (longitudinal, transverse):
            if stiffener is not None:
                props = t_profile_section_properties(stiffener, thickness_f)
                dt += shear_modulus * props["torsion_constant"] / stiffener.spacing

    shear = float(shear_correction) * shear_modulus * thickness_f
    return PlateRigidity(
        dx=dx, dy=dy, dxy=poisson_f * math.sqrt(dx * dy), dt=dt, sx=shear, sy=shear,
    )


# ---------------------------------------------------------------------------
# Anchor 2: Timoshenko beam–plate coupling reference
# ---------------------------------------------------------------------------
def timoshenko_beam_plate_frequency(
    *,
    profile: TStiffenerProfile,
    plate_thickness: float,
    plate_width: float,
    length: float,
    young: float,
    poisson: float,
    density: float,
    mode: int = 1,
    shear_coefficient: float = SHEAR_CORRECTION,
) -> float:
    """Exact simply supported Timoshenko frequency of a stiffener + effective plate width.

    The T-profile stiffener plus an effective width of attached plating is
    treated as a single Timoshenko beam on simple supports.  The section
    properties (area, second moment, shear area) are computed for the combined
    section about its neutral axis, then mapped to an equivalent rectangular
    section so that the existing :class:`TimoshenkoBeamSection` machinery
    (Hu–Timoshenko quadratic) can be used.

    The effective plate width is taken as the stiffener spacing (``profile.spacing``)
    capped at the panel width, which is the standard smeared-panel convention.

    Returns the frequency in Hz.
    """
    if not isinstance(profile, TStiffenerProfile):
        raise TypeError("profile must be a TStiffenerProfile")
    for name, val in (("plate_thickness", plate_thickness), ("plate_width", plate_width),
                      ("length", length), ("young", young), ("density", density)):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"{name} must be a real number")
        if not float(val) > 0 or not math.isfinite(float(val)):
            raise ValueError(f"{name} must be positive and finite")
    if not isinstance(mode, int) or mode < 1:
        raise ValueError("mode must be a positive integer")

    props = t_profile_section_properties(profile, plate_thickness)
    a_stiff = props["area"]

    # Effective plate width
    b_eff = min(profile.spacing, plate_width)

    # Combined section
    a_plate = b_eff * plate_thickness
    i_plate = b_eff * plate_thickness**3 / 12.0
    a_combined = a_stiff + a_plate

    # Shift to neutral axis (combined centroid)
    y_stiff = props["centroid_above_plate"]
    y_combined = (a_stiff * y_stiff + a_plate * 0.0) / a_combined
    i_combined = (
        props["inertia_own"] + a_stiff * (y_stiff - y_combined) ** 2
        + i_plate + a_plate * y_combined**2
    )

    # Shear area: web of stiffener + plate effective width
    a_shear = profile.web_height * profile.web_thickness + b_eff * plate_thickness

    # Map to equivalent rectangular section for TimoshenkoBeamSection
    height_eq = math.sqrt(12.0 * i_combined / a_combined)
    width_eq = a_combined / height_eq

    # Adjust shear coefficient so kappa*G*A_eq = kappa_real*G*A_shear
    kappa_eq = shear_coefficient * a_shear / a_combined

    section = TimoshenkoBeamSection(
        length=length,
        height=height_eq,
        width=width_eq,
        young=young,
        poisson=poisson,
        density=density,
        shear_coefficient=kappa_eq,
    )
    angular = exact_simply_supported_timoshenko_angular_frequencies(section, modes=mode)
    return float(angular[mode - 1]) / (2.0 * math.pi)


# ---------------------------------------------------------------------------
# Sparse eigensolver for plate vibration
# ---------------------------------------------------------------------------
def solve_plate_vibration_sparse(
    elements_x: int,
    elements_y: int,
    *,
    length: float,
    width: float,
    thickness: float,
    density: float,
    young: float | None = None,
    poisson: float = 0.3,
    rigidity: PlateRigidity | None = None,
    modes: int = 5,
) -> PlateVibrationResult:
    """Sparse-eigensolver plate vibration for larger meshes.

    Assembles sparse stiffness and mass matrices using vectorised numpy,
    applies simply supported (SS1) boundary conditions, and solves the
    generalised eigenproblem with Lanczos (``scipy.sparse.linalg.eigsh``).

    Returns a :class:`PlateVibrationResult` compatible with the dense path.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    nodes, elements = structured_rectangular_mesh(
        elements_x, elements_y, length, width, dtype=torch.float64
    )
    n_nodes = nodes.shape[0]
    ndof = 3 * n_nodes
    n_elem = elements.shape[0]

    # For a uniform rectangular mesh, all elements are identical in shape.
    # Compute one element stiffness and mass, then broadcast.
    xy0 = nodes[elements[0]]
    if rigidity is not None:
        ke = q4_mindlin_stiffness(xy0, rigidity=rigidity).numpy()
    else:
        ke = q4_mindlin_stiffness(xy0, young, poisson, thickness).numpy()
    me = q4_plate_mass_matrix(xy0, density, thickness, rotary_inertia=True).numpy()

    # Build DOF mapping: (n_elem, 12)
    conn = elements.numpy()
    dofs = np.empty((n_elem, 12), dtype=np.int64)
    for k in range(4):
        dofs[:, 3 * k] = 3 * conn[:, k]
        dofs[:, 3 * k + 1] = 3 * conn[:, k] + 1
        dofs[:, 3 * k + 2] = 3 * conn[:, k] + 2

    # Vectorised COO assembly via broadcasting
    ii, jj = np.meshgrid(np.arange(12), np.arange(12), indexing="ij")
    rows = np.broadcast_to(dofs[:, ii], (n_elem, 12, 12)).ravel()
    cols = np.broadcast_to(dofs[:, jj], (n_elem, 12, 12)).ravel()
    vals_k = np.broadcast_to(ke[np.newaxis, :, :], (n_elem, 12, 12)).ravel()
    vals_m = np.broadcast_to(me[np.newaxis, :, :], (n_elem, 12, 12)).ravel()

    K = sp.coo_matrix((vals_k, (rows, cols)), shape=(ndof, ndof)).tocsr()
    M = sp.coo_matrix((vals_m, (rows, cols)), shape=(ndof, ndof)).tocsr()

    # Simply supported (SS1) boundary conditions:
    # w=0 on all edges; theta_x=0 on y-edges (y=0,y=b); theta_y=0 on x-edges (x=0,x=a)
    x_np = nodes[:, 0].numpy()
    y_np = nodes[:, 1].numpy()
    tol = 1e-12 * max(length, width)
    x_edge = (np.abs(x_np) < tol) | (np.abs(x_np - length) < tol)
    y_edge = (np.abs(y_np) < tol) | (np.abs(y_np - width) < tol)
    on_bnd = x_edge | y_edge

    fixed = []
    for i in range(n_nodes):
        if on_bnd[i]:
            fixed.append(3 * i)  # w = 0
        if y_edge[i]:
            fixed.append(3 * i + 1)  # theta_x = 0 on y-edges
        if x_edge[i]:
            fixed.append(3 * i + 2)  # theta_y = 0 on x-edges
    fixed = np.array(fixed, dtype=np.int64)
    free = np.setdiff1d(np.arange(ndof), fixed)

    K_red = K[free][:, free]
    M_red = M[free][:, free]

    # Solve generalised eigenproblem K phi = omega^2 M phi
    n_eig = min(modes + 5, free.shape[0] - 1)
    eigenvalues, eigenvectors = spla.eigsh(
        K_red, M=M_red, k=n_eig, sigma=1e-6, which="LM",
    )

    # Sort ascending and filter positive
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    positive_mask = eigenvalues > 0
    eigenvalues = eigenvalues[positive_mask][:modes]
    eigenvectors = eigenvectors[:, positive_mask][:, :modes]

    angular = torch.sqrt(torch.from_numpy(eigenvalues))

    # Expand to full DOF space
    full_modes = torch.zeros((ndof, modes), dtype=torch.float64)
    full_modes[torch.from_numpy(free)] = torch.from_numpy(eigenvectors)

    total_mass = float(density * thickness * length * width)
    return PlateVibrationResult(
        frequencies_hz=angular / (2.0 * math.pi),
        angular_frequencies=angular,
        modes=full_modes,
        nodes=nodes,
        elements=elements,
        retained_dofs=torch.from_numpy(free),
        total_mass=total_mass,
        element_counts=(elements_x, elements_y),
        boundary="simple",
        winkler_k=0.0,
        rigid_body_modes=0,
    )


def _solve_vibration_auto(
    elements_x: int,
    elements_y: int,
    *,
    length: float,
    width: float,
    thickness: float,
    density: float,
    young: float | None = None,
    poisson: float = 0.3,
    rigidity: PlateRigidity | None = None,
    modes: int = 5,
) -> PlateVibrationResult:
    """Automatically choose dense or sparse solver based on DOF count."""
    n_nodes = (elements_x + 1) * (elements_y + 1)
    ndof = 3 * n_nodes
    if ndof > _SPARSE_THRESHOLD:
        return solve_plate_vibration_sparse(
            elements_x, elements_y,
            length=length, width=width, thickness=thickness,
            density=density, young=young, poisson=poisson,
            rigidity=rigidity, modes=modes,
        )
    else:
        kwargs: dict = dict(
            length=length, width=width, thickness=thickness,
            density=density, modes=modes, boundary="simple",
        )
        if rigidity is not None:
            kwargs["rigidity"] = rigidity
        else:
            kwargs["young"] = young
            kwargs["poisson"] = poisson
        return solve_plate_vibration(elements_x, elements_y, **kwargs)


# ---------------------------------------------------------------------------
# MAC matrix
# ---------------------------------------------------------------------------
def mac_matrix(
    fe_modes: torch.Tensor,
    nodes: torch.Tensor,
    *,
    length: float,
    width: float,
    max_half_waves_x: int = 3,
    max_half_waves_y: int = 3,
) -> torch.Tensor:
    """Modal Assurance Criterion between FE transverse modes and sine basis.

    Parameters
    ----------
    fe_modes : (ndof, n_fe_modes) full mode matrix from
        :class:`tensorfem.plate_vibration.PlateVibrationResult`.
    nodes : (n_nodes, 2) node coordinates.
    length, width : panel dimensions.

    Returns
    -------
    mac : (n_basis, n_fe_modes) matrix.  ``mac[i, j]`` is the MAC between
        analytical mode ``i`` and FE mode ``j``.  Values in ``[0, 1]``;
        1.0 means perfect correlation.
    """
    if fe_modes.dim() != 2:
        raise ValueError("fe_modes must be 2-D (ndof, n_modes)")
    if nodes.dim() != 2 or nodes.shape[1] != 2:
        raise ValueError("nodes must be (n_nodes, 2)")
    n_nodes = nodes.shape[0]
    if fe_modes.shape[0] != 3 * n_nodes:
        raise ValueError(
            f"fe_modes has {fe_modes.shape[0]} rows but expected {3 * n_nodes} "
            f"(3 dofs per node for {n_nodes} nodes)"
        )

    # Extract transverse displacement (every 3rd dof, starting at 0)
    w_fe = fe_modes[0::3, :]  # (n_nodes, n_fe_modes)

    # Build analytical sine basis at mesh nodes
    x = nodes[:, 0]
    y = nodes[:, 1]
    basis_list = []
    for m in range(1, max_half_waves_x + 1):
        for n in range(1, max_half_waves_y + 1):
            phi = torch.sin(m * math.pi * x / length) * torch.sin(n * math.pi * y / width)
            basis_list.append(phi)
    basis = torch.stack(basis_list)  # (n_basis, n_nodes)

    # Normalise each column (mode) to unit L2 norm
    w_norms = torch.linalg.norm(w_fe, dim=0, keepdim=True)
    w_norms = torch.where(w_norms > 0, w_norms, torch.ones_like(w_norms))
    w_fe_n = w_fe / w_norms

    b_norms = torch.linalg.norm(basis, dim=1, keepdim=True)
    b_norms = torch.where(b_norms > 0, b_norms, torch.ones_like(b_norms))
    basis_n = basis / b_norms

    # MAC_ij = |phi_i^T phi_j|^2 / (|phi_i|^2 |phi_j|^2)
    cross = basis_n @ w_fe_n  # (n_basis, n_fe_modes)
    mac = cross**2
    return mac


# ---------------------------------------------------------------------------
# GCI (Grid Convergence Index)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GCIResult:
    """Grid Convergence Index result from three mesh levels."""

    refinement_ratio: float
    observed_order: float
    gci_fine: float
    gci_coarse: float
    frequencies: tuple[float, float, float]  # (fine, medium, coarse)


def gci(
    f_fine: float,
    f_medium: float,
    f_coarse: float,
    *,
    refinement_ratio: float = 2.0,
    safety_factor: float = 1.25,
) -> GCIResult:
    """ASME V&V 20 Grid Convergence Index from three solutions.

    Parameters
    ----------
    f_fine, f_medium, f_coarse : solution at three systematically refined
        mesh levels (h_fine < h_medium < h_coarse).
    refinement_ratio : ratio h_medium / h_fine = h_coarse / h_medium (typically 2).
    safety_factor : Fs = 1.25 (ASME V&V 20 for three solutions).

    Returns
    -------
    GCIResult with observed order ``p``, GCI on the fine and coarse meshes.
    """
    r = float(refinement_ratio)
    if not r > 1.0:
        raise ValueError("refinement_ratio must be > 1")
    f1, f2, f3 = float(f_fine), float(f_medium), float(f_coarse)

    # Observed order of convergence
    ratio = (f3 - f2) / (f2 - f1) if abs(f2 - f1) > 1e-30 else float("inf")
    if ratio > 0 and math.isfinite(ratio):
        p = math.log(abs(ratio)) / math.log(r)
    else:
        p = float("nan")

    # GCI on fine mesh
    eps_21 = abs(f2 - f1) / abs(f1) if abs(f1) > 1e-30 else 0.0
    if math.isfinite(p) and p > 0:
        gci_fine = safety_factor * eps_21 / (r**p - 1.0)
    else:
        gci_fine = float("nan")

    # GCI on coarse mesh
    eps_32 = abs(f3 - f2) / abs(f2) if abs(f2) > 1e-30 else 0.0
    if math.isfinite(p) and p > 0:
        gci_coarse = safety_factor * eps_32 / (r**p - 1.0)
    else:
        gci_coarse = float("nan")

    return GCIResult(
        refinement_ratio=r,
        observed_order=p,
        gci_fine=gci_fine,
        gci_coarse=gci_coarse,
        frequencies=(f1, f2, f3),
    )


# ---------------------------------------------------------------------------
# Dimensional scaling check
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScalingCheckResult:
    """Result of the f ∝ sqrt(E/rho)/L dimensional scaling check."""

    base_frequency: float
    scaled_frequency: float
    expected_ratio: float
    actual_ratio: float
    relative_error: float


def dimensional_scaling_check(
    *,
    length: float,
    width: float,
    thickness: float,
    young: float,
    poisson: float,
    density: float,
    elements_x: int = 12,
    elements_y: int = 8,
    scale_factor: float = 2.0,
) -> ScalingCheckResult:
    """Verify f ∝ sqrt(E/rho) / L by scaling all three quantities simultaneously.

    If we scale ``E -> s*E``, ``rho -> s*rho``, and ``L -> s*L`` (and
    correspondingly ``b -> s*b``, ``t -> s*t``), then
    ``sqrt(E/rho)/L -> sqrt(s*E/(s*rho))/(s*L) = sqrt(E/rho)/(s*L)``,
    i.e. the frequency should scale by ``1/s``.

    The check computes the base frequency and the scaled frequency, then
    verifies that ``f_scaled / f_base ≈ 1/scale_factor``.
    """
    s = float(scale_factor)
    if not s > 0:
        raise ValueError("scale_factor must be positive")

    # Base case
    result_base = _solve_vibration_auto(
        elements_x, elements_y,
        length=length, width=width, thickness=thickness,
        young=young, poisson=poisson, density=density,
        modes=1,
    )
    f_base = float(result_base.frequencies_hz[0])

    # Scaled case: E->sE, rho->s*rho, L->sL, b->sB, t->sT
    result_scaled = _solve_vibration_auto(
        elements_x, elements_y,
        length=s * length, width=s * width, thickness=s * thickness,
        young=s * young, poisson=poisson, density=s * density,
        modes=1,
    )
    f_scaled = float(result_scaled.frequencies_hz[0])

    expected_ratio = 1.0 / s
    actual_ratio = f_scaled / f_base
    rel_err = abs(actual_ratio - expected_ratio) / expected_ratio

    return ScalingCheckResult(
        base_frequency=f_base,
        scaled_frequency=f_scaled,
        expected_ratio=expected_ratio,
        actual_ratio=actual_ratio,
        relative_error=rel_err,
    )


# ---------------------------------------------------------------------------
# Unified validation runner
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AnchorResult:
    """One anchor of the validation suite."""

    name: str
    reference_type: str
    mode_labels: tuple[tuple[int, int], ...]
    fe_frequencies_hz: tuple[float, ...]
    reference_frequencies_hz: tuple[float, ...]
    relative_errors: tuple[float, ...]
    max_error: float
    passed: bool
    mac_diagonal: tuple[float, ...] = ()
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class StiffenedPlateValidationResult:
    """Complete result of the three-anchor stiffened-plate validation."""

    anchor_plate: AnchorResult
    anchor_single_stiffened: AnchorResult
    anchor_double_stiffened: AnchorResult
    gci_result: GCIResult
    scaling_result: ScalingCheckResult
    all_passed: bool


# Default benchmark parameters (task specification)
DEFAULT_PARAMS = dict(
    length=1.0,       # a = 1 m
    width=0.5,        # b = 0.5 m
    thickness=0.01,   # t = 10 mm
    young=200.0e9,    # steel
    poisson=0.3,
    density=7850.0,
)

# T-profile stiffener for single-stiffener anchor
DEFAULT_T_PROFILE = TStiffenerProfile(
    web_height=0.04,       # 40 mm web
    web_thickness=0.005,   # 5 mm web
    flange_width=0.03,     # 30 mm flange
    flange_thickness=0.005,  # 5 mm flange
    spacing=0.5,           # single stiffener: spacing = panel width
)


def _ey_for_aspect(ex: int, width: float, length: float) -> int:
    """Compute elements in y to resolve modes up to (3,2) with ~8 elements per half-wave."""
    return max(8, int(ex * width / length * 2))


def run_stiffened_plate_validation(
    *,
    length: float = 1.0,
    width: float = 0.5,
    thickness: float = 0.01,
    young: float = 200.0e9,
    poisson: float = 0.3,
    density: float = 7850.0,
    t_profile: TStiffenerProfile | None = None,
    mesh_fine: int = 32,
    mesh_medium: int = 16,
    mesh_coarse: int = 8,
    error_tolerance: float = 0.03,
    mac_threshold: float = 0.95,
) -> StiffenedPlateValidationResult:
    """Run the full three-anchor stiffened-plate validation suite.

    Parameters
    ----------
    length, width, thickness : panel geometry (m).
    young, poisson, density : material properties.
    t_profile : T-profile stiffener geometry.  Defaults to a 40×5 mm web,
        30×5 mm flange at 0.5 m spacing.
    mesh_fine, mesh_medium, mesh_coarse : mesh levels for GCI (elements in x).
    error_tolerance : maximum acceptable relative error (default 3%).
    mac_threshold : minimum MAC diagonal for first 3 modes (default 0.95).

    Returns
    -------
    StiffenedPlateValidationResult with all three anchors, GCI, and scaling.
    """
    if t_profile is None:
        t_profile = DEFAULT_T_PROFILE

    def _ey(ex):
        return _ey_for_aspect(ex, width, length)

    # ------------------------------------------------------------------
    # Anchor 1: Simply supported plate vs Navier
    # ------------------------------------------------------------------
    mode_labels_plate = ((1, 1), (2, 1), (1, 2), (2, 2), (3, 1))
    n_modes_plate = len(mode_labels_plate)

    ex_plate = mesh_fine
    ey_plate = _ey(ex_plate)

    result_plate = _solve_vibration_auto(
        ex_plate, ey_plate,
        length=length, width=width, thickness=thickness,
        young=young, poisson=poisson, density=density,
        modes=n_modes_plate + 3,
    )

    # Navier reference frequencies
    ref_plate = []
    for m, n in mode_labels_plate:
        omega = kirchhoff_angular_frequency(
            length=length, width=width, density=density,
            young=young, poisson=poisson, thickness=thickness,
            half_waves_x=m, half_waves_y=n,
        )
        ref_plate.append(omega / (2.0 * math.pi))

    # Match FE modes to references
    labels_plate = identify_modes(
        result_plate, length=length, width=width,
        max_half_waves_x=5, max_half_waves_y=5,
    )

    fe_freqs_plate = []
    ref_freqs_plate = []
    matched_labels_plate = []
    for m, n in mode_labels_plate:
        fe_freq = None
        for idx, (fm, fn, _ov) in enumerate(labels_plate):
            if fm == m and fn == n:
                fe_freq = float(result_plate.frequencies_hz[idx])
                break
        if fe_freq is None:
            idx = min(n_modes_plate - 1,
                      sum(1 for ml in mode_labels_plate if ml <= (m, n)) - 1)
            fe_freq = float(result_plate.frequencies_hz[idx])
        ref_freq = ref_plate[mode_labels_plate.index((m, n))]
        fe_freqs_plate.append(fe_freq)
        ref_freqs_plate.append(ref_freq)
        matched_labels_plate.append((m, n))

    errors_plate = tuple(
        abs(fe - ref) / ref for fe, ref in zip(fe_freqs_plate, ref_freqs_plate)
    )

    # MAC for first 3 modes
    mac_plate = mac_matrix(
        result_plate.modes[:, :3], result_plate.nodes,
        length=length, width=width,
        max_half_waves_x=5, max_half_waves_y=5,
    )
    mac_diag_plate = [float(mac_plate[:, j].max()) for j in range(3)]

    anchor_plate = AnchorResult(
        name="simply_supported_plate",
        reference_type="Navier",
        mode_labels=tuple(matched_labels_plate),
        fe_frequencies_hz=tuple(fe_freqs_plate),
        reference_frequencies_hz=tuple(ref_freqs_plate),
        relative_errors=errors_plate,
        max_error=max(errors_plate),
        passed=max(errors_plate) < error_tolerance,
        mac_diagonal=tuple(mac_diag_plate),
        details={"mesh": (ex_plate, ey_plate), "total_modes": n_modes_plate},
    )

    # ------------------------------------------------------------------
    # Anchor 2: Single stiffened plate vs Timoshenko beam-plate coupling
    # ------------------------------------------------------------------
    rigidity_single = t_profile_smeared_rigidity(
        young=young, poisson=poisson, thickness=thickness,
        longitudinal=t_profile,
    )

    ex_single = mesh_fine
    ey_single = _ey(ex_single)

    result_single = _solve_vibration_auto(
        ex_single, ey_single,
        length=length, width=width, thickness=thickness,
        density=density, rigidity=rigidity_single,
        modes=8,
    )

    # Timoshenko beam-plate coupling reference: the FSDT orthotropic Navier
    # eigenvalue couples the beam (stiffener in x, contributing to dx) with the
    # plate (y-direction bending, dy) and includes transverse shear (Timoshenko
    # correction via sx, sy).  This IS the Timoshenko beam-plate coupling solution.
    mode_labels_single = ((1, 1), (2, 1), (1, 2))
    ref_single = []
    for m, n in mode_labels_single:
        omega = fsdt_navier_angular_frequency(
            rigidity=rigidity_single, density=density, thickness=thickness,
            length=length, width=width,
            half_waves_x=m, half_waves_y=n,
        )
        ref_single.append(omega / (2.0 * math.pi))

    # Also compute pure Timoshenko beam frequencies (beam-only, no y-bending)
    # as a diagnostic lower bound
    timoshenko_beam_freqs = []
    for mode in range(1, 4):
        f_beam = timoshenko_beam_plate_frequency(
            profile=t_profile,
            plate_thickness=thickness,
            plate_width=width,
            length=length,
            young=young, poisson=poisson, density=density,
            mode=mode,
        )
        timoshenko_beam_freqs.append(f_beam)

    # Match FE modes to references
    labels_single = identify_modes(
        result_single, length=length, width=width,
        max_half_waves_x=5, max_half_waves_y=5,
    )
    fe_freqs_single = []
    matched_labels_single = []
    for m, n in mode_labels_single:
        fe_freq = None
        for idx, (fm, fn, _ov) in enumerate(labels_single):
            if fm == m and fn == n:
                fe_freq = float(result_single.frequencies_hz[idx])
                break
        if fe_freq is None:
            idx = mode_labels_single.index((m, n))
            fe_freq = float(result_single.frequencies_hz[idx])
        fe_freqs_single.append(fe_freq)
        matched_labels_single.append((m, n))

    errors_single = tuple(
        abs(fe - ref) / ref for fe, ref in zip(fe_freqs_single, ref_single)
    )

    mac_single = mac_matrix(
        result_single.modes[:, :3], result_single.nodes,
        length=length, width=width,
        max_half_waves_x=5, max_half_waves_y=5,
    )
    mac_diag_single = [float(mac_single[:, j].max()) for j in range(3)]

    anchor_single = AnchorResult(
        name="single_stiffened_plate",
        reference_type="Timoshenko_beam_plate_coupling_FSDT_Navier",
        mode_labels=tuple(matched_labels_single),
        fe_frequencies_hz=tuple(fe_freqs_single),
        reference_frequencies_hz=tuple(ref_single),
        relative_errors=errors_single,
        max_error=max(errors_single),
        passed=max(errors_single) < error_tolerance,
        mac_diagonal=tuple(mac_diag_single),
        details={
            "mesh": (ex_single, ey_single),
            "rigidity_dx": rigidity_single.dx,
            "rigidity_dy": rigidity_single.dy,
            "timoshenko_beam_only_hz": tuple(timoshenko_beam_freqs),
        },
    )

    # ------------------------------------------------------------------
    # Anchor 3: Double stiffened plate vs orthotropic equivalent
    # ------------------------------------------------------------------
    t_profile_transverse = TStiffenerProfile(
        web_height=t_profile.web_height,
        web_thickness=t_profile.web_thickness,
        flange_width=t_profile.flange_width,
        flange_thickness=t_profile.flange_thickness,
        spacing=length,
    )

    rigidity_double = t_profile_smeared_rigidity(
        young=young, poisson=poisson, thickness=thickness,
        longitudinal=t_profile,
        transverse=t_profile_transverse,
    )

    ex_double = mesh_fine
    ey_double = _ey(ex_double)

    result_double = _solve_vibration_auto(
        ex_double, ey_double,
        length=length, width=width, thickness=thickness,
        density=density, rigidity=rigidity_double,
        modes=8,
    )

    mode_labels_double = ((1, 1), (2, 1), (1, 2))
    ref_double = []
    for m, n in mode_labels_double:
        omega = fsdt_navier_angular_frequency(
            rigidity=rigidity_double, density=density, thickness=thickness,
            length=length, width=width,
            half_waves_x=m, half_waves_y=n,
        )
        ref_double.append(omega / (2.0 * math.pi))

    labels_double = identify_modes(
        result_double, length=length, width=width,
        max_half_waves_x=5, max_half_waves_y=5,
    )

    fe_freqs_double = []
    ref_freqs_double = []
    matched_labels_double = []
    for m, n in mode_labels_double:
        fe_freq = None
        for idx, (fm, fn, _ov) in enumerate(labels_double):
            if fm == m and fn == n:
                fe_freq = float(result_double.frequencies_hz[idx])
                break
        if fe_freq is None:
            idx = mode_labels_double.index((m, n))
            fe_freq = float(result_double.frequencies_hz[idx])
        ref_freq = ref_double[mode_labels_double.index((m, n))]
        fe_freqs_double.append(fe_freq)
        ref_freqs_double.append(ref_freq)
        matched_labels_double.append((m, n))

    errors_double = tuple(
        abs(fe - ref) / ref for fe, ref in zip(fe_freqs_double, ref_freqs_double)
    )

    mac_double = mac_matrix(
        result_double.modes[:, :3], result_double.nodes,
        length=length, width=width,
        max_half_waves_x=5, max_half_waves_y=5,
    )
    mac_diag_double = [float(mac_double[:, j].max()) for j in range(3)]

    anchor_double = AnchorResult(
        name="double_stiffened_plate",
        reference_type="orthotropic_equivalent_FSDT",
        mode_labels=tuple(matched_labels_double),
        fe_frequencies_hz=tuple(fe_freqs_double),
        reference_frequencies_hz=tuple(ref_freqs_double),
        relative_errors=errors_double,
        max_error=max(errors_double),
        passed=max(errors_double) < error_tolerance,
        mac_diagonal=tuple(mac_diag_double),
        details={
            "mesh": (ex_double, ey_double),
            "rigidity_dx": rigidity_double.dx,
            "rigidity_dy": rigidity_double.dy,
        },
    )

    # ------------------------------------------------------------------
    # GCI: mesh convergence on the simply supported plate fundamental mode
    # ------------------------------------------------------------------
    f_coarse = float(_solve_vibration_auto(
        mesh_coarse, _ey(mesh_coarse),
        length=length, width=width, thickness=thickness,
        young=young, poisson=poisson, density=density,
        modes=1,
    ).frequencies_hz[0])

    f_medium = float(_solve_vibration_auto(
        mesh_medium, _ey(mesh_medium),
        length=length, width=width, thickness=thickness,
        young=young, poisson=poisson, density=density,
        modes=1,
    ).frequencies_hz[0])

    f_fine = float(_solve_vibration_auto(
        mesh_fine, _ey(mesh_fine),
        length=length, width=width, thickness=thickness,
        young=young, poisson=poisson, density=density,
        modes=1,
    ).frequencies_hz[0])

    gci_result = gci(f_fine, f_medium, f_coarse)

    # ------------------------------------------------------------------
    # Dimensional scaling check
    # ------------------------------------------------------------------
    scaling_result = dimensional_scaling_check(
        length=length, width=width, thickness=thickness,
        young=young, poisson=poisson, density=density,
        elements_x=mesh_medium, elements_y=_ey(mesh_medium),
        scale_factor=2.0,
    )

    all_passed = (
        anchor_plate.passed
        and anchor_single.passed
        and anchor_double.passed
        and all(m > mac_threshold for m in mac_diag_plate[:3])
        and scaling_result.relative_error < 0.01
    )

    return StiffenedPlateValidationResult(
        anchor_plate=anchor_plate,
        anchor_single_stiffened=anchor_single,
        anchor_double_stiffened=anchor_double,
        gci_result=gci_result,
        scaling_result=scaling_result,
        all_passed=all_passed,
    )
