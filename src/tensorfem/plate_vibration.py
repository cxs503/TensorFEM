"""Free vibration of Mindlin--Reissner plates, with or without a Winkler foundation.

Why this module exists
----------------------
Before this module the library could solve plate *bending*
(``tensorfem.plate``, ``tensorfem.plate_mindlin``) and *beam* vibration
(``tensorfem.modal``), but it had no plate or shell eigenvalue capability.
Panel natural frequencies are the entry quantity of any ship
vibro-acoustic or fatigue screening study, so the gap blocked a whole
discipline rather than a single benchmark.

Formulation
-----------
The element is the existing four-node selective-reduced-integration (SRI)
Mindlin plate of :func:`tensorfem.plate.q4_mindlin_stiffness`; only a
consistent mass matrix is added here:

    M_e = integral_Omega_e [ rho h N^T N  (w-w)  +  (rho h^3/12) N^T N  (theta-theta) ] dOmega

The rotary-inertia block is what keeps ``M`` positive definite: a
translation-only mass matrix is singular in ``theta`` and the generalised
eigenproblem cannot be reduced by Cholesky.  Keeping the block is also the
physically consistent Mindlin choice (Mindlin 1951).  For the thin plates
verified here (``a/h = 100``) the rotary block changes frequencies by less
than 0.05 %, so it is retained for conditioning, not to model thick plates.

Free vibration then solves

    K phi = omega^2 M phi,      K = K_plate + k_w M_foundation

with the fixed degrees of freedom eliminated before the eigen-solve and the
problem reduced to a standard symmetric form through the Cholesky factor of
``M``.

Independent references (all closed form, none of them produced by this code)
---------------------------------------------------------------------------
* Simply supported rectangular plate, Navier double-sine solution::

      omega_mn = pi^2 sqrt(D / (rho h)) [ (m/a)^2 + (n/b)^2 ]
      D        = E h^3 / (12 (1 - nu^2))

* Simply supported plate on a uniform Winkler foundation: the sine mode is
  still an exact eigenfunction of the foundation operator, hence::

      omega_mn^2 = ( D R_mn^2 + k_w ) / (rho h),   R_mn = (m pi/a)^2 + (n pi/b)^2

  A uniform foundation therefore *cannot* be used to hide discretisation
  error in the bending term: it is an additive exactness check.

* Free plate on a uniform Winkler foundation: the vertical translation mode is
  still exact, ``omega^2 = k_w / (rho h)``, with no boundary condition involved
  (``winkler_translation_angular_frequency``).  The remaining family is the
  free-plate spectrum shifted by ``k_w/(rho h)`` in ``omega^2``; that shift is
  exact in the Kirchhoff model and is supplied as an independent reference by
  ``tensorfem.plate_ritz_reference`` with ``winkler_parameter``.

* Clamped square plate, Leissa (NASA SP-160, 1969) tabulated frequency
  coefficient ``lambda_1 = 35.9853``::

      omega_1 = (lambda_1 / a^2) sqrt(D / (rho h))

* Shear-deformation bias of the thin-plate reference: the classical first
  order correction to the bending branch is
  ``omega^2 = omega_K^2 / (1 + D R^2/(kappa G h))``.  For ``a/h = 100`` this
  is ``-3e-4`` in ``omega``, i.e. the Kirchhoff reference is a fair target
  only to that level; the verification documents the residual explicitly.

Scope limits (do not over-read the qualification)
------------------------------------------------
* Uniform thickness, isotropic, small strain, linear elastic.
* ``simple`` support is the hard (SS1) convention: ``w = 0`` on all edges,
  ``theta_n = 0`` on the edges normal to each rotation.  This is the
  convention already used by :func:`tensorfem.plate.solve_simply_supported_sine_square`
  so bending and vibration friction-free comparisons share one BC set.
* No in-plane prestress, no added mass, no fluid coupling, no damping, no
  geometric imperfection.  Coupling to added mass (``tensorfem.added_mass``)
  or prestress (``tensorfem.shell4_buckling``) is deliberately out of scope.
* Dense algebra: cost is ``O(ndof^3)``.  Meshes beyond about ``24 x 24``
  elements should use a sparse eigensolver; the module refuses nothing but
  documents the limit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .plate import PlateRigidity, _shape, q4_mindlin_stiffness
from .plate_mindlin import q4_winkler_foundation_stiffness

#: Leissa (NASA SP-160, 1969) frequency coefficient for the fundamental mode of
#: a fully clamped square plate: ``omega = lambda a^-2 sqrt(D/(rho h))``.
LEISSA_CLAMPED_SQUARE_LAMBDA = 35.9853

#: Transverse shear correction factor shared with ``tensorfem.plate``.
SHEAR_CORRECTION = 5.0 / 6.0

BOUNDARY_SIMPLE = "simple"
BOUNDARY_CLAMPED = "clamped"
BOUNDARY_FREE = "free"
_SUPPORTED_BOUNDARIES = (BOUNDARY_SIMPLE, BOUNDARY_CLAMPED, BOUNDARY_FREE)


@dataclass(frozen=True)
class PlateVibrationResult:
    """Eigen-solution of a Mindlin plate on an optional Winkler foundation."""

    frequencies_hz: torch.Tensor      # (modes,)
    angular_frequencies: torch.Tensor  # (modes,)
    modes: torch.Tensor               # (ndof, modes) in the full dof space
    nodes: torch.Tensor               # (n_nodes, 2)
    elements: torch.Tensor            # (n_elements, 4)
    retained_dofs: torch.Tensor       # free dofs used in the eigen-solve
    total_mass: float                 # rho h A, exact for the consistent matrix
    element_counts: tuple[int, int]
    boundary: str
    winkler_k: float
    rigid_body_modes: int = 0         # zero-energy eigenpairs discarded before the elastic modes


@dataclass(frozen=True)
class ConvergenceRow:
    """One mesh level of a frequency convergence study."""

    elements_x: int
    elements_y: int
    frequency_hz: float
    relative_error: float


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _poisson(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("poisson must be a real number")
    value = float(value)
    if not math.isfinite(value) or not -1.0 < value < 0.5:
        raise ValueError("poisson must be finite and lie between -1 and 0.5")
    return value


def structured_rectangular_mesh(
    elements_x: int, elements_y: int, length: float = 1.0, width: float | None = None,
    *, dtype=torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a structured ``elements_x`` by ``elements_y`` Q4 mesh on ``[0,length] x [0,width]``."""
    elements_x = _count("elements_x", elements_x)
    elements_y = _count("elements_y", elements_y)
    length = _positive("length", length)
    width = length if width is None else _positive("width", width)
    x = torch.linspace(0.0, length, elements_x + 1, dtype=dtype)
    y = torch.linspace(0.0, width, elements_y + 1, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    nodes = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    stride = elements_x + 1
    elements = torch.tensor(
        [
            (j * stride + i, j * stride + i + 1, j * stride + i + stride + 1,
             j * stride + i + stride)
            for j in range(elements_y) for i in range(elements_x)
        ],
        dtype=torch.long,
    )
    return nodes, elements


def q4_plate_mass_matrix(
    xy: torch.Tensor, density: float | torch.Tensor, thickness: float | torch.Tensor,
    *, rotary_inertia: bool = True, shear_correction: float = SHEAR_CORRECTION,
) -> torch.Tensor:
    """Return the 12x12 consistent mass matrix of a Q4 Mindlin plate element.

    ``density`` and ``thickness`` may be tensors (they are broadcast like the
    stiffness builder).  The translatory block is ``rho h``; the rotation block
    is ``rho h^3/12`` when ``rotary_inertia`` is true, which is what makes the
    assembled mass matrix positive definite.
    """
    if xy.shape != (4, 2):
        raise ValueError("xy must have shape (4, 2) in counter-clockwise order")
    dtype, device = xy.dtype, xy.device
    rho = torch.as_tensor(density, dtype=dtype, device=device)
    t = torch.as_tensor(thickness, dtype=dtype, device=device)
    if bool((rho <= 0).item()):
        raise ValueError("density must be positive")
    if bool((t <= 0).item()):
        raise ValueError("thickness must be positive")
    if not bool(torch.as_tensor(rotary_inertia, dtype=torch.bool).item()):
        rotary_scale = torch.zeros_like(t)
    else:
        rotary_scale = rho * t**3 / 12.0
    translatory = rho * t
    mass = torch.zeros((12, 12), dtype=dtype, device=device)
    gp = 1.0 / 3.0**0.5
    for xif in (-gp, gp):
        for etaf in (-gp, gp):
            xi, eta = xy.new_tensor(xif), xy.new_tensor(etaf)
            N, dN_nat = _shape(xi, eta)
            jacobian = dN_nat.T @ xy
            det_j = torch.linalg.det(jacobian)
            if bool((det_j <= 0).item()):
                raise ValueError("plate element has non-positive Jacobian")
            shape_outer = torch.outer(N, N) * det_j
            blocks = mass.view(4, 3, 4, 3)
            blocks[:, 0, :, 0] += translatory * shape_outer
            blocks[:, 1, :, 1] += rotary_scale * shape_outer
            blocks[:, 2, :, 2] += rotary_scale * shape_outer
    return mass


def plate_rigidity(young: float, poisson: float, thickness: float) -> float:
    """Return the flexural rigidity ``D = E h^3 / (12 (1 - nu^2))``."""
    young = _positive("young", young)
    thickness = _positive("thickness", thickness)
    poisson = _poisson(poisson)
    return young * thickness**3 / (12.0 * (1.0 - poisson**2))


def kirchhoff_angular_frequency(
    *, length: float, width: float, density: float, young: float, poisson: float,
    thickness: float, half_waves_x: int = 1, half_waves_y: int = 1,
) -> float:
    """Navier exact angular frequency of a simply supported rectangular plate."""
    length = _positive("length", length)
    width = _positive("width", width)
    density = _positive("density", density)
    half_waves_x = _count("half_waves_x", half_waves_x)
    half_waves_y = _count("half_waves_y", half_waves_y)
    rigidity = plate_rigidity(young, poisson, thickness)
    wave_numbers = (half_waves_x * math.pi / length) ** 2 + (half_waves_y * math.pi / width) ** 2
    return math.sqrt(rigidity * wave_numbers**2 / (density * thickness))


def winkler_angular_frequency(
    *, length: float, width: float, density: float, young: float, poisson: float,
    thickness: float, winkler_k: float, half_waves_x: int = 1, half_waves_y: int = 1,
) -> float:
    """Exact angular frequency of a simply supported plate on a uniform Winkler bed.

    The Navier sine mode is an exact eigenfunction of the uniform foundation
    operator, so ``omega^2 = (D R^2 + k_w) / (rho h)`` holds without further
    approximation.  ``winkler_k`` is the foundation modulus per unit area
    (N/m^3).
    """
    if isinstance(winkler_k, bool) or not isinstance(winkler_k, (int, float)):
        raise TypeError("winkler_k must be a real number")
    winkler_k = float(winkler_k)
    if not math.isfinite(winkler_k) or winkler_k < 0.0:
        raise ValueError("winkler_k must be finite and non-negative")
    length = _positive("length", length)
    width = _positive("width", width)
    density = _positive("density", density)
    thickness = _positive("thickness", thickness)
    half_waves_x = _count("half_waves_x", half_waves_x)
    half_waves_y = _count("half_waves_y", half_waves_y)
    rigidity = plate_rigidity(young, poisson, thickness)
    wave_numbers = (half_waves_x * math.pi / length) ** 2 + (half_waves_y * math.pi / width) ** 2
    return math.sqrt((rigidity * wave_numbers**2 + winkler_k) / (density * thickness))


def winkler_translation_angular_frequency(
    *, density: float, thickness: float, winkler_k: float,
) -> float:
    """Exact angular frequency of the vertical translation mode on a Winkler bed.

    A plate that is held *only* by a uniform Winkler foundation has one exact
    eigenpair that does not depend on its boundary conditions, its bending
    stiffness or its planform: the vertical translation ``w = const``,
    ``theta_x = theta_y = 0`` gives ``U = k_w/2 int w^2`` and
    ``T = rho h/2 int w^2``, hence

        omega_translation^2 = k_w / (rho h),

    *exactly*, for a free edge as well.  Discretely the same is true of any
    consistent Q4 element set (the consistent foundation matrix and the
    translatory consistent mass share the same shape-function products), which
    is what makes this an anchor on the *assembly* rather than on the mesh:
    measured drift is at machine precision.  It is the reference used for the
    free-edge foundation verification in
    ``docs/free-plate-winkler-foundation-validation.md``.
    """
    if isinstance(winkler_k, bool) or not isinstance(winkler_k, (int, float)):
        raise TypeError("winkler_k must be a real number")
    winkler_k = float(winkler_k)
    if not math.isfinite(winkler_k) or winkler_k < 0.0:
        raise ValueError("winkler_k must be finite and non-negative")
    density = _positive("density", density)
    thickness = _positive("thickness", thickness)
    return math.sqrt(winkler_k / (density * thickness))


def clamped_square_angular_frequency(
    *, length: float, density: float, young: float, poisson: float, thickness: float,
) -> float:
    """Leissa's clamped-square fundamental angular frequency (literature value)."""
    length = _positive("length", length)
    density = _positive("density", density)
    thickness = _positive("thickness", thickness)
    rigidity = plate_rigidity(young, poisson, thickness)
    return LEISSA_CLAMPED_SQUARE_LAMBDA / length**2 * math.sqrt(rigidity / (density * thickness))


def shear_deformation_bias(
    *, length: float, width: float, young: float, poisson: float, thickness: float,
    half_waves_x: int = 1, half_waves_y: int = 1,
    shear_correction: float = SHEAR_CORRECTION,
) -> float:
    """Relative bias of the Kirchhoff reference caused by transverse shear.

    The first-order shear correction of the bending branch is
    ``omega^2 = omega_K^2 / (1 + D R^2 / (kappa G h))``; the returned value is
    ``omega / omega_K - 1`` (negative for real materials), which bounds how
    closely a Mindlin element may approach the thin-plate reference.
    """
    rigidity = plate_rigidity(young, poisson, thickness)
    poisson = _poisson(poisson)
    shear_modulus = float(young) / (2.0 * (1.0 + poisson))
    wave_numbers = (half_waves_x * math.pi / length) ** 2 + (half_waves_y * math.pi / width) ** 2
    return 1.0 / math.sqrt(1.0 + rigidity * wave_numbers / (shear_correction * shear_modulus * thickness)) - 1.0


def _edge_flags(nodes: torch.Tensor, length: float, width: float, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    tol = 1e-12 * max(length, width)
    x_edge = (nodes[:, 0].abs() < tol) | ((nodes[:, 0] - length).abs() < tol)
    y_edge = (nodes[:, 1].abs() < tol) | ((nodes[:, 1] - width).abs() < tol)
    return x_edge, y_edge


def _fixed_dofs(nodes: torch.Tensor, length: float, width: float, boundary: str, dtype) -> torch.Tensor:
    if boundary == BOUNDARY_FREE:  # every dof is free; three rigid-body modes remain
        return torch.zeros(0, dtype=torch.long)
    x_edge, y_edge = _edge_flags(nodes, length, width, dtype)
    on_boundary = x_edge | y_edge
    transverse = 3 * torch.nonzero(on_boundary, as_tuple=False).flatten()
    if boundary == BOUNDARY_SIMPLE:
        # Hard simple support: theta_x is restrained on the y-edges and theta_y
        # on the x-edges, i.e. the rotation about the edge normal vanishes.
        rotations = torch.cat((
            3 * torch.nonzero(y_edge, as_tuple=False).flatten() + 1,
            3 * torch.nonzero(x_edge, as_tuple=False).flatten() + 2,
        ))
    elif boundary == BOUNDARY_CLAMPED:
        rotations = torch.cat((
            3 * torch.nonzero(on_boundary, as_tuple=False).flatten() + 1,
            3 * torch.nonzero(on_boundary, as_tuple=False).flatten() + 2,
        ))
    else:  # pragma: no cover - guarded by solve_plate_vibration
        raise ValueError(f"unsupported boundary {boundary!r}")
    return torch.cat((transverse, rotations)).unique()


def solve_plate_vibration(
    elements_x: int = 8, elements_y: int = 8, *, length: float = 1.0,
    width: float | None = None, young: float = 200.0e9, poisson: float = 0.3,
    thickness: float = 0.01, density: float = 7850.0, winkler_k: float = 0.0,
    boundary: str = BOUNDARY_SIMPLE, modes: int = 4, rotary_inertia: bool = True,
    rigid_body_relative_tolerance: float = 1e-9, dtype=torch.float64,
    rigidity: PlateRigidity | None = None,
) -> PlateVibrationResult:
    """Solve the free-vibration eigenproblem ``K phi = omega^2 M phi``.

    Parameters mirror the bending entry point of ``tensorfem.plate``.  Fixed
    degrees of freedom are removed before the eigen-solve; the remaining
    problem is reduced with the Cholesky factor of the mass matrix.

    ``rigidity`` optionally replaces the isotropic ``(young, poisson,
    thickness)`` stiffness by an orthotropic :class:`tensorfem.plate.PlateRigidity`;
    ``thickness`` and ``density`` still define the consistent mass matrix, and
    ``young``/``poisson`` are then only validated, not used.
    """
    elements_x = _count("elements_x", elements_x)
    elements_y = _count("elements_y", elements_y)
    modes = _count("modes", modes)
    if rigidity is not None and not isinstance(rigidity, PlateRigidity):
        raise TypeError("rigidity must be a tensorfem.plate.PlateRigidity")
    if boundary not in _SUPPORTED_BOUNDARIES:
        raise ValueError(f"boundary must be one of {_SUPPORTED_BOUNDARIES}")
    if isinstance(winkler_k, bool) or not isinstance(winkler_k, (int, float)):
        raise TypeError("winkler_k must be a real number")
    winkler_k = float(winkler_k)
    if not math.isfinite(winkler_k) or winkler_k < 0.0:
        raise ValueError("winkler_k must be finite and non-negative")
    length = _positive("length", length)
    width = length if width is None else _positive("width", width)
    density = _positive("density", density)
    thickness = _positive("thickness", thickness)
    poisson = _poisson(poisson)

    nodes, elements = structured_rectangular_mesh(
        elements_x, elements_y, length, width, dtype=dtype
    )
    ndof = 3 * nodes.shape[0]
    stiffness = torch.zeros((ndof, ndof), dtype=dtype)
    mass = torch.zeros_like(stiffness)
    for connectivity in elements:
        xy = nodes[connectivity]
        element_stiffness = (
            q4_mindlin_stiffness(xy, rigidity=rigidity) if rigidity is not None
            else q4_mindlin_stiffness(xy, young, poisson, thickness)
        )
        if winkler_k > 0.0:
            element_stiffness = element_stiffness + q4_winkler_foundation_stiffness(xy, winkler_k)
        element_mass = q4_plate_mass_matrix(
            xy, density, thickness, rotary_inertia=rotary_inertia
        )
        dofs = torch.stack(
            (3 * connectivity, 3 * connectivity + 1, 3 * connectivity + 2), dim=1
        ).reshape(-1)
        stiffness[dofs[:, None], dofs] += element_stiffness
        mass[dofs[:, None], dofs] += element_mass

    fixed = _fixed_dofs(nodes, length, width, boundary, dtype)
    all_dofs = torch.arange(ndof)
    free = all_dofs[~torch.isin(all_dofs, fixed)]
    if modes > free.numel():
        raise ValueError("modes exceeds the number of free degrees of freedom")

    reduced_stiffness = stiffness[free[:, None], free]
    reduced_mass = mass[free[:, None], free]
    try:
        factor = torch.linalg.cholesky(reduced_mass)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(
            "reduced mass matrix is not positive definite; keep rotary_inertia=True"
        ) from exc
    inverse_factor = torch.linalg.inv(factor)
    standard = inverse_factor @ reduced_stiffness @ inverse_factor.T
    standard = 0.5 * (standard + standard.T)
    if isinstance(rigid_body_relative_tolerance, bool) or not isinstance(
        rigid_body_relative_tolerance, (int, float)
    ):
        raise TypeError("rigid_body_relative_tolerance must be a real number")
    rigid_body_relative_tolerance = float(rigid_body_relative_tolerance)
    if not math.isfinite(rigid_body_relative_tolerance) or rigid_body_relative_tolerance < 0.0:
        raise ValueError("rigid_body_relative_tolerance must be finite and non-negative")
    eigenvalues, eigenvectors = torch.linalg.eigh(standard)
    scale = float(eigenvalues[-1])
    threshold = rigid_body_relative_tolerance * scale
    rigid = int(torch.sum(eigenvalues <= threshold).item())
    selected = eigenvalues[rigid : rigid + modes]
    if selected.numel() < modes:
        raise ValueError(
            "requested modes exceed the elastic spectrum"
            f" ({eigenvalues.numel() - rigid} elastic modes available)"
        )
    if bool((selected <= 0.0).any().item()):
        raise ValueError("negative eigenvalue: retained set still holds a mechanism")
    if boundary != BOUNDARY_FREE and rigid:
        raise ValueError(
            "spurious near-zero eigenvalue on a supported boundary: "
            "the element set still holds a mechanism"
        )
    angular = torch.sqrt(selected)
    full_modes = torch.zeros((ndof, modes), dtype=dtype)
    full_modes[free] = inverse_factor.T @ eigenvectors[:, rigid : rigid + modes]
    total_mass = float(density * thickness * length * width)
    return PlateVibrationResult(
        frequencies_hz=angular / (2.0 * math.pi),
        angular_frequencies=angular,
        modes=full_modes,
        nodes=nodes,
        elements=elements,
        retained_dofs=free,
        total_mass=total_mass,
        element_counts=(elements_x, elements_y),
        boundary=boundary,
        winkler_k=winkler_k,
        rigid_body_modes=rigid,
    )


def mode_convergence(
    reference_hz: float, *,
    meshes: tuple[int, ...] = (6, 8, 12, 16), mode: int = 1, **kwargs,
) -> tuple[ConvergenceRow, ...]:
    """Refine a mesh and report the relative error of one mode per level.

    ``kwargs`` are forwarded to :func:`solve_plate_vibration`; the returned
    rows carry the error recomputed from raw frequencies, never from a stored
    constant.
    """
    reference_hz = _positive("reference_hz", reference_hz)
    mode = _count("mode", mode)
    rows = []
    for mesh in meshes:
        result = solve_plate_vibration(elements_x=mesh, elements_y=mesh, modes=mode, **kwargs)
        frequency = float(result.frequencies_hz[mode - 1])
        rows.append(
            ConvergenceRow(
                elements_x=mesh,
                elements_y=mesh,
                frequency_hz=frequency,
                relative_error=abs(frequency - reference_hz) / reference_hz,
            )
        )
    return tuple(rows)


def observed_convergence_order(rows: tuple[ConvergenceRow, ...], reference_hz: float) -> float:
    """Estimate the convergence order from the last two mesh levels.

    Uses the signed error so a sign change (error crossing zero) is reported
    as ``nan`` rather than silently fitted.
    """
    reference_hz = _positive("reference_hz", reference_hz)
    if len(rows) < 2:
        raise ValueError("at least two mesh levels are required")
    first, second = rows[-2], rows[-1]
    e1 = (first.frequency_hz - reference_hz) / reference_hz
    e2 = (second.frequency_hz - reference_hz) / reference_hz
    if e1 * e2 >= 0.0 and e1 != e2:
        ratio = second.elements_x / first.elements_x
        return math.log(abs(e1 / e2)) / math.log(ratio)
    return float("nan")