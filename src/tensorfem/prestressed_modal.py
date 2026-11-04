"""Stress-stiffened (prestressed) free vibration of beams and Mindlin plates.

Why this module exists
----------------------
Before this module the library could compute a *buckling* load
(:mod:`tensorfem.buckling`, :mod:`tensorfem.shell4_buckling`) and a
*stress-free* natural frequency (:mod:`tensorfem.modal`,
:mod:`tensorfem.plate_vibration`), but the two capabilities were never joined:
the natural frequency of a member that already carries an in-plane load could
not be computed at all.  Deck and side-shell panels are almost always under
compressive in-plane load when their vibration matters (machinery excitation,
whipping, springing), and a tensioned member (mooring line, stay, taut wire)
loses stiffness when the tension is removed.  The missing operator made that
whole question unanswerable.

Formulation
-----------
Linear (initial-stress, "Southwell") eigenproblem of the existing elements::

    (K + N * Kg) phi = omega^2 M phi

* ``K``  is the element elastic stiffness already verified by the bending,
  buckling and vibration benchmarks (Euler--Bernoulli beam::
  :func:`tensorfem.modal.beam_bending_stiffness`; Q4 selective-reduced
  Mindlin plate:: :func:`tensorfem.plate.q4_mindlin_stiffness`).
* ``M``  is the matching consistent mass (:func:`tensorfem.modal.beam_consistent_mass`,
  :func:`tensorfem.plate_vibration.q4_plate_mass_matrix`).
* ``Kg`` is the initial-stress (geometric) stiffness of the *same* mesh.  For
  the beam it is the standard consistent matrix already used for column
  buckling (:func:`tensorfem.buckling.beam_geometric_stiffness_unit`); for the
  plate it is integrated here from the bilinear transverse shape gradients.

Sign convention: ``N > 0`` is **tension** (stiffening), ``N < 0`` is
**compression**.  Note that :mod:`tensorfem.shell4_buckling` documents the
opposite (compression-positive) convention for its resultants; the two are
related by a sign and this module does not import that operator.

The prestress is a *given* constant force resultant; it is not updated after
deformation (no follower load, no nonlinear pre-buckling path) and the load is
applied through dead load only.

Independent closed-form references
----------------------------------
* Pinned--pinned Euler--Bernoulli beam carrying constant axial force ``N``::

      omega_n = sqrt( (n pi/L)^2 * (EI (n pi/L)^2 + N) / (rho A) )

  which vanishes exactly at the Euler load ``N_E,n = EI (n pi/L)^2`` and
  degenerates to the taut-string frequency ``(n pi/L) sqrt(N/(rho A))`` as
  tension dominates.

* Simply supported rectangular Kirchhoff plate with uniform biaxial
  in-plane force resultants (``Nx``, ``Ny`` positive in tension)::

      omega_mn^2 = ( D pi^4 R^2 + Nx (m pi/a)^2 + Ny (n pi/b)^2 ) / (rho h)
      R          = (m pi/a)^2 + (n pi/b)^2,   D = E h^3/[12(1-nu^2)]

  and the same expression with ``omega = 0`` gives the classical Navier
  buckling resultant ``N_cr = -D R`` for that half-wave pair.

Both references are exact continuum results; none of them is produced by this
code, and the finite-element comparison is reported with its discretisation
error instead of being asserted.

Exact discrete identity
-----------------------
For any mesh, if ``L`` is the smallest magnitude of the (negative) load
multiplier that makes the *reduced* elastic operator singular, then
``K + (-L) Kg`` has a null vector, so the fundamental frequency must be zero at
``N = -L``.  This identity is exact for the implemented operator pair and is
therefore a machine-precision test of the geometric-stiffness assembly, the
constraint reduction and the eigen-solver, independent of any continuum
reference.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .buckling import beam_geometric_stiffness_unit
from .modal import beam_bending_stiffness, beam_consistent_mass
from .plate import q4_mindlin_stiffness
from .plate_vibration import (
    BOUNDARY_CLAMPED,
    BOUNDARY_SIMPLE,
    _fixed_dofs,
    q4_plate_mass_matrix,
    structured_rectangular_mesh,
)

__all__ = [
    "StressStiffenedResult",
    "q4_plate_geometric_stiffness",
    "solve_stress_stiffened_modes",
    "beam_stress_stiffened_modes",
    "plate_stress_stiffened_modes",
    "plate_rigidity",
    "exact_beam_tension_frequency",
    "exact_beam_euler_load",
    "exact_plate_tension_frequency",
    "exact_plate_buckling_resultant",
]

_SUPPORTED_PLATE_BOUNDARIES = (BOUNDARY_SIMPLE, BOUNDARY_CLAMPED)

_ROOT3 = 1.0 / math.sqrt(3.0)
_FULL_POINTS = ((-_ROOT3, -_ROOT3), (_ROOT3, -_ROOT3), (_ROOT3, _ROOT3), (-_ROOT3, _ROOT3))
_PLATE_TRANSVERSE = (0, 3, 6, 9)


def _positive(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _finite(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _count(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


# ---------------------------------------------------------------------------
# Element geometric stiffness
# ---------------------------------------------------------------------------

def q4_plate_geometric_stiffness(xy: torch.Tensor, resultants, *, integration: str = "full") -> torch.Tensor:
    """Return the 12x12 initial-stress matrix of a flat Q4 Mindlin plate.

    ``resultants = (Nx, Ny, Nxy)`` are force per unit length with **tension
    positive**.  The matrix acts on the transverse displacement of the
    ``(w, theta_x, theta_y)`` node triple of :mod:`tensorfem.plate`; rotation
    degrees of freedom are not touched, matching the Kirchhoff/Mindlin
    assumption that the in-plane load does no work on rotations.

    ``integration="full"`` uses the 2x2 Gauss rule; ``"reduced"`` uses the
    single point at the element centre, which is what a one-point reduced
    integration of the transverse term would produce.  Bilinear gradients are
    integrated exactly by the 2x2 rule on a rectangle, so ``"full"`` is the
    default.
    """
    if not isinstance(xy, torch.Tensor) or xy.shape != (4, 2):
        raise ValueError("xy must be a 4x2 floating-point tensor")
    if not xy.dtype.is_floating_point:
        raise TypeError("xy must be a floating-point tensor")
    N = torch.as_tensor(resultants, dtype=xy.dtype, device=xy.device)
    if N.shape != (3,) or not bool(torch.all(torch.isfinite(N))):
        raise ValueError("resultants must be a finite (Nx, Ny, Nxy) triple")
    if integration not in ("full", "reduced"):
        raise ValueError("integration must be 'full' or 'reduced'")
    points = _FULL_POINTS if integration == "full" else ((0.0, 0.0),)
    resultants_matrix = torch.stack((torch.stack((N[0], N[2])), torch.stack((N[2], N[1]))))
    kg4 = xy.new_zeros((4, 4))
    for xi, eta in points:
        dnat = xy.new_tensor(
            [
                [-(1 - eta), -(1 - xi)],
                [(1 - eta), -(1 + xi)],
                [(1 + eta), (1 + xi)],
                [-(1 + eta), (1 - xi)],
            ]
        ) * 0.25
        jacobian = xy.T @ dnat
        determinant = torch.linalg.det(jacobian)
        if bool((determinant <= torch.finfo(xy.dtype).eps).item()):
            raise ValueError("degenerate or inverted plate element")
        gradient = dnat @ torch.linalg.inv(jacobian)
        weight = determinant if integration == "full" else 4.0 * determinant
        kg4 = kg4 + gradient @ resultants_matrix @ gradient.T * weight
    kg = xy.new_zeros((12, 12))
    transverse = torch.tensor(_PLATE_TRANSVERSE, dtype=torch.long)
    kg[transverse[:, None], transverse] = kg4
    return kg


# ---------------------------------------------------------------------------
# Closed-form references
# ---------------------------------------------------------------------------

def exact_beam_euler_load(mode: int, *, EI: float, length: float) -> float:
    """Euler load ``EI (n pi/L)^2`` of a pinned--pinned Euler--Bernoulli beam."""
    mode = _count("mode", mode)
    EI = _positive("EI", EI)
    length = _positive("length", length)
    return (mode * math.pi / length) ** 2 * EI


def exact_beam_tension_frequency(
    mode: int, *, EI: float, rho_a: float, length: float, axial_force: float = 0.0
) -> float:
    """Angular frequency of a pinned--pinned beam carrying constant axial force.

    ``axial_force > 0`` is tension.  The square root is real only while the
    member is stable (``axial_force > -EI (n pi/L)^2``); an unstable request is
    rejected instead of returning a NaN or an imaginary number.
    """
    mode = _count("mode", mode)
    EI = _positive("EI", EI)
    rho_a = _positive("rho_a", rho_a)
    length = _positive("length", length)
    axial_force = _finite("axial_force", axial_force)
    beta = mode * math.pi / length
    omega_squared = beta**2 * (EI * beta**2 + axial_force) / rho_a
    if omega_squared <= 0.0:
        raise ValueError(
            "axial_force is at or beyond the Euler load of this mode; the beam is unstable"
        )
    return math.sqrt(omega_squared)


def plate_rigidity(*, young: float, poisson: float, thickness: float) -> float:
    """Kirchhoff plate rigidity ``D = E h^3/[12 (1 - nu^2)]``."""
    young = _positive("young", young)
    thickness = _positive("thickness", thickness)
    poisson = _finite("poisson", poisson)
    if not -1.0 < poisson < 0.5:
        raise ValueError("poisson must lie in (-1, 0.5)")
    return young * thickness**3 / (12.0 * (1.0 - poisson**2))


def exact_plate_buckling_resultant(
    m: int,
    n: int,
    *,
    length: float,
    width: float,
    young: float,
    poisson: float,
    thickness: float,
    tension_y_ratio: float = 0.0,
    shear_ratio: float = 0.0,
) -> float:
    """Navier buckling resultant of a simply supported plate under a biaxial pattern.

    Returns the (negative) ``Nx`` at which the ``(m, n)`` half-wave pair becomes
    neutral for ``Ny = tension_y_ratio * Nx``.  Without shear the neutral
    condition ``D G^2 + Nx (m pi/a)^2 + Ny (n pi/b)^2 = 0`` gives::

        Nx,cr = -D G^2 / [ (m pi/a)^2 + tension_y_ratio (n pi/b)^2 ],
        G     = (m pi/a)^2 + (n pi/b)^2

    which reduces to the textbook ``-pi^2 D (m^2/a^2 + n^2/b^2)`` for uniaxial
    compression of a square plate at ``m = n = 1``.  A nonzero
    ``shear_ratio`` has no separable Navier solution and is rejected rather than
    approximated.
    """
    m = _count("m", m)
    n = _count("n", n)
    length = _positive("length", length)
    width = _positive("width", width)
    tension_y_ratio = _finite("tension_y_ratio", tension_y_ratio)
    shear_ratio = _finite("shear_ratio", shear_ratio)
    if shear_ratio != 0.0:
        raise ValueError("no separable Navier buckling reference exists for in-plane shear")
    rigidity = plate_rigidity(young=young, poisson=poisson, thickness=thickness)
    wave_x = (m * math.pi / length) ** 2
    wave_y = (n * math.pi / width) ** 2
    driver = wave_x + tension_y_ratio * wave_y
    if driver <= 0.0:
        raise ValueError("the load pattern does not destabilise this mode")
    return -rigidity * (wave_x + wave_y) ** 2 / driver


def exact_plate_tension_frequency(
    m: int,
    n: int,
    *,
    length: float,
    width: float,
    young: float,
    poisson: float,
    thickness: float,
    density: float,
    tension_x: float = 0.0,
    tension_y: float = 0.0,
) -> float:
    """Angular frequency of a simply supported prestressed Kirchhoff plate."""
    m = _count("m", m)
    n = _count("n", n)
    density = _positive("density", density)
    thickness = _positive("thickness", thickness)
    tension_x = _finite("tension_x", tension_x)
    tension_y = _finite("tension_y", tension_y)
    rigidity = plate_rigidity(young=young, poisson=poisson, thickness=thickness)
    wave_x = (m * math.pi / length) ** 2
    wave_y = (n * math.pi / width) ** 2
    membrane = tension_x * wave_x + tension_y * wave_y
    omega_squared = (rigidity * (wave_x + wave_y) ** 2 + membrane) / (density * thickness)
    if omega_squared <= 0.0:
        raise ValueError("prestress is at or beyond the buckling load of this mode; the plate is unstable")
    return math.sqrt(omega_squared)


# ---------------------------------------------------------------------------
# Prestressed eigen-solver
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StressStiffenedResult:
    """Prestressed free-vibration result of one load level."""

    load: float
    angular_frequencies: torch.Tensor
    frequencies_hz: torch.Tensor
    modes: torch.Tensor
    critical_tension: float
    """Negative ``load`` at which the reduced elastic operator becomes singular
    (the linear buckling load of the same discretisation)."""
    smallest_squared_frequency: float
    """Smallest eigenvalue of the standard-form problem, before any filtering.
    Zero (to round-off) exactly at ``load = critical_tension``."""
    discarded_modes: int
    retained_dofs: torch.Tensor | None = None
    nodes: torch.Tensor | None = None
    elements: torch.Tensor | None = None

    @property
    def critical_compression(self) -> float:
        """Magnitude of the compressive load that buckles this discretisation."""
        return -self.critical_tension

    def relative_errors(self, reference: "torch.Tensor | list[float] | tuple[float, ...]") -> torch.Tensor:
        """Relative error of the retained frequencies against ``reference``."""
        target = torch.as_tensor(reference, dtype=self.angular_frequencies.dtype)
        if target.shape != self.angular_frequencies.shape:
            raise ValueError("reference must match the number of retained frequencies")
        return torch.abs(self.angular_frequencies - target) / torch.abs(target)


def _reduced_system(stiffness, mass, geometric_unit, constrained_dofs):
    for name, matrix in (("stiffness", stiffness), ("mass", mass), ("geometric_unit", geometric_unit)):
        if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2:
            raise ValueError(f"{name} must be a square floating-point tensor")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} must be square")
    if not (stiffness.shape == mass.shape == geometric_unit.shape):
        raise ValueError("stiffness, mass and geometric_unit must have equal shape")
    n = stiffness.shape[0]
    constrained = {int(d) for d in constrained_dofs}
    if any(d < 0 or d >= n for d in constrained):
        raise ValueError("constrained_dofs contains an index outside the matrix")
    free = torch.tensor([i for i in range(n) if i not in constrained], dtype=torch.long)
    if free.numel() == 0:
        raise ValueError("at least one free degree of freedom is required")
    return (
        free,
        stiffness[free[:, None], free],
        mass[free[:, None], free],
        geometric_unit[free[:, None], free],
    )


def _unit_load_singularity(stiffness: torch.Tensor, geometric_unit: torch.Tensor) -> float:
    """Smallest magnitude negative load multiplier that makes ``K + s*Kg`` singular.

    Uses the reciprocal form ``K^-1/2 Kg K^-T/2``; ``K`` must be positive
    definite, which holds for every supported mesh used here.
    """
    try:
        chol = torch.linalg.cholesky(0.5 * (stiffness + stiffness.T))
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("elastic stiffness is not positive definite on the free dofs") from exc
    left = torch.linalg.solve_triangular(chol, geometric_unit, upper=False)
    standard = torch.linalg.solve_triangular(chol, left.T, upper=False).T
    standard = 0.5 * (standard + standard.T)
    mu = torch.linalg.eigvalsh(standard)
    mu_max = float(mu.max())
    if not math.isfinite(mu_max) or mu_max <= 0.0:
        raise ValueError("the geometric pattern cannot destabilise this discretisation")
    return 1.0 / mu_max


def solve_stress_stiffened_modes(
    stiffness: torch.Tensor,
    mass: torch.Tensor,
    geometric_unit: torch.Tensor,
    constrained_dofs,
    load: float,
    modes: int = 4,
    *,
    zero_tolerance: float = 1e-10,
    allow_mechanism: bool = False,
) -> StressStiffenedResult:
    """Solve ``(K + load*Kg) phi = omega^2 M phi`` on the free degrees of freedom.

    ``load > 0`` is tension.  The linear buckling load of the same
    discretisation is reported in ``critical_tension``; requesting a load at or
    beyond it is rejected instead of silently returning a post-buckling
    (indefinite) spectrum.  ``zero_tolerance`` is relative to the largest
    eigenvalue of the standard-form problem and is used only to count and drop
    numerically zero modes; ``smallest_squared_frequency`` keeps the raw value.
    """
    modes = _count("modes", modes)
    load = _finite("load", load)
    zero_tolerance = _finite("zero_tolerance", zero_tolerance)
    if zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be non-negative")
    free, stiffness_free, mass_free, geometric_free = _reduced_system(
        stiffness, mass, geometric_unit, constrained_dofs
    )
    critical_tension = -_unit_load_singularity(stiffness_free, geometric_free)
    margin = 1e-12 * max(1.0, abs(critical_tension))
    if load < critical_tension - margin:
        raise ValueError(
            f"load {load:g} is beyond the linear buckling load {critical_tension:g} "
            "of this discretisation; the prestressed operator is indefinite"
        )
    total = stiffness_free + load * geometric_free
    total = 0.5 * (total + total.T)
    if modes > free.numel():
        raise ValueError("modes exceeds the number of free degrees of freedom")
    factor = torch.linalg.cholesky(0.5 * (mass_free + mass_free.T))
    inverse_factor = torch.linalg.inv(factor)
    standard = inverse_factor @ total @ inverse_factor.T
    standard = 0.5 * (standard + standard.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(standard)
    scale = float(eigenvalues.abs().max())
    threshold = zero_tolerance * scale
    negative = int(torch.sum(eigenvalues < -threshold).item())
    if negative and not allow_mechanism:
        raise ValueError(
            f"prestressed operator has {negative} negative eigenvalue(s); "
            "the requested prestress is unstable"
        )
    keep = eigenvalues > threshold
    retained = eigenvalues[keep][:modes]
    if retained.numel() < modes:
        raise ValueError(
            f"requested modes exceed the elastic spectrum ({int(keep.sum().item())} available)"
        )
    vectors = eigenvectors[:, keep][:, :modes]
    angular = torch.sqrt(torch.clamp(retained, min=0.0))
    full_modes = torch.zeros((stiffness.shape[0], modes), dtype=stiffness.dtype)
    full_modes[free] = inverse_factor.T @ vectors
    return StressStiffenedResult(
        load=load,
        angular_frequencies=angular,
        frequencies_hz=angular / (2.0 * math.pi),
        modes=full_modes,
        critical_tension=float(critical_tension),
        smallest_squared_frequency=float(eigenvalues[0]),
        discarded_modes=int((~keep).sum().item()),
        retained_dofs=free,
    )


# ---------------------------------------------------------------------------
# Convenience models
# ---------------------------------------------------------------------------

def beam_stress_stiffened_modes(
    length: float = 1.0,
    elements: int = 8,
    *,
    EI: float = 1.0e4,
    rho_a: float = 10.0,
    axial_force: float = 0.0,
    modes: int = 4,
    boundary: str = "pinned",
) -> StressStiffenedResult:
    """Prestressed vibration of a uniform Euler--Bernoulli beam.

    ``boundary`` is ``"pinned"`` (transverse displacement fixed at both ends of
    the Euler--Bernoulli model) or ``"cantilever"`` (clamped at ``x=0``).
    ``axial_force > 0`` is tension.
    """
    length = _positive("length", length)
    elements = _count("elements", elements)
    EI = _positive("EI", EI)
    rho_a = _positive("rho_a", rho_a)
    axial_force = _finite("axial_force", axial_force)
    if boundary not in ("pinned", "cantilever"):
        raise ValueError("boundary must be 'pinned' or 'cantilever'")
    element_length = length / elements
    ke = beam_bending_stiffness(EI, element_length)
    me = beam_consistent_mass(rho_a, element_length)
    kge = beam_geometric_stiffness_unit(element_length)
    ndof = 2 * (elements + 1)
    stiffness = torch.zeros((ndof, ndof), dtype=ke.dtype)
    mass = torch.zeros_like(stiffness)
    geometric = torch.zeros_like(stiffness)
    for element in range(elements):
        dofs = torch.tensor([2 * element, 2 * element + 1, 2 * element + 2, 2 * element + 3])
        stiffness[dofs[:, None], dofs] += ke
        mass[dofs[:, None], dofs] += me
        geometric[dofs[:, None], dofs] += kge
    constrained = (0, 2 * elements) if boundary == "pinned" else (0, 1)
    result = solve_stress_stiffened_modes(
        stiffness, mass, geometric, constrained, axial_force, modes
    )
    return StressStiffenedResult(
        load=result.load,
        angular_frequencies=result.angular_frequencies,
        frequencies_hz=result.frequencies_hz,
        modes=result.modes,
        critical_tension=result.critical_tension,
        smallest_squared_frequency=result.smallest_squared_frequency,
        discarded_modes=result.discarded_modes,
        retained_dofs=result.retained_dofs,
    )


def plate_stress_stiffened_modes(
    elements_x: int = 8,
    elements_y: int = 8,
    *,
    length: float = 1.0,
    width: float | None = None,
    young: float = 200.0e9,
    poisson: float = 0.3,
    thickness: float = 0.01,
    density: float = 7850.0,
    tension_x: float = 0.0,
    tension_y_ratio: float = 0.0,
    shear_ratio: float = 0.0,
    boundary: str = BOUNDARY_SIMPLE,
    modes: int = 4,
    rotary_inertia: bool = True,
    integration: str = "full",
) -> StressStiffenedResult:
    """Prestressed vibration of a flat Q4 Mindlin plate.

    ``tension_x`` is the in-plane force resultant per unit length along ``x``
    (tension positive); ``tension_y = tension_y_ratio * tension_x`` and the
    in-plane shear resultant is ``shear_ratio * tension_x``.  The critical load
    reported by the result is the value of ``tension_x`` at which this
    discretisation becomes singular for the requested pattern.
    """
    elements_x = _count("elements_x", elements_x)
    elements_y = _count("elements_y", elements_y)
    length = _positive("length", length)
    width = length if width is None else _positive("width", width)
    thickness = _positive("thickness", thickness)
    density = _positive("density", density)
    young = _positive("young", young)
    poisson = _finite("poisson", poisson)
    if not -1.0 < poisson < 0.5:
        raise ValueError("poisson must lie in (-1, 0.5)")
    tension_x = _finite("tension_x", tension_x)
    tension_y_ratio = _finite("tension_y_ratio", tension_y_ratio)
    shear_ratio = _finite("shear_ratio", shear_ratio)
    if boundary not in _SUPPORTED_PLATE_BOUNDARIES:
        raise ValueError(f"boundary must be one of {_SUPPORTED_PLATE_BOUNDARIES}")
    modes = _count("modes", modes)

    nodes, elements = structured_rectangular_mesh(elements_x, elements_y, length, width)
    ndof = 3 * nodes.shape[0]
    stiffness = torch.zeros((ndof, ndof), dtype=nodes.dtype)
    mass = torch.zeros_like(stiffness)
    geometric = torch.zeros_like(stiffness)
    pattern = nodes.new_tensor([1.0, tension_y_ratio, shear_ratio])
    for connectivity in elements:
        xy = nodes[connectivity]
        element_stiffness = q4_mindlin_stiffness(xy, young, poisson, thickness)
        element_mass = q4_plate_mass_matrix(xy, density, thickness, rotary_inertia=rotary_inertia)
        element_geometric = q4_plate_geometric_stiffness(xy, pattern, integration=integration)
        dofs = torch.stack((3 * connectivity, 3 * connectivity + 1, 3 * connectivity + 2), dim=1).reshape(-1)
        stiffness[dofs[:, None], dofs] += element_stiffness
        mass[dofs[:, None], dofs] += element_mass
        geometric[dofs[:, None], dofs] += element_geometric

    fixed = _fixed_dofs(nodes, length, width, boundary, nodes.dtype)
    result = solve_stress_stiffened_modes(stiffness, mass, geometric, fixed, tension_x, modes)
    return StressStiffenedResult(
        load=result.load,
        angular_frequencies=result.angular_frequencies,
        frequencies_hz=result.frequencies_hz,
        modes=result.modes,
        critical_tension=result.critical_tension,
        smallest_squared_frequency=result.smallest_squared_frequency,
        discarded_modes=result.discarded_modes,
        retained_dofs=result.retained_dofs,
        nodes=nodes,
        elements=elements,
    )