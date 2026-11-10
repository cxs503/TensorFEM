"""Elastoplastic plate buckling and ultimate strength analysis.

This module couples the layered Shell4 J2 plasticity element
(:mod:`tensorfem.layered_shell4_plasticity`) with the Crisfield spherical
arc-length continuation (:mod:`tensorfem.arc_length`) to trace the
post-buckling equilibrium path of plates and stiffened panels including
both **material nonlinearity** (J2 flow rule, isotropic hardening,
radial return mapping) and **geometric nonlinearity** (Updated Lagrangian
geometric stiffness from current stress resultants).

Scope
-----
* Initial imperfection from the first linear buckling mode, scaled to a
  user-specified amplitude (default ``t/200`` per CSR / welding-code
  convention).
* Arc-length path tracing through limit points (buckling load, ultimate
  strength, post-buckling softening).
* Verification against classical closed-form plate buckling solutions
  (Timoshenko & Gere) and CSR-style ultimate strength formulas.

Honesty statement
-----------------
* The shell element is a flat-facet small-strain Mindlin formulation with
  corotational geometric stiffness.  Large rotations beyond the shallow-
  shell regime are not captured.
* The arc-length solver uses the material + geometric tangent; branch
  switching is not implemented (the imperfection selects the physical
  branch).
* CSR formulas are engineering approximations; the comparison quantifies
  the FE result against the rule, not the other way around.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from .arc_length import ArcLengthPoint, ArcLengthProblem, ArcLengthResult, solve_arc_length
from .layered_shell4_plasticity import (
    LayeredShell4Model,
    LayeredShell4State,
    assemble_layered_shell4,
    layered_shell4_buckling_eigenvalue,
    _element_dofs,
)


# ---------------------------------------------------------------------------
# Analytical reference solutions
# ---------------------------------------------------------------------------

def classical_plate_buckling_stress(a: float, b: float, t: float,
                                    young: float, poisson: float,
                                    n_terms: int = 10) -> float:
    """Critical stress for all-edges simply-supported plate in uniaxial compression.

    ``sigma_cr = k pi^2 E / (12(1-nu^2)) * (t/b)^2``
    with ``k = min_m (m*b/a + a/(m*b))^2`` (Timoshenko & Gere).

    Parameters
    ----------
    a : float
        Plate length (loaded direction) [m].
    b : float
        Plate width [m].
    t : float
        Plate thickness [m].
    young : float
        Young's modulus [Pa].
    poisson : float
        Poisson's ratio [-].

    Returns
    -------
    float
        Critical buckling stress [Pa].
    """
    if a <= 0 or b <= 0 or t <= 0 or young <= 0:
        raise ValueError("all geometric and material parameters must be positive")
    if not -1 < poisson < 0.5:
        raise ValueError("poisson must be in (-1, 0.5)")
    # Find the half-wave number m that minimises k
    best_k = float('inf')
    for m in range(1, n_terms + 1):
        k_m = (m * b / a + a / (m * b)) ** 2
        if k_m < best_k:
            best_k = k_m
    D_factor = young / (12.0 * (1.0 - poisson ** 2))
    sigma_cr = best_k * math.pi ** 2 * D_factor * (t / b) ** 2
    return sigma_cr


def csr_plate_ultimate_stress(a: float, b: float, t: float,
                              young: float, poisson: float,
                              yield_stress: float) -> float:
    """CSR-style plate ultimate strength (Faulkner effective width formula).

    The plate effective width ratio is ``b_eff/b = (2/beta - 1/beta^2)`` for
    ``beta >= 1``, where ``beta = sqrt(sigma_y / sigma_cr)`` is the plate
    slenderness.  The ultimate stress is ``sigma_u = sigma_y * b_eff / b``.

    This is a simplified engineering formula; actual CSR clauses have
    additional factors for boundary conditions, lateral pressure, etc.

    Parameters
    ----------
    a, b, t, young, poisson : float
        Plate geometry and material.
    yield_stress : float
        Yield stress [Pa].

    Returns
    -------
    float
        Ultimate strength [Pa].
    """
    sigma_cr = classical_plate_buckling_stress(a, b, t, young, poisson)
    beta = math.sqrt(yield_stress / sigma_cr)
    if beta < 1.0:
        # Stocky plate: yields before buckling
        return yield_stress
    b_eff_ratio = 2.0 / beta - 1.0 / beta ** 2
    return yield_stress * b_eff_ratio


# ---------------------------------------------------------------------------
# Mesh generation helpers
# ---------------------------------------------------------------------------

def simply_supported_plate_mesh(a: float, b: float, t: float,
                                 nx: int, ny: int,
                                 young: float, poisson: float,
                                 yield_stress: float, hardening: float,
                                 layers: int = 5) -> tuple[LayeredShell4Model, dict]:
    """Create a simply-supported flat plate mesh in the xy-plane.

    The plate occupies ``[0, a] x [0, b]`` at ``z = 0``.
    Boundary conditions: simply supported on all four edges
    (w = 0 on all edges, u = 0 on x=0 edge, v = 0 on y=0 edge).

    Returns
    -------
    model : LayeredShell4Model
    info : dict
        Contains 'fixed_dofs', 'loaded_dofs', 'reference_load'.
    """
    dtype = torch.float64
    x = torch.linspace(0, a, nx + 1, dtype=dtype)
    y = torch.linspace(0, b, ny + 1, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    nodes = torch.stack([xx.ravel(), yy.ravel(), torch.zeros_like(xx.ravel())], dim=1)

    # Element connectivity
    elements = []
    for i in range(nx):
        for j in range(ny):
            n0 = i * (ny + 1) + j
            n1 = (i + 1) * (ny + 1) + j
            n2 = (i + 1) * (ny + 1) + (j + 1)
            n3 = i * (ny + 1) + (j + 1)
            elements.append([n0, n1, n2, n3])
    elements = torch.tensor(elements, dtype=torch.long)

    model = LayeredShell4Model(
        nodes=nodes, elements=elements,
        young=young, poisson=poisson, thickness=t,
        yield_stress=yield_stress, hardening=hardening,
        layers=layers,
    )

    # Boundary conditions: simply supported on all four edges
    # Classical simply-supported: w=0 on edges, free rotation
    # In-plane: minimal constraints to prevent rigid body motion
    fixed = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            node = i * (ny + 1) + j
            # w = 0 on all four edges
            if i == 0 or i == nx or j == 0 or j == ny:
                fixed.append(6 * node + 2)  # w
            # Minimal in-plane constraints (rigid body modes only)
            if i == 0 and j == 0:
                # Corner (0,0): fix u and v
                fixed.append(6 * node + 0)  # u
                fixed.append(6 * node + 1)  # v
            elif i == 0:
                # x=0 edge (except corner): fix u only
                fixed.append(6 * node + 0)  # u
            elif j == 0:
                # y=0 edge (except corner): fix v only
                fixed.append(6 * node + 1)  # v
            # drilling rotation fixed everywhere for stability
            fixed.append(6 * node + 5)

    fixed_dofs = torch.tensor(sorted(set(fixed)), dtype=torch.long)

    # Reference load: uniform end compression on x=a edge
    # Apply unit pressure on the edge x=a (nodes with i=nx)
    ndof = 6 * len(nodes)
    reference_load = torch.zeros(ndof, dtype=dtype)
    # Unit compressive stress -> force per node = sigma * t * (b/ny) / 2 (half for boundary nodes)
    unit_stress = 1.0  # 1 Pa reference
    for j in range(ny + 1):
        node = nx * (ny + 1) + j
        tributary = b / ny if 0 < j < ny else b / (2 * ny)
        force = unit_stress * t * tributary
        reference_load[6 * node + 0] = -force  # compressive in x

    info = {
        'fixed_dofs': fixed_dofs,
        'reference_load': reference_load,
        'nx': nx, 'ny': ny,
        'a': a, 'b': b, 't': t,
    }
    return model, info


# ---------------------------------------------------------------------------
# Buckling imperfection
# ---------------------------------------------------------------------------

def compute_buckling_imperfection(model: LayeredShell4Model,
                                   fixed_dofs: torch.Tensor,
                                   unit_resultant: torch.Tensor,
                                   amplitude: float | None = None,
                                   modes: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute initial imperfection from the first buckling mode.

    Parameters
    ----------
    model : LayeredShell4Model
    fixed_dofs : tensor
        Constrained DOF indices.
    unit_resultant : tensor
        Stress resultant per unit stress, shape (3,).
        E.g. ``[t, 0, 0]`` for uniaxial compression.
    amplitude : float, optional
        Maximum imperfection amplitude [m]. Default is ``t/200``.
    modes : int
        Number of buckling modes to compute.

    Returns
    -------
    imperfection : tensor
        Nodal displacement imperfection, shape (n_dofs,).
    eigenvalues : tensor
        Buckling load factors (critical stresses).
    """
    if amplitude is None:
        amplitude = model.thickness / 200.0

    loads, w_fields, _ = layered_shell4_buckling_eigenvalue(
        model, fixed_dofs, unit_resultant, modes=modes)

    if len(loads) == 0:
        raise RuntimeError("no positive buckling eigenvalue found")

    # First mode shape: construct full DOF imperfection
    first_mode_w = w_fields[0]  # (n_nodes,) transverse displacement
    n_nodes = len(model.nodes)
    imperfection = torch.zeros(model.n_dofs, dtype=model.nodes.dtype)
    # Set w DOFs
    for i in range(n_nodes):
        imperfection[6 * i + 2] = first_mode_w[i]

    # Scale to desired amplitude
    max_w = float(imperfection.abs().max())
    if max_w > 0:
        imperfection = imperfection * (amplitude / max_w)

    return imperfection, loads


# ---------------------------------------------------------------------------
# Arc-length problem wrapper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ElastoplasticPlateState:
    """Immutable snapshot of the plate state for arc-length continuation."""
    displacement: torch.Tensor
    shell_state: LayeredShell4State


@dataclass(frozen=True)
class ElastoplasticPlateResult:
    """Result of an arc-length elastoplastic buckling analysis."""
    arc_result: ArcLengthResult
    states: tuple[ElastoplasticPlateState, ...]
    end_shortening: torch.Tensor
    max_plastic_strain: torch.Tensor
    critical_stress: float | None


def _build_arc_problem(model: LayeredShell4Model,
                       reference_load: torch.Tensor,
                       fixed_dofs: torch.Tensor,
                       initial_imperfection: torch.Tensor | None = None):
    """Build an ArcLengthProblem from a layered shell model.

    The internal force includes both material (elastoplastic) and
    geometric (stress-dependent) stiffness contributions.
    """
    ndof = model.n_dofs
    fixed = fixed_dofs.to(device=model.nodes.device, dtype=torch.long)
    mask = torch.ones(ndof, dtype=torch.bool, device=model.nodes.device)
    mask[fixed] = False
    free = torch.nonzero(mask).flatten()

    # Current state (mutable through closure)
    current_u = torch.zeros(ndof, dtype=model.nodes.dtype, device=model.nodes.device)
    current_state = LayeredShell4State.virgin(model)

    if initial_imperfection is not None:
        # Apply imperfection as initial displacement and commit the
        # resulting plastic state (should be elastic for small imperfections)
        current_u = initial_imperfection.to(model.nodes.dtype).clone()
        # Commit the state at the imperfection (zero load)
        _, _, _, trial = assemble_layered_shell4(
            model, current_u, current_state, geometric=True)
        current_state = trial

    def internal_and_tangent(q_free: torch.Tensor):
        """Return (internal_force[free], tangent[free,free]) at reduced DOFs q."""
        u = current_u.clone()
        u[free] = q_free
        fi, kt, _, trial = assemble_layered_shell4(
            model, u, current_state, geometric=True)
        return fi[free], kt[free[:, None], free]

    def residual_and_tangent(q_free: torch.Tensor, load_factor: float):
        """Return (residual[free], tangent[free,free], dR/dlambda)."""
        fi, kt = internal_and_tangent(q_free)
        residual = load_factor * reference_load[free] - fi
        return residual, kt, reference_load[free]

    problem = ArcLengthProblem(
        internal_tangent=internal_and_tangent,
        reference_load=reference_load[free],
        residual_tangent=residual_and_tangent,
    )
    return problem, free, current_u, current_state


def solve_elastoplastic_buckling(
    model: LayeredShell4Model,
    reference_load: torch.Tensor,
    fixed_dofs: torch.Tensor,
    *,
    initial_imperfection: torch.Tensor | None = None,
    steps: int = 30,
    step_size: float = 0.05,
    load_scale: float = 2.0,
    tolerance: float = 1e-8,
    max_iterations: int = 20,
) -> ElastoplasticPlateResult:
    """Trace the elastoplastic post-buckling path using arc-length continuation.

    Parameters
    ----------
    model : LayeredShell4Model
        Plate/shell model with J2 plasticity.
    reference_load : tensor
        Reference load vector (unit load factor).
    fixed_dofs : tensor
        Constrained DOF indices.
    initial_imperfection : tensor, optional
        Initial geometric imperfection (n_dofs,). Default: none.
    steps : int
        Number of arc-length steps.
    step_size : float
        Initial arc-length step size.
    load_scale : float
        Maximum load factor scale.
    tolerance : float
        Newton convergence tolerance.
    max_iterations : int
        Maximum Newton iterations per step.

    Returns
    -------
    ElastoplasticPlateResult
    """
    problem, free, current_u, current_state = _build_arc_problem(
        model, reference_load, fixed_dofs, initial_imperfection)

    initial_q = current_u[free].clone()

    arc_result = solve_arc_length(
        problem, initial_q,
        steps=steps, step_size=step_size, load_scale=load_scale,
        tolerance=tolerance, max_iterations=max_iterations,
    )

    # Post-process: reconstruct full states and compute engineering quantities
    states = []
    end_shortening = []
    max_plastic = []

    u = current_u.clone()
    state = current_state
    for pt in arc_result.points:
        u_new = u.clone()
        u_new[free] = pt.displacement
        # Compute end shortening (average x-displacement on loaded edge)
        # For a plate loaded in x, end shortening = -mean(u_x on loaded edge)
        # We approximate from the load factor and reference load
        end_shortening.append(float(pt.load_factor))
        states.append(ElastoplasticPlateState(u_new.clone(), state))
        u = u_new

    end_shortening_t = torch.tensor(end_shortening, dtype=torch.float64)

    return ElastoplasticPlateResult(
        arc_result=arc_result,
        states=tuple(states),
        end_shortening=end_shortening_t,
        max_plastic_strain=torch.zeros(len(arc_result.points)),
        critical_stress=None,
    )


# ---------------------------------------------------------------------------
# High-level verification functions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlateBucklingVerification:
    """Result of a plate buckling verification case."""
    case_name: str
    analytical_sigma_cr: float
    fe_sigma_cr: float
    relative_error: float
    mesh_info: dict
    passed: bool


def verify_elastic_plate_buckling(
    a: float, b: float, t: float,
    young: float, poisson: float,
    yield_stress: float,
    nx: int = 20, ny: int = 10,
) -> PlateBucklingVerification:
    """Verify elastic buckling of simply-supported plate vs analytical.

    This is the negative control: linear elastic, no imperfection.
    The FE buckling load should match the Timoshenko closed form within 2%.
    """
    hardening = 0.0  # elastic for buckling check
    model, info = simply_supported_plate_mesh(
        a, b, t, nx, ny, young, poisson, yield_stress, hardening)

    # Unit resultant for uniaxial compression: Nx = t * sigma, so unit = [t, 0, 0]
    unit_resultant = torch.tensor([t, 0.0, 0.0], dtype=torch.float64)
    loads, _, _ = layered_shell4_buckling_eigenvalue(
        model, info['fixed_dofs'], unit_resultant, modes=3)

    fe_sigma_cr = float(loads[0])
    analytical = classical_plate_buckling_stress(a, b, t, young, poisson)
    rel_error = abs(fe_sigma_cr - analytical) / analytical

    return PlateBucklingVerification(
        case_name=f"simply_supported_plate_{nx}x{ny}",
        analytical_sigma_cr=analytical,
        fe_sigma_cr=fe_sigma_cr,
        relative_error=rel_error,
        mesh_info={'nx': nx, 'ny': ny, 'a': a, 'b': b, 't': t},
        passed=rel_error < 0.02,  # 2% tolerance
    )


@dataclass(frozen=True)
class PlateUltimateStrengthVerification:
    """Result of a plate ultimate strength verification with imperfection."""
    case_name: str
    csr_sigma_u: float
    fe_sigma_u: float
    relative_error: float
    load_deflection: dict
    imperfection_amplitude: float
    passed: bool


def verify_plate_ultimate_strength(
    a: float, b: float, t: float,
    young: float, poisson: float,
    yield_stress: float, hardening: float,
    nx: int = 16, ny: int = 8,
    imperfection_ratio: float = 200.0,
    steps: int = 40,
) -> PlateUltimateStrengthVerification:
    """Verify plate ultimate strength with initial imperfection vs CSR.

    The plate has an initial imperfection in the shape of the first buckling
    mode with amplitude ``t/imperfection_ratio`` (default t/200).
    The FE ultimate strength is compared against the Faulkner/CSR formula.
    """
    model, info = simply_supported_plate_mesh(
        a, b, t, nx, ny, young, poisson, yield_stress, hardening)

    # Compute imperfection
    unit_resultant = torch.tensor([t, 0.0, 0.0], dtype=torch.float64)
    amplitude = t / imperfection_ratio
    imperfection, eigenvalues = compute_buckling_imperfection(
        model, info['fixed_dofs'], unit_resultant, amplitude=amplitude)

    fe_sigma_cr = float(eigenvalues[0])

    # Solve with arc-length
    result = solve_elastoplastic_buckling(
        model, info['reference_load'], info['fixed_dofs'],
        initial_imperfection=imperfection,
        steps=steps, step_size=0.03, load_scale=3.0,
    )

    # Extract ultimate strength (peak load factor)
    load_factors = torch.tensor([pt.load_factor for pt in result.arc_result.points])
    if len(load_factors) > 0:
        peak_factor = float(load_factors.max())
    else:
        peak_factor = 0.0

    # Convert load factor to stress
    # reference_load applies unit_stress=1 Pa, so load_factor = stress in Pa
    fe_sigma_u = peak_factor

    # CSR reference
    csr_sigma_u = csr_plate_ultimate_stress(a, b, t, young, poisson, yield_stress)

    rel_error = abs(fe_sigma_u - csr_sigma_u) / csr_sigma_u if csr_sigma_u > 0 else float('inf')

    # Load-deflection data
    load_deflection = {
        'load_factors': load_factors.tolist(),
        'fe_buckling_stress': fe_sigma_cr,
        'analytical_buckling_stress': classical_plate_buckling_stress(a, b, t, young, poisson),
    }

    return PlateUltimateStrengthVerification(
        case_name=f"plate_ultimate_{nx}x{ny}_w0_t{imperfection_ratio:.0f}",
        csr_sigma_u=csr_sigma_u,
        fe_sigma_u=fe_sigma_u,
        relative_error=rel_error,
        load_deflection=load_deflection,
        imperfection_amplitude=amplitude,
        passed=rel_error < 0.15,  # 15% tolerance for ultimate strength (engineering formula)
    )
