"""Verification layer that closes the 2-D near-incompressible locking debt.

The 3-D qualification (:mod:`tensorfem.incompressible_locking`) ended with an
explicit open item: the merged 2-D path had **no** B-bar / EAS counterpart, so
plane strain at ``nu >= 0.499`` was quantified but not curable.  With
:mod:`tensorfem.continuum_bbar` in place this module scores both formulations
against the same closed-form oracle and answers three separate questions:

1. *Is the oracle right?* - :func:`reference_self_check` re-derives the
   plane-strain Lame field from equilibrium, the two traction conditions and the
   isotropic law, and verifies that the stress state is Poisson-ratio
   independent (only the displacement field depends on ``nu``).
2. *Is B-bar locking-free?* - :func:`plane_locking_sweep` reports the error at
   ``nu = 0.3 ... 0.4999`` for both formulations on one fixed mesh.  A
   locking-free formulation keeps the **stress** error flat in ``nu``; the
   standard Q4 does not.
3. *Is B-bar convergent?* - :func:`plane_refinement_study` gives the observed
   order at ``nu = 0.4999``, which is what separates "coarse mesh" from
   "over-stiff element".

Element-level prerequisites are checked by :func:`patch_test` (a constant strain
field must produce identical internal forces for both formulations: for constant
strain ``Bv_bar == Bv``) and :func:`rigid_body_check` (rigid translation and
rotation must produce no internal force and no strain).

Geometry, mesh, consistent bore-pressure loads and symmetry constraints are taken
from :mod:`tensorfem.lame_reference`, the qualified thick-cylinder helper, so no
second load path is introduced.
"""
from __future__ import annotations

import math
import time

import torch

from .continuum import ContinuumModel, solve_continuum
from .continuum_bbar import q4_bbar_stiffness, solve_continuum_bbar, volumetric_constraint_rank
from .lame_reference import (cartesian_to_polar_stress, circular_pressure_loads, lame_state,
                             radial_component, sector_q4_mesh, symmetry_fixed_dofs)

FORMULATIONS = ("standard", "bbar")

#: Poisson ratios of the qualification sweep; ``0.4999`` is the value the merged
#: 3-D report quotes, ``0.3`` is the lock-free reference point.
DEFAULT_POISSONS = (0.3, 0.45, 0.49, 0.499, 0.4999)

#: Default geometry: thick cylinder, ``a/b = 0.625``, internal pressure only.
GEOMETRY = dict(inner_radius=0.5, outer_radius=0.8, inner_pressure=100.0e6,
                young_modulus=210.0e9)


#: Cached ``(nodes, elements) -> {formulation: rank}``; the global rank is a mesh
#: property but it costs an SVD, so it is computed once per mesh.
_CONSTRAINT_CACHE: dict = {}

#: Above this many degrees of freedom the 2x2-Gauss constraint rank is skipped:
#: its SVD is ``O(n_row * n_dof**2)`` and buys nothing that the meshes below do not
#: already show (documented as a limitation of the evidence, not of the element).
RANK_DOF_LIMIT = 1500


def constraint_counts(nodes: torch.Tensor, elements: torch.Tensor, *, skip_standard: bool = False) -> dict:
    """Independent volumetric constraints of both formulations, cached per mesh."""
    key = (len(nodes), len(elements), float(nodes.sum()), bool(skip_standard))
    if key not in _CONSTRAINT_CACHE:
        _CONSTRAINT_CACHE[key] = {
            "standard": (None if skip_standard
                         else volumetric_constraint_rank(nodes, elements, formulation="standard")),
            "bbar": volumetric_constraint_rank(nodes, elements, formulation="bbar"),
        }
    return dict(_CONSTRAINT_CACHE[key])


def _relative(weighted_error: torch.Tensor, weighted_reference: torch.Tensor) -> float:
    return float(torch.sqrt(weighted_error.sum() / weighted_reference.sum()))


def reference_self_check(poisson: float, *, samples: int = 33, **geometry) -> dict:
    """Independent residuals of the plane-strain Lame oracle (all expected tiny).

    ``equilibrium``   ``d sigma_r/d r + (sigma_r - sigma_theta)/r``
    ``traction``      ``sigma_r(a) = -p`` and ``sigma_r(b) = 0``
    ``constitutive``  ``eps_r = du/dr``, ``eps_theta = u/r`` fed through the
                      plane-strain isotropic law must return ``sigma_r``,
                      ``sigma_theta`` (and ``eps_z = 0``)
    ``stress_poisson_invariant``  the stress state must not move when ``nu``
                      changes, because it follows from tractions alone
    """
    spec = {**GEOMETRY, **geometry}
    a, b, p = spec["inner_radius"], spec["outer_radius"], spec["inner_pressure"]
    young = spec["young_modulus"]
    radius = torch.linspace(a, b, samples, dtype=torch.float64)
    state = lame_state(radius, inner_radius=a, outer_radius=b, inner_pressure=p,
                       young_modulus=young, poisson_ratio=poisson, plane="strain")
    sigma_r, sigma_theta = state.radial_stress, state.hoop_stress

    # (1) equilibrium of the analytic stress field, central differences
    h = 1.0e-7 * a
    hi = lame_state(radius + h, inner_radius=a, outer_radius=b, inner_pressure=p,
                    young_modulus=young, poisson_ratio=poisson, plane="strain").radial_stress
    lo = lame_state(radius - h, inner_radius=a, outer_radius=b, inner_pressure=p,
                    young_modulus=young, poisson_ratio=poisson, plane="strain").radial_stress
    residual = (hi - lo) / (2.0 * h) + (sigma_r - sigma_theta) / radius
    equilibrium = float(residual.abs().max()) / float((sigma_r.abs() / radius).max())

    # (2) tractions at both faces
    traction = max(abs(float(sigma_r[0]) + p) / p, abs(float(sigma_r[-1])) / p)

    # (3) plane-strain isotropic law from the analytic displacement
    h2 = 1.0e-7 * a
    def displacement(r):
        return lame_state(r, inner_radius=a, outer_radius=b, inner_pressure=p,
                          young_modulus=young, poisson_ratio=poisson, plane="strain").radial_displacement
    dudr = (displacement(radius + h2) - displacement(radius - h2)) / (2.0 * h2)
    epsilon_r = dudr
    epsilon_theta = state.radial_displacement / radius
    factor = young / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    sigma_r_law = factor * ((1.0 - poisson) * epsilon_r + poisson * epsilon_theta)
    sigma_theta_law = factor * (poisson * epsilon_r + (1.0 - poisson) * epsilon_theta)
    # sigma_r passes through zero at the free outer face, so normalise by the peak
    stress_scale = float(sigma_theta.abs().max())
    constitutive = max(float((sigma_r_law - sigma_r).abs().max()),
                       float((sigma_theta_law - sigma_theta).abs().max())) / stress_scale
    # plane strain requires eps_z = 0, i.e. sigma_z = nu (sigma_r + sigma_theta)
    axial = float((state.axial_stress - poisson * (sigma_r + sigma_theta)).abs().max()) / stress_scale

    other = lame_state(radius, inner_radius=a, outer_radius=b, inner_pressure=p,
                       young_modulus=young, poisson_ratio=min(poisson + 0.1, 0.49), plane="strain")
    invariant = max(float((other.radial_stress - sigma_r).abs().max()),
                    float((other.hoop_stress - sigma_theta).abs().max())) / stress_scale
    return {"equilibrium": equilibrium, "traction": traction, "constitutive": constitutive,
            "axial_stress_consistency": axial, "stress_poisson_invariant": invariant}


def _score(nodes, elements, displacement, stress, poisson, spec):
    radius = torch.linalg.vector_norm(nodes, dim=1)
    exact = lame_state(radius, inner_radius=spec["inner_radius"], outer_radius=spec["outer_radius"],
                       inner_pressure=spec["inner_pressure"], young_modulus=spec["young_modulus"],
                       poisson_ratio=poisson, plane="strain")
    fe_radial = radial_component(displacement, nodes)
    centres = nodes[elements].mean(1)
    sigma_r, sigma_theta, _ = cartesian_to_polar_stress(stress, centres)
    exact_centres = lame_state(torch.linalg.vector_norm(centres, dim=1),
                               inner_radius=spec["inner_radius"], outer_radius=spec["outer_radius"],
                               inner_pressure=spec["inner_pressure"], young_modulus=spec["young_modulus"],
                               poisson_ratio=poisson, plane="strain")
    errors = {
        "displacement_l2": _relative((fe_radial - exact.radial_displacement) ** 2,
                                     exact.radial_displacement ** 2),
        "radial_stress_l2": _relative((sigma_r - exact_centres.radial_stress) ** 2,
                                      exact_centres.radial_stress ** 2),
        "hoop_stress_l2": _relative((sigma_theta - exact_centres.hoop_stress) ** 2,
                                    exact_centres.hoop_stress ** 2),
    }
    bore = torch.abs(radius - spec["inner_radius"]) < 1.0e-12
    exact_bore = float(exact.radial_displacement[bore].mean())
    errors["bore_displacement_relative"] = abs(float(fe_radial[bore].mean()) - exact_bore) / abs(exact_bore)
    return errors


def solve_case(poisson: float, *, n_theta: int = 16, n_radial: int = 8,
               formulation: str = "bbar", **geometry):
    """Solve the pressurised thick cylinder in plane strain and score it."""
    if formulation not in FORMULATIONS:
        raise ValueError(f"formulation must be one of {FORMULATIONS}")
    spec = {**GEOMETRY, **geometry}
    a, b, p = spec["inner_radius"], spec["outer_radius"], spec["inner_pressure"]
    started = time.perf_counter()
    nodes, cells = sector_q4_mesh(a, b, n_theta, n_radial)
    forces = circular_pressure_loads(n_theta, n_radial, a, p, inner=True)
    fixed = symmetry_fixed_dofs(nodes, n_theta=n_theta, n_radial=n_radial)
    model = ContinuumModel(nodes, cells, torch.tensor(spec["young_modulus"], dtype=torch.float64),
                           torch.tensor(float(poisson), dtype=torch.float64),
                           torch.tensor(1.0, dtype=torch.float64), forces, fixed, plane="strain")
    solver = solve_continuum_bbar if formulation == "bbar" else solve_continuum
    result = solver(model)
    displacement = result.displacement.reshape(-1, 2)
    errors = _score(nodes, cells, displacement, result.stress, poisson, spec)
    return {
        "poisson_ratio": float(poisson),
        "formulation": formulation,
        "n_theta": int(n_theta),
        "n_radial": int(n_radial),
        "elements": int(len(cells)),
        "dofs": int(model.n_dofs),
        **{"volumetric_constraints": constraint_counts(
            nodes, cells, skip_standard=model.n_dofs > RANK_DOF_LIMIT)},
        "errors": errors,
        "solve_seconds": time.perf_counter() - started,
        "displacement": displacement,
        "stress": result.stress,
        "node_coordinates": nodes,
        "element_connectivity": cells,
    }


def plane_locking_sweep(poissons=DEFAULT_POISSONS, *, n_theta: int = 16, n_radial: int = 8,
                        formulations: tuple = FORMULATIONS, **geometry):
    """Error-vs-Poisson table for the 2-D plane-strain thick cylinder."""
    return [solve_case(nu, n_theta=n_theta, n_radial=n_radial, formulation=formulation, **geometry)
            for nu in poissons for formulation in formulations]


def plane_refinement_study(poisson: float = 0.4999, meshes=((8, 4), (16, 8), (24, 12), (32, 16), (48, 24)),
                           formulations: tuple = FORMULATIONS, **geometry):
    """Mesh-refinement record; ``h ~ 1/n_theta`` because the radial grading is uniform."""
    return [solve_case(poisson, n_theta=n_theta, n_radial=n_radial, formulation=formulation, **geometry)
            for n_theta, n_radial in meshes for formulation in formulations]


def observed_order(coarse: float, fine: float, ratio: float) -> float:
    """Classical log-log slope between two meshes with ``h`` ratio ``ratio``."""
    if coarse <= 0.0 or fine <= 0.0 or ratio <= 1.0:
        return float("nan")
    return math.log(coarse / fine) / math.log(ratio)


def _distorted_quad(distortion: float = 0.12) -> torch.Tensor:
    """One deliberately distorted counter-clockwise Q4 (``[1, 4, 2]``)."""
    return torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0 + distortion, 1.0],
                          [distortion, 1.0]]], dtype=torch.float64)


def _strain_dmat(young: float = 210.0e9, poisson: float = 0.4999) -> torch.Tensor:
    from .continuum import elasticity_matrix
    return elasticity_matrix(torch.tensor([young], dtype=torch.float64),
                             torch.tensor([poisson], dtype=torch.float64), "strain")


def patch_test(distortion: float = 0.12) -> dict:
    """Constant-strain consistency of the mean-dilatation element.

    Two checks on one deliberately distorted quadrilateral: (a) for a linear
    displacement field (constant strain) the B-bar operator equals the standard
    operator, so both elements must produce the same internal force vector to
    round-off; (b) the strain energy must equal ``0.5 * area * eps : D : eps``
    with the exact element area.  A formulation that fails (a) is not patch
    consistent and its refinement study would be meaningless.
    """
    from .continuum import q4_stiffness
    coordinates = _distorted_quad(distortion)
    dmat = _strain_dmat()
    thickness = torch.ones(1, dtype=torch.float64)
    stiffness_bbar, _ = q4_bbar_stiffness(coordinates, dmat, thickness)
    stiffness_standard, _ = q4_stiffness(coordinates, dmat, thickness)
    # linear (hence constant strain) displacement: u = A x, A = [[1e-3, 4e-4], [-3e-4, 8e-4]]
    gradient = torch.tensor([[1.0e-3, 4.0e-4], [-3.0e-4, 8.0e-4]], dtype=torch.float64)
    displacement = torch.einsum("ni,ij->nj", coordinates[0], gradient).reshape(-1)
    force_bbar = stiffness_bbar[0] @ displacement
    force_standard = stiffness_standard[0] @ displacement
    scale = float(force_standard.abs().max())
    engineering = torch.stack((gradient[0, 0], gradient[1, 1],
                               gradient[0, 1] + gradient[1, 0]))
    stress = dmat[0] @ engineering
    edge_a = coordinates[0][1] - coordinates[0][0]
    edge_b = coordinates[0][3] - coordinates[0][0]
    area = float(torch.abs(torch.linalg.det(torch.stack((edge_a, edge_b)))))
    exact_energy = 0.5 * area * float(engineering @ stress)
    energy_bbar = 0.5 * float(displacement @ force_bbar)
    return {"force_agreement_relative": float((force_bbar - force_standard).abs().max()) / scale,
            "energy_relative": abs(energy_bbar - exact_energy) / exact_energy,
            "element_area": area}


def rigid_body_check(distortion: float = 0.12) -> dict:
    """Translation and rotation must carry no internal force (objectivity).

    A single B-bar element has 8 degrees of freedom and must keep exactly three
    zero-energy modes (two translations, one rotation): the volumetric averaging
    must not delete the deviatoric stiffness, i.e. no hourglass mode.
    """
    coordinates = _distorted_quad(distortion)
    stiffness, _ = q4_bbar_stiffness(coordinates, _strain_dmat(), torch.ones(1, dtype=torch.float64))
    force_scale = float(torch.linalg.matrix_norm(stiffness[0], ord=2))
    translation = torch.tensor([0.7, -1.3], dtype=torch.float64).expand(4, 2).reshape(-1)
    omega = 0.35
    # infinitesimal rotation: u = Omega x with Omega antisymmetric, so du/dx is
    # purely antisymmetric and the small-strain tensor vanishes exactly.  (A finite
    # rotation matrix R would give u = (R - I) x whose symmetric gradient is
    # (cos w - 1) I -- a small-strain artifact, not a null mode.)
    omega_matrix = torch.tensor([[0.0, -omega], [omega, 0.0]], dtype=torch.float64)
    rotation = omega_matrix @ coordinates[0].T
    return {"translation_relative": float(torch.linalg.vector_norm(stiffness[0] @ translation)) / force_scale,
            "rotation_relative": float(torch.linalg.vector_norm(stiffness[0] @ rotation.T.reshape(-1))) / force_scale,
            "zero_modes": int(8 - torch.linalg.matrix_rank(stiffness[0]))}
