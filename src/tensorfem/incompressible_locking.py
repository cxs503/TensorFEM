"""Near-incompressible (``nu -> 1/2``) locking qualification of the HEX8 family.

The merged B-bar formulation (:mod:`tensorfem.solid3d_bbar`) was qualified by
:func:`tensorfem.solid3d_bbar_benchmarks.near_incompressible_cantilever`, which
scores it against Euler-Bernoulli beam deflection at ``nu = 0.4999``.  That is a
slender-beam formula for a problem whose compliance at ``nu -> 1/2`` is shear / 3-D
dominated, it exposes only one Poisson ratio, and it stores no deviatoric /
volumetric split, so it cannot separate "locking" from "coarse mesh".

This module removes that ambiguity by scoring the same elements against the
closed-form spherical Lame field, for which

* the **stress state is Poisson-ratio independent** (it depends on geometry and
  pressure only), and
* the **displacement state is Poisson-ratio dependent** in a known way,

so a volumetric over-stiffening (locking) bias appears directly as a
``nu``-dependent stress error.  A locking-free formulation keeps the stress error
flat in ``nu``; the standard displacement HEX8 does not.

Geometry, consistent pressure loads and the six tangential gauge constraints are
taken from :mod:`tensorfem.sphere_pressure_benchmark` (the qualified thick-sphere
case) so that no second, divergent load path is introduced.

Nothing here is a new solver: it is a verification layer.
"""
from __future__ import annotations

import time

import torch

from .sphere_pressure_benchmark import PROBLEM as SPHERE_PROBLEM
from .sphere_pressure_benchmark import _system as _sphere_system
from .solid3d import SolidModel, solve_solid
from .solid3d_bbar import solve_solid_bbar

FORMULATIONS = ("standard", "bbar")

#: Poisson ratios used by the qualification sweep.  ``0.4999`` is the value the
#: merged cantilever test quotes; ``0.3`` is the reference (lock-free) point.
DEFAULT_POISSONS = (0.3, 0.45, 0.49, 0.499, 0.4999)


def _problem(problem=None, poisson=None):
    base = dict(SPHERE_PROBLEM if problem is None else problem)
    if poisson is not None:
        base["poisson_ratio"] = float(poisson)
    return base


def lame_sphere_field(x: torch.Tensor, poisson: float, problem=None):
    """Exact 3-D Lame field of a thick sphere under internal pressure.

    Returns ``(u, sigma_r, sigma_theta)`` with ``u`` shaped ``[n, 3]``.  With
    ``a``/``b`` the inner/outer radius and ``p`` the internal pressure,

    ``A = p a^3 / (b^3 - a^3)``, ``B = p a^3 b^3 / (b^3 - a^3)``,

    ``sigma_r = A - B/r^3``, ``sigma_theta = A + B/(2 r^3)`` and
    ``u_r = ((1-2 nu) A r + (1+nu) B/(2 r^2)) / E``.
    """
    spec = _problem(problem, poisson)
    a = spec["inner_radius_m"]
    b = spec["outer_radius_m"]
    young = spec["young_modulus_pa"]
    p = spec["internal_pressure_pa"]
    if not -1.0 < poisson < 0.5:
        raise ValueError("poisson_ratio must lie in (-1, 0.5)")
    if not 0.0 < a < b:
        raise ValueError("inner_radius_m must be positive and smaller than outer_radius_m")
    r = torch.linalg.vector_norm(x, dim=1)
    direction = x / r[:, None]
    coefficient_a = p * a**3 / (b**3 - a**3)
    coefficient_b = p * a**3 * b**3 / (b**3 - a**3)
    radial = ((1.0 - 2.0 * poisson) * coefficient_a * r
              + (1.0 + poisson) * coefficient_b / (2.0 * r * r)) / young
    sigma_r = coefficient_a - coefficient_b / r**3
    sigma_theta = coefficient_a + coefficient_b / (2.0 * r**3)
    return radial[:, None] * direction, sigma_r, sigma_theta


def reference_self_check(poisson: float, *, problem=None, samples: int = 17):
    """Independently re-derive the reference: equilibrium, tractions, elasticity.

    Three residuals are returned (all relative, all expected at round-off):

    ``equilibrium``  ``d sigma_r/d r + 2 (sigma_r - sigma_theta)/r``
    ``traction``     ``sigma_r(a) = -p`` and ``sigma_r(b) = 0``
    ``constitutive`` ``epsilon_r = du/dr`` and ``epsilon_theta = u/r`` against
                     the isotropic law with the same ``E``, ``nu``
    """
    spec = _problem(problem, poisson)
    a, b = spec["inner_radius_m"], spec["outer_radius_m"]
    young, p = spec["young_modulus_pa"], spec["internal_pressure_pa"]
    radii = torch.linspace(a, b, samples, dtype=torch.float64)
    points = torch.stack((radii, torch.zeros_like(radii), torch.zeros_like(radii)), 1)
    displacement, sigma_r, sigma_theta = lame_sphere_field(points, poisson, spec)
    radial = displacement[:, 0]

    # (1) pointwise equilibrium of the analytic stress field, central differences
    h = 1e-6 * a
    hi = torch.stack((radii + h, torch.zeros_like(radii), torch.zeros_like(radii)), 1)
    lo = torch.stack((radii - h, torch.zeros_like(radii), torch.zeros_like(radii)), 1)
    _, sr_hi, st_hi = lame_sphere_field(hi, poisson, spec)
    _, sr_lo, st_lo = lame_sphere_field(lo, poisson, spec)
    derivative = (sr_hi - sr_lo) / (2.0 * h)
    equilibrium = derivative + 2.0 * (sigma_r - sigma_theta) / radii
    equilibrium_scale = float((sigma_r.abs() / radii).max())
    equilibrium_residual = float(equilibrium.abs().max()) / equilibrium_scale

    # (2) traction boundary conditions at both radii
    traction_residual = max(abs(float(sigma_r[0]) + p) / p, abs(float(sigma_r[-1])) / p)

    # (3) isotropic constitutive law, using analytic derivatives of u_r
    du = (((1.0 - 2.0 * poisson) * p * a**3 / (b**3 - a**3))
          - (1.0 + poisson) * p * a**3 * b**3 / (b**3 - a**3) / radii**3) / young
    hoop = radial / radii
    epsilon_r = (sigma_r - 2.0 * poisson * sigma_theta) / young
    epsilon_theta = (sigma_theta - poisson * (sigma_r + sigma_theta)) / young
    strain_scale = float(torch.maximum(epsilon_r.abs(), epsilon_theta.abs()).max())
    constitutive_residual = max(float((du - epsilon_r).abs().max()),
                                float((hoop - epsilon_theta).abs().max())) / strain_scale
    return {"equilibrium": equilibrium_residual,
            "traction": float(traction_residual),
            "constitutive": float(constitutive_residual),
            "poisson_ratio": float(poisson),
            "stress_state_poisson_invariant": _stress_invariance(spec)}


def _stress_invariance(problem):
    """Largest relative difference of the analytic stress field over ``nu``."""
    spec = _problem(problem)
    points = torch.stack((torch.linspace(spec["inner_radius_m"], spec["outer_radius_m"], 9),
                          torch.zeros(9, dtype=torch.float64),
                          torch.zeros(9, dtype=torch.float64)), 1)
    reference = None
    worst = 0.0
    for poisson in (0.0, 0.3, 0.4999):
        _, sigma_r, sigma_theta = lame_sphere_field(points, poisson, spec)
        if reference is None:
            reference = (sigma_r, sigma_theta)
            continue
        worst = max(worst,
                    float((sigma_r - reference[0]).abs().max()) / float(reference[0].abs().max()),
                    float((sigma_theta - reference[1]).abs().max()) / float(reference[1].abs().max()))
    return worst


def _relative(weighted_error: torch.Tensor, weighted_reference: torch.Tensor) -> float:
    return float(torch.sqrt(weighted_error.sum() / weighted_reference.sum()))


def _score(nodes, elements, volume, displacement, stress, poisson, problem):
    """Score a HEX8 solution against the exact field (element centres + nodes)."""
    spec = _problem(problem, poisson)
    exact_u, _, _ = lame_sphere_field(nodes, poisson, spec)
    centres = nodes[elements].mean(1)
    _, ref_sigma_r, ref_sigma_theta = lame_sphere_field(centres, poisson, spec)
    normal = centres / torch.linalg.vector_norm(centres, dim=1)[:, None]

    tensor = torch.zeros((len(centres), 3, 3), dtype=nodes.dtype, device=nodes.device)
    tensor[:, 0, 0], tensor[:, 1, 1], tensor[:, 2, 2] = stress[:, 0], stress[:, 1], stress[:, 2]
    tensor[:, 0, 1] = tensor[:, 1, 0] = stress[:, 3]
    tensor[:, 1, 2] = tensor[:, 2, 1] = stress[:, 4]
    tensor[:, 0, 2] = tensor[:, 2, 0] = stress[:, 5]
    sigma_r = torch.einsum("ei,eij,ej->e", normal, tensor, normal)
    mean = tensor.diagonal(dim1=1, dim2=2).sum(1) / 3.0
    reference_mean = (ref_sigma_r + 2.0 * ref_sigma_theta) / 3.0
    deviatoric = tensor - mean[:, None, None] * torch.eye(3, dtype=nodes.dtype)
    ref_deviatoric = (torch.zeros_like(tensor)
                      + (ref_sigma_r - reference_mean)[:, None, None] * normal[:, :, None] * normal[:, None, :]
                      + (ref_sigma_theta - reference_mean)[:, None, None]
                      * (torch.eye(3, dtype=nodes.dtype) - normal[:, :, None] * normal[:, None, :]))

    weights = volume
    errors = {
        "displacement_l2": _relative(((displacement - exact_u) ** 2).sum(1),
                                     (exact_u ** 2).sum(1)),
        "displacement_max_relative": float((torch.linalg.vector_norm(displacement - exact_u, dim=1)
                                            / torch.linalg.vector_norm(exact_u, dim=1)).max()),
        "radial_stress_l2": _relative(((sigma_r - ref_sigma_r) ** 2) * weights,
                                      (ref_sigma_r ** 2) * weights),
        "radial_stress_max_relative": float(((sigma_r - ref_sigma_r).abs() / ref_sigma_r.abs()).max()),
        "mean_stress_l2": _relative(((mean - reference_mean) ** 2) * weights,
                                    (reference_mean ** 2) * weights),
        "deviatoric_stress_l2": _relative(((deviatoric - ref_deviatoric) ** 2).sum((1, 2)) * weights,
                                          (ref_deviatoric ** 2).sum((1, 2)) * weights),
    }
    inner = torch.abs(torch.linalg.vector_norm(nodes, dim=1) - spec["inner_radius_m"]) < 1e-12
    radial_fe = torch.einsum("ij,ij->i", displacement, nodes
                             / torch.linalg.vector_norm(nodes, dim=1)[:, None])[inner].mean()
    radial_exact = float(torch.linalg.vector_norm(exact_u[inner], dim=1).mean())
    errors["inner_radius_displacement_relative"] = abs(float(radial_fe) - radial_exact) / abs(radial_exact)
    return errors


def solve_case(poisson: float, *, face_divisions: int = 6, radial_layers: int = 3,
               formulation: str = "bbar", problem=None):
    """Solve the thick-sphere case with one HEX8 formulation and score it."""
    if formulation not in FORMULATIONS:
        raise ValueError(f"formulation must be one of {FORMULATIONS}")
    spec = _problem(problem, poisson)
    started = time.perf_counter()
    nodes, elements, _k, force, fixed, _b, _d, _edofs, volume = _sphere_system(
        face_divisions, radial_layers, spec)
    fixed_dofs = torch.tensor(sorted(int(k) for k in fixed), dtype=torch.long)
    model = SolidModel(nodes, elements, nodes.new_tensor(spec["young_modulus_pa"]),
                       nodes.new_tensor(float(poisson)), force, fixed_dofs, "hex8")
    result = (solve_solid_bbar if formulation == "bbar" else solve_solid)(model)
    displacement = result.displacement.reshape(-1, 3)
    errors = _score(nodes, elements, volume, displacement, result.stress, poisson, spec)
    record = {
        "poisson_ratio": float(poisson),
        "formulation": formulation,
        "face_divisions": int(face_divisions),
        "radial_layers": int(radial_layers),
        "elements": int(len(elements)),
        "dofs": int(3 * len(nodes)),
        "errors": errors,
        "solve_seconds": time.perf_counter() - started,
        "displacement": displacement,
        "stress": result.stress,
        "node_coordinates": nodes,
        "element_connectivity": elements,
        "element_volume": volume,
    }
    return record


def locking_sweep(poissons=DEFAULT_POISSONS, *, face_divisions: int = 6, radial_layers: int = 3,
                  formulations=FORMULATIONS, problem=None):
    """Error-vs-Poisson tables for the requested formulations (3-D thick sphere)."""
    records = []
    for poisson in poissons:
        for formulation in formulations:
            records.append(solve_case(poisson, face_divisions=face_divisions,
                                      radial_layers=radial_layers,
                                      formulation=formulation, problem=problem))
    return records


PLANE_STRAIN_FORMULATIONS = ("standard", "bbar")


def plane_strain_case(poisson: float, *, formulation: str = "standard", n_theta: int = 16,
                      n_radial: int = 8, inner_radius: float = 0.5, outer_radius: float = 0.8,
                      pressure: float = 100.0e6, young_modulus: float = 210.0e9):
    """Score one 2-D Q4 plane-strain formulation against the plane-strain Lame field.

    ``formulation="standard"`` is the merged fully integrated Q4 path of
    :mod:`tensorfem.continuum`; ``"bbar"`` is the mean-dilatation counterpart of
    :mod:`tensorfem.continuum_bbar`.  Stress is recovered at element centres and
    compared with the exact field at the same centres, exactly as in the 3-D case,
    so the two families are scored by one metric.
    """
    if formulation not in PLANE_STRAIN_FORMULATIONS:
        raise ValueError(f"formulation must be one of {PLANE_STRAIN_FORMULATIONS}")
    from .continuum import ContinuumModel, solve_continuum
    from .lame_reference import (cartesian_to_polar_stress, circular_pressure_loads,
                                 lame_state, radial_component, sector_q4_mesh,
                                 symmetry_fixed_dofs)

    nodes, cells = sector_q4_mesh(inner_radius, outer_radius, n_theta, n_radial)
    forces = circular_pressure_loads(n_theta, n_radial, inner_radius, pressure, inner=True)
    fixed = symmetry_fixed_dofs(nodes, n_theta=n_theta, n_radial=n_radial)
    model = ContinuumModel(nodes, cells, torch.tensor(young_modulus),
                           torch.tensor(float(poisson)), torch.tensor(1.0),
                           forces, fixed, plane="strain")
    if formulation == "bbar":
        from .continuum_bbar import solve_continuum_bbar
        result = solve_continuum_bbar(model)
    else:
        result = solve_continuum(model)
    displacement = result.displacement.reshape(-1, 2)
    radius = torch.linalg.vector_norm(nodes, dim=1)
    state = lame_state(radius, inner_radius=inner_radius, outer_radius=outer_radius,
                       inner_pressure=pressure, young_modulus=young_modulus,
                       poisson_ratio=float(poisson), plane="strain")
    exact_radial = state.radial_displacement
    fe_radial = radial_component(displacement, nodes)
    errors = {"displacement_l2": _relative((fe_radial - exact_radial) ** 2, exact_radial ** 2)}
    centres = nodes[cells].mean(1)
    sigma_r, sigma_theta, _ = cartesian_to_polar_stress(result.stress, centres)
    exact_centres = lame_state(torch.linalg.vector_norm(centres, dim=1),
                               inner_radius=inner_radius, outer_radius=outer_radius,
                               inner_pressure=pressure, young_modulus=young_modulus,
                               poisson_ratio=float(poisson), plane="strain")
    errors["radial_stress_l2"] = _relative((sigma_r - exact_centres.radial_stress) ** 2,
                                           exact_centres.radial_stress ** 2)
    errors["hoop_stress_l2"] = _relative((sigma_theta - exact_centres.hoop_stress) ** 2,
                                         exact_centres.hoop_stress ** 2)
    # Plane strain: sigma_z = nu (sigma_r + sigma_theta), so the mean stress is a
    # pure volumetric indicator.  Locking shows up here first and hardest.
    mean_fe = (1.0 + float(poisson)) * (sigma_r + sigma_theta) / 3.0
    mean_exact = (1.0 + float(poisson)) * (exact_centres.radial_stress
                                           + exact_centres.hoop_stress) / 3.0
    errors["mean_stress_l2"] = _relative((mean_fe - mean_exact) ** 2, mean_exact ** 2)
    inner = torch.abs(radius - inner_radius) < 1e-12
    errors["inner_radius_displacement_relative"] = abs(
        float(fe_radial[inner].mean()) - float(exact_radial[inner].mean())) / abs(
            float(exact_radial[inner].mean()))
    return {"poisson_ratio": float(poisson),
            "formulation": "q4 plane strain" if formulation == "standard" else "q4 plane strain bbar",
            "n_theta": int(n_theta), "n_radial": int(n_radial),
            "dofs": int(2 * len(nodes)), "errors": errors}


def plane_strain_sweep(poissons=DEFAULT_POISSONS, *, n_theta: int = 16, n_radial: int = 8,
                       inner_radius: float = 0.5, outer_radius: float = 0.8,
                       pressure: float = 100.0e6, young_modulus: float = 210.0e9,
                       formulations=("standard",)):
    """Error-vs-Poisson tables for the 2-D Q4 plane-strain path.

    The merged element set ships only the fully integrated Q4, which locks; the
    default sweep still measures that debt.  Pass ``formulations=("standard",
    "bbar")`` to add the mean-dilatation element of
    :mod:`tensorfem.continuum_bbar`, which is the cure for the same problem.
    """
    for formulation in formulations:
        if formulation not in PLANE_STRAIN_FORMULATIONS:
            raise ValueError(f"formulation must be one of {PLANE_STRAIN_FORMULATIONS}")
    records = []
    for poisson in poissons:
        for formulation in formulations:
            records.append(plane_strain_case(
                poisson, formulation=formulation, n_theta=n_theta, n_radial=n_radial,
                inner_radius=inner_radius, outer_radius=outer_radius,
                pressure=pressure, young_modulus=young_modulus))
    return records
