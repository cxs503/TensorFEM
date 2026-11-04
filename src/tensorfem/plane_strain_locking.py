"""Plane-strain Lame qualification of the mean-dilatation (B-bar) Q4 element.

The question this module answers, with numbers instead of prose: on the exact
plane-strain Lame field, does the B-bar Q4 element (:mod:`tensorfem.plane_strain_bbar`)
actually remove the volumetric locking that
``docs/near-incompressible-locking.md`` recorded as an unmitigated debt
(radial-stress L2 error 155.2 % at ``nu = 0.4999``, growing without bound)?

Design of the study
-------------------
1. **Reference first.** :func:`lame_plane_strain_self_check` re-derives the
   closed-form field and scores it against three independent residuals:
   radial equilibrium, the two surface tractions, and the plane-strain
   constitutive law.  A benchmark that has not been checked against its own
   governing equations cannot convict a solver.
2. **Same mesh, same load, two formulations.** Only the element changes
   (``solve_continuum`` vs ``solve_continuum_bbar``); the annular-sector mesh,
   the consistent nodal pressure loads and the symmetry constraints all come
   from :mod:`tensorfem.lame_reference`, so no discrepancy can be blamed on the
   discretisation setup.
3. **Locking is a Poisson-ratio signature, not a single number.** The in-plane
   stress field of the Lame problem is *independent of the elastic constants*
   (``A`` and ``B`` follow from the tractions alone), so any variation of the
   stress error with ``nu`` is formulation error.  :func:`locking_sweep`
   therefore reports error-versus-``nu`` for both elements, and
   :func:`refinement_study` shows whether what is left converges with ``h``.

All arrays are ``float64``; nothing here is a stochastic or hard-coded value —
``benchmarks/plane_strain_bbar_locking.py`` writes the scalars it produces to
``benchmarks/plane_strain_bbar_locking_evidence.json``.
"""
from __future__ import annotations

import math

import torch

from .continuum import ContinuumModel, solve_continuum
from .lame_reference import (cartesian_to_polar_stress, circular_pressure_loads, lame_coefficients,
                             lame_state, radial_component, sector_q4_mesh, symmetry_fixed_dofs)
from .plane_strain_bbar import solve_continuum_bbar

#: Poisson ratios spanning the compressible range through the locking onset.
DEFAULT_POISSONS = (0.30, 0.45, 0.49, 0.499, 0.4999)
FORMULATIONS = ("standard", "bbar")


def _lame_plane_strain(radius: torch.Tensor, *, inner_radius: float, outer_radius: float,
                       inner_pressure: float, young_modulus: float, poisson_ratio: float):
    """Closed-form plane-strain Lame field (stresses do not depend on ``E`` or ``nu``)."""
    return lame_state(radius, inner_radius=inner_radius, outer_radius=outer_radius,
                      inner_pressure=inner_pressure, young_modulus=young_modulus,
                      poisson_ratio=poisson_ratio, plane="strain")


def lame_plane_strain_self_check(poisson: float = 0.3, *, inner_radius: float = 0.5,
                                 outer_radius: float = 0.8, inner_pressure: float = 100.0e6,
                                 young_modulus: float = 210.0e9, samples: int = 201) -> dict:
    """Independently re-verify the plane-strain Lame reference.

    Returns relative residuals for

    * ``equilibrium`` -- ``d sigma_r/d r + (sigma_r - sigma_theta)/r``;
    * ``traction`` -- ``sigma_r(a) + p`` and ``sigma_r(b)``;
    * ``constitutive`` -- ``sigma_r, sigma_theta`` rebuilt from ``d u/d r`` and
      ``u/r`` with the plane-strain Lame constants;
    * ``axial`` -- the plane-strain condition ``eps_z = 0`` implies
      ``sigma_z = nu (sigma_r + sigma_theta)``;
    * ``poisson_invariance`` -- the in-plane stress state must be identical at a
      different ``nu`` (this is the property the study exploits).
    """
    radius = torch.linspace(inner_radius, outer_radius, samples, dtype=torch.float64)
    radius = radius.clone().requires_grad_(True)
    state = _lame_plane_strain(radius, inner_radius=inner_radius, outer_radius=outer_radius,
                               inner_pressure=inner_pressure, young_modulus=young_modulus,
                               poisson_ratio=poisson)
    d_sigma_r = torch.autograd.grad(state.radial_stress.sum(), radius, create_graph=False)[0]
    d_u = torch.autograd.grad(state.radial_displacement.sum(), radius, create_graph=False)[0]
    with torch.no_grad():
        scale = float(state.radial_stress.abs().max()) + float(inner_pressure)
        equilibrium = (d_sigma_r + (state.radial_stress - state.hoop_stress) / radius).abs().max() / scale

        def boundary(r_value: float, target: float) -> torch.Tensor:
            radius_value = torch.tensor(r_value, dtype=torch.float64)
            field = _lame_plane_strain(radius_value, inner_radius=inner_radius,
                                       outer_radius=outer_radius, inner_pressure=inner_pressure,
                                       young_modulus=young_modulus, poisson_ratio=poisson)
            return (field.radial_stress - target).abs()

        traction = torch.maximum(boundary(inner_radius, -inner_pressure), boundary(outer_radius, 0.0)) / scale

        sheared = young_modulus / (2.0 * (1.0 + poisson))
        lame_lambda = young_modulus * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        eps_r = d_u
        eps_theta = state.radial_displacement / radius
        dilation = eps_r + eps_theta
        sigma_r_check = lame_lambda * dilation + 2.0 * sheared * eps_r
        sigma_theta_check = lame_lambda * dilation + 2.0 * sheared * eps_theta
        constitutive = torch.maximum((sigma_r_check - state.radial_stress).abs().max(),
                                     (sigma_theta_check - state.hoop_stress).abs().max()) / scale
        eps_z = (state.axial_stress - poisson * (state.radial_stress + state.hoop_stress)) / young_modulus
        axial = eps_z.abs().max() / (float(inner_pressure) / young_modulus)

        other = _lame_plane_strain(radius.detach(), inner_radius=inner_radius,
                                   outer_radius=outer_radius, inner_pressure=inner_pressure,
                                   young_modulus=young_modulus, poisson_ratio=0.45)
        invariance = torch.maximum((other.radial_stress - state.radial_stress.detach()).abs().max(),
                                   (other.hoop_stress - state.hoop_stress.detach()).abs().max()) / scale
    return {"equilibrium": float(equilibrium), "traction": float(traction),
            "constitutive": float(constitutive), "axial": float(axial),
            "poisson_invariance": float(invariance)}


def _relative(error: torch.Tensor, reference: torch.Tensor) -> float:
    return math.sqrt(float((error ** 2).sum() / (reference ** 2).sum()))


def solve_lame_case(poisson: float, *, formulation: str = "bbar", n_theta: int = 16,
                    n_radial: int = 8, inner_radius: float = 0.5, outer_radius: float = 0.8,
                    inner_pressure: float = 100.0e6, young_modulus: float = 210.0e9,
                    plane: str = "strain") -> dict:
    """Solve the annular-sector Lame case and score it against the closed form.

    ``plane="stress"`` is available for the compressible sanity check: the B-bar
    element must not degrade a case that does not lock.
    """
    if formulation not in FORMULATIONS:
        raise ValueError(f"formulation must be one of {FORMULATIONS}")
    nodes, cells = sector_q4_mesh(inner_radius, outer_radius, n_theta, n_radial)
    forces = circular_pressure_loads(n_theta, n_radial, inner_radius, inner_pressure, inner=True)
    fixed = symmetry_fixed_dofs(nodes, n_theta=n_theta, n_radial=n_radial)
    model = ContinuumModel(nodes, cells, torch.tensor(young_modulus, dtype=torch.float64),
                           torch.tensor(float(poisson), dtype=torch.float64),
                           torch.tensor(1.0, dtype=torch.float64), forces, fixed, plane=plane)
    solution = solve_continuum(model) if formulation == "standard" else solve_continuum_bbar(model)

    radius = torch.linalg.vector_norm(nodes, dim=1)
    exact_nodes = lame_state(radius, inner_radius=inner_radius, outer_radius=outer_radius,
                             inner_pressure=inner_pressure, young_modulus=young_modulus,
                             poisson_ratio=poisson, plane=plane)
    fe_radial = radial_component(solution.displacement.reshape(-1, 2), nodes)
    errors = {"displacement_l2": _relative(fe_radial - exact_nodes.radial_displacement,
                                           exact_nodes.radial_displacement)}

    centres = nodes[cells].mean(1)
    centre_radius = torch.linalg.vector_norm(centres, dim=1)
    sigma_r, sigma_theta, tau_rtheta = cartesian_to_polar_stress(solution.stress, centres)
    exact_centres = lame_state(centre_radius, inner_radius=inner_radius, outer_radius=outer_radius,
                               inner_pressure=inner_pressure, young_modulus=young_modulus,
                               poisson_ratio=poisson, plane=plane)
    errors["radial_stress_l2"] = _relative(sigma_r - exact_centres.radial_stress,
                                           exact_centres.radial_stress)
    errors["hoop_stress_l2"] = _relative(sigma_theta - exact_centres.hoop_stress,
                                         exact_centres.hoop_stress)
    # The in-plane trace is proportional to the dilatation, so it is the quantity
    # that volumetric locking corrupts; compare it as a field, not pointwise.
    exact_trace = exact_centres.radial_stress + exact_centres.hoop_stress
    errors["trace_stress_l2"] = _relative((sigma_r + sigma_theta) - exact_trace, exact_trace)
    errors["polar_shear_max"] = float(tau_rtheta.abs().max()) / float(
        exact_centres.radial_stress.abs().max())

    # Global equilibrium: the pressure resultant must be carried by the two
    # symmetry cuts (the bore nodes themselves are free, so their reaction is
    # zero by construction and is not a useful check).
    applied_resultant = float(forces.reshape(-1, 2).sum(dim=0).norm())
    support = solution.reaction[fixed.long()].reshape(-1, 2).sum(dim=0).norm()
    errors["support_equilibrium"] = abs(float(support) - applied_resultant) / applied_resultant

    return {"poisson_ratio": float(poisson), "formulation": formulation, "plane": plane,
            "n_theta": int(n_theta), "n_radial": int(n_radial),
            "elements": int(len(cells)), "dofs": int(2 * len(nodes)),
            "h": float(inner_radius * (outer_radius / inner_radius - 1.0) / n_radial),
            "errors": errors,
            "fields": {"centre_radius": centre_radius, "centre_x": centres[:, 0],
                       "centre_y": centres[:, 1], "sigma_r": sigma_r,
                       "sigma_theta": sigma_theta, "exact_radius": centre_radius,
                       "exact_sigma_r": exact_centres.radial_stress,
                       "exact_sigma_theta": exact_centres.hoop_stress}}


def conditioning_study(poissons=DEFAULT_POISSONS, *, n_theta: int = 16, n_radial: int = 8) -> list:
    """Condition number of the free-free stiffness (relevant for iterative solvers).

    The cured element is not free of cost: the mean-dilatation term still carries
    the bulk modulus, so the system becomes ill-conditioned as ``nu -> 1/2``.  The
    dense direct solver used here does not care, an iterative one would.
    """
    records = []
    for poisson in poissons:
        for formulation in FORMULATIONS:
            case = solve_lame_case(poisson, formulation=formulation, n_theta=n_theta,
                                   n_radial=n_radial)
            nodes, cells = sector_q4_mesh(0.5, 0.8, n_theta, n_radial)
            fixed = symmetry_fixed_dofs(nodes, n_theta=n_theta, n_radial=n_radial).long()
            mask = torch.ones(2 * len(nodes), dtype=torch.bool)
            mask[fixed] = False
            free = torch.arange(2 * len(nodes))[mask]
            model = ContinuumModel(nodes, cells, torch.tensor(210.0e9, dtype=torch.float64),
                                   torch.tensor(float(poisson), dtype=torch.float64),
                                   torch.tensor(1.0, dtype=torch.float64),
                                   torch.zeros(2 * len(nodes), dtype=torch.float64), fixed,
                                   plane="strain")
            stiffness = (solve_continuum(model) if formulation == "standard"
                         else solve_continuum_bbar(model)).stiffness[free][:, free]
            records.append({"poisson_ratio": float(poisson), "formulation": formulation,
                            "n_dofs_free": int(free.numel()),
                            "condition_number": float(torch.linalg.cond(stiffness)),
                            "radial_stress_l2": case["errors"]["radial_stress_l2"]})
    return records


def locking_sweep(poissons=DEFAULT_POISSONS, *, n_theta: int = 16, n_radial: int = 8) -> list:
    """Error versus Poisson ratio for both formulations on the same mesh."""
    return [solve_lame_case(poisson, formulation=formulation, n_theta=n_theta,
                            n_radial=n_radial)
            for poisson in poissons for formulation in FORMULATIONS]


def refinement_study(poisson: float = 0.4999, levels=((16, 8), (24, 12), (32, 16), (40, 20))) -> list:
    """Mesh refinement for both formulations at a fixed Poisson ratio."""
    return [solve_lame_case(poisson, formulation=formulation, n_theta=n_theta, n_radial=n_radial)
            for n_theta, n_radial in levels for formulation in FORMULATIONS]


def observed_order(coarse: dict, fine: dict, key: str = "radial_stress_l2") -> float:
    """Convergence order between two refinement levels (``h`` from the mesh)."""
    ratio = coarse["h"] / fine["h"]
    return math.log(coarse["errors"][key] / fine["errors"][key]) / math.log(ratio)