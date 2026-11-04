"""Plane-strain Q4 locking qualification against the closed-form Lame cylinder.

This is the 2-D counterpart of :mod:`tensorfem.incompressible_locking`, which
qualified the HEX8 family against the 3-D spherical Lame field and left the 2-D
plane-strain Q4 path as an explicitly recorded debt: ``nu >= 0.499`` gave 14 %-155 %
error with no mitigation available in the library.

The remedy now exists (:mod:`tensorfem.q4_volumetric`: B-bar projection and
selective reduced integration), so the same question can be answered rather than
merely measured:

* the **stress field of the thick cylinder is statically determinate**, hence
  identical for plane stress and plane strain and *independent of Poisson's ratio*;
* the **displacement field carries the Poisson ratio** in closed form,
  ``u_r = (1+nu)/E [(1-2nu) A r + B/r]``, and its volumetric part is ``2 (1+nu)(1-2nu) A / E``,
  which vanishes as ``nu -> 1/2`` -- i.e. the reference state really is
  incompressible in that limit.

Therefore a locking-free formulation keeps the *stress* error flat in ``nu``, and
the standard displacement Q4 does not.  Geometry, consistent arc pressure loads
and the symmetry constraints are taken from :mod:`tensorfem.lame_reference`; no
second discretisation or load path is introduced.

Nothing here is a new solver: it is a verification layer over
:func:`tensorfem.continuum.solve_continuum`.
"""
from __future__ import annotations

import math
import time

import torch

from .continuum import ContinuumModel, solve_continuum
from .lame_reference import (cartesian_to_polar_stress, circular_pressure_loads, lame_state,
                             radial_component, sector_q4_mesh, symmetry_fixed_dofs)

#: formulations compared by the qualification sweep
FORMULATIONS = ("standard", "bbar", "sri")
#: Poisson ratios of the sweep; ``0.4999`` is the value quoted by the merged
#: near-incompressible work, ``0.3`` is the lock-free reference point
DEFAULT_POISSONS = (0.3, 0.45, 0.49, 0.499, 0.4999)
#: reference (mesh, formulation independent) geometry of the case
DEFAULT_GEOMETRY = dict(inner_radius=0.5, outer_radius=0.8, pressure=100.0e6,
                        young_modulus=210.0e9, sector_angle=math.pi / 2.0)


def _relative(error: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.sqrt((error ** 2).sum() / (reference ** 2).sum()))


def _lame(**geometry):
    spec = dict(DEFAULT_GEOMETRY)
    spec.update({k: v for k, v in geometry.items() if v is not None})
    return spec


def solve_case(poisson: float, *, n_theta: int = 16, n_radial: int = 8,
               formulation: str = "bbar", **geometry):
    """Solve the sector annulus with one Q4 formulation and score it on the Lame field.

    Errors are the same unweighted point-wise RMS measures used by the recorded
    2-D debt table of :mod:`tensorfem.incompressible_locking`, so the numbers are
    directly comparable across the two studies.
    """
    if formulation not in FORMULATIONS:
        raise ValueError(f"formulation must be one of {FORMULATIONS}")
    spec = _lame(**geometry)
    started = time.perf_counter()
    nodes, cells = sector_q4_mesh(spec["inner_radius"], spec["outer_radius"], n_theta, n_radial)
    forces = circular_pressure_loads(n_theta, n_radial, spec["inner_radius"], spec["pressure"],
                                     inner=True, sector_angle=spec["sector_angle"])
    fixed = symmetry_fixed_dofs(nodes, n_theta=n_theta, n_radial=n_radial,
                               sector_angle=spec["sector_angle"])
    # NB: explicit float64 -- at ``nu -> 1/2`` the bulk modulus scales like
    # ``1/(1-2nu)``, so a float32 Poisson ratio (torch's default for a Python
    # float) is amplified into ~1e-4 relative stress error.  See the note in
    # docs/plane-strain-q4-locking.md.
    dtype = nodes.dtype
    model = ContinuumModel(nodes, cells, torch.tensor(spec["young_modulus"], dtype=dtype),
                           torch.tensor(float(poisson), dtype=dtype),
                           torch.tensor(1.0, dtype=dtype),
                           forces, fixed, plane="strain", formulation=formulation)
    result = solve_continuum(model)
    displacement = result.displacement.reshape(-1, 2)

    state = lame_state(torch.linalg.vector_norm(nodes, dim=1),
                       inner_radius=spec["inner_radius"], outer_radius=spec["outer_radius"],
                       inner_pressure=spec["pressure"], young_modulus=spec["young_modulus"],
                       poisson_ratio=float(poisson), plane="strain")
    exact_radial = state.radial_displacement
    fe_radial = radial_component(displacement, nodes)
    errors = {"displacement_l2": _relative(fe_radial - exact_radial, exact_radial),
              "displacement_max_relative": float(((fe_radial - exact_radial).abs()
                                                 / exact_radial.abs()).max())}

    centres = nodes[cells].mean(1)
    sigma_r, sigma_theta, _ = cartesian_to_polar_stress(result.stress, centres)
    exact_centres = lame_state(torch.linalg.vector_norm(centres, dim=1),
                              inner_radius=spec["inner_radius"], outer_radius=spec["outer_radius"],
                              inner_pressure=spec["pressure"], young_modulus=spec["young_modulus"],
                              poisson_ratio=float(poisson), plane="strain")
    errors["radial_stress_l2"] = _relative(sigma_r - exact_centres.radial_stress,
                                           exact_centres.radial_stress)
    errors["radial_stress_max_relative"] = float(((sigma_r - exact_centres.radial_stress).abs()
                                                 / exact_centres.radial_stress.abs()).max())
    errors["hoop_stress_l2"] = _relative(sigma_theta - exact_centres.hoop_stress,
                                         exact_centres.hoop_stress)

    # mean (volumetric) stress: in plane strain sigma_z = nu (sigma_x + sigma_y)
    mean_fe = (1.0 + poisson) * (result.stress[:, 0] + result.stress[:, 1]) / 3.0
    mean_exact = (1.0 + poisson) * (exact_centres.radial_stress + exact_centres.hoop_stress) / 3.0
    errors["mean_stress_l2"] = _relative(mean_fe - mean_exact, mean_exact)

    bore = torch.abs(torch.linalg.vector_norm(nodes, dim=1) - spec["inner_radius"]) < 1e-12
    bore_fe = float(fe_radial[bore].mean())
    errors["bore_displacement_relative"] = abs(bore_fe - float(exact_radial[bore].mean())) / abs(
        float(exact_radial[bore].mean()))

    # (1) assembly-level energy check: for a linear single-load step U = 1/2 f.u holds
    # exactly, so this is a consistency residual, expected at round-off
    external_work = 0.5 * float(result.displacement @ forces)
    errors["energy_balance_relative"] = abs(float(result.strain_energy) - external_work) / abs(
        external_work)
    # (2) the *analytic* boundary work 1/2 p u_r(a) (a * sector) differs from the
    # consistent-arc load resultant by O(dtheta^2) redistribution; recorded, not gated
    analytic_work = 0.5 * spec["pressure"] * bore_fe * spec["inner_radius"] * spec["sector_angle"]
    errors["analytic_work_relative"] = abs(external_work - analytic_work) / abs(analytic_work)

    return {
        "poisson_ratio": float(poisson),
        "formulation": formulation,
        "n_theta": int(n_theta),
        "n_radial": int(n_radial),
        "elements": int(len(cells)),
        "dofs": int(2 * len(nodes)),
        "errors": errors,
        "solve_seconds": time.perf_counter() - started,
        "displacement": displacement,
        "stress": result.stress,
        "nodes": nodes,
        "cells": cells,
        "bore_displacement": bore_fe,
        "exact_bore_displacement": float(exact_radial[bore].mean()),
    }


def locking_sweep(poissons=DEFAULT_POISSONS, *, n_theta: int = 16, n_radial: int = 8,
                  formulations=FORMULATIONS, **geometry):
    """Error-vs-Poisson records for every requested formulation."""
    records = []
    for poisson in poissons:
        for formulation in formulations:
            record = solve_case(poisson, n_theta=n_theta, n_radial=n_radial,
                                formulation=formulation, **geometry)
            records.append({k: v for k, v in record.items()
                            if k not in ("displacement", "stress", "nodes", "cells")})
    return records


def refinement_study(poisson: float, *, meshes=((8, 4), (16, 8), (32, 16), (64, 32)),
                     formulations=FORMULATIONS, **geometry):
    """Mesh-refinement records at a fixed Poisson ratio."""
    records = []
    for n_theta, n_radial in meshes:
        for formulation in formulations:
            record = solve_case(poisson, n_theta=n_theta, n_radial=n_radial,
                                formulation=formulation, **geometry)
            records.append({k: v for k, v in record.items()
                            if k not in ("displacement", "stress", "nodes", "cells")})
    return records


def _mesh_size(n_theta: int, n_radial: int, **geometry) -> float:
    spec = _lame(**geometry)
    return (spec["outer_radius"] - spec["inner_radius"]) / n_radial


def observed_order(records, metric: str, formulation: str, **geometry):
    """Observed convergence order of ``metric`` from the two finest records."""
    subset = [r for r in records if r["formulation"] == formulation]
    subset.sort(key=lambda r: r["n_radial"])
    if len(subset) < 2:
        raise ValueError("need at least two refinements")
    coarse, fine = subset[-2], subset[-1]
    ratio = _mesh_size(coarse["n_theta"], coarse["n_radial"], **geometry) / _mesh_size(
        fine["n_theta"], fine["n_radial"], **geometry)
    return math.log(coarse["errors"][metric] / fine["errors"][metric]) / math.log(ratio)


def flatness(records, metric: str, formulation: str):
    """Ratio of the worst to the best ``metric`` value over the Poisson sweep."""
    values = [r["errors"][metric] for r in records if r["formulation"] == formulation]
    return max(values) / min(values)

def reference_self_check(poisson: float, *, samples: int = 17, **geometry) -> dict:
    """Re-derive the plane-strain Lame reference from independent residuals.

    Mirrors the discipline used for the 3-D field in
    :func:`tensorfem.incompressible_locking.reference_self_check`: the reference
    is not taken on trust, it is re-checked against

    ``equilibrium``    ``d sigma_r/d r + (sigma_r - sigma_theta)/r = 0``
    ``traction``       ``sigma_r(a) = -p`` and ``sigma_r(b) = 0``
    ``constitutive``   ``eps_r = du/dr``, ``eps_theta = u/r`` and ``eps_z = 0``
                       against the isotropic law with the same ``E, nu``
    ``volumetric``     the volumetric strain ``eps_r + eps_theta = 2(1+nu)(1-2nu)A/E``,
                       i.e. the reference state really is incompressible as ``nu -> 1/2``
    """
    spec = _lame(**geometry)
    a, b, young, p = (spec["inner_radius"], spec["outer_radius"],
                      spec["young_modulus"], spec["pressure"])
    radii = torch.linspace(a, b, samples, dtype=torch.float64)
    state = lame_state(radii, inner_radius=a, outer_radius=b, inner_pressure=p,
                       young_modulus=young, poisson_ratio=float(poisson), plane="strain")
    sigma_r, sigma_theta, sigma_z = state.radial_stress, state.hoop_stress, state.axial_stress

    h = 1e-6 * a
    hi = radii + h
    lo = radii - h
    sr_hi = lame_state(hi, inner_radius=a, outer_radius=b, inner_pressure=p,
                       young_modulus=young, poisson_ratio=float(poisson), plane="strain").radial_stress
    sr_lo = lame_state(lo, inner_radius=a, outer_radius=b, inner_pressure=p,
                       young_modulus=young, poisson_ratio=float(poisson), plane="strain").radial_stress
    equilibrium = (sr_hi - sr_lo) / (2.0 * h) + (sigma_r - sigma_theta) / radii
    equilibrium_residual = float(equilibrium.abs().max()) / float((sigma_r.abs() / radii).max())

    traction_residual = max(abs(float(sigma_r[0]) + p) / p, abs(float(sigma_r[-1])) / p)

    # analytic du/dr of u_r = (1+nu)/E [(1-2nu) A r + B/r]
    a_coef, b_coef = (float(v) for v in
                      __import__("tensorfem.lame_reference", fromlist=["x"]).lame_coefficients(a, b, p))
    du = (1.0 + poisson) / young * ((1.0 - 2.0 * poisson) * a_coef - b_coef / radii ** 2)
    hoops = state.radial_displacement / radii
    epsilon_r = (sigma_r - poisson * (sigma_theta + sigma_z)) / young
    epsilon_theta = (sigma_theta - poisson * (sigma_r + sigma_z)) / young
    epsilon_z = (sigma_z - poisson * (sigma_r + sigma_theta)) / young
    strain_scale = float(torch.maximum(epsilon_r.abs(), epsilon_theta.abs()).max())
    constitutive_residual = max(float((du - epsilon_r).abs().max()),
                                float((hoops - epsilon_theta).abs().max()),
                                float(epsilon_z.abs().max())) / strain_scale
    volumetric = float((epsilon_r + epsilon_theta).max()) / strain_scale
    return {"equilibrium": equilibrium_residual,
            "traction": float(traction_residual),
            "constitutive": constitutive_residual,
            "volumetric_strain_over_strain_scale": volumetric,
            "poisson_ratio": float(poisson)}
