"""Qualification cases for :mod:`tensorfem.axisym_shell`.

Each case pairs a closed-form (or independently coded) reference with a
shell-of-revolution solve and returns the numbers a gate can assert on.

``membrane_cylinder``
    Pressurised cylinder with open or closed ends.  The exact state is pure
    membrane with no bending boundary layer: ``N_theta = pR``, ``N_s = pR/2``
    (closed) or ``0`` (open) and ``w = pR^2 (1 - nu/2) / (E t)`` (closed) or
    ``pR^2/(E t)`` (open).  A straight-generatrix element represents that
    geometry exactly, so the expected error is round-off.

``membrane_cone``
    Closed truncated cone under internal pressure.  Membrane theory for a
    straight generatrix gives ``N_theta = p r / sin(phi)`` and
    ``N_phi = p r / (2 sin(phi))`` at every point, with ``phi`` the angle
    between the outward normal and the axis (``sin(phi) = dr/ds``).

``sphere_pressure``
    Closed sphere under internal pressure: the case on which the faceted
    ``Shell4`` path failed at ~1006% (``docs/benchmark-qualification-status.md``).
    ``exact_geometry=True`` uses circular-arc meridians (``R_s = R``).

``torus_pressure``
    Quarter torus under internal pressure: a curved meridian with **no pole**,
    so it isolates discretisation error from the axis singularity.  Membrane
    theory for a torus of tube radius ``a`` and ring radius ``R`` gives the
    constant hoop force ``N_theta = p a / 2`` and
    ``N_phi = p a (R + a sin(phi)/2) / (R + a sin(phi))``, derived here
    independently from axial equilibrium plus normal equilibrium.

``clamped_cylinder``
    Pressurised open-ended cylinder clamped in the radial/rotational sense at
    both ends, i.e. bending boundary layers at both ends.  The reference is
    :func:`tensorfem.cylindrical_pressure_reference.exact_axisymmetric_solution`
    -- an independently derived closed form used by the *Shell4* qualification,
    with autodiff derivatives -- so agreement here is not a re-evaluation of the
    same code path.
"""
from __future__ import annotations

import math

import torch

from .axisym_shell import AxisymShell


def _max_rel(values, reference) -> float:
    if isinstance(reference, (int, float)):
        reference = [reference] * len(values)
    return max(abs(v - r) / abs(r) for v, r in zip(values, reference))


def membrane_cylinder(*, radius: float, length: float, thickness: float, young: float,
                      poisson: float, pressure: float, closed_ends: bool,
                      facets: int = 8) -> dict:
    """Cylinder under internal pressure; the exact state is pure membrane."""
    nodes = [(radius, length - length * i / facets) for i in range(facets + 1)]
    model = AxisymShell(nodes, young=young, poisson=poisson, thickness=thickness,
                        pressure=pressure)
    if closed_ends:
        model.apply_axial_force(0, pressure * math.pi * radius ** 2)
        model.apply_axial_force(facets, -pressure * math.pi * radius ** 2)
    mid = facets // 2
    u = model.solve({model.dof(mid, "u"): 0.0})
    field = model.field(u)
    w = [float(u[model.dof(j, "w")]) for j in range(facets + 1)]
    w_exact = (pressure * radius ** 2 / (young * thickness) *
               (1.0 - 0.5 * poisson if closed_ends else 1.0))
    n_s_exact = pressure * radius / 2.0 if closed_ends else 0.0
    return {
        "case": "cylinder_closed" if closed_ends else "cylinder_open",
        "facets": facets,
        "N_theta": field["N_theta"],
        "N_theta_exact": pressure * radius,
        "N_theta_max_rel_error": _max_rel(field["N_theta"], pressure * radius),
        "N_s_max_abs_error": max(abs(v - n_s_exact) for v in field["N_s"]),
        "w": w,
        "w_exact": w_exact,
        "w_max_rel_error": max(abs(v - w_exact) / abs(w_exact) for v in w),
        "axial_reaction": float(model.reaction(u)[model.dof(mid, "u")]),
        "pressure_axial_load": (pressure * math.pi * radius ** 2 if closed_ends else 0.0),
    }


def membrane_cone(*, r_small: float, r_large: float, rise: float, thickness: float,
                  young: float, poisson: float, pressure: float,
                  facets: int = 12) -> dict:
    """Closed truncated cone under internal pressure (membrane exact state).

    ``r_large - r_small`` is the radial rise and ``rise`` the axial drop, so the
    generatrix makes ``cos(phi) = dr/ds`` with the normal.
    """
    dr, dz = r_large - r_small, -abs(rise)
    length = math.hypot(dr, dz)
    nodes = [(r_small + dr * i / facets, dz * i / facets) for i in range(facets + 1)]
    model = AxisymShell(nodes, young=young, poisson=poisson, thickness=thickness,
                        pressure=pressure)
    model.apply_axial_force(0, pressure * math.pi * r_small ** 2)
    model.apply_axial_force(facets, -pressure * math.pi * r_large ** 2)
    sn = abs(dz) / length                      # sin(phi) = dr'... = -dz/ds
    u = model.solve({model.dof(facets // 2, "u"): 0.0})
    field = model.field(u)
    n_theta_exact = [pressure * r / sn for r in field["r"]]
    n_phi_exact = [pressure * r / (2.0 * sn) for r in field["r"]]
    return {
        "case": "cone_closed",
        "facets": facets,
        "sn": sn,
        "r": field["r"],
        "N_theta": field["N_theta"],
        "N_theta_exact": n_theta_exact,
        "N_theta_max_rel_error": _max_rel(field["N_theta"], n_theta_exact),
        "N_phi": field["N_s"],
        "N_phi_exact": n_phi_exact,
        "N_phi_max_rel_error": _max_rel(field["N_s"], n_phi_exact),
        "axial_reaction": float(model.reaction(u)[model.dof(facets // 2, "u")]),
        "pressure_axial_load": pressure * math.pi * r_small ** 2,
    }


def sphere_nodes(radius: float, facets: int) -> list[tuple[float, float]]:
    """Meridional nodes of a hemisphere from the pole to the equator."""
    return [(radius * math.sin(0.5 * math.pi * i / facets),
             radius * math.cos(0.5 * math.pi * i / facets))
            for i in range(facets + 1)]


def sphere_pressure(*, radius: float, thickness: float, young: float, poisson: float,
                    pressure: float, facets: int = 8,
                    exact_geometry: bool = True) -> dict:
    """Closed sphere under internal pressure; pole-to-equator symmetry model."""
    nodes = sphere_nodes(radius, facets)
    model = AxisymShell(nodes, young=young, poisson=poisson, thickness=thickness,
                        pressure=pressure,
                        meridian_radius=radius if exact_geometry else None)
    u = model.solve({
        model.dof(0, "u"): 0.0, model.dof(0, "beta"): 0.0,
        model.dof(facets, "u"): 0.0, model.dof(facets, "beta"): 0.0,
    })
    field = model.field(u)
    membrane = pressure * radius / 2.0
    w_exact = pressure * radius ** 2 * (1.0 - poisson) / (2.0 * young * thickness)
    w = float(u[model.dof(0, "w")])
    anisotropy = max(abs(a - b) for a, b in zip(field["N_s"], field["N_theta"]))
    return {
        "case": "sphere_exact_geometry" if exact_geometry else "sphere_faceted",
        "facets": facets,
        "geometry": "circular_arc" if exact_geometry else "chord_facets",
        "N_membrane": list(field["N_theta"]),
        "N_phi": list(field["N_s"]),
        "N_theta_max_rel_error": _max_rel(field["N_theta"], membrane),
        "N_phi_max_rel_error": _max_rel(field["N_s"], membrane),
        "anisotropy_rel_error": anisotropy / membrane,
        "w": w,
        "w_exact": w_exact,
        "w_rel_error": abs(w - w_exact) / abs(w_exact),
        "N_membrane_exact": membrane,
        "sigma_exact": membrane / thickness,
    }


def torus_nodes(ring_radius: float, tube_radius: float, facets: int
                ) -> list[tuple[float, float]]:
    """Quarter torus from the crown (phi=0) to the outer equator (phi=pi/2)."""
    out = []
    for i in range(facets + 1):
        phi = 0.5 * math.pi * i / facets
        out.append((ring_radius + tube_radius * math.sin(phi),
                    tube_radius * math.cos(phi)))
    return out


def torus_membrane(ring_radius: float, tube_radius: float, pressure: float,
                   phi) -> tuple[list[float], list[float]]:
    """Exact torus membrane forces ``(N_phi, N_theta)`` at the normal angle ``phi``."""
    a, r_ring, p = tube_radius, ring_radius, pressure
    n_phi = [p * a * (r_ring + 0.5 * a * math.sin(x)) / (r_ring + a * math.sin(x))
             for x in phi]
    return n_phi, [p * a / 2.0] * len(list(phi))


def torus_pressure(*, ring_radius: float, tube_radius: float, thickness: float,
                   young: float, poisson: float, pressure: float,
                   facets: int = 8, exact_geometry: bool = True) -> dict:
    """Quarter torus under internal pressure (no pole, curved meridian)."""
    nodes = torus_nodes(ring_radius, tube_radius, facets)
    model = AxisymShell(nodes, young=young, poisson=poisson, thickness=thickness,
                        pressure=pressure,
                        meridian_radius=tube_radius if exact_geometry else None)
    u = model.solve({
        model.dof(0, "u"): 0.0, model.dof(0, "beta"): 0.0,
        model.dof(facets, "u"): 0.0, model.dof(facets, "beta"): 0.0,
    })
    field = model.field(u)
    n_phi_exact, n_theta_exact = torus_membrane(ring_radius, tube_radius, pressure,
                                                field["phi"])
    return {
        "case": "torus_exact_geometry" if exact_geometry else "torus_faceted",
        "facets": facets,
        "geometry": "circular_arc" if exact_geometry else "chord_facets",
        "phi": field["phi"],
        "N_theta": field["N_theta"],
        "N_theta_exact": n_theta_exact,
        "N_theta_max_rel_error": _max_rel(field["N_theta"], n_theta_exact),
        "N_phi": field["N_s"],
        "N_phi_exact": n_phi_exact,
        "N_phi_max_rel_error": _max_rel(field["N_s"], n_phi_exact),
    }


def clamped_cylinder(*, radius: float, length: float, thickness: float, young: float,
                     poisson: float, pressure: float, facets: int = 200,
                     shear_integration: str = "reduced") -> dict:
    """Open-ended pressurised cylinder with radial-clamped ends (boundary layers).

    Reference: :func:`cylindrical_pressure_reference.exact_axisymmetric_solution`
    (clamped--clamped closed form of the Shell4 qualification, an independent
    element family and an independent derivation).
    """
    from .cylindrical_pressure_reference import (cylinder_parameters,
                                                 exact_axisymmetric_solution)

    nodes = [(radius, length - length * i / facets) for i in range(facets + 1)]
    model = AxisymShell(nodes, young=young, poisson=poisson, thickness=thickness,
                        pressure=pressure, shear_integration=shear_integration)
    u = model.solve({
        model.dof(0, "w"): 0.0, model.dof(0, "beta"): 0.0,
        model.dof(facets, "w"): 0.0, model.dof(facets, "beta"): 0.0,
        model.dof(facets, "u"): 0.0,
    })
    xs = torch.tensor([length - length * i / facets for i in range(facets + 1)],
                      dtype=torch.float64)
    x_sorted, order = torch.sort(xs)
    w_fe = torch.tensor([float(u[model.dof(j, "w")]) for j in range(facets + 1)],
                        dtype=torch.float64)[order]
    parameters = cylinder_parameters(radius=radius, thickness=thickness, young=young,
                                     poisson=poisson, pressure=pressure, length=length)
    exact = exact_axisymmetric_solution(x_sorted, parameters)
    w_mem = parameters.membrane_radial_displacement
    err = (w_fe - exact.displacement).abs() / w_mem
    return {
        "case": "cylinder_clamped",
        "facets": facets,
        "shear_integration": shear_integration,
        "x": x_sorted.tolist(),
        "w_fe": w_fe.tolist(),
        "w_exact": exact.displacement.tolist(),
        "w_membrane": w_mem,
        "boundary_layer": parameters.boundary_layer_thickness,
        "w_rel_l2_error": float(torch.linalg.vector_norm(w_fe - exact.displacement)
                                / torch.linalg.vector_norm(exact.displacement)),
        "w_max_scaled_error": float(err.max()),
    }