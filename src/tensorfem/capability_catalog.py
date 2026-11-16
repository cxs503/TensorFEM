"""TensorFEM capability catalog — ``@capability``-decorated pilot declarations.

This module is the *single source of truth* for the TensorFEM capability
registry (the TensorLBM capability-registry pilot, applied to TensorFEM as the
first solver-repo of the list).  Each capability is declared by applying the
``@capability`` decorator (from :mod:`tensorfem.capability_registry`) directly
to a wrapper function that delegates to the real public-API kernel.

Because every ``@capability`` argument here is a *literal*, the declaration is
discoverable both by ``import``-mode introspection (the live registry) and by
``ast``-mode parsing (which never imports this module) — see
:func:`tensorfem.capability_registry.dump_capabilities`.

Importing this module registers every capability in the process-wide registry.
"""
from __future__ import annotations

from . import (
    solvers,
    frame2d,
    continuum,
    modal,
    buckling,
    thermal,
    sparse_core,
    solid_plasticity,
    plate_mindlin,
    shell4,
    contact3d,
    marine_structures,
    marine_hydrodynamics,
)
from .capability_registry import capability

__all__ = [
    "solve_linear_static",
    "solve_frame_static",
    "solve_continuum",
    "solve_modes",
    "solve_linear_buckling",
    "solve_steady_thermal",
    "solve_sparse_static",
    "solve_load_steps",
    "solve_plate_point_load",
    "shell4_stiffness",
    "update_node_facet_contact",
    "solve_hull_girder_uniform_load",
    "morison_base_actions",
]


# --------------------------------------------------------------------------- #
# Capability declarations (literal @capability decorators)
# --------------------------------------------------------------------------- #

@capability(
    "capability.linear_static",
    title="Linear Static Solver",
    domain="structural / statics",
    maturity="mature",
    has_benchmark=True,
    has_test=True,
    in_public_api=True,
)
def solve_linear_static(model):
    """Linear static solver — delegates to ``tensorfem.solvers.solve_linear_static``."""
    return solvers.solve_linear_static(model)


@capability(
    "capability.frame_static",
    title="2-D Frame Static Solver",
    domain="structural / statics",
    maturity="mature",
    has_benchmark=True,
    has_test=True,
    in_public_api=True,
)
def solve_frame_static(model):
    """2-D frame static solver — delegates to ``tensorfem.frame2d.solve_frame_static``."""
    return frame2d.solve_frame_static(model)


@capability(
    "capability.continuum",
    title="Q4 Continuum Solver",
    domain="structural / continuum",
    maturity="mature",
    has_benchmark=True,
    has_test=True,
    in_public_api=True,
)
def solve_continuum(model):
    """Q4 continuum solver — delegates to ``tensorfem.continuum.solve_continuum``."""
    return continuum.solve_continuum(model)


@capability(
    "capability.modal",
    title="Modal Analysis",
    domain="dynamics / vibration",
    maturity="mature",
    has_benchmark=True,
    has_test=True,
    in_public_api=True,
)
def solve_modes(stiffness, mass, constrained_dofs, modes=6):
    """Modal analysis — delegates to ``tensorfem.modal.solve_modes``."""
    return modal.solve_modes(stiffness, mass, constrained_dofs, modes=modes)


@capability(
    "capability.buckling",
    title="Linear Buckling",
    domain="structural / stability",
    maturity="mature",
    has_benchmark=True,
    has_test=True,
    in_public_api=True,
)
def solve_linear_buckling(stiffness, geometric_unit, constrained_dofs, modes=3):
    """Linear buckling — delegates to ``tensorfem.buckling.solve_linear_buckling``."""
    return buckling.solve_linear_buckling(
        stiffness, geometric_unit, constrained_dofs, modes=modes
    )


@capability(
    "capability.thermal",
    title="Thermal Analysis",
    domain="thermal / heat transfer",
    maturity="mature",
    has_benchmark=True,
    has_test=True,
    in_public_api=True,
)
def solve_steady_thermal(model):
    """Steady thermal — delegates to ``tensorfem.thermal.solve_steady_thermal``."""
    return thermal.solve_steady_thermal(model)


@capability(
    "capability.sparse",
    title="Sparse Assembly & Iterative Solves",
    domain="numerics / linear algebra",
    maturity="mature",
    has_benchmark=True,
    has_test=True,
    in_public_api=True,
)
def solve_sparse_static(K, force, **kwargs):
    """Sparse static solve — delegates to ``tensorfem.sparse_core.solve_sparse_static``."""
    return sparse_core.solve_sparse_static(K, force, **kwargs)


@capability(
    "capability.plasticity",
    title="J2 Elastoplasticity",
    domain="material / plasticity",
    maturity="mature",
    has_benchmark=True,
    has_test=True,
    in_public_api=True,
)
def solve_load_steps(model, external_force, load_factors, **kwargs):
    """J2 load-path solver — delegates to ``tensorfem.solid_plasticity.solve_load_steps``."""
    return solid_plasticity.solve_load_steps(
        model, external_force, load_factors, **kwargs
    )


@capability(
    "capability.plate_mindlin",
    title="Mindlin Thick Plate",
    domain="structural / plate",
    maturity="mature",
    has_benchmark=True,
    has_test=True,
    in_public_api=True,
)
def solve_plate_point_load(nx, side, young, poisson, thickness, winkler_k, point_load, **kwargs):
    """Mindlin plate point-load — delegates to ``tensorfem.plate_mindlin.solve_plate_point_load``."""
    return plate_mindlin.solve_plate_point_load(
        nx, side, young, poisson, thickness, winkler_k, point_load, **kwargs
    )


@capability(
    "capability.shell4",
    title="Shell4 (MITC) Shell",
    domain="structural / shell",
    maturity="developing",
    has_benchmark=False,
    has_test=True,
    in_public_api=True,
)
def shell4_stiffness(xyz, young, poisson, thickness, **kwargs):
    """Shell4 stiffness — delegates to ``tensorfem.shell4.shell4_stiffness``."""
    return shell4.shell4_stiffness(xyz, young, poisson, thickness, **kwargs)


@capability(
    "capability.contact_3d",
    title="3-D Contact Mechanics",
    domain="contact / mechanics",
    maturity="developing",
    has_benchmark=False,
    has_test=True,
    in_public_api=True,
)
def update_node_facet_contact(point, vertices, faces, state, **kwargs):
    """3-D node-to-facet contact — delegates to ``tensorfem.contact3d.update_node_facet_contact``."""
    return contact3d.update_node_facet_contact(
        point, vertices, faces, state, **kwargs
    )


@capability(
    "capability.marine_structures",
    title="Marine Structures",
    domain="marine / structures",
    maturity="mature",
    has_benchmark=True,
    has_test=True,
    in_public_api=True,
)
def solve_hull_girder_uniform_load(**kwargs):
    """Hull-girder uniform load — delegates to ``tensorfem.marine_structures.solve_hull_girder_uniform_load``."""
    return marine_structures.solve_hull_girder_uniform_load(**kwargs)


@capability(
    "capability.marine_hydrodynamics",
    title="Marine Hydrodynamics",
    domain="marine / hydrodynamics",
    maturity="developing",
    has_benchmark=False,
    has_test=True,
    in_public_api=True,
)
def morison_base_actions(wave, member, x, time, **kwargs):
    """Morison base actions — delegates to ``tensorfem.marine_hydrodynamics.morison_base_actions``."""
    return marine_hydrodynamics.morison_base_actions(
        wave, member, x, time, **kwargs
    )
