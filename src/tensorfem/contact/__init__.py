"""Contact sub-package: contact mechanics formulations.

Two formulation families live here:

* :mod:`tensorfem.contact.unilateral` -- unilateral contact solved as a
  complementarity problem (node-to-rigid-plane gaps, Winkler foundation);
* :mod:`tensorfem.contact.planes` -- small-deformation frictionless contact
  with rigid planes (active-set and penalty methods) plus the Hertz sphere
  reference.

Both were reachable as ``tensorfem.contact`` at different times: the plane
contact formulation used to be the flat module ``tensorfem/contact.py``, which
a package of the same name shadows on import. The formulation now lives inside
the package so ``tensorfem.contact.solve_rigid_plane_contact`` and friends are
importable again.
"""
from .planes import (
    ContactResult,
    HertzSphereReference,
    hertz_sphere_on_halfspace,
    solve_rigid_plane_contact,
)
from .unilateral import (
    UnilateralContactResult,
    complementarity_residual,
    solve_unilateral_contact,
    solve_winkler_unilateral,
)

__all__ = [
    "ContactResult",
    "HertzSphereReference",
    "UnilateralContactResult",
    "complementarity_residual",
    "hertz_sphere_on_halfspace",
    "solve_rigid_plane_contact",
    "solve_unilateral_contact",
    "solve_winkler_unilateral",
]
