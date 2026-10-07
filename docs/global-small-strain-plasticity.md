# Global multi-TET4 small-strain plasticity

`global_plasticity.py` turns the existing TET4 J2 return mapping into a reusable
multi-element global analysis path. It assembles internal force and consistent
algorithmic tangent, stores one integration-point state per constant-strain
tetrahedron, performs adaptive Newton increments, and commits history only
after convergence. Rejected increments and terminal failures leave the caller's
checkpoint unchanged. A converged result can be supplied as the next run's
restart state, including for unloading.

The qualification model is a unit bar represented by two cubes and twelve
conforming TET4 elements. Symmetry boundaries and a uniform end traction give
the independent homogeneous uniaxial solution

`epsilon_x = sigma/E + (sigma-sigma_y)/H`.

Loading crosses the elastic/plastic transition, reaches 400 stress units, then
unloads to zero. Tip displacement, element stress and residual strain agree
with the analytical solution far inside the 3% limit. An uninterrupted path
and a restarted path are bitwise identical. Tests also force Newton failure and
verify transactional rollback.

This is a **small-strain**, quasi-static, isotropic-hardening J2 formulation
with one integration point per TET4. It does not claim finite deformation,
finite rotation, mixed pressure-displacement treatment, contact coupling, or
general non-proportional production qualification.
