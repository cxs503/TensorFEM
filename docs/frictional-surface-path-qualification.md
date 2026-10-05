# Double-deformable frictional surface path

This path couples the frictional Mortar residual and frozen-return-branch
algorithmic tangent to independent three-direction foundation stiffness on
both surfaces. A nonmatching 2/3 faceted parabolic-cap/plane model is advanced
through normal closure, sticking shear and sliding shear load steps. Coulomb
history is committed only after Newton convergence; a forced failed increment
must preserve both displacement and integration-point history.

The qualification checks stick-to-slip transition, the sliding `|T|=mu*N`
oracle below 3%, two-sided force balance and the moment of the complete nodal
equilibrium residual. Rigid-motion objectivity and tangent consistency are
covered by the frictional-Mortar assembly qualification.

This is a real curved, nonmatching, double-deformable Newton path, but
`general_surface_to_surface` remains blocked pending curved frictional mesh
convergence, symmetric two-pass friction history and finite-strain contact.
