# Finite-strain symmetric-friction contact closure

This closure combines two real compressible Neo-Hookean TET4 bodies with the
equal-weight, two-history frictional Mortar interface. Total-Lagrangian solid
and algorithmic contact tangents share one Newton matrix. Normal increments
close the gap, followed by sticking and sliding shear increments. Histories
are committed only after equilibrium.

Executable gates cover stick-to-slip transition, nonnegative and nonduplicated
dissipation, positive deformation Jacobians, interface balance, a centred
difference of the combined solid/contact tangent and exact failed-step
rollback. Constitutive and contact rigid-rotation covariance are independently
gated by their component qualifications.

This qualifies a matching planar finite-strain friction integration subset.
Curved finite-strain friction, self-contact, impact and production segmentation
remain outside the general claim.
