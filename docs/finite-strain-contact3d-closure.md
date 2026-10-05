# Finite-strain double-deformable contact closure

Two compressible Neo-Hookean blocks are discretised by real TET4 elements.
Their opposed QUAD4 faces use current-configuration Mortar projection. Total-
Lagrangian solid internal force/tangent and contact residual/tangent enter the
same Newton system; line search rejects inverted deformation gradients.

The load path closes the initial gap and then applies finite tangential
relative motion. Gates cover both-body deformation, positive Jacobians,
strain/contact energy, interface force balance, equilibrium and failed-step
rollback.

This is a matching planar frictionless finite-strain integration closure. It
does not yet qualify curved finite-strain contact, finite-strain friction,
self-contact, impact or production large-sliding segmentation, so the general
surface-to-surface scope remains blocked.
