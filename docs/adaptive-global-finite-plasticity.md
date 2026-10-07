# Adaptive finite plasticity in the global TET4 Newton loop

Each integration point stores both multiplicative plastic history and its last
committed total deformation gradient. Every global Newton trial integrates from
that immutable state using the full-step/two-half-step controller. Only a
converged load increment commits all point states and cumulative dissipation;
rejected increments and terminal failures roll back transactionally. A result
is a complete checkpoint for reversed or cyclic continuation.

The material algorithm is coupled to a Richardson-centred element residual
tangent. A TET4 tangent requires 48 perturbed material integrations plus one
base response per Newton evaluation. This cost is recorded as 49 evaluations
per element; it is robust and auditable but substantially more expensive than a
closed-form consistent tangent.

Qualification combines a shear--stretch--reverse--unload material cycle against
500 uniform substeps per segment with a force-controlled global reverse cycle.
It checks stress and `Fp` below 3%, nonnegative hysteretic dissipation,
`det(Fp)=1`, Newton iteration bounds, split-run restart and forced-failure
rollback. Objectivity is inherited from and regression-tested in the underlying
adaptive material suite. General multiaxial production use still requires a
closed-form/implicit consistent tangent and broader cyclic benchmark coverage.
