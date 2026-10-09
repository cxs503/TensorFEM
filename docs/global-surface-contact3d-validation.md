# Global 3-D surface-contact validation

`tensorfem.global_contact3d` is a compact reference for coupling deformable
three-dimensional contact to a global FE residual. It is intentionally limited
to slave-node quadrature against a TRI3 search surface; it is **not** a mortar,
self-contact, CCD, or general production contact implementation.

## Discretisation and state

Each `NodeTrianglePair` carries a positive tributary weight. At every residual
evaluation the slave point is projected onto all candidate triangles in the
current configuration. The selected TRI3 barycentric shape functions distribute
the opposite force to the three master nodes. Thus the assembled nodal contact
force has zero resultant to roundoff. For frictionless contact its moment also
vanishes to roundoff because the slave-master separation and normal traction are
collinear.

Normal contact uses a penalty law. Tangential contact uses an elastic predictor
and Coulomb return with accumulated dissipated energy. Relative slave/master
increments remove common rigid translation. Trial histories are rebuilt from an
immutable committed state at every Newton iteration and are committed only after
convergence; failed increments therefore cannot contaminate later steps.

The tangent is the algorithmic Jacobian of the assembled structural/contact
residual within a fixed active/search branch. Facet switches and stick-slip
transitions remain nonsmooth, so the reference solver uses backtracking.

## Automated evidence

`tests/test_global_contact3d.py` checks:

- contact force entering global equilibrium rather than being post-processed;
- TRI3 shape-function reactions, resultant and frictionless moment balance;
- current-configuration facet switching during large sliding;
- stick-slip history and positive frictional dissipation;
- invariance to a common rigid translation and covariance under finite rotation;
- mesh-triangulation and load-step agreement;
- monotone penalty convergence toward the rigid-contact reaction;
- algorithmic tangent agreement with centred differences; and
- transactional rollback after a deliberately failed Newton increment.

With friction and finite penalty penetration, applying equal/opposite tangential
forces at the separated slave and projected master points creates a small couple
of order `penetration * tangential_force`. It converges to zero with penetration;
the code reports `moment_imbalance` rather than hiding this low-order limitation.

## Explicit capability boundary

The module does not provide dual mortar spaces, surface-to-surface integration,
unbiased two-pass contact, augmented Lagrange updates, self-contact exclusion,
continuous collision detection, contact stabilization, or scalable broad-phase
search. Those capabilities require a separate production discretisation and
must not be inferred from this qualification slice.
