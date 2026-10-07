# Curved nonmatching finite-strain friction foundation

The structured generator creates two independent three-dimensional TET4
blocks. The lower contact surface is planar; the upper surface is a shallow
paraboloid embedded directly in the reference solid mesh. Master and slave
surface resolutions are independent, and every contact node belongs to real
Neo-Hookean volume elements.

The first executable closure uses 1x1 master and 2x2 slave surfaces with
normal closure followed by tangential stick/slip loading. Total-Lagrangian
solid tangents and equal-weight two-pass frictional Mortar tangents share the
same Newton solve. It gates equilibrium, positive Jacobians, interface force
balance and nonduplicated dissipation. A 2x2/3x3 construction gate proves the
mesh family is extensible.

The opt-in two-level qualification now closes both 1x1/2x2 and 2x2/3x3 real
solid paths.  The recorded run completed in 138.02 s.  Coulomb relative errors
were `1.54e-5` and `7.01e-6`; interface force imbalance stayed below `9e-16`,
minimum `det(F)` exceeded `0.9984`, and master/slave interchange error was
`7.05e-13`.  Its evidence hash is
`8722cc556009bd3e5644152e32d25dd6ccd44a5c48c71ce604f6261719d03e6d`.

Stick/slip branch crossings use a signed secant displacement predictor based
on the last two committed load states.  It changes only the Newton initial
guess, checks `det(F)>0`, and retains transactional history.  If a target still
fails, adaptive bisection inserts fully converged load states but always returns
to the original requested target.

This qualifies the curved, nonmatching, double-deformable finite-strain
friction subset. It does not claim self-contact, impact, or production contact
segmentation; those remain independent capability boundaries.

The opt-in qualification executes complete 1/2 and 2/3 paths and repeats the
2/3 path after exchanging contact roles. It gates the analytic Coulomb cone,
interface action-reaction, positive `det(F)`, nonnegative weighted dissipation,
equilibrium and rollback, then records a canonical SHA-256 evidence hash.

Run with `TENSORFEM_RUN_SLOW_FINITE_STRAIN_FRICTION=1 PYTHONPATH=src python scripts/run_curved_finite_strain_friction_qualification.py`.
