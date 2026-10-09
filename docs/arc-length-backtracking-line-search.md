# Opt-in arc-length backtracking line search

The generic arc-length solver supports an explicit
`line_search="backtracking"` mode.  The default remains `None`, preserving the
existing full-Newton trajectory.  Backtracking scales only the Newton
correction; it does not change the predictor, arc metric, Newton tangent, or
branch selection.

The merit function combines a force-scaled equilibrium residual and the
normalized spherical arc constraint.  Trial corrections use factors
`1, 1/2, ...` down to the configured minimum.  A step for which no trial
reduces merit fails closed and enters the existing rejected-step path.  No
trial state is committed.  Accepted factors and evaluation counts are exposed
in solver diagnostics.

## Public von Mises arch qualification

A 24-point path with radius 0.005 captures the first analytic limit load with
0.0255% error.  Immediately beyond that peak, a deliberately difficult but
reproducible radius-0.4 continuation step is limited to six Newton iterations.

| Strategy | Accepted | Rejected/failed | Terminal load |
|---|---:|---:|---:|
| Fixed full Newton | 0 | 1 | no committed state |
| Backtracking Newton | 1 | 0 | 11.7789907558 |
| 20-iteration reference | 1 | 0 | 11.7789907558 |

Backtracking selects factor 0.5 on the first correction and reaches the same
equilibrium as the high-iteration fixed-Newton reference.  Relative load and
displacement differences are below `1e-12`, comfortably inside the 1%
qualification boundary.  It therefore eliminates one of one fixed-Newton
failures in this controlled difficult step.

An uninterrupted three-step line-search path and a one-step plus restarted
two-step path agree to `1e-12`.  New line-search checkpoints record the
strategy identity; attempting to resume one with fixed Newton is rejected.
Legacy checkpoints may explicitly opt into line search, but once written in
that mode cannot silently switch back.

This qualifies the minimal generic arc-length loop.  It does not yet qualify
automatic activation or the finite-rotation Shell4-specific corrector; both
remain separate future gates.

