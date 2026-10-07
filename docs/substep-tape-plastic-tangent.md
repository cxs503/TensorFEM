# Chained tangent across an accepted plastic substep tape

Adaptive control first records normalized endpoints for every accepted pair of
half steps. Stress and `dP/dF` are then replayed through that immutable sequence,
so AD explicitly chains every return map without differentiating recursive
control flow. The tape is deterministic and restartable.

Before use at a neighboring Newton iterate, every recorded interval repeats its
full-step/two-half-step estimator. If any error exceeds 1.05 times the original
tolerance, replay raises `RetapeRequired`; rebuilding the tape is explicit and
does not mutate committed state. Thus the derivative remains piecewise smooth
and never pretends the tree-selection boundary is differentiable.

Large non-coaxial tests compare stress and tangent to adaptive AD and Richardson
below 3%, while checking Newton convergence, plastic volume, dissipation and
transactional failure.
