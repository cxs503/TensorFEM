# Hybrid tangent global cyclic solver

The global driver selects hybrid implicit/adaptive-AD tangents independently at
every TET4 integration point and supports increasing, reversed and cyclic load
targets. It also runs pure adaptive AD or Richardson under the identical step
controller for qualification. Every assembly records material path count,
strategy count and fallback reason. Only converged increments commit point
history and cumulative dissipation; results are restart checkpoints.

CI uses two TET4 integration domains and compares the full cycle against both
pure AD and Richardson below 3%. It requires no Newton iteration regression,
fewer material paths than Richardson, nonnegative dissipation, restart
equivalence and forced-failure rollback. The coincident two-element topology is
a solver/state isolation benchmark, not a physical mesh convergence claim.
