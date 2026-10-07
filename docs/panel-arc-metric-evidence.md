# Panel continuation metric and checkpoint evidence

The original panel continuation norm mixed translations in metres and nodal
rotations in radians with unit weights.  On the 8x8 path at 480.2 kN,
99.9934% of the squared increment was an unconstrained drilling rotation.  The
apparent load plateau was therefore numerical gauge drift, not a physical
limit point.

The `dimensionally_scaled` metric follows the through-thickness displacement
field `u + z x theta`: translations and the scaled load coordinate have unit
weight, bending rotations have weight `t^2/12`, and drilling rotation has zero
arc-length weight while remaining in the equilibrium equations.  Metric name
and solver-step cap are checkpoint identity fields, so incompatible paths
cannot be resumed silently.

The 8x8 state was migrated through a hash-verified generation at point 181.
With the corrected metric it advanced to point 340 and 580.889 kN with no
rejected point, maximum relative equilibrium residual `1.38e-13`, and terminal
absolute energy residual `5.25e-4 J`.  The path remains monotone, so no 8x8
peak is claimed.  The corresponding 12x12 controlled migration reached point
20 and 309.197 kN with all energy gates passing.

Checkpoints are immutable generation files.  A manifest atomically publishes
the filename and SHA-256 of a completed generation.  Fault injection between
generation publication and manifest commit leaves the prior generation valid.
The interrupted 8x8 incident was recovered to point 181 only after both the
load value and complete mechanical-state hash matched the portable manifest.
