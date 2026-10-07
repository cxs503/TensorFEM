# Experimental global nonlinear cylindrical-shell step

The global step assembles the exact element energy, internal force and
consistent 24x24 tangent from `shell_consistent`. It provides shared-node DOF
mapping, proportional loads and prescribed displacements, full Newton updates,
potential-energy backtracking, adaptive increment growth/cutback, rollback,
and schema-tagged JSON checkpoint/restart. Singular or nonconverged increments
are reduced; exhaustion below the minimum increment returns `converged=False`
rather than accepting an invalid state.

The auditable integration case is a two-element cylindrical strip of length
2, arc width `R*DeltaTheta=1`, thickness 0.02, `E=2e8`, `nu=0`, and total axial
force 1000. All non-axial DOFs are restrained. Its membrane response reduces
exactly to the axial-bar formula `u=F L/(E t R DeltaTheta)`. Acceptance requires
relative displacement error below 3%, reaction balance, successful adaptive
increments, restart equivalence, and whole-mesh finite-rigid-motion energy and
internal force at roundoff.

This is an integration verification, not a substitute for the published
Scordelis--Lo test (separately below 3% at 8x8). The feature remains
**experimental**: production sparse assembly and a full nonlinear Roof solve
are not yet connected. Pinched Cylinder and Hemispherical Shell are not
implemented or claimed.
