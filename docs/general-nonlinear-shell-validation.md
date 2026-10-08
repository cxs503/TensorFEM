# Experimental general doubly curved nonlinear Shell4

The general shell uses arbitrary three-dimensional reference facets, a proper
best-fit corotational frame, exponential-map nodal rotations and the projected
selective-integration Shell4 stiffness. Internal force and the complete
material/geometric tangent are first and second derivatives of one energy.
The global step provides shared-node assembly, proportional constraints and
loads, full Newton, energy backtracking, adaptive cutback, rollback and JSON
restart.

An independent two-facet spherical panel exercises normals changing in both
surface directions. Its directional tangent is checked by central difference
within `3e-6`; its load-displacement path must be monotonic and restart must
reproduce the accepted state. No external reference is asserted for this
integration panel.

Spatial accuracy remains tied to the public MacNeal--Harder hemisphere with an
18-degree hole: corrected 20x20, 24x24 and 28x28 errors are 2.693%, 2.473%
and 2.317%. The last two displacements differ by approximately 0.152%.
The skew-gradient correction supersedes the earlier 12/16/20 results.
Scordelis--Lo Roof and Pinched
Cylinder retain their separate sourced gates.

This does **not** yet qualify a universal nonlinear curved shell: the three
classical benchmarks are small-strain responses, while the nonlinear driver is
an independently converged engineering integration case. Finite-strain shell
materials, through-thickness integration, follower pressure and nonlinear
published snap-through benchmarks remain future qualification work.
