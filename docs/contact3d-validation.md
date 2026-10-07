# Three-dimensional node-to-facet contact

## Scope

`contact3d.py` is a finite-sliding local contact kernel for multiple slave
nodes and oriented TRI3 or QUAD4 master facets in the current configuration.
QUAD4 facets are searched as two triangles, so warped facets remain
well-defined.  The kernel provides:

- global closest-facet search and barycentric force distribution;
- penalty or augmented-Lagrangian normal enforcement;
- vector Coulomb stick/slip return mapping with committed history;
- slave/master action--reaction forces, recoverable energy and dissipation;
- explicit exclusion of incident facets for building self-contact candidate
  filters.

This is not mortar or segment-to-segment contact.  It does not provide a
broad-phase spatial index, automatic self-contact topology generation, a
global nonlinear contact solver, or a Hertz continuous-pressure solution.

## Verification contracts

1. A point penetrating an interior TRI3 location reproduces
   `lambda = k_n delta`; barycentric forces preserve resultant force and
   moment to round-off.
2. On a 30-degree plane, stick gives `|t| = k_t |s_e|`, sliding gives
   `|t| = mu lambda`, and plastic slip produces positive dissipation.
3. A slave crosses a QUAD4 split and then crosses onto the adjacent face;
   the current-configuration global search recovers the analytical face,
   projected point and signed gap to round-off.
4. Two slave nodes assemble equal and opposite master reactions.  A second
   augmented-Lagrangian update doubles the multiplier for unchanged
   penetration, exactly matching `lambda_(n+1)=lambda_n-k_n g`.
5. Degenerate geometry and an empty filtered candidate set fail closed.

All scalar force, projection and energy-oracle errors are below 3%; the
current automated cases attain floating-point round-off.  The reference laws
are the classical unilateral Kuhn--Tucker conditions, penalty/augmented
Lagrangian update, Coulomb cone `||t|| <= mu lambda`, and barycentric virtual
work identity.

