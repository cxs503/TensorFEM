# Beam, stability, and transient benchmarks

Acceptance is strict relative error below 3%.

## Timoshenko cantilever

A rectangular cantilever is loaded by a transverse tip force. Both a slender
beam (`L/h=100`) and a deep beam (`L/h=2`) are checked using 1, 2, and 4
elements against

`delta = P L^3/(3 E I) + P L/(kappa G A)`.

This is the classical Timoshenko result (bending plus shear deformation). The
closed-form shear-flexible element stiffness is locking-free; equilibrium and
mesh invariance are also tested.

## Euler columns

Pinned-pinned, fixed-free, fixed-pinned, and fixed-fixed columns are checked
against `Pcr = pi^2 E I/(K L)^2`, with effective-length factors 1, 2,
0.6991556596, and 0.5. Four- and sixteen-element results establish convergence.

## Newmark transient

An undamped SDOF oscillator is integrated for five cycles with the
average-acceleration method (`beta=1/4`, `gamma=1/2`). Displacement is compared
with `u(t)=u0 cos(omega t)`, and total mechanical energy drift is tested.

References: S. Timoshenko, *Strength of Materials*, beam deflection and column
chapters; N. M. Newmark, “A Method of Computation for Structural Dynamics,”
ASCE Journal of the Engineering Mechanics Division, 1959.
