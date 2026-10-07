# Stiffener instability and local-pit screening

Two independently auditable screening cases extend the marine verification
set without claiming a complete collapse analysis.

## Lateral-torsional bifurcation

A simply supported, doubly symmetric line stiffener under uniform moment uses
the classical elastic lateral-torsional buckling reference

```text
Mcr = (pi/L) sqrt(E Iy GJ) sqrt(1 + pi^2 E Iw/(L^2 GJ)).
```

The numerical result uses the lowest eigenvalue of a centred-difference
Dirichlet Laplacian in place of `(pi/L)^2`. The 4/8/16/32-segment sequence must
converge monotonically and finish below 3% relative error.

## Sharp circular pit

A constant-curvature coupon has `D=D0` outside one circular pit and
`D=(1-loss)D0` inside it. Cell-centre material tagging is compared with the
independent exact pit area `pi*r^2`. Because a discontinuous boundary can
produce non-monotone cell-count oscillations, qualification requires both the
64 and 128 grids to remain below 3%, rather than hiding the 16/32-grid errors.
The equivalent pit thickness ratio is `(1-loss)^(1/3)`.

Run:

```bash
PYTHONPATH=src python -m pytest -q tests/test_marine_stiffener_failure.py
PYTHONPATH=src python examples/marine/stiffener_failure_screening.py
```

## Claim boundary

These cases qualify an ideal isolated stiffener's linear elastic uniform-
moment bifurcation and constant-curvature sensitivity to one sharp circular
rigidity-loss patch. They do not qualify plate--stiffener interaction, weld
failure, eccentric monosymmetric sections, nonlinear tripping, crack
initiation, postbuckling, load redistribution, or complete ultimate collapse.
