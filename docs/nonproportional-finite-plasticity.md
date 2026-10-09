# Non-proportional finite-plastic integration

The qualified path first applies finite simple shear and then a rotated
isochoric principal stretch. Each total-deformation segment is integrated by a
transactional full-step/two-half-step estimator. State error in `Fp` and
`alpha` controls recursive subdivision; exhausting the depth limit fails
without modifying the caller's state. A 1,200-uniform-substep implementation is
used as an independent reference and both stress and `Fp` errors are below 3%.

The algorithmic `dP/dF` is computed with centred differences at `h` and `h/2`
and fourth-order Richardson extrapolation. It is checked against a third step
size and drives a material Newton solve. This is more accurate and stable than
the earlier single-step centred tangent, but remains numerical rather than a
closed-form implicit derivative.

Tests cover superposed-rotation objectivity, yield consistency, plastic volume,
nonnegative dissipation, loading-order dependence, residual stress after a
closed total-deformation cycle, split-run restart and transactional failure.
These results qualify only the stated shear/rotated-stretch history; they do not
generalize automatically to arbitrary non-coaxial cyclic histories.
