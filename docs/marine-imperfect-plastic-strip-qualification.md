# Imperfect plastic plated-strip qualification

`marine_imperfect_strip` is a deliberately limited verification model for
auditing the interaction of an initial local imperfection, welding-like
self-equilibrated residual stress and bilinear material plasticity. It is a
one-mode strip reduction with midpoint fibres across the width. It is **not**
a general shell finite element, does not represent stiffener/tripping modes,
and must not be used as a ship-collapse or class-rule result. TensorLBM is not
used.

The assumed buckle shape is sinusoidal. Its squared slope adds membrane
shortening at every fibre. The generalized mode equilibrium is

`k_b (q-q0) - N(q) q = 0`.

Each fibre carries the initial field `sigma_r cos(2 pi x)`, whose midpoint
sum is zero for the required even fibre count. Subsequent states use an exact
uniaxial bilinear return map with isotropic hardening. Each returned
`StripState` records the committed load step, amplitude, membrane force,
yield residual, generalized equilibrium residual, work, stored energy,
plastic dissipation, energy residual and yielded fraction.

## Verification evidence

The fully elastic response is checked against an independent continuum cubic
obtained by analytically integrating the squared sine-mode slope. The plastic
response is checked against a 1,280-step, 512-fibre reference. Production
qualification resolutions are 80/32, 160/64 and 320/128 (steps/fibres); their
combined response errors must decrease monotonically and the fine error must
be below 3%. The same gate requires equilibrium below `1e-10`, yield-surface
violation below `1e-12`, and energy residual below 3% of external work.

```python
from tensorfem.marine_imperfect_strip import run_imperfect_strip_qualification

report = run_imperfect_strip_qualification()
assert report["passed"]
```

Inputs reject booleans masquerading as counts, non-finite/non-positive
properties, odd fibre counts (which would spoil exact residual-stress
balance), and residual stress at or above initial yield.
