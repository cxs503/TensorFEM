# Discrete stiffener and nonuniform-corrosion qualification

This qualification adds two deliberately bounded marine-panel screening cases.

The stiffened case retains each longitudinal stiffener as an individual
Euler--Bernoulli line member at an explicit attachment coordinate. Plate and
beam share the transverse sinusoidal displacement at that line. Its critical
line load is checked against the exact identity
`sum(sin(j*pi/(n+1))^2)=(n+1)/2`; this verifies the minimal plate--beam
compatibility and stiffness assembly without replacing the beams by a smeared
orthotropic rigidity.

The corrosion case uses the smooth field
`t=t0*(1-eta*sin²(pi*x/a)*sin²(pi*y/b))^(1/3)`. Consequently its local bending
rigidity is linear in the prescribed field and the sine-mode Ritz numerator
has an analytic integral. A 2/3/6/12 midpoint sequence demonstrates an
under-resolved first grid followed by exact integration of the finite Fourier
content to roundoff, comfortably below the 3% relative-error gate.

Run:

```bash
PYTHONPATH=src python -m pytest -q tests/test_marine_discrete_panel.py
PYTHONPATH=src python examples/marine/discrete_stiffener_corrosion.py
```

## Claim boundary

This qualifies linear compatible transverse plate/line-beam bending and smooth
nonuniform-thickness sensitivity only. It does **not** qualify stiffener
tripping or torsion, weld flexibility/failure, isolated-pit resolution,
material yielding, nonlinear postbuckling, or ultimate load.
