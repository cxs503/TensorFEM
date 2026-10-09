# Marine local-plate buckling validation

This TensorFEM-only qualification compares a numerical finite-difference
eigenvalue with the independent continuous Navier solution for a simply
supported rectangular orthotropic plate under uniform uniaxial compression.
Line load uses N/m, stress uses Pa, dimensions use m, and bending rigidities use
N m. TensorLBM and CFD are neither imported nor executed.

For half-wave integers `m,n`, the analytical reference is

`Nx = [Dx alpha^4 + 2 H alpha^2 beta^2 + Dy beta^4] / alpha^2`,

where `alpha=m*pi/a`, `beta=n*pi/b`, and `H=D12+2D66`. The numerical solver
instead uses the eigenvalues of second-order central-difference operators on
the interior grid. It searches every resolvable half-wave pair, so aspect-ratio
and orthotropic mode changes are tested rather than prescribed.

The formal suite contains square and 2:1 isotropic plates plus a 2:1 panel with
smeared longitudinal stiffeners. Grids of 8x8, 16x16 and 32x32 must converge
monotonically; every final result must have relative critical-load error below
3%. The stiffener model adds `Es As z^2/s` to `Dx`. It is an equivalent
orthotropic idealization, not a discrete plate/stiffener junction model.

This evidence qualifies critical elastic local-buckling loads and mode
selection only for simply supported plates with uniform compression. It does
not qualify shell-element geometric stiffness, initial imperfections, welding
residual stress, discrete stiffener tripping, nonlinear postbuckling, or
ultimate strength.

Run:

```bash
PYTHONPATH=src python -m pytest -q tests/test_marine_plate_buckling.py
PYTHONPATH=src python examples/marine/local_plate_buckling.py
```
