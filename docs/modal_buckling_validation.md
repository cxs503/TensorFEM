# Modal and linear buckling validation

The implementation uses two-node Euler--Bernoulli bending elements with two
degrees of freedom per node (`w`, `theta`). Double precision is used for the
eigenvalue transformations.

## Cantilever vibration

The reference solution is

`omega_1 = 1.875104^2 sqrt(EI / (rho A L^4))`.

An eight-element mesh with a consistent mass matrix is tested against this
value. The automated acceptance threshold is a relative error below 3%.

## Pin-ended Euler column

The reference critical load is

`P_cr,n = (n pi)^2 EI / L^2`.

The geometric stiffness corresponds to a unit uniform compressive force. An
eight-element mesh is used for the first-mode acceptance test; a sixteen-element
mesh checks the first three modes. Every tested load must be within 3% of the
analytical value.

Run only these benchmarks with:

```bash
PYTHONPATH=src python -m pytest -q tests/test_modal_buckling.py
PYTHONPATH=src python examples/modal_buckling_benchmarks.py
```
