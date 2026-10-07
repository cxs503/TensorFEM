# Coupled elastoplastic bar and unilateral contact validation

## Scope

`coupled_nonlinear.py` assembles two nonlinear mechanisms in one global
residual and tangent: a backward-Euler bilinear elastoplastic bar and a
unilateral penalty rigid stop. Both are solved inside the transactional
adaptive load-step driver. Material state is integrated from the last committed
checkpoint on every Newton/line-search evaluation and is committed only after
equilibrium. Rejected increments therefore cannot contaminate plastic history.

This qualifies **material plasticity plus small-sliding frictionless contact**.
It does not claim simultaneous geometric nonlinearity, finite-sliding contact,
or general three-dimensional coupling.

## Independent benchmark

Parameters are `L=2`, `A=0.01`, `E=200000`, `sigma_y=250`, isotropic hardening
`H=10000`, stop clearance `c=0.004`, penalty `k_c=500`, and final force `F=8`.
The final state is both yielded and in contact.

For monotonic loading, after yield the consistent axial tangent is

`E_t = E H / (E + H)`.

The exact force equilibrium on the active post-yield branch is

`F = A [sigma_y + E_t (u/L - sigma_y/E)] + k_c (u-c)`.

Solving this affine equation gives `u=0.0139130434782609`. The global solver is
compared directly with this expression and must remain below the project-wide
strict 3% gate; the recorded regression is near floating-point precision.

Separate tests force a rejected increment while crossing the yield/contact
kinks and confirm automatic subdivision, immutable virgin state, line-search
compatibility, and fail-closed exhaustion of the minimum increment.

## Run

```bash
PYTHONPATH=src python -m pytest -q tests/test_coupled_nonlinear.py
```
