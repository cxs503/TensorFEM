# Coupled three-dimensional nonlinear demonstration

## Claim boundary

This demonstrator couples three verified mechanisms in one scalar global
equilibrium problem:

- a spatial inclined bar undergoing finite rotation and finite stretch;
- backward-Euler one-dimensional isotropic-hardening plasticity;
- the three-dimensional node-to-TRI3 contact kernel and its master reaction.

Only global-X motion is free, making every term independently auditable.  It
is not claimed as a general three-dimensional continuum, finite-strain
plasticity, surface-to-surface contact, or arbitrary multi-DOF nonlinear
solver.

## Model and independent reference

The initial bar joins `(0,0,0)` to `(0.8,0.6,0.3)`.  Its moving end approaches
an oriented rigid triangular plane at `x=0.804`.  Parameters are
`A=0.01`, `E=200000`, yield stress `250`, hardening `10000`, contact penalty
`500`, and final X load `8`.

At each displacement `u`, the independent monotonic reference evaluates:

1. exact current length and direction cosine of the inclined spatial bar;
2. the closed-form elastic/bilinear monotonic stress;
3. `k_n max(u-0.004,0)` contact force.

A bracketed bisection solves this scalar force law.  This reference does not
call the incremental return mapping, Newton residual, tangent, or contact
update used by the numerical solution.

## Qualification contracts

- final displacement versus independent reference: relative error `<3%`;
- assembled external/bar/contact equilibrium: relative residual `<3%`;
- plasticity and contact must both be active and the bar must rotate;
- solving `0 -> 0.5 -> 1.0` must reproduce a direct `0 -> 1.0` analysis;
- an injected singular iteration with deliberately corrupt plastic and
  contact trial histories must trigger automatic cutback without committing
  either history;
- exhausted minimum increment must fail closed and leave the input checkpoint
  unchanged.

The executable tests use much tighter tolerances than the 3% qualification
threshold.  Run:

```bash
PYTHONWARNINGS=error PYTHONPATH=src python -m pytest -q \
  tests/test_coupled_3d_demo.py
PYTHONPATH=src python examples/coupled_3d_bar_contact.py
```
