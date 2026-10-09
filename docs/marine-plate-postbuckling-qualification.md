# Prestressed plate and imperfect postbuckling qualification

This TensorFEM-only qualification contains two deliberately narrow models.

The linear model solves the generalized buckling equation for a simply
supported orthotropic plate under prescribed uniform compressive line loads
`Nx` and `Ny`. Its central-difference eigenvalues on 8, 16, and 32 interior-node
grids are compared with an independently evaluated continuous Navier series.
Both uniaxial and biaxial prestress cases must converge monotonically, and the
32-grid error must be below 3%.

The nonlinear model uses one sine mode and the nondimensional potential

```text
V = (q-w0)^2/2 - lambda*q^2/2 + c*(q^2-w0^2)^2/4
```

where `w0` is the stress-free initial modal imperfection, `q` is total modal
amplitude, and `c` is a positive membrane-stiffening coefficient. This is the
standard pitchfork structure of a one-mode von Karman--Koiter reduction. The
positive equilibrium branch is integrated by second-order load continuation
with 40, 80, and 160 steps. An independent bracketed root solution of
`dV/dq=0` is the oracle. Maximum path error must decrease monotonically and be
below 3%.

This evidence validates the stated classical-plate eigenproblem and the
auditable reduced-order imperfect path only. It is **not** a general shell
finite element, does not compute membrane prestress from an in-plane FE solve,
and does not qualify multi-mode interaction, plasticity, residual stress,
stiffener tripping, snap-through, arc-length continuation, or collapse. It
does not use TensorLBM or CFD.

Run the evidence with:

```bash
PYTHONPATH=src python -m pytest -q tests/test_marine_plate_postbuckling.py
PYTHONPATH=src python examples/marine/plate_postbuckling.py
```
