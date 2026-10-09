# Two-dimensional continuum validation

The continuum module implements small-strain isotropic CST and fully integrated
Q4 elements for plane stress and plane strain. Q4 uses the standard 2x2 Gauss
rule. Element matrices are assembled with differentiable PyTorch operations.

The automated validation suite contains:

1. A constant-strain Q4 patch test with an affine prescribed displacement field.
   The displacement relative error tolerance is `1e-11`.
2. A slender cantilever benchmark against the Euler--Bernoulli analytical tip
   displacement, using a 40x8 mesh. The acceptance threshold is **3 percent**.
3. A CST constant uniaxial stress test against the analytical plane-stress state.

The cantilever test deliberately reports the centre node of the loaded edge and
uses consistent nodal forces. It is a convergence/verification case, not a claim
that Euler--Bernoulli theory is an exact two-dimensional elasticity solution.
