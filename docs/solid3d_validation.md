# TET4 and HEX8 validation

The `solid3d` module implements small-strain isotropic TET4 and 2x2x2
Gauss-integrated HEX8 elements in double-precision PyTorch.

The automated benchmarks are:

- A three-dimensional affine patch on a 2x2x2 mesh with a perturbed interior
  node. TET4 and distorted HEX8 reproduce all normal and shear modes with
  relative displacement error below `1e-11`.
- Analytical uniaxial tension, using `ux=sigma*x/E`, `uy=-nu*sigma*y/E`, and
  `uz=-nu*sigma*z/E`. Recovered stress `[sigma,0,0,0,0,0]` and resultant
  `sigma*A` are checked below `1e-11` relative error.
- An autograd check from strain energy to Young's modulus and rejection of
  degenerate/non-positive Jacobians.

These exact linear-elastic references are substantially stricter than the
project-wide 3% acceptance gate. A non-affine 3-D bending convergence benchmark
is intentionally tracked as the next validation batch rather than conflated
with the element patch test.
