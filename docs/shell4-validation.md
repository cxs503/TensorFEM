# Flat Q4 Mindlin shell validation

`shell4_stiffness` is a four-node **flat-facet** shell with six global DOFs per
node. It combines the existing fully integrated plane-stress Q4 membrane and
selective-reduced-integration Mindlin plate after constructing an orthonormal
element frame. A difference-only drilling penalty removes relative drilling
modes without penalising uniform rigid spin.

## Formal automated benchmarks

| Check | Reference | Relative error / residual | Limit |
|---|---|---:|---:|
| Six rigid-body modes | zero strain energy | < 1e-12 normalised | < 3% |
| Membrane constant-strain patch | `0.5 A t eps^T D eps` | < 1e-12 | < 3% |
| Cylindrical pure bending | `0.5 A D kappa^2` | < 1e-12 | < 3% |
| Arbitrary 3-D rotation | tensor-coordinate invariance | < 1e-12 | < 3% |

The membrane patch uses an affine displacement field. The bending check uses
`w=kappa*x^2/2` and the compatible rotation at the four nodes; for this
rectangular element the interpolated rotation has constant curvature and the
centre shear strain is exactly zero. These are independent analytical energy
comparisons rather than self-comparisons against another FE implementation.

## Explicit scope

This establishes membrane–bending coupling, coordinate transformation, rigid
objectivity, and flat-plate degeneration. It does **not** qualify warped or
curved shell geometry. Scordelis–Lo roof, pinched cylinder and hemispherical
shell benchmarks remain pending a curved-shell formulation and mesh-convergence
study; no accuracy claim is made for them.
