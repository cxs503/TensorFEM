# B-bar HEX8 verification

The B-bar element uses full 2x2x2 Gauss integration for the deviatoric strain
and replaces only volumetric strain by its element-volume average. It therefore
addresses volumetric locking without the hourglass modes of one-point reduced
integration.

## Evidence

* Affine 3-D constant-strain patch at `nu=0.4999`: energy relative error below
  `1e-10`; this checks consistency in the nearly incompressible limit.
* A 2x2x2 cube under uniform uniaxial traction at `nu=0.4999`, with symmetry
  planes and free lateral contraction, reproduces `u_x=traction/E` to numerical
  precision.
* A free element has exactly six numerical zero eigenvalues (the three
  translations and three rotations); its seventh eigenvalue is positive. This
  is the fail-closed hourglass/rank check.
* A `L/h=10` cantilever at `nu=0.4999`, `E=1e6`, unit end resultant, and the
  same `48x4x4` mesh gives the comparison below. The analytical slender-beam
  reference is `PL^3/(3EI)=0.004`.

| formulation | tip displacement | relative error |
|---|---:|---:|
| standard fully integrated HEX8 | 0.001572526 | 60.6869% |
| B-bar fully integrated HEX8 | 0.003922943 | **1.9264%** |

The standard result is retained as a negative diagnostic. Mesh, material,
loading and constraints are identical, so the improvement comes from the
formulation rather than selecting a different grid. The formal B-bar result is
strictly below the project 3% threshold.

Reference for the method: T. J. R. Hughes, *The Finite Element Method: Linear
Static and Dynamic Finite Element Analysis*, Dover, 2000, Sec. 4.5 (mean
dilatation/B-bar method). The cantilever limit follows Euler-Bernoulli theory.
