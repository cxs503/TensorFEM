# TET4 / TET10 h-convergence and GCI evidence

`docs/h-convergence-gci-verification.md` §6 records that the discrete-uncertainty
evidence set covered `hex8` only and that `tet4`, TET10, shells and plates were
unmeasured. This note closes the tetrahedral half of that gap: it measures the
observed order of accuracy and the Roache grid-convergence-index (GCI) band of
both tetrahedral families against two *independent* references.

## 1. Cases and oracles

| item | value |
|---|---|
| box | `L=2.0`, `w=0.4`, `h=0.6` |
| material | `E=210e9`, `nu=0.3` |
| mesh family | `(2n, n, 3)` box, six TET4 per hex, `n = 2, 4, 8` |
| TET10 mesh | conforming upgrade of the same TET4 mesh (midside nodes shared between elements) |
| manufactured field | `u_i = a cos/sin(kx)cos/sin(ly)cos/sin(mz)`, `a=1e-4`, `planar=True` (z-independent) |
| body force | `b = -div(sigma)` by automatic differentiation of the analytic field |
| pure bending | `u_x=-k x yc`, `u_y=k/2(x^2+nu yc^2-nu zc^2)`, `u_z=k nu yc zc`, `k=1e-3` |

The consistent nodal loads are integrated with a collapsed (Duffy) Gauss rule on
each tetrahedron (`tensorfem.tet_convergence.tet_consistent_body_force`): only
affine corner geometry and shape-function values enter, never the `B` matrix or
the stiffness of the element under test. The rule reproduces the reference
moments `int 1 = 1/6`, `int r = 1/24`, `int r^2 = 1/60`, `int r s = 1/120` to
1e-12. The strain-energy reference is the analytic `int sigma:eps/2 dV`
(16978.9999 J), cross-checked at a higher rule order (16979.5256 J, 0.0031%).

Reproduce with `python benchmarks/tet_h_convergence_gci.py`; machine-readable
evidence and the figure are in `docs/assets/tet-h-convergence-gci/`.

## 2. Measured convergence (manufactured solution)

| family | mesh | free DOF | rel. L2 displacement error | rel. strain-energy error |
|---|---|---:|---:|---:|
| TET4 | 4×2×3 | 18 | 5.5657e-03 | 2.3210e-01 |
| TET4 | 8×4×3 | 126 | 2.4850e-03 | 6.3819e-02 |
| TET4 | 16×8×3 | 630 | 6.9102e-04 | 1.6371e-02 |
| TET10 | 4×2×3 | 315 | 1.4113e-02 | 1.9746e-02 |
| TET10 | 8×4×3 | 1575 | 2.7490e-03 | 2.3041e-03 |
| TET10 | 16×8×3 | 6975 | 4.4370e-04 | 3.0440e-04 |

| family | L2 order (coarse→fine) | energy order (coarse→fine) | theory |
|---|---|---|---|
| TET4 | 1.163, 1.846 | 1.863, 1.963 | 2 (L2), 2 (energy) |
| TET10 | 2.360, 2.631 | 3.099, 2.920 | 3 (L2), 3-4 (energy) |

## 3. GCI band against the analytic energy

| family | Richardson extrapolation | analytic | deviation | observed order | relative GCI |
|---|---:|---:|---:|---:|---:|
| TET4 | 17017.37 | 16979.00 | +0.226% | 1.826 | 2.368% |
| TET10 | 16978.23 | 16979.00 | -0.005% | 3.125 | 0.032% |

Both studies pass the fail-closed evaluator
(`tensorfem.convergence_verification.evaluate_convergence_study`) with the
analytic energy as the reference: the finest-mesh energy error is 1.637% (TET4)
and 0.030% (TET10), and the GCI band contains the analytic value in both cases.

## 4. Pure-bending Dirichlet consistency

| family | rel. L2 displacement error (n=2,4,8) | rel. strain-energy error |
|---|---|---|
| TET4 | 7.49e-17, 3.46e-16, 1.47e-15 | 2.593, 0.6677, 0.1864 |
| TET10 | 2.93e-16, 1.11e-15, 2.85e-15 | 1.7e-12, 5.1e-12, 2.7e-11 |

Both families reproduce the quadratic bending field *nodally* on this structured
split, so the displacement comparison is at roundoff and carries no order — the
same superconvergence the HEX8 note reports. TET4's constant-strain energy is
nevertheless 259% low at the coarsest mesh, decaying at roughly second order,
while TET10's energy is at roundoff. Nodal exactness must not be quoted as
bending accuracy: for TET4 the energy, not the nodal displacement, exposes the
missing transverse resolution.

## 5. Notable traps

* **A mesh with no free DOF reports a perfect L2 error.** With one element
  through the thickness every vertex is a boundary node, the "solution" equals
  the nodal interpolant and `relative_l2_displacement_error` is exactly 0.0 for
  any element type. The study now fails closed (`no free DOF`) instead of
  publishing that artifact; three layers are the minimum for this mesh family.
  The TET10 upgrade is unaffected because midside nodes on the hex main
  diagonals still lie inside the volume.
* **Pre-asymptotic rates.** At `n=2,4` TET4 shows L2 order 1.16 and TET10 2.36;
  only the finest pair approaches the theoretical 2 and 3. Reporting a single
  coarse triple would under-state both families.
* **The manufactured body force of this field is self-equilibrated**, so its
  integral over the box vanishes and only the first moment of the load vector
  carries information. The load-vector test checks `sum_a x_a (x) f_a` against an
  independent tensor-product Gauss rule, not the (vanishing) resultant.

## 6. Scope and limitations

* Verification, not validation: agreement proves the discrete operator is
  consistent and converges; it says nothing about tetrahedral suitability for
  bending-dominated shells (see `sphere-membrane-validation.md`, where curved
  Shell4 fails the 3% membrane gate), volumetric locking, or contact.
* The field is z-independent (`planar=True`) and the mesh keeps three layers
  through the thickness, so the study exercises a 2-D-like state in a 3-D mesh;
  a fully three-dimensional TET10 sequence was not run because the dense solver
  would exceed the available budget.
* Structured six-tet splits are used. Unstructured/centroid splits change the
  constant in the error but not the order; that was not measured here.
* The energy order of TET10 (3.10, 2.92) is below the 3-4 asymptote, so no
  claim is made about the asymptotic energy rate of the quadratic tet beyond
  what the three meshes show.
