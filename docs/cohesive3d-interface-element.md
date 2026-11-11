# 3-D zero-thickness cohesive interface element: formulation and verification

Module: `src/tensorfem/cohesive3d.py`
Tests: `tests/test_cohesive3d.py`
Verification tables: `benchmarks/cohesive3d_verification_tables.py` →
`benchmarks/cohesive3d_verification_tables.{json,md}`
Parameter set used everywhere: structural adhesive, `k_n = 1e12 Pa/m`,
`k_s = 4e11 Pa/m`, `t_n0 = 5e6 Pa`, `t_s0 = 3e6 Pa`, `G_Ic = 500 J/m^2`,
`G_IIc = 1200 J/m^2`, BK exponent `eta = 1.6`.

## 1. What the element is

An 8-node, zero-thickness cohesive element with Abaqus `COH3D8` topology: nodes
0–3 on one face, nodes 4–7 on the opposite face, 2×2 Gauss quadrature on the
mid-surface.  The interface is pre-meshed (standard cohesive practice): the
element does not search for the crack path.

Public API:

| symbol | purpose |
|---|---|
| `BKCohesive3DLaw` | irreversible bilinear law on `(delta_n, delta_s1, delta_s2)` with BK mixed-mode propagation, quadratic nominal-stress initiation, Macaulay-bracketed damage driving, secant unloading and an analytic consistent tangent |
| `cohesive_separations(coords, u, kinematics=...)` | per-Gauss-point local separation `(4, 3)` for either kinematics |
| `cohesive_hex8_response(coords, u, law, history, kinematics=...)` | internal force `(8, 3)`, tangent `(24, 24)`, updated history `(4,)`, diagnostics |
| `assemble_cohesive_interface(coords, conn, u, law, histories, kinematics=...)` | dense assembly of a pre-meshed interface patch |
| `interface_area`, `mid_surface_frame`, `cohesive_shape_functions` | geometry helpers |

Node ordering of the mid-surface must be counter-clockwise seen from the
positive normal side: `xi_i = (-1, 1, 1, -1)`, `eta_i = (-1, -1, 1, 1)`.

## 2. Two kinematics

### 2.1 `kinematics="small_strain"` (default, unchanged)

The local frame `R_ref = (n, s1, s2)` is built from the **reference**
mid-surface and the separation is the frame-projected displacement jump

```
delta = sum_a N_a [ R_ref (u_a^top - u_a^bot) ]        a = 1..4
```

Force `f = int B^T t dA` and analytic tangent `K = int B^T D B dA`.

**Scope:** valid only for infinitesimal interface rotation.  A rigid rotation of
the *body* in a fixed frame is not a change of frame; the reference-frame
separation then sees the rotation-induced shear `k_s * g * theta` and reports a
spurious traction that grows without bound (measured, see the KNOWN_LIMIT row of
the verification table: `max |f| / (t_n0 A) = 2.2e2` at `theta = 0.3 rad` and
`9.9e3` at `theta = 3.0 rad`).  Rotating the coordinates *and* the
displacements together is a genuine change of frame and is reproduced exactly
(`test_rigid_rotation_frame_indifference`) — it does **not** demonstrate
physical objectivity, which is why the corotational formulation exists.

### 2.2 `kinematics="corotational"`

The frame `R_cur` and the mid-surface Jacobian are rebuilt from the **current**
configuration `x = X + u`, and the separation is the gap between the
frame-projected current gap vector and the frame-projected reference gap vector:

```
delta = sum_a N_a [ R_cur (x_a^top - x_a^bot) - R_ref (X_a^top - X_a^bot) ]
```

Under a rigid rotation `x_top - x_bot = R (X_top - X_bot)` and
`R_cur = R_ref R^T`, so both terms coincide and `delta = 0` identically.  Under a
rigid translation both gaps are unchanged.  This is the classical
corotational (updated-Lagrangian-frame) cohesive kinematics: the interface is
objective at any rotation angle while the *adherends* still use the small-strain
kinematics that the host solids provide.

Integration is over the **reference** area (`w * jac_ref`), so the interface
neither gains nor loses area and the formulation stays conservative in the
reference configuration; area change of the interface itself is neglected.

#### Exact consistent tangent without nested differentiation

With `B = d(delta)/du`, the internal force is `f = sum A B^T t`.  Its derivative splits
into a material and a geometric part,

```
K = sum_gp A_gp B^T D B  +  d/du ( sum_gp A_gp B^T t ) |_{t const}
  = sum_gp A_gp B^T D B  +  Hess_u ( sum_gp A_gp delta_gp . t_gp )
```

because `sum A B^T t` is the gradient of the scalar `sum A delta . t` for frozen
`t`.  The Hessian carries every geometric term (the sensitivity of the local
frame to the nodal displacements, i.e. exactly what a frame-frozen tangent
misses).  Cost is one 24-wide Jacobian of the separation plus one Hessian of a
scalar per element (~0.15 s in float64 with `vectorize=True`); a naively nested
Jacobian of the force costs ~5 s per element for the same result, which is why
the potential/Hessian split is used.

Consequences pinned by tests:

* elastic regime ⇒ the tangent is symmetric to `1.4e-16` (Hessian of a scalar);
* softening branch ⇒ the tangent is intentionally **unsymmetric** for
  `k_n != k_s` (the law is then not a gradient field) and matches central
  differences of the force to `3.0e-08`.

## 3. Verification table (generated, `benchmarks/cohesive3d_verification_tables.md`)

| check | oracle (closed form) | measured | tolerance | status |
|---|---|---|---|---|
| `law_reduces_to_2d_bk` | 3-D law == merged 2-D BK law on `delta_s2 = 0` | 2.1e-16 | 1e-12 | PASS |
| `law_energy_mode_I` | `int t . d(delta) == G_Ic` | 500.000000 vs 500.000000 J/m² (3.4e-16) | 1e-5 | PASS |
| `law_energy_mode_II` | `int t . d(delta) == G_IIc` | 1199.999919 vs 1200.000000 (6.7e-08) | 1e-5 | PASS |
| `law_energy_mode_III` | mode III behaves as mode II | 1199.999919 vs 1200.000000 (6.7e-08) | 1e-5 | PASS |
| `law_energy_mixed_45` | `int t . d(delta) == G_c(BK)` | 594.316809 vs 594.316824 (2.6e-08) | 1e-5 | PASS |
| `law_initiation_quadratic` | `(t_n/t_n0)^2 + (t_s/t_s0)^2 == 1` at onset | 4.4e-16 | 1e-12 | PASS |
| `law_tangent_vs_ad` | analytic `dt/d(delta)` == autograd Jacobian of the traction | 5.5e-16 | 1e-10 | PASS |
| `law_batch_matches_pointwise` | batched update (16 softening points) == 16 independent point calls | max relative difference 0.0e+00 at damage 0.991 | 1e-12 | PASS |
| `element_patch_small_strain` | uniform opening: reaction == `t * A` | 4615384.615385 N vs 4615384.615385 N (0.0) | 1e-12 | PASS |
| `element_rigid_rotation_small_strain` | rigid rotation in a fixed frame ⇒ zero traction | `max |f|/(t_n0 A) = 9.9e3` | — | **KNOWN_LIMIT** |
| `element_rigid_rotation_corotational` | rigid rotation in a fixed frame ⇒ zero traction | 2.9e-12 over `theta = 0.01 … 3.0 rad` | 1e-10 | PASS |
| `element_rotation_objectivity_corotational` | `f(RX, Ru) == R f(X, u)` | 1.1e-11 | 1e-9 | PASS |
| `corotational_elastic_tangent_symmetry` | Hessian of a potential ⇒ symmetric | 1.4e-16 | 1e-12 | PASS |
| `corotational_tangent_vs_finite_difference` | `df/du` == central differences | 3.0e-08 at damage 0.762 | 1e-6 | PASS |
| `corotational_dissipation_rotation_invariance` | dissipated work independent of rigid rotation | 7.9e-14 over `theta = 0, 0.4, 1.2 rad` | 1e-12 | PASS |
| `corotational_dissipation_vs_fracture_energy` | dissipated work == `G_Ic * A` | 498.8077 J vs 500 J (2.4e-03, 40-step trapezoid) | 5e-3 | PASS |
| `small_strain_dissipation_at_rotation` | same opening at `theta = 0.4 rad`, reference frame | −1559043 J vs 500 J (3.1e+03) | — | **KNOWN_LIMIT** |
| `corotational_mesh_objectivity` | 1×1, 2×2, 4×4 patches ⇒ same reaction | 1.0e-12 | 1e-11 | PASS |
| `corotational_assembly_consistency` | assembly == sum of element contributions | 0.0 (force and tangent) | 1e-12 | PASS |
| `corotational_compression_undamaged` | closing uses `k_n`, never damages | −9999.999995 N vs −10000 N (5.3e-10) | 1e-8 | PASS |

18 PASS, 0 FAIL, 2 KNOWN_LIMIT.  Regenerate with
`PYTHONPATH=src python benchmarks/cohesive3d_verification_tables.py`; the JSON
carries the raw values, the Markdown table is the one pasted above.

## 4. Pitfalls (measured while building this)

1. **"Rotate coordinates and displacements together" is not an objectivity
   test.**  It is a change of frame that any reference-frame formulation
   reproduces exactly.  The physical test is a rigid rotation expressed purely in
   displacements, which the `small_strain` path fails by three orders of
   magnitude at 0.3 rad.  The verification table keeps both rows so the
   limitation cannot silently come back.
2. **Frame ordering.**  The law is written in `(n, s1, s2)` order; stacking the
   rotation as `(e1, e2, n)` silently swaps normal and shear.  The triad rows
   must be `(n, e1, e2)` — a cyclic (det = +1) permutation.
3. **Nested differentiation is affordable only through the potential.**  A
   Jacobian *of the force* (whose own definition contains a Jacobian) needs a
   nested graph and costs ~35× more than the Hessian-of-a-scalar identity used
   in the implementation.
4. **Precision floor of the corotational separation.**  The measure subtracts two
   frame-projected gaps of order `gap = 0.1 m` to produce openings of order
   `1e-8 … 1e-6 m`, so ~7 decimal digits are lost before the law is evaluated.
   The force is consequently reproduced to ~`5e-10` relative (see the
   compression row), not to machine precision.  If sub-nanometre openings ever
   matter, move the reference gap into the material description instead of
   relying on cancellation.
5. **Post-failure singularity.**  After the interface fully fails the upper
   adherend keeps an unconstrained rigid translation and the Newton operator
   becomes singular.  The benchmark driver adds a `1e-12` relative diagonal
   regularization; this is a property of the *specimen*, not of the element.
6. **Compression never damages.**  Closing openings use the penalty stiffness
   `k_n` and leave the history variable untouched; friction and contact
   degradation are out of scope.

7. **The solver never calls the law point by point.**  `cohesive_hex8_response`
   evaluates all `4 * n_elements` Gauss points of the patch in one batched call,
   so a bug in *batch* broadcasting is a bug in the only path the solver uses.
   One was present in the BK mixing weight (`frac ** (eta-1)` and both `dF`
   denominators must be broadcast with `[..., None]`): with exactly three points
   in the batch the trailing axis aliased the *component* axis and each point
   picked up its neighbour's weight, with any other batch size the update raised
   `RuntimeError: The size of tensor a (16) must match the size of tensor b (3)`.
   Every single-point check passed throughout.  `law_batch_matches_pointwise`
   (verification table) and `test_batched_update_matches_pointwise_in_softening`
   / `test_vectorised_assembly_with_damage_matches_production_element` now pin the
   batch path itself, with finite differences -- not the pointwise path -- as the
   oracle for the tangent.
8. **The interface patch is conforming.**  Adjacent `COH3D8` elements share the
   four nodes of their common in-plane edge, so a *global* force or tangent entry
   is the sum over every incident element.  Comparing one element against the
   global vector therefore disagrees at the shared nodes even when the assembly is
   exact; the reference must be assembled over the whole patch
   (`tests/test_cohesive3d_dcb.py`), which is also what
   `corotational_assembly_consistency` does for the explicit assembly path.

## 5. Scope limits (not claimed)

* Small-strain adherends: no geometric (initial-stress) stiffness of the *host
  solid*; only the interface rotation is treated exactly (corotational).
* No friction, no contact damage, no cyclic degradation.
* No mixed-mode **non-proportional** path validation: the law is a total
  (history-of-maximum-separation) formulation, so a load path that reverses the
  mode mix is outside the verified envelope.
* Prescribed crack path (pre-meshed interface); no crack-path search, no
  remeshing, no adaptive insertion.
* Rate and temperature dependence are not modelled.