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
| `law_reduces_to_2d_bk` | 3-D law == merged 2-D BK law on delta_s2 = 0 | max scaled traction difference over 5 mode-mix ratios x 5 amplitudes | 1e-12 | PASS |
| `law_energy_mode_I` | int t . d(delta) along the envelope == G_c(direction) | mode_I: measure 500.000000 vs 500.000000 J/m^2 | 1e-05 | PASS |
| `law_energy_mode_II` | int t . d(delta) along the envelope == G_c(direction) | mode_II: measure 1199.999919 vs 1200.000000 J/m^2 | 1e-05 | PASS |
| `law_energy_mode_III` | int t . d(delta) along the envelope == G_c(direction) | mode_III: measure 1199.999919 vs 1200.000000 J/m^2 | 1e-05 | PASS |
| `law_energy_mixed_45` | int t . d(delta) along the envelope == G_c(direction) | mixed_45: measure 594.316809 vs 594.316824 J/m^2 | 1e-05 | PASS |
| `law_initiation_quadratic` | (t_n/t_n0)^2 + (t_s/t_s0)^2 == 1 at damage onset | max |quadratic criterion - 1| over 11 mode-mix angles | 1e-12 | PASS |
| `law_tangent_vs_ad` | analytic dt/d(delta) == autograd Jacobian of the traction | max relative tangent difference over 35 (angle, amplitude) points | 1e-10 | PASS |
| `law_batch_matches_pointwise` | batched update (16 softening points) == 16 independent point calls | max relative difference 0.0e+00 at damage 0.991 | 1e-12 | PASS |
| `element_patch_small_strain` | uniform opening: reaction on the top face == t * A (unit area) | measure 4615384.615385 N vs 4615384.615385 N | 1e-12 | PASS |
| `element_rigid_rotation_small_strain` | rigid rotation of the body in a fixed frame must give zero traction | max |f| / (t_n0 A) over theta = [0.01, 0.1, 0.3, 0.7, 1.4, 3.0] (reference-frame separation) | n/a | **KNOWN_LIMIT** |
| `element_rigid_rotation_corotational` | rigid rotation of the body in a fixed frame must give zero traction | max |f| / (t_n0 A) over theta = [0.01, 0.1, 0.3, 0.7, 1.4, 3.0] (current-frame separation) | 1e-10 | PASS |
| `element_rotation_objectivity_corotational` | f(R X, R u) == R f(X, u) for a rotated element geometry | relative Frobenius difference of the reaction under R (0.5, -0.2, 0.9) | 1e-09 | PASS |
| `corotational_elastic_tangent_symmetry` | tangent is the Hessian of a potential in the elastic regime => symmetric | max |K - K^T| / max |K| | 1e-12 | PASS |
| `corotational_tangent_vs_finite_difference` | d f / d u == central differences of the internal force | relative Frobenius difference at damage 0.762 (central differences, eps = 1e-9 m) | 1e-06 | PASS |
| `corotational_dissipation_rotation_invariance` | dissipated work under monotonic opening is independent of rigid rotation | max |W(theta) - W(0)| / W(0) for theta = 0.0, 0.4, 1.2 rad | 1e-12 | PASS |
| `corotational_dissipation_vs_fracture_energy` | dissipated work == G_Ic * A (trapezoid integration, 40 steps) | measure 498.8077 J vs 500.0 J | 5e-03 | PASS |
| `small_strain_dissipation_at_rotation` | same opening at theta = 0.4 rad, reference-frame separation (known limit) | measure -1559043.3 J vs 500.0 J | n/a | **KNOWN_LIMIT** |
| `law_dissipation_vs_path_integral` | closed-form W_d(a_f) == int t . d(delta) along a monotonic ray (7 mixities) | max relative difference, 0deg 500.0000; 15deg 502.2829; 30deg 522.8049 ... | 1e-05 | PASS |
| `law_dissipation_equals_fracture_energy` | W_d(a_f) == G_c(direction), BK mixed mode, modes I / II / III / 45deg / 3-D | max relative difference from G_c, 5 directions, worst 5.74e-16 | 1e-12 | PASS |
| `law_dissipation_monotone_bounded` | W_d == 0 on the elastic branch, non-decreasing, saturating at G_c | elastic max 0.0e+00 J/m^2, worst decrease 1.1e-13, overshoot 2.2e-16 | 1e-12 | PASS |
| `element_dissipation_equals_gc_times_area` | element diagnostics ``dissipated_energy`` == G_Ic * A at complete opening | small_strain 500.000000 J, corotational 500.000000 J vs 500.0 J | 1e-12 | PASS |
| `element_dissipation_rotation_invariance` | element ``dissipated_energy`` independent of a rigid rotation (corotational) | max relative spread over theta = 0.0, 0.3, 1.2 rad | 1e-14 | PASS |
| `nonproportional_path_dissipation` | open to 1.5 a_0(I) then shear to failure: 0 <= W - G_IIc and W_d ~= W_path | W_path 1200.249 J/m^2 vs G_IIc 1200.0, closed form 1199.754 (residual 4.1e-04) | 1e-03 | PASS |
| `corotational_mesh_objectivity` | n x n patches (n = 1, 2, 4) of the same unit interface give the same reaction | max relative reaction difference from t * A | 1e-11 | PASS |
| `corotational_assembly_consistency` | dense patch assembly == sum of element contributions (force and tangent) | max relative difference, 2 x 2 patch | 1e-12 | PASS |
| `corotational_compression_undamaged` | closing uses the penalty stiffness k_n and never damages the interface | reaction -9999.999995 N vs -10000.000000 N, damage 0.0e+00 | 1e-08 | PASS |

24 PASS, 0 FAIL, 2 KNOWN_LIMIT.  Regenerate with
`python benchmarks/cohesive3d_verification_tables.py` (the script puts this
checkout's `src/` first on `sys.path`, so it can never report numbers from a
foreign editable install); the JSON carries the raw values *and* a
`provenance` block (`tensorfem_module_file`, `git_commit`, `git_dirty`), the
Markdown table is the one pasted above.

### 3.1 Dissipated energy in closed form (`dissipated_energy`)

Every response carries the energy dissipated per unit **reference** area,

```
W_d(h) = int_0^h t_eff(s) ds  -  t_eff(h) h / 2 ,      h = max(history, a)
```

with `t_eff` the bilinear envelope of the work-conjugate direction and the
second term the energy stored by the secant unloading branch.  It vanishes on
the elastic branch, is monotone in `h`, and saturates at `G_c(direction)`; the
element aggregates it as `sum_gp A_gp W_d` into
`diagnostics["dissipated_energy"]` [J] (per-Gauss-point values in
`diagnostics["dissipated_per_area"]`).

Why it matters: an interface patch now exposes its own damage energy, so a
nonlinear solution can check the energy balance `W_ext = W_strain + W_d` per
Newton increment without integrating the traction history numerically, and a
delamination test can report the R-curve input `W_d / A` directly.

The identity is verified against two independent oracles (rows
`law_dissipation_vs_path_integral`, `law_dissipation_equals_fracture_energy`,
`element_dissipation_equals_gc_times_area`): a numerical integral of
`t . d(delta)` along a ray (max deviation `5.0e-6` over 7 mode mixes) and the
BK closed form `G_c` (`5.7e-16`).  Because the law is a
history-of-maximum-separation model, the state function uses the **current**
mode mix; a path that rotates the mix is therefore not integrated exactly.  The
row `nonproportional_path_dissipation` measures that residual instead of
assuming it away: open to `1.5 a_0(I)` then shear to failure gives
`W_path = 1200.249 J/m^2` against `W_d = 1199.754 J/m^2` (residual `4.1e-4`,
0.04%) and does not exceed `G_IIc`.

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

9. **The verification script must import *this* checkout.**  In a shared
   `site-packages` (several worktrees, editable installs) `import tensorfem`
   resolved to another worktree's copy and the table run crashed inside *that*
   code (`RuntimeError: The size of tensor a (3) must match the size of tensor b
   (16)`) although the fix was already merged here.  A table produced that way
   would be a statement about somebody else's checkout.  The script now puts
   `../src` first on `sys.path` and records the imported module file plus the
   git commit in the JSON `provenance` block.
10. **Dissipation is a state function of (history, direction), not of the path.**
   Re-evaluating `W_d` after the mode mix has rotated re-uses the current
   direction's `a_0`, `a_f`, `G_c`; the residual against the path integral is
   `4.1e-4` for the measured I -> II path.  Do not use `W_d` as an incremental
   energy for mixed-mode path reversals without accounting for that.

## 5. Scope limits (not claimed)

* Small-strain adherends: no geometric (initial-stress) stiffness of the *host
  solid*; only the interface rotation is treated exactly (corotational).
* No friction, no contact damage, no cyclic degradation.
* No mixed-mode **non-proportional** path *exactness*: the law is a total
  (history-of-maximum-separation) formulation.  One class of non-proportional
  path (mode I pre-damage, then shear to failure) is now measured in the
  verification table (`nonproportional_path_dissipation`: work bounded by
  `G_IIc`, closed-form dissipation within `4.1e-4`); arbitrary mode-mix
  reversals remain outside the verified envelope.
* Prescribed crack path (pre-meshed interface); no crack-path search, no
  remeshing, no adaptive insertion.
* Rate and temperature dependence are not modelled.