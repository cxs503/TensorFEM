# 8-node three-dimensional cohesive interface element

Anchor: `GAP-FEM-3D-COHESIVE` (3-D cohesive zone element + verification case).
Module: `src/tensorfem/cohesive3d.py`; tests: `tests/test_cohesive3d.py`
(19 tests); tables/figure: `benchmarks/cohesive3d_verification_tables.py` →
`outputs/cohesive3d/verification_table.json`, `outputs/cohesive3d/traction_separation.png`.

## 1. What already existed and what this adds

Before this change the repository had cohesive capability only at two levels:

| Existing module | Scope |
|---|---|
| `tensorfem.cohesive` | scalar mode-I bilinear law + **two-node** normal interface |
| `tensorfem.mixed_mode_cohesive` | 2-D BK mixed-mode law on proportional paths (material point only) |

Neither could be used inside a 3-D mesh. `tensorfem.cohesive3d` adds the missing
structural element and its consistent tangent:

* 8-node (4+4) zero-thickness interface, bilinear interpolation of the
  separation field over the mid-surface, Gauss integration in an orthonormal
  mid-surface frame (`R` rows: normal, in-plane tangent 1, in-plane tangent 2);
* three-dimensional BK mixed-mode law (normal + two shear directions) as the
  natural extension of `mixed_mode_cohesive`: quadratic stress onset criterion,
  ``lam0``, ``lamf = 2 Gc / (w lam0)``, ``Gc = GIc + (GIIc-GIc) (Gs/w)^eta``;
* irreversible damage (secant unloading at the historical maximum equivalent
  separation), normal-compression penalty without tensile damage;
* exact consistent tangent by automatic differentiation of the law
  (`torch.autograd.functional.jacobian`), so the element tangent differentiates
  the *implemented* law including its direction dependence;
* diagnostics used by the verification: per-Gauss-point damage, equivalent
  separation history, and dissipated cohesive energy.

## 2. Formulation

Separation at a Gauss point, with face shape functions ``N_i``:

```
delta(ksi,eta) = R(ksi,eta) @ sum_i N_i (u_{i+4} - u_i)   (local: n, s, t)
lam             = |delta|  (normal component clipped at 0)
c               = delta / lam                             (mix direction)
lam0            = 1 / sqrt((kn c_n/tn0)^2 + (ks c_s/ts0)^2 + (ks c_t/ts0)^2)
w               = kn c_n^2 + ks (c_s^2 + c_t^2)
Gc(c)           = GIc + (GIIc - GIc) (ks (c_s^2+c_t^2) / w)^eta
lamf            = 2 Gc(c) / (w lam0)
traction        = (kn dn, ks ds, ks dt) * g(lam_max),  g = lambda0 (lamf-lam)/(lam (lamf-lam0))
f_e             = int_A B^T R^T t dA ,    K_e = int_A B^T R^T (dt/ddelta) R B dA
```

The envelope integral is closed form: `int t . d(delta) = w lam0 lamf / 2 = Gc`,
so each mix direction dissipates exactly its BK fracture energy — this is what
the verification measures rather than assumes.

``eta = 2.284`` is the Benzeggagh–Kenane (1996) fit for IM7/8552; the strength
onset uses the quadratic stress criterion of Camanho & Dávila
(NASA/TM-2002-211737). Both are the standard reference conventions used by the
existing 2-D module, so the two model levels stay consistent.

## 3. Verification (all numbers from `outputs/cohesive3d/verification_table.json`)

Model: `kn=1e5, ks=6e4, tn0=10, ts0=8, GIc=0.5, GIIc=1.2, eta=2.284`.

| Check | Computed | Closed form | Relative error |
|---|---:|---:|---:|
| mode-I peak traction | 10.000000 | 10 | 0 |
| mode-I traction at `2 GIc/t0` | 1.39e-15 | 0 | 1.4e-15 |
| mode-II peak traction | 8.000000 | 8 | 0 |
| mode-III (out-of-plane shear) peak | 8.000000 | 8 | 0 |
| BK `Gc` at mix 3:4 | 6.545390e-1 | 6.545390e-1 | 0 |
| BK `Gc` at mix 1:1:1 | 6.753293e-1 | 6.753293e-1 | 0 |
| work integral, mix 3:4 | 6.545386e-1 | 6.545390e-1 | 5.3e-7 |
| work integral, mix 1:1:1 | 6.753289e-1 | 6.753293e-1 | 7.2e-7 |
| element dissipated energy (4 mm² interface) | 2.298020 | `Gc*A = 2.298020` | 1.9e-16 |
| rigid translation residual | 0 | 0 | 0 |
| flat interface area | 1.000000 | 1 | 0 |
| element tangent vs central differences (h=1e-8) | — | — | 2.2e-12 |
| element tangent vs central differences (h=1e-9) | — | — | 1.9e-11 |

The two work-integral rows are trapezoid quadrature of the piecewise-linear
envelope; the exact identity `w lam0 lamf / 2 = Gc` is asserted separately in
`tests/test_cohesive3d.py` at 1e-13. **Every exact check is below 3% by a
margin of 11 orders of magnitude; the largest observed error over all exact
checks is 7.2e-7 (quadrature limited).**

Additional tests (not tabulated because they assert identities, not values):

* pure mode-I uniform opening reproduces the existing **two-node mode-I law**
  (`tensorfem.cohesive`, `BilinearCohesiveLaw`) at every point of the envelope;
* order-2 Gauss reproduces a bilinear elastic separation field exactly (order-3
  rule agrees to 1e-8 on a warped interface);
* action–reaction: the nodal force vector sums to zero and the upper-face
  resultant equals an independently re-implemented quadrature (1e-13);
* irreversibility: unload/reload on the secant, no history loss, zero traction
  after failure, no damage under normal compression;
* input validation and degenerate-geometry detection.

Figure (`outputs/cohesive3d/traction_separation.png`): left panel — normalised
BK envelopes for mode I, mode II, mix 3:4 and mix 1:1:1 with onset (dotted) and
failure (dashed) separations; right panel — dissipated energy normalised by
`Gc(direction)`, which reaches exactly 1.0 for every mix direction.

## 4. Limitations (not claimed)

1. **Small-strain, reference-configuration frame.** A rigid *translation* is
   stress free (verified) but a rigid *rotation* produces a spurious shear
   separation of order `omega x h`, because the frame is not updated. Objective
   formulation (corotational mid-surface) is not implemented.
2. **Proportional paths only.** Non-proportional mixed-mode histories are
   represented by the single scalar history `lam_max`; mode-mix history
   dependence is not modelled.
3. **Prescribed crack path.** The interface mesh is user supplied; no crack
   initiation, tracking, insertion or branching.
4. **Bilinear geometry, no finite sliding, no mortar tying.**
5. **Assembled multi-element Newton qualification is still open.** An H8 block
   split by this interface and driven in displacement control was implemented to
   compare against the exact series-spring solution (`u = delta + t(delta) H/E`,
   peak load `t0 A`). The Newton loop converged (5–8 iterations) but the
   reaction bookkeeping did not reproduce the exact series solution, so the
   script was **not** shipped; it is recorded here as open work rather than as a
   passing benchmark. The element kernel itself is qualified by the table above
   and by the assembly-level consistent-tangent check (finite differences of the
   24×24 element tangent).
6. No DCB/ENF comparison against corrected beam theory yet: a cohesive DCB needs
   `l_cz/h_element >= 3` and root-rotation corrections to be within a few
   percent of CBT, which is a separate (larger) benchmark effort.

## 5. Reproduction

```
PYTHONPATH=src python -m pytest tests/test_cohesive3d.py -q          # 19 passed
PYTHONPATH=src python benchmarks/cohesive3d_verification_tables.py   # table + figure
```