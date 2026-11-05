# h-convergence order and Roache GCI evidence (SOLID3D / HEX8)

Status: **implemented and machine-verified** for the `hex8` element;
`tet4` and curved/shell families are **not covered** by this evidence set.

## 1. Why this page exists

The suite already contained consistency checks (patch tests, uniaxial exact
fields) and peak-value mesh-convergence gates
(`tensorfem.panel_mesh_convergence`, an *adjacent-mesh* gate that deliberately
refuses to extrapolate).  What it did **not** contain was any statement about
the *rate* at which discretisation error decays, nor any error band around a
fine-mesh result -- exactly the two quantities an independent reviewer asks for
first ("is the observed order the theoretical one?" and "how large is the
remaining discretisation uncertainty?").  A converged-looking study can still be
converged to the wrong answer, so the gate also requires a comparison against an
*independent* reference.

| component | module | role |
| --- | --- | --- |
| observed order, Richardson extrapolation, Roache GCI, gate reports | `tensorfem.convergence_verification` | solver-agnostic, evidence-only |
| closed-form pure-bending oracle, manufactured solution, study drivers | `tensorfem.elasticity_convergence` | independent analytic references |
| reproducible evidence + figure | `benchmarks/h_convergence_gci.py` | writes `docs/assets/h-convergence-gci/` |
| acceptance tests | `tests/test_convergence_verification.py`, `tests/test_elasticity_analytic_verification.py` | run in the normal suite |

## 2. Method

For a refinement sequence with constant ratio `r = h_coarse / h_fine` the
observed order of accuracy of three consecutive levels is

```
p = ln |(f_fine - f_mid) / (f_mid - f_coarse)| / ln r          (Richardson)
```

the limit value is extrapolated with
`f_0 = f_fine + (f_fine - f_mid) / (r^p - 1)`, and the Roache Grid Convergence
Index of the finest pair is

```
GCI_fine = Fs * |f_fine - f_coarse| / |f_fine| / (r^p - 1),   Fs = 1.25
```

`Fs = 1.25` is used because the order is observed rather than assumed.  A study
is `passed` only when **all** of the following hold: an order is measurable, the
order drift between the first and last triple stays inside a band, the finest
order matches the theoretical order, the GCI band contracts with refinement,
and -- when a reference is supplied -- the extrapolated value is inside
tolerance *and* the finest-mesh true error lies inside the GCI band.  A
non-monotone or non-constant-ratio sequence is reported as `blocked` and is
**never** extrapolated; three identical levels are reported as machine-precision
convergence with no fabricated order.

### Independent references (no self-generated truth)

1. **Pure bending of a prismatic bar** (Navier solution).  For curvature `k`,
   `u_x = -k x y_c`, `u_y = k/2 (x^2 + nu y_c^2 - nu z_c^2)`, `u_z = k nu y_c z_c`
   is an exact elasticity state (`sigma_xx = -E k y_c`, all shears zero) with the
   closed-form energy `U = E k^2 L I / 2`.
2. **Manufactured trigonometric field** with body force `b = -div(sigma)`
   obtained by automatic differentiation of the analytic displacement and
   consistent nodal loads integrated with an independent 4x4x4 Gauss rule (no
   `B`-matrix, Jacobian or stiffness from the code under test is used).
   Cross-checks used in the tests: `b` agrees with `-div(sigma)` computed through
   the analytic stress expression (rel. 1e-16), and
   `1/2 int u.b dV = int sigma:eps/2 dV` (rel. 5e-5).

## 3. Case and parameters

| quantity | pure bending | manufactured |
| --- | --- | --- |
| box `L x w x h` [m] | 2.0 x 0.4 x 0.6 | 2.0 x 0.4 x 0.6 |
| `E` [Pa] / `nu` [-] | 2.100e+11 / 0.3 | 2.100e+11 / 0.3 |
| loading | imposed curvature `k` = 0.001 1/m (Dirichlet field) | body force from the manufactured field (Dirichlet field on the boundary) |
| meshes | `(2n, n, n)` HEX8, `n` = 1, 2, 4, `r = 2` | `(2n, n, 3)` HEX8, `n` = 2, 4, 8, `r = 2` |
| analytic energy [J] | 672.000000 | 16978.999929 |

The manufactured field is run in the `planar` variant (independent of `z`, three
element layers through the thickness so that interior nodes exist): the `z`
direction is then discretised exactly and the study stays two-dimensional, which
keeps the suite fast without weakening the measured order.

## 4. Results

### 4.1 Pure bending -- consistency (superconvergence)

| n | mesh | h [m] | free dofs | rel. L2 displacement error | U_FE [J] | U_exact [J] | rel. U error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `[2, 1, 1]` | 1.0000 | 0 | 0.000e+00 | 2421.138462 | 672.000000 | 2.603e+00 |
| 2 | `[4, 2, 2]` | 0.5000 | 9 | 5.729e-17 | 1109.284615 | 672.000000 | 6.507e-01 |
| 4 | `[8, 4, 4]` | 0.2500 | 189 | 1.582e-16 | 781.321154 | 672.000000 | 1.627e-01 |

The nodal field is reproduced to machine precision
(max rel. L2 error 1.58e-16), i.e. the
element is consistent for a bending state.  The strain-energy error column is the
*interpolation* energy of the element (it measures the spurious shear of the
trilinear interpolant, not a Galerkin error) and decays at the observed order
2.00, which is the theoretical `h^2` for the energy
of a linear element.

### 4.2 Manufactured solution -- measured rate and error band

| n | mesh | h [m] | free dofs | rel. L2 displacement error | U_FE [J] | U_exact [J] | rel. U error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | `[4, 2, 3]` | 0.5000 | 18 | 3.477e-02 | 12668.825653 | 16978.999929 | 2.539e-01 |
| 4 | `[8, 4, 3]` | 0.2500 | 126 | 1.042e-02 | 15880.368255 | 16978.999929 | 6.471e-02 |
| 8 | `[16, 8, 3]` | 0.1250 | 630 | 2.725e-03 | 16704.573628 | 16978.999929 | 1.616e-02 |

| gate | value | verdict |
| --- | --- | --- |
| observed L2 order per triple | 1.738, 1.935 | trending to 2 |
| observed energy order (finest triple) | 1.962 | theory 2, deviation 1.9% (< 15 %) |
| Richardson limit of `U` [J] | 16989.1227 | vs analytic 16978.9999, rel. error 5.96e-04 |
| GCI band of the finest pair (relative) | 0.0213 | `Fs` = 1.25, band [16348.88734, 17060.25992] J |
| true error at the finest mesh vs analytic | 1.616e-02 | inside the GCI band (True) |
| FE energy vs analytic | 16704.5736 < 16978.9999 J | displacement FE approaches the exact energy from below, as projection theory requires |

Evidence file: `docs/assets/h-convergence-gci/evidence.json`; figure:
`docs/assets/h-convergence-gci/convergence.png` (panel a: element-centre
`sigma_xx` of the bending state against the analytic value; panel b: log-log
error decay with a slope-2 reference; panel c: true error against the GCI band).

## 5. How to use it on a new study

```python
from tensorfem.convergence_verification import ConvergenceLevel, evaluate_convergence_study

levels = [ConvergenceLevel(h, peak_load, f"n={n}") for h, peak_load, n in runs]
report = evaluate_convergence_study(
    levels,
    qoi="peak load [N]",
    expected_order=2.0,        # theoretical order of the element
    order_band=0.15,           # accepted relative deviation of the observed order
    reference_value=41.2e3,    # independent reference: experimental or published
    reference_tolerance=0.03,  # project acceptance threshold
)
assert report["status"] == "passed", report["checks"]
print(report["gci"]["relative_gci"])   # discretisation uncertainty band
```

Run the reproducible evidence with
`python benchmarks/h_convergence_gci.py` (it adds `src/` to `sys.path` itself).

## 6. Limitations and what is *not* claimed

* **Verification, not validation.** The references are analytic; agreement
  proves the discrete operator is consistent and converges at the theoretical
  rate.  It says nothing about whether the model (material, boundary conditions,
  element suitability) matches reality -- that still needs experimental or
  published benchmark data.
* **`hex8` here, tetrahedra in a companion note.** `tet4` and TET10 are measured
  in [tet-h-convergence-gci-validation.md](tet-h-convergence-gci-validation.md)
  (L2 orders 1.85 / 2.63 at the finest pair, energy GCI 2.37% / 0.032%, analytic
  energy reference). Shells, plates and contact are still not covered by this
  evidence set; the evaluator itself is element-agnostic and can be reused once
  a manufactured/analytic oracle exists for them.
* **No deal.II / external FE cross-check.** No `deal.II` installation is
  available in this environment, so the independent baseline is analytic rather
  than a second implementation.  A second-code comparison remains an open item.
* **GCI assumptions.** Roache's estimator presumes a monotone, constant-ratio
  refinement sequence in the asymptotic range; the pre-asymptotic L2 order here
  (1.74 -> 1.93) shows
  the gate is sensitive to that, which is why the drift and expected-order bands
  are checked explicitly instead of trusting a single triple.
* **Machine-precision levels** (e.g. patch tests) produce no order; the evaluator
  reports them as `machine_precision_convergence` rather than inventing a rate.
* The bending oracle is nodally exact on uniform meshes, so it is a consistency
  test; the manufactured solution is the case that actually measures the rate.

## 7. Suite status for this change

The new tests are green (`26 passed` for the two files above) and the benchmark
reproduces the evidence.  The full-suite baseline was executed with
`pytest tests -q -n 8 --timeout=900` on a host whose `load average` was 150-290
(shared machine): **9 failed, 896 passed, 6 skipped, 4 xpassed** in 2420 s.  None
of the 9 red items imports the modules added here; they are 3 host-load
timeouts, 3 wall-clock threshold assertions, one `PYTHONWARNINGS=error` image
issue (a `pynvml` FutureWarning raised by torch's CUDA import makes the public
API audit script exit non-zero; without warnings-as-errors it reports `ok:
true`), and the remaining items pass when re-run standalone
(`3 passed in 487.55s`).  The per-test triage is recorded in
`runs/m2106/test_baseline.md` - listed rather than hidden.
