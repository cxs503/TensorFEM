# Verification note: Kerr three-parameter foundation and the `kerr_equivalent_pasternak` reduction

Anchor: `GAP-FEM-KERR-PLATE`. Code under verification: `src/tensorfem/kerr_foundation.py`
(single-field Pasternak operator `p = k w - G_p lap w`) and
`tensorfem.foundation_plate_bending.kerr_equivalent_pasternak` (three-parameter Kerr →
Pasternak reduction). Tests: `tests/test_kerr_foundation_verification.py`.
All numbers below are produced mechanically by `scripts/kerr_reduction_validity.py`
into `artifacts/kerr_reduction_validity.json` (no hand-copied values); figure:
`docs/assets/kerr-foundation-reduction-validity.png`.

## 1. What was verified, and why it needed verifying

`kerr_equivalent_pasternak` reduces Kerr's three-parameter idealisation
`(k_u, k_l, G_p)` (springs to the plate, springs to the rigid base, shear layer between
them) to the two-parameter Pasternak pair

```
k_eq = k_u k_l / (k_u + k_l),      G_eq = k_u G_p / (k_u + k_l),
```

and its own docstring states that this holds *to first order in `G_p`*. Until now that
statement was untested: the existing anchors in
`tests/test_kerr_foundation_verification.py` (a)–(f) and the whole-plate anchors in
`tests/test_kerr_plate_verification.py` only ever exercise **one** field, so they cannot
see a reduction error at all — a single-field model compared with the single-field
closed form is self-consistent by construction, whatever `(k_eq, G_eq)` it was fed.

This note adds the **exact two-field Kerr model** as an independent reference and
measures the reduction against it.

## 2. Reference model (exact Kerr, two fields)

Total potential energy of Kerr's idealisation, with `w` the plate deflection and `w_1`
the shear-layer deflection,

```
Pi = 1/2 ∫ k_u (w - w_1)^2 + 1/2 ∫ k_l w_1^2 + 1/2 ∫ G_p |grad w_1|^2 - ∫ q w .
```

Its three stiffness blocks are *the production element*
`q4_kerr_foundation_stiffness` evaluated with three parameter triples, so no foundation
code is re-implemented in the reference:

```
K_ww   = k_u ∫ N^T N                            = q4_kerr(xy, k_u, 0)
K_w1w1 = (k_u+k_l) ∫ N^T N + G_p ∫ gradN^TgradN = q4_kerr(xy, k_u+k_l, G_p)
K_ww1  = -k_u ∫ N^T N                           = -q4_kerr(xy, k_u, 0)
```

The plate uses the production `q4_mindlin_stiffness`; `w_1` is pinned on the supported
edge, i.e. the boundary condition of the closed form it is compared against.

For the single sinusoidal load `q0 sin(pi x/L) sin(pi y/L)` on a simply supported square
(`lap -> -2 lambda`, `lambda = (pi/L)^2`) the exact model is closed-form:

```
w_1 = k_u w / (k_u + k_l + 2 G_p lambda)
p   = k_u (k_l + 2 G_p lambda) / (k_u + k_l + 2 G_p lambda) * w
w   = q0 / (4 D lambda^2 + p/w)
```

whereas the production reduction gives `p_eq = k_eq + 2 G_eq lambda`. The ratio is an
exact algebraic factor, the **reduction bias**

```
p_eq / p_exact = 1 + 2 lambda G_p / (k_u + k_l).
```

## 3. Case parameters

| quantity | symbol | value |
|---|---|---|
| plate side | `L` | 1.0 m |
| Young's modulus | `E` | 2.1e11 Pa |
| Poisson's ratio | `nu` | 0.30 |
| thickness | `t` | 0.01 m |
| flexural rigidity | `D` | 1.923e4 N·m |
| load amplitude | `q0` | 1.0e4 Pa |
| upper spring layer | `k_u` | 4.0e7 N/m³ |
| lower spring layer | `k_l` | 2.0e7 N/m³ |
| shear layer (sweep) | `G_p` | 1e4 … 3e7 N/m |

## 4. Results

### 4.1 The reference model is itself verified (h-convergence, `G_p = 1e6` N/m)

| n (els/side) | `w_FE` (m) | `w_1/w` | rel. error vs closed form |
|---|---|---|---|
| 8 | 3.6974e-4 | 0.50004 | 1.410% |
| 16 | 3.6593e-4 | 0.50124 | 0.365% |
| 24 | 3.6522e-4 | 0.50146 | 0.171% |

Closed form: `w = 3.64596e-4 m`, `w_1/w = 0.50164`. Observed order between `n = 8, 16`:
**1.95** (expected 2 for Q4 Mindlin). So the reference is a converged, independent
solution of the exact Kerr problem, not a second implementation of the reduction.

### 4.2 Reduction accuracy (`k_u = 4e7`, `k_l = 2e7` N/m³)

| `G_p` (N/m) | bias `p_eq/p_exact` | `w_exact` (m) | `w_reduced` (m) | rel. error (closed form) | rel. error (FE, n=8) |
|---|---|---|---|---|---|
| 1e4 | 1.0033 | 4.7815e-4 | 4.7715e-4 | 0.211% | 0.211% |
| 3e4 | 1.0099 | 4.7423e-4 | 4.7123e-4 | 0.632% | 0.633% |
| **1e5** | 1.0329 | 4.6135e-4 | 4.5162e-4 | **2.107%** | 2.110% |
| 3e5 | 1.0987 | 4.3063e-4 | 4.0365e-4 | 6.266% | 6.280% |
| **1e6** | 1.3290 | 3.6460e-4 | 2.9424e-4 | **19.30%** | 19.37% |
| 3e6 | 1.9870 | 2.9350e-4 | 1.6582e-4 | 43.50% | 43.65% |
| 1e7 | 4.2899 | 2.4227e-4 | 6.5608e-5 | 72.92% | 73.06% |
| 3e7 | 10.8696 | 2.2203e-4 | 2.4061e-5 | 89.16% | 89.23% |

The closed-form and the two independent FE models agree to 0.15% (absolute), so the
error column is the reduction, not the discretisation.

### 4.3 Validity range (bisection on the 3% / 15% acceptance lines)

| line | `G_p` (N/m) | `G_p/(k_u+k_l)` | bias |
|---|---|---|---|
| 3% error (accept) | 1.425e5 | 2.37e-3 | 1.047 |
| 15% error (reject) | 7.522e5 | 1.25e-2 | 1.247 |

Validity criterion (single mode): `2 lambda G_p / (k_u + k_l) ≲ 0.05`, equivalently
`G_p ≲ 0.025 (k_u + k_l) / lambda`. For multi-mode loading every mode `mn` is biased by
its own `1 + 2 lambda_mn G_p/(k_u+k_l)` with `lambda_mn = pi^2(m^2+n^2)/L^2`, so the
fundamental mode carries the smallest bias and the single-mode study is the optimistic
(least-stiffened) case in terms of bias — while deflection is still dominated by the
fundamental mode, so the table above is the relevant error scale.

### 4.4 Limits (documented and now locked by tests)

| limit | `(k_eq, G_eq)` | bias | measured reduction error |
|---|---|---|---|
| `G_p = 0` | `(k_u k_l/(k_u+k_l), 0)` | 1 (machine precision) | < 1e-14 |
| `k_l -> inf` (rigid base pins the layer) | `-> (k_u, 0)` = Winkler `k_u` | `-> 1` | 4.2e-7 at `k_l = 1e6 k_u` |
| `k_u -> inf` (plate tied to layer) | `-> (k_l, G_p)` = Pasternak | – | – |

## 5. Verdict

* Element and algebra anchors (a)–(f) hold: symmetry, PSD, `G_p = 0` identity, monotone
  shear stiffening, fail-closed inputs, plus new closed-form quadrature identities
  (`1/2 k A w^2` for constant `w`, `1/2 G_p A |grad w|^2` for linear `w`, both to
  `1e-12`) — **REAL**.
* The exact three-parameter Kerr problem is solved and verified to **0.17%** at
  `n = 24` (order 1.95) — **REAL**.
* `kerr_equivalent_pasternak` is **first-order accurate in `G_p` only**: its pressure
  bias is exactly `1 + 2 lambda G_p/(k_u+k_l)`. It stays under the 3% acceptance line
  only for `G_p ≲ 1.4e5` N/m with the representative triple, and it is **not
  acceptable** (>15%, up to a factor 9.23 in deflection at `G_p = 3e7` N/m) outside
  that range. Any workflow (e.g. `benchmarks/verified/fem_kerr_plate`, whose
  representative `G_p = 5e4` N/m is inside the range (reduction error 1.05%); the element anchors' `G_p = 3e7`
  N/m is *not*) must therefore check `2 lambda G_p/(k_u+k_l)` before using the
  reduction, or solve the exact two-field model.
* Therefore the docstring statement "reduces … to first order in `G_p`" is **true but
  weak**: downstream users should treat the equivalent-Pasternak model as a
  small-shear-parameter approximation, not as an equivalent model.

## 6. Limitations of this study

* Single sinusoidal load term and simply supported edges (the configuration with a
  closed form). Uniform/point loads are covered only qualitatively (mode-dependent
  bias), not tabulated.
* The reference model shares the production element with the reduced model; only the
  assembly topology (two fields) and the closed forms are independent. A fully
  independent implementation (e.g. analytic Hankel solution for the infinite Kerr
  plate) is not provided here.
* `w_1` edge boundary condition (pinned) is a modelling choice; it matches the closed
  form's homogeneous boundary conditions. Free-edge foundations are out of scope.
* No experimental reference exists for this synthetic case; the anchors are
  closed-form/consistency, not experimental.

## 7. Recommendation

Add the reduction-validity guard to any caller of `kerr_equivalent_pasternak`:

```
bias = 1 + 2*pi**2*G_p/((k_u+k_l)*L**2)   # single dominant mode
if bias > 1.05: use the exact two-field model (or warn / refuse)
```

The helper needed for the exact model is in `scripts/kerr_reduction_validity.py`
(`solve_two_field`) and in the test module; promoting it into
`tensorfem.kerr_foundation` as `assemble_kerr_two_field` is the natural follow-up.

## Reproduction

```
PYTHONPATH=src python scripts/kerr_reduction_validity.py    # ~3.5 min, writes JSON + PNG
python -m pytest tests/test_kerr_foundation_verification.py -q
```