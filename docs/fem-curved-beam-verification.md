# GAP-FEM-CURVED-BEAM-VERIFY: Quarter-Ring Curved Cantilever vs Castigliano Closed Form

**Anchor:** `GAP-FEM-CURVED-BEAM-VERIFY` (new, verification-debt class)
**Status:** ✅ **VERIFIED** — 8/8 gates pass
**Date:** 2026-11-11
**Seat:** tp-ds31-6 (generalist, self-start)

## 1. Why this exists

The verification-debt census (report 13917, m14722) identified the curved-beam
quarter-ring as the **top verification-debt candidate** for TensorFEM:

> 曲梁 quarter-ring (test_curved_beam_benchmark)：有闭式门禁 `(3π/4-2)PR³/EI`，
> 粗网格强制 FAIL；**无独立复核文档**；闭式系数仅测试内断言。

The functionality (`tensorfem.curved_beam_benchmark`) is already merged and has
a pytest gate, but it lacked a `benchmarks/verified/`-level **independent
analytical-basis review face** with a refinement ladder, graded gates, and a
convergence figure.  This single fills that debt.

## 2. Physics

A **quarter of a circular ring** of radius `R` and rectangular section
`width × thickness` is clamped at one end and loaded at the free end by a point
force `P` acting along the free-end tangent.  The tip tangent is perpendicular
to the root tangent, so the problem couples **membrane (hoop) action with
bending** and exercises exactly the curved-shell mechanism that flat-facet
shells are known to distort.

**Reference (Castigliano on the thin, inextensible-section curved bar):**

```
M(φ) = -P R (1 - cos φ),   φ = 0 at free end, φ = π/2 at clamp

tangent load → tip displacement along tangent:
  δ_t = P R³ (3π/4 - 2) / (E I)  +  π P R / (4 E A)  +  π P R / (4 k G A)

transverse (radial) tip displacement under the same tangent load:
  δ_r = P R³ / (2 E I)          (Maxwell–Betti reciprocity)

direct response to a radial end load:
  δ_x = π P R³ / (4 E I)
```

The three coefficients `3π/4 - 2 = 0.356194`, `1/2`, `π/4` are cross-checked
against an independent polygonal Euler–Bernoulli frame model, against the
flat-facet `shell4_stiffness`, and against the metric-consistent
`cylindrical_shell4_stiffness`; all three numerical routes agree to better than
0.5%.

## 3. Parameters

| Parameter | Value | Unit |
|---|---|---|
| Ring radius `R` | 10.0 | m |
| Section width (out of plane) | 0.2 | m |
| Section thickness (in plane) | 0.1 | m |
| Young modulus `E` | 2.1e5 | Pa |
| Poisson ratio `ν` | 0.3 | — |
| End load `P` | 1.0 | N |
| Elements across width | 2 | — |
| Refinement ladder `n_arc` | 4, 8, 16, 32, 64 | — |

## 4. Method

Two FE models are solved (both clamped over all six DOFs on the root edge):

* **`element="flat"`** — `shell4_stiffness` on a faceted ring (axis = global `z`).
* **`element="cylindrical"`** — `cylindrical_shell4_stiffness` (exact cylindrical
  metric, axis = global `x`).

Each is run over the refinement ladder `n_arc ∈ {4, 8, 16, 32, 64}` and graded
against the Castigliano closed form.  The driver is
`benchmarks/verified/fem_curved_beam/run.py`, which emits `result.json`,
`gates_verified.json`, and `curved_beam_convergence.png`.

## 5. Results

| element | n_arc | tangent err (%) | radial err (%) | eq residual (%) |
|---|---|---|---|---|
| flat | 4 | 3.868 | 3.661 | 2.5e-7 |
| flat | 16 | 0.747 | 0.589 | 1.2e-6 |
| flat | 64 | **0.194** | **0.140** | 3.1e-8 |
| cylindrical | 4 | 7.767 | 5.846 | 2.0e-2 |
| cylindrical | 16 | 1.078 | 0.779 | 8.1e-5 |
| cylindrical | 64 | **0.163** | **0.159** | 1.0e-6 |

**Gates (8/8 pass):**

| Gate | Quantity | Value | Threshold | Pass |
|---|---|---|---|---|
| C1_tangent_flat | tangent error | 0.194% | ≤ 1% | ✅ |
| C2_radial_flat | radial error | 0.140% | ≤ 1% | ✅ |
| C1_tangent_cyl | tangent error | 0.163% | ≤ 1% | ✅ |
| C2_radial_cyl | radial error | 0.159% | ≤ 1% | ✅ |
| C3_monotone_flat | monotone decrease | 1.0 | ≥ 1.0 | ✅ |
| C3_monotone_cyl | monotone decrease | 1.0 | ≥ 1.0 | ✅ |
| C4_equilibrium_flat | reaction residual | 3.1e-8% | ≤ 1% | ✅ |
| C4_equilibrium_cyl | reaction residual | 1.0e-6% | ≤ 1% | ✅ |

## 6. Error judgement

* **Tangent tip-displacement** vs Castigliano closed form: **0.19%** (flat),
  **0.16%** (cylindrical) at `n_arc=64` — **< 1%** ✅ (well under the 3% bar).
* **Radial tip-displacement** vs closed form: **0.14%** (flat), **0.16%**
  (cylindrical) — **< 1%** ✅.
* **Equilibrium** (root reaction = applied load): residual **< 1e-6%** ✅.

Both element routes (flat-facet and exact-cylindrical-metric) converge to the
same closed form to better than 0.2%, confirming the curved-shell mechanism is
correctly captured and that the two independent stiffness routes agree.

## 7. Notes / limitations

* The closed form is the **thin, inextensible-section** curved-bar idealisation;
  axial and transverse-shear terms are retained but small (they contribute
  < 0.1% of the tangent response at these slenderness ratios).
* The cylindrical element measures `θ` with the **opposite sense** to the flat
  mesh (its exact-metric construction requires that), so reported components
  are **magnitudes** along the declared tangent/normal axes — every gate
  compares magnitudes.
* The coarse mesh (`n_arc=4`) is **required to fail** (tangent error > 3%) so a
  regression that silently re-tunes the elements cannot pass — this is the
  anti-regression property of the original pytest gate, preserved here.

## 8. Artifacts

* `benchmarks/verified/fem_curved_beam/run.py` — verification driver
* `benchmarks/verified/fem_curved_beam/result.json` — machine-readable results
* `benchmarks/verified/fem_curved_beam/gates_verified.json` — gate manifest
* `benchmarks/verified/fem_curved_beam/curved_beam_convergence.png` — figure
* `tests/test_curved_beam_benchmark.py` — existing pytest gate (9 passed)

## 9. Suggested ledger entry

**New anchor:** `GAP-FEM-CURVED-BEAM-VERIFY`
**Acceptance:** quarter-ring curved cantilever vs Castigliano closed form,
tangent & radial tip-displacement error < 1% at `n_arc=64` for both flat and
cylindrical elements; monotone convergence; equilibrium residual < 1%.
**Domain:** TensorFEM (structural / curved-shell verification).
