# Hertz contact, Cattaneo–Mindlin partial slip and unilateral complementarity verification (P45-N02)

**Repository:** TensorFEM · **Module under test:** `tensorfem.cattaneo_hertz`
**Study driver:** `benchmarks/contact_hertz_cattaneo_study.py`
**Artifacts:** `artifacts/contact_hertz_cattaneo/{result.json, gates.json, *.png}`
**Environment:** Python 3.11, torch CPU (loong64 node), float64

---

## 1. Case description

Three contact-mechanics verification faces are implemented and graded against
closed-form solutions.  All three concern a steel indenter on an elastic body
(`E = 210 GPa`, `ν = 0.3`), small strain, small sliding, rigid indenter:

| id | configuration | effective radius | effective modulus | configurations × loads |
|---|---|---|---|---|
| H1 | sphere–sphere | `R_eff = R/2 = 25 mm` | `E* = E/(2(1−ν²)) = 115.38 GPa` | 3 loads |
| H2 | sphere–plane | `R_eff = R = 50 mm` | `E* = E/(1−ν²) = 230.77 GPa` | 3 loads |
| H3 | cylinder–plane (plane strain) | `R = 50 mm` | `E* = E/(1−ν²)` | 3 loads |
| C1 | Cattaneo–Mindlin partial slip | H2, `N = 100 N`, `μ = 0.3` | — | 9 tangential levels |
| K1 | unilateral complementarity | H2, `N = 100 N` | — | penalty `k_n` × 8, ALM × 6 |

Loads `F ∈ {10, 100, 1000} N`; the line contact uses `P' = F/10 mm`
(10 mm strip width, declared in `result.json → parameters.line_strip_m`).

The verification is *not* circular: the Hertz radius is used only to **place**
the graded mesh (standard a-priori grading) and to fix the dimensionless
scaling.  No Hertz formula enters the equilibrium, the contact detection or the
recovery of `a` and `p0`; both are read off the computed pressure field by a
least-squares fit of `p² = p0²(1 − (r/a)²)` over nodes carrying ≥ 0.5 % of the
peak pressure.

## 2. Method

1. **Condensed FE compliance.** An axisymmetric Q4 mesh (point contact) or a
   plane-strain Q4 mesh (line contact) is assembled once with unit modulus and
   lengths in units of the Hertz contact radius `a_H` (half-width `b_H`), then
   statically condensed onto the contact-surface degrees of freedom:
   `u_C = H f_C` (exact linear algebra, one LU with one right-hand side per
   contact DOF).  Linear elasticity fixes the scaling exactly
   (`H_phys = H_dim/(E_fe a_H)`, `H_dim/E_fe`), so a single factorisation per
   mesh serves every load and both point configurations.
2. **Two unilateral solvers on the same condensed problem.**
   * *Penalty*: `f = k_n max(−g, 0)` with an exact active-set loop; the
     condition number of `(I + k_n H_AA)` is recorded at every solve so the
     residual/conditioning trade-off is visible rather than hidden.
   * *Augmented Lagrangian*: the subproblem is algebraically the penalty
     problem for the shifted offset `offset − λ/ρ`, so each subproblem reuses
     the exact active-set solver while the multiplier update
     `λ ← max(0, λ − ρ g)` removes the remaining penetration; the converged
     state satisfies `f ≥ 0`, `g ≥ 0`, `f·g = 0`.
3. **Cattaneo–Mindlin tangential solver.** Rigid stick set (`u_r = s`) plus a
   Coulomb slip annulus (`q = μp`), monotone loading, active-set removal of
   nodes whose stick traction would exceed `μp`; `c/a` is a nodal estimate and
   is reported with its mesh ladder.
4. **Independent oracles.** Hertz point/line closed forms; the line-contact
   approach recomputed from the **Flamant kernel** quadrature
   (`hertz_line_approach_quadrature`) so no logarithm convention is fitted;
   the point-contact approach referenced to a surface point at `4a` and
   evaluated from the **Boussinesq kernel** (`hertz_point_approach`, elliptic
   integral by AGM) so the finite domain enters both sides identically;
   Cattaneo `c/a`, `u_t(Q)` and the Mindlin small-`Q` stiffness
   `8Ga/(2−ν)`.
5. **Mesh ladder.** `n_in = 12, 17, 24` uniform elements over `[0, 2a]`
   (`h/a = 1/6, 1/8.5, 1/12`), geometric growth 1.3 to a `100a` far field,
   patch depth `2a`; a coarser ladder `6, 9, 12` is used for order estimation;
   the tangential study additionally uses `24, 32, 40`.

## 3. Results

### 3.1 Hertz gates (finest mesh `h/a = 1/12`)

| configuration | `a_sim/a_H − 1` | `p0_sim/p0_H − 1` | `δ_sim/δ_ref − 1` |
|---|---|---|---|
| sphere–sphere, 10/100/1000 N | **+0.325 %** | **−0.526 %** | **−0.106 %** |
| sphere–plane, 10/100/1000 N | **+0.325 %** | **−0.526 %** | **−0.106 %** |
| cylinder–plane, 10/100/1000 N | **+0.422 %** | **+0.554 %** | **−0.248 %** |

The three loads give *identical* relative errors to machine precision: in the
dimensionless formulation the problem is exactly self-similar (`a ∝ F^{1/3}`,
`p0 ∝ F^{1/3}`), so the load ladder verifies the scaling law rather than new
physics.  This is stated explicitly instead of presenting the 9 points as 9
independent tests.

Gate verdicts: **A1 PASS (0.42 % ≤ 2 %)**, **A2 PASS (0.55 % ≤ 3 %)**,
**A3 PASS (0.25 % ≤ 3 %)**.

### 3.2 Complementarity gates

| `k_n` / median(H⁻¹) | `‖p·g‖₁/(N δ)` | penetration/δ | cond(`I+k_n H_AA`) |
|---|---|---|---|
| 10 | 4.30e−06 | 8.6e−03 | 36 |
| 20 | 2.16e−06 | 4.3e−03 | 39 |
| 100 | 4.34e−07 | 8.7e−04 | 42 |
| 200 | 2.17e−07 | 4.4e−04 | 42 |
| 1000 | 4.34e−08 | 8.7e−05 | 42 |
| 2000 | 2.17e−08 | 4.4e−05 | 42 |
| 10000 | 4.34e−09 | 8.7e−06 | 42 |
| 20000 | 2.17e−09 | 4.4e−06 | 42 |

ALM: residual `3.3e−13` in 10 iterations, `min g = −1.4e−11` (round-off).
**A6 PASS**: ALM `≤ 1e−8` and the penalty residual halves on every doubling
(measured ratios 0.5024, 0.5003, 0.5000, 0.5000 — a 2 % band is allowed for
discrete active-set round-off, documented in `gates.json`).

### 3.3 Energy and negative controls

* Contact strain energy `½ f'H f` versus `(2/5)Fδ`: **−1.08 %**
  (independent cross-check by the indenter work `∫F dδ`: −1.13 %).
* **A8 PASS**: gaps strictly positive ⇒ contact force exactly `0.0` for both
  solvers; zero tangential shift ⇒ traction exactly `0.0`.

### 3.4 Cattaneo–Mindlin: honest FAIL with a diagnosed root cause

`Q(u_t)` is off by up to +249 % and the stick radius collapses at
`Q ≈ 0.19 μN`, while the closed forms are `c/a = 0.93` at that load.  The
failure is **not** a discretisation issue:

| `n_in` (nodes in contact) | `Q/s` (flat-punch check) | flat-punch limit `8(G/E)/(2−ν)` | error |
|---|---|---|---|
| 24 (11) | 5.565 | 1.810 | **+207 %** |
| 32 (15) | 5.529 | 1.810 | **+206 %** |
| 40 (19) | 5.508 | 1.810 | **+204 %** |

The offset is **mesh-independent** (it does not shrink from 11 to 19 nodes in
the contact), which identifies a formulation error: the axisymmetric `n = 0`
sector imposes a *radial expansion* of the contact patch (`u_r = s`
axisymmetrically), whereas Cattaneo's tangential problem is a *unidirectional
rigid shift* — the first (`n = 1`) harmonic of the same geometry.  The `n = 0`
sector is stiffened by hoop action, hence the factor ≈ 3.  The plane-strain
alternative is not usable either: the half-model fixes `u_x = 0` at the
symmetry plane (the correct symmetry for *indentation*), which forbids the rigid
tangential shift, and the tangential problem instead requires `u_z = 0` with
`u_x` free there.

Structural properties of the tangential solver *are* verified
(`tests/test_contact_hertz_cattaneo.py`): the Coulomb bound holds node by node,
`Q` is monotone in `s`, the slip annulus only grows, and full sliding saturates
exactly at `Q = μN` (rel. 1e−9).  What is missing is a FE kernel that can carry
the unidirectional tangential field.

### 3.5 Grid convergence (A7): honest FAIL

| config | `a` rel. error (h/a = 1/6, 1/8.5, 1/12) | order `p0` | order `a` | order `δ` |
|---|---|---|---|---|
| sphere–sphere/plane | +0.94 %, −0.08 %, +0.32 % | 4.19 | not estimable | −1.52 |
| cylinder–plane | +2.43 %, +0.84 %, +0.42 % | 2.10 | −1.08 | not estimable |

The `a` recovery carries a ~0.5 % fit noise floor (lumped pressure field), so
the differences change sign and the three-mesh order estimator is undefined;
the approach is already at 0.1–0.25 %, i.e. below the truncation floor, so its
order is negative.  The peak-pressure order is well behaved (4.19 / 2.10), and
the *absolute* errors at the finest mesh satisfy A1–A3 with a factor 5–12
margin.

## 4. Figures

| file | content |
|---|---|
| `hertz_pressure_profiles.png` | `p/p0` vs `r/a` for the three configurations (finest mesh) against `√(1−(r/a)²)` |
| `hertz_convergence.png` | relative error of `a`, `p0`, `δ` vs `h/a` (F = 1000 N) with observed orders |
| `complementarity.png` | penalty residual + penetration vs `k_n` with the `1/k_n` reference and the condition number; ALM residual vs iterations with the 1e−8 gate |
| `energy_and_traction.png` | computed indentation curve vs Hertz (energy integral) and the stick/slip traction partition `q(r)` vs `μp(r)` |
| `cattaneo_mindlin.png` | `Q/(μN)` vs `u_t/a` and `c/a` vs `Q/(μN)`, FE against closed form (documents the FAIL) |

## 5. Notes, limitations, applicability

1. **Finite-domain bias.** The approach and the stored energy are systematically
   low by `O(a/W)` because the far field is clamped: measured `δ` bias
   −2.29 % at `W = 30a`, −0.86 % at `W = 100a`, −0.2 % (est.) beyond.  The
   approach gate therefore uses a *reference-relative* measurement (contact
   centre versus a surface point at `4a`, both evaluated from the same kernel),
   which removes the bias from the comparison instead of hiding it.  The energy
   gate is left exposed to it — that is why A9 misses 1 % by 0.08 %.
2. **Penalty conditioning.** `cond(I + k_n H_AA)` saturates near 42 because the
   active-set size, not `k_n`, dominates at these levels; the residual still
   follows `1/k_n` over five decades.  No stiffness-matrix pathology was
   observed, so the honest-FAIL branch reserved for "condition number explosion"
   was not needed.
3. **Point versus line approach conventions.** The line-contact approach is
   convention-dependent (logarithmic reference distance); Johnson's
   `2 ln(4R/b) − 1` and the Flamant quadrature at `ref = R` differ by 4.7 %
   (measured, `tests`), which is why the gate uses the quadrature at the
   *measured* reference node.  For the point contact no convention is needed.
4. **`c/a` resolution.** The stick radius is a nodal estimate; with 11–19 nodes
   in the contact the quantisation is ±1 node spacing.  The mesh table is in
   `result.json → cattaneo_mesh`.
5. **Not claimed.** Nothing here establishes finite sliding, deformable–deformable
   contact, plastic contact, or 3-D (non-axisymmetric) tangential contact.
   Unloading/reverse slip in the Cattaneo solver is out of scope (monotone path
   only).

## 6. Error judgement (does every key quantity stay below 3 %?)

| quantity | reference | error | `< 3 %`? |
|---|---|---|---|
| contact radius, sphere–sphere | Hertz | +0.325 % | **yes** |
| contact radius, sphere–plane | Hertz | +0.325 % | **yes** |
| contact radius, cylinder–plane | Hertz | +0.422 % | **yes** |
| peak pressure (all three) | Hertz | ≤ 0.554 % | **yes** |
| approach (all three) | Hertz / Flamant–Boussinesq | ≤ 0.248 % | **yes** |
| ALM complementarity residual | `≤ 1e−8` | 3.3e−13 | **yes** |
| energy `U` vs `(2/5)Fδ` | Hertz | −1.08 % | **yes** (but misses the 1 % sub-gate by 0.08 %) |
| `u_t(Q)` Cattaneo | Cattaneo/Mindlin | up to +249 % | **NO — honest FAIL** |
| `c/a` Cattaneo | `(1−Q/μN)^{1/3}` | up to 100 % | **NO — honest FAIL** |
| convergence order of `a`, `δ` | GCI ≥ 1.5 | not estimable | **NO — honest FAIL** |

Reproduce: `PYTHONPATH=src python benchmarks/contact_hertz_cattaneo_study.py`
(cached meshes are rebuilt in ≈ 3 min on first use; the study itself takes
≈ 40 s) and `pytest tests/test_contact_hertz_cattaneo.py` (16 tests, 68 s).