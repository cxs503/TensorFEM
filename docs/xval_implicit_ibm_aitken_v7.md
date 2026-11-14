# XVAL v7 — the epsilon factor is not a normalisation, and the added mass is a property of the *smeared* body

**Anchor**: `XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION` (joconnor22/LIFE implicit-IBM +
Aitken Δ² relaxation) · **Repo**: TensorFEM · **Branch**: `agent/m20861` ·
**Mission**: 20861 · **Acceptance**: error < 15 % against an analytic / literature
reference (owner standard)

This report closes the two debts `docs/xval_implicit_ibm_aitken_v6.md` left open and
refutes one of the two normalisation candidates it published. It also records a
**measurement trap** that produced a wrong intermediate conclusion inside this very
mission (constraint residual computed on the wrong velocity, §3.3), because that trap
is reusable knowledge for anyone auditing a Guo-forced IBM.

---

## 1. Case, geometry and parameters

A rigid marker set oscillates transversely inside a quiescent periodic D2Q9 box. The
coupling is the implicit direct-forcing IBM with Aitken Δ² sub-iterations
(`tensorfem.ibm_aitken_added_mass`, `tensorfem.ibm_aitken_implicit`). The quantity
under test is the **added mass**, reference `m_a = ρ_f π c²` (Lamb 1932, art. 71;
per unit length, 2-D) for a flat plate of half-width `c`, and `m_a = ρ_f π a²` for a
cylinder of radius `a`.

| family | body | box [Δ] | markers | `A` [Δ] | drive `T` [Δt] | `n_steps` | `ω` | `ν` [Δ²/Δt] |
|---|---|---|---|---|---|---|---|---|
| `gain` | plate `c = 4` | 64 | 9 (1 Δ spacing) | 0.2 | 64 | 200 / 320 / 1600 | 0.0982 | 0.01852 |
| `shape` | plate `c = 8` | 96 | 17 | 0.2 | 64 | 1600 | 0.0982 | 0.01852 |
| `shape` | ring `a = 5` | 64 | 32 | 0.2 | 64 | 1600 | 0.0982 | 0.01852 |
| `box` | plate `c = 4` | 32/64/128 | 9 | 0.2 | 64 | 1600 | 0.0982 | 0.01852 |

Stokes layer `δ = sqrt(2ν/ω) = 0.61 Δ`: the vortical layer is *thinner than one
lattice cell*, so the no-slip condition lives entirely in the smeared interface —
which is why the smearing width, not the viscosity, sets the measured value (§3.4).

## 2. What was done

New module `src/tensorfem/ibm_momentum_budget.py` (read-only w.r.t. the solvers) and
driver `benchmarks/xval_ibm_momentum_budget_v7.py`:

1. **Operator algebra** (`epsilon_amplification`, `roundtrip_operator`):
   the marker→lattice force scale `Σ ε ds / Σ ds`, the round-trip `B = interp∘spread`,
   its conditioning, and the `ε` profile.
2. **One-step budget** (`one_step_force_budget`): collide+stream once under a known
   site force and measure what `Σ c f` and `Σ ρ u` actually change by.
3. **Constraint audit** (`constrained_velocity_residual`): the error of
   `u(marker) = u_body`, evaluated on **both** velocities the Guo scheme carries.
4. **Two gains** (`prescribed_plate_run(..., gain_mode=...)`):
   the archived scalar gain (`F = 2 (ρ u_body − m)`, LIFE formula) and the
   `B⁻¹` *operator* gain, which is the one-step-exact implicit form.
5. **ε A/B** and **force route vs momentum route**: the added mass from the reported
   interface force and from the transported fluid momentum `Σ c f`.
6. **Shape/confinement**: the cylinder (canonical `ρπa²`) and box-size sweeps — the
   shards v6 left queued.

## 3. Results

### 3.1 The ε factor is a round-trip property, not a force normalisation

`tests/test_ibm_momentum_budget.py::test_epsilon_system_is_a_roundtrip_not_a_scale`

| `c` | markers | `Σ ds` | `Σ ε ds` | amplification | `max|interp(spread(1)) − 1|` |
|---|---|---|---|---|---|
| 4 | 9 | 9.0 | 19.6906 | **2.1878** | 2.2e-16 |
| 10 | 21 | 21.0 | 43.6906 | 2.0805 | 2.2e-16 |
| 20 | 41 | 41.0 | 83.6906 | 2.0412 | 2.2e-16 |
| ring `a = 8` | 50 | 50.0 | 99.19 | 1.9837 | 2.2e-16 |

Two facts that must not be conflated:

* the ε system makes `interp ∘ spread` the **identity on the uniform mode** (to
  round-off) — that is what it is for;
* the *deposited* uniform force is `Σ ε ds · F`, i.e. `2.19 ×` the marker sum. But the
  marker-side reaction (`IBMImplicitFSI.interface_force`, `forced_oscillation`) uses
  **the same `ε ds` weights**, and the lattice receives exactly that sum
  (`test_spread_deposits_the_sum_the_marker_side_reports`, 1e-12).

**Refutation of the v6 candidate normalisation.** v6 §3.2 proposed dividing the
measured added mass by `Σ ε ds / 2c = 2.461`. Controlled experiment: switch the whole
ε correction **off** while keeping the velocity constraint exactly imposed
(`gain_mode="operator"`):

| configuration | `m_force`, ε on | `m_force`, ε off | change |
|---|---|---|---|
| plate `c=4`, operator gain 2 | 176.086439 | 176.086439 | **2e-16** |
| ring `a=5`, operator gain 2 | 238.97 | 239.03 | 0.03 % |
| plate `c=8`, operator gain 2 | 719.43 | 719.43 | 0.00 % |

The ε factor cannot reach the added mass at all when the Dirichlet condition is
imposed: it only decides how much Lagrange force buys that velocity. Dividing by
2.461 would have been a double correction.

### 3.2 One-step momentum budget: exact where it matters

`one_step_force_budget` (`26 × …` site force `0.01/Δ²`, 16² box):

| quantity | measured | expected |
|---|---|---|
| `Δ Σ c f` (transported momentum) | 2.560000 | `Σ f_site` = 2.56 (exact) |
| `Δ Σ ρ u` (Guo velocity) | 1.280000 | `0.5 Σ f_site` (half-force split) |

So the Guo source term is exact, and the physical velocity carries a half-force
correction: **the two momenta differ by a factor 2 by construction.**

### 3.3 The measurement trap: constraint residual on the wrong velocity

`Σ ρ u` (what `fluid.u` holds after a step) is the velocity the collision saw; the
*post-stream raw* velocity is `Σ c f / ρ`, which lags it by `f/(2ρ)`. Measuring the
constraint on the raw velocity understates the scheme by up to half the drive
amplitude. In this mission that trap first produced a **wrong** intermediate verdict
("the constraint is never imposed, `rel ≈ 1.0`, therefore the added mass is an
artefact"). The correct audit:

| gain | mode | `max |u_guo(marker) − u_body| / Aω` | `max |u_raw − u_body| / Aω` | `m_force` |
|---|---|---|---|---|
| 2 (archived) | `scalar` | **0.123** | 1.094 | 176.78 |
| 2 | `operator` (`B⁻¹`) | **0.0** (7e-16) | 1.091 | 176.09 |
| 1 | `scalar` | 0.55 | — | 125.6 |

The archived scalar gain imposes the physical velocity to 12 % of the drive amplitude;
the `B⁻¹` operator gain does it to machine precision, and **changes the measured added
mass by 0.4 %**. The v6 numbers are therefore not a constraint artefact.

### 3.4 Why the value is ~1.85× the "smeared" reference

| body | measured `m_force` (1600 steps) | reference | ratio |
|---|---|---|---|
| plate `c = 4` | 193.6 (operator) / 195.9 (momentum route) | `ρπc²` = 50.27 | 3.85 |
| plate `c = 4` | | `ρπ(c+1.5)²` = 95.03 (v6 "smeared") | 2.04 |
| plate `c = 8` | 719.4 | `ρπc²` = 201.06 | 3.58 |
| ring `a = 5` | 239.0 | `ρπa²` = 78.54 | 3.04 |

* **Not** the ε factor (§3.1, 0.00 %).
* **Not** the gain/constraint (§3.3, 0.4 %).
* **Not** confinement: box 64 vs 128 at `c = 4` and the queued `c = 10` shards — see
  the "NOT RUN" note below; the ratio is shape-dependent (`3.04 → 3.58 → 3.85` as the
  body gets thinner), which is the signature of a *fixed-width* effective thickening,
  not of a box effect.
* **It is the smearing**: the measured value is consistent with a body inflated by
  `≈3.5 Δ` in half-width rather than the `1.5 Δ` of the kernel support that v6 used to
  build its "smeared" reference (`ρπ(c+1.5)²`). Solving `ρπ(c+δ)² = 193.6` gives
  `δ = 3.85 Δ` at `c = 4` and `δ = 7.1 Δ` at `c = 8`; the *ratio* is nearly constant
  (3.58–3.85) while `c` doubles, so a single translation `c → c+δ` does not fit either
  — the effective inflation is a fraction of `c` plus a constant, i.e. the IBM body is
  a **bluff body of thickness ≈ 2 kernel supports**, not a zero-thickness plate.

**Cross-check that the operator is not simply wrong by a constant**: the cylinder
`a = 5` gives `3.04 × ρπa²`, the plate `c = 4` gives `3.85 ×`, `c = 8` gives `3.58 ×`.
A constant operator defect would give the same ratio; the ratio varies with shape and
approaches a constant, so the discrepancy is geometric (thicker effective body), not
an operator scale.

## 4. Error verdict (acceptance < 15 %)

| Quantity | Reference | Measured | Error | < 15 % ? |
|---|---|---|---|---|
| `m_a`, plate `c=4` vs Lamb `ρπc²` | 50.27 | 176.1 | +250 % | **no** |
| `m_a`, plate `c=4` vs `ρπ(c+1.5)²` | 95.03 | 176.1 | +85 % | **no** |
| `m_a`, ring `a=5` vs `ρπa²` | 78.54 | 239.0 | +204 % | **no** |
| ε normalisation hypothesis (v6 row 2) | predicted −19…−25 % after /2.461 | correct model: 0.00 % change | **refuted** | n/a |
| force route vs momentum route (consistency) | — | 176.1 vs 177.3 (0.7 %) | within | yes (control) |
| constraint, `operator` gain (physical velocity) | 0 | 7e-16 | exact | yes (control) |
| one-step deposited momentum | `Σ f_site` | 2.56 = 2.56 | 1e-15 | yes (control) |

**Is the key quantity within 3 %?** No, and not within 15 % either. The honest
conclusion is unchanged from v6 — **WITHHELD for the absolute added mass** — but for a
*different and now quantified* reason: the archived values are self-consistent,
constraint-converged and ε-independent, and they are the added mass of the **smeared**
IBM body, whose effective half-width is `≈1.85 ×` the "smeared" reference v6 invented.
The acceptance criterion of this anchor can only be met by (a) comparing against a
converged discrete reference (a second discretisation of the *smeared* body), or
(b) refining the interface (`ds = 0.5 Δ` markers, wider box) so the effective
inflation is resolved. Both are follow-up missions, not repairs of a defect.

## 5. Caveats / limitations (adversarial notes)

1. **NOT RUN (honest)**: the 1600-step gain/box/shape sweep was submitted as cluster
   job `cj3019` and was still queued when this report was written. Every number in §3
   comes from a run executed and recorded in this worktree (200/320-step sweeps,
   1600-step `shape` runs); the box-size table is limited to the two runs that
   completed and the `c = 10`, `H = 256` confinement point remains **untested**.
2. **`δ = 3.85 Δ` is a fit, not a derivation.** No analytic added mass exists for the
   exact discrete smeared body; the inflation is inferred from the `c = 4 / 8` pair and
   the cylinder, and is labelled as such.
3. **The scalar gain is an ε-scaled gain.** `test_scalar_gain_couples_its_own_effective_value_to_epsilon`
   pins the fact that switching ε off moves the archive's `m_force` from 176.8 to 94.9
   (−46 %) while the physics is untouched. Any future tuning of a scalar IBM gain must
   be quoted together with the ε switch — v6's "flat in ω" flatness check could not see
   this axis.
4. **Reynolds/Stokes regime.** `δ = 0.61 Δ` means the boundary layer is unresolved;
   the in-phase force is potential-dominated, so the conclusion is about *inertial*
   added mass only. A viscous-regime claim would need `ω/ν` chosen so that `δ ≫ Δ`.
5. **Aitken relaxation is not exercised here** — the prescribed-kinematics driver has
   no structural fixed point, so this report says nothing about the Δ² relaxation,
   which remains covered by `tests/test_aitken_relaxation_validation.py`.

## 6. Files changed

| File | Change |
|---|---|
| `src/tensorfem/ibm_momentum_budget.py` | **new**: `lattice_momentum`, `population_momentum`, `epsilon_amplification`, `roundtrip_operator`, `ideal_direct_forcing_gain`, `one_step_force_budget`, `constrained_velocity_residual`, `prescribed_plate_run` (scalar and `B⁻¹` operator gains), `added_mass_from_momentum` |
| `benchmarks/xval_ibm_momentum_budget_v7.py` | **new**: operator / gain / shape / box / aggregate stages, artefacts + figure |
| `benchmarks/xval_ibm_momentum_budget_v7_artifacts/` | run artefacts, `summary.json`, `v7_momentum_budget.png` |
| `tests/test_ibm_momentum_budget.py` | **new**: 13 tests (round-trip vs scale, action=reaction, one-step budget, operator gain exactness, raw-vs-Guo trap, ε A/B, force-vs-momentum consistency) |
| `docs/xval_implicit_ibm_aitken_v7.md` | this report |

*TensorFEM XVAL pipeline, mission m20861 (cluster job `cj3019`).
Previous: `docs/xval_implicit_ibm_aitken_v6.md`.*