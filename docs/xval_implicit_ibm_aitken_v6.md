# XVAL v6 — the added-mass attribute of the implicit IBM + Aitken plate, re-derived

**Anchor**: `XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION` (joconnor22/LIFE implicit-IBM
+ Aitken Δ² relaxation) · **Repo**: TensorFEM · **Branch**: `agent/m20796` ·
**Mission**: 20796
**Acceptance**: error < 15 % against analytic / literature reference (owner standard)

This report closes the two debts `docs/xval_implicit_ibm_aitken_v5.md` left open
(§5.2 "frequency-level PASS, mass-attribute-level FAIL" and §5.3 "sign-changing
bias unexplained"). It also reports a **defect in the archived evidence** that
invalidates one published column of `benchmarks/xval_added_mass_artifacts/`.

---

## 1. Case and what was measured

The physical case is the v5 one: a thin rigid plate of half-width `c` (marker
segment spaced `1 Δ`) spring-mounted with `k`, translating transversely inside a
quiescent periodic D2Q9 box, coupled by implicit direct forcing with Aitken
Δ² sub-iterations (`tensorfem.ibm_aitken_added_mass`). Reference:
`m_a = ρ_f π c²` (Lamb 1932, art. 71), `f/f_vac = sqrt(m_s/(m_s+m_a))`.

Three protocols are exercised in this mission:

| Protocol | Observable | What it can certify |
|---|---|---|
| free decay (v5) | coupled frequency `f` | `f` directly; `m_a` only via amplification |
| legacy forced fit (`{y'', y', y, 1}`) | interface force | **nothing** (rank deficient, §2) |
| harmonic sweep (new) | force in phase/quadrature vs `ω` | `m_a` and `k_a` *if* the flow is settled (§3) |

### Parameter table (all runs of this mission)

| Family | `c` [Δ] | box [Δ] | markers | `A` [Δ] | drive `T` [Δt] | `n_steps` | `ω_lbm` | `ν` [Δ²/Δt] | `c²/ν` [Δt] |
|---|---|---|---|---|---|---|---|---|---|
| `h160_c10` | 10 | 160 | 21 | 0.25 / 0.125 | 48, 96, 192 | 240–960 | 1.8 | 0.01852 | 5400 |
| `h064_c04` | 4 | 64 | 9 | 0.20 | 64, 128 | 3200 | 1.8 | 0.01852 | 864 |

`c²/ν` is the viscous settling time; `n_steps/c²/ν` is 0.04–0.18 for the `c=10`
family and 3.7 for the `c=4` family. That ratio decides which records are
physically settled and which are not (§3.2).

---

## 2. Defect: the legacy forced-oscillation fit has zero information

`forced_oscillation` fitted the interface force onto `{y'', y', y, 1}` at a
**single** drive frequency. But the kinematics are prescribed,
`y = A sin(ωt)`, so `y'' = -ω² y` **exactly**: the first and third columns are
collinear and the least-squares problem is rank deficient
(`κ = 2.07e19`). `np.linalg.lstsq` then returns the minimum-norm split, which
puts the whole in-phase force into the `y` column.

Evidence (all reproducible, `tests/test_added_mass_harmonic.py`):

| Synthetic force built from | legacy fit returns | legacy `R²` |
|---|---|---|
| `m_a = 314.16` (exact Lamb) | `6.0e-5` | 1.000 |
| `m_a = 560.00` | `1.1e-4` | 1.000 |

The estimator is blind to `m_a` at the 1e-6 level for a 78 % change of the true
value. The **archived** record shows the same artefact:

```text
benchmarks/xval_added_mass_artifacts/evidence.json :: E1_forced
  "added_mass_fit": -0.0, "added_mass_ref": 314.159, "fit_r2": 1.0
  verdict.added_mass_fit_within_tolerance: false
```

`-0.0` was archived as a *physics* failure; it is a *linear-algebra* artefact.
The verdict flag must be read as "no measurement was taken", not as "the IBM
added mass is wrong".

**Fix (shipped).** `forced_oscillation` now reports, with the same simulation
loop, the **apparent** mass

```text
m_app(ω) = (m_a - k_a/ω²) = K/(A ω²),   K = force component in phase with sin(ωt)
c_a      = -C/(A ω),                    C = quadrature component
```

plus `added_mass_identifiable = False`, an explicit
`added_mass_definition = "apparent (m_a - k_a/omega^2)"`, and the legacy value
under `legacy_added_mass_fit` so the defect stays auditable. `m_a` and `k_a`
separate only across frequencies —
`tensorfem.ibm_added_mass_harmonic.fit_added_mass_sweep` regresses
`K/A = m_a ω² - k_a` and refuses (`identifiable = False`) when
`ω_max/ω_min < 1.5`.

---

## 3. Results

### 3.1 v5 §5.3 resolved: the "sign-changing mass bias" is algebra

The free-decay mass attribute is a *derived* quantity,
`m_a = k/(2πf)² - m_s`, so its error is fixed by the frequency error:

```text
m_a = k/(2πf)² - m_s,  m_s = m_a/r,   e = δf/f
δm_a/m_a = ((1+e)^-2 - 1)(1 + 1/r)          (exact)
         ≈ -2 (1 + 1/r) e                   (infinitesimal gain)
```

| v5 row | `r = m_a/m_s` | `δf/f` | predicted `δm_a/m_a` | archived | residual |
|---|---|---|---|---|---|
| `c10_H288` | 0.600 | −12.109 % | **+78.54 %** | +78.54 % | **−0.00 pp** |
| `c10_H160` | 0.600 | −9.4 % | +58.21 % | +56.70 % | +1.51 pp |
| `c20_H160` | 2.510 | +11.9 % | **−28.16 %** | −28.10 % | **−0.06 pp** |

The two rows with unrounded frequencies are reproduced to < 0.1 pp; the third
differs only because its 9.4 % was quoted to one decimal. Consequences:

1. The **sign change** of the mass error needs no fluid mechanism — it is the
   sign of the frequency error (`−` for `c=10`, `+` for `c=20`) carried through a
   ratio-dependent gain. v5 §5.3's hypothesis (iii) "fluid added-stiffness /
   damping that biases `m_a_eff` either way" is **not needed** and, in a
   translation-invariant periodic box, is **forbidden** for a static term
   (`k_a ≡ 0` under `y → y + const`).
2. The mass attribute carries **no independent information** beyond the 12 %
   frequency error. Certifying `m_a` to 15 % through free decay at `r = 0.6`
   would need `δf/f < 2.81 %` (gain 5.33) — i.e. 4.3× better frequency accuracy
   than the coupled solver delivers; at `r = 2.51` it needs 5.36 % (gain 2.80).
   v5's "mass-attribute FAIL" is therefore a statement about the free-decay
   *experiment design*, not about the solver, and it is retired here.

### 3.2 Forced-oscillation sweep: where the protocol is valid and where it is not

`benchmarks/xval_added_mass_harmonic_v6.py` (artefacts in
`benchmarks/xval_added_mass_harmonic_v6_artifacts/`):

| tag | `c` | box | `T` | cycles | `n/c²ν` | `m_app` | damping | `R²` |
|---|---|---|---|---|---|---|---|---|
| `h160_c10_T048_A025` | 10 | 160 | 48 | 5 | 0.044 | 47.34 | 27.16 | 1.0000 |
| `h160_c10_T096_A025` | 10 | 160 | 96 | 5 | 0.089 | 146.85 | 8.41 | 0.9384 |
| `h160_c10_T192_A025` | 10 | 160 | 192 | 5 | 0.178 | 396.67 | 2.28 | 0.9004 |
| `h160_c10_T192_A0125` | 10 | 160 | 192 | 5 | 0.178 | 397.25 | 2.28 | 0.9000 |
| `h064_c04_T064_A020` | 4 | 64 | 64 | 50 | 3.70 | 188.46 | 3.99 | 0.9974 |
| `h064_c04_T128_A020` | 4 | 64 | 128 | 25 | 3.70 | 175.23 | 2.47 | 0.9860 |

**Amplitude linearity check.** At `T = 192`, `A = 0.125` vs `0.25` gives
`m_app = 397.25` vs `396.67` — 0.15 % apart, so the measurement is in the linear
regime. This is the one control that passes cleanly.

**The `c = 10` family is transient-limited, and the regression says so.** The
records are 5 drive periods long — 0.04–0.18 of the viscous time `c²/ν = 5400`.
Regressing `K/A` on `ω²` over the three periods yields

```text
m_a = 21.23,  k_a = -0.4625,  R² = 0.8715,  ω-span 4.0   (Lamb: 314.16)
```

`m_a = 21` is absurd (it is 7 % of Lamb, and `k_a < 0` contradicts the
translation invariance of the box), so the three-term force model
`-m_a y'' - c_a y' - k_a y` does **not** describe these records: the plate's
wake is still growing and the "added mass" is being read off a transient. The
sweep is therefore reported as **not valid at `c = 10`**, and the v5-parameter
route to `m_a` is closed rather than "failing by 78 %". A settled `c = 10,
box 160` record would need `~3 c²/ν = 16000` steps per frequency (≈ 14 min per
frequency on one core), out of single-mission scope.

**The `c = 4` family is settled** (record = 3.7 × viscous time, 25–50 periods;
`R²` 0.986–0.997) and its apparent mass is *flat*: 188.46 → 175.23 for a
doubling of `ω` (7.2 %), so `k_a/ω²` leakage is small and `m_app ≈ m_a`.

**Where does that leave the absolute value?** `m_a ≈ 175–188` against
`ρπc² = 50.27` is 3.5–3.7×. Two candidate normalisations are quantified, neither
decided:

| candidate | factor | resulting `m_a` | vs reference |
|---|---|---|---|
| smeared body `ρπ(c+1.5Δ)²` | reference `95.0` | 175–188 | +84 % … +98 % |
| ε-conservation force scale `Σ ε ds /(2c)` | `19.69/8 = 2.461` | 71–77 | −19 % … −25 % vs 95.0 |

The ε row is a **measured** operator property, not a guess:
`IBMInterface`'s Pinelli/LIFE ε-conservation with markers spaced exactly `1 Δ`
and a 3-point kernel gives `Σ_i ε_i ds_i = 19.69` for a plate of length
`2c = 8 Δ` — `2.461 ×` the naive `Σ ds`. The sum has a constant offset form,
`Σ ε ds = 2c + 3.691` (measured at `c = 4, 10, 20`: ratios 2.461, 2.184, 2.092),
so the excess is largest for the short plate used in the settled regime. Whether
that `ε` normalisation belongs inside the added-mass definition is the open item. Dividing it out leaves the
`c = 4` value 19–25 % below `ρπ(c+1.5Δ)²`. The decisive discriminator is a
momentum-budget test (apply a known Lagrange force, integrate the lattice
momentum change, compare with `Σ f ε ds`), which is **NOT_RUN** here.

### 3.3 Figures

`benchmarks/xval_added_mass_harmonic_v6_artifacts/v6_harmonic.png`
(three panels, matplotlib, generated programmatically from `summary.json`):

* **(a)** apparent added mass vs drive frequency for both families, with
  Lamb `ρπc²` and the v5 free-decay inference (560.91) as horizontals: shows the
  `c = 10` family sweeping across the reference (transient) and the `c = 4`
  family sitting flat and high (settled);
* **(b)** the separated terms: in-phase coefficient `K/A` vs `ω²` with the sweep
  regression and the Lamb slope — the `c = 10` points are visibly not on a
  Lamb-slope line;
* **(c)** the legacy defect: `m_a` returned by the `{y'',y',y,1}` fit against the
  true `m_a` used to build the synthetic force — a flat line at 0 while the
  identity would be `y = x` (`κ ~ 1e19`).

No field cloud is claimed in this report: the quantity under test is an
*integral* (interface force vs frequency), and the v5 field figure
(`v5_field_vorticity.png`) already documents the field at these parameters.

---

## 4. Error verdict (acceptance < 15 %)

| Quantity | Reference | Measured | Error | < 15 % ? |
|---|---|---|---|---|
| coupled frequency `f`, `c=10`/`H=288` (v5) | 1.6667e-3 | 1.4649e-3 | 12.1 % | **yes** |
| mass attribute from free decay, `r=0.6` | 314.16 | 560.91 | 78.5 % | **no — derived, see §3.1** |
| mass attribute explained by §3.1 algebra | 78.54 % predicted | 78.54 % archived | **0.00 pp** | yes (identity) |
| `m_app` amplitude linearity, `A=0.125` vs `0.25` | 397.25 | 396.67 | 0.15 % | yes |
| `m_app(ω)` flatness, `c=4` (settled) | 188.46 (`T=64`) | 175.23 (`T=128`) | 7.2 % | yes |
| `m_a` absolute, `c=4` settled | 50.27 (Lamb) / 95.0 (smeared) | 175–188 (or 71–77 after the ε factor 2.46) | +40…+280 % / −25…+98 % | **no — WITHHELD** |
| legacy forced fit, any `m_a` | 314.16 / 560 | 6e-5 / 1e-4 | −100 % | **no — rank artefact, fixed** |

**Is the key quantity within 3 %?** No. The only 3 %-level agreements are
*identities and controls*: the algebraic amplification (0.00 pp), the amplitude
linearity (0.15 %), the frequency-flatness of the settled `c = 4` measurement
(7.2 % is the best physical flatness achieved) and the exact synthetic recovery
of the harmonic projector (< 1e-6). The added-mass *value* at `c = 10` v5
parameters is **WITHHELD**: the free-decay route cannot certify it (§3.1) and
the forced route is transient-limited there (§3.2).

---

## 5. Caveats / limitations (adversarial notes)

1. **The archived `-0.0` was invisible.** A perfect `R² = 1.0` next to a −100 %
   error should have been read as degeneracy, not as physics. Any estimator in
   this repo that reports `R²` without a conditioning/rank check is suspect —
   `legacy_basis_condition` and `identifiable` are now reported for this one.
2. **The ε-conservation factor 2.46 is not yet attributed.** It is measured for
   1 Δ marker spacing and is constant in `c`; whether it is the missing
   normalisation of the added mass or an intended consequence of exact-uniform
   force transmission (Pinelli et al. 2010) is **NOT RUN** here and must not be
   quoted as an error of the IBM operator.
3. **Shard coverage is incomplete (honest).** Three designed configurations
   (`h160_c10_T288`, `h160_c10_T384`, `h256_c10_T096/T192`) were submitted as
   cluster shards (`cj2964`, 4 shards) and were still queued (7.5 min, 18 ahead)
   when this report was written; they are **not** part of the results. Every
   number above comes from an archived artefact file in
   `benchmarks/xval_added_mass_harmonic_v6_artifacts/`. The `h256` pair would
   have tested box-size (confinement) sensitivity, which remains **untested**.
4. **The `c = 10` regression is reported as invalid on purpose.** Publishing
   `m_a = 21.23` as an "independent measurement" would be worse than the defect
   it replaces: the sweep machinery is correct (it recovers synthetic `m_a`,
   `k_a` to 1e-6, `tests/test_added_mass_harmonic.py`), the *records* are not
   settled. Frequency sweeps must carry the `n_steps/c²ν` ratio next to every
   point.
5. **Static `k_a = 0` argument.** The claim "`k_a ≡ 0` by translation invariance"
   holds for the periodic box used here. In a walled channel or with a free
   surface it does not, and then a single-frequency apparent mass is *not* the
   added mass — the sweep is mandatory, not optional.

---

## 6. Files changed

| File | Change |
|---|---|
| `src/tensorfem/ibm_added_mass_harmonic.py` | **new**: harmonic projection, apparent mass + damping, `fit_added_mass_sweep` (identifiability gate), `added_mass_gain`, `amplify_frequency_error` (exact), `required_frequency_accuracy`, `legacy_basis_condition`, `legacy_single_frequency_fit`, `measure_added_mass_sweep` |
| `src/tensorfem/ibm_aitken_added_mass.py` | `forced_oscillation` defect fix: harmonic projection replaces the rank-deficient `{y'',y',y,1}` fit; reports `apparent` mass + `identifiable=False` + legacy value; docstrings corrected |
| `benchmarks/xval_added_mass_harmonic_v6.py` | **new** sweep benchmark (shard / aggregate / figures), 10 configurations, per-point `n_steps/c²ν` diagnostics |
| `benchmarks/xval_added_mass_harmonic_v6_artifacts/` | 6 run artefacts + `summary.json` + `v6_harmonic.png` |
| `tests/test_added_mass_harmonic.py` | **new** 12 tests (defect, projector, sweep separation, refusal, identity vs archived rows, LBM labelling) |
| `docs/xval_implicit_ibm_aitken_v6.md` | this report |

*Generated by TensorFEM XVAL pipeline, mission m20796 (cluster submission `cj2964`).
Previous: `docs/xval_implicit_ibm_aitken_v5.md`.*