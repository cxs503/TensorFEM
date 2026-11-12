# W24 `ice-coupled-6dof` - impact pressure-area fit vs. the ISO 19906 area law

campaign node `paper-exp-ice-coupled-6dof-4` - generated 2026-11-12T07:15:40Z - commit `9ac00e601eae33bc0354371aae1f0b38a24e1378`

Every number below is read back from `pressure_area_fit.csv` / `iso19906_comparison.json` by `run_w24_pressure_area_fit.py`; nothing is transcribed by hand.

## 1. Question and experiment design

The W22/W23 chain uses the ISO 19906 area law `p = C_R (A/A_0)^ex` **as the contact pressure source**. A power law fitted to its own (p, A) samples therefore has no discriminating power: it returns the input exponent. This study adds the arms that make the statement testable:

| arm | role | pressure source |
|---|---|---|
| `C1_iso_area_law` | closure (imposed ISO area law as pressure source) | p = C_R (A/A_0)^ex with the declared normalisation (C_R = 1 MPa, A_0 = 1 m^2, ex = -0.28) |
| `C0_random_strength` | negative control (no area dependence) | p = sigma_floe, sigma_floe ~ lognormal(mean = 1.9 MPa, cv = 0.35); no area dependence |
| `T_rate_n015` | treatment (rate-dependent crushing, n = 0.15) | p = p_ref (epsdot/epsdot_ref)^n, epsdot = V_close/(4D), n = 0.15, p_ref = 1.900 MPa at epsdot_ref = 1 1/s (one-constant level calibration, no area exponent) |
| `T_rate_n033` | treatment (rate-dependent crushing, n = 0.33) | p = p_ref (epsdot/epsdot_ref)^n, epsdot = V_close/(4D), n = 0.33, p_ref = 1.900 MPa at epsdot_ref = 1 1/s (one-constant level calibration, no area exponent) |
| `T_rate_n050` | treatment (rate-dependent crushing, n = 0.5) | p = p_ref (epsdot/epsdot_ref)^n, epsdot = V_close/(4D), n = 0.5, p_ref = 1.900 MPa at epsdot_ref = 1 1/s (one-constant level calibration, no area exponent) |

Pre-declared: concentrations 0.3, 0.5, 0.7 x seeds 1101, 1102, 1103, 1104, 1105, 40 s per run, dt = 0.01 s, commanded speed 5 m/s, at least 15 impacts per concentration, |relative deviation| > 15% -> FAIL. Metric: per impact the peak-force sample `(A_peak, p_peak = N/A)`; secondary the impulse-weighted `p_bar = int N dt / int A dt`.

## 2. Pressure-area fits (criterion 1)

| arm | C | ex | C (MPa) | R2 | n impacts | A range [m2] |
|---|---|---|---|---|---|---|
| `C1_iso_area_law` | 0.3 | -0.2800 | 1 | 1.000 | 762 | 0.000589 - 0.648 |
| `C1_iso_area_law` | 0.5 | -0.2800 | 1 | 1.000 | 1267 | 0.000211 - 0.721 |
| `C1_iso_area_law` | 0.7 | -0.2800 | 1 | 1.000 | 1776 | 0.00041 - 0.651 |
| `C0_random_strength` | 0.3 | -0.1260 | 1.361 | 0.082 | 762 | 0.00166 - 0.633 |
| `C0_random_strength` | 0.5 | -0.1174 | 1.429 | 0.063 | 1267 | 0.000758 - 0.662 |
| `C0_random_strength` | 0.7 | -0.1205 | 1.42 | 0.064 | 1776 | 0.00169 - 0.618 |
| `T_rate_n015` | 0.3 | -0.0353 | 1.801 | 0.174 | 762 | 0.00201 - 0.683 |
| `T_rate_n015` | 0.5 | -0.0394 | 1.783 | 0.189 | 1267 | 0.00162 - 0.769 |
| `T_rate_n015` | 0.7 | -0.0436 | 1.77 | 0.222 | 1776 | 0.0021 - 0.707 |
| `T_rate_n033` | 0.3 | -0.0864 | 1.964 | 0.197 | 762 | 0.00154 - 0.614 |
| `T_rate_n033` | 0.5 | -0.0970 | 1.914 | 0.225 | 1267 | 0.00127 - 0.687 |
| `T_rate_n033` | 0.7 | -0.1068 | 1.877 | 0.263 | 1776 | 0.00162 - 0.644 |
| `T_rate_n050` | 0.3 | -0.1573 | 2.338 | 0.242 | 762 | 0.00108 - 0.502 |
| `T_rate_n050` | 0.5 | -0.1734 | 2.241 | 0.263 | 1267 | 0.00094 - 0.569 |
| `T_rate_n050` | 0.7 | -0.1878 | 2.169 | 0.297 | 1776 | 0.00116 - 0.527 |

### 2.1 Does the chain recover the law it was fed? (closure arm)

Fitted exponent on the imposed-law arm: -0.280000, -0.280000, -0.280000 (input -0.28); maximum absolute deviation 5.55e-17, R2 = 1.000000. That is the proof that this chain cannot test the law: the (p, A) relation is an identity there, not a measurement.

### 2.2 Negative control: how much apparent exponent does the coupling manufacture on its own?

Per-floe random strength with **no** area dependence (cv = 0.35, exact material exponent 0) gives a *measured* exponent of -0.126 (se 0.015, R2 0.08), -0.117 (se 0.013, R2 0.06), -0.120 (se 0.011, R2 0.06). The slope is **not** zero and its sign is systematic across seeds: the contact area is an *endogenous* variable of the coupled dynamics (a stronger floe deflects the hull earlier, so it reaches a smaller penetration and hence a smaller peak area), so high pressure correlates with small area even when the material has no area effect at all. That value is the **endogeneity floor** of the chain: an apparent exponent of that magnitude carries no information about a pressure-area law.

## 3. Ensemble parameter distribution (criterion 3)

| arm | C | ex median | ex p05 | ex p95 | impacts |
|---|---|---|---|---|---|
| `C1_iso_area_law` | 0.3 | -0.2800 | -0.2800 | -0.2800 | 762 |
| `C1_iso_area_law` | 0.5 | -0.2800 | -0.2800 | -0.2800 | 1267 |
| `C1_iso_area_law` | 0.7 | -0.2800 | -0.2800 | -0.2800 | 1776 |
| `C0_random_strength` | 0.3 | -0.1258 | -0.1345 | -0.1182 | 762 |
| `C0_random_strength` | 0.5 | -0.1175 | -0.1226 | -0.1129 | 1267 |
| `C0_random_strength` | 0.7 | -0.1221 | -0.1385 | -0.1056 | 1776 |
| `T_rate_n015` | 0.3 | -0.0383 | -0.0432 | -0.0288 | 762 |
| `T_rate_n015` | 0.5 | -0.0389 | -0.0492 | -0.0307 | 1267 |
| `T_rate_n015` | 0.7 | -0.0483 | -0.0497 | -0.0349 | 1776 |
| `T_rate_n033` | 0.3 | -0.0927 | -0.1146 | -0.0641 | 762 |
| `T_rate_n033` | 0.5 | -0.0992 | -0.1235 | -0.0733 | 1267 |
| `T_rate_n033` | 0.7 | -0.1139 | -0.1235 | -0.0871 | 1776 |
| `T_rate_n050` | 0.3 | -0.1663 | -0.2042 | -0.1200 | 762 |
| `T_rate_n050` | 0.5 | -0.1740 | -0.2130 | -0.1373 | 1267 |
| `T_rate_n050` | 0.7 | -0.2035 | -0.2207 | -0.1504 | 1776 |

Quantiles are taken over the per-seed fits (each seed fitting its own impacts); the criterion-1 fit pools all impacts of a concentration.

## 4. What the exponent actually is (confounding)

For every arm the raw slope and the mechanism-level partial exponents are related by the exact identity


    dlnp/dlnA|total = a_A + a_h * beta_h + a_V * beta_V,

with `a` the partial exponents (conditioned on the other two variables) and `beta_h`, `beta_V` the auxiliary regressions `dln h/dln A`, `dln V_c/dln A` of the ensemble. For a pressure law that is deterministic in `(A, h, V_c)` the identity closes to machine precision (residual column), so the *raw* slope can be attributed term by term:

| arm | C | raw ex | a_A | a_h | a_V | beta_h | beta_V | residual |
|---|---|---|---|---|---|---|---|---|
| `C1_iso_area_law` | 0.3 | -0.2800 | -0.2800 | -0.0000 | -0.0000 | +0.100 | +0.314 | +5.55e-17 |
| `C1_iso_area_law` | 0.5 | -0.2800 | -0.2800 | +0.0000 | -0.0000 | +0.118 | +0.319 | +0.00e+00 |
| `C1_iso_area_law` | 0.7 | -0.2800 | -0.2800 | -0.0000 | +0.0000 | +0.120 | +0.210 | +0.00e+00 |
| `C0_random_strength` | 0.3 | -0.1260 | -0.1701 | +0.2164 | +0.0465 | +0.104 | +0.462 | -8.33e-17 |
| `C0_random_strength` | 0.5 | -0.1174 | -0.1583 | +0.1688 | +0.0630 | +0.129 | +0.303 | -2.78e-17 |
| `C0_random_strength` | 0.7 | -0.1205 | -0.1574 | +0.1221 | +0.0622 | +0.132 | +0.334 | -4.44e-16 |
| `T_rate_n015` | 0.3 | -0.0353 | -0.1500 | +0.1500 | +0.1500 | +0.110 | +0.654 | -1.46e-16 |
| `T_rate_n015` | 0.5 | -0.0394 | -0.1500 | +0.1500 | +0.1500 | +0.130 | +0.607 | -2.98e-16 |
| `T_rate_n015` | 0.7 | -0.0436 | -0.1500 | +0.1500 | +0.1500 | +0.129 | +0.581 | +2.57e-16 |
| `T_rate_n033` | 0.3 | -0.0864 | -0.3300 | +0.3300 | +0.3300 | +0.106 | +0.632 | -3.33e-16 |
| `T_rate_n033` | 0.5 | -0.0970 | -0.3300 | +0.3300 | +0.3300 | +0.125 | +0.582 | +1.94e-16 |
| `T_rate_n033` | 0.7 | -0.1068 | -0.3300 | +0.3300 | +0.3300 | +0.123 | +0.554 | +9.71e-17 |
| `T_rate_n050` | 0.3 | -0.1573 | -0.5000 | +0.5000 | +0.5000 | +0.101 | +0.585 | +8.33e-17 |
| `T_rate_n050` | 0.5 | -0.1734 | -0.5000 | +0.5000 | +0.5000 | +0.117 | +0.536 | +6.38e-16 |
| `T_rate_n050` | 0.7 | -0.1878 | -0.5000 | +0.5000 | +0.5000 | +0.115 | +0.510 | +1.47e-15 |

For the rate-dependent arms `a_A = -n` to machine precision (the declared law is `p ~ (V_c h / A)**n`), while the raw slope is far smaller in magnitude: the population covariance term `a_h beta_h + a_V beta_V` cancels most of it. A single number `ex` measured on a population is therefore **not** the mechanism exponent, and comparing it with a design exponent requires the same population and the same calibre. The random-strength control shows the remaining bias channel (endogeneity of A under feedback), which no regression on `A, h, V` can remove.

## 5. ISO 19906 comparison (criteria 2 and 5)

Anchor `AN-ISO19906` - **WITHHELD**. the catalogue entry carries the algebraic form only and no licensed copy of ISO 19906 is reachable from this work tree, so C_R, ex and the clause/table/page that carry them cannot be transcribed; writing a recalled number would be fabrication

Required literature items (document / clause / table / page):

| document | locator needed | carries |
|---|---|---|
| ISO 19906:2010 (or :2019) -- Annex A, pressure-area clause | clause number + table number + page | global pressure-area exponent ex_g and coefficient C_R (with its A_0 normalisation and validity range in A) |
| ISO 19906:2010 (or :2019) -- local pressure clause/table | clause number + table number + page | local (high-pressure-zone) exponent ex_l and coefficient, plus the definition of the local area |
| ISO 19906:2010 (or :2019) -- global ice action clauses | clause number (+ any design table) | whether a tabulated design global ice force exists and how the area law is combined with the pressure factors |

Search scope already performed:

- repository grep for '19906' over src/, benchmarks/, docs/, papers/ -> code and prose mentions only, no transcription of the coefficients
- ecosystem anchor catalogue lookup by id AN-ISO19906 -> id + statement (algebraic form), no numbers
- sibling-repository anchor registries (ice-abrasion-extremes, ice-hull-propeller-threefield, ice-polygon-floe-field) -> certify that the primary text is absent and keep candidate values in a comparison-only registry

Consequence: the deviation from the **normative** exponent cannot be computed, so every row of criterion 2 is WITHHELD. What the artefacts carry - clearly labelled non-normative - is the deviation from the declared reference normalisation (`ex = -0.28`, `C_R = 1 MPa`, provenance: W22/W23 defaults) and from the ecosystem's unverified candidate (`ex = -0.3`, quote = null).

| arm | C | ex | dev vs declared ref | dev vs unverified candidate |
|---|---|---|---|---|
| `C1_iso_area_law` | 0.3 | -0.2800 | +0.0% | +6.7% |
| `C1_iso_area_law` | 0.5 | -0.2800 | -0.0% | +6.7% |
| `C1_iso_area_law` | 0.7 | -0.2800 | +0.0% | +6.7% |
| `C0_random_strength` | 0.3 | -0.1260 | +55.0% | +58.0% |
| `C0_random_strength` | 0.5 | -0.1174 | +58.1% | +60.9% |
| `C0_random_strength` | 0.7 | -0.1205 | +57.0% | +59.8% |
| `T_rate_n015` | 0.3 | -0.0353 | +87.4% | +88.2% |
| `T_rate_n015` | 0.5 | -0.0394 | +85.9% | +86.9% |
| `T_rate_n015` | 0.7 | -0.0436 | +84.4% | +85.5% |
| `T_rate_n033` | 0.3 | -0.0864 | +69.1% | +71.2% |
| `T_rate_n033` | 0.5 | -0.0970 | +65.4% | +67.7% |
| `T_rate_n033` | 0.7 | -0.1068 | +61.9% | +64.4% |
| `T_rate_n050` | 0.3 | -0.1573 | +43.8% | +47.6% |
| `T_rate_n050` | 0.5 | -0.1734 | +38.1% | +42.2% |
| `T_rate_n050` | 0.7 | -0.1878 | +32.9% | +37.4% |

## 6. Calibre bridge: global vs local pressure (criterion 4)

| item | value | comparable with ISO? | status |
|---|---|---|---|
| contact-area definition (this model) | A = w*h with w the penetrating width of the wedged bow and h the floe thickness (identical to the W23 chain) | only if the standard also uses a nominal contact area | CONDITIONAL |
| pressure calibre (this model) | p = N/A with a single uniform pressure across the contact patch -> p is the GLOBAL (area-average) calibre, participation factor kappa = 1 | same family as the standard's global-pressure calibre; not comparable with the local calibre | CONDITIONAL |
| local (high-pressure-zone) calibre | the model carries no sub-patch pressure field, so neither the high-pressure-zone area A_loc nor its internal pressure distribution exists; p_loc cannot be sampled | not comparable | WITHHELD - a non-uniform pressure field inside the contact patch (high-pressure zones / non-simultaneous contact) would be needed; the chain is a uniform-pressure idealisation, so using the global pressure as a local one would systematically under-predict local peaks and that bias cannot be quantified from this model |
| validity range of the area law (A range) | simulated A in [0.001274, 0.687] m2 (concentration 0.5, peak samples) | the standard states its validity range with the clause/table; that range is absent here, so whether the simulation lies inside it cannot be decided | WITHHELD - the validity range is part of the clause/table text, not transcribed |
| population semantics of ex | the ex reported here is a regression slope over a SPECIFIC ensemble (fixed C, floe-diameter, thickness and speed distributions); the multifactor decomposition shows that the raw slope mixes in the co-variation of h and closing speed | the standard's ex is a design constant that corresponds to no particular ice population; a comparison must declare the population | CONDITIONAL |
| C_R dimension and normalisation point | C is reported at A_0 = 1 m2 (MPa); the standard's A_0 and units must be confirmed from the clause | if the normalisation point differs, C is not directly comparable | WITHHELD - the reference area/units of C_R are not transcribed |

the simulated side has the global calibre only; the local calibre and the standard's validity range are WITHHELD. Any ex comparison must first align (i) the area definition, (ii) the pressure calibre, (iii) the population / A range and (iv) the C normalisation point -- missing any one makes the comparison meaningless.

The analytic bridge is `p_global = kappa p_loc` with `kappa = N_hp A_loc / A_nom`; the simulated model has `kappa = 1` by construction (one uniform pressure over the contact patch).

## 7. Criteria ledger

| # | criterion | status | detail |
|---|---|---|---|
| 1 | per-concentration fitted ex, C, R2 for at least 3 concentrations, listed one by one; primary arm = rate-dependent crushing n = 0.33 | **PASS** | C is the prefactor at A_0 = 1 m2 (MPa); R2 is the coefficient of determination of the log-log regression; the per-concentration numbers are in rows and in pressure_area_fit.csv |
| 2 | relative deviation of the fitted ex/C from the ISO 19906 value, per concentration; > 15 % is a FAIL with an attribution | **WITHHELD** | the normative numbers are not available (no primary text, no transcription with clause/table locators -- see anchor.required_documents), so the deviation from the standard cannot be computed. The table reports the deviation from the DECLARED reference normalisation (C_R = 1 MPa, ex = -0.28; not normative) and from the ecosystem's UNVERIFIED candidate (ex = -0.30, quote = null, comparison only); neither may be presented as the standard's value. |
| 3 | ensemble of at least 15 impacts per concentration; parameter distribution reported as median with 5-95 % quantiles | **PASS** | impact counts are pooled over all seeds of a concentration; quantiles are taken over the per-seed fits (each seed fitting its own impacts) with quantile_summary (linear interpolation) |
| 4 | global vs local pressure calibre bridged item by item; incomparable items marked WITHHELD with a reason | **PASS** | 6 items itemised (3 CONDITIONAL, 3 WITHHELD); the analytic bridge is p_global = kappa p_loc with kappa = N_hp A_loc / A_nom, and kappa = 1 in this model by construction, so no local-calibre number is claimed |
| 5 | normative values must carry a clause/table locator; if unobtainable, report WITHHELD and list the search scope | **WITHHELD** | the catalogue entry carries the algebraic form only and no licensed copy of ISO 19906 is reachable from this work tree, so C_R, ex and the clause/table/page that carry them cannot be transcribed; writing a recalled number would be fabrication |
| 6 (supplementary) | negative control: the apparent exponent produced with a material law that has no area dependence (endogeneity floor of the coupled contact) | **PASS** |  |

Criterion 5 (the normative numbers must carry a clause/table locator; if unobtainable, report WITHHELD and list the search scope) is answered by the anchor block of section 5: **WITHHELD**, with the three required document items and the search scope already performed.

## 8. Figures

- `benchmarks/ice_coupled_6dof/figures/w24_pressure_area_scatter.png` - impact pressure vs nominal contact area (log-log) for the closure arm C1, the negative control C0 and the primary rate-dependent arm T (n = 0.33), with the fitted power laws and the declared normalisation line; the imposed-law arm collapses exactly onto its own line (R2 = 1), the rate-dependent arm does not
- `benchmarks/ice_coupled_6dof/figures/w24_exponent_by_concentration.png` - fitted exponent per ice concentration and arm: median with 5-95 % quantiles over the per-seed fits; dotted grey lines are the analytic -n of the rate law and the dashed line the declared (non-normative) normalisation ex = -0.28
- `benchmarks/ice_coupled_6dof/figures/w24_exponent_decomposition.png` - multifactor log-log decomposition of the rate-dependent arms: the partial area exponent dlnp/dlnA recovers -n while the raw p(A) slope is confounded by the thickness and the closing speed (confounding is the main finding)

## 9. Limitations / honesty notes

- the contact patch carries a SINGLE uniform pressure: the model's p can only be the global (area-average) calibre, there is no high-pressure-zone / non-simultaneous contact structure, so a local-calibre ex cannot be sampled from this chain
- the pressure source is a declared engineering constitutive law: p_ref, epsdot_ref and n are declared inputs and the pressure level scales exactly linearly in p_ref; n is swept (0.15 / 0.33 / 0.50) as a sensitivity, not calibrated to ice measurements
- floes are point masses, the ice is level and homogeneous and no floe-size statistics of the crushed layer are resolved
- ex is a POPULATION-dependent quantity: the raw p-A slope mixes in the co-variation of thickness and closing speed, and the multifactor decomposition separates the mechanism-level partial exponent dlnp/dlnA; any reported ex must state its population
- the primary ISO 19906 text is not reachable -> the quantitative deviation from the standard is WITHHELD; every 'standard value' slot is left empty and in-ecosystem candidates are used for comparison only (quote = null)
- the hydrodynamic / radiation model is still the declared synthetic W23 model: this node is about the ice contact statistics, so the hull response quantities are not a full-scale validation

## 10. Reproduce

```bash
python benchmarks/ice_coupled_6dof/run_w24_pressure_area_fit.py
python -m pytest tests/test_ice_pressure_area_w24.py -q
```

Artefact hashes: `pressure_area_fit.csv` sha256 `3f7ad27cb181c0ffd7e5ee7d789540ddc4469e707a722bab3cf8b4ea76a90523`, `iso19906_comparison.json` sha256 `ded6b243cd2dc45e9a6ac2b529f6580980052d7af31cf5786e0d9dc4300975d6` (a JSON file cannot contain its own digest, so this hash is quoted here and computed after the final write).
