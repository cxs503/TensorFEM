# H4-ICE-BELT-FEM round 1 — pressure-area law × plate-panel cross-validation × design-pressure zone division

**Branch:** `agent/m14884` · **Module:** `src/tensorfem/ice_belt_pressure_area.py` ·
**Run artifact:** `artifacts/H4-ICE-BELT-FEM/pressure_area_zoned_panel.json` ·
**Figure:** `artifacts/H4-ICE-BELT-FEM/fig_h4_pressure_area.png` ·
**Tests:** `tests/test_ice_belt_pressure_area.py` (28 tests)

## 1. What was asked and what this round delivers

| # | Criterion | Status | Basis |
|---|---|---|---|
| ① | Pressure-area curve implemented and compared point by point with the IACS UR (Polar Class) rule values | **PARTIAL / WITHHELD** | Curve implementation verified point by point on **four independent evaluation routes** to ≤2.2e-11; the comparison **against the IACS UR I2 table is withheld** — the rule table is not available offline and no rule constant may be invented (see §6). The comparator `compare_with_rule_table()` is ready and refuses to run on non-rule data. |
| ② | Plate-panel response cross-validated by two independent discretisations, <10 % | **PASS** | MITC4 layered shell vs Q4-SRI torch plate **0.97 %** (full field) / **2.19 %** (centred local patch); both vs the Kirchhoff Navier series 1.08 % / 1.91 %; Q4-SRI vs Navier 0.11 % / 0.27 %. |
| ③ | Design-pressure zone division correct | **PASS** | Z1–Z4 partition (tiling / contiguity / endpoints / factors), Z5 belt containment, Z6 pressure ordering, Z7 zone-factor linearity (≤2.2e-16), Z8 force accounting (0.0), Z9 class ordering against the ship-platform ice-class registry (55 ordered pairs, PC1→PC7/IAS→IC). |

Overall bundle status: **PARTIAL_WITHHELD** — criteria ② and ③ pass, ① passes for the
implementation and is withheld for the rule-table comparison. This is reported as
withheld, never as passed.

## 2. Case

Ice-belt panel of a Polar-Class ship hull, demonstration geometry:

| Item | Value |
|---|---|
| Ship length / depth / scantling draught | 120 m / 12 m / 7.0 m |
| Ice belt (vertical extent) | z = 4.10 … 9.60 m (height 5.5 m), from caller-supplied offsets −2.9 m / +2.6 m |
| Frame spacing s | 0.35 m |
| Panel spans | plate field 1.6 m, stiffener 3.2 m, primary member 6.4 m |
| Load patches A = s·ℓ | 0.56 / 1.12 / 2.24 m² |
| Hull-area zones (end fractions, factors) | stern ≤0.20 (0.75), midbody ≤0.60 (0.85), bow-intermediate ≤0.85 (1.00), bow ≤1.00 (1.25) |
| Plate thickness (cross-validation panel) | 10 mm, E = 206 GPa, ν = 0.3, SSSS |
| Pressure-area anchor | p_ref = 2.0 MPa at A_ref = 1 m², α = 0.3, C_d = C_f = 1 — **DEMONSTRATION**, not a rule value |

The zone boundaries, the belt offsets and the pressure anchor are **inputs**; the module
hard-codes none of them (`PressureAreaLaw.provenance` is gated by
`check_pressure_provenance`, which fails closed on `DEMONSTRATION*` data when
`require_design_data=True`).

## 3. Method

### 3.1 The pressure-area law and its four independent routes

`p(A) = C_d C_f p_ref (A/A_ref)^(-α)`, `F(A) = p(A)·A = C A^(1-α)`.

| route | how `p` is obtained | what it would catch |
|---|---|---|
| `direct` | `p = C_d C_f p_ref (A/A_ref)**-α` | — (reference) |
| `log` | `p = exp(log C − α (log A − log A_ref))` | any use of the wrong power/area path |
| `marginal` | `p = (dF/dA)/(1−α)`, `dF/dA` by central differences of `F` only | a wrong `F–p` relation, i.e. exactly the trap below |
| `radial` | mean of a self-similar local field `p_local(r) = c r^(−β)`, `β = 2α`, by 2-D Gauss–Legendre with the substitution `r = R t^(1/(2−β))` | a wrong exponent/area scaling of the averaged quantity |

Additional checks: the scaling invariant `p·A^α` must stay equal to `C_d C_f p_ref A_ref^α`
along the curve; a least-squares log-log fit must return the designed exponent; the
inverse `F → A → F` must round-trip; and `∫ p dA = ΔF/(1−α)` by Gauss–Legendre on 32
geometric panels.

**Trap found and fixed in this round.** `dF/dA` is *not* the pressure: with
`F = pA` and `p ∝ A^(−α)`, `dF/dA = (1−α) p`. The first implementation of the "inverse"
route compared `dF/dA` with `p` and reported a 30 % inconsistency — the artefact of an
incorrect identity, not of the law. Both the marginal route and the quadrature identity
now carry the explicit `(1−α)` factor, and `test_marginal_route_needs_the_one_minus_alpha_factor`
pins the trap.

### 3.2 Panel cross-validation (three mutually independent discretisations)

* **MITC4** — `tensorfem.ice_belt_collapse.PlateModel` (numpy layered 5-DOF flat shell,
  assumed transverse-shear field), the element used by the collapse chain;
* **Q4-SRI** — `tensorfem.plate_mindlin.solve_plate_on_winkler_foundation` with `k = 0`
  (independent torch implementation, one-point shear rule);
* **Navier** — Kirchhoff double series for the SSSS plate, with the double Fourier
  coefficients of a centred rectangular patch (`navier_patch_centre`), no mesh at all.
  With `patch_area=None` the series reduces to the classical uniform-load solution
  (verified against `ice_belt_collapse.navier_ssss_centre`, rel. diff <1e-10).

Both load cases take their magnitude from the pressure-area law: the full-field case at
`p(A_panel)`, the local case at `p(0.25 A_panel)`. Because at the *design* pressure the
panel is far past first yield, the cross-validation runs at a factor
`λ = 9.394e-05` (full field) / `1.181e-04` (patch) of the design pressure, chosen so the
peak deflection is 2 % of the thickness; `check_load_linearity` then proves
`w(2p)/w(p) = 2.0000` (tol 1e-4), i.e. the comparison measures the discretisation and not
a nonlinear solution path.

### 3.3 Zone division

`partition_hull_zones` (boundaries must be strictly increasing, last = L, so a gap or an
overlap cannot be constructed), `ice_belt_extent` (offsets as inputs, belt must start above
the base line), `panel_patches` (areas ordered plate ≤ stiffener ≤ primary) →
`design_pressure_zones` (12 cells: 4 zones × 3 patches) → `check_design_pressure_zones`
(Z1–Z8) + `check_class_ordering` (Z9, ranks injected from
`GET /api/v1/polaris/ice-classes` on the ship platform, port 8002; when the registry is
unreachable the check is reported withheld, never assumed).

## 4. Results (programmatically injected from the run artifact)

### 4.1 Pressure-area curve, point by point (Pa→MPa)

| A [m²] | p direct | p log route | p marginal route | p radial-mean route | F = p·A [N] | q = F/√A [N/m] | p·A^α |
|---|---|---|---|---|---|---|---|
| 0.05 | 4.912912 | 4.912912 | 4.912912 | 4.912912 | 245 645.6 | 1 098 560.5 | 2 000 000.0 |
| 0.10 | 3.990525 | 3.990525 | 3.990525 | 3.990525 | 399 052.5 | 1 261 914.7 | 2 000 000.0 |
| 0.25 | 3.031433 | 3.031433 | 3.031433 | 3.031433 | 757 858.3 | 1 515 716.6 | 2 000 000.0 |
| 0.50 | 2.462289 | 2.462289 | 2.462289 | 2.462289 | 1 231 144.4 | 1 741 101.1 | 2 000 000.0 |
| 1.00 | 2.000000 | 2.000000 | 2.000000 | 2.000000 | 2 000 000.0 | 2 000 000.0 | 2 000 000.0 |
| 2.00 | 1.624505 | 1.624505 | 1.624505 | 1.624505 | 3 249 009.6 | 2 297 396.7 | 2 000 000.0 |
| 4.00 | 1.319508 | 1.319508 | 1.319508 | 1.319508 | 5 278 031.6 | 2 639 015.8 | 2 000 000.0 |
| 8.00 | 1.071773 | 1.071773 | 1.071773 | 1.071773 | 8 574 187.7 | 3 031 433.1 | 2 000 000.0 |
| 16.00 | 0.870551 | 0.870551 | 0.870551 | 0.870551 | 13 928 809.0 | 3 482 202.3 | 2 000 000.0 |

Error summary (all relative): log route ≤ **1.33e-15**, marginal route ≤ **2.22e-11**,
radial-mean route ≤ **2.22e-16**, invariant spread ≤ **2.22e-16**, inverse-area round trip
≤ **2.22e-16**, fitted exponent **−0.3000000000000002** (error −2.2e-16, max log residual
5.3e-15), quadrature identity `∫p dA = ΔF/(1−α)` rel. error **0.0** (single naive panel
would be 5.5e-5, reported alongside to show the refinement is real).

### 4.2 Panel cross-validation

| case | λ | p [MPa] | w/t | MITC4 w [m] | Q4-SRI w [m] | Navier w [m] | MITC4↔Q4-SRI | MITC4↔Navier | Q4-SRI↔Navier |
|---|---|---|---|---|---|---|---|---|---|
| full field | 9.394e-05 | 1.4172e-04 | 0.0200 | 1.978480e-04 | 1.997877e-04 | 2.000000e-04 | **0.97 %** | 1.08 % | 0.11 % |
| centred patch (A/4) | 1.181e-04 | 2.7000e-04 | 0.0204 | 2.038258e-04 | 1.994657e-04 | 2.000000e-04 | **2.19 %** | 1.91 % | 0.27 % |

Mesh study (MITC4 vs Navier, full field): 7 → −5.23 %, 11 → −2.08 %, 15 → −1.08 %,
21 → −0.49 % (monotone convergence, roughly second order in the deflection error).

### 4.3 Zone division (design pressure [MPa] / patch force [kN])

| zone (factor) | plate field A = 0.56 m² | stiffener A = 1.12 m² | primary A = 2.24 m² |
|---|---|---|---|
| stern (0.75) | 1.7850 / 999.6 | 1.4499 / 1623.8 | 1.1777 / 2637.9 |
| midbody (0.85) | 2.0230 / 1132.9 | 1.6432 / 1840.4 | 1.3347 / 2989.7 |
| bow-intermediate (1.00) | 2.3800 / 1332.8 | 1.9331 / 2165.1 | 1.5702 / 3517.3 |
| bow (1.25) | 2.9750 / 1666.0 | 2.4164 / 2706.4 | 1.9628 / 4396.6 |

Checks: Z1 tiling 120.000 m (gap 0.0), Z2 contiguity, Z3 endpoints, Z4 factors > 0, Z5
belt 4.10–9.60 m inside depth 12 m, Z6 pressure strictly decreasing / force increasing
across the patch hierarchy in every zone, Z7 zone-factor ratio reproduced to ≤2.2e-16,
Z8 force accounting 0.0, Z9 class ordering strictly monotone in the injected registry rank
(55 ordered pairs).

## 5. Error judgement against a reference

| key quantity | reference | got | error | <3 %? |
|---|---|---|---|---|
| p(A) at 9 areas | direct power law (same implementation family) | log / marginal / radial routes | ≤2.2e-11 | yes (implementation, **not** an accuracy claim vs IACS) |
| exponent α | 0.3 (documented IACS local-pressure exponent) | fit −0.3000000000000002 | 2.2e-16 | yes |
| panel peak deflection | Kirchhoff Navier series | MITC4 / Q4-SRI | 1.08 % / 0.11 % full field; 1.91 % / 0.27 % patch | yes |
| **p vs IACS UR (PC) rule values** | **IACS UR I2 table** | — | **not computed** | **WITHHELD — no basis to answer** |

The `<3 %` question is therefore answered *yes for the panel/discretisation and curve
implementation* items and *no basis* for the rule-value comparison. Nothing in this round
supports a design-use claim.

## 6. Withheld item (round-2 input)

```
criterion_1.rule_table_comparison.status = WITHHELD
fail_stage = rule-table ingestion
failure_reason = the IACS UR I2 local-pressure table (p_ref per Polar Class at the rule
  reference area, with C_d/C_f definitions) is not available offline in this worktree and
  is not exposed by the ship platform API; repository policy forbids embedding invented
  rule constants, so no ref-vs-got numbers against the rule are claimed.
ready_api = compare_with_rule_table(law, RulePressureTable(rows=..., provenance="IACS UR
  I2 Rev.x <date> clause 3.2", is_rule_data=True))
```

Round-2 options, in order of cost: (a) ingest the UR I2 table as a data file carrying
edition + clause (no code change — the comparator and the fail-closed gate already exist);
(b) expose it through the ship platform as a rule-data endpoint so TensorFEM keeps zero
domain semantics; (c) only if (a)/(b) are impossible, transcribe with two-person
verification and `is_rule_data=True` + clause citation.

## 7. Notes, limits, applicability

* **Linear-range cross-validation.** At the demonstration design pressure (1.51 MPa) the
  10 mm, 1.6 m panel is far beyond elasticity; the comparison therefore runs at λ ≈ 1e-4
  of the design pressure with an explicit linearity proof. The discretisation agreement is
  a property of the elements, not of the load level — but this must be stated, because a
  reader could otherwise mistake the 1 % for a large-deflection result.
* **Square panel only for the Q4-SRI route**, because that solver is a square-mesh plate
  routine. The real plate field is 0.35 × (≤1.6) m; the cross-validation geometry is the
  square panel of the same thickness/material, and the Navier route (which handles any
  aspect ratio) is the one that anchors the actual bay.
* **The pressure-area law is geometry-agnostic in the rule sense** (only `A` matters). A
  self-similar radial field would give a *shape-dependent* mean, so the radial route is a
  consistency check of the exponent/averaging, not a proof of the rule's geometry
  independence.
* **Kirchhoff Navier** is thin-plate theory; at `a/h = 160` the thickness correction is
  far below the 10 % band, but for `a/h < 20` the Q4-SRI route would be the reference and
  Navier would need a Mindlin series.
* **Patch convention**: the analysis patch preserves the plate aspect ratio
  (`ratio = √(A_patch/A_panel)`), identical to `ice_belt_collapse.patch_pressure_load`.
  A rule patch of a different shape would shift `w` by a few percent; the comparator would
  need the patch geometry as data.
* **No plastic / collapse content in this module** — that is the collapse chain
  (`ice_belt_collapse`) and the cyclic chain (`ice_belt_ratcheting`); this round only
  qualifies the pressure law, the panel discretisation and the zone division they consume.

## 8. Reproduce

```bash
# full bundle (writes the artifact) — ~65 s
PYTHONPATH=src python -m tensorfem.ice_belt_pressure_area \
    --out artifacts/H4-ICE-BELT-FEM/pressure_area_zoned_panel.json
# figure from the artifact (no number typed in the script)
PYTHONPATH=src python scripts/fig_h4_ice_belt_pressure_area.py
# acceptance tests
PYTHONPATH=src python -m pytest tests/test_ice_belt_pressure_area.py -q   # 28 passed
```

External anchors: IACS UR I2 "Design ice pressure" (power-law form, α = 0.3 structure);
ISO 19906:2010 local ice action (same pressure-area form); Timoshenko & Woinowsky-Krieger
(1959) *Theory of Plates and Shells* §30 (uniform-load Navier series); Dvorkin & Bathe
(1984) MITC4; Hughes, Cohen & Haroun (1978) one-point shear integration; Bree (1967) —
not used here. The IACS numeric table is **not** an anchor of this round because it could
not be obtained.