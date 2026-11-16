# CASE JOUR-ICE-STRUCT-CALIB-PROV - calibration-factor provenance (ice structure)

Generated 2026-11-16T01:23:26Z by `benchmarks/jour_ice_struct_calib_prov/run_calib_prov.py`.

## 1. Problem setting and baseline source

The live ice-structure response is `total = raw * calibration_factor` with the
factor a **per-ship constant** (this ship: `cf = 0.753028`) whose fleet value scatters
over `0.670077 .. 0.901199`.  No provenance is attached to the response, so no accuracy claim
about a calibrated total is auditable.  Baselines used here:

- live-state ledger record `live_state.json` (transcribed from `JOUR-ICE-STRUCT-CALIB-PROV`);
- IACS UR I2 Rev.5 design ice load, rebuilt at run time from the sha256-bound clause
  transcription `benchmarks/anchors/iacs-ur-i-ice-load.json` (Table 1 class factors are
  re-parsed from the quote, the quote hash is re-verified);
- ISO 19906 global pressure-area family at **declared** coefficients (`C_R = 1.0 MPa`,
  `A_0 = 1.0 m2`), exponent swept over `[-0.5, -0.28, -0.2]` because the standard is WITHHELD here.

## 2. Method and implementation

`src/tensorfem/calibration_provenance.py` (new) provides `ProvenanceRecord`
(source kind + citation + document/dataset SHA-256 + estimator + n + declared
uncertainty + validity envelope), a fail-closed `audit()`, the weakest-link rule
`combine_provenance()` (a ratio is never more auditable than its weaker operand,
GUM Sec. 5 quadrature for the uncertainty), `CalibrationFactor` (checks that the
value reproduces `reference/raw`), and `calibrated_total()` which either refuses
(`UnauditableCalibration`) or carries the unaudited bias.  Fleet-level helpers:
`fleet_dispersion()`, `constant_penalty()`, `best_single_constant()`.

Because `total = raw * cf`, the relative error of a calibrated total **equals** the
relative error of `cf` for every `raw`; that identity is machine-checked below, which
lets the whole audit run at the level of `cf` without per-ship `raw` values.

## 3. Parameter table (dimensionless)

| quantity | value | note |
|---|---|---|
| shipped cf (this ship) | 0.753028 | constant, no source |
| fleet cf min / max | 0.670077 / 0.901199 | ledger |
| fleet dispersion (max-min)/min | 34.49% | reported 34.50% |
| acceptance tolerance | 15% | platform standard |
| code-implied dlnP/dlnA (IACS chain) | +0.2821 | algebraic, fixed AR/CFD |
| ISO exponents swept | -0.50, -0.28, -0.20 | declared, WITHHELD |
| DUI = 40 kt: F_IACS | 15.264 MN | sub-region 1 |
| DUI = 40 kt: patch b x w | 1.269 x 3.545 m | area 4.4979 m2 |
| DUI = 120 kt: F_IACS | 17.586 MN | sub-region 1 |
| DUI = 120 kt: patch b x w | 1.341 x 3.747 m | area 5.0231 m2 |

## 4. Comparison table

### 4.1 Live-state audit (criterion: |member error| <= 15%)

| item | got | reference | error | verdict |
|---|---|---|---|---|
| fleet dispersion | 34.492% | 34.500% | -0.0081 pp | PASS |
| member error at cf_min | -16.442% | 0 | 16.442% | FAIL |
| member error at cf_max | +12.379% | 0 | 12.379% | PASS |
| best single constant | 0.768640 | - | minimax 14.709% | PASS |
| scaling identity d(total)/total = d(cf)/cf | max dev 1.11e-16 | 0 | - | PASS |

### 4.2 Model-form ratio cf = F_ISO / F_IACS (documented, parameter-swept)

| DUI [kt] | A [m2] | F_IACS [MN] | ISO ex | p_ISO [MPa] | F_ISO [MN] | cf |
|---|---|---|---|---|---|---|
| 40 | 4.4979 | 15.264 | -0.50 | 0.4715 | 2.121 | 0.1389 |
| 40 | 4.4979 | 15.264 | -0.28 | 0.6564 | 2.952 | 0.1934 |
| 40 | 4.4979 | 15.264 | -0.20 | 0.7403 | 3.330 | 0.2181 |
| 120 | 5.0231 | 17.586 | -0.50 | 0.4462 | 2.241 | 0.1274 |
| 120 | 5.0231 | 17.586 | -0.28 | 0.6364 | 3.197 | 0.1818 |
| 120 | 5.0231 | 17.586 | -0.20 | 0.7241 | 3.637 | 0.2068 |

Ratio range **0.1274 .. 0.2181** (spread 71.16% over the swept parameters).

The ratio *level* (0.127-0.218) does **not** overlap the observed fleet band (0.670-0.901), and that is not evidence about the live constant in either
direction: the ISO side is a declared normalisation that may be rescaled arbitrarily, so only the
**parameter dependence** of the ratio is used here, never its absolute level.

### 4.3 Provenance gate

- IACS operand: `AUDITABLE`
- ISO operand: `UNAUDITABLE` -- a judgement value carries no external evidence and cannot support a quantitative claim; replace it with a code rule, an experiment or a regression dataset before quoting a calibrated total
- derived ratio: `UNAUDITABLE` -- weakest link
- `calibrated_total(..., require_auditable=True)` raised `UnauditableCalibration`: **True**
- permissive path carries `unaudited_bias_rel = 0.0000`

Remediation ladder (generated from the audit, filled, re-audited):

| stage | verdict | missing fields |
|---|---|---|
| as_shipped | UNAUDITABLE | - |
| regression_n4 | UNAUDITABLE | sample_size>=5 |
| regression_n8 | AUDITABLE | - |

## 5. Sensitivity / convergence

- Two displacements (40 / 120 kt) x three ISO exponents: the ratio moves from 0.1274 to 0.2181; the sign and size of the parameter dependence do not depend on the exponent choice alone.
- `C_R` invariance: doubling/tripling the declared reference pressure leaves the
  parameter dispersion at 71.165% (checked at C_R = 1.0 and 3.0 MPa, invariant = True); only the absolute
  ratio level scales with `C_R`, which is why no absolute level is claimed.
- Best-constant floor 14.709% vs tolerance 15%: a constant is admissible on this fleet spread.

## 6. Failure modes and artefacts

| artefact | path |
|---|---|
| audit payload | `artifacts/jour_ice_struct_calib_prov/calib_prov_audit.json` |
| fleet band figure | `artifacts/jour_ice_struct_calib_prov/fig_cf_fleet_band.png` |
| model-ratio figure | `artifacts/jour_ice_struct_calib_prov/fig_cf_model_ratio.png` |

Reproduce: `python benchmarks/jour_ice_struct_calib_prov/run_calib_prov.py`

Limits: the audit sees only the fleet (min, max) and one ship constant, so no
distributional statement is made; the ISO side is WITHHELD, so the derived ratio is
non-normative and the gate is expected to refuse it - that is the finding, not a
shortfall of the run.
