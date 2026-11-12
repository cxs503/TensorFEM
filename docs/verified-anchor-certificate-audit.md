# Verified-anchor certificate audit: the evidence registry cannot lie about itself

Status: **shipped** · engine `tools/verified_anchor_audit.py` · fail-closed gate
`tests/test_verified_anchor_certificates.py` (16 tests) · machine record
`benchmarks/verified/anchor_certificates.json` · shrink-only debt registry
`benchmarks/verified/anchor_certificate_debt.json` · generated table
`docs/verified-anchor-certificate-table.md` · figure
`docs/assets/anchor-certificates/certificate_audit.png`

## 1. Case — what the registry is, and what was wrong with it

`benchmarks/verified/` is TensorFEM's qualification surface: 32 anchor
directories, each holding a driver (`run.py` / `run_verification.py`), a
`result.json` with the numbers, and a `gates_verified.json` in schema
`tensorfem.verified-gates.v1` whose gates are claims of the form

```
quantity   value  <operator>  threshold    declared passed: true/false
```

The registry had three structural weaknesses, all found in this session by
reading the shipped artifacts (no solver was re-run to find them):

1. **Missing evidence.** 13 of 32 anchors declared figure artifacts in the
   `artifacts` map that **do not exist anywhere in the repository**. A
   certificate that names evidence which was never committed cannot be
   re-inspected — it can only be believed.
2. **Silent skips.** `benchmarks/verified/test_verification_ladder.py` covers a
   handful of anchors and calls `pytest.skip` when an anchor's artifacts are
   absent. Nine anchors have **no `gates_verified.json` at all**; the suite that
   is supposed to guard them stays green because it never looks at them.
3. **Unfalsifiable passes.** Gates were found that declare `passed: true` while
   carrying no numeric threshold, a non-finite value, or a comparison written
   only in prose. Nothing can ever turn such a gate red.

The lever that makes all three machine-checkable is that `value` and `threshold`
are **numbers inside the artifact**: the verdict is recomputable without the
solver, so "the certificate contradicts its own numbers" becomes a decidable
property rather than a review opinion.

## 2. Method

`tools/verified_anchor_audit.py` walks `benchmarks/verified/*/`, normalises every
gate (list-shaped, dict-shaped and flat legacy shapes; `passed`/`status`/`pass`/
`verdict` spellings; `threshold`/`limit` spelling; `value`/`actual`), and applies
this rule set:

| rule | classification |
|---|---|
| explicit `operator` present, `declared != recomputed` | `E_GATE_CONTRADICTION` (hard) |
| operator only in prose (`rel err < 3%`), adjacent number read as the bound | recomputed, `W_OPERATOR_INFERRED` / `W_THRESHOLD_FROM_PROSE` (soft) |
| operator absent **and** no numeric criterion, declared PASS | `E_PASS_WITHOUT_CRITERION` (hard) |
| same, declared FAIL | `W_UNRECOMPUTABLE_FAIL` (soft — fail-closed is honest) |
| `value` is NaN/inf and declared PASS | `E_GATE_VALUE_NOT_FINITE` (hard) |
| top-level `passed` absent / contradicting the gates it aggregates | `E_OVERALL_ABSENT` / `E_OVERALL_CONTRADICTION` |
| declared artifact missing on disk / not tracked by git | `E_ARTIFACT_MISSING` / `E_ARTIFACT_UNTRACKED` |
| declared "artifact" is prose, not a path | `W_ARTIFACT_NOT_A_PATH` (soft) |
| schema tag ≠ `tensorfem.verified-gates.v1`, or no envelope at all | `E_SCHEMA` |

Design choices worth stating, because they bound the claim:

* **Prose inference is limited to an adjacent comparison symbol.** A trend
  statement such as `e(4) > e(8) > e(32)` carries no numeric bound, so no
  threshold is invented for it (unit test
  `test_trend_statement_is_not_read_as_a_bound`). Guessing a bound would make the
  audit itself unfalsifiable.
* **Inferred operators never produce a hard defect.** They are reported as
  `W_CONTRADICTION_INFERRED`, because my reading of the prose is not evidence.
* **The audit never re-runs a solver**, so it costs ~0.3 s and can gate every
  commit. It therefore cannot detect "the physics is wrong" — only "the
  certificate contradicts itself, or its evidence is absent".

`tests/test_verified_anchor_certificates.py` runs the same code and fails closed:
the set of uncertified anchors must equal the shrink-only debt registry
(`anchor_certificate_debt.json`). New debt is rejected; a repaired anchor must be
deleted from the registry (`stale` check), so the debt count cannot quietly grow
back.

## 3. Results

### 3.1 Registry state before and after the repair wave

Measured with the same auditor in this session (before/after differ only in the
artifacts restored in §3.2):

| metric | before | after |
|---|---:|---:|
| anchors | 32 | 32 |
| certified (self-consistent certificate) | 5 | **15** |
| contradicted (hard defect inside the certificate) | 18 | **8** |
| uncertified (no `gates_verified.json`) | 9 | 9 |
| anchors with `E_ARTIFACT_MISSING` | 13 | **0** |
| gates total | 229 | 229 |
| gates whose verdict was recomputed from `value`/`threshold` | 187 | 187 (81.7 %) |
| gates with a **hard** contradiction (declared operator) | 0 | **0** |
| gates contradicted under a prose-inferred operator | 2 | 2 |
| gates no machine can grade | 42 | 42 |

The complete per-anchor table is regenerated into
`docs/verified-anchor-certificate-table.md`; every number comes from
`anchor_certificates.json`, none is typed by hand.

### 3.2 Repair wave 1 — 28 declared-but-uncommitted figures restored

The `artifacts` maps named PNGs that the anchors' **own committed drivers**
produce. Re-running those drivers regenerated them; 28 figures across 12 anchors
are now committed (`fem_beam_static`, `fem_cantilever`, `fem_curved_beam`,
`fem_hertz_contact`, `fem_kirsch_plate`, `fem_modal`, `fem_panel_buckling`,
`fem_plate_modal`, `fem_plate_postbuckling`, `fem_plate_static`,
`fem_structural_benchmarks`, `fem_winkler_plate`). The anchor JSON evidence was
left at its published revision (`git checkout -- benchmarks/verified/*.json`):
the figures are new, the numbers are untouched.

### 3.3 Reproducibility side-observation

Re-running the eight drivers was also a reproducibility probe. Six anchors
reproduced **bit-identically**; two differed at floating-point noise and no gate
verdict changed:

| anchor | quantity | published | rerun | relative deviation |
|---|---|---:|---:|---:|
| `fem_plate_modal` | a residual output | 3.4906e-11 | 1.3578e-11 | 1.6×10⁻¹ on a 1e-11 residual (both ≪ threshold) |
| `fem_structural_benchmarks` | case-B metric | 3.6109361694e-3 | 3.6109361533e-3 | 4.5×10⁻⁹ |

`fem_cantilever` reproduced every deflection/stress digit with only
`solve_seconds` changing.

### 3.4 Findings that stay open (booked in the debt registry)

| # | finding | anchors | remedy |
|---|---|---|---|
| F1 | no certificate at all | `fem_axisym_shell`, `fem_beam`, `fem_elastoplastic_plate_buckling`, `fem_hertz_contact_loadcontrol`, `fem_kerr_plate`, `fem_plate`, `fem_sent_j_integral`, `fem_thermal`, `rainflow_fix_spec` | publish `gates_verified.json` from a fresh run |
| F2 | PASS with no numeric criterion | `fem_shell_modal` (`G3.shell`), `fem_sphere_pressure_membrane` (`G5`, `G6`), `fem_shell_scordelis_lo` (`cylindrical.monotone_qoi`) | give the gate a threshold, or mark it informational |
| F3 | non-finite value declared PASS | `fem_structural_acoustics` (`C5_cylinder_gap_trend`, value `NaN`) | replace with the measured quantity |
| F4 | schema deviation (`partial` tag, bare list, dict-of-gates, flat map, no aggregate verdict) | `fem_structural_acoustics`, `fem_shell_modal`, `fem_sphere_pressure_membrane`, `fem_wave_dispersion` | re-emit in the canonical envelope |
| F5 | `anchor` field ≠ directory name | `fem_modal`, `fem_morley_skew_plate` | align the field |
| F6 | declared PASS that the prose criterion reverses | `fem_structural_acoustics` `B4_plate_high_ka_plateau` (value 1.034 vs prose bound 0.05), `C2_esm_resolution_convergence` (0.0209 vs 0.05 read as `>`) | a human must re-read the criterion — booked, not "fixed" |

F6 is deliberately left open: my operator was **inferred from prose**, so
declaring a contradiction there would be my error, not the artifact's.

## 4. Figures

`docs/assets/anchor-certificates/certificate_audit.png`, two panels:
(a) per anchor, the gates split into "verdict recomputed and agrees" /
"not machine-checkable" / "contradiction under a declared operator";
(b) the registry grades. Generated by `tools/verified_anchor_audit_figure.py`
from `anchor_certificates.json` only.

## 5. Notes / limitations

* **No gate was rewritten and no number was edited** in any anchor's
  `result.json` / `gates_verified.json`. Fixing F1–F5 means re-running the
  corresponding benchmark, which is a separate (heavier) wave; normalising an
  evidence file in place would destroy the provenance that makes auditing
  possible.
* The audit cannot tell whether the oracle named in a gate is the right one: it
  reads `value`/`threshold`/`operator`, not the physics. `fem_wave_dispersion`
  carries three recomputed FAIL gates and no aggregate verdict; that is booked as
  `E_OVERALL_ABSENT`, not as an error in the dispersion results.
* Units inside prose criteria are not machine-checkable: the
  `fem_sphere_pressure_membrane` control gate states "> 3%" while its value is a
  percentage (133.17), so the comparison runs in mixed units. It passes either
  way (the gate is an element control), but the mixed reading is a trap.
* `git ls-files` is the tracking oracle. When git is unavailable the artifact
  check degrades to existence-only and the JSON records
  `git_tracking_checked: false`.
* The debt registry is shrink-only by test, but it is generated: editing it by
  hand to hide a defect makes `defect_drift` fire.

## 6. Error determination

The audited quantities are exact integers (gate counts) and exact booleans
(verdict agreement), so there is no tolerance to report for them: 187 of 187
recomputable gates agree with their declared verdict, 0 contradictions under a
declared operator — that is an integer identity, not an approximation.

The one measured quantity with a numeric comparison is reproducibility (§3.3):

| quantity | reference | measured | error | < 3 %? |
|---|---|---:|---:|---|
| `fem_structural_benchmarks` case-B metric rerun | published 3.6109361694e-3 | 3.6109361533e-3 | 4.5×10⁻⁹ relative | **yes** |
| `fem_plate_modal` residual rerun | published 3.4906e-11 | 1.3578e-11 | 1.6×10⁻¹ relative, absolute 2.1e-11 | **no** on relative terms — declared as such |

The `fem_plate_modal` residual is a **1e-11-scale quantity**: relative deviation
is large, absolute deviation is 2×10⁻¹¹ and both readings are ~10⁹ below the
gate threshold, so no verdict moves. It is reported as a relative-error failure
rather than hidden behind the absolute scale.

No physical QoI (deflection, stress, frequency) is claimed in this document;
where physical numbers appear (§3.3) they are the anchors' own published values,
quoted for the reproducibility comparison only.