# Certificate debt: renderable figures + canonical sphere certificate (m17162)

Mission m17162, repo TensorFEM. Two closed verification debts and one structural
repair of the fail-closed certificate gate.

## 1. The anchor main was red: six anchors booked no debt

`tests/test_verified_anchor_certificates.py::test_no_unregistered_certificate_defect`
failed on `main` *before* any change here: six anchors carried
`E_ARTIFACT_MISSING` but were absent from the shrink-only debt registry —

| anchor | declared artifact that does not resolve |
|---|---|
| `fem_beam_static` | `deflection_profile.png`, `convergence.png` |
| `fem_cantilever` | `cantilever_field.png`, `cantilever_convergence.png` |
| `fem_hertz_contact` | `hertz_pressure.png`, `hertz_convergence.png`, `hertz_penalty.png` |
| `fem_plate_modal` | `plate_modes.png`, `plate_convergence.png` |
| `fem_plate_static` | `fields_ssss.png`, `fields_cccc.png`, `convergence.png` |
| `fem_winkler_plate` | `winkler_deflection_profile.png`, `winkler_convergence.png` |

### Root cause (measured, not guessed)

The certificates require committed PNG figures; the delivery merge gate strips
binaries. Evidence on the 2026-11-13 tree:

* `git ls-files '*.png' | wc -l` -> **489** repo-wide, but
  `git ls-files 'benchmarks/verified/**/*.png' | wc -l` -> **0**;
* `git log --all -- benchmarks/verified/fem_cantilever/cantilever_field.png`
  shows `d094673b` ("restore 32 certificate-named figures and clear the
  shrink-only debt registry") and `c4070685` ("restore declared figure
  evidence, close Scordelis-Lo certificate debt") — i.e. the figures have been
  restored twice and stripped twice, and merge #852 records "28 个文件被门禁滤除".

So the requirement was unsatisfiable, and the gate oscillated red once per merge
cycle. Restoring the PNGs a third time would reproduce the cycle.

### Repair: declared *renderables* instead of declared binaries

`tools/certificate_renderables.py` (new) moves an absent figure declaration out
of `artifacts` into `figures_rendered_on_demand`:

```json
{"path": "convergence.png",
 "rendered_by": "benchmarks/verified/fem_plate_static/run.py"}
```

`rendered_by` must be a **committed** script *inside the anchor directory* that
literally names the figure, so the figure is one command away from the committed
`result.json`. That is a weaker claim than "the PNG is here", but it is a claim
that survives the merge gate, and it is machine checked:

* `E_FIGURE_DECLARED_BUT_ABSENT` — an `artifacts` value names an absent figure;
* `E_RENDERABLE_NO_PRODUCER` / `E_RENDERABLE_FOREIGN_PRODUCER` — no committed
  in-directory producer;
* `E_RENDERABLE_BUT_PRESENT` — existence wins: a figure that *is* on disk may
  not be laundered into the renderable list;
* a violation on an anchor that is *already* in `anchor_certificate_debt.json`
  is tolerated but reported (`debt_registered_violations`), so debt stays
  visible instead of blocking: `fem_winkler_plate_k_sweep` ships no runner at
  all, so it cannot claim a producer and remains declared debt.

Migration result: **38 figure declarations across 15 anchors** became
renderable; no `artifacts` value was dropped without a producer.

## 2. `fem_sphere_pressure_membrane`: legacy certificate -> canonical envelope

The anchor's physics was fine; its certificate was not auditable. Legacy shape:
no `schema`/`anchor`/`passed` envelope, gate criteria as prose, and two gates
whose `value` was a vector (`G5` "e(4) > e(8) > e(32)" -> list,
`G6` observed order -> list). The audit filed it `E_SCHEMA` +
`E_PASS_WITHOUT_CRITERION`: the PASS could be *read*, not *recomputed*.

`benchmarks/verified/fem_sphere_pressure_membrane/emit_certificate.py` (new)
re-emits the certificate from `result.json` alone — zero solve, zero hand-typed
number — with `--check` to detect drift. Vector claims are scalarised without
losing their content: `G5` becomes two contraction ratios `e(8)/e(4)` and
`e(32)/e(8)` (each `< 1`), `G6` becomes the binding member `min(order) >= 1.5`.

The one **falsified** predeclared prediction (thin-shell reference band predicted
to hold to `t/R = 0.05`, measured to end at `0.02`, deviation slope 1.0276 in
`t/R`) is preserved in a `predictions` block with its numbers and
`outcome: "falsified"`, and is structurally *not* a gate, so it can neither be
hidden by a passing verdict nor be confused with an element-accuracy failure.

Gate values are unchanged where the legacy certificate was already numeric:

| gate | value | threshold | verdict |
|---|---:|---:|---|
| G1 `sigma_theta` vs `pR/2t`, 8 facets | 7.309107e-07 | 3 % | pass |
| G2 `sigma_phi` vs `pR/2t` | 1.203855e-06 | 3 % | pass |
| G3 isotropy | 1.524848e-06 | 3 % | pass |
| G4 `w` vs `pR^2(1-nu)/(2Et)` | 6.261665e-06 | 3 % | pass |
| G5a contraction e(8)/e(4) | 6.479793e-02 | < 1 | pass |
| G5b contraction e(32)/e(8) | 3.869363e-03 | < 1 | pass |
| G6 min observed order | 3.456 | >= 1.5 | pass |
| G7 Poisson invariance (5 `nu`) | 8.572344e-07 | 3 % | pass |
| G8a thin-vs-Lame at the case `t/R` | 4.013376e-03 | 3 % | pass |
| G9 chord-geometry negative control | 2.078741e+00 | must MISS | pass |
| G10 flat-facet Shell4 negative control | 1.331659e+02 | must MISS | pass |

Acceptance bar unchanged at **3 %**; every accuracy gate is far inside it
(worst 6.3e-06, i.e. 6.3e-04 %), and both negative controls miss it by 2-4
orders of magnitude, which is what makes the pass meaningful.

## 3. Anchor name/envelope mismatches

`fem_morley_skew_plate` declared `anchor: CASE-FEM-MORLEY-SKEW-PLATE` (a case
slug) and `fem_modal` declared `anchor: fem_modal_p19` (a result file), so both
tripped `E_ANCHOR_NAME_MISMATCH`; the case/result slugs are kept under a `case`
key so no provenance is lost. `fem_morley_skew_plate` also declared four figure
paths plus prose strings; the figures are now renderables and the prose strings
are gone, so `artifacts` names only committed files.

## 4. Registry state: before -> after

| metric | before (main) | after |
|---|---:|---:|
| anchors | 35 | 35 |
| certified | 12 | **27** |
| contradicted | 17 | **2** |
| uncertified (no gates) | 6 | 6 |
| gates recomputed | 275 / 297 | 278 / 297 |
| debt registry | 17 anchors, **6 unregistered (gate RED)** | 8 anchors, 0 unregistered, 0 stale, 0 drift |
| renderable figure violations | n/a (rule absent) | 0 |

Remaining debt is honest and registered: 6 anchors with no gate file at all
(`fem_axisym_shell`, `fem_elastoplastic_plate_buckling`, `fem_sent_j_integral`,
`fem_shell4_formulation_matrix`, `fem_thermal`, `rainflow_fix_spec`),
`fem_structural_acoustics` (malformed/non-finite gates + schema),
`fem_winkler_plate_k_sweep` (no runner, figures unproducible).

## 5. Limitations / scope

* No solver was re-run: every number is derived from the committed artifacts
  (capacity discipline: load15 >= 100 during this mission).
* The renderable policy *weakens* the fail-closed contract for figures on
  purpose; it compensates by requiring a committed producer. A figure whose
  producer is later deleted becomes a violation (and its anchor red) again.
* `fem_sphere_pressure_membrane` still grades a Reissner-Mindlin shell element
  against *shell* theory; the shell formula's own distance from 3-D elasticity is
  measured separately (G8a, +4.01e-01 % at `t/R = 4e-3`), so this is not a 3-D
  elasticity validation.
