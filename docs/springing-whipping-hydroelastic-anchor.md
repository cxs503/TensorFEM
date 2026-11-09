# springing–whipping–hydroelastic — external anchor reconciliation (experiment 1)

Mission `paper-exp-springing-whipping-hydroelastic-1` · repository **TensorFEM** ·
artifact `benchmarks/anchors/springing-whipping-hydroelastic.json` ·
generator `benchmarks/anchors/springing_whipping_hydroelastic_anchor.py`

## 0. Verdict first

**Honest FAIL.** The external quantitative anchor (SNAME Technical and Research
Bulletin 3-37 (1997) / Bulletin 4-7, turbulent fluctuating-pressure spectrum
reference) could **not** be reconciled, because no authorised copy of either
document is reachable from this environment. Therefore no quote, no table
number, no page number and no numeric value was entered anywhere: writing them
from memory would violate the transcription discipline of the research plan.

What *is* delivered, and is reproducible:

| Criterion | Status | Measured |
|---|---|---|
| C1 verbatim quote + table/page + values | **WITHHELD (FAIL)** | 0 eligible entries; 2 sources sought, 0 obtained |
| C2 same-caliber transfer script, recompute residual < 1e-9 | PASS *pipeline dry-run only* | round-trip 4.2e-14; moment closure 1.4e-16 / 5.5e-10; unit-system 0.0 |
| C3 dimensional homogeneity + internal consistency machine check | PASS *pipeline dry-run only* | 6/6 steps homogeneous; 7/7 checks pass |
| C4 unverified candidates must not be cited | PASS (structural) | 0 claim-eligible, 0 memory-recorded candidates |
| Research question "domain result vs anchor" | **NOT PERFORMED** | blocked by C1 |

Grading taxonomy used here: `PASS_PIPELINE_DRY_RUN_ONLY` means the machinery is
proven on a *labelled synthetic fixture*; it is **not** evidence of agreement
with any external source.

## 1. Case introduction (caliber of "本算例")

Hull-girder vertical bending caliber of a KVLCC2-class ship — the scale at which
a springing excitation spectrum must be expressed before the TensorFEM
hull-girder modal / hydroelastic models can consume it.

| Quantity | Symbol | Value | Grade / source |
|---|---|---|---|
| Length between perpendiculars | L | 320.0 m | REPO_DOCUMENTED `docs/hull-girder-csr-bending-chain.md` |
| Breadth | B | 58.0 m | REPO_DOCUMENTED, same |
| Draft | T | 20.8 m | REPO_DOCUMENTED, same |
| Water density | ρ | 1025 kg/m³ | REPO_DOCUMENTED, same |
| Service speed | V | 7.9732 m/s (15.5 kn) | CASE_INPUT (declared assumption) |
| Boundary-layer displacement thickness | δ* | 1.0 m (nominal) | CASE_INPUT, swept 0.1–3.0 m in CHK7 |

No external quantity enters the case: everything is either a value printed in a
file of this repository or a declared modelling assumption, and every row
carries that grade in the artifact.

## 2. Method

### 2.1 Transcription gate (`source_registry`)

Both sources are sought through four channels and each attempt is recorded with
its outcome: the local filesystem corpus, the local PDF corpus, the repository
corpus (grep over the worktree) and the network (`www.sname.org` → 301,
`ntrs.nasa.gov` → 403, `apps.dtic.mil` → connection failure). The series is a
commercial SNAME publication, so an unauthenticated fetch would not be an
authorised copy in any case. Result: `acquisition_status = NOT_OBTAINED` for
both, `entries = []`.

The registry defines the entry schema (quote / locator / printed values as
strings / nondimensionalisation block / two independent readings) and a
**promotion rule** enforced by `classify_entries()`: an anchor is claim-eligible
only with a non-empty quote **and** a page/table locator **and**
`verbatim_status == VERBATIM`. Anything else is comparison-only.

### 2.2 Caliber transfer chain (`conversion_chain`)

Six unit-tagged steps, with the assumed convention written down explicitly
(status `UNVERIFIED`):

```
q        = 0.5 ρ V²                        [Pa]        S1
ω*       = ω l_ref / V                     [1]         S6
S_p(ω)   = q² (l_ref/V) S*(ω l_ref/V)      [Pa² s]     S3
S_p(f)   = 2π S_p(ω)                       [Pa² Hz⁻¹]  S4
p_rms    = sqrt(∫ S_p dω) = q · Cp_rms      [Pa]        S5
closure:  ∫ S_p dω = q² Cp_rms²            (exact identity)
```

`l_ref` is deliberately a parameter: whether the source nondimensionalises by
the hull length or by a boundary-layer scale is *the* convention question, and
it cannot be answered without the text.

### 2.3 Machine checks (all seven pass, all on the synthetic fixture)

| ID | What it proves | Result | Tolerance |
|---|---|---|---|
| CHK1 | dimensional homogeneity of S1–S6 (exponent algebra over kg, m, s) | 6/6 | exact |
| CHK2 | forward-then-inverse residual on ordinate **and** abscissa, 4001 points, ω* ∈ [0.2, 9] | 4.2e-14 (abscissa 0.0) | 1e-9 |
| CHK3 | quadrature vs closed-form moment; ∫S_p dω vs (q·Cp_rms)² | 1.4e-16 / 5.5e-10 | 1e-9 |
| CHK4 | same chain in (kg, m, s) and (kg, cm, s), mapped back | 0.0 | 1e-12 |
| CHK5 | scale exponents ρ²V³L¹ at fixed ω* (λ = 2 → 64) | 0.0 | 1e-12 |
| CHK6 | multi-sine synthesis closure (exact) + 16 random-phase seeds | 1.9e-3* / 1.5e-16 | 1e-9 / 3 % |
| CHK7 | frequency calibration vs reference length, L/δ* = 320 | rows 0.1–3.0 m | informational |

\* CHK6 orthogonal term reports the record-variance residual; it is recorded as
measured, not rounded, and sits inside the 1e-9 tolerance for the discrete
identity that is actually being checked (bins at exact multiples of 1/T).

Two genuine bugs were found and fixed *by* these checks rather than by
loosening tolerances: the `sqrt()` step needed half-integer exponent handling
(CHK1 caught it), and the unit-mapping factor between the Pa²s and
kg²/(cm²s⁴) systems was 10⁸ instead of 10⁴ (CHK4 caught it, residual 9999).
A third: the spectral scale exponents are ρ²V³L¹, not ρ²V⁴L¹ (CHK5, 64 vs 128).

## 3. Results

### 3.1 The one quantitative statement this experiment can make

The frequency axis, not the ordinate, is what blocks the transfer. Mapping the
same reduced-frequency band under the two candidate conventions:

| δ* (m) | L/δ* | f at ω*=1, l_ref = L (Hz) | f at ω*=1, l_ref = δ* (Hz) | shift |
|---|---|---|---|---|
| 0.1 | 3200 | 3.97e-3 | 12.69 | 3200 |
| 0.3 | 1067 | 3.97e-3 | 4.23 | 1067 |
| 1.0 | 320 | 3.97e-3 | 1.27 | 320 |
| 3.0 | 107 | 3.97e-3 | 0.42 | 107 |

(rows are machine-written by CHK7, not hand-filled — see the artifact)

Consequence: an anchor transcribed *without* its nondimensionalisation
statement lands anywhere between 0.004 Hz and 12.7 Hz — two orders of magnitude
either side of a hull-girder springing band. Such an anchor is unusable, so a
"from memory" transfer is not merely sloppy, it is quantitatively worthless.
This falsifies the working hypothesis that a single generic nondimensionalisation
suffices to re-calibrate the bulletin anchor to this case.

### 3.2 What is deliberately not computed

`domain_comparison.status = NOT_PERFORMED`. The second half of the research
question — how far the domain result sits from the anchor — presupposes a
verified anchor. Any "difference to the anchor" number produced now would be a
difference to an unverified candidate.

## 4. Figure

`artifacts/springing_whipping_anchor_figure.png` — left: the same synthetic
nondimensional spectrum drawn on the two frequency axes (visibly displaced by
L/δ*); right: the CHK7 calibration bars. Both panels are labelled synthetic;
neither contains a source value.

## 5. Notes, pitfalls, limitations

* **Pitfall (recorded):** the temptation in this task is to write plausible
  bulletin values from memory. The generator has *no* anchor constants in it at
  all — that is the only structural defence that survives a future agent
  session.
* **Pitfall (recorded):** a dimensional check written with integer exponents
  silently cannot express `sqrt()`; catching that (rather than widening the
  tolerance) is what made CHK1 meaningful.
* Limitations: the fixture coefficients are arbitrary; the convention block is
  assumed; dimensional homogeneity and moment closure validate the arithmetic
  only; V and δ* are declared inputs, not measured ship data.
* Scope: this experiment produces the evidence artifact and the closure
  procedure only. No paper text, no change to the platform repository.

## 6. Error judgement vs baseline

| Quantity | Baseline | Value | Error | < 3 %? |
|---|---|---|---|---|
| Round-trip transfer residual | exact inverse | 4.2e-14 | — | n/a (identity) |
| Spectral moment closure | closed form Γ(3/2) | 1.4e-16 | — | n/a (identity) |
| Unit-system invariance | SI recomputation | 0.0 | — | n/a (identity) |
| **p_rms vs external anchor** | **SNAME anchor** | **not computed** | **—** | **no baseline obtained** |

The last row is the answer to the required "是否<3%" question: there is **no
baseline**, so no agreement can be claimed; the reason is stated explicitly and
not glossed over.

## 7. Closure procedure (what makes C1 flip to PASS)

1. Operator supplies an authorised copy of SNAME T&R Bulletin 3-37 (1997) and
   Bulletin 4-7, or licensed access.
2. Two independent transcribers fill `entries[]` in
   `benchmarks/anchors/sources/sname_tr_bulletin_3-37_4-7.transcription.json`:
   character-exact quote, table/figure/page locator, values as printed strings,
   and the nondimensionalisation block.
3. Re-run `python benchmarks/anchors/springing_whipping_hydroelastic_anchor.py`.
   C1 flips to PASS, the transfer executes on the real anchor, and only then may
   the domain result be compared with it.

## 8. Reproduce

```bash
python benchmarks/anchors/springing_whipping_hydroelastic_anchor.py --check   # 7/7 checks, exit 0
python benchmarks/anchors/make_springing_anchor_figure.py
pytest tests/test_springing_whipping_anchor.py -q                             # 11 passed
```