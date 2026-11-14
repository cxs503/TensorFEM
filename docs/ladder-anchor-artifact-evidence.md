# Ladder anchors: the certificate now names committed evidence

Mission m20321 (re-open of TensorFEM#848, closed owner 2026-11-05). The original
PR repaired the *numbers* of the two refinement-ladder anchors so that every gate
carries an `operator`/`threshold` pair a third party can recompute. That code-level
repair is already on `main` (`src/tensorfem/refinement_gate.py` emits `operator`,
grades monotonicity as `violations <= 0` and reports the measured finest-mesh error
against the GCI band instead of copying the band into `actual`). What was still
missing on `main` is the other half of a certificate: the **evidence the numbers
were read from**.

## What the registry looked like before this change

`tools/verified_anchor_audit.py` found, on the unchanged `main`:

| anchor | artifact state | audit code |
|---|---|---|
| `fem_heat_conduction_l2` | declared `fig_convergence.png`, `fig_field.png`, neither committed | `E_ARTIFACT_MISSING` |
| `fem_pinched_cylinder` | wrote `convergence.png`, `displacement_comparison.png`, declared nothing | `W_UNDECLARED_ARTIFACTS` |
| `fem_shell_scordelis_lo` | wrote `scordelis_lo_convergence.png`, declared nothing | `W_UNDECLARED_ARTIFACTS` |

No directory under `benchmarks/verified/` shipped a committed image at all, so every
figure a verifier would want to look at existed only in the run directory of whoever
produced it, and `tests/test_verified_anchor_certificates.py` was red on `main` because
`fem_heat_conduction_l2` carried an unregistered defect.

## What this change does

1. Both ladder runners now emit an `artifacts` map naming the files that back the gate
   verdicts (`result.json`, `gates_verified.json`, the ladder chart, and for
   Scordelis-Lo the two diagnostic studies). The declaration is written by the
   generator, so it cannot drift from the run.
2. The five figures are committed next to the numbers they visualise:
   * `fem_pinched_cylinder/convergence.png`, `displacement_comparison.png`
   * `fem_shell_scordelis_lo/scordelis_lo_convergence.png`
   * `fem_heat_conduction_l2/fig_convergence.png`, `fig_field.png`
3. `tests/test_anchor_artifact_evidence.py` pins the contract: a certificate must
   name a figure, every named path must exist and be git-tracked, the figure must be a
   non-trivial PNG, any committed figure must be declared (auto-discovery), and the
   auditor must grade the artifacts clean.

## Reproduction (every number below was produced by these commands)

```bash
PYTHONPATH=src python benchmarks/verified/fem_pinched_cylinder/run_verification.py
PYTHONPATH=src python benchmarks/verified/fem_shell_scordelis_lo/run_verification.py
PYTHONPATH=src python benchmarks/verified/fem_heat_conduction_l2/run.py
python tools/verified_anchor_audit.py --write     # refresh certificates + debt + table
python tools/verified_anchor_audit.py --check     # 0 registered defect drift
```

The reruns reproduced the committed numbers to the last digit reproduced by the
blas layer (differences at 1e-16 relative, e.g. `-6.260527590930111e-06` ->
`-6.260527590930112e-06`); no gate verdict changed.

## Verdicts are unchanged and remain honest FAILs

| anchor | quantity | value | criterion | verdict |
|---|---|---:|---|---|
| `fem_pinched_cylinder` | finest-mesh deflection error (32x32) | 15.97 % | <= 3 % | FAIL |
| `fem_pinched_cylinder` | Richardson limit error | 163.1 % | <= 3 % | FAIL |
| `fem_pinched_cylinder` | observed order (8x8 -> 32x32) | 0.177 | >= 1.0 | FAIL |
| `fem_shell_scordelis_lo` | flat-facet finest error | 49.40 % | <= 3 % | FAIL |
| `fem_shell_scordelis_lo` | cylindrical finest error | 54.47 % | <= 3 % | FAIL |
| `fem_heat_conduction_l2` | relative L2 error (32x32) | 4.618e-04 | <= 3 % | PASS |
| `fem_heat_conduction_l2` | observed L2 order | 1.876 | >= 1.8 | PASS |

Committing the figures does not upgrade any claim: the two shell ladders stay
honest FAILs against the 3 % criterion, and the images now make the reason visible
(pre-asymptotic / non-monotone ladders).

## Residual debt this change does **not** touch

`fem_thermal` and `fem_sent_j_integral` still declare artifacts that are not
committed (`E_ARTIFACT_MISSING`), four anchors still ship no gate file, and
`fem_structural_acoustics` still fails the schema/`E_PASS_WITHOUT_CRITERION`
checks. All of them are registered in `anchor_certificate_debt.json`; this change
adds no new debt (the registry is byte-identical) and removes none.

## Known pre-existing reds (not introduced, not fixed here)

`tests/test_pinched_cylinder_maturity.py` (stale `cylindrical_shell4.py` hash pin,
missing `case_*.png`), `tests/test_verified_fem_anchors.py` (pins obsolete gate ids
`G1_finest_accuracy` / `G2_monotone_convergence`), and
`tests/test_scordelis_lo_verification.py::test_cylindrical_shell4_passes_accuracy_gate`
(current cylindrical Shell4 measures 88.7 % error on that 4x4 case) fail on `main`
for reasons unrelated to this evidence change; they belong to the element-repair
line of work, not to the certificate line.
