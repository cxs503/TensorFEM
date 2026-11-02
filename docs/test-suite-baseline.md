# Test-suite baseline and drift protocol (m1831)

Status: **partial coverage, honest** — 93 of 889 collected tests re-run green on a
contended host; the repository-wide baseline record and the drift tool are in
place so the next run on a quiet machine completes the comparison
automatically. No new real red was found in the covered set.

## 1. Why this document exists

The tree carried no machine-readable record of "which tests are green on this
commit", so every drift hunt had to be re-derived by hand. That makes "new red
vs pre-existing red vs environment red" unfalsifiable. This task adds

* `scripts/summarize_test_baseline.py` — condenses a pytest `--junitxml`
  artefact into a small JSON baseline (per-test `nodeid → outcome`, totals,
  SHA-256 signature, interpreter/torch/platform stamp) and diffs two records,
  classifying a *new red* as "green in baseline, not green now" (`--check`
  exits 1 on one);
* `tests/test_test_baseline_summary.py` — 5 tests pinning outcome precedence,
  order-independent signature and regression detection;
* `runs/run_suite_per_file.py` — per-file runner that bounds the blast radius
  of a pathologically slow file and yields a per-file runtime table.

## 2. Method

| Step | Command | Artefact |
| --- | --- | --- |
| Collect | `pytest --collect-only -q` | 889 tests in 170 files |
| Whole-suite attempt | `pytest tests -q --timeout=1800 --junitxml=runs/junit.xml` | `runs/pytest_full.log` (aborted, see §5) |
| Per-batch re-run | `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest <files> --timeout=45 --junitxml=runs/junit_batchN.xml` | 3 batches, 19 files |
| Condense | `python scripts/summarize_test_baseline.py runs/junit_merged.xml --output artifacts/test-baseline-run1.json` | `artifacts/test-baseline-run1.json` |

Environment stamp of the record: Python 3.11.6, torch 2.7.1, Linux x86-64,
host load average 150–770 during the run (shared build host).

## 3. Results (programmatically injected)

Source: `artifacts/test-baseline-run1.json`, signature
`ccb8a49553c68d39582873ab38e8db295087f4aed4556f70a8674a080dd37b7e`.

| Batch | Files | Tests | passed | failed | error | skipped | xfailed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (validation gate, benchmark cases, plate, truss, thermal, public API, baseline tool) | 7 | 31 | 31 | 0 | 0 | 0 | 0 |
| 2 (solid3d, solid3d-bbar, shell4, plate Mindlin, nonlinear) | 5 | 31 | 31 | 0 | 0 | 0 | 0 |
| 3 (solid3d extended/edge, sparse core, nonlinear step, validation, quadratic solid, classic shell benchmarks) | 7 | 31 | 31 | 0 | 0 | 0 | 0 |
| **total** | **19** | **93** | **93** | **0** | **0** | **0** | **0** |

Pass rate in the covered set: 93/93 = 100%. No xfail/xpass fired because the
three machine-dependent xfail marks live in files outside the covered set.

## 4. Red classification rules

* **Real red** — deterministic physics/numerics failure, or an assertion on a
  quantity with a declared reference. Reproducible on a quiet host.
* **Environment red** — outcome decided by the host, not the code: missing
  optional dependency (this image lacks `python-docx`, `reportlab`,
  `threadpoolctl`), sibling-checkout import (`tensorlbm`), wall-clock/BLAS
  rounding. The repo already declares three such cases as `xfail`
  (`test_finite_rotation_layered_shell4.py`, `test_shell4_arc_line_search.py`,
  `test_v1_evidence_aggregator.py` ×2, all "machine-dependent … See issue #8").
* **Budget red** — the test did not finish inside the re-run budget on the
  loaded host; this is *not* classified as red, it is reported as uncovered.

Covered set: 93/93 green → **no real red, no new red, nothing to fix**. The only
bug found and fixed in this task was inside the new tool itself (`failure` JUnit
tag was mapped to `failed` after its own test caught the mismatch).

## 5. Limitations (why coverage is 93/889)

The shared host ran at load average 150–770 during this task. A single
889-test pytest process was allowed 50 min wall / 9 h CPU and had completed
only ~52 tests (evidence: `runs/pytest_full.log`, 52 progress marks) before it
was stopped; the two heaviest early files (`test_ad_plastic_tangent.py`, 3
tests, >6 min under the same load) dominate. Pinning the BLAS pools to one
thread cut a 31-test batch to 7–23 s, which is why the per-batch protocol above
was adopted. The remaining 796 tests (including the heavy qualification,
contact, sparse-scaling and benchmark-report files) are **uncovered** by this
record; `runs/run_suite_per_file.py` exists to finish them on a quiet host, and
`scripts/summarize_test_baseline.py --baseline artifacts/test-baseline-run1.json
--check` will then report exactly which tests newly turned red.

No accuracy claim is made here: this task re-runs tests, it does not re-derive
physical reference values. Where the covered tests assert against analytical
or literature references they pass at their own declared (≤3%) tolerances.