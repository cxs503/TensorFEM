# Re-grading a refinement-ladder certificate from its own artifact

Mission m16857 (#848) rebuilt the refinement-ladder anchor certificates so that a
reader who never runs the solver can re-derive their verdict. This page records
the remaining half of that contract and the guard that enforces it.

## The defect this closes

An anchor ships two files that must agree:

| file | role |
|---|---|
| `benchmarks/verified/<anchor>/result.json` | the measured ladder (`levels`, `reference`) and the graded gates |
| `benchmarks/verified/<anchor>/gates_verified.json` | the `tensorfem.verified-gates.v1` envelope the certificate audit reads |

`tools/verified_anchor_audit.py` checks that every **gate** is recomputable from
its own `actual`/`threshold`/`operator` pair. That is a per-gate self-consistency
test, and it cannot see the failure mode that matters for a published anchor:

* a gate that agrees with itself but no longer matches the ladder it claims to
  grade (stale gate after a refreshed `result.json`);
* a threshold edited by hand (`0.03` → `0.30`) — every gate still "agrees with
  itself", the certificate silently gets 10× looser;
* a `criteria` block that only exists as prose in `gates_verified.json`, so the
  grading thresholds the run actually used are nowhere in the machine-readable
  artifact. The reader is then forced to recover `min_order` / `coarsest_sanity`
  / `tolerance` *from the gates they are supposed to check* — circular, and
  unfalsifiable by construction.

Before this change all three shipped ladder blocks (`fem_pinched_cylinder`,
`fem_shell_scordelis_lo` flat-facet, `fem_shell_scordelis_lo` cylindrical) were
in the third state: the thresholds lived only in the gates, so
"recomputable" meant "recomputable if you already trust the answer".

## What changed

1. `tensorfem.refinement_gate.evaluate_reference_ladder` now emits a
   machine-readable `criteria` block next to `levels`/`reference`:
   `qoi`, `reference_label`, `reference_value`, `tolerance`, `min_order`,
   `coarsest_sanity`. Everything the grader needs is in the artifact; nothing is
   recovered from the gates.
2. `tensorfem.refinement_gate.regrade_ladder(report)` re-grades a persisted block
   from `levels` + `criteria` only (no solver, no benchmark module, no per-anchor
   recipe), and `certificate_drift(report)` returns every field that disagrees
   with the re-grade. A block without `criteria`, with fewer than three levels,
   or with `reference.value != criteria.reference_value` raises
   `LadderCertificateError` — it fails closed instead of grading against guessed
   thresholds.
3. `tools/regrade_ladder_certificate.py` sweeps every
   `tensorfem.refinement-gate/1.0` block under `benchmarks/verified/*/result.json`,
   re-grades it, and cross-checks the `tensorfem.verified-gates.v1` envelope gate
   by gate (`status`, `operator`, `actual`, `threshold`) and criterion by
   criterion (`tolerance`, `min_order`, `coarsest_sanity`).

   ```
   python tools/regrade_ladder_certificate.py            # table
   python tools/regrade_ladder_certificate.py --check    # exit 1 on any drift
   python tools/regrade_ladder_certificate.py --json     # machine record
   ```

4. Both anchors were re-run through their own `run_verification.py`
   (`PYTHONPATH=src python benchmarks/verified/<anchor>/run_verification.py`) so
   the shipped blocks carry `criteria`. The re-run reproduced **every** measured
   number bit-for-bit; the only artifact change is the added `criteria` block
   (+8 lines pinched cylinder, +16 lines Scordelis-Lo).

## Result

Sweep record (`tools/regrade_ladder_certificate.py`, mission m20291):

| metric | before | after |
|---|---:|---:|
| ladder blocks discovered | 3 | 3 |
| ladder gates re-graded | 0 (no criteria → fail-closed) | 18 |
| gates reproduced bit-for-bit | — | 18 |
| drift fields | 3 blocks ungradeable | **0** |
| envelope gates disagreeing with their ladder | 12 (blocked by (1)) | **0** |
| `--check` exit code | 1 | **0** |

Per-block status is unchanged and stays honest-FAIL:

| block | status | finest error | certified (Richardson) error |
|---|---|---:|---:|
| `fem_pinched_cylinder` | FAIL | 15.97 % | 163.14 % |
| `fem_shell_scordelis_lo/flat_facet` | FAIL | 49.40 % | 53.19 % |
| `fem_shell_scordelis_lo/cylindrical` | FAIL | 54.47 % | n/a (order p < 0) |

The delta is **certificate reproducibility, not accuracy**: no threshold was
loosened, no verdict was promoted. `tools/verified_anchor_audit.py --check` still
passes and `anchor_certificates.json` is byte-identical, because the envelopes did
not change.

## Guard, not hope

`tests/test_ladder_certificate_regrade.py` parametrises over every discovered
block and adds negative controls, so the guard cannot rot into a no-op:

* every block must carry a complete `criteria` block;
* the shipped gates must be reproduced bit-for-bit, including
  `reported_accuracy_finest` and `certified_accuracy`;
* a hand-edited threshold, a flipped verdict, a stale level value or a changed
  `tolerance` must each produce drift (four separate controls);
* a missing `criteria`, a mismatched `reference.value`, or a wrong schema must
  raise `LadderCertificateError`;
* an envelope whose `actual` disagrees with its ladder must be reported;
* the audit sweep and `--check` must stay clean on the shipped registry.

One matcher trap is pinned by its own test: Scordelis-Lo ships *two* ladders under
one envelope (`flat_facet.*` and `cylindrical.*`), whose gate ids differ only by
the ladder-name prefix. Stripping the prefix collapses them and compares the
cylindrical block against flat-facet gates — a false "ok". The tool keys the
envelope lookup by the block's position in `result.json`
(`/ladder_grading/cylindrical` → `cylindrical.`) and the test asserts both ladders
match.

## Limitations

* **Scope.** Only `benchmarks/verified/*/result.json` carrying a
  `tensorfem.refinement-gate/1.0` block is covered (3 blocks, 18 gates today).
  Anchors certified by `tools/certify_ladder_anchor.py` use a per-anchor recipe
  and are out of scope; so is the `tensorfem.verified-gates.v1` envelope of
  `fem_structural_acoustics`, which is not ladder-shaped.
* **WITHHELD gates stay number-less.** `accuracy_asymptotic` and
  `asymptotic_range_consistency` on the Scordelis-Lo cylindrical ladder are
  `WITHHELD` with `actual=None` (no Richardson limit exists for a negative
  observed order). The re-grade reproduces that absence exactly. Writing a
  surrogate number into a withheld gate would make the verdict recompute to
  `FAIL ≠ WITHHELD` and be flagged as a certificate contradiction, and would
  overstate the evidence; the audit counts these as
  `W_UNRECOMPUTABLE_FAIL` *warnings* on a still-certified anchor, which is the
  correct classification for missing evidence.
* **Reproducibility, not validity.** A certificate that reproduces bit-for-bit
  proves the artifact is self-consistent; it says nothing about whether the
  element matches the experiment. These two anchors remain 15.97 % / 49–54 %
  away from the published reference — the accuracy debt is unchanged and lives
  in the shell element, not in the certificate.