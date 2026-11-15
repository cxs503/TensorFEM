# CASE — GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE, round-2 reopen (PR #1050)

Anchor `GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE`. Repository TensorFEM. Owner closed
PR **#1050** on 2026-11-05; this note is the round-2 reopen accounting ("管到底",
second pass): what of #1050 is already in `main`, what was genuinely missing, and
what this pass fixed. Numbers in §3 are produced by the commands in §6, not typed
by hand.

## 1. Status of the #1050 deliverable in `main`

The five files named by the mission are present on `main` (checked with
`git ls-tree main`, `git log main -- <path>`):

| file | lines | landed in |
| --- | --- | --- |
| `src/tensorfem/grillage_plastic_collapse.py` | 876 | `4d1d118f` (m19472), merged as #938 |
| `src/tensorfem/grillage_plastic_verification.py` | 694 | `4d1d118f`, merged as #938 |
| `src/tensorfem/grillage_ice_ultimate.py` | 1070 | `6401143e` (#938), updated by `fdd5a9ca` (#1022) |
| `tests/test_grillage_plastic_collapse.py` | 149 | `6401143e` (#938) |
| `docs/CASE-GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE-round2.md` | 209 | `6401143e` (#938) |

So the *code and tests* of #1050 reached `main` through the m19472 deliverables in
#938 (and #1022 later touched the sibling elastic module); what did **not** reach
`main` is the machine-readable evidence the report points at, because
`artifacts/` is filtered by the merge gate. That is audit finding **F6** of
`docs/AUDIT-GAP-PP-5-GRILLAGE-ULTIMATE-independent-review.md`, and it is the one
piece of the #1050 acceptance that was still open.

## 2. What this pass changed (minimal, one feature point)

1. **The "no hand-typed number" claim is now a check.**
   `tests/test_grillage_plastic_doc_artifact_gate.py` parses the report's five
   printed tables and re-derives, from the report's own cells, every relation it
   asserts:
   * each gate row's printed error must sit inside its printed budget, and the
     printed verdict must agree with that comparison;
   * the reported ultimate state must equal `min(member grillage, plate-field
     limit)` and the printed governing mode must name the branch that produced
     the minimum;
   * the printed margin must equal `ultimate / 4.000 MPa`;
   * the 3x3 sensitivity sweep must keep its design and the same union rule in
     all nine rows;
   * the mesh-convergence sweep must decrease and stay in the `O(h^2)` band.
   These run in the fast lane (5 tests, 0.20 s, no solver call).
2. **The report↔code link F6 could not check is the slow-lane gate.**
   `test_round2_report_numbers_reproduce_from_current_code` (§4 of the module)
   reruns `tensorfem.grillage_plastic_verification` from the current code and
   requires **every** printed number — 21 gate rows, both case-study tables, the
   convergence sweep, the 3x3 sensitivity sweep — to be reproduced to printing
   precision. Registered in `tests/slow_lane_manifest.txt`.
3. **The stale artifact pointer is fixed at the source.** §8 of the round-2
   report now says that the module's output directory is a command-line argument,
   that the default `artifacts/` is merge-gate filtered, and that passing a
   tracked directory (`docs/grillage_plastic_collapse`, the convention
   `docs/grillage_ice_ultimate/` already uses) keeps the evidence in the tree.

## 3. Verification actually run

```
PYTHONPATH=src python -m pytest tests/test_grillage_plastic_doc_artifact_gate.py -q -m "not slow"
# 5 passed, 1 deselected in 0.20 s

PYTHONPATH=src python -m pytest tests/test_grillage_plastic_collapse.py -q
# 10 passed in 74.74 s
```

The slow-lane test (`tests/test_grillage_plastic_doc_artifact_gate.py::test_round2_report_numbers_reproduce_from_current_code`)
was **not** completed inside this pass: its recompute was submitted to the
scheduler twice (`cj3582` 8 cores, `cj3598` 1 core, `stage_dir` = this worktree)
and both were still `queued` behind ~50 in-flight jobs when the pass ended, and the
local machine is barred from the multi-minute run by the compute-dispatch order.
Honest status: the fast-lane half is green, the slow half is registered and
unrun-here — it runs in CI's full lane (`pytest -q --timeout=1800`).

## 4. Error judgement of the gated quantities (existing 21 gates, re-read)

The round-2 report's own table, re-checked cell by cell by the fast lane, records
these worst errors against the predeclared budgets:

| quantity | reference | worst printed error | budget | < 3 % ? |
| --- | --- | --- | --- | --- |
| SS beam collapse load (V1) | 8 Mp/L² closed form | −0.2597 % | 1 % | yes |
| SS central-patch collapse (V2) | 8 Mp/(2Lc−c²) | −0.3576 % | 1 % | yes |
| clamped beam collapse (V3) | 16 Mp/L² | −0.2597 % | 1 % | yes |
| two-span mechanism (V4) | virtual-work minimum | −0.0441 % | 2 % | yes |
| elastic path (V5) | qL²/8, 5qL⁴/384EI | +0.0651 % | 0.2 % / 1 % | yes |
| axial interaction (V6) | Mp(1−(P/Py)²) | −0.2597 % | 2 % | yes |
| case-study mesh (V7b) | n=8 vs n=16 | +2.1496 % | 5 % | yes |
| frame line vs closed form (V10) | F(2L−c)/8 | −1.0989 % | 2 % | yes |

No gated quantity exceeds 3 %; the largest is the 16-vs-8 element mesh difference
at +2.15 %, which is a discretisation difference, not a physics error, and the
physical limit load is gated by V1–V4 at ≤0.36 %.

## 5. Limitations of this pass (honest)

* **No physics was redone and no number was re-derived analytically**: this pass
  audits the report against the code, it does not re-derive the rigid-plastic
  theory. Point-wise independence of `orthotropic_plate_fd` and the closed forms
  was audited earlier and is not repeated here.
* **The evidence JSON is not committed by this pass**, so F6 is fixed in the
  report text and in CI behaviour rather than by adding the file. `artifacts/`
  output is genuinely reproducible, but the copy under a tracked path needs a
  scheduler-completed run; that remains the one open item (see §3).
* **`MemberSection._parts` web-from-mid-plane convention (audit F5)** is untouched:
  changing it would move every number already cited by rounds 2–6.
* The gate covers the round-2 tables only. Round 3–6 tables (eccentric patch
  V11–V15, patch-position map) are gated by their own documents and by
  `tests/test_grillage_plastic_collapse.py`.

## 6. Reproduction

```
PYTHONPATH=src python -m tensorfem.grillage_plastic_verification            # 33 gates, writes artifacts/
PYTHONPATH=src python -m tensorfem.grillage_plastic_verification docs/grillage_plastic_collapse  # keep evidence in-tree
PYTHONPATH=src python -m pytest tests/test_grillage_plastic_doc_artifact_gate.py -q                 # report<->code gate
PYTHONPATH=src python -m pytest tests/test_grillage_plastic_collapse.py tests/test_grillage_ice_ultimate.py -q
```