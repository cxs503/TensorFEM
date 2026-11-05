# PR #130 CI cure: PR #87 cross-check must be reproducible, not shipped

## Symptom
`test (3.11)` and `test (3.12)` both failed on PR #130 (head `e4b44dc`) with

```
tests/test_marine_large_angle_stability.py::test_both_evidence_schemas_survive_the_merge
FileNotFoundError: .../artifacts/pr87_merge_cross_check.json
```

`lint (advisory)` stays red on the tree-wide pre-existing ruff backlog
(continue-on-error, not a merge gate).

## Root cause
The delivery gate filtered `artifacts/pr87_merge_cross_check.json` out of the
published branch ("1 个文件被门禁滤除"). The test loaded the payload straight
from that file, so the gate-filtered checkout could never pass: the gate that
protects the branch and the test that verifies it disagreed about whether the
JSON exists. The gate is right (artifacts are regenerable), so the test had to
stop depending on the shipped file.

## Fix (minimal, two files)
* `benchmarks/pr87_merge_cross_check.py`: the payload builder is extracted as
  `build_payload(verbose=True)`; `main()` now only calls it, writes the JSON and
  the figure. CLI output and artifact bytes are unchanged (same aggregation,
  same `verdict` key).
* `tests/test_marine_large_angle_stability.py`: `_pr87_cross_check()` imports
  the benchmark by path and recomputes the payload in-process; if the JSON is
  present it is used only as a witness (`schema`, `overall`, `verdict` must
  agree), never as the sole source. The cross-check is deterministic and runs
  in well under a second, so the recomputation is the primary route.

## Evidence
Reproduced the CI failure locally first on the real PR head with `artifacts/`
absent (1 failed), then after the fix:

```
pytest tests/test_marine_large_angle_stability.py           -> 53 passed
pytest ...::test_both_evidence_schemas_survive_the_merge    -> passes both with
                                                               and without the
                                                               artifact present
python benchmarks/pr87_merge_cross_check.py                 -> verdict: PASS
```

Cross-check numbers (max |closed form - centroid| over 41 heels per geometry):

| case | limit (deg) | |closed-centroid| (m) | |oracle-centroid| (m) | linear-model err |
|---|---|---|---|---|
| slender_deck_limited | 11.310 | 1.665e-15 | 2.442e-15 | 2.67% |
| wide_shallow_low_gm | 21.801 | 2.670e-14 | 3.419e-14 | 47.62% |
| narrow_deep_section | 18.435 | 1.088e-14 | 1.665e-14 | 6.37% |

Gate is 1e-9 relative: achieved 1.3e-14 worst case (relative), i.e. five
orders of magnitude margin. The large linear-model error on the wide/shallow
low-GM barge is the documented limitation the module measures, not a
regression: `linear_gz` is the merged hydrostatics model, the exact arm is the
reference.
