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

## Round 2 (fix mission 4154): land the cure on the PR ref, verified on the PR head tree

Round 1 produced this same two-file cure on `agent/m4100`.  That ref is not PR
#130's ref, so the PR stayed red: its head `e4b44dc` still did

```python
cross = json.loads((ROOT / "artifacts" / "pr87_merge_cross_check.json").read_text(...))
```

and that file is not in the published tree.  Round 2 applies the cure directly on
`e4b44dc`.  Both touched files are byte-identical between `73f976b^` (round 1's
base) and `e4b44dc`, so the cure transfers as a clean cherry-pick: no conflict
resolution, no unrelated file touched (3 files: benchmark, test, this note).

### Local reproduction of the red, on PR head `e4b44dc`, `artifacts/` absent exactly as published

```
$ python3 -m pytest tests/test_marine_large_angle_stability.py -x -q
FAILED tests/test_marine_large_angle_stability.py::test_both_evidence_schemas_survive_the_merge
  FileNotFoundError: [Errno 2] No such file or directory:
  '.../artifacts/pr87_merge_cross_check.json'
1 failed, 52 passed in 15.03s
```

This is the same traceback the two CI test jobs reported, reproduced from the
branch tree alone (no `artifacts/` directory contents).

### Verification after the cure (same tree, python 3.11.6 / torch 2.7.1)

| check | command | result |
|---|---|---|
| artifact absent (published-branch shape) | `pytest tests/test_marine_large_angle_stability.py -q` | 53 passed |
| artifact present (witness path) | shipped JSON copied in, same test file | 53 passed |
| benchmark CLI still writes the same bytes | `main()` with `ARTIFACTS`/`ASSETS` redirected to a temp dir | `verdict: PASS`, JSON byte-identical (after strip) to `artifacts/pr87_merge_cross_check.json` on the delivery branch `agent/m3893` |
| figure still produced | same run | `cross_check.png` written (72 355 B tracked copy unchanged) |
| no new lint findings | `ruff check` on both touched files | 8 findings before, 8 after (pre-existing duplicated-import block); benchmark file clean before and after |

The witness path is what keeps the check honest when the artifact *is* present:
`schema`, `overall` and `verdict` of the shipped JSON must agree with the
recomputed payload, so a stale or hand-edited artifact still fails.

### Lint (advisory): pre-existing, tree-wide, and not a merge gate

The `lint (advisory)` job carries `continue-on-error: true` in
`.github/workflows/ci.yml` precisely because the tree has a large pre-existing
ruff backlog; on PR head `e4b44dc` `ruff check src tests scripts examples`
reports 4276 findings and `ruff format --check` wants 616 files reformatted.
Both files touched here already wanted reformatting *before* the patch, and the
added lines do not sit in those hunks; making that job green would mean
reformatting 616 unrelated files, which this mission is not allowed to do.

### Residual risk / not done
* Only the touched test file was re-run to completion here; the rest of the tree
  is untouched by this patch (the benchmark module is imported by no library
  code, only by the test via `importlib`), so the CI-wide suite is the backstop.
* The artifact is regenerable and stays out of the branch on purpose; do not
  re-add `artifacts/pr87_merge_cross_check.json` to a published branch.

### Exact local coverage (shared runner under memory stress, 2026-11-05)
* `tests/test_marine_large_angle_stability.py`: 53 passed with the artifact absent
  and 53 passed with it present.
* Targeted subset `tests/test_marine_*.py`, `tests/test_benchmark*.py`,
  `tests/test_external_benchmark_contracts.py`, `tests/test_verification_ledger.py`,
  `tests/test_engineering_cli.py` (26 files): 28 tests executed with zero failures
  before the runner was SIGTERM-ed at 11:22 under the shared machine's memory
  pressure (batched full-suite attempts were SIGKILL-ed, rc=137). No failure was
  observed in either run; the untouched tree is covered by CI, not by this run.
