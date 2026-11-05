# PR #70 merge cure (mission m3835): lossless three-way merge of `agent/m3297` onto `main`

**Repository**: TensorFEM · **PR**: #70 (`agent/m3297` → `main`)
**Cure branch**: `agent/m3835` (identical tree to the cured `agent/m3297`; pushed back to the PR head branch)
**Artefacts**: `artifacts/pr70cure/` — `union_audit.py`, `union_audit.json`, `baseline_targeted.json`, `make_figures.py`, `figs/`

## 1. Case: what was dirty and why

`merge_governor` reported PR #70 as `mergeable=false`. The PR head had been re-created on an
older main base while `main` kept advancing, so the two histories diverged:

| role | commit | note |
|---|---|---|
| merge base | `14ca2fdc1fd0` | common ancestor of the two tips |
| `main` (target) | `7accc4b9cfb4` | 191 paths changed relative to the merge base |
| `agent/m3297` (PR head, before cure) | `092f326f9bb4` | 9 paths, 10 files: plate bending-buckling feature |
| merge commit produced by this cure | `5c10574ad843` | first parent = PR head, second parent = `main` |

The PR content is the non-uniform (linearly varying) in-plane prebuckling stress closure:
`src/tensorfem/plate_bending_buckling.py` (Ritz + independent Hermite C1 reference),
`src/tensorfem/plate_shear_buckling.py` (shared pencil congruent transform fix),
`tests/test_plate_bending_buckling.py`, `benchmarks/plate_bending_buckling*`,
`docs/plate-bending-buckling-validation.md`, three figures, and one line of
`docs/benchmark-qualification-status.md`.

## 2. Method (no mechanical overlay)

1. Fetch both tips from `origin`, recompute the merge base; never trust the cached
   `origin/main` of another worktree (`/root/tensor-ecosystem/TensorFEM` was stale at an
   *older* main, `e48fea8`, which is an ancestor, not a descendant, of the real tip).
2. `git merge --no-ff origin/main` on a branch cut from the true PR head → exactly **one**
   conflicted file, `docs/benchmark-qualification-status.md` (a single paragraph; 191 files
   were main-only additions, 9 branch-only additions, none else overlapped).
3. Resolve by **semantic union** at sentence granularity, not by choosing a side:
   * main's newer verdict is kept verbatim — the open-hole hemisphere shell now *has* an
     independent HEX20 oracle and **fails** the 3 % gate (mean 5.09 %, max 9.00 %);
   * the branch's claim replaces one clause of the "still missing" list — non-uniform
     prebuckling stress (pure in-plane bending) is **closed**, `k_b = 23.8818` vs literature
     23.9, see `plate-bending-buckling-validation.md`.
   The sentence "开孔半球壳仍缺独立应力参考" (base text, not branch-new) is superseded because
   it is now factually false; it is declared explicitly in the audit's `SUPERSEDED` table with
   this reason, so the supersession is auditable rather than silent.
4. Verify the resolution mechanically with `artifacts/pr70cure/union_audit.py`, a git-plumbing
   three-way audit over **all 1237 paths** (see §3), including two negative controls.

## 3. Results

### 3.1 Lossless content-union audit (`union_audit.json`)

| verdict class | paths |
|---|---|
| branch-only change, merged == branch blob | 9 |
| main-only change, merged == main blob | 191 |
| content conflict, sentence-level union audited | 1 |
| cure-only additions (this audit's own artefacts) | 2 |
| **silent content losses** | **0** |
| audited paths / verdict | 1237 / **PASS** |

Conflict-file detail (`docs/benchmark-qualification-status.md`):
base `72ffd9d` → branch `00f415f`, main `6d1ca69`, merged `b090fbb`;
segments added by branch 3, added by main 8, **dropped (branch) 0, dropped (main) 0,
resurrected 0**.

Negative controls (the checker must fail on lossy merges; both constructed from the same
inputs, audited with the same script):

| control | construction | audit outcome |
|---|---|---|
| A `5a32b7a` | conflict resolved by taking the branch side whole | **FAIL** — `CONFLICT_TAKEN_WHOLE_FROM_ONE_SIDE`, main's oracle verdict lost |
| B `b4c81fd` | mechanical overlay = main tree wholesale | **FAIL** — 9 × `LOST_BRANCH_CONTENT` + 1 × `BRANCH_ONLY_MODIFIED_IN_MERGE` (all PR #70 files gone) |

So the PASS in §3.1 is a discriminating result, not a vacuous one.

### 3.2 Tests (same runner, same env, no `-p no:randomly` shuffling)

| run | result |
|---|---|
| merge commit, full `tests/` (`-n 24`, `--timeout=900`) | **1702 passed, 2 failed, 7 skipped, 1 xpassed** in 558.6 s |
| `main` baseline, plate/buckling scope | 159 passed in 83.5 s |
| merge commit, plate/buckling scope | **170 passed** in 88.2 s (159 + the 11 new `test_plate_bending_buckling.py` tests) |

Both full-suite failures are **pre-existing on `main`** and were reproduced individually on a
clean `git archive` export of `origin/main`:

| test | on main | on merge |
|---|---|---|
| `test_release_quality.py::test_api_audit_script_passes_with_warnings_as_errors` | fails (1.5 s) | fails |
| `test_finite_strain_curved_friction_contact3d.py::test_real_curved_nonmatching_tet4_friction_level_one_closure` | fails (30.7 s) | fails |

No test that passes on `main` fails on the merge. The structural reason is in §3.1: for every
non-conflicting path the merged blob is byte-identical to the winning side's blob, and the
single union-resolved path is a `.md` file that is never executed.

**Honest limitation**: a *full-suite* baseline run on `main` was started with identical
settings but did not finish inside the mission window (one worker stalled at 98 % on the
shared host, 128 cores under concurrent load from other missions). The baseline is therefore
established by the identical-scope plate/buckling run plus individual reproduction of both
failures on `main`, not by a completed full-suite baseline. This is recorded in
`artifacts/pr70cure/baseline_targeted.json`, not papered over.

### 3.3 Physical re-verification of the merged-in content

The PR's own claim survives the merge (values read programmatically from
`benchmarks/plate_bending_buckling_evidence.json`):

| quantity | numerical | reference | relative error | 3 % gate |
|---|---|---|---|---|
| bending-buckling coefficient, a/b = 10 | 23.881813 | 23.900 (Timoshenko & Gere §9.2) | **0.0761 %** | pass |
| same, independent Hermite C1 reference | 23.880627 | — | vs Ritz **4.97e-5** | pass |
| uniform-compression limit k | 4.000000 | 4.000 | 2.2e-16 | pass |
| shear coefficient a/b = 1 regression | 9.32467 | 9.34 | 0.164 % | pass |

## 4. Figures

* `artifacts/pr70cure/figs/merge_topology.png` — path classes of the three-way merge.
* `artifacts/pr70cure/figs/test_outcomes.png` — merged vs baseline test counts (identical scope).
* `artifacts/pr70cure/figs/kb_reverification.png` — left: `k_b` vs aspect ratio collapsing onto
  23.9; right: independent Hermite C1 reference error and eigen-residual vs refinement.

## 5. Notes / pitfalls

* **Stale `origin/main` in sibling worktrees.** `/root/tensor-ecosystem/TensorFEM` reported
  `main = e48fea8`, which is an *ancestor* of the real tip `7accc4b`. Always `ls-remote` the
  live remote before deciding what "dirty against main" means.
* **Never resolve a doc paragraph by side selection.** Both sides edited the same paragraph for
  different reasons (one added a negative verdict, the other closed a different item); taking
  either side silently deletes the other's science. Sentence-level union is the only lossless
  answer, and the negative controls in §3.1 show what the lossy alternatives look like.
* **Audit must be able to fail.** A union audit that cannot detect the `-X ours` overlay is
  worthless; hence the two synthetic lossy merges.
* Scope limit: this cure only makes the histories mergeable and preserves content. It does not
  re-qualify the shell geometric-stiffness gap, Hertz contact, or the two pre-existing failing
  tests, which remain open on `main` independently of PR #70.

## 6. Reproduce

```bash
MB=$(git merge-base origin/main origin/agent/m3297)
git checkout -B cure origin/agent/m3297 && git merge --no-ff origin/main   # 1 conflict
# ... resolve docs/benchmark-qualification-status.md as the sentence union ...
python3 artifacts/pr70cure/union_audit.py --base $MB --branch origin/agent/m3297 \
        --main origin/main --merged HEAD --json-out artifacts/pr70cure/union_audit.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python3 -m pytest tests -q -n 24 --timeout=900
python3 artifacts/pr70cure/make_figures.py /tmp/merged.log /tmp/base.log
```