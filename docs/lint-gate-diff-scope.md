# The `lint (advisory)` red: what it actually measured, and the diff-scoped cure

Status: **PR #97's lint red is closed** (its own files are clean and the job now
gates what a change touches). §6 lists what this deliberately does *not* fix.
Gate script: `scripts/lint_scope.py`. Workflow: `.github/workflows/ci.yml`
(`lint` job). Tests: `tests/test_lint_scope.py` (12). Evidence:
`artifacts/pr97-lint/before.json`, `artifacts/pr97-lint/after-vs-main.json`,
`artifacts/pr97-lint/after-push-base.json`, `artifacts/pr97-lint/gate_scope.png`.

## 1. Why this case

PR #97 (branch `agent/m3593`, head `5e1ca50`) had `test (3.11)`, `test (3.12)`
green and `lint (advisory)` red. The job ran two repo-wide commands:

```yaml
run: ruff check src tests scripts examples
run: ruff format --check src tests scripts examples
```

Neither number has anything to do with the pull request:

| measurement on the PR head `5e1ca50` | result |
|---|---|
| `ruff check src tests scripts examples` | **4270** findings (pre-existing backlog) |
| `ruff format --check src tests scripts examples` | **623** files would be reformatted, 13 already formatted |
| `ruff check` on the three Python files the PR touches | 1 finding (`E702`, `abaqus_io.py:112`) |
| `ruff format --check` on the same three files | 3 of 3 would be reformatted |

So the red mixed one genuine defect of the change with a tree-wide backlog that
fails identically on `main`. Because the job is `continue-on-error: true`, the
run conclusion is `success` while the check run is `failure` — a signal that is
both uninformative and impossible to make green by any single pull request.

## 2. Method

Two separate actions, kept separate on purpose.

**(a) The change's own debt: paid.** In `src/tensorfem/abaqus_io.py` the
`GENERATE` branch of the set reader packed three statements onto one line
(`E702`); it is now three statements. The PR's three Python files are then
`ruff format`-ed. Formatting is confined to files the PR already owns, so the
diff carries no unrelated churn (102/116/71 changed lines, all inside the three
files).

**(b) The measurement: scoped to the diff.** `scripts/lint_scope.py`:

* resolves the comparison base the way the CI event requires — pull requests
  against `origin/<base_ref>`, pushes against `github.event.before`, `HEAD^` as
  the last resort — and then takes the **merge base**, so a branch that merged a
  moved `main` is not blamed for `main`'s own history;
* collects the `*.py` files touched between that merge base and `HEAD`
  (`--diff-filter=ACMRT`, so deleted files are not linted), plus staged,
  unstaged and untracked files with `--worktree` for local use;
* runs `ruff check` **and** `ruff format --check` on exactly that list;
* measures the grandfathered tree-wide backlog with `--measure-backlog` and
  writes it into `lint-report.json`, but **never gates on it**;
* exit codes: `0` clean, `1` violations, `2` environment error — a missing
  `ruff`, an unresolvable base or a refused `git diff` is a *failure*, never a
  silent "clean".

`ci.yml` gains `fetch-depth: 0` (the merge base is unreachable in a depth-1
clone), a base-resolution step, the gate itself, `ruff==0.15.0` pinned in the
lint job so the format verdict is reproducible, and `lint-report.json` as an
artifact. The job keeps its name (`lint (advisory)`) and its
`continue-on-error: true`; the backlog is not touched.

## 3. Results

| quantity | before (`5e1ca50`) | after (this branch) | how measured |
|---|---:|---:|---|
| `ruff check` findings in the changed files | **1** (`E702`) | **0** | `scripts/lint_scope.py`, `artifacts/pr97-lint/*.json` |
| `ruff format --check` files needing rewrite, in scope | **3 / 3** | **0 / 5** | same |
| tree-wide `ruff check` findings (**not gated**) | 4270 | 4270 | `--measure-backlog` |
| tree-wide files `ruff format` rewrites (**not gated**) | 623 | 623 | `--measure-backlog` |
| files the lint job actually judges on this PR | 636 (all) | 5 (the change) | `lint-report.json: files` |
| `tests/test_abaqus_io.py` | 13 tests (head) | **14 passed** | `pytest -q tests/test_abaqus_io.py` |
| `tests/test_lint_scope.py` | — | **12 passed** | `pytest -q tests/test_lint_scope.py` |

Both report files show `"clean": true` with the same file list, for the two
bases the workflow can pick:

* `after-vs-main.json` — base `origin/main` (pull-request path);
* `after-push-base.json` — base `286fc59`, the previous head of this branch,
  i.e. what a push event resolves from `github.event.before`: it sees only the
  six files of the lint cure, and they are clean.

### What the tests pin

`tests/test_lint_scope.py` uses a real temporary git repository and a stub
`ruff`, so it tests the gate's *decisions*, not ruff: the scope is the diff and
not the tree; a merged-in trunk commit stays out of scope; a clean change exits
0 and touches only the changed file; a violation exits 1; no Python change exits
0 without invoking ruff; untracked files are picked up with `--worktree`;
deleted files are dropped; a missing `ruff`, a missing base and an unresolvable
base all exit 2; the backlog is measured but does not gate.

## 4. Figure

`artifacts/pr97-lint/gate_scope.png` (generated by
`scripts/plot_lint_gate_scope.py`, numbers read from the two JSON reports):
left panel — findings and unformatted files for the change itself, before versus
after, i.e. the part the pull request is responsible for; right panel — the
grandfathered backlog versus the number of files the job judges, before (5 130
files, 4270 findings) versus after (5 files, 0 findings), with the backlog bar
annotated as *not gated* so the two effects cannot be confused.

## 5. Caveats

* This does **not** make `ruff check src tests scripts examples` pass. The 4270
  findings and 623 unformatted files are still there; they are only excluded
  from *gating*, and they are still *counted on every run*.
* The gate only protects future commits. Any file already carrying debt that a
  later change touches becomes that change's responsibility, which is intended
  (ratchet) but will surprise a change that edits one line of a legacy file —
  restrict such a change or format the file in the same commit.
* `ruff` is pinned in the lint job (`0.15.0`) because `ruff format`'s output
  moves between releases; a pin bump is a deliberate change, not an incidental
  one. The dev extra still declares `ruff>=0.6` so nothing else breaks.
* The three artifact files (two PNG figures and `audit.json`) that the earlier
  head carried were dropped by the repository's artifact-size gate and are not
  restored here; they are recompute-able from
  `scripts/audit_abaqus_keyword_gate.py`.
* To make lint blocking again: `python scripts/lint_scope.py --all` must be
  clean. Until then `continue-on-error: true` stays, and the honest statement
  is "the gate is live for new debt, the legacy debt is untouched".

## 6. Error judgement

This is a CI-measurement change; there is no physical quantity, so the usual
"< 3 %" test has no referent. The quantitative claims are exact counts, which is
a strictly stronger criterion than a 3 % band:

| quantity | baseline | value | < 3 % ? |
|---|---|---|---|
| `ruff check` findings owned by PR #97's files | 1 (`E702`) | **0** | yes — exact zero, not a tolerance |
| PR #97 files failing `ruff format --check` | 3 of 3 | **0 of 5** | yes — exact zero |
| lint findings the job can attribute to a change | 0 % (all-or-nothing, tree-wide) | **100 %** (only changed files) | yes — categorical |
| tree-wide backlog removed | 4270 / 623 | **0 removed** | **no, by design** — explicitly not claimed as fixed |
| `tests/test_abaqus_io.py` | 13 head / 14 pre-existing expectations | 14 passed, 0 failed | n/a (pass/fail) |
| gate self-tests | — | 12 passed, 0 failed | n/a (pass/fail) |

Honest remainder: the repository's lint debt is not reduced by one diagnostic;
what changes is that a red lint check now always names a file the change itself
introduced, and the backlog number is printed on every run instead of being
guessed at.