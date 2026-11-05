# Change-scoped lint gate (ruff)

Status: implemented in `.github/workflows/ci.yml` (`lint` job) with
`scripts/lint_scope.sh`. Evidence: `artifacts/lint-gate/evidence.json`,
`artifacts/lint-gate/ci_lint_job_simulation.log`.

## Why this exists

The `lint (advisory)` job used to run

```
ruff check src tests scripts examples
ruff format --check src tests scripts examples
```

over the whole tree. That question has two answers that are both useless:

| state (trees `src tests scripts examples`) | whole-tree `ruff check` | whole-tree `ruff format --check` |
| --- | --- | --- |
| `main` (`c5168e4`) | 4271 errors | 624 of 632 files would be reformatted |
| PR #107 head (`0fd69f8`) | 4215 errors (E702 ×2985, E701 ×964, …) | 561 of 569 files would be reformatted |

The backlog is pre-existing, so the job reported failure for *every* commit —
including the four files PR #107 added, which were already clean under
`ruff check` and only needed formatting. A red X that means "the legacy tree is
dirty" carries no information about the change under review, and it made
PR #107's CI red even though the change itself was sound.

The CI comment itself anticipated the two possible fixes: "Fix the backlog,
then drop this continue-on-error and promote it to a gate." Reformatting
several hundred unrelated files is neither reviewable nor in scope for a
benchmark PR, so we take the standard alternative: **gate the change, keep the
legacy debt visible, pay it down file by file.**

## How it works

1. `scripts/lint_scope.sh` prints the repository-relative Python files the
   change touches inside `src/ tests/ scripts/ examples/ benchmarks/`.
   * `pull_request*` → `git diff --diff-filter=ACMR origin/$GITHUB_BASE_REF...HEAD`
     (three-dot: the PR's own changes, not main's drift).
   * `push` / `merge_group` → diff against `github.event.before`; a new branch
     (`before` = all-zero SHA) or an unresolvable base falls back to
     `git merge-base HEAD origin/main`.
   * no base at all (`workflow_dispatch`) → the whole tree, i.e. the old audit
     mode, which is why the job is still `continue-on-error: true`: that mode
     is *expected* to be red against the backlog.
2. The job feeds that list to `ruff check` and `ruff format --check` through
   `xargs -r -a lint-scope.txt`. An empty scope skips both steps: "nothing
   changed" must not be confused with "nothing to check" or with "check
   everything".
3. `fetch-depth: 0` is required — a depth-1 clone cannot compute the diff base.

Consequences, stated plainly:

* New Python code is held to `ruff check` + `ruff format` from the first
  commit, because an added file is always inside the diff.
* Touching a legacy file makes that file's backlog the author's problem; the
  fix is `ruff check --fix <file>` / `ruff format <file>`, still scoped.
* The debt does not disappear: the whole-tree numbers stay in
  `artifacts/lint-gate/evidence.json` (`whole_tree_backlog`: 4242 errors and
  576 unformatted files over the five linted trees at the fixed head) and are
  regenerated on demand, so "how dirty is the tree" remains a measurable
  quantity instead of a permanently-red check.

## Reproducing locally

```bash
# what will CI lint for this branch?
bash scripts/lint_scope.sh $(git merge-base HEAD origin/main)

# the job's two commands, verbatim
xargs -r -a lint-scope.txt -d '\n' ruff check
xargs -r -a lint-scope.txt -d '\n' ruff format --check

# the evidence record (before/after per scoped file + whole-tree backlog)
python scripts/lint_gate_evidence.py --out artifacts/lint-gate/evidence.json
```

`scripts/lint_gate_evidence.py` measures the scoped files as they are *now* and
as they were at `--before` (default: the parent of `HEAD`), so the record
demonstrates the red → green transition rather than asserting it.

Guard tests: `tests/test_lint_gate_scope.py` runs the scope script inside
throwaway repositories, covering the three behaviours the workflow depends on
(diff-limited scope, empty scope on an unchanged tree, whole-tree fallback when
`origin/<base>` cannot be resolved) plus the workflow wiring.

## Limits

* The gate cannot see lint debt in files a change does not touch; that is the
  point, but it means a whole-tree audit is still needed before the job can be
  promoted to blocking (`continue-on-error: false`).
* `ruff` is unpinned (`ruff>=0.6` in the `dev` extra) — a new ruff release that
  adds rules or changes formatting can redden a previously green PR. Pinning
  `ruff` in the `dev` extra is the follow-up if that noise appears.
* Files renamed out of the linted trees are dropped from the scope
  (`--diff-filter=ACMR`), which is intended but worth knowing when reading
  scope sizes.