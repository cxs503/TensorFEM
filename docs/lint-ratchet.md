# Lint ratchet: why CI lint was red and how it is gated now

Status: implemented in `scripts/lint_changed.py`, wired into the `lint (advisory)` job of
`.github/workflows/ci.yml`, regression-tested by `tests/test_lint_changed_scope.py`.

## 1. Symptom

Every run of the `lint (advisory)` job on PR #103 (`agent/m3671`) reported `failure`, so the
check carried no information about the pull request under review. Reproduced locally on the PR
head `c5f66d2` with ruff 0.15.0:

| command | result on `c5f66d2` |
|---|---|
| `ruff check src tests scripts examples` | `Found 4218 errors` (exit 1) |
| `ruff format --check src tests scripts examples` | `560 files would be reformatted, 8 files already formatted` (exit 1) |
| `ruff check src/tensorfem/tet_convergence.py tests/test_tet_convergence.py benchmarks/tet_h_convergence_gci.py` | `Found 3 errors` (exit 1) |

## 2. Two distinct defects

1. **New debt in this PR.** Three fresh findings in the two linted files the PR adds:
   * `src/tensorfem/tet_convergence.py:216` `F841` -- `references = (exact_energy, reference_energy)`
     assigned and never used;
   * `tests/test_tet_convergence.py:2` `F401` -- `import math` unused;
   * `tests/test_tet_convergence.py:132` `F841` -- dead tuple assignment
     `nodes, weights = torch.linspace(0.0, 1.0, 2), None` (`weights` is rebound below, `nodes` unused).
   Both files also needed `ruff format`. Those three lines were removed, the two files were
   formatted, and the PR's own tests were re-run to show the removal is behaviour-neutral
   (`tests/test_tet_convergence.py`, see `artifacts/pr103-lint/pr-tests.log`).

2. **The job itself could never pass.** The backlog is repository-wide and pre-existing:

   | rule | findings | selected meaning |
   |---|---|---|
   | `E702` | 2985 | multiple statements on one line (semicolon) |
   | `E701` | 964 | multiple statements on one line (colon) |
   | `E741` | 65 | ambiguous variable name |
   | `F401` | 63 | unused import |
   | `E401` | 56 | multiple imports on one line |
   | `E731` | 30 | lambda assignment |
   | `F405` / `F403` | 24 | star-import fallout |
   | `F841` | 22 | unused variable |
   | `E703` / `E402` / `F541` | 9 | misc |

   4218 findings across 326 of the 568 files under `src/ tests/ scripts/ examples/`, plus 560
   files that `ruff format` would rewrite. Only 125 of the findings have a ruff autofix, and the
   two dominant rules (`E702`, `E701`, 94% of the backlog) have **none**: clearing them means
   hand-splitting thousands of statements inside numerical kernels. Doing that inside PR #103
   would have rewritten most of the repository, which is exactly the unrelated churn the review
   rules forbid -- and it would have made the PR unreviewable and unrevertable.
   `continue-on-error: true` did not help either: the job-level flag keeps the *workflow* green
   but the job's own check run still concludes `failure`, which is what the PR gate saw.

## 3. The fix: gate the diff, grandfather the backlog

`scripts/lint_changed.py` lints exactly the Python files a change touches:

```bash
# what CI runs for a pull request
python scripts/lint_changed.py --base origin/main --report lint-report.json
# what a contributor runs before pushing (default base HEAD^)
python scripts/lint_changed.py
# the pre-ratchet, repo-wide invocation (use once the backlog is clean)
python scripts/lint_changed.py --all
```

Selection rules:

* **base** = PR target branch (`origin/<base_ref>`) for `pull_request`, the previous pushed
  commit (`github.event.before`) for `push`, `HEAD^` otherwise; overridable with `--base`.
* **changed set** = `git diff --name-only --diff-filter=ACMR <merge-base(base, HEAD)>`, which
  compares against the working tree, so staged, unstaged **and untracked** `*.py` files are all
  covered -- a brand-new file is gated before its first commit.
* **roots** = `src/ tests/ scripts/ examples/`, byte-for-byte the same scope the old CI step
  claimed to lint. Note `benchmarks/` was never linted by CI and still is not.
* both commands run on that set: `ruff check` and `ruff format --check`, identical flags to the
  commands they replace.
* exit codes: `0` clean (including "no Python changed"), `1` findings, `2` environment error
  (unresolvable base, ruff missing). Code `2` fails **closed**: a broken base resolution is a
  visible red, never a silent pass.

Consequences: a change that adds no new debt is now green, a change that adds debt is red with
only its own files named, and the 4218-finding legacy baseline is untouched (no mass rewrite, no
`# noqa` carpet, no rules switched off in `pyproject.toml` -- the backlog stays measurable and
shrinks as files get touched).

## 4. Promotion path

The ratchet is a bridge, not the destination:

1. burn down the backlog by directory (`E702`/`E701` are mechanical: split statements, then
   `ruff format`), landing a directory only when `ruff check <dir>` is clean;
2. keep `LINT_CHANGED` green throughout, so touched files cannot regress;
3. when `python scripts/lint_changed.py --all` is clean, replace the CI step's `--base` argument
   with `--all` and delete `continue-on-error: true` to make lint blocking.

`--all` exists precisely so step 3 is a two-line edit, and `--fail-on-empty` guards against a
misconfigured `--paths` silently linting nothing.

## 5. Verification

| check | command | result |
|---|---|---|
| red reproduced | `ruff check src tests scripts examples` on `c5f66d2` | 4218 errors (see `artifacts/pr103-lint/baseline.txt`) |
| ratchet gate on this PR | `python scripts/lint_changed.py --base origin/main` | 4 files gated, `ruff check` + `ruff format --check` exit 0 (`artifacts/pr103-lint/after-vs-main.json`) |
| PR's own tests still pass | `pytest tests/test_tet_convergence.py` | 9 of 12 passed, 0 failed, in 9 min; the 3 remaining TET10 h-convergence sweeps were still running when this PR was handed off (the host was shared with other agents' full-suite runs). Partial log: `artifacts/pr103-lint/pr-tests.log`; CI's `test` job covers the rest |
| selection logic regression net | `pytest tests/test_lint_changed_scope.py` | 9 passed |
| YAML validity | `yaml.safe_load('.github/workflows/ci.yml')` | jobs `test`, `lint`; 6 lint steps |

## 6. Limitations

* The ratchet measures *scope*, not absolute tidiness: an old file that still carries findings
  passes until someone edits it, at which point the whole file must be clean. That is the point,
  but it means the gate gets stricter per file as the backlog is touched.
* Findings introduced by a merge commit are attributed to the merge, not to the branch that
  authored them; the diff is taken from the merge base, so a branch that merges `main` still
  shows only its own contribution.
* `E741`/`E731` style findings in untouched files remain for the backlog campaign; nothing here
  reduces them.
* This document does not change `pyproject.toml`'s ruff configuration; no rule is ignored and no
  `noqa` was added, so the 4218-finding number remains an honest measurement of the tree.