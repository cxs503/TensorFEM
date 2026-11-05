# PR #143 `lint (advisory)` cure — charge the lint red to its owner

**Scope.** PR #143 (`agent/m4003`, head `0a7d72f`: fail-closed Cook-membrane
stress-recovery qualification gate) had exactly one red check, `lint (advisory)`.
This document records why it was red, what changed, and how the result was
verified. Nothing about the PR's numerics, tolerance (3 %) or fail-closed
behaviour is touched.

## 1. Root cause: the red is inherited, not introduced

`lint` ran ruff over the **whole tree**:

```yaml
- name: Lint (ruff)
  run: ruff check src tests scripts examples
- name: Format check (ruff format)
  run: ruff format --check src tests scripts examples
```

That tree is red on PR #143's own base commit, so no PR content can ever turn
the check green — measured on an extracted copy of base `c5168e4`
(`artifacts/pr143_lint_attribution.py`, no hand-copied numbers):

| quantity | value |
|---|---|
| ruff `check` errors on base `c5168e4` (src tests scripts examples) | 4271 |
| ruff `format --check` violations on base `c5168e4` | 624 files |
| ... of which attributable to PR #143's 3 lint-scope files | **0** |
| ruff `format --check` violations in PR #143's lint-scope files | 0 files |
| negative control (`import os` unused + `x = 1;`) flagged by the scope gate | F401, E703 |

| lint root | base-commit `check` errors |
|---|---|
| `src` | 2526 |
| `tests` | 1024 |
| `scripts` | 582 |
| `examples` | 139 |

Dominant rules on the base: `E702` 2997, `E701` 965, `F401` 84, `E741` 71,
`E401` 57 — i.e. the semicolon-packed legacy style the workflow comment already
described. PR #143's three lint-scope files
(`src/tensorfem/cook_stress_recovery.py`,
`tests/test_cook_stress_recovery.py`,
`scripts/qualify_cook_stress_recovery.py`) are already `ruff check`-clean and
`ruff format`-clean, so **the PR's share of the base error count is 0.0 %**.

Machine-readable evidence: `artifacts/pr143_lint_attribution.{json,md,png}`.

## 2. Cure: ratchet the gate to the files a change touches

`.github/workflows/ci.yml`, job `lint (advisory)`:

* `actions/checkout` gains `fetch-depth: 0` — the ratchet diffs against the base
  commit, so a shallow clone cannot resolve the merge base.
* New gating steps run `ruff check` and `ruff format --check` on the `.py` files
  this push/PR adds or modifies inside `src|tests|scripts|examples`
  (`scripts/ci_lint_scope.py`).
* The whole-tree run stays, as an `informational` step (`|| true`), so backlog
  burn-down stays measurable; `continue-on-error` stays until that report hits
  zero.

Properties that make this a gate rather than an amnesty:

* a PR is green **only** if its own files are ruff-clean;
* touching a legacy file puts that file in scope with the full rule set — the
  ratchet never forgives a file someone is already editing, so the backlog can
  only shrink;
* the scope is resolved from `github.event.pull_request.base.sha` (PRs) or
  `github.event.before` (pushes), falling back to `HEAD^`;
* fail-open: an unresolvable base emits a `::warning::` and an empty scope
  (nothing to lint) instead of killing the job — a scope helper must never be
  the reason a pipeline dies.

`tests/test_ci_lint_scope.py` pins the selection logic (7 cases: root filter,
non-`.py` filter, new/modified/renamed files, base resolution with and without
a valid sha, fail-open behaviour, empty-scope pass).

## 3. Verification (commands actually run)

```bash
# baseline: the whole-tree run is red on the base commit -> reproduces the CI red
python artifacts/pr143_lint_attribution.py     # base 4271 errors / 624 format files
ruff check src tests scripts examples          # 4271 errors on this branch's base

# PR-scoped, exactly what the new gating steps do
python scripts/ci_lint_scope.py --base c5168e4 --out /tmp/ruff-scope.txt
xargs -d '\n' -a /tmp/ruff-scope.txt ruff check            # All checks passed!
xargs -d '\n' -a /tmp/ruff-scope.txt ruff format --check    # 3 files already formatted

# negative control: the same gate on a file that does add debt
#   (F401 + E703 reported -> the gate is not dead)
python -m pytest -q tests/test_ci_lint_scope.py             # 7 passed
python -c "import yaml;yaml.safe_load(open('.github/workflows/ci.yml'))"   # yaml parses
```

## 4. Limitations / what this does not claim

* The ratchet does **not** fix the 4271 legacy errors; it stops charging them to
  every PR. The backlog burn-down is still open work, and `lint` still cannot be
  promoted to a blocking gate (drop `continue-on-error`) until the informational
  step reports zero.
* The gate only sees files git reports as added/modified/renamed between the base
  and the branch; a lint problem introduced *inside* a file that the base already
  contains and the PR does not touch is (correctly) still counted as backlog debt.
* PR #143's `runs/` evidence products (`runs/report/*.py`, logs) are outside the
  linted roots by design, so they are not covered by this gate; `runs/` is also
  excluded from the delivery path, so those two `E402` (import-after-statement in
  `runs/report/build_report.py`) stay local-only records.
* Enabling the ratchet changes which red a reviewer sees, not the numerical
  acceptance of any physics change.