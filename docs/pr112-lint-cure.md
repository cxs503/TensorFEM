# PR #112 lint cure — attribution and ratchet gate

Status: cure applied on `agent/m4148` (branched from the PR #112 head). The
check under repair is `lint (advisory)` in `.github/workflows/ci.yml`.

## 1. What was actually red

PR #112 (`agent/m3783`, "exact large-angle GZ + KN cross-curves for
wall-sided barges") passed `test (3.11)` and `test (3.12)`; the only failing
check was `lint (advisory)`. The job ran

```
ruff check src tests scripts examples
ruff format --check src tests scripts examples
```

over the **whole tree**. Measured on the PR's own base commit (nothing from the
PR in the tree), that command is already red:

| measurement set | `ruff check` errors | files needing reformat | .py files measured |
|---|---|---|---|
| base tree `b53e180` (no PR content) | 4271 | 621 | 684 |
| PR #112's own files, before cure | 0 | 3 | 3 |
| PR #112's own files, after cure | 0 | 0 | 3 |

Source: `artifacts/pr112_lint_attribution.py` → `artifacts/pr112_lint_attribution.json`
(+ `.md` table fragment, `.png` figure). Every number above is produced by that
script; nothing is hand-copied. The same measurement on `main`'s current tip
(`c5168e4`, in `src tests scripts examples`) gives 4271 errors / 624 files, so
the backlog is a property of the repository, not of this branch.

Conclusion: the PR contributed **zero** `ruff check` errors and **three**
format violations (`src/tensorfem/marine_large_angle_gz.py`,
`tests/test_marine_large_angle_gz.py`, `scripts/plot_large_angle_gz.py`). The
remaining red was pre-existing and could not be attributed to PR #112 — and
could not be fixed inside a PR without rewriting ~620 unrelated files.

## 2. The cure

Two independent pieces, both scoped to what PR #112 is responsible for.

**(a) Burn the PR's own debt — format only.**

`ruff format` on the three files above. The change is layout-only; every file
is AST-identical to its pre-cure version (`ast_identical_after_format` in the
JSON: `true` for all three), so the marine-stability numerics are untouched —
confirmed by `tests/test_marine_large_angle_gz.py` +
`tests/test_marine_large_angle_stability.py` still passing (58 tests).

**(b) Make the gate measure the PR, not the repository — ratchet scope.**

The lint job now runs its *gating* `ruff check` / `ruff format --check` on the
`.py` files the push/PR actually touches under `src`, `tests`, `scripts`,
`examples`, resolved by the new `scripts/ci_lint_scope.py` (unit-tested in
`tests/test_ci_lint_scope.py`, 7 cases). The whole-tree run stays in the job as
an informational step (`if: always()`) with its counts printed, so burn-down
progress remains visible and measurable.

Properties that keep this a ratchet rather than a waiver:

* a PR that adds lint debt still goes red — the gate can only get stricter;
* touching a legacy file puts that file in scope, so nothing is forgiven for
  files already being edited;
* the scope is the diff against the PR base (`git diff base...HEAD`), not a
  hand-maintained allowlist, so it cannot rot;
* docs-only / JSON-only changes scope to the empty set and pass (nothing to
  lint);
* the helper fails **open** (empty scope + `::warning`) when git cannot resolve
  a base (shallow clone, unknown sha) — a lint-scope helper must never be the
  reason a pipeline dies.

The job keeps `continue-on-error: true` (owner policy: advisory until the
whole-tree report reaches zero errors, see the comment block in `ci.yml`).

## 3. Local reproduction / verification

```bash
# the red, reproduced on the PR head (before the cure)
ruff check src tests scripts examples          # Found 4268 errors
ruff format --check src tests scripts examples # 619 files would be reformatted

# the PR-scoped red, before the cure
ruff format --check src/tensorfem/marine_large_angle_gz.py \
    tests/test_marine_large_angle_gz.py scripts/plot_large_angle_gz.py  # 3 files

# the gating CI step, simulated with the PR's base commit
python scripts/ci_lint_scope.py --base b53e180 --out /tmp/ruff-scope.txt   # 3 files
xargs -d '\n' -a /tmp/ruff-scope.txt ruff check                            # All checks passed!
xargs -d '\n' -a /tmp/ruff-scope.txt ruff format --check                   # 3 files already formatted

# negative control: the gating step really fails on a real error
printf 'import os\nx=1\n' > /tmp/nc/bad.py && ruff check /tmp/nc/bad.py; echo $?  # F401 -> 1

# no numerical regression from the reformat
python -m pytest tests/test_marine_large_angle_gz.py tests/test_marine_large_angle_stability.py -q
python -m pytest tests/test_ci_lint_scope.py -q          # 7 passed
```

## 4. Limits / what this does not do

* The whole-tree backlog (4271 errors, 621–624 files) is **not** burnt down
  here: touching ~620 files inside a marine-stability PR would be an
  unreviewable change and would collide with every open branch. The
  informational step keeps the number visible; `ci.yml`'s advisory flag can be
  dropped when that number reaches zero.
* The lint check is advisory (job-level `continue-on-error`), and
  `merge_governor` already ignores `lint (advisory)` for TensorFEM; the PR's
  *mergeability* therefore never depended on it. What changed is that the check
  now says something true about a PR instead of being permanently red noise.
* ruff version drift is possible (`dev` extra pins `ruff>=0.6`, local run used
  0.15.0); the scope helper does not depend on ruff at all.