# CI lint gate: what blocks, what is ratcheted, what is only counted

The lint CI job used to run two tree-wide commands:

```
ruff check src tests scripts examples            # default rules: E4, E7, E9, F
ruff format --check src tests scripts examples
```

Both were red on **every** branch, including branches that touch no Python at
all, because of a pre-existing backlog (4268 `ruff check` findings, 616 files
that `ruff format` would rewrite). A gate that is always red carries no signal:
"lint (advisory) = failure" was the permanent state of the checks page, and a
real regression was indistinguishable from the backlog.

The job now separates three different contracts.

## 1. Correctness rules — tree-wide, blocking

```toml
[tool.ruff.lint]
select = ["E9", "F63", "F7", "F81", "F82"]
```

| code | catches |
|---|---|
| `E9` | syntax / IO / tokenisation errors (`E999`, `E902`, …) |
| `F63` | invalid `%` and f-string format strings |
| `F7` | `break` / `continue` / `return` outside a loop or function |
| `F81` | redefinition of an unused name (`F811`) |
| `F82` | undefined name / undefined `__all__` export (`F821`, `F822`) |

The whole tree passes this set with **zero** findings, so a finding is always a
regression, never noise. Run it exactly as CI does:

```
ruff check src tests scripts examples
```

## 2. Formatting — ratcheted on the files a change touches

```
python scripts/lint_gate.py --base origin/main
```

`scripts/lint_gate.py` diffs the working tree against the merge base with
`origin/main` and, for every Python file the change adds or modifies:

* **added file not formatted** → fail;
* **file clean at the merge base, dirty now** → fail (formatting regression);
* **file already dirty at the merge base** → warn only.

Consequences, both intended:

* no-change-to-Python changes (docs, CI, config) pass without any reformatting;
* touching one legacy file does **not** drag a whole-file reformat into a small
  diff, so concurrent branches do not collide on formatting churn.

`ruff` is pinned (`ruff==0.15.0` in the `dev` extra) so that the formatter and
the gate cannot drift underneath each other. Bump the pin in its own change.

Escape hatch for a deliberate full sweep (never used as a gate):

```
python scripts/lint_gate.py --all --count      # prints the remaining backlog
ruff format src tests scripts examples         # burn-down commit
```

## 3. Legacy style backlog — counted, never failing

```
ruff check --select E4,E7,E9,F --statistics src tests scripts examples
python scripts/lint_gate.py --all --count
```

Current counters: **4268** `ruff check` findings, **616** files that
`ruff format` would rewrite. Dominant categories: `E702`/`E701`
semicolon/one-line statements (3962), `F401` unused imports (81), `E741`
ambiguous names (71).

## Burn-down order (each step shrinks the counter, then widens the gate)

1. `F401` unused imports — 81 findings, 76 auto-fixable
   (`ruff check --select F401 --fix src tests`); review `__init__.py` re-exports
   by hand, `--fix` can drop public re-exports.
2. `F841` unused variables (27) and `E703`/`E401` (62, mostly auto-fixable).
3. `E731` lambda assignments (33), `E741` ambiguous names (71).
4. `E7xx` one-liners (3962) and the 616-file `ruff format` backlog — mechanical,
   but only worth doing as one deliberate repo-wide commit, not per branch.

When the counters reach zero: widen `[tool.ruff.lint] select` to the full
default, drop `continue-on-error` from the lint job in
`.github/workflows/ci.yml`, and let both steps block.

## Why the job is still `continue-on-error`

The job stays advisory at the *job* level until the counters above reach zero:
that is what lets branches opened before this split stay mergeable. The two
steps in 1. and 2. are hard gates inside the job — a new undefined name or a
formatting regression shows up as a red check on the PR.