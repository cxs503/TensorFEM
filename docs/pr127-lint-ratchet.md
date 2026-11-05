# PR #127 lint ratchet -- the check now measures the change, not the inheritance

Round 1 of this cure proved the red check was inherited; round 2 removes the
inheritance. The `lint (advisory)` job used to run `ruff check src tests scripts
examples` and `ruff format --check src tests scripts examples` over the **whole
tree**. The tree carries a backlog of 4268 ruff findings and 616
files that `ruff format` would rewrite, all of it predating PR #127, so the job
was red on every ref -- including `main` -- and told a reviewer nothing about the
pull request. The fix is a ratchet: `scripts/lint_ratchet.py` measures only the
Python files the change touches and compares each one with **its own version at
the merge base**.

## What the gate does

| rule | behaviour |
|---|---|
| scope | `src`, `tests`, `scripts`, `examples`; only `.py` files added or modified by `git diff base...HEAD` |
| findings | per file, the count of each ruff rule code may not grow (`ruff check --output-format json`) |
| formatting | a file that was already `ruff format` clean may not become unformatted; a **new** file starts from clean, so it must be clean |
| live probe | a seeded broken file (unused import + packed statement, unformatted) is linted on every run; if it is not reported the gate exits 2 |
| blind gate | unresolvable merge base, missing ruff, unparsable ruff output, ruff exit >= 2 all exit 2 (failure) |
| backlog | whole-tree totals are printed for the record, never used to fail the job |
| exit codes | `0` no regression, `1` this change adds debt, `2` the gate could not measure |

`continue-on-error: true` is kept so the check name stays stable; deleting that
one line promotes the job to a blocking gate, which is safe now because the job
no longer carries `main`'s debt.

## Measurements

Whole-tree backlog at the time of writing (ruff 0.15.0):

| rule | findings |
|---|---|
| `E702` multiple-statements-on-one-line-semicolon | 2997 |
| `E701` multiple-statements-on-one-line-colon | 965 |
| `F401` unused-import | 81 |
| `E741` ambiguous-variable-name | 71 |
| `E401` multiple-imports-on-one-line | 57 |
| `E731` lambda-assignment | 33 |
| `F841` unused-variable | 27 |
| `F405` undefined-local-with-import-star-usage | 22 |
| `E402` module-import-not-at-top-of-file | 5 |
| `E703` useless-semicolon | 5 |
| `F541` f-string-missing-placeholders | 3 |
| `F403` undefined-local-with-import-star | 2 |

Total: **4268 findings / 616 unformatted files** -- reported, not gated.

PR #127 (merge base `7accc4b9` -> this branch head; the exact head SHA is
`pr_gate.head_sha` in the evidence JSON, so this document does not go stale on
every regeneration), in-scope files only:

| file | findings at merge base | findings at PR head | format | verdict |
|---|---|---|---|---|
| `scripts/audit_pr127_lint_cure.py` (added) | 0 | 0 | clean | ok |
| `scripts/lint_ratchet.py` (added) | 0 | 0 | clean | ok |
| `scripts/plot_large_angle_gz.py` (added) | 0 | 0 | clean | ok |
| `scripts/pr127_lint_ratchet_evidence.py` (added) | 0 | 0 | clean | ok |
| `src/tensorfem/marine_large_angle_gz.py` (added) | 0 | 0 | clean | ok |
| `tests/test_lint_ratchet.py` (added) | 0 | 0 | clean | ok |
| `tests/test_marine_large_angle_gz.py` (added) | 0 | 0 | clean | ok |

Verdict: **no new debt** (gate exit `0`, 0 regression(s)).

The fix commits (`5f2d3290` -> this branch head) add in-scope files, all clean:

| file | findings | format | verdict |
|---|---|---|---|
| `scripts/lint_ratchet.py` (added) | 0 | clean | ok |
| `scripts/pr127_lint_ratchet_evidence.py` (added) | 0 | clean | ok |
| `tests/test_lint_ratchet.py` (added) | 0 | clean | ok |

## Fail-closed controls (measured, not asserted)

| control | expected exit | measured exit |
|---|---|---|
| new finding in a changed file | 1 | 1 |
| formatting regression on a clean file | 1 | 1 |
| unresolvable base ref (blind gate) | 2 | 2 |

## Tests

`python -m pytest -q tests/test_lint_ratchet.py` -> 7 passed, 0 failed.
The suite builds throw-away git repositories with a debt-laden `main` and covers:
inherited debt passes; touching a debt-carrying file without growing its debt
passes; a new finding in a new file fails; a new finding in a legacy file fails;
an unformatted new file fails; an out-of-scope change passes without measuring;
an unresolvable base ref exits 2.

Plus the 39 tests of the PR's own subject (`tests/test_marine_large_angle_gz.py`),
which the lint cure does not touch.

## Reproduce

```
python scripts/lint_ratchet.py --base <merge-base-ref>
python scripts/pr127_lint_ratchet_evidence.py   # regenerates this file, the JSON and the figure
```

## Limitations

* The ratchet gates *growth*, not absolute cleanliness. A file that is already
  unformatted stays unformatted until someone formats it; that is deliberate --
  reformatting 616 legacy files inside a feature PR is the "unrelated
  files" failure mode this project forbids.
* The ratchet cannot see debt that arrives through files it does not lint
  (`artifacts/*.py` are outside the CI scope).
* `ruff` is not pinned in the repository; results above were taken with
  0.15.0, the same default rule set (`E4/E7/E9/F`) CI uses.
* Burning the whole-tree backlog down is still a separate mission: the splitter
  for `E701`/`E702`, `--fix --unsafe-fixes` for the safe remainder, manual
  `E741`/`F841`/`F403`/`F405` fixes. Only then can `continue-on-error` be dropped.
