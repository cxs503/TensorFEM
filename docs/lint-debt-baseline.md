# Lint debt ledger and ratchet gate (PR #137)

Status: **lint CI is green again on the PR head, and any *increase* in lint debt now
fails the job.** The legacy backlog itself is untouched and still visible in every
run's summary - it is frozen in a ledger, not hidden.

## 1. Why this document exists

PR #137 (branch `agent/m3902`) reported a red CI run: `lint (advisory)=failure`.
Diagnosis first, because the PR obviously did not write the backlog:

| check | value |
| --- | --- |
| files changed by the PR vs `main` | 1 (`docs/pr72_merge_report.md`, markdown) |
| `ruff check src tests scripts examples` on that head | **4271 violations**, exit 1 |
| `ruff format --check src tests scripts examples` | **624 files** would be rewritten, exit 1 |
| tool / config | `ruff 0.15.0`, line-length 100, target py311, ruff default ruleset (E4/E7/E9/F), scopes `src tests scripts examples` |

So the red is pre-existing repository debt (E702 semicolon-packed statements and
E701 one-line `if`s alone are 3962 of
4271, i.e. 93%).
Two cures were considered and rejected:

* **keep `continue-on-error: true`** (the status quo): every PR keeps showing a red
  check, and genuinely new debt (a freshly introduced E702/F401) enters unnoticed -
  the check conveys nothing.
* **fix the backlog before gating**: 4271 violations across
  347 files plus 624 reformatted
  files is a whole-tree rewrite. It would bury every other CI signal, make focused bug-fix
  PRs unreviewable, and directly violates the "do not touch unrelated files" rule that a
  CI-fix task must obey.

Adopted instead: a **debt ratchet**. The current state is frozen per file and per rule;
new debt fails, paid-down debt is reported as progress and re-frozen deliberately.

## 2. What was added

| artefact | role |
| --- | --- |
| `scripts/lint_gate.py` | collects `ruff check --output-format=json` + `ruff format --check`, compares against the ledger, exits 0/1 (2 on tooling failure) |
| `docs/lint-baseline.json` | frozen ledger: `{file: {rule: count}}`, the unformatted-file set, totals, SHA-256 signature `b9dded40eea50d4c...` |
| `tests/test_lint_gate.py` | 15 tests: pure comparison logic, error paths, and end-to-end runs against a throwaway tree with the real ruff binary |
| `.github/workflows/ci.yml` | lint job promoted from `lint (advisory)`/`continue-on-error` to `lint (debt ratchet)`, ruff pinned to `0.15.0` so the ledger stays comparable |

### Gate semantics

| situation | verdict |
| --- | --- |
| frozen `file -> rule` pair unchanged | pass |
| a pair's count **grows** | **FAIL** `path: CODE before -> now` |
| a rule appears in a file that did not have it | **FAIL** `path: CODE new (nx)` |
| a file needs `ruff format` and is not in the frozen set | **FAIL** |
| debt paid down (count drops, file reformatted) | pass, listed under "debt burned down" |
| ruff missing / unparsable output / bad baseline | exit 2 - never silently green |

## 3. Frozen ledger (injected from `runs/lint-debt-scan.json`, `docs/lint-baseline.json`)

| metric | value |
| --- | ---: |
| python files scanned | 634 |
| `ruff check` violations | 4271 |
| files carrying check debt | 347 |
| distinct `file -> rule` pairs | 652 |
| files `ruff format` would rewrite | 624 |
| files already ruff-formatted | 10 |
| ruff | `ruff 0.15.0` |
| ledger signature | `b9dded40eea50d4c917f91b21e88ada425dbd7225a655b61446616a6d609c219` |

### Debt by rule

| rule | count | share |
| --- | ---: | ---: |
| `E702` | 2997 | 70.2% |
| `E701` | 965 | 22.6% |
| `F401` | 84 | 2.0% |
| `E741` | 71 | 1.7% |
| `E401` | 57 | 1.3% |
| `E731` | 33 | 0.8% |
| `F841` | 27 | 0.6% |
| `F405` | 22 | 0.5% |
| `E402` | 5 | 0.1% |
| `E703` | 5 | 0.1% |
| `F541` | 3 | 0.1% |
| `F403` | 2 | 0.0% |

### Debt by scope

| scope | check violations | python files | share | unformatted files |
| --- | ---: | ---: | ---: | ---: |
| `src` | 2526 | 229 | 59.1% | 228 |
| `tests` | 1024 | 231 | 24.0% | 224 |
| `scripts` | 582 | 93 | 13.6% | 91 |
| `examples` | 139 | 81 | 3.3% | 81 |

### Worst offenders (top 10 files)

| # | file | violations | rules |
| ---: | --- | ---: | --- |
| 1 | `src/tensorfem/suboff_ice_simulation.py` | 116 | E701:19, E702:97 |
| 2 | `src/tensorfem/mesh_project.py` | 99 | E401:1, E701:38, E702:53, E703:5, E741:1, F841:1 |
| 3 | `src/tensorfem/mesh_pipeline.py` | 81 | E701:46, E702:34, E731:1 |
| 4 | `src/tensorfem/finite_strain_contact3d.py` | 75 | E701:24, E702:47, F841:4 |
| 5 | `src/tensorfem/result_db_v2.py` | 75 | E401:1, E701:30, E702:42, E731:1, F841:1 |
| 6 | `src/tensorfem/project_workflow.py` | 71 | E401:1, E701:22, E702:47, F401:1 |
| 7 | `src/tensorfem/suboff_ice_collision.py` | 66 | E702:66 |
| 8 | `src/tensorfem/benchmark_registry.py` | 62 | E701:4, E702:57, E741:1 |
| 9 | `src/tensorfem/engineering_cases.py` | 54 | E401:1, E701:7, E702:46 |
| 10 | `src/tensorfem/initial_fields.py` | 50 | E701:22, E702:28 |

## 4. Verification (all numbers programmatic)

| evidence | artefact | result |
| --- | --- | --- |
| red reproduction on the PR head | `runs/lint-red-repro.log`, `runs/lint-format-red-repro.log` | `ruff check` exit 1 / 4271 errors; `ruff format --check` exit 1 / 624 files |
| gate green on the same tree | `runs/lint-gate-ci.log`, `runs/lint-report.json` | exit 0, `pass: true`, 0 regressions |
| negative control (one injected E702 + unformatted state in a real file) | `runs/lint-gate-red.log` | exit 1, 2 new-debt entries; revert -> exit 0 |
| detection dose-response | `runs/lint-ratchet-probe.json` | see table below |
| unit/integration tests | `pytest tests/test_lint_gate.py -q` | 15 passed |
| workflow syntax | `yaml.safe_load(.github/workflows/ci.yml)` | 2 jobs, lint job has no `continue-on-error` |

Dose-response of the gate (probe file `tests/_lint_ratchet_probe.py`, injected then removed):

| injected E702 | detected E702 | reported entries | gate rc |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 |
| 1 | 1 | 2 | 1 |
| 2 | 2 | 2 | 1 |
| 3 | 3 | 2 | 1 |
| 4 | 4 | 2 | 1 |
| 5 | 5 | 2 | 1 |

Detection is exact (slope 1, zero at zero) and the tree returns to exit 0 once the probe
file is removed: the gate reacts to debt, not to the probe.

## 5. Caveats and scope

* The ledger is **ruff-version specific** - a newer ruff with new rules would look like new
  debt. The lint job therefore pins `ruff==0.15.0`; bumping the pin is a
  deliberate commit that also re-freezes the ledger (`--update-baseline`).
  The pin lives in **two** places that must agree - the lint job's
  `pip install "ruff==0.15.0"` and the `dev` extra in `pyproject.toml` (the `test`
  job installs the gate's tests from that extra and drives the real binary
  end-to-end).  They are held together by
  `tests/test_lint_gate.py::test_dev_extra_pins_the_same_ruff_as_the_lint_gate`,
  and `lint_gate.py --check` refuses to compare across versions (exit code 2)
  instead of reporting fake new debt.
* The ratchet blocks *new* debt, not *all* debt: 4271 violations remain. They are
  visible in every CI job summary (`Lint debt ratchet` table) and in
  `artifacts/lint-report.json`.
* Granularity is `(file, rule)`, matching how the debt actually gets paid down. Moving
  code between files therefore needs a re-freeze; that is intended, since a move is a
  reviewable event.
* Renaming the job from `lint (advisory)` to `lint (debt ratchet)` removes it from the
  merge governor's ignore list, so it is a real merge gate from now on. Any stale
  `"lint (advisory)"` ignore entry on the platform side is now a no-op.
* Nothing here changes library behaviour: the diff is CI + a new script + its tests + this
  document. `pytest --collect-only` still collects the whole suite.

## 6. Burn-down plan (next rounds)

| phase | target | why this order |
| ---: | --- | --- |
| 1 | E702/E701 (3962 items) | 93% of the backlog; a codemod that splits `;`/`:` statements is mechanically checkable by comparing `ast.dump` before/after per file |
| 2 | F401/F841/E401/E703 (~173 items) | ruff `--fix` candidates; needs the `__init__` re-export review |
| 3 | E741/E731/F405/F403/E402 (~133 items) | renames and `lambda -> def`; semantic, needs per-file review |
| 4 | `ruff format` sweep (624 files) | last, single mechanical commit, AST-equality check per file |

Each phase lands as its own PR with `python scripts/lint_gate.py --update-baseline`, so the
ledger only ever moves down and the diff of `docs/lint-baseline.json` is the receipt.
