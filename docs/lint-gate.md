# Lint gates and the ruff ratchet

## Why this exists

The TensorFEM tree predates any lint discipline: 4268 ruff violations are spread
over 344 files (`E702`/`E701` semicolon-packed statements, `F401` unused
imports, `E741` ambiguous names, star imports) and 616 files are not
`ruff format` clean. CI used to run

```yaml
- run: ruff check src tests scripts examples
- run: ruff format --check src tests scripts examples
```

inside a job marked `continue-on-error: true`. Two things were wrong with that:

* the job could never pass, so the check reported `failure` on **every** pull
  request — a permanently red signal that cannot be told apart from a real
  regression (this is what blocked PR #137);
* `continue-on-error` only keeps the *workflow* green — GitHub still publishes
  the failing job conclusion, which is exactly what a pull-request gate reads.

Making the backlog advisory therefore hides the signal instead of removing the
red, and gating on a clean tree would require an unreviewable reformat of 616
files in one commit.

## What runs now

The `lint` job has two blocking gates and two advisory reporters:

| step | blocking | scope |
|---|---|---|
| `ruff check --select E9,F63,F7,F82,F811 src tests scripts examples` | yes | bug-class rules (syntax, undefined names, comprehension rebindings, `is` literals). Clean tree-wide today, so it is required outright. |
| `python scripts/lint_ratchet.py` | yes | every default rule, compared against the frozen backlog in `scripts/lint_baseline.json`; new debt fails. |
| `ruff check src tests scripts examples` | no | full backlog, printed so the drain stays visible. |
| `ruff format --check src tests scripts examples` | no | unformatted files that pre-date the ratchet. |

The ratchet is the gate that can be satisfied today *and* still refuses
regressions: pre-existing violations are frozen per file and per rule, so a file
may keep its 40 semicolons but may not gain a 41st, may not grow a new rule, and
may not become the first unformatted file since the freeze.

## Day-to-day

```bash
pip install -e ".[dev]"            # ruff is pinned, see below
python scripts/lint_ratchet.py     # same check CI runs; exit 1 = new debt
pytest -q tests/test_lint_ratchet.py
```

If the ratchet fails, fix the reported items (the message names
`<path>: <rule> <budget> -> <count>`). Do **not** re-freeze to silence a real
regression.

## Paying debt down / re-freezing

`scripts/lint_baseline.json` records the *exact* current state, so it must be
regenerated whenever violations are actually removed:

```bash
ruff check --fix src tests scripts examples      # safe autofixes (144 of them)
python scripts/lint_ratchet.py --update          # re-freeze the smaller backlog
git diff --stat scripts/lint_baseline.json       # the numbers must go down
```

The `--update` diff is the audit trail: a freeze that *raises* a count is a
regression being laundered, and reviewers should treat it as such.

Draining the backlog to zero lets the two advisory steps be promoted to gates
and the ratchet file to be deleted; until then the bug-class gate and the
ratchet are what protect the tree.

## Why ruff is pinned

`pyproject.toml` pins `ruff==0.15.0` for the `dev` extra. Rule counts are a
function of the ruff release: a new release can add findings to the default rule
set and would make the frozen baseline meaningless. The ratchet refuses to
compare numbers produced by a different release, so the pin and the baseline
must be bumped in the same commit.