# Test lanes: the `slow` marker contract and the fast lane

TensorFEM's suite is one body set with two very different cost profiles:

| population | examples | cost |
|---|---|---|
| core | element formulations, analytic/reference benchmarks, API and evidence contracts | ≪ 1 s each |
| qualification | mesh-convergence sweeps, long nonlinear/contact paths, industrial sparse factorisations, signed-evidence reproduction | 5 s – 4 min each |

On the reference run used to seed the registry (1967 items, full lane), the
**114 heaviest cases carried 3375 s of the 3855 s total test time (88 %)**;
the remaining ~1850 cases cost ~480 s together. Running the tail on every edit
is what eventually tempts a maintainer to delete or `xfail` the heavy cases.
The lanes buy the ergonomics without paying that price.

The registry is re-measured against the tree it ships with (2044 collected
items): the 129 entries below are the earlier 118 plus the 11 cases the newer
suite pushed past the 5 s cut-off (the `test_torsion_rectangular_section`
convergence family at 6-131 s per case, and
`test_elastica_cantilever.py::test_bvp_route_is_independent_of_quadrature` at
~10 s). Nothing was unregistered: a case gets cheaper on a faster host, and
over-registering only costs an extra full-lane run. One inert legacy decorator
(`tests/test_elastica_cantilever.py`, documented there as "one cheap
load-controlled FE case") was folded out instead of being registered, because
`--strict-markers` turns a stray decorator into a real deselection and that
case is neither slow nor passing: hiding a red test behind the fast lane would
be exactly the abuse the lanes exist to prevent.

## What the lanes are

| lane | command | semantics |
|---|---|---|
| full | `pytest` | nothing deselected; **this is what CI and the release evidence run certify** |
| fast | `pytest --fast` | identical to `pytest -m "not slow"`; registered slow cases deselected |

The lanes change *selection only*. No test is deleted, skipped, `xfail`-ed, and
no assertion or tolerance is touched — so the fast lane remains a real gate for
every unmarked test: an injected defect in an unmarked test still turns it red
(pinned by `tests/test_slow_lane.py::test_fast_lane_still_reddens_on_a_defect_in_an_unmarked_test`,
and mirrored by the full-lane counterpart for a registered test).

`--fast` is opt-in. `pyproject.toml` sets `addopts = "--strict-markers"` and a
bare `pytest` never deselects anything, so the reduced lane can only be
requested explicitly — CI cannot drift onto it silently
(`test_default_lane_is_the_full_lane`, `test_ci_runs_the_full_lane`).

## Where the marker comes from

`slow` is registered in `[tool.pytest.ini_options].markers`, and the set of
slow tests is **data**, not a scattering of decorators:

```
tests/slow_lane_manifest.txt      # one base node id per line, sorted, reviewable as a diff
tests/conftest.py                 # attaches pytest.mark.slow from the manifest at collection
```

A central registry was chosen over per-test decorators deliberately: the
alternative touches ~64 legacy test modules, which would put 64 files into the
change-scoped `ruff format --check` gate that they currently (pre-existing)
fail, and it would let the "which tests are slow" answer drift across the tree.
With one registry, the answer is one file, and the marker is still a real
pytest marker — `pytest -m slow`, `pytest --collect-only -m "not slow"`,
`--strict-markers`, and IDE marker lists all behave normally.

`tests/conftest.py` attaches the marker in
`pytest_collection_modifyitems(..., tryfirst=True)`. The ordering is load
bearing: `-m` filtering is itself implemented as a `pytest_collection_modifyitems`
hook, so marking must happen before pytest evaluates the mark expression —
otherwise `--fast` would work while the documented `-m "not slow"` equivalent
silently selected everything. `test_fast_lane_is_exactly_the_not_slow_expression`
and `test_registry_matches_a_fresh_collection` pin the equivalence.

## Registration policy

A test belongs in the registry when either

1. it measured **≥ 5 s** on the reference run (`--junitxml` full lane, 12
   worker local run), or
2. it is heavyweight *by construction* — the four opt-in
   `TENSORFEM_RUN_SLOW_*` qualification studies, which are minute-scale
   whenever they actually run (and are 0 s when gated off, so measurement
   alone would never surface them).

The measurement is a floor, not a ceiling: the loong64 CI node is slower than
the x86 reference, so a case measured just under the cut-off may still be
registered. Over-registering costs full-lane coverage nothing; under-registering
silently slows every fast-lane run.

## Maintaining the registry

```bash
# full lane, once, with machine-readable timings
pytest -q -n 12 --dist loadfile --junitxml=artifacts/junit_full.xml

# what is heavy, what is registered, what drifted
python scripts/slow_lane_audit.py --junitxml artifacts/junit_full.xml --threshold 5
python scripts/slow_lane_audit.py --junitxml artifacts/junit_full.xml --threshold 5 --check

# or from a --durations log
pytest -q --durations=0 > artifacts/durations.log
python scripts/slow_lane_audit.py artifacts/durations.log --threshold 5
```

`--check` exits non-zero when a case above the threshold is **not** registered
(the drift mode that matters), and the human report also lists registered
entries that are no longer measured (renamed/cheapened cases). The audit
aggregates parametrised ids onto their base node id, matching the registry
format.

One case is deliberately kept out of the registry even when a loaded host
measures it above the cut-off: the hermetic lane guards themselves
(`tests/test_slow_lane.py`, ~1 s of real work plus two `pytest` subprocesses
each, so 6 s on a fully loaded x86 node). Registering them would remove the
negative controls from the fast lane, which is the lane they protect;
`--check` on a busy machine therefore reports that one row and nothing else.

Registry integrity is enforced in the suite itself:

* every entry must name a live test function in the tree
  (`test_every_registry_entry_names_a_live_test_function`) — so a rename fails
  loudly instead of silently unregistering;
* entries must be base node ids, sorted and unique
  (`test_registry_entries_are_canonical_and_sorted`);
* the opt-in heavyweight studies must stay registered, and the registry must
  still span the heavy families (`test_registry_covers_the_opt_in_heavyweight_studies`,
  `test_registry_covers_the_heavy_tail`);
* `pytest --fast` and `pytest -m slow` must partition the real suite
  (`test_registry_matches_a_fresh_collection`, itself registered because it
  collects the suite twice).

## Interaction with `pytest-timeout` and xdist

The lanes are orthogonal to both: `--timeout` still applies per test, and the
fast lane composes with `-n auto` (`pytest --fast -n 12`) since selection
happens before distribution.