"""Repository-wide pytest configuration: shared session caches and the test lanes.

Two independent concerns live here; both are repository-wide mechanics that no
single test module owns.

1. Session-scoped caches for expensive shared reference solutions
-----------------------------------------------------------------
Background (see ``docs/test-duplication-audit.md``): the Cook-membrane audit
family (``tests/test_cook_*.py``, five modules) repeatedly recomputes two
expensive objects

* ``tensorfem.cook_stress_reference.solve_q8(divisions)`` - the independent Q8
  reference solve (Jacobi-PCG, tens of seconds at ``divisions=24``), and
* ``tensorfem.additional_benchmark_fields.cook_fields(divisions)`` - the
  production Q4 field (seconds at ``divisions=8``, minutes at ``divisions=64``),

and the call sites are identical: ``solve_q8(12)`` occurs five times and
``solve_q8(24)`` four times across three modules.  Nothing in the read-only
consumers (``probe_displacement``, ``element_centre_accuracy``,
``loaded_edge_agreement``, ``cross_check_pointwise_recovery``, ...) mutates the
returned dictionaries, so one computation per ``divisions`` value per session is
sufficient.

The fixtures return a *lookup function* rather than a value, so a test asks for the
discretisation it needs (``cook_q8(24)``) instead of the conftest hard-coding a grid.
Every call hands back a deep copy: a future test that mutates the field it received
cannot poison the cache for the rest of the session.

Correctness of the cache is itself pinned by
``tests/test_cook_consolidation_equivalence.py`` (a cached entry must equal a fresh
computation, and distinct keys must not alias).

2. The two test lanes of TensorFEM
----------------------------------
The suite mixes two populations that are both essential but have very
different cost profiles: element/benchmark/contract checks that finish in
well under a second each, and *qualification* studies (mesh-convergence
sweeps, long nonlinear and contact paths, industrial sparse factorisations)
that need tens of seconds to minutes each.  Running everything on every edit is
what eventually tempts a maintainer to delete or xfail the heavy cases -- the
regression this repository refuses to buy speed with.

The lanes keep every assertion and every test in place and only change *which*
tests run:

* full lane -- plain ``pytest``.  Nothing is deselected.  This is the lane CI
  and the release evidence run; the only lane that can certify the tree.
* fast lane -- ``pytest --fast``, exactly equivalent to ``pytest -m "not
  slow"``.  Tests registered as slow are deselected.  An injected defect in
  any *unmarked* test still turns the fast lane red, because the lane never
  touches a body or an assertion.

Registration is data, not scattered decorators: ``tests/slow_lane_manifest.txt``
lists the node ids, this conftest marks them ``pytest.mark.slow`` at collection
time, and ``scripts/slow_lane_audit.py`` re-derives the candidate list from a
``--durations``/JUnit measurement so the registry can be reviewed as a diff.
A marker registered centrally cannot drift out of step with itself, and it
leaves a test file's own content -- including its formatting -- untouched.
``tests/test_slow_lane.py`` pins the contract from both ends: the hermetic
lane mechanics (deselect, equivalence with ``-m``, red-on-injected-defect in
both lanes) and the registry (every entry names a live test, nothing heavy is
missing).
"""

from __future__ import annotations

import copy
import functools
from pathlib import Path

import pytest

# --- shared session caches ------------------------------------------------


def _memoised(compute, copy_result):
    """Wrap ``compute`` in a per-session memo that hands out copies of the result."""

    cache: dict[int, object] = {}

    @functools.wraps(compute)
    def lookup(divisions):
        key = int(divisions)
        if key not in cache:
            cache[key] = compute(key)
        result = cache[key]
        return copy.deepcopy(result) if copy_result else result

    lookup.cache = cache  # introspection for the cache-invariant tests
    return lookup


@pytest.fixture(scope="session")
def cook_q8():
    """Session cache of ``cook_stress_reference.solve_q8`` keyed by ``divisions``."""
    from tensorfem import cook_stress_reference

    return _memoised(cook_stress_reference.solve_q8, copy_result=True)


@pytest.fixture(scope="session")
def cook_case():
    """Session cache of the production Q4 Cook field, ``cook_fields(divisions)``."""
    from tensorfem.additional_benchmark_fields import cook_fields

    return _memoised(cook_fields, copy_result=True)


# --- the two lanes --------------------------------------------------------

LANE_OPTION = "--fast"
SLOW_MARKER = "slow"
REGISTRY_PATH = Path(__file__).with_name("slow_lane_manifest.txt")

_registry_cache: frozenset[str] | None = None


def base_node_id(node_id: str) -> str:
    """Strip a parametrisation suffix: ``...::test_x[a-1]`` -> ``...::test_x``."""
    return node_id.split("[", 1)[0]


def read_registry(path: Path = REGISTRY_PATH) -> frozenset[str]:
    """Registered slow node ids; ``#`` comments and blank lines are ignored."""
    global _registry_cache
    if path == REGISTRY_PATH and _registry_cache is not None:
        return _registry_cache
    entries: set[str] = set()
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                entries.add(line)
    result = frozenset(entries)
    if path == REGISTRY_PATH:
        _registry_cache = result
    return result


def pytest_addoption(parser):
    group = parser.getgroup("tensorfem-lanes")
    if any(option.dest() == "fast_lane" for option in group.options):
        # A re-imported/re-copied conftest must not double-register the option.
        return
    group.addoption(
        LANE_OPTION,
        action="store_true",
        dest="fast_lane",
        default=False,
        help=(
            "fast lane: deselect registered slow tests (equivalent to "
            "-m 'not slow').  No test is skipped or weakened, so a defect in "
            "any unmarked test still fails the run."
        ),
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Attach the ``slow`` marker from the registry, then apply the lane.

    ``tryfirst`` matters: ``-m`` selection is itself implemented as a
    ``pytest_collection_modifyitems`` hook, so the markers have to be attached
    before pytest evaluates the mark expression -- otherwise ``--fast`` would
    work while the documented ``-m "not slow"`` equivalent silently selected
    everything.
    """
    registry = read_registry()
    if registry:
        for item in items:
            if base_node_id(item.nodeid) in registry:
                item.add_marker(pytest.mark.slow)

    if not config.getoption("fast_lane"):
        return
    deselected = [item for item in items if item.get_closest_marker(SLOW_MARKER) is not None]
    if not deselected:
        return
    keep = set(map(id, deselected))
    config.hook.pytest_deselected(items=deselected)
    items[:] = [item for item in items if id(item) not in keep]