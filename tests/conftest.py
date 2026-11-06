"""Session-scoped caches for expensive shared reference solutions.

Background (see ``docs/test-duplication-audit.md``): the Cook-membrane audit family
(``tests/test_cook_*.py``, five modules) repeatedly recomputes two expensive objects

* ``tensorfem.cook_stress_reference.solve_q8(divisions)`` - the independent Q8
  reference solve (Jacobi-PCG, tens of seconds at ``divisions=24``), and
* ``tensorfem.additional_benchmark_fields.cook_fields(divisions)`` - the production
  Q4 field (seconds at ``divisions=8``, minutes at ``divisions=64``),

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
"""

from __future__ import annotations

import copy
import functools

import pytest


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