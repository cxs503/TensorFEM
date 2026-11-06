"""Equivalence guards for the m4718 Cook-membrane test consolidation.

Eleven inline ``reference.solve_q8(...)`` / ``cook_fields(...)`` call sites in the
four ``test_cook_*.py`` modules (a 40 s solve at ``divisions=24`` and a 172 s field at
``divisions=32`` on the development node) now run once per session through the
``cook_q8`` / ``cook_case`` fixtures in ``tests/conftest.py``, and two tests that
evaluated the *same* ``cross_check_pointwise_recovery(cook_fields(8))`` result were
merged into one.  Sharing a computed reference across tests is only sound if

1. a cached entry is bit-for-bit what a fresh computation returns,
2. distinct discretisations never alias each other,
3. a test that mutates the payload it received cannot poison the session cache, and
4. the merged assertion set is still *sensitive* - it must fail when the quantity it
   guards is wrong (otherwise the merge bought wall clock with coverage).

Items 1-3 are the usual failure modes of a memoised fixture; item 4 is the negative
control for the merge itself, kept as a permanent test rather than a one-off check.
"""

from __future__ import annotations

import ast
import json
import pathlib

import numpy as np

from tensorfem import cook_stress_reference as reference

HERE = pathlib.Path(__file__).resolve().parent
CONSOLIDATION = HERE.parent / "artifacts" / "cook_assertion_consolidation.json"

SURVIVOR_MODULE = "test_cook_oracle_operator.py"
SURVIVOR_TEST = "test_pointwise_recovery_reproduces_the_stored_q4_centre_stress"
ABSORBED_MODULE = "test_cook_stress_recovery_verification.py"
ABSORBED_TEST = "test_cross_check_of_pointwise_recovery_passes"


def _canonical_asserts(path: pathlib.Path, test_name: str) -> list[str]:
    """Canonical ``assert`` predicates of one test function (message and ``is True`` dropped)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == test_name:
            predicates = []
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assert):
                    expression = " ".join(ast.unparse(sub.test).split())
                    if expression.endswith(" is True"):
                        expression = expression[: -len(" is True")]
                    predicates.append(expression)
            return predicates
    raise AssertionError(f"{test_name} not found in {path.name}")


def test_cached_q8_solution_equals_a_fresh_solve(cook_q8):
    """The session cache must not change the reference solution in any way."""
    cached = cook_q8(2)
    fresh = reference.solve_q8(2)
    assert cached["divisions"] == fresh["divisions"] == 2
    assert np.array_equal(cached["nodes"], fresh["nodes"])
    assert np.array_equal(cached["elements"], fresh["elements"])
    assert np.array_equal(cached["displacement"], fresh["displacement"])
    assert cached["relative_residual"] == fresh["relative_residual"]
    assert cached["load_resultant"] == fresh["load_resultant"]


def test_cached_case_equals_a_fresh_production_field(cook_case):
    """The cached production Q4 field must equal a fresh ``cook_fields`` payload."""
    from tensorfem.additional_benchmark_fields import cook_fields

    cached = cook_case(2)
    fresh = cook_fields(2)
    assert float(cached["mesh"]) == float(fresh["mesh"])
    assert np.array_equal(np.asarray(cached["nodes"]), np.asarray(fresh["nodes"]))
    assert np.array_equal(np.asarray(cached["elements"]), np.asarray(fresh["elements"]))
    assert np.array_equal(
        np.asarray(cached["displacement"]), np.asarray(fresh["displacement"])
    )
    assert np.allclose(np.asarray(cached["stress"]), np.asarray(fresh["stress"]), rtol=0, atol=0)
    assert cached["stress_accuracy_qualified"] == fresh["stress_accuracy_qualified"]


def test_one_computation_per_discretisation(cook_q8):
    """A repeated key must not recompute; a new key must add exactly one entry."""
    before = len(cook_q8.cache)
    cook_q8(3)
    after_first = len(cook_q8.cache)
    cook_q8(3)
    assert after_first == before + 1
    assert len(cook_q8.cache) == after_first


def test_distinct_discretisations_do_not_alias(cook_q8):
    """Keying is per discretisation: two keys must return their own solution."""
    coarse = cook_q8(2)
    fine = cook_q8(3)
    assert coarse["divisions"] == 2 and fine["divisions"] == 3
    assert coarse["nodes"].shape != fine["nodes"].shape
    assert coarse["nodes"].shape[0] != fine["nodes"].shape[0]


def test_a_mutating_caller_cannot_poison_the_session_cache(cook_case):
    """Fixtures hand out copies: in-place edits by one test must stay local."""
    pristine = np.asarray(cook_case(2)["displacement"], dtype=np.float64).copy()
    assert np.abs(pristine).max() > 0.0

    poisoned = cook_case(2)
    assert poisoned["displacement"] is not cook_case(2)["displacement"]
    poisoned["displacement"][0] = 123.0
    poisoned["nodes"].clear()
    poisoned["mesh"] = -1

    recovered = cook_case(2)
    assert recovered["mesh"] == 2
    assert len(recovered["nodes"]) > 0
    assert np.array_equal(np.asarray(recovered["displacement"], dtype=np.float64), pristine)
    assert np.abs(np.asarray(recovered["displacement"], dtype=np.float64)).max() > 0.0


def test_merged_cross_check_assertions_reject_a_corrupted_stored_field(cook_case):
    """Negative control: the merged assertions still fail on a wrong stored field."""
    case = cook_case(8)
    good = reference.cross_check_pointwise_recovery(case)
    assert good["passed"] is True
    assert good["max_relative_difference"] < 1e-10

    corrupted = dict(case)
    corrupted["stress"] = np.asarray(case["stress"], dtype=np.float64) * 1.05
    bad = reference.cross_check_pointwise_recovery(corrupted)
    assert bad["passed"] is False
    assert bad["max_relative_difference"] > bad["tolerance"]
    assert bad["max_relative_difference"] > 1e-10


def test_the_merged_test_carries_every_assertion_of_the_absorbed_test():
    """The absorbed test's claims must all be present in the surviving test."""
    record = json.loads(CONSOLIDATION.read_text(encoding="utf-8"))
    assert record["coverage_preserved"] is True
    assert record["lost_assertions"] == []

    absorbed = [
        row for row in record["removed"] if row["removed_from"] == f"{ABSORBED_MODULE}::{ABSORBED_TEST}"
    ]
    assert absorbed, "the consolidation record no longer mentions the absorbed test"
    inherited = set()
    for row in absorbed:
        assert row["survives_in"] == [f"{SURVIVOR_MODULE}::{SURVIVOR_TEST}"], row
        inherited.update(row["matched_asserts"])
    assert inherited == {
        "check['passed']",
        "check['max_relative_difference'] <= check['tolerance']",
    }

    predicates = set(_canonical_asserts(HERE / SURVIVOR_MODULE, SURVIVOR_TEST))
    assert inherited <= predicates, (inherited - predicates)

    absorbed_module = (HERE / ABSORBED_MODULE).read_text(encoding="utf-8")
    assert f"def {ABSORBED_TEST}(" not in absorbed_module


def test_the_consolidated_modules_share_one_reference_solution():
    """After consolidation every cook module goes through the session cache."""
    for module in (
        "test_cook_oracle_operator.py",
        "test_cook_stress_audit_closure.py",
        "test_cook_stress_reference.py",
        "test_cook_stress_recovery_verification.py",
    ):
        source = (HERE / module).read_text(encoding="utf-8")
        tree = ast.parse(source)
        inline = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            owner = getattr(getattr(func, "value", None), "id", "")
            if name in {"solve_q8", "cook_fields"} and owner in {"reference", "cook_reference"}:
                inline.append(f"{owner}.{name}")
        assert not inline, f"{module} still recomputes {inline} inline"

