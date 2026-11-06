"""Pipeline end-to-end smoke test (mission 4622).

Verifies that the repository root module ``smoke_marker_m4622`` is importable
through the pytest ``pythonpath = ["src", "."]`` configuration and that it
carries the expected marker constant.  Kept intentionally trivial: its job is
to prove the code + test path is wired end to end, not to exercise numerics.
"""

from smoke_marker_m4622 import PIPELINE_SMOKE_M4622


def test_marker_constant_value() -> None:
    """The marker constant must equal the mission number as a string."""
    assert PIPELINE_SMOKE_M4622 == "4622"


def test_arithmetic_sanity() -> None:
    """Trivial arithmetic smoke assertion."""
    assert 1 + 1 == 2