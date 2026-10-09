import pytest

from tensorfem.validation import Reference, require_all, validate


def test_formal_tolerance_cannot_exceed_three_percent():
    with pytest.raises(ValueError, match="3%"):
        Reference("case", "u", 1.0, "m", "analytical", tolerance=0.031)


def test_validation_gate_is_strictly_below_three_percent():
    reference = Reference("case", "u", 100.0, "mm", "analytical")
    assert validate(reference, 97.01).passed
    assert not validate(reference, 97.0).passed


def test_suite_fails_closed():
    reference = Reference("case", "u", 1.0, "m", "analytical")
    with pytest.raises(AssertionError, match="accuracy gate"):
        require_all([validate(reference, 0.96)])
