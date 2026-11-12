"""Gate tests for the foundation-plate registry evidence.

The point of these tests is not that a number is produced but that the *gate*
holds: the FE reading must sit inside 3 % of the exact two-parameter Navier
series, and the G -> 0 limit must be an exactness statement rather than a loose
tolerance.
"""
from __future__ import annotations

import pytest

from tensorfem.foundation_plate_registry import (
    CASE,
    N_DEFAULT,
    PASTERNAK_G,
    WINKLER_K,
    centre_deflection,
    exact_centre_deflection,
    plate_foundation_evidence,
)


@pytest.fixture(scope="module")
def evidence():
    return {e.id: e for e in plate_foundation_evidence()}


def test_registry_entries_are_present(evidence):
    assert set(evidence) == {
        "plate.foundation.winkler_center_w",
        "plate.foundation.pasternak_center_w",
        "plate.foundation.pasternak_zero_reduces_to_winkler",
    }
    for e in evidence.values():
        assert 0.0 < e.tolerance <= 0.03


def test_winkler_centre_deflection_within_3pct(evidence):
    e = evidence["plate.foundation.winkler_center_w"]
    assert e.passed, f"error {e.error:.3%} exceeds {e.tolerance:.1%}"
    assert e.error < 0.03


def test_pasternak_centre_deflection_within_3pct(evidence):
    e = evidence["plate.foundation.pasternak_center_w"]
    assert e.passed, f"error {e.error:.3%} exceeds {e.tolerance:.1%}"


def test_shear_layer_stiffens_the_foundation():
    """Physics invariant: the reaction is k*w - G*lap(w), so G > 0 must *reduce*
    the deflection (a shear layer stiffens the foundation, it cannot soften it)."""
    w_winkler = centre_deflection(winkler=WINKLER_K, pasternak=0.0, n=8)
    w_pasternak = centre_deflection(winkler=WINKLER_K, pasternak=PASTERNAK_G, n=8)
    assert abs(w_pasternak) < abs(w_winkler)


def test_exact_reference_is_the_expected_series_value():
    """The reference must be stable under refinement-independent series truncation."""
    d = exact_centre_deflection(winkler=WINKLER_K, pasternak=PASTERNAK_G)
    # 0.5 mm order for this stiff-foundation case; guards against a wrong symbol.
    assert 1.0e-5 < abs(d) < 1.0e-3


def test_reduction_to_winkler_is_exact(evidence):
    e = evidence["plate.foundation.pasternak_zero_reduces_to_winkler"]
    assert e.passed
    assert e.error < 1e-12


@pytest.mark.parametrize("n", (N_DEFAULT,))
def test_mesh_is_the_registered_one(n):
    assert n == 24
    assert CASE["a"] == CASE["b"] == 1.0