"""Guard on the mechanism attribution of ``cylindrical_shell4`` (CASE-FEM-SHELL-MODAL).

The modal anchor ``benchmarks/verified/fem_shell_modal`` reports two honest FAILs
(spectrum 21-51 % below the Donnell oracle; rotation-only modes in the first ten).
The attribution recorded in ``docs/CASE-FEM-SHELL-MODAL-mechanism-attribution.md``
is: the production element carries 5 spurious mechanisms per element, they come
from the **one-point membrane** integration, and switching the shear rule alone
does not remove them.

These assertions pin that attribution so a later formulation change cannot pass
silently: the day the production element becomes rank sufficient this test fails
and forces the debt entry to be revisited.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[1] / "benchmarks" / "verified"
          / "fem_shell_modal" / "rank_census.py")


def _census_module():
    spec = importlib.util.spec_from_file_location("fem_shell_modal_rank_census", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def census():
    return _census_module().run()


def test_production_element_is_rank_deficient(census):
    entry = census["variants"]["production_membrane1pt_shear1pt"]
    assert entry["nullspace_dimension"] == 11, (
        "production cylindrical_shell4 should own 6 rigid + 5 spurious modes; "
        f"got {entry['nullspace_dimension']}"
    )
    assert entry["spurious_mechanisms"] == 5


def test_shear_rule_alone_does_not_cure_the_mechanisms(census):
    """MITC4/2x2 shear keeps 2-3 mechanisms, so the membrane is the source."""
    assert census["variants"]["membrane1pt_shear_mitc"]["nullspace_dimension"] == 9
    assert census["variants"]["membrane1pt_shear2x2"]["nullspace_dimension"] == 8


def test_full_membrane_integration_removes_them(census):
    """Rank sufficiency is reachable, but see the next test: it is not accuracy."""
    entry = census["variants"]["membrane2x2_shear1pt"]
    assert entry["nullspace_dimension"] == 6
    assert entry["spurious_mechanisms"] == 0


def test_full_integration_penalises_a_rigid_mode(census):
    """null < 6 is *not* a better element: one rigid mode acquires energy."""
    for name in ("membrane2x2_shear2x2", "membrane2x2_shear_mitc"):
        assert census["variants"][name]["nullspace_dimension"] < 6, (
            f"{name} is expected to penalise one rigid-body mode (straight-facet "
            "geometry + drilling penalty); if this changes, re-derive the doc"
        )


def test_recorded_variant_spectrum_matches_the_log(census):
    """The figure/JSON pair must agree with the raw run log (no hand transcription)."""
    import json

    data = json.loads((SCRIPT.parent / "integration_variants.json").read_text())
    prod = data["rules"]["membrane=1pt shear=1pt"]
    assert [row["mesh"] for row in prod] == ["4x8", "8x8", "8x16"]
    assert prod[0]["lowest_physical_hz"] == pytest.approx(100.35, abs=0.01)
    assert prod[-1]["error_vs_oracle_pct"] == pytest.approx(20.95, abs=0.01)
    # no variant reaches the 15 % anchor band
    for rows in data["rules"].values():
        assert min(row["error_vs_oracle_pct"] for row in rows) > 15.0