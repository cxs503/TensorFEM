import copy

import pytest

from tensorfem.marine_structural_qualification import (
    STRUCTURAL_IDS, run_marine_structural_qualification,
    validate_marine_structural_qualification,
)


@pytest.fixture(scope="module")
def report():
    return run_marine_structural_qualification()


def test_tensorfem_only_qualification_is_traceable_and_passes(report):
    assert validate_marine_structural_qualification(report) is report
    assert report["passed"] and "no TensorLBM" in report["solver_scope"]
    assert len(report["scalar_evidence"]) == len(STRUCTURAL_IDS) == 15
    assert report["classical_shell_suite"]["passed"]
    assert all(row["error"] < row["tolerance"] <= 0.03 for row in report["scalar_evidence"])


def test_capability_boundaries_are_explicit_and_fail_closed(report):
    categories = report["categories"]
    assert categories["plate_shell_linear_and_large_rotation"]["status"] == "qualified"
    assert categories["local_plate_buckling"]["status"] == "qualified_prototype"
    assert categories["ultimate_hull_girder_strength"]["status"] == "qualified_section_prototype"
    assert categories["fatigue_damage_primitives"]["status"] == "qualified_prototype"
    assert categories["imperfect_plate_postbuckling"]["status"] == "qualified_reduced_order"
    assert categories["multi_component_progressive_yielding"]["status"] == "qualified_section_prototype"
    assert categories["spectrum_fatigue_and_crack_growth"]["status"] == "qualified_primitives"
    assert categories["full_shell_progressive_collapse"]["status"] == "not_qualified"
    tampered = copy.deepcopy(report)
    tampered["categories"]["full_shell_progressive_collapse"]["status"] = "qualified"
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_marine_structural_qualification(tampered)


def test_classical_shell_evidence_has_primary_doi_and_coarse_negative_evidence(report):
    for case in report["classical_shell_suite"]["cases"]:
        assert case["case"]["citation"]["doi"].startswith("10.")
        assert any(not row["passed"] for row in case["results"])
        assert all(row["passed"] for row in case["results"] if row["role"] == "qualification")
