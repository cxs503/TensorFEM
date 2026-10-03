import copy

import pytest

from tensorfem.industrial_p0_phase3_qualification import (
    run_industrial_p0_phase3_qualification,
    validate_industrial_p0_phase3_qualification,
)


@pytest.fixture(scope="module")
def report():
    return run_industrial_p0_phase3_qualification()


def test_phase3_report_is_reproducible_and_signed(report):
    assert validate_industrial_p0_phase3_qualification(report) is report
    rerun = run_industrial_p0_phase3_qualification()
    assert rerun["report_hash"] == report["report_hash"]
    assert rerun["hertz_axisymmetric"]["evidence_hash"] == report["hertz_axisymmetric"]["evidence_hash"]


def test_axisymmetric_hertz_passes_every_declared_gate(report):
    evidence = report["hertz_axisymmetric"]
    assert evidence["passed"] is True
    assert all(evidence["gates"].values())
    assert evidence["thresholds"]["relative_error_strict_upper_bound"] == 0.03
    assert [r["radial_elements"] for r in evidence["mesh_sequence"]] == [16, 24, 48]
    fine = evidence["mesh_sequence"][-1]["relative_errors"]
    for name in ("load", "contact_radius", "peak_pressure", "pressure_l2", "normalized_overlap"):
        assert fine[name] < 0.03


def test_unproved_3d_and_panel_claims_remain_blocked(report):
    categories = report["categories"]
    assert categories["axisymmetric_deformable_halfspace_hertz"]["status"] == "qualified"
    assert categories["finite_rotation_layered_shell_arc_integration"]["status"] == "qualified"
    assert categories["deformable_to_deformable_contact_3d"]["status"] == "blocked"
    assert categories["marine_panel_peak_and_postpeak"]["status"] == "blocked"
    assert report["marine_panel_ultimate"]["status"] == "blocked"
    assert report["qualified_count"] == 2 and report["blocked_count"] == 2


def test_finite_rotation_integration_is_qualified_without_promoting_panel(report):
    evidence = report["finite_rotation_layered_shell"]
    assert evidence["passed"] is True
    assert evidence["accepted_equilibrium_norm"] < evidence["equilibrium_strict_upper_bound"]
    assert evidence["rejected_path_converged"] is False
    assert evidence["rejected_path_preserved_committed_history"] is True
    assert "not panel peak/post-peak" in evidence["boundary"]


def test_nested_and_report_tampering_fail_closed(report):
    changed = copy.deepcopy(report)
    changed["hertz_axisymmetric"]["mesh_sequence"][-1]["relative_errors"]["load"] = 0.0
    with pytest.raises(ValueError, match="Hertz evidence hash mismatch"):
        validate_industrial_p0_phase3_qualification(changed)

    promoted = copy.deepcopy(report)
    promoted["categories"]["marine_panel_peak_and_postpeak"]["status"] = "qualified"
    with pytest.raises(ValueError, match="panel ultimate claim must remain blocked"):
        validate_industrial_p0_phase3_qualification(promoted)
