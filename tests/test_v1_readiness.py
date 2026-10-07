import copy

import pytest

from tensorfem.v1_readiness import evaluate_v1_readiness, validate_v1_readiness


def _capabilities():
    base = {"passed": True, "maximum_relative_error": .001,
            "restart_or_rollback": True}
    return {
        "nonlinear_shell_robustness": {**base, "difficult_step_improved": True},
        "general_double_deformable_contact_3d": {
            **base, "scope": "general_surface_to_surface"},
        "shell_sparse_scalability": {
            **base, "qualified_dofs": 10_000, "storage_reduction": .2},
        "release_quality": {
            **base, "full_regression": True, "offline_install": True,
            "api_audit": True},
    }


def _panels():
    return [{"divisions": n, "peak_force_n": force,
             "peak_confirmed": True, "post_peak_observed": True}
            for n, force in ((4, 100.), (8, 99.), (12, 98.))]


def test_v1_readiness_passes_only_complete_scoped_evidence():
    report = evaluate_v1_readiness(_capabilities(), _panels())
    assert report["ready_for_1_0"] and report["blockers"] == []
    assert validate_v1_readiness(report) == report


def test_v1_readiness_blocks_partial_contact_and_missing_panel_peak():
    capabilities = _capabilities()
    capabilities["general_double_deformable_contact_3d"]["scope"] = "planar_node_triangle"
    report = evaluate_v1_readiness(capabilities, _panels()[:2])
    assert not report["ready_for_1_0"]
    assert "general_double_deformable_contact_3d" in report["blockers"]
    assert "marine_panel_4_8_12_peak_postpeak" in report["blockers"]


def test_v1_readiness_fails_closed_on_tampering_and_bad_thresholds():
    report = evaluate_v1_readiness(_capabilities(), _panels())
    broken = copy.deepcopy(report); broken["ready_for_1_0"] = False
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_v1_readiness(broken)
    with pytest.raises(ValueError, match="at most 3%"):
        evaluate_v1_readiness({}, [], benchmark_tolerance=.031)
