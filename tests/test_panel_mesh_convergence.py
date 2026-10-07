import pytest

from tensorfem.panel_mesh_convergence import evaluate_panel_mesh_convergence


def manifest(divisions, peak, *, confirmed=True, post_peak=True):
    return {"divisions": divisions, "peak_force_n": peak,
            "peak_confirmed": confirmed, "post_peak_observed": post_peak}


def test_gate_passes_only_complete_adjacent_mesh_evidence():
    result = evaluate_panel_mesh_convergence([
        manifest(4, 509_000.), manifest(8, 505_000.), manifest(12, 503_000.),
    ])
    assert result["status"] == "passed"
    assert result["passed"]
    assert len(result["comparisons"]) == 2


def test_gate_blocks_incomplete_or_unconfirmed_mesh_without_extrapolation():
    result = evaluate_panel_mesh_convergence([
        manifest(4, 509_000.), manifest(8, 505_000., post_peak=False),
    ])
    assert result["status"] == "blocked"
    assert result["missing_divisions"] == [8, 12]
    assert result["rejected"][8] == "post_peak_not_observed"


def test_gate_reports_completed_but_nonconverged_sequence_as_failed():
    result = evaluate_panel_mesh_convergence([
        manifest(4, 509_000.), manifest(8, 450_000.), manifest(12, 400_000.),
    ])
    assert result["status"] == "failed"
    assert not result["passed"]


@pytest.mark.parametrize("tolerance", [0., -1., float("inf")])
def test_gate_rejects_invalid_tolerance(tolerance):
    with pytest.raises(ValueError):
        evaluate_panel_mesh_convergence([], relative_tolerance=tolerance)
