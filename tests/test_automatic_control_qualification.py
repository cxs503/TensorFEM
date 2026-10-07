import copy

from tensorfem.automatic_control_qualification import qualify_automatic_step_path


def _point(step, x, force, displacement, external, internal, iterations=3):
    return {"step": step, "edge_shortening_m": x, "force_n": force,
            "centre_deflection_m": displacement, "external_work_j": external,
            "internal_energy_j": internal, "newton_iterations": iterations}


def test_automatic_path_qualification_interpolates_and_fails_closed():
    fixed = {"point_history": [
        _point(1, 1., 10., .1, 2., 2.01),
        _point(2, 2., 20., .2, 4., 4.01),
        _point(3, 3., 30., .3, 6., 6.01),
    ], "chunks": [{"attempted": 3, "rejected": 1}]}
    decision = {"automatic_algorithm_switching_enabled": False,
                "step_application": {"current_step_size": .1,
                                     "applied_step_size": .05}}
    automatic = {"point_history": [
        _point(1, 1., 10., .1, 2., 2.01),
        _point(2, 2.5, 25.1, .2501, 5.001, 5.011),
    ], "chunks": [{"attempted": 2, "rejected": 0}],
        "nonlinear_controller_decisions": [decision]}
    restarted = copy.deepcopy(automatic)
    report = qualify_automatic_step_path(automatic, fixed, restarted)
    assert report["passed"] is True
    assert report["restart_exact"] is True
    assert report["step_scaling_events"] == 1
    assert report["rejected_step_reduction"] == 1
    assert report["newton_iteration_reduction"] == 3
    assert len(report["evidence_sha256"]) == 64

    restarted["point_history"][-1]["force_n"] = 26.
    assert qualify_automatic_step_path(automatic, fixed, restarted)["passed"] is False

