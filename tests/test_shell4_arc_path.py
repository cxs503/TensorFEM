import torch

from tensorfem.shell4_arc_path import (
    cantilever_shell_problem, run_shell4_arc_path_evidence,
    trace_cantilever_shell,
)


def test_shell4_arc_points_satisfy_actual_discrete_equilibrium():
    _mesh, _load, problem, free = cantilever_shell_problem()
    result, result_free = trace_cantilever_shell(step_size=0.1, steps=4)
    assert result.converged and torch.equal(free, result_free)
    load_norm = torch.linalg.vector_norm(problem.reference_load)
    for point in result.points:
        internal, _ = problem.internal_tangent(point.displacement)
        residual = internal - point.load_factor * problem.reference_load
        assert torch.linalg.vector_norm(residual) / load_norm < 1e-7


def test_shell4_arc_path_has_three_percent_step_convergence():
    report = run_shell4_arc_path_evidence()
    assert report["passed"]
    assert report["relative_step_change"]["load_factor"] < 0.03
    assert report["relative_step_change"]["tip_displacement"] < 0.03
    assert abs(report["rows"][-1]["tip_displacement"]) > 0.1
    assert report["rows"][-1]["maximum_residual_norm"] < 1e-6
