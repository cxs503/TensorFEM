import torch

from tensorfem.finite_rotation_layered_shell4 import solve_finite_rotation_arc_path
from tensorfem.layered_shell4_plasticity import LayeredShell4Model, LayeredShell4State
from tensorfem.marine_panel_execution import execute_panel_chunked_job
from tensorfem.panel_path_evidence import shell_stored_energy


D = torch.float64


def _cantilever():
    nodes = torch.tensor([[0., 0., 0.], [1., 0., 0.],
                          [1., 1., 0.], [0., 1., 0.]], dtype=D)
    model = LayeredShell4Model(
        nodes, torch.tensor([[0, 1, 2, 3]]), 200000., .3, .08,
        250., 1400., layers=3,
    )
    fixed = torch.tensor(list(range(6))+list(range(18, 24)))
    load = torch.zeros(24, dtype=D)
    load[8] = load[14] = -100.
    return model, fixed, load, LayeredShell4State.virgin(model)


def _options(state):
    return dict(steps=1, step_size=.8, maximum_step=.8, minimum_step=.792,
                load_scale=.001, initial_state=state, tolerance=2e-7,
                max_iterations=15)


def test_shell4_backtracking_closes_failure_preserves_path_energy_and_transaction():
    model, fixed, load, virgin = _cantilever()
    before = tuple(point.plastic_strain.clone() for element in virgin.points
                   for gauss in element for point in gauss)
    fixed_result = solve_finite_rotation_arc_path(
        model, load, fixed, line_search=None, **_options(virgin),
    )
    trace = []
    line = solve_finite_rotation_arc_path(
        model, load, fixed, line_search="backtracking", diagnostics=trace,
        **_options(virgin),
    )
    reference = solve_finite_rotation_arc_path(
        model, load, fixed, line_search="backtracking",
        **{**_options(virgin), "tolerance": 2e-9},
    )

    assert not fixed_result.converged and line.converged and reference.converged
    assert min(item.get("line_search_alpha", 1.)
               for item in trace[-1]["iterations"]) < 1.
    assert abs(line.load_factor/reference.load_factor-1.) < .01
    path_error = (float(torch.linalg.vector_norm(line.displacement-reference.displacement))
                  / float(torch.linalg.vector_norm(reference.displacement)))
    assert path_error < .01
    line_energy = shell_stored_energy(
        model, line.displacement, line.committed_state).recoverable
    reference_energy = shell_stored_energy(
        model, reference.displacement, reference.committed_state).recoverable
    assert abs(line_energy/reference_energy-1.) < .01

    # Neither a rejected fixed step nor line-search trial evaluations can
    # mutate the caller-owned checkpoint.
    after = tuple(point.plastic_strain for element in virgin.points
                  for gauss in element for point in gauss)
    assert all(torch.equal(left, right) for left, right in zip(before, after))
    rejected = tuple(point.plastic_strain
                     for element in fixed_result.committed_state.points
                     for gauss in element for point in gauss)
    assert all(torch.equal(left, right) for left, right in zip(before, rejected))


def test_shell4_line_search_default_restart_and_panel_identity(tmp_path):
    model, fixed, load, virgin = _cantilever()
    options = dict(step_size=.05, maximum_step=.05, load_scale=.001,
                   initial_state=virgin, tolerance=2e-7)
    default = solve_finite_rotation_arc_path(model, load, fixed, steps=2, **options)
    explicit = solve_finite_rotation_arc_path(
        model, load, fixed, steps=2, line_search=None, **options)
    first = solve_finite_rotation_arc_path(
        model, load, fixed, steps=1, line_search="backtracking", **options)
    resumed = solve_finite_rotation_arc_path(
        model, load, fixed, steps=1, step_size=.05, maximum_step=.05,
        load_scale=.001, initial_state=first.committed_state,
        initial_displacement=first.displacement,
        initial_load_factor=first.load_factor,
        initial_previous_increment=first.previous_increment,
        tolerance=2e-7, line_search="backtracking",
    )
    whole = solve_finite_rotation_arc_path(
        model, load, fixed, steps=2, line_search="backtracking", **options)
    assert torch.equal(default.displacement, explicit.displacement)
    assert resumed.load_factor == whole.load_factor
    assert torch.allclose(resumed.displacement, whole.displacement,
                          rtol=1e-11, atol=1e-12)

    fixed_job = execute_panel_chunked_job(
        2, .02, steps=1, cache_dir=tmp_path/"fixed", resume=False,
        maximum_wall_seconds=0.,
    )
    line_job = execute_panel_chunked_job(
        2, .02, steps=1, cache_dir=tmp_path/"line", resume=False,
        maximum_wall_seconds=0., line_search="backtracking",
    )
    assert fixed_job["job_key"] != line_job["job_key"]
    assert "line_search" not in fixed_job
    assert line_job["line_search"] == "backtracking"
