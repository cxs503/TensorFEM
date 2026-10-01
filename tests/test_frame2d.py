import torch

from tensorfem.frame2d import FrameModel, solve_frame_static
from tensorfem.frame_benchmarks import cantilever_tip_load, cantilever_uniform_load


TOL = 0.03


def relative_error(actual, expected):
    return torch.abs((actual - expected) / expected)


def test_cantilever_tip_load_benchmark_below_three_percent():
    result, expected = cantilever_tip_load()
    actual = result.displacement[torch.tensor([4, 5])]
    assert torch.all(relative_error(actual, expected) < TOL)


def test_cantilever_uniform_load_benchmark_below_three_percent():
    result, expected = cantilever_uniform_load()
    actual = result.displacement[torch.tensor([4, 5])]
    assert torch.all(relative_error(actual, expected) < TOL)
    # Global equilibrium: vertical base reaction equals -qL.
    assert torch.isclose(result.reaction[1], torch.tensor(10_000., dtype=torch.float64))


def test_rotated_member_axial_response_and_reactions():
    E, A, L, P = 70e9, 3e-3, 2.0, 5_000.0
    c = 2**-0.5
    nodes = torch.tensor([[0., 0.], [L*c, L*c]], dtype=torch.float64)
    force = torch.zeros(6, dtype=torch.float64)
    force[3:5] = torch.tensor([P*c, P*c], dtype=torch.float64)
    model = FrameModel(nodes, torch.tensor([[0, 1]]), torch.tensor(E), torch.tensor(A),
                       torch.tensor(1e-5), force, torch.tensor([0, 1, 2, 5]))
    result = solve_frame_static(model)
    extension = result.displacement[3]*c + result.displacement[4]*c
    assert relative_error(extension, torch.tensor(P*L/(E*A), dtype=torch.float64)) < TOL
    assert torch.allclose(result.reaction[:2], -force[3:5], rtol=1e-10, atol=1e-8)


def test_frame_solution_is_differentiable_in_inertia():
    inertia = torch.tensor(8e-6, dtype=torch.float64, requires_grad=True)
    model = FrameModel(torch.tensor([[0., 0.], [3., 0.]], dtype=torch.float64), torch.tensor([[0, 1]]),
                       torch.tensor(210e9), torch.tensor(.02), inertia,
                       torch.tensor([0., 0., 0., 0., -1e4, 0.], dtype=torch.float64), torch.tensor([0, 1, 2]))
    compliance = -solve_frame_static(model).displacement[4]
    compliance.backward()
    assert inertia.grad is not None and torch.isfinite(inertia.grad) and inertia.grad < 0
