import torch

from tensorfem.assembly import assemble_truss_stiffness
from tensorfem.benchmarks import run_single_bar, single_bar, three_bar_truss
from tensorfem.solvers import solve_linear_static


def test_single_bar_matches_analytical_solution():
    result = run_single_bar()
    assert result.passed
    assert result.relative_error < 1e-12


def test_stiffness_is_symmetric_and_has_rigid_modes():
    matrix, _, _ = assemble_truss_stiffness(three_bar_truss())
    assert torch.allclose(matrix, matrix.T, atol=1e-10)
    eigenvalues = torch.linalg.eigvalsh(matrix)
    assert torch.sum(torch.abs(eigenvalues) < 1e-6) >= 3


def test_reactions_balance_applied_load():
    model = three_bar_truss()
    result = solve_linear_static(model)
    total_x = result.reaction[0::2].sum() + model.forces[0::2].sum()
    total_y = result.reaction[1::2].sum() + model.forces[1::2].sum()
    assert torch.allclose(total_x, torch.tensor(0., dtype=model.dtype), atol=1e-8)
    assert torch.allclose(total_y, torch.tensor(0., dtype=model.dtype), atol=1e-8)


def test_compliance_is_differentiable_by_area():
    model, _ = single_bar()
    area = torch.tensor(3e-4, dtype=torch.float64, requires_grad=True)
    model = type(model)(model.nodes, model.elements, model.young_modulus, area,
                        model.forces, model.fixed_dofs)
    result = solve_linear_static(model)
    compliance = model.forces @ result.displacement
    compliance.backward()
    assert area.grad is not None
    assert area.grad < 0
