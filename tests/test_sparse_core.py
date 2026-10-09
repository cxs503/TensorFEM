import torch
import pytest

from tensorfem.sparse_core import (DofManager, MPC, assemble_coo,
                                   conjugate_gradient, solve_sparse_static)

D = torch.float64


def chain_stiffness(n, k=1000.0):
    ke = torch.tensor([[1., -1.], [-1., 1.]], dtype=D).repeat(n - 1, 1, 1) * k
    ed = torch.stack((torch.arange(n - 1), torch.arange(1, n)), 1)
    return assemble_coo(ke, ed, n)


def test_dof_numbering_and_equations():
    d = DofManager(3, ("ux", "uy"))
    assert d.dof(2, "uy") == 5
    assert d.element_dofs(torch.tensor([[0, 2]])).tolist() == [[0, 1, 4, 5]]
    assert d.equations([0, 3]).tolist() == [-1, 0, 1, -1, 2, 3]


def test_sparse_assembly_matches_dense_and_preserves_gradient():
    values = torch.tensor([2., 3.], dtype=D, requires_grad=True)
    base = torch.tensor([[1., -1.], [-1., 1.]], dtype=D)
    K = assemble_coo(values[:, None, None] * base, torch.tensor([[0, 1], [1, 2]]), 3)
    expected = torch.tensor([[2., -2., 0.], [-2., 5., -3.], [0., -3., 3.]], dtype=D)
    assert torch.allclose(K.to_dense(), expected)
    K.values().sum().backward()
    assert values.grad is not None


def test_cg_matches_dense_spd():
    A = torch.tensor([[4., -1., 0.], [-1., 4., -1.], [0., -1., 3.]], dtype=D)
    b = torch.tensor([1., 4., 2.], dtype=D)
    r = conjugate_gradient(A.to_sparse(), b, rtol=1e-13)
    assert r.converged
    assert torch.allclose(r.x, torch.linalg.solve(A, b), rtol=1e-11, atol=1e-12)


def test_analytical_axial_bar_below_three_percent_and_reactions():
    n, length, area, young, load = 41, 2., .01, 200e9, 1e5
    K = chain_stiffness(n, young * area / (length / (n - 1)))
    f = torch.zeros(n, dtype=D); f[-1] = load
    result = solve_sparse_static(K, f, dirichlet={0: 0.}, rtol=1e-12)
    exact = load * length / (young * area)
    assert abs(float(result.displacement[-1]) - exact) / exact < .03
    assert abs(float(result.reaction[0]) + load) / load < 1e-10


def test_nonzero_dirichlet_and_mpc_elimination():
    # Three springs; prescribe u0, and tie u2=0.5*u1. Compare KKT-equivalent minimum.
    K = chain_stiffness(3, 10.)
    f = torch.tensor([0., 2., 0.], dtype=D)
    result = solve_sparse_static(K, f, dirichlet={0: .1},
                                 mpcs=(MPC(2, ((1, .5),), .02),), rtol=1e-13)
    # Reduced energy gives (T'KT)q=T'(f-Ku0).
    T = torch.tensor([[0.], [1.], [.5]], dtype=D); u0 = torch.tensor([.1, 0., .02], dtype=D)
    q = torch.linalg.solve(T.T @ K.to_dense() @ T, T.T @ (f - K.to_dense() @ u0))
    assert torch.allclose(result.displacement, T @ q + u0, atol=1e-12)


def test_constraints_fail_closed_on_ambiguous_or_chained_mpc():
    K = chain_stiffness(3); f = torch.zeros(3, dtype=D)
    with pytest.raises(ValueError, match="both prescribed"):
        solve_sparse_static(K, f, dirichlet={2: 0.}, mpcs=(MPC(2, ((1, 1.),)),))
    with pytest.raises(ValueError, match="independent"):
        solve_sparse_static(K, f, dirichlet={0: 0.},
                            mpcs=(MPC(1, ((0, 1.),)),))
