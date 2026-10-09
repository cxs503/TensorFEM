import torch

from tensorfem.finite_rotation_layered_shell4 import (
    assemble_finite_rotation_layered_shell4,
    assemble_finite_rotation_layered_shell4_sparse,
)
from tensorfem.layered_shell4_plasticity import LayeredShell4Model, LayeredShell4State
from tensorfem.matrix_free_arc import solve_matrix_free_finite_rotation_shell4
from tensorfem.marine_panel_ultimate_fe import build_panel_case


D = torch.float64


def _two_element_shell():
    nodes = torch.tensor([
        [0., 0., 0.], [1., 0., 0.], [2., 0., 0.],
        [0., 1., 0.], [1., 1., 0.], [2., 1., 0.],
    ], dtype=D)
    elements = torch.tensor([[0, 1, 4, 3], [1, 2, 5, 4]])
    return LayeredShell4Model(nodes, elements, 2.e5, .3, .08, 250., 1400., layers=3)


def test_reduced_sparse_shell_assembly_is_identical_to_dense_force_and_tangent():
    model = _two_element_shell()
    state = LayeredShell4State.virgin(model)
    q = torch.zeros(model.n_dofs, dtype=D)
    q[8] = 2.e-4
    q[14] = -1.e-4
    active = torch.tensor([6, 7, 8, 12, 13, 14, 24, 25, 26, 30, 31, 32])
    dense = assemble_finite_rotation_layered_shell4(model, q, state)
    sparse = assemble_finite_rotation_layered_shell4_sparse(
        model, q, state, active_dofs=active,
    )
    assert torch.equal(sparse.active_dofs, active)
    assert torch.allclose(sparse.internal_force, dense.internal_force, rtol=0., atol=0.)
    expected = dense.tangent[active[:, None], active]
    assert torch.allclose(sparse.tangent.to_dense(), expected, rtol=1e-13, atol=1e-10)
    direction = torch.linspace(-1., 1., len(active), dtype=D)
    assert torch.allclose(sparse.tangent @ direction, expected @ direction,
                          rtol=2e-13, atol=1e-10)
    # Element assembly stores only the connectivity graph, not ndof**2.
    assert sparse.tangent._nnz() < model.n_dofs**2


def test_sparse_frozen_shell_path_converges_without_tangent_actions():
    nodes = torch.tensor([[0., 0., 0.], [1., 0., 0.],
                          [1., 1., 0.], [0., 1., 0.]], dtype=D)
    model = LayeredShell4Model(nodes, torch.tensor([[0, 1, 2, 3]]),
                               2.e5, .3, .08, 250., 1400., layers=3)
    fixed = torch.tensor([0,1,2,3,4,5, 8,9,10,11, 14,15,16,17,
                          18,19,20,21,22,23])
    load = torch.zeros(model.n_dofs, dtype=D)
    load[6] = load[12] = 2.
    sparse = solve_matrix_free_finite_rotation_shell4(
        model, load, fixed, linearization="frozen_tangent",
        preconditioner="jacobi", steps=1, step_size=.01, maximum_step=.01,
        load_scale=.1, tolerance=3e-7, krylov_rtol=2e-7,
        krylov_maxiter=30,
    )
    assert sparse.converged
    assert sparse.points[0].equilibrium_relative_norm <= 3e-7
    assert sparse.tangent_actions == 0


def test_four_by_four_panel_sparse_storage_and_action_gate():
    case = build_panel_case(4)
    state = LayeredShell4State.virgin(case.model)
    q = torch.zeros(case.model.n_dofs, dtype=D)
    mask = torch.ones(case.model.n_dofs, dtype=torch.bool)
    mask[case.fixed_dofs] = False
    free = torch.nonzero(mask).flatten()
    dense = assemble_finite_rotation_layered_shell4(case.model, q, state)
    sparse = assemble_finite_rotation_layered_shell4_sparse(
        case.model, q, state, active_dofs=free,
    )
    reduced_dense = dense.tangent[free[:, None], free]
    direction = torch.linspace(-1., 1., len(free), dtype=D)
    exact = reduced_dense @ direction
    relative = torch.linalg.vector_norm(sparse.tangent @ direction-exact) / torch.linalg.vector_norm(exact)
    sparse_bytes = (sparse.tangent.values().numel()*sparse.tangent.values().element_size()
                    + sparse.tangent.indices().numel()*sparse.tangent.indices().element_size())
    full_dense_bytes = dense.tangent.numel()*dense.tangent.element_size()
    assert float(relative) < 1e-12
    assert sparse_bytes < .7*full_dense_bytes
