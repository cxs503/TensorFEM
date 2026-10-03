import torch

from tensorfem.general_shell_nonlinear import GeneralShellMesh, assemble_general_shell
from tensorfem.shell4_state_path import (recover_elastic_section_state,
    shell_with_initial_imperfection, solve_shell4_state_path)

D = torch.float64


def _two_facet_problem():
    # A genuinely assembled, doubly curved two-facet patch.
    phi = torch.tensor([.45, .75], dtype=D)
    theta = torch.tensor([0., .35, .7], dtype=D)
    gp, gt = torch.meshgrid(phi, theta, indexing="ij")
    nodes = 5 * torch.stack((torch.sin(gp) * torch.cos(gt),
                             torch.sin(gp) * torch.sin(gt), torch.cos(gp)), -1).reshape(-1, 3)
    elements = torch.tensor([[0, 3, 4, 1], [1, 4, 5, 2]])
    base = GeneralShellMesh(nodes, elements, 2e7, .25, .03)
    imperfection = torch.zeros_like(nodes)
    imperfection[4] = 2e-4 * nodes[4] / torch.linalg.vector_norm(nodes[4])
    mesh = shell_with_initial_imperfection(base, imperfection)
    load = torch.zeros(36, dtype=D)
    load[24:27] = -.05 * mesh.nodes[4] / torch.linalg.vector_norm(mesh.nodes[4])
    fixed = list(range(18)) + [6 * 3 + 5, 6 * 4 + 5, 6 * 5 + 5]
    return base, mesh, imperfection, load, fixed


def test_imperfection_is_stress_free_reference_and_section_energy_is_exact():
    base, mesh, imperfection, _, _ = _two_facet_problem()
    assert torch.allclose(mesh.nodes, base.nodes + imperfection)
    q = torch.zeros(36, dtype=D)
    state = recover_elastic_section_state(mesh, q)
    # SVD frame recovery leaves only round-off, not physical prestress.
    assert torch.linalg.vector_norm(state.generalized_deformation) < 2e-15
    assert torch.linalg.vector_norm(state.generalized_resultant) < 2e-9
    assert state.energy.sum() < 1e-23
    q[24] = 1e-5
    state = recover_elastic_section_state(mesh, q)
    assembled_energy, _, _ = assemble_general_shell(mesh, q, tangent=False)
    assert torch.allclose(state.energy.sum(), assembled_energy, rtol=2e-12, atol=1e-18)


def test_multifacet_arc_path_balance_and_committed_state():
    _, mesh, _, load, fixed = _two_facet_problem()
    out = solve_shell4_state_path(mesh, load, fixed, steps=2, step_size=.02,
                                  load_scale=.1, tolerance=2e-7)
    assert out.continuation.converged and len(out.continuation.points) == 2
    _, internal, _ = assemble_general_shell(mesh, out.committed.dofs, tangent=False)
    residual = internal - out.committed.load_factor * load
    assert torch.linalg.vector_norm(residual[out.free_dofs]) < 3e-7
    assert torch.all(out.committed.section.energy >= 0)


def test_multifacet_assembled_tangent_matches_directional_difference():
    _, mesh, _, _, _ = _two_facet_problem()
    q = torch.linspace(-2e-5, 2e-5, 36, dtype=D)
    direction = torch.cos(torch.arange(36, dtype=D))
    direction /= torch.linalg.vector_norm(direction)
    _, _, tangent = assemble_general_shell(mesh, q, tangent=True)
    h = 2e-7
    _, fp, _ = assemble_general_shell(mesh, q + h * direction, tangent=False)
    _, fm, _ = assemble_general_shell(mesh, q - h * direction, tangent=False)
    exact = tangent @ direction
    relative = torch.linalg.vector_norm((fp - fm) / (2 * h) - exact) / torch.linalg.vector_norm(exact)
    assert relative < 2e-5


def test_failed_step_rolls_back_global_and_section_state():
    _, mesh, _, load, fixed = _two_facet_problem()
    out = solve_shell4_state_path(mesh, load, fixed, steps=1, step_size=.2,
                                  load_scale=.1, tolerance=1e-14,
                                  max_iterations=1, minimum_step=.11)
    assert not out.continuation.converged
    assert out.committed.load_factor == 0.
    assert torch.count_nonzero(out.committed.dofs) == 0
    assert out.committed.section.energy.sum() < 1e-23


def test_imperfection_validation_fails_closed():
    base, _, _, _, _ = _two_facet_problem()
    bad = torch.zeros((len(base.nodes), 2), dtype=D)
    try:
        shell_with_initial_imperfection(base, bad)
    except ValueError as error:
        assert "same shape" in str(error)
    else:
        raise AssertionError("invalid imperfection was accepted")
