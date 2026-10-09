import torch
from tensorfem.solid3d import elasticity_matrix_3d, structured_hex_mesh
from tensorfem.solid3d_bbar import hex8_bbar_stiffness, bbar_free_element_eigenvalues
from tensorfem.solid3d_bbar_benchmarks import (near_incompressible_cantilever,
                                                near_incompressible_uniaxial_patch)

D = torch.float64


def test_bbar_affine_patch_energy_at_near_incompressibility():
    nodes, hexes = structured_hex_mesh(1., .8, .6, 1, 1, 1, dtype=D)
    x = nodes[hexes]
    constitutive = elasticity_matrix_3d(torch.tensor([2e6], dtype=D),
                                        torch.tensor([.4999], dtype=D))
    stiffness, _ = hex8_bbar_stiffness(x, constitutive)
    strain = torch.tensor([2e-4, -1e-4, -.999e-4, 3e-5, -2e-5, 4e-5], dtype=D)
    u = torch.zeros((8, 3), dtype=D)
    u[:, 0] = strain[0]*x[0, :, 0] + .5*strain[3]*x[0, :, 1] + .5*strain[5]*x[0, :, 2]
    u[:, 1] = strain[1]*x[0, :, 1] + .5*strain[3]*x[0, :, 0] + .5*strain[4]*x[0, :, 2]
    u[:, 2] = strain[2]*x[0, :, 2] + .5*strain[4]*x[0, :, 1] + .5*strain[5]*x[0, :, 0]
    computed = .5*u.reshape(-1) @ stiffness[0] @ u.reshape(-1)
    reference = .5*strain @ constitutive[0] @ strain * (.8*.6)
    assert abs(float(computed-reference))/float(reference) < 1e-10


def test_bbar_has_six_and_only_six_zero_energy_modes():
    nodes, hexes = structured_hex_mesh(1., .8, .6, 1, 1, 1, dtype=D)
    eigenvalues = bbar_free_element_eigenvalues(nodes[hexes[0]])
    scale = eigenvalues[-1]
    assert float(torch.max(torch.abs(eigenvalues[:6]))/scale) < 1e-10
    assert float(eigenvalues[6]/scale) > 1e-7


def test_bbar_cures_near_incompressible_locking_under_three_percent():
    standard, bbar, reference = near_incompressible_cantilever()
    standard_error = abs(standard-reference)/reference
    bbar_error = abs(bbar-reference)/reference
    assert standard_error > .50
    assert bbar_error < .03
    assert bbar_error < standard_error/20


def test_near_incompressible_uniaxial_traction_is_exact():
    _, _, error = near_incompressible_uniaxial_patch()
    assert error < 1e-8
