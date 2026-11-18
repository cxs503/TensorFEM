"""Analytical and assembly checks for the closed Shell4 sphere."""
import torch

from tensorfem.sphere_membrane_benchmark import (
    PROBLEM, element_membrane_resultants, facet_geometry, membrane_reference,
    mesh_defect, pressure_load, solve_sphere_membrane, sphere_shell_mesh)

D = torch.float64


def test_exact_membrane_reference_values():
    reference = membrane_reference(PROBLEM)
    assert abs(reference['resultant_n_per_m'] - 5.0) < 1e-12
    assert abs(reference['stress_pa'] - 125.0) < 1e-12
    assert abs(reference['radial_displacement_m'] - 1.2820512820512822e-05) < 1e-18


def test_flat_facet_membrane_recovery_is_exact():
    """Manufactured uniform in-plane strain on one flat quad."""
    nodes = torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]],
                         dtype=D)
    elements = torch.tensor([[0, 1, 2, 3]])
    ex, ey = 1e-5, -.5e-5
    displacement = torch.zeros(24, dtype=D)
    for a, (x, y) in enumerate(((0., 0.), (1., 0.), (1., 1.), (0., 1.))):
        displacement[6 * a + 0] = ex * x
        displacement[6 * a + 1] = ey * y
    resultants = element_membrane_resultants(nodes, elements, displacement, PROBLEM)
    E, nu, t = 6.825e7, .3, .04
    C11 = E / (1. - nu ** 2)
    assert abs(float(resultants['nx']) - C11 * (ex + nu * ey) * t) < 1e-12
    assert abs(float(resultants['ny']) - C11 * (ey + nu * ex) * t) < 1e-12
    assert abs(float(resultants['nxy'])) < 1e-18


def test_sphere_mesh_is_a_closed_quadrangulated_surface():
    nodes, elements = sphere_shell_mesh(4, PROBLEM)
    assert mesh_defect(elements) == {'boundary_or_nonmanifold_edges': 0,
                                     'orientation_defects': 0, 'euler_characteristic': 2}
    radius = torch.linalg.vector_norm(nodes, dim=1)
    assert float(radius.min()) > 10. - 1e-12 and float(radius.max()) < 10. + 1e-12
    # A flat-facet polyhedron always encloses slightly less area than the sphere.
    area = float(facet_geometry(nodes, elements)['area'].sum())
    assert area < 4. * torch.pi * 100.


def test_pressure_load_is_self_equilibrated_and_matches_facet_area():
    nodes, elements = sphere_shell_mesh(4, PROBLEM)
    geometry = facet_geometry(nodes, elements)
    force = pressure_load(nodes, elements, 1., geometry)
    vector = force.reshape(-1, 6)[:, :3]
    assert float(torch.linalg.vector_norm(vector.sum(0))) < 1e-10
    # Each facet contributes p*A split equally over its four corners.
    assert abs(float(vector.sum(0).abs().max())) < 1e-10
    total = float(force.reshape(-1, 6)[:, :3].norm(dim=1).sum())
    assert total >= .9 * float(geometry['area'].sum())


def test_pressure_scatter_matches_independent_sum_over_faces():
    nodes, faces = sphere_shell_mesh(4)
    direct = torch.zeros_like(nodes)
    for face in faces:
        local = pressure_load(nodes[face], torch.arange(4).reshape(1,4), 1.)
        direct[face] += local.view(4, 6)[:, :3]
    torch.testing.assert_close(pressure_load(nodes, faces, 1.).view(-1,6)[:, :3], direct)


def test_expansion_fit_removes_arbitrary_rigid_motion():
    from tensorfem.sphere_membrane_benchmark import uniform_expansion_fit
    nodes, _ = sphere_shell_mesh(4)
    q = torch.zeros(len(nodes),6,dtype=D)
    omega = nodes.new_tensor([-.1,.2,.3]).expand_as(nodes)
    q[:, :3] = .0123*nodes/10 + nodes.new_tensor([1.,3.,-2.]) + torch.linalg.cross(omega, nodes)
    fit = uniform_expansion_fit(nodes, q.flatten())
    assert abs(fit["radial_displacement"]-.0123) < 1e-12
    assert fit["relative_residual"] < 1e-11


def test_coarse_mesh_resolves_average_before_local_stress():
    coarse = solve_sphere_membrane(2).radius_ratios
    finer = solve_sphere_membrane(6).radius_ratios
    assert coarse["resultant_error_vs_exact"] > .03 > finer["resultant_error_vs_exact"]
    assert finer["principal_min_max_rel_error"] > .03


def test_refined_shell_passes_pointwise_membrane_gates():
    result = solve_sphere_membrane(16)
    ratios = result.radius_ratios
    for key in ("principal_min", "principal_max", "hoop", "meridional"):
        assert ratios[key+"_max_rel_error"] < .03
    assert ratios["radial_displacement_error"] < .03
    assert ratios["uniform_expansion_relative_residual"] < .03
    assert ratios["reaction_norm_n"]/ratios["applied_force_total_n"] < 1e-6
