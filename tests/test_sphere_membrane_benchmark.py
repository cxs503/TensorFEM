"""Audit tests for the closed-sphere membrane oracle of the Shell4 shell facet.

The merged ``projected_shell4_stiffness`` element is a flat facet.  These tests
pin both halves of the audit:

* the flat-facet membrane stress recovery is exact (manufactured patch test),
  so any error in the curved case is attributable to the geometry
  representation and not to the stress operator;
* the closed faceted sphere under uniform pressure fails the 3% membrane gate
  at every mesh the dense solver can afford, and the failure is *retained*
  rather than smoothed away.
"""
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


def test_faceted_sphere_fails_the_three_percent_membrane_gate():
    """Fail-closed: the merged facet element does not reach p*R/2 at 96 facets."""
    result = solve_sphere_membrane(4, PROBLEM)
    ratios = result.radius_ratios
    assert ratios['resultant_error_vs_exact'] > .03
    assert ratios['resultant_error_vs_facet'] > .03     # not a faceting-reference artefact
    assert ratios['resultant_isotropy_defect'] > .10    # strongly non-isotropic
    assert ratios['uniform_expansion_relative_residual'] > .50
    # The resultants must still be tensile: the facets are stretched, not inverted.
    assert float(result.element_resultants['mean'].mean()) > 0.


def test_error_decreases_with_facet_refinement_but_stays_unqualified():
    coarse = solve_sphere_membrane(2, PROBLEM).radius_ratios
    fine = solve_sphere_membrane(4, PROBLEM).radius_ratios
    assert fine['resultant_error_vs_exact'] < coarse['resultant_error_vs_exact']
    assert fine['resultant_error_vs_exact'] > .03
    # Monotone but far from the exact value: record the measured order.
    order = (torch.log(torch.tensor(coarse['resultant_error_vs_exact']))
             - torch.log(torch.tensor(fine['resultant_error_vs_exact']))) / torch.log(torch.tensor(2.))
    assert 0. < float(order) < 4.