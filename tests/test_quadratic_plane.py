"""Verification of the production Q8 (serendipity) plane element.

Every criterion below is machine-checkable and independent of any stored artefact:

* shape-function derivatives against central differences,
* constant-strain patch test on a distorted Q8 element (exact to round-off),
* rigid-body/null-space dimension of one element (3 modes) plus a reduced-integration
  negative control that must expose spurious mechanisms,
* Cook membrane against the published scalar reference 23.96 and against the
  independently coded Q8 Richardson reference,
* plane-strain Lame annulus against the closed-form stress field,
* global equilibrium (reaction resultant equals the applied load).
"""

import functools
import math

import pytest
import torch

from tensorfem import cook_stress_reference as reference
from tensorfem import quadratic_plane as qp
from tensorfem.continuum import elasticity_matrix

DTYPE = torch.float64
PLANE_STRESS = dict(young=1.0, poisson=1.0 / 3.0, plane="stress")


def distorted_element() -> torch.Tensor:
    return torch.tensor(
        [[-1.0, -1.0], [1.2, -0.9], [1.1, 1.15], [-0.9, 1.05],
         [-0.05, -0.97], [1.15, 0.1], [0.05, 1.1], [-0.95, 0.02]],
        dtype=DTYPE,
    )


def plane_stress_matrix() -> torch.Tensor:
    return elasticity_matrix(
        torch.tensor([1.0], dtype=DTYPE), torch.tensor([1.0 / 3.0], dtype=DTYPE), "stress"
    )


def cook_problem(divisions: int):
    """Cook membrane as a Q8 problem (same geometry and load as the Q4 case)."""
    corners = torch.tensor(((0.0, 0.0), (48.0, 44.0), (48.0, 60.0), (0.0, 44.0)), dtype=DTYPE)
    nodes4, cells4 = qp.mapped_q4_grid(corners, divisions, divisions)
    nodes, cells = qp.q8_mesh(nodes4, cells4)
    forces = torch.zeros(2 * len(nodes), dtype=DTYPE)
    edge = torch.tensor(48.0, dtype=DTYPE)
    on_edge = torch.nonzero(torch.isclose(nodes[:, 0], edge)).reshape(-1)
    per_element = 1.0 / divisions
    # Consistent quadratic edge load per element: w/6, 4w/6, w/6 on bottom/mid/top.
    for cell in cells.tolist():
        edge_nodes = [
            node for node in cell if abs(float(nodes[node, 0]) - 48.0) < 1e-12
        ]
        if len(edge_nodes) < 3:
            continue
        bottom, mid, top = sorted(edge_nodes, key=lambda node: float(nodes[node, 1]))
        forces[2 * bottom + 1] += per_element / 6.0
        forces[2 * mid + 1] += 4.0 * per_element / 6.0
        forces[2 * top + 1] += per_element / 6.0
    left = torch.nonzero(torch.isclose(nodes[:, 0], torch.tensor(0.0, dtype=DTYPE))).reshape(-1)
    dirichlet = {}
    for node in left.tolist():
        dirichlet[2 * node] = 0.0
        dirichlet[2 * node + 1] = 0.0
    return nodes, cells, forces, dirichlet, on_edge


@functools.lru_cache(maxsize=1)
def reference_solutions():
    return reference.build_reference((12, 24, 48))["solutions"]


def reference_von_mises(xi: float, eta: float) -> float:
    solutions = reference_solutions()
    values = [reference.field_stress(s, xi, eta)["von_mises"] for s in solutions]
    fit = reference.richardson(values, [s["divisions"] for s in solutions])
    return float(fit["extrapolated"])


def probe_local(xi: float, eta: float, divisions: int):
    """Element index and local parent coordinates of a global parent probe."""
    column = min(max(int(math.floor(xi * divisions)), 0), divisions - 1)
    row = min(max(int(math.floor(eta * divisions)), 0), divisions - 1)
    return (
        row * divisions + column,
        (2.0 * (xi * divisions - column) - 1.0, 2.0 * (eta * divisions - row) - 1.0),
    )


# --------------------------------------------------------------------------- #
# algebra
# --------------------------------------------------------------------------- #
def test_shape_derivatives_match_central_differences():
    step = 1e-7
    for xi, eta in ((0.37, -0.21), (-0.8, 0.6), (0.0, 0.0), (0.55, 0.9)):
        finite = torch.stack(
            (
                (qp.q8_shape(xi + step, eta) - qp.q8_shape(xi - step, eta)) / (2 * step),
                (qp.q8_shape(xi, eta + step) - qp.q8_shape(xi, eta - step)) / (2 * step),
            ),
            1,
        )
        assert float((finite - qp.q8_shape_derivative(xi, eta)).abs().max()) < 1e-8


def test_shape_functions_partition_unity_and_nodal_delta():
    for xi, eta in ((0.3, 0.4), (-0.2, 0.75), (0.0, 0.0)):
        assert abs(float(qp.q8_shape(xi, eta).sum()) - 1.0) < 1e-14
        assert float(qp.q8_shape_derivative(xi, eta).sum(0).abs().max()) < 1e-13
    for index, (xi, eta) in enumerate(qp.NODE_ORDER):
        values = qp.q8_shape(xi, eta)
        assert abs(float(values[index]) - 1.0) < 1e-14
        assert float(values.abs().sum()) - 1.0 < 1e-12


def test_constant_strain_patch_is_exact_on_distorted_element():
    coordinates = distorted_element()
    gradient = torch.tensor([[0.013, 0.007], [-0.004, 0.011]], dtype=DTYPE)
    translation = torch.tensor([0.3, -0.2], dtype=DTYPE)
    displacement = torch.zeros(16, dtype=DTYPE)
    for node in range(8):
        displacement[2 * node] = translation[0] + gradient[0] @ coordinates[node]
        displacement[2 * node + 1] = translation[1] + gradient[1] @ coordinates[node]
    exact = torch.tensor(
        [gradient[0, 0], gradient[1, 1], gradient[0, 1] + gradient[1, 0]], dtype=DTYPE
    )
    for xi, eta in ((0.0, 0.0), (0.4, -0.7), (-0.9, 0.8), (0.9, 0.9)):
        matrix, det = qp.q8_strain_matrix(xi, eta, coordinates.unsqueeze(0))
        assert float(det[0]) > 0.0
        assert float((matrix[0] @ displacement - exact).abs().max()) < 1e-14


def test_one_element_has_exactly_three_rigid_body_modes():
    stiffness, _ = qp.q8_stiffness(
        distorted_element().unsqueeze(0), plane_stress_matrix(), torch.tensor([1.0], dtype=DTYPE)
    )
    eigenvalues = torch.linalg.eigvalsh(stiffness[0])
    modes = int((eigenvalues <= float(eigenvalues.max()) * 1e-10).sum())
    assert modes == qp.RIGID_BODY_MODES, eigenvalues.tolist()


def test_reduced_two_by_two_rule_exposes_spurious_modes():
    """Negative control: under-integration must show more than three null modes."""
    coordinates = distorted_element()
    dmat = plane_stress_matrix()
    rule = ((-1 / math.sqrt(3), -1 / math.sqrt(3)), (1 / math.sqrt(3), -1 / math.sqrt(3)),
            (1 / math.sqrt(3), 1 / math.sqrt(3)), (-1 / math.sqrt(3), 1 / math.sqrt(3)))
    stiffness = torch.zeros((16, 16), dtype=DTYPE)
    for xi, eta in rule:
        matrix, det = qp.q8_strain_matrix(xi, eta, coordinates.unsqueeze(0))
        stiffness = stiffness + matrix[0].T @ dmat @ matrix[0] * float(det[0])
    eigenvalues = torch.linalg.eigvalsh(stiffness)
    modes = int((eigenvalues <= float(eigenvalues.max()) * 1e-10).sum())
    assert modes > qp.RIGID_BODY_MODES, eigenvalues.tolist()


# --------------------------------------------------------------------------- #
# Cook membrane
# --------------------------------------------------------------------------- #
def test_cook_load_resultant_is_one():
    for divisions in (2, 4, 8):
        _, _, forces, _, _ = cook_problem(divisions)
        assert abs(float(forces.sum()) - 1.0) < 1e-13


def test_cook_displacement_matches_published_reference():
    published = 23.96
    errors = []
    for divisions in (4, 8, 16):
        nodes, cells, forces, dirichlet, on_edge = cook_problem(divisions)
        result = qp.solve_q8_plane(nodes, cells, forces=forces, dirichlet=dirichlet, **PLANE_STRESS)
        probe = [int(n) for n in on_edge.tolist() if abs(float(nodes[n, 1]) - 52.0) < 1e-12][0]
        errors.append(abs(float(result.displacement[2 * probe + 1]) - published) / published)
        assert result.relative_residual < 1e-10
    assert errors[0] < 0.03
    assert errors[-1] < 0.005
    assert errors == sorted(errors, reverse=True)


def test_cook_equilibrium_reaction_equals_applied_load():
    """The support reaction must carry the whole applied resultant with opposite sign."""
    nodes, cells, forces, dirichlet, _ = cook_problem(8)
    result = qp.solve_q8_plane(nodes, cells, forces=forces, dirichlet=dirichlet, **PLANE_STRESS)
    supported = sorted(dirichlet)
    carried = float(result.reaction[torch.tensor(supported)].sum())
    assert abs(carried + float(forces.sum())) < 1e-10
    free = torch.ones(len(result.reaction), dtype=torch.bool)
    free[torch.tensor(supported)] = False
    assert float(result.reaction[free].abs().max()) < 1e-9


@pytest.mark.parametrize("divisions,gate", [(4, 0.06), (8, 0.02), (16, 0.01)])
def test_cook_stress_agrees_with_independent_q8_reference(divisions, gate):
    """Pointwise von Mises at interior parent points against the independent reference."""
    nodes, cells, forces, dirichlet, _ = cook_problem(divisions)
    result = qp.solve_q8_plane(nodes, cells, forces=forces, dirichlet=dirichlet, **PLANE_STRESS)
    scale = 0.0
    errors = []
    for nominal in reference.SAMPLE_POINTS:
        xi, eta = nominal[0] + 0.0121, nominal[1] + 0.0073
        element, local = probe_local(xi, eta, divisions)
        target = reference_von_mises(xi, eta)
        scale = max(scale, abs(target))
        stress = qp.q8_stress_at(nodes, cells, result.displacement, element, *local, **PLANE_STRESS)
        errors.append(abs(float(qp.von_mises(stress)) - target) / max(abs(target), 0.05 * scale))
    assert max(errors) < gate, errors


# --------------------------------------------------------------------------- #
# Lame annulus (plane strain, curved geometry, non-polynomial exact field)
# --------------------------------------------------------------------------- #
def lame_constants(inner=1.0, outer=2.0, pressure=1.0):
    denominator = outer ** 2 - inner ** 2
    return pressure * inner ** 2 / denominator, pressure * inner ** 2 * outer ** 2 / denominator


def lame_stress(radius, coefficient_a, coefficient_b):
    return (
        coefficient_a - coefficient_b / (radius * radius),
        coefficient_a + coefficient_b / (radius * radius),
    )


def lame_displacement(radius, coefficient_a, coefficient_b, young, poisson):
    return (1.0 + poisson) / young * (
        coefficient_a * (1.0 - 2.0 * poisson) * radius + coefficient_b / radius
    )


def lame_problem(n_theta: int, n_radial: int, inner=1.0, outer=2.0, young=1.0, poisson=0.3):
    nodes, cells = qp.polar_q8_mesh(inner, outer, n_theta, n_radial)
    coefficient_a, coefficient_b = lame_constants(inner, outer, 1.0)
    radius = torch.linalg.norm(nodes, dim=1)
    dirichlet = {}
    for node in range(len(nodes)):
        value = float(radius[node])
        if min(abs(value - inner), abs(value - outer)) < 1e-12:
            displacement = lame_displacement(value, coefficient_a, coefficient_b, young, poisson)
            angle = math.atan2(float(nodes[node, 1]), float(nodes[node, 0]))
            dirichlet[2 * node] = displacement * math.cos(angle)
            dirichlet[2 * node + 1] = displacement * math.sin(angle)
    result = qp.solve_q8_plane(
        nodes, cells, young=young, poisson=poisson, plane="strain", dirichlet=dirichlet
    )
    return nodes, cells, result, (coefficient_a, coefficient_b)


def lame_worst_error(nodes, cells, result, constants) -> float:
    coefficient_a, coefficient_b = constants
    worst, scale = 0.0, 1.0
    for element in range(len(cells)):
        connectivity = cells[element]
        for slot, (xi, eta) in enumerate(qp.GAUSS_POINTS_3X3):
            stress = result.gauss_stress[element, slot]
            physical = (qp.q8_shape(xi, eta).to(DTYPE) @ nodes[connectivity]).tolist()
            r = math.hypot(physical[0], physical[1])
            radial_exact, hoop_exact = lame_stress(r, coefficient_a, coefficient_b)
            angle = math.atan2(physical[1], physical[0])
            c, s = math.cos(angle), math.sin(angle)
            sxx, syy, sxy = float(stress[0]), float(stress[1]), float(stress[2])
            radial = c * c * sxx + s * s * syy + 2 * c * s * sxy
            hoop = s * s * sxx + c * c * syy - 2 * c * s * sxy
            scale = max(scale, abs(hoop_exact), abs(radial_exact))
            worst = max(worst, abs(radial - radial_exact) / scale, abs(hoop - hoop_exact) / scale)
    return worst


def test_lame_analytic_field_satisfies_boundary_pressures():
    coefficient_a, coefficient_b = lame_constants()
    assert abs(lame_stress(1.0, coefficient_a, coefficient_b)[0] + 1.0) < 1e-12
    assert abs(lame_stress(2.0, coefficient_a, coefficient_b)[0]) < 1e-12


@pytest.mark.parametrize("n_theta,n_radial,gate", [(16, 4, 0.02), (32, 8, 0.01)])
def test_lame_annulus_stress_matches_closed_form(n_theta, n_radial, gate):
    nodes, cells, result, constants = lame_problem(n_theta, n_radial)
    assert lame_worst_error(nodes, cells, result, constants) < gate