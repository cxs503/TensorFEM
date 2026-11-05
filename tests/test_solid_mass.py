import math

import pytest
import torch

from tensorfem.solid3d import hex_to_tet_mesh, structured_hex_mesh
from tensorfem.solid_mass import (
    assemble_solid_mass,
    confined_modulus,
    hex8_consistent_mass,
    mass_properties,
    rod_axial_angular_frequency,
    tet4_consistent_mass,
)
from tensorfem.solid3d import SolidModel, solve_solid
from tensorfem.solid_mass_audit import (
    audit_axial_rod,
    axial_rod_case,
    mass_invariant_audit,
)

D = torch.float64


def _brick(a=2.0, b=1.0, c=0.8):
    x = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [a, 0.0, 0.0],
            [a, b, 0.0],
            [0.0, b, 0.0],
            [0.0, 0.0, c],
            [a, 0.0, c],
            [a, b, c],
            [0.0, b, c],
        ],
        dtype=D,
    )
    return x, a * b * c


def test_hex8_brick_matches_separable_closed_form():
    """For a brick the consistent mass has the closed form
    ``m_ij = rho V / 8 * prod_d (1 + xi_i xi_j / 3) / 2``."""
    x, volume = _brick()
    rho = 7850.0
    me = hex8_consistent_mass(x.reshape(1, 8, 3), rho)[0][0::3, 0::3]
    signs = torch.tensor(
        [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
         [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
        dtype=D,
    )
    closed = torch.zeros((8, 8), dtype=D)
    for i in range(8):
        for j in range(8):
            factor = torch.tensor(1.0, dtype=D)
            for d in range(3):
                factor = factor * (1.0 + signs[i, d] * signs[j, d] / 3.0) / 2.0
            closed[i, j] = rho * volume / 8.0 * factor
    assert torch.allclose(me, closed, rtol=1e-13, atol=1e-11)
    # diagonal equals rho V / 27 for an 8-node brick
    assert math.isclose(float(me[0, 0]), rho * volume / 27.0, rel_tol=1e-13)
    # row sums reproduce the nodal mass share rho V / 8 (partition of unity)
    assert torch.allclose(me.sum(1), torch.full((8,), rho * volume / 8.0, dtype=D), rtol=1e-13)


def test_tet4_closed_form_and_quadrature_agree():
    """Linear TET: ``m_ii = rho V / 10`` and ``m_ij = rho V / 20``."""
    x = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.8]], dtype=D)
    rho = 1000.0
    volume = 2.0 * 1.0 * 0.8 / 6.0
    me = tet4_consistent_mass(x.reshape(1, 4, 3), rho)[0]
    assert math.isclose(float(me[0, 0]), rho * volume / 10.0, rel_tol=1e-14)
    assert math.isclose(float(me[0, 3]), rho * volume / 20.0, rel_tol=1e-14)
    assert math.isclose(float(me[0, 0] / me[0, 3]), 2.0, rel_tol=1e-14)
    # block structure: x/y/z components are decoupled
    assert float(me[0, 1]) == 0.0 and float(me[0, 4]) == 0.0


@pytest.mark.parametrize("element_type", ["hex8", "tet4"])
def test_assembly_symmetry_positivity_and_dof_ordering(element_type):
    nodes, hexes = structured_hex_mesh(2.0, 1.0, 0.8, 3, 2, 2)
    elements = hexes if element_type == "hex8" else hex_to_tet_mesh(nodes, hexes)[1]
    mass = assemble_solid_mass(nodes, elements, 7850.0, element_type=element_type)
    assert torch.allclose(mass, mass.T, atol=1e-8)
    assert float(torch.linalg.eigvalsh(mass).min()) > 0.0
    # DOF ordering must match solid3d: a rigid translation has no elastic force
    model = SolidModel(
        nodes,
        elements.long(),
        torch.tensor(210e9, dtype=D),
        torch.tensor(0.3, dtype=D),
        torch.zeros(3 * len(nodes), dtype=D),
        torch.tensor([0, 1, 2]),
        element_type,
    )
    stiffness = solve_solid(model).stiffness
    translation = torch.zeros((len(nodes), 3), dtype=D)
    translation[:, 1] = 1.0
    translation = translation.reshape(-1)
    assert float(torch.linalg.vector_norm(stiffness @ translation)) < 1e-3
    energy = float(translation @ mass @ translation)
    exact = mass_properties(nodes, elements, 7850.0, element_type=element_type).exact_mass
    assert math.isclose(energy, exact, rel_tol=1e-12)


@pytest.mark.parametrize("element_type", ["hex8", "tet4"])
def test_mass_and_polar_inertia_invariants(element_type):
    """Consistent mass reproduces rho V and rho*int(x^2+y^2) to round-off."""
    for distortion in (0.0, 0.05):
        cases = {
            (c.element_type, c.mass_form): c
            for c in mass_invariant_audit(distortion=distortion)
        }
        case = cases[(element_type, "consistent")]
        assert case.mass_error < 1e-12
        assert case.polar_inertia_error < 1e-12
        lumped = cases[(element_type, "lumped")]
        assert lumped.mass_error < 1e-12          # row-sum lumping conserves mass
        assert 0.05 < lumped.polar_inertia_error < 0.3  # but inflates rho int r^2 dV


@pytest.mark.parametrize("element_type", ["hex8", "tet4"])
def test_lumped_mass_is_diagonal_row_sum(element_type):
    nodes, hexes = structured_hex_mesh(1.0, 0.5, 0.4, 2, 2, 2)
    elements = hexes if element_type == "hex8" else hex_to_tet_mesh(nodes, hexes)[1]
    consistent = assemble_solid_mass(nodes, elements, 2700.0, element_type=element_type)
    lumped = assemble_solid_mass(
        nodes, elements, 2700.0, element_type=element_type, lumped=True
    )
    assert torch.allclose(lumped.sum(1), consistent.sum(1), rtol=1e-12)
    assert torch.allclose(lumped, torch.diag_embed(torch.diagonal(lumped)))
    assert bool((torch.diagonal(lumped) > 0).all())


def test_axial_rod_reference_and_confined_modulus():
    e, nu, rho, length = 210e9, 0.3, 7850.0, 10.0
    assert math.isclose(confined_modulus(e, 0.0), e, rel_tol=1e-15)
    assert math.isclose(confined_modulus(e, nu), e * (1 - nu) / ((1 + nu) * (1 - 2 * nu)), rel_tol=1e-14)
    w1 = rod_axial_angular_frequency(1, length, e, rho)
    assert math.isclose(w1, math.pi / (2 * length) * math.sqrt(e / rho), rel_tol=1e-15)
    assert math.isclose(
        rod_axial_angular_frequency(1, length, e, rho, fixed_free=False),
        2 * w1,
        rel_tol=1e-14,
    )


@pytest.mark.parametrize("element_type", ["hex8", "tet4"])
def test_axial_rod_mesh_convergence_below_three_percent(element_type):
    """3-D solid meshes reproduce the exact confined rod frequencies."""
    audit = audit_axial_rod(element_type=element_type, mass_form="consistent", meshes=(4, 8, 16))
    errors = [case.relative_errors for case in audit.cases]
    # second-order convergence in every mode
    for mode, order in enumerate(audit.observed_order):
        assert 1.6 < order < 2.4, (element_type, mode, order)
        assert all(
            errors[i + 1][mode] < errors[i][mode] for i in range(len(errors) - 1)
        ), (element_type, mode, errors)
    # the finest mesh clears the 3% acceptance gate on all three modes
    assert audit.finest_max_error < 0.03, (element_type, audit.finest_max_error)


def test_lumped_rod_frequencies_stay_within_three_percent():
    audit = audit_axial_rod(element_type="hex8", mass_form="lumped", meshes=(8, 16))
    assert audit.finest_max_error < 0.03
    # Row-sum lumping reproduces the axial rod about as well as consistent mass.
    consistent = audit_axial_rod(element_type="hex8", mass_form="consistent", meshes=(8, 16))
    assert abs(audit.finest_max_error - consistent.finest_max_error) < 5e-3


def test_convergence_rate_is_h_squared():
    coarse = axial_rod_case(8, modes=1)
    fine = axial_rod_case(16, modes=1)
    ratio = coarse.relative_errors[0] / fine.relative_errors[0]
    assert 3.6 < ratio < 4.4, ratio


def test_invalid_inputs_are_rejected():
    nodes, hexes = structured_hex_mesh(1.0, 1.0, 1.0, 1, 1, 1)
    with pytest.raises(ValueError):
        assemble_solid_mass(nodes, hexes, -1.0)
    with pytest.raises(ValueError):
        assemble_solid_mass(nodes, hexes, 1000.0, element_type="hex20")
    with pytest.raises(ValueError):
        assemble_solid_mass(nodes, hex_to_tet_mesh(nodes, hexes)[1], 1000.0, element_type="hex8")
    with pytest.raises(ValueError):
        assemble_solid_mass(nodes, torch.tensor([[0, 1, 2, 99, 4, 5, 6, 7]]), 1000.0)
    with pytest.raises(ValueError):
        hex8_consistent_mass(nodes[hexes][:, :4], 1000.0)
    with pytest.raises(ValueError):
        axial_rod_case(2, mass_form="hrz")
    with pytest.raises(ValueError):
        axial_rod_case(1, modes=5)
    with pytest.raises(ValueError):
        rod_axial_angular_frequency(0, 1.0, 1.0, 1.0)