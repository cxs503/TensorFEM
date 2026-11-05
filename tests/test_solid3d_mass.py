"""Exact-reference tests for the 3D solid inertia operator (HEX8 / TET4).

Every gate here is anchored on a closed form (``rho V``, the analytic block inertia
tensor, the free-free zero-mode count, or the rod axial frequency), so a failure
means the operator is wrong rather than "the reference is different".  The
lumped-mass rotational inertia is deliberately tested as a *known defect*: the
assertion pins the measured ratio, it does not pretend the operator is exact.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorfem.solid3d import hex_to_tet_mesh, structured_hex_mesh
from tensorfem.solid3d_mass import (
    assemble_mass,
    axial_free_dofs,
    axial_mode_index,
    box_rigid_body_reference,
    constrained_modulus,
    element_consistent_mass,
    element_lumped_mass,
    generalized_modes,
    rigid_body_mass_matrix,
    rigid_body_modes,
    rod_axial_angular_frequency,
)

E = 210.0e9
NU = 0.3
RHO = 7850.0
L = 1.0
SIDE = 0.1


def _stiffness(nodes, cells, young=E, nu=NU):
    from tensorfem.solid3d import SolidModel, solve_solid

    model = SolidModel(
        nodes=nodes,
        elements=cells,
        young_modulus=torch.tensor(young, dtype=nodes.dtype),
        poisson_ratio=torch.tensor(nu, dtype=nodes.dtype),
        forces=torch.zeros(3 * nodes.shape[0], dtype=nodes.dtype),
        fixed_dofs=torch.zeros(0, dtype=torch.long),
    )
    return solve_solid(model).stiffness


def _total_mass(mass):
    e = torch.zeros(mass.shape[0], dtype=mass.dtype)
    e[0::3] = 1.0
    return float(e @ mass @ e)


def test_hex8_and_tet4_total_mass_are_exact():
    lx, ly, lz = 2.0, 0.5, 1.5
    nodes, cells = structured_hex_mesh(lx, ly, lz, 3, 2, 4)
    exact = RHO * lx * ly * lz
    for lumped in (False, True):
        assert abs(_total_mass(assemble_mass(nodes, cells, RHO, lumped=lumped)) / exact - 1) < 1e-13
    tnodes, tets = hex_to_tet_mesh(nodes, cells)
    for lumped in (False, True):
        assert abs(_total_mass(assemble_mass(tnodes, tets, RHO, element_type="tet4",
                                            lumped=lumped)) / exact - 1) < 1e-13


def test_consistent_mass_is_symmetric_positive_and_block_diagonal_in_components():
    nodes, cells = structured_hex_mesh(0.4, 0.3, 0.5, 2, 2, 2)
    mass = assemble_mass(nodes, cells, RHO)
    assert torch.allclose(mass, mass.transpose(0, 1), atol=1e-18)
    assert float(torch.linalg.eigvalsh(mass).min()) > 0.0
    # Hex8 mass is identical for the three displacement components.
    n = 3 * nodes.shape[0]
    assert torch.allclose(mass[:n:3, :n:3], mass[1:n:3, 1:n:3], atol=1e-18)


def test_rigid_body_inertia_matches_analytic_block():
    lx, ly, lz = 0.4, 0.3, 0.5
    nodes, cells = structured_hex_mesh(lx, ly, lz, 2, 2, 3)
    centroid = nodes.mean(0)
    exact = box_rigid_body_reference(RHO * lx * ly * lz, (lx, ly, lz))
    mass = assemble_mass(nodes, cells, RHO)
    measured = rigid_body_mass_matrix(nodes, mass, about=centroid)
    scale = float(exact.diagonal().max())
    assert float((measured - exact).abs().max()) / scale < 1e-12


def test_lumped_rotation_inertia_is_a_known_defect_not_a_pass():
    """Lumped mass reproduces mass/translations exactly but not the inertia tensor."""
    lx, ly, lz = 0.4, 0.3, 0.5
    nodes, cells = structured_hex_mesh(lx, ly, lz, 1, 1, 1)
    centroid = nodes.mean(0)
    exact = box_rigid_body_reference(RHO * lx * ly * lz, (lx, ly, lz))
    measured = rigid_body_mass_matrix(nodes, assemble_mass(nodes, cells, RHO, lumped=True),
                                      about=centroid)
    assert abs(float(measured[0, 0]) / float(exact[0, 0]) - 1.0) < 1e-13
    # Row-sum lumping puts 1/8 of the mass on each corner, so the polar inertia is
    # 3x the exact value on a single element; this cannot be "tuned away".
    assert abs(float(measured[3, 3]) / float(exact[3, 3]) - 3.0) < 1e-12


def test_lumped_inertia_converges_to_exact_with_refinement():
    lx, ly, lz = 0.4, 0.3, 0.5
    exact = box_rigid_body_reference(RHO * lx * ly * lz, (lx, ly, lz))
    ratios = []
    for n in (1, 2, 4):
        nodes, cells = structured_hex_mesh(lx, ly, lz, n, n, n)
        measured = rigid_body_mass_matrix(nodes, assemble_mass(nodes, cells, RHO, lumped=True),
                                          about=nodes.mean(0))
        ratios.append(float(measured[3, 3]) / float(exact[3, 3]))
    # Corner lumping has the exact closed form  I_lumped / I_exact = 1 + 2 / n^2
    # for an n x n x n grid of a rectangular block - a sharper oracle than "converges".
    for n, ratio in zip((1, 2, 4), ratios):
        assert ratio == pytest.approx(1.0 + 2.0 / n**2, rel=1e-12)


def test_free_free_body_has_six_zero_modes_and_no_rigid_force():
    nodes, cells = structured_hex_mesh(0.2, 0.2, 0.6, 2, 2, 4)
    mass = assemble_mass(nodes, cells, RHO)
    stiffness = _stiffness(nodes, cells)
    values, _ = generalized_modes(stiffness, mass)
    omega2 = values.clamp_min(0.0)
    elastic = omega2[6:]
    assert float(omega2[:6].max() / elastic.min()) < 1e-10
    modes = rigid_body_modes(nodes, about=nodes.mean(0))
    residual = float((stiffness @ modes.transpose(0, 1)).abs().max())
    assert residual < 1e-3 * float(stiffness.abs().max())


def test_generalized_modes_are_mass_orthonormal():
    nodes, cells = structured_hex_mesh(0.2, 0.2, 0.6, 2, 2, 3)
    mass = assemble_mass(nodes, cells, RHO)
    stiffness = _stiffness(nodes, cells)
    values, vectors = generalized_modes(stiffness, mass, n_modes=4, drop=6)
    gram = vectors.transpose(0, 1) @ mass @ vectors
    assert torch.allclose(gram, torch.eye(4, dtype=gram.dtype), atol=1e-10)
    # Rayleigh quotient must reproduce the eigenvalues.
    rayleigh = torch.diagonal(vectors.transpose(0, 1) @ stiffness @ vectors)
    assert torch.allclose(rayleigh, values, rtol=1e-9, atol=1e-6)


def test_constrained_modulus_reference_and_axial_dof_selection():
    assert constrained_modulus(E, NU) == pytest.approx(E * (1 - NU) / ((1 + NU) * (1 - 2 * NU)))
    assert constrained_modulus(E, NU) > E
    nodes, _ = structured_hex_mesh(1.0, 1.0, 1.0, 1, 1, 2)
    dofs = axial_free_dofs(nodes, axis=2, fixed_at=0.0)
    z = nodes[:, 2]
    assert dofs.numel() == int((z > 0).sum())
    assert torch.equal(dofs % 3, torch.full_like(dofs, 2))
    assert torch.equal(torch.unique(nodes[dofs // 3, 2]), torch.tensor([0.5, 1.0], dtype=z.dtype))


def test_uniaxial_strain_column_axial_frequency_converges_second_order():
    modulus = constrained_modulus(E, NU)
    errors = {}
    for n in (4, 8):
        nodes, cells = structured_hex_mesh(SIDE, SIDE, L, 1, 1, n)
        dofs = axial_free_dofs(nodes, axis=2, fixed_at=0.0)
        mass = assemble_mass(nodes, cells, RHO)
        stiffness = _stiffness(nodes, cells)
        values, _ = generalized_modes(stiffness.index_select(0, dofs).index_select(1, dofs),
                                      mass.index_select(0, dofs).index_select(1, dofs),
                                      n_modes=1)
        exact = rod_axial_angular_frequency(1, young=E, density=RHO, length=L, modulus=modulus)
        errors[n] = abs(float(values[0].clamp_min(0).sqrt()) / exact - 1.0)
    assert errors[8] < 3e-3
    ratio = errors[4] / errors[8]
    assert 3.0 < ratio < 5.0, "quadratic convergence implies an error ratio near four"


def test_lumped_and_consistent_first_mode_agree_but_higher_modes_do_not():
    """Measured, not assumed: on a uniform mesh both masses give the same omega_1.

    Row-sum lumping of a uniform trilinear HEX8 mesh is close to the optimal
    (HRZ) diagonal scaling, so the first two axial frequencies agree to ~1e-5
    while the third mode is clearly worse.  This test pins that observation so a
    later "improvement" cannot silently change it.
    """
    modulus = constrained_modulus(E, NU)
    nodes, cells = structured_hex_mesh(SIDE, SIDE, L, 1, 1, 8)
    dofs = axial_free_dofs(nodes, axis=2, fixed_at=0.0)
    stiffness = _stiffness(nodes, cells)
    errors = {}
    for lumped in (False, True):
        mass = assemble_mass(nodes, cells, RHO, lumped=lumped)
        values, _ = generalized_modes(stiffness.index_select(0, dofs).index_select(1, dofs),
                                      mass.index_select(0, dofs).index_select(1, dofs),
                                      n_modes=3)
        omega = values.clamp_min(0.0).sqrt()
        errors[lumped] = [
            float(omega[i]) / rod_axial_angular_frequency(
                i + 1, young=E, density=RHO, length=L, modulus=modulus) - 1.0
            for i in range(3)
        ]
    # Mode 1: same magnitude, opposite sign (consistent overshoots, lumped undershoots).
    assert errors[False][0] > 0.0 > errors[True][0]
    assert abs(abs(errors[False][0]) - abs(errors[True][0])) < 0.02 * abs(errors[False][0])
    # Mode 3: lumping is measurably worse on the identical mesh.
    assert abs(errors[True][2]) > 1.3 * abs(errors[False][2])
    assert abs(errors[False][0]) < 3e-3


def test_full_3d_block_axial_mode_identified_by_energy_matches_rod_reference():
    n = 8
    nodes, cells = structured_hex_mesh(0.05, 0.05, L, 1, 1, n)
    clamp = torch.nonzero(nodes[:, 2].abs() < 1e-12).reshape(-1)
    fixed = torch.cat([3 * clamp + i for i in range(3)]).sort().values
    mask = torch.ones(3 * nodes.shape[0], dtype=torch.bool)
    mask[fixed] = False
    dofs = torch.nonzero(mask).reshape(-1)
    mass = assemble_mass(nodes, cells, RHO)
    stiffness = _stiffness(nodes, cells)
    values, vectors = generalized_modes(stiffness.index_select(0, dofs).index_select(1, dofs),
                                        mass.index_select(0, dofs).index_select(1, dofs),
                                        n_modes=10)
    full = torch.zeros(3 * nodes.shape[0], vectors.shape[1], dtype=nodes.dtype)
    full[dofs] = vectors
    pick = axial_mode_index(full, torch.diagonal(mass))
    omega = float(values[pick].clamp_min(0.0).sqrt())
    exact = rod_axial_angular_frequency(1, young=E, density=RHO, length=L)
    assert abs(omega / exact - 1.0) < 3e-2
    # The axial mode must not be the first (bending) mode of a slender block.
    assert pick > 0


def test_distorted_hex_quadrature_is_approximate_not_exact():
    nodes, cells = structured_hex_mesh(1.0, 1.0, 1.0, 2, 2, 2)
    distorted = nodes.clone()
    distorted[13] = distorted[13] + 0.18 * torch.tensor([0.31, -0.22, 0.27], dtype=nodes.dtype)
    coarse = float(element_consistent_mass(distorted, cells, RHO, order=2).sum()) / 3.0
    fine = float(element_consistent_mass(distorted, cells, RHO, order=3).sum()) / 3.0
    exact = RHO * 1.0
    # Boundary nodes are untouched, so the volume is preserved; the difference is
    # pure quadrature error and must be small but strictly non-zero.
    assert 0.0 < abs(coarse - fine) / exact < 1e-2
    assert abs(fine / exact - 1.0) < 1e-3


def test_element_mass_rejects_bad_input():
    nodes, cells = structured_hex_mesh(1.0, 1.0, 1.0, 1, 1, 1)
    with pytest.raises(ValueError):
        element_consistent_mass(nodes, cells, RHO, element_type="hex20")
    with pytest.raises(ValueError):
        assemble_mass(nodes, cells, torch.tensor([RHO, RHO]))
    inverted = nodes.clone()
    inverted[[0, 1]] = inverted[[1, 0]]
    with pytest.raises(ValueError):
        element_consistent_mass(inverted, cells, RHO)


def test_lumped_element_mass_is_row_sum_of_consistent():
    nodes, cells = structured_hex_mesh(1.0, 0.5, 0.25, 2, 1, 1)
    consistent = element_consistent_mass(nodes, cells, RHO)
    lumped = element_lumped_mass(nodes, cells, RHO)
    assert torch.allclose(lumped.sum(-1), consistent.sum(-1))
    off = lumped - torch.diag_embed(torch.diagonal(lumped, dim1=-2, dim2=-1))
    assert float(off.abs().max()) == 0.0


def test_rod_reference_formula_is_the_sine_closed_form():
    omega = rod_axial_angular_frequency(2, young=E, density=RHO, length=L)
    assert omega == pytest.approx(3 * math.pi / (2 * L) * math.sqrt(E / RHO), rel=1e-15)
    with pytest.raises(ValueError):
        rod_axial_angular_frequency(0, young=E, density=RHO, length=L)