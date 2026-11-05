"""Gates for the B-bar plane-strain Q4 (:mod:`tensorfem.continuum_bbar`).

Order of checks follows the repository convention: element-level properties first
(no spurious zero-energy modes, exact constant-strain reproduction, symmetry),
then the exact Lamé oracle, then the fail-closed statement about the uncured
fully integrated Q4 that stays in the tree.
"""
import pytest
import torch

from tensorfem.continuum import ContinuumModel, elasticity_matrix
from tensorfem.continuum_bbar import (bbar_free_element_eigenvalues, q4_bbar_stiffness,
                                      solve_continuum_bbar)
from tensorfem.incompressible_locking import plane_strain_case, plane_strain_sweep

SQUARE = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=torch.float64)
DISTORTED = torch.tensor([[0.0, 0.0], [2.7, 0.4], [3.1, 2.3], [-0.3, 1.7]], dtype=torch.float64)
FAST = dict(n_theta=16, n_radial=8)


def _model(nodes, cells, poisson, forces, fixed, prescribed=None, element_type="q4"):
    return ContinuumModel(nodes, cells, torch.tensor(210.0e9), torch.tensor(float(poisson)),
                          torch.tensor(1.0), forces, fixed, plane="strain",
                          element_type=element_type, prescribed_values=prescribed)


def test_no_hourglass_mode_for_square_and_distorted_element():
    """Exactly three zero eigenvalues: two translations plus one rotation."""
    for coordinates in (SQUARE, DISTORTED):
        eigenvalues = bbar_free_element_eigenvalues(coordinates)
        zero = int((eigenvalues.abs() < 1e-9 * eigenvalues.abs().max()).sum())
        assert zero == 3, eigenvalues.tolist()
        assert float(eigenvalues.min()) > -1e-9 * float(eigenvalues.abs().max())


def test_stiffness_is_symmetric_positive_semidefinite():
    dmat = elasticity_matrix(torch.tensor([210.0e9]), torch.tensor([0.4999]), "strain")
    stiffness, bbar = q4_bbar_stiffness(DISTORTED.reshape(1, 4, 2), dmat, torch.tensor([1.0]))
    k = stiffness[0]
    assert torch.allclose(k, k.T, atol=1e-6 * k.abs().max())
    eigenvalues = torch.linalg.eigvalsh((k + k.T) / 2.0)
    assert float(eigenvalues.min()) > -1e-9 * float(eigenvalues.max())
    assert bbar.shape == (1, 3, 8)


def test_constant_strain_is_reproduced_exactly_on_a_distorted_element():
    """A mean-dilatation element must not perturb an affine displacement field."""
    poisson = 0.4999
    dmat = elasticity_matrix(torch.tensor([210.0e9], dtype=torch.float64),
                             torch.tensor([poisson], dtype=torch.float64), "strain")
    # Isochoric on purpose: at nu = 0.4999 a dilatational affine field carries a
    # bulk response of order 1e14 Pa, which would turn any 1e-16 strain
    # difference into a visible stress and make the patch test meaningless.
    strain = torch.tensor([1.3e-4, -1.3e-4, 0.7e-4], dtype=torch.float64)
    # u_x = a x + b y, u_y = c x + d y with the requested engineering shear.
    gradient = torch.tensor([[strain[0], 0.5 * strain[2] - 0.4e-4],
                             [0.5 * strain[2] + 0.4e-4, strain[1]]], dtype=torch.float64)
    prescribed = (DISTORTED @ gradient.T).reshape(-1)
    model = _model(DISTORTED, torch.tensor([[0, 1, 2, 3]]), poisson,
                   torch.zeros(8, dtype=torch.float64), torch.arange(8),
                   prescribed=prescribed)
    result = solve_continuum_bbar(model)
    expected = dmat[0] @ strain
    assert torch.allclose(result.stress[0], expected, rtol=1e-11, atol=1e-6 * float(expected.abs().max()))


def test_rigid_body_motion_produces_zero_stress():
    """Small-strain rigid motion: translation plus an infinitesimal rotation.

    A finite rotation matrix would inject the O(theta^2) diagonal term
    (R - I)xx = cos(theta) - 1, which is a genuine small-strain stretch and not a
    zero-strain motion, so the rotation is applied in skew-symmetric form.
    """
    poisson = 0.4999
    omega, translation = 0.37, torch.tensor([1.2, -3.4], dtype=torch.float64)
    spin = torch.stack((-omega * DISTORTED[:, 1], omega * DISTORTED[:, 0]), 1)
    prescribed = (spin + translation).reshape(-1)
    model = _model(DISTORTED, torch.tensor([[0, 1, 2, 3]]), poisson,
                   torch.zeros(8, dtype=torch.float64), torch.arange(8),
                   prescribed=prescribed)
    result = solve_continuum_bbar(model)
    # Stress of a spurious strain the size of the displacement over the element.
    scale = 210.0e9 * float(prescribed.abs().max()) / 2.0
    assert float(result.stress.abs().max()) < 1e-12 * scale


def test_plane_strain_thick_cylinder_matches_lame_below_one_percent():
    record = plane_strain_case(0.4999, formulation="bbar", **FAST)
    assert record["errors"]["displacement_l2"] < 0.01
    assert record["errors"]["radial_stress_l2"] < 0.01
    assert record["errors"]["hoop_stress_l2"] < 0.01
    assert record["errors"]["inner_radius_displacement_relative"] < 0.01


def test_bbar_error_is_flat_in_poisson_while_the_field_stays_accurate():
    records = {(r["poisson_ratio"]): r for r in
               plane_strain_sweep((0.3, 0.4999), formulations=("bbar",), **FAST)}
    soft, stiff = records[0.3], records[0.4999]
    growth = stiff["errors"]["displacement_l2"] / soft["errors"]["displacement_l2"]
    # Displacement carries a (1 - 2 nu) factor, so some growth is physical; the
    # fully integrated element grows by a factor of 400 on the same mesh.
    assert growth < 2.0
    assert stiff["errors"]["mean_stress_l2"] < 2.0 * soft["errors"]["mean_stress_l2"]
    assert stiff["errors"]["mean_stress_l2"] < 1e-3


def test_fully_integrated_q4_locking_is_fail_closed_and_not_reported_as_passing():
    soft = plane_strain_case(0.3, formulation="standard", **FAST)
    stiff = plane_strain_case(0.4999, formulation="standard", **FAST)
    cured = plane_strain_case(0.4999, formulation="bbar", **FAST)
    assert stiff["errors"]["displacement_l2"] > 0.5
    assert stiff["errors"]["radial_stress_l2"] > 1.0
    assert stiff["errors"]["mean_stress_l2"] > 100.0 * cured["errors"]["mean_stress_l2"]
    assert cured["errors"]["displacement_l2"] < stiff["errors"]["displacement_l2"] / 100.0
    # The uncured element's volumetric error also blows up with nu: 0.03 % -> 26 %.
    assert stiff["errors"]["mean_stress_l2"] > 100.0 * soft["errors"]["mean_stress_l2"]


def test_bbar_converges_with_order_close_to_two():
    coarse = plane_strain_case(0.4999, formulation="bbar", n_theta=16, n_radial=8)
    fine = plane_strain_case(0.4999, formulation="bbar", n_theta=32, n_radial=16)
    for key in ("displacement_l2", "radial_stress_l2"):
        order = torch.log(torch.tensor(coarse["errors"][key] / fine["errors"][key])) \
            / torch.log(torch.tensor(2.0))
        assert float(order) > 1.8, (key, float(order))


def test_reaction_forces_balance_the_applied_pressure():
    from tensorfem.lame_reference import (circular_pressure_loads, sector_q4_mesh,
                                          symmetry_fixed_dofs)
    n_theta, n_radial = FAST["n_theta"], FAST["n_radial"]
    nodes, cells = sector_q4_mesh(0.5, 0.8, n_theta, n_radial)
    forces = circular_pressure_loads(n_theta, n_radial, 0.5, 100.0e6, inner=True)
    fixed = symmetry_fixed_dofs(nodes, n_theta=n_theta, n_radial=n_radial)
    model = _model(nodes, cells, 0.4999, forces, fixed)
    result = solve_continuum_bbar(model)
    total = (result.reaction + forces).reshape(-1, 2).sum(0)
    scale = float(forces.abs().sum())
    assert float(total.abs().max()) < 1e-9 * scale


def test_plane_stress_and_cst_are_refused():
    nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    cells = torch.tensor([[0, 1, 2, 3]])
    plane_stress = ContinuumModel(nodes, cells, torch.tensor(210.0e9), torch.tensor(0.3),
                                  torch.tensor(1.0), torch.zeros(8, dtype=torch.float64),
                                  torch.arange(8, dtype=torch.long), plane="stress")
    with pytest.raises(ValueError, match="plane strain"):
        solve_continuum_bbar(plane_stress)
    triangle = _model(nodes[:3], torch.tensor([[0, 1, 2]]), 0.3,
                      torch.zeros(6, dtype=torch.float64), torch.arange(6, dtype=torch.long),
                      element_type="cst")
    with pytest.raises(ValueError, match="Q4"):
        solve_continuum_bbar(triangle)