"""Fail-closed gates for the 2-D plane-strain near-incompressible locking closure.

Order of checks: the analytic oracle first (equilibrium / traction / plane-strain
constitutive law / Poisson invariance of the stress state), then the element
prerequisites (patch consistency, exactly three rigid-body null modes), and only
then the locked-vs-corrected comparison.  The standard Q4 failure is **asserted**
at ``nu = 0.4999`` so that no later change can silently re-declare the locking
debt as a pass.
"""
import pytest
import torch

# dense Q4 assembly + rank SVDs: keep the OpenMP fan-out small on many-core hosts
torch.set_num_threads(8)

from tensorfem.continuum_bbar import (averaged_dilatation_row, q4_bbar_stiffness,
                                     solve_continuum_bbar, volumetric_constraint_rank)
from tensorfem.plane_strain_locking import (patch_test, plane_locking_sweep,
                                           plane_refinement_study, reference_self_check,
                                           rigid_body_check, solve_case)

FAST = dict(n_theta=8, n_radial=4)


def test_reference_oracle_is_equilibrium_traction_and_constitutive_consistent():
    for poisson in (0.3, 0.45, 0.4999):
        check = reference_self_check(poisson)
        assert check["equilibrium"] < 1e-7
        assert check["traction"] < 1e-12
        assert check["constitutive"] < 1e-4
        assert check["axial_stress_consistency"] < 1e-12
        assert check["stress_poisson_invariant"] < 1e-12


def test_reference_stress_state_brackets_the_bore_pressure():
    """Closed-form sanity: sigma_r(a) = -p, sigma_r(b) = 0, hoop stress positive."""
    from tensorfem.lame_reference import lame_state
    radius = torch.tensor([0.5, 0.8], dtype=torch.float64)
    state = lame_state(radius, inner_radius=0.5, outer_radius=0.8, inner_pressure=100.0e6,
                       young_modulus=210.0e9, poisson_ratio=0.4999, plane="strain")
    assert abs(float(state.radial_stress[0]) + 100.0e6) < 1e-3
    assert abs(float(state.radial_stress[1])) < 1e-3
    assert bool((state.hoop_stress > 0).all())


def test_element_is_patch_consistent_and_has_no_hourglass_mode():
    patch = patch_test()
    assert patch["force_agreement_relative"] < 1e-12   # B-bar == standard for constant strain
    assert patch["energy_relative"] < 1e-12            # exact constant-strain energy
    rigid = rigid_body_check()
    assert rigid["translation_relative"] < 1e-12
    assert rigid["rotation_relative"] < 1e-12
    assert rigid["zero_modes"] == 3                     # 2 translations + 1 rotation, no hourglass


def test_averaged_dilatation_row_measures_dilatation_exactly():
    """The averaged row must annihilate translations and return 2c for u = c*x.

    That is the defining property of the dilatation operator; it holds for any
    element shape because it is a property of the strain-displacement relation (a
    distorted quad is used on purpose, so a wrong Jacobian averaging cannot pass).
    """
    from tensorfem.continuum_bbar import averaged_dilatation_row
    coordinates = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.2, 1.0], [0.05, 0.9]]], dtype=torch.float64)
    averaged = averaged_dilatation_row(coordinates)
    translation = coordinates.new_tensor([0.7, -1.3]).expand(4, 2).reshape(-1)
    assert abs(float(averaged[0] @ translation)) < 1e-12
    c = 3.0e-3
    dilatation = (c * coordinates[0]).reshape(-1)   # u = c*x  =>  eps_xx = eps_yy = c
    assert abs(float(averaged[0] @ dilatation) - 2.0 * c) < 1e-12


def test_volumetric_constraint_count_is_one_per_element():
    from tensorfem.lame_reference import sector_q4_mesh
    nodes, cells = sector_q4_mesh(0.5, 0.8, 8, 4)
    bbar = volumetric_constraint_rank(nodes, cells, formulation="bbar")
    standard = volumetric_constraint_rank(nodes, cells, formulation="standard")
    assert bbar == len(cells)
    assert standard > 0.8 * (2 * len(nodes))   # 2x2 Gauss nearly saturates the displacement space


def test_bbar_cures_plane_strain_locking_and_standard_q4_fails_closed():
    bbar_compressible = solve_case(0.3, formulation="bbar", **FAST)
    bbar_incompressible = solve_case(0.4999, formulation="bbar", **FAST)
    standard_incompressible = solve_case(0.4999, formulation="standard", **FAST)
    # (a) the B-bar stress error is flat in nu: the stress field does not depend on nu,
    #     so a locking-free element must give the same error at both ratios
    ratio = (bbar_incompressible["errors"]["radial_stress_l2"]
             / bbar_compressible["errors"]["radial_stress_l2"])
    assert ratio < 1.05
    # (b) B-bar is accurate where the standard element is not
    assert bbar_incompressible["errors"]["radial_stress_l2"] < 0.05
    assert bbar_incompressible["errors"]["displacement_l2"] < 0.05
    # (c) the standard Q4 locks: > 50 % radial stress error at nu = 0.4999
    assert standard_incompressible["errors"]["radial_stress_l2"] > 0.5
    assert standard_incompressible["errors"]["displacement_l2"] > 0.2


def test_bbar_converges_while_the_standard_q4_stalls():
    meshes = ((8, 4), (16, 8), (32, 16))
    records = plane_refinement_study(0.4999, meshes=meshes)
    for formulation in ("bbar", "standard"):
        points = [r for r in records if r["formulation"] == formulation]
        errors = [r["errors"]["radial_stress_l2"] for r in points]
        assert errors[1] < errors[0]           # monotone decrease
        assert errors[2] < errors[1]
    bbar = [r["errors"]["radial_stress_l2"] for r in records if r["formulation"] == "bbar"]
    standard = [r["errors"]["radial_stress_l2"] for r in records if r["formulation"] == "standard"]
    assert standard[-1] > 10.0 * bbar[-1]      # locking leaves an order-one residual error


def test_sweep_table_is_complete_and_reaches_the_gate():
    sweep = plane_locking_sweep(poissons=(0.3, 0.4999), **FAST)
    assert len(sweep) == 4
    worst_bbar = max(r["errors"]["radial_stress_l2"] for r in sweep if r["formulation"] == "bbar")
    assert worst_bbar < 0.03                   # 3 % gate at the coarse 8x4 mesh, all nu


def test_solver_rejects_non_q4_and_gates_plane_stress_behind_opt_in():
    """Union of the two merged qualifications: plane stress is refused by default
    (plane-strain-only volumetric split) *and* reachable through an explicit opt-in."""
    from tensorfem.continuum import ContinuumModel
    from tensorfem.lame_reference import circular_pressure_loads, sector_q4_mesh, symmetry_fixed_dofs
    nodes, cells = sector_q4_mesh(0.5, 0.8, 8, 4)
    forces = circular_pressure_loads(8, 4, 0.5, 100.0e6, inner=True)
    fixed = symmetry_fixed_dofs(nodes, n_theta=8, n_radial=4)
    with pytest.raises(ValueError):
        solve_continuum_bbar(ContinuumModel(
            nodes, torch.tensor([[0, 1, 5], [1, 2, 6]], dtype=torch.long),
            torch.tensor(210.0e9), torch.tensor(0.3), torch.tensor(1.0), forces, fixed,
            element_type="cst", plane="strain"))
    # plane stress is not a locking case (finite bulk modulus), but the path must run
    model = ContinuumModel(nodes, cells, torch.tensor(210.0e9, dtype=torch.float64),
                           torch.tensor(0.3, dtype=torch.float64),
                           torch.tensor(1.0, dtype=torch.float64), forces, fixed, plane="stress")
    with pytest.raises(ValueError, match="plane strain"):
        solve_continuum_bbar(model)      # default is fail-closed, not silent
    result = solve_continuum_bbar(model, allow_plane_stress=True)
    assert torch.isfinite(result.displacement).all()
    assert result.displacement.abs().max() > 0.0