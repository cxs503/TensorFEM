"""Gates for the near-incompressible locking qualification (:mod:`tensorfem.incompressible_locking`).

The oracle is checked first (equilibrium, tractions, constitutive law, and the
Poisson-invariance of the stress state).  Only then are the element results
scored, and the standard HEX8 failure is asserted rather than tolerated so that
nobody can quote it as a pass.
"""
import pytest
import torch

from tensorfem.incompressible_locking import (lame_sphere_field, locking_sweep,
                                              plane_strain_sweep, reference_self_check,
                                              solve_case)

FAST = dict(face_divisions=6, radial_layers=3)


def test_reference_field_is_equilibrium_and_traction_consistent():
    for poisson in (0.3, 0.4999):
        check = reference_self_check(poisson)
        assert check["equilibrium"] < 1e-8
        assert check["traction"] < 1e-12
        assert check["constitutive"] < 1e-9
        assert check["stress_state_poisson_invariant"] < 1e-12


def test_reference_hoop_and_radial_stress_bracket_the_pressure():
    """Closed-form sanity: sigma_r(a) = -p, sigma_r(b) = 0, hoop stress positive."""
    _, sigma_r, sigma_theta = lame_sphere_field(
        torch.tensor([[8.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=torch.float64), 0.3)
    assert abs(float(sigma_r[0]) + 1.0e6) < 1e-6
    assert abs(float(sigma_r[1])) < 1e-6
    assert bool((sigma_theta > 0).all())


def test_bbar_stress_error_is_poisson_invariant_while_standard_hex8_locks():
    reference = solve_case(0.3, formulation="bbar", **FAST)
    incompressible = solve_case(0.4999, formulation="bbar", **FAST)
    standard = solve_case(0.4999, formulation="standard", **FAST)

    flat = incompressible["errors"]["radial_stress_l2"]
    base = reference["errors"]["radial_stress_l2"]
    # B-bar: the discretisation error does not care about nu (locking-free).
    assert abs(flat - base) / base < 0.05
    assert incompressible["errors"]["displacement_l2"] < 0.05
    # The stress state itself is nu-independent, so any growth is a bias.
    assert standard["errors"]["radial_stress_l2"] > 10.0 * flat
    assert standard["errors"]["displacement_l2"] > 0.5
    assert standard["errors"]["displacement_l2"] > 20.0 * incompressible["errors"]["displacement_l2"]


def test_standard_hex8_near_incompressible_blowup_is_reported_not_tolerated():
    """Locking corrupts the whole field, the mean-stress error fastest.

    Both the volumetric (mean) and the deviatoric parts degrade under the standard
    displacement HEX8; the volumetric part degrades hardest, which is the locking
    signature.  The gated claim is only about magnitude, not about "deviatoric is
    unaffected", which the numbers here do not support.
    """
    soft = solve_case(0.3, formulation="standard", **FAST)
    stiff = solve_case(0.4999, formulation="standard", **FAST)
    bbar = solve_case(0.4999, formulation="bbar", **FAST)

    mean_growth = stiff["errors"]["mean_stress_l2"] / soft["errors"]["mean_stress_l2"]
    deviatoric_growth = (stiff["errors"]["deviatoric_stress_l2"]
                         / soft["errors"]["deviatoric_stress_l2"])
    assert mean_growth > 10.0
    assert mean_growth > 2.0 * deviatoric_growth
    assert stiff["errors"]["inner_radius_displacement_relative"] > 0.5
    assert bbar["errors"]["inner_radius_displacement_relative"] < 0.05
    assert bbar["errors"]["mean_stress_l2"] < 0.1 * stiff["errors"]["mean_stress_l2"]


def test_bbar_error_converges_under_mesh_refinement_at_nu_half():
    coarse = solve_case(0.4999, formulation="bbar", face_divisions=4, radial_layers=2)
    fine = solve_case(0.4999, formulation="bbar", **FAST)
    for key in ("displacement_l2", "radial_stress_l2"):
        assert fine["errors"][key] < coarse["errors"][key]


def test_sweep_records_are_complete_and_reproducible():
    records = locking_sweep((0.3, 0.4999), **FAST)
    assert len(records) == 4
    lookup = {(r["poisson_ratio"], r["formulation"]): r for r in records}
    required = {"displacement_l2", "radial_stress_l2", "mean_stress_l2",
                "deviatoric_stress_l2", "inner_radius_displacement_relative"}
    for record in records:
        assert required <= set(record["errors"])
        assert all(torch.isfinite(torch.tensor(value)) for value in record["errors"].values())
    repeat = solve_case(0.4999, formulation="bbar", **FAST)
    assert abs(repeat["errors"]["displacement_l2"]
               - lookup[(0.4999, "bbar")]["errors"]["displacement_l2"]) < 1e-10


@pytest.mark.parametrize("poisson", [0.4999])
def test_plane_strain_q4_locking_debt_is_quantified(poisson):
    """The 2-D Q4 path has no B-bar counterpart; record the debt instead of hiding it."""
    soft = plane_strain_sweep((0.3,))[0]
    stiff = plane_strain_sweep((poisson,))[0]
    assert stiff["errors"]["displacement_l2"] > soft["errors"]["displacement_l2"]
    assert stiff["errors"]["radial_stress_l2"] > soft["errors"]["radial_stress_l2"]
    # Fail-closed: the uncured 2-D path is worse than the 3 % neighbourhood.
    assert stiff["errors"]["radial_stress_l2"] > 0.03