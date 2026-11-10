"""Gates for the three-case J2 elastoplastic qualification surface.

Each test pins one of the acceptance criteria of the campaign node
``p337w1-n07-fem-elastoplastic`` (C1..C7).  The tests assert the *criterion*
(its band from the campaign table) on a configuration small enough for the fast
lane; the full production run lives in
``benchmarks/j2_elastoplastic_three_case/run.py`` and is registered in the slow
lane.  Two of the gates are deliberately tighter than the criterion, because the
implemented kernel is exact on those paths (material point / homogeneous bar) --
a loose assertion there would hide a regression.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorfem.j2_elastoplastic_three_case import (
    CylinderSpec,
    TorsionSpec,
    UniaxialSpec,
    bilinear_shear_stress,
    bilinear_uniaxial_stress,
    cylinder_elastic_band,
    mesh_convergence_gci,
    pi_plane_yield_surface,
    plastic_pressure_levels,
    pure_shear_material_point,
    pure_shear_patch_test,
    run_qualification,
    thick_wall_cylinder,
    uniaxial_tension_fem,
    uniaxial_tension_material_point,
)

torch.set_num_threads(2)

UNIAXIAL = UniaxialSpec()
TORSION = TorsionSpec()
CYLINDER = CylinderSpec()


# --------------------------------------------------------------------------- #
# material card identities (the oracles themselves)
# --------------------------------------------------------------------------- #
def test_material_card_identities_are_physics_not_hand_copied_numbers():
    assert UNIAXIAL.yield_strain == pytest.approx(250.0e6 / 210.0e9)
    assert UNIAXIAL.tangent_modulus == pytest.approx(
        210.0e9 * 1000.0e6 / (210.0e9 + 1000.0e6))
    # bilinear branches join continuously at yield
    eps_y = UNIAXIAL.yield_strain
    assert bilinear_uniaxial_stress(eps_y, 210.0e9, 250.0e6, 1000.0e6) == pytest.approx(250.0e6)
    assert bilinear_uniaxial_stress(0.5 * eps_y, 210.0e9, 250.0e6, 1000.0e6) == pytest.approx(
        125.0e6)
    # pure shear: tau_y = sigma_y / sqrt(3); plastic slope G H / (3G + H)
    assert TORSION.shear_yield_stress == pytest.approx(250.0e6 / math.sqrt(3.0))
    gamma_y = TORSION.shear_yield_strain
    assert bilinear_shear_stress(gamma_y, TORSION.shear_modulus, 250.0e6, 1000.0e6) \
        == pytest.approx(TORSION.shear_yield_stress)
    assert TORSION.plastic_shear_modulus == pytest.approx(
        1000.0e6 * TORSION.shear_modulus / (3.0 * TORSION.shear_modulus + 1000.0e6))
    # cylinder: first yield and plastic collapse of the plane-strain tube
    k = 250.0e6 / math.sqrt(3.0)
    assert CYLINDER.pressure_first_yield == pytest.approx(k * (1.0 - 0.25))
    assert CYLINDER.pressure_limit == pytest.approx(2.0 * k * math.log(2.0))
    # the mission's stated 0..100 MPa band is *below* first yield for this card
    assert CYLINDER.pressure_first_yield > 100.0e6


# --------------------------------------------------------------------------- #
# C1 / C2 - uniaxial tension
# --------------------------------------------------------------------------- #
def test_c1_uniaxial_material_point_matches_independent_oracle():
    strains = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
    report = uniaxial_tension_material_point(UNIAXIAL, strains)
    assert report["max_relative_error_oracle"] < 1e-10
    assert report["max_relative_error_closed_form"] < 1e-10
    # lateral strain must equal -nu*eps on the elastic branch and -eps/2 in the
    # plastic branch (plastic incompressibility, both lateral directions)
    elastic = report["levels"][0]
    assert elastic["lateral_strain"] == pytest.approx(-UNIAXIAL.poisson * elastic["strain"],
                                                      rel=1e-6)
    plastic = report["levels"][-1]
    # uniaxial stress + plastic incompressibility:
    #   eps_lat = -nu sigma/E - eps_pl/2,  eps_pl = eps - sigma/E
    elastic_axial = plastic["stress_kernel"] / UNIAXIAL.young
    eps_pl = plastic["strain"] - elastic_axial
    assert plastic["lateral_strain"] == pytest.approx(
        -UNIAXIAL.poisson * elastic_axial - 0.5 * eps_pl, rel=1e-6)
    assert plastic["alpha"] == pytest.approx(eps_pl, rel=1e-6)
    assert plastic["alpha"] > 0.04


def test_c1_c2_tet4_bar_reproduces_the_card_and_reads_back_the_yield_stress():
    strains = [0.002, 0.01, 0.03, 0.05]
    report = uniaxial_tension_fem(UNIAXIAL, strains, mesh=(4, 1, 1))
    # The exact solution is homogeneous, so a TET4 model must be exact to solver
    # tolerance: a loose bound here would let a constitutive bug through.
    assert report["max_relative_error_element"] < 1e-6
    assert report["max_relative_error_reaction"] < 1e-6
    assert report["yield_stress_estimate"] == pytest.approx(UNIAXIAL.yield_stress, rel=0.02)


# --------------------------------------------------------------------------- #
# C3 - pure shear
# --------------------------------------------------------------------------- #
def test_c3_pure_shear_curve_and_yield_point():
    shears = [0.0004, 0.0009, 0.0015, 0.0021, 0.003, 0.006, 0.012, 0.025, 0.05]
    report = pure_shear_material_point(TORSION, shears)
    assert report["max_relative_error"] < 1e-10
    assert report["tau_y_relative_error"] < 1e-8
    assert report["tau_y"] == pytest.approx(TORSION.shear_yield_stress, rel=1e-8)
    assert report["plastic_shear_modulus_numeric"] == pytest.approx(
        TORSION.plastic_shear_modulus, rel=1e-8)
    assert report["torque_at_yield"] == pytest.approx(
        TORSION.shear_yield_stress * 2.0 * math.pi * TORSION.radius ** 2 * TORSION.thickness)


def test_pure_shear_tet4_patch_test_matches_material_point():
    report = pure_shear_patch_test(TORSION, 0.003)
    assert report["n_interior_nodes"] >= 1
    assert report["relative_error"] < 1e-8
    assert report["stress_spread"] < 1e-6 * report["tau_reference"]


# --------------------------------------------------------------------------- #
# C6 - pi-plane circularity
# --------------------------------------------------------------------------- #
def test_c6_yield_surface_is_a_circle_in_the_pi_plane():
    report = pi_plane_yield_surface(UNIAXIAL, directions=6)
    assert len(report["directions"]) == 6
    assert report["max_relative_deviation"] < 1e-6  # criterion band is 1e-2
    for row in report["directions"]:
        assert row["equivalent_stress_at_onset"] == pytest.approx(UNIAXIAL.yield_stress,
                                                                 rel=1e-6)


# --------------------------------------------------------------------------- #
# case C - elastic band, plastic zone, GCI
# --------------------------------------------------------------------------- #
def test_cylinder_0_to_100_mpa_band_is_elastic_and_matches_lame():
    report = cylinder_elastic_band(CYLINDER, n_r=16, grading=0.1)
    assert report["max_plastic_strain"] == 0.0
    assert max(row["pressure"] for row in report["levels"]) <= 100.0e6
    assert report["max_u_bore_relative_error"] < 0.01


def test_c4_c5_plastic_zone_and_displacement_against_hill():
    report = thick_wall_cylinder(CYLINDER, [1.0 * CYLINDER.shear_yield_stress,
                                            1.35 * CYLINDER.shear_yield_stress],
                                 mesh_sizes=(16, 32), steps=6)
    assert report["max_r_p_relative_error"] < 0.05
    assert report["max_u_bore_relative_error"] < 0.10
    assert report["max_u_profile_relative_error"] < 0.10
    for row in report["levels"]:
        assert 0.0 < row["yielded_fraction"] < 1.0
        assert row["r_p_reference"] > CYLINDER.inner_radius
        # the interface is bracketed by the discretisation: |r_p - bracket| <= h
        h = (CYLINDER.outer_radius - CYLINDER.inner_radius) / row["n_r"]
        assert abs(row["r_p_fem"] - row["r_p_bracket"]) <= h


def test_c7_gci_is_reported_and_not_blocked():
    pressure = 1.2 * CYLINDER.shear_yield_stress
    rows = [row for row in thick_wall_cylinder(CYLINDER, [pressure],
                                               mesh_sizes=(8, 16, 32), steps=6)["levels"]]
    report = mesh_convergence_gci(rows, CYLINDER)
    assert report["status"] != "blocked"
    assert report["gci"]["relative_gci"] < 0.05
    assert report["observed_orders"]  # an order was actually measured
    # the residual offset against the closed form is a property of the classical
    # plane-strain solution, not of the mesh: it is smaller than the C5 band
    assert abs(report["richardson_relative_offset_vs_analytic"]) < 0.10


def test_plastic_pressure_levels_stay_below_collapse():
    levels = plastic_pressure_levels(CYLINDER)
    assert len(levels) == 5
    assert all(level > CYLINDER.pressure_first_yield for level in levels)
    assert all(level < CYLINDER.pressure_limit for level in levels)


# --------------------------------------------------------------------------- #
# end-to-end (slow lane)
# --------------------------------------------------------------------------- #
def test_full_qualification_all_criteria_pass():
    report = run_qualification(
        uniaxial=UNIAXIAL, torsion=TORSION, cylinder=CYLINDER,
        uniaxial_strains=[0.002, 0.01, 0.03, 0.05],
        shear_levels=[0.0009, 0.0021, 0.006, 0.025],
        cylinder_meshes=(8, 16, 32), bar_mesh=(6, 1, 1), steps=6)
    criteria = report["criteria"]
    failed = {name: entry for name, entry in criteria.items() if entry["passed"] is False}
    assert not failed, f"criteria failed: {failed}"
    for name in ("C1_uniaxial_curve", "C2_yield_stress", "C3_shear_yield",
                 "C4_plastic_radius", "C5_radial_displacement", "C6_pi_plane_circle"):
        assert criteria[name]["passed"] is True, name
    assert criteria["C7_gci"]["status"] == "passed"
    # the hardening branch carries no closed form: it must at least be reported
    hardening = report["case_C_cylinder"]["hardening_reference"]
    assert hardening is not None
    assert hardening["closed_form_available"] is False
    # no Hill oracle for H > 0: the comparison columns must be None, not numbers
    assert hardening["max_r_p_relative_error"] is None
    assert hardening["max_u_profile_relative_error"] is None
    assert all(row["r_p_fem"] > CYLINDER.inner_radius for row in hardening["levels"])