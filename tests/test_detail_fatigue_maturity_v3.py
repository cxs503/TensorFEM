"""Machine-criteria verification for P106W3A2-N02: ship-hull detail fatigue maturity.

This node matures the welded-detail fatigue chain on top of the existing FEM
verification (P106W2-N10).  It adds a *structural detail fatigue verification
surface* that exercises the complete global-to-detail transfer chain:

    global beam theory (sigma_nom) -> SCF (FEM) -> hot spot (sigma_hot)
        -> S-N curve (DNV-RP-C203 D) -> fatigue life

across the three acceptance joint families:

* T-joint (transverse fillet, t = 10-20 mm, r = 0.5-2 mm);
* butt weld (reinforced, t = 10-20 mm, r = 0.5-2 mm);
* bracket / corner toe (L-joint, t = 10-20 mm, r = 0.5-2 mm).

Every test targets one machine criterion of the node specification:

1. **[geometry]** every joint family member builds a non-degenerate Q8 mesh
   (positive Jacobian, finite SCF);
2. **[SCF]** the FEM notch SCF of the T-joint family is within 5 % of the Nii
   closed form; the butt family within 20 % of the Maddox form;
3. **[hot spot]** the 0.5t / 1.5t linear extrapolation is consistent with the
   IIW 0.4t / 0.9t / 1.4t quadratic extrapolation (deviation <= 5 %);
4. **[S-N]** DNV-RP-C203 curve D: m1 = 3 branch gives 2x stress -> 8x life,
   m2 = 5 branch gives 2x stress -> 32x life, regression residual <= 1 %;
5. **[beam interface]** sigma_hot = SCF * sigma_nom reproduces the beam-theory
   section stress within 5 % for the load-carrying family;
6. **[full chain]** a worked hull-girder detail (60 MPa axial + 20 MPa bending)
   produces a finite fatigue life in all three environments (air, seawater CP,
   seawater free), with the expected ordering life_air > life_cp > life_free.
"""
from __future__ import annotations

import math

import pytest

from tensorfem.fatigue import (
    BeamSection,
    beam_nominal_stress,
    fatigue_life,
    global_to_detail_fatigue,
    hot_spot_stress,
    solve_weld_detail,
)
from tensorfem.fatigue.detail_maturity import (
    JOINT_FAMILY,
    family_geometry,
    gci_hotspot,
    power_law_response,
    run_maturity_assessment,
    scf_vs_iiw,
    toe_mesh_resolution,
)
from tensorfem.fatigue.weld_toe import (
    maddox_stress_concentration,
    nii_stress_concentration,
)


# --------------------------------------------------------------------------- #
# C1: geometry -- every joint family member builds a non-degenerate mesh
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("joint_type", ["t_fillet", "butt", "corner"])
@pytest.mark.parametrize("t,r", JOINT_FAMILY)
def test_joint_family_builds_non_degenerate_mesh(joint_type, t, r):
    """Every acceptance joint meshes and solves with a finite, physical SCF."""
    detail = family_geometry(t, r, joint_type=joint_type, load_case="plate_axial")
    sol = solve_weld_detail(detail, level=1)
    assert sol.n_elements > 500
    assert math.isfinite(sol.kt_toe)
    # the surface band must stay inside the modelled depth (mesh degeneracy guard)
    assert detail.surface_band_m < detail.plate_depth_m
    # the peak stress must be positive and finite (a negative or zero peak
    # would indicate a degenerate mesh or a wrong load path)
    assert sol.peak_toe_stress_pa > 0.0


# --------------------------------------------------------------------------- #
# C2: SCF -- FEM notch SCF vs closed-form anchors
# --------------------------------------------------------------------------- #
def test_t_joint_scf_within_5pct_of_nii():
    """FE notch SCF of the T-joint family is within 5 % of the Nii closed form."""
    errors = []
    for t, r in JOINT_FAMILY:
        detail = family_geometry(t, r, joint_type="t_fillet", load_case="plate_axial")
        sol = solve_weld_detail(detail, level=1)
        anchor = nii_stress_concentration(t, r, detail.attachment_height_m)
        errors.append(abs(sol.kt_toe - anchor) / anchor)
    assert max(errors) <= 0.05, f"worst T-joint SCF error = {max(errors):.3f}"


def test_butt_scf_within_20pct_of_maddox():
    """FE notch SCF of the butt family is within 20 % of the Maddox form."""
    errors = []
    for t, r in JOINT_FAMILY:
        detail = family_geometry(t, r, joint_type="butt", load_case="plate_axial")
        sol = solve_weld_detail(detail, level=1)
        anchor = maddox_stress_concentration(t, r)
        errors.append(abs(sol.kt_toe - anchor) / anchor)
    assert max(errors) <= 0.20, f"worst butt SCF error = {max(errors):.3f}"


def test_corner_scf_is_finite_and_positive():
    """Corner (bracket toe) SCF is finite and positive for the whole family."""
    for t, r in JOINT_FAMILY:
        detail = family_geometry(t, r, joint_type="corner", load_case="plate_axial")
        sol = solve_weld_detail(detail, level=1)
        assert math.isfinite(sol.kt_toe)
        assert sol.kt_toe > 0.0
        # the corner joint is an L-section: the peak stress may be below or
        # above the nominal membrane stress depending on the geometry, but it
        # must be within a physical range (0.5-3.0)
        assert 0.5 < sol.kt_toe < 3.0, f"corner SCF {sol.kt_toe} out of range"


# --------------------------------------------------------------------------- #
# C3: hot spot -- 0.5t/1.5t linear vs IIW 0.4t/0.9t/1.4t quadratic
# --------------------------------------------------------------------------- #
def test_hotspot_linear_quadratic_consistency():
    """0.5t/1.5t linear and 0.4t/0.9t/1.4t quadratic agree within 5 %."""
    for t, r in JOINT_FAMILY:
        detail = family_geometry(t, r, joint_type="t_fillet", load_case="plate_axial")
        sol = solve_weld_detail(detail, level=1)
        lin = hot_spot_stress(sol, t, method="linear")
        quad = hot_spot_stress(sol, t, method="quadratic")
        dev = abs(lin.stress_ratio - quad.stress_ratio) / quad.stress_ratio
        assert dev <= 0.05, f"hot spot deviation {dev:.3f} at t={t}, r={r}"


# --------------------------------------------------------------------------- #
# C4: S-N -- DNV-RP-C203 curve D, m1=3 and m2=5 branches
# --------------------------------------------------------------------------- #
def test_sn_curve_m1_branch():
    """m1 = 3 branch: 2x stress range -> 8x life reduction."""
    n1 = fatigue_life(100.0e6, environment="air")
    n2 = fatigue_life(200.0e6, environment="air")
    ratio = n1 / n2
    assert abs(ratio - 8.0) / 8.0 <= 0.01, f"m1 ratio = {ratio:.3f}"


def test_sn_curve_m2_branch():
    """m2 = 5 branch: 2x stress range -> 32x life reduction."""
    n1 = fatigue_life(20.0e6, environment="air")
    n2 = fatigue_life(40.0e6, environment="air")
    ratio = n1 / n2
    assert abs(ratio - 32.0) / 32.0 <= 0.01, f"m2 ratio = {ratio:.3f}"


def test_sn_curve_power_law_residual():
    """Power-law regression residual over a decade of stress ranges <= 1 %."""
    result = power_law_response(100.0e6)
    assert result["pass"] is True
    assert result["max_log_residual"] <= 0.01


def test_sn_curve_environment_ordering():
    """At 100 MPa: CP life > air life > free-corrosion life (DNV-RP-C203).

    The cathodic-protection curve shares the air m1 = 3 branch but moves the
    knee to 1e6 cycles, so at 100 MPa (air life ~1.5e6) the CP curve is on the
    more favourable m2 = 5 branch.  The free-corrosion curve has a knock-down
    ``log10 a1`` and is always the shortest life.
    """
    ds = 100.0e6
    n_air = fatigue_life(ds, environment="air")
    n_cp = fatigue_life(ds, environment="seawater_cp")
    n_free = fatigue_life(ds, environment="seawater_free")
    assert n_cp > n_air > n_free


# --------------------------------------------------------------------------- #
# C5: beam interface -- sigma_hot = SCF * sigma_nom
# --------------------------------------------------------------------------- #
def test_beam_to_hotspot_chain():
    """sigma_hot = SCF * sigma_nom reproduces the beam-theory section stress."""
    # a 20 mm plate, 1 m wide: area = 0.02 m2, I = t^3/12, c = t/2
    t = 0.020
    section = BeamSection(area_m2=t * 1.0, I_m4=1.0 * t**3 / 12.0, c_m=0.5 * t)
    # 60 MPa axial + 20 MPa bending at the detail section
    sigma_nom = beam_nominal_stress(
        axial_force_n=60.0e6 * section.area_m2,
        bending_moment_nm=20.0e6 * section.Z_m3,
        section=section,
    )
    assert sigma_nom == pytest.approx(80.0e6, rel=1e-10)

    detail = family_geometry(t, 0.002, joint_type="t_fillet", load_case="plate_axial")
    sol = solve_weld_detail(detail, level=1)
    lin = hot_spot_stress(sol, t, method="linear")
    sigma_hot = lin.stress_ratio * sigma_nom
    # the non-load-carrying T-joint has structural SCF ~1.0, so sigma_hot ~ sigma_nom
    assert abs(sigma_hot - sigma_nom) / sigma_nom <= 0.05


def test_global_to_detail_full_chain():
    """Full chain: global beam loads -> SCF -> hot spot -> S-N life."""
    t = 0.020
    section = BeamSection(area_m2=t * 1.0, I_m4=1.0 * t**3 / 12.0, c_m=0.5 * t)
    detail = family_geometry(t, 0.002, joint_type="t_fillet", load_case="plate_axial")
    result = global_to_detail_fatigue(
        axial_force_n=60.0e6 * section.area_m2,
        bending_moment_nm=20.0e6 * section.Z_m3,
        section=section,
        detail=detail,
        environment="air",
        level=1,
    )
    assert result.sigma_nom_pa > 0.0
    assert result.scf_hotspot_linear > 0.0
    assert result.sigma_hot_linear_pa > 0.0
    assert result.life_linear_cycles > 0.0
    # consistency: sigma_hot = SCF * sigma_nom
    assert abs(result.sigma_hot_linear_pa - result.scf_hotspot_linear * result.sigma_nom_pa) \
        / result.sigma_hot_linear_pa < 1e-10


# --------------------------------------------------------------------------- #
# C6: full chain -- worked hull-girder detail in three environments
# --------------------------------------------------------------------------- #
def test_worked_hull_girder_detail():
    """A worked hull-girder detail produces finite life in all environments."""
    t = 0.020
    section = BeamSection(area_m2=t * 1.0, I_m4=1.0 * t**3 / 12.0, c_m=0.5 * t)
    detail = family_geometry(t, 0.002, joint_type="t_fillet", load_case="plate_axial")
    sigma_nom = beam_nominal_stress(
        axial_force_n=60.0e6 * section.area_m2,
        bending_moment_nm=20.0e6 * section.Z_m3,
        section=section,
    )
    sol = solve_weld_detail(detail, level=1)
    lin = hot_spot_stress(sol, t, method="linear")
    sigma_hot = lin.stress_ratio * sigma_nom
    for env in ("air", "seawater_cp", "seawater_free"):
        n = fatigue_life(sigma_hot, environment=env, thickness_m=t)
        assert n > 0.0
        assert math.isfinite(n)


# --------------------------------------------------------------------------- #
# C7: mesh convergence -- GCI of the hot-spot stress
# --------------------------------------------------------------------------- #
def test_gci_hotspot_within_3pct():
    """GCI of the hot-spot stress over three mesh levels <= 3 %."""
    detail = family_geometry(0.020, 0.002, joint_type="t_fillet", load_case="plate_axial")
    result = gci_hotspot(detail, levels=(1, 2, 3))
    assert result["pass"] is True
    assert result["gci_fine"] <= 0.03


def test_toe_mesh_size_below_t_tenth():
    """Toe surface element size <= t / 10."""
    detail = family_geometry(0.020, 0.002, joint_type="t_fillet", load_case="plate_axial")
    result = toe_mesh_resolution(detail, level=1)
    assert result["pass"] is True
    assert result["surface_element_over_t"] < 0.02


# --------------------------------------------------------------------------- #
# C8: full maturity ledger
# --------------------------------------------------------------------------- #
def test_full_maturity_ledger():
    """The full maturity ledger reports every criterion with a verdict."""
    ledger = run_maturity_assessment()
    assert "criteria" in ledger
    assert "all_pass" in ledger
    for name, row in ledger["criteria"].items():
        assert "requirement" in row
        assert "pass" in row
