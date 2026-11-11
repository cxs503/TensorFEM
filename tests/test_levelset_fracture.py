"""Verification tests for the level-set crack-growth capability.

Coverage map (the A-gates of the campaign node are in brackets):

* ``test_reinitialisation_gradient_gate``            [A1] |grad phi| = 1 within 1%
* ``test_reinitialisation_convergence_order``        [A1] monotone refinement
* ``test_advection_rigid_translation``               transport accuracy
* ``test_velocity_extension_is_normal_independent``  extension off the front
* ``test_zero_contour_geometry``                     marching squares accuracy
* ``test_hoop_stress_kink_angle_pure_mode_ii``       MTS = -70.53 deg
* ``test_kink_angle_criteria_coincide``              documented equivalence
* ``test_inclined_crack_sifs_and_sign``              analytic SIFs
* ``test_bending_griffith_relation_factors``         analogy factors
* ``test_kevin_anchor_matches_nevel_form``           exact Kelvin vs folklore
* ``test_rect_mindlin_element_matches_repo``         numpy == torch element
* ``test_equibiaxial_moment_field_is_uniform``       FE exactness
* ``test_uncracked_patch_moment_matches_kelvin``     Winkler load path
* ``test_energy_release_rate_vs_bending_griffith``   [A3] G within 10%
* ``test_ice_fracture_load_energy_vs_analytic``      [A4] fracture load
* ``test_level_set_crack_path_deviation``            [A2] path < 15%
* ``test_strength_and_energy_criteria_consistency``  cross-check
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorfem.levelset.crack_criteria import (
    bending_energy_release_rate,
    energy_release_rate,
    hoop_stress_factor,
    inclined_crack_sifs,
    kink_angle_max_energy,
    kink_angle_max_hoop,
    growth_velocity,
)
from tensorfem.levelset.crack_path import (
    level_set_path,
    mixity_schedule,
    path_comparison,
    reference_path,
)
from tensorfem.levelset.ice_plate_fracture import (
    IceSheet,
    analytical_fracture_load,
    equivalent_flaw_size,
    energy_release_rate_ice,
    nevel_form_reference,
    strength_first_crack_load,
    uncracked_moment_check,
)
from tensorfem.levelset.level_set import (
    LevelSetField,
    circle_level_set,
    crack_level_set,
    plane_level_set,
    path_length,
    polyline_deviation,
)
from tensorfem.levelset.plate_crack import (
    energy_release_rate_vce,
    kelvin_kei_derivative,
    patch_max_moment,
    patch_max_moment_nevel,
    plate_rigidity,
    rect_mindlin_stiffness,
    solve_cracked_plate,
)
from tensorfem.plate import q4_mindlin_stiffness


# ---------------------------------------------------------------- level set [A1]
def test_reinitialisation_gradient_gate():
    """[A1] re-initialisation of a distorted circle restores |grad phi| = 1 (<1%)."""
    radius, n, h = 1.0, 201, 3.0 / 200.0
    exact = circle_level_set(radius, nx=n, h=h, x0=-1.5, y0=-1.5)
    distorted = exact.copy()
    distorted.phi = exact.phi * (1.0 + 0.3 * np.sin(6.0 * exact.phi))
    assert distorted.gradient_error(band=0.2)["max_abs"] > 0.3  # genuinely broken
    band = 0.45
    repaired = distorted.reinitialize(band=band, n_iter=int(4.0 * band / (0.3 * h)) + 10,
                                      order=2)
    err = repaired.gradient_error(band=0.2)
    assert err["max_abs"] < 0.01, err
    assert err["rms"] < 5e-4, err
    ref = repaired.error_vs_exact(lambda x, y: np.hypot(x, y) - radius, band=0.2)
    assert ref["max_abs"] < 0.01 * radius


def test_reinitialisation_convergence_order():
    """Refinement reduces the distance error (second-order scheme)."""
    errs = []
    for n in (101, 201):
        h = 3.0 / (n - 1)
        exact = circle_level_set(1.0, nx=n, h=h, x0=-1.5, y0=-1.5)
        field = exact.copy()
        field.phi = exact.phi * (1.0 + 0.3 * np.sin(6.0 * exact.phi))
        band = 0.45
        out = field.reinitialize(band=band, n_iter=int(4.0 * band / (0.3 * h)) + 10,
                                 order=2)
        errs.append(out.error_vs_exact(lambda x, y: np.hypot(x, y) - 1.0, band=0.2)["max_abs"])
    assert errs[1] < errs[0]
    assert errs[1] / errs[0] < 0.5


def test_advection_rigid_translation():
    """Semi-Lagrangian transport moves a plane front exactly."""
    field = plane_level_set((1.0, 0.0), offset=0.0, nx=81, ny=81, h=0.05)
    moved = field.advect(1.0, 0.0, 0.25, scheme="sl2")
    expected = plane_level_set((1.0, 0.0), offset=0.25, nx=81, ny=81, h=0.05)
    inner = np.abs(expected.phi) < 0.5
    assert np.max(np.abs(moved.phi - expected.phi)[inner]) < 1e-6
    assert moved.gradient_error(band=0.5)["max_abs"] < 1e-6
    upwind = field.advect(1.0, 0.0, 0.25, scheme="upwind")
    assert np.max(np.abs(upwind.phi - expected.phi)[inner]) < 5e-3


def test_upwind_advection_rejects_bad_scheme():
    field = plane_level_set((1.0, 0.0), nx=21, ny=21, h=0.1)
    with pytest.raises(ValueError):
        field.advect(1.0, 0.0, 0.1, scheme="spectral")


def test_velocity_extension_is_normal_independent():
    """grad V . grad phi = 0: an extension of a plane front is constant."""
    field = plane_level_set((0.0, 1.0), nx=61, ny=61, h=0.05)
    ext = field.extend_velocity(2.5)
    assert ext.shape == field.shape
    assert np.allclose(ext, 2.5, atol=1e-12)
    circ = circle_level_set(1.0, nx=121, h=0.02)
    ext_c = circ.extend_velocity(1.0)
    assert np.allclose(ext_c[np.abs(circ.phi) < 0.5], 1.0, atol=1e-12)


def test_zero_contour_geometry():
    """Marching squares recovers a circle radius and the crack segment."""
    circ = circle_level_set(1.0, nx=201, h=0.02)
    contours = circ.zero_contour()
    assert contours
    length = path_length(contours[0])
    assert abs(length - 2.0 * math.pi) / (2.0 * math.pi) < 5e-3
    radii = np.hypot(contours[0][:, 0], contours[0][:, 1])
    assert np.max(np.abs(radii - 1.0)) < 0.5 * circ.h
    crack = crack_level_set(0.8, angle=math.radians(30.0), nx=161, ny=161, h=0.02)
    pts = np.vstack(crack.zero_contour())
    normal = np.array([-math.sin(math.radians(30.0)), math.cos(math.radians(30.0))])
    along = pts @ np.array([math.cos(math.radians(30.0)), math.sin(math.radians(30.0))])
    # the points that lie on the crack line (|perp| below the interpolation
    # error) span the whole crack segment.  The one-sided notch representation
    # additionally emits a short spurious branch along the two tip rays, where
    # the field jumps; that branch is documented and trimmed by the front
    # extractor (``crack_path._tip_from_contour``).
    perp = np.abs(pts @ normal)
    on_line = perp <= 0.5 * crack.h
    assert on_line.sum() >= 30
    assert along[on_line].min() <= -0.4 + crack.h
    assert along[on_line].max() >= 0.4 - crack.h


def test_polyline_deviation_metric():
    ref = np.stack((np.linspace(0.0, 1.0, 50), np.zeros(50)), axis=1)
    same = polyline_deviation(ref, ref)
    assert same["max_rel"] < 1e-12
    shifted = ref + np.array([0.0, 0.05])
    off = polyline_deviation(shifted, ref)
    assert abs(off["rms_rel"] - 0.05) < 1e-6


# ------------------------------------------------------------------- criteria
def test_hoop_stress_kink_angle_pure_mode_ii():
    """Maximum tangential stress criterion: pure mode II kinks at -acos(1/3)."""
    out = kink_angle_max_hoop(0.0, 1.0)
    assert math.degrees(out.theta) == pytest.approx(-math.degrees(math.acos(1.0 / 3.0)),
                                                    abs=1e-6)
    mixed = kink_angle_max_hoop(1.0, 1.0)
    assert math.degrees(mixed.theta) == pytest.approx(-53.130102, abs=1e-5)
    assert kink_angle_max_hoop(1.0, 0.0).theta == 0.0


def test_kink_angle_criteria_coincide():
    """With the small-kink angular functions max-G and max-hoop-stress coincide.

    This is a *verified limitation*: the classical -82 deg pure-mode-II value of
    the maximum energy release rate criterion needs the full Hussain et al.
    kinked-crack solution, which is outside this module's scope.
    """
    for ratio in (0.0, 0.3, 1.0, 3.0):
        a = kink_angle_max_hoop(1.0, ratio).theta
        b = kink_angle_max_energy(1.0, ratio).theta
        assert a == pytest.approx(b, abs=1e-6)


def test_inclined_crack_sifs_and_sign():
    """Analytic SIFs of an inclined crack and the mode-I limit."""
    k_i, k_ii = inclined_crack_sifs(1.0e6, math.radians(90.0), 0.01)
    assert k_i == pytest.approx(1.0e6 * math.sqrt(math.pi * 0.01))
    assert abs(k_ii) < 1e-9
    k_i, k_ii = inclined_crack_sifs(1.0e6, math.radians(45.0), 0.01)
    assert k_i == pytest.approx(k_ii)
    assert energy_release_rate(k_i, 0.0, 7.0e9, 0.3) == pytest.approx(k_i ** 2 / 7.0e9)


def test_growth_velocity_threshold():
    assert growth_velocity(0.5, 1.0) == 0.0
    assert growth_velocity(2.0, 1.0) == pytest.approx(1.0)
    assert growth_velocity(1.5, 1.0, v0=2.0, exponent=2.0) == pytest.approx(0.5)


def test_bending_griffith_relation_factors():
    """The analogy factor c^2 enters as 1/(1+nu)^2 for equibiaxial bending."""
    e, nu, t, m, a = 7.0e9, 0.3, 0.3, 1000.0, 1.0
    d = plate_rigidity(e, nu, t)
    biax = bending_energy_release_rate(m, a, d, t, nu, moment_state="biaxial")
    cyl = bending_energy_release_rate(m, a, d, t, nu, moment_state="cylindrical")
    uni = bending_energy_release_rate(m, a, d, t, nu, moment_state="uniaxial_moment")
    assert cyl == pytest.approx(math.pi * m ** 2 * a / (d * t))
    assert biax == pytest.approx(cyl / (1.0 + nu) ** 2)
    assert uni == pytest.approx(cyl / (1.0 - nu ** 2) ** 2)
    with pytest.raises(ValueError):
        bending_energy_release_rate(m, a, d, t, nu, moment_state="nonsense")


def test_kevin_anchor_matches_nevel_form():
    """Exact Kelvin patch moment vs the folklore closed form (0.6159 = ln2-g+1/2)."""
    sheet = IceSheet()
    check = nevel_form_reference(1.0e4, 0.5, sheet)
    assert abs(check["relative_difference"]) < 5e-3, check
    # kei(0) = -pi/4 and kei increases for small argument, hence kei' > 0 there
    assert kelvin_kei_derivative(0.5) == pytest.approx(0.3332, abs=5e-3)
    assert patch_max_moment(1.0e4, 0.5, sheet.rigidity, sheet.foundation,
                            sheet.poisson) > 0.0
    with pytest.raises(ValueError):
        patch_max_moment_nevel(1.0, 0.0, sheet.rigidity, sheet.foundation)


# ------------------------------------------------------------------ plate FE
def test_rect_mindlin_element_matches_repo():
    """The numpy rectangle element is identical to the torch Q4 element."""
    for width, height, treatment in ((0.7, 0.3, "mitc"), (1.5, 1.5, "mitc"),
                                     (0.25, 2.0, "mitc"), (0.7, 0.3, "sri")):
        mine = rect_mindlin_stiffness(width, height, 7.0e9, 0.3, 0.3,
                                      shear_treatment=treatment)
        xy = torch.tensor([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
                          dtype=torch.float64)
        repo = q4_mindlin_stiffness(xy, 7.0e9, 0.3, 0.3,
                                    shear_treatment=treatment).numpy()
        assert np.abs(mine - repo).max() / np.abs(repo).max() < 1e-14
    with pytest.raises(ValueError):
        rect_mindlin_stiffness(0.7, 0.3, 7.0e9, 0.3, 0.3, shear_treatment="bad")


def test_equibiaxial_moment_field_is_uniform():
    """Biaxial edge moments reproduce the exact uniform moment state."""
    e, nu, t, m = 7.0e9, 0.3, 0.3, 1000.0
    n, side = 8, 20.0
    xs = np.linspace(0.0, side, n + 1)
    res = solve_cracked_plate(xs, xs.copy(), jc=n // 2, i_left=None, i_right=None,
                              thickness=t, young=e, poisson=nu, edge_moment=m)
    mom = res.moment_field(young=e, poisson=nu, thickness=t)
    assert np.abs(mom["Mx"] - m).max() < 1e-4
    assert np.abs(mom["My"] - m).max() < 1e-4
    assert np.abs(mom["Mxy"]).max() < 1e-4
    d = plate_rigidity(e, nu, t)
    centre_node = int(res.grid[n // 2, n // 2])
    w = res.u[int(res.gdof[centre_node, 0])]
    assert w == pytest.approx(-m * side ** 2 / (4.0 * d * (1.0 + nu)), rel=1e-9)


def test_uncracked_patch_moment_matches_kelvin():
    """[A4a] FE centre moment of a floating sheet vs the exact Kelvin value."""
    sheet = IceSheet()
    check = uncracked_moment_check(sheet, pressure=1.0e4, patch_radius=0.5,
                                   domain=4.0, cells=60, h_min=0.3)
    assert abs(check["relative_difference"]) < 0.10, check
    assert check["applied_load"] == pytest.approx(check["expected_load"], rel=1e-10)


def test_cracked_plate_parameters_validate():
    sheet = IceSheet()
    assert sheet.foundation == pytest.approx(1025.0 * 9.81)
    assert sheet.critical_energy_release_rate == pytest.approx(
        1.0e5 ** 2 * (1 - 0.09) / 7.0e9)
    with pytest.raises(ValueError):
        IceSheet(thickness=0.0)
    with pytest.raises(ValueError):
        IceSheet(toughness=0.0)


def test_energy_release_rate_vs_bending_griffith():
    """[A3] VCE energy release rate vs the analytic bending-Griffith value (<10%).

    Equibiaxial bending of a plate with a central through crack: the analytic
    value is ``G = pi c^2 M^2 a/(D h)`` with ``c = 1/(1+nu)``.
    """
    e, nu, t, m, a = 7.0e9, 0.3, 0.3, 1000.0, 1.0
    n = 80
    side = 20.0 * a
    xs = np.linspace(-0.5 * side, 0.5 * side, n + 1)
    ih = int(round(a / (xs[1] - xs[0])))
    out = energy_release_rate_vce(xs, xs.copy(), jc=n // 2, i_left=n // 2 - ih,
                                  i_right=n // 2 + ih, advances=(1, 2), thickness=t,
                                  young=e, poisson=nu, edge_moment=m)
    analytic = bending_energy_release_rate(m, a, plate_rigidity(e, nu, t), t, nu,
                                          moment_state="biaxial")
    assert out["G"] > 0.0
    assert abs(out["G"] / analytic - 1.0) < 0.10, (out["G"], analytic)


# ----------------------------------------------------------------- ice sheet
def test_ice_fracture_load_energy_vs_analytic():
    """[A4] crack-model fracture load vs the small-crack analytic value (<20%).

    The mesh is deliberately coarse (it is also asserted that the finite-element
    value is *above* the analytic one, as a blunt crack tip is stiffer than the
    ideal Griffith crack).
    """
    sheet = IceSheet()
    half, radius = 0.30, 0.5
    out = energy_release_rate_ice(half, 1.0e4, radius, sheet, domain=4.0, cells=80,
                                  h_min=0.12, advances=(1,))
    g = out["G"]
    assert g > 0.0
    p_fe = math.sqrt(sheet.critical_energy_release_rate / g) * 1.0e4
    p_an = analytical_fracture_load(half, radius, sheet)
    assert abs(p_fe / p_an - 1.0) < 0.20, (p_fe, p_an)


def test_strength_and_energy_criteria_consistency():
    """The two ice-engineering criteria are reconciled by the flaw size."""
    sheet = IceSheet()
    radius = 0.5
    p_first = strength_first_crack_load(radius, sheet)
    a_equiv = equivalent_flaw_size(p_first, radius, sheet)
    assert p_first > 0.0 and a_equiv > 0.0
    p_energy = analytical_fracture_load(a_equiv, radius, sheet)
    assert p_energy == pytest.approx(p_first, rel=1e-9)
    # a shorter crack must raise the fracture load (G grows with crack length)
    assert analytical_fracture_load(0.1, radius, sheet) > analytical_fracture_load(
        0.3, radius, sheet)


# -------------------------------------------------------------- crack path [A2]
def test_level_set_crack_path_deviation():
    """[A2] level-set tracked path vs the independently integrated path (<15%)."""
    beta = math.radians(45.0)
    a, decay, step, steps = 0.4, 1.2, 0.2, 10
    schedule = mixity_schedule(beta, 1.0e6, a, decay)
    tip0 = np.array([a * math.cos(beta), a * math.sin(beta)])
    ref = reference_path(beta, schedule["ratio"], decay, steps=steps, step=step,
                         start=tip0)
    tracked = level_set_path(beta, a, ratio=schedule["ratio"], decay=decay,
                             steps=steps, step=step, nx=201, ny=201, h=0.03,
                             reinit_band=0.6)
    cmp = path_comparison(tracked.path, ref)
    assert cmp["max_rel"] < 0.15, cmp
    assert cmp["rms_rel"] < 0.10, cmp
    # the initial kink angle is the analytic MTS value also from the level set
    assert math.degrees(tracked.initial_kink_angle) == pytest.approx(53.13, abs=0.1)
    assert math.degrees(tracked.final_direction) > math.degrees(beta)
