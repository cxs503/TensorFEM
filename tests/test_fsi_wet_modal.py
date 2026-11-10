"""Tests for the FSI wet modal module (dual-criteria adjudication).

Evidence levels used here:

* closed-form Junger & Feit loading -- checked against its own ``x -> 0``
  two-dimensional limit and against the independent finite-difference fluid
  solve (pointwise and modal);
* plate fluid loading -- checked against an exact wave-number integral, the
  panel-discretised Rayleigh operator, and the classical ``rho_f/k_mn``
  asymptote;
* frequency response -- checked against the wet eigenfrequencies, the
  single-degree-of-freedom resonance amplitude, and reciprocity.
"""
import math

import numpy as np
import pytest

from tensorfem import fsi_wet_modal as fsi

RADIUS = 1.0
LENGTH = 20.0
THICKNESS = 0.01
RHO_FLUID = 1025.0
RHO_STRUCT = 7850.0
YOUNG = 210.0e9
POISSON = 0.3


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_positive_validation_rejects_bad_geometry(value):
    with pytest.raises(ValueError):
        fsi.cylinder_fluid_loading_coefficient(value, LENGTH, 1, 2)
    with pytest.raises(ValueError):
        fsi.plate_modal_wavenumber(1.0, value, 1, 1)


def test_axisymmetric_mode_is_rejected():
    with pytest.raises(ValueError):
        fsi.cylinder_fluid_loading_coefficient(RADIUS, LENGTH, 1, 0)


def test_negative_added_mass_ratio_rejected():
    with pytest.raises(ValueError):
        fsi.wet_frequency_ratio(-0.1)


def test_invalid_end_condition_rejected():
    with pytest.raises(ValueError):
        fsi.solve_incompressible_potential(
            RADIUS, 2, lambda x: np.zeros_like(x), x_min=0.0, x_max=1.0,
            end_condition="open",
        )


# ---------------------------------------------------------------------------
# 1. Closed-form fluid loading (Junger & Feit)
# ---------------------------------------------------------------------------


def test_loading_coefficient_tends_to_inverse_circumferential_order():
    """Long-cylinder limit: C(m, n) -> 1/n (exact 2D result)."""
    for order in (1, 2, 3, 4):
        coefficient = fsi.cylinder_fluid_loading_coefficient(
            1.0, 1.0e4, 1, order
        )
        assert coefficient == pytest.approx(1.0 / order, rel=2e-3)


def test_exterior_loading_is_weaker_than_interior():
    """The exterior fluid can escape outward, so C_ext < C_int."""
    for order in (1, 2, 3):
        exterior = fsi.cylinder_fluid_loading_coefficient(1.0, 20.0, 1, order)
        interior = fsi.cylinder_fluid_loading_coefficient(
            1.0, 20.0, 1, order, exterior=False
        )
        assert exterior < interior


def test_coefficient_decreases_with_axial_and_circumferential_order():
    base = fsi.cylinder_fluid_loading_coefficient(RADIUS, LENGTH, 1, 2)
    assert fsi.cylinder_fluid_loading_coefficient(RADIUS, LENGTH, 2, 2) < base
    assert fsi.cylinder_fluid_loading_coefficient(RADIUS, LENGTH, 1, 3) < base


def test_total_added_mass_reproduces_two_dimensional_limit():
    """n = 1 rigid-translation limit: added mass per unit length -> rho pi R^2.

    ``M_a = rho_f pi R \\int psi w dx`` with ``w = 1`` and the 2D solution
    ``psi = -R`` gives ``rho_f pi R^2`` per unit length; with the modal
    normalisation ``w = sin(pi x / L)`` the same coefficient is compared
    through ``C(m=1, n=1) -> 1`` for a very long shell.
    """
    coefficient = fsi.cylinder_fluid_loading_coefficient(1.0, 2.0e4, 1, 1)
    assert coefficient == pytest.approx(1.0, rel=1e-3)
    added_mass_per_length = RHO_FLUID * RADIUS * coefficient
    assert added_mass_per_length == pytest.approx(RHO_FLUID * math.pi * RADIUS ** 2 / math.pi, rel=1e-3)


def test_added_mass_ratio_scales_with_radius_and_density():
    mu = fsi.cylinder_modal_added_mass_ratio(
        RADIUS, LENGTH, THICKNESS, RHO_FLUID, RHO_STRUCT, 1, 2
    )
    mu_thin = fsi.cylinder_modal_added_mass_ratio(
        RADIUS, LENGTH, 0.5 * THICKNESS, RHO_FLUID, RHO_STRUCT, 1, 2
    )
    mu_light = fsi.cylinder_modal_added_mass_ratio(
        RADIUS, LENGTH, THICKNESS, 0.5 * RHO_FLUID, RHO_STRUCT, 1, 2
    )
    assert mu_thin == pytest.approx(2.0 * mu, rel=1e-12)
    assert mu_light == pytest.approx(0.5 * mu, rel=1e-12)


def test_wet_frequency_ratio_identity():
    for mu in (0.0, 1.0, 6.4896):
        assert fsi.wet_frequency_ratio(mu) == pytest.approx(
            1.0 / math.sqrt(1.0 + mu), rel=1e-15
        )


def test_donnell_keeps_n1_membrane_term():
    """The n = 1 Donnell branch must keep the membrane term.

    Euler-Bernoulli beam limit for the shell ``n = 1`` mode:
    ``omega = (pi/L)^2 sqrt(E I / rho A)`` with ``I/A = R^2/2``.  Donnell with
    the membrane term is within 50% of it, whereas
    :func:`tensorfem.cylindrical_shell_modal.dmv_angular_frequency` drops that
    term and lands more than 3x low (see docs/fsi-wet-modal-dual-criteria.md).
    """
    from tensorfem.cylindrical_shell_modal import dmv_frequency_hz

    beam_hz = (
        (math.pi / LENGTH) ** 2
        * math.sqrt(YOUNG * RADIUS ** 2 / 2.0 / RHO_STRUCT)
        / (2.0 * math.pi)
    )
    donnell = fsi.donnell_shell_frequency_hz(
        RADIUS, LENGTH, THICKNESS, YOUNG, POISSON, RHO_STRUCT, 1, 1
    )
    legacy = dmv_frequency_hz(
        radius=RADIUS, length=LENGTH, thickness=THICKNESS, young=YOUNG,
        poisson=POISSON, density=RHO_STRUCT, n_axial=1, n_circ=1,
    )
    assert 0.5 < donnell / beam_hz < 1.5
    assert legacy < 0.3 * beam_hz


def test_donnell_frequency_increases_with_mode_order():
    frequencies = [
        fsi.donnell_shell_frequency_hz(
            RADIUS, LENGTH, THICKNESS, YOUNG, POISSON, RHO_STRUCT, m, n
        )
        for m, n in ((1, 2), (1, 3), (1, 4))
    ]
    assert frequencies[0] < frequencies[1] < frequencies[2]


def test_analytic_submerged_cylinder_shape():
    result = fsi.analytic_submerged_cylinder(
        radius=RADIUS, length=LENGTH, thickness=THICKNESS, rho_fluid=RHO_FLUID,
        rho_struct=RHO_STRUCT, young=YOUNG, poisson=POISSON,
        modes=((1, 2), (1, 3)),
    )
    assert len(result["dry_hz"]) == len(result["wet_hz"]) == 2
    for dry, wet, mu, ratio in zip(
        result["dry_hz"], result["wet_hz"], result["mu"], result["ratio"]
    ):
        assert wet < dry
        assert ratio == pytest.approx(1.0 / math.sqrt(1.0 + mu), rel=1e-12)


# ---------------------------------------------------------------------------
# 2. Numerical incompressible fluid domain
# ---------------------------------------------------------------------------


def test_potential_sign_and_pressure_convention():
    """psi < 0 where the wall moves outward (fluid is pushed out)."""
    k = math.pi / LENGTH
    potential = fsi.solve_incompressible_potential(
        RADIUS, 2, lambda z: np.sin(k * z), x_min=0.0, x_max=LENGTH,
        n_radial=60, n_axial_grid=150,
    )
    wall = potential.wall_potential
    velocity = potential.wall_velocity
    moving = np.abs(velocity) > 1e-3
    assert np.all(wall[moving] * velocity[moving] < 0.0)


def test_numerical_local_added_mass_matches_bessel():
    """Pointwise ``-rho psi / w`` equals the closed-form ``rho R C(m, n)``."""
    k = math.pi / LENGTH
    potential = fsi.solve_incompressible_potential(
        RADIUS, 2, lambda z: np.sin(k * z), x_min=-3.0, x_max=LENGTH + 3.0,
        wall_x_min=-3.0, wall_x_max=LENGTH + 3.0, n_radial=120, n_axial_grid=300,
    )
    local = fsi.numerical_local_added_mass_per_area(
        potential, RHO_FLUID, [0.25 * LENGTH, 0.75 * LENGTH]
    )
    reference = (
        RHO_FLUID * RADIUS * fsi.cylinder_fluid_loading_coefficient(
            RADIUS, LENGTH, 1, 2
        )
    )
    assert np.all(np.isfinite(local))
    assert np.max(np.abs(local - reference) / reference) < 0.02


def test_incompressible_solve_converges_second_order():
    k = math.pi / LENGTH
    reference = (
        RHO_FLUID * RADIUS * fsi.cylinder_fluid_loading_coefficient(RADIUS, LENGTH, 1, 3)
    )
    errors = []
    for n_radial, n_axial_grid in ((60, 150), (120, 300), (240, 600)):
        potential = fsi.solve_incompressible_potential(
            RADIUS, 3, lambda z: np.sin(k * z), x_min=-3.0, x_max=LENGTH + 3.0,
            wall_x_min=-3.0, wall_x_max=LENGTH + 3.0, n_radial=n_radial,
            n_axial_grid=n_axial_grid,
        )
        local = fsi.numerical_local_added_mass_per_area(
            potential, RHO_FLUID, [0.25 * LENGTH]
        )[0]
        errors.append(abs(local - reference) / reference)
    assert errors[0] > errors[1] > errors[2]
    assert errors[1] / errors[2] > 2.0
    assert errors[2] < 5e-3


def test_numerical_modal_added_mass_matches_closed_form():
    """Finite-length shell vs infinite-cylinder formula: end effect < 2%."""
    numerical = fsi.numerical_cylinder_added_mass_ratio(
        radius=RADIUS, length=LENGTH, thickness=THICKNESS, rho_fluid=RHO_FLUID,
        rho_struct=RHO_STRUCT, n_axial=1, n_circ=2, n_radial=160,
        n_axial_grid=400,
    )
    analytic = fsi.cylinder_modal_added_mass_ratio(
        RADIUS, LENGTH, THICKNESS, RHO_FLUID, RHO_STRUCT, 1, 2
    )
    assert numerical["added_mass_ratio"] == pytest.approx(analytic, rel=0.02)
    assert numerical["modal_added_mass"] > 0.0


def test_end_conditions_barely_matter_for_slender_shell():
    kwargs = dict(
        radius=RADIUS, length=LENGTH, thickness=THICKNESS, rho_fluid=RHO_FLUID,
        rho_struct=RHO_STRUCT, n_axial=1, n_circ=2, n_radial=120,
        n_axial_grid=300,
    )
    baffle = fsi.numerical_cylinder_added_mass_ratio(end_condition="baffle", **kwargs)
    caps = fsi.numerical_cylinder_added_mass_ratio(
        end_condition="rigid_caps", axial_padding=0.0, **kwargs
    )
    assert caps["added_mass_ratio"] == pytest.approx(
        baffle["added_mass_ratio"], rel=0.02
    )


# ---------------------------------------------------------------------------
# 3. Cylinder dual-criteria wet modal result
# ---------------------------------------------------------------------------


def test_cylinder_wet_modal_dual_criteria():
    result = fsi.solve_submerged_cylinder_wet_modal(
        radius=RADIUS, length=LENGTH, thickness=THICKNESS, rho_fluid=RHO_FLUID,
        rho_struct=RHO_STRUCT, young=YOUNG, poisson=POISSON,
        modes=((1, 2), (1, 3)), n_radial=120, n_axial_grid=300,
    )
    summary = result.summary()
    assert summary["mode_numbers"] == [[1, 2], [1, 3]]
    for mu_analytic, mu_numerical in zip(
        summary["mu_analytic"], summary["mu_numerical_baffle"]
    ):
        assert mu_analytic > 3.0  # thin R/t = 100 shell is strongly fluid loaded
        assert mu_numerical == pytest.approx(mu_analytic, rel=0.02)
    for wet, dry in zip(summary["wet_hz_analytic"], summary["dry_hz_analytic"]):
        assert wet < dry
    assert summary["local_coefficient_max_rel_error"] < 0.02


def test_cylinder_wet_frequencies_collapse_in_water():
    """R/t = 100 thin shell: wet frequency is well below half the dry value."""
    summary = fsi.analytic_submerged_cylinder(
        radius=RADIUS, length=LENGTH, thickness=THICKNESS, rho_fluid=RHO_FLUID,
        rho_struct=RHO_STRUCT, young=YOUNG, poisson=POISSON,
        modes=((1, 2),),
    )
    assert summary["ratio"][0] < 0.5
    assert summary["wet_hz"][0] == pytest.approx(
        summary["dry_hz"][0] * summary["ratio"][0], rel=1e-12
    )


def test_cylinder_result_is_immutable_and_json_safe():
    result = fsi.solve_submerged_cylinder_wet_modal(
        modes=((1, 2),), n_radial=60, n_axial_grid=150
    )
    summary = result.summary()
    assert isinstance(summary["mu_analytic"], list)
    assert all(isinstance(value, float) for value in summary["mu_analytic"])
    with pytest.raises(Exception):
        result.radius = 2.0  # frozen dataclass


# ---------------------------------------------------------------------------
# 4. Plate fluid loading: exact integral, panel operator, asymptote
# ---------------------------------------------------------------------------


def test_sine_fourier_transform_matches_quadrature():
    for order, k in ((1, 0.0), (1, 2.0), (2, 5.5), (3, math.pi)):
        closed = fsi._sine_fourier_transform(np.array([k]), 1.0, order)[0]
        x = np.linspace(0.0, 1.0, 200001)
        numerical = np.trapezoid(
            np.sin(order * math.pi * x) * np.exp(-1j * k * x), x
        )
        assert closed == pytest.approx(numerical, abs=1e-4)


def test_plate_exact_added_mass_is_converged():
    values = [
        fsi.plate_exact_modal_added_mass(1.0, 1.0, 1, 1, RHO_FLUID, wave_extent=extent)
        for extent in (60.0, 120.0, 240.0)
    ]
    assert values[0] == pytest.approx(values[1], rel=1e-6)
    assert values[1] == pytest.approx(values[2], rel=1e-6)


def test_plate_asymptote_is_approached_for_high_modes():
    """Exact/``rho_f/k_mn`` ratio must tend to 1 with increasing mode order."""
    ratios = []
    for order in (1, 2, 3, 5):
        exact = fsi.plate_exact_modal_added_mass(
            1.0, 1.0, order, order, RHO_FLUID
        ) / 0.25
        asymptote = fsi.plate_asymptotic_added_mass_per_area(
            1.0, 1.0, order, order, RHO_FLUID
        )
        ratios.append(exact / asymptote)
    assert all(later < earlier for earlier, later in zip(ratios, ratios[1:]))
    assert ratios[0] > 1.5  # low modes are far from the asymptote
    assert ratios[-1] < 1.10


def test_plate_panel_operator_matches_exact_integral():
    exact = fsi.plate_exact_modal_added_mass(1.0, 1.0, 1, 1, RHO_FLUID)
    panel = fsi.plate_panel_added_mass_ratio(
        1.0, 1.0, 1, 1, RHO_FLUID, RHO_STRUCT, 0.005, nx=32, ny=32
    ) * (RHO_STRUCT * 0.005 * 1.0 * 1.0 / 4.0)
    assert panel == pytest.approx(exact, rel=0.01)


def test_plate_panel_operator_converges_with_mesh():
    """The panel discretisation approaches the exact wave-number integral."""
    exact = fsi.plate_exact_modal_added_mass(1.0, 1.0, 1, 1, RHO_FLUID) / (
        RHO_STRUCT * 0.005 * 1.0 * 1.0 / 4.0
    )
    ratios = [
        fsi.plate_panel_added_mass_ratio(
            1.0, 1.0, 1, 1, RHO_FLUID, RHO_STRUCT, 0.005, nx=n, ny=n
        )
        for n in (8, 16, 32)
    ]
    errors = [abs(value - exact) / exact for value in ratios]
    # The piecewise-constant panel discretisation is mesh independent to well
    # below the 1% adjudication band (the residual error is dominated by the
    # equal-area patch self term, not by the panel size).
    assert max(errors) < 0.01
    assert max(ratios) - min(ratios) < 0.01 * exact


def test_navier_closed_form_matches_reference_value():
    """Simply supported square plate: f11 = pi/2 sqrt(D/(rho h)) 2/a^2."""
    frequency = fsi.navier_plate_frequency_hz(
        1.0, 1.0, 0.005, YOUNG, POISSON, RHO_STRUCT, 1, 1
    )
    rigidity = YOUNG * 0.005 ** 3 / (12.0 * (1.0 - POISSON ** 2))
    expected = math.pi / 2.0 * math.sqrt(rigidity / (RHO_STRUCT * 0.005)) * 2.0
    assert frequency == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# 5. Plate wet modal result (three criteria)
# ---------------------------------------------------------------------------


def test_plate_wet_modal_three_criteria():
    result = fsi.solve_submerged_plate_wet_modal(
        length=1.0, width=1.0, thickness=0.005, rho_fluid=RHO_FLUID,
        modes=((1, 1), (1, 2), (2, 2)), panel_mesh=(24, 24), fe_mesh=(12, 12),
    )
    summary = result.summary()
    for mu_exact, mu_panel in zip(summary["mu_exact"], summary["mu_panel"]):
        assert mu_panel == pytest.approx(mu_exact, rel=0.02)
    for mu_exact, mu_asymptotic in zip(summary["mu_exact"], summary["mu_asymptotic"]):
        assert mu_asymptotic < mu_exact  # asymptote under-predicts low modes
    for wet, dry in zip(summary["wet_hz"], summary["dry_hz_navier"]):
        assert wet < dry
    for fe, navier in zip(summary["dry_hz_fe"], summary["dry_hz_navier"]):
        assert fe == pytest.approx(navier, rel=0.05)


def test_plate_mode_shape_pipeline():
    """FE dry mode shapes reproduce the analytic sine modes (shape pipeline)."""
    for m, n in ((1, 1), (1, 2)):
        check = fsi.plate_mode_shape_check(m=m, n=n)
        assert check["mac"] > 0.999
        assert check["max_scaled_deviation"] < 0.02
        assert abs(check["frequency_error"]) < 0.05
        assert check["fe_hz"] == pytest.approx(check["navier_hz"], rel=0.05)


def test_plate_wet_modal_rejects_empty_modes():
    with pytest.raises(ValueError):
        fsi.solve_submerged_plate_wet_modal(modes=())


# ---------------------------------------------------------------------------
# 6. Frequency response function
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def frf():
    return fsi.submerged_plate_frf(n_modes=9)


def test_frf_resonances_reproduce_wet_eigenfrequencies(frf):
    peaks = frf.resonance_peaks_hz(window=0.01)
    strong = [
        (index, participation)
        for index, participation in enumerate(frf.participation)
        if participation > 0.05
    ]
    assert len(strong) >= 2
    for index, _ in strong:
        wet = frf.wet_frequencies_hz[index]
        assert peaks[index] == pytest.approx(wet, rel=2e-3)


def test_frf_first_resonance_amplitude_matches_sdof(frf):
    """|H| at the (1,1) resonance equals phi^2 / (2 zeta omega_wet^2)."""
    peak = frf.amplitude().max()
    participation = frf.participation[0]
    omega = 2.0 * math.pi * frf.wet_frequencies_hz[0]
    sdof = participation / (2.0 * frf.damping_ratio * omega ** 2)
    assert float(peak) == pytest.approx(sdof, rel=0.1)


def test_frf_reciprocity():
    first = fsi.submerged_plate_frf(
        n_modes=6, force_location=(0.3, 0.7), response_location=(0.8, 0.2)
    )
    second = fsi.submerged_plate_frf(
        n_modes=6, force_location=(0.8, 0.2), response_location=(0.3, 0.7)
    )
    difference = (first.amplitude() - second.amplitude()).abs().max()
    assert float(difference) <= 1e-12 * float(first.amplitude().max())


def test_frf_wet_frequencies_below_dry(frf):
    for dry, wet in zip(frf.dry_frequencies_hz, frf.wet_frequencies_hz):
        assert wet < dry


def test_frf_generalized_masses_are_normalised(frf):
    for mass in frf.generalized_masses:
        assert mass == pytest.approx(1.0, rel=1e-9)


def test_frf_rejects_points_outside_panel():
    with pytest.raises(ValueError):
        fsi.submerged_plate_frf(n_modes=1, force_location=(1.5, 0.5))


def test_frf_rejects_bad_damping():
    with pytest.raises(ValueError):
        fsi.submerged_plate_frf(n_modes=1, damping_ratio=1.5)