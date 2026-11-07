"""Tests for the panel moment--curvature (Smith) collapse surface."""

from __future__ import annotations

import math

import pytest

from tensorfem.panel_mchi_qualification import (
    LoadShorteningCurve,
    PanelFibre,
    PanelSection,
    axial_ultimate,
    column_euler_fem_check,
    increment_refinement_study,
    moment_curvature,
    plate_beta_fem_check,
    section_from_panel,
)
from tensorfem.stiffened_panel_ultimate import StiffenedPanel


def tee_panel() -> StiffenedPanel:
    return StiffenedPanel(
        span=0.800,
        spacing=0.800,
        plate_thickness=0.012,
        web_height=0.18,
        web_thickness=0.009,
        flange_width=0.080,
        flange_thickness=0.012,
    )


def ice_belt_panel() -> StiffenedPanel:
    return StiffenedPanel(
        span=2.400,
        spacing=0.800,
        plate_thickness=0.020,
        web_height=0.20,
        web_thickness=0.010,
        flange_width=0.080,
        flange_thickness=0.012,
    )


def twin_fibre_section(curve: LoadShorteningCurve) -> PanelSection:
    return PanelSection(
        fibres=(PanelFibre("upper", 0.002, 0.05, curve), PanelFibre("lower", 0.002, -0.05, curve)),
        young=curve.young,
        yield_stress=curve.yield_stress,
        label="twin",
    )


# --------------------------------------------------------------------------
# fibre law
# --------------------------------------------------------------------------
def test_fibre_curve_stress_branches_and_tension_cap():
    yield_stress, young = 235.0e6, 206.0e9
    curve = LoadShorteningCurve(
        young=young,
        yield_stress=yield_stress,
        limit_stress=0.7 * yield_stress,
        softening_strain=2.0 * yield_stress / young,
        residual_ratio=0.6,
    )
    yield_strain = yield_stress / young
    limit_strain = 0.7 * yield_strain
    assert curve.stress(0.5 * yield_strain) == pytest.approx(0.5 * yield_stress)
    # Tension saturates at yield (no hardening model in this module).
    assert curve.stress(5.0 * yield_strain) == pytest.approx(yield_stress)
    # Compression: elastic, softening, residual plateau.
    assert curve.stress(-0.5 * limit_strain) == pytest.approx(-0.5 * 0.7 * yield_stress)
    assert curve.stress(-limit_strain) == pytest.approx(-0.7 * yield_stress)
    midpoint = curve.stress(-(limit_strain + 1.0 * yield_strain))
    assert -0.7 * yield_stress < midpoint < -0.42 * yield_stress
    assert curve.stress(-10.0 * yield_strain) == pytest.approx(-0.42 * yield_stress)
    # A softening branch makes the fibre stress fall again past the limit: the
    # equilibrium solve has to know this, so the shape is asserted, not assumed.
    grid = [(-6.0 + 0.01 * i) * yield_strain for i in range(1201)]
    values = [curve.stress(strain) for strain in grid]
    descending = [
        (grid[i], values[i], values[i + 1])
        for i in range(len(grid) - 1)
        if values[i + 1] < values[i] - 1e-9
    ]
    assert descending, "a softening branch must produce a descending segment"
    assert all(strain < -limit_strain for strain, _a, _b in descending)


def test_plateau_curve_is_monotone_in_strain():
    yield_stress, young = 235.0e6, 206.0e9
    curve = LoadShorteningCurve(
        young=young, yield_stress=yield_stress, limit_stress=0.7 * yield_stress
    )
    yield_strain = yield_stress / young
    grid = [(-6.0 + 0.01 * i) * yield_strain for i in range(1201)]
    values = [curve.stress(strain) for strain in grid]
    assert all(b >= a - 1e-9 for a, b in zip(values, values[1:]))


def test_fibre_curve_energy_matches_numerical_integral():
    yield_stress, young = 235.0e6, 206.0e9
    curve = LoadShorteningCurve(
        young=young,
        yield_stress=yield_stress,
        limit_stress=0.7 * yield_stress,
        softening_strain=1.5 * yield_stress / young,
        residual_ratio=0.55,
    )
    yield_strain = yield_stress / young
    for strain in (-8.0, -3.2, -1.0, -0.4, -0.05, 0.3, 1.4, 4.0):
        target = strain * yield_strain
        samples = 20000
        step = target / samples
        numeric = 0.5 * (curve.stress(0.0) + curve.stress(target)) * step
        numeric += sum(curve.stress(step * i) * step for i in range(1, samples))
        analytic = curve.energy(target)
        assert analytic == pytest.approx(numeric, rel=2e-6), f"strain {strain} eps_y"


def test_fibre_curve_input_validation():
    with pytest.raises(ValueError):
        LoadShorteningCurve(young=206e9, yield_stress=235e6, limit_stress=300e6)
    with pytest.raises(ValueError):
        LoadShorteningCurve(young=206e9, yield_stress=235e6, limit_stress=200e6, residual_ratio=0.5)
    with pytest.raises(ValueError):
        LoadShorteningCurve(
            young=206e9,
            yield_stress=235e6,
            limit_stress=200e6,
            softening_strain=1e-3,
            residual_ratio=1.0,
        )
    with pytest.raises(ValueError):
        LoadShorteningCurve(young=-1.0, yield_stress=235e6, limit_stress=200e6)


# --------------------------------------------------------------------------
# section construction
# --------------------------------------------------------------------------
def test_section_from_panel_preserves_area_and_centroid():
    panel = tee_panel()
    section = section_from_panel(panel, plate_strips=8)
    assert section.total_area == pytest.approx(panel.total_area, rel=1e-12)
    # The lumped offsets reproduce the panel centroid chain.
    assert section.elastic_neutral_axis == pytest.approx(panel.centroid_offset, rel=1e-9)
    # Subdividing the web through its thickness recovers its own bending
    # inertia; the residual ~1 % is the layered approximation of h^3 t / 12.
    assert section.inertia == pytest.approx(panel.column_inertia, rel=2e-2)
    assert len(section.fibres) == 8 + 4 + 2
    modes = {fibre.name: fibre.mode for fibre in section.fibres}
    assert modes["web[0]"] == "column"
    assert modes["flange[0]"] in {"column", "flange_local"}


def test_web_layering_improves_the_lumped_inertia():
    panel = tee_panel()
    single = section_from_panel(panel, web_layers=1, flange_layers=1)
    layered = section_from_panel(panel, web_layers=4, flange_layers=2)
    target = panel.column_inertia
    assert abs(layered.inertia / target - 1.0) < abs(single.inertia / target - 1.0)
    assert abs(layered.inertia / target - 1.0) < 2e-2


def test_section_from_panel_tripping_option_lowers_the_flange_limit():
    panel = tee_panel()
    plain = section_from_panel(panel)
    tripped = section_from_panel(panel, include_tripping=True)
    plain_flange = next(f for f in plain.fibres if f.name == "flange[0]")
    tripped_flange = next(f for f in tripped.fibres if f.name == "flange[0]")
    assert tripped_flange.curve.limit_stress <= plain_flange.curve.limit_stress + 1e-9
    assert (
        tripped_flange.mode == panel.governing_stiffener_mode
        or tripped_flange.mode == "flange_local"
        or tripped_flange.mode == "tripping"
    )


# --------------------------------------------------------------------------
# moment--curvature
# --------------------------------------------------------------------------
def test_elastic_limit_reproduces_section_rigidity():
    section = section_from_panel(tee_panel())
    collapse = moment_curvature(section, steps=12, maximum_strain_factor=0.2)
    for point in collapse.points:
        expected = section.young * section.inertia * point.curvature
        assert point.moment == pytest.approx(expected, rel=1e-6)
        assert abs(point.axial_residual) <= 1e-9 * section.total_area * section.yield_stress


def test_curvature_sign_compresses_the_stiffener_side():
    panel = tee_panel()
    section = section_from_panel(panel)
    collapse = moment_curvature(section, steps=20, maximum_strain_factor=3.0)
    last = collapse.points[-1]
    flange = next(f for f in section.fibres if f.name == "flange[0]")
    flange_strain = last.reference_strain + last.curvature * flange.offset
    assert flange_strain < 0.0
    assert "flange[0]" in last.fibres_at_limit
    assert last.moment > 0.0


def test_energy_ledger_closes_and_refines():
    section = section_from_panel(tee_panel())
    study = increment_refinement_study(section, steps=(100, 200, 400, 800))
    residuals = [row["energy_residual"] for row in study["rows"]]
    assert residuals[0] < 1e-3
    assert residuals[-1] < 1e-5
    assert study["energy_residual_decreasing"] is True
    # Second-order trapezoid convergence: each doubling should cut the residual
    # to well below half of the previous value.
    assert all(second < 0.5 * first for first, second in zip(residuals, residuals[1:]))
    assert study["all_peaks_unique"] is True
    assert study["all_monotonic_to_peak"] is True


def test_plateau_law_saturates_at_the_plastic_moment():
    section = section_from_panel(tee_panel())
    collapse = moment_curvature(section, steps=200)
    plastic = section.fully_plastic_moment()
    assert collapse.ultimate_moment <= plastic * (1.0 + 1e-12)
    assert collapse.ultimate_moment > 0.98 * plastic
    # A limiting-stress plateau cannot produce a limit point; that must be
    # reported rather than dressed up as one.
    assert collapse.saturating is True
    assert collapse.peak_unique is True


def test_softening_branch_gives_unique_limit_point_and_post_peak():
    panel = tee_panel()
    yield_strain = panel.yield_stress / panel.young
    section = section_from_panel(panel, softening_strain=4.0 * yield_strain, residual_ratio=0.70)
    collapse = moment_curvature(section, steps=400)
    assert collapse.peak_unique is True
    assert collapse.monotonic_to_peak is True
    assert collapse.post_peak is True
    assert collapse.saturating is False
    assert collapse.ultimate_moment < section.fully_plastic_moment()
    # Bounded below by the moment at first fibre yield: the section cannot be
    # weaker than the elastic path up to the first limit.
    axis = section.elastic_neutral_axis
    extent = max(abs(fibre.offset - axis) for fibre in section.fibres)
    first_limit_strain = min(fibre.curve.limit_strain for fibre in section.fibres)
    assert collapse.ultimate_moment > section.young * section.inertia * (
        first_limit_strain / extent
    )
    assert collapse.energy_residual < 1e-4
    # Monotone moment after the peak (no secondary branch above the first peak).
    tail = [p.moment for p in collapse.points[collapse.ultimate_index :]]
    assert all(b <= a + 1e-9 * abs(a) for a, b in zip(tail, tail[1:]))


def test_softening_keeps_ultimate_below_the_plateau_value():
    panel = tee_panel()
    yield_strain = panel.yield_stress / panel.young
    plateau = moment_curvature(section_from_panel(panel), steps=400)
    softened = moment_curvature(
        section_from_panel(panel, softening_strain=4.0 * yield_strain, residual_ratio=0.70),
        steps=400,
    )
    assert softened.ultimate_moment < plateau.ultimate_moment


def test_axial_ultimate_matches_components_at_ultimate():
    for panel in (tee_panel(), ice_belt_panel()):
        section = section_from_panel(panel)
        result = axial_ultimate(section, steps=2000)
        assert result["ultimate_over_yield"] == pytest.approx(
            panel.panel_ultimate_stress / panel.yield_stress, rel=1e-6
        )
        assert result["ultimate_over_yield"] < 1.0


def test_moment_curvature_input_validation():
    section = section_from_panel(tee_panel())
    with pytest.raises(ValueError):
        moment_curvature(section, steps=4)
    with pytest.raises(ValueError):
        moment_curvature(section, steps=100, maximum_strain_factor=0.0)
    with pytest.raises(ValueError):
        moment_curvature(section, steps=100, axial_force=1.0e9)
    with pytest.raises(ValueError):
        increment_refinement_study(section, steps=(100,))


# --------------------------------------------------------------------------
# cross-verification against TensorFEM discretisations
# --------------------------------------------------------------------------
def test_plate_eigenvalue_converges_to_the_k4_anchor():
    result = plate_beta_fem_check(tee_panel())
    coefficients = [row["buckling_coefficient_fe"] for row in result["mesh_train"]]
    errors = [row["coefficient_error_vs_k4"] for row in result["mesh_train"]]
    assert coefficients[-1] == pytest.approx(4.0, rel=5e-3)
    assert all(second < first for first, second in zip(errors, errors[1:]))
    assert result["finest_coefficient_error_vs_k4"] < 5e-3


def test_column_eigenvalue_reproduces_euler_and_is_an_upper_bound():
    result = column_euler_fem_check(tee_panel(), elements=(8, 16, 32, 64))
    train = result["mesh_train"]
    assert train[-1]["relative_error"] < 1e-3
    assert all(
        second["relative_error"] < first["relative_error"]
        for first, second in zip(train, train[1:])
    )
    # A Rayleigh-quotient estimate converges from above.
    assert all(
        row["fe_critical_load"] >= row["euler_critical_load"] * (1.0 - 1e-12) for row in train
    )
    panel = tee_panel()
    jo = result["johnson_ostenfeld_stress"]
    assert 0.0 < jo <= panel.yield_stress
    # Johnson--Ostenfeld is never above yield and never below the lesser of the
    # Euler stress and half the yield stress (its two branches).
    euler = result["column_euler_stress"]
    assert min(euler, 0.5 * panel.yield_stress) - 1e-6 <= jo <= panel.yield_stress + 1e-6


@pytest.mark.slow
def test_fe_panel_compression_path_smoke():
    from tensorfem.panel_mchi_qualification import fe_panel_compression_path

    panel = tee_panel()
    result = fe_panel_compression_path(panel, nx=2, ny=1, layers=1, steps=4, include_points=False)
    assert result["converged"] is True
    assert result["points_completed"] == 4
    assert result["free_dofs"] > 0
    assert result["element_counts"] == [2, 2, 2]
    assert result["ultimate_over_yield"] > 0.0
    assert math.isfinite(result["ultimate_stress"])
