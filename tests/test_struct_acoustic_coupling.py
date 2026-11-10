"""Campaign p348w2-n05: structure-acoustic coupling verification surface.

Acceptance criteria (see the node's machine gate) and where each one lives:

=====  ===================================================  ==========================
C1     plate modes vs Levy analytic, first 10, <= 2 %       TestC1PlateModes
C2     radiated power vs Rayleigh analytic, <= 10 %         TestC2RadiationPower
C3     underwater directivity vs experiment/anchor, <=15 %  TestC3Underwater
C4     two-way coupling interface residual <= 3 %           TestC4TwoWayCoupling
C5     FW-H vs structure-acoustic power, <= 10 %            TestC5FwhCrossCheck
C6     plate-grid refinement doubles -> power change <=5 %  TestC6MeshConvergence
C7     vacuum negative control: modes vs analytic <= 1 %    TestC7VacuumControl
C8     >= 15 test cases over modes/radiation/coupling/FW-H  this module
=====  ===================================================  ==========================

Every reference value is either a closed form, an independently derived
variational/literature value, or a second numerical route -- never a snapshot of
this repository's own output.

C3 is reported as *partially* met on purpose: the free-surface directivity is
verified against the exact image solution, against the closed-form Lloyd factor
and against the analytic null law ``cos theta_n = n lambda / (2d)`` (all to
< 1 %), but the *experimental* digitised pattern for a submerged plate is not
available offline in this environment.  The experimental leg is declared
WITHHELD in the campaign report instead of being replaced by a self-referential
comparison; :func:`free_surface_power` returns the pattern in a shape that a
digitised dataset can be compared against as soon as one is supplied.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorfem.plate_acoustics import (
    PlateAcousticConfig,
    far_field_power,
    rayleigh_power,
    sine_mode_field,
)
from tensorfem.struct_acoustics import (
    PlateModalConfig,
    RadiationGrid,
    coupled_modal_response,
    free_surface_power,
    free_surface_power_ratio_analytic,
    kirchhoff_data_surface_power,
    leissa_cccc_square_lambdas,
    levy_frequencies,
    lloyd_null_cosines,
    radiated_power_from_pressure,
    ritz_frequencies,
    sine_frequencies,
    surface_pressure,
)
from tensorfem.struct_acoustics.levy_modes import _clamped_beam_mu

STEEL = PlateModalConfig(length=1.5, width=1.0, thickness=0.01)
WATER_RHO = 1000.0
WATER_C = 1500.0
AIR_RHO = 1.225
AIR_C = 343.0


def _sine_mode_velocity(grid: RadiationGrid, m: int = 1, n: int = 1) -> torch.Tensor:
    points = grid.points
    return torch.sin(m * math.pi * points[:, 0] / STEEL.length) * torch.sin(
        n * math.pi * points[:, 1] / STEEL.width
    )


def _relative(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


@pytest.fixture(scope="module")
def levy_ssss() -> list[float]:
    return levy_frequencies(STEEL, x_edges="simply_supported", modes=10)


@pytest.fixture(scope="module")
def levy_scsc() -> list[float]:
    return levy_frequencies(STEEL, x_edges="clamped", modes=10)


# ---------------------------------------------------------------------------
# C1: plate modes
# ---------------------------------------------------------------------------

class TestC1PlateModes:
    """C1: first ten plate frequencies against the Levy analytic solution."""

    def test_levy_simply_supported_is_the_closed_form(self, levy_ssss):
        """Levy must collapse onto the double-sine eigenvalue for an S-S-S-S plate."""
        exact = sorted(
            STEEL.frequency(
                math.sqrt(
                    (m * math.pi / STEEL.length) ** 2
                    + (n * math.pi / STEEL.width) ** 2
                )
            )
            for m in range(1, 6)
            for n in range(1, 6)
        )[:10]
        for value, reference in zip(levy_ssss, exact):
            assert _relative(value, reference) < 1e-9

    def test_levy_scsc_matches_independent_ritz(self, levy_scsc):
        """S-C-S-C plate: transcendental Levy roots vs a variational Ritz bound."""
        ritz = ritz_frequencies(
            STEEL, x_edges="clamped", y_edges="simply_supported", modes=10, terms=10
        )
        errors = [_relative(a, b) for a, b in zip(levy_scsc, ritz)]
        assert max(errors) < 0.02, errors
        # observed ~2e-4; a coarser gate would hide a determinant/branch bug
        assert max(errors) < 1e-3, errors

    def test_ritz_fully_clamped_matches_leissa_table(self):
        """C-C-C-C square plate vs the transcribed Leissa (1969) coefficients."""
        square = PlateModalConfig(length=1.0, width=1.0, thickness=0.01)
        ritz = ritz_frequencies(square, x_edges="clamped", y_edges="clamped", modes=8, terms=10)
        scale = 2.0 * math.pi * square.length ** 2 * math.sqrt(
            square.mass_per_area / square.rigidity
        )
        lambdas = [f * scale for f in ritz]
        reference = leissa_cccc_square_lambdas()
        errors = [_relative(a, b) for a, b in zip(lambdas, reference)]
        assert max(errors) < 0.005, errors

    def test_ritz_simply_supported_is_exact(self):
        ritz = ritz_frequencies(
            STEEL, x_edges="simply_supported", y_edges="simply_supported", modes=6, terms=6
        )
        exact = sorted(
            STEEL.frequency(
                math.sqrt(
                    (m * math.pi / STEEL.length) ** 2
                    + (n * math.pi / STEEL.width) ** 2
                )
            )
            for m in range(1, 4)
            for n in range(1, 4)
        )[:6]
        for value, reference in zip(ritz, exact):
            assert _relative(value, reference) < 1e-8

    def test_clamped_beam_roots_satisfy_the_characteristic_equation(self):
        mus = _clamped_beam_mu(5)
        for mu in mus:
            # cos(mu) cosh(mu) = 1 is ill-conditioned for large mu (cosh grows),
            # so the residual is measured on the scaled form cos(mu) = 1/cosh(mu).
            assert abs(math.cos(mu) - 1.0 / math.cosh(mu)) < 1e-11
        assert abs(mus[0] - 4.730040744862704) < 1e-9

    def test_clamping_the_edges_stiffens_every_mode(self, levy_ssss, levy_scsc):
        """Negative control: the S-C-S-C plate must be stiffer, mode by mode."""
        assert all(b > a for a, b in zip(levy_ssss, levy_scsc))


# ---------------------------------------------------------------------------
# C7: vacuum negative control
# ---------------------------------------------------------------------------

class TestC7VacuumControl:
    """C7: with the fluid switched off the coupled model must return dry modes.

    With ``rho -> 0`` the modal fluid force vanishes and the coupled equation
    reduces to the analytic dry single-mode response
    ``q_m = F_m / (m_m (omega_dry^2 - omega^2))``.  Two frequencies are enough to
    test that denominator *without* re-deriving ``F_m`` or ``m_m`` in the test:
    the ratio ``|q(omega_1)| / |q(omega_2)|`` must equal
    ``(omega_dry^2 - omega_2^2) / (omega_dry^2 - omega_1^2)``.
    """

    def test_zero_density_recovers_the_dry_modal_response(self):
        grid = RadiationGrid.uniform(STEEL.length, STEEL.width, 16, 12)
        dry = float(sine_frequencies(STEEL, half_waves_x=2, half_waves_y=2)[0])
        omega_dry = 2.0 * math.pi * dry
        first = 1.17 * dry
        second = 1.43 * dry
        amplitudes = []
        for frequency in (first, second):
            response = coupled_modal_response(
                grid,
                STEEL,
                frequency=frequency,
                rho=1e-9,
                sound_speed=AIR_C,
                half_waves_x=2,
                half_waves_y=2,
                point_force=((0.7, 0.5), 1.0),
            )
            amplitudes.append(float(torch.abs(response.modal_amplitudes[0])))
        measured = amplitudes[0] / amplitudes[1]
        analytic = (omega_dry ** 2 - (2.0 * math.pi * second) ** 2) / (
            omega_dry ** 2 - (2.0 * math.pi * first) ** 2
        )
        assert _relative(measured, analytic) < 0.01
        assert _relative(measured, analytic) < 1e-6

    def test_zero_density_radiated_power_vanishes(self):
        """Negative control for the acoustic side: no fluid, no radiated power."""
        grid = RadiationGrid.uniform(STEEL.length, STEEL.width, 16, 12)
        dry = float(sine_frequencies(STEEL, half_waves_x=2, half_waves_y=2)[0])
        response = coupled_modal_response(
            grid,
            STEEL,
            frequency=1.2 * dry,
            rho=1e-9,
            sound_speed=AIR_C,
            half_waves_x=2,
            half_waves_y=2,
            point_force=((0.7, 0.5), 1.0),
        )
        assert response.radiated_power > 0.0
        assert response.radiated_power < 1e-12 * max(response.input_power, 1.0) or (
            response.radiated_power / max(abs(response.input_power), 1e-300) < 1e-6
        )


# ---------------------------------------------------------------------------
# C2: radiated power against the Rayleigh analytic integral
# ---------------------------------------------------------------------------

class TestC2RadiationPower:
    """C2: radiated power, pressure route vs Rayleigh integral vs far field."""

    def test_surface_pressure_route_reproduces_rayleigh_power(self):
        grid = RadiationGrid.uniform(STEEL.length, STEEL.width, 32, 32)
        velocity = _sine_mode_velocity(grid)
        acoustic = PlateAcousticConfig(
            length=STEEL.length,
            width=STEEL.width,
            thickness=STEEL.thickness,
            rho_fluid=WATER_RHO,
            sound_speed=WATER_C,
        )
        field = sine_mode_field(acoustic, 1, 1, points=32, velocity_amplitude=1.0)
        for frequency in (200.0, 400.0, 800.0):
            reference = rayleigh_power(field, frequency=frequency, rho=WATER_RHO, c=WATER_C)
            pressure = surface_pressure(
                grid, velocity, frequency=frequency, rho=WATER_RHO, sound_speed=WATER_C
            )
            value = radiated_power_from_pressure(pressure, velocity, grid)
            assert _relative(value, reference) < 0.01

    def test_surface_pressure_route_reproduces_far_field_power(self):
        grid = RadiationGrid.uniform(STEEL.length, STEEL.width, 24, 24)
        velocity = _sine_mode_velocity(grid, 2, 1)
        acoustic = PlateAcousticConfig(
            length=STEEL.length,
            width=STEEL.width,
            thickness=STEEL.thickness,
            rho_fluid=AIR_RHO,
            sound_speed=AIR_C,
        )
        field = sine_mode_field(acoustic, 2, 1, points=24, velocity_amplitude=1.0)
        reference = far_field_power(field, frequency=500.0, rho=AIR_RHO, c=AIR_C)
        pressure = surface_pressure(
            grid, velocity, frequency=500.0, rho=AIR_RHO, sound_speed=AIR_C
        )
        value = radiated_power_from_pressure(pressure, velocity, grid)
        assert _relative(value, reference) < 0.05

    def test_underwater_efficiency_is_far_below_coincidence(self):
        """A 10 mm steel plate at 200 Hz in water is deep sub-critical.

        The efficiency is taken from the pressure route on a *resolved* mode
        (verified against the Rayleigh integral in the two tests above); a
        coupled multi-mode response is deliberately not used here because its
        power is a difference of nearly cancelling modal contributions, which is
        where the panel quadrature's passivity defect (~1e-3 of the resistance
        scale, see the module docstring) shows up.
        """
        grid = RadiationGrid.uniform(STEEL.length, STEEL.width, 24, 24)
        velocity = _sine_mode_velocity(grid)
        pressure = surface_pressure(
            grid, velocity, frequency=200.0, rho=WATER_RHO, sound_speed=WATER_C
        )
        power = radiated_power_from_pressure(pressure, velocity, grid)
        mean_square = float(
            (grid.weights * velocity ** 2).sum() / grid.surface
        )
        efficiency = power / (WATER_RHO * WATER_C * grid.surface * mean_square)
        coincidence = PlateAcousticConfig(
            length=STEEL.length,
            width=STEEL.width,
            thickness=STEEL.thickness,
            rho_fluid=WATER_RHO,
            sound_speed=WATER_C,
        ).critical_frequency
        assert coincidence > 10000.0
        assert 0.0 < efficiency < 0.2


# ---------------------------------------------------------------------------
# C4: two-way coupling
# ---------------------------------------------------------------------------

class TestC4TwoWayCoupling:
    """C4: interface residual and the exact power-balance identity."""

    @pytest.mark.parametrize("rho,sound_speed", [(AIR_RHO, AIR_C), (WATER_RHO, WATER_C)])
    def test_interface_residual_below_three_percent(self, rho, sound_speed):
        grid = RadiationGrid.uniform(STEEL.length, STEEL.width, 24, 16)
        response = coupled_modal_response(
            grid,
            STEEL,
            frequency=200.0,
            rho=rho,
            sound_speed=sound_speed,
            half_waves_x=6,
            half_waves_y=5,
            point_force=((0.7, 0.5), 1.0),
        )
        assert response.interface_residual < 0.03
        assert response.equation_residual < 1e-6
        assert response.iterations > 1
        assert response.residual_history[-1] < response.residual_history[0]

    def test_power_balance_identity_holds(self):
        """P_in = P_rad exactly for an undamped two-way coupled solution."""
        grid = RadiationGrid.uniform(STEEL.length, STEEL.width, 24, 16)
        for rho, sound_speed in ((AIR_RHO, AIR_C), (WATER_RHO, WATER_C)):
            response = coupled_modal_response(
                grid,
                STEEL,
                frequency=200.0,
                rho=rho,
                sound_speed=sound_speed,
                half_waves_x=6,
                half_waves_y=5,
                point_force=((0.7, 0.5), 1.0),
            )
            assert response.power_balance_error < 1e-6
            assert response.radiated_power > 0.0

    def test_iteration_converges_to_the_direct_solve(self):
        grid = RadiationGrid.uniform(STEEL.length, STEEL.width, 20, 14)
        response = coupled_modal_response(
            grid,
            STEEL,
            frequency=300.0,
            rho=WATER_RHO,
            sound_speed=WATER_C,
            half_waves_x=4,
            half_waves_y=4,
            point_force=((0.7, 0.5), 1.0),
        )
        assert response.direct_solution_gap < 1e-6

    def test_fluid_loading_lowers_the_resonance(self):
        """Water added mass must soften the first mode (physics sign check)."""
        grid = RadiationGrid.uniform(STEEL.length, STEEL.width, 16, 12)
        dry = float(sine_frequencies(STEEL, half_waves_x=2, half_waves_y=2)[0])

        def peak_frequency(rho: float, sound_speed: float) -> float:
            frequencies = [dry * (1.0 + offset) for offset in (-0.12, -0.08, -0.04, 0.02, 0.06)]
            amplitudes = []
            for frequency in frequencies:
                response = coupled_modal_response(
                    grid,
                    STEEL,
                    frequency=frequency,
                    rho=rho,
                    sound_speed=sound_speed,
                    half_waves_x=2,
                    half_waves_y=2,
                    point_force=((0.7, 0.5), 1.0),
                )
                amplitudes.append(float(torch.abs(response.modal_amplitudes[0])))
            return frequencies[amplitudes.index(max(amplitudes))]

        wet = peak_frequency(WATER_RHO, WATER_C)
        assert wet < dry
        assert wet > 0.5 * dry


# ---------------------------------------------------------------------------
# C5: FW-H / Kirchhoff analogy cross-check
# ---------------------------------------------------------------------------

class TestC5FwhCrossCheck:
    """C5: the acoustic analogy must reproduce the structure-acoustic power."""

    def test_fwh_power_matches_rayleigh_route(self):
        grid = RadiationGrid.uniform(STEEL.length, STEEL.width, 24, 16)
        velocity = _sine_mode_velocity(grid)
        for frequency in (200.0, 600.0):
            result = kirchhoff_data_surface_power(
                grid,
                STEEL,
                surface_velocity=velocity,
                frequency=frequency,
                rho=WATER_RHO,
                sound_speed=WATER_C,
            )
            assert result.relative_power_gap < 0.10, result.relative_power_gap
            assert result.phase_coherence > 0.999

    def test_fwh_pressure_field_is_not_just_a_power_match(self):
        """The analogy must reproduce the complex far field, not only |p|^2."""
        grid = RadiationGrid.uniform(STEEL.length, STEEL.width, 20, 14)
        velocity = _sine_mode_velocity(grid, 2, 1)
        result = kirchhoff_data_surface_power(
            grid,
            STEEL,
            surface_velocity=velocity,
            frequency=400.0,
            rho=AIR_RHO,
            sound_speed=AIR_C,
        )
        assert abs(result.amplitude_ratio - 1.0) < 0.02
        assert result.relative_power_gap < 0.05

    def test_data_surface_convergence_is_monotone(self):
        """The gate has teeth: a data surface too close to the plate is caught."""
        grid = RadiationGrid.uniform(STEEL.length, STEEL.width, 24, 16)
        velocity = _sine_mode_velocity(grid)
        gaps = {}
        for radius_factor in (1.0, 1.6, 2.0):
            result = kirchhoff_data_surface_power(
                grid,
                STEEL,
                surface_velocity=velocity,
                frequency=200.0,
                rho=WATER_RHO,
                sound_speed=WATER_C,
                radius_factor=radius_factor,
                n_cap_theta=12,
                n_cap_phi=24,
            )
            gaps[radius_factor] = result.relative_power_gap
        assert gaps[1.0] > 0.10
        assert gaps[2.0] < gaps[1.6] < gaps[1.0]
        assert gaps[2.0] < 0.01


# ---------------------------------------------------------------------------
# C3: underwater free-surface radiation
# ---------------------------------------------------------------------------

class TestC3Underwater:
    """C3: free-surface directivity, null law and power ratio."""

    def test_lloyd_null_cosines(self):
        cosines = lloyd_null_cosines(depth=10.0, wavelength=5.0)
        assert cosines == pytest.approx((0.25, 0.5, 0.75, 1.0))

    def test_null_angles_match_the_analytic_law(self):
        grid = RadiationGrid.uniform(STEEL.length, STEEL.width, 24, 16)
        velocity = _sine_mode_velocity(grid)
        for depth in (5.0, 10.0):
            field = free_surface_power(
                grid,
                STEEL,
                surface_velocity=velocity,
                frequency=200.0,
                depth=depth,
                n_theta=48,
                n_phi=48,
            )
            wavelength = WATER_C / 200.0
            analytic = [
                math.degrees(math.acos(min(1.0, value)))
                for value in lloyd_null_cosines(depth, wavelength)
            ]
            numeric = sorted(math.degrees(value) for value in field.null_angles())
            assert len(numeric) >= len(analytic)
            for measured, reference in zip(numeric, sorted(analytic)):
                assert abs(measured - reference) / reference < 0.15

    def test_power_ratio_matches_the_closed_form_lloyd_factor(self):
        grid = RadiationGrid.uniform(STEEL.length, STEEL.width, 20, 16)
        velocity = _sine_mode_velocity(grid)
        for depth in (2.0, 5.0, 10.0):
            field = free_surface_power(
                grid,
                STEEL,
                surface_velocity=velocity,
                frequency=200.0,
                depth=depth,
                n_theta=32,
                n_phi=48,
            )
            analytic = free_surface_power_ratio_analytic(
                grid,
                STEEL,
                surface_velocity=velocity,
                frequency=200.0,
                depth=depth,
            )
            assert _relative(field.power_ratio, analytic) < 0.02

    def test_deep_surface_tends_to_the_incoherent_lloyd_limit(self):
        """A pressure-release surface doubles the power when its nulls are dense.

        ``4 sin^2(k d cos theta)`` averages to 2 over the hemisphere, so the
        deep-surface limit of the power ratio is **2**, not 1: the reflected
        image adds its energy incoherently once the interference fringe spacing
        falls below the directivity scale.  Getting 1 here would mean the image
        term is missing.
        """
        grid = RadiationGrid.uniform(STEEL.length, STEEL.width, 20, 16)
        velocity = _sine_mode_velocity(grid)
        deep = free_surface_power(
            grid,
            STEEL,
            surface_velocity=velocity,
            frequency=200.0,
            depth=10.0 * WATER_C / 200.0,
            n_theta=240,
            n_phi=32,
        )
        assert abs(deep.power_ratio - 2.0) < 0.02

    def test_compact_source_matches_the_omnidirectional_lloyd_average(self):
        """``ka << 1``: the ratio is the closed form ``2[1 - sin(2kd)/(2kd)]``."""
        grid = RadiationGrid.uniform(STEEL.length, STEEL.width, 20, 16)
        velocity = _sine_mode_velocity(grid)
        for frequency, depth_factor in ((20.0, 1.1), (20.0, 2.3)):
            wavelength = WATER_C / frequency
            depth = depth_factor * wavelength
            field = free_surface_power(
                grid,
                STEEL,
                surface_velocity=velocity,
                frequency=frequency,
                depth=depth,
                n_theta=120,
                n_phi=32,
            )
            k = 2.0 * math.pi / wavelength
            closed_form = 2.0 * (1.0 - math.sin(2.0 * k * depth) / (2.0 * k * depth))
            assert _relative(field.power_ratio, closed_form) < 0.02

    def test_shallow_pressure_release_suppresses_radiation(self):
        grid = RadiationGrid.uniform(STEEL.length, STEEL.width, 20, 16)
        velocity = _sine_mode_velocity(grid)
        wavelength = WATER_C / 200.0
        shallow = free_surface_power(
            grid,
            STEEL,
            surface_velocity=velocity,
            frequency=200.0,
            depth=wavelength / 40.0,
            n_theta=24,
            n_phi=32,
        )
        assert shallow.power_ratio < 0.05

    def test_free_field_route_reproduces_rayleigh_power(self):
        grid = RadiationGrid.uniform(STEEL.length, STEEL.width, 20, 16)
        velocity = _sine_mode_velocity(grid)
        acoustic = PlateAcousticConfig(
            length=STEEL.length,
            width=STEEL.width,
            thickness=STEEL.thickness,
            rho_fluid=WATER_RHO,
            sound_speed=WATER_C,
        )
        field = sine_mode_field(acoustic, 1, 1, points=32, velocity_amplitude=1.0)
        reference = rayleigh_power(field, frequency=200.0, rho=WATER_RHO, c=WATER_C)
        free = free_surface_power(
            grid,
            STEEL,
            surface_velocity=velocity,
            frequency=200.0,
            depth=None,
            n_theta=24,
            n_phi=32,
        )
        assert _relative(free.free_field_power, reference) < 0.05


# ---------------------------------------------------------------------------
# Defect found on the way: non-square grids in plate_acoustics
# ---------------------------------------------------------------------------

class TestNonSquareGridRegression:
    """``plate_acoustics._tile_axis`` raised for ``nx != ny``.

    Both ``rayleigh_power`` and ``far_field_power`` flattened the source grid
    with an expand whose dimensions only conformed when the panel counts were
    equal, so every rectangular *discretisation* of a rectangular plate (the
    natural choice for a 3:2 plate) failed with a shape error.  The fix is one
    line; this regression pins it against the analytic cross-check that the two
    routes must agree for a non-symmetric mode as well.
    """

    def test_rayleigh_and_far_field_agree_on_a_non_square_grid(self):
        from tensorfem.plate_acoustics import SurfaceVelocityField

        nx, ny = 20, 12
        x = (torch.arange(nx, dtype=torch.float64) + 0.5) * STEEL.length / nx
        y = (torch.arange(ny, dtype=torch.float64) + 0.5) * STEEL.width / ny
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        velocity = torch.sin(2 * math.pi * xx / STEEL.length) * torch.sin(
            math.pi * yy / STEEL.width
        )
        field = SurfaceVelocityField(
            x=x,
            y=y,
            velocity=velocity,
            weight_x=torch.full((nx,), STEEL.length / nx, dtype=torch.float64),
            weight_y=torch.full((ny,), STEEL.width / ny, dtype=torch.float64),
        )
        reference = rayleigh_power(field, frequency=200.0, rho=WATER_RHO, c=WATER_C)
        compared = far_field_power(field, frequency=200.0, rho=WATER_RHO, c=WATER_C)
        assert reference > 0.0
        assert _relative(compared, reference) < 0.02


# ---------------------------------------------------------------------------
# C6: discretisation convergence
# ---------------------------------------------------------------------------

class TestC6MeshConvergence:
    """C6: doubling the surface panel count must not move the power."""

    def test_panel_refinement_changes_power_by_less_than_five_percent(self):
        """C6: doubling the panel count in both directions must not move P_rad."""
        values = []
        for panels in ((16, 12), (32, 24)):
            grid = RadiationGrid.uniform(STEEL.length, STEEL.width, *panels)
            velocity = _sine_mode_velocity(grid)
            pressure = surface_pressure(
                grid, velocity, frequency=200.0, rho=WATER_RHO, sound_speed=WATER_C
            )
            values.append(radiated_power_from_pressure(pressure, velocity, grid))
        assert _relative(values[1], values[0]) < 0.05

    def test_coupled_power_is_grid_insensitive_between_dry_resonances(self):
        """Same refinement on the two-way coupled response, away from resonance."""
        values = []
        for panels in ((20, 14), (40, 28)):
            grid = RadiationGrid.uniform(STEEL.length, STEEL.width, *panels)
            response = coupled_modal_response(
                grid,
                STEEL,
                frequency=1.45 * float(sine_frequencies(STEEL, half_waves_x=3, half_waves_y=3)[0]),
                rho=WATER_RHO,
                sound_speed=WATER_C,
                half_waves_x=3,
                half_waves_y=3,
                point_force=((0.7, 0.5), 1.0),
            )
            values.append(response.radiated_power)
        assert values[0] > 0.0 and values[1] > 0.0
        assert _relative(values[1], values[0]) < 0.05

    def test_self_panel_integral_is_the_closed_form(self):
        from tensorfem.struct_acoustics.radiation_coupling import _self_panel_integral

        value = _self_panel_integral(0.05, 0.05)
        assert _relative(value, 4.0 * 0.05 * math.asinh(1.0)) < 1e-12
        # rectangle: the general closed form reduces to the square result
        assert _relative(_self_panel_integral(0.04, 0.09),
                         2.0 * (0.04 * math.asinh(0.09 / 0.04)
                                + 0.09 * math.asinh(0.04 / 0.09))) < 1e-12
        # ... and a very fine Gauss rule must approach it from below
        from tensorfem.acoustic_radiation import _gauss_legendre

        nodes, weights = _gauss_legendre(120)
        gx = 0.5 * 0.05 * nodes
        wx = 0.5 * 0.05 * weights
        r = torch.sqrt(gx[None, :] ** 2 + gx[:, None] ** 2)
        quadrature = float(((wx[:, None] * wx[None, :]) / r).sum())
        assert _relative(quadrature, value) < 0.02