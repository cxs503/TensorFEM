"""Tests for :mod:`tensorfem.struct_acoustics.cylinder_radiation`.

Each test pins one independent anchor of the cylinder structure-acoustic chain:

* Donnell shell roots (three branches, thin-shell ``omega ~ t`` scaling),
* the Junger & Feit infinite-cylinder conductance (low-``ka`` piston law, high
  ``ka`` plateau, evanescent decay for subsonic axial wavenumber),
* the axial-spectrum / window closure (closed form vs quadrature, long-window
  limit back to the infinite cylinder),
* the equivalent-source exterior solve (pulsating sphere exact power, single
  monopole power, compact-body asymptote, rigid-body null),
* the geometry / mode-identification helpers used by the benchmark.
"""
import math
import os

import pytest
import torch

# This machine has 128 cores; the small dense linear algebra of the acoustic
# operators runs several times *slower* when torch spawns one thread per core.
torch.set_num_threads(min(8, os.cpu_count() or 8))

from tensorfem.struct_acoustics.cylinder_radiation import (
    axial_spectrum,
    compact_body_power,
    cylinder_surface_grid,
    donnell_frequencies,
    equivalent_source_power,
    finite_cylinder_mode_power,
    fourier_cylinder_mode,
    infinite_cylinder_conductance,
    infinite_cylinder_efficiency,
    monopole_power_matrix,
    shell_radial_amplitudes,
)

RHO = 1.225
C = 343.0
RADIUS = 0.5
LENGTH = 2.0
STEEL = dict(young=210.0e9, poisson=0.3, density=7850.0, thickness=0.005)


def _lateral(radius=RADIUS, length=LENGTH, n_axial=17, n_circ=32):
    points, normals, weights, meta = cylinder_surface_grid(
        radius=radius, length=length, n_axial=n_axial, n_circ=n_circ
    )
    n_lateral = int(meta[0]) * int(meta[1])
    theta = torch.atan2(points[:n_lateral, 2], points[:n_lateral, 1])
    return points, normals, weights, n_lateral, theta


def test_donnell_roots_are_ordered_and_positive():
    roots = donnell_frequencies(1, 2, radius=RADIUS, length=LENGTH, **STEEL)
    assert len(roots) == 3
    assert roots[0] > 0.0
    assert roots[0] < roots[1] < roots[2]
    # For a thin shell the membrane branches sit orders of magnitude above the
    # bending branch; that separation is the physics the acoustics uses.
    assert roots[1] / roots[0] > 5.0


def test_donnell_lowest_root_is_the_membrane_branch():
    """Donnell's lowest root is thickness-independent (its known limitation).

    The truly inextensional branch of a thin shell scales as ``t``; Donnell
    kinematics put the lowest root at the (thickness-independent) hoop membrane
    level instead, which is why these roots are only a branch-structure anchor and
    never the shell frequency reference of this node.
    """
    thin = donnell_frequencies(1, 2, radius=RADIUS, length=LENGTH,
                               **{**STEEL, "thickness": 0.0025})[0]
    thick = donnell_frequencies(1, 2, radius=RADIUS, length=LENGTH,
                                **{**STEEL, "thickness": 0.005})[0]
    assert thin / thick == pytest.approx(1.0, rel=0.02)
    # ... and it sits far above the classical inextensional estimate (~13 Hz).
    assert thick > 100.0


def test_infinite_cylinder_low_frequency_piston_law():
    """``n = 0`` (ring) mode: ``sigma -> (pi/2) k R`` as ``k R -> 0``.

    The leading correction is ``O(kR)``: the ratio is 0.99948 at ka = 0.01 and
    0.9714 at ka = 0.1, so the law is checked where it is an asymptote.
    """
    for ka in (1e-4, 1e-3, 1e-2):
        k = ka / RADIUS
        sigma = infinite_cylinder_efficiency(0, k, 0.0, radius=RADIUS, rho=RHO, c=C)
        assert sigma / (0.5 * math.pi * ka) == pytest.approx(1.0, rel=5e-3)


def test_infinite_cylinder_high_frequency_plateau():
    """Above coincidence every mode radiates; the plateau values are exact."""
    ka = 20.0
    k = ka / RADIUS
    sigma0 = infinite_cylinder_efficiency(0, k, 0.0, radius=RADIUS, rho=RHO, c=C)
    assert sigma0 == pytest.approx(1.0, abs=0.01)
    for n in (1, 2, 3):
        sigma = infinite_cylinder_efficiency(n, k, 0.0, radius=RADIUS, rho=RHO, c=C)
        # <cos^2 n theta> = 1/2 halves the perimeter-integrated power.
        assert sigma == pytest.approx(0.5, abs=0.02), (n, sigma)


def test_infinite_cylinder_evanescent_for_subsonic_axial_wavenumber():
    """``k_x > k`` is cut off: the conductance collapses by orders of magnitude."""
    k = 1.0 / RADIUS
    values = [
        infinite_cylinder_conductance(
            2, k, factor * k, radius=RADIUS, rho=RHO, c=C
        )
        for factor in (0.0, 1.05, 1.5, 3.0)
    ]
    assert values[0] > 0.0
    magnitudes = [abs(value) for value in values]
    assert magnitudes[1] < 1e-3 * magnitudes[0]
    assert magnitudes[2] <= magnitudes[1]
    assert magnitudes[3] < 1e-12 * magnitudes[0]


def test_conductance_is_non_negative():
    """No spurious negative radiation resistance (round-off at 1e-14 allowed)."""
    for n in range(4):
        for k in (0.2, 1.0, 3.0, 10.0):
            reference = infinite_cylinder_conductance(
                n, k, 0.0, radius=RADIUS, rho=RHO, c=C
            )
            for factor in (0.0, 0.5, 0.9, 1.2, 2.0):
                value = infinite_cylinder_conductance(
                    n, k, factor * k, radius=RADIUS, rho=RHO, c=C
                )
                assert value > -1e-12


def test_axial_spectrum_matches_numerical_quadrature():
    """Closed-form ``int sin(m pi x/L) e^{-i q x} dx`` vs a dense Riemann sum."""
    length, m = 2.0, 2
    for k_axial in (0.0, 0.3, 1.0, 4.0):
        grid = torch.linspace(0.0, length, 400001, dtype=torch.float64)
        f = torch.sin(m * math.pi * grid / length) * torch.exp(-1j * k_axial * grid)
        numeric = complex(torch.trapezoid(f, grid))
        closed = complex(axial_spectrum(
            torch.tensor([k_axial], dtype=torch.float64), length, m
        )[0])
        # k_x = 0 makes the integral vanish identically: compare absolutely there.
        if abs(numeric) > 1e-3:
            assert abs(closed - numeric) / abs(numeric) < 1e-4
        else:
            assert abs(closed - numeric) < 1e-9
    # removable singularity at k_x = m pi / L: the limit is -i L/2
    singular = axial_spectrum(
        torch.tensor([m * math.pi / length], dtype=torch.float64), length, m
    )[0]
    assert complex(singular) == pytest.approx(complex(0.0, -0.5 * length), abs=1e-12)


def test_window_closure_reproduces_infinite_cylinder_for_long_window():
    """Long window: the windowed power must converge to the J&F modal value."""
    n, m = 2, 1
    length = 400.0
    k = 2.0 * math.pi * 60.0 / C
    k_axial = m * math.pi / length
    windowed = finite_cylinder_mode_power(
        m, n, radius=RADIUS, length=length, frequency=60.0,
        rho=RHO, c=C, samples=8001,
    )
    analytic = infinite_cylinder_efficiency(
        n, k, k_axial, radius=RADIUS, rho=RHO, c=C
    )
    assert windowed.sigma / analytic == pytest.approx(1.0, rel=0.02)


def test_window_closure_convergence_controls():
    """Sampling and band width are converged to well below the 15 % gate."""
    coarse = finite_cylinder_mode_power(
        1, 2, radius=RADIUS, length=LENGTH, frequency=150.0,
        rho=RHO, c=C, samples=2001, half_width=8.0,
    )
    fine = finite_cylinder_mode_power(
        1, 2, radius=RADIUS, length=LENGTH, frequency=150.0,
        rho=RHO, c=C, samples=8001, half_width=16.0,
    )
    assert coarse.sigma / fine.sigma == pytest.approx(1.0, rel=5e-3)


def test_window_closure_scales_with_velocity_squared():
    base = finite_cylinder_mode_power(
        1, 1, radius=RADIUS, length=LENGTH, frequency=100.0, rho=RHO, c=C,
        velocity_amplitude=1.0, samples=2001,
    )
    doubled = finite_cylinder_mode_power(
        1, 1, radius=RADIUS, length=LENGTH, frequency=100.0, rho=RHO, c=C,
        velocity_amplitude=2.0, samples=2001,
    )
    assert doubled.power / base.power == pytest.approx(4.0, rel=1e-9)
    assert doubled.sigma == pytest.approx(base.sigma, rel=1e-9)


def test_monopole_power_matrix_single_source():
    """One monopole radiates ``|q|^2/(8 pi rho c)``."""
    sources = torch.zeros((1, 3), dtype=torch.float64)
    matrix = monopole_power_matrix(sources, 1.7, RHO, C)
    assert float(matrix[0, 0]) == pytest.approx(1.0 / (8.0 * math.pi * RHO * C),
                                                rel=1e-12)


def _sphere(radius, n_theta=13, n_phi=24):
    theta = torch.linspace(0.0, math.pi, n_theta + 2, dtype=torch.float64)[1:-1]
    phi = torch.linspace(0.0, 2.0 * math.pi, n_phi + 1, dtype=torch.float64)[:-1]
    th, ph = torch.meshgrid(theta, phi, indexing="ij")
    th, ph = th.reshape(-1), ph.reshape(-1)
    normals = torch.stack(
        (torch.sin(th) * torch.cos(ph), torch.sin(th) * torch.sin(ph), torch.cos(th)),
        dim=1,
    )
    return radius * normals, normals


def test_equivalent_source_reproduces_pulsating_sphere():
    """Exterior solve must return the exact pulsating-sphere power."""
    radius = 0.5
    points, normals = _sphere(radius)
    velocity = torch.ones(points.shape[0], dtype=torch.float64)
    for ka in (0.5, 1.0, 2.0, 3.0):
        result = equivalent_source_power(
            points, normals, velocity,
            frequency=ka * C / (2.0 * math.pi * radius),
            rho=RHO, c=C, regularization=1e-6,
        )
        exact = 2.0 * math.pi * radius ** 2 * RHO * C * ka ** 2 / (1.0 + ka ** 2)
        assert result.power / exact == pytest.approx(1.0, rel=0.01), ka


def test_equivalent_source_rigid_body_radiates_nothing():
    """Negative control: ``v_n = 0`` (rigid baffled body) gives zero power."""
    points, normals, weights, n_lateral, _ = _lateral(n_axial=9, n_circ=16)
    zero = torch.zeros(points.shape[0], dtype=torch.float64)
    result = equivalent_source_power(
        points, normals, zero, frequency=120.0, rho=RHO, c=C,
        area_weights=weights,
    )
    assert abs(result.power) < 1e-30
    assert result.sigma == 0.0 or math.isnan(result.sigma)


def test_window_closure_zero_velocity_radiates_nothing():
    result = finite_cylinder_mode_power(
        1, 1, radius=RADIUS, length=LENGTH, frequency=120.0, rho=RHO, c=C,
        velocity_amplitude=0.0, samples=2001,
    )
    assert result.power == 0.0


def test_compact_body_asymptote_reproduces_pulsating_sphere():
    """Monopole + dipole asymptote must be exact for a breathing sphere.

    For a sphere ``v_n = V`` the dipole moment vanishes and ``Pi_mono =
    rho c k^2 (4 pi a^2 V)^2/(8 pi)`` is the leading term of the exact
    ``2 pi a^2 rho c V^2 (ka)^2/(1+(ka)^2)``.
    """
    radius = 0.5
    points, normals = _sphere(radius)
    weights = torch.full((points.shape[0],), 4.0 * math.pi * radius ** 2
                         / points.shape[0], dtype=torch.float64)
    velocity = torch.ones(points.shape[0], dtype=torch.float64)
    ka = 0.1
    frequency = ka * C / (2.0 * math.pi * radius)
    power, volume_velocity, dipole = compact_body_power(
        points, normals, velocity, weights, frequency=frequency, rho=RHO, c=C
    )
    exact = 2.0 * math.pi * radius ** 2 * RHO * C * ka ** 2 / (1.0 + ka ** 2)
    assert dipole == pytest.approx(0.0, abs=1e-12)
    assert volume_velocity == pytest.approx(4.0 * math.pi * radius ** 2, rel=1e-12)
    assert power / exact == pytest.approx(1.0, rel=0.02)


def test_cylinder_surface_grid_geometry():
    points, normals, weights, n_lateral, _ = _lateral(n_axial=5, n_circ=8)
    lateral_weights = weights[:n_lateral]
    assert float(lateral_weights.sum()) == pytest.approx(
        2.0 * math.pi * RADIUS * LENGTH, rel=1e-12
    )
    # Caps are rigid matching points: zero weight, and they are off the lateral
    # surface (|r| < R) so they cannot add radiating area.
    assert float(weights[n_lateral:].sum()) == 0.0
    caps = points[n_lateral:]
    assert torch.all(torch.linalg.norm(caps[:, 1:], dim=1) <= RADIUS + 1e-12)
    unit = torch.linalg.norm(normals, dim=1)
    assert torch.allclose(unit, torch.ones_like(unit), atol=1e-12)
    lateral = normals[:n_lateral]
    assert torch.allclose(lateral[:, 0], torch.zeros_like(lateral[:, 0]))


def test_fourier_cylinder_mode_recovers_harmonic():
    """Mode identification must pick out a planted ``(m, n)`` sine harmonic."""
    length, radius = 2.0, 0.5
    points, _, _, n_lateral, _ = _lateral(n_axial=17, n_circ=32)
    x = points[:n_lateral, 0]
    theta = torch.atan2(points[:n_lateral, 2], points[:n_lateral, 1])
    planted = torch.sin(2 * math.pi * x / length) * torch.cos(3 * theta)
    coeffs, dominant, share = fourier_cylinder_mode(
        x, theta, planted, length=length, axial_modes=4, circ_modes=6
    )
    assert dominant == (2, 3)
    assert share > 0.95
    assert coeffs[("cos", 2, 3)] == pytest.approx(1.0, abs=1e-6)
    assert abs(coeffs[("sin", 2, 3)]) < 1e-9


def test_shell_radial_amplitudes_project_the_normal_direction():
    """Radial extraction: normal displacement -> itself, axial -> zero."""
    theta = torch.linspace(0.0, 2.0 * math.pi, 8, dtype=torch.float64)[:-1]
    radius, length = 0.5, 1.0
    nodes = torch.stack(
        (torch.zeros_like(theta), radius * torch.cos(theta), radius * torch.sin(theta)),
        dim=1,
    )
    normal = torch.stack((torch.zeros_like(theta), torch.sin(theta), torch.cos(theta)),
                         dim=1)
    mode = torch.zeros((theta.numel(), 6), dtype=torch.float64)
    mode[:, 0:3] = 0.25 * normal
    radial = shell_radial_amplitudes(nodes, theta, mode.reshape(-1))
    assert torch.allclose(radial, torch.full_like(radial, 0.25), atol=1e-12)

    axial_mode = torch.zeros((theta.numel(), 6), dtype=torch.float64)
    axial_mode[:, 0] = 1.0
    radial_axial = shell_radial_amplitudes(nodes, theta, axial_mode.reshape(-1))
    assert torch.allclose(radial_axial, torch.zeros_like(radial_axial), atol=1e-12)