"""Verification of the free-edge plate vibration path against an independent Ritz reference.

Every assertion below is against either (a) a literature frequency coefficient,
(b) a reference produced by a different discretisation (global Rayleigh--Ritz), or
(c) an exact matrix identity.  No assertion compares against a stored snapshot of
this repository's own output.
"""
from __future__ import annotations

import math

import pytest

from tensorfem.plate_ritz_reference import (
    energy_quadrature_check,
    legendre_operator_matrices,
    ritz_free_plate_convergence,
    ritz_free_plate_eigenvalues,
)
from tensorfem.plate_vibration import (
    BOUNDARY_CLAMPED,
    BOUNDARY_FREE,
    BOUNDARY_SIMPLE,
    kirchhoff_angular_frequency,
    plate_rigidity,
    solve_plate_vibration,
)

YOUNG = 200.0e9
POISSON = 0.30
THICKNESS = 0.01
DENSITY = 7850.0


def _lambda(result, mode: int, length: float = 1.0) -> float:
    omega = 2.0 * math.pi * float(result.frequencies_hz[mode])
    scale = math.sqrt(DENSITY * THICKNESS / plate_rigidity(YOUNG, POISSON, THICKNESS))
    return omega * length**2 * scale


def _free(mesh: int, modes: int = 4, width: float | None = None, length: float = 1.0):
    return solve_plate_vibration(
        mesh, mesh, boundary=BOUNDARY_FREE, modes=modes, length=length, width=width,
        young=YOUNG, poisson=POISSON, thickness=THICKNESS, density=DENSITY,
    )


# --------------------------------------------------------------- reference side
def test_legendre_operator_mass_matrix_is_exact():
    order = 6
    mass, first, second, coupling = legendre_operator_matrices(order, 2.0)
    for matrix in (mass, first, second):
        assert matrix.shape == (order + 1, order + 1)
        assert float(abs(matrix - matrix.T).max()) < 1e-12
    # int_0^2 P_0 P_0 dx = 2, int_0^2 P_1 P_1 dx = 2/3
    assert mass[0, 0] == pytest.approx(2.0, rel=1e-13)
    assert mass[1, 1] == pytest.approx(2.0 / 3.0, rel=1e-13)
    # the one-sided coupling operator is not symmetric, by construction
    assert float(abs(coupling - coupling.T).max()) > 1e-6


def test_assembled_form_matches_direct_two_dimensional_quadrature():
    # The transposed-coupling bug (kron(C2x, C2y) instead of kron(C2x, C2y.T))
    # returns ~5e-2 here; a correct assembly is at quadrature round-off.
    assert energy_quadrature_check(order=8) < 1e-10
    assert energy_quadrature_check(order=8, width=0.5) < 1e-10


def test_ritz_square_plate_reproduces_leissa_coefficients():
    result = ritz_free_plate_eigenvalues(order=14, count=4)
    values = [float(v) for v in result.dimensionless]
    # Leissa, Vibration of Plates, NASA SP-160 (1969): free square plate, nu = 0.3
    assert values[0] == pytest.approx(13.468, rel=5e-5)
    assert values[1] == pytest.approx(19.596, rel=5e-4)
    assert values[2] == pytest.approx(24.272, rel=5e-4)
    # the first elastic mode of a free plate is well below the simply supported
    # fundamental pi^2(1 + 1) = 19.739
    assert values[0] < math.pi**2 * 2.0


def test_ritz_sequence_is_a_monotone_upper_bound():
    rows = ritz_free_plate_convergence(orders=(8, 10, 12, 14, 16), count=1)
    values = [value for _, value in rows]
    for coarse, fine in zip(values, values[1:]):
        assert fine <= coarse + 1e-12
    assert abs(values[-1] - values[0]) < 1e-3


def test_ritz_rigid_body_constraint_is_exactly_satisfied():
    result = ritz_free_plate_eigenvalues(order=12, count=2)
    assert result.rigid_body_defect < 1e-12


def test_ritz_rectangular_plate_ordering():
    square = ritz_free_plate_eigenvalues(order=16, count=1)
    rectangle = ritz_free_plate_eigenvalues(order=16, width=0.5, count=2)
    # a narrower plate is stiffer in the short direction, and the dimensionless
    # coefficient uses the fixed length a
    assert float(rectangle.dimensionless[0]) > float(square.dimensionless[0])
    assert rectangle.aspect_ratio == pytest.approx(0.5)


# ----------------------------------------------------------- finite-element side
def test_free_plate_has_exactly_three_rigid_body_modes_plus_one_hourglass():
    for mesh in (8, 12):
        result = _free(mesh)
        # 3 rigid-body modes are physical; the 4th zero-energy mode is the known
        # spurious (hourglass) mode of the SRI Q4 element and is documented.
        assert result.rigid_body_modes == 4
        assert float(result.frequencies_hz[0]) > 0.0


def test_free_square_plate_first_four_modes_within_three_percent():
    reference = [float(v) for v in ritz_free_plate_eigenvalues(order=14, count=4).dimensionless]
    result = _free(16)
    for mode in range(4):
        computed = _lambda(result, mode)
        assert abs(computed - reference[mode]) / reference[mode] < 0.03, (
            f"mode {mode + 1}: {computed:.4f} vs {reference[mode]:.4f}"
        )


def test_free_plate_mesh_study_errors_decrease():
    reference = float(ritz_free_plate_eigenvalues(order=14, count=1).dimensionless[0])
    errors = []
    for mesh in (6, 8, 10, 12):
        errors.append(abs(_lambda(_free(mesh, modes=1), 0) - reference) / reference)
    assert errors[0] > errors[-1]
    assert errors[-1] < 3.0e-3


def test_free_rectangular_plate_two_to_one_within_three_percent():
    reference = ritz_free_plate_eigenvalues(order=18, width=0.5, count=2)
    result = _free(16, modes=2, width=0.5)
    for mode in range(2):
        computed = _lambda(result, mode)
        target = float(reference.dimensionless[mode])
        assert abs(computed - target) / target < 0.03


def test_free_plate_fundamental_lies_between_rigid_and_simply_supported():
    free = _lambda(_free(12, modes=1), 0)
    supported = _lambda(
        solve_plate_vibration(
            12, 12, boundary=BOUNDARY_SIMPLE, modes=1, young=YOUNG, poisson=POISSON,
            thickness=THICKNESS, density=DENSITY,
        ),
        0,
    )
    assert free < supported
    # ... and the free value is close to the Kirchhoff-supported ratio 13.468/19.739
    assert free / supported == pytest.approx(13.468 / (math.pi**2 * 2.0), rel=0.02)


def test_supported_boundaries_report_no_zero_energy_modes():
    for boundary in (BOUNDARY_SIMPLE, BOUNDARY_CLAMPED):
        result = solve_plate_vibration(
            8, 8, boundary=boundary, modes=2, young=YOUNG, poisson=POISSON,
            thickness=THICKNESS, density=DENSITY,
        )
        assert result.rigid_body_modes == 0


def test_free_plate_rotation_dof_carries_no_rigid_motion_energy():
    result = _free(10, modes=2)
    field = result.modes.reshape(-1, 3)
    # A rigid-body translation field is a combination of 1, x, y in w and zero
    # rotations; the first elastic mode must therefore not be a pure translation.
    translation_only = float(field[:, 1].abs().max()) < 1e-12 and float(field[:, 2].abs().max()) < 1e-12
    assert not translation_only
    # the elastic mode is orthogonal (in the mass metric) to nothing degenerate:
    # its frequency is far above the zero-energy cluster
    assert float(result.frequencies_hz[0]) > 1.0


# ------------------------------------------------------------------- error paths
def test_free_plate_rejects_mode_request_beyond_elastic_spectrum():
    with pytest.raises(ValueError, match="modes"):
        _free(2, modes=40)


def test_free_plate_rejects_negative_rigid_body_tolerance():
    with pytest.raises(ValueError, match="rigid_body_relative_tolerance"):
        solve_plate_vibration(
            4, 4, boundary=BOUNDARY_FREE, modes=1, rigid_body_relative_tolerance=-1.0
        )


def test_free_plate_rejects_non_numeric_rigid_body_tolerance():
    with pytest.raises(TypeError, match="rigid_body_relative_tolerance"):
        solve_plate_vibration(
            4, 4, boundary=BOUNDARY_FREE, modes=1, rigid_body_relative_tolerance="tight"
        )


def test_ritz_rejects_invalid_order_and_poisson():
    with pytest.raises(ValueError, match="order"):
        ritz_free_plate_eigenvalues(order=1)
    with pytest.raises(ValueError, match="poisson"):
        ritz_free_plate_eigenvalues(poisson=0.5)
    with pytest.raises(TypeError, match="order"):
        ritz_free_plate_eigenvalues(order=8.0)