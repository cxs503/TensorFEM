"""Verification of the linear stress-gradient plate buckling reference.

The load case (linearly varying in-plane compression) is verified against four
independent legs: the exact uniform Navier solution, the tabulated coefficients,
Gauss--Legendre quadrature of the load potential, and basis refinement.
"""
import math

import numpy as np
import pytest

from tensorfem.marine_plate_buckling import (
    isotropic_plate_rigidities,
    navier_uniaxial_buckling_load,
)
from tensorfem.plate_shear_buckling import (
    _modes,
    ritz_inplane_buckling_load,
)
from tensorfem.plate_stress_gradient_buckling import (
    LITERATURE_GRADIENT_COEFFICIENTS,
    gradient_buckling_mode_field,
    gradient_cos_integral,
    plate_buckling_coefficient,
    reference_gradient_sweep,
    ritz_gradient_buckling_load,
    run_stress_gradient_qualification,
    sine_gradient_geometric_matrix,
    sine_gradient_geometric_matrix_quadrature,
    stress_gradient_ramp,
)

E, NU, T = 210.0e9, 0.3, 0.012


def _rigidities():
    return isotropic_plate_rigidities(E, NU, T)


def _coefficient(psi, aspect=1.0, basis=24):
    d, _, h = _rigidities()
    return ritz_gradient_buckling_load(
        length=aspect, width=1.0, rigidity_x=d, rigidity_y=d,
        coupling_rigidity=h, stress_gradient=psi, basis_x=basis, basis_y=basis,
    ).coefficient


def test_uniform_limit_reproduces_the_exact_navier_solution():
    """psi = 1 must collapse onto the exact uniform case, on all aspect ratios.

    This is the leg that proves the ramp machinery did not shift the load: both
    the merged Navier oracle and the merged uniform Galerkin path are independent
    of the code added here.
    """
    d, _, h = _rigidities()
    for aspect in (0.5, 0.75, 1.0, 2.0, 3.0):
        gradient = ritz_gradient_buckling_load(
            length=aspect, width=1.0, rigidity_x=d, rigidity_y=d,
            coupling_rigidity=h, stress_gradient=1.0, basis_x=24, basis_y=24,
        )
        navier_load, _, _ = navier_uniaxial_buckling_load(
            length=aspect, width=1.0, rigidity_x=d, rigidity_y=d, coupling_rigidity=h,
        )
        uniform = ritz_inplane_buckling_load(
            length=aspect, width=1.0, rigidity_x=d, rigidity_y=d,
            coupling_rigidity=h, compression_ratio=0.0, basis_x=24, basis_y=24,
        )
        navier_k = navier_load / (math.pi**2 * d)
        assert gradient.coefficient == pytest.approx(navier_k, rel=1e-10)
        assert gradient.multiplier == pytest.approx(uniform.multiplier, rel=1e-10)
        # and the closed single-mode identity k = ((a/m b) + (m b/a))^2
        identity = min((aspect / m + m / aspect) ** 2 for m in range(1, 9))
        assert gradient.coefficient == pytest.approx(identity, rel=1e-10)


def test_uniform_geometric_matrix_is_the_uniaxial_diagonal():
    d, _, _ = _rigidities()
    matrix, modes = sine_gradient_geometric_matrix(
        length=2.0, width=1.0, basis_x=5, basis_y=3, stress_gradient=1.0)
    expected = np.diag([(m * math.pi / 2.0) ** 2 for m, _ in modes])
    assert np.abs(matrix - expected).max() == 0.0


def test_block_assembly_matches_a_direct_mode_loop():
    """Regression for the Kronecker inflation bug.

    A plain ``kron(diag(alpha), ramp)`` produces a ``(M*N)^2`` matrix; the
    assembly must be block diagonal in the axial half-wave number.  The direct
    loop below is written from the definition ``G[(m,n),(p,q)] = alpha_m
    delta_mp Yhat[n,q]`` and shares no code with the vectorised assembly.
    """
    d, _, _ = _rigidities()
    a, b, nx, ny, psi = 1.7, 1.0, 4, 3, -0.4
    matrix, modes = sine_gradient_geometric_matrix(
        length=a, width=b, basis_x=nx, basis_y=ny, stress_gradient=psi)
    assert matrix.shape == (nx * ny, nx * ny)
    for row, (m, n) in enumerate(modes):
        for column, (p, q) in enumerate(modes):
            expected = 0.0
            if m == p:
                ramp = (gradient_cos_integral(abs(n - q), psi)
                        - gradient_cos_integral(n + q, psi))
                expected = (m * math.pi / a) ** 2 * ramp
            assert matrix[row, column] == pytest.approx(expected, abs=1e-12)


def test_closed_form_matches_gauss_quadrature_of_the_load_potential():
    for psi in (1.0, 0.5, 0.0, -0.5, -1.0):
        closed, modes = sine_gradient_geometric_matrix(
            length=1.3, width=1.0, basis_x=4, basis_y=4, stress_gradient=psi)
        numeric, numeric_modes = sine_gradient_geometric_matrix_quadrature(
            length=1.3, width=1.0, basis_x=4, basis_y=4, stress_gradient=psi)
        assert modes == numeric_modes
        scale = np.abs(closed).max()
        assert np.abs(closed - numeric).max() / scale < 1e-10


def test_geometric_matrix_is_symmetric_and_traceless_at_pure_bending():
    matrix, _ = sine_gradient_geometric_matrix(
        length=2.0, width=1.0, basis_x=6, basis_y=6, stress_gradient=-1.0)
    assert np.abs(matrix - matrix.T).max() == 0.0
    # sign reversal of the ramp removes the net compression, so the diagonal
    # (the uniform-compression part) must vanish identically
    assert np.abs(np.diag(matrix)).max() == 0.0
    # and the matrix must be indefinite, not positive semidefinite
    assert np.linalg.eigvalsh(0.5 * (matrix + matrix.T)).min() < 0.0


def test_triangular_load_matches_the_tabulated_coefficient():
    assert _coefficient(0.0, 1.0) == pytest.approx(7.81, rel=0.03)


def test_pure_bending_reproduces_the_tabulated_minimum():
    """``k = 23.9`` is the minimum over ``a/b``; a square panel is not it.

    Reporting the square value (25.53) against 23.9 would have produced a 6.8 %
    discrepancy and a false failure, so the aspect dependence is pinned here.
    """
    assert _coefficient(-1.0, 2.0) == pytest.approx(23.9, rel=0.03)
    assert _coefficient(-1.0, 0.7) == pytest.approx(23.9, rel=0.03)
    square = _coefficient(-1.0, 1.0)
    assert square == pytest.approx(25.528, rel=0.01)
    assert square > 23.9


def test_uniform_load_anchors_the_remaining_tabulated_coefficient():
    assert _coefficient(1.0, 1.0) == pytest.approx(4.00, rel=1e-6)


def test_coefficient_grows_as_the_gradient_becomes_more_adverse():
    sweep = reference_gradient_sweep(young=E, poisson=NU, thickness=T, basis=16)
    gradients = [row["stress_gradient"] for row in sweep]
    coefficients = [row["coefficient_k"] for row in sweep]
    assert gradients == sorted(gradients, reverse=True)
    assert all(later > earlier for earlier, later in zip(coefficients, coefficients[1:]))


def test_critical_coefficient_converges_from_above():
    for psi, aspect in ((0.0, 1.0), (-1.0, 2.0)):
        series = [_coefficient(psi, aspect, basis) for basis in (4, 8, 12, 24)]
        assert all(later <= earlier * (1.0 + 1e-12)
                   for earlier, later in zip(series, series[1:]))
        assert abs(series[-1] / series[-2] - 1.0) < 1e-6


def test_mode_field_rayleigh_quotient_reproduces_the_eigenvalue():
    d, _, h = _rigidities()
    field = gradient_buckling_mode_field(
        length=2.0, width=1.0, rigidity_x=d, rigidity_y=d, coupling_rigidity=h,
        stress_gradient=-1.0, basis_x=20, basis_y=20, samples=41,
    )
    assert field["energy_ratio_relative_deviation"] < 1e-10
    assert field["critical_load"] == pytest.approx(field["energy_ratio_load"], rel=1e-10)
    assert field["coefficient_k"] == pytest.approx(23.8818, rel=1e-3)
    # simply supported edges: the field vanishes on the boundary
    shape = field["shape"]
    assert np.abs(shape[0, :]).max() < 1e-12
    assert np.abs(shape[-1, :]).max() < 1e-12
    assert np.abs(shape[:, 0]).max() < 1e-12
    assert np.abs(shape[:, -1]).max() < 1e-12
    assert np.abs(shape).max() == pytest.approx(1.0, rel=1e-12)


def test_literature_table_keys_are_gated_at_three_percent():
    report = run_stress_gradient_qualification(young=E, poisson=NU, thickness=T)
    assert report["passed"] is True
    assert report["tolerance"] == 0.03
    for row in report["literature"]:
        assert row["relative_error"] < 0.03
    assert set((row["stress_gradient"], row["aspect_ratio"])
               for row in report["literature"]) == set(LITERATURE_GRADIENT_COEFFICIENTS)


def test_qualification_fails_closed_when_the_gate_is_impossible():
    """The gate must raise, not silently downgrade, when it cannot be met."""
    with pytest.raises(AssertionError):
        run_stress_gradient_qualification(
            young=E, poisson=NU, thickness=T, tolerance=1.0e-12)


def test_stress_gradient_ratio_is_validated():
    assert stress_gradient_ramp("psi", -1.0) == -1.0
    assert stress_gradient_ramp("psi", 1) == 1.0
    for bad in (-1.0001, 1.0001, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            stress_gradient_ramp("psi", bad)
    with pytest.raises(TypeError):
        stress_gradient_ramp("psi", True)
    with pytest.raises(TypeError):
        stress_gradient_ramp("psi", "0.5")


def test_basis_and_load_validation():
    d, _, _ = _rigidities()
    with pytest.raises(ValueError):
        sine_gradient_geometric_matrix(
            length=1.0, width=1.0, basis_x=1, basis_y=4, stress_gradient=0.0)
    with pytest.raises(ValueError):
        plate_buckling_coefficient(critical_load=0.0, rigidity=d, width=1.0)
    with pytest.raises(ValueError):
        gradient_buckling_mode_field(
            length=1.0, width=1.0, rigidity_x=d, rigidity_y=d,
            coupling_rigidity=d, stress_gradient=0.0, samples=2)
    assert _modes(2, 2) == [(1, 1), (1, 2), (2, 1), (2, 2)]