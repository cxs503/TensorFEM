"""Shear / combined in-plane buckling of simply supported orthotropic panels."""
import math

import numpy as np
import pytest

from tensorfem.marine_plate_buckling import (
    isotropic_plate_rigidities,
    navier_uniaxial_buckling_load,
)
from tensorfem.plate_shear_buckling import (
    LITERATURE_KS_LONG_PANEL,
    LITERATURE_KS_SQUARE,
    classical_shear_coefficient,
    critical_shear_stress,
    ritz_inplane_buckling_load,
    run_shear_buckling_qualification,
    shear_buckling_coefficient,
    sine_energy_matrices,
    sine_energy_quadrature,
)

E, NU, T = 210.0e9, 0.3, 0.012


def _rigidities():
    return isotropic_plate_rigidities(E, NU, T)


def test_closed_form_shear_matrix_matches_quadrature():
    """Regression: the sine/cosine index order of the shear coupling integral.

    ``int w,x^i w,y^j`` pairs the x derivative of one mode with the y
    derivative of the other; swapping the two sine arguments silently perturbs
    the critical load by 18 %, so the closed assembly is pinned against
    Gauss-Legendre quadrature of the same functional here.
    """
    d, _, h = _rigidities()
    closed = sine_energy_matrices(length=1.0, width=1.0, rigidity_x=d,
                                  rigidity_y=d, coupling_rigidity=h,
                                  basis_x=4, basis_y=4)
    quadrature = sine_energy_quadrature(length=1.0, width=1.0, rigidity_x=d,
                                        rigidity_y=d, coupling_rigidity=h,
                                        basis_x=4, basis_y=4)
    bending_c, uniaxial_c, shear_c, modes_c = closed
    bending_q, uniaxial_q, shear_q, modes_q = quadrature
    assert modes_c == modes_q
    scale = np.abs(bending_c).max()
    assert np.abs(bending_c - bending_q).max() / scale < 1e-10
    assert np.abs(uniaxial_c - uniaxial_q).max() / np.abs(uniaxial_c).max() < 1e-10
    assert np.abs(shear_c - shear_q).max() / np.abs(shear_c).max() < 1e-10
    # the shear coupling must be non-trivial (a fully zero matrix would make
    # every quadrature comparison vacuous)
    assert np.abs(shear_c).max() > 0.0
    assert np.abs(shear_c - shear_c.T).max() == 0.0


def test_shear_matrix_off_diagonal_couples_opposite_parity_modes():
    d, _, h = _rigidities()
    _, _, shear, modes = sine_energy_matrices(
        length=1.0, width=1.0, rigidity_x=d, rigidity_y=d,
        coupling_rigidity=h, basis_x=3, basis_y=3)
    index = {pair: i for i, pair in enumerate(modes)}
    assert shear[index[(1, 1)], index[(2, 2)]] != 0.0
    # same parity in both directions -> orthogonal integrals
    assert shear[index[(1, 1)], index[(3, 3)]] == pytest.approx(0.0, abs=1e-30)


def test_uniaxial_limit_reproduces_the_navier_solution():
    """With zero shear the sine machinery must collapse to the uniaxial oracle."""
    d, dy, h = _rigidities()
    for aspect in (1.0, 2.0, 3.5):
        mode = ritz_inplane_buckling_load(
            length=aspect, width=1.0, rigidity_x=d, rigidity_y=dy,
            coupling_rigidity=h, basis_x=24, basis_y=24, compression_ratio=0.0)
        reference, m, n = navier_uniaxial_buckling_load(
            length=aspect, width=1.0, rigidity_x=d, rigidity_y=dy,
            coupling_rigidity=h)
        assert mode.multiplier == pytest.approx(reference, rel=1e-10)
        assert mode.shear == 0.0
        assert mode.compression_x == pytest.approx(reference, rel=1e-10)


def test_square_plate_shear_coefficient_matches_literature():
    """Timoshenko & Gere k_s = 9.34 for a simply supported square plate."""
    d, _, h = _rigidities()
    mode = ritz_inplane_buckling_load(length=1.0, width=1.0, rigidity_x=d,
                                      rigidity_y=d, coupling_rigidity=h,
                                      basis_x=16, basis_y=16)
    coefficient = shear_buckling_coefficient(
        shear_load=mode.multiplier, rigidity=d, width=1.0)
    assert abs(coefficient / LITERATURE_KS_SQUARE - 1.0) < 0.03
    assert mode.buckling
    # τ_cr = k_s π² D / (b² t) with the same data
    stress = critical_shear_stress(shear_load=mode.multiplier, thickness=T)
    assert stress == pytest.approx(coefficient * math.pi**2 * d / (1.0**2 * T),
                                   rel=1e-12)


def test_long_panel_shear_coefficient_approaches_the_asymptote():
    d, _, h = _rigidities()
    coefficients = []
    for aspect in (4.0, 6.0, 10.0):
        mode = ritz_inplane_buckling_load(
            length=aspect, width=1.0, rigidity_x=d, rigidity_y=d,
            coupling_rigidity=h, basis_x=20, basis_y=16)
        coefficients.append(shear_buckling_coefficient(
            shear_load=mode.multiplier, rigidity=d, width=1.0))
    assert abs(coefficients[1] / LITERATURE_KS_LONG_PANEL - 1.0) < 0.03
    assert abs(coefficients[2] / LITERATURE_KS_LONG_PANEL - 1.0) < 0.02
    # monotone decrease towards 5.34 and never below the classical asymptote
    assert coefficients[0] > coefficients[1] > coefficients[2] > LITERATURE_KS_LONG_PANEL
    # the classical closed approximation is only an approximation: at a/b = 4
    # the converged value sits 0.6 % above it, so agreement is bounded, not one-sided
    for aspect, value in zip((4.0, 6.0, 10.0), coefficients):
        assert abs(value / classical_shear_coefficient(aspect) - 1.0) < 0.05


def test_basis_refinement_is_monotone_and_converged():
    d, _, h = _rigidities()
    values = []
    for basis in (4, 6, 8, 10, 16, 20):
        mode = ritz_inplane_buckling_load(
            length=1.0, width=1.0, rigidity_x=d, rigidity_y=d,
            coupling_rigidity=h, basis_x=basis, basis_y=basis)
        values.append(shear_buckling_coefficient(
            shear_load=mode.multiplier, rigidity=d, width=1.0))
    assert all(first > second for first, second in zip(values, values[1:]))
    assert values[-1] / values[-2] - 1.0 < 1e-3


def test_combined_compression_and_shear_within_the_interaction_envelope():
    """N_x/N_x,cr + (N_xy/N_xy,cr)^2 = 1 is the classical interaction."""
    d, _, h = _rigidities()
    pure_shear = ritz_inplane_buckling_load(
        length=1.0, width=1.0, rigidity_x=d, rigidity_y=d, coupling_rigidity=h,
        basis_x=16, basis_y=16)
    uniaxial = ritz_inplane_buckling_load(
        length=1.0, width=1.0, rigidity_x=d, rigidity_y=d, coupling_rigidity=h,
        basis_x=16, basis_y=16, compression_ratio=0.0)
    shear_capacity = pure_shear.multiplier
    compression_capacity = uniaxial.multiplier
    for ratio in (0.5, 1.0, 2.0):
        mode = ritz_inplane_buckling_load(
            length=1.0, width=1.0, rigidity_x=d, rigidity_y=d,
            coupling_rigidity=h, basis_x=16, basis_y=16,
            compression_ratio=ratio)
        assert mode.buckling
        assert mode.compression_x > 0.0 and mode.shear > 0.0
        assert mode.shear == pytest.approx(ratio * mode.multiplier, rel=1e-12)
        # both components shrink when the other load is present
        assert mode.compression_x < compression_capacity
        assert mode.shear < shear_capacity
        assert abs(mode.compression_x / compression_capacity
                   + (mode.shear / shear_capacity) ** 2 - 1.0) < 0.06


def test_transverse_stiffening_increases_shear_capacity():
    d, _, h = _rigidities()
    results = [
        ritz_inplane_buckling_load(length=2.0, width=1.0, rigidity_x=d,
                                   rigidity_y=dy, coupling_rigidity=h,
                                   basis_x=16, basis_y=16)
        for dy in (d, 2.0 * d, 4.0 * d)
    ]
    loads = [item.multiplier for item in results]
    assert loads[0] < loads[1] < loads[2]


def test_qualification_suite_passes():
    report = run_shear_buckling_qualification(tolerance=0.03, basis=16)
    assert report["passed"] is True
    assert report["schema"] == "tensorfem.plate-shear-buckling/1.0"
    assert {row["case"] for row in report["cases"]} == {
        "square_shear", "long_panel_shear", "orthotropic_panel_shear"}
    for row in report["cases"]:
        assert row["passed"] is True
        if row["relative_error"] is not None:
            assert row["relative_error"] < 0.03


@pytest.mark.parametrize("kwargs", [
    {"length": 0.0}, {"width": -1.0}, {"rigidity_x": 0.0},
    {"coupling_rigidity": -2.0},
])
def test_invalid_geometry_is_rejected(kwargs):
    d, _, h = _rigidities()
    arguments = dict(length=1.0, width=1.0, rigidity_x=d, rigidity_y=d,
                     coupling_rigidity=h)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        ritz_inplane_buckling_load(**arguments)


def test_invalid_basis_and_ratio_are_rejected():
    d, _, h = _rigidities()
    with pytest.raises(ValueError):
        ritz_inplane_buckling_load(length=1.0, width=1.0, rigidity_x=d,
                                   rigidity_y=d, coupling_rigidity=h, basis_x=1)
    with pytest.raises(ValueError):
        ritz_inplane_buckling_load(length=1.0, width=1.0, rigidity_x=d,
                                   rigidity_y=d, coupling_rigidity=h,
                                   compression_ratio=-1.0)
    with pytest.raises(TypeError):
        ritz_inplane_buckling_load(length=1.0, width=1.0, rigidity_x=d,
                                   rigidity_y=d, coupling_rigidity=h,
                                   compression_ratio="1.0")


def test_classical_coefficient_limits():
    assert classical_shear_coefficient(1.0) == pytest.approx(9.34)
    assert classical_shear_coefficient(1.0e6) == pytest.approx(5.34, abs=1e-8)
    with pytest.raises(ValueError):
        classical_shear_coefficient(0.0)