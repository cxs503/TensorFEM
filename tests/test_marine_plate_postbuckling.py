import math

import pytest

from tensorfem.marine_plate_postbuckling import (
    finite_difference_prestress_buckling, imperfect_postbuckling_path,
    navier_prestress_buckling_factor, postbuckling_reference_amplitude,
    run_plate_postbuckling_qualification,
)
from tensorfem.marine_plate_buckling import isotropic_plate_rigidities


def test_equal_biaxial_square_plate_has_closed_form_factor():
    d, dy, h = isotropic_plate_rigidities(210e9, 0.3, 0.01)
    factor, m, n = navier_prestress_buckling_factor(
        length=1.0, width=1.0, rigidity_x=d, rigidity_y=dy,
        coupling_rigidity=h, prestress_x=2.0, prestress_y=2.0)
    assert factor == pytest.approx(2.0 * math.pi**2 * d / 2.0, rel=1e-14)
    assert (m, n) == (1, 1)


def test_prestress_eigenproblem_converges_to_navier_oracle():
    d, dy, h = isotropic_plate_rigidities(200e9, 0.29, 0.012)
    results = [finite_difference_prestress_buckling(
        length=1.5, width=1.0, rigidity_x=d, rigidity_y=dy,
        coupling_rigidity=h, prestress_x=8e5, prestress_y=3e5,
        grid_x=n, grid_y=n) for n in (8, 16, 32)]
    assert results[2].relative_error < results[1].relative_error < results[0].relative_error
    assert results[2].relative_error < 0.03


def test_unloaded_imperfect_path_starts_at_stress_free_shape():
    path = imperfect_postbuckling_path(maximum_load_factor=1.2, steps=160,
                                      imperfection=0.1, coefficient=1.0)
    assert path[0].amplitude == pytest.approx(0.1)
    assert path[-1].amplitude > path[0].amplitude
    assert max(point.relative_error for point in path) < 0.03


def test_reference_root_satisfies_reduced_von_karman_equilibrium():
    q = postbuckling_reference_amplitude(load_factor=1.1, imperfection=0.08,
                                         coefficient=0.7)
    residual = (1.0 - 1.1) * q - 0.08 + 0.7 * (q*q - 0.08**2) * q
    assert residual == pytest.approx(0.0, abs=1e-14)


def test_qualification_enforces_grid_and_step_convergence():
    report = run_plate_postbuckling_qualification()
    assert report["passed"] is True
    for row in report["prestress_eigenbuckling"]:
        assert row["errors"][2] < row["errors"][1] < row["errors"][0]
        assert row["errors"][2] < report["tolerance"]
    errors = report["imperfect_postbuckling"]["maximum_errors"]
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < report["tolerance"]


@pytest.mark.parametrize("kwargs,exception", [
    ({"prestress_x": -1.0}, ValueError), ({"prestress_x": 0.0, "prestress_y": 0.0}, ValueError),
    ({"grid_x": True}, TypeError), ({"grid_y": 1}, ValueError),
])
def test_prestress_solver_rejects_invalid_inputs(kwargs, exception):
    values = dict(length=1.0, width=1.0, rigidity_x=1.0, rigidity_y=1.0,
                  coupling_rigidity=1.0, prestress_x=1.0, prestress_y=0.0,
                  grid_x=8, grid_y=8)
    values.update(kwargs)
    with pytest.raises(exception):
        finite_difference_prestress_buckling(**values)


@pytest.mark.parametrize("kwargs,exception", [
    ({"imperfection": 0.0}, ValueError), ({"steps": 1}, ValueError),
    ({"samples": 42}, ValueError), ({"maximum_load_factor": float("nan")}, ValueError),
])
def test_postbuckling_path_rejects_invalid_inputs(kwargs, exception):
    values = dict(maximum_load_factor=1.2, steps=20, imperfection=0.1)
    values.update(kwargs)
    with pytest.raises(exception):
        imperfect_postbuckling_path(**values)
