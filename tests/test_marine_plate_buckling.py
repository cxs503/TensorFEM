import math

import pytest

from tensorfem.marine_plate_buckling import (
    finite_difference_uniaxial_buckling, isotropic_plate_rigidities,
    navier_uniaxial_buckling_load, run_local_plate_buckling_qualification,
    smeared_longitudinal_stiffener_rigidity,
)


def test_square_isotropic_oracle_is_classical_four_pi_squared_d():
    d, dy, h = isotropic_plate_rigidities(210e9, 0.3, 0.01)
    load, m, n = navier_uniaxial_buckling_load(
        length=1.0, width=1.0, rigidity_x=d, rigidity_y=dy,
        coupling_rigidity=h)
    assert load == pytest.approx(4.0 * math.pi**2 * d, rel=1e-14)
    assert (m, n) == (1, 1)


def test_aspect_ratio_selects_multiple_longitudinal_half_waves():
    d, dy, h = isotropic_plate_rigidities(200e9, 0.3, 0.01)
    result = finite_difference_uniaxial_buckling(
        length=2.0, width=1.0, thickness=0.01, rigidity_x=d,
        rigidity_y=dy, coupling_rigidity=h, grid_x=32, grid_y=32)
    assert (result.longitudinal_half_waves, result.transverse_half_waves) == (2, 1)
    assert result.relative_error < 0.03
    assert result.critical_stress == pytest.approx(result.critical_line_load / 0.01)


def test_qualification_has_monotone_convergence_and_strict_gate():
    report = run_local_plate_buckling_qualification()
    assert report["passed"] is True
    assert len(report["cases"]) == 3
    for case in report["cases"]:
        assert case["errors"][2] < case["errors"][1] < case["errors"][0]
        assert case["errors"][2] < report["tolerance"]
    assert report["cases"][1]["mode"] == [2, 1]
    assert report["cases"][2]["mode"] != report["cases"][1]["mode"]


@pytest.mark.parametrize("kwargs,exception", [
    ({"length": 0.0}, ValueError), ({"rigidity_x": float("nan")}, ValueError),
    ({"grid_x": 2.5}, TypeError), ({"grid_y": 1}, ValueError),
    ({"maximum_half_waves": True}, TypeError),
])
def test_solver_rejects_invalid_inputs(kwargs, exception):
    values = dict(length=1.0, width=1.0, thickness=0.01, rigidity_x=1.0,
                  rigidity_y=1.0, coupling_rigidity=1.0, grid_x=8, grid_y=8)
    values.update(kwargs)
    with pytest.raises(exception):
        finite_difference_uniaxial_buckling(**values)


def test_stiffener_validation_and_rigidity_increment():
    value = smeared_longitudinal_stiffener_rigidity(
        plate_rigidity=10.0, stiffener_young=200.0, stiffener_area=2.0,
        eccentricity=-0.5, spacing=4.0)
    assert value == 35.0
    with pytest.raises(ValueError):
        smeared_longitudinal_stiffener_rigidity(
            plate_rigidity=10.0, stiffener_young=200.0, stiffener_area=2.0,
            eccentricity=float("inf"), spacing=4.0)
