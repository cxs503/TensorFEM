import pytest
import torch

from tensorfem.crack_tip_j_integral import (
    ContourField, contour_j_integral, mode_i_j_reference,
    sample_williams_mode_i_contour,
)


@pytest.mark.parametrize("plane_strain", [True, False])
def test_numerical_j_matches_independent_k_relation(plane_strain):
    k, e, nu = 42e6, 210e9, 0.29
    field = sample_williams_mode_i_contour(0.012, 720, k, e, nu,
                                           plane_strain=plane_strain)
    value = contour_j_integral(field).j_j_m2
    reference = mode_i_j_reference(k, e, nu, plane_strain=plane_strain)
    assert value == pytest.approx(reference, rel=3e-3)


def test_path_independence_for_radii_and_non_circular_contour():
    args = (35e6, 200e9, 0.3)
    fields = [sample_williams_mode_i_contour(radius, 512, *args,
              radial_perturbation=perturb) for radius, perturb in
              ((0.004, 0.0), (0.015, 0.12), (0.05, -0.18))]
    values = torch.tensor([contour_j_integral(field).j_j_m2 for field in fields])
    assert float((values.max()-values.min())/values.mean()) < 0.005


def test_quadrature_converges_to_reference():
    k, e, nu = 31e6, 70e9, 0.33
    ref = mode_i_j_reference(k, e, nu)
    errors = []
    for n in (32, 64, 128, 256):
        value = contour_j_integral(sample_williams_mode_i_contour(.01, n, k, e, nu)).j_j_m2
        errors.append(abs(value-ref))
    assert errors[-1] < errors[0]/50
    assert errors[-1]/ref < 0.003


def test_invalid_contour_fails_closed():
    field = sample_williams_mode_i_contour(.01, 32, 1e6, 1e9)
    reversed_field = ContourField(field.coordinates_m.flip(0), field.stress_pa.flip(0),
                                  field.displacement_gradient.flip(0))
    with pytest.raises(ValueError, match="counter-clockwise"):
        contour_j_integral(reversed_field)
    with pytest.raises(ValueError, match="points"):
        sample_williams_mode_i_contour(.01, 8, 1e6, 1e9)
