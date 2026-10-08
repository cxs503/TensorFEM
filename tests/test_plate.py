import pytest
import torch

from tensorfem.plate import (kirchhoff_sine_center_deflection,
                             mindlin_sine_center_deflection,
                             q4_mindlin_stiffness,
                             solve_simply_supported_sine_square)


def _center_w(result, n):
    center = (n//2)*(n+1) + n//2
    return result.displacement[3*center].item()


def test_element_is_symmetric_and_differentiable():
    xy = torch.tensor([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=torch.float64)
    E = torch.tensor(2.1e11, dtype=torch.float64, requires_grad=True)
    k = q4_mindlin_stiffness(xy, E, .3, .1)
    assert torch.allclose(k, k.T, atol=1e-8)
    k.square().sum().backward()
    assert E.grad is not None and torch.isfinite(E.grad)


@pytest.mark.parametrize("thickness", [0.01, 0.001])
def test_thin_plate_sine_benchmark_below_three_percent(thickness):
    n, E, nu, q = 16, 1.0e7, .3, 1.0
    result = solve_simply_supported_sine_square(n, young=E, poisson=nu,
                                                thickness=thickness, q0=q)
    exact = kirchhoff_sine_center_deflection(1., E, nu, thickness, q)
    error = abs(_center_w(result, n)-exact)/exact
    assert error < .03


def test_mesh_convergence_and_no_shear_locking():
    E, nu, t, q = 1.0e7, .3, .001, 1.
    exact = kirchhoff_sine_center_deflection(1., E, nu, t, q)
    errors = []
    for n in (4, 8, 16):
        errors.append(abs(_center_w(solve_simply_supported_sine_square(
            n, young=E, poisson=nu, thickness=t, q0=q), n)-exact)/exact)
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < .03


def test_thick_mindlin_plate_below_three_percent():
    n, E, nu, t, q = 16, 1.0e7, .3, .2, 1.
    result = solve_simply_supported_sine_square(n, young=E, poisson=nu,
                                                thickness=t, q0=q)
    exact = mindlin_sine_center_deflection(1., E, nu, t, q)
    assert abs(_center_w(result, n)-exact)/exact < .03


def test_skew_plate_affine_rigid_tilt_has_no_shear_or_bending_energy():
    # This test detects a transpose error hidden by axis-aligned square meshes.
    xy = torch.tensor([[0., 0.], [2., .3], [2.4, 1.3], [.4, 1.]], dtype=torch.float64)
    q = torch.zeros((4, 3), dtype=torch.float64)
    q[:, 0] = .2*xy[:, 0]-.3*xy[:, 1]+.4
    q[:, 1] = -.2
    q[:, 2] = .3
    k = q4_mindlin_stiffness(xy, 210e9, .3, .01)
    residual = torch.linalg.vector_norm(k@q.flatten())
    scale = torch.linalg.matrix_norm(k)*torch.linalg.vector_norm(q)
    assert float(residual/scale) < 1e-13
