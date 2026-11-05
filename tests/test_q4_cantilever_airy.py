"""Verification of the Q4 plane-stress element against an exact elasticity field.

Two independent layers are checked:

1. the closed-form reference in :mod:`tensorfem.airy_cantilever` is itself
   self-consistent (pointwise equilibrium, strain compatibility, exact section
   resultants and an independently integrated strain energy);
2. the finite-element solution of the library solver reproduces that field
   inside the declared scope, with a measured convergence order close to the
   nominal second order of a fully integrated bilinear quadrilateral.

The corner band is deliberately excluded from the pointwise gate and its
failure is locked here as a known scope limit, so that no future change can
silently upgrade the claim to "3% everywhere".
"""
import functools
import math

import pytest
import torch

from tensorfem.airy_cantilever import (AiryCantilever, boundary_nodal_forces,
                                       traction_resultant)
from tensorfem.continuum import ContinuumModel, rectangular_q4_mesh, solve_continuum
from scripts.audit_q4_cantilever_airy import INTERIOR_X, INTERIOR_Y, errors as _errors

errors = functools.lru_cache(maxsize=None)(_errors)

D = torch.float64
MODEL = AiryCantilever()
TOL = 1e-8


def _finite_difference(field, points, h=1e-7):
    ex = torch.tensor([h, 0.0], dtype=D)
    ey = torch.tensor([0.0, h], dtype=D)
    return ((field(points + ex) - field(points - ex)) / (2 * h),
            (field(points + ey) - field(points - ey)) / (2 * h))


def test_stress_field_is_in_pointwise_equilibrium():
    points = torch.tensor([[0.2, 0.031], [0.55, -0.022], [0.9, 0.004]], dtype=D)
    dsx, dsy = _finite_difference(MODEL.stress, points)
    scale = float(MODEL.stress(points)[:, 0].abs().max()) / MODEL.length
    assert torch.max(torch.abs(dsx[:, 0] + dsy[:, 2])) / scale < 1e-6
    assert torch.max(torch.abs(dsx[:, 2] + dsy[:, 1])) / scale < 1e-6


def test_strain_field_is_compatible_with_displacement():
    points = torch.tensor([[0.4, 0.03], [0.75, -0.043]], dtype=D)
    dux, duy = _finite_difference(MODEL.displacement, points)
    from_displacement = torch.stack((dux[:, 0], duy[:, 1], dux[:, 1] + duy[:, 0]), 1)
    from_stress = MODEL.strain(points)
    assert torch.max(torch.abs(from_displacement - from_stress)) < 1e-12


def test_closed_form_strain_energy_matches_independent_integral():
    c, inertia, load = MODEL.half_height, MODEL.second_moment, MODEL.end_shear
    t, e, nu = MODEL.thickness, MODEL.young_modulus, MODEL.poisson
    shear_g = e / (2 * (1 + nu))
    bending = t / (2 * e) * (load**2 / inertia**2) * (MODEL.length**3 / 3) * (2 * c**3 / 3)
    shear = t / (2 * shear_g) * (load**2 / (4 * inertia**2)) * MODEL.length * (16 * c**5 / 15)
    assert MODEL.strain_energy() == pytest.approx(bending + shear, rel=1e-12)
    assert float(MODEL.stress(torch.tensor([[0.0, 0.0]]))[0, 1]) == 0.0


def test_boundary_tractions_are_self_equilibrated_and_carry_the_end_shear():
    nodes, elements = rectangular_q4_mesh(MODEL.length, MODEL.height, 8, 4)
    forces = boundary_nodal_forces(nodes, elements, MODEL)
    scale = abs(MODEL.end_shear)
    # A body in equilibrium: the *net* traction resultant must vanish ...
    assert abs(float(forces[:, 0].sum())) / scale < 1e-12
    assert abs(float(forces[:, 1].sum())) / scale < 1e-12
    moment = (nodes[:, 0] * forces[:, 1] - nodes[:, 1] * forces[:, 0]).sum()
    assert abs(float(moment)) / (scale * MODEL.length) < 1e-12
    # ... while the loaded end section alone carries the full shear load.
    end = torch.isclose(nodes[:, 0], torch.tensor(MODEL.length, dtype=D))
    assert float(forces[end, 1].sum()) / MODEL.end_shear == pytest.approx(1.0, rel=1e-12)
    assert float(forces[end, 0].sum()) / scale < 1e-12
    resultant, bending = traction_resultant(MODEL)
    assert float(resultant[1]) == pytest.approx(MODEL.end_shear, rel=1e-12)
    assert abs(float(bending[2])) < abs(MODEL.end_shear) * MODEL.length


@pytest.mark.parametrize("mesh", [(20, 4), (80, 16)])
def test_finite_element_field_reproduces_the_exact_solution(mesh):
    case = errors(mesh, MODEL)
    fine = mesh == (80, 16)
    gate = .03 if fine else .10
    assert case["sigma_x"]["interior_max_relative"] < gate
    assert case["tau_xy"]["interior_max_relative"] < gate
    assert case["displacement"]["max_relative"] < gate
    assert case["strain_energy_relative"] < gate
    assert 0 < case["interior_element_fraction"] < 1


def test_measured_convergence_order_is_second_order():
    coarse = errors((40, 8), MODEL)
    fine = errors((80, 16), MODEL)
    order = math.log(coarse["sigma_x"]["interior_max_relative"]
                     / fine["sigma_x"]["interior_max_relative"]) / math.log(2)
    energy_order = math.log(coarse["strain_energy_relative"]
                            / fine["strain_energy_relative"]) / math.log(2)
    assert order > 1.8, order
    assert energy_order > 1.8, energy_order


def test_corner_band_failure_is_locked_as_scope_limitation():
    """Element centres next to a corner do *not* meet 3% on the shear stress.

    The exact data is discontinuous at the corners (parabolic edge shear meets
    the end-section bending stress), so the local interpolation error is a
    property of the reference problem, not of the element.  Locking the failure
    keeps the published qualification scope honest.
    """
    case = errors((80, 16), MODEL)
    excluded = 1 - case["interior_element_fraction"]
    assert excluded > 0
    assert case["tau_xy"]["all_elements_max_relative"] > .03
    worst = case["tau_xy"]["worst_point_m"]
    x, y = worst
    # the worst point sits in the corner band excluded by the gate
    assert x < INTERIOR_X[0] * MODEL.length or x > INTERIOR_X[1] * MODEL.length \
        or abs(y) > INTERIOR_Y * MODEL.half_height


def test_euler_bernoulli_reference_explains_only_part_of_the_tip_deflection():
    beam_tip = MODEL.end_shear * MODEL.length**3 / (3 * MODEL.young_modulus * MODEL.second_moment)
    model_error = abs(beam_tip / MODEL.tip_deflection() - 1)
    assert 0.005 < model_error < 0.02      # ~1% shear-deformation model error
    # sigma_x of Euler--Bernoulli *is* the exact elasticity field
    y = MODEL.half_height
    beam_root = -MODEL.end_shear * MODEL.length * y / MODEL.second_moment
    exact_root = float(MODEL.stress(torch.tensor([[0.0, y]]))[0, 0])
    # identical up to float64 rounding of the half-height coordinate
    assert beam_root == pytest.approx(exact_root, rel=1e-7)