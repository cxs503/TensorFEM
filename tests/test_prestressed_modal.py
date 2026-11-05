"""Tests for prestressed (stress-stiffened) free vibration.

Every reference here is a closed form or an exact discrete identity; none is a
stored snapshot of this repository.
"""
import math

import pytest
import torch

from tensorfem.buckling import euler_pinned_critical_load, pinned_pinned_column_buckling
from tensorfem.modal import simply_supported_beam_modes
from tensorfem.plate_vibration import solve_plate_vibration
from tensorfem.prestressed_modal import (
    beam_stress_stiffened_modes,
    exact_beam_euler_load,
    exact_beam_tension_frequency,
    exact_plate_buckling_resultant,
    exact_plate_tension_frequency,
    plate_rigidity,
    plate_stress_stiffened_modes,
    q4_plate_geometric_stiffness,
    solve_stress_stiffened_modes,
)

TOL = 0.03


def relative(actual: float, reference: float) -> float:
    return abs(actual - reference) / abs(reference)


# ---------------------------------------------------------------------------
# Element operator
# ---------------------------------------------------------------------------

def _unit_square():
    return torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)


def test_element_operator_is_symmetric_linear_and_transverse_only():
    xy = _unit_square() * 0.5
    kg = q4_plate_geometric_stiffness(xy, (3.0, -1.0, 0.7))
    assert torch.allclose(kg, kg.T, atol=1e-14)
    # Grade-2 (transverse) dof motion is the only work-carrying field.
    rotation_dofs = torch.tensor([1, 2, 4, 5, 7, 8, 10, 11])
    assert torch.allclose(kg[rotation_dofs], torch.zeros_like(kg[rotation_dofs]))
    # Linearity in the resultants.
    a = q4_plate_geometric_stiffness(xy, (2.0, 1.0, 0.0))
    b = q4_plate_geometric_stiffness(xy, (1.0, 0.5, 0.0))
    assert torch.allclose(2.0 * b, a, atol=1e-14)
    assert torch.count_nonzero(q4_plate_geometric_stiffness(xy, (0.0, 0.0, 0.0))) == 0
    # Uniform tension gives a positive semidefinite operator; a compressive
    # resultant makes it indefinite, which is exactly how buckling enters.
    tension = q4_plate_geometric_stiffness(xy, (1.0, 1.0, 0.0))
    assert float(torch.linalg.eigvalsh(tension).min()) > -1e-12
    assert float(torch.linalg.eigvalsh(kg).min()) < -1e-3


def test_element_operator_rejects_degenerate_and_malformed_input():
    with pytest.raises(ValueError):
        q4_plate_geometric_stiffness(_unit_square(), (1.0, 1.0))
    collapsed = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    with pytest.raises(ValueError):
        q4_plate_geometric_stiffness(collapsed, (1.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        q4_plate_geometric_stiffness(_unit_square(), (1.0, 0.0, 0.0), integration="bogus")


def test_element_operator_reduces_to_bilinear_gradient_energy():
    # Independent scalar integral of the bilinear-gradient form on a 1x1 square.
    xy = _unit_square()
    kg = q4_plate_geometric_stiffness(xy, (1.0, 1.0, 0.0), integration="full")
    w = torch.tensor([0.0, 1.0, 3.0, 2.0], dtype=torch.float64)  # w = x + 2y (linear)
    dofs = torch.tensor([0, 3, 6, 9])
    energy = float(w @ kg[dofs[:, None], dofs] @ w)
    assert math.isclose(energy, 1.0**2 + 2.0**2, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Zero prestress must reproduce the already-verified stress-free solvers
# ---------------------------------------------------------------------------

def test_zero_prestress_reproduces_verified_beam_and_plate_modes():
    beam = beam_stress_stiffened_modes(2.0, 16, EI=5.0e4, rho_a=12.0, modes=3)
    reference = simply_supported_beam_modes(2.0, 16, 5.0e4, 12.0, 3)
    assert torch.allclose(beam.angular_frequencies, reference.angular_frequencies, rtol=1e-8)
    plate = plate_stress_stiffened_modes(6, 6, length=1.0, thickness=0.01, modes=2)
    plate_reference = solve_plate_vibration(6, 6, length=1.0, thickness=0.01, modes=2)
    assert torch.allclose(plate.angular_frequencies, plate_reference.angular_frequencies, rtol=1e-8)


# ---------------------------------------------------------------------------
# Closed-form continuum laws
# ---------------------------------------------------------------------------

def test_beam_tension_frequency_converges_to_closed_form():
    length, EI, rho_a = 2.0, 5.0e4, 12.0
    load = 0.5 * exact_beam_euler_load(1, EI=EI, length=length)
    errors = []
    for elements in (6, 12, 24):
        result = beam_stress_stiffened_modes(length, elements, EI=EI, rho_a=rho_a, axial_force=load, modes=1)
        exact = exact_beam_tension_frequency(1, EI=EI, rho_a=rho_a, length=length, axial_force=load)
        errors.append(relative(float(result.angular_frequencies[0]), exact))
    assert errors[-1] < TOL and errors[0] > errors[-1]
    order = math.log(errors[0] / errors[-1]) / math.log(4.0)
    assert order > 1.8


def test_beam_compression_softens_and_tension_stiffens_by_the_exact_law():
    length, EI, rho_a = 2.0, 5.0e4, 12.0
    load = 0.4 * exact_beam_euler_load(1, EI=EI, length=length)
    unloaded = beam_stress_stiffened_modes(length, 16, EI=EI, rho_a=rho_a, modes=1)
    tensioned = beam_stress_stiffened_modes(length, 16, EI=EI, rho_a=rho_a, axial_force=load, modes=1)
    compressed = beam_stress_stiffened_modes(length, 16, EI=EI, rho_a=rho_a, axial_force=-load, modes=1)
    assert float(unloaded.angular_frequencies[0]) < float(tensioned.angular_frequencies[0])
    assert float(compressed.angular_frequencies[0]) < float(unloaded.angular_frequencies[0])
    for load, result in ((load, tensioned), (-load, compressed)):
        exact = exact_beam_tension_frequency(1, EI=EI, rho_a=rho_a, length=length, axial_force=load)
        assert relative(float(result.angular_frequencies[0]), exact) < TOL


def test_beam_taut_string_limit():
    length, rho_a = 4.0, 7.5
    load = 4.0e6
    result = beam_stress_stiffened_modes(length, 32, EI=1.0e-2, rho_a=rho_a, axial_force=load, modes=1)
    string = (math.pi / length) * math.sqrt(load / rho_a)
    assert relative(float(result.angular_frequencies[0]), string) < TOL


def test_beam_critical_load_matches_euler_load():
    length, EI = 3.0, 2.0e5
    result = beam_stress_stiffened_modes(length, 32, EI=EI, rho_a=10.0, modes=1)
    assert relative(-result.critical_tension, euler_pinned_critical_load(length, EI, 1)) < 1e-3


def test_plate_uniaxial_prestress_follows_the_navier_law():
    kwargs = dict(length=1.0, width=0.8, young=200.0e9, poisson=0.3, thickness=0.01, density=7850.0)
    section = {k: v for k, v in kwargs.items() if k != "density"}
    load = 0.3 * exact_plate_buckling_resultant(1, 1, **section)
    errors = []
    for elements in (4, 6, 8):
        result = plate_stress_stiffened_modes(elements, elements, tension_x=load, modes=1, **kwargs)
        exact = exact_plate_tension_frequency(1, 1, tension_x=load, **kwargs)
        errors.append(relative(float(result.angular_frequencies[0]), exact))
    assert errors[-1] < TOL and errors[0] > errors[-1]


def test_plate_biaxial_prestress_follows_the_navier_law():
    kwargs = dict(length=1.0, width=1.0, young=70.0e9, poisson=0.33, thickness=0.004, density=2700.0)
    section = {k: v for k, v in kwargs.items() if k != "density"}
    load = 0.25 * exact_plate_buckling_resultant(1, 1, tension_y_ratio=0.5, **section)
    result = plate_stress_stiffened_modes(8, 8, tension_x=load, tension_y_ratio=0.5, modes=1, **kwargs)
    exact = exact_plate_tension_frequency(1, 1, tension_x=load, tension_y=0.5 * load, **kwargs)
    assert relative(float(result.angular_frequencies[0]), exact) < TOL


def test_plate_critical_load_matches_navier_buckling():
    kwargs = dict(length=1.0, width=1.0, young=200.0e9, poisson=0.3, thickness=0.01, density=7850.0)
    section = {k: v for k, v in kwargs.items() if k != "density"}
    result = plate_stress_stiffened_modes(10, 10, modes=1, **kwargs)
    exact = exact_plate_buckling_resultant(1, 1, **section)
    assert relative(result.critical_tension, exact) < TOL


# ---------------------------------------------------------------------------
# Exact discrete identity: the fundamental vanishes at the critical load
# ---------------------------------------------------------------------------

def test_fundamental_vanishes_at_the_critical_load_beam_and_plate():
    beam_zero = beam_stress_stiffened_modes(2.0, 12, EI=5.0e4, rho_a=12.0, modes=1)
    beam_crit = beam_stress_stiffened_modes(
        2.0, 12, EI=5.0e4, rho_a=12.0, axial_force=beam_zero.critical_tension, modes=1
    )
    scale = beam_zero.smallest_squared_frequency
    assert beam_crit.smallest_squared_frequency / scale < 1e-9

    plate_zero = plate_stress_stiffened_modes(6, 6, length=1.0, thickness=0.01, modes=1)
    plate_crit = plate_stress_stiffened_modes(
        6, 6, length=1.0, thickness=0.01, tension_x=plate_zero.critical_tension, modes=1
    )
    plate_scale = plate_zero.smallest_squared_frequency
    assert plate_crit.smallest_squared_frequency / plate_scale < 1e-9


def test_critical_load_is_consistent_with_the_buckling_module():
    length, EI = 2.0, 5.0e4
    stress = beam_stress_stiffened_modes(length, 16, EI=EI, rho_a=10.0, modes=1)
    buckling = pinned_pinned_column_buckling(length, 16, EI, modes=1)
    assert relative(-stress.critical_tension, float(buckling.load_factors[0])) < 1e-10


# ---------------------------------------------------------------------------
# Fail-closed behaviour and a negative control
# ---------------------------------------------------------------------------

def test_load_beyond_the_critical_load_is_rejected():
    plate = plate_stress_stiffened_modes(6, 6, length=1.0, thickness=0.01, modes=1)
    with pytest.raises(ValueError, match="buckling load"):
        plate_stress_stiffened_modes(
            6, 6, length=1.0, thickness=0.01, tension_x=1.05 * plate.critical_tension, modes=1
        )
    with pytest.raises(ValueError):
        exact_beam_tension_frequency(1, EI=1.0, rho_a=1.0, length=1.0, axial_force=-1.1 * math.pi**2)
    with pytest.raises(ValueError):
        exact_plate_buckling_resultant(1, 1, length=1.0, width=1.0, young=1.0, poisson=0.3,
                                       thickness=0.1, shear_ratio=0.2)


def test_a_wrong_sign_geometric_operator_breaks_the_closed_form_law():
    # Negative control: the same machinery with the geometric matrix mis-scaled
    # by 50% (a plausible assembly bug) must miss the exact law by more than the gate.
    length, EI, rho_a = 2.0, 5.0e4, 12.0
    load = 0.5 * exact_beam_euler_load(1, EI=EI, length=length)
    elements = 16
    kg = torch.zeros((2 * (elements + 1), 2 * (elements + 1)), dtype=torch.float64)
    from tensorfem.buckling import beam_geometric_stiffness_unit
    from tensorfem.modal import beam_bending_stiffness, beam_consistent_mass

    le = length / elements
    ke = beam_bending_stiffness(EI, le)
    me = beam_consistent_mass(rho_a, le)
    kge = 0.5 * beam_geometric_stiffness_unit(le)
    k = torch.zeros_like(kg)
    m = torch.zeros_like(kg)
    for element in range(elements):
        dofs = torch.tensor([2 * element, 2 * element + 1, 2 * element + 2, 2 * element + 3])
        k[dofs[:, None], dofs] += ke
        m[dofs[:, None], dofs] += me
        kg[dofs[:, None], dofs] += kge
    wrong = solve_stress_stiffened_modes(k, m, kg, (0, 2 * elements), load, modes=1)
    exact = exact_beam_tension_frequency(1, EI=EI, rho_a=rho_a, length=length, axial_force=load)
    assert relative(float(wrong.angular_frequencies[0]), exact) > 0.05


def test_plate_rigidity_and_reference_guards():
    assert math.isclose(plate_rigidity(young=200.0e9, poisson=0.3, thickness=0.01),
                        200.0e9 * 0.01**3 / (12.0 * (1.0 - 0.09)), rel_tol=1e-12)
    with pytest.raises(ValueError):
        plate_rigidity(young=1.0, poisson=0.5, thickness=0.1)
    with pytest.raises(ValueError):
        plate_stress_stiffened_modes(4, 4, boundary="free")
    with pytest.raises(ValueError):
        exact_plate_tension_frequency(1, 1, length=1.0, width=1.0, young=1.0, poisson=0.3,
                                      thickness=0.1, density=1.0, tension_x=-1.0e9)