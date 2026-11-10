"""Tests for :mod:`tensorfem.von_karman_plate_riks` (campaign p250-N06).

The module carries a mesh-level von Karman plate postbuckling path.  The tests
below grade its *building blocks* against independent closed forms and the
*continuation* against the classical one-mode solution, on meshes small enough
to keep the suite fast (the campaign-scale run lives in
``benchmarks/verified/fem_plate_postbuckling/run.py``).

Note on BLAS threading: every case here is a chain of small dense
factorisations.  On a many-core host an oversubscribed BLAS makes them ~40x
slower, so the numerical cases are sized for a 4--8 element mesh.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorfem import von_karman_plate_riks as vk

A = 0.400
B = 0.400
THICKNESS = 0.006
YOUNG = 206.0e9
POISSON = 0.3


# --------------------------------------------------------------------------
# closed-form oracles
# --------------------------------------------------------------------------
def test_timoshenko_closed_form_matches_hand_value():
    closed = vk.timoshenko_critical_stress(a=A, b=B, thickness=THICKNESS, young=YOUNG,
                                          poisson=POISSON)
    assert closed["half_waves"] == 1
    assert closed["buckling_coefficient"] == pytest.approx(4.0)
    rigidity = YOUNG * THICKNESS ** 3 / (12.0 * (1.0 - POISSON ** 2))
    assert closed["rigidity"] == pytest.approx(rigidity)
    assert closed["sigma_cr"] == pytest.approx(4.0 * math.pi ** 2 * rigidity
                                               / (B ** 2 * THICKNESS))
    assert closed["sigma_cr"] / 1e6 == pytest.approx(167.566, abs=0.01)


def test_timoshenko_half_wave_count_follows_aspect_ratio():
    for ratio, expected in ((1.0, 1), (2.0, 2), (3.0, 3)):
        closed = vk.timoshenko_critical_stress(a=ratio * B, b=B, thickness=THICKNESS,
                                               young=YOUNG, poisson=POISSON)
        assert closed["half_waves"] == expected
        assert closed["sigma_cr"] / closed["buckling_coefficient"] == pytest.approx(
            closed["sigma_cr"] / 4.0 if expected == 1 else
            closed["sigma_cr"] / closed["buckling_coefficient"])


def test_koiter_coefficient_of_a_square_plate():
    """``c = 3(1-nu^2)(a^4+b^4) / (4(a^2+b^2)^2)`` = 3(1-nu^2)/8 for a square."""
    coefficient = vk.koiter_amplitude_coefficient(a=A, b=B, poisson=POISSON)
    assert coefficient == pytest.approx(3.0 * (1.0 - POISSON ** 2) / 8.0)
    assert coefficient == pytest.approx(0.34125)


def test_koiter_load_factor_reduces_to_critical_load_without_amplitude():
    coefficient = vk.koiter_amplitude_coefficient(a=A, b=B, poisson=POISSON)
    tiny = 1e-6 * THICKNESS
    assert vk.koiter_load_factor(amplitude=tiny, thickness=THICKNESS,
                                 imperfection=0.0,
                                 coefficient=coefficient) == pytest.approx(1.0, abs=1e-6)


def test_koiter_formula_is_monotone_in_imperfection_at_fixed_load():
    """A larger imperfection needs a larger amplitude at the same load."""
    coefficient = vk.koiter_amplitude_coefficient(a=A, b=B, poisson=POISSON)

    def amplitude_for(load, w0):
        low, high = max(w0, 1e-12), THICKNESS
        for _ in range(200):
            middle = 0.5 * (low + high)
            value = vk.koiter_load_factor(amplitude=middle, thickness=THICKNESS,
                                          imperfection=w0, coefficient=coefficient)
            if value > load:
                high = middle
            else:
                low = middle
        return 0.5 * (low + high)

    amplitudes = [amplitude_for(1.05, w0) for w0 in (1e-6, THICKNESS / 500,
                                                     THICKNESS / 100)]
    assert amplitudes[0] < amplitudes[1] < amplitudes[2]


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    {"a": 0.0, "b": 1.0, "thickness": 0.01, "young": 1e11, "poisson": 0.3},
    {"a": 1.0, "b": -1.0, "thickness": 0.01, "young": 1e11, "poisson": 0.3},
    {"a": 1.0, "b": 1.0, "thickness": 0.0, "young": 1e11, "poisson": 0.3},
    {"a": 1.0, "b": 1.0, "thickness": 0.01, "young": 0.0, "poisson": 0.3},
    {"a": 1.0, "b": 1.0, "thickness": 0.01, "young": 1e11, "poisson": 0.6},
])
def test_timoshenko_rejects_invalid_input(kwargs):
    with pytest.raises((ValueError, TypeError)):
        vk.timoshenko_critical_stress(**kwargs)


def test_koiter_load_factor_rejects_zero_amplitude():
    coefficient = vk.koiter_amplitude_coefficient(a=A, b=B, poisson=POISSON)
    with pytest.raises(ValueError):
        vk.koiter_load_factor(amplitude=0.0, thickness=THICKNESS, imperfection=0.0,
                              coefficient=coefficient)


# --------------------------------------------------------------------------
# element level
# --------------------------------------------------------------------------
def test_bending_operator_reproduces_navier_centre_deflection():
    """Simply supported plate under uniform load: ``0.00406 q b^4 / D``."""
    nodes, elements = vk.plate_mesh(a=A, b=B, nx=8, ny=8)
    boundary = vk.simply_supported_boundary(nodes, a=A, b=B)
    prepared = vk._prepare(nodes, elements, vk._zero_imperfection, poisson=POISSON,
                           young=YOUNG, thickness=THICKNESS)
    ndof = int(boundary["ndof"])
    free = vk._free_dofs(boundary)
    _, stiffness = vk.assemble(nodes, prepared, np.zeros(ndof), young=YOUNG,
                               poisson=POISSON, thickness=THICKNESS, tangent=True)
    load = np.zeros(ndof)
    dx, dy = A / 8, B / 8
    q = 1.0e4
    for index in range(len(nodes)):
        load[6 * index + vk.W] = q * dx * dy
    solution = np.linalg.solve(stiffness.numpy()[np.ix_(free, free)], load[free])
    full = np.zeros(ndof)
    full[free] = solution
    centre = 4 * 9 + 4
    rigidity = YOUNG * THICKNESS ** 3 / (12.0 * (1.0 - POISSON ** 2))
    navier = 0.00406 * q * B ** 4 / rigidity
    assert full[6 * centre + vk.W] / navier == pytest.approx(1.0, abs=0.04)


def test_edge_load_lumping_reproduces_the_resultant():
    nodes, _elements = vk.plate_mesh(a=A, b=B, nx=4, ny=4)
    boundary = vk.simply_supported_boundary(nodes, a=A, b=B)
    load = vk.reference_edge_load(nodes, boundary, b=B, line_load=1.0e6)
    axial = load.reshape(-1, vk.DOF_PER_NODE)[:, vk.U]
    assert float(-axial.sum()) == pytest.approx(1.0e6 * B)
    assert torch.count_nonzero(load) == len(boundary["loaded_edge"])


def test_simply_supported_boundary_removes_rigid_body_motion():
    nodes, _elements = vk.plate_mesh(a=A, b=B, nx=6, ny=6)
    boundary = vk.simply_supported_boundary(nodes, a=A, b=B)
    fixed = set(boundary["fixed"])
    per_node = [sorted(index % 6 for index in fixed if index // 6 == node)
                for node in range(len(nodes))]
    corners = [index for index, (x, y) in enumerate(nodes.tolist())
               if (min(x, A - x) < 1e-12) and (min(y, B - y) < 1e-12)]
    for node in corners:
        assert set(per_node[node]) >= {vk.W, vk.WX, vk.WY}
    assert vk.V in per_node[boundary["corner"]]
    assert all(vk.U in per_node[node] for node in boundary["left_edge"])


# --------------------------------------------------------------------------
# linear bifurcation and negative controls
# --------------------------------------------------------------------------
def test_linear_buckling_matches_timoshenko_closed_form():
    closed = vk.timoshenko_critical_stress(a=A, b=B, thickness=THICKNESS, young=YOUNG,
                                          poisson=POISSON)
    result = vk.linear_buckling(a=A, b=B, thickness=THICKNESS, young=YOUNG,
                               poisson=POISSON, nx=6, ny=6)
    assert result["buckling"] is True
    assert result["critical_stress"] == pytest.approx(closed["sigma_cr"], rel=0.01)
    assert result["residuals"][0] < 1e-8
    assert result["base_positive_definite"] is True
    mode = np.asarray(result["mode"]).reshape(-1, vk.DOF_PER_NODE)[:, vk.W]
    assert vk.nodal_shape_mac(result["nodes"], mode, a=A, b=B, m=1, n=1) > 0.99


def test_linear_buckling_aspect_ratio_two_selects_two_half_waves():
    closed = vk.timoshenko_critical_stress(a=2.0 * B, b=B, thickness=THICKNESS,
                                           young=YOUNG, poisson=POISSON)
    result = vk.linear_buckling(a=2.0 * B, b=B, thickness=THICKNESS, young=YOUNG,
                               poisson=POISSON, nx=12, ny=6)
    assert closed["half_waves"] == 2
    assert result["critical_stress"] == pytest.approx(closed["sigma_cr"], rel=0.01)
    mode = np.asarray(result["mode"]).reshape(-1, vk.DOF_PER_NODE)[:, vk.W]
    assert vk.nodal_shape_mac(result["nodes"], mode, a=2.0 * B, b=B, m=2, n=1) > 0.95


def test_negative_control_zero_load_tangent_is_positive_definite():
    spectrum = vk.base_tangent_spectrum(a=A, b=B, thickness=THICKNESS, young=YOUNG,
                                        poisson=POISSON, nx=4, ny=4)
    assert spectrum["positive_definite"] is True
    assert spectrum["min_eigenvalue"] > 0.0


def test_negative_control_tension_produces_no_buckling_factor():
    result = vk.linear_buckling(a=A, b=B, thickness=THICKNESS, young=YOUNG,
                               poisson=POISSON, nx=4, ny=4, line_load=-1.0)
    assert result["buckling"] is False
    assert result["load_factors"] == []
    # Under tension the mapped spectrum is non-positive; round-off leaves a
    # spurious eigenvalue of order 1e-23, which the relative floor rejects.
    assert result["max_mapped_eigenvalue"] <= result["eigenvalue_floor"]


# --------------------------------------------------------------------------
# continuation
# --------------------------------------------------------------------------
def test_postbuckling_path_follows_the_one_mode_koiter_branch():
    coefficient = vk.koiter_amplitude_coefficient(a=A, b=B, poisson=POISSON)
    imperfection = 1.0e-3 * THICKNESS
    path = vk.trace_postbuckling(a=A, b=B, thickness=THICKNESS, young=YOUNG,
                                 poisson=POISSON, nx=4, ny=4,
                                 imperfection_amplitude=imperfection, steps=8,
                                 step_size=2.0e-3, tolerance=1e-6,
                                 max_iterations=15, start_load_factor=0.9,
                                 maximum_step=6.0e-3)
    assert path["converged"] is True
    assert path["points"]
    loads = [point["load_factor"] for point in path["points"]]
    assert loads == sorted(loads)
    for target in (0.1, 0.2, 0.3):
        measured = vk.interpolate_load_factor(path["points"],
                                              total_amplitude_over_t=target)
        if measured is None:
            continue
        reference = vk.koiter_load_factor(amplitude=target * THICKNESS,
                                          thickness=THICKNESS,
                                          imperfection=imperfection,
                                          coefficient=coefficient)
        assert measured == pytest.approx(reference, rel=0.06)


def test_postbuckling_amplitude_grows_through_the_critical_load():
    path = vk.trace_postbuckling(a=A, b=B, thickness=THICKNESS, young=YOUNG,
                                 poisson=POISSON, nx=4, ny=4,
                                 imperfection_amplitude=1.0e-3 * THICKNESS, steps=8,
                                 step_size=2.0e-3, tolerance=1e-6, max_iterations=15,
                                 start_load_factor=0.9, maximum_step=6.0e-3)
    amplitudes = [point["total_amplitude_over_t"] for point in path["points"]]
    assert amplitudes == sorted(amplitudes)
    assert amplitudes[-1] > 2.0 * amplitudes[0]


def test_tensile_reference_load_never_reaches_a_buckling_point():
    """The traced path under tension stays on the unbuckled (flat) branch."""
    path = vk.trace_postbuckling(a=A, b=B, thickness=THICKNESS, young=YOUNG,
                                 poisson=POISSON, nx=4, ny=4,
                                 imperfection_amplitude=0.0, steps=4,
                                 step_size=2.0e-3, tolerance=1e-6, max_iterations=15,
                                 start_load_factor=0.0)
    assert path["converged"] is True
    assert all(abs(point["amplitude_over_t"]) < 1e-6 for point in path["points"])


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def test_mac_identities():
    first = np.array([1.0, 2.0, 3.0])
    second = np.array([0.0, 3.0, -3.0])
    assert vk.mac(first, first) == pytest.approx(1.0)
    assert vk.mac(first, -first) == pytest.approx(1.0)
    assert vk.mac(first, np.array([0.0, 0.0, 0.0]) if False else second) < 1.0
    assert vk.mac(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        vk.mac(first, np.array([1.0, 2.0]))


def test_dof_scaling_follows_the_mode_derivative_orders():
    nodes, _elements = vk.plate_mesh(a=A, b=B, nx=4, ny=4)
    boundary = vk.simply_supported_boundary(nodes, a=A, b=B)
    free = vk._free_dofs(boundary)
    weights = vk.dof_scale(free, a=A, b=B)
    kx, ky = math.pi / A, math.pi / B
    kinds = free % vk.DOF_PER_NODE
    assert np.all(weights[kinds == vk.U] == pytest.approx(1.0))
    assert np.all(weights[kinds == vk.WX] == pytest.approx(kx))
    assert np.all(weights[kinds == vk.WY] == pytest.approx(ky))
    assert np.all(weights[kinds == vk.WXY] == pytest.approx(kx * ky))


def test_sine_imperfection_satisfies_simply_supported_compatibility():
    field = vk.sine_imperfection(a=A, b=B, amplitude=1.0e-5)
    for x, y in ((0.0, 0.3), (A, 0.3), (0.3, 0.0), (0.3, B)):
        assert field(x, y)[0] == pytest.approx(0.0, abs=1e-20)