"""Gates for the plane-stress continuum oracle of the quarter-ring benchmark.

The oracle exists to grade the curved-beam closed form in
``tensorfem.curved_beam_benchmark`` from a different theory family (2-D
elasticity, Q8 elements) and to grade the stress recovery the shell elements
feed into strength checks.  The tests below assert *invariants* of that oracle,
not merely that it runs:

* the Q8 element reproduces an exact constant-strain state (patch consistency)
  and every element of the structured mesh has a positive Jacobian;
* statics: the reaction force and reaction moment of the clamped ring equal the
  applied load and ``P R`` to machine precision;
* Maxwell--Betti reciprocity holds between the two load cases (a symmetry of the
  assembled stiffness that would expose a load-assembly bug);
* in the thin limit the Richardson-extrapolated continuum compliance agrees with
  the curved-beam closed form to better than 0.5 % -- the independent grading of
  the coefficient ``3 pi/4 - 2`` and its shear/axial companions;
* the continuum hoop stress at the inner/outer surface agrees with the
  Winkler--Bach curved-beam profile to better than 1 %, while the flat-beam
  formula under-reports the inner fibre by an amount that grows with ``t/R``.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorfem.curved_beam_benchmark import quarter_ring_reference
from tensorfem.curved_ring_continuum import (
    Q8_GAUSS,
    _plane_stress_matrix,
    _strain_matrix,
    hoop_stress_section,
    q8_shape,
    quarter_ring_q8_mesh,
    richardson_limit,
    section_forces,
    solve_quarter_ring,
    straight_beam_stress,
    winkler_bach_stress,
)

RADIUS = 1.0
WIDTH = 0.2
YOUNG = 2.0e11
POISSON = 0.3
LOAD = 1000.0


def _reference(ratio: float) -> dict:
    return quarter_ring_reference(RADIUS, WIDTH, ratio * RADIUS, YOUNG, POISSON, LOAD)


@pytest.fixture(scope="module")
def thin_ring():
    return solve_quarter_ring(
        24, 6, RADIUS, 0.05 * RADIUS, WIDTH, YOUNG, POISSON, LOAD
    )


def test_mesh_geometry_and_jacobians():
    mesh = quarter_ring_q8_mesh(8, 2, RADIUS, 0.2)
    assert mesh.elements.shape == (8 * 2, 8)
    # boundary line of a Q8 grid: (n_thick+1) corners + n_thick mid-side nodes
    assert mesh.root_nodes.size == 2 * mesh.n_thick + 1
    assert mesh.free_end_nodes.size == mesh.root_nodes.size
    radii = np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1])
    assert radii.min() == pytest.approx(0.9)
    assert radii.max() == pytest.approx(1.1)
    assert mesh.nodes[mesh.centroid_node].tolist() == pytest.approx([1.0, 0.0])
    # every element must be positively oriented (the Q8 ordering is radial-first)
    for element in mesh.elements:
        coordinates = mesh.nodes[element]
        for xi, eta, _ in Q8_GAUSS:
            _, determinant = _strain_matrix(xi, eta, coordinates)
            assert determinant > 0.0


def test_q8_constant_strain_patch():
    """Affine displacement field on a distorted element -> exact constant stress."""
    coordinates = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.85, 0.45],
            [0.25, 0.4],
            [0.5, 0.0],
            [0.96 + 0.1, 0.225],  # distorted mid-side nodes: the patch must still pass
            [0.55, 0.425],
            [0.125, 0.2],
        ]
    )
    strain = np.array([1.0e-3, -5.0e-4, 2.0e-4])
    constitutive = _plane_stress_matrix(YOUNG, POISSON)
    displacement = np.zeros(16)
    for node, (x, y) in enumerate(coordinates):
        displacement[2 * node] = strain[0] * x + 0.5 * strain[2] * y
        displacement[2 * node + 1] = strain[1] * y + 0.5 * strain[2] * x
    for xi, eta, _ in Q8_GAUSS:
        matrix, _ = _strain_matrix(xi, eta, coordinates)
        assert matrix @ displacement == pytest.approx(strain, rel=1e-12, abs=1e-18)
        shape, _ = q8_shape(xi, eta)
        assert shape.sum() == pytest.approx(1.0)


def test_section_forces_statics():
    phi = 0.25 * math.pi
    axial, moment = section_forces(RADIUS, LOAD, phi)
    assert axial == pytest.approx(-LOAD * math.cos(phi))
    assert moment == pytest.approx(-LOAD * RADIUS * (1.0 - math.cos(phi)))
    axial_root, moment_root = section_forces(RADIUS, LOAD, 0.5 * math.pi)
    assert axial_root == pytest.approx(0.0, abs=1e-12)
    assert moment_root == pytest.approx(-LOAD * RADIUS)


def test_equilibrium_and_reciprocity(thin_ring):
    assert thin_ring.equilibrium_error() < 1e-8
    assert thin_ring.moment_error() < 1e-8
    radial_case = solve_quarter_ring(
        24, 6, RADIUS, 0.05 * RADIUS, WIDTH, YOUNG, POISSON, LOAD, direction="radial"
    )
    cross = radial_case.tip_tangent
    direct = thin_ring.tip_radial
    assert cross == pytest.approx(direct, rel=1e-6)
    assert radial_case.equilibrium_error() < 1e-8
    # direct radial compliance must also match its own closed form
    assert radial_case.tip_radial == pytest.approx(
        _reference(0.05)["delta_radial_load"], rel=5e-3
    )


def test_thin_limit_matches_curved_beam_closed_form(thin_ring):
    reference = _reference(0.05)
    assert thin_ring.tip_tangent == pytest.approx(reference["delta_tangent"], rel=5e-3)
    assert thin_ring.tip_radial == pytest.approx(reference["delta_radial"], rel=5e-3)


def test_richardson_limit_reproduces_known_order():
    # x(n) = X + c/n^2 sampled at n = 8, 16, 32 -> second order recovered
    target, coefficient = 1.2345, -7.0
    values = [target + coefficient / n**2 for n in (8, 16, 32)]
    fit = richardson_limit(values)
    assert fit["observed_order"] == pytest.approx(2.0, rel=1e-9)
    assert fit["limit"] == pytest.approx(target, rel=1e-9)
    non_monotone = richardson_limit([1.0, 2.0, 1.5])
    assert math.isnan(non_monotone["observed_order"])


def test_winkler_bach_limits():
    """Curved-beam stress degenerates onto the flat-beam one as t/R -> 0."""
    moment = 1.0
    gaps = []
    for ratio in (0.02, 0.01, 0.005):
        thickness = ratio * RADIUS
        radii = np.array([RADIUS - 0.5 * thickness, RADIUS + 0.5 * thickness])
        curved = winkler_bach_stress(radii, RADIUS, thickness, WIDTH, 0.0, moment)
        flat = straight_beam_stress(radii, RADIUS, thickness, WIDTH, 0.0, moment)
        gaps.append(float(np.max(np.abs(curved / flat - 1.0))))
    assert gaps[0] > gaps[1] > gaps[2]  # the difference is O((t/R)^2)
    assert gaps[-1] < 1e-2
    # axial force alone gives a uniform hoop stress
    thickness = 0.5 * RADIUS
    radii = np.array([0.8, 1.0, 1.2])
    assert np.allclose(
        winkler_bach_stress(radii, RADIUS, thickness, WIDTH, LOAD, 0.0),
        LOAD / (WIDTH * thickness),
    )


def test_surface_stress_grades_winkler_bach(thin_ring):
    """Continuum surface hoop stress vs curved-beam theory and the flat-beam formula."""
    thickness = 0.2 * RADIUS
    solution = solve_quarter_ring(
        32, 8, RADIUS, thickness, WIDTH, YOUNG, POISSON, LOAD
    )
    section = hoop_stress_section(solution, row=16, fibres=0.95)
    axial, moment = section_forces(RADIUS, LOAD, section["angle"])
    for index in (0, -1):
        radius = section["radii"][index]
        hoop_fe = section["hoop_stress"][index]
        curved = winkler_bach_stress(
            np.array([radius]), RADIUS, thickness, WIDTH, axial, moment
        )[0]
        assert abs(hoop_fe / curved - 1.0) < 1e-2
    # the flat-beam formula is systematically low at the inner surface
    inner_radius = section["radii"][0]
    flat = straight_beam_stress(
        np.array([inner_radius]), RADIUS, thickness, WIDTH, axial, moment
    )[0]
    assert abs(flat / section["hoop_stress"][0] - 1.0) > 3.0e-2


def test_thick_section_drift_is_bounded():
    """Closed form stays within 3 % of the continuum limit up to t/R = 0.3."""
    values = []
    for n_arc, n_thick in ((16, 4), (32, 8)):
        solution = solve_quarter_ring(
            n_arc, n_thick, RADIUS, 0.3 * RADIUS, WIDTH, YOUNG, POISSON, LOAD
        )
        values.append(solution.tip_tangent)
    reference = _reference(0.3)["delta_tangent"]
    # the finest mesh alone is already inside the envelope
    assert abs(values[-1] / reference - 1.0) < 3e-2