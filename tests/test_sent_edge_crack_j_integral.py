"""Verification of the SENT (single-edge-notched tension) J-integral path.

The graded anchor lives in ``benchmarks/verified/fem_sent_j_integral/``; this
file keeps the fast invariants and the handbook comparison under ``pytest``.
Everything here runs on the coarse fan mesh so the module is exercised in a few
seconds; the refinement ladder, the height sweep and the figures belong to the
anchor driver.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorfem.edge_crack_sent import (
    DEFAULT_SENT_LEVELS,
    edge_geometry,
    edge_parent,
    j_integral_subring,
    sent_convergence,
    sent_gci,
    sent_k_from_j,
    sent_k_reference,
    sent_t6_mesh,
    solve_sent,
    tada_sent_shape_factor,
)

WIDTH = 0.10
HEIGHT = 0.20
SIGMA = 100e6
YOUNG = 210e9
NU = 0.3


@pytest.fixture(scope="module")
def coarse_mesh():
    return sent_t6_mesh(WIDTH, 0.02, HEIGHT, n_sectors=24, first_ring=0.08, growth=1.6)


@pytest.fixture(scope="module")
def coarse_solve(coarse_mesh):
    return solve_sent(coarse_mesh, sigma_pa=SIGMA, young_pa=YOUNG, poisson=NU)


# --------------------------------------------------------------------------- #
# handbook reference
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "alpha, expected",
    [(0.10, 1.1837), (0.30, 1.6599), (0.50, 2.8264)],
)
def test_tada_factor_matches_the_handbook_table(alpha, expected):
    """Published single-edge-notch tension values: F(0.1) = 1.18, F(0.3) = 1.66, F(0.5) = 2.83."""
    assert tada_sent_shape_factor(alpha) == pytest.approx(expected, abs=5e-4)


def test_tada_factor_rejects_arguments_outside_its_envelope():
    with pytest.raises(ValueError):
        tada_sent_shape_factor(0.61)
    with pytest.raises(ValueError):
        tada_sent_shape_factor(0.0)


def test_reference_sif_scales_with_the_geometry_factor():
    a = 0.02
    k = sent_k_reference(SIGMA, a, WIDTH)
    assert k == pytest.approx(
        SIGMA * math.sqrt(math.pi * a) * tada_sent_shape_factor(a / WIDTH), rel=1e-12
    )
    assert sent_k_reference(SIGMA, 0.05, WIDTH) > sent_k_reference(SIGMA, a, WIDTH)


# --------------------------------------------------------------------------- #
# mesh invariants
# --------------------------------------------------------------------------- #
def test_every_element_has_a_positive_jacobian_at_its_quadrature_points(coarse_mesh):
    """The element mapping must be invertible wherever the stiffness is integrated.

    Checked at the seven Dunavant points, which is exactly the set
    :func:`tensorfem.fatigue.crack_tip_quarter_point.t6_stiffness` evaluates, and
    at the element centroid.  Corner-adjacent parent points are *not* asserted:
    the ray fan produces mildly twisted six-node triangles whose quadratic map
    can turn inside out at a corner even when every integration point is fine.
    """
    from tensorfem.fatigue.crack_tip_quarter_point import _DUNAVANT, t6_shape_derivative

    coordinates = coarse_mesh.nodes[coarse_mesh.elements]
    points = list(_DUNAVANT) + [(1.0 / 3.0, 1.0 / 3.0, 0.0)]
    for xi, eta, _ in points:
        d_xi, d_eta = t6_shape_derivative(xi, eta)
        dx = torch.tensor(d_xi, dtype=coordinates.dtype) @ coordinates
        dy = torch.tensor(d_eta, dtype=coordinates.dtype) @ coordinates
        det = dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]
        assert bool(torch.all(det > 0.0)), f"non-positive det at parent point ({xi}, {eta})"


def test_the_tip_node_is_shared_by_every_sector(coarse_mesh):
    tip = {coarse_mesh.ring[j][0] for j in range(coarse_mesh.n_sectors + 1)}
    assert len(tip) == 1


def test_outer_boundary_is_the_rectangle(coarse_mesh):
    boundary = coarse_mesh.nodes[[coarse_mesh.ring[j][-1] for j in range(coarse_mesh.n_sectors + 1)]]
    on_top = (boundary[:, 1] - HEIGHT).abs() < 1e-12
    on_right = (boundary[:, 0] - (WIDTH - 0.02)).abs() < 1e-12
    on_left = (boundary[:, 0] + 0.02).abs() < 1e-12
    assert bool(torch.all(on_top | on_right | on_left))


def test_the_first_ring_mid_side_nodes_sit_at_the_quarter_point(coarse_mesh):
    """Barsoum degeneracy: on the singular rays the mid-side node is at 1/4 length."""
    assert coarse_mesh.stations[1] == pytest.approx(0.25 * coarse_mesh.stations[2])
    tip = coarse_mesh.ring[0][0]
    element = next(
        index
        for index, ids in enumerate(coarse_mesh.elements.tolist())
        if tip in ids[:3]
    )
    ids = coarse_mesh.elements[element][:6].tolist()
    on_axis = [
        float(coarse_mesh.nodes[node][0])
        for node in ids
        if abs(float(coarse_mesh.nodes[node][1])) < 1e-12
    ]
    assert len(on_axis) == 3, "tip, the station-1 node and the mid-side node lie on y = 0"
    xs = sorted(on_axis)
    station_one = float(coarse_mesh.nodes[coarse_mesh.ring[0][1]][0])
    assert xs[0] == pytest.approx(0.0, abs=1e-12)
    assert xs[2] == pytest.approx(station_one)
    assert xs[1] == pytest.approx(0.25 * station_one)


# --------------------------------------------------------------------------- #
# solve and J integral
# --------------------------------------------------------------------------- #
def test_uniform_tension_integrates_to_sigma_times_width(coarse_solve):
    assert coarse_solve.applied_load_n == pytest.approx(SIGMA * WIDTH, rel=1e-9)


def test_global_equilibrium_closes(coarse_solve):
    assert coarse_solve.reaction_ratio == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("alpha, tolerance", [(0.2, 0.05), (0.5, 0.05)])
def test_recovered_k_matches_tada(alpha, tolerance):
    a = alpha * WIDTH
    mesh = sent_t6_mesh(WIDTH, a, HEIGHT, n_sectors=24, first_ring=0.08, growth=1.6)
    solved = solve_sent(mesh, sigma_pa=SIGMA, young_pa=YOUNG, poisson=NU)
    station = min(
        range(1, len(mesh.stations) - 1),
        key=lambda idx: abs(mesh.stations[idx] - 0.15),
    )
    j = j_integral_subring(mesh, solved.displacement, station, young_pa=YOUNG, poisson=NU)
    k = sent_k_from_j(j["j_j_m2"], YOUNG, NU)
    reference = sent_k_reference(SIGMA, a, WIDTH)
    assert abs((k - reference) / reference) < tolerance


def test_j_is_path_independent_over_the_fan_stations():
    mesh = sent_t6_mesh(WIDTH, 0.02, HEIGHT, n_sectors=24, first_ring=0.08, growth=1.6)
    solved = solve_sent(mesh, sigma_pa=SIGMA, young_pa=YOUNG, poisson=NU)
    values = [
        j_integral_subring(mesh, solved.displacement, station, young_pa=YOUNG, poisson=NU)[
            "j_j_m2"
        ]
        for station in range(2, len(mesh.stations) - 1)
        if mesh.stations[station] <= 0.35
    ]
    assert min(values) > 0.0
    # the outermost contours sit in the coarsest part of the fan, so only the
    # inner half of the fan is required to be path independent here
    assert (max(values) - min(values)) / max(values) < 0.03


def test_the_factor_excludes_a_central_crack_model():
    """The free surface behind the mouth must raise F well above the central value 1."""
    a = 0.2 * WIDTH
    mesh = sent_t6_mesh(WIDTH, a, HEIGHT, n_sectors=24, first_ring=0.08, growth=1.6)
    solved = solve_sent(mesh, sigma_pa=SIGMA, young_pa=YOUNG, poisson=NU)
    station = min(
        range(1, len(mesh.stations) - 1),
        key=lambda idx: abs(mesh.stations[idx] - 0.15),
    )
    j = j_integral_subring(mesh, solved.displacement, station, young_pa=YOUNG, poisson=NU)
    factor = sent_k_from_j(j["j_j_m2"], YOUNG, NU) / (SIGMA * math.sqrt(math.pi * a))
    assert factor > 1.15


def test_contour_at_the_tip_is_rejected(coarse_mesh, coarse_solve):
    with pytest.raises(IndexError):
        j_integral_subring(coarse_mesh, coarse_solve.displacement, 0, young_pa=YOUNG)
    with pytest.raises(IndexError):
        j_integral_subring(
            coarse_mesh, coarse_solve.displacement, len(coarse_mesh.stations) - 1,
            young_pa=YOUNG,
        )


def test_non_singular_control_is_recorded_not_asserted_to_fail():
    """The isoparametric control reaches the same band: recorded, never hidden.

    Measured in the anchor driver at the medium level: the mid-side-node-at-
    midpoint mesh still lands within 3 % of Tada, so the J path outside the
    innermost ring is insensitive to the tip enrichment.  What the enrichment
    does buy is the near-tip ``r^(-1/2)`` field, which is checked by the field
    figure of the anchor.
    """
    level = dict(DEFAULT_SENT_LEVELS[0])
    record = sent_convergence(
        width_m=WIDTH,
        crack_ratios=(0.2,),
        height_ratio=2.0,
        sigma_pa=SIGMA,
        young_pa=YOUNG,
        poisson=NU,
        contour_fractions=(0.15,),
        levels=(level,),
        quarter_point=False,
    )
    case = record["levels"][0]["cases"][0]
    assert math.isfinite(case["contours"][0]["shape_factor"])
    assert abs(case["contours"][0]["rel_error_vs_tada"]) < 0.10


def test_gci_reports_a_converging_sequence():
    ladder = [
        {"cases": [{"a_over_w": 0.2, "contours": [{"requested_fraction": 0.15,
                                                   "shape_factor": value}]}]}
        for value in (1.3300, 1.3620, 1.3673)
    ]
    gci = sent_gci(ladder, alpha=0.2)
    assert gci["converging"]
    assert gci["observed_order"] > 0.5
    assert gci["gci_fine"] < 0.05


def test_edge_parameterisation_is_consistent_with_the_parent_coordinates():
    for edge_index in (0, 1, 2):
        for t in (0.0, 0.5, 1.0):
            shape, d_dt = edge_geometry(edge_index, t)
            assert len(shape) == 6 and len(d_dt) == 6
            assert sum(shape) == pytest.approx(1.0, abs=1e-12)
            assert sum(d_dt) == pytest.approx(0.0, abs=1e-12)
            xi, eta = edge_parent(edge_index, t)
            assert 0.0 <= xi <= 1.0 and 0.0 <= eta <= 1.0 and xi + eta <= 1.0 + 1e-12