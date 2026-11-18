"""Tests for the discrete mechanism (yield-line LP) upper bound and for the
acceptance-criterion discrimination gate.

The mechanism LP is used as an *independent* yardstick for the closed-form
plastic bounds of ``tensorfem.plate_patch_plastic_limit``; the tests therefore
pin the two cases whose collapse load is known exactly and pin the hinge
assembly against an independently computed dihedral angle.
"""
import numpy as np
import pytest

from scripts.plate_mechanism_lp_bound import _clip_area, mechanism_upper_bound
from tensorfem.plate_patch_plastic_limit import (
    criterion_discrimination,
    plastic_bound_bracket,
    plastic_moment,
    strip_lower_bound,
)

A, B, T = 2.0, 1.0, 0.020
PATCH = (0.5, 0.5)
SY = 400e6


def _independent_dihedral(result, x_slacks):
    """Dihedral of every hinge from node values only (no LP rows involved)."""
    nodes, tris, grads, w = (result["nodes"], result["tris"],
                             result["grads"], result["w"])
    out = []
    for (p, q), members in result["hinge_list"]:
        normal = np.array([nodes[q][1] - nodes[p][1], -(nodes[q][0] - nodes[p][0])])
        normal /= np.linalg.norm(normal)
        slopes = [float((grads[ti] @ normal) @ np.array([w[n] for n in tris[ti]]))
                  for ti, _opp in members]
        out.append(abs(slopes[0] - slopes[1]))
    return np.array(out), np.array(x_slacks)


@pytest.mark.parametrize("n", [6, 8, 12])
def test_square_plate_recovers_exact_johansen_collapse(n):
    """Uniformly loaded simply supported square plate: p_u = 24 m_p/a^2 exactly."""
    res = mechanism_upper_bound(1.0, 1.0, 1.0, (1.0, 1.0), nx=n, ny=n)
    assert res["pressure"] == pytest.approx(24.0, rel=1e-9)


@pytest.mark.parametrize("nx,ny", [(4, 4), (8, 8), (12, 12)])
def test_square_plate_bound_is_never_below_the_exact_load(nx, ny):
    res = mechanism_upper_bound(1.0, 1.0, 1.0, (1.0, 1.0), nx=nx, ny=ny)
    assert res["pressure"] >= 24.0 - 1e-9


def test_hinge_assembly_matches_independent_dihedral():
    """Guards the ``theta = |s1 - s2|`` sign: summing the two triangles'
    coefficient vectors cancels them on symmetric mechanisms and would report a
    zero-dissipation mechanism (p_u = 0)."""
    res = mechanism_upper_bound(2.0, 1.0, 1.0, (0.5, 0.5), nx=8, ny=4)
    indep, slacks = _independent_dihedral(res, res["slacks"])
    assert np.abs(indep - slacks).max() < 1e-9
    assert indep.max() > 0.0


def test_long_plate_bound_sits_inside_the_closed_form_bracket():
    """4x0.5 uniformly loaded plate: the exact one-way value 8 m_p/b^2 = 32 is
    the a -> inf limit, the strip field gives the lower bound 32.5 here, so a
    valid bound must lie above it (and well below the crude one-way guess)."""
    mp = 1.0
    res = mechanism_upper_bound(4.0, 0.5, mp, (4.0, 0.5), nx=16, ny=4)
    lower = 8 * mp / 0.25 + 8 * mp / 16.0
    assert lower <= res["pressure"] < 1.5 * lower


def test_anchor_geometry_bound_is_admissible_but_not_tighter_than_closed_form():
    """The closed-form pyramid ridge is not a grid edge, so at reachable
    resolutions the LP bound is looser than the closed form: the LP confirms the
    closed-form value as the reported reference instead of replacing it."""
    mp = plastic_moment(T, SY)
    res = mechanism_upper_bound(A, B, mp, PATCH, nx=24, ny=12)
    bracket = plastic_bound_bracket(A, B, T, SY, PATCH, terms=3)
    assert res["pressure"] > bracket["tightest_admissible_upper"] * 1.03
    assert res["pressure"] > strip_lower_bound(A, B, T, SY, PATCH)["pressure"]


def test_patch_overlap_area_is_exact():
    inside = np.array([[0.2, 0.2], [0.4, 0.2], [0.2, 0.4]])
    assert _clip_area(inside, 0.1, 0.5, 0.1, 0.5) == pytest.approx(0.02)
    # half of a unit right triangle cut by the vertical line x = 0.5
    half = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert _clip_area(half, 0.5, 2.0, -1.0, 2.0) == pytest.approx(0.125)
    assert _clip_area(inside, 0.5, 0.6, 0.5, 0.6) == 0.0


# ---------------------------------------------------------------------------
# acceptance-criterion discrimination
# ---------------------------------------------------------------------------

def test_discrimination_flags_a_bracket_too_wide_for_the_tolerance():
    gate = criterion_discrimination(upper=2.194, lower=0.6095, tolerance=0.10)
    assert gate["bracket_width_ratio"] == pytest.approx(2.194 / 0.6095)
    assert gate["required_width_ratio"] == pytest.approx(1.0 / 0.9)
    assert gate["best_possible_relative_error"] == pytest.approx(
        1.0 - 0.6095 / 2.194)
    assert gate["discriminative"] is False
    assert gate["verdict"] == "not_discriminative"
    assert gate["required_tightening_factor"] > 3.0


def test_discrimination_accepts_a_tight_bracket():
    gate = criterion_discrimination(upper=1.04, lower=1.0, tolerance=0.10)
    assert gate["discriminative"] is True
    assert gate["verdict"] == "decidable"
    # already tighter than the 1/(1-tol) gate: no tightening required (<= 1)
    assert gate["required_tightening_factor"] <= 1.0


def test_discrimination_rejects_inverted_or_degenerate_brackets():
    with pytest.raises(ValueError):
        criterion_discrimination(upper=1.0, lower=2.0, tolerance=0.10)
    with pytest.raises(ValueError):
        criterion_discrimination(upper=1.0, lower=1.0, tolerance=0.0)


def test_anchor_bracket_is_not_discriminative_for_the_ledger_criterion():
    """GAP-PP-3 acceptance asks for <10 % against the upper bound; with the
    admissible bracket of the anchor geometry no correct solver can satisfy it."""
    bracket = plastic_bound_bracket(A, B, T, SY, PATCH, terms=3)
    gate = criterion_discrimination(
        upper=bracket["tightest_admissible_upper"],
        lower=bracket["lower_strip"], tolerance=0.10)
    assert gate["verdict"] == "not_discriminative"
    assert gate["best_possible_relative_error"] > 0.5