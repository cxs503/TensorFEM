"""Scordelis--Lo / flat-facet Shell4: shear-rule rank diagnosis.

`benchmarks/verified/fem_shell_scordelis_lo` originally concluded that flat
facets "cannot capture curved-shell membrane-bending coupling", because the
qualified flat-facet ladder misses the published reference w = -0.3024 by
~50 %.

That attribution is falsified here.  With the *same* facets, load and
diaphragms -- only the transverse-shear integration rule changed -- the assumed
natural shear field brings the ladder inside 3 % at 16x16 and inside 1 % at
32x32, while the one-point rule carries three spurious zero-energy mechanisms
per element.  The operative defect is therefore a rank deficiency of the shear
block, i.e. an implementation property of the element, not a property of
faceting.

These tests pin the three legs of that claim: the rank census, the accuracy the
fix buys, and the fail-closed status of the default rule.
"""
import math

import pytest
import torch

from tensorfem.shell4_rank import element_rank_census, rigid_body_modes
from tensorfem.shell_benchmarks import scordelis_lo_roof

D = torch.float64
REFERENCE = -0.3024


def _facet() -> torch.Tensor:
    """One planar facet of the Scordelis--Lo (x, theta) mesh."""
    R = 25.0
    dx, dt = 1.0, math.radians(2.0)
    x0, t0 = 0.0, math.radians(-1.0)
    return torch.tensor(
        [
            [x0, R * math.sin(t0), R * math.cos(t0)],
            [x0 + dx, R * math.sin(t0), R * math.cos(t0)],
            [x0 + dx, R * math.sin(t0 + dt), R * math.cos(t0 + dt)],
            [x0, R * math.sin(t0 + dt), R * math.cos(t0 + dt)],
        ],
        dtype=D,
    )


@pytest.fixture(scope="module")
def census():
    xyz = _facet()
    return {
        shear: element_rank_census(xyz, 4.32e8, 0.0, 0.25, plate_shear=shear)
        for shear in ("sri", "mitc", "full")
    }


@pytest.fixture(scope="module")
def ladders():
    return {
        "sri": {n: scordelis_lo_roof(n, n, dtype=D) for n in (8, 16)},
        "mitc": {
            n: scordelis_lo_roof(n, n, dtype=D, plate_shear="mitc") for n in (8, 16)
        },
    }


def test_rigid_body_modes_are_six_and_span_the_null_space_of_rigid_body(census):
    for shear, c in census.items():
        assert c.rigid_basis_residual < 1e-10, shear
        assert c.zero_modes >= 6, shear


def test_default_one_point_shear_leaves_three_spurious_mechanisms(census):
    c = census["sri"]
    assert c.zero_modes == 9
    assert c.spurious_modes == 3
    # All three are the classic plate-shear hourglass: nodal w alternates
    # +a, -a, +a, -a around the element.
    for mechanism in c.mechanisms:
        w = mechanism[2::6]
        a = float(w[0])
        assert abs(a) > 1e-6
        assert abs(float(w[1]) + a) < 1e-9
        assert abs(float(w[2]) - a) < 1e-9
        assert abs(float(w[3]) + a) < 1e-9


def test_assumed_natural_shear_removes_the_two_plate_hourglass_modes(census):
    c = census["mitc"]
    assert c.zero_modes == 7
    assert c.spurious_modes == 1
    # The surviving mechanism is in-plane (no transverse displacement, no
    # section rotation): it does not corrupt the bending response measured here.
    v = c.mechanisms[0]
    assert float(torch.linalg.vector_norm(v[2::6])) < 1e-9
    assert float(torch.linalg.vector_norm(v[3::6])) < 1e-9
    assert float(torch.linalg.vector_norm(v[4::6])) < 1e-9
    assert float(torch.linalg.vector_norm(v[0::6])) > 1e-6


def test_equal_order_full_shear_is_locking_not_a_remedy(census, ladders):
    # Full integration also removes the plate hourglass modes, but it locks:
    # the discrete response collapses far below the reference.
    assert census["full"].spurious_modes == 1
    w = float(scordelis_lo_roof(8, 8, dtype=D, plate_shear="full").probe_displacement)
    assert abs(w) < 0.1 * abs(REFERENCE)


def test_flat_facet_passes_scordelis_lo_with_assumed_natural_shear(ladders):
    w16 = ladders["mitc"][16].probe_displacement
    err16 = abs(w16 / REFERENCE - 1.0)
    assert err16 < 0.03, f"16x16 mitc error {err16:.3%}"
    # and the ladder is genuinely convergent, not a lucky coarse point
    err8 = abs(ladders["mitc"][8].probe_displacement / REFERENCE - 1.0)
    assert err16 < err8
    assert err16 < 0.03 < err8


def test_default_one_point_shear_stays_fail_closed(ladders):
    # Guard: the historical default must not be advertised as passing until the
    # default itself is deliberately changed.  If this ever fails, the default
    # was improved -- update the anchor README and this guard together.
    err8 = abs(ladders["sri"][8].probe_displacement / REFERENCE - 1.0)
    assert err8 > 0.30


def test_plate_shear_keyword_is_backward_compatible():
    default = scordelis_lo_roof(4, 4, dtype=D)
    explicit = scordelis_lo_roof(4, 4, dtype=D, plate_shear="sri")
    assert default.probe_displacement == explicit.probe_displacement
    assert torch.equal(default.displacement, explicit.displacement)


def test_rigid_body_modes_layout_matches_the_element_dof_order():
    basis = rigid_body_modes(_facet())
    assert basis.shape == (24, 6)
    # first three columns: translations, one unit per node in a single component
    assert torch.allclose(basis[0::6, 0], torch.ones(4, dtype=D))
    assert torch.allclose(basis[1::6, 1], torch.ones(4, dtype=D))
    assert torch.allclose(basis[2::6, 2], torch.ones(4, dtype=D))
    assert float(basis[:, 3:].abs().sum()) > 0.0