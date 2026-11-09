"""Verification of ``tensorfem.open_section_torsion`` (open thin-walled profiles).

The module exists because ``marine_stiffener_failure`` had to be handed
``torsion_constant`` / ``warping_constant`` by hand.  The calculators below are
therefore pinned against quantities that do **not** come from the same code:

* closed forms of thin-walled theory (I-section ``Cw = tf b^3 h^2/24``, channel
  ``e = 3b^2/(6b+h)`` and ``Cw = (t b^3 h^2/12)(3b+2h)/(6b+h)``, zero warping for
  a T and for an equal-leg angle) -- machine precision is expected because the
  quadrature is exact for these integrands;
* structural symmetry (SC of an I / Z section is the centroid, T on the flange
  line, angle at the leg-centreline corner);
* traversal invariance (a different root wall must give the same SC and Cw);
* the exact Saint-Venant rectangle series for the flat-bar ``J``;
* the *finite difference* multiply connected Prandtl solver of
  ``tensorfem.box_torsion`` for the closed-contour Bredt branch;
* a 3-D HEX8 torsion solve for ``J`` of a real profile, which knows nothing
  about thin-wall theory.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem import box_torsion as bt
from tensorfem.marine_stiffener_failure import lateral_torsional_buckling_screen
from tensorfem.open_section_torsion import (
    angle_section,
    box_section,
    channel_section,
    fem_torsion_constant,
    flat_bar,
    i_section,
    prism_extrude,
    properties,
    rect_block_mesh,
    section_properties,
    stiffener_properties,
    tee_section,
    z_section,
)
from tensorfem.saint_venant_torsion import RectSection

H, B, TW, TF = 0.30, 0.12, 0.008, 0.012


def test_i_section_warping_and_inertia_match_closed_forms():
    props = properties(i_section(H, B, TW, TF))
    cw_exact = TF * B ** 3 * H ** 2 / 24.0
    assert props.warping_constant / cw_exact - 1.0 == pytest.approx(0.0, abs=1e-12)
    ix_exact = 2.0 * TF * B * (0.5 * H) ** 2 + TW * H ** 3 / 12.0
    assert props.i_xx / ix_exact - 1.0 == pytest.approx(0.0, abs=1e-12)
    assert props.i_xy == pytest.approx(0.0, abs=1e-18)
    # double symmetry: the shear centre is the centroid
    assert props.shear_centre == pytest.approx(props.centroid, abs=1e-12)
    assert props.shear_centre[1] == pytest.approx(0.5 * H, abs=1e-12)


def test_channel_shear_centre_and_warping_match_timoshenko():
    t = 0.01
    props = properties(channel_section(H, B, t, t))
    # shear centre behind the web, distance 3b^2/(6b+h)
    assert props.shear_centre[0] == pytest.approx(-3.0 * B ** 2 / (6.0 * B + H), abs=1e-12)
    assert props.shear_centre[1] == pytest.approx(0.5 * H, abs=1e-12)
    cw_exact = (t * B ** 3 * H ** 2 / 12.0) * (3.0 * B + 2.0 * H) / (6.0 * B + H)
    assert props.warping_constant / cw_exact - 1.0 == pytest.approx(0.0, abs=1e-12)
    # equal thickness makes the centroid offset b^2/(h+2b) from the web
    assert props.centroid[0] == pytest.approx(B ** 2 / (H + 2.0 * B), abs=1e-12)


def test_tee_and_angle_have_zero_warping_in_thin_wall_theory():
    tee = properties(tee_section(H, B, TW, TF))
    # every wall of a T passes through the flange centreline -> w == 0 there
    assert tee.shear_centre == pytest.approx((0.0, H), abs=1e-12)
    assert tee.warping_constant == pytest.approx(0.0, abs=1e-30)
    angle = properties(angle_section(0.10, 0.12, 0.008))
    # equal-leg-like angle: shear centre at the leg-centreline corner
    assert angle.shear_centre == pytest.approx((0.0, 0.0), abs=1e-12)
    assert angle.warping_constant == pytest.approx(0.0, abs=1e-30)


def test_z_section_point_symmetry_puts_shear_centre_at_centroid():
    props = properties(z_section(H, B, TW, TF))
    assert props.shear_centre == pytest.approx(props.centroid, abs=1e-12)
    assert props.warping_constant > 0.0
    # 180-degree rotation maps the section onto itself and leaves w invariant
    w = np.asarray(props.node_warping)
    assert w[2] == pytest.approx(w[3], rel=1e-12)


def test_properties_are_invariant_to_the_traversal_root():
    t = 0.01
    a = properties(channel_section(H, B, t, t))
    nodes = [(B, 0.0), (0.0, 0.0), (0.0, H), (B, H)]
    b = section_properties(nodes, edges=[(0, 1, t), (1, 2, t), (2, 3, t)])
    assert b.shear_centre == pytest.approx(a.shear_centre, abs=1e-12)
    assert b.warping_constant / a.warping_constant - 1.0 == pytest.approx(0.0, abs=1e-12)


def test_flat_bar_torsion_constant_against_exact_rectangle_series():
    width, thick = 0.10, 0.01
    props = properties(flat_bar(width, thick))
    exact = RectSection(0.5 * width, 0.5 * thick).j
    # j_rect *is* the rectangle series, j_thin is the leading-order bt^3/3
    assert props.j_rect / exact - 1.0 == pytest.approx(0.0, abs=1e-12)
    assert props.j_thin == pytest.approx(width * thick ** 3 / 3.0, rel=1e-14)
    assert props.j_thin / exact - 1.0 == pytest.approx(0.06675, abs=2e-3)
    # a single straight wall is a degenerate pole: documented, not raised
    assert props.notes["degenerate_pole"] is True
    assert props.warping_constant == 0.0


def test_closed_box_bredt_against_finite_difference_prandtl():
    t = 0.02
    props = properties(box_section(0.20, 0.20, t))
    assert props.closed and props.j_bredt is not None
    fd = bt.solve_box_prandtl(0.11, 0.11, t, 0.005)      # multiply connected Prandtl
    for j_fd in (fd.j_energy, fd.j_torque):              # two independent FD identities
        # Bredt is the thin-wall limit; a 20 %-thick wall still adds its own
        # open-section torsion, so the exact value runs a few per cent higher
        assert abs(props.j_bredt / j_fd - 1.0) < 0.10
        assert j_fd > props.j_bredt


def test_hex8_torsion_solve_brackets_the_flat_bar():
    xs = np.linspace(0.0, 0.10, 5)
    ys = np.linspace(0.0, 0.01, 2)
    nodes2d, quads = rect_block_mesh(xs, ys)
    nodes, hexes = prism_extrude(nodes2d, quads, 0.5, 4)
    res = fem_torsion_constant(nodes, hexes)
    exact = RectSection(0.05, 0.005).j
    thin = 0.10 * 0.01 ** 3 / 3.0
    assert exact < res["j_fem"] < thin
    assert res["j_fem"] / exact - 1.0 < 0.05
    assert res["dofs"] == 3 * len(nodes)


def test_stiffener_scantlings_feed_the_lateral_torsional_buckling_screen():
    props = stiffener_properties(0.25, 0.010, 0.090, 0.012)
    assert props.j > 0.0 and props.warping_constant >= 0.0
    screen = lateral_torsional_buckling_screen(
        segments=40, length=3.0, torsion_constant=props.j,
        warping_constant=props.warping_constant, weak_axis_inertia=props.i_yy)
    assert screen["critical_moment"] > 0.0
    assert screen["relative_error"] < 0.01
    # the T profile has (thin-wall) zero warping, so the LTB moment reduces to
    # the pure Saint-Venant form; the old placeholder Cw = 2.5e-8 was 3x larger
    # than the real warping constant of a 250x90 T stiffener
    assert props.warping_constant < 1e-8


def test_input_validation():
    with pytest.raises(ValueError):
        section_properties([(0, 0), (1, 0)], edges=[(0, 1, 0.0)])
    with pytest.raises(ValueError):
        section_properties([(0, 0), (0, 0)], edges=[(0, 1, 0.01)])
    with pytest.raises(ValueError):
        section_properties([(0, 0), (1, 0)], edges=[(0, 5, 0.01)])
    with pytest.raises(ValueError):
        # a loop is not an open section
        section_properties([(0, 0), (1, 0), (1, 1), (0, 1)],
                           edges=[(0, 1, 0.01), (1, 2, 0.01), (2, 3, 0.01), (3, 0, 0.01)])
    with pytest.raises(ValueError):
        # closed sections must repeat the first node
        section_properties([(0, 0), (1, 0), (1, 1), (0, 1)],
                           edges=[(0, 1, 0.01), (1, 2, 0.01), (2, 3, 0.01), (3, 0, 0.01)],
                           closed=True)
    with pytest.raises(ValueError):
        section_properties([(0, 0), (1, 0), (2, 0)], edges=[(0, 1, 0.01)], name="disconnected")
