"""Gates for :mod:`tensorfem.multicell_torsion` (multi-cell box torsion).

Every assertion re-derives its reference inside the test: the single-cell
degeneracy against the merged ``box_torsion`` solver, the two ``J`` identities
against the Prandtl field, the shared-web flow against the multi-cell Bredt
compatibility solution.  Nothing is read from the evidence JSON, so a drifted
number cannot pass.
"""
import math

import numpy as np
import pytest

from tensorfem import multicell_torsion as mc
from tensorfem.box_torsion import solve_box_prandtl


SINGLE = mc.RectMultiCellSection(a=1.0, b=0.6, wall=0.1)
SYM2 = mc.RectMultiCellSection(a=1.0, b=0.6, wall=0.1, webs=((0.0, 0.1),))
ASYM2 = mc.RectMultiCellSection(a=1.0, b=0.6, wall=0.1, webs=((0.6, 0.1),))
THREE = mc.RectMultiCellSection(a=1.5, b=0.6, wall=0.1,
                                webs=((-0.5, 0.1), (0.5, 0.1)))


# --------------------------------------------------------------------------- #
# geometry and validation
# --------------------------------------------------------------------------- #
def test_geometry_and_area():
    assert SYM2.n_cells == 2 and THREE.n_cells == 3
    holes = SYM2.holes()
    assert len(holes) == 2
    assert holes[0] == pytest.approx((-0.9, -0.05, -0.5, 0.5))
    assert holes[1] == pytest.approx((0.05, 0.9, -0.5, 0.5))
    # material area = outer rectangle minus the two compartments
    assert SYM2.area == pytest.approx(2.4 - 2 * 0.85 * 1.0)
    # mid-line cells share the web centre line
    mids = SYM2.cell_midlines()
    assert mids[0][1] == pytest.approx(0.0) and mids[1][0] == pytest.approx(0.0)
    assert mids[0][0] == pytest.approx(-0.95)


def test_geometry_validation():
    with pytest.raises(ValueError):
        mc.RectMultiCellSection(a=0.1, b=0.6, wall=0.1)
    with pytest.raises(ValueError):
        mc.RectMultiCellSection(a=1.0, b=0.6, wall=0.1, webs=((0.6, 0.65),))
    with pytest.raises(ValueError):
        mc.RectMultiCellSection(a=1.0, b=0.6, wall=0.1,
                                webs=((0.6, 0.1), (0.2, 0.1)))
    with pytest.raises(ValueError):
        SYM2.grid(0.07)  # does not divide the geometry


def test_mesh_is_conforming_and_positive():
    xy, quads = mc.multicell_cross_section(ASYM2, 0.05)
    p = xy.numpy()
    q = quads.numpy()
    # no duplicated nodes
    assert len({(round(x, 9), round(y, 9)) for x, y in p}) == len(p)
    # every quad is counter-clockwise with positive area
    for quad in q:
        pts = p[quad]
        area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1)
                            - np.roll(pts[:, 0], -1) * pts[:, 1])
        assert area > 0.0
    # total quad area equals the analytic material area
    total = 0.0
    for quad in q:
        pts = p[quad]
        total += 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1)
                              - np.roll(pts[:, 0], -1) * pts[:, 1])
    assert total == pytest.approx(ASYM2.area, rel=1e-12)


# --------------------------------------------------------------------------- #
# the FD oracle
# --------------------------------------------------------------------------- #
def test_single_cell_matches_box_torsion_oracle():
    """n_cell = 1 must reproduce the merged single-cell module bit for bit."""
    field = mc.solve_multicell_prandtl(SINGLE, 0.05)
    box = solve_box_prandtl(1.0, 0.6, 0.1, 0.05)
    assert field.j_torque == pytest.approx(box.j_torque, rel=1e-10)
    assert field.j_energy == pytest.approx(box.j_energy, rel=1e-10)
    assert field.constants[0] == pytest.approx(box.c, rel=1e-10)


def test_two_identities_and_circulation():
    for section, h in ((SYM2, 0.05), (ASYM2, 0.025), (THREE, 0.05)):
        field = mc.solve_multicell_prandtl(section, h)
        assert field.compat_residual < 1e-10
        # energy and torque identities are independent functionals of phi
        assert abs(field.j_energy / field.j_torque - 1.0) < 3.5e-2
        # the hole constants are the Dirichlet values actually realised
        for rect, c in zip(section.holes(), field.constants):
            x0, x1, y0, y1 = rect
            j0 = int(round((x0 - field.xs[0]) / h))
            j1 = int(round((x1 - field.xs[0]) / h))
            i0 = int(round((y0 - field.ys[0]) / h))
            i1 = int(round((y1 - field.ys[0]) / h))
            assert field.phi[i0, j0] == pytest.approx(c, abs=1e-12)
            assert field.phi[i1, j1] == pytest.approx(c, abs=1e-12)


def test_symmetric_case_is_even_in_x():
    field = mc.solve_multicell_prandtl(SYM2, 0.025)
    nx = len(field.xs)
    left = field.phi[:, : nx // 2]
    right = field.phi[:, nx: nx // 2: -1]
    assert np.abs(left - right).max() < 1e-12
    assert field.constants[0] == pytest.approx(field.constants[1], rel=1e-12)


def test_grid_convergence_is_monotone():
    js = [mc.solve_multicell_prandtl(ASYM2, h).j_torque
          for h in (0.05, 0.025)]
    # the coarse grid is the stiffer-singular one; both must agree to a few %
    assert abs(js[0] / js[1] - 1.0) < 1e-2


# --------------------------------------------------------------------------- #
# the multi-cell Bredt reference
# --------------------------------------------------------------------------- #
def test_bredt_single_cell_limit():
    bredt = mc.bredt_multicell(SINGLE)
    area = (2.0 * 1.0 - 0.1) * (2.0 * 0.6 - 0.1)
    perim = 4.0 * ((1.0 - 0.05) + (0.6 - 0.05))
    assert bredt.j == pytest.approx(4.0 * area ** 2 * 0.1 / perim, rel=1e-12)


def test_bredt_compatibility_equations_hold():
    bredt = mc.bredt_multicell(ASYM2)
    for j, (x0, x1, y0, y1) in enumerate(ASYM2.cell_midlines()):
        area = (x1 - x0) * (y1 - y0)
        total = 0.0
        for wall in bredt.walls:
            if j not in wall["cells"]:
                continue
            # wall_flow is measured along the first member cell's direction
            sign = 1.0 if j == wall["cells"][0] else -1.0
            total += sign * bredt.wall_flow(wall) * wall["length"] / wall["thickness"]
        assert total == pytest.approx(2.0 * area, rel=1e-12)
    # torque identity: T = 2 sum A_i q_i
    torque = 2.0 * sum(a * q for a, q in
                       zip(bredt.cell_areas, bredt.cell_flows))
    assert bredt.j == pytest.approx(torque, rel=1e-12)


def test_symmetric_section_web_flow_vanishes():
    bredt = mc.bredt_multicell(SYM2)
    web = [w for w in bredt.walls if w["label"] == "web0"][0]
    assert abs(bredt.wall_flow(web)) < 1e-12
    # two identical cells: the same flow as the single cell of the same outline
    assert bredt.j == pytest.approx(mc.bredt_multicell(SINGLE).j, rel=1e-12)


def test_naive_independent_cell_sum_is_wrong():
    """Treating the cells as independent tubes double-counts the shared wall."""
    bredt = mc.bredt_multicell(SYM2)
    naive = 0.0
    for (x0, x1, y0, y1) in SYM2.cell_midlines():
        area = (x1 - x0) * (y1 - y0)
        perim = 2.0 * ((x1 - x0) + (y1 - y0))
        naive += 4.0 * area ** 2 * 0.1 / perim
    assert naive < 0.8 * bredt.j  # documented design-office trap, not a gate


# --------------------------------------------------------------------------- #
# the FD field reproduces the multi-cell Bredt mechanics
# --------------------------------------------------------------------------- #
def test_shared_web_flow_matches_bredt():
    field = mc.solve_multicell_prandtl(ASYM2, 0.025)
    fd = mc.wall_band_taus(ASYM2, field)
    bredt = mc.bredt_multicell(ASYM2)
    web = [w for w in bredt.walls if w["label"] == "web0"][0]
    bredt_tau = abs(bredt.wall_flow(web)) / web["thickness"]
    assert bredt_tau > 0.1  # the web genuinely carries flow in this layout
    assert abs(fd["web0"] / bredt_tau - 1.0) < 0.10


def test_symmetric_web_carries_little_flow():
    field = mc.solve_multicell_prandtl(SYM2, 0.025)
    fd = mc.wall_band_taus(SYM2, field)
    assert fd["web0"] < 0.1 * fd["outer_left"]
    assert fd["outer_left"] == pytest.approx(fd["outer_right"], rel=0.02)


def test_bredt_is_a_lower_bound_and_thin_wall_limited():
    thin = mc.RectMultiCellSection(a=1.0, b=0.6, wall=0.02, webs=((0.6, 0.02),))
    coarse = mc.RectMultiCellSection(a=1.0, b=0.6, wall=0.1, webs=((0.6, 0.1),))
    e_thin = mc.solve_multicell_prandtl(thin, 0.01).j_torque \
        / mc.bredt_multicell(thin).j - 1.0
    e_coarse = mc.solve_multicell_prandtl(coarse, 0.025).j_torque \
        / mc.bredt_multicell(coarse).j - 1.0
    assert 0.0 < e_thin < e_coarse          # Bredt underestimates, less as t -> 0
    assert e_thin < 0.02                    # 2 % band at t/2a = 1 %
    assert e_coarse < 0.10


# --------------------------------------------------------------------------- #
# the merged HEX8 production path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("section", [SYM2, ASYM2])
def test_hex8_reproduces_fd_torsional_rigidity(section):
    field = mc.solve_multicell_prandtl(section, 0.025)
    fem = mc.multicell_torsion_fem(section, 0.05, layers_z=2)
    assert abs(fem["j_fe"] / field.j_torque - 1.0) < 0.03
    assert fem["warping_max"] > 0.0 and fem["dofs"] == 3 * fem["nodes"]


def test_hex8_single_cell_agrees_with_box_module():
    fem = mc.multicell_torsion_fem(SINGLE, 0.05, layers_z=2)
    field = mc.solve_multicell_prandtl(SINGLE, 0.025)
    assert abs(fem["j_fe"] / field.j_torque - 1.0) < 0.03