"""Regression gate for the thin-walled closed-box tip-deflection case.

The case-pool anchor ``CASE-CANTILEVER-TIP`` certifies a *solid* rectangular
cantilever.  These tests cover the closed thin-walled box added alongside it
(``src/tensorfem/thinwall_box.py``):

* section properties against a brute-force numerical integration;
* the closed-ring shear flow against the two properties it must have --
  ``oint q/t ds = 0`` (zero twist) and ``A_s`` equal to the strain-energy
  derivative (an independent quadrature route);
* the mesher: the hollow mesh must reproduce the ring area and second moment
  exactly, and the void must really be void;
* one cheap 3-D solve must land inside the case-pool tolerance of the shear-flow
  oracle, so a regression in ``hex20`` or in the mesh builder is caught without
  paying for the full benchmark sweep
  (``benchmarks/verified/fem_cantilever_shear/box_section_shear.py``).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorfem.thinwall_box import (  # noqa: E402
    BoxSection,
    box_hex20_mesh,
    delta_bending,
    delta_naive_timoshenko,
    delta_thin_wall,
    shear_area,
    shear_coefficient,
    shear_energy_check,
    shear_flow_closed_form,
    shear_flow_q0_per_unit_shear,
    solve_box_cantilever,
)

E = 200e9
NU = 0.3
P = 5.0


def _brute_force(section: BoxSection, n: int = 801) -> tuple[float, float]:
    """Area and ``int z^2 dA`` of the ring by midpoint sampling (numpy)."""
    import numpy as np

    dy = section.width / n
    dz = section.depth / n
    y = -0.5 * section.width + (np.arange(n) + 0.5) * dy
    z = -0.5 * section.depth + (np.arange(n) + 0.5) * dz
    inside = (np.abs(y) < 0.5 * section.inner_width)[:, None] & (
        np.abs(z) < 0.5 * section.inner_depth
    )[None, :]
    cell = dy * dz
    area = float((~inside).sum() * cell)
    inertia = float((np.where(inside, 0.0, 1.0) * cell * z[None, :] ** 2).sum())
    return area, inertia


@pytest.mark.parametrize("t_ratio", [0.20, 0.10, 0.05])
def test_section_properties_match_brute_force(t_ratio):
    section = BoxSection(0.010, 0.010, t_ratio * 0.010)
    area, inertia = _brute_force(section)
    assert area == pytest.approx(section.area, rel=2e-3)
    assert inertia == pytest.approx(section.inertia_y, rel=2e-3)


def test_box_section_rejects_impossible_walls():
    with pytest.raises(ValueError):
        BoxSection(0.010, 0.010, 0.005)
    with pytest.raises(ValueError):
        BoxSection(0.010, 0.010, 0.0)
    with pytest.raises(ValueError):
        BoxSection(0.010, 0.010, -0.001)


def test_shear_flow_is_twist_free():
    """``oint q/t ds = 0`` along the closed path, evaluated from the returned flow."""
    import sympy as sp

    form = shear_flow_closed_form()
    s = form["s"]
    section = BoxSection(0.010, 0.010, 0.001)
    subs = {
        form["a"]: 0.5 * section.centreline_width,
        form["c"]: 0.5 * section.centreline_depth,
        form["t"]: section.thickness,
    }
    circulation = 0.0
    for length, Q in form["pieces"]:
        expr = (Q / section.thickness).subs(subs)
        circulation += float(sp.integrate(expr, (s, 0, float(sp.nsimplify(length).subs(subs)))))
    # q = V Q / I, so the twist-free condition is scale free; V = I = 1 here.
    assert abs(circulation) < 1e-12


def test_closed_ring_has_nonzero_indeterminate_constant():
    section = BoxSection(0.010, 0.010, 0.001)
    assert abs(shear_flow_q0_per_unit_shear(section)) > 0.0


def test_shear_area_matches_strain_energy():
    section = BoxSection(0.010, 0.010, 0.001)
    check = shear_energy_check(section, 0.04, E, NU)
    assert check["relative_difference"] < 1e-4


def test_effective_shear_area_below_web_only_and_gross():
    for t_ratio in (0.20, 0.10, 0.05):
        section = BoxSection(0.010, 0.010, t_ratio * 0.010)
        As = shear_area(section)
        assert 0.0 < As < section.area
        # shear lag: the flange flows add energy without carrying net shear
        assert As < section.web_only_shear_area()
        assert shear_coefficient(section) > 5.0 / 6.0


def test_naive_kappa_underestimates_shear_deflection():
    section = BoxSection(0.010, 0.010, 0.001)
    length = 0.04
    tw = delta_thin_wall(section, length, P, E, NU)
    naive = delta_naive_timoshenko(section, length, P, E, NU)
    assert naive < tw
    assert (tw - naive) / tw > 0.03


def test_hollow_mesh_geometry_is_exact():
    section = BoxSection(0.010, 0.010, 0.001)
    mesh = box_hex20_mesh(section, 0.04, axial_cells=4, wall_cells=2, inner_cells=6)
    assert mesh["mesh_area"] == pytest.approx(section.area, rel=1e-12)
    assert mesh["mesh_inertia_y"] == pytest.approx(section.inertia_y, rel=1e-12)
    assert mesh["removed_cells"] > 0
    # the void really is void: no node strictly inside the inner surface
    nodes = mesh["nodes"]
    inner_y = 0.5 * section.inner_width - 1e-12
    inner_z = 0.5 * section.inner_depth - 1e-12
    inside = ((nodes[:, 1].abs() < inner_y) & (nodes[:, 2].abs() < inner_z)).sum().item()
    assert inside == 0


def test_coarse_box_solve_matches_shear_flow_oracle():
    section = BoxSection(0.010, 0.010, 0.002)
    length = 0.02
    result = solve_box_cantilever(
        section,
        length,
        load=P,
        young=E,
        nu=NU,
        axial_cells=6,
        wall_cells=1,
        inner_cells=6,
    )
    oracle = delta_thin_wall(section, length, P, E, NU)
    assert result["force_residual"] < 1e-6
    assert result["moment_residual"] < 1e-6
    # deliberately coarse: the tight (3 %) claim is carried by the benchmark sweep
    assert abs(result["tip_deflection_m"] / oracle - 1.0) < 0.08
    assert result["tip_deflection_m"] > delta_bending(section, length, P, E)