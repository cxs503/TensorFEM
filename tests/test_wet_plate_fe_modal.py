"""Tests for the coupled wet-plate modal kernel (nonlocal fluid operator).

Evidence levels:

* ``piston_added_mass_per_area`` / ``prolate_spheroid_transverse_added_mass``
  -- closed-form limits (sphere ``0.5 m_displaced``, long spheroid ``2/3``);
* ``solve_wet_plate_fe_modal`` -- dry branch against the Navier closed form,
  added-mass diagonal against the exact baffled Rayleigh integral of
  :func:`tensorfem.fsi_wet_modal.plate_exact_modal_added_mass`, wet/dry ratio
  against the single-mode ``1/sqrt(1+mu)`` prediction, and the fluid panel
  mesh as an independent convergence parameter.
"""
import math

import pytest

from tensorfem import fsi_wet_modal as fsi
from tensorfem.wet_plate_fe_modal import (
    piston_added_mass_per_area,
    prolate_spheroid_transverse_added_mass,
    solve_wet_plate_fe_modal,
)

LENGTH = 10.0
WIDTH = 4.0
THICKNESS = 0.02
YOUNG = 210.0e9
POISSON = 0.3
RHO_STRUCT = 7850.0
RHO_FLUID = 1000.0


@pytest.fixture(scope="module")
def baseline():
    """One 8x8 structural / 16x16 fluid solve reused by several assertions."""
    return solve_wet_plate_fe_modal(
        length=LENGTH,
        width=WIDTH,
        thickness=THICKNESS,
        young=YOUNG,
        poisson=POISSON,
        rho_struct=RHO_STRUCT,
        rho_fluid=RHO_FLUID,
        modes=6,
        fe_mesh=(8, 8),
        panel_mesh=(16, 16),
        dry_mode_numbers=((1, 1), (2, 1), (1, 2)),
    )


# ---------------------------------------------------------------------------
# closed-form approximations
# ---------------------------------------------------------------------------


def test_piston_added_mass_scales_with_equivalent_radius():
    m1 = piston_added_mass_per_area(4.0, 1000.0)
    m2 = piston_added_mass_per_area(16.0, 1000.0)
    assert m1 == pytest.approx(8.0 * 1000.0 * math.sqrt(4.0 / math.pi) / (3.0 * math.pi))
    assert m2 / m1 == pytest.approx(2.0)  # area x4 -> a_eq x2 -> m_a x2


def test_prolate_spheroid_sphere_limit():
    radius = 1.0
    length = 2.0 * radius * 1.0005  # e -> 0: sphere
    result = prolate_spheroid_transverse_added_mass(radius, length, 1000.0)
    assert result["added_mass"] / result["displaced_mass"] == pytest.approx(0.5, rel=1e-2)


def test_prolate_spheroid_long_cylinder_limit():
    radius = 1.0
    length = 2000.0  # e -> 1
    result = prolate_spheroid_transverse_added_mass(radius, length, 1000.0)
    two_dimensional = 1000.0 * math.pi * radius ** 2 * length
    assert result["added_mass"] / two_dimensional == pytest.approx(2.0 / 3.0, rel=2e-3)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_piston_validation(bad):
    with pytest.raises(ValueError):
        piston_added_mass_per_area(bad, 1000.0)
    with pytest.raises(ValueError):
        prolate_spheroid_transverse_added_mass(bad, 10.0, 1000.0)


# ---------------------------------------------------------------------------
# wet modal solve
# ---------------------------------------------------------------------------


def test_dry_branch_matches_navier(baseline):
    navier = fsi.navier_plate_frequency_hz(
        LENGTH, WIDTH, THICKNESS, YOUNG, POISSON, RHO_STRUCT, 1, 1
    )
    assert baseline.dry_hz_fe[0] == pytest.approx(navier, rel=0.03)
    assert baseline.dry_hz_fe[0] == pytest.approx(baseline.dry_hz_navier[0], rel=0.03)


def test_added_mass_diagonal_matches_exact_rayleigh(baseline):
    structural = RHO_STRUCT * THICKNESS * LENGTH * WIDTH / 4.0
    exact = fsi.plate_exact_modal_added_mass(
        LENGTH, WIDTH, 1, 1, RHO_FLUID
    ) / structural
    assert baseline.mu_modal_diag[0] == pytest.approx(exact, rel=0.02)


def test_wet_ratio_matches_single_mode_prediction(baseline):
    structural = RHO_STRUCT * THICKNESS * LENGTH * WIDTH / 4.0
    exact = fsi.plate_exact_modal_added_mass(
        LENGTH, WIDTH, 1, 1, RHO_FLUID
    ) / structural
    prediction = fsi.wet_frequency_ratio(exact)
    assert baseline.ratio_fe[0] == pytest.approx(prediction, rel=0.02)
    # the wet mode is always slower than the dry one
    assert baseline.wet_hz_fe[0] < baseline.dry_hz_fe[0]


def test_wet_ratio_independent_of_fluid_mesh():
    kwargs = dict(
        length=LENGTH,
        width=WIDTH,
        thickness=THICKNESS,
        young=YOUNG,
        poisson=POISSON,
        rho_struct=RHO_STRUCT,
        rho_fluid=RHO_FLUID,
        modes=6,
        fe_mesh=(8, 8),
        dry_mode_numbers=((1, 1),),
    )
    coarse = solve_wet_plate_fe_modal(panel_mesh=(10, 10), **kwargs)
    fine = solve_wet_plate_fe_modal(panel_mesh=(20, 20), **kwargs)
    assert fine.ratio_fe[0] == pytest.approx(coarse.ratio_fe[0], rel=0.02)


def test_mode_shapes_and_pressure_are_exposed(baseline):
    assert baseline.nodes is not None and baseline.elements is not None
    n_nodes = baseline.nodes.shape[0]
    assert baseline.dry_mode_shapes.shape[0] == n_nodes
    assert baseline.wet_mode_shapes.shape == baseline.dry_mode_shapes.shape
    assert baseline.panel_pressure.shape[1] == baseline.dry_mode_shapes.shape[1]
    assert baseline.collapsed_modes == 0


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_geometry_validation(bad):
    with pytest.raises(ValueError):
        solve_wet_plate_fe_modal(length=bad)
    with pytest.raises(ValueError):
        solve_wet_plate_fe_modal(rho_fluid=bad)