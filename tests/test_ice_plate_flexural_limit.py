"""Verification of the ice-plate flexural limit-load anchor (H3-ICE-PLATE-FEM).

The chain under test is

  Winkler plate kernel (unchanged) -> FE moment field
      -> independent Rankine surface-stress criterion -> first-crack load,

with the exact Kelvin patch moment and the Westergaard/Nevel closed form as the
analytic references.  The tests also guard the two failure modes of this kind of
anchor: a wrong element Jacobian (factor two in the moments) and a load vector
that does not represent the circular patch.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorfem.foundation_plate_bending import _gauss_kinematics
from tensorfem.ice_plate_flexural_limit import (
    characteristic_moment_per_load,
    failure_mode,
    first_crack_load,
    foundation_reaction,
    nevel_moment_applicability,
    principal_moments,
    q4_shape_kinematics,
    quarter_disc_loads,
    solve_circular_patch,
    surface_tensile_stress,
)
from tensorfem.levelset.plate_crack import patch_max_moment

SHEET = dict(young=7.0e9, poisson=0.30, thickness=0.30, foundation=1025.0 * 9.81)
RIGIDITY = 7.0e9 * 0.30 ** 3 / (12.0 * (1.0 - 0.30 ** 2))
LENGTH = (RIGIDITY / SHEET["foundation"]) ** 0.25
STRENGTH = 6.0e5


# ------------------------------------------------------------------ kinematics
def test_q4_kinematics_matches_torch_plate_kernel():
    """Own bilinear kinematics == the torch kernel's, on a distorted element."""
    quads = (
        np.array([[0.0, 0.0], [0.7, 0.0], [0.7, 0.3], [0.0, 0.3]]),
        np.array([[0.0, 0.0], [0.9, 0.1], [0.8, 0.7], [-0.1, 0.5]]),
        np.array([[0.0, 0.0], [2.0, -0.4], [2.3, 1.1], [0.2, 1.6]]),
    )
    for xy in quads:
        for xi, eta in ((-1 / math.sqrt(3), -1 / math.sqrt(3)), (1 / math.sqrt(3),) * 2):
            n, dn, det_j = q4_shape_kinematics(xy, xi, eta)
            n_t, dn_t, det_t = _gauss_kinematics(
                torch.tensor(xy, dtype=torch.float64)[None, :, :], xi, eta
            )
            assert np.allclose(n, n_t.numpy(), atol=1e-14)
            assert np.allclose(dn, dn_t.numpy()[0], atol=1e-12)
            assert det_j == pytest.approx(float(det_t[0]), rel=1e-12)


def test_jacobian_is_half_the_side_length():
    """d x / d xi = side/2: the factor that silently doubles all moments."""
    xy = np.array([[0.0, 0.0], [0.8, 0.0], [0.8, 0.6], [0.0, 0.6]])
    _, _, det_j = q4_shape_kinematics(xy, 0.0, 0.0)
    assert det_j == pytest.approx(0.8 * 0.6 / 4.0)


# ------------------------------------------------------------------ load vector
@pytest.mark.parametrize("frac", [0.15, 0.5, 1.0])
def test_circular_patch_load_is_exact(frac):
    """Consistent quarter-disc loads integrate to pressure * pi a^2 / 4."""
    from tensorfem.foundation_plate_bending import graded_square_mesh

    a = frac * LENGTH
    nodes_t, elements_t = graded_square_mesh(60, 20.0 * LENGTH, 5.0)
    force = quarter_disc_loads(nodes_t.numpy(), elements_t.numpy(), a, 2.5e3, subcells=8)
    total = force[0::3].sum()
    assert total / (2.5e3 * math.pi * a ** 2 / 4.0) == pytest.approx(1.0, abs=5e-4)


def test_patch_load_zero_outside_patch():
    """No element beyond one diagonal of the patch receives a force."""
    from tensorfem.foundation_plate_bending import graded_square_mesh

    a = 0.4 * LENGTH
    nodes_t, elements_t = graded_square_mesh(40, 20.0 * LENGTH, 4.0)
    force = quarter_disc_loads(nodes_t.numpy(), elements_t.numpy(), a, 1.0).reshape(-1, 3)
    loaded = np.nonzero(np.abs(force[:, 0]) > 0.0)[0]
    r = np.hypot(*nodes_t.numpy()[loaded].T)
    assert r.max() > a  # nodes of the boundary element carry part of the load
    assert r.max() <= 2.0 * a


# ------------------------------------------------------------------ equilibrium
@pytest.mark.parametrize("frac", [0.3, 1.0, 4.0])
def test_foundation_reaction_balances_the_patch_load(frac):
    sol = solve_circular_patch(patch_radius=frac * LENGTH, cells=80, beta=5.0, **SHEET)
    assert foundation_reaction(sol) / sol.total_load == pytest.approx(1.0, abs=1e-6)
    assert sol.total_load / (math.pi * (frac * LENGTH) ** 2 / 4.0) == pytest.approx(
        1.0, abs=5e-3
    )


# ------------------------------------------------------------------ anchor
@pytest.mark.parametrize("frac", [0.1, 0.3, 0.5, 1.0])
def test_centre_moment_matches_kelvin_anchor(frac):
    """FE centre moment vs the exact Kelvin patch moment (acceptance: 3%)."""
    a = frac * LENGTH
    sol = solve_circular_patch(patch_radius=a, cells=80, beta=5.0, **SHEET)
    specs = characteristic_moment_per_load(sol)
    anchor = patch_max_moment(1.0, a, RIGIDITY, SHEET["foundation"], SHEET["poisson"])
    assert specs["centre_moment"] / anchor == pytest.approx(1.0, abs=1e-2)
    assert specs["bottom_max_moment"] >= specs["centre_moment"] * (1.0 - 1e-12)
    assert specs["bottom_at_centre"]


def test_centre_state_is_axisymmetric():
    sol = solve_circular_patch(patch_radius=0.3 * LENGTH, cells=80, beta=5.0, **SHEET)
    m = sol.moments[int(np.argmin(np.hypot(*sol.gauss_points.T)))]
    assert m[0] == pytest.approx(m[1], rel=2e-3)
    assert abs(m[2]) < 1e-3 * abs(m[0])


# ------------------------------------------------------------------ criterion
def test_principal_moment_criterion_is_rankine():
    # uniaxial moment: bottom tension 6M/h^2, top fibre stress zero
    m1, m2 = principal_moments(np.array([5.0]), np.array([0.0]), np.array([0.0]))
    bottom, top = surface_tensile_stress(m1, m2, 0.5)
    assert bottom[0] == pytest.approx(6.0 * 5.0 / 0.25)
    assert top[0] == pytest.approx(0.0)
    # pure twist: principal moments +-|Mxy|
    m1, m2 = principal_moments(np.array([0.0]), np.array([0.0]), np.array([3.0]))
    assert (m1[0], m2[0]) == pytest.approx((3.0, -3.0))
    # hogging: the bottom fibre is in compression, the top in tension
    bottom, top = surface_tensile_stress(np.array([-1.0]), np.array([-2.0]), 0.3)
    assert bottom[0] == 0.0
    assert top[0] == pytest.approx(6.0 * 2.0 / 0.3 ** 2)
    # zero moment state: no tension at either fibre
    bottom, top = surface_tensile_stress(np.array([0.0]), np.array([0.0]), 0.3)
    assert bottom[0] == 0.0 and top[0] == 0.0


@pytest.mark.parametrize("frac", [0.2, 0.5])
def test_first_crack_load_within_ten_percent_of_plate_theory(frac):
    """Acceptance criterion: limit load within 10% of the analytic plate theory."""
    out = first_crack_load(
        patch_radius=frac * LENGTH, flexural_strength=STRENGTH, cells=80, beta=5.0, **SHEET
    )
    assert abs(out["error_vs_kelvin"]) < 0.03
    assert abs(out["error_vs_nevel"]) < 0.03
    assert out["critical_fibre"] == "bottom"


def test_first_crack_load_reproduces_strength_at_failure():
    """Scaling the FE field to P_f must put the critical fibre exactly at sigma_f."""
    frac = 0.3
    out = first_crack_load(
        patch_radius=frac * LENGTH, flexural_strength=STRENGTH, cells=80, beta=5.0, **SHEET
    )
    pressure = out["first_crack_load_fem"] / (math.pi * (frac * LENGTH) ** 2)
    sol = solve_circular_patch(
        patch_radius=frac * LENGTH, cells=80, beta=5.0, pressure=pressure, **SHEET
    )
    bottom, _ = sol.bottom_top_stress()
    # 1e-3: the equivalent pressure uses the analytic patch area while the load
    # vector integrates the sub-cell quadrature of that area (<3e-4 here)
    assert bottom.max() / STRENGTH == pytest.approx(1.0, rel=1e-3)


# ------------------------------------------------------------------ mode
def test_small_patch_fails_at_the_bottom_centre_radially():
    mode = failure_mode(patch_radius=0.3 * LENGTH, cells=80, beta=5.0, **SHEET)
    mode.pop("solution")
    assert mode["critical_fibre"] == "bottom"
    assert mode["critical_at_centre"]
    assert mode["hoop_over_radial"] == pytest.approx(1.0, abs=1e-2)
    assert mode["mode"] == "radial_bottom"
    assert mode["top_over_bottom"] < 0.3


def test_large_patch_migrates_inside_the_patch_and_prefers_radial_moment():
    mode = failure_mode(patch_radius=6.0 * LENGTH, cells=80, beta=5.0, **SHEET)
    mode.pop("solution")
    assert mode["critical_fibre"] == "bottom"
    assert 0.5 < mode["critical_radius_over_patch"] < 0.95
    assert mode["hoop_over_radial"] < 0.9
    assert mode["mode"] == "circumferential_bottom"


def test_critical_location_is_mesh_converged():
    coarse = failure_mode(patch_radius=4.0 * LENGTH, cells=80, beta=5.0, **SHEET)
    fine = failure_mode(patch_radius=4.0 * LENGTH, cells=120, beta=6.0, **SHEET)
    coarse.pop("solution"), fine.pop("solution")
    assert coarse["critical_radius_over_patch"] == pytest.approx(
        fine["critical_radius_over_patch"], abs=0.03
    )
    assert coarse["top_over_bottom"] == pytest.approx(fine["top_over_bottom"], abs=0.02)


# ------------------------------------------------------------------ Nevel window
def test_nevel_form_validity_window():
    rows = {r["radius_over_length"]: r for r in nevel_moment_applicability(**SHEET)}
    assert abs(rows[0.3]["relative_error"]) < 0.01
    assert abs(rows[1.0]["relative_error"]) > 0.10      # outside its validity
    assert rows[2.0]["nevel"] < 0.0                     # log form changes sign