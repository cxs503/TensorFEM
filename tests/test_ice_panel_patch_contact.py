"""Tests for the CZM-contact ice patch load -> panel stress pipeline.

Coverage, in the order the physics is chained:

* the rectangular panel solver reproduces the Timoshenko Table 8 coefficients
  (the caliber both the JOUR-ICE-STRUCT baseline and the H3-ICE-PLATE-FEM anchor
  are quoted at),
* the contact -> load projection conserves force exactly on a grid that is *not*
  aligned with the panel mesh,
* the CZM contact solve hits the prescribed normal force and never exceeds the
  crushing ceiling, with a contact area that grows with load,
* a flat ice edge collapses onto the uniform patch (self-check of the chain),
* the two-way coupling loop reaches a fixed point and keeps the force,
* the finite-element peak moment agrees with the independent Kirchhoff Navier
  series for a non-uniform pressure field,
* the non-uniformity scan returns machine-readable numbers with >= 3
  configurations.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorfem.ice_panel_patch_contact import (
    BilinearCohesiveLaw,
    ContactPatch,
    coupled_patch_solve,
    czm_contact_patch,
    default_law,
    navier_rect_moments,
    nonuniformity_scan,
    patch_from_contact,
    solve_rect_panel,
    uniform_patch_pressure,
)

DP = dict(young=206.0e9, poisson=0.3, thickness=0.022)


def _flat_edge(half: float = 0.10) -> dict:
    return {"kind": "flat", "amplitude": 0.0, "box_half": half}


def _spherical_edge() -> dict:
    return {"kind": "spherical", "radius": 1.0, "amplitude": 0.02, "box_half": 0.10}


# ---------------------------------------------------------------- references --
def test_uniform_square_matches_timoshenko_table8():
    """SSSS square, uniform load, nu = 0.3: M_max/(q a^2) = 0.0479, w D/(q a^4) = 0.00406."""
    sol = solve_rect_panel(24, 24, 1.0, 1.0, load=1.0e6, **DP)
    peak = sol.stress_at(0.5, 0.5)
    m_coef = peak["moment_x_Nm_per_m"] / 1.0e6
    assert m_coef == pytest.approx(0.0479, rel=0.01)

    D = DP["young"] * DP["thickness"] ** 3 / (12.0 * (1.0 - DP["poisson"] ** 2))
    from tensorfem.ice_panel_patch_contact import deflection_at

    w = float(deflection_at(sol, torch.tensor([0.5]), torch.tensor([0.5]))[0])
    w_coef = w * D / 1.0e6
    assert w_coef == pytest.approx(0.00406, rel=0.02)


def test_navier_series_reproduces_timoshenko_for_full_plate():
    """The independent series must reproduce the tabulated coefficient exactly."""
    uni = uniform_patch_pressure(
        total_force=1.0e6, centre=(0.5, 0.5), half_x=0.5, half_y=0.5,
        panel=(1.0, 1.0), grid=81,
    )
    nav = navier_rect_moments(
        uni, 1.0, 1.0, x=0.5, y=0.5, terms=161, **DP
    )
    assert nav["moment_x"] / 1.0e6 == pytest.approx(0.0478864, rel=2e-3)


# ------------------------------------------------------------- projection ----
def test_projection_conserves_force_on_non_aligned_grid():
    """Exact sub-cell overlap integration: force balance to machine precision."""
    law = default_law()
    patch = czm_contact_patch(
        law=law, ceiling=2.0e6, edge=_spherical_edge(), total_force=4.0e4,
        panel=(0.8, 0.7), grid=61,
    )
    # deliberately not aligned with the panel mesh (40 elements over 0.8 m)
    _, _, f, info = patch_from_contact(patch, 37, 26, 0.8, 0.7)
    assert info["force_error_rel"] < 1e-12
    assert info["projected_force"] == pytest.approx(patch.total_force, rel=1e-12)
    assert info["loaded_area"] == pytest.approx(0.2 * 0.2, rel=1e-12)
    # only the vertical (w) dofs carry load
    w_dofs = f.reshape(-1, 3)[:, 0]
    assert float(f.reshape(-1, 3)[:, 1:].abs().sum()) == 0.0
    assert float(w_dofs.sum()) == pytest.approx(patch.total_force, rel=1e-12)


def test_pressure_field_is_zero_outside_the_contact_grid():
    patch = uniform_patch_pressure(
        total_force=1.0e4, centre=(0.2, 0.2), half_x=0.05, half_y=0.05,
        panel=(0.8, 0.7), grid=41,
    )
    x = torch.tensor([0.6, 0.6])
    y = torch.tensor([0.2, 0.6])
    assert float(patch.pressure_at(x, y).abs().sum()) == 0.0


# ------------------------------------------------------------- CZM contact ---
@pytest.mark.parametrize("force", [1.0e4, 4.0e4, 8.0e4])
def test_czm_contact_hits_target_force_and_respects_ceiling(force):
    law = default_law()
    patch = czm_contact_patch(
        law=law, ceiling=2.0e6, edge=_spherical_edge(), total_force=force,
        panel=(0.8, 0.7), grid=81,
    )
    assert patch.total_force == pytest.approx(force, rel=1e-7)
    assert patch.peak_pressure <= 2.0e6 * (1.0 + 1e-12)
    assert patch.peak_pressure > 0.0
    assert 0.0 < patch.contact_area <= 0.2 * 0.2
    # the pressure centroid of the centred spherical edge is the patch centre
    assert patch.centroid[0] == pytest.approx(0.4, abs=1e-9)
    assert patch.centroid[1] == pytest.approx(0.35, abs=1e-9)


def test_contact_area_and_peak_pressure_grow_with_load():
    law = default_law()
    areas, peaks = [], []
    for force in (1.0e4, 4.0e4, 1.0e5):
        p = czm_contact_patch(
            law=law, ceiling=2.0e6, edge=_spherical_edge(), total_force=force,
            panel=(0.8, 0.7), grid=81,
        )
        areas.append(p.contact_area)
        peaks.append(p.peak_pressure)
    assert areas == sorted(areas)
    assert peaks == sorted(peaks)
    # the pressure field is non-uniform: peak well above the mean
    p = czm_contact_patch(
        law=law, ceiling=2.0e6, edge=_spherical_edge(), total_force=4.0e4,
        panel=(0.8, 0.7), grid=81,
    )
    assert p.peak_pressure / (p.total_force / p.contact_area) > 1.2


# ------------------------------------------------------- uniform limit case --
def test_flat_edge_reduces_to_the_uniform_patch():
    """Self-check of the chain: a flat ice edge must give a uniform pressure."""
    law = default_law()
    F = 4.0e4
    patch = czm_contact_patch(
        law=law, ceiling=2.0e6, edge=_flat_edge(), total_force=F,
        panel=(0.8, 0.7), grid=61,
    )
    mean = patch.total_force / patch.contact_area
    assert patch.peak_pressure == pytest.approx(mean, rel=1e-12)

    _, _, f, _ = patch_from_contact(patch, 32, 28, 0.8, 0.7)
    sol = solve_rect_panel(32, 28, 0.8, 0.7, load=f, **DP)

    uni = uniform_patch_pressure(
        total_force=F, centre=patch.centroid, half_x=0.10, half_y=0.10,
        panel=(0.8, 0.7), grid=61,
    )
    _, _, f_uni, _ = patch_from_contact(uni, 32, 28, 0.8, 0.7)
    sol_uni = solve_rect_panel(32, 28, 0.8, 0.7, load=f_uni, **DP)

    a = sol.stress_at(0.4, 0.35)["sigma_Mx_MPa"]
    b = sol_uni.stress_at(0.4, 0.35)["sigma_Mx_MPa"]
    assert a == pytest.approx(b, rel=1e-9)


# --------------------------------------------------------- two-way coupling --
def test_two_way_coupling_converges_and_keeps_force():
    law = default_law()
    patch, sol, coup = coupled_patch_solve(
        nx=24, ny=20, law=law, ceiling=2.0e6, edge=_spherical_edge(),
        total_force=4.0e4, grid=61, tol=1e-4, **DP,
    )
    assert coup["converged"], coup["history"]
    assert coup["sweeps"] >= 2
    assert patch.total_force == pytest.approx(4.0e4, rel=1e-9)
    # the plate deflects towards the load
    from tensorfem.ice_panel_patch_contact import deflection_at

    w_centre = float(
        deflection_at(sol, torch.tensor([0.4]), torch.tensor([0.35]))[0]
    )
    assert w_centre > 0.0
    # the coupling changes the pressure field relative to the one-way solve
    one_way = czm_contact_patch(
        law=law, ceiling=2.0e6, edge=_spherical_edge(), total_force=4.0e4,
        panel=(0.8, 0.7), grid=61,
    )
    assert patch.approach > one_way.approach


# -------------------------------------------- finite element vs series check --
def test_fem_peak_moment_matches_navier_series_for_nonuniform_patch():
    law = default_law()
    patch = czm_contact_patch(
        law=law, ceiling=2.0e6, edge=_spherical_edge(), total_force=4.0e4,
        panel=(0.8, 0.7), grid=81,
    )
    _, _, f, _ = patch_from_contact(patch, 40, 35, 0.8, 0.7)
    sol = solve_rect_panel(40, 35, 0.8, 0.7, load=f, **DP)
    peak = sol.peak_stress()
    nav = navier_rect_moments(
        patch, 0.8, 0.7,
        x=peak["peak_sigma_Mx_at_m"][0], y=peak["peak_sigma_Mx_at_m"][1],
        terms=121, **DP,
    )
    sig_nav = 6.0 * abs(nav["moment_x"]) / DP["thickness"] ** 2 / 1e6
    assert peak["sigma_Mx_MPa"] == pytest.approx(sig_nav, rel=0.02)


# ------------------------------------------------------------------- scan ----
def test_scan_is_machine_readable_and_has_at_least_three_configurations():
    scan = nonuniformity_scan(nx=20, ny=16, grid=49)
    assert len(scan["configurations"]) >= 3
    for row in scan["configurations"]:
        for key in (
            "name", "total_force_N", "contact_area_m2", "peak_pressure_MPa",
            "nonuniform_patch", "uniform_reference_patch",
            "delta_nonuniform_over_uniform", "coupling",
        ):
            assert key in row, key
        for key in ("sigma_vm", "sigma_Mx", "sigma_vm_pct", "sigma_Mx_pct"):
            val = row["delta_nonuniform_over_uniform"][key]
            assert math.isfinite(val)
            assert val > 0.0
        assert row["force_residual"] < 1e-7
        assert row["force_projection_error_rel"] < 1e-10
        assert row["nonuniform_patch"]["sigma_Mx_MPa"] > 0.0
        assert row["uniform_reference_patch"]["sigma_Mx_MPa"] > 0.0
    # the flat-edge limit case must sit on the uniform reference
    c1 = next(r for r in scan["configurations"] if r["name"].startswith("C1"))
    assert c1["delta_nonuniform_over_uniform"]["sigma_Mx"] == pytest.approx(1.0, rel=2e-3)
    # a curved (non-uniform) patch must move the peak stress
    curved = [
        r for r in scan["configurations"]
        if not r["name"].startswith("C1")
    ]
    assert any(
        abs(r["delta_nonuniform_over_uniform"]["sigma_Mx"] - 1.0) > 1e-2 for r in curved
    )
    assert scan["summary"]["max_force_projection_error_rel"] < 1e-10