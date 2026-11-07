"""Tests for the elastic hull-panel / ice-contact flexibility feedback module."""
from __future__ import annotations

import math

import pytest
import torch

from tensorfem.ice_panel_fsi import (
    CoupledConfig,
    HullPanel,
    IceConfig,
    PanelConfig,
    bending_capped_pressure,
    compare_rigid_elastic,
    ice_bending_operator,
    navier_ss_rect_centre,
    rect_mesh,
    run_ice_panel_case,
    theory_centre_deflection,
    thickness_sensitivity,
    uniform_load_centre_deflection,
)

D = torch.float64


# ---------------------------------------------------------------- geometry
def test_rect_mesh_geometry():
    nodes, elems = rect_mesh(0.75, 2.4, 8, 16)
    assert nodes.shape == (9 * 17, 2)
    assert elems.shape == (8 * 16, 4)
    # node (i, j) at index j*(nx+1)+i, x fastest
    assert torch.allclose(nodes[1], torch.tensor([0.75 / 8, 0.0], dtype=D))
    assert torch.allclose(nodes[9], torch.tensor([0.0, 2.4 / 16], dtype=D))
    # counter-clockwise connectivity
    conn = elems[0]
    xy = nodes[conn]
    area = 0.5 * abs(float(torch.linalg.det(torch.stack(
        (xy[1] - xy[0], xy[2] - xy[0])))))
    assert area > 0
    with pytest.raises(ValueError):
        rect_mesh(-1.0, 1.0, 4, 4)
    with pytest.raises(ValueError):
        rect_mesh(1.0, 1.0, 0, 4)


def test_panel_config_validation():
    with pytest.raises(ValueError):
        PanelConfig(boundary="free").validate()
    with pytest.raises(ValueError):
        PanelConfig(poisson=0.6).validate()
    with pytest.raises(ValueError):
        PanelConfig(thickness=-0.01).validate()
    with pytest.raises(ValueError):
        CoupledConfig(blocks=1).validate()


# -------------------------------------------------- static plate verification
def test_simply_supported_centre_deflection_matches_plate_theory():
    """Acceptance (a): FE centre deflection vs plate theory within 10%."""
    cases = [(1.0, 1.0, 0.010), (0.75, 2.4, 0.010), (0.75, 2.4, 0.025)]
    errors = []
    for span_x, span_y, thickness in cases:
        cfg = PanelConfig(span_x=span_x, span_y=span_y, thickness=thickness,
                          boundary="simply_supported", nx=8, ny=16)
        fem = uniform_load_centre_deflection(cfg, 1.0e5)
        theory = theory_centre_deflection(cfg, 1.0e5)
        errors.append(abs(fem - theory) / theory)
    assert max(errors) < 0.10
    assert max(errors) < 0.05


def test_clamped_to_simply_supported_ratio():
    """Clamped/SS centre-deflection ratios match textbook limits."""
    square_ss = uniform_load_centre_deflection(
        PanelConfig(span_x=1.0, span_y=1.0, thickness=0.01, nx=8, ny=8,
                    boundary="simply_supported"), 1.0e5)
    square_cl = uniform_load_centre_deflection(
        PanelConfig(span_x=1.0, span_y=1.0, thickness=0.01, nx=8, ny=8,
                    boundary="clamped"), 1.0e5)
    assert abs(square_cl / square_ss - 0.31) < 0.03      # 0.00126/0.00406
    panel_ss = uniform_load_centre_deflection(
        PanelConfig(thickness=0.025, boundary="simply_supported"), 1.0e5)
    panel_cl = uniform_load_centre_deflection(
        PanelConfig(thickness=0.025, boundary="clamped"), 1.0e5)
    assert abs(panel_cl / panel_ss - 0.20) < 0.02        # one-way strip limit


def test_navier_reference_closed_form():
    """The Navier series must reproduce the classic 0.00406 coefficient."""
    rigidity = 206e9 * 0.01 ** 3 / (12 * (1 - 0.3 ** 2))
    w = navier_ss_rect_centre(1.0, 1.0, 206e9, 0.3, 0.01, 1.0e5)
    assert abs(w / (0.00406 * 1.0e5 / rigidity) - 1.0) < 0.01


# ------------------------------------------------------- contact operators
def test_contact_compliance_is_symmetric_positive():
    cfg = PanelConfig()
    contacts = torch.tensor([[0.375, 0.1], [0.375, 1.2], [0.375, 2.3]], dtype=D)
    panel = HullPanel(cfg, contacts)
    R = panel.compliance
    assert torch.allclose(R, R.T, atol=1e-18)
    assert torch.all(torch.diagonal(R) > 0)
    assert float(R[1, 1]) > float(R[0, 0])          # mid-bay softer than edge
    # a unit force at the centre deflects the centre most
    w = panel.contact_deflection(torch.tensor([0.0, 1.0, 0.0], dtype=D))
    assert int(torch.argmax(w)) == 1


def test_strain_energy_consistent_with_stiffness():
    cfg = PanelConfig()
    contacts = torch.tensor([[0.375, 0.4], [0.375, 1.2], [0.375, 2.0]], dtype=D)
    panel = HullPanel(cfg, contacts)
    f = torch.tensor([3.0e4, -1.0e4, 2.0e4], dtype=D)
    u = panel.solve_reduced(panel._L @ f)
    from_stiffness = 0.5 * float(u @ (panel.stiffness @ u))
    assert abs(panel.strain_energy(f) - from_stiffness) < 1e-6 * abs(from_stiffness)


def test_ice_bending_operator_ignores_uniform_pressure():
    ice = IceConfig()
    n = 12
    positions = (torch.arange(n, dtype=D) + 0.5) * (2.4 / n)
    B = ice_bending_operator(ice, positions)
    sigma = B @ torch.ones(n, dtype=D)
    assert float(torch.max(torch.abs(sigma))) < 1.0e-6


def test_ice_bending_operator_matches_periodic_plate_solution():
    ice = IceConfig()
    n, span = 12, 2.4
    positions = (torch.arange(n, dtype=D) + 0.5) * (span / n)
    B = ice_bending_operator(ice, positions)
    kappa = 2.0 * math.pi / span
    rigidity = ice.flexural_rigidity
    foundation = ice.water_density * ice.gravity
    amplitude = (6.0 / ice.thickness ** 2) * rigidity * kappa ** 2 / (
        rigidity * kappa ** 4 + foundation)
    p = 1.0e6 * torch.cos(kappa * positions)
    sigma = B @ p
    assert abs(float(torch.max(torch.abs(sigma))) / (amplitude * 1.0e6) - 1.0) < 0.08


def test_bending_cap_respects_flexural_strength():
    ice = IceConfig()
    n, span = 12, 2.4
    positions = (torch.arange(n, dtype=D) + 0.5) * (span / n)
    B = ice_bending_operator(ice, positions)
    pressure = 3.0e6 * torch.cos(2.0 * math.pi * positions / span)
    capped = bending_capped_pressure(B, pressure, ice.flexural_strength)
    sigma = torch.abs(B @ capped)
    assert float(torch.max(sigma)) <= ice.flexural_strength * 1.02
    assert float(torch.max(capped)) < float(torch.max(pressure))


def test_crush_pressure_rate_dependence():
    ice = IceConfig()
    slow = float(ice.crush_pressure(torch.tensor(0.01, dtype=D)))
    fast = float(ice.crush_pressure(torch.tensor(1.0, dtype=D)))
    assert slow >= fast
    assert 0.5 * ice.compressive_strength - 1.0 < fast < 2.0 * ice.compressive_strength + 1.0


def test_pressure_area_factor():
    ice = IceConfig(pressure_area_exponent=0.0)
    assert abs(float(ice.pressure_area_factor(torch.tensor(0.05, dtype=D))) - 1.0) < 1e-12
    ice2 = IceConfig(pressure_area_exponent=0.2)
    small = float(ice2.pressure_area_factor(torch.tensor(0.01, dtype=D)))
    large = float(ice2.pressure_area_factor(torch.tensor(1.0, dtype=D)))
    assert small > large > 0


# ------------------------------------------------------- coupled simulations
@pytest.fixture(scope="module")
def representative():
    cfg = CoupledConfig(ice=IceConfig(velocity=0.3), block_mass=233.0, steps=400)
    return compare_rigid_elastic(cfg)


def test_rigid_panel_peak_is_energy_limited(representative):
    """The rigid-panel peak is set by the impact energy, below the crush cap."""
    rigid = representative["rigid"]
    n = rigid["config"].blocks
    cap_load = n * rigid["contact_area"] * rigid["p_cap"]
    ice = rigid["config"].ice
    stiffness = n * rigid["contact_area"] * ice.young / ice.elastic_length()
    energy_limited = (2.0 * rigid["kinetic_initial"] * stiffness) ** 0.5
    assert abs(rigid["peak_force"] - energy_limited) / energy_limited < 0.05
    assert rigid["peak_force"] < cap_load
    assert rigid["bent_blocks"] == 0
    assert rigid["bending_fraction"] == 0.0
    assert float(torch.max(rigid["max_deflection"])) == 0.0


def test_elastic_panel_reduces_peak_load(representative):
    reduction = representative["load_reduction"]
    assert 0.05 < reduction < 0.50
    assert representative["peak_elastic"] < representative["peak_rigid"]


def test_elastic_energy_closure(representative):
    assert representative["elastic"]["energy_closure_error"] < 0.05
    assert representative["rigid"]["energy_closure_error"] < 0.05


def test_high_energy_case_activates_crushing_and_bending():
    cfg = CoupledConfig(ice=IceConfig(velocity=0.3), block_mass=4000.0, steps=800)
    row = compare_rigid_elastic(cfg)
    elastic = row["elastic"]
    assert elastic["crush_energy"] > 0.1 * elastic["kinetic_initial"]
    assert elastic["bent_blocks"] > 0
    assert elastic["energy_closure_error"] < 0.05
    assert row["peak_elastic"] < row["peak_rigid"]


def test_thickness_sensitivity_trend():
    cfg = CoupledConfig(ice=IceConfig(velocity=0.3), block_mass=233.0, steps=400)
    rows = thickness_sensitivity(cfg, (0.015, 0.030))
    assert rows[0]["load_reduction"] > rows[1]["load_reduction"]
    assert rows[0]["panel_frequency"] < rows[1]["panel_frequency"]
    for row in rows:
        assert row["energy_closure_error"] < 0.05


def test_bending_fraction_grows_with_flexibility():
    """Acceptance (c): a more flexible bay converts crushing into ice bending."""
    cfg = CoupledConfig(ice=IceConfig(velocity=0.3), block_mass=4000.0, steps=800)
    thin = compare_rigid_elastic(CoupledConfig(**{
        **cfg.__dict__,
        "panel": PanelConfig(thickness=0.012)}))
    thick = compare_rigid_elastic(CoupledConfig(**{
        **cfg.__dict__,
        "panel": PanelConfig(thickness=0.040)}))
    assert thin["bending_fraction_elastic"] >= thick["bending_fraction_elastic"]
    assert thin["load_reduction"] > thick["load_reduction"]


def test_run_is_deterministic():
    cfg = CoupledConfig(ice=IceConfig(velocity=0.3), block_mass=233.0,
                        steps=120, rigid=False)
    a = run_ice_panel_case(cfg)
    b = run_ice_panel_case(cfg)
    assert a["peak_force"] == b["peak_force"]
    assert torch.equal(a["total_force"], b["total_force"])


def test_panel_frequency_supports_quasi_static_reading(representative):
    """Rayleigh frequency of the bay is reported and physically plausible."""
    elastic = representative["elastic"]
    f1 = elastic["panel_frequency"]
    assert 50.0 < f1 < 5000.0
    # thinner bay -> lower frequency (softer), thicker -> higher
    thick = run_ice_panel_case(CoupledConfig(
        panel=PanelConfig(thickness=0.040), ice=IceConfig(velocity=0.3),
        block_mass=233.0, steps=60))
    assert thick["panel_frequency"] > f1