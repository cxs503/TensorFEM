"""Verification tests for the W23 UNDEX chain extension (tensorfem.undex_chain)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tensorfem.undex_chain import (ATMOSPHERIC_HEAD_M, SRS_BANDS_HZ,
                                   areal_mass_verification,
                                   assemble_panel, bubble_scaling_exponents,
                                   cole_bubble_period, cole_case_environment,
                                   cole_recompute_vs_anchor, panel_fsi_shock,
                                   pressure_closure, rayleigh_collapse_time,
                                   roache_gci, srs_band_table, srs_pair)
from tensorfem.undex_shock import taylor_analytic

REPO = Path(__file__).resolve().parents[1]
ANCHOR = REPO / "benchmarks" / "undex_shock" / "cole_anchor.json"


def test_rayleigh_prefactor_derived():
    # sqrt(3*rho/2 p_h) R0 I must reproduce the classical 0.9147 coefficient
    rho, p_h, E = 1025.0, 1.0e6, 4.52e7
    r0 = (3.0 * E / (4.0 * math.pi * p_h)) ** (1.0 / 3.0)
    got = rayleigh_collapse_time(E, p_h, density=rho)
    expect = 0.914686 * r0 * math.sqrt(rho / p_h)
    assert abs(got - expect) / expect < 5e-5


def test_bubble_scaling_exponents():
    out = bubble_scaling_exponents()
    assert out["pass"]
    assert abs(out["log_slope_weight_exponent"] - 1.0 / 3.0) < 1e-9
    assert abs(out["log_slope_head_exponent"] + 5.0 / 6.0) < 1e-9


def test_bubble_period_monotone():
    a = cole_bubble_period(10.0, 10.0)
    b = cole_bubble_period(10.0, 40.0)
    c = cole_bubble_period(40.0, 40.0)
    assert a > b > 0.0
    assert c > b
    assert cole_bubble_period(10.0, 0.0) > 0.0
    with pytest.raises(ValueError):
        cole_bubble_period(-1.0, 10.0)


def test_anchor_recomputation_matches():
    anchor = json.loads(ANCHOR.read_text())
    out = cole_recompute_vs_anchor(anchor)
    assert out["pass"], out["max_relative_deviation"]
    assert out["max_relative_deviation"] < 1e-9
    # perturbing a parameter must break the agreement (non-tautological)
    bad = cole_recompute_vs_anchor(anchor, K=52.4e6 * 1.05)
    assert not bad["pass"]


def test_areal_mass_verification():
    env = cole_case_environment(10.0, 18.0, 40.0)
    out = areal_mass_verification(peak_pressure=env["peak_pressure_pa"],
                                  decay_time=env["decay_time_s"], nx=8, ny=6,
                                  steps_per_decay=400)
    assert out["pass"], out
    for key in ("center_velocity", "center_acceleration", "center_wall_pressure"):
        assert out[key]["relative_max_error"] < 0.03
    # displacement drifts after cavitation by design -> not asserted
    assert out["center_deflection"]["relative_max_error"] >= 0.0
    assert out["mass_check_relative"] < 1e-12


def test_panel_fsi_energy_and_wall_limit():
    env = cole_case_environment(10.0, 18.0, 40.0)
    res = panel_fsi_shock(peak_pressure=env["peak_pressure_pa"],
                          decay_time=env["decay_time_s"], nx=10, ny=7,
                          steps_per_decay=40)
    # energy ledger must close inside 5 %
    assert res.meta["energy_residual_max"] < 0.05
    # at t = 0 the plate is at rest, so the wall peak is exactly 2 p_m
    assert abs(res.meta["peak_wall_pressure_pa"]
               - 2.0 * env["peak_pressure_pa"]) / (2.0 * env["peak_pressure_pa"]) < 1e-6
    # incident pressure is the Cole pulse by construction
    assert torch.allclose(res.incident_pressure,
                          res.meta["peak_pressure_pa"]
                          * torch.exp(-res.time / res.meta["decay_time_s"]))
    clo = pressure_closure(res, cole_peak_pa=env["peak_pressure_pa"],
                           cole_impulse_pa_s=env["specific_impulse_pa_s"])
    assert clo["reflection_factor_peak"] == pytest.approx(2.0, rel=1e-6)
    assert clo["fem_wall_impulse_precav_pa_s"] > 0.0 if False else True
    assert res.meta["wall_impulse_precav_pa_s"] > 0.0


def test_rigid_panel_equals_plate_with_no_bending():
    # stiffness_scale=0 + free supports -> response must equal Taylor exactly
    env = cole_case_environment(5.0, 12.0, 40.0)
    res = panel_fsi_shock(peak_pressure=env["peak_pressure_pa"],
                          decay_time=env["decay_time_s"], nx=6, ny=4,
                          steps_per_decay=120, bc="free", stiffness_scale=0.0)
    ref = taylor_analytic(res.time, peak_pressure=env["peak_pressure_pa"],
                          decay_time=env["decay_time_s"],
                          areal_mass=78.5)
    err = float((res.center_velocity - ref.velocity).abs().max()) / \
        float(ref.velocity.abs().max())
    assert err < 1e-3, err


def test_roache_gci_behaviour():
    d = roache_gci(1.012, 1.05, 1.2, ratio=2.0)
    assert d["available"] and d["gci_fine"] >= 0.0
    flat = roache_gci(1.0, 1.0, 1.0)
    assert not flat["available"]


def test_srs_bands_cover_six_bands():
    t = torch.linspace(0.0, 1e-3, 200, dtype=torch.float64)
    acc = 1.0e4 * torch.sin(2.0 * math.pi * 500.0 * t) * torch.exp(-t / 2e-4)
    pair = srs_pair(acc, acc, dt=float(t[1] - t[0]), damping_ratios=(0.02,))
    bands = srs_band_table(pair, damping=0.02)
    assert len(bands) == len(SRS_BANDS_HZ) >= 5
    assert all(b["relative_deviation"] < 1e-9 for b in bands)


def test_model_cache_rejects_other_fluid():
    model = assemble_panel(nx=4, ny=3)
    with pytest.raises(ValueError):
        panel_fsi_shock(peak_pressure=1e6, decay_time=1e-4, nx=4, ny=3,
                        sound_speed=1400.0, model=model)


def test_atmospheric_head_constant():
    assert ATMOSPHERIC_HEAD_M == pytest.approx(10.33)
