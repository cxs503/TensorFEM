"""Artifact gates for the ``fem_cylinder_radiation`` verification anchor.

The anchor driver (``benchmarks/verified/fem_cylinder_radiation/run.py``) writes
``result.json``, ``gates_verified.json`` and the figures.  This module does three
things:

1. re-checks that the committed evidence is internally consistent (every driver
   gate appears in the manifest, is graded PASS, and the numbers agree);
2. re-derives the anchor relations **independently of the driver** from the
   public API (piston law, high-``ka`` plateau, pulsating-sphere Rayleigh power,
   lateral area, rigid-body null, ``|V|^2`` scaling), so a silently edited
   evidence file fails here instead of passing;
3. pins the axial-cutoff convention of ``infinite_cylinder_conductance``
   (regression for the ``nan`` the sweep returned at ``k_x == k``, fixed here).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest
import torch

from tensorfem.struct_acoustics.cylinder_radiation import (
    cylinder_surface_grid,
    equivalent_source_power,
    finite_cylinder_mode_power,
    infinite_cylinder_conductance,
    infinite_cylinder_efficiency,
)

ANCHOR = Path(__file__).resolve().parents[1] / "benchmarks" / "verified" / "fem_cylinder_radiation"
RHO = 1.225
C = 343.0
RADIUS = 0.5


def _evidence():
    result = json.loads((ANCHOR / "result.json").read_text(encoding="utf-8"))
    gates = json.loads((ANCHOR / "gates_verified.json").read_text(encoding="utf-8"))
    return result, gates


def test_anchor_evidence_exists_and_all_gates_pass():
    result, gates = _evidence()
    assert result["passed"] is True
    assert gates["passed"] is True
    assert not gates.get("failed_gates")
    manifest = {g["id"]: g for g in gates["gates"]}
    for key, value in result["gates"].items():
        assert key in manifest, f"driver gate {key} missing from gates_verified.json"
        assert manifest[key]["passed"] is True, key
        assert float(manifest[key]["value"]) == pytest.approx(float(value), rel=1e-12, abs=1e-30), key
    assert len(manifest) == len(gates["gates"]) >= 15


def test_piston_law_low_ka():
    """sigma -> (pi/2) k a for the ring mode n = 0 at small ka."""
    worst = 0.0
    for ka in (1e-4, 1e-3, 1e-2):
        sigma = infinite_cylinder_efficiency(0, ka / RADIUS, 0.0, radius=RADIUS, rho=RHO, c=C)
        worst = max(worst, abs(sigma / (0.5 * math.pi * ka) - 1.0))
    assert worst <= 5e-3


def test_high_ka_plateau():
    """sigma_0 -> 1 and sigma_n -> 1/2 (n >= 1) once ka >> 1."""
    k = 20.0 / RADIUS
    assert infinite_cylinder_efficiency(0, k, 0.0, radius=RADIUS, rho=RHO, c=C) == pytest.approx(1.0, rel=2e-2)
    for n in (1, 2, 3):
        assert infinite_cylinder_efficiency(n, k, 0.0, radius=RADIUS, rho=RHO, c=C) == pytest.approx(0.5, rel=2e-2)


def test_pulsating_sphere_rayleigh_power():
    """ESM on a pulsating sphere reproduces Pi = 2 pi a^2 rho c V^2 (ka)^2/(1+(ka)^2)."""
    n_theta, n_phi = 13, 24
    theta = torch.linspace(0.0, math.pi, n_theta + 2, dtype=torch.float64)[1:-1]
    phi = torch.linspace(0.0, 2.0 * math.pi, n_phi + 1, dtype=torch.float64)[:-1]
    th, ph = torch.meshgrid(theta, phi, indexing="ij")
    th, ph = th.reshape(-1), ph.reshape(-1)
    normals = torch.stack((torch.sin(th) * torch.cos(ph), torch.sin(th) * torch.sin(ph), torch.cos(th)), dim=1)
    points = RADIUS * normals
    velocity = torch.ones(points.shape[0], dtype=torch.float64)
    # spherical quadrature weights: sin(theta) dtheta dphi times a^2
    dth = math.pi / (n_theta + 1)
    dph = 2.0 * math.pi / n_phi
    weights = (RADIUS**2) * torch.sin(th) * dth * dph
    for ka in (0.5, 1.0, 2.0, 3.0):
        res = equivalent_source_power(points, normals, velocity, frequency=ka * C / (2.0 * math.pi * RADIUS),
                                      rho=RHO, c=C, area_weights=weights)
        exact = 2.0 * math.pi * RADIUS**2 * RHO * C * ka**2 / (1.0 + ka**2)
        assert res.power / exact == pytest.approx(1.0, rel=1e-2), ka


def test_window_closure_reaches_infinite_cylinder_value():
    """A very long cylinder mode must recover the infinite-cylinder modal sigma."""
    n, m, length, freq = 2, 1, 400.0, 60.0
    res = finite_cylinder_mode_power(m, n, radius=RADIUS, length=length, frequency=freq,
                                     rho=RHO, c=C, samples=8001, half_width=12.0)
    k = 2.0 * math.pi * freq / C
    ref = infinite_cylinder_efficiency(n, k, m * math.pi / length, radius=RADIUS, rho=RHO, c=C)
    assert res.sigma / ref == pytest.approx(1.0, rel=2e-2)


def test_rigid_body_null_and_velocity_squared_scaling():
    kw = dict(radius=RADIUS, length=2.0, frequency=150.0, rho=RHO, c=C, samples=2001, half_width=16.0)
    zero = finite_cylinder_mode_power(1, 2, velocity_amplitude=0.0, **kw)
    assert abs(zero.power) <= 1e-30
    one = finite_cylinder_mode_power(1, 2, velocity_amplitude=1.0, **kw)
    two = finite_cylinder_mode_power(1, 2, velocity_amplitude=2.0, **kw)
    assert two.power / one.power == pytest.approx(4.0, rel=1e-9)


def test_lateral_area_and_rigid_caps():
    points, normals, weights, meta = cylinder_surface_grid(radius=RADIUS, length=2.0, n_axial=12, n_circ=32)
    n_lateral = int(meta[0]) * int(meta[1])
    lateral = float(weights[:n_lateral].sum())
    assert lateral == pytest.approx(2.0 * math.pi * RADIUS * 2.0, rel=1e-12)
    assert abs(float(weights[n_lateral:].sum())) <= 1e-14


def test_axial_cutoff_returns_zero_not_nan():
    """Regression: at k_x == k the conductance divided 0/0 in the Hankel ratio."""
    k = 1.0 / RADIUS
    for n in (0, 1, 2, 3):
        value = infinite_cylinder_conductance(n, k, k, radius=RADIUS, rho=RHO, c=C)
        assert math.isfinite(value)
        assert value == 0.0
    for n in (0, 2):
        above = infinite_cylinder_conductance(n, k, k * (1.0 + 1e-6), radius=RADIUS, rho=RHO, c=C)
        assert abs(above) <= 1e-9
