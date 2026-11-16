"""Tests for the geometrical-wetting (Akai 2018 / Wenzel) audit.

Honest gates:
  G1  the vectorized bounce-back step reproduces the reference to machine
      precision (max |df| == 0)
  G2  the Wenzel effective-angle helper is exact against acos(r*cos(theta))
  G3  the full droplet LBM field is *not* trusted: this test documents that
      the merged geometric-wetting feature is numerically unstable (NaN) and
      that measure_contact_angle returns a hard-coded default 90.0 when no
      valid interface is found -- i.e. the existing "contact-angle validation"
      is a false pass, not real evidence.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfem.lbm_wetting import (
    _lbm_step,
    _lbm_step_vec,
    _wenzel_effective_angle,
    GeometricalWettingConfig,
    run_geometrical_wetting_validation,
)


class TestVectorizedStep:
    def test_reproduces_reference(self):
        ny, nx = 24, 24
        rng = np.random.default_rng(0)
        f = rng.uniform(0.05, 0.5, size=(9, ny, nx))
        solid = np.zeros((ny, nx), dtype=bool)
        solid[:2, :] = True
        solid[10:14, 8:16] = True
        f_ref = _lbm_step(f, 1.0, solid)
        f_vec = _lbm_step_vec(f, 1.0, solid)
        assert np.max(np.abs(f_ref - f_vec)) < 1e-12


class TestWenzelHelper:
    def test_exact(self):
        for theta in (30.0, 45.0, 60.0, 90.0, 120.0, 150.0):
            for r in (1.0, 1.2, 1.5, 2.0):
                ce = max(-1.0, min(1.0, r * math.cos(math.radians(theta))))
                ref = math.degrees(math.acos(ce))
                assert abs(_wenzel_effective_angle(theta, r) - ref) < 1e-12

    def test_flat_wall_identity(self):
        # r = 1 -> effective angle == intrinsic
        assert _wenzel_effective_angle(90.0, 1.0) == pytest.approx(90.0)


class TestHonestInstability:
    def test_field_is_nan_not_trusted(self):
        """The merged geometric-wetting droplet solver is unstable (NaN).

        This is a *documentation* test: it asserts the honest state so that a
        future fix that makes the field finite will flip this test and be
        caught.  It must NOT be read as a passing validation.
        """
        cfg = GeometricalWettingConfig(
            nx=48, ny=48, droplet_radius=10.0,
            droplet_center_x=24.0, droplet_center_y=18.0,
            pillar_width=8, pillar_height=4,
            intrinsic_angle_deg=90.0, max_steps=800,
        )
        res = run_geometrical_wetting_validation(cfg)
        rho = res["rho_field"]
        # The field is NaN -> the "measured" angle is the hard-coded default 90.0
        assert not np.isfinite(rho).all(), (
            "field became finite -- the instability was fixed; "
            "re-run the full Akai2018/Wenzel benchmark before trusting it"
        )
        assert res["measured_angle_deg"] == 90.0, (
            "measure_contact_angle returned a non-default value on a NaN field"
        )
