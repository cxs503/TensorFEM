"""XVAL-LBM-AKAI2018-GEOMETRICAL-WETTING -- compute stage.

Geometrical (structured-surface) wetting in D2Q9 LBM: a droplet rests on a
surface carrying a central rectangular pillar.  The geometric wetting BC
enforces the *intrinsic* contact angle on the flat wall; the pillar makes the
surface geometrically rough, so the *effective* (apparent) contact angle
measured at the droplet edge is compared against the Wenzel (1936) analytical
prediction ``cos(theta_eff) = r * cos(theta)`` with ``r = 1 + 2*h/w`` (2-D
pillar: top + two side walls over projected width ``w``).

This is the "geometrical wetting" validation pattern of lanl/MF-LBM's
``akai2018-geometrical-wetting`` case: the contact angle is *not* imposed as a
free parameter but emerges from the geometry of the solid surface.

Predeclared gates (fixed before the final sweep):

  G1  the vectorized bounce-back step ``_lbm_step_vec`` reproduces the
      reference ``_lbm_step`` field to machine precision (max |df| < 1e-12)
  G2  the Wenzel effective-angle helper is exact against the closed form
      ``acos(r*cos(theta))`` (max |error| == 0)
  G3  for a flat wall (r = 1) the measured effective angle equals the
      intrinsic angle to < 3 %
  G4  for a rough wall (r = 1 + 2h/w) the measured effective angle tracks the
      Wenzel prediction to < 15 % (owner precision standard)
  G5  the sweep over intrinsic angles {60, 90, 120} deg is monotone in the
      Wenzel direction (theta_eff increases with theta)

Run:  python benchmarks/verified/lbm_akai2018_geometrical_wetting/run.py
Out:  benchmarks/verified/lbm_akai2018_geometrical_wetting/result.json
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np  # noqa: E402

from tensorfem.lbm_wetting import (  # noqa: E402
    _lbm_step,
    _lbm_step_vec,
    _wenzel_effective_angle,
    geometrical_wetting_xval,
    run_geometrical_wetting_validation,
    GeometricalWettingConfig,
)


def _check_g1() -> dict:
    """Vectorized step reproduces the reference step to machine precision."""
    ny, nx = 24, 24
    rng = np.random.default_rng(0)
    f = rng.uniform(0.05, 0.5, size=(9, ny, nx))
    solid = np.zeros((ny, nx), dtype=bool)
    solid[:2, :] = True
    solid[10:14, 8:16] = True
    tau = 1.0
    f_ref = _lbm_step(f, tau, solid)
    f_vec = _lbm_step_vec(f, tau, solid)
    max_df = float(np.max(np.abs(f_ref - f_vec)))
    return {"max_abs_df": max_df, "passed": max_df < 1e-12}


def _check_g2() -> dict:
    """Wenzel helper is exact against the closed form."""
    worst = 0.0
    for theta in (30.0, 45.0, 60.0, 90.0, 120.0, 150.0):
        for r in (1.0, 1.2, 1.5, 2.0):
            import math
            cos_eff = r * math.cos(math.radians(theta))
            cos_eff = max(-1.0, min(1.0, cos_eff))
            ref = math.degrees(math.acos(cos_eff))
            got = _wenzel_effective_angle(theta, r)
            worst = max(worst, abs(got - ref))
    return {"max_abs_error": worst, "passed": worst == 0.0}


def _check_g3() -> dict:
    """Flat wall (r=1): measured effective angle == intrinsic to < 3 %."""
    cfg = GeometricalWettingConfig(
        nx=64, ny=64, droplet_radius=14.0,
        droplet_center_x=32.0, droplet_center_y=26.0,
        pillar_width=64, pillar_height=0,  # no pillar -> flat wall
        intrinsic_angle_deg=90.0, max_steps=1500,
    )
    res = run_geometrical_wetting_validation(cfg)
    return {
        "intrinsic": res["intrinsic_angle_deg"],
        "measured": res["measured_angle_deg"],
        "error_pct": res["error_pct"],
        "passed": res["error_pct"] < 3.0,
    }


def _check_g4() -> dict:
    """Rough wall: measured effective angle tracks Wenzel to < 15 %."""
    cfg = GeometricalWettingConfig(
        nx=96, ny=96, droplet_radius=18.0,
        droplet_center_x=48.0, droplet_center_y=34.0,
        pillar_width=12, pillar_height=6,
        intrinsic_angle_deg=90.0, max_steps=2500,
    )
    res = run_geometrical_wetting_validation(cfg)
    return {
        "intrinsic": res["intrinsic_angle_deg"],
        "roughness": res["roughness"],
        "wenzel": res["wenzel_angle_deg"],
        "measured": res["measured_angle_deg"],
        "error_pct": res["error_pct"],
        "passed": res["error_pct"] < 15.0,
    }


def _check_g5() -> dict:
    """Sweep over intrinsic angles is monotone in the Wenzel direction."""
    results = geometrical_wetting_xval(
        intrinsic_angles=[60.0, 90.0, 120.0],
        nx=96, ny=96, max_steps=2500,
    )
    measured = [r["measured_angle_deg"] for r in results]
    wenzel = [r["wenzel_angle_deg"] for r in results]
    monotone = all(
        measured[i] < measured[i + 1] for i in range(len(measured) - 1)
    )
    return {
        "intrinsic": [r["intrinsic_angle_deg"] for r in results],
        "wenzel": wenzel,
        "measured": measured,
        "errors_pct": [r["error_pct"] for r in results],
        "monotone": monotone,
        "passed": monotone,
    }


def main() -> None:
    gates = {
        "G1_vectorized_step_repro": _check_g1(),
        "G2_wenzel_exact": _check_g2(),
        "G3_flat_wall_intrinsic": _check_g3(),
        "G4_rough_wall_wenzel": _check_g4(),
        "G5_sweep_monotone": _check_g5(),
    }
    all_pass = all(g["passed"] for g in gates.values())
    out = {
        "anchor": "XVAL-LBM-AKAI2018-GEOMETRICAL-WETTING",
        "all_gates_passed": all_pass,
        "gates": gates,
    }
    path = os.path.join(HERE, "result.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    print("ALL GATES PASSED:", all_pass)


if __name__ == "__main__":
    main()
