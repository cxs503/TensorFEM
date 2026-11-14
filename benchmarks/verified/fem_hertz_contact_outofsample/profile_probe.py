#!/usr/bin/env python3
"""Field-level probe at the *out-of-sample* point a/R = 0.075.

The certificate's field verification (``docs/CASE-FEM-HERTZ-CONTACT-field-
verification.md``) checks the recovered surface pressure and the elastic field
only at the two points it also certifies (a/R = 0.05, 0.10).  This probe
records the same surface quantities at the interpolation point a/R = 0.075 so
the report can show a *field* comparison that the anchor was never graded on.

Run:
    PYTHONPATH=src python3 benchmarks/verified/fem_hertz_contact_outofsample/profile_probe.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))

from tensorfem.hertz_halfspace_loadcontrol import (  # noqa: E402
    graded_depth,
    hertz_load_reference,
    solve_axisymmetric_contact_load_control,
    two_zone_axis,
)

R, E, NU = 10.0, 1.0e5, 0.3
RATIO = 0.075
H_REL = 0.0075
DEPTH_FACTOR = 96.0


def main():
    e_star = E / (1.0 - NU**2)
    a_hertz = RATIO * R
    load = 4.0 * e_star * a_hertz**3 / (3.0 * R)
    ref = hertz_load_reference(load, R, E, NU)
    a, h = ref.contact_radius, H_REL * ref.contact_radius
    r_axis = two_zone_axis(3.0 * a, DEPTH_FACTOR * a, int(round(3.0 * a / h)), 26)
    z_axis = graded_depth(DEPTH_FACTOR * a, 4.0 * a, 24, 16)
    t0 = time.time()
    ref, state, diag = solve_axisymmetric_contact_load_control(
        load=load, radius=R, young=E, poisson=NU,
        r_axis=r_axis, z_axis=z_axis, rho_factor=1.0, load_tol=1e-10,
    )
    radii = np.asarray(state.surface_radii)
    inside = radii <= 1.05 * ref.contact_radius
    out = {
        "schema": "tensorfem.hertz.profile-probe.v1",
        "point": {"a_over_R": RATIO, "h_over_a": H_REL,
                  "depth_over_a": DEPTH_FACTOR, "n_elements": diag["n_elements"]},
        "analytic": {"contact_radius": ref.contact_radius,
                     "indentation": ref.indentation,
                     "peak_pressure": ref.peak_pressure},
        "numeric": {"contact_radius_linear": state.contact_radius_linear,
                    "indentation": state.prescribed_indentation,
                    "peak_pressure_consistent": float(
                        state.metadata["peak_pressure_consistent"]),
                    "peak_pressure_lumped": float(state.peak_pressure)},
        "wall_s": round(time.time() - t0, 2),
        "surface": {
            "r_over_a": (radii[inside] / ref.contact_radius).tolist(),
            "pressure_consistent_over_p0": (
                np.asarray(state.surface_pressure_consistent)[inside]
                / ref.peak_pressure).tolist(),
            "pressure_lumped_over_p0": (
                np.asarray(state.surface_pressure)[inside]
                / ref.peak_pressure).tolist(),
            "analytic_over_p0": (
                ref.pressure(radii[inside]) / ref.peak_pressure).tolist(),
        },
    }
    path = os.path.join(HERE, "probes", "profile_aR0075.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    print(f"wrote {path}  ({out['wall_s']}s, {diag['n_elements']} elements)")


if __name__ == "__main__":
    main()