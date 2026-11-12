"""Specimen-level energy-release-rate verification of the 3-D cohesive DCB.

Closes the *energy release rate* half of the ``GAP-FEM-3D-COHESIVE``
acceptance criterion ("P-delta curve **and energy release rate** vs
analytical < 10 %").  The peak-load half is documented as strength-controlled
(LEFM is not the applicable reference; see
``docs/GAP-FEM-3D-COHESIVE-displacement-control-repair.md``).

Physical anchor: for a cohesive interface the work over a complete separation
of one unit area is, by construction of the law, ``G_Ic``.  Hence a
quasi-statically advancing front must release ``G_Ic`` per unit of newly
cracked area::

    G = d W_interface / d A_cracked  ==  G_Ic        (pure Mode I)

The test drives the production HEX8 + COH3D8 DCB (same specimen, law and mesh
family as the displacement-control probe) far enough that the front propagates
a measurable length, then checks the local slope of ``W_interface`` vs
``A_cracked`` over the steady-propagation window against ``G_Ic``.

Registered in the slow lane (``tests/slow_lane_manifest.txt``): it is a
minute-scale nonlinear path-follower run.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorfem.cohesive3d import BKCohesive3DLaw
from tensorfem.cohesive3d_dcb import (
    DCBGeometry,
    build_dcb_model,
    solve_dcb,
)

torch.set_num_threads(1)

# --- specimen / law: identical to the displacement-control probe -----------
E_ADH, NU_ADH, H_ARM = 70.0e9, 0.30, 3.0e-3
TOUGHNESS = 300.0          # G_Ic [J/m^2]
LENGTH, WIDTH, CRACK = 60e-3, 3e-3, 30e-3
T_N0, K_N, BK_ETA = 5.0e6, 1e12, 2.2
NX, NZ = 60, 2
DELTA_MAX, N_STEPS, TOL = 4.0e-3, 160, 1e-8
TOL_G = 0.10                # main criterion: |G - G_Ic| / G_Ic <= 10 %
TOL_GLOBAL = 0.15           # supporting bound for the global W/A ratio


def _law() -> BKCohesive3DLaw:
    return BKCohesive3DLaw(K_N, K_N / 2.5, T_N0, 0.6 * T_N0,
                          TOUGHNESS, 800.0, BK_ETA)


def _geometry() -> DCBGeometry:
    return DCBGeometry(length=LENGTH, width=WIDTH, arm_thickness=H_ARM,
                       initial_crack=CRACK, nx=NX, ny=1, nz=NZ)


@pytest.mark.slow
def test_energy_release_rate_matches_gic():
    """The incremental cohesive work per unit of newly cracked area equals
    ``G_Ic`` (the area under the traction-separation curve) within 10 %."""
    res = solve_dcb(build_dcb_model(_geometry(), _law()),
                    delta_max=DELTA_MAX, n_steps=N_STEPS, tol=TOL)
    st = res.stats
    assert st["final_failed_area"] > 0.0, "no interface failure recorded"

    disp = np.array([c["disp"] for c in res.curve])
    w_if = np.array([c["interface_work"] for c in res.curve])
    a_fail = np.array([c["failed_area"] for c in res.curve])
    front = np.array([c["crack_front_m"] for c in res.curve])
    ext = front - CRACK

    # steady-propagation window: front moved > 1 mm and stays > 2 mm from the
    # end of the bonded region (avoid the initiation transient and end-effect)
    idx = np.flatnonzero((ext > 1.0e-3) & (front < LENGTH - 2.0e-3))
    assert idx.size >= 3, (
        f"crack did not propagate into the steady window "
        f"(max extension {ext.max():.3e} m)")
    lo, hi = int(idx[0]), int(idx[-1])

    assert a_fail[hi] > a_fail[lo] + 1e-12
    g_local = (w_if[hi] - w_if[lo]) / (a_fail[hi] - a_fail[lo])
    rel = (g_local - TOUGHNESS) / TOUGHNESS
    assert math.isfinite(rel)
    assert abs(rel) <= TOL_G, (
        f"energy release rate {g_local:.2f} J/m^2 deviates "
        f"{rel * 100:+.1f} % from G_Ic = {TOUGHNESS} J/m^2 "
        f"(tol {TOL_G * 100:.0f} %)")

    # supporting evidence: the global ratio at the final state is an upper
    # bound (the numerator also carries the process-zone work) and stays
    # inside a loose bound of G_Ic
    g_global = st["final_interface_work"] / st["final_failed_area"]
    rel_g = (g_global - TOUGHNESS) / TOUGHNESS
    assert abs(rel_g) <= TOL_GLOBAL, (
        f"global W/A = {g_global:.2f} J/m^2 deviates {rel_g * 100:+.1f} % "
        f"from G_Ic (loose bound {TOL_GLOBAL * 100:.0f} %)")
