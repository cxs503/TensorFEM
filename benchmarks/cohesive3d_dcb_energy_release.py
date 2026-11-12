#!/usr/bin/env python
"""Specimen-level *energy release rate* verification of the 3-D cohesive DCB.

Closes the second half of the ``GAP-FEM-3D-COHESIVE`` acceptance criterion:
"*P-delta curve* **and energy release rate** vs analytical < 10 %".

The peak-load half is documented as strength-controlled (LEFM is not the
applicable reference; see ``docs/GAP-FEM-3D-COHESIVE-displacement-control-
repair.md``).  The energy-release-rate half was never measured: the committed
evidence JSON carries ``interface_work_per_area = NaN`` because the quick runs
stop before any interface element fully fails.

Physical anchor (independent of the FE)
---------------------------------------
For a cohesive interface the work done by the traction over a *complete*
separation of one unit area is, by construction of the law, the critical
energy release rate ``G_Ic`` (the area under the traction-separation curve).
Therefore, when a crack front advances quasi-statically through the width,
the incremental cohesive work per unit of newly cracked area must equal
``G_Ic``::

    G = d W_interface / d A_cracked  ==  G_Ic        (pure Mode I)

This script drives the production HEX8 + COH3D8 DCB (same specimen, law and
mesh family as the displacement-control probe) far enough that the front
propagates a measurable length, then measures ``G`` three ways and compares
each to ``G_Ic = 300 J/m^2``:

1. **local slope**  -- ``(W_hi - W_lo) / (A_hi - A_lo)`` over the
   steady-propagation window (the physically correct derivative);
2. **window fit**   -- least-squares slope of ``W_interface`` vs ``A_cracked``
   over the same window (robust to step noise);
3. **global ratio** -- ``W_interface / A_cracked`` at the final state
   (upper bound: the numerator also carries the process-zone work).

Usage::

    PYTHONPATH=src python benchmarks/cohesive3d_dcb_energy_release.py

Exit status 0 when the local-slope criterion is inside its predeclared
tolerance; the other two are reported as supporting evidence.
"""
from __future__ import annotations

import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.cohesive3d import BKCohesive3DLaw  # noqa: E402
from tensorfem.cohesive3d_dcb import (  # noqa: E402
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
NX, NZ = 60, 2             # same mesh family as the converged peak run
DELTA_MAX, N_STEPS, TOL = 4.0e-3, 160, 1e-8
TOL_G = 0.10                # predeclared: |G - G_Ic| / G_Ic <= 10 %


def law() -> BKCohesive3DLaw:
    return BKCohesive3DLaw(K_N, K_N / 2.5, T_N0, 0.6 * T_N0,
                          TOUGHNESS, 800.0, BK_ETA)


def geometry() -> DCBGeometry:
    return DCBGeometry(length=LENGTH, width=WIDTH, arm_thickness=H_ARM,
                       initial_crack=CRACK, nx=NX, ny=1, nz=NZ)


def _enc(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main() -> int:
    t0 = time.time()
    model = build_dcb_model(geometry(), law())

    res = solve_dcb(model, delta_max=DELTA_MAX, n_steps=N_STEPS, tol=TOL)
    curve = res.curve
    stats = res.stats

    a0 = CRACK
    disp = np.array([c["disp"] for c in curve])
    load = np.array([c["load"] for c in curve])
    w_if = np.array([c["interface_work"] for c in curve])
    a_fail = np.array([c["failed_area"] for c in curve])
    front = np.array([c["crack_front_m"] for c in curve])
    ext = front - a0                       # crack extension [m]

    # steady-propagation window: front has moved > 1 mm and stays > 2 mm away
    # from the end of the bonded region (avoid the initiation transient and
    # the final end-effect)
    mask = (ext > 1.0e-3) & (front < LENGTH - 2.0e-3)
    idx = np.flatnonzero(mask)
    if idx.size < 3:
        print("crack did not propagate into the steady window; "
              f"max extension = {ext.max():.4e} m", file=sys.stderr)
        return 2
    lo, hi = int(idx[0]), int(idx[-1])
    win = slice(lo, hi + 1)

    g_local = (w_if[hi] - w_if[lo]) / (a_fail[hi] - a_fail[lo]) \
        if a_fail[hi] > a_fail[lo] + 1e-12 else float("nan")
    g_fit = float(np.polyfit(a_fail[win], w_if[win], 1)[0]) \
        if np.ptp(a_fail[win]) > 1e-12 else float("nan")
    g_global = w_if[-1] / a_fail[-1] if a_fail[-1] > 0 else float("nan")

    rel_local = (g_local - TOUGHNESS) / TOUGHNESS
    rel_fit = (g_fit - TOUGHNESS) / TOUGHNESS
    rel_global = (g_global - TOUGHNESS) / TOUGHNESS

    passed = bool(math.isfinite(rel_local) and abs(rel_local) <= TOL_G)

    out = {
        "schema": "tensorfem.cohesive3d-dcb-energy-release/1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "benchmarks/cohesive3d_dcb_energy_release.py",
        "host": {"platform": platform.platform(),
                 "python": platform.python_version(),
                 "torch": torch.__version__},
        "parameters": {"E_Pa": E_ADH, "nu": NU_ADH, "arm_thickness_m": H_ARM,
                       "width_m": WIDTH, "length_m": LENGTH,
                       "initial_crack_m": CRACK, "G_Ic_J_m2": TOUGHNESS,
                       "t_n0_Pa": T_N0, "k_n_Pa_per_m": K_N, "bk_eta": BK_ETA,
                       "nx": NX, "nz": NZ, "delta_max_m": DELTA_MAX,
                       "n_steps": N_STEPS, "tol": TOL},
        "run": {"converged_all": stats["converged_all"],
                 "steps": stats["steps"],
                 "newton_iterations": stats["newton_iterations"],
                 "n_dofs": stats["n_dofs"], "n_iface": stats["n_iface"],
                 "reach_delta_m": stats["reach_delta"],
                 "final_failed_area_m2": stats["final_failed_area"],
                 "final_interface_work_J": stats["final_interface_work"],
                 "final_strain_energy_J": stats["final_strain_energy"],
                 "final_external_work_J": stats["final_external_work"],
                 "energy_balance_rel": stats["energy_balance_rel"],
                 "crack_extension_m": float(ext[-1]),
                 "peak_load_N": float(load.max()),
                 "peak_disp_m": float(disp[int(np.argmax(load))])},
        "criterion": {
            "name": "energy_release_rate_vs_GIc",
            "criterion": "dW_interface/dA_cracked (steady window) vs G_Ic",
            "ref_J_m2": TOUGHNESS,
            "got_local_J_m2": g_local,
            "got_fit_J_m2": g_fit,
            "got_global_J_m2": g_global,
            "rel_error_local": rel_local,
            "rel_error_fit": rel_fit,
            "rel_error_global": rel_global,
            "tolerance": "|rel_error_local| <= 0.10",
            "passed": passed,
        },
        "window": {"lo_step": lo, "hi_step": hi,
                    "ext_lo_m": float(ext[lo]), "ext_hi_m": float(ext[hi]),
                    "a_fail_lo_m2": float(a_fail[lo]),
                    "a_fail_hi_m2": float(a_fail[hi])},
        "curve": [{"disp": float(d), "load": float(p),
                   "interface_work_J": float(w), "failed_area_m2": float(a),
                   "crack_front_m": float(f)}
                  for d, p, w, a, f in zip(disp, load, w_if, a_fail, front)],
        "runtime_s": time.time() - t0,
    }

    here = Path(__file__).resolve().parent
    (here / "cohesive3d_dcb_energy_release.json").write_text(
        json.dumps(out, indent=1, default=_enc))
    print(json.dumps({k: out[k] for k in ("run", "criterion", "window")},
                     indent=1, default=_enc))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
