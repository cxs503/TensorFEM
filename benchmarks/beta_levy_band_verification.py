"""Verification of the elastic-rotational-restraint panel against exact references.

Anchor: ``GAP-PP-4-ICE-STRUCT-ELASTIC-RESTRAINT`` (paper JOUR-ICE-STRUCT section
4.2 boundary convention).  The two ends of the *beta* band were already pinned
(beta = 0 to an independent Navier baseline, beta = 1 to Timoshenko's clamped
coefficients); the **interior of the band had no independent reference at all**,
which is exactly the quantity the manuscript needs in order to report a range
instead of an upper bound.

This benchmark closes that gap:

1. ``canonical`` sweep - square Kirchhoff plate, ``beta_x in {0, ..., 1}``,
   ``beta_y = 0``; reference = **exact Levy solution** of the same boundary
   value problem (:func:`tensorfem.plate_levy_restraint.levy_restrained_plate`),
   which involves no discretisation at all.
2. ``panel`` sweep - the manuscript panel (0.8 m x 0.7 m x 22 mm steel plate,
   ice patch over the full breadth), restraint on **all four** edges; reference
   = double-sine Rayleigh-Ritz (:func:`ritz_restrained_plate`) with a mode-count
   sensitivity bracket, because the Levy solution only handles one sprung pair.
3. ``mesh`` study on the canonical case so every reported error is a *converged*
   error, not a mesh coincidence.

Run::

    python benchmarks/beta_levy_band_verification.py            # full (~4 min)
    python benchmarks/beta_levy_band_verification.py --quick    # coarse
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from tensorfem.plate_elastic_restraint import (  # noqa: E402
    beta_to_zeta,
    moments_at_nodes,
    solve_restrained_panel,
)
from tensorfem.plate_levy_restraint import (  # noqa: E402
    flexural_rigidity,
    kirchhoff_strip_deflection,
    kirchhoff_strip_end_moment,
    levy_restrained_plate,
    ritz_restrained_plate,
)

ARTIFACTS = ROOT / "benchmarks" / "artifacts"

BETAS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _centroid_node(nodes: np.ndarray, a: float, b: float) -> int:
    return int(np.argmin(np.abs(nodes[:, 0] - a / 2.0) + np.abs(nodes[:, 1] - b / 2.0)))


def _fem(nx, ny, a, b, E, nu, t, q, beta_x, beta_y):
    res = solve_restrained_panel(
        nx, ny, a, b, E, nu, t, q, beta_x=beta_x, beta_y=beta_y
    )
    nodes = res.nodes.numpy()
    mom = moments_at_nodes(res).numpy()
    k = _centroid_node(nodes, a, b)
    return res, k, float(res.w().numpy()[k]), float(mom[k, 0]), float(mom[k, 1])


# ---------------------------------------------------------------------------
# 1. canonical plate: exact Levy reference
# ---------------------------------------------------------------------------

def canonical_sweep(meshes, n_modes=2001):
    """Square thin plate, x-edges sprung, y-edges simple; Levy is exact."""
    a = b = 1.0
    E, nu, t, q = 200.0e9, 0.3, 0.01, 1000.0
    D = flexural_rigidity(E, nu, t)

    rows = []
    for beta in BETAS:
        K = beta_to_zeta(beta) * D / a
        exact = levy_restrained_plate(a, b, D, nu, K, q, n_modes=n_modes)
        w_ex, mx_ex = exact.w_centre(), exact.moments_centre()[0]
        entry = {
            "beta": beta,
            "K_rot_Nm_per_m_per_rad": K,
            "exact_levy": {
                "w_centre_m": w_ex,
                "Mx_centre_Nm_per_m": mx_ex,
                "w_normalised_wD_over_qa4": w_ex * D / (q * a**4),
                "edge_moment_Nm_per_m": exact.moments(0.0, b / 2.0)[0],
                "edge_slope_dw_dx": exact.slope(0.0, b / 2.0),
            },
            "fem": [],
        }
        for nx, ny in meshes:
            _, _, w, mx, _ = _fem(nx, ny, a, b, E, nu, t, q, beta, 0.0)
            entry["fem"].append(
                {
                    "nx": nx,
                    "ny": ny,
                    "w_centre_m": w,
                    "Mx_centre_Nm_per_m": mx,
                    "err_w_pct": 100.0 * (w / w_ex - 1.0),
                    "err_Mx_pct": 100.0 * (mx / mx_ex - 1.0),
                }
            )
        rows.append(entry)
    return {
        "case": {
            "description": "square Kirchhoff plate, x-edges rotationally sprung, y-edges simple",
            "a_m": a,
            "b_m": b,
            "t_m": t,
            "E_Pa": E,
            "nu": nu,
            "q_Pa": q,
            "D_Nm": D,
            "beta_y": 0.0,
            "reference": "exact Levy series (no discretisation)",
        },
        "betas": list(BETAS),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# 2. manuscript panel: all four edges restrained, Ritz reference
# ---------------------------------------------------------------------------

PANEL = dict(
    a=0.8,       # span between transverse frames [m]
    b=0.7,       # frame spacing / breadth [m]
    t=0.022,     # plate thickness [m]
    E=206.0e9,   # steel
    nu=0.3,
    q=1.0e6,     # 1 MPa patch pressure over the full breadth
)


def panel_sweep(nx=80, ny=70, ritz_modes=(30, 40, 50)):
    a, b, t, E, nu, q = (PANEL[k] for k in ("a", "b", "t", "E", "nu", "q"))
    D = flexural_rigidity(E, nu, t)

    rows = []
    for beta in BETAS:
        Kx = beta_to_zeta(beta) * D / a
        Ky = beta_to_zeta(beta) * D / b
        ritz = {}
        for m in ritz_modes:
            sol = ritz_restrained_plate(a, b, D, nu, Kx, Ky, q, m, m)
            ritz[str(m)] = {
                "w_centre_m": sol.w_centre(),
                "Mx_centre_Nm_per_m": sol.moments_centre()[0],
            }
        refined = ritz[str(ritz_modes[-1])]
        _, _, w, mx, _ = _fem(nx, ny, a, b, E, nu, t, q, beta, beta)
        rows.append(
            {
                "beta": beta,
                "K_rot_x_Nm_per_m_per_rad": Kx,
                "K_rot_y_Nm_per_m_per_rad": Ky,
                "ritz": ritz,
                "fem": {
                    "nx": nx,
                    "ny": ny,
                    "w_centre_m": w,
                    "Mx_centre_Nm_per_m": mx,
                },
                "err_w_pct": 100.0 * (w / refined["w_centre_m"] - 1.0),
                "err_Mx_pct": 100.0 * (mx / refined["Mx_centre_Nm_per_m"] - 1.0),
                "ritz_truncation_spread_w_pct": 100.0
                * (ritz[str(ritz_modes[0])]["w_centre_m"] / refined["w_centre_m"] - 1.0),
            }
        )
    return {
        "case": {
            "description": "manuscript ice-belt panel, uniform pressure 1 MPa over full breadth",
            **PANEL,
            "D_Nm": D,
            "reference": "double-sine Rayleigh-Ritz (variational, upper bound in stiffness)",
        },
        "ritz_modes": list(ritz_modes),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# 3. cylindrical-bending strip: independent closed form
# ---------------------------------------------------------------------------

def strip_check():
    """b >> a limit of the Levy solution against beam closed forms."""
    L = 1.0
    D = 1.0
    nu = 0.3
    q = 1.0
    out = []
    for K in (0.0, 5.0, 50.0, 1e6):
        sol = levy_restrained_plate(L, 100.0 * L, D, nu, K, q, n_modes=6401)
        w_ref = kirchhoff_strip_deflection(q, L, D, K)
        m_ref = -kirchhoff_strip_end_moment(q, L, D, K)
        m_levy = sol.moments(0.0, 50.0 * L)[0]
        out.append(
            {
                "K": K,
                "w_centre_levy": sol.w_centre(),
                "w_centre_beam": w_ref,
                "err_w_pct": 100.0 * (sol.w_centre() / w_ref - 1.0),
                "edge_moment_levy": m_levy,
                "edge_moment_beam": m_ref,
                "err_M_pct": (
                    100.0 * (m_levy / m_ref - 1.0) if abs(m_ref) > 0 else float("nan")
                ),
            }
        )
    return {
        "case": {
            "description": "cylindrical bending limit (b = 100 a) vs beam closed form",
            "L_m": L,
            "D_Nm": D,
            "nu": nu,
            "q_Pa": q,
        },
        "rows": out,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="coarse meshes, short run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    meshes = [(10, 10), (20, 20), (40, 40)] if not args.quick else [(10, 10), (20, 20)]
    nx, ny = (80, 70) if not args.quick else (40, 35)
    ritz_modes = (30, 40, 50) if not args.quick else (20, 30, 40)

    print("[1/3] canonical Levy sweep ...", flush=True)
    canonical = canonical_sweep(meshes)
    print("[2/3] manuscript panel vs Ritz ...", flush=True)
    panel = panel_sweep(nx=nx, ny=ny, ritz_modes=ritz_modes)
    print("[3/3] strip limit ...", flush=True)
    strip = strip_check()

    payload = {
        "generated_by": "benchmarks/beta_levy_band_verification.py",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "canonical_levy": canonical,
        "panel_ritz": panel,
        "strip_limit": strip,
        "summary": {
            "max_abs_err_w_pct_finest_mesh": max(
                abs(r["fem"][-1]["err_w_pct"]) for r in canonical["rows"]
            ),
            "max_abs_err_Mx_pct_finest_mesh": max(
                abs(r["fem"][-1]["err_Mx_pct"]) for r in canonical["rows"]
            ),
            "max_abs_err_w_pct_panel_vs_ritz": max(abs(r["err_w_pct"]) for r in panel["rows"]),
            "max_abs_err_Mx_pct_panel_vs_ritz": max(abs(r["err_Mx_pct"]) for r in panel["rows"]),
            "strip_max_abs_err_w_pct": max(abs(r["err_w_pct"]) for r in strip["rows"]),
            "strip_max_abs_err_M_pct": max(
                abs(r["err_M_pct"]) for r in strip["rows"] if not math.isnan(r["err_M_pct"])
            ),
            "deflection_range_ssss_to_clamped_pct": 100.0
            * (
                1.0
                - canonical["rows"][-1]["exact_levy"]["w_centre_m"]
                / canonical["rows"][0]["exact_levy"]["w_centre_m"]
            ),
        },
    }

    out = Path(args.out) if args.out else ARTIFACTS / "beta_levy_band_verification.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())