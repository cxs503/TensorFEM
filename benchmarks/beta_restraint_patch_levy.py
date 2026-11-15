#!/usr/bin/env python3
"""Exact patch-load reference for the elastically restrained ice-belt panel.

Why this exists
---------------
``docs/CASE-FEM-ELASTIC-RESTRAINT-LEVY-VERIFICATION.md`` pinned the interior of
the restraint band with an *exact* Kirchhoff Levy solution - but only for a
**uniform** load.  The load case the paper (JOUR-ICE-STRUCT) actually applies is
a **pressure patch** of finite length, and
``docs/CASE-FEM-ELASTIC-RESTRAINT-TWO-WAY-PEAK.md`` explicitly left the
patch-load convergence table unmeasured after the patch-integration defect was
fixed.  So the two things a reader needs (a converged centre stress and an
independent reference for it at every ``beta``) were missing.

This benchmark supplies both:

1. exact reference - :func:`tensorfem.plate_levy_restraint.levy_restrained_plate_patch`
   solves the Kirchhoff plate with a piecewise-constant load in ``x`` in closed
   form (one ODE per sine mode in ``y``, ``C0..C2`` continuity at the load
   breaks, rotational spring on ``x = 0, a``).  No FEM, no discretisation in
   ``x``.
2. FEM mesh ladder - the existing Mindlin Q4 panel
   (:func:`tensorfem.plate_elastic_restraint.solve_restrained_panel`) at
   ``nx = 20, 40, 80, 160`` for each ``beta``, compared with (1) term by term.

Three independent consistency checks are recorded with the numbers:
``patch = full width`` must reproduce the uniform-load Levy solution to machine
precision, ``beta = 0`` must reproduce the published 401-term Navier baseline of
``panel_patch_load_results.json``, and :func:`ritz_restrained_plate` (a
variational solution of the same patch problem) must agree with the exact
series.

Usage
-----
One ``beta`` per invocation (so the sweep can be sharded on the cluster)::

    python benchmarks/beta_restraint_patch_levy.py --beta 0.4 --nx 20,40,80,160 \
        --out outputs/beta_patch_levy_beta0.4.json

    python benchmarks/beta_restraint_patch_levy.py --merge outputs/*.json \
        --out benchmarks/artifacts/beta_restraint_patch_levy.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.plate_elastic_restraint import (  # noqa: E402
    beta_to_zeta,
    peak_bending_stress,
    rect_patch_pressure,
    rotational_stiffness,
    solve_restrained_panel,
)
from tensorfem.plate_levy_restraint import (  # noqa: E402
    flexural_rigidity,
    levy_restrained_plate,
    levy_restrained_plate_patch,
    ritz_restrained_plate,
)

# paper design point (papers/jour-ice-struct/data/panel_patch_load_results.json)
L, S, T = 0.8, 0.7, 0.022
E, NU = 206.0e9, 0.3
P_REF = 1.0e6
PATCH_C = 0.1 * L                                  # 8 cm patch length
X1, X2 = 0.5 * (L - PATCH_C), 0.5 * (L + PATCH_C)
BASELINE_SIGMA_MPA = 73.37101604723352             # Navier N=401, centre vM
LEVY_MODES = 401
RITZ_MODES = (30, 80)


def von_mises_bending(Mx: float, My: float, t: float, Mxy: float = 0.0) -> float:
    """``sigma_vM = 6/t^2 sqrt(Mx^2 - Mx My + My^2 + 3 Mxy^2)`` [Pa]."""
    return 6.0 / t**2 * math.sqrt(
        Mx * Mx - Mx * My + My * My + 3.0 * Mxy * Mxy
    )


def reference_row(beta: float) -> dict:
    """Exact Levy + Ritz reference for one ``beta`` on the x-edges (``beta_y=0``)."""
    D = flexural_rigidity(E, NU, T)
    K = rotational_stiffness(beta, E, NU, T, L)
    levy = levy_restrained_plate_patch(
        L, S, D, NU, K, P_REF, (X1, X2), n_modes=LEVY_MODES
    )
    Mx, My = levy.moments_centre()
    ritz = ritz_restrained_plate(
        L, S, D, NU, K, 0.0, P_REF, m_modes=RITZ_MODES[0], n_modes=RITZ_MODES[1],
        patch=(X1, X2, 0.0, S),
    )
    ritz_Mx, ritz_My = ritz.moments_centre()
    return {
        "beta_x": beta,
        "beta_y": 0.0,
        "zeta": beta_to_zeta(beta),
        "K_rot_Nm_per_rad": K,
        "levy_n_modes": LEVY_MODES,
        "levy_dropped_modes": len(levy.dropped_modes),
        "levy_w_centre_mm": levy.w_centre() * 1e3,
        "levy_Mx_centre_Nm_per_m": Mx,
        "levy_My_centre_Nm_per_m": My,
        "levy_sigma_vm_centre_MPa": von_mises_bending(Mx, My, T) / 1e6,
        "ritz_modes": list(RITZ_MODES),
        "ritz_w_centre_mm": ritz.w_centre() * 1e3,
        "ritz_sigma_vm_centre_MPa": von_mises_bending(ritz_Mx, ritz_My, T) / 1e6,
        "levy_vs_ritz_w_rel_pct": (levy.w_centre() / ritz.w_centre() - 1.0) * 100.0,
        "levy_vs_ritz_sigma_vm_rel_pct": (
            von_mises_bending(Mx, My, T) / von_mises_bending(ritz_Mx, ritz_My, T) - 1.0
        ) * 100.0,
    }


def fem_row(beta: float, nx: int) -> dict:
    """One Mindlin panel solve on the paper case, load = the ice patch."""
    ny = max(1, int(round(nx * S / L)))
    pressure = rect_patch_pressure(P_REF, X1, X2, 0.0, S)
    res = solve_restrained_panel(
        nx, ny, L, S, E, NU, T, pressure, beta_x=beta, beta_y=0.0
    )
    peak = peak_bending_stress(res)
    return {
        "beta_x": beta,
        "nx": nx,
        "ny": ny,
        "h_mm": L / nx * 1e3,
        "K_rot_x_Nm_per_rad": res.K_rot_x,
        "sigma_vm_centre_MPa": peak["centre_von_mises"] / 1e6,
        "Mx_centre_Nm_per_m": peak["centre_Mx"],
        "My_centre_Nm_per_m": peak["centre_My"],
        "w_centre_mm": peak["w_centre"] * 1e3,
        # global max is kept for the record only: it sits on the patch edge and
        # is mesh-sensitive (see docs/...TWO-WAY-PEAK.md Sec. 5.1)
        "sigma_vm_max_MPa": peak["max_von_mises"] / 1e6,
    }


def consistency_checks() -> dict:
    """Machine-precision identities that the exact reference must satisfy."""
    D = flexural_rigidity(E, NU, T)
    K = rotational_stiffness(0.4, E, NU, T, L)
    full = levy_restrained_plate_patch(L, S, D, NU, K, P_REF, (0.0, L), n_modes=LEVY_MODES)
    uni = levy_restrained_plate(L, S, D, NU, K, P_REF, n_modes=LEVY_MODES)
    beta0 = levy_restrained_plate_patch(L, S, D, NU, 0.0, P_REF, (X1, X2), n_modes=LEVY_MODES)
    Mx0, My0 = beta0.moments_centre()
    sigma0 = von_mises_bending(Mx0, My0, T) / 1e6
    return {
        "full_width_patch_vs_uniform_w_rel": abs(full.w_centre() / uni.w_centre() - 1.0),
        "full_width_patch_vs_uniform_Mx_rel": abs(
            full.moments_centre()[0] / uni.moments_centre()[0] - 1.0
        ),
        "beta0_patch_sigma_vm_centre_MPa": sigma0,
        "published_navier_N401_sigma_vm_centre_MPa": BASELINE_SIGMA_MPA,
        "beta0_vs_published_rel_pct": (sigma0 / BASELINE_SIGMA_MPA - 1.0) * 100.0,
    }


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:  # pragma: no cover - diagnostics only
        return "unknown"


def run_beta(beta: float, meshes: list[int]) -> dict:
    ref = reference_row(beta)
    rows = [fem_row(beta, nx) for nx in meshes]
    for r in rows:
        r["sigma_vm_err_vs_levy_pct"] = (
            r["sigma_vm_centre_MPa"] / ref["levy_sigma_vm_centre_MPa"] - 1.0
        ) * 100.0
        r["w_err_vs_levy_pct"] = (r["w_centre_mm"] / ref["levy_w_centre_mm"] - 1.0) * 100.0
    # observed order between consecutive meshes (centre stress)
    for i in range(1, len(rows)):
        e0 = abs(rows[i - 1]["sigma_vm_err_vs_levy_pct"])
        e1 = abs(rows[i]["sigma_vm_err_vs_levy_pct"])
        if e0 > 0.0 and e1 > 0.0:
            ratio = rows[i]["nx"] / rows[i - 1]["nx"]
            rows[i]["observed_order_sigma"] = math.log(e0 / e1) / math.log(ratio)
    return {
        "generated_by": "benchmarks/beta_restraint_patch_levy.py",
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "case": {
            "description": "JOUR-ICE-STRUCT design point, full-breadth ice patch",
            "L_m": L, "S_m": S, "t_m": T, "E_Pa": E, "nu": NU,
            "patch_length_m": PATCH_C, "patch_x_m": [X1, X2],
            "patch_breadth_m": S, "p_MPa": P_REF / 1e6,
            "D_Nm": flexural_rigidity(E, NU, T),
            "restraint": "x-edges sprung with beta_x, y-edges simply supported",
        },
        "reference": ref,
        "fem": rows,
    }


def merge(paths: list[str]) -> dict:
    parts = [json.loads(Path(p).read_text()) for p in paths]
    parts.sort(key=lambda d: d["reference"]["beta_x"])
    out = {
        "generated_by": "benchmarks/beta_restraint_patch_levy.py --merge",
        "git_sha": parts[0]["git_sha"],
        "python": parts[0]["python"],
        "numpy": parts[0]["numpy"],
        "case": parts[0]["case"],
        "reference_by_beta": [p["reference"] for p in parts],
        "fem_rows": [r for p in parts for r in p["fem"]],
    }
    refs = out["reference_by_beta"]
    if refs:
        s0 = refs[0]["levy_sigma_vm_centre_MPa"]
        s1 = refs[-1]["levy_sigma_vm_centre_MPa"]
        out["band_exact_levy"] = {
            "sigma_vm_beta0_MPa": s0,
            "sigma_vm_beta1_MPa": s1,
            "reduction_vs_beta0_pct": (1.0 - s1 / s0) * 100.0,
            "note": "beta=1 on the x-edges only; y-edges remain simply supported",
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--beta", type=float, help="single embedding coefficient")
    ap.add_argument("--nx", default="20,40,80,160", help="comma separated mesh ladder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge", nargs="*", help="merge these part files instead")
    ap.add_argument("--checks-only", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.merge is not None:
        paths = [p for pattern in args.merge for p in sorted(glob.glob(pattern))]
        payload = merge(paths)
        payload["consistency_checks"] = consistency_checks()
        out.write_text(json.dumps(payload, indent=2))
        print(f"merged {len(paths)} parts -> {out}")
        return 0

    if args.checks_only:
        payload = {"checks": consistency_checks()}
    else:
        if args.beta is None:
            ap.error("--beta is required unless --merge/--checks-only is used")
        meshes = [int(v) for v in args.nx.split(",") if v.strip()]
        payload = run_beta(args.beta, meshes)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload.get("reference", payload.get("checks")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())