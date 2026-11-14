#!/usr/bin/env python3
"""Beta band for an ice-belt plate field with elastically restrained edges.

Physical question (JOUR-ICE-STRUCT Sec. 4.2)
-------------------------------------------
The panel-level ice-pressure benchmark in the paper assumes a *simply
supported* (SSSS) plate field.  A real ice-belt plate field is bounded by two
longitudinals and two transverse frames, which provide a *finite* rotational
restraint: not zero (SSSS) and not infinite (clamped).  Consequently the SSSS
stress is only an upper bound; the paper cannot state how much of that bound is
actually realised.

This benchmark quantifies the band.  For an embedding coefficient
``beta in [0, 1]`` (0 = simply supported, 1 = clamped on the same edges) we
solve the Mindlin panel with rotational springs of stiffness
``K_rot = zeta(beta) D / span`` and report the centre von Mises bending stress.
Because the two spans are restrained by *different* members, ``beta_x`` (edges
restrained by the longitudinals) and ``beta_y`` (edges restrained by the
transverse frames) are swept independently.

Cases
-----
P  paper design point: 0.8 m x 0.7 m x 22 mm steel panel (E = 206 GPa,
   nu = 0.3), local ice patch of 1 MPa, patch = full breadth x 0.1 l
   (8 cm x 70 cm, A = 0.056 m^2) -- the row
   ``lambda_c_over_l = 0.1, patch_mode = full_breadth`` of
   ``papers/jour-ice-struct/data/panel_patch_load_results.json``.
S  the same mesh/material as the literature anchors: square SSSS plate under
   uniform load, compared with Timoshenko Table 8 coefficients.

Usage
-----
    python benchmarks/beta_restraint_panel_band.py [--nx 160] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorfem.plate_elastic_restraint import (  # noqa: E402
    peak_bending_stress,
    rect_patch_pressure,
    solve_restrained_panel,
)

# paper design point (papers/jour-ice-struct/data/panel_patch_load_results.json)
L, S, T = 0.8, 0.7, 0.022
E, NU = 206.0e9, 0.3
P_REF = 1.0e6
PATCH_C = 0.1 * L                      # patch length along the panel
BASELINE_SIGMA_MPA = 73.37101604723352  # Navier N=401, centre von Mises

BETA_DIAG = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BETA_X_ONLY = [0.0, 0.25, 0.5, 0.75, 1.0]


def paper_patch(nx: int, beta_x: float, beta_y: float) -> dict:
    ny = max(1, int(round(nx * S / L)))
    x1, x2 = 0.5 * (L - PATCH_C), 0.5 * (L + PATCH_C)
    press = rect_patch_pressure(P_REF, x1, x2, 0.0, S)
    res = solve_restrained_panel(
        nx, ny, L, S, E, NU, T, press, beta_x=beta_x, beta_y=beta_y,
    )
    pk = peak_bending_stress(res)
    return {
        "nx": nx, "ny": ny,
        "beta_x": beta_x, "beta_y": beta_y,
        "K_rot_x_Nm_per_rad": res.K_rot_x,
        "K_rot_y_Nm_per_rad": res.K_rot_y,
        "sigma_vm_centre_MPa": pk["centre_von_mises"] / 1e6,
        "sigma_x_centre_MPa": pk["centre_sigma_x"] / 1e6,
        "sigma_vm_max_MPa": pk["max_von_mises"] / 1e6,
        "w_centre_mm": pk["w_centre"] * 1e3,
        "sigma_ratio_to_ssss": pk["centre_von_mises"] / 1e6 / BASELINE_SIGMA_MPA,
        "rel_err_vs_navier_pct": 100.0 * (pk["centre_von_mises"] / 1e6 / BASELINE_SIGMA_MPA - 1.0),
    }


def literature_square(nx: int) -> dict:
    """Square SSSS / CCCC plates vs Timoshenko Table 8 (uniform load)."""
    a = 1.0
    D = E * T**3 / (12.0 * (1.0 - NU**2))
    q = P_REF
    out = {"nx": nx, "uniform_square_SSSS": {}, "uniform_square_CCCC": {}}
    res = solve_restrained_panel(nx, nx, a, a, E, NU, T, q, beta_x=0.0)
    pk = peak_bending_stress(res)
    out["uniform_square_SSSS"] = {
        "w_coef": abs(pk["w_centre"]) * D / (q * a**4),
        "M_coef": abs(pk["centre_Mx"]) / (q * a**2),
        "ref_w_coef_Timoshenko": 0.00406,
        "ref_M_coef_Timoshenko": 0.0479,
    }
    res = solve_restrained_panel(nx, nx, a, a, E, NU, T, q, beta_x=1.0)
    pk = peak_bending_stress(res)
    M = abs(pk["centre_Mx"])
    # edge moment: max |M| over the x = 0 edge nodes
    from tensorfem.plate_elastic_restraint import moments_at_nodes
    Mn = moments_at_nodes(res)
    edge = torch.nonzero(res.nodes[:, 0].abs() < 1e-9, as_tuple=False).flatten()
    medge = float(Mn[edge, 0].abs().max())
    out["uniform_square_CCCC"] = {
        "w_coef": abs(pk["w_centre"]) * D / (q * a**4),
        "M_coef_centre": M / (q * a**2),
        "M_coef_edge": medge / (q * a**2),
        "ref_w_coef_Timoshenko": 0.00126,
        "ref_M_coef_centre_Timoshenko": 0.0231,
        "ref_M_coef_edge_Timoshenko": 0.0513,
    }
    return out


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=160)
    ap.add_argument("--out", default=str(Path(__file__).parent / "artifacts"
                                         / "beta_restraint_panel_band.json"))
    args = ap.parse_args()

    diag = [paper_patch(args.nx, b, b) for b in BETA_DIAG]
    per_edge = [paper_patch(args.nx, b, 0.0) for b in BETA_X_ONLY]
    lit = literature_square(80)

    ssss = diag[0]["sigma_vm_centre_MPa"]
    cccc = diag[-1]["sigma_vm_centre_MPa"]
    payload = {
        "generated_by": "benchmarks/beta_restraint_panel_band.py",
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "case": {
            "description": "ice-belt panel field, local ice patch",
            "L_m": L, "S_m": S, "t_m": T, "E_Pa": E, "nu": NU,
            "patch_length_m": PATCH_C, "patch_breadth_m": S,
            "patch_area_m2": PATCH_C * S, "p_MPa": P_REF / 1e6,
            "D_Nm": E * T**3 / (12 * (1 - NU**2)),
        },
        "baseline_navier_N401_sigma_vm_centre_MPa": BASELINE_SIGMA_MPA,
        "mesh": {"nx": args.nx, "ny": max(1, int(round(args.nx * S / L)))},
        "diagonal_beta": diag,
        "beta_x_only_beta_y_0": per_edge,
        "band": {
            "sigma_vm_SSSS_MPa": ssss,
            "sigma_vm_clamped_MPa": cccc,
            "reduction_vs_SSSS_pct": 100.0 * (1.0 - cccc / ssss),
        },
        "literature_square": lit,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))

    print(f"case: {L}x{S}x{T} m, patch {PATCH_C*100:.0f}x{S*100:.0f} cm, "
          f"{P_REF/1e6:.1f} MPa, mesh {args.nx}x{payload['mesh']['ny']}")
    print(f"{'beta':>6} {'sigma_vm ctr [MPa]':>20} {'vs SSSS':>9} {'vs Navier':>10}")
    for row in diag:
        print(f"{row['beta_x']:6.2f} {row['sigma_vm_centre_MPa']:20.4f} "
              f"{row['sigma_ratio_to_ssss']:9.4f} {row['rel_err_vs_navier_pct']:+9.3f}%")
    print(f"band: SSSS {ssss:.3f} MPa -> clamped {cccc:.3f} MPa "
          f"(-{payload['band']['reduction_vs_SSSS_pct']:.1f}%)")
    print("literature square (nx=80):", json.dumps(lit, indent=1))
    print("written", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())