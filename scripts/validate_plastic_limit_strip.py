"""Validate the plastic *limit load* of the layered plate element on an exact case.

The anchor evidence (``plate_patch_plastic_limit``) compares the FE collapse load
of a *patch-loaded square panel* with rigid-plastic **bounds**.  Bounds are not an
exact reference: the admissible bracket for that case is wide
(``strip_lb = 0.76 MPa`` to ``sine_ub = 1.40 MPa``, ratio 1.85), so agreement
within 10 % of the upper bound does not by itself show that the FE collapse load
is right.  This script supplies the missing exact anchor:

a **narrow uniformly loaded strip** (``b/a = 0.2``) whose collapse is governed by
the one-way (beam) mechanism across the **short** direction ``b``.  The cylindrical
field ``w = w(y)`` is kinematically admissible for the simply supported plate and
dissipates ``4 m_p w0 / b`` per unit width at the mid-span hinge, so

    p_exact = 8 m_p / b^2        (beam mechanism, no empirical constant)

is an *exact* rigid-plastic limit for the strip and the FE result must approach it
**from above** (a displacement-based element is stiffer than the continuum, so its
limit load is an upper estimate that decreases under mesh refinement).  The path is
run with the updated geometric stiffness and in-plane restrained edges
(``geometric=True, inplane_restrained=True``): that is the boundary condition the
beam mechanism assumes, and it is what makes the load-deflection curve turn over
so that a limit point exists at all.

Run one span mesh per process::

    python scripts/validate_plastic_limit_strip.py --nx 16 --ny 2 --out out/strip16.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfem.plate_patch_plastic_limit import run_patch_plastic_path  # noqa: E402

A = 1.0          # span (long direction)
B = 0.2          # width (narrow direction) -> one-way mechanism
T = 0.02
E = 210.0e9
NU = 0.3
SY = 355.0e6


def exact_strip_limit(b: float = B, t: float = T, sy: float = SY) -> float:
    """``p = 8 m_p / b^2`` -- simply supported beam mechanism, m_p = sy t^2/4.

    The strip spans across its **short** direction ``b`` (a/b = 5, one-way
    action): the mid-span moment per unit width is ``p b^2 / 8`` and the
    plastic limit is reached when it equals the full plastic moment
    ``m_p = sy t^2 / 4``.  (An earlier revision of this script used the long
    direction ``a`` as the span, which is off by ``(a/b)^2 = 25`` and left
    the ramp far below the true limit -- the plate was still elastic at the
    end of the ramp, so no limit load was measured at all.)
    """
    return 8.0 * (sy * t * t / 4.0) / (b * b)


def _interp_crossing(steps: list[dict], target: float) -> float | None:
    prev = None
    for s in steps:
        if prev is None and s["sigma"] >= target:
            return s["factor"]
        if prev is not None and prev["sigma"] < target <= s["sigma"]:
            frac = (target - prev["sigma"]) / (s["sigma"] - prev["sigma"])
            return prev["factor"] + frac * (s["factor"] - prev["factor"])
        prev = s
    return None


def run(nx: int, ny: int, *, hardening: float = 1.0e6, layers: int = 3,
        fractions=(0.3, 0.6, 0.8, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5)) -> dict:
    pex = exact_strip_limit()
    patch = (A, B)                     # uniform load over the whole strip
    probe = run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=SY,
                                   hardening=hardening, patch=patch,
                                   load_factors=[0.05 * pex], nx=nx, ny=ny,
                                   layers=layers,
                                   geometric=True, inplane_restrained=True)
    p0 = probe.loading[0]
    path = run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=SY,
                                  hardening=hardening, patch=patch,
                                  load_factors=[f * pex for f in fractions],
                                  nx=nx, ny=ny, layers=layers,
                                  geometric=True, inplane_restrained=True)
    loading = [{"factor": s.load_factor, "w": s.centre_deflection,
                "w_over_t": s.centre_deflection / T,
                "sigma": s.max_surface_mises, "alpha": s.max_alpha}
               for s in path.loading]
    k0 = None
    for i in range(1, len(loading)):
        dp = loading[i]["factor"] - loading[i - 1]["factor"]
        dw = loading[i]["w"] - loading[i - 1]["w"]
        if dp > 0 and dw > 0:
            k0 = dp / dw
            break
    tangent = []
    for i in range(1, len(loading)):
        dp = loading[i]["factor"] - loading[i - 1]["factor"]
        dw = loading[i]["w"] - loading[i - 1]["w"]
        tangent.append({"factor": loading[i]["factor"],
                        "k": (dp / dw) / k0 if (k0 and dw > 0) else 0.0})
    last = loading[-1]["factor"] if loading else None
    return {
        "case": {"a": A, "b": B, "t": T, "aspect": A / B, "nx": nx, "ny": ny,
                 "layers": layers, "hardening": hardening},
        "exact_limit": pex,
        "elastic": {"w_per_pa": p0.centre_deflection / p0.load_factor,
                    "sigma_per_pa": p0.max_surface_mises / p0.load_factor},
        "loading": loading, "tangent": tangent,
        "ramp_max": fractions[-1] * pex,
        "collapse_last_converged": last,
        "ramp_completed": bool(last is not None and last >= fractions[-1] * pex - 1e-9),
        "limit_point_searched": bool(path.limit_point_searched),
        "measured_limit_load": path.measured_limit_load,
        "collapse_knee": next((t["factor"] for t in tangent if t["k"] < 0.05), None),
        "first_yield_fe": _interp_crossing(loading, SY),
        "error_last_converged_vs_exact_pct": (None if last is None
                                              else 100.0 * (last / pex - 1.0)),
        "error_limit_vs_exact_pct": (None if path.measured_limit_load is None
                                      else 100.0 * (path.measured_limit_load / pex - 1.0)),
        "w_over_t_at_last": loading[-1]["w_over_t"] if loading else None,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=16)
    ap.add_argument("--ny", type=int, default=2)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--hardening", type=float, default=1.0e6)
    ap.add_argument("--fractions", type=str, default="0.3,0.6,0.8,0.9,1.0,1.05,1.1,1.15,1.2,1.3,1.4,1.5")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    fractions = tuple(float(x) for x in args.fractions.split(","))
    out = run(args.nx, args.ny, hardening=args.hardening, layers=args.layers,
              fractions=fractions)
    text = json.dumps(out, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        open(args.out, "w", encoding="utf-8").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())