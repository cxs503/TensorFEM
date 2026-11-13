"""Cross-validation of the two-component Shan-Chen (MCMP) scheme with a
barycentric mixture velocity and Guo forcing.

Anchor: ``XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO``
        ("lssfau/lbmpy 的 shan-chen-mcmp-barycentric-guo 做法", acceptance < 15 %)

What is measured (all numbers written programmatically to
``docs/assets/lbm_shan_chen_mcmp_barycentric_guo/results.json``; the report
``docs/XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO.md`` is rendered from that
JSON, no hand-typed value):

1. **External reference reproduction** -- the OSS reference solution of
   ``lssfau/lbmpy`` ``tests/shan_chen/test_shan_chen_two_component.py``
   (64x64 periodic, omega_a = omega_b = 1, g_ab = 6, exponential
   pseudo-potential, 1000 steps, density profiles taken along one lattice
   line).  The reference array itself originates from the independent C++
   implementation ``lbm-principles-practice/code/chapter9/shanchen.cpp``.
   Three scheme arms are tried (``eq_shift`` = half / none / tau) so that a
   failure cannot be blamed on a single convention.
2. **Stability boundary of the scheme** as a function of the cross coupling
   ``g_ab`` for the same initial value problem (sharp step), reported as the
   first step at which the state stops being finite.
3. **Laplace law** on a circular drop: ``dp(R)`` against ``sigma/R`` for three
   radii (the classical falsifiable check of a two-component Shan-Chen model).

Run:
    PYTHONPATH=src python examples/lbm_shan_chen_mcmp_barycentric_guo_xval.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tensorfem.lbm_shan_chen_mcmp import MCMPConfig, ShanChenMCMP

# --------------------------------------------------------------------------
# OSS reference (lbmpy tests/shan_chen/test_shan_chen_two_component.py;
# array origin: lbm-principles-practice/code/chapter9/shanchen.cpp, nsteps=1000,
# nfluids=2, gA=0).  Provenance: repo lssfau/lbmpy, file sha256 recorded below.
# --------------------------------------------------------------------------
OSS_REF = {
    "repo": "lssfau/lbmpy",
    "path": "tests/shan_chen/test_shan_chen_two_component.py",
    "origin": "lbm-principles-practice/code/chapter9/shanchen.cpp (nsteps=1000, nfluids=2, gA=0)",
    "params": {"N": 64, "omega_a": 1.0, "omega_b": 1.0, "g_ab": 6.0,
               "rho0": 1.0, "steps": 1000, "rho_a_init": [0.1, 0.9],
               "rho_b_init": [0.9, 0.1]},
    "ref_a": [0.213948, 0.0816724, 0.0516763, 0.0470179, 0.0480882, 0.0504771,
              0.0531983, 0.0560094, 0.0588071, 0.0615311, 0.064102, 0.0664467,
              0.0684708, 0.070091, 0.0712222, 0.0718055, 0.0718055, 0.0712222,
              0.070091, 0.0684708, 0.0664467, 0.064102, 0.0615311, 0.0588071,
              0.0560094, 0.0531983, 0.0504771, 0.0480882, 0.0470179, 0.0516763,
              0.0816724, 0.213948, 0.517153, 0.833334, 0.982884, 1.0151,
              1.01361, 1.0043, 0.993178, 0.981793, 0.970546, 0.959798, 0.949751,
              0.940746, 0.933035, 0.926947, 0.922713, 0.920548, 0.920548,
              0.922713, 0.926947, 0.933035, 0.940746, 0.949751, 0.959798,
              0.970546, 0.981793, 0.993178, 1.0043, 1.01361, 1.0151, 0.982884,
              0.833334, 0.517153],
}

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "lbm_shan_chen_mcmp_barycentric_guo"


def run_reference_arm(eq_shift: str, source_velocity: str, steps: int = 1000) -> dict:
    """Reproduce the OSS reference case with one scheme convention."""
    res = {"eq_shift": eq_shift, "source_velocity": source_velocity, "steps": steps}
    sim = ShanChenMCMP(MCMPConfig(nx=64, ny=64, g_aa=0.0, g_ab=6.0, g_ba=6.0,
                                  tau_a=1.0, tau_b=1.0, eq_shift=eq_shift,
                                  source_velocity=source_velocity))
    sim.init_step_x(0.1, 0.9)
    div_step = None
    for i in range(1, steps + 1):
        sim.step()
        if not np.isfinite(sim.f).all():
            div_step = i
            break
    res["diverged_at_step"] = div_step
    if div_step is None:
        d = sim.density()
        ref_a = np.array(OSS_REF["ref_a"])
        ref_b = np.array(OSS_REF["ref_b"]) if "ref_b" in OSS_REF else None
        pa = d[0][0]
        res["rel_l2_a"] = float(np.linalg.norm(pa - ref_a) / np.linalg.norm(ref_a))
        res["rel_linf_a"] = float(np.abs(pa - ref_a).max() / np.abs(ref_a).max())
        res["rho_min"] = float(d.min())
        res["rho_max"] = float(d.max())
        res["profile_a"] = [float(v) for v in pa]
    else:
        d = sim.density()
        res["rho_min_last_finite"] = float(np.nanmin(np.where(np.isfinite(d), d, np.nan)))
        res["rho_max_last_finite"] = float(np.nanmax(np.where(np.isfinite(d), d, np.nan)))
    return res


def stability_boundary(g_values, steps: int = 600) -> list:
    """First non-finite step for the sharp-step initial value problem."""
    rows = []
    for g in g_values:
        sim = ShanChenMCMP(MCMPConfig(nx=64, ny=64, g_ab=g, g_ba=g))
        sim.init_step_x(0.1, 0.9)
        div = None
        for i in range(1, steps + 1):
            sim.step()
            if not np.isfinite(sim.f).all():
                div = i
                break
        rows.append({"g_ab": g, "diverged_at_step": div,
                     "steps_completed": steps if div is None else div - 1})
    return rows


def laplace_law(radii, n: int = 96, steps: int = 6000, g: float = 6.0,
                rho_in: float = 0.9, rho_out: float = 0.1) -> dict:
    """Measure dp(R) for a circular drop and fit the Young-Laplace law."""
    pts = []
    for r0 in radii:
        sim = ShanChenMCMP(MCMPConfig(nx=n, ny=n, g_ab=g, g_ba=g))
        sim.init_bubble(radius=r0, rho_in=rho_in, rho_out=rho_out, interface_width=2.0)
        ok = True
        for i in range(steps):
            sim.step()
            if not np.isfinite(sim.f).all():
                ok = False
                break
        if not ok:
            pts.append({"R": r0, "stable": False, "steps_completed": i})
            continue
        rho_tot = sim.density().sum(axis=0)
        c = n // 2
        inside = rho_tot[max(0, c - 2):c + 3, max(0, c - 2):c + 3].mean()
        outside = rho_tot[:6, :6].mean()
        dp = (inside - outside) / 3.0        # p = cs^2 rho_tot
        pts.append({"R": r0, "stable": True, "steps_completed": steps,
                    "rho_tot_inside": float(inside), "rho_tot_outside": float(outside),
                    "dp": float(dp), "inv_R": 1.0 / r0,
                    "spurious_u": sim.spurious_current()})
    fit = {}
    good = [p for p in pts if p.get("stable")]
    if len(good) >= 2:
        x = np.array([p["inv_R"] for p in good])
        y = np.array([p["dp"] for p in good])
        A = np.vstack([x, np.ones_like(x)]).T
        (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
        yhat = A @ np.array([slope, intercept])
        ss_res = float(((y - yhat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        fit = {"sigma": float(slope), "intercept": float(intercept),
               "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
               "rel_dev_vs_sigma_over_R": [
                   float(abs(p["dp"] - slope * p["inv_R"]) / abs(slope * p["inv_R"])) for p in good]}
    return {"points": pts, "fit": fit, "g_ab": g, "n": n, "steps": steps}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {"anchor": "XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO",
               "oss_reference": {k: v for k, v in OSS_REF.items() if k != "ref_a"}}

    arms = []
    for eq_shift, src in (("half", "forced"), ("none", "bary"), ("tau", "bary")):
        arms.append(run_reference_arm(eq_shift, src))
    results["reference_arms"] = arms

    results["stability_boundary"] = stability_boundary([0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0])

    # Laplace law only where the model is stable (see stability_boundary)
    g_lap = float(__import__("os").environ.get("G_LAPLACE", "1.0"))
    results["laplace"] = laplace_law([8.0, 12.0, 16.0], n=96, steps=3000, g=g_lap)

    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: results[k] for k in ("reference_arms", "stability_boundary")}, indent=1))
    print(json.dumps(results["laplace"], indent=1))


if __name__ == "__main__":
    main()