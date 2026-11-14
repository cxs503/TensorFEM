"""Cross-validation v5: coupled added-mass frequency of the implicit IBM + Aitken solver.

Anchor ``XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION`` (reference side:
joconnor22/LIFE -- implicit direct forcing with Aitken Delta^2 sub-iterations).

Why v5 exists
-------------
``docs/xval_implicit_ibm_aitken_v4.md`` closed with a **failing** check:
``f_fsi / f_vac = 1.5811`` against the analytic ``0.7906`` (error 100 %), and
attributed it to "the plate's initial condition ``y0 = 0.2`` being interpreted
as lattice units rather than physical units".  That diagnosis is wrong, and the
check itself was never executable:

* the v4 plate run used ``n_steps = 300``, ``dt = 1`` while the analytic
  reference period of the very same configuration is ``T = 600`` time units, so
  the record is **half a period long**;
* the zero-crossing branch needs two up-crossings (it has none) and the
  Hann-windowed FFT branch bottoms out at its fundamental bin, so the estimate
  saturates at ``1 / (n_steps * dt) = 1/300`` -- exactly the number in the v4
  evidence file (``f_measured = 0.003333``);
* the reported ratio ``1.5811 = sqrt(2.5)`` is therefore the window length, not
  a physical frequency, and the "2x" coincidence is an artefact of ``T`` having
  been chosen such that ``T = 2 * n_steps * dt``.

The defect is in the *validation harness* (window shorter than one period,
no validity guard), not in the solver: with a 4-period record the same case
gives ``T = 674.5`` vs ``600`` (11 % on frequency), inside the 15 % acceptance.

What v5 measures
----------------
For a thin plate of half-width ``c`` in 2-D potential flow the added mass for
broadside translation is exactly ``m_a = rho_f pi c^2`` (Lamb 1932, art. 71), so
the spring-mounted plate obeys ``f = sqrt(k/(m_s+m_a))/2pi``.  Two systematic
effects make a *single* lattice run differ from that idealisation, and both are
measurable rather than fudge factors:

* **kernel smearing** -- the 3-point delta has support ``1.5 dx`` beyond the end
  markers, so the force field acts on a plate of apparent half-width
  ``c + delta`` with ``delta`` ~ the kernel half-width;
* **finite domain** -- the box is periodic in both directions, so the plate sees
  its images; the blockage ratio ``B = (2c + 1)/H`` biases ``m_a`` upward.

v5 therefore sweeps ``c`` at fixed box size (``delta`` and ``B`` both change) and
the box size at fixed ``c``, recovers ``m_a_eff`` from the fitted period, and
reports (i) the raw error against Lamb, (ii) the error after the *a priori*
smearing correction ``m_a -> rho_f pi (c + 1.5 dx)^2``, and (iii) a two-parameter
model-form fit (``delta``, blockage slope) as a consistency check -- the fitted
``delta`` must be of order the kernel half-width, not a free knob.  A sweep over
the added-mass ratio and over the relaxation mode adds the coupling-side check:
Aitken vs fixed ``omega = 1`` sub-iteration counts at ``m_a/m_s = 1.0``, the
partitioned-scheme instability band that v4 never reached (its ratios were <= 0.6).

Usage
-----
``python benchmarks/xval_plate_added_mass_v5.py --config K``  (one shard)
``python benchmarks/xval_plate_added_mass_v5.py --aggregate`` (table + figures)
``python benchmarks/xval_plate_added_mass_v5.py --quick``     (local smoke)
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfem.ibm_aitken_added_mass import (  # noqa: E402
    IBMImplicitFSI,
    LBMFluid2D,
    SpringMountedBody,
    analytic_added_mass,
    estimate_period,
    estimate_period_damped,
    estimate_period_fft,
    oscillator_frequency,
    period_window_ok,
    plate_markers,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "xval_plate_added_mass_v5_artifacts")
TOL = 15.0          # owner acceptance: cross-validation error < 15 %
T_TARGET = 600.0    # analytic period of every configuration (time units)
N_CYCLES = 4.0      # record length in periods (v4 used 0.5 -- the defect)

#: ``(name, half_width, box, added_mass_ratio, relax_mode)``
CONFIGS = [
    ("c07_H288", 7.0, 288, 0.6, "aitken"),
    ("c10_H288", 10.0, 288, 0.6, "aitken"),
    ("c14_H288", 14.0, 288, 0.6, "aitken"),
    ("c20_H288", 20.0, 288, 0.6, "aitken"),
    ("c10_H144", 10.0, 144, 0.6, "aitken"),
    ("c10_H288_ratio1p0", 10.0, 288, 1.0, "aitken"),
    ("c10_H288_ratio1p0_fixed", 10.0, 288, 1.0, "fixed"),
]
DELTA_APRIORI = 1.5   # 3-point kernel half-width, in lattice units


def run_config(name, half_width, box, ratio, relax_mode, dt=1.0, n_steps=None,
               omega_lbm=1.9, sub_tol=1e-6, max_sub_iter=30, rho_f=1.0):
    """One plate free-vibration run; returns the record and its fitted metrics."""
    if n_steps is None:
        n_steps = int(round(N_CYCLES * T_TARGET / dt))
    c = float(half_width)
    m_a = analytic_added_mass(c, rho_f)             # rho_f * pi * c^2
    m_s = m_a / float(ratio)
    k = (m_s + m_a) * (2.0 * math.pi / T_TARGET) ** 2

    center = (box / 2.0, box / 2.0)
    body = SpringMountedBody(plate_markers(center, c, dx=1.0), mass=m_s,
                             stiffness=k, y0=0.2)
    fluid = LBMFluid2D((box, box), omega=omega_lbm, rho0=rho_f)
    fsi = IBMImplicitFSI(body, fluid, dt=dt, dx=1.0, ref_length=2.0 * c,
                         relax_mode=relax_mode, fixed_omega=1.0,
                         sub_tol=sub_tol, max_sub_iter=max_sub_iter)

    t0 = time.time()
    ys, subs, resids = [], [], []
    diverged = False
    for _ in range(int(n_steps)):
        try:
            rec = fsi.step()
        except (ValueError, FloatingPointError) as exc:
            diverged = str(exc)
            break
        ys.append(rec["y"])
        subs.append(rec["sub_iterations"])
        resids.append(rec["residual"])
        if not rec["finite"]:
            diverged = "interface state left the lattice"
            break
    wall = time.time() - t0

    fit = estimate_period_damped(ys, dt)
    p_old = estimate_period(ys, dt)          # zero-crossing / FFT legacy path
    p_fft = estimate_period_fft(ys, dt)
    T = fit["period"]
    f_meas = 1.0 / T if T == T and T > 0.0 else float("nan")
    f_vac = oscillator_frequency(k, m_s)
    f_lamb = oscillator_frequency(k, m_s + m_a)
    m_a_eff = k / (2.0 * math.pi * f_meas) ** 2 - m_s if f_meas == f_meas else float("nan")
    c_eff = math.sqrt(max(m_a_eff, 0.0) / (rho_f * math.pi)) if m_a_eff == m_a_eff else float("nan")
    m_a_smear = analytic_added_mass(c + DELTA_APRIORI, rho_f)

    def _pct(got, ref):
        return float("nan") if ref == 0 else 100.0 * (got - ref) / abs(ref)

    return {
        "name": name, "half_width": c, "box": box, "added_mass_ratio": float(ratio),
        "relax_mode": relax_mode, "dt": dt, "n_steps": int(n_steps),
        "steps_completed": len(ys), "wall_seconds": wall, "diverged": diverged or False,
        "lattice": [box, box], "omega_lbm": omega_lbm, "sub_tol": sub_tol,
        "m_a_analytic": m_a, "m_a_smeared": m_a_smear, "m_s": m_s, "k": k,
        "blockage": (2.0 * c + 1.0) / box,
        "period": T, "period_tau": fit["tau"], "period_fit_rms": fit["rms"],
        "period_fit_quality": fit["rms_over_amplitude"],
        "period_valid": fit["valid"], "window_cycles": fit["cycles"],
        "period_zero_cross": p_old, "period_fft": p_fft,
        "window_ok_2cyc": period_window_ok(len(ys), dt, T, 2.0),
        "frequency": f_meas, "frequency_vac": f_vac, "frequency_lamb": f_lamb,
        "frequency_ratio": f_meas / f_vac if f_vac else float("nan"),
        "frequency_ratio_lamb": f_lamb / f_vac,
        "frequency_error_pct": _pct(f_meas, f_lamb),
        "added_mass_measured": m_a_eff, "added_mass_error_pct": _pct(m_a_eff, m_a),
        "added_mass_error_smeared_pct": _pct(m_a_eff, m_a_smear),
        "effective_half_width": c_eff, "delta_fitted": c_eff - c,
        "damping_per_cycle": 1.0 - math.exp(-T / fit["tau"])
        if fit["tau"] == fit["tau"] and fit["tau"] > 0 else float("nan"),
        "mean_sub_iterations": float(np.mean(subs)) if subs else float("nan"),
        "max_sub_iterations": int(max(subs)) if subs else 0,
        "maxit_hits": int(fsi.n_maxit_hits),
        "converged_all": bool(all(np.array(subs) < max_sub_iter)) if subs else False,
        "y_history": [float(v) for v in ys],
        "sub_iter_history": [int(v) for v in subs],
        "residual_history": [float(v) for v in resids],
    }


# ---------------------------------------------------------------------------
def _fit_model_form(recs):
    """Least squares ``m_a_eff = rho pi (c + delta)^2 (1 + beta B)`` over the c-sweep."""
    sweep = [r for r in recs if r["added_mass_ratio"] == 0.6
             and r["added_mass_measured"] == r["added_mass_measured"]]
    if len(sweep) < 3:
        return None
    c = np.array([r["half_width"] for r in sweep])
    H = np.array([r["box"] for r in sweep], dtype=float)
    B = (2.0 * c + 1.0) / H
    ma = np.array([r["added_mass_measured"] for r in sweep])
    best = None
    for delta in np.linspace(0.0, 4.0, 41):
        for beta in np.linspace(0.0, 3.0, 61):
            pred = np.pi * (c + delta) ** 2 * (1.0 + beta * B)
            rel = np.abs(pred - ma) / ma
            score = float(np.max(rel))
            if best is None or score < best[0]:
                best = (score, float(delta), float(beta))
    _, delta, beta = best
    pred = np.pi * (c + delta) ** 2 * (1.0 + beta * B)
    return {"delta": delta, "beta": beta,
            "max_rel_error": float(np.max(np.abs(pred - ma) / ma)),
            "rms_rel_error": float(np.sqrt(np.mean(((pred - ma) / ma) ** 2))),
            "n_points": int(len(sweep))}


def aggregate(recs):
    """Verdict table over all configs."""
    rows = []
    for r in recs:
        err = r["frequency_error_pct"]
        rows.append({
            "config": r["name"], "c": r["half_width"], "H": r["box"],
            "blockage": r["blockage"], "m_a/m_s": r["added_mass_ratio"],
            "relax": r["relax_mode"], "period": r["period"],
            "window_cycles": r["window_cycles"],
            "f_ratio": r["frequency_ratio"],
            "f_ratio_lamb": r["frequency_ratio_lamb"],
            "freq_err_pct": err,
            "m_a_err_pct": r["added_mass_error_pct"],
            "m_a_err_smeared_pct": r["added_mass_error_smeared_pct"],
            "delta": r["delta_fitted"],
            "valid": r["period_valid"] and np.isfinite(err),
            "pass": bool(r["period_valid"] and np.isfinite(err) and abs(err) <= TOL),
            "mean_sub_iter": r["mean_sub_iterations"],
            "maxit_hits": r["maxit_hits"],
        })
    n_pass = sum(1 for x in rows if x["pass"])
    return {"rows": rows, "n_configs": len(rows), "n_pass": n_pass,
            "tolerance_pct": TOL,
            "all_pass": n_pass == len(rows),
            "model_form": _fit_model_form(recs)}


# ---------------------------------------------------------------------------
def _json_default(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and obj != obj:
        return None
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    os.makedirs(OUT, exist_ok=True)
    quick = "--quick" in argv
    if "--config" in argv:
        k = int(argv[argv.index("--config") + 1])
        name, c, box, ratio, relax = CONFIGS[k]
        rec = run_config(name, c, box, ratio, relax,
                         n_steps=600 if quick else None)
        path = os.path.join(OUT, f"config_{k:02d}_{name}.json")
        with open(path, "w") as fh:
            json.dump(rec, fh, indent=1, default=_json_default)
        print(json.dumps({kk: vv for kk, vv in rec.items()
                          if not kk.endswith("_history")}, indent=1,
                         default=_json_default))
        return rec
    if "--aggregate" in argv:
        recs = []
        for k in range(len(CONFIGS)):
            name = CONFIGS[k][0]
            path = os.path.join(OUT, f"config_{k:02d}_{name}.json")
            if os.path.exists(path):
                recs.append(json.load(open(path)))
        ev = {"anchor": "XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION",
              "reference": "Lamb 1932 art. 71: m_a = rho_f pi c^2 (thin plate, 2-D)",
              "tolerance_pct": TOL, "n_cycles_min": N_CYCLES,
              "delta_apriori": DELTA_APRIORI,
              "configs": ev_rows(recs), "verdict": aggregate(recs)}
        with open(os.path.join(OUT, "evidence.json"), "w") as fh:
            json.dump(ev, fh, indent=1, default=_json_default)
        print(json.dumps(ev["verdict"], indent=1, default=_json_default))
        return ev
    if quick:
        rec = run_config("quick", 10.0, 96, 0.6, "aitken", n_steps=600)
        print(json.dumps({kk: vv for kk, vv in rec.items()
                          if not kk.endswith("_history")}, indent=1,
                         default=_json_default))
        return rec
    print(__doc__)
    return None


def ev_rows(recs):
    keep = ("name", "half_width", "box", "blockage", "added_mass_ratio",
            "relax_mode", "period", "window_cycles", "period_valid",
            "period_zero_cross", "period_fft", "frequency", "frequency_vac",
            "frequency_lamb", "frequency_ratio", "frequency_ratio_lamb",
            "frequency_error_pct", "m_a_analytic", "m_a_smeared",
            "added_mass_measured", "added_mass_error_pct",
            "added_mass_error_smeared_pct", "delta_fitted",
            "damping_per_cycle", "mean_sub_iterations", "maxit_hits",
            "converged_all", "steps_completed", "wall_seconds", "diverged")
    return [{k: r.get(k) for k in keep} for r in recs]


if __name__ == "__main__":
    main()