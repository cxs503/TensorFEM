"""XVAL — implicit IBM + Aitken relaxation: added-mass cross-validation.

Anchor ``XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION`` (reference side: joconnor22/LIFE).
Reference values are analytic, not copied:

* ``m_a = rho_f pi c^2``          thin plate in 2-D potential flow (Lamb 1932, art. 71);
* ``f = sqrt(k/(m_s + m_a))/2pi`` spring-mounted rigid body;
* ``f / f_vacuum = sqrt(m_s/(m_s+m_a))``.

Experiments
-----------
E0 control     structure alone (implicit Newmark) vs analytic frequency.
E1 operator    prescribed oscillation -> least-squares added mass ``m_a``.
E2 coupled     free vibration of the spring-mounted plate driven by the
               implicit IBM/Aitken solver -> frequency vs analytic;
               sweep over the added-mass ratio ``m_a/m_s`` and over the
               relaxation mode (``aitken`` vs fixed ``omega = 1``).

Outputs (``benchmarks/xval_added_mass_artifacts/``): ``evidence.json``,
``xval_added_mass_history.png``, ``xval_added_mass_frequency.png``,
``xval_added_mass_force.png``, ``xval_added_mass_summary.md``.

Run: ``python benchmarks/xval_aitken_added_mass.py [--quick]``
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensorfem.ibm_aitken_added_mass import (  # noqa: E402
    analytic_added_mass,
    coupled_free_vibration,
    forced_oscillation,
    oscillator_frequency,
    structural_probe,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "xval_added_mass_artifacts")
TOL = 15.0          # owner acceptance: cross-validation error < 15 %


def _cfg(quick):
    if quick:
        return dict(half_width=10.0, box=96, period=300.0, n_steps=600,
                    drive_period=300.0, ratios=(0.60, 1.00))
    return dict(half_width=20.0, box=160, period=600.0, n_steps=1500,
                drive_period=600.0, ratios=(0.15, 0.35, 0.60, 1.00, 1.57))


def _case(ratio, cfg, mode, n_steps=None):
    m_a = analytic_added_mass(cfg["half_width"])
    m_s = m_a / ratio
    k = (m_s + m_a) * (2.0 * math.pi / cfg["period"]) ** 2
    rec = coupled_free_vibration(
        n_steps=n_steps or cfg["n_steps"], half_width=cfg["half_width"],
        center=(cfg["box"] / 2.0, cfg["box"] / 2.0),
        lattice=(cfg["box"], cfg["box"]), mass=m_s, stiffness=k, y0=0.2,
        omega_lbm=1.9, relax_mode=mode, fixed_omega=1.0, sub_tol=1e-6,
        max_sub_iter=25)
    keep = ("relax_mode", "added_mass_ratio", "mass", "stiffness", "period",
            "period_fft", "frequency", "frequency_ref", "frequency_ratio",
            "frequency_ratio_ref", "frequency_error_pct",
            "frequency_error_fft_pct", "added_mass_measured",
            "added_mass_measured_error_pct", "steps_completed", "diverged",
            "diverged_reason", "mean_sub_iterations", "max_sub_iterations",
            "maxit_hits", "converged_all")
    out = {k2: rec[k2] for k2 in keep}
    out["frequency_from"] = "zero_crossing"
    if not np.isfinite(out["frequency"]) and np.isfinite(out["period_fft"]):
        out["frequency"] = 1.0 / out["period_fft"]
        out["frequency_error_pct"] = 100.0 * (
            out["frequency"] - out["frequency_ref"]) / out["frequency_ref"]
        out["frequency_from"] = "fft"
    out["y_history"] = rec["y_history"]
    out["iter_history"] = rec["iter_history"]
    return out


def run(quick=False):
    os.makedirs(OUT, exist_ok=True)
    cfg = _cfg(quick)
    t0 = time.time()
    ev = {"anchor": "XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION",
          "reference": "joconnor22/LIFE (implicit IBM + Aitken relaxation)",
          "config": {k: (list(v) if isinstance(v, tuple) else v)
                     for k, v in cfg.items()},
          "tolerance_pct": TOL}
    print("[E0] structural control", flush=True)
    ev["E0_structural"] = structural_probe(mass=_case_mass(cfg, 0.6),
                                           stiffness=_case_k(cfg, 0.6),
                                           n_steps=3000)
    print("  f=%.6f ref=%.6f err=%+.3f%%" % (
        ev["E0_structural"]["frequency"], ev["E0_structural"]["frequency_ref"],
        ev["E0_structural"]["error_pct"]), flush=True)

    print("[E1] forced oscillation", flush=True)
    ev["E1_forced"] = forced_oscillation(
        n_steps=cfg["n_steps"], half_width=cfg["half_width"],
        center=(cfg["box"] / 2.0, cfg["box"] / 2.0),
        lattice=(cfg["box"], cfg["box"]), amplitude=0.2,
        drive_period=cfg["drive_period"], omega_lbm=1.9)
    print("  m_a fit=%.2f ref=%.2f err=%+.2f%% r2=%.4f" % (
        ev["E1_forced"]["added_mass_fit"], ev["E1_forced"]["added_mass_ref"],
        ev["E1_forced"]["added_mass_error_pct"],
        ev["E1_forced"]["fit_r2"]), flush=True)

    print("[E2] coupled sweep", flush=True)
    cases = []
    for ratio in cfg["ratios"]:
        for mode in ("aitken", "fixed"):
            rec = _case(ratio, cfg, mode)
            print("  ratio=%.2f mode=%-6s T=%.1f err=%+.2f%% it=%.2f maxit=%d "
                  "div=%s" % (ratio, mode, rec["period"],
                              rec["frequency_error_pct"],
                              rec["mean_sub_iterations"], rec["maxit_hits"],
                              rec["diverged"]), flush=True)
            cases.append(rec)
    ev["E2_coupled"] = cases
    ev["runtime_s"] = time.time() - t0

    # ---- verdict ---------------------------------------------------------
    freq_err = [abs(c["frequency_error_pct"]) for c in cases
                if np.isfinite(c["frequency_error_pct"]) and not c["diverged"]]
    ev["verdict"] = {
        "coupled_frequency_max_abs_error_pct": max(freq_err) if freq_err else None,
        "coupled_frequency_all_within_tolerance": bool(
            freq_err) and max(freq_err) <= TOL,
        "structural_control_within_0p1pct":
            abs(ev["E0_structural"]["error_pct"]) < 0.1,
        "added_mass_fit_within_tolerance":
            abs(ev["E1_forced"]["added_mass_error_pct"]) <= TOL,
        "cases_diverged": [{"ratio": c["added_mass_ratio"],
                            "mode": c["relax_mode"],
                            "reason": c["diverged_reason"]}
                           for c in cases if c["diverged"]],
    }
    with open(os.path.join(OUT, "evidence.json"), "w") as fh:
        json.dump(ev, fh, indent=2)
    _figures(ev)
    _summary(ev)
    return ev


def _case_mass(cfg, ratio):
    m_a = analytic_added_mass(cfg["half_width"])
    return m_a / ratio


def _case_k(cfg, ratio):
    m_a = analytic_added_mass(cfg["half_width"])
    m_s = m_a / ratio
    return (m_s + m_a) * (2.0 * math.pi / cfg["period"]) ** 2


# ---------------------------------------------------------------------------
# figures (SQL-free, straight from evidence.json in memory)
# ---------------------------------------------------------------------------
def _figures(ev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = ev["E2_coupled"]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for c in cases:
        if c["relax_mode"] != "aitken" or c["diverged"]:
            continue
        ax.plot(np.arange(len(c["y_history"])), c["y_history"],
                label="m_a/m_s=%.2f (T=%.0f)" % (c["added_mass_ratio"], c["period"]))
    ax.set_xlabel("LBM step")
    ax.set_ylabel("tip displacement y (lattice units)")
    ax.set_title("Coupled free vibration, Aitken-relaxed implicit IBM")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "xval_added_mass_history.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ratios = sorted({c["added_mass_ratio"] for c in cases})
    for mode, colour in (("aitken", "tab:blue"), ("fixed", "tab:orange")):
        xs, ys = [], []
        for r in ratios:
            sel = [c for c in cases if c["added_mass_ratio"] == r
                   and c["relax_mode"] == mode and not c["diverged"]]
            if sel:
                xs.append(r)
                ys.append(100.0 * (sel[0]["frequency_ratio"]
                                   / sel[0]["frequency_ratio_ref"] - 1.0))
        ax.plot(xs, ys, "o-", color=colour, label=mode)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(ev["tolerance_pct"], color="r", ls="--", lw=0.8, label="+15 %")
    ax.axhline(-ev["tolerance_pct"], color="r", ls="--", lw=0.8)
    ax.set_xlabel("added-mass ratio m_a/m_s")
    ax.set_ylabel("frequency error vs analytic (%)")
    ax.set_title("Coupled frequency error (analytic m_a = rho pi c^2)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "xval_added_mass_frequency.png"), dpi=130)
    plt.close(fig)

    f = ev["E1_forced"]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    t = np.arange(len(f["force_history"]))
    ax.plot(t, f["force_history"], lw=0.9, color="tab:green")
    ax.set_xlabel("LBM step")
    ax.set_ylabel("interface force F_y")
    ax.set_title("Prescribed oscillation: fitted m_a=%.1f vs analytic %.1f (r2=%.3f)"
                 % (f["added_mass_fit"], f["added_mass_ref"], f["fit_r2"]))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "xval_added_mass_force.png"), dpi=130)
    plt.close(fig)


def _summary(ev):
    c = ev["config"]
    lines = ["# XVAL — implicit IBM + Aitken: added-mass cross-validation",
             "",
             "Anchor `XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION`; reference analytic"
             " (Lamb 1932): `m_a = rho_f pi c^2`, `f = sqrt(k/(m_s+m_a))/2pi`.",
             "",
             "Configuration: plate half-width c=%.1f, lattice %dx%d, drive/analytic"
             " period %.0f steps, %d steps per case."
             % (c["half_width"], c["box"], c["box"], c["period"], c["n_steps"]),
             "",
             "| experiment | quantity | reference | got | error | verdict |",
             "|---|---|---|---|---|---|"]
    e0 = ev["E0_structural"]
    lines.append("| E0 control | f (structure only) | %.6f | %.6f | %+.3f%% | %s |"
                 % (e0["frequency_ref"], e0["frequency"], e0["error_pct"],
                    "PASS" if abs(e0["error_pct"]) < 0.1 else "FAIL"))
    e1 = ev["E1_forced"]
    lines.append("| E1 operator | m_a (force fit, r2=%.3f) | %.1f | %.1f | %+.1f%% | %s |"
                 % (e1["fit_r2"], e1["added_mass_ref"], e1["added_mass_fit"],
                    e1["added_mass_error_pct"],
                    "PASS" if abs(e1["added_mass_error_pct"]) <= TOL else "FAIL"))
    for case in ev["E2_coupled"]:
        if case["diverged"]:
            verdict = "DIVERGED"
        elif case["frequency_error_pct"] != case["frequency_error_pct"]:
            verdict = "n/a"
        else:
            verdict = "PASS" if abs(case["frequency_error_pct"]) <= TOL else "FAIL"
        lines.append("| E2 coupled | f, m_a/m_s=%.2f, %s | %.6f | %.6f | %+.1f%% | %s |"
                     % (case["added_mass_ratio"], case["relax_mode"],
                        case["frequency_ref"],
                        case["frequency"] if case["frequency"] == case["frequency"]
                        else float("nan"),
                        case["frequency_error_pct"], verdict))
    v = ev["verdict"]
    lines += ["",
              "Max |frequency error| over converged cases: %s %% (tolerance %.0f %%)."
              % (("%.2f" % v["coupled_frequency_max_abs_error_pct"])
                 if v["coupled_frequency_max_abs_error_pct"] is not None else "n/a",
                 TOL),
              "",
              "Runtime: %.0f s." % ev["runtime_s"]]
    with open(os.path.join(OUT, "xval_added_mass_summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ev = run(quick="--quick" in sys.argv[1:])
    print(json.dumps(ev["verdict"], indent=2))