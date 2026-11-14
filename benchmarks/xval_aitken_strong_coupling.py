"""XVAL round 4 — does the implicit IBM + Aitken coupling survive *strong* added mass?

Anchor ``XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION`` (reference side: joconnor22/LIFE,
implicit direct forcing with Aitken Delta^2 sub-iterations).

Motivation (round 3 left this open)
-----------------------------------
``docs/xval_implicit_ibm_aitken_v3.md`` section 4.3 reports an honest negative
result: over the swept band ``m_a/m_s in [0.15, 1.57]`` Aitken and fixed
``omega = 1`` give *bit-identical* displacement records, because the interface
fixed point already converges in ~1 sub-iteration.  The conclusion drawn there
("the ratios reachable in the stable band are too mild for the relaxation to
matter") is *not* an explanation: nothing in that report predicts **where** the
relaxation must start to matter, so the anchor's central mechanism -- Aitken
relaxation of the implicit IBM sub-iterations -- is still unvalidated on the
coupled problem.

This round derives the missing threshold and tests it.

Theory (this report)
--------------------
In this driver the interface force at the frozen pre-force fluid state is

    F(y) = -sum_i w_i * 2 rho_i (u_body,y(y) - m_i,y) / dt ,
    u_body,y(y) = gamma/(beta dt) (y - y_n) + const          (Newmark)

so with the Newmark solve ``y_new = (F + m beta^-1 dt^-2 y_pred)/(m beta^-1 dt^-2 + k)``
the interface map is linear with **constant gain**

    g = d y_new / d y = -4 S / (4 m_s + k beta dt^2),  S == sum_i rho_i w_i ,
    S = -(dF/dy) beta dt^2 / (2 gamma)                       (stencil mass)

i.e. ``g ~= -S / m_s`` for a soft spring.  Two consequences:

* the fixed-point iteration ``omega = 1`` diverges for ``|g| > 1``, i.e. for an
  added-mass ratio above the **critical** value
  ``ratio_crit = m_a / S``  (independent of ``k`` to leading order);
* the Aitken optimum for a scalar linear map is ``omega* = 1/(1-g)``
  (the recursion already verified in ``docs/xval_implicit_ibm_aitken_v2.md``
  section 3.2), so the converged relaxation factor is a *measurable prediction*
  of the measured gain.

Because ``S ~ O(c)`` while ``m_a = rho pi c^2``, ``ratio_crit ~ pi c / (S/c)``
grows with the plate half-width: the graded test below measures ``ratio_crit``
and then sweeps through it.

Experiments
-----------
S1 gain      ``dF/dy`` on a frozen fluid state -> ``S`` and ``ratio_crit`` for
             ``c`` in {10, 16, 20}; analytic cross-check of the linear map.
S2 sweep     coupled free vibration over ``m_a/m_s`` straddling ``ratio_crit``,
             modes ``aitken`` / fixed ``omega = 1`` / fixed ``omega`` = 0.5;
             per-case sub-iteration statistics, divergence, frequency error.
S3 relax     converged Aitken ``omega`` vs the prediction ``1/(1-g)``.

Run: ``python benchmarks/xval_aitken_strong_coupling.py [--quick]``
Outputs: ``benchmarks/xval_aitken_strong_artifacts/``.
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
    IBMImplicitFSI,
    LBMFluid2D,
    SpringMountedBody,
    analytic_added_mass,
    coupled_free_vibration,
    plate_markers,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "xval_aitken_strong_artifacts")
TOL = 15.0                      # owner acceptance on the frequency
BETA, GAMMA = 0.25, 0.5         # Newmark parameters used by the driver


# ---------------------------------------------------------------------------
# S1: interface gain / stencil mass on a frozen fluid state
# ---------------------------------------------------------------------------
def interface_gain(half_width=10.0, box=96, omega_lbm=1.9, eps=1.0e-6):
    """Return ``(dF/dy, S, ratio_crit)`` measured on a quiescent fluid state.

    ``dF/dy`` is measured by a central difference of the interface force with
    respect to the trial displacement ``y`` at *fixed* fluid state -- exactly the
    quantity that sets the interface-map gain ``g = dF/dy / (m beta^-1 dt^-2 + k)``.
    """
    body = SpringMountedBody(plate_markers((box / 2.0, box / 2.0), half_width),
                             mass=1.0, stiffness=0.0, y0=0.0)
    fluid = LBMFluid2D((box, box), omega=omega_lbm, rho0=1.0)
    solver = IBMImplicitFSI(body, fluid, sub_tol=0.0, max_sub_iter=2)
    rho, u = fluid.macroscopic(include_force=False)
    y0 = 0.0
    f_plus, _ = solver.interface_force(rho, u, y0 + eps)
    f_minus, _ = solver.interface_force(rho, u, y0 - eps)
    dFdy = (f_plus - f_minus) / (2.0 * eps)
    # S = -dF/dy * beta dt^2 / (2 gamma)  (dt = 1)
    stencil_mass = -dFdy * BETA / (2.0 * GAMMA)
    m_a = analytic_added_mass(half_width)
    return {"half_width": half_width, "lattice": box, "dF_dy": dFdy,
            "stencil_mass": stencil_mass,
            "stencil_mass_per_half_width": stencil_mass / half_width,
            "added_mass": m_a,
            "ratio_crit": m_a / stencil_mass,
            "markers": int(len(body.positions0))}


def map_gain(stencil_mass, mass, stiffness, dt=1.0):
    """Predicted interface-map gain ``g = dF/dy / (m beta^-1 dt^-2 + k)``.

    ``dF/dy = -4 S`` for Newmark ``beta = 1/4, gamma = 1/2, dt = 1``.
    """
    return -4.0 * stencil_mass / (4.0 * mass + stiffness * BETA * dt ** 2)


# ---------------------------------------------------------------------------
# S2/S3: coupled sweep through the critical ratio
# ---------------------------------------------------------------------------
def _case(ratio, mode, fixed_omega, cfg, n_steps=None):
    c = cfg["half_width"]
    m_a = analytic_added_mass(c)
    m_s = m_a / ratio
    k = (m_s + m_a) * (2.0 * math.pi / cfg["period"]) ** 2
    rec = coupled_free_vibration(
        n_steps=n_steps or cfg["n_steps"], half_width=c,
        center=(cfg["box"] / 2.0, cfg["box"] / 2.0),
        lattice=(cfg["box"], cfg["box"]), mass=m_s, stiffness=k, y0=0.2,
        omega_lbm=cfg["omega_lbm"], relax_mode=mode, fixed_omega=fixed_omega,
        sub_tol=cfg["sub_tol"], max_sub_iter=cfg["max_sub_iter"])
    keep = ("relax_mode", "added_mass_ratio", "mass", "stiffness", "period",
            "period_fft", "frequency", "frequency_ref", "frequency_fft",
            "frequency_ratio", "frequency_ratio_ref", "frequency_error_pct",
            "frequency_error_fft_pct", "added_mass_measured",
            "added_mass_measured_error_pct", "steps_completed", "diverged",
            "diverged_reason", "mean_sub_iterations", "max_sub_iterations",
            "maxit_hits", "converged_all")
    out = {k2: rec[k2] for k2 in keep}
    out["fixed_omega"] = fixed_omega
    g = -4.0 * cfg["stencil_mass"] / (4.0 * m_s + k)      # exact interface gain
    out["gain_pred"] = g
    out["omega_star_pred"] = 1.0 / (1.0 - g)
    out["frequency_from"] = "zero_crossing"
    if not np.isfinite(out["frequency"]) and np.isfinite(out["period_fft"]):
        out["frequency"] = 1.0 / out["period_fft"]
        out["frequency_error_pct"] = 100.0 * (
            out["frequency"] - out["frequency_ref"]) / out["frequency_ref"]
        out["frequency_from"] = "fft"
    # converged relaxation factor actually used (last sub-iteration of each step)
    rh = np.asarray(rec["relax_history"], dtype=float)
    rh = rh[np.isfinite(rh)]          # NaN = step converged, no relaxation
    out["omega_used_mean"] = float(np.mean(rh)) if rh.size else float("nan")
    out["omega_used_last_quarter"] = (float(np.mean(rh[rh.size * 3 // 4:]))
                                      if rh.size else float("nan"))
    out["omega_n_samples"] = int(rh.size)
    out["y_history"] = rec["y_history"]
    out["iter_history"] = rec["iter_history"]
    out["residual_history"] = rec["residual_history"]
    return out


def _cfg(quick):
    if quick:
        return dict(half_width=10.0, box=96, period=300.0, n_steps=400,
                    omega_lbm=1.9, sub_tol=1e-6, max_sub_iter=40,
                    ratios=(1.0, 4.0, 8.0, 12.0))
    return dict(half_width=10.0, box=96, period=300.0, n_steps=1500,
                omega_lbm=1.9, sub_tol=1e-6, max_sub_iter=40,
                ratios=(1.0, 3.0, 6.0, 8.0, 10.0, 12.0, 16.0))


def run(quick=False):
    os.makedirs(OUT, exist_ok=True)
    cfg = _cfg(quick)
    t0 = time.time()
    ev = {"anchor": "XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION",
          "round": 4, "reference": "joconnor22/LIFE (implicit IBM + Aitken)",
          "tolerance_pct": TOL,
          "config": {k: (list(v) if isinstance(v, tuple) else v)
                     for k, v in cfg.items()}}

    print("[S1] interface gain / stencil mass", flush=True)
    s1 = []
    for c, box in ((10.0, 96), (16.0, 144), (20.0, 160)):
        rec = interface_gain(half_width=c, box=box)
        # linearity check of the map: second difference must vanish
        body = SpringMountedBody(plate_markers((box / 2.0, box / 2.0), c),
                                 mass=1.0, stiffness=0.0, y0=0.0)
        fluid = LBMFluid2D((box, box), omega=cfg["omega_lbm"], rho0=1.0)
        solver = IBMImplicitFSI(body, fluid, sub_tol=0.0, max_sub_iter=2)
        rho, u = fluid.macroscopic(include_force=False)
        f_a, _ = solver.interface_force(rho, u, 0.0)
        f_b, _ = solver.interface_force(rho, u, 0.01)
        f_c, _ = solver.interface_force(rho, u, 0.02)
        rec["linearity_residual"] = abs((f_c - f_b) - (f_b - f_a)) \
            / max(abs(f_b - f_a), 1e-30)
        s1.append(rec)
        print("  c=%4.1f S=%8.3f (%.2f per c) m_a=%8.2f ratio_crit=%6.2f lin=%.2e"
              % (c, rec["stencil_mass"], rec["stencil_mass_per_half_width"],
                 rec["added_mass"], rec["ratio_crit"],
                 rec["linearity_residual"]), flush=True)
    ev["S1_gain"] = s1
    cfg["stencil_mass"] = [r for r in s1
                           if r["half_width"] == cfg["half_width"]][0]["stencil_mass"]
    ev["config"]["stencil_mass"] = cfg["stencil_mass"]

    print("[S2] coupled sweep through ratio_crit", flush=True)
    cases = []
    for ratio in cfg["ratios"]:
        for mode, fw in (("aitken", 1.0), ("fixed", 1.0), ("fixed", 0.5)):
            rec = _case(ratio, mode, fw, cfg)
            print("  ratio=%5.1f %-6s w=%.1f steps=%3d it=%5.2f maxit=%3d div=%d "
                  "T=%7.1f err=%+7.2f%%" %
                  (ratio, mode, fw, rec["steps_completed"],
                   rec["mean_sub_iterations"], rec["maxit_hits"],
                   int(rec["diverged"]), rec["period"],
                   rec["frequency_error_pct"]), flush=True)
            cases.append(rec)
    ev["S2_coupled"] = cases

    # ---- S3: Aitken omega vs 1/(1-g) in the iterating regime -------------
    s3 = []
    for rec in cases:
        if rec["relax_mode"] != "aitken" or rec["diverged"]:
            continue
        if not rec["mean_sub_iterations"] > 1.5:
            continue                      # fixed point converged in 1 iteration
        s3.append({"ratio": rec["added_mass_ratio"],
                   "gain_pred": rec["gain_pred"],
                   "omega_star_pred": rec["omega_star_pred"],
                   "omega_used_mean": rec["omega_used_mean"],
                   "omega_used_last_quarter": rec["omega_used_last_quarter"],
                   "n_samples": rec["omega_n_samples"],
                   "error_pct": 100.0 * (rec["omega_used_mean"]
                                         / rec["omega_star_pred"] - 1.0),
                   "error_last_quarter_pct": 100.0 * (
                       rec["omega_used_last_quarter"]
                       / rec["omega_star_pred"] - 1.0)})
    ev["S3_relaxation"] = s3

    # ---- verdict ---------------------------------------------------------
    def _abs_err(rec):
        v = rec["frequency_error_fft_pct"]
        if not np.isfinite(v):
            v = rec["frequency_error_pct"]
        return abs(v) if np.isfinite(v) else float("inf")

    ok = [c for c in cases if not c["diverged"] and np.isfinite(_abs_err(c))]
    ratio_crit = [r for r in s1 if r["half_width"] == cfg["half_width"]][0]["ratio_crit"]
    strong = [c for c in cases if c["added_mass_ratio"] > ratio_crit]
    ev["verdict"] = {
        "ratio_crit": ratio_crit,
        "strong_coupling_cases": len(strong),
        "fixed_omega1_diverged_above_crit": all(
            c["diverged"] or c["maxit_hits"] > 0.5 * c["steps_completed"]
            for c in strong if c["relax_mode"] == "fixed"
            and c["fixed_omega"] == 1.0),
        "aitken_stable_above_crit": all(not c["diverged"] for c in strong
                                        if c["relax_mode"] == "aitken"),
        "coupled_frequency_max_abs_error_pct": max((_abs_err(c) for c in ok),
                                                   default=None),
        "coupled_frequency_all_within_tolerance": bool(ok) and
            max(_abs_err(c) for c in ok) <= TOL,
        "n_cases": len(cases),
        "diverged_cases": [{"ratio": c["added_mass_ratio"],
                            "mode": c["relax_mode"],
                            "omega": c["fixed_omega"],
                            "reason": c["diverged_reason"]}
                           for c in cases if c["diverged"]],
    }
    ev["runtime_s"] = time.time() - t0
    with open(os.path.join(OUT, "evidence.json"), "w") as fh:
        json.dump(ev, fh, indent=2)
    _figures(ev)
    _summary(ev)
    return ev


# ---------------------------------------------------------------------------
# S4: re-measurement of the round-3 plate16 case (untraceable in its artifacts)
# ---------------------------------------------------------------------------
def recheck_plate(half_width=16.0, box=144, period=600.0, n_steps=3000,
                  ratio=0.6, omega_lbm=1.9, sub_tol=1e-9, max_sub_iter=40):
    """Re-run round 3's ``c = 16, m_a/m_s = 0.6, T = 600`` case and save the numbers.

    ``docs/xval_implicit_ibm_aitken_v3.md`` section 4.2 reports ``T = 682.7``
    (``-12.1 %``) for this configuration, but the artifact saved next to it
    (``benchmarks/xval_added_mass_artifacts/evidence_plate16.json``) stores
    ``null`` for period, frequency error and sub-iteration count, so the claim
    could not be traced to evidence.  This re-measures it on a 5-period record
    and writes ``plate16_recheck.json``.
    """
    m_a = analytic_added_mass(half_width)
    m_s = m_a / ratio
    k = (m_s + m_a) * (2.0 * math.pi / period) ** 2
    f_ref = math.sqrt(k / (m_s + m_a)) / (2.0 * math.pi)
    out = {"half_width": half_width, "box": box, "target_period": period,
           "n_steps": n_steps, "added_mass_ratio": ratio, "mass": m_s,
           "stiffness": k, "added_mass_ref": m_a, "frequency_ref": f_ref,
           "period_ref": 1.0 / f_ref, "cases": []}
    for mode, fw in (("aitken", 1.0), ("fixed", 1.0)):
        rec = _case(ratio, mode, fw,
                    {"half_width": half_width, "box": box, "period": period,
                     "n_steps": n_steps, "omega_lbm": omega_lbm,
                     "sub_tol": sub_tol, "max_sub_iter": max_sub_iter,
                     "stencil_mass": 0.0})
        for key in ("y_history", "iter_history", "residual_history"):
            rec.pop(key, None)
        out["cases"].append(rec)
        print("  [S4] c=%.0f %-6s T=%8.2f T_ref=%8.2f err=%+7.2f%% it=%5.2f "
              "maxit=%d div=%d"
              % (half_width, mode, rec["period"], out["period_ref"],
                 rec["frequency_error_pct"], rec["mean_sub_iterations"],
                 rec["maxit_hits"], int(rec["diverged"])), flush=True)
    with open(os.path.join(OUT, "plate16_recheck.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    return out



# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _figures(ev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = ev["S2_coupled"]

    # (1) sub-iteration count / maxit hits vs ratio, per mode
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for mode, fw, colour in (("aitken", 1.0, "tab:blue"),
                             ("fixed", 1.0, "tab:red"),
                             ("fixed", 0.5, "tab:orange")):
        sel = [c for c in cases if c["relax_mode"] == mode
               and abs(c["fixed_omega"] - fw) < 1e-12]
        if not sel:
            continue
        xs = [c["added_mass_ratio"] for c in sel]
        ys = [c["mean_sub_iterations"] for c in sel]
        ax.plot(xs, ys, "o-", color=colour,
                label="%s, omega=%.1f" % (mode, fw))
    crit = ev["verdict"]["ratio_crit"]
    ax.axvline(crit, color="k", ls=":", lw=1.0,
               label="ratio_crit = %.1f (|g|=1)" % crit)
    ax.set_xlabel("added-mass ratio m_a/m_s")
    ax.set_ylabel("mean sub-iterations per step")
    ax.set_title("Interface fixed point: sub-iteration cost vs added-mass ratio")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "subiter_vs_ratio.png"), dpi=130)
    plt.close(fig)

    # (2) measured gain vs prediction, and Aitken omega vs 1/(1-g)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axs[0]
    sel = [c for c in cases if not c["diverged"]]
    xs = [c["added_mass_ratio"] for c in sel]
    ax.plot(xs, [c["gain_pred"] for c in sel], "k--", lw=1.0,
            label="prediction g = -S/m_s")
    ax.axhline(-1.0, color="r", ls=":", lw=1.0, label="|g| = 1")
    ax.set_xlabel("added-mass ratio m_a/m_s")
    ax.set_ylabel("interface gain g")
    ax.set_title("Predicted interface-map gain")
    ax.legend(fontsize=8)

    ax = axs[1]
    s3 = ev["S3_relaxation"]
    if s3:
        xs = [r["ratio"] for r in s3]
        ax.plot(xs, [r["omega_star_pred"] for r in s3], "k--o", lw=1.0,
                label="Aitken optimum 1/(1-g)")
        ax.plot(xs, [r["omega_used_mean"] for r in s3], "s", color="tab:blue",
                label="measured (mean over steps)")
        ax.set_xlabel("added-mass ratio m_a/m_s")
        ax.set_ylabel("Aitken relaxation factor omega")
        ax.set_title("Aitken omega vs analytic optimum")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no iterating case", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "gain_and_omega.png"), dpi=130)
    plt.close(fig)

    # (3) displacement histories above the critical ratio
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for c in cases:
        if c["relax_mode"] != "aitken" or c["diverged"]:
            continue
        ax.plot(np.arange(len(c["y_history"])), c["y_history"], lw=0.9,
                label="m_a/m_s=%.1f (T=%.0f)" % (c["added_mass_ratio"],
                                                 c["period"]))
    ax.set_xlabel("LBM step")
    ax.set_ylabel("plate displacement y")
    ax.set_title("Aitken-relaxed implicit IBM above the critical added-mass ratio")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "history_strong.png"), dpi=130)
    plt.close(fig)

    # (4) frequency error vs ratio
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for mode, fw, colour in (("aitken", 1.0, "tab:blue"),
                             ("fixed", 1.0, "tab:red"),
                             ("fixed", 0.5, "tab:orange")):
        sel = [c for c in cases if c["relax_mode"] == mode
               and abs(c["fixed_omega"] - fw) < 1e-12 and not c["diverged"]]
        if not sel:
            continue
        xs = [c["added_mass_ratio"] for c in sel]
        ys = [c["frequency_error_fft_pct"] if np.isfinite(c["frequency_error_fft_pct"])
              else c["frequency_error_pct"] for c in sel]
        ax.plot(xs, ys, "o-", color=colour, label="%s, omega=%.1f" % (mode, fw))
    ax.axhline(0.0, color="k", lw=0.8)
    for s in (+1, -1):
        ax.axhline(s * ev["tolerance_pct"], color="r", ls="--", lw=0.8)
    ax.set_xlabel("added-mass ratio m_a/m_s")
    ax.set_ylabel("coupled frequency error vs analytic (%)")
    ax.set_title("Frequency accuracy vs added-mass ratio (dashed = +/-15 %)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "frequency_vs_ratio.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# summary markdown
# ---------------------------------------------------------------------------
def _summary(ev):
    c = ev["config"]
    v = ev["verdict"]
    L = ["# XVAL round 4 — implicit IBM + Aitken in the strong added-mass regime",
         "",
         "Anchor `XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION`; reference analytic "
         "(Lamb 1932): `m_a = rho_f pi c^2`, `f = sqrt(k/(m_s+m_a))/2pi`.",
         "",
         "Configuration: plate half-width c=%.1f, lattice %dx%d, target coupled "
         "period %.0f steps, %d steps per case, `sub_tol=%.0e`, `max_sub_iter=%d`."
         % (c["half_width"], c["box"], c["box"], c["period"], c["n_steps"],
            c["sub_tol"], c["max_sub_iter"]),
         "",
         "## S1 — interface gain / stencil mass (frozen fluid state)",
         "",
         "| c | S = -dF/dy beta dt^2/(2 gamma) | S/c | m_a = rho pi c^2 | ratio_crit = m_a/S | linearity residual |",
         "|---|---|---|---|---|---|"]
    for r in ev["S1_gain"]:
        L.append("| %.0f | %.3f | %.3f | %.2f | **%.2f** | %.1e |"
                 % (r["half_width"], r["stencil_mass"],
                    r["stencil_mass_per_half_width"], r["added_mass"],
                    r["ratio_crit"], r["linearity_residual"]))
    L += ["",
          "The interface force is exactly linear in the trial displacement at "
          "fixed fluid state (second difference / first difference below 1e-12), "
          "so the interface map has the *constant* gain `g = -S/m_s`.",
          "",
          "## S2 — coupled sweep through `ratio_crit = %.2f`" % v["ratio_crit"],
          "",
          "| m_a/m_s | mode | omega | steps | mean sub-it | maxit hits | diverged | T | f error | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for cs in ev["S2_coupled"]:
        err = cs["frequency_error_fft_pct"] if np.isfinite(cs["frequency_error_fft_pct"]) \
            else cs["frequency_error_pct"]
        if cs["diverged"]:
            verdict = "DIVERGED"
        elif not np.isfinite(err):
            verdict = "n/a"
        else:
            verdict = "PASS" if abs(err) <= ev["tolerance_pct"] else "FAIL"
        L.append("| %.1f | %s | %.1f | %d | %.2f | %d | %s | %.1f | %+.2f%% | %s |"
                 % (cs["added_mass_ratio"], cs["relax_mode"], cs["fixed_omega"],
                    cs["steps_completed"], cs["mean_sub_iterations"],
                    cs["maxit_hits"], "yes" if cs["diverged"] else "no",
                    cs["period"] if np.isfinite(cs["period"]) else float("nan"),
                    err if np.isfinite(err) else float("nan"), verdict))
    L += ["",
          "## S3 — Aitken relaxation factor vs analytic optimum `1/(1-g)`",
          "",
          "| m_a/m_s | g = -S/m_s | omega* = 1/(1-g) | measured omega | error |",
          "|---|---|---|---|---|"]
    for r in ev["S3_relaxation"]:
        L.append("| %.1f | %+.3f | %.4f | %.4f | %+.1f%% |"
                 % (r["ratio"], r["gain_pred"], r["omega_star_pred"],
                    r["omega_used_mean"], r["error_pct"]))
    L += ["",
          "## Verdict",
          "",
          "* critical added-mass ratio (measured gain crossing |g| = 1): "
          "**%.2f**" % v["ratio_crit"],
          "* fixed-point iteration with `omega = 1` above the critical ratio: "
          "fails (maxit hits / divergence): **%s**"
          % ("confirmed" if v["fixed_omega1_diverged_above_crit"] else "NOT confirmed"),
          "* Aitken-relaxed implicit IBM above the critical ratio: stable: "
          "**%s**" % ("confirmed" if v["aitken_stable_above_crit"] else "NOT confirmed"),
          "* max |frequency error| over converged cases: **%.2f %%** "
          "(acceptance %.0f %%)" % (v["coupled_frequency_max_abs_error_pct"]
                                    if v["coupled_frequency_max_abs_error_pct"]
                                    is not None else float("nan"),
                                    ev["tolerance_pct"]),
          "", "Runtime: %.0f s." % ev["runtime_s"]]
    with open(os.path.join(OUT, "xval_strong_coupling_summary.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    if "--recheck-plate" in sys.argv[1:]:
        recheck_plate()
        raise SystemExit(0)
    ev = run(quick="--quick" in sys.argv[1:])
    print(json.dumps(ev["verdict"], indent=2))