"""XVAL v11 -- closed-form cost and stability law for the implicit-IBM Aitken loop.

Anchor: ``XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION`` (joconnor22/LIFE implicit IBM +
Aitken delta^2 relaxation).  Rounds v1-v10 verified the *coupled frequency* against
Lamb (1932) but left the relaxation itself entirely **qualitative** -- v6 records
``"Aitken convergence vs fixed omega = 1 | qualitative (3 it vs divergence) | n/a"``
and v10 repeats that row in its acceptance table.  This round replaces it with a law
that is written down in closed form *before* it is measured.

Theory (derived here, not fitted)
--------------------------------
Within one time step the fluid state is frozen, so the interface map is exactly
affine in the trial displacement ``y_k`` (linearity of the direct-forcing operator is
already pinned by ``tests/test_aitken_strong_coupling.py``)::

    y_new = G(y) = g * y + b,        g = -4 S_grad / (4 m_s + k)

so the sub-iteration is the scalar fixed point of Irons & Tuck (1969) / Kuettler &
Wall (2008).  With ``r_k = G(y_k) - y_k`` (the solver's convention) and factor
``omega``::

    r_{k+1} = q * r_k,      q = 1 + omega (g - 1) = 1 - omega (1 + R),    R := -g

The production ``AitkenRelaxation.omega_for`` returns, for any base factor,

    omega* = -1/(g-1) = 1/(1 + R)                                  (exact, not asymptotic)

so ``q = 0`` and the fixed point is reached **exactly after two relaxations**, i.e.
three sub-iterations, for *every* added-mass ratio ``R``.  Fixed relaxation instead
converges only while ``|1 - omega(1+R)| < 1``, i.e. for

    omega < omega_crit = 2/(1+R)    <=>    R_crit(omega) = 2/omega - 1     (exact)

The ``omega = 1`` case ``R_crit = 1`` is the added-mass criterion of Causin, Gerbeau
& Nobile (2005).  At tolerance ``tol`` the fixed-relaxation cost is the integer law
``n(tol) = 1 + ceil(ln(tol/|r_1|)/ln|q|)``.

Deliverables
------------
A. scalar model driven by the *production* class: exactness of ``omega*``, finite
   termination count, geometric rate ``|q|``, the cost law and ``R_crit(omega)``;
   the same law re-checked on the repo's second, independent Aitken class
   (``fsi_lbm_strong_coupling``); and the stale-base penalty implied by the
   ``omega_prev`` docstring.
B. the real coupled solver ``IBMImplicitFSI`` at the anchor's strong-coupling
   configuration (``m_a/m_s = 12``) and four other mass ratios: the residual sequence
   is recovered non-invasively through the public ``max_sub_iter`` argument (fresh
   solver per truncation length; the fluid is frozen inside a step), ``R`` is
   measured, and the Aitken factor, the Aitken cost, the fixed-relaxation cost law
   and the stability boundary are each compared with the closed form.

Single core, a few seconds of wall time; no cluster submission needed.

Usage::

    python benchmarks/xval_aitken_rate_law.py            # tables + JSON + PNGs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfem.ibm_aitken_added_mass import (  # noqa: E402
    IBMImplicitFSI,
    LBMFluid2D,
    SpringMountedBody,
    analytic_added_mass,
    plate_markers,
)
from tensorfem.ibm_aitken_implicit import AitkenRelaxation  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "xval_aitken_rate_law_artifacts")

BOX = 64
C = 8.0
DT = 1.0
PERIOD = 400.0


# ---------------------------------------------------------------------------
# A. scalar affine fixed point driven by the production class
# ---------------------------------------------------------------------------
def scalar_loop(g, b, mode="aitken", omega0=1.0, tol=1e-12, maxit=400, x0=0.0,
                aitken=None):
    """Run the scalar interface fixed point exactly as ``IBMImplicitFSI.step`` does.

    Returns ``(n_iterations, x_final, residual_sequence, omega_sequence)``; the
    iteration count includes the converged residual evaluation, matching the
    ``sub_iterations`` convention of the coupled solver.
    """
    if aitken is None:
        aitken = AitkenRelaxation(relax_max=1.0e6, relax0=omega0)
    x = x0
    r_prev = None
    w_applied = float("nan")
    res, omegas = [], []
    for it in range(1, maxit + 1):
        x_new = g * x + b
        r = x_new - x
        res.append(r)
        if abs(r) <= tol:
            return it, x_new, res, omegas
        if mode == "aitken" and it >= 2:
            w = aitken.omega_for(np.array([r_prev]), np.array([r]),
                                 omega_prev=w_applied)
        else:
            w = omega0
        w_applied = w
        omegas.append(w)
        r_prev = r
        x = x + w * (x_new - x)
    return maxit, x, res, omegas


def predict_count(r1, q, tol):
    """Closed-form sub-iteration count: smallest n with |r1| |q|^(n-1) <= tol."""
    if abs(r1) <= tol:
        return 1
    if abs(q) == 0.0:                 # r_2 = 0 exactly: one relaxation suffices
        return 2
    if abs(q) >= 1.0:
        return math.inf
    return max(1, 1 + math.ceil(math.log(tol / abs(r1)) / math.log(abs(q)) - 1e-12))


def part_a_scalar(ratios, omegas, tol):
    rows = []
    for R in ratios:
        g = -R
        b = 1.0 - g                     # fixed point at x* = 1
        n, x, res, om = scalar_loop(g, b, mode="aitken", tol=1e-13)
        w_star = 1.0 / (1.0 + R)
        w2 = om[1] if len(om) > 1 else float("nan")
        rows.append({"test": "A1_A2_aitken_exact", "R": R, "g": g,
                     "omega_star_closed_form": w_star, "omega_star_measured": w2,
                     "rel_err": abs(w2 - w_star) / abs(w_star),
                     "n_iter_measured": n, "n_iter_predicted": 3,
                     "final_err": abs(x - 1.0)})

    for R in ratios:
        for w in omegas:
            g = -R
            b = 1.0 - g
            q = 1.0 + w * (g - 1.0)
            n, _x, res, _ = scalar_loop(g, b, mode="fixed", omega0=w, tol=tol)
            q_meas = abs(res[1]) / abs(res[0]) if len(res) > 1 and res[0] else float("nan")
            diverges = abs(q) >= 1.0
            n_pred = predict_count(abs(res[0]), q, tol)
            agrees = (n == 400) if diverges else (n == int(n_pred))
            rows.append({"test": "A3_fixed_rate", "R": R, "omega": w,
                         "q_closed_form": q, "q_measured": q_meas,
                         "q_rel_err": (abs(q_meas - abs(q)) / abs(q)) if q else 0.0,
                         "n_iter_measured": n,
                         "n_iter_predicted": (math.inf if diverges else int(n_pred)),
                         "count_agrees": bool(agrees),
                         "diverges_predicted": bool(diverges)})

    bnd = []
    for w in omegas:
        R_crit_pred = 2.0 / w - 1.0
        if R_crit_pred <= 0.0:
            bnd.append({"test": "A4_critical_ratio", "omega": w,
                        "R_crit_closed_form": R_crit_pred,
                        "note": "no R >= 0 converges",
                        "stable_at_R_1e-6": bool(abs(1.0 + w * (-1e-6 - 1.0)) < 1.0),
                        "stable_at_R_50": bool(abs(1.0 + w * (-50.0 - 1.0)) < 1.0)})
            continue
        lo, hi = 1e-6, 50.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if abs(1.0 + w * (-mid - 1.0)) >= 1.0:
                hi = mid
            else:
                lo = mid
        R_crit_meas = 0.5 * (lo + hi)
        bnd.append({"test": "A4_critical_ratio", "omega": w,
                    "R_crit_closed_form": R_crit_pred,
                    "R_crit_bisected": R_crit_meas,
                    "abs_err": abs(R_crit_meas - R_crit_pred),
                    "rel_err": abs(R_crit_meas - R_crit_pred) / abs(R_crit_pred)})

    from tensorfem.fsi_lbm_strong_coupling import AitkenRelaxation as Aitken2

    second = []
    for R in ratios:
        g = -R
        b = 1.0 - g
        a = Aitken2(omega0=1.0)
        x, hist, it = 0.0, [], 0
        for it in range(1, 200):
            x_new = g * x + b
            r = x_new - x
            if abs(r) <= 1e-13:
                break
            w = a.update(np.array([r]))
            hist.append(w)
            x = x + w * (x_new - x)
        w_star = 1.0 / (1.0 + R)
        w2 = hist[1] if len(hist) > 1 else float("nan")
        second.append({"test": "A5_second_implementation", "R": R,
                       "omega_star_closed_form": w_star,
                       "omega_star_measured": w2,
                       "rel_err": abs(w2 - w_star) / abs(w_star),
                       "n_iter_measured": it, "n_iter_predicted": 3})

    a = AitkenRelaxation(relax_max=1.0e6, relax0=1.0)
    r_km1, r_k = np.array([1.0]), np.array([-0.5])
    w_correct = a.omega_for(r_km1, r_k, omega_prev=0.4)
    a2 = AitkenRelaxation(relax_max=1.0e6, relax0=0.25)
    a2.omega = 0.25                      # stale factor carried over from the last step
    w_stale = a2.omega_for(r_km1, r_k)   # base defaults to the stale store
    stale = {"test": "A6_stale_base", "omega_with_applied": w_correct,
             "omega_with_stale": w_stale,
             "ratio_measured": w_stale / w_correct,
             "ratio_predicted": 0.25 / 0.4,
             "rel_err": abs(w_stale / w_correct - 0.25 / 0.4) / (0.25 / 0.4)}
    return rows, bnd, second, stale


# ---------------------------------------------------------------------------
# B. the real coupled solver
# ---------------------------------------------------------------------------
def make_solver(m_s, k, mode="aitken", omega=1.0, sub_tol=1e-8, max_sub_iter=40,
                y0=0.2):
    body = SpringMountedBody(plate_markers((BOX / 2.0, BOX / 2.0), C),
                             mass=m_s, stiffness=k, y0=y0)
    fluid = LBMFluid2D((BOX, BOX), omega=1.9, rho0=1.0)
    return IBMImplicitFSI(body, fluid, dt=DT, sub_tol=sub_tol,
                          max_sub_iter=max_sub_iter, relax_mode=mode,
                          fixed_omega=omega)


def config_for_mass_ratio(mass_ratio, period=PERIOD):
    """Anchor configuration: added mass from Lamb, stiffness from the target period."""
    m_a = analytic_added_mass(C)
    m_s = m_a / mass_ratio
    k = (m_s + m_a) * (2.0 * math.pi / period) ** 2
    return m_s, k


def residual_norms(m_s, k, omega, n_warm=0, nmax=6):
    """Sub-iteration residual norms recovered through the public API only.

    ``sub_tol = 0, max_sub_iter = m`` makes ``step()`` report the m-th residual; a
    fresh solver per m replays the same deterministic prefix.
    """
    seq = []
    for m in range(1, nmax + 1):
        solver = make_solver(m_s, k, mode="aitken", sub_tol=1e-8, max_sub_iter=40)
        if n_warm:
            solver.run(n_warm)
        solver.relax_mode = "fixed"
        solver.fixed_omega = omega
        solver.sub_tol = 0.0
        solver.max_sub_iter = m
        solver.step()
        seq.append(float(solver.residuals[-1]))
    return seq


def gain_from_norms(m_s, k, n_warm=0, nmax=6):
    """R = |q| measured with a unit relaxation factor, plus constancy across k."""
    seq = residual_norms(m_s, k, omega=1.0, n_warm=n_warm, nmax=nmax)
    ratios = [seq[i + 1] / seq[i] for i in range(len(seq) - 1)]
    return seq, ratios, float(np.mean(ratios)), float(np.ptp(ratios) / np.mean(ratios))


def step_cost(m_s, k, mode, omega, tol=1e-8, maxit=60):
    """One step at a fixed setting; a divergent excursion counts as failure.

    A diverging interface walk -- the physically expected outcome above the
    critical ratio -- can push a marker stencil off the lattice, and the IBM
    operator raises ``ValueError`` there.  That *is* the instability, so it is
    caught and reported as non-convergence instead of aborting the study.
    """
    solver = make_solver(m_s, k, mode=mode, omega=omega, sub_tol=tol,
                         max_sub_iter=maxit)
    try:
        solver.step()
    except (ValueError, FloatingPointError, OverflowError) as exc:
        return maxit, False, float("inf")
    n = int(solver.sub_iterations[-1])
    res = float(solver.residuals[-1])
    converged = bool(n < maxit and res <= tol and np.isfinite(solver.body.y)
                     and abs(solver.body.y) < 1.0e3)
    return n, converged, res


def decays(m_s, k, omega, r1, m_iter=120, factor=0.95):
    """True when the fixed-relaxation loop strictly decays within ``m_iter``.

    Judging the boundary by "reaches ``sub_tol``" would be resolution limited: near
    ``|q| = 1`` the required count diverges, so any finite ``max_sub_iter`` would
    place the measured boundary *inside* the true one by a bias that depends on the
    initial residual.  Requiring only that the residual shrink below ``factor * r1``
    resolves ``|q|`` up to ``factor**(1/m_iter)`` = 0.99957 at the default settings,
    i.e. ~0.04 % in ``omega_crit``, independent of ``r1``.
    """
    solver = make_solver(m_s, k, mode="fixed", omega=omega, sub_tol=0.0,
                         max_sub_iter=m_iter)
    try:
        solver.step()
    except (ValueError, FloatingPointError, OverflowError):
        return False
    res = float(solver.residuals[-1])
    return bool(np.isfinite(res) and res <= factor * r1
                and np.isfinite(solver.body.y) and abs(solver.body.y) < 1.0e3)


def bisect_omega_crit(m_s, k, r1, m_iter=120, factor=0.95):
    """Largest fixed factor for which the loop still decays: the stability edge."""
    lo, hi = 0.02, 4.0
    if decays(m_s, k, lo, r1, m_iter, factor):
        pass
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if decays(m_s, k, mid, r1, m_iter, factor):
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-5 * hi:
            break
    return 0.5 * (lo + hi), factor ** (1.0 / m_iter)


def part_b_real(mass_ratios, tol):
    rows = {k: [] for k in ("gain", "invariance", "aitken", "staircase", "boundary")}
    for mr in mass_ratios:
        m_s, k = config_for_mass_ratio(mr)
        seq, ratios, R_meas, spread = gain_from_norms(m_s, k, n_warm=0)
        seq5, ratios5, R5, spread5 = gain_from_norms(m_s, k, n_warm=5)
        rows["gain"].append({"mass_ratio_param": mr, "m_s": m_s, "k": k,
                             "residual_norms": seq, "ratios": ratios,
                             "R_measured": R_meas, "ratio_spread_rel": spread})
        rows["invariance"] += [
            {"mass_ratio_param": mr, "n_warm": 0, "R_measured": R_meas,
             "residual_norms": seq, "spread_rel": spread},
            {"mass_ratio_param": mr, "n_warm": 5, "R_measured": R5,
             "residual_norms": seq5, "spread_rel": spread5},
        ]
        nA, convA, resA = step_cost(m_s, k, "aitken", 1.0, tol=tol, maxit=40)
        solver = make_solver(m_s, k, mode="aitken", sub_tol=tol, max_sub_iter=40)
        solver.step()
        wA = float([w for w in solver.relaxes if np.isfinite(w)][-1])
        w_star = 1.0 / (1.0 + R_meas)
        rows["aitken"].append({"mass_ratio_param": mr, "R_measured": R_meas,
                               "omega_star_closed_form": w_star,
                               "omega_star_measured": wA,
                               "rel_err": abs(wA - w_star) / w_star,
                               "n_iter_measured": nA, "n_iter_predicted": 3,
                               "converged": convA, "res_final": resA})
        for w in (0.5, 1.0):
            q = 1.0 - w * (1.0 + R_meas)
            for tt in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
                n, conv, res = step_cost(m_s, k, "fixed", w, tol=tt, maxit=80)
                n_pred = predict_count(seq[0], q, tt)
                rows["staircase"].append({
                    "mass_ratio_param": mr, "omega": w, "R": R_meas, "tol": tt,
                    "q": q, "n_iter_measured": n,
                    "n_iter_predicted": (math.inf if not math.isfinite(n_pred)
                                         else int(n_pred)),
                    "converged": conv, "res_final": res})
        w_crit_meas, q_floor = bisect_omega_crit(m_s, k, seq[0])
        w_crit_pred = 2.0 / (1.0 + R_meas)
        rows["boundary"].append({"mass_ratio_param": mr, "R_measured": R_meas,
                                 "r1_normalised": seq[0],
                                 "omega_crit_closed_form": w_crit_pred,
                                 "omega_crit_bisected": w_crit_meas,
                                 "rel_err": abs(w_crit_meas - w_crit_pred) / w_crit_pred,
                                 "rate_floor_resolved": q_floor,
                                 # the bisection classifies |q| >= q_floor as "not
                                 # decaying", so the measured edge sits at
                                 # omega = (1+q_floor)/(1+R); undo that bias
                                 "omega_crit_bias_corrected":
                                     2.0 * w_crit_meas / (1.0 + q_floor),
                                 "rel_err_bias_corrected":
                                     abs(2.0 * w_crit_meas / (1.0 + q_floor)
                                         - w_crit_pred) / w_crit_pred,
                                 "R_crit_from_measured_omega": 2.0 / w_crit_meas - 1.0})
    return rows


def stair_agrees(r):
    if isinstance(r["n_iter_predicted"], int) and r["converged"]:
        return r["n_iter_measured"] == r["n_iter_predicted"]
    return (not r["converged"]) and r["n_iter_predicted"] == math.inf


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_rate_law(rows, path):
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    conv = [r for r in rows["staircase"] if r["converged"] and r["q"] != 0.0]
    q = [abs(r["q"]) for r in conv]
    n = [r["n_iter_measured"] for r in conv]
    ax.loglog(q, n, "o", ms=6, alpha=0.75, label="measured (real solver)")
    qq = np.logspace(math.log10(min(q)), math.log10(max(q)), 120)
    ax.loglog(qq, [max(1.0, -math.log10(1e-8) / math.log10(v)) for v in qq], "k--",
              lw=1, label=r"$\ln(tol)/\ln|q|$ trend")
    # one-parameter fit through the measured cloud for a quantitative check
    slope = np.polyfit(np.log10(q), np.log10(n), 1)[0]
    ax.plot(qq, 10 ** (np.polyval(np.polyfit(np.log10(q), np.log10(n), 1),
                                  np.log10(qq))), ":", lw=1.2,
            label=rf"measured slope {slope:.2f} (closed form $-1$) ")
    ax.set_xlabel(r"$|q| = |1-\omega(1+R)|$")
    ax.set_ylabel(r"sub-iterations to $10^{-8}$")
    ax.set_title("fixed-relaxation cost obeys the closed-form rate")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_boundary(rows, path):
    ws = np.linspace(0.15, 2.6, 300)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot(ws, 2.0 / ws - 1.0, "k-", lw=1.6, label=r"closed form $R=2/\omega-1$")
    ax.plot([r["omega_crit_bisected"] for r in rows["boundary"]],
            [r["R_measured"] for r in rows["boundary"]], "o", ms=9, mfc="none",
            mew=1.6, label="measured boundary (real solver)")
    for r in rows["boundary"]:
        ax.annotate(rf"$m_a/m_s$={r['mass_ratio_param']:g}",
                    (r["omega_crit_bisected"], r["R_measured"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax.axhline(1.0, color="tab:gray", ls=":", lw=1)
    ax.text(0.18, 1.04, r"Causin et al. (2005): $R_{crit}=1$ at $\omega=1$",
            fontsize=8, color="tab:gray")
    ax.set_xlabel(r"fixed relaxation factor $\omega$")
    ax.set_ylabel(r"interface gain $R=-g$")
    ax.set_title("stability boundary of the implicit-IBM sub-iteration")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_cost(rows, path):
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    R = [r["R_measured"] for r in rows["gain"]]
    nA = [next(x["n_iter_measured"] for x in rows["aitken"]
               if x["mass_ratio_param"] == m) for m in
          [r["mass_ratio_param"] for r in rows["gain"]]]
    ax.semilogy(R, nA, "^", ms=9, color="tab:green", label="measured, Aitken")
    ax.axhline(3.0, color="tab:green", ls="--", lw=1,
               label=r"Aitken closed form: 3 sub-iterations, any $R$")
    for w, st in ((0.5, "s"), (1.0, "o")):
        sel = [r for r in rows["staircase"]
               if abs(r["omega"] - w) < 1e-12 and r["converged"]]
        if sel:
            ax.semilogy([r["R"] for r in sel], [r["n_iter_measured"] for r in sel],
                        st, ms=6, mfc="none",
                        label=rf"measured, fixed $\omega={w}$")
    ax.set_xlabel(r"interface gain $R=-g$")
    ax.set_ylabel(r"sub-iterations per time step (tol $10^{-8}$)")
    ax.set_title("cost of the implicit-IBM coupling")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main(argv=None):  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tol", type=float, default=1e-8)
    args = ap.parse_args(argv)

    os.makedirs(OUT, exist_ok=True)
    scalar_ratios = [0.1, 0.3, 0.6, 1.0, 2.5, 4.0, 12.0]
    omegas = [0.5, 1.0, 1.5, 2.0, 2.5]
    mass_ratios = [1.0, 2.0, 4.0, 12.0, 30.0]

    a_rows, a_bnd, a_second, a_stale = part_a_scalar(scalar_ratios, omegas, args.tol)
    print("=" * 78)
    print("A. scalar affine fixed point, production AitkenRelaxation")
    print("=" * 78)
    print(f"{'R':>7} {'omega*cf':>10} {'omega*meas':>11} {'rel_err':>10} "
          f"{'n_meas':>7} {'n_pred':>7} {'|x-x*|':>9}")
    for r in a_rows:
        if r["test"] != "A1_A2_aitken_exact":
            continue
        print(f"{r['R']:>7.2f} {r['omega_star_closed_form']:>10.6f} "
              f"{r['omega_star_measured']:>11.6f} {r['rel_err']:>10.2e} "
              f"{r['n_iter_measured']:>7d} {r['n_iter_predicted']:>7d} "
              f"{r['final_err']:>9.1e}")
    fixed = [r for r in a_rows if r["test"] == "A3_fixed_rate"]
    worst_q = max(r["q_rel_err"] for r in fixed)
    n_hits = sum(1 for r in fixed if r["count_agrees"])
    print(f"\nfixed-omega: worst |q| error {worst_q:.2e}; cost/divergence law "
          f"agrees on {n_hits}/{len(fixed)} (omega,R) pairs")
    for r in a_bnd:
        if "note" in r:
            print(f"critical ratio omega={r['omega']:.2f}: closed form "
                  f"{r['R_crit_closed_form']:+.3f} -> {r['note']}; "
                  f"stable at R=1e-6: {r['stable_at_R_1e-6']}, "
                  f"at R=50: {r['stable_at_R_50']}")
        else:
            print(f"critical ratio omega={r['omega']:.2f}: closed form "
                  f"{r['R_crit_closed_form']:+.6f}, bisected "
                  f"{r['R_crit_bisected']:+.6f}, err {r['abs_err']:.2e}")
    print("second implementation (fsi_lbm_strong_coupling): worst omega* error "
          f"{max(r['rel_err'] for r in a_second):.2e}, termination "
          f"{[r['n_iter_measured'] for r in a_second]}")
    print(f"stale-base ratio measured {a_stale['ratio_measured']:.6f} vs predicted "
          f"{a_stale['ratio_predicted']:.6f} (rel_err {a_stale['rel_err']:.2e})")

    b = part_b_real(mass_ratios, args.tol)
    print("\n" + "=" * 78)
    print(f"B. real coupled solver IBMImplicitFSI (box {BOX}, c = {C}, dt = {DT})")
    print("=" * 78)
    print(f"{'m_a/m_s':>8} {'m_s':>9} {'R_meas':>8} {'spread':>8} {'R(nw=5)':>9} "
          f"{'omega*cf':>9} {'omega*ms':>9} {'n_it':>5} {'w_err':>8}")
    inv5 = [r for r in b["invariance"] if r["n_warm"] == 5]
    for g_row, a_row, inv in zip(b["gain"], b["aitken"], inv5):
        print(f"{g_row['mass_ratio_param']:>8.1f} {g_row['m_s']:>9.4f} "
              f"{g_row['R_measured']:>8.4f} {g_row['ratio_spread_rel']:>8.1e} "
              f"{inv['R_measured']:>9.4f} {a_row['omega_star_closed_form']:>9.6f} "
              f"{a_row['omega_star_measured']:>9.6f} {a_row['n_iter_measured']:>5d}",
              f" {a_row['rel_err']:.1e}")
    n_stair = len(b["staircase"])
    n_ok = sum(1 for r in b["staircase"] if stair_agrees(r))
    print(f"\nfixed-relaxation cost law: integer-exact on {n_ok}/{n_stair} "
          f"(R, omega, tol) points")
    print(f"{'m_a/m_s':>8} {'omega':>6} {'tol':>8} {'n_meas':>7} {'n_pred':>7} {'ok':>4}")
    for r in b["staircase"]:
        np_s = "-" if r["n_iter_predicted"] == math.inf else str(r["n_iter_predicted"])
        print(f"{r['mass_ratio_param']:>8.1f} {r['omega']:>6.2f} {r['tol']:>8.0e} "
              f"{r['n_iter_measured']:>7d} {np_s:>7} "
              f"{str(stair_agrees(r)):>4}")
    print("\nstability boundary (fixed relaxation), bisected in omega:")
    for r in b["boundary"]:
        print(f"  m_a/m_s={r['mass_ratio_param']:>5.1f}  R_meas={r['R_measured']:.4f}  "
              f"r1={r['r1_normalised']:.3e}  "
              f"omega_crit cf={r['omega_crit_closed_form']:.5f}  "
              f"measured={r['omega_crit_bisected']:.5f}  "
              f"rel_err={r['rel_err']:.2e}  "
              f"(|q| resolved to {r['rate_floor_resolved']:.5f}; "
              f"bias-corrected rel_err {r['rel_err_bias_corrected']:.2e})")

    payload = {"anchor": "XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION", "round": "v11",
               "tol": args.tol,
               "scalar": {"rows": a_rows, "boundary": a_bnd,
                          "second_implementation": a_second, "stale_base": a_stale},
               "real": b,
               "grid": {"box": BOX, "half_width": C, "dt": DT,
                        "scalar_ratios": scalar_ratios, "omegas": omegas,
                        "mass_ratios": mass_ratios}}
    with open(os.path.join(OUT, "xval_aitken_rate_law.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    fig_rate_law(b, os.path.join(OUT, "xval_aitken_v11_rate.png"))
    fig_boundary(b, os.path.join(OUT, "xval_aitken_v11_boundary.png"))
    fig_cost(b, os.path.join(OUT, "xval_aitken_v11_cost.png"))
    print(f"\nartifacts -> {OUT}")

    ok = (worst_q < 1e-10 and n_hits == len(fixed)
          and max(r["rel_err"] for r in a_rows
                  if r["test"] == "A1_A2_aitken_exact") < 1e-12
          and all(r["n_iter_measured"] == 3 for r in b["aitken"] if r["converged"])
          and n_ok == n_stair)
    print("\nVERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())