"""Cross-validation v4: implicit-IBM + Aitken relaxation (XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION).

Reference side: joconnor22/LIFE (Lattice Boltzmann – immersed boundary – finite
element; implicit direct forcing + Aitken Δ² sub-iterations).

Improvements over v3:
* Vacuum frequency uses ≥ 3 periods (FFT fallback when < 2 zero crossings).
* Submerged frequency uses Hann-windowed FFT peak (robust for short records).
* ε=1 force-transmission row no longer conflates the published LIFE band with the
  sanity window actually applied: the row reports both, is judged OUT_OF_REF_BAND
  when the measurement sits outside LIFE's published 11–37 %, and is excluded from
  the cross-validation pass count (our centreline-marker geometry differs from
  LIFE's surface markers, so the row is a sanity check, not an agreement test).
* Added-mass comparison uses the *plate* geometry (analytic m_a = ρ_f π c²)
  instead of the beam (whose effective thickness is set by the delta support).

Run: ``python benchmarks/xval_implicit_ibm_aitken_v4.py [--quick]``
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfem.ibm_aitken_implicit import (  # noqa: E402
    AitkenRelaxation,
    BeamProperties,
    CorotationalBeam2D,
    IBMInterface,
    beam_natural_frequencies,
    cantilever_free_vibration,
    dirac_delta_3pt,
)
from tensorfem.ibm_aitken_added_mass import (  # noqa: E402
    SpringMountedBody,
    IBMImplicitFSI,
    LBMFluid2D,
    estimate_period_damped,
    period_window_ok,
    plate_markers,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "xval_implicit_ibm_aitken_v4_artifacts")
TOL = 0.15  # owner acceptance: cross-validation error < 15 %

# Published LIFE band (``src/Objects.cpp`` epsilon-conservation study, surface
# markers) and the sanity window applied to *our* centreline-marker geometry.
# These are different objects and must never be printed as one criterion: the
# shipped v4 table showed ``ref = 11-37 %`` next to a 61 % measurement marked
# PASS, which is a 65 % deviation from the published band.
FXT_REF_BAND = (0.11, 0.37)
FXT_SANITY_BAND = (0.02, 0.70)


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and obj != obj:  # NaN
        return None
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ---------------------------------------------------------------------------
def _estimate_period_fft(series, dt):
    """Period from Hann-windowed FFT peak (works even with < 2 zero crossings)."""
    y = np.asarray(series, dtype=float)
    y = y - y.mean()
    n = len(y)
    if n < 4:
        return float("nan")
    w = np.hanning(n)
    Y = np.fft.rfft(y * w)
    freqs = np.fft.rfftfreq(n, d=dt)
    peak_idx = np.argmax(np.abs(Y[1:])) + 1
    if peak_idx >= len(freqs):
        return float("nan")
    f_peak = freqs[peak_idx]
    if f_peak <= 0:
        return float("nan")
    return 1.0 / f_peak


def _estimate_period_zc(series, dt):
    """Period from upward zero crossings (original method)."""
    y = np.asarray(series, dtype=float)
    y = y - y.mean()
    idx = np.where((y[:-1] <= 0.0) & (y[1:] > 0.0))[0]
    if len(idx) < 2:
        return float("nan")
    return float((idx[-1] - idx[0]) * dt / (len(idx) - 1))


def estimate_period(series, dt):
    """Robust period: zero crossings if ≥ 2, else FFT."""
    p_zc = _estimate_period_zc(series, dt)
    if p_zc == p_zc:
        return p_zc
    return _estimate_period_fft(series, dt)


# ---------------------------------------------------------------------------
def band_deviation(got, band):
    """Relative distance outside ``band`` (0 when inside, NaN when undefined)."""
    lo, hi = band
    if got != got:
        return float("nan")
    if got < lo:
        return float((lo - got) / lo)
    if got > hi:
        return float((got - hi) / hi)
    return 0.0


def force_band_verdict(got, ref_band=FXT_REF_BAND, sanity_band=FXT_SANITY_BAND,
                       tol=TOL):
    """Three-way verdict for the ε=1 force-transmission row.

    ``IN_BAND``     the measurement agrees with LIFE's published band within ``tol``;
    ``OUT_OF_REF_BAND`` the measurement is inside the sanity window but deviates
                    from the published band (a *configuration* difference, not a
                    cross-validation pass);
    ``OUTSIDE_SANITY`` the measurement is not even inside the sanity window.

    Only ``IN_BAND`` counts as a cross-validation pass.
    """
    dev = band_deviation(got, ref_band)
    in_sanity = bool(sanity_band[0] <= got <= sanity_band[1]) if got == got else False
    if not in_sanity:
        verdict = "OUTSIDE_SANITY"
    elif dev <= tol:
        verdict = "IN_BAND"
    else:
        verdict = "OUT_OF_REF_BAND"
    return {"verdict": verdict, "pass": verdict == "IN_BAND",
            "ref_band": list(ref_band), "sanity_band": list(sanity_band),
            "ref_band_deviation": float(dev)}


# ---------------------------------------------------------------------------
def kernel_and_epsilon_checks():
    """Kernel partition of unity and IBM force-transmission conservation."""
    offs = np.linspace(0.0, 1.0, 401)
    best = 0.0
    for x in offs:
        i = np.arange(-3, 4)
        best = max(best, abs(float(dirac_delta_3pt(x - i).sum()) - 1.0))

    n = 41
    s = np.linspace(0.0, 1.0, n)
    markers = np.stack((20.0 + 40.0 * s, 32.0 + 3.0 * np.sin(2.0 * np.pi * s)), axis=1)
    ibm_raw = IBMInterface(markers, (64, 64), dx=1.0, correct_epsilon=False)
    ibm_eps = IBMInterface(markers, (64, 64), dx=1.0, correct_epsilon=True)
    raw_err = ibm_raw.uniform_force_transmission_error()
    corr_err = ibm_eps.uniform_force_transmission_error()

    rng = np.random.default_rng(7)
    f = rng.normal(size=(n, 2))
    tot_grid = ibm_eps.spread(f).sum(axis=(0, 1))
    tot_marker = (f * ibm_eps.eps_used[:, None] * ibm_eps.ds[:, None]).sum(axis=0)
    total_f_rel = float(np.linalg.norm(tot_grid - tot_marker) / np.linalg.norm(tot_marker))

    return {
        "partition_of_unity_error": best,
        "partition_ref": 2.2e-16,
        "force_transmission_error_eps1": raw_err,
        "force_transmission_ref_band": list(FXT_REF_BAND),
        "force_transmission_sanity_band": list(FXT_SANITY_BAND),
        "force_transmission_error_corrected": corr_err,
        "force_transmission_ref_corrected": 9e-16,
        "total_marker_force_rel_error": total_f_rel,
        "n_markers": n,
    }


# ---------------------------------------------------------------------------
def aitken_scalar_study(gains=(1.5, 2.5, 4.0), tol=1e-10, max_it=200):
    """Scalar interface fixed point U -> g U + b: Aitken vs fixed relaxation."""
    rows = []
    for g in gains:
        b = 1.0
        omega_star = 1.0 / (1.0 - g)
        for mode in ("aitken", "fixed1", "fixed_opt"):
            ait = AitkenRelaxation(relax_max=5.0, relax0=1.0)
            u, r_km1 = 0.0, None
            resid = []
            omega = 1.0
            it = 0
            for it in range(1, max_it + 1):
                u_new = g * u + b
                r = u_new - u
                res = abs(r)
                resid.append(res)
                if res < tol:
                    u = u_new
                    break
                if mode == "aitken":
                    if it == 1:
                        omega = 1.0
                    else:
                        num = r_km1 * (r - r_km1)
                        den = (r - r_km1) ** 2
                        omega = -omega * num / den if den > 0 else omega
                        omega = math.copysign(min(abs(omega), 5.0), omega)
                elif mode == "fixed1":
                    omega = 1.0
                else:
                    omega = omega_star
                u = u + omega * (u_new - u)
                r_km1 = r
            converged = it < max_it and resid[-1] < tol
            rows.append({"gain": g, "mode": mode, "iterations": it,
                         "converged": bool(converged),
                         "omega_final": float(omega),
                         "omega_star": float(omega_star),
                         "relax_error": (abs(omega - omega_star) / abs(omega_star)
                                         if mode == "aitken" and converged else None),
                         "residual_final": float(resid[-1])})
    return rows


# ---------------------------------------------------------------------------
def euler_bernoulli_f1(E, I, rho, A, length):
    """First clamped-free Euler-Bernoulli frequency [Hz].

    ``f1 = (1.87510407**2 / 2 pi) * sqrt(E I / (rho A L**4))``.

    The **area A belongs in the denominator**.  The original v4 harness wrote
    ``rho * length**4`` only, which over-estimates ``f1`` by ``1/sqrt(A)`` for
    every section of non-unit area (for ``A = 1/50``: a factor 7.07).  The
    formula lives in a named function so a unit test can pin it.
    """
    return (1.87510407 ** 2) / (2.0 * math.pi) * math.sqrt(
        float(E) * float(I) / (float(rho) * float(A) * float(length) ** 4))


def structure_checks(n_elements=10, dt=5.0e-3, n_steps=3000, amplitude=0.02,
                     props=None, min_cycles=2.0):
    """Clamped-beam eigen-frequency and vacuum free-vibration frequency.

    Two defects of the first v4 run are fixed here:

    * **Section propagation.**  The section handed to
      :func:`cantilever_free_vibration` is now the section the analytic
      reference is computed from (``props=``).  The pre-fix harness let the
      library fall back to its unit section (``A = 1``, ``I = 1/12``) and then
      compared the 43.048 Hz release of *that* beam with the 0.144486 Hz
      reference of the ``t = 1/50`` section -- the "300x mismatch" recorded in
      the shipped evidence file.
    * **Window gating.**  The vacuum period comes from the window-safe
      damped-sinusoid fit (:func:`estimate_period_damped`) and is accepted only
      if :func:`period_window_ok` reports at least ``min_cycles`` periods in the
      record; otherwise the estimate is ``nan`` and the row is reported as
      ``INVALID_WINDOW`` instead of as a physics failure.
    """
    t = 1.0 / 50.0
    props = props or BeamProperties(E=1.0e5, A=t, I=t ** 3 / 12.0, rho=1.0,
                                    length=1.0, thickness=t)
    beam = CorotationalBeam2D(n_elements=n_elements, props=props)
    # The generalised eigen-solve is not portable for slender sections (t = 1/50):
    # LAPACK's ``eigvalsh`` on the loong64 nodes raises ``Eigenvalues did not
    # converge`` on the Cholesky-whitened pencil, while x86 converges.  It is kept
    # as a diagnostic only; the acceptance check for f1 is the free-vibration
    # record, which integrates the same operator and is platform independent.
    try:
        f_eig = beam_natural_frequencies(beam, n_modes=2)
        eigen_status = "ok"
    except np.linalg.LinAlgError as exc:  # pragma: no cover - platform dependent
        f_eig = np.array([float("nan")])
        eigen_status = f"LinAlgError: {exc}"
    f_analytic = euler_bernoulli_f1(props.E, props.I, props.rho, props.A,
                                    props.length)
    period_ref = 1.0 / f_analytic
    vac = cantilever_free_vibration(n_steps=n_steps, n_elements=n_elements, dt=dt,
                                    amplitude=amplitude, fluid_rho=0.0, props=props)
    tips = vac["tip_history"]
    fit = estimate_period_damped(tips, dt)
    period_legacy = estimate_period(tips, dt)
    window_ok = bool(period_window_ok(len(tips), dt, period_ref, min_cycles)
                     and fit["valid"])
    period = float(fit["period"]) if window_ok else float("nan")
    freq = 1.0 / period if period == period else float("nan")
    sim = vac.get("section", {})
    section_consistent = all(abs(float(sim.get(k, float("nan"))) - float(getattr(props, k)))
                             <= 1e-12 * max(1.0, abs(float(getattr(props, k))))
                             for k in ("E", "A", "I", "rho", "length", "thickness"))
    return {
        "f1_analytic": f_analytic,
        "f1_analytic_basis": "Euler-Bernoulli f1 = 1.8751^2/(2pi) sqrt(E I/(rho A L^4))",
        "f1_fem_eigen": float(f_eig[0]),
        "f1_eigen_rel_error": (abs(float(f_eig[0]) - f_analytic) / f_analytic
                               if np.isfinite(float(f_eig[0])) else float("nan")),
        "f1_eigen_status": eigen_status,
        "period_reference": period_ref,
        "vacuum_period_fit": float(fit["period"]),
        "vacuum_period_fit_valid": bool(fit["valid"]),
        "vacuum_period": period,
        "vacuum_frequency": freq,
        "vacuum_freq_rel_error": (abs(freq - f_analytic) / f_analytic
                                  if freq == freq else float("nan")),
        "vacuum_period_legacy_zc": float(period_legacy),
        "vacuum_frequency_legacy_zc": (1.0 / period_legacy
                                       if period_legacy == period_legacy
                                       else float("nan")),
        "vacuum_cycles_in_record": float(fit["cycles"]),
        "vacuum_min_cycles": float(min_cycles),
        "vacuum_window_ok": window_ok,
        "section_requested": {k: float(getattr(props, k)) for k in
                             ("E", "A", "I", "rho", "length", "thickness")},
        "section_simulated": sim,
        "section_consistent": bool(section_consistent),
        "tip_history": tips,
        "dt": dt,
        "n_elements": n_elements,
        "n_steps": n_steps,
        "slenderness": props.length / props.thickness,
        "structural_props": {"E": props.E, "A": props.A, "I": props.I,
                             "rho": props.rho, "length": props.length,
                             "thickness": props.thickness},
    }


# ---------------------------------------------------------------------------
def vacuum_verdict(st, tol=TOL):
    """Three-way verdict for the beam vacuum free-vibration row.

    Same contract as :func:`plate_verdict`: a record that does not hold
    ``min_cycles`` periods is ``INVALID_WINDOW`` (the estimator would return the
    window length), never a physics ``FAIL``.
    """
    window_ok = bool(st.get("vacuum_window_ok", False))
    err = st.get("vacuum_freq_rel_error", float("nan"))
    if not window_ok:
        verdict = "INVALID_WINDOW"
    elif not (err == err):
        verdict = "FAIL"
    elif err <= tol:
        verdict = "PASS"
    else:
        verdict = "FAIL"
    return {"verdict": verdict, "pass": verdict == "PASS", "window_ok": window_ok,
            "freq_rel_error": float(err)}


# ---------------------------------------------------------------------------
def plate_verdict(pm, tol=TOL):
    """Three-way verdict for the plate added-mass row.

    ``INVALID_WINDOW`` is a distinct outcome from ``FAIL``: when the record is
    shorter than two analytic periods (or the damped-sinusoid fit rejects it),
    every period estimator collapses onto the window length, so the measured
    frequency says nothing about the physics.  The first v4 run shipped
    ``f_ratio = 1.5811`` (exactly ``T_analytic / (n_steps dt) = 600/300 = 2``
    times the analytic value scaled by ``f_vac``) and labelled it a 100 %
    physics error.
    """
    if "error" in pm:
        return {"verdict": "ERROR", "pass": False, "window_ok": False,
                "f_ratio_error": float("nan"), "detail": str(pm["error"])}
    window_ok = bool(pm.get("window_ok_2cyc", True) and pm.get("period_fit_valid", True))
    err = pm.get("f_ratio_error", float("nan"))
    if not window_ok:
        verdict = "INVALID_WINDOW"
    elif not (err == err):
        verdict = "FAIL"
    elif err <= tol:
        verdict = "PASS"
    else:
        verdict = "FAIL"
    return {"verdict": verdict, "pass": verdict == "PASS", "window_ok": window_ok,
            "f_ratio_error": float(err), "detail": ""}


def submerged_study(n_steps, dt, amplitude=0.01, lattice=(160, 120),
                    relax_mode="aitken", fixed_omega=1.0, n_elements=10):
    """Beam released in an at-rest D2Q9 box: Aitken-coupled implicit IBM."""
    t0 = time.time()
    rec = cantilever_free_vibration(n_steps=n_steps, n_elements=n_elements, dt=dt,
                                    amplitude=amplitude, fluid_rho=0.2,
                                    lattice=lattice, relax_mode=relax_mode,
                                    fixed_omega=fixed_omega)
    rec["wall_seconds"] = time.time() - t0
    period = estimate_period(rec["tip_history"], dt)
    rec["period_fft"] = period
    rec["frequency_fft"] = 1.0 / period if period == period else float("nan")
    return rec


# ---------------------------------------------------------------------------
def plate_added_mass_study(c=16, lattice=(144, 144), n_steps=900, dt=1.0,
                           sub_tol=1e-6, max_sub_iter=30):
    """Thin rigid plate, spring-mounted, 1 transverse DOF.

    Analytic added mass: m_a = ρ_f π c² (Lamb 1932, art. 71).
    """
    rho_f = 1.0
    omega_lbm = 1.9
    m_a_analytic = rho_f * math.pi * c ** 2
    m_s = m_a_analytic / 0.6
    T_target = 600.0
    k = (m_s + m_a_analytic) * (2.0 * math.pi / T_target) ** 2

    cx, cy = lattice[0] / 2.0, lattice[1] / 2.0
    positions = plate_markers((cx, cy), c, dx=1.0)

    plate = SpringMountedBody(positions, mass=m_s, stiffness=k, y0=0.2)
    fluid = LBMFluid2D(lattice, omega=omega_lbm, rho0=rho_f)
    fsi = IBMImplicitFSI(plate, fluid, dt=dt, dx=1.0, ref_length=2.0 * c,
                         sub_tol=sub_tol, max_sub_iter=max_sub_iter,
                         relax_mode="aitken")

    hist = []
    for step in range(1, n_steps + 1):
        try:
            rec = fsi.step()
            hist.append(rec)
        except (ValueError, FloatingPointError):
            break

    tips = [h["y"] for h in hist]
    # Window guard: a record shorter than one period makes every estimator below
    # collapse onto the window length (see docs/xval_implicit_ibm_aitken_v5.md).
    # The window-safe damped-sinusoid fit is the primary estimate; the legacy
    # zero-crossing / FFT path is kept only for the record.
    fit = estimate_period_damped(tips, dt)
    period_legacy = estimate_period(tips, dt)
    # No fallback to the legacy estimator: an invalid fit means the record cannot
    # support a period, and the row must say INVALID_WINDOW rather than carry the
    # window length into the cross-validation table as a "measurement".
    period = float(fit["period"]) if fit["valid"] else float("nan")
    freq = 1.0 / period if period == period else float("nan")
    f_vac = math.sqrt(k / m_s) / (2.0 * math.pi)
    f_fsi_analytic = math.sqrt(k / (m_s + m_a_analytic)) / (2.0 * math.pi)

    return {
        "c": c, "lattice": list(lattice), "n_steps": n_steps,
        "steps_completed": len(hist),
        "m_a_analytic": m_a_analytic, "m_s": m_s, "k": k,
        "f_vac": f_vac, "f_fsi_analytic": f_fsi_analytic,
        "f_measured": freq,
        "f_measured_legacy": 1.0 / period_legacy if period_legacy == period_legacy
        else float("nan"),
        "f_ratio_measured_legacy": (1.0 / period_legacy / f_vac
                                    if period_legacy == period_legacy
                                    and period_legacy > 0.0 else float("nan")),
        "period_fit": fit["period"], "period_fit_valid": fit["valid"],
        "period_legacy": period_legacy,
        "window_cycles": fit["cycles"],
        "period_reference": 1.0 / f_fsi_analytic,
        "window_ok_2cyc": period_window_ok(len(tips), dt,
                                          1.0 / f_fsi_analytic, 2.0),
        "f_ratio_measured": freq / f_vac if freq == freq else float("nan"),
        "f_ratio_analytic": f_fsi_analytic / f_vac,
        "f_ratio_error": abs(freq / f_vac - f_fsi_analytic / f_vac) / (f_fsi_analytic / f_vac)
        if freq == freq else float("nan"),
        "tip_history": tips,
        "sub_iterations": [h["sub_iterations"] for h in hist],
        "maxit_hits": fsi.n_maxit_hits,
        "converged_all": all(h["converged"] for h in hist),
    }


# ---------------------------------------------------------------------------
def main(argv=None):
    quick = "--quick" in (argv if argv is not None else sys.argv)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "runs"), exist_ok=True)

    ev = {"anchor": "XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION",
          "reference": "joconnor22/LIFE (implicit IBM + epsilon conservation + "
                       "block Gauss-Seidel with Aitken relaxation), GPLv3",
          "repo": "TensorFEM", "tolerance": TOL,
          "version": "v4",
          "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    ev["run_mode"] = "quick" if quick else "full"
    ev["kernel_epsilon"] = kernel_and_epsilon_checks()
    ev["aitken_scalar"] = aitken_scalar_study()
    ev["structure"] = structure_checks(n_steps=3000 if not quick else 1500)

    n_sub = 10 if quick else 25
    dt = 5.0e-3
    ev["submerged_aitken"] = submerged_study(n_steps=n_sub, dt=dt)
    ev["submerged_fixed_omega1"] = submerged_study(n_steps=n_sub, dt=dt,
                                                   relax_mode="fixed",
                                                   fixed_omega=1.0)

    plate_c = 10 if quick else 16
    # The analytic period of the plate case is T = 600 lattice steps (k is set to
    # that end).  Four periods make the damped-sinusoid fit well-posed; the quick
    # mode deliberately keeps the old 300-step (0.5 period) record so the
    # INVALID_WINDOW verdict stays exercised.
    plate_steps = 300 if quick else 2400
    try:
        ev["plate_added_mass"] = plate_added_mass_study(c=plate_c, n_steps=plate_steps)
    except Exception as exc:
        ev["plate_added_mass"] = {"error": str(exc)}

    # ---- cross-validation table -------------------------------------------
    ke = ev["kernel_epsilon"]
    st = ev["structure"]
    vv = vacuum_verdict(st)
    ev["vacuum_verdict"] = vv
    table = [
        {"item": "kernel partition of unity", "ref": ke["partition_ref"],
         "got": ke["partition_of_unity_error"], "kind": "absolute",
         "pass": ke["partition_of_unity_error"] <= 1e-14},
        {"item": "force transmission error, eps = 1 (sanity window only; "
                 "not an agreement test)",
         "ref": "LIFE 11-37 % (surface markers); our pass band 2-70 % is a "
                "sanity window, not agreement",
         "got": ke["force_transmission_error_eps1"], "kind": "band",
         **(fbv := force_band_verdict(ke["force_transmission_error_eps1"]))},
        {"item": "force transmission error, eps corrected",
         "ref": ke["force_transmission_ref_corrected"],
         "got": ke["force_transmission_error_corrected"], "kind": "absolute",
         "pass": ke["force_transmission_error_corrected"] <= 1e-13},
        {"item": "vacuum free-vibration f vs analytic", "ref": st["f1_analytic"],
         "got": st["vacuum_frequency"], "kind": "relative",
         "window_ok": vv["window_ok"], "verdict": vv["verdict"], "pass": vv["pass"]},
    ]

    by = {(r["gain"], r["mode"]): r for r in ev["aitken_scalar"]}
    for g in sorted({r["gain"] for r in ev["aitken_scalar"]}):
        a_row, f_row = by[(g, "aitken")], by[(g, "fixed1")]
        o_row = by[(g, "fixed_opt")]
        # The previous v4 row printed the iteration cap (200) in the ``ref``
        # column of a ``relative`` row, which reads as "200 reference iterations
        # vs 3" = a 98.5 % error that was nevertheless labelled PASS.  For
        # g > 1 the map u <- g u + b diverges under omega = 1 by construction
        # (omega = 1 is outside the stable range 0 < omega < 2/g), so this is a
        # qualitative statement, not a measurement.
        table.append({"item": f"#sub-iterations, gain {g}: Aitken converged while "
                              f"fixed omega=1 did not (qualitative)",
                      "ref": f"{f_row['iterations']} = iteration cap; omega=1 is "
                             f"outside 0 < omega < 2/g, divergence is guaranteed",
                      "got": a_row["iterations"], "kind": "qualitative",
                      "pass": bool(a_row["converged"]) and not f_row["converged"]})
        opt_err = (abs(a_row["iterations"] - o_row["iterations"])
                   / max(1.0, float(o_row["iterations"])))
        table.append({"item": f"#sub-iterations, gain {g}: Aitken vs optimal fixed "
                              f"relaxation omega*=1/(1-g)",
                      "ref": o_row["iterations"], "got": a_row["iterations"],
                      "kind": "relative",
                      "pass": bool(a_row["converged"] and o_row["converged"]
                                   and opt_err <= TOL)})

    pm = ev.get("plate_added_mass", {})
    pv = plate_verdict(pm)
    if "error" not in pm:
        table.append({"item": "plate f_fsi/f_vac vs analytic added-mass",
                      "ref": pm["f_ratio_analytic"], "got": pm["f_ratio_measured"],
                      "kind": "relative", "window_ok": pv["window_ok"],
                      "verdict": pv["verdict"], "pass": pv["pass"]})
    else:
        table.append({"item": "plate f_fsi/f_vac vs analytic added-mass",
                      "ref": "N/A", "got": pm["error"], "kind": "error",
                      "verdict": "ERROR", "pass": False})
    ev["plate_verdict"] = pv

    ev["table"] = table

    sub = ev["submerged_aitken"]
    fix = ev["submerged_fixed_omega1"]
    ev["coupling_summary"] = {
        "aitken": {"mean_sub_iterations": float(np.mean(sub["sub_iterations"])) if sub["sub_iterations"] else 0,
                   "max_sub_iterations": int(np.max(sub["sub_iterations"])) if sub["sub_iterations"] else 0,
                   "maxit_hits": sub["maxit_hits"],
                   "converged_all": sub["converged_all"]},
        "fixed_omega1": {"mean_sub_iterations": float(np.mean(fix["sub_iterations"])) if fix["sub_iterations"] else 0,
                         "max_sub_iterations": int(np.max(fix["sub_iterations"])) if fix["sub_iterations"] else 0,
                         "maxit_hits": fix["maxit_hits"],
                         "converged_all": fix["converged_all"]},
    }

    with open(os.path.join(OUT, "evidence.json"), "w") as fh:
        json.dump(ev, fh, indent=2, default=_json_default)

    _figures(ev)
    _summary_md(ev)

    print(json.dumps({k: v for k, v in ev.items()
                      if k in ("table", "kernel_epsilon", "plate_added_mass",
                               "coupling_summary")}, indent=2, default=_json_default))
    ev["status"] = {"n_checks": len(table), "n_pass": sum(1 for r in table if r["pass"]),
                    "all_pass": all(r["pass"] for r in table),
                    "open_failures": [r["item"] for r in table if not r["pass"]],
                    "out_of_ref_band": [r["item"] for r in table
                                        if r.get("verdict") == "OUT_OF_REF_BAND"],
                    "qualitative_rows": [r["item"] for r in table
                                         if r.get("kind") == "qualitative"],
                    "invalid_windows": [r["item"] for r in table
                                        if r.get("verdict") == "INVALID_WINDOW"]}
    with open(os.path.join(OUT, "evidence.json"), "w") as fh:
        json.dump(ev, fh, indent=2, default=_json_default)
    n_pass = sum(1 for r in table if r["pass"])
    print(f"\n[XVAL v4] {n_pass}/{len(table)} checks within tolerance")
    return ev


def _figures(ev):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("matplotlib unavailable:", exc)
        return

    st = ev["structure"]
    sub = ev["submerged_aitken"]
    dt = st["dt"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(np.arange(len(st["tip_history"])) * dt, st["tip_history"],
               label="vacuum (FEM only)")
    if sub["tip_history"]:
        ax[0].plot(np.arange(len(sub["tip_history"])) * dt, sub["tip_history"],
                   label="submerged (implicit IBM + Aitken)")
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("tip displacement [m]")
    ax[0].set_title("clamped beam release: tip history")
    ax[0].legend(fontsize=8)

    if sub["sub_iterations"]:
        ax[1].plot(np.arange(1, len(sub["sub_iterations"]) + 1),
                   sub["sub_iterations"], "o-", ms=3, label="Aitken")
    fix = ev["submerged_fixed_omega1"]
    if fix["sub_iterations"]:
        ax[1].plot(np.arange(1, len(fix["sub_iterations"]) + 1),
                   fix["sub_iterations"], "s--", ms=3, label="fixed omega = 1")
    ax[1].set_xlabel("time step")
    ax[1].set_ylabel("sub-iterations")
    ax[1].set_title("interface fixed-point cost")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "xval_tip_history.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    rows = [r for r in ev["aitken_scalar"] if r["mode"] == "aitken"]
    g = [r["gain"] for r in rows]
    ax.plot(g, [r["omega_star"] for r in rows], "k--o", label="analytic 1/(1-g)")
    ax.plot(g, [r["omega_final"] for r in rows], "rs", label="Aitken estimate")
    ax.set_xlabel("interface gain g")
    ax.set_ylabel("relaxation omega")
    ax.set_title("Aitken recovery of the optimal relaxation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "xval_aitken_relaxation.png"), dpi=140)
    plt.close(fig)

    pm = ev.get("plate_added_mass", {})
    if "error" not in pm and pm.get("tip_history"):
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        tips = pm["tip_history"]
        ax.plot(np.arange(len(tips)) * 1.0, tips, "b-", lw=0.8, label="plate displacement")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_xlabel("time step")
        ax.set_ylabel("y [lattice units]")
        ax.set_title(f"plate free vibration (c={pm['c']}, m_a/m_s={pm['m_a_analytic']/pm['m_s']:.2f})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "xval_plate_vibration.png"), dpi=140)
        plt.close(fig)


def _summary_md(ev):
    ke = ev["kernel_epsilon"]
    st = ev["structure"]
    pm = ev.get("plate_added_mass", {})
    pv = ev.get("plate_verdict", {})
    cs = ev["coupling_summary"]

    def fmt(x):
        if isinstance(x, (int, float)):
            return f"{x:.6g}"
        return str(x)

    lines = [
        "# XVAL — implicit IBM + Aitken relaxation (TensorFEM vs LIFE recipe) — v4",
        "",
        f"Generated: {ev['generated_at_utc']}  |  tolerance: {ev['tolerance']:.0%}",
        "",
        "## Reference side",
        "joconnor22/LIFE (GPLv3): implicit IBM direct forcing"
        " (`src/IBMNode.cpp:51-58`), epsilon conservation (`src/Objects.cpp:235-321`),",
        "block Gauss-Seidel sub-iterations with Aitken dynamic relaxation"
        " (`src/Objects.cpp:26-60,150-213`), corotational beam (`src/FEMElement.cpp:99-116`).",
        "Documented in TensorPlatform `docs/oss_inventory/FEAT-OSS-INVENTORY-LBM_LIFE.md`.",
        "",
        "## Cross-validation table",
        "",
        "| item | ref | got | kind | verdict |",
        "|---|---|---|---|---|",
    ]
    for r in ev["table"]:
        verdict = r.get("verdict", "PASS" if r["pass"] else "FAIL")
        lines.append(f"| {r['item']} | {fmt(r['ref'])} | {fmt(r['got'])} | "
                     f"{r['kind']} | {verdict} |")
    lines += [
        "",
        "`INVALID_WINDOW` = the measurement record is too short for the period",
        "estimator (see `period_window_ok`); it is not a physics result.",
        "`OUT_OF_REF_BAND` = inside our sanity window but outside LIFE's published",
        "band: a configuration difference, *not* a cross-validation pass.",
        "`qualitative` rows are convergence statements, not measured agreements.",
        "",
        f"Run mode: **{ev.get('run_mode', 'full')}** "
        f"(structure {st['n_steps']} steps, plate {pm.get('n_steps', 'n/a')} steps).",
        "",
        "## Verdict accounting",
        "",
        f"- checks: {ev['status']['n_checks']}, pass: {ev['status']['n_pass']}, "
        f"all_pass: {ev['status']['all_pass']}",
        f"- open failures: {ev['status']['open_failures']}",
        f"- out of reference band (not counted as pass): "
        f"{ev['status'].get('out_of_ref_band', [])}",
        f"- invalid windows: {ev['status']['invalid_windows']}",
        "",
        "## Coupling cost",
        "",
        "| mode | mean sub-iter | max sub-iter | maxit hits | converged all |",
        "|---|---|---|---|---|",
        f"| Aitken | {cs['aitken']['mean_sub_iterations']:.2f} | "
        f"{cs['aitken']['max_sub_iterations']} | {cs['aitken']['maxit_hits']} | "
        f"{cs['aitken']['converged_all']} |",
        f"| fixed ω=1 | {cs['fixed_omega1']['mean_sub_iterations']:.2f} | "
        f"{cs['fixed_omega1']['max_sub_iterations']} | {cs['fixed_omega1']['maxit_hits']} | "
        f"{cs['fixed_omega1']['converged_all']} |",
        "",
        "## Vacuum frequency (beam, clamped-free)",
        "",
        f"- Analytic f1 (Euler-Bernoulli, with area): {st['f1_analytic']:.6f} Hz"
        f" (T = {st['period_reference']:.4f} s)",
        f"- FEM eigen: {st['f1_fem_eigen']:.6f} Hz "
        f"(error: {st['f1_eigen_rel_error']*100:.4f} %)",
        f"- Section requested == simulated: {st['section_consistent']} "
        f"(A = {st['structural_props']['A']:.4g})",
        f"- Record: {st['n_steps']} steps x dt = {st['dt']:.4g} s = "
        f"{st['vacuum_cycles_in_record']:.2f} periods (gate {st['vacuum_min_cycles']:.1f})",
        f"- Window valid: {st['vacuum_window_ok']}",
        f"- Free-vibration f (damped-sinusoid fit): {st['vacuum_frequency']:.6f} Hz "
        f"(error: {st['vacuum_freq_rel_error']*100:.4f} %)",
        f"- Free-vibration f (legacy mean-removed zero crossings, record only): "
        f"{st['vacuum_frequency_legacy_zc']:.6f} Hz",
        "",
    ]
    if "error" not in pm:
        lines += [
            "## Plate added-mass frequency ratio",
            "",
            f"- c = {pm['c']}, lattice = {pm['lattice']}, steps = {pm['n_steps']}",
            f"- Analytic f_vac: {pm['f_vac']:.6g} Hz, f_fsi: {pm['f_fsi_analytic']:.6g} Hz",
            f"- Analytic f_fsi/f_vac: {pm['f_ratio_analytic']:.6f}",
            f"- Measured f_fsi/f_vac (damped fit, valid = {pm['period_fit_valid']}): "
            f"{fmt(pm['f_ratio_measured'])}",
            f"- Measured f_fsi/f_vac (legacy zero crossings): "
            f"{fmt(pm.get('f_ratio_measured_legacy'))}",
            f"- Error (damped fit): "
            f"{pm['f_ratio_error']*100:.2f} %" if pm['f_ratio_error'] == pm['f_ratio_error']
            else "- Error (damped fit): n/a",
            f"- Record length: {pm.get('window_cycles', float('nan')):.2f} periods "
            f"(gate 2.0), window valid: {pv.get('window_ok')}",
            f"- Verdict: **{pv.get('verdict', 'n/a')}**",
            "",
        ]
    lines += [
        "## Notes",
        "",
        "1. ε=1 force-transmission error (61 %) exceeds LIFE's 11–37 % band by 65 %",
        "   because our test uses centreline markers (spacing ≈ 1 Δ) rather than LIFE's",
        "   surface markers with exact element shape-function integration. The row is",
        "   therefore reported as OUT_OF_REF_BAND (sanity window 2–70 %) and is excluded",
        "   from the pass count: it cannot certify IBM force conservation against LIFE.",
        "2. The beam submerged frequency remains open: the beam's effective thickness in",
        "   the IBM is set by the 3-point delta support (≈ 3 Δ), not a physical thickness,",
        "   so the analytic added-mass estimate (m_a = ρ_f π b²/4) does not apply directly.",
        "   The plate geometry (v4) closes this gap with an analytic reference.",
        "3. Aitken relaxation shows no advantage over ω=1 at the tested added-mass ratios",
        "   (≤ 0.23 for the beam, 0.6 for the plate) because the interface fixed point",
        "   converges in 1 sub-iteration. The scalar-probe evidence (ω* = 1/(1−g)) stands.",
        "4. v4.1 (mission 22704) fixes two harness defects that produced the shipped",
        "   43.048 Hz / 1.5811 rows: the free-vibration run now receives the section its",
        "   analytic reference describes, and both frequency rows are gated by the record",
        "   length instead of being reported as physics failures.",
    ]
    with open(os.path.join(OUT, "xval_summary.md"), "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
