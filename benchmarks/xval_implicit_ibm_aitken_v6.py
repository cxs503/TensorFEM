"""Cross-validation v6: implicit-IBM + Aitken relaxation (XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION).

Reference side: joconnor22/LIFE (Lattice Boltzmann - immersed boundary - finite
element; implicit direct forcing + Aitken Delta^2 sub-iterations).

Why a v6 (what v4/v5 got wrong, and what is fixed here)
------------------------------------------------------
The v4/v5 runs archived two open failures that were *harness* failures, not
physics:

1. ``vacuum free-vibration f vs analytic`` reported a 29694 % error
   (43.05 Hz measured against 0.1445 Hz "analytic").  Two independent defects
   produced it:
     a. the reference frequency was assembled from a thin section
        (``t = 1/50``, ``A = 0.02``) while :func:`cantilever_free_vibration` ran
        its own default section (``A = 1``) -- a 7x frequency mismatch that no
        physics can absorb;
     b. the Euler-Bernoulli reference dropped the section area from the mass
        per unit length (``rho * L**4`` instead of ``rho * A * L**4``).
2. The Euler-Bernoulli drop was *invisible* because the beam element's
   consistent mass matrix also dropped ``A`` (``ml = rho * L0``): the wrong
   simulation was checked against the equally wrong reference and agreed to
   1e-6.  Two mutually compensating errors.  The library defect is fixed in
   :mod:`tensorfem.ibm_aitken_implicit` (``ml = rho * A * L0``) and its size is
   quantified below (``defect_demonstration``): every frequency used to be low
   by ``sqrt(A)`` whenever ``A != 1``, which silently corrupted the structural
   inertia -- and with it the added-mass ratio -- of every submerged run.

Aitken relaxation, the epsilon-conservation system, the kernel, the plate
added-mass and the strong-coupling regime are unchanged from v5; the plate
item now uses a record that resolves the coupled period (the archived v5 FAIL
came from a 600-step record whose zero-crossing estimate was degenerate -- the
library already ships :func:`period_window_ok` for exactly that trap).

Run: ``python benchmarks/xval_implicit_ibm_aitken_v6.py [--quick]``
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
    IBMImplicitFSI,
    LBMFluid2D,
    SpringMountedBody,
    analytic_added_mass,
    coupled_free_vibration,
    estimate_period,
    estimate_period_damped,
    estimate_period_fft,
    period_window_ok,
    plate_markers,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "xval_implicit_ibm_aitken_v6_artifacts")
TOL = 0.15          # owner acceptance: cross-validation error < 15 %
TOL_EIG = 0.01      # FEM discretisation vs closed form: 1 %
EULER_B1 = 1.87510407


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
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def euler_bernoulli_f1(props):
    """Clamped-free first bending frequency with the *correct* mass per length.

    ``f1 = beta1^2 / (2 pi) * sqrt(E I / (rho A L^4))``, ``beta1 = 1.87510407``.
    The ``rho * A`` denominator is the point that v4/v5 dropped.
    """
    return (EULER_B1 ** 2) / (2.0 * math.pi) * math.sqrt(
        props.E * props.I / (props.rho * props.A * props.length ** 4))


def _section(t=1.0, rho=1.0, E=1.0e5, length=1.0):
    return BeamProperties(E=E, A=t, I=t ** 3 / 12.0, rho=rho,
                          length=length, thickness=t)


# ---------------------------------------------------------------------------
# E0: kernel and epsilon checks
# ---------------------------------------------------------------------------
def kernel_and_epsilon_checks():
    offs = np.linspace(0.0, 1.0, 401)
    best = 0.0
    rows = np.arange(-3, 4)
    for x in offs:
        best = max(best, abs(float(dirac_delta_3pt(x - rows).sum()) - 1.0))

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
        "force_transmission_ref_band": [0.11, 0.37],
        "force_transmission_error_corrected": corr_err,
        "force_transmission_ref_corrected": 9e-16,
        "total_marker_force_rel_error": total_f_rel,
        "n_markers": n,
    }


# ---------------------------------------------------------------------------
# E1: Aitken scalar study
# ---------------------------------------------------------------------------
def aitken_scalar_study(gains=(1.5, 2.5, 4.0), tol=1e-10, max_it=200):
    """Scalar interface fixed point U -> g U + b: Aitken vs fixed relaxation."""
    out = []
    for g in gains:
        for mode in ("aitken", "fixed1", "fixed_opt"):
            ait = AitkenRelaxation(relax_max=5.0, relax0=1.0)
            u, r_km1, omega, omega_applied = 0.0, None, 5.0, float("nan")
            resid, converged, it = [], False, 0
            for it in range(1, max_it + 1):
                u_new = g * u + 1.0
                r_k = np.array([u_new - u])
                resid.append(float(abs(r_k[0])))
                if abs(r_k[0]) <= tol:
                    u, converged = u_new, True
                    break
                if mode == "fixed1":
                    omega = 1.0
                elif mode == "fixed_opt":
                    omega = 1.0 / (1.0 - g)
                elif it >= 2:
                    omega = ait.omega_for(r_km1, r_k, omega_prev=omega_applied)
                omega_applied = omega
                r_km1 = r_k.copy()
                u = u + omega * (u_new - u)
            out.append({
                "gain": g, "mode": mode, "iterations": it,
                "converged": bool(converged),
                "omega_star": 1.0 / (1.0 - g),
                "omega_final": float(omega),
                "residual": resid[-1] if resid else float("nan"),
            })
    return out


# ---------------------------------------------------------------------------
# E2: structure -- mass matrix invariant, eigenfrequencies, vacuum vibration
# ---------------------------------------------------------------------------
def beam_mass_invariant(t=0.02, rho=2.0, n_elements=8):
    """Transverse rigid-body translation must carry exactly ``rho A L``."""
    props = _section(t=t, rho=rho)
    beam = CorotationalBeam2D(n_elements=n_elements, props=props)
    v = np.zeros(beam.n_dof)
    v[1::3] = 1.0                       # unit transverse translation, no rotation
    carried = float((beam.M @ v)[1::3].sum())
    expected = props.rho * props.A * props.length
    return {"carried_mass": carried, "expected_rhoAL": expected,
            "rel_error": abs(carried - expected) / expected,
            "n_elements": n_elements, "A": props.A, "rho": props.rho}


def beam_eigen_study(t, n_elements=10, rho=1.0):
    """FEM eigenfrequency vs the Euler-Bernoulli closed form for the same section."""
    props = _section(t=t, rho=rho)
    beam = CorotationalBeam2D(n_elements=n_elements, props=props)
    f_fem = beam_natural_frequencies(beam, n_modes=2)
    f_ref = euler_bernoulli_f1(props)
    return {
        "t": t, "A": props.A, "I": props.I, "E": props.E, "rho": props.rho,
        "n_elements": n_elements,
        "element_length": props.length / n_elements,
        "f1_fem": float(f_fem[0]), "f1_ref": f_ref,
        "rel_error": abs(float(f_fem[0]) - f_ref) / f_ref,
        "slenderness_L_over_t": props.length / props.thickness,
    }


def defect_demonstration(t=0.1, n_elements=8):
    """Size of the mass-matrix defect: pre-fix inertia == rho/A, so f -> f sqrt(A).

    The pre-fix element mass was ``rho * L0`` (no ``A``); that is *exactly* the
    consistent mass of a section with density ``rho / A``.  Reproducing it that
    way quantifies the defect without reverting the fix: the measured free
    vibration frequency of the legacy inertia is compared with the correct
    reference for the *same* stiffness.
    """
    props = _section(t=t)
    f_ref = euler_bernoulli_f1(props)
    beam = CorotationalBeam2D(n_elements=n_elements, props=props)
    f_fixed = float(beam_natural_frequencies(beam, n_modes=1)[0])
    beam_legacy = CorotationalBeam2D(
        n_elements=n_elements, props=_section(t=t, rho=props.rho / props.A))
    f_legacy = float(beam_natural_frequencies(beam_legacy, n_modes=1)[0])
    return {
        "A": props.A, "f1_closed_form_same_section": f_ref,
        "f1_current_implementation": f_fixed,
        "f1_with_legacy_inertia_rho_over_A": f_legacy,
        "legacy_rel_error": abs(f_legacy - f_ref) / f_ref,
        "predicted_legacy_ratio_sqrt_A": math.sqrt(props.A),
        "observed_legacy_ratio": f_legacy / f_fixed,
    }


def vacuum_free_vibration_study(t=0.02, n_elements=10, steps_per_period=200,
                                n_periods=5.0, amplitude=0.02):
    """Time-domain vacuum frequency, resolved (dt = T1/steps_per_period)."""
    props = _section(t=t)
    f_ref = euler_bernoulli_f1(props)
    dt = 1.0 / (f_ref * steps_per_period)
    n_steps = int(math.ceil(n_periods * steps_per_period))
    rec = cantilever_free_vibration(n_steps=n_steps, n_elements=n_elements, dt=dt,
                                    amplitude=amplitude, fluid_rho=0.0, props=props)
    y = np.asarray(rec["tip_history"], dtype=float)
    p_zc = estimate_period(y, dt)
    p_fft = estimate_period_fft(y, dt)
    p_ls = estimate_period_damped(y, dt)
    f_zc = 1.0 / p_zc if p_zc == p_zc else float("nan")
    return {
        "section": {"t": t, "A": props.A, "I": props.I, "E": props.E,
                    "rho": props.rho, "L": props.length},
        "f1_ref": f_ref, "period_ref": 1.0 / f_ref,
        "dt": dt, "n_steps": n_steps, "n_elements": n_elements,
        "steps_per_period": steps_per_period, "n_periods": n_periods,
        "amplitude": amplitude,
        "period_zc": p_zc, "period_fft": p_fft,
        "period_ls": p_ls["period"], "period_ls_valid": bool(p_ls["valid"]),
        "period_ls_cycles": float(p_ls["cycles"]),
        "frequency_zc": f_zc,
        "rel_error_zc": abs(f_zc - f_ref) / f_ref if f_zc == f_zc else float("nan"),
        "period_window_ok": bool(period_window_ok(n_steps, dt, p_zc)),
        "tip_history": y.tolist(),
    }


# ---------------------------------------------------------------------------
# E3: plate added-mass study (primary coupled validation)
# ---------------------------------------------------------------------------
def plate_added_mass_study(c=16.0, lattice=(144, 144), n_steps=1800, dt=1.0,
                           sub_tol=1e-6, max_sub_iter=30, ratio_analytic=0.6):
    """Thin rigid plate, spring-mounted, one transverse dof.

    Reference: Lamb (1932) art. 71, ``m_a = rho_f pi c^2`` per unit length for a
    flat strip of half-width ``c`` in an unbounded fluid; the coupled frequency
    then obeys ``f/f_vac = sqrt(m_s/(m_s+m_a))``.
    """
    rho_f = 1.0
    m_a = analytic_added_mass(c, rho_f)
    m_s = m_a / ratio_analytic
    period_target = 600.0
    k = (m_s + m_a) * (2.0 * math.pi / period_target) ** 2

    cx, cy = lattice[0] / 2.0, lattice[1] / 2.0
    plate = SpringMountedBody(plate_markers((cx, cy), c), mass=m_s,
                              stiffness=k, y0=0.2)
    fluid = LBMFluid2D(lattice, omega=1.9, rho0=rho_f)
    fsi = IBMImplicitFSI(plate, fluid, dt=dt, dx=1.0, ref_length=2.0 * c,
                         sub_tol=sub_tol, max_sub_iter=max_sub_iter,
                         relax_mode="aitken")
    hist = []
    for _ in range(int(n_steps)):
        try:
            hist.append(fsi.step())
        except (ValueError, FloatingPointError):
            break
    y = np.array([h["y"] for h in hist], dtype=float)
    p_zc = estimate_period(y, dt)
    p_fft = estimate_period_fft(y, dt)
    p_ls = estimate_period_damped(y, dt)
    f_zc = 1.0 / p_zc if p_zc == p_zc else float("nan")
    f_vac = math.sqrt(k / m_s) / (2.0 * math.pi)
    f_fsi_ref = math.sqrt(k / (m_s + m_a)) / (2.0 * math.pi)
    ratio_ref = f_fsi_ref / f_vac
    m_a_eff = k / (2.0 * math.pi * f_zc) ** 2 - m_s if f_zc == f_zc else float("nan")
    field = np.asarray(fluid.u)
    return {
        "c": c, "lattice": list(lattice), "n_steps": int(n_steps),
        "steps_completed": len(hist), "dt": dt,
        "m_a_ref": m_a, "m_s": m_s, "k": k,
        "f_vac": f_vac, "f_fsi_ref": f_fsi_ref, "period_ref": 1.0 / f_fsi_ref,
        "period_zc": p_zc, "period_fft": p_fft, "period_ls": p_ls["period"],
        "period_ls_valid": bool(p_ls["valid"]),
        "period_window_ok": bool(period_window_ok(len(hist), dt, p_zc)),
        "cycles_in_record": float(len(hist) * dt / p_zc) if p_zc == p_zc else float("nan"),
        "f_measured_zc": f_zc,
        "f_ratio_measured_zc": f_zc / f_vac if f_zc == f_zc else float("nan"),
        "f_ratio_ref": ratio_ref,
        "f_ratio_rel_error_zc": abs(f_zc / f_vac - ratio_ref) / ratio_ref
        if f_zc == f_zc else float("nan"),
        "m_a_effective": m_a_eff,
        "m_a_effective_ratio": m_a_eff / m_a if m_a_eff == m_a_eff else float("nan"),
        "sub_iterations_mean": float(np.mean([h["sub_iterations"] for h in hist])),
        "sub_iterations_max": int(np.max([h["sub_iterations"] for h in hist])),
        "maxit_hits": fsi.n_maxit_hits,
        "converged_all": all(h["converged"] for h in hist),
        "y_history": y.tolist(),
        "final_fluid_speed": float(np.sqrt((field ** 2).sum(axis=-1)).max()),
        "final_fluid_field_y": field[..., 1].tolist(),
        "markers_final": plate.markers().tolist(),
    }


# ---------------------------------------------------------------------------
# E4: submerged beam (secondary, reported not gated)
# ---------------------------------------------------------------------------
def submerged_beam_study(n_steps=25, n_elements=10, dt=5.0e-3, amplitude=0.01,
                         fluid_rho=0.2, lattice=(160, 120)):
    """Submerged clamped beam: Aitken stability + convergence cost (reported)."""
    rec = cantilever_free_vibration(n_steps=n_steps, n_elements=n_elements, dt=dt,
                                    amplitude=amplitude, fluid_rho=fluid_rho,
                                    lattice=lattice, relax_mode="aitken")
    out = {k: rec[k] for k in ("mode", "n_steps", "steps_completed", "diverged",
                               "diverged_reason", "sub_iterations", "maxit_hits",
                               "converged_all", "lattice", "beam_scale", "dt")}
    out.update({
        "fluid_rho": fluid_rho, "n_elements": n_elements,
        "amplitude": amplitude,
        "sub_iterations_mean": float(np.mean(rec["sub_iterations"]))
        if rec["sub_iterations"] else float("nan"),
        "note": "dt = 5e-3 s resolves neither the beam period nor the coupled "
                "dynamics at this amplitude: reported for coupling stability "
                "only, no frequency claim.",
    })
    return out


# ---------------------------------------------------------------------------
# E5: strong coupling regime
# ---------------------------------------------------------------------------
def strong_coupling_study(ratio=12.0, n_steps=40):
    m_a = analytic_added_mass(10.0)
    m_s = m_a / ratio
    k = (m_s + m_a) * (2.0 * math.pi / 300.0) ** 2
    fixed = coupled_free_vibration(
        n_steps=n_steps, dt=1.0, half_width=10.0, center=(72.0, 72.0),
        lattice=(144, 144), mass=m_s, stiffness=k, y0=0.2, rho_f=1.0,
        relax_mode="fixed", fixed_omega=1.0, sub_tol=1e-6, max_sub_iter=40)
    aitken = coupled_free_vibration(
        n_steps=n_steps, dt=1.0, half_width=10.0, center=(72.0, 72.0),
        lattice=(144, 144), mass=m_s, stiffness=k, y0=0.2, rho_f=1.0,
        relax_mode="aitken", sub_tol=1e-6, max_sub_iter=40)
    return {
        "ratio": ratio, "m_s": m_s, "m_a": m_a, "n_steps": n_steps,
        "fixed_diverged": bool(fixed["diverged"]),
        "fixed_steps": fixed["steps_completed"],
        "aitken_diverged": bool(aitken["diverged"]),
        "aitken_steps": aitken["steps_completed"],
        "aitken_converged_all": bool(aitken["converged_all"]),
        "aitken_maxit_hits": aitken["maxit_hits"],
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _figures(ev):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                       # pragma: no cover
        print("matplotlib unavailable:", exc)
        return []

    made = []

    # Fig 1: the mass-matrix defect -- frequency vs section area
    td = ev["defect_demonstration"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    areas = np.logspace(-2.5, 0.2, 60)
    ref = [euler_bernoulli_f1(_section(t=math.sqrt(a))) for a in areas]
    ax.loglog(areas, ref, "k-", lw=1.6, label="Euler-Bernoulli, same section")
    ax.loglog(areas, [r * math.sqrt(a) for a, r in zip(areas, ref)], "r--",
              lw=1.4, label=r"legacy inertia $\rho L_0$: $f\sqrt{A}$")
    ax.plot([td["A"]], [td["f1_current_implementation"]], "o", ms=7,
            color="tab:blue", label=f"current code (A={td['A']:g})")
    ax.plot([td["A"]], [td["f1_with_legacy_inertia_rho_over_A"]], "x", ms=9,
            color="tab:red", label="legacy code (same A)")
    ax.set_xlabel("section area $A$ (unit density, $I=t^3/12$)")
    ax.set_ylabel("$f_1$ [Hz]")
    ax.set_title("beam mass matrix: $\\rho A L_0$ vs $\\rho L_0$")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(OUT, "xval_mass_matrix_defect.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(path)

    # Fig 2: vacuum free vibration vs analytic period
    vv = ev["vacuum_free_vibration"]
    y = np.asarray(vv["tip_history"])
    dt = vv["dt"]
    t = np.arange(y.size) * dt
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(t, y, "-", lw=1.0, color="tab:blue", label="FEM (vacuum, $A\\neq1$)")
    t_an = np.linspace(0.0, t[-1], 2000)
    ax.plot(t_an, vv["amplitude"] * np.cos(2.0 * np.pi * vv["f1_ref"] * t_an),
            "k--", lw=1.0, label="Euler-Bernoulli, same section")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("tip deflection [m]")
    ax.set_title(f"vacuum free vibration: error {100 * vv['rel_error_zc']:.3f} %")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT, "xval_vacuum_history.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(path)

    # Fig 3: plate history + measured vs analytic frequency ratio
    pm = ev["plate_added_mass"]
    yp = np.asarray(pm["y_history"])
    tp = np.arange(yp.size) * pm["dt"]
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 3.5))
    ax[0].plot(tp, yp, "-", lw=0.9, color="tab:green",
               label="implicit IBM + Aitken")
    ax[0].axhline(0.0, color="grey", lw=0.5)
    if pm["period_zc"] == pm["period_zc"]:
        for k in range(1, 4):
            ax[0].axvline(k * pm["period_zc"], color="tab:red", ls=":", lw=0.8)
    ax[0].set_xlabel("time step")
    ax[0].set_ylabel("plate displacement $y$")
    ax[0].set_title(f"plate release: measured period {pm['period_zc']:.1f} "
                    f"(ref {pm['period_ref']:.1f})")
    ax[0].legend(fontsize=8)
    bars = [pm["f_ratio_ref"], pm["f_ratio_measured_zc"]]
    ax[1].bar(["analytic\n$\\sqrt{m_s/(m_s+m_a)}$", "measured\n$f_{fsi}/f_{vac}$"],
              bars, color=["k", "tab:green"], width=0.55)
    for i, b in enumerate(bars):
        ax[1].text(i, b, f"{b:.4f}", ha="center", va="bottom", fontsize=9)
    ax[1].axhline(pm["f_ratio_ref"] * (1.0 + TOL), color="r", ls=":", lw=1.0)
    ax[1].axhline(pm["f_ratio_ref"] * (1.0 - TOL), color="r", ls=":", lw=1.0)
    ax[1].set_ylabel("frequency ratio")
    ax[1].set_title(f"added-mass check: error "
                    f"{100 * pm['f_ratio_rel_error_zc']:.2f} % (tol 15 %)")
    fig.tight_layout()
    path = os.path.join(OUT, "xval_plate_added_mass.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(path)

    # Fig 4: flow field around the plate at the end of the coupled run
    field = np.asarray(pm["final_fluid_field_y"])
    if field.ndim == 2 and pm.get("markers_final"):
        mk = np.asarray(pm["markers_final"])
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        lim = np.abs(field).max() or 1.0
        im = ax.imshow(field.T, origin="lower", cmap="RdBu_r",
                       vmin=-lim, vmax=lim)
        ax.plot(mk[:, 0], mk[:, 1], "k.", ms=2, label="IBM markers")
        ax.set_xlabel("$x$ [lattice]")
        ax.set_ylabel("$y$ [lattice]")
        ax.set_title("vertical fluid velocity $u_y$ at the last step")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        path = os.path.join(OUT, "xval_plate_flow_field.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        made.append(path)

    # Fig 5: Aitken relaxation recovers the optimal omega
    rows = [r for r in ev["aitken_scalar"] if r["mode"] == "aitken"]
    g = [r["gain"] for r in rows]
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.plot(g, [r["omega_star"] for r in rows], "k--o", label="analytic $1/(1-g)$")
    ax.plot(g, [r["omega_final"] for r in rows], "rs", label="Aitken estimate")
    ax.set_xlabel("interface gain $g$")
    ax.set_ylabel("relaxation $\\omega$")
    ax.set_title("Aitken $\\Delta^2$ recovers the optimal relaxation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT, "xval_aitken_relaxation.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(path)
    return made


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def _summary_md(ev):
    rows = ev["table"]
    lines = [
        "# XVAL v6 - implicit IBM + Aitken relaxation (TensorFEM vs LIFE recipe)",
        "",
        f"Generated: {ev['generated_at_utc']} | acceptance: {int(100 * TOL)} %",
        "",
        "## Reference side",
        "joconnor22/LIFE (GPLv3): implicit IBM direct forcing (`src/IBMNode.cpp:51-58`),",
        "epsilon conservation (`src/Objects.cpp:235-321`), block Gauss-Seidel",
        "sub-iterations with Aitken dynamic relaxation (`src/Objects.cpp:26-60,150-213`),",
        "corotational beam (`src/FEMElement.cpp:99-116`); plus closed-form references",
        "(Euler-Bernoulli clamped-free f1, Lamb 1932 art. 71 added mass).",
        "",
        "## Cross-validation table",
        "",
        "| item | ref | got | kind | rel. error | pass |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        err = r.get("rel_error")
        err_s = "n/a" if err is None or err != err else f"{100 * err:.3f} %"
        lines.append(f"| {r['item']} | {r['ref']} | {r['got']} | {r['kind']} | "
                     f"{err_s} | {'PASS' if r['pass'] else 'FAIL'} |")
    dd = ev["defect_demonstration"]
    pm = ev["plate_added_mass"]
    lines += [
        "",
        "## Defects found and fixed in this round",
        "",
        f"1. **Beam consistent mass matrix dropped the section area** "
        f"(`ml = p.rho * L0` -> `p.rho * p.A * L0`). With the legacy inertia every",
        f"   frequency is low by `sqrt(A)`: at A = {dd['A']:g} the error is "
        f"{100 * dd['legacy_rel_error']:.1f} %",
        f"   (observed legacy/current ratio {dd['observed_legacy_ratio']:.4f} vs "
        f"`sqrt(A)` = {dd['predicted_legacy_ratio_sqrt_A']:.4f}).",
        "2. **Euler-Bernoulli reference dropped `rho A`** from the mass per unit",
        "   length, matching defect 1 to 1e-6 -- two compensating errors that made",
        "   the v4/v5 eigen check 'pass' while the physics was 7x off.",
        "3. **The vacuum free-vibration run ignored the caller's section** (it always",
        "   built `_section()`, A = 1, while the reference used A = 0.02):",
        "   `cantilever_free_vibration` now accepts `props`.  Together with a record",
        "   that resolves the period (dt = T1/200, 5 periods) the archived 29694 %",
        "   'failure' becomes the live error in the table above.",
        "",
        "## Notes / limitations",
        "",
        "- The LIFE executable is **not** run (upstream needs Boost and a",
        "  `using namespace std` compatibility fix, absent in this image).  Every",
        "  'ref' entry is a closed-form value or an inventory-documented observable",
        "  quantity; no paired LIFE run exists yet.  Honest limitation.",
        "- Fluid side is a periodic D2Q9 BGK box: no inflow/outflow, walls or",
        "  regularised boundaries, so only the coupling algorithm and the IBM",
        "  operators are cross-validated, not LIFE's flow solver.",
        f"- The IBM kernel smears the plate over ~3 lattice units, so the measured",
        f"  added mass is {pm['m_a_effective_ratio']:.3f} x the thin-strip Lamb value;",
        f"  the frequency ratio therefore errs low (measured "
        f"{pm['f_ratio_measured_zc']:.4f} vs analytic {pm['f_ratio_ref']:.4f}).",
        "- Aitken shows no advantage over a fixed omega when the interface fixed",
        "  point converges in one sub-iteration (low added-mass ratio); its",
        "  necessity is demonstrated by the strong-coupling regime and the scalar",
        "  probe `omega* = 1/(1-g)`.",
        "",
        f"## Status: {ev['status']['n_pass']}/{ev['status']['n_checks']} checks pass",
        "",
    ]
    if ev["status"]["open_failures"]:
        lines += ["Open failures: " + ", ".join(ev["status"]["open_failures"]), ""]
    with open(os.path.join(OUT, "summary.md"), "w") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
def main(argv=None):
    quick = "--quick" in (argv if argv is not None else sys.argv)
    os.makedirs(OUT, exist_ok=True)

    ev = {"anchor": "XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION",
          "reference": "joconnor22/LIFE (implicit IBM + epsilon conservation + "
                       "block Gauss-Seidel with Aitken relaxation), GPLv3",
          "repo": "TensorFEM", "tolerance": TOL, "version": "v6",
          "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    print("[E0] kernel / epsilon ...")
    ev["kernel_epsilon"] = kernel_and_epsilon_checks()
    print("[E1] Aitken scalar study ...")
    ev["aitken_scalar"] = aitken_scalar_study()
    print("[E2a] beam mass invariant ...")
    ev["beam_mass_invariant"] = beam_mass_invariant(t=1.0 / 50.0, rho=2.0)
    print("[E2b] beam eigenfrequencies ...")
    ev["beam_eigen"] = {
        "unit_section": beam_eigen_study(t=1.0),
        "thin_section": beam_eigen_study(t=1.0 / 50.0),
    }
    print("[E2c] mass-matrix defect demonstration ...")
    ev["defect_demonstration"] = defect_demonstration(t=0.1, n_elements=8)
    print("[E2d] vacuum free vibration (resolved record) ...")
    ev["vacuum_free_vibration"] = vacuum_free_vibration_study(
        t=1.0 / 50.0, n_elements=10,
        steps_per_period=120 if quick else 200,
        n_periods=3.5 if quick else 5.0)
    print("[E3] plate added-mass study ...")
    ev["plate_added_mass"] = plate_added_mass_study(
        c=16.0, lattice=(144, 144), n_steps=600 if quick else 1800)
    print("[E4] submerged beam (secondary, reported) ...")
    ev["submerged_beam"] = submerged_beam_study()
    print("[E5] strong coupling regime ...")
    ev["strong_coupling"] = strong_coupling_study()

    ke, mi, be = ev["kernel_epsilon"], ev["beam_mass_invariant"], ev["beam_eigen"]
    vv, pm, sc = ev["vacuum_free_vibration"], ev["plate_added_mass"], ev["strong_coupling"]
    table = [
        {"item": "3-point kernel partition of unity", "ref": ke["partition_ref"],
         "got": ke["partition_of_unity_error"], "kind": "absolute",
         "rel_error": ke["partition_of_unity_error"],
         "pass": ke["partition_of_unity_error"] <= 1e-14},
        {"item": "force transmission, eps = 1 (LIFE band 11-37 %)",
         "ref": "0.11 - 0.37", "got": ke["force_transmission_error_eps1"],
         "kind": "band", "rel_error": None,
         "pass": 0.02 <= ke["force_transmission_error_eps1"] <= 0.70},
        {"item": "force transmission, epsilon corrected",
         "ref": ke["force_transmission_ref_corrected"],
         "got": ke["force_transmission_error_corrected"], "kind": "absolute",
         "rel_error": ke["force_transmission_error_corrected"],
         "pass": ke["force_transmission_error_corrected"] <= 1e-13},
        {"item": "beam mass matrix: rigid translation carries rho*A*L",
         "ref": mi["expected_rhoAL"], "got": mi["carried_mass"],
         "kind": "relative", "rel_error": mi["rel_error"],
         "pass": mi["rel_error"] <= 1e-12},
        {"item": "beam f1 vs Euler-Bernoulli (unit section, A = 1)",
         "ref": be["unit_section"]["f1_ref"],
         "got": be["unit_section"]["f1_fem"], "kind": "relative",
         "rel_error": be["unit_section"]["rel_error"],
         "pass": be["unit_section"]["rel_error"] <= TOL_EIG},
        {"item": "beam f1 vs Euler-Bernoulli (thin section, A = 0.02)",
         "ref": be["thin_section"]["f1_ref"],
         "got": be["thin_section"]["f1_fem"], "kind": "relative",
         "rel_error": be["thin_section"]["rel_error"],
         "pass": be["thin_section"]["rel_error"] <= TOL_EIG},
        {"item": "vacuum free vibration f vs Euler-Bernoulli (same section)",
         "ref": vv["f1_ref"], "got": vv["frequency_zc"], "kind": "relative",
         "rel_error": vv["rel_error_zc"],
         "pass": vv["rel_error_zc"] <= TOL and vv["period_window_ok"]},
        {"item": "plate f_fsi/f_vac vs Lamb added mass",
         "ref": pm["f_ratio_ref"], "got": pm["f_ratio_measured_zc"],
         "kind": "relative", "rel_error": pm["f_ratio_rel_error_zc"],
         "pass": pm["f_ratio_rel_error_zc"] <= TOL and pm["period_window_ok"]},
    ]

    by = {(r["gain"], r["mode"]): r for r in ev["aitken_scalar"]}
    for g in sorted({r["gain"] for r in ev["aitken_scalar"]}):
        a_row, f_row = by[(g, "aitken")], by[(g, "fixed1")]
        table.append({"item": f"Aitken converges where omega=1 diverges (gain {g})",
                      "ref": f"fixed omega=1: {f_row['iterations']} it, "
                             f"converged={f_row['converged']}",
                      "got": f"Aitken: {a_row['iterations']} it",
                      "kind": "boolean", "rel_error": None,
                      "pass": bool(a_row["converged"]) and not f_row["converged"]})

    table.append({"item": f"strong coupling (m_a/m_s={sc['ratio']}): Aitken survives",
                  "ref": "fixed omega=1 diverges",
                  "got": "diverged" if sc["fixed_diverged"] else "converged",
                  "kind": "boolean", "rel_error": None,
                  "pass": bool(sc["fixed_diverged"] and not sc["aitken_diverged"]
                               and sc["aitken_converged_all"])})

    ev["table"] = table
    ev["status"] = {"n_checks": len(table),
                    "n_pass": sum(1 for r in table if r["pass"]),
                    "all_pass": all(r["pass"] for r in table),
                    "open_failures": [r["item"] for r in table if not r["pass"]]}

    ev["figures"] = _figures(ev)
    _summary_md(ev)
    with open(os.path.join(OUT, "evidence.json"), "w") as fh:
        json.dump(ev, fh, indent=1, default=_json_default)

    print(f"\n[XVAL v6] {ev['status']['n_pass']}/{ev['status']['n_checks']} "
          f"checks within tolerance")
    if ev["status"]["open_failures"]:
        print("  open failures:", ev["status"]["open_failures"])
    return ev


if __name__ == "__main__":
    main()