"""Harmonic (frequency-sweep) identification of IBM added mass / damping.

Anchor ``XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION`` (joconnor22/LIFE implicit-IBM
+ Aitken recipe).  This module repairs the *identification* half of the anchor:
:func:`tensorfem.ibm_aitken_added_mass.forced_oscillation` fits the interface
force onto the basis ``{y'', y', y, 1}`` at a **single** drive frequency.  For a
prescribed harmonic motion that basis is rank deficient:

    y = A sin(wt)   =>   y'' = -w^2 y   exactly,

so the ``y''`` and ``y`` columns are collinear to machine precision and the
least-squares problem has no unique solution.  ``numpy.linalg.lstsq`` returns the
minimum-norm solution, which puts the whole in-phase force into the ``y`` column:
the archived record ``benchmarks/xval_added_mass_artifacts/evidence.json`` shows
``added_mass_fit = -0.0`` with ``fit_r2 = 1.0`` against a reference of 314.16.
That ``-0.0`` is a rank artefact, not a measurement -- it is independent of the
physics, of the solver quality and of the mesh.

What is identifiable at one frequency
-------------------------------------
For a rigid plate whose motion is dictated, the hydrodynamic force is

    F(t) = -m_a y'' - c_a y' - k_a y  (+ const)

with ``y = A sin(wt)`` this is

    F(t) = (m_a A w^2 - k_a A) sin(wt) - c_a A w cos(wt).

A single record therefore yields one in-phase number
``K = A (m_a w^2 - k_a)`` and one quadrature number ``C = -c_a A w``:
the **apparent mass**

    m_app(w) = K / (A w^2) = m_a - k_a / w^2

conflates ``m_a`` and ``k_a``.  ``m_a`` and ``k_a`` separate only across
frequencies: ``K/A`` is linear in ``w^2`` with slope ``m_a`` and intercept
``-k_a``.  :func:`fit_added_mass_sweep` performs that regression and refuses
(``identifiable = False``) when the frequency span is too narrow.

Why this matters for the anchor
-------------------------------
A free-decay test measures one frequency too, and the map from that frequency to
the mass attribute is *amplified* by

    d m_a / m_a = -2 (1 + r) / r * (d f / f),     r = m_a / m_s,

i.e. by 5.3x at ``r = 0.6`` and 2.8x at ``r = 2.5`` (see :func:`added_mass_gain` and the exact
finite-error form :func:`amplify_frequency_error`).  A 12 % frequency error therefore *must* appear as a
64-78 % mass error at low mass ratio and as a ~30 % error of the opposite sign at
high mass ratio -- which is exactly the sign-changing "unexplained bias" of
``docs/xval_implicit_ibm_aitken_v5.md`` §5.3.  The mass attribute recovered from
free decay is not an independent measurement; the frequency sweep here is.

Public API
----------
:func:`harmonic_projection`, :func:`apparent_added_mass`,
:func:`fit_added_mass_sweep`, :func:`added_mass_gain`,
:func:`required_frequency_accuracy`, :func:`measure_added_mass_sweep`.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "harmonic_projection",
    "apparent_added_mass",
    "fit_added_mass_sweep",
    "added_mass_gain",
    "amplify_frequency_error",
    "required_frequency_accuracy",
    "measure_added_mass_sweep",
    "legacy_basis_condition",
    "legacy_single_frequency_fit",
]


# ---------------------------------------------------------------------------
# signal identification
# ---------------------------------------------------------------------------
def harmonic_projection(t, series, omega, skip_cycles=2.0):
    """Project ``series`` onto ``{sin(omega t), cos(omega t), 1}``.

    Parameters
    ----------
    t : array_like
        Sample times (uniform, ``dt`` = ``t[1] - t[0]``).
    series : array_like
        Signal to analyse.
    omega : float
        Drive angular frequency (rad per unit time).
    skip_cycles : float
        Number of drive periods discarded from the front of the record
        (start-up transient).  If the record is too short for the requested
        skip, the first half of the record is dropped instead and
        ``valid = False`` is reported only when fewer than one period is left.

    Returns
    -------
    dict with ``in_phase`` (coefficient of ``sin``), ``quadrature``
    (coefficient of ``cos``), ``offset``, ``r2``, ``n_samples``,
    ``periods_in_window`` and ``valid``.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(series, dtype=float)
    if t.size != y.size or t.size < 8:
        raise ValueError("harmonic_projection needs >= 8 samples")
    n_skip = int(round(skip_cycles * 2.0 * math.pi / float(omega) /
                       float(t[1] - t[0])))
    if n_skip >= t.size - 4:
        n_skip = t.size // 2
    ts, ys = t[n_skip:], y[n_skip:]
    basis = np.stack([np.sin(omega * ts), np.cos(omega * ts),
                      np.ones_like(ts)], axis=1)
    coef, *_ = np.linalg.lstsq(basis, ys, rcond=None)
    resid = ys - basis @ coef
    denom = float(np.sum((ys - ys.mean()) ** 2)) or 1.0
    window = float(ts[-1] - ts[0]) * float(omega) / (2.0 * math.pi)
    return {
        "in_phase": float(coef[0]),
        "quadrature": float(coef[1]),
        "offset": float(coef[2]),
        "r2": float(1.0 - np.sum(resid ** 2) / denom),
        "n_samples": int(ts.size),
        "periods_in_window": window,
        "valid": bool(window >= 1.0),
        "condition": float(np.linalg.cond(basis)),
    }


def apparent_added_mass(force, displacement, t, omega, amplitude=None,
                        skip_cycles=2.0):
    """Single-frequency apparent mass ``m_app = (m_a - k_a/w^2)`` and damping.

    ``m_app`` is what a one-frequency forced test measures; it equals ``m_a``
    only when the added stiffness ``k_a`` vanishes (true for a flat plate by
    translational symmetry, *not* true in general).  ``separated`` is always
    ``False`` here -- use :func:`fit_added_mass_sweep` for ``m_a`` proper.
    """
    proj = harmonic_projection(t, force, omega, skip_cycles=skip_cycles)
    y = np.asarray(displacement, dtype=float)
    amp = float(amplitude) if amplitude is not None else \
        0.5 * float(y.max() - y.min())
    w2 = float(omega) ** 2
    return {
        "omega": float(omega),
        "amplitude": amp,
        "in_phase": proj["in_phase"],
        "quadrature": proj["quadrature"],
        "apparent_added_mass": proj["in_phase"] / (amp * w2),
        "damping": -proj["quadrature"] / (amp * float(omega)),
        "offset": proj["offset"],
        "r2": proj["r2"],
        "periods_in_window": proj["periods_in_window"],
        "valid": proj["valid"],
        "separated": False,
    }


def fit_added_mass_sweep(points, min_omega_span=1.5):
    """Separate ``m_a`` and ``k_a`` from ``K/A = m_a w^2 - k_a`` across w.

    Parameters
    ----------
    points : sequence of dict
        One entry per drive frequency, each with ``omega``, ``amplitude`` and
        ``in_phase`` (as returned by :func:`harmonic_projection` /
        :func:`apparent_added_mass`).  Invalid points are dropped.
    min_omega_span : float
        Required ``omega_max / omega_min`` for the regression to be called
        identifiable.  Two frequencies close together give a slope dominated by
        noise; a factor 1.5 is the smallest span used in the archived sweeps.

    Returns
    -------
    dict with ``added_mass`` (slope), ``added_stiffness`` (``-intercept``),
    ``identifiable``, ``omega_span``, ``r2``, ``per_point`` and ``reason``.
    """
    pts = [p for p in points if p.get("valid", True)]
    if len(pts) < 2:
        return {"added_mass": float("nan"), "added_stiffness": float("nan"),
                "identifiable": False, "omega_span": float("nan"),
                "r2": float("nan"), "per_point": [],
                "reason": "fewer than two valid drive frequencies"}
    w = np.array([p["omega"] for p in pts], dtype=float)
    amp = np.array([p["amplitude"] for p in pts], dtype=float)
    k = np.array([p["in_phase"] for p in pts], dtype=float) / amp
    span = float(w.max() / w.min())
    if span < min_omega_span:
        return {"added_mass": float("nan"), "added_stiffness": float("nan"),
                "identifiable": False, "omega_span": span,
                "r2": float("nan"), "per_point": [],
                "reason": "frequency span %.3f < %.3f" % (span, min_omega_span)}
    basis = np.stack([w ** 2, -np.ones_like(w)], axis=1)
    coef, *_ = np.linalg.lstsq(basis, k, rcond=None)
    resid = k - basis @ coef
    denom = float(np.sum((k - k.mean()) ** 2)) or 1.0
    return {
        "added_mass": float(coef[0]),
        "added_stiffness": float(coef[1]),
        "identifiable": True,
        "omega_span": span,
        "r2": float(1.0 - np.sum(resid ** 2) / denom),
        "residual_rms": float(np.sqrt(np.mean(resid ** 2))),
        "per_point": [dict(p, in_phase_over_amplitude=float(v))
                      for p, v in zip(pts, k)],
        "reason": "ok",
    }


# ---------------------------------------------------------------------------
# free-decay identifiability (amplification of the frequency error)
# ---------------------------------------------------------------------------
def added_mass_gain(added_mass_ratio):
    """``d m_a/m_a = -gain * df/f`` with ``gain = 2 (1 + r) / r``, ``r=m_a/m_s``.

    Derived from ``m_a = k/(2 pi f)^2 - m_s``::

        dm_a/df = -2 k/(2 pi)^2 f^-3 = -2 (m_s + m_a) / f
        dm_a/m_a = -2 (m_s + m_a)/m_a * df/f.
    """
    r = float(added_mass_ratio)
    if r <= 0.0:
        raise ValueError("added mass ratio must be positive")
    return 2.0 * (1.0 + r) / r


def amplify_frequency_error(added_mass_ratio, frequency_error_pct):
    """Exact mass-attribute error induced by a frequency error (percent in/out).

    The infinitesimal gain (:func:`added_mass_gain`) is not enough at the 10 %
    level the anchor actually exhibits; the exact map from ``f`` to ``m_a`` is

        m_a = k/(2 pi f)^2 - m_s,   m_s = m_a / r

        dm_a/m_a = ((1 + e)^-2 - 1) * (1 + 1/r),      e = df/f

    This is a *derived* quantity, not an independent measurement: it reproduces
    the three archived free-decay rows of ``xval_implicit_ibm_aitken_v5`` to
    < 0.1 percentage point (78.56 vs 78.54; -28.12 vs -28.10), which is the
    evidence that the "sign-changing bias" of that report is algebraic, not
    physical.
    """
    r = float(added_mass_ratio)
    e = float(frequency_error_pct) / 100.0
    if r <= 0.0:
        raise ValueError("added mass ratio must be positive")
    if e >= 1.0:
        raise ValueError("frequency error must be < 100%")
    return 100.0 * ((1.0 + e) ** -2 - 1.0) * (1.0 + 1.0 / r)


def required_frequency_accuracy(added_mass_ratio, target_pct=15.0):
    """Frequency accuracy (percent) needed to certify ``m_a`` to ``target_pct``."""
    return float(target_pct) / added_mass_gain(added_mass_ratio)


# ---------------------------------------------------------------------------
# legacy defect: reproduce and expose the rank deficiency
# ---------------------------------------------------------------------------
def legacy_basis_condition(omega, dt=1.0, n_steps=600, skip=300):
    """Condition number of the ``{y'', y', y, 1}`` basis used by the legacy fit."""
    t = dt * np.arange(1, int(n_steps) + 1, dtype=float)
    amp, w = 0.2, float(omega)
    basis = np.stack([-amp * w ** 2 * np.sin(w * t),
                      amp * w * np.cos(w * t),
                      amp * np.sin(w * t),
                      np.ones_like(t)], axis=1)[int(skip):]
    return float(np.linalg.cond(basis))


def legacy_single_frequency_fit(m_a, k_a=0.0, c_a=0.0, omega=2 * math.pi / 300.0,
                                amplitude=0.2, dt=1.0, n_steps=600,
                                skip=300, noise=0.0, seed=0):
    """Reproduce the legacy ``{y'', y', y, 1}`` fit on a *synthetic* exact record.

    The signal is a pure harmonic force built from a known ``m_a``, ``k_a``,
    ``c_a`` -- exactly representable by the legacy basis.  If the legacy fit
    returned the physics, it would return ``m_a`` here.  It does not: it returns
    the minimum-norm split of a rank-deficient problem.
    """
    t = dt * np.arange(1, int(n_steps) + 1, dtype=float)
    w = float(omega)
    y = amplitude * np.sin(w * t)
    v = amplitude * w * np.cos(w * t)
    a = -amplitude * w ** 2 * np.sin(w * t)
    f = -float(m_a) * a - float(c_a) * v - float(k_a) * y
    if noise:
        rng = np.random.default_rng(seed)
        f = f + noise * float(np.std(f)) * rng.standard_normal(f.size)
    basis = np.stack([a, v, y, np.ones_like(y)], axis=1)[skip:]
    coef, *_ = np.linalg.lstsq(basis, f[skip:], rcond=None)
    return {"added_mass_fit": -float(coef[0]),
            "damping_fit": -float(coef[1]),
            "stiffness_fit": -float(coef[2]),
            "constant_fit": float(coef[3]),
            "true_added_mass": float(m_a), "true_stiffness": float(k_a),
            "condition": legacy_basis_condition(omega, dt, n_steps, skip)}


# ---------------------------------------------------------------------------
# LBM drive
# ---------------------------------------------------------------------------
def measure_added_mass_sweep(half_width=10.0, periods=(96.0, 128.0, 192.0),
                             lattice=(160, 160), amplitude=0.25,
                             center=None, omega_lbm=1.8, dt=1.0,
                             cycles_settle=2.0, cycles_measure=3.0,
                             sub_tol=1e-10, max_sub_iter=40, rho_f=1.0):
    """Drive the implicit-IBM plate at several frequencies; return the sweep fit.

    Each frequency is an independent prescribed-kinematics LBM run
    (:func:`tensorfem.ibm_aitken_added_mass.forced_oscillation`, which owns the
    solver); this function only re-analyses the returned histories with
    :func:`apparent_added_mass` and :func:`fit_added_mass_sweep`.

    Returns the :func:`fit_added_mass_sweep` result plus ``points`` (raw
    per-frequency records, histories trimmed) and ``lamb_added_mass``.
    """
    from .ibm_aitken_added_mass import analytic_added_mass, forced_oscillation

    if center is None:
        center = (0.5 * lattice[0], 0.5 * lattice[1])
    points = []
    for period in periods:
        n_steps = int(round((cycles_settle + cycles_measure) * period))
        rec = forced_oscillation(n_steps=n_steps, dt=dt, half_width=half_width,
                                 center=center, lattice=lattice,
                                 amplitude=amplitude, rho_f=rho_f,
                                 drive_period=period, omega_lbm=omega_lbm,
                                 sub_tol=sub_tol, max_sub_iter=max_sub_iter)
        t = dt * np.arange(1, int(n_steps) + 1, dtype=float)
        pt = apparent_added_mass(rec["force_history"], rec["y_history"], t,
                                 omega=2.0 * math.pi / float(period),
                                 amplitude=amplitude,
                                 skip_cycles=cycles_settle)
        pt.update({"drive_period": float(period), "n_steps": int(n_steps),
                   "lattice": list(lattice), "half_width": half_width})
        points.append(pt)
    fit = fit_added_mass_sweep(points)
    fit["points"] = points
    fit["lamb_added_mass"] = analytic_added_mass(half_width, rho_f)
    fit["lamb_error_pct"] = 100.0 * (fit["added_mass"] -
                                     fit["lamb_added_mass"]) / \
        fit["lamb_added_mass"] if fit["identifiable"] else float("nan")
    return fit