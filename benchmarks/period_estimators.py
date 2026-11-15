"""High-precision period estimator for (possibly distorted) decay records.

Why this exists
---------------
The convergence study needs the coupled frequency to ~0.05 %, because the
measured added mass goes as ``T^-2``: a 0.5 % period bias is a 1 % added-mass
bias, i.e. 0.15 % of coupled-frequency error -- acceptable -- but
``estimate_period_damped`` returns the FFT seed unless its internal pattern
search actually moves, which quantises the answer to FFT bins (observed: two
different configurations reported *identical* added-mass ratios to 4 digits).

This module fits

    y(t) = c0 + exp(-t/tau) * sum_{k=1..K} [a_k cos(k w t) + b_k sin(k w t)]

with ``w = 2 pi / T``.  For fixed ``(T, tau)`` the model is **linear** in
``c0, a_k, b_k`` (separable least squares), so ``(T, tau)`` are the only two
non-linear parameters and can be minimised to machine-limited tolerance by a
coarse grid followed by alternating golden-section refinement.  ``K > 1`` lets
the fit absorb the second harmonic that a direct-forcing IBM generates, instead
of letting that distortion bias the fundamental.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["multiharmonic_period", "period_spread"]


def _design(t, T, tau, n_harm):
    w = 2.0 * math.pi / T
    env = np.exp(-t / tau)
    cols = [np.ones_like(t)]
    for k in range(1, int(n_harm) + 1):
        cols.append(env * np.cos(k * w * t))
        cols.append(env * np.sin(k * w * t))
    return np.column_stack(cols)


def _rss(t, y, T, tau, n_harm):
    A = _design(t, T, tau, n_harm)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = A @ coef - y
    return float(res @ res), coef


def _golden(f, lo, hi, tol=1e-6, itmax=80):
    """Minimise a unimodal scalar function on ``[lo, hi]``."""
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = float(lo), float(hi)
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(itmax):
        if abs(b - a) <= tol * max(1.0, abs(a) + abs(b)):
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    return (a + b) / 2.0


def multiharmonic_period(series, dt=1.0, T0=None, n_harm=2, tau_bounds=None,
                         span=0.35):
    """Fundamental period of a decaying, possibly distorted oscillation.

    ``T0`` seeds the search (default: the record span / 2 — the sweep uses a
    known design period).  Returns ``period``, ``tau``, ``amplitude`` (the
    fundamental's envelope), ``rms``, ``rms_over_amplitude``, ``harmonics``
    (relative amplitude of the k >= 2 terms), ``valid`` and the residual at the
    returned optimum.
    """
    y = np.asarray(series, dtype=float)
    n = int(y.size)
    out = {"period": float("nan"), "tau": float("nan"), "amplitude": float("nan"),
           "rms": float("nan"), "rms_over_amplitude": float("nan"),
           "harmonics": [], "offset": float("nan"), "valid": False,
           "method": f"multiharmonic_ls_K{int(n_harm)}"}
    if n < 16 or not np.all(np.isfinite(y)):
        return out
    t = np.arange(n, dtype=float) * float(dt)
    if T0 is None or not (T0 == T0) or T0 <= 0.0:
        crossings = _zero_crossings(y)
        T0 = (float(np.mean(np.diff(crossings))) if len(crossings) > 2
              else float(t[-1]) / 2.0)
    lo_T, hi_T = T0 * (1.0 - span), T0 * (1.0 + span)
    if tau_bounds is None:
        tau_bounds = (float(t[-1]) / 6.0, 60.0 * float(t[-1]))
    lo_tau, hi_tau = tau_bounds

    # coarse grid, then alternating golden-section on T and log(tau)
    best = None
    for T in np.linspace(lo_T, hi_T, 17):
        for tau in np.geomspace(lo_tau, hi_tau, 13):
            r, _ = _rss(t, y, T, tau, n_harm)
            if best is None or r < best[0]:
                best = (r, float(T), float(tau))
    _, T, tau = best
    for _ in range(6):
        T = _golden(lambda x: _rss(t, y, x, tau, n_harm)[0],
                    max(lo_T, T * 0.98), min(hi_T, T * 1.02), tol=1e-7)
        tau = _golden(lambda x: _rss(t, y, T, x, n_harm)[0],
                      max(lo_tau, tau / 2.0), min(hi_tau, tau * 2.0), tol=1e-7)
    rss, coef = _rss(t, y, T, tau, n_harm)
    amp = math.hypot(coef[1], coef[2])
    rms = math.sqrt(rss / n)
    harmonics = []
    for k in range(2, int(n_harm) + 1):
        ak, bk = coef[2 * k - 1], coef[2 * k]
        harmonics.append(float(math.hypot(ak, bk) / amp) if amp > 0 else float("nan"))
    out.update({"period": float(T), "tau": float(tau), "amplitude": float(amp),
                "offset": float(coef[0]), "rms": float(rms),
                "rms_over_amplitude": float(rms / amp) if amp > 0 else float("inf"),
                "harmonics": harmonics, "offset_ratio": float(abs(coef[0]) / amp)
                if amp > 0 else float("inf")})
    out["valid"] = bool(np.isfinite(T) and amp > 0.0
                        and out["rms_over_amplitude"] < 0.05
                        and (hi_T - lo_T) > 0.0)
    return out


def _zero_crossings(y):
    """Upward zero crossings (mean-removed) with linear interpolation."""
    z = y - float(np.mean(y))
    idx = np.where((z[:-1] <= 0.0) & (z[1:] > 0.0))[0]
    out = []
    for i in idx:
        denom = z[i + 1] - z[i]
        out.append(i + (0.0 - z[i]) / denom if denom else float(i))
    return np.array(out, dtype=float)


def period_spread(estimates):
    """Relative spread of a set of period estimates (measurement uncertainty)."""
    vals = [float(v) for v in estimates if v == v and v > 0.0]
    if len(vals) < 2:
        return float("nan")
    return (max(vals) - min(vals)) / float(np.mean(vals))