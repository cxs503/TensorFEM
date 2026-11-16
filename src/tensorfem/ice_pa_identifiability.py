"""Identifiability of the pressure--area exponent of the ice-coupled chain (W31).

Why this module exists
----------------------
``tensorfem.ice_pressure_area`` reports an exponent ``ex`` from a log--log fit of
``p = N_peak / A_peak`` on ``A_peak`` over an ensemble of ice impacts, and the
campaign node ``paper-exp-ice-coupled-6dof-4`` compares that number with a
declared reference (``ex = -0.28``).  The W24 run made the comparison
quantitative but left the central question open: the treatment arm returned
``ex ~ -0.10`` while a material law with *no* area dependence returned up to
``|ex| = 0.126`` -- i.e. the physical signal was inside its own null.  This
module supplies the missing piece: an explicit statement of *what the fitted
exponent measures*, and an estimator that separates the mechanism exponent from
the population (design-calibre) exponent.

The identity that has to be faced
---------------------------------
Write the rate-dependent crushing law of the treatment arm as

    p = p_ref * (epsdot / epsdot_ref) ** n ,      epsdot = V_close / (4 w),
    w = A / h                                                   (1)

with ``A`` the nominal contact area, ``h`` the floe thickness (so ``w`` is the
contact width).  In log variables (1) is *linear*:

    ln p = const + n ln V_close + n ln h - n ln A.               (2)

Hence for any population on which ``A`` co-varies with ``h`` and ``V_close``,
the raw (population) log--log slope is exactly

    ex_pop = -n + n (beta_h + beta_V) = n (beta_h + beta_V - 1),  (3)
    beta_h = d ln h / d ln A ,  beta_V = d ln V_close / d ln A .

Equation (3) is :func:`population_exponent_identity`.  It says three things
that are easy to get wrong:

* the *population* exponent is **not** a material constant: it is the declared
  rate exponent ``n`` multiplied by the ensemble's covariance structure.  The
  same contact law yields any value in ``[-n, +inf)`` depending on how the area
  variation of the ensemble is produced;
* if the area variation comes from the thickness (``A = w h`` with ``w`` fixed,
  ``beta_h -> 1``) the two mechanisms cancel and ``ex_pop -> 0`` for **every**
  ``n``.  That is the "endogenous floor": it is not numerical noise and not an
  estimator artefact, it is the algebra of ``A = w h``;
* if the area variation comes from the width (``beta_h -> 0``, ``beta_V -> 0``)
  the population exponent collapses to ``-n``, which is the mechanism exponent
  the treatment arm actually predicts.

The estimator
-------------
:func:`within_transform_slope` (fixed-effects / within transformation) bins the
sample on the two confounders ``h`` and ``V_close`` and fits the log--log slope
*inside* the bins, so that only width-driven area variation is used.  With
``n_h_bins = n_v_bins = 1`` it is exactly the raw population OLS slope, so a
single sweep :func:`conditioning_convergence` shows the measured exponent
migrating from ``ex_pop`` towards the mechanism exponent ``-n`` as the
confounders are resolved -- and the same sweep on the negative control (a law
with no area dependence) must stay at ``0``.  That pair of statements is what
"identifiability" means operationally: an estimator whose value tracks the
mechanism when a mechanism exists and returns zero when it does not.

Inversion
---------
:func:`implied_rate_exponent` inverts (3): if a design exponent ``ex_target``
were produced by an indentation-rate mechanism, the rate exponent it implies is
``n = ex_target / (beta_h + beta_V - 1)``; :func:`admissible_beta_window` gives
the covariance window in which a declared ``n`` range can reproduce it.  These
are the tools that let a mismatch be *attributed* (to ``n``, or to the ensemble)
instead of merely reported.

Honesty notes
-------------
* Nothing here calibrates or tunes anything: ``n``, ``p_ref``, ``epsdot_ref``
  stay the declared inputs of the W24 study; the estimator never sees ``n``.
* Equation (3) assumes the pressure law is deterministic in ``(A, h, V_close)``
  and holds up to the clamping / flooring branches of
  :class:`tensorfem.ice_pressure_area.RateDependentCrushing` (``max_pressure``
  and the ``1e-12`` strain-rate floor); binning the sample on ``h`` and ``V``
  removes the first-order effect of both, and the residual of the identity is
  reported by the driver rather than assumed to be zero.
* The within transformation conditions on the *recorded* confounders.  An
  unrecorded confounder that acts inside a bin would still bias the slope; the
  ``area_variance_explained_by_cells`` and ``se_*`` outputs bound how much of
  the area variation is actually being used, so an over-conditioned sample
  (bins with almost no width variation left) shows up as a large standard error
  instead of a confident wrong number.
* No normative (ISO 19906) number is stored, derived or implied anywhere.

All quantities SI; pressures in MPa where the field name says so.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "population_exponent_identity",
    "covariance_betas",
    "within_transform_slope",
    "conditioning_convergence",
    "implied_rate_exponent",
    "admissible_beta_window",
    "separation_sigma",
]


# --------------------------------------------------------------------------- #
# the identity
# --------------------------------------------------------------------------- #
def covariance_betas(area_m2, thickness_m, closing_speed_mps) -> dict:
    """Auxiliary population slopes ``beta_h``, ``beta_V`` of equation (3).

    ``beta_h = d ln h / d ln A`` and ``beta_V = d ln V_close / d ln A`` are
    properties of the *ensemble* (how the area variation is produced), not of
    the contact law.
    """
    a = np.asarray(area_m2, dtype=float)
    h = np.asarray(thickness_m, dtype=float)
    v = np.asarray(closing_speed_mps, dtype=float)
    good = (np.isfinite(a) & np.isfinite(h) & np.isfinite(v)
            & (a > 0) & (h > 0) & (v > 0))
    a, h, v = a[good], h[good], v[good]
    n = int(a.size)
    if n < 3:
        return {"n_points": n, "fit_ok": False,
                "note": "too few points for the auxiliary regressions"}
    la = np.log(a)
    beta_h = float(np.polyfit(la, np.log(h), 1)[0])
    beta_v = float(np.polyfit(la, np.log(v), 1)[0])
    return {
        "n_points": n,
        "fit_ok": True,
        "beta_h_vs_A": beta_h,
        "beta_V_vs_A": beta_v,
        "sum_betas": beta_h + beta_v,
        "log_area_var": float(np.var(la, ddof=1)),
    }


def population_exponent_identity(rate_exponent: float, beta_h: float,
                                 beta_v: float) -> dict:
    """Closed-form population exponent of a rate-dependent crushing law.

    ``ex_pop = -n + n (beta_h + beta_V)`` (equation 3), with the mechanism
    partial exponents ``(dlnp/dlnA, dlnp/dlnh, dlnp/dlnV) = (-n, +n, +n)``
    written out so the caller can see where each term comes from.
    """
    n = float(rate_exponent)
    if not np.isfinite(n):
        raise ValueError("rate_exponent must be finite")
    bh = float(beta_h)
    bv = float(beta_v)
    ex_pop = -n + n * (bh + bv)
    return {
        "rate_exponent": n,
        "partial_exponents": {"A": -n, "h": n, "V_close": n},
        "beta_h_vs_A": bh,
        "beta_V_vs_A": bv,
        "attenuation_factor": bh + bv - 1.0,
        "population_exponent": ex_pop,
        "mechanism_exponent": -n,
        "floor_limit_thickness_dominated": 0.0,
        "floor_limit_width_dominated": -n,
        "note": ("ex_pop = n (beta_h + beta_V - 1); the population exponent is a "
                 "material exponent times the ensemble's covariance structure"),
    }


# --------------------------------------------------------------------------- #
# fixed-effects (within) estimator
# --------------------------------------------------------------------------- #
def _quantile_bin_codes(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bin index of ``values`` with ``1 <= n_bins`` (0 -> single bin)."""
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if n_bins == 1:
        return np.zeros(values.size, dtype=np.int64)
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(values, qs))
    if edges.size == 0:
        return np.zeros(values.size, dtype=np.int64)
    return np.searchsorted(edges, values, side="right").astype(np.int64)


def within_transform_slope(area_m2, thickness_m, closing_speed_mps,
                           pressure_MPa, *, n_h_bins: int = 8,
                           n_v_bins: int = 8, clusters=None,
                           min_cell_size: int = 2) -> dict:
    """Log--log slope of ``p`` on ``A`` *within* cells of fixed ``(h, V_close)``.

    The sample is binned on the two confounders of equation (2); each cell is
    demeaned before the regression, so the estimator reacts only to area
    variation that is not explained by the thickness or the closing speed
    (i.e. to width-driven area variation).

    ``n_h_bins = n_v_bins = 1`` reproduces the raw population OLS slope exactly.

    Returns the slope, a classical standard error and (when ``clusters`` is
    given, e.g. the seed of each impact) a cluster-robust one, the number of
    cells used/dropped and how much of the log-area variance the cells absorb.
    """
    a = np.asarray(area_m2, dtype=float)
    h = np.asarray(thickness_m, dtype=float)
    v = np.asarray(closing_speed_mps, dtype=float)
    p = np.asarray(pressure_MPa, dtype=float)
    good = (np.isfinite(a) & np.isfinite(h) & np.isfinite(v) & np.isfinite(p)
            & (a > 0) & (h > 0) & (v > 0) & (p > 0))
    if clusters is not None:
        cl = np.asarray(clusters)
        if cl.shape[0] != a.shape[0]:
            raise ValueError("clusters must have one entry per sample")
        cl = cl[good]
    else:
        cl = None
    a, h, v, p = a[good], h[good], v[good], p[good]
    n_total = int(a.size)
    if n_total < 6:
        return {"fit_ok": False, "n_points": n_total,
                "note": "too few usable points for the within estimator"}

    x_all = np.log(a)
    y_all = np.log(p)
    hc = _quantile_bin_codes(np.log(h), n_h_bins)
    vc = _quantile_bin_codes(np.log(v), n_v_bins)
    cell = hc * (int(vc.max()) + 1) + vc

    # keep cells that carry at least ``min_cell_size`` points and some width
    # variation; report what was dropped instead of silently shrinking the data
    keep = np.zeros(n_total, dtype=bool)
    n_cells_used = 0
    n_cells_dropped = 0
    for c in np.unique(cell):
        idx = np.flatnonzero(cell == c)
        if idx.size < max(int(min_cell_size), 2):
            n_cells_dropped += 1
            continue
        xs = x_all[idx]
        if float(np.var(xs, ddof=1)) <= 1.0e-18:
            n_cells_dropped += 1
            continue
        keep[idx] = True
        n_cells_used += 1
    n = int(keep.sum())
    if n_cells_used < 1 or n < 6:
        return {"fit_ok": False, "n_points": n_total,
                "note": "no cell carries enough width variation"}

    x = x_all[keep]
    y = y_all[keep]
    cell = cell[keep]
    if cl is not None:
        cl = cl[keep]

    xd = np.empty_like(x)
    yd = np.empty_like(y)
    for c in np.unique(cell):
        idx = np.flatnonzero(cell == c)
        xd[idx] = x[idx] - x[idx].mean()
        yd[idx] = y[idx] - y[idx].mean()

    sxx = float(xd @ xd)
    if sxx <= 1.0e-18:
        return {"fit_ok": False, "n_points": n,
                "note": "degenerate design: no within-cell area variation"}
    slope = float(xd @ yd) / sxx
    resid = yd - slope * xd
    dof = max(n - n_cells_used - 1, 1)
    sigma2 = float(resid @ resid) / dof
    se_classical = float(np.sqrt(sigma2 / sxx))

    se_cluster = None
    n_clusters = None
    if cl is not None:
        groups = np.unique(cl)
        n_clusters = int(groups.size)
        if n_clusters >= 2:
            scores = np.zeros(n_clusters)
            for k, g in enumerate(groups):
                m = cl == g
                scores[k] = float(xd[m] @ resid[m])
            var_cluster = (float(scores @ scores) / sxx ** 2
                           * n_clusters / (n_clusters - 1.0))
            se_cluster = float(np.sqrt(var_cluster))

    # share of the (used) log-area variance that the cells absorb:
    # SS_within / SS_total of a one-way layout on the cell id
    ss_within = 0.0
    for c in np.unique(cell):
        idx = np.flatnonzero(cell == c)
        ss_within += float(np.sum((x[idx] - x[idx].mean()) ** 2))
    ss_total = float(np.sum((x - x.mean()) ** 2))
    return {
        "fit_ok": True,
        "slope": slope,
        "se_classical": se_classical,
        "se_cluster": se_cluster,
        "n_clusters": n_clusters,
        "n_points": n,
        "n_points_total": n_total,
        "n_cells_used": n_cells_used,
        "n_cells_dropped": n_cells_dropped,
        "n_h_bins": int(n_h_bins),
        "n_v_bins": int(n_v_bins),
        "r2_within": (float(1.0 - (resid @ resid) / float(yd @ yd))
                      if float(yd @ yd) > 0 else float("nan")),
        "area_var_explained_by_cells": (1.0 - ss_within / ss_total
                                        if ss_total > 0.0 else 0.0),
        "se_used": se_cluster if se_cluster is not None else se_classical,
        "note": ("log-log slope of p on A inside cells of fixed (thickness, "
                 "closing speed); n_h_bins = n_v_bins = 1 is the raw "
                 "population slope"),
    }


def conditioning_convergence(area_m2, thickness_m, closing_speed_mps,
                             pressure_MPa, *, resolutions=((1, 1), (2, 2),
                                                           (4, 4), (8, 8),
                                                           (16, 16)),
                             clusters=None) -> dict:
    """Sweep :func:`within_transform_slope` over conditioning resolutions.

    ``(1, 1)`` is the raw population slope; each later entry conditions away
    more of the thickness / closing-speed confounder.  A treatment arm with a
    real mechanism-level area dependence must drift towards its mechanism
    exponent and away from the population value; a law without one must stay
    at zero.
    """
    rows = []
    for nb in resolutions:
        nb = tuple(int(v) for v in nb)
        fit = within_transform_slope(area_m2, thickness_m, closing_speed_mps,
                                     pressure_MPa, n_h_bins=nb[0],
                                     n_v_bins=nb[1], clusters=clusters)
        fit["resolution"] = list(nb)
        rows.append(fit)
    ok = [r for r in rows if r.get("fit_ok")]
    raw = rows[0].get("slope") if rows[0].get("fit_ok") else None
    fine = ok[-1].get("slope") if ok else None
    return {
        "rows": rows,
        "raw_population_slope": raw,
        "finest_conditioned_slope": fine,
        "migration": (None if (raw is None or fine is None) else fine - raw),
        "note": ("convergence of the measured exponent as the confounders of "
                 "equation (3) are resolved"),
    }


# --------------------------------------------------------------------------- #
# inversion / attribution
# --------------------------------------------------------------------------- #
def implied_rate_exponent(target_exponent: float, beta_h: float,
                          beta_v: float) -> dict:
    """Rate exponent a rate-driven mechanism would need for ``target_exponent``.

    Inverts ``ex_pop = n (beta_h + beta_V - 1)``.  A value outside the
    physically cited brittle-crushing window is the signature that the target
    exponent is *not* reproducible by this mechanism on this population.
    """
    e = float(target_exponent)
    denom = float(beta_h) + float(beta_v) - 1.0
    out = {
        "target_exponent": e,
        "beta_h_vs_A": float(beta_h),
        "beta_V_vs_A": float(beta_v),
        "attenuation_factor": denom,
    }
    if abs(denom) < 1.0e-12:
        out["n_implied"] = None
        out["note"] = ("the attenuation factor vanishes: no rate exponent "
                       "reproduces the target on this population")
        return out
    out["n_implied"] = e / denom
    out["note"] = ("n_implied = ex_target / (beta_h + beta_V - 1); compare with "
                   "the cited brittle-crushing window before drawing any "
                   "conclusion")
    return out


def admissible_beta_window(target_exponent: float, n_range) -> dict:
    """Covariance window ``beta_h + beta_V`` admitting a ``n`` inside ``n_range``.

    From ``ex_target = n (beta - 1)``: ``beta = 1 + ex_target / n``, evaluated
    at the two ends of the rate-exponent window (``n > 0``).
    """
    e = float(target_exponent)
    lo, hi = (float(n_range[0]), float(n_range[1]))
    if lo <= 0.0 or hi <= 0.0:
        raise ValueError("rate exponents must be positive")
    b_lo = 1.0 + e / hi
    b_hi = 1.0 + e / lo
    return {
        "target_exponent": e,
        "rate_exponent_window": [lo, hi],
        "beta_sum_low": b_lo,
        "beta_sum_high": b_hi,
        "note": ("the target exponent is reproducible by a rate mechanism only "
                 "if the ensemble's beta_h + beta_V falls inside this window"),
    }


def separation_sigma(estimate: float, se_estimate: float, floor_estimate: float,
                     se_floor: float) -> dict:
    """How many combined standard errors separate an estimate from a floor.

    ``floor_estimate`` is the apparent exponent produced by a material law that
    has no area dependence at all (the negative control).  A physical exponent
    is only *resolved* when the two are distinguishable, so the criterion used
    across the ice-load campaign is expressed here in sigma of the combined
    uncertainty.
    """
    se = float(np.hypot(float(se_estimate), float(se_floor)))
    delta = abs(float(estimate) - float(floor_estimate))
    return {
        "estimate": float(estimate),
        "floor": float(floor_estimate),
        "abs_difference": delta,
        "se_estimate": float(se_estimate),
        "se_floor": float(se_floor),
        "se_combined": se,
        "sigma": (delta / se) if se > 0.0 else float("inf"),
        "ci95_non_overlapping": delta > 1.96 * se,
        "resolved_3sigma": delta > 3.0 * se,
    }