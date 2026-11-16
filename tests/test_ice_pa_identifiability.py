"""Tests for :mod:`tensorfem.ice_pa_identifiability` (W31).

The tests are synthetic on purpose: the point of the module is an algebraic
statement (``ex_pop = n (beta_h + beta_V - 1)``) plus an estimator, so the
reference values are injected exactly and the estimator must return them.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorfem.ice_pa_identifiability import (
    admissible_beta_window,
    conditioning_convergence,
    covariance_betas,
    implied_rate_exponent,
    population_exponent_identity,
    separation_sigma,
    within_transform_slope,
)


# --------------------------------------------------------------------------- #
# synthetic ensemble builder
# --------------------------------------------------------------------------- #
def _rate_law_ensemble(n, *, n_floes=4000, seed=7, width_frac=0.35):
    """Impacts of ``p = p_ref (V h / (4 A))**n`` with a declared covariance.

    ``width_frac`` splits the log-area variance between the thickness (``1 -
    width_frac``) and the contact width (``width_frac``), so the population
    exponent can be moved between ``0`` (thickness dominated) and ``-n``
    (width dominated) while the contact law stays identical.
    """
    rng = np.random.default_rng(seed)
    z_h = rng.normal(size=n_floes)
    z_w = rng.normal(size=n_floes)
    z_v = rng.normal(size=n_floes)
    ln_h = np.log(1.0) + 0.60 * z_h
    ln_w = np.log(2.0) + 0.80 * (np.sqrt(width_frac) * z_w
                                 + np.sqrt(1.0 - width_frac) * z_h * 0.0
                                 + width_frac * z_h * 0.0)
    # make the split exact: area variation = (1 - width_frac) from h,
    # width_frac from w, with w varying by 0.80 * sqrt(1) scale
    ln_w = np.log(2.0) + 0.80 * np.sqrt(width_frac) * z_w
    ln_h = np.log(1.0) + 0.60 * np.sqrt(1.0 - width_frac) * z_h
    ln_v = np.log(0.5) + 0.30 * z_v
    h = np.exp(ln_h)
    w = np.exp(ln_w)
    v = np.exp(ln_v)
    area = w * h
    p = np.exp(n * (ln_v + ln_h - np.log(4.0) - np.log(area)))
    return area, h, v, p


# --------------------------------------------------------------------------- #
# the identity
# --------------------------------------------------------------------------- #
def test_identity_matches_synthetic_rate_law():
    for n in (0.15, 0.28, 0.33, 0.50):
        a, h, v, p = _rate_law_ensemble(n)
        betas = covariance_betas(a, h, v)
        pred = population_exponent_identity(n, betas["beta_h_vs_A"],
                                            betas["beta_V_vs_A"])
        raw = float(np.polyfit(np.log(a), np.log(p), 1)[0])
        # the law is log-linear -> the identity is exact
        assert pred["population_exponent"] == pytest.approx(raw, abs=1e-9)
        assert pred["mechanism_exponent"] == pytest.approx(-n)


def test_identity_thickness_dominated_floor_is_zero():
    # pure thickness variation, width (nearly) fixed -> ex_pop -> 0 for any n
    for n in (0.15, 0.33, 0.50):
        a, h, v, p = _rate_law_ensemble(n, width_frac=0.02)
        raw = float(np.polyfit(np.log(a), np.log(p), 1)[0])
        assert abs(raw) < 0.05 * n, "thickness confounder must cancel the law"


def test_identity_width_dominated_recovers_mechanism():
    for n in (0.15, 0.33):
        a, h, v, p = _rate_law_ensemble(n, width_frac=0.98)
        raw = float(np.polyfit(np.log(a), np.log(p), 1)[0])
        assert raw == pytest.approx(-n, rel=0.10)


# --------------------------------------------------------------------------- #
# the estimator
# --------------------------------------------------------------------------- #
def test_single_cell_equals_raw_population_slope():
    a, h, v, p = _rate_law_ensemble(0.33, width_frac=0.5)
    raw = float(np.polyfit(np.log(a), np.log(p), 1)[0])
    fit = within_transform_slope(a, h, v, p, n_h_bins=1, n_v_bins=1)
    assert fit["fit_ok"]
    assert fit["slope"] == pytest.approx(raw, rel=1e-9)
    assert fit["area_var_explained_by_cells"] == pytest.approx(0.0, abs=1e-12)


def test_conditioned_slope_recovers_injected_exponent():
    for n in (0.15, 0.33, 0.50):
        a, h, v, p = _rate_law_ensemble(n, width_frac=0.35)
        conv = conditioning_convergence(a, h, v, p)
        raw = conv["raw_population_slope"]
        fine = conv["finest_conditioned_slope"]
        assert abs(raw) < 0.7 * n, "population slope should be attenuated"
        assert fine == pytest.approx(-n, rel=0.06), (
            "within estimator must recover the injected exponent")


def test_conditioned_slope_is_zero_for_area_free_law():
    # negative control: pressure independent of area, correlated with h by
    # accident through the per-floe draw only
    rng = np.random.default_rng(11)
    n_floes = 3000
    h = np.exp(rng.normal(0.0, 0.6, n_floes))
    w = np.exp(rng.normal(np.log(2.0), 0.8, n_floes))
    v = np.exp(rng.normal(np.log(0.5), 0.3, n_floes))
    area = w * h
    p = np.exp(rng.normal(np.log(1.9), 0.35, n_floes))
    conv = conditioning_convergence(area, h, v, p)
    assert abs(conv["finest_conditioned_slope"]) < 0.05
    assert abs(conv["raw_population_slope"]) < 0.08


def test_cluster_standard_error_is_reported():
    a, h, v, p = _rate_law_ensemble(0.33)
    clusters = np.repeat(np.arange(10), a.size // 10)
    fit = within_transform_slope(a, h, v, p, n_h_bins=4, n_v_bins=4,
                                 clusters=clusters)
    assert fit["fit_ok"]
    assert fit["n_clusters"] == 10
    assert fit["se_cluster"] is not None and fit["se_cluster"] > 0.0
    assert fit["se_used"] == fit["se_cluster"]


def test_degenerate_inputs_are_flagged_not_invented():
    a, h, v, p = _rate_law_ensemble(0.33)
    fit = within_transform_slope(a[:3], h[:3], v[:3], p[:3])
    assert fit["fit_ok"] is False
    # zero area variance -> no width information at all
    fit2 = within_transform_slope(np.full(a.size, a[0]), h, v, p, n_h_bins=1,
                                  n_v_bins=1)
    assert fit2["fit_ok"] is False
    with pytest.raises(ValueError):
        within_transform_slope(a, h, v, p, n_h_bins=0)


# --------------------------------------------------------------------------- #
# inversion / attribution
# --------------------------------------------------------------------------- #
def test_implied_rate_exponent_inverts_the_identity():
    n = 0.33
    betas = {"beta_h_vs_A": 0.45, "beta_V_vs_A": 0.25}
    pred = population_exponent_identity(n, betas["beta_h_vs_A"],
                                        betas["beta_V_vs_A"])
    inv = implied_rate_exponent(pred["population_exponent"],
                                betas["beta_h_vs_A"], betas["beta_V_vs_A"])
    assert inv["n_implied"] == pytest.approx(n, rel=1e-12)


def test_implied_rate_exponent_null_attenuation():
    inv = implied_rate_exponent(-0.28, 0.5, 0.5)
    assert inv["n_implied"] is None


def test_admissible_beta_window_brackets_the_target():
    win = admissible_beta_window(-0.28, (0.2, 0.4))
    # n = 0.4 gives beta = 1 - 0.7 = 0.30 ; n = 0.2 gives beta = -0.40
    assert win["beta_sum_low"] == pytest.approx(1.0 + (-0.28) / 0.4)
    assert win["beta_sum_high"] == pytest.approx(1.0 + (-0.28) / 0.2)
    # a population inside the window must invert back inside the n window
    beta = 0.5 * (win["beta_sum_low"] + win["beta_sum_high"])
    inv = implied_rate_exponent(-0.28, beta, 0.0)
    assert 0.2 <= inv["n_implied"] <= 0.4


def test_separation_sigma_flags_resolved_and_unresolved():
    resolved = separation_sigma(-0.33, 0.01, 0.02, 0.02)
    assert resolved["resolved_3sigma"] is True
    assert resolved["ci95_non_overlapping"] is True
    inside = separation_sigma(-0.10, 0.02, -0.12, 0.03)
    assert inside["resolved_3sigma"] is False