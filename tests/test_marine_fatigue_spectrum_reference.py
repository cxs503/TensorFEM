"""Fail-closed spectral verification of the rainflow + Miner fatigue path.

These cases compare the *merged* ``marine_fatigue_advanced`` primitives with
standalone mathematics (Rayleigh narrow-band law, ASTM E1049 bookkeeping
identity, closed-form Paris integral) rather than with a second in-house
counter.  The expected-value locks below also pin the two documented traps
(sampling loss, start-point dependence) so the write-up cannot drift.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem.marine_fatigue_advanced import (
    CrackGrowthBlock,
    ParisLaw,
    paris_crack_growth,
    rainflow_cycles,
    spectrum_miner_damage,
)
from tensorfem.marine_fatigue_qualification import SNLogLogCurve
from tensorfem.rainflow_spectrum_reference import (
    narrow_band_damage,
    narrow_band_damage_case,
    narrow_band_process,
    paris_power_law_closed_form,
    rainflow_count_invariant,
    rayleigh_range_moment,
    rayleigh_range_pdf,
    sampled_peak_damage_factor,
)

CURVE = SNLogLogCurve(100.0e6, 2.0e6, 3.0)
RMS = 40.0e6


def test_rayleigh_moment_matches_direct_integration_of_its_own_density():
    # Independent check of the oracle itself: integrate S^m p(S) dS numerically
    # with the density returned by the module and compare with the closed form.
    grid = np.linspace(1e-3, 12.0 * RMS, 400001)
    density = rayleigh_range_pdf(grid, RMS)
    for slope in (1.0, 3.0, 5.0):
        numerical = float(np.trapezoid(grid**slope * density, grid))
        closed = rayleigh_range_moment(RMS, slope)
        # trapezoid on a truncated grid: 1e-4 is grid-limited, not model-limited
        assert abs(numerical / closed - 1.0) < 1e-4


def test_narrow_band_rainflow_damage_matches_rayleigh_law_within_3pct():
    case = narrow_band_damage_case(
        curve=CURVE, relative_bandwidth=0.05, cycles=20000.0, samples_per_cycle=40.0
    )
    assert case["spectral_width"] < 0.02
    assert case["relative_error"] < 0.03, case
    # finite-bandwidth rainflow damage may not fall below the narrow-band law
    assert case["damage_rate_ratio"] >= 1.0, case
    # the counter must reproduce the spectral cycle rate for a narrow band
    assert case["count_rate_error"] < 0.01, case


def test_undersampled_narrow_band_peaks_bias_damage_low():
    # Documented trap: 6 samples per cycle loses ~11% of the damage.
    coarse = narrow_band_damage_case(
        curve=CURVE, relative_bandwidth=0.05, cycles=20000.0, samples_per_cycle=6.0
    )
    assert coarse["damage_ratio"] < 0.93, coarse
    assert coarse["damage_ratio"] < coarse["sampling_factor_estimate"] + 0.05
    assert sampled_peak_damage_factor(6.0, 3.0) == pytest.approx(0.8691, abs=5e-4)


def test_counting_identity_holds_for_closed_records():
    rng = np.random.default_rng(11)
    for k in range(60):
        walk = np.cumsum(np.random.default_rng(4000 + k).normal(size=int(rng.integers(4, 12))))
        walk -= walk.mean()
        history = [float(walk[0]), *map(float, walk[1:]), float(walk[0])]
        stats = rainflow_count_invariant(history)
        assert stats["closed"] == 1.0
        assert stats["defect"] < 1e-12, (history, stats)


def test_start_point_changes_only_the_residual_decomposition():
    history, _ = narrow_band_process(
        relative_bandwidth=0.05, cycles=300.0, samples_per_cycle=40.0
    )
    damages = []
    counts = []
    for fraction in (0.0, 0.13, 0.41, 0.77):
        shifted = np.roll(history, -int(fraction * history.size))
        cycles = rainflow_cycles(shifted.tolist())
        damages.append(spectrum_miner_damage(cycles, CURVE).damage)
        counts.append(math.fsum(row.count for row in cycles))
    # an open record may lose or gain at most one residual half cycle
    assert max(counts) - min(counts) <= 1.0 + 1e-9
    spread = max(damages) / min(damages) - 1.0
    assert spread < 0.02, spread             # damage sensitivity stays sub-2%


@pytest.mark.parametrize(
    "slope,coefficient",
    [(2.0, 1.0e-24), (2.5, 1.0e-24), (3.0, 1.0e-30), (4.0, 1.0e-38)],
)
def test_paris_power_law_branch_matches_closed_form(slope, coefficient):
    law = ParisLaw(coefficient, slope, 1.0)
    blocks = (
        CrackGrowthBlock(100.0e6, 1000.0),
        CrackGrowthBlock(150.0e6, 500.0),
        CrackGrowthBlock(80.0e6, 2000.0),
    )
    produced = paris_crack_growth(0.01, blocks, law).final_crack_m
    oracle = paris_power_law_closed_form(0.01, blocks, law)
    assert produced < 0.05                      # subcritical, physically meaningful
    assert abs(produced / oracle - 1.0) < 1e-12


def test_paris_closed_form_reports_runaway_growth():
    law = ParisLaw(1.0e-24, 3.0, 1.0)
    blocks = (CrackGrowthBlock(100.0e6, 10000.0),)
    assert math.isinf(paris_power_law_closed_form(0.01, blocks, law))
    with pytest.raises(ValueError):
        paris_crack_growth(0.01, blocks, law)


def test_sampling_factor_vanishes_as_rate_grows():
    factors = [sampled_peak_damage_factor(n, 3.0) for n in (4, 8, 20, 60, 120)]
    assert factors == sorted(factors)
    assert factors[-1] > 0.999
    assert narrow_band_damage(RMS, CURVE, 10.0) > narrow_band_damage(RMS, CURVE, 1.0)