"""Reproductions for the rainflow counter defect (see docs/rainflow-counting-defect.md).

Green tests here lock the invariants the *current* counter does satisfy (arc
conservation, equivalent-cycle content on a triangle, sine count) so that the
defect is not confused with a broken record.  The two ``xfail(strict=True)``
tests are the defect itself: they fail today and will turn into failures of the
suite the moment somebody marks them ``xpass`` - i.e. strict xfail is the
regression tripwire for the fix.

Scope: counting structure only.  Damage from the wrong pairing is quantified in
``test_defect_nested_damage_is_inflated``.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from tensorfem.marine_fatigue_advanced import _reversals, rainflow_cycles

CONST_AMP_3 = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0]
NESTED = [0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 0.0]


def _range_total(history) -> float:
    return math.fsum(row.count * row.range_pa for row in rainflow_cycles(history))


def _arc_total(history) -> float:
    reversals = _reversals(history)
    return 0.5 * math.fsum(abs(b - a) for a, b in itertools.pairwise(reversals))


# --------------------------------------------------------------------------- #
# invariants that DO hold today (green) - the counter is not simply broken
# --------------------------------------------------------------------------- #
def test_arc_conservation_holds_for_random_closed_records():
    """sum(count * range) == half the path length: no range is lost or doubled."""
    rng = np.random.default_rng(2026)
    for k in range(80):
        walk = np.cumsum(
            np.random.default_rng(7000 + k).normal(size=int(rng.integers(4, 60)))
        )
        walk -= walk.mean()
        history = [float(walk[0]), *map(float, walk[1:]), float(walk[0])]
        assert _range_total(history) == pytest.approx(_arc_total(history), rel=1e-12)


def test_equivalent_cycle_content_of_a_triangle_is_one():
    cycles = rainflow_cycles((0.0, 100e6, 0.0))
    assert math.fsum(row.count for row in cycles) == pytest.approx(1.0)


def test_sine_dwell_keeps_one_equivalent_cycle_per_period():
    dt = 1.0 / 200.0
    t = np.arange(int(20.0 / dt)) * dt
    signal = 50.0e6 * np.sin(2.0 * math.pi * 2.0 * t)
    cycles = rainflow_cycles((signal - signal.mean()).tolist())
    assert math.fsum(row.count for row in cycles) == pytest.approx(40.0, rel=0.05)
    assert max(row.range_pa for row in cycles) == pytest.approx(100.0e6, rel=0.02)


# --------------------------------------------------------------------------- #
# the defect (strict xfail: tripwire for the fix)
# --------------------------------------------------------------------------- #
def test_fixed_constant_amplitude_block_yields_three_full_cycles():
    """Was the strict-xfail tripwire; closed by the ASTM E1049 four-point counter.

    See docs/rainflow-four-point-adjudication.md (every closed loop is now
    reported as a full cycle, so a three-period block is three full cycles).
    """
    cycles = rainflow_cycles(CONST_AMP_3)
    assert [row.count for row in cycles] == [1.0, 1.0, 1.0]
    assert [row.range_pa for row in cycles] == [10.0, 10.0, 10.0]


@pytest.mark.xfail(
    strict=True,
    reason="the audit panel asked for {1,1,2,2,3,3}; that is a level-crossing "
    "reading, not rainflow - the ASTM E1049 four-point rule gives seven half "
    "cycles here (no loop closes while the amplitude keeps growing). Kept as a "
    "strict xfail so the mis-specified expectation cannot be silently adopted; "
    "see docs/rainflow-four-point-adjudication.md",
)
def test_misspecified_level_crossing_expectation_is_not_rainflow():
    ranges = sorted(row.range_pa for row in rainflow_cycles(NESTED))
    assert ranges == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]


def test_defect_nested_damage_is_inflated():
    """Documents the magnitude: interval decomposition over-counts m=3 damage 3.25x.

    This is *green today* because it measures the current (defective) output; the
    fix must flip the two xfail tests above and this assertion together.
    """
    reversals = _reversals(NESTED)
    interval_damage = 0.5 * math.fsum(
        abs(b - a) ** 3 for a, b in itertools.pairwise(reversals)
    )
    current = math.fsum(
        row.count * row.range_pa**3 for row in rainflow_cycles(NESTED)
    )
    assert current == pytest.approx(interval_damage)
    assert current == pytest.approx(234.0)
    # the "72 / +225 %" figure in docs/rainflow-counting-defect.md came from a
    # level-crossing decomposition (1,1,2,2,3,3), which is *not* rainflow: the
    # ASTM E1049 four-point reference returns the same seven intervals and hence
    # the same 234, i.e. the damage claim is falsified while the structural
    # labelling defect was real.  See docs/rainflow-four-point-adjudication.md.
    assert 1 + 1 + 8 + 8 + 27 + 27 == 72