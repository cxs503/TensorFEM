#!/usr/bin/env python3
"""Adjudicate the rainflow counting conventions behind docs/rainflow-counting-defect.md.

Independent reference: the ASTM E1049 four-point rule applied by *repeated scan*
of the reversal list (no shared stack code with ``rainflow_cycles``), plus the
level-crossing ("interpolated closure") decomposition the older audit panel used
as its expectation.  The script measures, for the same records:

* cycle structure (full/half, ranges) of the shipped counter vs both references,
* Miner damage with m = 3 (mean-stress free, so range-only),
* the arc-conservation residual ``sum(count*range) / (0.5 * sum|ds|) - 1``.

Outputs ``artifacts/rainflow_convention/audit.json`` and ``.../convention.png``.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorfem.marine_fatigue_advanced import _reversals, rainflow_cycles  # noqa: E402

OUT = REPO / "artifacts" / "rainflow_convention"

CONST_AMP_3 = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0]
NESTED = [0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 0.0]
TRIANGLE = [0.0, 100.0, 0.0]
TWO_LEVEL = [0.0, 5.0, 0.0, 5.0, 0.0, 20.0, 0.0, 20.0, 0.0]


def sine_40() -> list[float]:
    dt = 1.0 / 200.0
    t = np.arange(int(20.0 / dt)) * dt
    s = 50.0e6 * np.sin(2.0 * math.pi * 2.0 * t)
    return (s - s.mean()).tolist()


def four_point_reference(history) -> tuple[tuple[float, float, float], ...]:
    """ASTM E1049 four-point rainflow by repeated scan (independent of the stack)."""
    rev = list(_reversals(history))
    closed: list[tuple[float, float, float]] = []
    while True:
        hit = False
        for i in range(len(rev) - 3):
            s1, s2, s3, s4 = rev[i : i + 4]
            inner = abs(s3 - s2)
            if inner <= abs(s2 - s1) and inner <= abs(s4 - s3) and inner > 0.0:
                closed.append((inner, 0.5 * (s2 + s3), 1.0))
                del rev[i + 1 : i + 3]
                hit = True
                break
        if not hit:
            break
    out = list(closed)
    intervals = [
        (abs(b - a), 0.5 * (a + b))
        for a, b in itertools.pairwise(rev)
        if abs(b - a) > 0.0
    ]
    is_closed = len(rev) >= 2 and abs(rev[-1] - rev[0]) <= 1e-12 * max(
        1.0, abs(rev[0]), abs(rev[-1])
    )
    if is_closed and len(intervals) >= 2:
        head, tail = intervals[0][0], intervals[-1][0]
        if abs(head - tail) <= 1e-9 * max(1.0, head):
            out.append((head, 0.5 * (rev[0] + rev[1]), 1.0))
            intervals = intervals[1:-1]
    out.extend((rng, mean, 0.5) for rng, mean in intervals)
    return tuple(out)


def level_crossing_expectation(history) -> tuple[float, ...]:
    """Loops closed by interpolating a return to the last crossed level.

    This is what produced the ``{1,1,2,2,3,3}`` expectation in the old audit: walk
    the record and, whenever a reversal interval crosses the running level, close
    a full cycle of that excursion range.
    """
    rev = list(_reversals(history))
    ranges: list[float] = []
    lo = hi = rev[0]
    for a, b in itertools.pairwise(rev):
        if b > a:  # travelling up
            while hi + 1e-12 < b:
                ranges.append(b - lo)
                lo, hi = b, b
                break
            lo = min(lo, a)
            hi = max(hi, b)
        else:
            while lo - 1e-12 > b:
                ranges.append(hi - b)
                lo, hi = b, b
                break
            lo = min(lo, b)
            hi = max(hi, a)
    return tuple(sorted(ranges))


def summarise(cycles) -> dict:
    rows = sorted((float(r), float(c)) for r, _, c in cycles)
    return {
        "n_cycles": len(rows),
        "full_cycles": sum(1 for _, c in rows if c == 1.0),
        "half_cycles": sum(1 for _, c in rows if c == 0.5),
        "ranges": [r for r, _ in rows],
        "range_sum": math.fsum(c * r for r, c in rows),
        "damage_m3": math.fsum(c * r ** 3 for r, c in rows),
    }


def arc_half(history) -> float:
    rev = list(_reversals(history))
    return 0.5 * math.fsum(abs(b - a) for a, b in itertools.pairwise(rev))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sine = sine_40()
    rng = np.random.default_rng(2026)
    random_records = []
    for k in range(120):
        n = int(rng.integers(4, 60))
        walk = np.cumsum(np.random.default_rng(7000 + k).normal(size=n))
        walk -= walk.mean()
        random_records.append([float(walk[0]), *map(float, walk[1:]), float(walk[0])])

    records = {
        "constant_amplitude_3_periods": CONST_AMP_3,
        "expanding_zigzag": NESTED,
        "triangle": TRIANGLE,
        "two_level_block": TWO_LEVEL,
        "sine_40_periods": sine,
    }
    table = {}
    for name, history in records.items():
        shipped = summarise(
            [(row.range_pa, row.mean_pa, row.count) for row in rainflow_cycles(history)]
        )
        reference = summarise(four_point_reference(history))
        crossing = level_crossing_expectation(history)
        arc = arc_half(history)
        table[name] = {
            "shipped": shipped,
            "four_point_reference": reference,
            "structural_match": (
                shipped["n_cycles"] == reference["n_cycles"]
                and np.allclose(shipped["ranges"], reference["ranges"])
                and shipped["full_cycles"] == reference["full_cycles"]
            ),
            "damage_relative_gap": abs(shipped["damage_m3"] / reference["damage_m3"] - 1.0)
            if reference["damage_m3"]
            else 0.0,
            "range_sum_relative_gap": abs(shipped["range_sum"] / reference["range_sum"] - 1.0)
            if reference["range_sum"]
            else 0.0,
            "arc_half_path_length": arc,
            "shipped_arc_conservation_error": (
                abs(shipped["range_sum"] / arc - 1.0) if arc else 0.0
            ),
            "reference_arc_conservation_error": (
                abs(reference["range_sum"] / arc - 1.0) if arc else 0.0
            ),
            "level_crossing_ranges": list(crossing),
            "level_crossing_damage_m3": math.fsum(r ** 3 for r in crossing),
        }

    random_rows = []
    for history in random_records:
        shipped_sum = math.fsum(
            row.count * row.range_pa for row in rainflow_cycles(history)
        )
        reference_sum = math.fsum(r * c for r, _, c in four_point_reference(history))
        arc = arc_half(history)
        random_rows.append(
            {
                "arc": arc,
                "shipped_error": abs(shipped_sum / arc - 1.0) if arc else 0.0,
                "reference_error": abs(reference_sum / arc - 1.0) if arc else 0.0,
            }
        )

    payload = {
        "schema": "tensorfem.rainflow-convention-audit/1",
        "shipped_rule": "ASTM E1049 four-point, closed-record residual merged",
        "records": table,
        "random_closed_records": {
            "count": len(random_rows),
            "shipped_max_arc_error": max(r["shipped_error"] for r in random_rows),
            "reference_max_arc_error": max(r["reference_error"] for r in random_rows),
        },
        "conclusion": {
            "structural_defect_fixed": all(
                row["structural_match"] for row in table.values()
            ),
            "damage_claim_falsified": {
                "record": "expanding_zigzag",
                "old_audit_expected_damage_m3": 72.0,
                "four_point_reference_damage_m3": table["expanding_zigzag"][
                    "four_point_reference"
                ]["damage_m3"],
                "note": "the 72 came from the level-crossing decomposition, which "
                "is not rainflow; ASTM E1049 keeps seven half cycles on an "
                "expanding zigzag (no loop closes)",
            },
        },
    }
    (OUT / "audit.json").write_text(json.dumps(payload, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    names = list(table)
    x = np.arange(len(names))
    ax.bar(
        x - 0.2,
        [table[n]["shipped"]["full_cycles"] for n in names],
        0.4,
        label="shipped four-point (full cycles)",
    )
    ax.bar(
        x + 0.2,
        [table[n]["shipped"]["half_cycles"] for n in names],
        0.4,
        label="shipped four-point (half cycles)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n", 1) for n in names], fontsize=7)
    ax.set_ylabel("cycle count")
    ax.set_title("Cycle structure per record")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(
        x - 0.2,
        [table[n]["shipped"]["damage_m3"] for n in names],
        0.4,
        label="shipped",
    )
    ax.bar(
        x + 0.2,
        [table[n]["four_point_reference"]["damage_m3"] for n in names],
        0.4,
        label="four-point reference",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n", 1) for n in names], fontsize=7)
    ax.set_ylabel("Miner damage (m = 3, log)")
    ax.set_title("Damage: shipped vs independent reference")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "convention.png", dpi=140, bbox_inches="tight")

    print(json.dumps(payload["conclusion"], indent=2))
    print("structural/ damage gaps:")
    for n, row in table.items():
        print(
            f"  {n:32s} struct_match={row['structural_match']} "
            f"damage_gap={row['damage_relative_gap']:.3e} "
            f"arc_err={row['shipped_arc_conservation_error']:.2e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
