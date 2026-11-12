#!/usr/bin/env python3
"""Is the published rainflow fix spec satisfiable?  (verification of a spec)

``docs/rainflow-counting-defect.md`` §5 fixes a defect in
``tensorfem.marine_fatigue_advanced.rainflow_cycles`` and states three
criteria that a fix must meet *simultaneously*.  This script grades candidate
counters against those criteria and against the ASTM E1049-85 bookkeeping
invariant (arc conservation), and writes ``result.json``.

Candidates
----------
``production``   merged ``rainflow_cycles`` (3-point stack, ``len(stack)==3``
                 special case; the documented defect).
``four_point``   ASTM E1049-85 four-point rule: the excursion ``S2-S3`` closes
                 when ``|S3-S2| <= |S2-S1|`` **and** ``|S3-S2| <= |S4-S3|``;
                 closed records are scanned cyclically and their residual is
                 paired end to end.  This is the rule §5 recommends.
``remainder``    half-cycle pairing with a live remainder ("pagoda"/range-pair
                 form): an excursion is closed at the *interpolated* level it
                 returns to, and the unused part of the enclosing half cycle
                 stays on the stack.

Criteria (verbatim from §5)
---------------------------
C1  structured anchors: constant amplitude -> n full; nested
    ``0,1,-1,2,-2,3,-3,0`` -> full ranges ``1,1,2,2,3,3``; triangle -> 1 full;
    two-level block -> ``5,5,20,20`` full.
C2  narrow-band Rayleigh damage gate ``relative_error < 0.03`` and cycle rate
    within 1 % (``narrow_band_damage_case``).
C3  40-period sine -> 40 equivalent cycles, max range ``2A``.

No guideline factor or reference value enters a counter; references are used
only in the grading step.  Run::

    PYTHONPATH=src python benchmarks/verified/rainflow_fix_spec/run.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

import tensorfem.rainflow_spectrum_reference as ref  # noqa: E402
from tensorfem.marine_fatigue_advanced import (  # noqa: E402
    RainflowCycle,
    _reversals,
    rainflow_cycles,
)
from tensorfem.marine_fatigue_qualification import SNLogLogCurve  # noqa: E402

CURVE = SNLogLogCurve(100.0e6, 2.0e6, 3.0)
SINE_A = 50.0


# --------------------------------------------------------------------------- #
# candidate counters
# --------------------------------------------------------------------------- #
def production(history):
    return [(c.range_pa, c.mean_pa, c.count) for c in rainflow_cycles(history)]


def _cyclic_turns(points):
    pts = [p for i, p in enumerate(points) if i == 0 or p != points[i - 1]]
    while len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    if len(pts) < 3:
        return pts
    keep = [
        pts[i]
        for i in range(len(pts))
        if (pts[i] - pts[i - 1]) * (pts[(i + 1) % len(pts)] - pts[i]) < 0.0
    ]
    return keep if len(keep) >= 2 else pts


def four_point(history):
    rev = _reversals([float(v) for v in history])
    closed = len(rev) >= 3 and rev[0] == rev[-1]
    pts = list(rev[:-1]) if closed else list(rev)
    if closed:
        k = max(range(len(pts)), key=lambda i: (abs(pts[i]), pts[i]))
        pts = pts[k:] + pts[:k]
    out = []
    while len(pts) >= 4:
        n = len(pts)
        span = n if closed else n - 3
        hit = -1
        for i in range(span):
            if closed:
                a, b, c, d = (pts[(i + j) % n] for j in range(4))
            else:
                a, b, c, d = pts[i : i + 4]
            y = abs(c - b)
            if y > 0.0 and y <= abs(b - a) and y <= abs(d - c):
                hit = i
                break
        if hit < 0:
            break
        if closed:
            a, b, c, d = (pts[(hit + j) % n] for j in range(4))
            out.append((y, 0.5 * (b + c), 1.0))
            for j in sorted({(hit + 1) % n, (hit + 2) % n}, reverse=True):
                del pts[j]
        else:
            a, b, c, d = pts[hit : hit + 4]
            out.append((abs(c - b), 0.5 * (b + c), 1.0))
            del pts[hit + 1 : hit + 3]
    if closed:
        chain = _cyclic_turns(pts)
        segs = [
            (abs(b - a), 0.5 * (a + b))
            for a, b in zip(chain, chain[1:] + chain[:1])
        ]
        k = len(segs)
        for i in range(k // 2):
            (ra, ma), (rb, mb) = segs[i], segs[k - 1 - i]
            if ra > 0.0 and rb > 0.0:
                out.append((min(ra, rb), 0.5 * (ma + mb), 1.0))
        if k % 2:
            r, m = segs[k // 2]
            if r > 0.0:
                out.append((r, m, 0.5))
    else:
        for a, b in zip(pts, pts[1:]):
            if abs(b - a) > 0.0:
                out.append((abs(b - a), 0.5 * (a + b), 0.5))
    return out


def remainder(history):
    rev = _reversals([float(v) for v in history])
    stack = []
    out = []
    for a, b in zip(rev, rev[1:]):
        cr, cs, cstart = abs(b - a), (1.0 if b > a else -1.0), a
        while stack and stack[-1][1] != cs and stack[-1][0] <= cr:
            tr, ts, tstart = stack.pop()
            lo, hi = (tstart, tstart + tr) if ts > 0.0 else (tstart - tr, tstart)
            out.append((tr, 0.5 * (lo + hi), 1.0))
            cr, cs, cstart = cr - tr, cs, tstart
        if cr > 1e-12:
            stack.append((cr, cs, cstart))
    for r, s, start in stack:
        out.append((r, start + 0.5 * r * s, 0.5))
    return out


CANDIDATES = {
    "production": production,
    "four_point": four_point,
    "remainder": remainder,
}


# --------------------------------------------------------------------------- #
# grading
# --------------------------------------------------------------------------- #
def _structure(cycles):
    full = sorted(round(r, 6) for r, _, c in cycles if c == 1.0)
    half = sorted(round(r, 6) for r, _, c in cycles if c == 0.5)
    return {
        "full_ranges": full,
        "half_ranges": half,
        "sum_count": math.fsum(c for _, _, c in cycles),
        "arc": math.fsum(c * r for r, _, c in cycles),
        "max_range": max((r for r, _, _ in cycles), default=0.0),
    }


def _half_path(history):
    rev = _reversals([float(v) for v in history])
    return 0.5 * math.fsum(abs(b - a) for a, b in zip(rev, rev[1:]))


def anchors():
    t = np.arange(0.0, 20.0, 1.0 / 200.0)
    sine = (SINE_A * np.sin(2.0 * math.pi * 2.0 * t)).tolist()
    return {
        "const_amplitude_3": [0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0],
        "nested_zigzag": [0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 0.0],
        "triangle": [0.0, 100.0, 0.0],
        "two_level_block": [0.0, 5.0, 0.0, 5.0, 0.0, 20.0, 0.0, 20.0, 0.0],
        "sine_40_periods": sine,
    }


EXPECT = {
    "const_amplitude_3": {"full_ranges": [10.0, 10.0, 10.0], "half_ranges": []},
    "nested_zigzag": {"full_ranges": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0], "half_ranges": []},
    "triangle": {"full_ranges": [100.0], "half_ranges": []},
    "two_level_block": {"full_ranges": [5.0, 5.0, 20.0, 20.0], "half_ranges": []},
}


def grade_anchors():
    rows = {}
    for name, history in anchors().items():
        half_path = _half_path(history)
        rows[name] = {"half_path": half_path, "counters": {}}
        for cname, fn in CANDIDATES.items():
            st = _structure(fn(history))
            arc_residual = st["arc"] - half_path
            entry = dict(st)
            entry["arc_conservation_residual"] = arc_residual
            if name in EXPECT:
                entry["c1_pass"] = (
                    abs(arc_residual) < 1e-6
                    and st["full_ranges"] == EXPECT[name]["full_ranges"]
                    and st["half_ranges"] == EXPECT[name]["half_ranges"]
                )
            else:  # sine: criterion C3, equivalent cycles and 2A max range
                entry["c3_pass"] = (
                    abs(arc_residual) < 1e-6
                    and abs(st["sum_count"] - 40.0) / 40.0 < 0.05
                    and abs(st["max_range"] - 2.0 * SINE_A) / (2.0 * SINE_A) < 0.02
                )
            rows[name]["counters"][cname] = entry
    return rows


def grade_gate():
    """C2: narrow-band Rayleigh gate with each candidate as the production counter."""
    original = ref.rainflow_cycles
    rows = {}
    try:
        for cname, fn in CANDIDATES.items():
            if cname == "production":
                ref.rainflow_cycles = original
            else:
                ref.rainflow_cycles = lambda h, _fn=fn: tuple(
                    RainflowCycle(r, m, c) for r, m, c in _fn(h)
                )
            case = ref.narrow_band_damage_case(
                curve=CURVE, relative_bandwidth=0.05, cycles=2000.0
            )
            rows[cname] = {
                "relative_error": case["relative_error"],
                "count_rate_error": case["count_rate_error"],
                "damage_rate_ratio": case["damage_rate_ratio"],
                "cycles_counted": case["cycles_counted"],
                "c2_pass": case["relative_error"] < 0.03
                and case["count_rate_error"] < 0.01,
            }
    finally:
        ref.rainflow_cycles = original
    return rows


def main() -> int:
    anchor_rows = grade_anchors()
    gate_rows = grade_gate()

    verdict = {}
    for cname in CANDIDATES:
        c1 = all(
            row["counters"][cname].get("c1_pass", False)
            for name, row in anchor_rows.items()
            if name in EXPECT
        )
        c3 = anchor_rows["sine_40_periods"]["counters"][cname]["c3_pass"]
        c2 = gate_rows[cname]["c2_pass"]
        verdict[cname] = {
            "C1_structured_anchors": c1,
            "C2_rayleigh_gate": c2,
            "C3_sine_40_periods": c3,
            "satisfies_all_three": c1 and c2 and c3,
        }

    payload = {
        "anchor": "NEW:rainflow-fix-spec-satisfiability",
        "title": "Satisfiability of the published rainflow fix spec (three criteria)",
        "references": {
            "counting": "ASTM E1049-85 (reapproved 2017) four-point / stack cycle counting",
            "spectral_oracle": "Rayleigh narrow-band law (Bendat 1964; DNV-RP-C203 App. B)",
            "spec_under_test": "docs/rainflow-counting-defect.md §5",
        },
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
        "anchors": anchor_rows,
        "narrow_band_gate": gate_rows,
        "verdict": verdict,
        "conclusion": (
            "No candidate satisfies C1 and C3 together: the nested-zigzag "
            "expectation (full 1,1,2,2,3,3) requires closing loops at the "
            "interpolated record-start level, while the 40-period sine "
            "expectation forbids exactly that closure (it halves every cycle "
            "range and doubles the count). C1 and C3 are mutually "
            "inconsistent as written."
        ),
    }
    (HERE / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    for name, row in anchor_rows.items():
        print(f"\n{name}  half_path={row['half_path']:.6g}")
        for cname, st in row["counters"].items():
            key = "c1_pass" if name in EXPECT else "c3_pass"
            print(
                f"  {cname:11s} full={st['full_ranges'][:7]} half={st['half_ranges'][:5]} "
                f"sum_count={st['sum_count']:.3g} arc_resid={st['arc_conservation_residual']:+.3g} "
                f"max_range={st['max_range']:.4g}  {key}={st[key]}"
            )
    print("\nnarrow-band Rayleigh gate (C2):")
    for cname, row in gate_rows.items():
        print(f"  {cname:11s} rel_err={row['relative_error']:.4f} "
              f"count_rate_err={row['count_rate_error']:.4f} "
              f"damage_rate_ratio={row['damage_rate_ratio']:.4f} C2={row['c2_pass']}")
    print("\nverdict:")
    for cname, row in verdict.items():
        print(f"  {cname:11s} {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())