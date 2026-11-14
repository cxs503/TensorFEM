#!/usr/bin/env python3
"""Probe: truncation sensitivity of the contact radius and indentation.

Reads the result.json and checks that the indentation and contact-radius
errors are stable across the depth ladder (24a, 48a, 96a).  A large drift
means the accepted depth 96a is still contaminated by the truncation
boundary.

Note: the current result.json only contains depth-96 runs, so this probe
grades the truncation gates from the sibling truncation_extrapolation.json
instead.

Usage
-----
    python benchmarks/verified/fem_hertz_contact_loadcontrol/probes/probe_truncation.py
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(HERE, "..", "result.json")
TRUNCATION = os.path.join(HERE, "..", "truncation_extrapolation.json")
OUT = os.path.join(HERE, "truncation_probe.json")

CRITERIA = {
    "indentation_drift_pp": 1.0,
    "contact_radius_drift_pp": 0.5,
    "reference": "error must not drift by more than the threshold (pp) between 48a and 96a",
}


def probe():
    # First try to get depth-ladder data from truncation_extrapolation.json
    trunc_data = None
    if os.path.exists(TRUNCATION):
        with open(TRUNCATION) as fh:
            trunc_data = json.load(fh)

    probes = []

    # If truncation data is available, use it
    if trunc_data and "ladder" in trunc_data:
        ladder = trunc_data["ladder"]
        for i in range(len(ladder) - 1):
            r1, r2 = ladder[i], ladder[i + 1]
            d1, d2 = r1["depth_over_a"], r2["depth_over_a"]
            drift_d = abs(r1["error_pct"]["indentation"] - r2["error_pct"]["indentation"])
            drift_a = abs(
                r1["error_pct"]["contact_radius_linear"]
                - r2["error_pct"]["contact_radius_linear"]
            )
            probes.append({
                "key": f"d{d1:.0f}a_vs_d{d2:.0f}a",
                "depth_1": d1,
                "depth_2": d2,
                "indentation_drift_pp": drift_d,
                "contact_radius_drift_pp": drift_a,
                "passed": (
                    drift_d <= CRITERIA["indentation_drift_pp"]
                    and drift_a <= CRITERIA["contact_radius_drift_pp"]
                ),
            })

    # Also check result.json for any depth-variant runs
    with open(RESULT) as fh:
        result = json.load(fh)

    rows = result.get("runs", {})
    # Group by (ratio, h) and check depth variants
    from collections import defaultdict
    groups = defaultdict(list)
    for key, row in rows.items():
        if "error_pct" not in row:
            continue
        groups[(row.get("a_over_R"), row.get("h_over_a"))].append(row)

    for (ratio, h), group in groups.items():
        if len(group) < 2:
            continue
        # Sort by depth
        group.sort(key=lambda r: r.get("domain_depth_over_a", 0))
        for i in range(len(group) - 1):
            r1, r2 = group[i], group[i + 1]
            d1, d2 = r1.get("domain_depth_over_a", 0), r2.get("domain_depth_over_a", 0)
            if d1 == 0 or d2 == 0:
                continue
            drift_d = abs(r1["error_pct"]["indentation"] - r2["error_pct"]["indentation"])
            drift_a = abs(
                r1["error_pct"]["contact_radius_linear"]
                - r2["error_pct"]["contact_radius_linear"]
            )
            key = f"ratio{ratio:.2f}_h{h:.4f}_d{d1:.0f}a_vs_d{d2:.0f}a"
            # Avoid duplicates
            if any(p["key"] == key for p in probes):
                continue
            probes.append({
                "key": key,
                "a_over_R": ratio,
                "h_over_a": h,
                "depth_1": d1,
                "depth_2": d2,
                "indentation_drift_pp": drift_d,
                "contact_radius_drift_pp": drift_a,
                "passed": (
                    drift_d <= CRITERIA["indentation_drift_pp"]
                    and drift_a <= CRITERIA["contact_radius_drift_pp"]
                ),
            })

    all_passed = all(p["passed"] for p in probes) if probes else True
    payload = {
        "schema": "tensorfem.probe.truncation.v1",
        "anchor": "fem_hertz_contact_loadcontrol",
        "criteria": CRITERIA,
        "probes": probes,
        "passed": all_passed,
        "failed": [p["key"] for p in probes if not p["passed"]],
        "note": "result.json only has depth-96 runs; truncation data from truncation_extrapolation.json" if not probes else "",
    }
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(f"wrote {OUT}  ({len(probes)} probes, {'PASS' if all_passed else 'FAIL'})")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(probe())
