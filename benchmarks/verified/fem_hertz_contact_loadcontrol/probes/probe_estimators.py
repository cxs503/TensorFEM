#!/usr/bin/env python3
"""Probe: contact-radius estimator consistency across the three methods.

The anchor records three estimators for the contact radius:
  * ``contact_radius_linear``  -- linear zero-crossing between the last two
    active nodes (the primary estimator used in the gates);
  * ``contact_radius_fit``     -- least-squares fit of p^2 vs r^2 over the
    active set;
  * ``contact_radius_fit_consistent`` -- same fit but using the mass-matrix
    consistent pressure field.

This probe grades the *spread* among the three estimators on the **finest
mesh** (h/a = 0.0075), where all three should agree.  On coarser meshes the
consistent-pressure estimator is known to be noisy and is excluded from the
spread check.

Usage
-----
    python benchmarks/verified/fem_hertz_contact_loadcontrol/probes/probe_estimators.py
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(HERE, "..", "result.json")
OUT = os.path.join(HERE, "estimators_probe.json")

CRITERIA = {
    "spread_pct_finest": 2.0,
    "reference": "on the finest mesh (h/a=0.0075) all three estimators must agree within 2% of the mean",
}


def probe():
    with open(RESULT) as fh:
        result = json.load(fh)

    rows = result.get("runs", {})
    probes = []
    for key, row in sorted(rows.items()):
        num = row.get("numeric", {})
        if not all(k in num for k in ("contact_radius_linear", "contact_radius_fit",
                                       "contact_radius_fit_consistent")):
            continue
        a_lin = num["contact_radius_linear"]
        a_fit = num["contact_radius_fit"]
        a_fit_c = num["contact_radius_fit_consistent"]
        mean_a = (a_lin + a_fit + a_fit_c) / 3.0
        spread = max(abs(a_lin - mean_a), abs(a_fit - mean_a), abs(a_fit_c - mean_a))
        spread_pct = 100.0 * spread / mean_a if mean_a > 0 else float("inf")
        h_over_a = row.get("h_over_a", 1.0)
        is_finest = abs(h_over_a - 0.0075) < 1e-6
        # Only grade the finest mesh; coarser meshes are informational
        passed = spread_pct <= CRITERIA["spread_pct_finest"] if is_finest else True
        probes.append({
            "key": key,
            "a_over_R": row.get("a_over_R"),
            "h_over_a": h_over_a,
            "contact_radius_linear": a_lin,
            "contact_radius_fit": a_fit,
            "contact_radius_fit_consistent": a_fit_c,
            "mean": mean_a,
            "spread_pct": spread_pct,
            "is_finest": is_finest,
            "passed": passed,
        })

    # Only the finest-mesh probes count for the overall verdict
    finest_probes = [p for p in probes if p["is_finest"]]
    all_passed = all(p["passed"] for p in finest_probes) if finest_probes else True
    payload = {
        "schema": "tensorfem.probe.estimators.v1",
        "anchor": "fem_hertz_contact_loadcontrol",
        "criteria": CRITERIA,
        "probes": probes,
        "finest_probes": finest_probes,
        "passed": all_passed,
        "failed": [p["key"] for p in finest_probes if not p["passed"]],
    }
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(f"wrote {OUT}  ({len(probes)} probes, {len(finest_probes)} finest, {'PASS' if all_passed else 'FAIL'})")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(probe())
