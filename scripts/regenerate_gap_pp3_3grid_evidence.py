#!/usr/bin/env python
"""Regenerate the GAP-PP-3 3-grid evidence bundle's *derived* columns.

Defect being pinned (verification debt, mission m23702)
--------------------------------------------------------
The committed ``docs/assets/gap_pp3_3grid_results.json`` was produced by an
older ``gap_pp3_3grid_study.run_mesh`` that computed ``limit_load_error_vs_ub``
against the **Ilyushin sine upper bound** (``bounds.ilyushin_sine_upper_pressure``)
instead of the **yield-line upper bound** (``analytic.yield_line_upper_pressure``)
that the criteria block and the auditor use.  The result is an internally
inconsistent bundle:

* ``meshes[<n>].limit_load_error_vs_ub`` = ``(limit - sine_UB)/sine_UB``
  (e.g. -0.1743 for 24x12) while
* ``meshes[<n>].limit_load_measured`` and ``criteria.limit_lt_10pct`` are
  expressed against the yield-line UB (``analytic.yield_line_upper_pressure``).

The independent auditor (``scripts/audit_gap_pp3_3grid_bundle.py``, check
``M4-limit-error``) therefore flags every mesh as BLOCKING-inconsistent:
``(limit - yield_line_UB)/yield_line_UB`` does not equal the stored
``limit_load_error_vs_ub``.

This driver re-derives the *derived* column from the bundle's own primitives
(the same identity the auditor checks) and re-runs ``finalize`` so the
``criteria`` block is regenerated from the corrected per-mesh entries.  It does
**not** touch any raw FE number (``limit_load_measured``, ``first_yield_factor``,
``permanent_deflection``, the ``path``) -- those are the solved primitives and
stay as they are.

Usage
-----
    python scripts/regenerate_gap_pp3_3grid_evidence.py \
        docs/assets/gap_pp3_3grid_results.json --out docs/assets/gap_pp3_3grid_results.json

Exit code 0 iff the regenerated bundle passes the independent auditor.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gap_pp3_3grid_study import finalize, write  # noqa: E402
from audit_gap_pp3_3grid_bundle import audit_bundle  # noqa: E402


def regenerate(bundle: dict) -> dict:
    """Re-derive ``limit_load_error_vs_ub`` and the ``criteria`` block.

    The reference upper bound is the one the criteria block and the auditor
    use: ``analytic.yield_line_upper_pressure``.  Every mesh's
    ``limit_load_error_vs_ub`` is recomputed as
    ``(limit_load_measured - yield_line_UB)/yield_line_UB`` and the ``criteria``
    block is regenerated from the corrected per-mesh entries by ``finalize``.
    """
    ub = bundle["analytic"]["yield_line_upper_pressure"]
    for name, mesh in bundle["meshes"].items():
        limit = mesh.get("limit_load_measured")
        if limit is not None:
            mesh["limit_load_error_vs_ub"] = (float(limit) - float(ub)) / float(ub)
    # finalize regenerates the criteria block from the (now consistent) entries
    finalize(bundle)
    return bundle


def main(path: str, out: str) -> int:
    with open(path) as handle:
        bundle = json.load(handle)
    before = audit_bundle(bundle)
    print(f"before: ok={before['ok']}  "
          f"blocking={[f['check'] for f in before['findings'] if not f['ok'] and f['severity']=='blocking']}")
    regenerate(bundle)
    after = audit_bundle(bundle)
    print(f"after : ok={after['ok']}  "
          f"blocking={[f['check'] for f in after['findings'] if not f['ok'] and f['severity']=='blocking']}")
    if not after["ok"]:
        print("regenerated bundle still fails the auditor; not writing")
        return 1
    write(bundle, out)
    print(f"wrote {os.path.abspath(out)}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="input bundle JSON")
    ap.add_argument("--out", default=None, help="output bundle JSON (default: in place)")
    args = ap.parse_args()
    out = args.out or args.path
    sys.exit(main(args.path, out))
