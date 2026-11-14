#!/usr/bin/env python3
"""CASE-KIRSCH-PLATE refinement probe: does the finest-level point error survive n -> 96?

Why this probe
--------------
The anchor `run.py` reports, for `d/W = 0.02`, a *point* error
`|K_t(n=48) - 3|/3 = 4.69e-4` (0.047 %) while its own GCI on the same ladder is
`3.24e-3` (0.32 %) and the two-level Richardson limit sits at `3.0092` — i.e.
*further* from 3 than the finest level itself.  A point error smaller than the
band that is supposed to bound it can only be either (a) a genuinely converged
value whose GCI was computed from a coarse pair (n = 12/24, still in the
pre-asymptotic regime), or (b) a lucky cancellation at n = 48 that reverses on
the next refinement.  The m15393 audit recorded this as finding **F6** and left
the n = 96 probe un-dispatched (cluster job cj1099 never ran).

Predeclared decision rule (fixed before any new solve)
-----------------------------------------------------
Decision: how is the `K_t = 3` accuracy of this case to be *quoted* downstream?
  H1 (point): convergence is asymptotically monotone at order p >= 1.5 beyond
      n = 48 -> quote `|K_t-3|/3` at the finest level plus a Richardson limit.
  H2 (band): the refinement stalls or reverses (error(96) >= error(48)) ->
      the 0.047 % is a lucky point, quote the GCI band instead.
Verdict:
  * H1 if error(96) < error(48) AND observed order p(24,48,96) >= 1.5.
  * H2 if error(96) >= error(48).
  * otherwise inconclusive (reported as such, no reinterpretation after the fact).
Quantitative prediction to be falsified (from the existing 24/48 pair, p = 1.76):
  err(96) ~= err(48) * 2^-1.76 = 1.4e-4.  Accepted if within a factor 2.

Run::

    OMP_NUM_THREADS=6 PYTHONPATH=src python benchmarks/verified/fem_kirsch_plate/refinement_probe_n96.py
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tensorfem import __version__  # noqa: E402
from tensorfem.finite_width_hole import FiniteWidthPlate, solve_case  # noqa: E402

KT = 3.0                  # Kirsch (1898) infinite-plate limit, machine-evaluated
RATIO = 0.02             # closest to the infinite plate in the anchor ladder
LEVELS = (24, 48, 96)     # refinement ratio r = 2, extends the anchor's 12/24/48
GRADING = 6.0
HALF_LENGTH_OVER_WIDTH = 1.5
OUT = HERE / "refinement_probe_n96.json"


def richardson(fine: float, mid: float, coarse: float, r: float = 2.0) -> tuple:
    """Three-level Richardson: observed order, extrapolated limit, fine GCI (FoS 1.25)."""
    d1, d2 = mid - coarse, fine - mid
    if d1 == 0.0 or d2 / d1 <= 0.0:
        return float("nan"), float("nan"), float("nan")
    p = math.log(abs(d1 / d2)) / math.log(r)
    limit = fine + (fine - mid) / (r**p - 1.0)
    gci_fine = 1.25 * abs(fine - mid) / (r**p - 1.0) / abs(fine)
    return p, limit, gci_fine


def main() -> int:
    plate = FiniteWidthPlate(diameter=RATIO, half_length=HALF_LENGTH_OVER_WIDTH)
    rows = []
    for level in LEVELS:
        started = time.time()
        case = solve_case(plate, n_boundary=level, grading=GRADING)
        rows.append(
            {
                "n_boundary": level,
                "ktg": case["ktg_fe"],
                "ktn": case["ktn_fe"],
                "rel_error_vs_kirsch": abs(case["ktg_fe"] - KT) / KT,
                "load_balance_rel": case["load_balance"]["relative_error"],
                "wall_seconds": time.time() - started,
            }
        )
        print(f"[probe] d/W={RATIO} n={level:3d} K_t={case['ktg_fe']:.9f} "
              f"err={rows[-1]['rel_error_vs_kirsch']*100:.4f}% "
              f"({rows[-1]['wall_seconds']:.1f} s)")

    e24, e48, e96 = (r["rel_error_vs_kirsch"] for r in rows)
    p, limit, gci = richardson(rows[2]["ktg"], rows[1]["ktg"], rows[0]["ktg"])
    pred96 = e48 * 2.0**-1.76

    if e96 < e48 and p >= 1.5:
        verdict, which = "H1_point_defensible", "point"
    elif e96 >= e48:
        verdict, which = "H2_point_luck__quote_band", "band"
    else:
        verdict, which = "inconclusive", "band"

    payload = {
        "schema": "tensorfem.kirsch-refinement-probe.v1",
        "probe": "CASE-FEM-KIRSCH-PLATE refinement probe n=96 (closes m15393 finding F6)",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "host": platform.node(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "tensorfem": __version__,
        },
        "setup": {
            "ratio_d_over_W": RATIO,
            "levels": list(LEVELS),
            "refinement_ratio": 2.0,
            "grading": GRADING,
            "half_length_over_width": HALF_LENGTH_OVER_WIDTH,
            "reference_kt": KT,
            "reference": "Kirsch (1898) VDI 42, 797; Timoshenko & Goodier 3rd ed. Art. 39 (evaluated in code)",
        },
        "predeclared": {
            "hypotheses": {
                "H1_point": "error(96) < error(48) and p(24,48,96) >= 1.5 -> point value quotable",
                "H2_band": "error(96) >= error(48) -> 0.047 % is lucky, quote the GCI band",
            },
            "prediction_err96_from_p1.76": pred96,
            "prediction_accept_factor": 2.0,
        },
        "rows": rows,
        "summary": {
            "error_n24": e24,
            "error_n48": e48,
            "error_n96": e96,
            "error_ratio_successive_24_48": e48 / e24,
            "error_ratio_successive_48_96": e96 / e48,
            "observed_order_24_48_96": p,
            "richardson_limit": limit,
            "gci_fine_band": gci,
            "richardson_limit_rel_error": abs(limit - KT) / KT,
            "prediction_met": abs(e96 - pred96) / pred96 <= 2.0,
        },
        "verdict": verdict,
        "quoted_accuracy_basis": which,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[probe] e24={e24*100:.4f}% e48={e48*100:.4f}% e96={e96*100:.4f}% "
          f"p={p:.3f} richardson={limit:.6f} GCI={gci*100:.3f}% -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())