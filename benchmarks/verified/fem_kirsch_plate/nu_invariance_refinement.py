#!/usr/bin/env python3
"""CASE-KIRSCH-PLATE: does the `nu`/`E` invariance of `K_t` survive refinement?

The anchor gates `nu_E_invariance` on a single mesh (n = 24) and measures a
0.71 % spread of `K_t` between `nu = 0`, 0.3 and 0.45.  That number is *not* a
physical dependence: for a plane problem with traction-only boundary data the
stress field is independent of `E` and `nu`, so `K_t = 3` for every elastic
constant.  The spread is the Q4 discretisation's dependence on `nu` at fixed
mesh.  If it is discretisation, it must vanish under refinement; if it is a bug
(wrong constitutive factor, e.g. plane strain creeping into a plane-stress path)
it will stay.

Predeclared decision rule (fixed before the runs)
-------------------------------------------------
  * PASS  (discretisation, asymptotic): ratio <= 0.5, i.e. a p >= 1 decay.
  * INCONCLUSIVE (decays slower than p = 1): 0.5 < ratio < 1.0 -- the spread is
    still shrinking, so a constitutive bug (which would keep it constant or make
    it grow) is not what is being seen, but the mesh is not asymptotic either.
  * FAIL  (constitutive bug): ratio >= 1.0 -- spread does not decay, or grows.
  * Prediction (p = 2 for the recovered Q4 edge stress): ratio ~ 0.25.

Run::

    OMP_NUM_THREADS=6 PYTHONPATH=src python benchmarks/verified/fem_kirsch_plate/nu_invariance_refinement.py
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tensorfem import __version__  # noqa: E402
from tensorfem.finite_width_hole import FiniteWidthPlate, solve_case  # noqa: E402

RATIO = 0.02
LEVELS = tuple(int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ("24", "48")))
POISSONS = (0.0, 0.30, 0.45)
YOUNG_REF = 200.0e9
YOUNG_ALT = 2.0e9
GRADING = 6.0
OUT = HERE / (sys.argv[2] if len(sys.argv) > 2 else "nu_invariance_refinement.json")


def spread(values: list) -> float:
    lo, hi = min(values), max(values)
    mid = 0.5 * (lo + hi)
    return (hi - lo) / mid


def main() -> int:
    rows = []
    for level in LEVELS:
        for nu in POISSONS:
            case = solve_case(FiniteWidthPlate(diameter=RATIO, poisson=nu), n_boundary=level, grading=GRADING)
            rows.append({"n_boundary": level, "poisson": nu, "young_pa": YOUNG_REF, "ktg": case["ktg_fe"]})
        case = solve_case(FiniteWidthPlate(diameter=RATIO, poisson=0.30, young=YOUNG_ALT),
                          n_boundary=level, grading=GRADING)
        rows.append({"n_boundary": level, "poisson": 0.30, "young_pa": YOUNG_ALT, "ktg": case["ktg_fe"]})
        print(f"[nu-refine] n={level:3d} ktg=" +
              " ".join(f"{r['ktg']:.6f}(nu={r['poisson']},E={r['young_pa']:.0e})"
                       for r in rows if r["n_boundary"] == level))

    summary = {}
    for level in LEVELS:
        sel = [r for r in rows if r["n_boundary"] == level]
        nu_rows = [r for r in sel if r["young_pa"] == YOUNG_REF]
        e_row = [r for r in sel if r["young_pa"] == YOUNG_ALT][0]
        nu_ref = [r for r in nu_rows if r["poisson"] == 0.30][0]
        summary[level] = {
            "nu_spread_ktg": spread([r["ktg"] for r in nu_rows]),
            "young_rel_change_ktg": abs(e_row["ktg"] - nu_ref["ktg"]) / nu_ref["ktg"],
        }

    ratio = summary[LEVELS[-1]]["nu_spread_ktg"] / summary[LEVELS[0]]["nu_spread_ktg"]
    if ratio <= 0.5:
        verdict = "PASS_discretisation_asymptotic"
    elif ratio < 1.0:
        verdict = "INCONCLUSIVE_decays_slower_than_p1"
    else:
        verdict = "FAIL_constitutive_bug"
    payload = {
        "schema": "tensorfem.kirsch-nu-invariance-refinement.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tensorfem": __version__,
        "setup": {"ratio_d_over_W": RATIO, "levels": list(LEVELS), "poissons": list(POISSONS),
                  "young_ref": YOUNG_REF, "young_alt": YOUNG_ALT, "grading": GRADING,
                  "physics_claim": "plane stress, traction-only: sigma independent of E and nu"},
        "predeclared": {"pass_if_ratio_le": 0.5, "fail_if_ratio_ge": 1.0,
                        "predicted_ratio_p2": 0.25,
                        "observed_order": math.log(1.0 / ratio) / math.log(2.0)},
        "rows": rows,
        "summary": summary,
        "spread_ratio_48_over_24": ratio,
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[nu-refine] spread " + " ".join(
        f"{lv}={summary[lv]['nu_spread_ktg']*100:.3f}%" for lv in LEVELS) +
        f" ratio={ratio:.3f} p={math.log(1.0/ratio)/math.log(2.0):.3f} -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())