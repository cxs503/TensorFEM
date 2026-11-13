#!/usr/bin/env python3
"""Arm 3: does the indentation offset vanish when the *vertical* near field is
refined AND the domain depth is extrapolated -- i.e. is it discretisation, or a
residual model bias of the axisymmetric Q4 contact model?

Arms 1 and 2 (``farfield_discretisation.py``, ``nearfield_resolution.py``)
isolated two candidates at the accepted depth 768a:

* far-field stretching (26 radial / 16 vertical graded layers) : rejected,
  refining them moves the indentation error by only ~0.02 pp;
* vertical near-field spacing ``dz = a/6`` (never refined by the ``h/a``
  ladder, which only refines r) : accepted as *a* contributor, -0.480% ->
  -0.310% for ``dz = a/24``.

This arm closes the question on the depth axis:

  A) vertical ladder continued: ``dz = a/48`` at 768a (same radial near field);
  B) depth ladder 96a/192a/384a/768a at ``dz = a/24`` with a refined far field
     (52 radial, 32 vertical), extrapolated with ``_harness.richardson``.

Pre-declared rule (before the numbers are seen, ``CRITERIA``):

  * if ``|e_inf(delta)| <= 0.15%`` -> the whole offset is discretisation, the
    earlier "irreducible model bias" verdict is falsified;
  * if ``|e_inf(delta)|`` stays above 0.15% -> a residual model bias is real and
    is *quantified* here; the arm is then reported as a partial explanation
    (share of the 0.428% that the vertical refinement removes) and the residual
    is declared open instead of being dressed up as converged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))

from _harness import richardson, utc_now, write_json  # noqa: E402

import farfield_discretisation as fd  # noqa: E402

CRITERIA = {
    "discretisation_explained_pct": 0.15,
    "case_pool_pct": 3.0,
    "reference": "Hertz 1882 J. reine angew. Math. 92:156; "
                 "Johnson 1985 Contact Mechanics 4.2 DOI:10.1017/cbo9781139171731",
}
PRIOR_EXTRAPOLATED_BIAS_PCT = 0.42780983887758905  # truncation study, (26,16) far field
JSONL = os.path.join(HERE, "converged_model_rows.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "converged_model.json"))
    args = ap.parse_args()
    open(JSONL, "w").close()

    def dump(row, tag):
        with open(JSONL, "a") as fh:
            fh.write(json.dumps({"tag": tag, "row": row}) + "\n")
        print(f"  {tag:16s} depth={row['depth_over_a']:5.0f}a n_zfine={row['n_zfine']:3d} "
              f"far=({row['n_outer']},{row['n_grow']}) els={row['n_elements']:6d} "
              f"e_a={row['error_pct']['contact_radius_linear']:+.4f}% "
              f"e_d={row['error_pct']['indentation']:+.4f}% "
              f"e_p={row['error_pct']['peak_pressure_consistent']:+.4f}% "
              f"{row['wall_s']:.1f}s", flush=True)

    print("[A] vertical ladder continued: dz = a/48 at 768a")
    fine = fd.run_one(768.0, fd.BASE_OUTER, fd.BASE_GROW, n_zfine=192)
    fine["label"] = "dz_a48"
    dump(fine, "zres_a48")

    print("[B] depth ladder at dz = a/24, far field (52, 32)")
    zrows = []
    for depth in (96.0, 192.0, 384.0, 768.0):
        r = fd.run_one(depth, 52, 32, n_zfine=96)
        r["label"] = f"depth_{depth:.0f}a_dz_a24"
        dump(r, "depth_dz_a24")
        zrows.append(r)

    errs = [r["error_pct"]["indentation"] for r in zrows]
    err_a = [r["error_pct"]["contact_radius_linear"] for r in zrows]
    err_p = [r["error_pct"]["peak_pressure_consistent"] for r in zrows]
    res = richardson(errs[-3], errs[-2], errs[-1], 2.0)
    lim = res.get("richardson_limit")
    e_inf = None if lim is None or not math.isfinite(lim) else float(lim)
    # a and p0 are near-field quantities: check truncation insensitivity instead
    spread_a = max(err_a) - min(err_a)
    spread_p = max(err_p) - min(err_p)

    explained = abs(e_inf) <= CRITERIA["discretisation_explained_pct"] if e_inf is not None else False
    payload = {
        "schema": "tensorfem.hertz-converged-model.v1",
        "anchor": "fem_hertz_contact_loadcontrol",
        "case_pool": "FEM-HERTZ-CONTACT (id 104)",
        "generated_at": utc_now(),
        "question": "does the indentation offset vanish once the vertical near "
                    "field is refined and the depth is extrapolated?",
        "criteria": CRITERIA,
        "config": {"h_over_a": fd.H_REL, "a_over_R": fd.RATIO,
                   "fine_band_depth_over_a": fd.BAND},
        "vertical_ladder_a48": fine,
        "depth_ladder_dz_a24": zrows,
        "extrapolation": {"depths": [r["depth_over_a"] for r in zrows],
                          "errors_pct": errs,
                          "richardson": res,
                          "e_inf_pct": e_inf},
        "near_field_truncation_spread_pp": {"contact_radius": spread_a,
                                            "peak_pressure": spread_p},
        "prior_extrapolated_bias_pct": PRIOR_EXTRAPOLATED_BIAS_PCT,
        "share_of_prior_bias_explained_pct": (
            None if e_inf is None else
            100.0 * (PRIOR_EXTRAPOLATED_BIAS_PCT - abs(e_inf)) / PRIOR_EXTRAPOLATED_BIAS_PCT
        ),
        "verdict": {
            "discretisation_explained": bool(explained),
            "e_inf_delta_pct": e_inf,
            "statement": ("fully discretisation" if explained else
                          "residual model bias quantified; vertical refinement "
                          "removes only part of the prior 0.428%"),
        },
    }
    write_json(Path(args.out), payload)
    print(f"e_inf(delta) = {e_inf} %  (prior -0.4278%), order={res.get('observed_order')}, "
          f"share explained = {payload['share_of_prior_bias_explained_pct']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())