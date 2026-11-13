#!/usr/bin/env python3
"""Arm 2 of the mechanism hunt for the residual Hertz indentation offset:
**is the -0.43% plateau the vertical near-field resolution that the h/a ladder
never refined?**

Background.  The accepted Hertz anchor model builds its mesh as

* radial: uniform ``h = (h/a) a`` over the contact band ``r <= 3a`` -> the
  ``h/a`` ladder 0.03 / 0.015 / 0.0075 refines THIS direction only;
* vertical: a fixed fine band of thickness ``4a`` discretised with a **fixed 24
  layers**, i.e. ``dz = a/6`` regardless of ``h/a``.

So the "mesh-independence" of the indentation plateau was only ever tested in
one direction, at a fixed vertical spacing that is ~11x coarser than the finest
radial spacing.  A bias produced by the vertical discretisation is therefore
*exactly* mesh-independent under the existing ladder -- which is what the
previous studies observed and could not explain.

Hypothesis H_zres (this arm)
    refining only the vertical near field (dz: a/6 -> a/12 -> a/24), with the
    radial near field and the stretched far field bit-identical, moves the
    indentation error below 0.15%.
Alternative H_bias
    the offset survives any near-field refinement -> contact model bias
    (nodal multiplier patch / paraboloid chord / ALM tolerance) stays the
    explanation.

Thresholds are fixed before the numbers are seen (``CRITERIA``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))

from _harness import utc_now, write_json  # noqa: E402

import farfield_discretisation as fd  # noqa: E402

CRITERIA = {
    "zres_accept_pct": 0.15,
    "h_bias_equivalence_pp": 0.05,
    "case_pool_pct": 3.0,
    "reference": "Hertz 1882 J. reine angew. Math. 92:156; "
                 "Johnson 1985 Contact Mechanics 4.2 DOI:10.1017/cbo9781139171731",
}
HYPOTHESIS = (
    "H_zres: the plateau is the vertical near-field spacing dz = a/6, which the "
    "h/a ladder never refined; dz -> a/12, a/24 with everything else identical "
    "moves |e_delta| below 0.15%."
)
ALTERNATIVE = (
    "H_bias: |e_delta| stays within 0.05 pp of the control for every dz -> the "
    "offset is a bias of the contact model itself."
)

#: (n_zfine, n_outer, n_grow, label) -- n_zfine = 24 is the accepted model.
ARMS = [
    (24, fd.BASE_OUTER, fd.BASE_GROW, "control_dz_a6"),
    (48, fd.BASE_OUTER, fd.BASE_GROW, "dz_a12"),
    (96, fd.BASE_OUTER, fd.BASE_GROW, "dz_a24"),
    (96, 52, 32, "dz_a24_far_refined"),
]

JSONL = os.path.join(HERE, "nearfield_vertical_rows.jsonl")


def _solve_arms():
    rows = []
    for n_zfine, n_outer, n_grow, label in ARMS:
        r = fd.run_one(768.0, n_outer, n_grow, n_zfine=n_zfine)
        r["label"] = label
        with open(JSONL, "a") as fh:
            fh.write(json.dumps({"tag": "zres", "row": r}) + "\n")
        rows.append(r)
        print(f"  {label:22s} n_zfine={n_zfine:3d} els={r['n_elements']:6d} "
              f"dz={r['near_field_dz_over_a']:.4f}a dr={r['near_field_dr_over_a']:.4f}a "
              f"e_a={r['error_pct']['contact_radius_linear']:+.4f}% "
              f"e_d={r['error_pct']['indentation']:+.4f}% "
              f"e_p={r['error_pct']['peak_pressure_consistent']:+.4f}% "
              f"{r['wall_s']:.1f}s", flush=True)
    return rows


def combined_verdict(rows):
    """Pre-declared two-sided decision rule for the vertical arm."""
    control = rows[0]["error_pct"]["indentation"]
    final = rows[2]["error_pct"]["indentation"]
    shift = abs(final - control)
    accept = abs(final) <= CRITERIA["zres_accept_pct"]
    reject = shift <= CRITERIA["h_bias_equivalence_pp"]
    d = {
        "control_e_delta_pct": control,
        "dz_a24_e_delta_pct": final,
        "shift_pp": shift,
        "combined_e_delta_pct": rows[3]["error_pct"]["indentation"],
        "accepted": "H_zres" if accept else ("H_bias" if reject else "inconclusive"),
        "rule": "H_zres iff |e_delta(dz=a/24)| <= 0.15%; H_bias iff the dz "
                "refinement shifts e_delta by <= 0.05 pp",
        "sequence_e_delta_pct": [r["error_pct"]["indentation"] for r in rows],
        "sequence_e_a_pct": [r["error_pct"]["contact_radius_linear"] for r in rows],
        "sequence_e_p_pct": [r["error_pct"]["peak_pressure_consistent"] for r in rows],
    }
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "nearfield_resolution.json"))
    ap.add_argument("--from-jsonl", action="store_true",
                    help="rebuild the artifact from nearfield_vertical_rows.jsonl")
    args = ap.parse_args()
    if args.from_jsonl:
        rows = []
        with open(JSONL) as fh:
            for line in fh:
                rows.append(json.loads(line)["row"])
        print(f"[rebuild] {len(rows)} vertical rows")
    else:
        open(JSONL, "w").close()
        rows = _solve_arms()

    arm_verdict = combined_verdict(rows)
    payload = {
        "schema": "tensorfem.hertz-nearfield-vertical.v1",
        "anchor": "fem_hertz_contact_loadcontrol",
        "case_pool": "FEM-HERTZ-CONTACT (id 104)",
        "generated_at": utc_now(),
        "question": "is the residual indentation offset the vertical near-field "
                    "spacing dz=a/6 that the h/a ladder never refined?",
        "hypothesis": HYPOTHESIS,
        "alternative": ALTERNATIVE,
        "criteria": CRITERIA,
        "config": {
            "depth_over_a": 768.0,
            "h_over_a": fd.H_REL,
            "a_over_R": fd.RATIO,
            "fine_band_depth_over_a": fd.BAND,
            "radial_near_field_identical_across_arms": True,
            "far_field_accepted_model": [fd.BASE_OUTER, fd.BASE_GROW],
        },
        "arms": rows,
        "arm_verdict": arm_verdict,
    }
    write_json(Path(args.out), payload)
    print(f"arm_verdict: {arm_verdict['accepted']}  seq e_d="
          f"{[round(v, 4) for v in arm_verdict['sequence_e_delta_pct']]}%")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())