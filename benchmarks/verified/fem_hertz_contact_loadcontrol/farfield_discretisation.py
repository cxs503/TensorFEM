#!/usr/bin/env python3
"""Is the residual indentation bias of the Hertz case a *model* bias or a
far-field discretisation artefact?

Background (``docs/CASE-FEM-HERTZ-CONTACT-truncation-extrapolation.md``):
the load-controlled Hertz anchor leaves a **-0.428%** indentation error after
Richardson extrapolation of the domain depth to infinity.  That number was
recorded as an *irreducible model bias* but its mechanism was never identified;
the study's own ``limitations`` list says so.  The depth ladder moved the
truncation boundary by **stretching 16 fixed far-field layers** (and 26 fixed
outer radial layers) instead of refining them, so the ladder confounds two
effects:

* H_trunc : boundary too close (decays as depth -> inf)          [tested before]
* H_far   : the far field is *stretched*, not refined; a bilinear
            Q4 layer of thickness ~460a cannot represent a field that decays
            like 1/z, and the discretisation error of a self-similar stretch
            SATURATES as depth -> inf (i.e. it survives extrapolation, and it
            is independent of the near-field h/a).                  [this study]
* H_bias  : genuine bias of the contact model itself (nodal multiplier patch,
            paraboloid chord, ALM tolerance).

Design (thresholds fixed before the numbers are seen -- see ``CRITERIA``):

  D  : decide which of H_far / H_bias explains the extrapolated -0.428%.
  C  : the existing accepted model, depth 768a, h/a=0.015, far field
       (26 radial, 16 vertical) = the *control*; it must reproduce -0.4804%.
  I1 : identical near field, far field refined to (52, 32)
  I2 : identical near field, far field refined to (104, 64)
  I3 : depth ladder 96a/192a/384a/768a with the refined far field (I2 knobs),
       re-extrapolated with the repo harness ``richardson``.

Predeclared decision rule
-------------------------
  at fixed depth 768a:  |e_delta| drops below 0.15%  -> H_far accepted
                        |e_delta| stays within +-0.05 pp of the control
                                                   -> H_bias accepted
                        anything in between        -> inconclusive (reported)
  and the re-extrapolated limit |e_inf(delta)| < 0.10% corroborates H_far.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))

import numpy as np  # noqa: E402

from _harness import richardson, utc_now, write_json  # noqa: E402

from tensorfem.hertz_halfspace_loadcontrol import (  # noqa: E402
    graded_depth,
    hertz_load_reference,
    solve_axisymmetric_contact_load_control,
    two_zone_axis,
)

R = 10.0
E = 1.0e5
NU = 0.3
H_REL = 0.015            # accepted near-field spacing h/a
RATIO = 0.05             # a/R of the primary load point
BAND = 4.0               # thickness of the fine vertical band, in units of a
N_ZFINE = 24             # surface-band vertical elements (unchanged: near field)
BASE_OUTER, BASE_GROW = 26, 16          # accepted model (control)

#: Thresholds, declared before any solve of this study is run.
CRITERIA = {
    "far_bias_accept_pct": 0.15,
    "h_bias_equivalence_pp": 0.05,
    "extrapolated_limit_accept_pct": 0.10,
    "case_pool_pct": 3.0,
    "reference": "Hertz 1882 J. reine angew. Math. 92:156; "
                 "Johnson 1985 Contact Mechanics 4.2 DOI:10.1017/cbo9781139171731",
}
HYPOTHESIS = (
    "H_far: the extrapolated -0.428% indentation offset is the discretisation "
    "error of the *stretched* far field (26 radial / 16 vertical geometrically "
    "graded Q4 layers spanning up to 768a); refining only that region, with the "
    "near field bit-identical, moves |e_delta| below 0.15%."
)
ALTERNATIVE = (
    "H_bias: the offset is a bias of the contact model itself and is invariant "
    "to far-field refinement (|e_delta| stays within 0.05 pp of -0.4804%)."
)


def load_for_radius_ratio(ratio):
    e_star = E / (1.0 - NU**2)
    a = ratio * R
    return 4.0 * e_star * a**3 / (3.0 * R)


def mesh_for(depth_factor, h_rel, n_outer, n_grow, growth=1.6, n_zfine=N_ZFINE):
    """Same construction as ``run.py::run_case`` with the two far-field layer
    counts exposed.  ``n_outer``/``n_grow`` only change the stretched outer
    region: the contact band (3a of uniform h/a) and the 4a x 24 fine vertical
    band are untouched, so the near-field resolution is bit-identical."""
    ref = hertz_load_reference(load_for_radius_ratio(RATIO), R, E, NU)
    a = ref.contact_radius
    h = h_rel * a
    n_inner = int(round(3.0 * a / h))
    r_axis = two_zone_axis(3.0 * a, depth_factor * a, n_inner, n_outer)
    z_axis = graded_depth(depth_factor * a, BAND * a, n_zfine, n_grow, growth=growth)
    return ref, r_axis, z_axis


def run_one(depth_factor, n_outer, n_grow, rho_factor=1.0, growth=1.6, n_zfine=N_ZFINE):
    ref, r_axis, z_axis = mesh_for(depth_factor, H_REL, n_outer, n_grow, growth, n_zfine)
    t0 = time.time()
    ref, state, diag = solve_axisymmetric_contact_load_control(
        load=ref.load, radius=R, young=E, poisson=NU,
        r_axis=r_axis, z_axis=z_axis, rho_factor=rho_factor, load_tol=1e-10,
    )
    wall = time.time() - t0
    e_a = 100.0 * (state.contact_radius_linear / ref.contact_radius - 1.0)
    e_d = 100.0 * (state.prescribed_indentation / ref.indentation - 1.0)
    e_p = 100.0 * (state.metadata["peak_pressure_consistent"] / ref.peak_pressure - 1.0)
    # largest far-field element aspect ratio (radial width / vertical height)
    dr = np.diff(r_axis)
    dz = -np.diff(z_axis)
    return {
        "depth_over_a": depth_factor,
        "n_outer": n_outer,
        "n_grow": n_grow,
        "n_zfine": n_zfine,
        "near_field_dz_over_a": float(-(z_axis[1] - z_axis[0]) / ref.contact_radius),
        "near_field_dr_over_a": float((r_axis[1] - r_axis[0]) / ref.contact_radius),
        "growth": growth,
        "rho_factor": rho_factor,
        "h_over_a": H_REL,
        "a_over_R": RATIO,
        "load": ref.load,
        "n_elements": diag["n_elements"],
        "ndof": diag["ndof"],
        "wall_s": wall,
        "load_residual": diag["load_residual"],
        "penetration_over_delta": diag["penetration_over_indentation"],
        "equilibrium_residual_rel": state.metadata["equilibrium_residual_rel"],
        "max_far_radial_element": float(dr[-1]),
        "max_far_vertical_element": float(dz[-1]),
        "far_aspect_ratio": float(dr[-1] / dz[-1]),
        "analytic": {
            "contact_radius": ref.contact_radius,
            "indentation": ref.indentation,
            "peak_pressure": ref.peak_pressure,
        },
        "numeric": {
            "contact_radius_linear": state.contact_radius_linear,
            "indentation": state.prescribed_indentation,
            "peak_pressure_consistent": state.metadata["peak_pressure_consistent"],
            "reaction": state.reaction,
        },
        "error_pct": {
            "contact_radius_linear": e_a,
            "indentation": e_d,
            "peak_pressure_consistent": e_p,
        },
    }


JSONL = os.path.join(HERE, "farfield_rows.jsonl")


def _append(row, tag):
    """Incremental dump so a killed shell never costs a finished solve."""
    with open(JSONL, "a") as fh:
        fh.write(json.dumps({"tag": tag, "row": row}) + "\n")


def farfield_ladder():
    """Control vs two far-field refinements at the accepted depth 768a."""
    rows = []
    for n_outer, n_grow in ((BASE_OUTER, BASE_GROW), (52, 32), (104, 64)):
        rows.append(run_one(768.0, n_outer, n_grow))
        _append(rows[-1], "farfield")
        r = rows[-1]
        print(f"  depth 768a far=({n_outer},{n_grow}) els={r['n_elements']:6d} "
              f"e_a={r['error_pct']['contact_radius_linear']:+.4f}% "
              f"e_d={r['error_pct']['indentation']:+.4f}% "
              f"e_p={r['error_pct']['peak_pressure_consistent']:+.4f}% "
              f"{r['wall_s']:.1f}s", flush=True)
    return rows


def depth_ladder(n_outer, n_grow):
    rows = []
    for depth in (96.0, 192.0, 384.0, 768.0):
        rows.append(run_one(depth, n_outer, n_grow))
        _append(rows[-1], "depth_refined")
        r = rows[-1]
        print(f"  refined far=({n_outer},{n_grow}) depth {depth:5.0f}a "
              f"els={r['n_elements']:6d} e_d={r['error_pct']['indentation']:+.4f}% "
              f"{r['wall_s']:.1f}s", flush=True)
    return rows


def extrapolate(rows):
    """Richardson limit of the indentation error as depth -> inf.

    The ladder doubles the domain depth (refinement ratio 2 in ``1/depth`` is
    not a mesh ratio, so the order is reported as an *empirical decay order*);
    the arithmetic itself is the repo harness ``_harness.richardson`` on the
    last three members of the ladder, exactly as the earlier truncation study
    did, so the two extrapolations are comparable.
    """
    depths = [r["depth_over_a"] for r in rows]
    errs = [r["error_pct"]["indentation"] for r in rows]
    out = {"depths": depths, "errors_pct": errs}
    if len(errs) >= 3:
        res = richardson(errs[-3], errs[-2], errs[-1], 2.0)
        out["richardson"] = res
        lim = res.get("richardson_limit")
        out["e_inf_pct"] = None if lim is None or not math.isfinite(lim) else float(lim)
        out["order"] = res.get("observed_order")
        out["member_of_ladder"] = depths[-3:]
    else:
        out["e_inf_pct"] = None
    return out


def verdict(far_rows, ext_base, ext_refined):
    control = far_rows[0]["error_pct"]["indentation"]
    refined = max(far_rows, key=lambda r: r["n_elements"])["error_pct"]["indentation"]
    shift_pp = abs(refined - control)
    accept_far = abs(refined) <= CRITERIA["far_bias_accept_pct"]
    accept_bias = shift_pp <= CRITERIA["h_bias_equivalence_pp"]
    if accept_far and ext_refined.get("e_inf_pct") is not None:
        accept_far = abs(ext_refined["e_inf_pct"]) <= CRITERIA["extrapolated_limit_accept_pct"]
    return {
        "control_e_delta_pct": control,
        "refined_e_delta_pct": refined,
        "shift_pp": shift_pp,
        "extrapolated_limit_baseline_pct": ext_base.get("e_inf_pct"),
        "extrapolated_limit_refined_pct": ext_refined.get("e_inf_pct"),
        "accepted": "H_far" if accept_far else ("H_bias" if accept_bias else "inconclusive"),
        "rule": "H_far iff |e_delta| <= 0.15% at 768a and |e_inf| <= 0.10%; "
                "H_bias iff the far refinement shifts e_delta by <= 0.05 pp",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "farfield_discretisation.json"))
    ap.add_argument("--skip-depth-ladder", action="store_true")
    ap.add_argument("--from-jsonl", action="store_true",
                    help="rebuild the artifact from farfield_rows.jsonl (zero compute)")
    args = ap.parse_args()

    if not args.from_jsonl:
        open(JSONL, "w").close()
    if args.from_jsonl:
        # Zero-compute rebuild: every solve of this study already appended its
        # full row to the JSONL, so the artifact is recoverable without re-running.
        far, ref_rows = [], []
        with open(JSONL) as fh:
            for line in fh:
                rec = json.loads(line)
                (far if rec["tag"] == "farfield" else ref_rows).append(rec["row"])
        print(f"[rebuild] {len(far)} far-field rows, {len(ref_rows)} depth rows")
    else:
        print("[1] far-field refinement at fixed depth 768a, h/a=0.015, a/R=0.05")
        far = farfield_ladder()

    ext_base = {"e_inf_pct": -0.42780983887758905}  # prior study, 4 levels @ (26,16)
    ext_ref = extrapolate(ref_rows) if (args.from_jsonl and ref_rows) else {"e_inf_pct": None}
    if not args.from_jsonl and not args.skip_depth_ladder:
        print("[2] depth ladder with refined far field (104, 64)")
        ref_rows = depth_ladder(104, 64)
        ext_ref = extrapolate(ref_rows)
        print(f"  refined Richardson limit e_inf(delta) = {ext_ref['e_inf_pct']} %")
    else:
        ref_rows = []

    v = verdict(far, ext_base, ext_ref)
    payload = {
        "schema": "tensorfem.hertz-farfield-study.v1",
        "anchor": "fem_hertz_contact_loadcontrol",
        "case_pool": "FEM-HERTZ-CONTACT (id 104)",
        "generated_at": utc_now(),
        "question": "is the extrapolated -0.428% indentation offset a model bias "
                    "or the discretisation error of the stretched far field?",
        "hypothesis": HYPOTHESIS,
        "alternative": ALTERNATIVE,
        "criteria": CRITERIA,
        "config": {
            "R": R, "E": E, "nu": NU, "h_over_a": H_REL, "a_over_R": RATIO,
            "fine_band_depth_over_a": BAND, "n_zfine": N_ZFINE,
            "accepted_model_far_field": [BASE_OUTER, BASE_GROW],
            "near_field_identical_across_all_runs": True,
        },
        "farfield_ladder": far,
        "depth_ladder_refined": ref_rows,
        "extrapolation_baseline_prior": ext_base,
        "extrapolation_refined": ext_ref,
        "arm_verdict": v,
    }
    write_json(Path(args.out), payload)
    print(f"arm_verdict: {v['accepted']}  control={v['control_e_delta_pct']:+.4f}% "
          f"refined={v['refined_e_delta_pct']:+.4f}% shift={v['shift_pp']:.4f}pp "
          f"e_inf_refined={v['extrapolated_limit_refined_pct']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())