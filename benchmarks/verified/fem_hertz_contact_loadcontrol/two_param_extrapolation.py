#!/usr/bin/env python3
"""Is the residual Hertz indentation offset a *model* bias, or the joint
discretisation error of two parameters that were never varied together?

State of the evidence (``docs/CASE-FEM-HERTZ-CONTACT-plateau-mechanism.md``,
``docs/CASE-FEM-HERTZ-CONTACT-truncation-extrapolation.md``):

* the depth ladder (domain depth / a = 96 -> 768) was only ever run at **one**
  vertical near-field spacing ``dz = a/24``; extrapolated to infinite depth it
  left ``e_delta = -0.2092%``, recorded as an *unlocated* residual (the study's
  own F2 is an honest FAIL: "mechanism not identified");
* the vertical ladder (``dz = a/6 -> a/48``) was only ever run at **one** depth
  (768a); it converged to ``e_delta ~ -0.295%`` in ``dz`` alone;
* the two ladders were additionally run with *different far-field
  stretching* ((26,16) vs (52,32) layers), a 0.022 pp confound.

So the "-0.209%" number is the limit of a **one-parameter** extrapolation of a
function of two parameters: ``e_delta = f(dz, a/H)``.  Neither parameter alone
reaches zero, and the joint limit was never evaluated.  This study evaluates it.

Model (pre-declared before any solve):

    e_delta(dz, H) = e0 + A (dz/a)^p + B (a/H)^q ,   p, q, A, B, e0 free

``e0`` is the residual that no amount of near-field refinement and no amount of
domain depth can remove -- the only well-posed definition of "model bias".

Pre-declared decision rule (``CRITERIA``; fixed before the numbers are seen):

  H_disc : the claim "there is an irreducible model bias of ~0.2%" is
           **falsified**, i.e. ``|e0| <= 0.05%`` *and* the propagated
           uncertainty band around ``e0`` contains 0;
  H_bias : the model bias is **confirmed**, ``|e0| >= 0.15%`` *and* the band
           excludes 0;
  otherwise ``inconclusive`` (reported honestly, no upgrade of either claim).

Both are reported together with the pool criterion (``<3%`` on ``a`` and
``delta``), which is decided on the accepted model, not on ``e0``.

Grid (radial near field, load, ``a/R``, band thickness bit-identical to the
accepted model; far field fixed at the (52,32) configuration that the far-field
study showed to be converged in the far field itself -- this removes the
0.022 pp confound of the earlier ladders):

    dz ladder  (H = 768a) : dz = a/6, a/12, a/24, a/48
    H  ladder  (dz = a/12): H  = 96a, 192a, 384a, 768a
    H  ladder  (dz = a/24): H  = 96a, 192a, 384a, 768a   [reused artifacts]
    H  ladder  (dz = a/48): H  = 96a, 192a, 384a, 768a

The ``dz = a/24`` depth ladder already exists in
``converged_model_rows.jsonl`` with the same far field, so it is *read* back
(no hand-copied numbers) instead of recomputed.

Usage
-----
    python3 two_param_extrapolation.py --list
    python3 two_param_extrapolation.py --shard I --of N     # solve one cell
    python3 two_param_extrapolation.py --assemble           # zero-compute
    python3 two_param_extrapolation.py --assemble --figure  # + PNG
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))

from _harness import richardson, utc_now, write_json  # noqa: E402

CELLDIR = os.path.join(HERE, "two_param_cells")
ARTIFACT = os.path.join(HERE, "two_param_extrapolation.json")
REUSE_JSONL = os.path.join(HERE, "converged_model_rows.jsonl")
FIGDIR = os.path.join(HERE, "..", "..", "..", "docs", "assets",
                      "case-fem-hertz-contact")

#: far-field stretching held fixed across the whole study (see module docstring)
FAR = (52, 32)

#: (label, depth_over_a, n_zfine)  -- n_zfine=depth-band layers; dz = 4a/n_zfine
CELLS = [
    ("dz_a6_H768", 768.0, 24),
    ("dz_a12_H768", 768.0, 48),
    ("dz_a48_H96", 96.0, 192),
    ("dz_a48_H192", 192.0, 192),
    ("dz_a48_H384", 384.0, 192),
    ("dz_a48_H768", 768.0, 192),
    ("dz_a12_H96", 96.0, 48),
    ("dz_a12_H192", 192.0, 48),
    ("dz_a12_H384", 384.0, 48),
]

CRITERIA = {
    "disc_abs_pct": 0.05,
    "bias_abs_pct": 0.15,
    "case_pool_pct": 3.0,
    "reference": "Hertz 1882 J. reine angew. Math. 92:156; "
                 "Johnson 1985 Contact Mechanics 4.2 DOI:10.1017/cbo9781139171731",
}


# --------------------------------------------------------------------------- #
# solve side
# --------------------------------------------------------------------------- #
def cell_path(label):
    return os.path.join(CELLDIR, label + ".json")


def run_cell(idx):
    """Solve one grid cell and persist it (idempotent: existing cells are kept)."""
    import farfield_discretisation as fd

    label, depth, n_zfine = CELLS[idx]
    path = cell_path(label)
    if os.path.exists(path):
        print(f"[skip] {label} already solved")
        return 0
    n_outer, n_grow = FAR
    row = fd.run_one(depth, n_outer, n_grow, n_zfine=n_zfine)
    row["label"] = label
    row["far_field_layers"] = [n_outer, n_grow]
    os.makedirs(CELLDIR, exist_ok=True)
    write_json(Path(path), {"schema": "tensorfem.hertz-two-param-cell.v1",
                      "label": label, "far_field_layers": [n_outer, n_grow],
                      "generated_at": utc_now(), "row": row})
    print(f"[done] {label} depth={depth}a nz={n_zfine} "
          f"els={row['n_elements']} e_d="
          f"{row['error_pct']['indentation']:+.4f}% "
          f"e_a={row['error_pct']['contact_radius_linear']:+.4f}% "
          f"{row['wall_s']:.0f}s", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# assemble side (zero compute)
# --------------------------------------------------------------------------- #
def load_cells():
    """All (dz/a, a/H, e_delta%, e_a%, e_p%, source) points of the grid."""
    pts = []
    for label, depth, n_zfine in CELLS:
        path = cell_path(label)
        if not os.path.exists(path):
            continue
        row = json.load(open(path))["row"]
        pts.append({
            "label": label,
            "depth_over_a": row["depth_over_a"],
            "dz_over_a": row["near_field_dz_over_a"],
            "e_delta_pct": row["error_pct"]["indentation"],
            "e_a_linear_pct": row["error_pct"]["contact_radius_linear"],
            "e_a_fit_pct": row["error_pct"].get("contact_radius_fit"),
            "e_p_pct": row["error_pct"]["peak_pressure_consistent"],
            "n_elements": row["n_elements"],
            "wall_s": row["wall_s"],
            "source": "cell",
        })
    # reuse the dz=a/24 depth ladder from the committed artifact (same far field)
    if os.path.exists(REUSE_JSONL):
        for line in open(REUSE_JSONL):
            rec = json.loads(line)
            row = rec.get("row", rec)
            if rec.get("tag") != "depth_dz_a24":
                continue
            if [row.get("n_outer"), row.get("n_grow")] != list(FAR):
                continue
            pts.append({
                "label": row.get("label", "depth_dz_a24"),
                "depth_over_a": row["depth_over_a"],
                "dz_over_a": row["near_field_dz_over_a"],
                "e_delta_pct": row["error_pct"]["indentation"],
                "e_a_linear_pct": row["error_pct"]["contact_radius_linear"],
                "e_a_fit_pct": row["error_pct"].get("contact_radius_fit"),
                "e_p_pct": row["error_pct"]["peak_pressure_consistent"],
                "n_elements": row["n_elements"],
                "wall_s": row["wall_s"],
                "source": "reuse:converged_model_rows.jsonl",
            })
    return pts


def _three(points, key):
    """Pick the three finest available approximations in one parameter."""
    return [p[key] for p in points]


def depth_limit(points, dz, window=(192.0, 384.0, 768.0)):
    """One-parameter depth extrapolation at a fixed dz.

    ``window`` selects which three domain depths feed the Richardson
    extrapolation; running two windows is the honest sensitivity of the
    extrapolation to a model assumption the data cannot decide.
    """
    sub = [p for p in points if abs(p["dz_over_a"] - dz) < 1e-9]
    got = {p["depth_over_a"]: p for p in sub}
    if not all(d in got for d in window):
        return None
    e = [got[d]["e_delta_pct"] for d in window]
    r = richardson(e[0], e[1], e[2], refinement_ratio=2.0)
    r["depths_over_a"] = list(window)
    r["e_delta_pct"] = e
    return r


WINDOWS = {"upper": (192.0, 384.0, 768.0), "lower": (96.0, 192.0, 384.0)}
DZ_LEVELS = (1.0 / 12.0, 1.0 / 24.0, 1.0 / 48.0)


def nested(points, window_key="upper"):
    """Nested (depth, then dz) extrapolation -- the primary estimate of e0."""
    w = WINDOWS[window_key]
    limits = {}
    for dz in DZ_LEVELS:
        limits[dz] = depth_limit(points, dz, window=w)
    tag = {1.0 / 12.0: "dz_a12", 1.0 / 24.0: "dz_a24", 1.0 / 48.0: "dz_a48"}
    out = {"window": window_key, "window_depths": list(w),
           "depth_limits": {tag[dz]: limits[dz] for dz in DZ_LEVELS}}
    if not all(limits.values()):
        out["status"] = "incomplete grid"
        return out
    lim = [limits[1.0 / 12.0]["richardson_limit"],
           limits[1.0 / 24.0]["richardson_limit"],
           limits[1.0 / 48.0]["richardson_limit"]]
    r = richardson(lim[0], lim[1], lim[2], refinement_ratio=2.0)
    e0 = r["richardson_limit"]
    # Roache GCI of the outer (dz) extrapolation is relative -> absolute band
    gci = r.get("gci_fine")
    band_gci = abs(gci * e0) if (gci is not None and math.isfinite(gci)) else float("nan")
    out.update({
        "status": "ok",
        "e0_pct": e0,
        "dz_observed_order": r["observed_order"],
        "dz_gci_relative": gci,
        "band_gci_pct": band_gci,
        "limits_by_dz_pct": lim,
        "depth_orders": [limits[dz]["observed_order"] for dz in DZ_LEVELS],
    })
    return out


def dz_limit_at(points, depth):
    """dz extrapolation at a fixed depth from (a/6, a/12, a/24)."""
    sub = sorted([p for p in points if abs(p["depth_over_a"] - depth) < 1e-9],
                 key=lambda p: -p["dz_over_a"])
    by = {round(p["dz_over_a"], 6): p for p in sub}
    want = [1.0 / 6.0, 1.0 / 12.0, 1.0 / 24.0]
    if not all(round(w, 6) in by for w in want):
        return None
    e = [by[round(w, 6)]["e_delta_pct"] for w in want]
    r = richardson(e[0], e[1], e[2], refinement_ratio=2.0)
    r["dz_over_a"] = want
    r["e_delta_pct"] = e
    return r


def interim(points):
    """Corner-additivity estimate of the joint limit, used *only* when the
    finest-dz depth ladder did not land.

    With one-parameter limits known in each direction the first-order correction
    of the corner point is  e0 ~ depth_limit(dz) + [dz_limit(H) - e(dz,H)].
    This is an estimate, NOT the pre-declared quantity: it is reported with the
    intermediate numbers so the reader can see it is a linearisation of a
    two-parameter surface, and the verdict stays "inconclusive" while the grid
    is incomplete.
    """
    dl = depth_limit(points, 1.0 / 24.0)
    zl = dz_limit_at(points, 768.0)
    corner = None
    for p in points:
        if abs(p["depth_over_a"] - 768.0) < 1e-9 and abs(p["dz_over_a"] - 1.0 / 24.0) < 1e-9:
            corner = p["e_delta_pct"]
    if not (dl and zl) or corner is None:
        return {"status": "unavailable"}
    corr = zl["richardson_limit"] - corner
    return {
        "status": "interim (corner linearisation)",
        "depth_limit_pct": dl["richardson_limit"],
        "dz_limit_pct": zl["richardson_limit"],
        "corner_e_delta_pct": corner,
        "corner_correction_pp": corr,
        "e0_interim_pct": dl["richardson_limit"] + corr,
        "dz_order": zl["observed_order"],
        "depth_order": dl["observed_order"],
    }


def analyse(points):
    """Nested limit with an honest uncertainty band + global-fit cross-check."""
    prim = nested(points, "upper")
    alt = nested(points, "lower")
    fit = global_fit(points)
    band = None
    if prim.get("status") == "ok":
        parts = {"gci": prim["band_gci_pct"]}
        if alt.get("status") == "ok":
            parts["window_shift"] = abs(prim["e0_pct"] - alt["e0_pct"])
        if fit.get("status") == "ok":
            parts["global_fit_shift"] = abs(prim["e0_pct"] - fit["e0_pct"])
        band = max(v for v in parts.values() if v is not None and math.isfinite(v))
        prim["band_components_pct"] = parts
        prim["band_pct"] = band
    return prim, alt, fit, band


def global_fit(points):
    """Least-squares fit of e = e0 + A (dz/a)^p + B (a/H)^q (p, q free).

    Grid search over (p, q) with a linear solve for (e0, A, B) at each node --
    small, dependency-free, and it reports the residual so the reader can judge
    whether the two-term model actually describes the data.
    """
    import numpy as np

    x = np.array([[p["dz_over_a"], p["depth_over_a"], p["e_delta_pct"]]
                  for p in points], dtype=float)
    if len(x) < 5:
        return {"status": "need >=5 points"}
    dz, H, e = x[:, 0], x[:, 1], x[:, 2]
    best = None
    for p_ in np.arange(0.5, 3.01, 0.05):
        for q_ in np.arange(0.5, 3.01, 0.05):
            M = np.column_stack([np.ones_like(e), dz ** p_, (1.0 / H) ** q_])
            coef, *_ = np.linalg.lstsq(M, e, rcond=None)
            res = e - M @ coef
            sse = float(res @ res)
            if best is None or sse < best["sse"]:
                best = {"sse": sse, "e0_pct": float(coef[0]),
                        "A": float(coef[1]), "B": float(coef[2]),
                        "p": float(p_), "q": float(q_),
                        "rms_pct": math.sqrt(sse / len(e))}
    best["n_points"] = int(len(e))
    best["status"] = "ok"
    return best


def decide(e0, band):
    """Pre-declared two-sided decision rule (see module docstring)."""
    if e0 is None or not math.isfinite(e0):
        return "inconclusive (grid incomplete)"
    lo, hi = e0 - band, e0 + band
    if abs(e0) <= CRITERIA["disc_abs_pct"] and lo <= 0.0 <= hi:
        return "H_disc"          # model-bias claim falsified
    if abs(e0) >= CRITERIA["bias_abs_pct"] and not (lo <= 0.0 <= hi):
        return "H_bias"          # model bias confirmed
    return "inconclusive"


def make_figure(payload, path):
    from _harness import pyplot

    plt = pyplot()
    pts = payload["points"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    for dz, style in ((1 / 6, "o:"), (1 / 12, "s--"), (1 / 24, "^"),
                      (1 / 48, "d-")):
        sub = sorted([p for p in pts if abs(p["dz_over_a"] - dz) < 1e-9],
                     key=lambda p: p["depth_over_a"])
        if len(sub) < 2:
            continue
        ax.plot([p["depth_over_a"] for p in sub],
                [p["e_delta_pct"] for p in sub], style,
                label=r"$\Delta z/a=1/%d$" % round(1.0 / dz))
    dl = payload["nested"]["depth_limits"]
    if dl.get("dz_a48"):
        ax.axhline(dl["dz_a48"]["richardson_limit"], color="k", lw=0.8, ls="-.",
                   label="depth $\\to\\infty$ ($\\Delta z=a/48$)")
    ax.set_xscale("log")
    ax.set_xlabel(r"domain depth $H/a$")
    ax.set_ylabel(r"$e_\delta$  [%]")
    ax.set_title("(a) depth ladder, one curve per $\\Delta z$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)

    ax = axes[1]
    for depth, style in ((768.0, "o-"), (384.0, "s--"), (192.0, "^:"),
                         (96.0, "d-.")):
        sub = sorted([p for p in pts if abs(p["depth_over_a"] - depth) < 1e-9],
                     key=lambda p: p["dz_over_a"])
        if len(sub) < 2:
            continue
        ax.plot([p["dz_over_a"] for p in sub],
                [p["e_delta_pct"] for p in sub], style,
                label=r"$H/a=%d$" % depth)
    ax.axhline(0.0, color="gray", lw=0.8)
    e0 = payload["nested"].get("e0_pct")
    band = payload["nested"].get("band_pct") or 0.0
    if e0 is not None and math.isfinite(e0):
        ax.axhspan(e0 - band, e0 + band, color="tab:red", alpha=0.15,
                   label="joint limit band")
        ax.axhline(e0, color="tab:red", lw=1.0,
                   label=r"$e_0=%.3f\%%$" % e0)
    ax.set_xlabel(r"$\Delta z/a$")
    ax.set_ylabel(r"$e_\delta$  [%]")
    ax.set_title("(b) vertical ladder, one curve per depth")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.suptitle("Hertz indentation error: joint ($\\Delta z$, $H$) limit",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    print(f"figure -> {path}")



def make_summary_figure(payload, path):
    """Residual budget: which refinement buys how much of the 1.01% at the
    accepted model, and what is left at the joint limit."""
    from _harness import pyplot

    plt = pyplot()
    pool = payload.get("pool") or {}
    prim = payload.get("nested") or {}
    labels, vals, errs = [], [], []
    labels.append("accepted model\n($\\Delta z=a/6$, 96a)")
    vals.append(pool.get("e_delta_pct") or 0.0)
    errs.append(0.0)
    if prim.get("dz_a48_depth_limit_pct") is not None:
        labels.append("$\\Delta z$ refinement only")
        vals.append(prim["dz_a48_depth_limit_pct"])
        errs.append(0.0)
    lim = prim.get("limits_by_dz_pct") or []
    if len(lim) >= 2 and lim[1] is not None:
        labels.append("depth extrapolation only\n($\\Delta z=a/24$)")
        vals.append(lim[1])
        errs.append(0.0)
    if payload.get("e0_pct") is not None:
        labels.append("joint limit $e_0$")
        vals.append(payload["e0_pct"])
        errs.append(payload.get("band_pct") or 0.0)
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    idx = range(len(vals))
    colors = ["tab:gray"] + ["tab:blue"] * (len(vals) - 2) + ["tab:red"]
    ax.bar(idx, vals, yerr=errs, capsize=5,
           color=colors[:len(vals)], alpha=0.85)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhspan(-payload["criteria"]["case_pool_pct"],
               payload["criteria"]["case_pool_pct"],
               color="tab:green", alpha=0.08, label="case-pool band $\\pm3\\%$")
    for i, v in enumerate(vals):
        ax.annotate(f"{v:+.3f}%", (i, v), textcoords="offset points",
                    xytext=(0, 6 if v > 0 else -14), ha="center", fontsize=8)
    ax.set_xticks(list(idx))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"$e_\delta$  [%]")
    ax.set_title("Hertz indentation residual budget")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"figure -> {path}")

def assemble(figure=False):
    pts = load_cells()
    prim, alt, fit, band = analyse(pts)
    e0 = prim.get("e0_pct")
    verdict = decide(e0, band)
    pool = {}
    cert = os.path.join(HERE, "result.json")
    if os.path.exists(cert):
        d = json.load(open(cert))
        fine = d["runs"]["ratio0.05_h0.0075_d96"]["error_pct"]
        pool = {
            "accepted_model": "h/a=0.0075, dz=a/6, H=96a (case-pool certificate)",
            "e_delta_pct": fine["indentation"],
            "e_a_linear_pct": fine["contact_radius_linear"],
            "e_a_fit_pct": fine.get("contact_radius_fit"),
            "within_3pct": bool(abs(fine["indentation"]) < CRITERIA["case_pool_pct"]
                                and abs(fine["contact_radius_linear"])
                                < CRITERIA["case_pool_pct"]),
        }
    payload = {
        "schema": "tensorfem.hertz-two-param-extrapolation.v1",
        "anchor": "fem_hertz_contact_loadcontrol",
        "case_pool": "FEM-HERTZ-CONTACT (id 104)",
        "generated_at": utc_now(),
        "question": "does the residual indentation offset of the Hertz anchor "
                    "survive the joint (dz, 1/H) limit, i.e. is it a model bias?",
        "criteria": CRITERIA,
        "far_field_layers": list(FAR),
        "n_points": len(pts),
        "cells_expected": len(CELLS),
        "cells_missing": [c[0] for c in CELLS
                          if not os.path.exists(cell_path(c[0]))],
        "points": pts,
        "nested": prim,
        "nested_alt_window": alt,
        "global_fit": fit,
        "interim_estimate": interim(pts),
        "e0_pct": e0,
        "band_pct": band,
        "verdict": verdict,
        "pool": pool,
    }
    write_json(Path(ARTIFACT), payload)
    print(f"points={len(pts)} nested e0={e0} band={band} "
          f"fit e0={fit.get('e0_pct')} verdict={verdict}")
    if figure:
        os.makedirs(FIGDIR, exist_ok=True)
        make_figure(payload, os.path.join(FIGDIR,
                                          "hertz_two_param_limit.png"))
        make_summary_figure(payload, os.path.join(FIGDIR,
                                                  "hertz_two_param_budget.png"))
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--shard", type=int, default=None,
                    help="1-based shard index (or 0-based; both cover the grid)")
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--figure", action="store_true")
    args = ap.parse_args()
    if args.list:
        for i, c in enumerate(CELLS):
            print(i, c)
        return 0
    if args.shard is not None:
        n = len(CELLS)
        idx = (args.shard - 1) % n
        print(f"[shard {args.shard}/{args.of}] -> cell {idx} {CELLS[idx]}")
        return run_cell(idx)
    if args.assemble:
        assemble(figure=args.figure)
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())