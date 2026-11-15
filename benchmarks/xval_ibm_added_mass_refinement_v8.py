#!/usr/bin/env python3
"""XVAL v8 -- is the implicit-IBM added-mass excess a resolution artefact?

v1-v7 of the anchor measured ``m_a ~ 3.6 rho pi c^2`` for the discrete plate and
attributed the factor to *geometry* ("the IBM body is a bluff body of thickness
~2 kernel supports", ``docs/xval_implicit_ibm_aitken_v7.md``).  That claim was
never tested on the one axis that separates geometry from discretisation: the
lattice spacing.  This script does the two measurements that settle it.

Stages
------
``bem``    potential-flow added mass (independent boundary-element reference,
           :mod:`tensorfem.added_mass_bem2d`): validated on the circle/ellipse
           closed forms, then ``C_a = m/(rho pi c^2)`` for rectangles of
           increasing thickness -- i.e. how much added mass a body of the
           *claimed* thickness really has.
``refine`` grid refinement at fixed physical geometry (``dx = 1, 1/2, 1/4, 1/8``)
           with the acoustic scaling ``dt = dx``; measured from the fluid
           momentum (force-free route) and from the marker force.
``equiv``  unit-convention control: the ``dx = 1/2`` run re-expressed in
           ``dx = 1`` units (double the body, amplitude and period) must give the
           same dimensionless coefficient -- otherwise the refinement series is
           comparing two different physics problems.
``csw``    body-size sweep at fixed ``dx = 1`` (the axis v7 used), for the
           "fixed-cell-thickness" prediction ``C = ((c+delta)/c)^2``.
``all``    every stage + aggregate + figure.

    python3 benchmarks/xval_ibm_added_mass_refinement_v8.py --stage all
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from tensorfem.added_mass_bem2d import (  # noqa: E402
    added_mass_matrix,
    benchmarks as bem_benchmarks,
    circle_panels,
    ellipse_panels,
    rectangle_added_mass_coefficient,
    rectangle_panels,
)
from tensorfem.ibm_added_mass_refinement import lattice_added_mass  # noqa: E402

ART = os.path.join(HERE, "xval_ibm_added_mass_refinement_v8_artifacts")
C_PHYS = 4.0          # plate half-width, dx=1 units
BOX_PHYS = 64.0
AMPLITUDE = 0.2       # 5 % of the half-width
PERIOD_PHYS = 64.0
KERNEL_HALF = 1.5     # cells: half-support of the 3-point delta (dirac_delta_3pt)


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _write(name, payload):
    os.makedirs(ART, exist_ok=True)
    path = os.path.join(ART, name)
    with open(path, "w") as fh:
        json.dump(_clean(payload), fh, indent=1, sort_keys=True)
    return path


# ---------------------------------------------------------------------------
def bem_stage():
    """Independent potential-flow reference: shape validation + thick-plate C_a."""
    rows = bem_benchmarks(n_circle=720, n_ellipse=720)
    thickness = []
    for hc in (0.02, 0.05, 0.1, 0.1875, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0):
        thickness.append({
            "h_over_c": hc,
            "C_a": rectangle_added_mass_coefficient(hc, per_edge=220),
        })
    # panel-count convergence for the thickest and thinnest sections
    convergence = []
    for hc in (0.375, 1.0):
        for per_edge in (60, 120, 240, 480):
            convergence.append({
                "h_over_c": hc, "per_edge": per_edge,
                "C_a": rectangle_added_mass_coefficient(hc, per_edge=per_edge),
            })
    # the smeared disk a marker ring really represents: annulus of half-width
    # KERNEL_HALF cells around radius a, outer added mass (disk) + entrained core
    smear = []
    for a_cells, delta in ((5.0, KERNEL_HALF), (5.0, 0.0), (4.0, KERNEL_HALF)):
        outer = math.pi * (a_cells + delta) ** 2
        smear.append({
            "a_cells": a_cells, "delta_cells": delta,
            "outer_disk": outer, "entrained_core": outer,
            "total": 2.0 * outer,
            "C_ring_smeared": 2.0 * (a_cells + delta) ** 2 / a_cells ** 2,
        })
    payload = {"validation": rows, "thickness": thickness,
               "panel_convergence": convergence, "ring_smearing": smear}
    _write("bem.json", payload)
    return payload


def _row(r):
    keys = ("refine", "dx", "n_cell", "c_lat", "n_markers", "ds_median", "t_lat",
            "n_steps", "gain", "gain_mode", "amp_cells", "m_momentum_lat",
            "m_force_lat", "analytic_lat", "C_momentum", "C_force",
            "err_momentum", "err_force", "r2_momentum", "r2_force",
            "v_achieved_ratio", "constraint_rel_max", "eps_amplification",
            "periods_in_window")
    out = {k: r[k] for k in keys if k in r}
    out["config"] = {k: r.get(k) for k in ("c_phys", "box_phys", "amplitude",
                                           "period_phys")}
    return out


def refine_stage(refines=(1, 2, 4, 8), controls=((1, "scalar", 2.0), (2, "scalar", 2.0))):
    rows = []
    for refine in refines:
        t0 = time.time()
        r = lattice_added_mass(refine=refine, c_phys=C_PHYS, box_phys=BOX_PHYS,
                               amplitude=AMPLITUDE, period_phys=PERIOD_PHYS,
                               n_cycles=4.0, gain=2.0, gain_mode="operator")
        row = _row(r)
        row["wall_s"] = round(time.time() - t0, 2)
        rows.append(row)
        print(f"  refine 1/{refine:d}  H={r['n_cell']:4d}  "
              f"C_mom={r['C_momentum']:.4f}  C_force={r['C_force']:.4f}  "
              f"({row['wall_s']:.1f}s)", flush=True)
        _write(f"refine_{refine}.json", r)
    ctrl = []
    for refine, mode, gain in controls:
        r = lattice_added_mass(refine=refine, c_phys=C_PHYS, box_phys=BOX_PHYS,
                               amplitude=AMPLITUDE, period_phys=PERIOD_PHYS,
                               n_cycles=4.0, gain=gain, gain_mode=mode)
        row = _row(r)
        ctrl.append(row)
        print(f"  control refine 1/{refine:d} {mode} gain={gain}: "
              f"C_mom={r['C_momentum']:.4f}", flush=True)
    payload = {"series": rows, "controls": ctrl}
    _write("refine.json", payload)
    return payload


def equiv_stage():
    """Same discrete problem, two unit conventions (dx = 1 vs dx = 1/2)."""
    rows = []
    base = lattice_added_mass(refine=2, c_phys=C_PHYS, box_phys=BOX_PHYS,
                              amplitude=AMPLITUDE, period_phys=PERIOD_PHYS,
                              n_cycles=4.0, gain=2.0, gain_mode="operator")
    rows.append({"convention": "dx=1/2", **_row(base)})
    # dx = 1, everything doubled in cells: identical lattice numbers
    twin = lattice_added_mass(refine=1, c_phys=2.0 * C_PHYS,
                              box_phys=2.0 * BOX_PHYS,
                              amplitude=2.0 * AMPLITUDE,
                              period_phys=2.0 * PERIOD_PHYS,
                              n_cycles=4.0, gain=2.0, gain_mode="operator")
    rows.append({"convention": "dx=1 (doubled)", **_row(twin)})
    same = {
        "C_momentum_rel_diff": abs(twin["C_momentum"] / base["C_momentum"] - 1.0),
        "C_force_rel_diff": abs(twin["C_force"] / base["C_force"] - 1.0),
        "lattice_config_identical": (
            twin["n_cell"] == base["n_cell"] and twin["n_markers"] == base["n_markers"]
            and abs(twin["amp_cells"] - base["amp_cells"]) < 1e-12
            and abs(twin["t_lat"] - base["t_lat"]) < 1e-12),
    }
    payload = {"rows": rows, "consistency": same}
    _write("equiv.json", payload)
    print(f"  unit-equivalence: C_mom rel diff {same['C_momentum_rel_diff']:.2e}, "
          f"identical lattice config {same['lattice_config_identical']}", flush=True)
    return payload


def csw_stage(cs=(4.0, 8.0, 16.0), box_per_c=8.0):
    """Body-size sweep at fixed ``dx = 1`` with the *confinement ratio* held.

    ``box = box_per_c * c`` so that ``box/c = 8`` at every point: this isolates
    the "fixed-cell-thickness" prediction ``C = ((c+delta)/c)^2`` from the
    confinement change that confounded the v7 ``c``-sweep (fixed box, growing
    body).  Amplitude is 5 % of the half-width, as everywhere else.
    """
    rows = []
    for c in cs:
        r = lattice_added_mass(refine=1, c_phys=c, box_phys=box_per_c * c,
                               amplitude=0.05 * c, period_phys=PERIOD_PHYS,
                               n_cycles=4.0, gain=2.0, gain_mode="operator")
        row = _row(r)
        rows.append(row)
        print(f"  c_lat={c:5.1f}  C_mom={r['C_momentum']:.4f}", flush=True)
        _write(f"csw_{int(c)}.json", r)
    payload = {"series": rows}
    _write("csw.json", payload)
    return payload


# ---------------------------------------------------------------------------
def aggregate(refine, bem, equiv, csw):
    series = refine["series"]
    xs = np.array([r["c_lat"] for r in series], dtype=float)
    ys = np.array([r["C_momentum"] for r in series], dtype=float)
    # C = C_inf + k / c_lat  (fit on the two finest points, reported for all)
    fit = {}
    if len(xs) >= 3:
        A = np.stack([np.ones_like(1.0 / xs), 1.0 / xs], axis=1)
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        fit = {"C_inf": float(coef[0]), "k": float(coef[1]),
               "residual_max": float(np.max(np.abs(A @ coef - ys)))}
    thick = {row["h_over_c"]: row["C_a"] for row in bem["thickness"]}
    # which rectangle thickness reproduces the dx = 1 measurement?
    implied_hc = None
    target = float(series[0]["C_momentum"])
    hs = sorted(thick)
    for h0, h1 in zip(hs[:-1], hs[1:]):
        c0, c1 = thick[h0], thick[h1]
        if c0 <= target <= c1:
            implied_hc = float(h0 + (h1 - h0) * (target - c0) / (c1 - c0))
            break
    if implied_hc is None and target > thick[hs[-1]]:
        h0, h1 = hs[-2], hs[-1]
        c0, c1 = thick[h0], thick[h1]
        implied_hc = float(h0 + (h1 - h0) * (target - c0) / (c1 - c0))
    out = {
        "series_C_momentum": fit,
        "measured_C_dx1": target,
        "reference_C_thin_plate": 1.0,
        "reference_C_kernel_thickness": thick.get(0.375),
        "reference_C_one_chord_thick": thick.get(1.0),
        "implied_h_over_c_at_dx1": implied_hc,
        "implied_thickness_cells_at_dx1": (None if implied_hc is None
                                          else implied_hc * C_PHYS),
        "unit_equivalence": equiv["consistency"],
        "csw": [{"c_lat": r["c_lat"], "C_momentum": r["C_momentum"],
                 "pred_fixed_delta_3p5": ((r["c_lat"] + 3.5) / r["c_lat"]) ** 2}
                for r in csw["series"]],
    }
    _write("aggregate.json", out)
    return out


def figure(refine, bem, agg):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                      # pragma: no cover
        print(f"  (no figure: {exc})", flush=True)
        return None
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))

    # (a) BEM reference: added mass of a thick body
    hc = [r["h_over_c"] for r in bem["thickness"]]
    ca = [r["C_a"] for r in bem["thickness"]]
    ax[0].plot(hc, ca, "o-", ms=3.5, lw=1.4, label="rectangle (BEM)")
    ax[0].axhline(1.0, color="k", ls=":", lw=1, label="thin plate, $C_a=1$")
    ax[0].scatter([0.375], [bem["thickness"][3]["C_a"]], marker="s", s=55,
                  color="tab:green", zorder=5,
                  label="kernel thickness $3\\Delta$")
    ax[0].scatter([agg["implied_h_over_c_at_dx1"]], [agg["measured_C_dx1"]],
                  marker="*", s=170, color="tab:red", zorder=6,
                  label="measured $\\Delta$-lattice")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("$h/c$ (half-thickness / half-chord)")
    ax[0].set_ylabel("$C_a = m_a/(\\rho\\pi c^2)$")
    ax[0].set_title("(a) how thick a body would have to be")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=7.5)

    # (b) refinement
    xs = [r["c_lat"] for r in refine["series"]]
    ax[1].plot(xs, [r["C_momentum"] for r in refine["series"]], "o-",
               label="momentum route")
    ax[1].plot(xs, [r["C_force"] for r in refine["series"]], "s--",
               label="force route")
    ax[1].axhline(1.0, color="k", ls=":", label="thin plate")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("$c/\\Delta$  (half-chord in cells)")
    ax[1].set_ylabel("$C = m_a/(\\rho\\pi c^2)$")
    ax[1].set_title("(b) refinement at fixed physical geometry")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)

    # (c) force/momentum histories at the finest level
    try:
        with open(os.path.join(ART, f"refine_{refine['series'][-1]['refine']}.json")) as fh:
            finest = json.load(fh)
        ax[2].plot(finest["momentum_history"], lw=0.9, label="$\\Pi_y$ (fluid momentum)")
        ax[2].plot(finest["force_history"], lw=0.9, alpha=0.8, label="$F_y$ (marker force)")
        ax[2].set_xlabel("step")
        ax[2].set_ylabel("lattice units")
        ax[2].set_title(f"(c) finest run ($c/\\Delta={finest['c_lat']:.0f}$)")
        ax[2].grid(alpha=0.3)
        ax[2].legend(fontsize=8)
    except Exception:
        pass
    fig.tight_layout()
    path = os.path.join(ART, "v8_refinement.png")
    fig.savefig(path, dpi=150)
    print(f"  figure -> {path}", flush=True)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["bem", "refine", "equiv", "csw", "all", "aggregate"])
    ap.add_argument("--refines", default="1,2,4,8")
    args = ap.parse_args(argv)
    refines = tuple(int(x) for x in args.refines.split(","))
    os.makedirs(ART, exist_ok=True)

    def load(name):
        with open(os.path.join(ART, name)) as fh:
            return json.load(fh)

    if args.stage in ("bem", "all"):
        print("bem stage", flush=True)
        bem_stage()
    if args.stage in ("refine", "all"):
        print("refine stage", flush=True)
        refine_stage(refines=refines)
    if args.stage in ("equiv", "all"):
        print("equiv stage", flush=True)
        equiv_stage()
    if args.stage in ("csw", "all"):
        print("csw stage", flush=True)
        csw_stage()
    if args.stage in ("aggregate", "all"):
        bem = load("bem.json")
        refine = load("refine.json")
        equiv = load("equiv.json")
        csw = load("csw.json")
        agg = aggregate(refine, bem, equiv, csw)
        print(json.dumps(_clean(agg), indent=1), flush=True)
        fig = figure(refine, bem, agg)
        return {"aggregate": agg, "figure": fig}
    return None


if __name__ == "__main__":
    main()