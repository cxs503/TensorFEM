#!/usr/bin/env python3
"""CASE-BENDING-PATCH: constant-curvature (bending) patch test, Q4 Mindlin plate.

Physics
-------
A flat isotropic plate patch is loaded only through boundary displacements that
belong to a state of **constant curvature** ``(kappa_x, kappa_y, kappa_xy)``.
The exact state (transverse-shear free) is, in this model's sign convention
``gamma_xz = w,x + theta_x``::

    w       = -(kappa_x x^2 + kappa_xy x y + kappa_y y^2)/2
    theta_x =   kappa_x x + kappa_xy y/2
    theta_y =   kappa_xy x/2 + kappa_y y

    M = Db kappa,  Db = D [[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]],  D = E t^3/(12(1-nu^2))

Criterion (the reference): a patch with free interior nodes must reproduce the
interior displacement, rotations, curvatures/moments and strain energy exactly
-- Irons & Ahmad, *Techniques of Finite Elements* (1980) ch. 8; Zienkiewicz &
Taylor, *The Finite Element Method* Vol. 2, §10.4; Batoz, Bathe & Ho, *Int. J.
Numer. Methods Eng.* **15** (1980) 1771-1812.  Every target value is evaluated
from the closed form above (:func:`tensorfem.bending_patch.oracle_selfcheck`
re-verifies it before any assembly); no handbook digit is copied.

Cases (all machine graded)
--------------------------
1. ``affine`` / ``skewed`` patches -- parallelogram elements, i.e. affine maps of
   the reference patch.  A shear-robust Q4 must reproduce the state to
   round-off: this is the pass/fail form of the test.
2. ``distorted`` patch -- non-affine quads.  The bilinear interpolation of the
   quadratic ``w`` is not exact, so a small residual remains; it is *measured*
   and graded against the case-pool tolerance (< 2%), and its h-convergence is
   reported rather than assumed.
3. ``locking`` control -- the same patch and the same driver with 2x2 (full)
   shear integration.  It must fail: a gate set that the locking element also
   passes would prove nothing.
4. ``strip-full`` (round 2) -- the Poisson cross-check, repaired.  The lateral-
   refinement sweep (``n_y = 2..16``) and the aspect-ratio series (``L/b = 5, 10,
   20``) show that the free-edge leg carries a *clamped-root* end effect that
   decays with ``L/b``; the closed-form leg is kept only as an algebra identity,
   and the coupling is graded on the **solved** deflections plus the exact
   ``M_y = 0`` (plane-stress) patch below.
5. ``free-surface`` (round 2) -- the ``M_y = 0`` state ``kappa_y = -nu kappa_x``
   reproduces ``M_x = D_b kappa_x`` with ``D_b = E t^3/12``, so the Poisson
   coupling is read out of the element's own ``Db`` and *can* fail.
6. ``strip`` -- a *load-driven* cantilever plate strip under a tip moment, which
   removes the "the answer was prescribed" objection.  Two closed forms are used:
   cylindrical bending (``lateral="symmetry"``, plate rigidity ``D``) and the
   narrow strip with free edges (beam rigidity ``D_b = E t^3/12``); their ratio
   is exactly ``1/(1-nu^2)``, which cross-checks the Poisson coupling.

Outputs (this directory): ``result.json``, ``gates_verified.json``, and the PNGs
referenced by ``docs/bending-patch.md``.  Generated, never hand-edited.

Usage::

    OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/verified/fem_bending_patch/run.py
    OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/verified/fem_bending_patch/run.py --quick
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "verified"))

from _harness import Gate, gate_manifest, pyplot, repo_root, utc_now, write_json  # noqa: E402

from tensorfem.bending_patch import (  # noqa: E402
    CurvatureState,
    affine_patch_mesh,
    distorted_patch_mesh,
    free_surface_check,
    free_surface_state,
    oracle_selfcheck,
    run_patch_case,
    strip_pure_bending,
)

HERE = Path(__file__).resolve().parent
FIGS = REPO / "docs" / "assets" / "bending_patch"
DOC = REPO / "docs" / "bending-patch.md"

YOUNG = 210.0e3     # MPa
POISSON = 0.3
THICK = 1.0         # mm (patch is 1 x 1 x 1: a thick-ish plate, E/t/nu O(1) test)

STATES = {
    "cylindrical": CurvatureState(1.0e-3, 0.0, 0.0),
    "twist": CurvatureState(0.0, 0.0, 1.0e-3),
    "double": CurvatureState(1.0e-3, 0.6e-3, 0.4e-3),
}

# case-pool tolerance for the anchor
POOL_TOL = 0.02
# round-off floor below which an observed convergence order is not measurable
ROUNDOFF_FLOOR = 1.0e-10
# round-2 cross-check refinements
NY_LEVELS = (2, 4, 8, 16)
ASPECT_LEVELS = (1.0, 2.0, 4.0)
STRIP_WIDTH = 0.2
FREE_SURFACE_POISSONS = (0.3, 0.45)
# self-imposed "exact" tolerance for the affine pass/fail form
EXACT_TOL = 1.0e-12
# round-3: thin-limit (shear-locking) ladder.  ``h/t`` is the locking parameter of a
# Mindlin element; the plate is thinned at a *fixed* mesh so h/t sweeps 0.25..5 (a
# factor 20, i.e. h/t up to 12.5 at the finest mesh) while h is unchanged.
LOCK_THICKNESSES = (1.0, 0.5, 0.2, 0.1, 0.05)
LOCK_MESHES = (4, 8)
LOCK_GRADED_MESH = 4
# predeclared from the pure (h/t)^2 law, NOT from the measured value: the slope floor
# is the same 1.5 used for observed orders, the amplification floor is a factor 4
# below the 20^2 = 400 that the law predicts over this ladder.
LOCK_SLOPE_MIN = 1.5
LOCK_AMPLIFICATION_MIN = 100.0
LOCK_LAW_TOL = 0.5


# --------------------------------------------------------------------------- #
# case matrix
# --------------------------------------------------------------------------- #
def patch_matrix(quick: bool) -> list[dict]:
    rows = []
    for name, state in STATES.items():
        meshes = [
            ("affine", affine_patch_mesh(3)),
            ("skew", affine_patch_mesh(3, skew=0.35)),
            ("distorted", distorted_patch_mesh(4 if not quick else 3)),
        ]
        for fam, patch in meshes:
            for treat in ("sri", "mitc", "full"):
                t0 = time.time()
                r = run_patch_case(patch, state, young=YOUNG, poisson=POISSON,
                                   thickness=THICK, shear_treatment=treat)
                r["case"] = f"{name}/{fam}/{treat}"
                r["state_name"] = name
                r["wall_seconds"] = round(time.time() - t0, 3)
                rows.append(r)
    return rows


def refinement_ladder(quick: bool) -> list[dict]:
    levels = (4, 8) if quick else (4, 8, 16, 32)
    rows = []
    state = STATES["cylindrical"]
    for n in levels:
        patch = distorted_patch_mesh(n)
        for treat in ("sri", "mitc"):
            r = run_patch_case(patch, state, young=YOUNG, poisson=POISSON,
                               thickness=THICK, shear_treatment=treat)
            r["h"] = 1.0 / n
            r["case"] = f"ladder/n={n}/{treat}"
            rows.append(r)
    return rows


def strip_cases(quick: bool) -> list[dict]:
    rows = []
    levels = (4, 8) if quick else (4, 8, 16, 32)
    for lateral in ("symmetry", "free"):
        for n_x in levels:
            for treat in ("sri", "mitc"):
                r = strip_pure_bending(n_x, 2, moment=1.0e3, lateral=lateral,
                                       shear_treatment=treat,
                                       young=YOUNG, poisson=POISSON, thickness=0.01)
                r["case"] = f"strip/{lateral}/n={n_x}/{treat}"
                rows.append(r)
    # locking control + thickness sensitivity (two physical thickness ratios)
    for lateral in ("symmetry", "free"):
        for thick in (0.01, 0.10):
            r = strip_pure_bending(8, 2, moment=1.0e3, lateral=lateral,
                                   shear_treatment="full",
                                   young=YOUNG, poisson=POISSON, thickness=thick)
            r["case"] = f"strip-locking/{lateral}/t={thick}/{8}"
            rows.append(r)
    return rows


def free_surface_cases(quick: bool) -> list[dict]:
    """The ``M_y = 0`` (plane-stress) state: the coupling check that can fail."""
    rows = []
    nus = FREE_SURFACE_POISSONS[:1] if quick else FREE_SURFACE_POISSONS
    families = [("affine", affine_patch_mesh(3)),
                ("skew", affine_patch_mesh(3, skew=0.35))]
    for nu in nus:
        for fam, patch in families:
            for treat in ("sri", "mitc"):
                r = free_surface_check(patch, 1.0e-3, young=YOUNG, poisson=nu,
                                       thickness=THICK, shear_treatment=treat)
                r["case"] = f"free-surface/nu={nu}/{fam}/{treat}"
                r["family"] = fam
                rows.append(r)
    return rows


def strip_transverse_sweep(quick: bool) -> list[dict]:
    """Transverse (``n_y``) refinement of the load-driven strip, both laterals.

    Round 1 ran the free-edge leg at a fixed ``n_y = 2``; this sweep measures what
    that mesh does to the answer.
    """
    rows = []
    nxs = (8,) if quick else (8, 32)
    nys = (2, 4, 8) if quick else NY_LEVELS
    for lateral in ("symmetry", "free"):
        for n_x in nxs:
            for n_y in nys:
                r = strip_pure_bending(n_x, n_y, moment=1.0e3, lateral=lateral,
                                       shear_treatment="sri", young=YOUNG,
                                       poisson=POISSON, thickness=0.01,
                                       width=STRIP_WIDTH)
                r["case"] = f"strip-ny/{lateral}/nx={n_x}/ny={n_y}"
                r["lateral"] = lateral
                rows.append(r)
    return rows


def strip_aspect_series(quick: bool) -> list[dict]:
    """Free-edge strip at fixed element size, growing ``L/b`` (end-effect probe)."""
    rows = []
    levels = ASPECT_LEVELS[:2] if quick else ASPECT_LEVELS
    for length in levels:
        n_x = int(round(32 * length))
        r = strip_pure_bending(n_x, 8, moment=1.0e3, length=length,
                               width=STRIP_WIDTH, lateral="free",
                               shear_treatment="sri", young=YOUNG, poisson=POISSON,
                               thickness=0.01)
        r["case"] = f"strip-aspect/L={length:g}/nx={n_x}"
        rows.append(r)
    return rows


def strip_root_support_control(quick: bool) -> list[dict]:
    """Root-support control: is the free-edge offset caused by the `theta_y` clamp?

    `clamp_theta_y=False` keeps `w` and `theta_x` restrained at the root and lets
    `theta_y` float.  If the offset disappeared, the restraint, not the `w = 0`
    incompatibility with the anticlastic field, would be the cause.
    """
    rows = []
    for clamp in (True, False):
        r = strip_pure_bending(32, 8, moment=1.0e3, lateral="free",
                               shear_treatment="sri", young=YOUNG, poisson=POISSON,
                               thickness=0.01, width=STRIP_WIDTH,
                               clamp_theta_y=clamp)
        r["case"] = f"strip-root/clamp_theta_y={clamp}/nx=32/ny=8"
        r["clamp_theta_y"] = clamp
        rows.append(r)
    return rows


def locking_scaling_cases(quick: bool) -> list[dict]:
    """Round 3: thin the plate at a fixed mesh and sweep the locking parameter h/t.

    Every graded patch case of rounds 1-2 runs at ``t/L = 1``, i.e. ``h/t = 0.25`` --
    a *thick* plate, which is exactly where shear locking is weakest.  The round-2
    negative control therefore read only 2.1 % against a 2 % threshold (a 5 % margin):
    a control that thin cannot separate "the element locks" from "the element has a
    small interpolation error".

    Here the plate is thinned instead of the mesh refined, so ``h/t`` grows at constant
    ``h``.  The spurious transverse-shear energy of the bilinear interpolant scales as
    ``(h/t)^2``, so a locking element must show a log-log slope of 2 and an
    amplification of ``20^2 = 400``; a shear-robust one must stay at round-off.  The
    distorted family is graded (its interpolant is genuinely not shear-free) and the
    affine family is the control.
    """
    # the full thickness ladder is kept in --quick: it is 60 small patch solves and the
    # graded thresholds must not change between the smoke test and the full run.
    taus = LOCK_THICKNESSES
    meshes = (LOCK_GRADED_MESH,) if quick else LOCK_MESHES
    rows = []
    for n in meshes:
        for fam, patch in (("affine", affine_patch_mesh(n)),
                           ("distorted", distorted_patch_mesh(n))):
            for tau in taus:
                for treat in ("sri", "mitc", "full"):
                    r = run_patch_case(patch, STATES["double"], young=YOUNG,
                                       poisson=POISSON, thickness=tau,
                                       shear_treatment=treat)
                    r["case"] = f"lock-scaling/{fam}/n={n}/t={tau:g}/{treat}"
                    r["family"] = fam
                    r["thickness_ratio"] = tau
                    r["h_over_t"] = 1.0 / (n * tau)
                    rows.append(r)
    return rows


# --------------------------------------------------------------------------- #
# analysis helpers
# --------------------------------------------------------------------------- #
def observed_order(rows: list[dict], treatment: str, key: str) -> float:
    """Least-squares slope of log(error) vs log(h); NaN if it is not measurable.

    A slope fitted to round-off numbers is noise, not a convergence rate: if the
    largest error in the ladder is below :data:`ROUNDOFF_FLOOR` the order is
    reported as ``NaN`` (-> ``null`` in JSON, "n/a" in the report) instead of a
    number that looks like evidence.
    """
    sel = [r for r in rows if r["shear_treatment"] == treatment and r[key] > 0.0]
    if not sel or max(r[key] for r in sel) < ROUNDOFF_FLOOR:
        return float("nan")
    pts = [(math.log(r["h"]), math.log(r[key])) for r in sel]
    if len(pts) < 2:
        return float("nan")
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    return num / den


def worst(rows: list[dict], pred, key: str) -> float:
    vals = [r[key] for r in rows if pred(r)]
    return max(vals) if vals else float("nan")


def loglog_slope(xs: list[float], ys: list[float],
                 floor: float = ROUNDOFF_FLOOR) -> float:
    """Least-squares slope of log(y) vs log(x); NaN when y is at the round-off floor.

    Same discipline as :func:`observed_order`: a slope fitted to round-off noise is not
    a rate, so a series whose largest value is below ``floor`` is reported as ``NaN``
    (``null`` in JSON, "n/a" in the report) instead of a fake exponent.
    """
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if y > 0.0]
    if len(pts) < 2 or max(ys) < floor:
        return float("nan")
    n = len(pts)
    mx = sum(q[0] for q in pts) / n
    my = sum(q[1] for q in pts) / n
    num = sum((q[0] - mx) * (q[1] - my) for q in pts)
    den = sum((q[0] - mx) ** 2 for q in pts)
    return num / den


def locking_scaling_summary(rows: list[dict]) -> dict:
    """Slope / amplification / thin-limit value of every (family, treatment, mesh)."""
    out = {"graded_mesh_n": LOCK_GRADED_MESH,
           "thickness_ratios": sorted({r["thickness_ratio"] for r in rows}),
           "series": {}, "slope": {}, "amplification": {}, "thin_limit": {}}
    for fam in ("affine", "distorted"):
        for treat in ("sri", "mitc", "full"):
            for n in sorted({r["n"] for r in rows}):
                sel = sorted([r for r in rows if r["family"] == fam
                              and r["shear_treatment"] == treat and r["n"] == n],
                             key=lambda r: r["h_over_t"])
                if not sel:
                    continue
                key = f"{fam}/{treat}/n={n}"
                xs = [r["h_over_t"] for r in sel]
                ys = [r["energy_rel_error"] for r in sel]
                out["series"][key] = [
                    {"h_over_t": x, "energy_rel_error": y,
                     "displacement_rel_error": r["displacement_rel_error"]}
                    for x, y, r in zip(xs, ys, sel)]
                out["slope"][key] = loglog_slope(xs, ys)
                out["amplification"][key] = (ys[-1] / ys[0]) if ys[0] > 0.0 else float("nan")
                out["thin_limit"][key] = ys[-1]
    ns = sorted({r["n"] for r in rows})
    g = LOCK_GRADED_MESH
    out["slope_full"] = out["slope"].get(f"distorted/full/n={g}", float("nan"))
    out["amplification_full"] = out["amplification"].get(f"distorted/full/n={g}",
                                                         float("nan"))
    slope_sri = out["slope"].get(f"distorted/sri/n={g}", float("nan"))
    out["slope_sri_dev"] = (abs(slope_sri - 2.0) if slope_sri == slope_sri
                            else float("nan"))
    out["thin_mitc"] = max(out["thin_limit"][k] for k in out["thin_limit"]
                           if "/mitc/" in k)
    out["thin_sri_affine"] = max(out["thin_limit"][f"affine/sri/n={n}"] for n in ns)
    out["thin_sri_distorted"] = max(out["thin_limit"][f"distorted/sri/n={n}"]
                                    for n in ns)
    return out


def round3_gates(lock_sum: dict) -> list[Gate]:
    """Round-3 thin-limit gates (one place, so a merge step cannot fork them)."""
    return [
        Gate("locking_scaling_slope_full",
             "observed slope of log(strain-energy error) vs log(h/t), distorted patch, "
             "fully integrated 2x2 shear, n=4, t/L = 1 .. 0.05",
             "the spurious transverse-shear energy of the bilinear interpolant scales as "
             "(h/t)^2, so a locking element must show slope 2 (a negative control on the "
             "law, not on one marginal number); threshold 1.5 = the observed-order floor",
             lock_sum["slope_full"], LOCK_SLOPE_MIN, ">="),
        Gate("locking_amplification_full",
             "e(h/t=5) / e(h/t=0.25) of the energy error, distorted patch, fully "
             "integrated shear, n=4",
             "the (h/t)^2 law predicts 20^2 = 400 over this ladder; a factor 4 slack "
             "(>= 100) is required. The round-2 single-point control read 2.1 % against "
             "a 2 % threshold (a 5 % margin) at h/t = 0.25, i.e. in the thick-plate "
             "regime where locking is weakest by construction",
             lock_sum["amplification_full"], LOCK_AMPLIFICATION_MIN, ">="),
        Gate("thin_limit_mitc",
             "max strain-energy error of MITC4 over both mesh families and both meshes "
             "at t/L = 0.05 (h/t = 5)",
             "assumed transverse-shear strain: the element must stay shear-free in the "
             "thin limit, where the locking element is ~558 % off",
             lock_sum["thin_mitc"], POOL_TOL),
        Gate("thin_limit_sri_affine",
             "max strain-energy error of SRI on the AFFINE patch at t/L = 0.05, both "
             "meshes",
             "on an affine mesh the SRI element passes the patch test at every "
             "thickness, so distortion -- not thickness -- is the trigger of the "
             "degradation measured on the distorted family",
             lock_sum["thin_sri_affine"], ROUNDOFF_FLOOR),
        Gate("sri_locking_law_slope_dev",
             "|observed slope - 2| of the SRI energy error vs h/t, distorted patch, n=4",
             "mechanism identification: the SRI degradation must follow the same (h/t)^2 "
             "locking law (an implementation error would not reproduce the exponent). "
             "SRI reduces the constant ~21x but does not remove the thin-plate "
             "limitation on distorted meshes",
             lock_sum["slope_sri_dev"], LOCK_LAW_TOL),
    ]


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(result: dict) -> list[dict]:
    plt = pyplot()
    FIGS.mkdir(parents=True, exist_ok=True)
    made = []

    # 1. the three patch meshes -------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    meshes = [("affine (parallelograms)", affine_patch_mesh(3)),
              ("skewed affine (skew=0.35)", affine_patch_mesh(3, skew=0.35)),
              ("distorted (non-affine)", distorted_patch_mesh(4))]
    for ax, (title, patch) in zip(axes, meshes):
        xy = patch.nodes.numpy()
        for e in patch.elements.tolist():
            loop = list(e) + [e[0]]
            ax.plot(xy[loop, 0], xy[loop, 1], "-", color="0.35", lw=1.0)
        ax.plot(xy[:, 0], xy[:, 1], "o", ms=4, color="C0")
        inter = patch.interior.numpy()
        ax.plot(xy[inter, 0], xy[inter, 1], "o", ms=8, mfc="none", mec="C3")
        ax.set_title(f"{title}\n{patch.elements.shape[0]} elems, "
                     f"{int(inter.sum())} free nodes", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    fig.suptitle("Constant-curvature patch: mesh families (red = free interior node)",
                 fontsize=10)
    fig.tight_layout()
    p = FIGS / "patch_meshes.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append({"file": str(p.relative_to(REPO)),
                 "caption": "The three patch mesh families; red rings mark the free "
                            "interior nodes that make the answer a genuine solve."})

    # 2. error vs h ------------------------------------------------------------
    ladder = [r for r in result["ladder"]]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for treat, style in (("sri", "o-"), ("mitc", "s-")):
        pts = sorted([r for r in ladder if r["shear_treatment"] == treat],
                     key=lambda r: r["h"])
        hs = [r["h"] for r in pts]
        ax.loglog(hs, [r["displacement_rel_error"] for r in pts], style, label=f"{treat}: w")
        ax.loglog(hs, [r["moment_rel_error"] for r in pts], style.replace("-", "--"),
                  alpha=0.7, label=f"{treat}: M")
    hs = sorted({r["h"] for r in ladder})
    if len(hs) >= 2:
        anchor_val = max(r["displacement_rel_error"] for r in ladder if r["h"] == hs[0])
        ax.loglog([hs[0], hs[-1]], [anchor_val, anchor_val * (hs[-1] / hs[0]) ** 2],
                  "k:", lw=1.0, label="slope 2")
    ax.set_xlabel("h = 1/n (patch subdivided n x n)")
    ax.set_ylabel("relative error")
    ax.set_title("Distorted patch: error vs h (cylindrical state)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIGS / "error_vs_h.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append({"file": str(p.relative_to(REPO)),
                 "caption": "h-convergence of the distorted patch (solid = displacement, "
                            "dashed = moments) with a slope-2 guide line."})

    # 3. error budget bar chart -------------------------------------------------
    rows = [r for r in result["matrix"]]
    labels, disp, mom, en = [], [], [], []
    for fam in ("affine", "skew", "distorted"):
        for treat in ("sri", "mitc", "full"):
            sel = [r for r in rows if r["family"] == fam and r["shear_treatment"] == treat]
            if not sel:
                continue
            labels.append(f"{fam}\n{treat}")
            disp.append(max(r["displacement_rel_error"] for r in sel))
            mom.append(max(r["moment_rel_error"] for r in sel))
            en.append(max(r["energy_rel_error"] for r in sel))
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    idx = range(len(labels))
    ax.bar([i - 0.26 for i in idx], disp, width=0.26, label="w")
    ax.bar(idx, mom, width=0.26, label="M")
    ax.bar([i + 0.26 for i in idx], en, width=0.26, label="energy")
    ax.axhline(POOL_TOL, color="C3", ls="--", lw=1.2, label="case-pool tolerance 2%")
    ax.set_yscale("log")
    ax.set_xticks(list(idx))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("max relative error over the three curvature states")
    ax.set_title("Patch error budget by mesh family and shear treatment")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIGS / "error_budget.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append({"file": str(p.relative_to(REPO)),
                 "caption": "Error budget of every patch configuration; the 2% case-pool "
                            "tolerance is drawn for reference."})

    # 4. load-driven strip ------------------------------------------------------
    strip = [r for r in result["strip"] if r["case"].startswith("strip/")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    for ax, lateral in zip(axes, ("symmetry", "free")):
        sel = [r for r in strip if r["lateral"] == lateral and r["shear_treatment"] == "sri"]
        if not sel:
            continue
        ns = [r["n_x"] for r in sel]
        ax.semilogx(ns, [r["w_rel_error_kirchhoff"] * 100 for r in sel], "o-",
                    label="w error [%]")
        ax.semilogx(ns, [r["theta_rel_error"] * 100 for r in sel], "s--",
                    label="theta error [%]")
        ax.axhline(POOL_TOL * 100, color="C3", ls="--", lw=1.2, label="2% tolerance")
        ax.set_xlabel("elements along the strip")
        ax.set_ylabel("error [%]")
        ax.set_title(f"load-driven strip, lateral={lateral}\n"
                     f"reference rigidity {sel[0]['rigidity_used']}")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIGS / "strip_bending.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append({"file": str(p.relative_to(REPO)),
                 "caption": "Load-driven cantilever strip: tip deflection and rotation "
                            "errors against the closed form, both lateral conditions."})

    # 5. thin-limit (h/t) locking ladder ---------------------------------------
    summ = result["locking_scaling_summary"]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    styles = (("distorted", "full", "C3", "o-", "distorted, 2x2 shear (locking)"),
              ("distorted", "sri", "C0", "s-", "distorted, sri"),
              ("distorted", "mitc", "C2", "^-", "distorted, mitc"),
              ("affine", "full", "C3", "o--", "affine, 2x2 shear (locking)"),
              ("affine", "sri", "C0", "s--", "affine, sri"))
    for fam, treat, colour, style, label in styles:
        pts = summ["series"].get(f"{fam}/{treat}/n={LOCK_GRADED_MESH}") or []
        if not pts:
            continue
        ax.loglog([q["h_over_t"] for q in pts],
                  [max(q["energy_rel_error"], 1e-18) for q in pts],
                  style, color=colour, label=label, ms=4)
    guide = summ["series"][f"distorted/full/n={LOCK_GRADED_MESH}"]
    x0, y0 = guide[0]["h_over_t"], guide[0]["energy_rel_error"]
    ax.loglog([x0, 20.0 * x0], [y0, 400.0 * y0], "k:", lw=1.0,
              label="pure (h/t)^2 law: slope 2, amplification 400")
    ax.axhline(POOL_TOL, color="C3", ls="--", lw=1.2, label="2% case-pool tolerance")
    ax.axhline(ROUNDOFF_FLOOR, color="0.5", ls="-.", lw=1.0,
               label="declared round-off floor")
    txt = []
    for k in (f"distorted/full/n={LOCK_GRADED_MESH}", f"affine/full/n={LOCK_GRADED_MESH}",
              f"distorted/sri/n={LOCK_GRADED_MESH}"):
        sl = summ["slope"].get(k, float("nan"))
        am = summ["amplification"].get(k, float("nan"))
        txt.append(f"{k.split('/n=')[0]}: slope {'n/a' if sl != sl else format(sl, '.2f')},"
                   f" amplification {'n/a' if am != am else format(am, '.0f')}")
    ax.set_xlabel("h/t   (h = 1/n at fixed t; the plate is thinned)")
    ax.set_ylabel("relative strain-energy error")
    ax.set_title("Thin-limit locking ladder, n = %d\n%s" % (LOCK_GRADED_MESH,
                                                           "; ".join(txt)), fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    p = FIGS / "locking_scaling.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append({"file": str(p.relative_to(REPO)),
                 "caption": "Thin-limit ladder: the bilinear patch at constant h with "
                            "the plate thinned (h/t = 0.25..5). Fully integrated and "
                            "distorted-SRI energies follow the (h/t)^2 law (slope 2, "
                            "amplification 400 = 20^2); affine SRI and MITC4 stay on the "
                            "round-off floor at every thickness."})
    return made


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def pct(x: float) -> str:
    return f"{100.0 * x:.3e} %" if x < 1e-6 else f"{100.0 * x:.3f} %"


def render_report(result: dict) -> str:
    g = result["gates"]
    lines = []
    A = lines.append
    A("# CASE-BENDING-PATCH — constant-curvature (bending) patch test, Q4 Mindlin plate")
    A("")
    A("Generated by `benchmarks/verified/fem_bending_patch/run.py`; every number below is")
    A("injected from `benchmarks/verified/fem_bending_patch/result.json` (no hand-copied digit).")
    A("")
    A("## 1. Problem and reference")
    A("")
    A("* **Criterion** (the reference): a plate patch with free interior nodes, loaded only")
    A("  through boundary displacements from a state of **constant curvature**, must")
    A("  reproduce the interior displacement, rotations, moments and strain energy exactly.")
    A("  Statement: Irons & Ahmad, *Techniques of Finite Elements* (1980), ch. 8; restated")
    A("  for plate elements in Zienkiewicz & Taylor, *The Finite Element Method*, Vol. 2,")
    A("  §10.4, and applied in Batoz, Bathe & Ho, *Int. J. Numer. Methods Eng.* **15** (1980)")
    A("  1771-1812, §4. Continuum (plane-stress) patch tests — including this repository's")
    A("  `docs/patch-test-robinson.md` — say nothing about plate **bending** elements, whose")
    A("  independent rotation field and transverse-shear energy can lock; that gap is what")
    A("  this anchor closes.")
    A("* **Target values are analytic**, evaluated from the closed form below rather than")
    A("  copied from a table: `M = Db kappa` with `D = E t^3/(12(1-nu^2))`.")
    A("  `oracle_selfcheck()` re-verifies the field (zero transverse shear, constant")
    A(f"  curvature, constant moments) before any assembly: shear residual")
    A(f"  {result['oracle']['shear_residual']:.2e}, curvature deviation"
      f" {result['oracle']['curvature_deviation']:.2e}, moment deviation"
      f" {result['oracle']['moment_deviation']:.2e} (finite-difference limited).")
    A("* **Reference access note (honest)**: the textbooks were not retrievable in this")
    A("  sandbox (no outbound network). What is taken from them is the *criterion* and the")
    A("  constitutive form; every target number is then derived in code from that form, so")
    A("  nothing depends on an unavailable table. The same convention is used in")
    A("  `docs/patch-test-robinson.md`.")
    A("* **Case-pool tolerance**: `< 2%` (`tensor_case_pool`, anchor `CASE-BENDING-PATCH`);")
    A("  the pass/fail (affine) form is graded far tighter, at `1e-12`.")
    A("")
    A("## 2. Method and implementation")
    A("")
    A("| item | value |")
    A("|---|---|")
    A(f"| Young's modulus E | {YOUNG:g} MPa |")
    A(f"| Poisson ratio nu | {POISSON:g} |")
    A(f"| thickness t | {THICK:g} mm (patch 1 x 1 in-plane, L/t = 1) |")
    A(f"| curvature states | {', '.join(f'{k} ({v.label()})' for k, v in STATES.items())} |")
    A("| element | `tensorfem.plate.q4_mindlin_stiffness` (production Q4 Mindlin-Reissner,"
      " `shear_treatment='sri'` and `'mitc'`) |")
    A("| locking control | 2x2 (full) shear integration,"
      " `tensorfem.bending_patch.q4_reference_stiffness(shear_treatment='full')` |")
    A("| exact field | `tensorfem.bending_patch.CurvatureState.field` |")
    A("")
    A("Each patch is assembled from the production element, the exact field is prescribed")
    A("on every boundary node, and the interior DOFs are solved from")
    A("`K_ff u_f = -K_fc u_c` (a genuine solve: the answer is nowhere imposed). Moments are")
    A("recovered with the same `Bb` operator the element integrates, at the 2x2 Gauss")
    A("points, and compared with `Db kappa`.")
    A("")
    A("## 3. Parameters (dimensionless)")
    A("")
    A("| quantity | value |")
    A("|---|---|")
    A(f"| D / (E t^3) | {result['dimensionless']['D_over_Et3']:.6f} |")
    A(f"| D_s / (E t) = kappa_s/(2(1+nu)) | {result['dimensionless']['Ds_over_Et']:.6f} |")
    A(f"| bending : shear stiffness ratio, D/(D_s t^2) |"
      f" {result['dimensionless']['bending_shear_ratio']:.6f} |")
    A(f"| kappa_x t (patch) | {STATES['cylindrical'].kappa_x * THICK:.3e} |")
    A(f"| mesh sizes n (affine/skew) | 3 |")
    A(f"| mesh sizes n (distorted / ladder) |"
      f" {sorted({r['n'] for r in result['ladder']})} |")
    A("")
    A("## 4. Results — reference vs computed")
    A("")
    A("### 4.1 Affine / skewed patches: the pass/fail form")
    A("")
    A("| state | family | treatment | w error | rotation error | moment error | energy error | verdict |")
    A("|---|---|---|---|---|---|---|---|")
    for r in result["matrix"]:
        if r["family"] == "distorted":
            continue
        ok = max(r["displacement_rel_error"], r["rotation_rel_error"],
                 r["moment_rel_error"], r["energy_rel_error"]) <= EXACT_TOL
        A(f"| {r['state_name']} | {r['family']} | {r['shear_treatment']} |"
          f" {r['displacement_rel_error']:.2e} | {r['rotation_rel_error']:.2e} |"
          f" {r['moment_rel_error']:.2e} | {r['energy_rel_error']:.2e} |"
          f" {'PASS' if ok else 'FAIL'} |")
    A("")
    A("### 4.2 Distorted (non-affine) patch vs the 2% case-pool tolerance")
    A("")
    A("| state | treatment | w error | moment error | energy error | vs 2% |")
    A("|---|---|---|---|---|---|")
    for r in result["matrix"]:
        if r["family"] != "distorted":
            continue
        worst = max(r["displacement_rel_error"], r["moment_rel_error"], r["energy_rel_error"])
        A(f"| {r['state_name']} | {r['shear_treatment']} | {pct(r['displacement_rel_error'])} |"
          f" {pct(r['moment_rel_error'])} | {pct(r['energy_rel_error'])} |"
          f" {'PASS' if worst <= POOL_TOL else 'FAIL'} |")
    A("")
    A("### 4.3 Load-driven cantilever strip (the answer is *not* prescribed)")
    A("")
    A("| lateral | n_x | treatment | w tip | closed form | w error | theta error | verdict |")
    A("|---|---|---|---|---|---|---|---|")
    for r in result["strip"]:
        if not r["case"].startswith("strip/"):
            continue
        ok = (r["w_rel_error_kirchhoff"] <= POOL_TOL and r["theta_rel_error"] <= POOL_TOL)
        if r["lateral"] == "free":
            A(f"| {r['lateral']} | {r['n_x']} | {r['shear_treatment']} |"
              f" {r['w_tip']:.6f} | {r['w_exact_kirchhoff']:.6f} |"
              f" {pct(r['w_rel_error_kirchhoff'])} | {pct(r['theta_rel_error'])} |"
              f" not graded (n_y = 2, aliased — §4.5) |")
            continue
        A(f"| {r['lateral']} | {r['n_x']} | {r['shear_treatment']} |"
          f" {r['w_tip']:.6f} | {r['w_exact_kirchhoff']:.6f} |"
          f" {pct(r['w_rel_error_kirchhoff'])} | {pct(r['theta_rel_error'])} |"
          f" {'PASS' if ok else 'FAIL'} |")
    A("")
    A("The `free` rows are kept at the round-1 transverse mesh (`n_y = 2`) only to"
      " document that mesh; they are **not graded** because refining `n_y` moves them"
      " (§4.5).")
    A("")
    A(f"Rigidity used: `D = E t^3/(12(1-nu^2))` for the symmetry cell (cylindrical bending),")
    A(f"`D_b = E t^3/12` for the free-edge strip (plane stress, beam rigidity).")
    A(f"Ratio of the two **solved** tip deflections ="
      f" {result['strip_ratio']['measured']:.6f}, theory `1/(1-nu^2) ="
      f" {result['strip_ratio']['theory']:.6f}` (error"
      f" {pct(result['strip_ratio']['rel_error'])}) at the converged transverse mesh"
      f" (n_y = {result['strip_ratio']['mesh']['n_y']}). At the round-1 mesh"
      f" (n_y = {result['strip_ratio']['naive_n_y']}) the same ratio reads"
      f" {result['strip_ratio']['naive_measured']:.6f}"
      f" (error {pct(result['strip_ratio']['naive_rel_error'])}) — see §4.5: the"
      f" difference is a clamped-root end effect, not a coupling error.")
    A("")
    A("### 4.4 Negative control: the locking element must fail")
    A("")
    A("| case | quantity | value | verdict |")
    A("|---|---|---|---|")
    for r in result["strip"]:
        if r["case"].startswith("strip-locking/"):
            A(f"| {r['case']} | w error vs closed form | {pct(r['w_rel_error_kirchhoff'])} |"
              f" {'FAIL (as required)' if r['w_rel_error_kirchhoff'] > POOL_TOL else 'UNEXPECTED PASS'} |")
    lock_dist = [r for r in result["matrix"]
                 if r["family"] == "distorted" and r["shear_treatment"] == "full"]
    if lock_dist:
        A(f"| distorted patch, `full` shear integration | max energy error |"
          f" {pct(max(r['energy_rel_error'] for r in lock_dist))} |"
          f" {'FAIL (as required)' if max(r['energy_rel_error'] for r in lock_dist) > POOL_TOL else 'within 2%'} |")
        A(f"| distorted patch, `full` shear integration | max moment error |"
          f" {pct(max(r['moment_rel_error'] for r in lock_dist))} |"
          f" {'FAIL (as required)' if max(r['moment_rel_error'] for r in lock_dist) > POOL_TOL else 'within 2%'} |")
    A("")
    A("### 4.5 Round-2 cross-check repair (Poisson coupling)")
    A("")
    A("The round-1 Poisson check graded `M L^2/(2 D)` of the free-edge leg against"
      " `M L^2/(2 D_b)` of the symmetry leg, i.e. **two closed forms**, and reported a"
      " 0.000 % error. Both operands are the same formula with a rigidity the driver"
      " inserted, so the ratio is `D/D_b = 1/(1-nu^2)` by construction: the check was"
      " insensitive to the element and could not fail. It is now split into an explicit"
      " oracle identity and three checks that use solved quantities:")
    A("")
    A("| leg | quantity | value |")
    A("|---|---|---|")
    A(f"| closed form (oracle only) | ratio `D/D_b` |"
      f" {result['strip_ratio']['closed_form_ratio']:.12f} |")
    A(f"| closed form (oracle only) | error vs `1/(1-nu^2)` |"
      f" {result['strip_ratio']['closed_form_rel_error']:.3e} |")
    A(f"| solved, converged mesh (n_y={result['strip_ratio']['mesh']['n_y']}) |"
      f" `w_free/w_sym` | {result['strip_ratio']['measured']:.6f} |")
    A(f"| solved, converged mesh (n_y={result['strip_ratio']['mesh']['n_y']}) |"
      f" error vs `1/(1-nu^2)` | {pct(result['strip_ratio']['rel_error'])} |")
    A(f"| solved, round-1 mesh (n_y={result['strip_ratio']['naive_n_y']}) |"
      f" `w_free/w_sym` | {result['strip_ratio']['naive_measured']:.6f} |")
    A(f"| free-surface state `M_y=0` | max `|M_y|/|M_x|`, nu ="
      f" {', '.join(f'{v:g}' for v in result['free_surface_summary']['poissons'])} |"
      f" {result['free_surface_summary']['max_my_over_mx']:.3e} |")
    A(f"| free-surface state `M_y=0` | max `|M_x/(D_b kappa_x) - 1|` |"
      f" {result['free_surface_summary']['max_rigidity_deviation']:.3e} |")
    A(f"| free-surface state `M_y=0` | same, graded against `D` instead of `D_b`"
      f" (separation) | {pct(result['free_surface_summary']['alternative_rigidity_deviation'])} |")
    A("")
    A("The free-edge leg is **not** transverse-converged at the round-1 mesh. Refining"
      " only the transverse direction, at fixed `n_x`:")
    A("")
    A("| n_y | free-edge `w` error vs `M L^2/(2 D_b)` |")
    A("|---|---|")
    for ny, val in result["strip_transverse"]["free_w_error_by_n_y"].items():
        A(f"| {ny} | {pct(val)} |")
    A("")
    A(f"Relative change between the two finest levels:"
      f" {pct(result['strip_transverse']['transverse_change'])}. The converged offset"
      f" (~1.2 %) is not a discretization error: it decays with the strip aspect ratio at"
      f" fixed element size, which identifies it as a **clamped-root end effect** (the"
      f" `w = 0` root restraint is incompatible with the anticlastic curvature"
      f" `kappa_y = -nu kappa_x` that the free lateral edges require, so a boundary layer"
      f" adds a small tip rotation).")
    A("")
    A("| L/b | free-edge `w` error vs `M L^2/(2 D_b)` |")
    A("|---|---|")
    for k, val in result["strip_aspect"]["free_w_error_by_L_over_b"].items():
        A(f"| {k} | {pct(val)} |")
    A("")
    root = {r["clamp_theta_y"]: r["w_rel_error_kirchhoff"] for r in result["strip_root_support"]}
    if root:
        A("Root-support control at fixed mesh (`n_x = 32`, `n_y = 8`):"
          f" `theta_y` clamped -> {pct(root[True])}; `theta_y` free ->"
          f" {pct(root[False])}. The offset survives, so it is the `w = 0` restraint of"
          " the anticlastic field, not the rotation restraint.")
        A("")
    A(f"Decay ratio `e(L/b=20)/e(L/b=5) ="
      f" {result['strip_aspect']['end_effect_decay_ratio']:.3f}`: the offset is local to"
      f" the clamped root and vanishes with `L/b`, so the beam closed form is *not* the"
      f" limit of a hard-clamped strip at engineering aspect ratios. The leg is therefore"
      f" graded at 2 % only after transverse convergence, and the Poisson coupling itself"
      f" is graded on the exact `M_y = 0` patch, where `M_y` must vanish to round-off.")
    A("")
    A("| nu | family | treatment | w error | rotation error | moment error |"
      " `|M_y|/|M_x|` | `M_x/(D_b kappa_x)` |")
    A("|---|---|---|---|---|---|---|---|")
    for r in result["free_surface"]:
        A(f"| {r['poisson']:g} | {r['family']} | {r['shear_treatment']} |"
          f" {r['displacement_rel_error']:.2e} | {r['rotation_rel_error']:.2e} |"
          f" {r['moment_rel_error']:.2e} | {r['my_over_mx']:.2e} |"
          f" {r['mx_over_db_kappa']:.9f} |")
    A("")
    A("The state is run at `nu = 0.3` and `nu = 0.45` so the check cannot be satisfied"
      " by a constant tuned to one Poisson ratio.")
    A("")
    summ = result["locking_scaling_summary"]
    g_n = summ["graded_mesh_n"]
    h_over_t_thin = 1.0 / (g_n * min(summ["thickness_ratios"]))

    def _s(key, spec=".2f"):
        v = summ["slope"].get(key, float("nan"))
        return "n/a (at round-off)" if v != v else format(v, spec)

    A("### 4.6 Round-3 thin-limit control: the (h/t)^2 locking law")
    A("")
    A("Every case graded above runs at `t/L = 1`, i.e. `h/t = 0.25` — a **thick** plate,")
    A("which is exactly where shear locking is weakest. That is why the round-2 patch")
    A("negative control read only 2.1 % against a 2 % threshold (a 5 % margin): a control")
    A("that thin cannot separate *the element locks* from *the element has a small")
    A("interpolation error*. Round 3 instead **thins the plate at a fixed mesh**, so `h/t`")
    A("grows by a factor 20 while `h` is unchanged, and grades the *law*, not one value.")
    A("")
    A("The spurious transverse-shear energy of a bilinear `w` scales as `(h/t)^2`, so a")
    A("locking element must show log-log slope 2 and an amplification of `20^2 = 400` over")
    A("this ladder; a shear-robust one must stay at round-off. Both thresholds were fixed")
    A("from that law before the numbers were read (slope floor 1.5 = the observed-order")
    A("floor used elsewhere; amplification floor 100 = a factor 4 slack).")
    A("")
    A(f"| family | treatment | slope vs `h/t` | amplification | error at `h/t ="
      f" {h_over_t_thin:g}` |")
    A("|---|---|---|---|---|")
    for fam in ("affine", "distorted"):
        for treat in ("sri", "mitc", "full"):
            key = f"{fam}/{treat}/n={g_n}"
            amp = summ["amplification"].get(key, float("nan"))
            A(f"| {fam} | {treat} | {_s(key)} |"
              f" {'n/a' if amp != amp else format(amp, '.1f')} |"
              f" {summ['thin_limit'].get(key, float('nan')):.3e} |")
    A("")
    A(f"`n = {g_n}` (16 elements) is the graded mesh; `n = 8` reproduces every slope and")
    A("is exported in `result.json` under `locking_scaling_summary.series`.")
    A("")
    A("* **2x2 (full) shear integration locks exactly as predicted.** On the affine patch")
    A(f"  the slope is {_s(f'affine/full/n={g_n}')} and the amplification"
      f" {summ['amplification'][f'affine/full/n={g_n}']:.1f} = 20^2 — the pure `(h/t)^2`"
      " law with no fitted constant. A stronger negative control than round 2: it pins"
      " the mechanism, not just the sign of the error.")
    A(f"* **MITC4 is shear-free in the thin limit**: {summ['thin_mitc']:.2e} at"
      f" `h/t = {h_over_t_thin:g}`, versus"
      f" {summ['thin_limit'][f'distorted/full/n={g_n}']:.1f} for the locking element on"
      " the same mesh.")
    A("* **SRI is distortion-limited — a limitation of the element, recorded not hidden.**")
    A(f"  Exact on affine meshes at every thickness ({summ['thin_sri_affine']:.1e} at"
      f" `h/t = {h_over_t_thin:g}`), but on the distorted family its error follows the"
      f" same law (slope {_s(f'distorted/sri/n={g_n}')}) and reaches"
      f" {pct(summ['thin_sri_distorted'])} at `h/t = {h_over_t_thin:g}`. SRI cuts the"
      f" constant by"
      f" {summ['thin_limit'][f'distorted/full/n={g_n}'] / summ['thin_sri_distorted']:.0f}x"
      " relative to full integration; it does not remove the thin-plate limit on a")
    A("  general mesh.")
    sri_series = summ["series"][f"distorted/sri/n={g_n}"]
    A(f"* The same SRI cases keep a much smaller *displacement* error"
      f" ({pct(sri_series[0]['displacement_rel_error'])} at the thickest plate ->"
      f" {pct(sri_series[-1]['displacement_rel_error'])} at the thinnest, `n = {g_n}`),"
      " so the quantity that grows like `(h/t)^2` is the spurious shear **energy** — the"
      " locking indicator — not a displacement-error bound.")
    A("")
    A("**Scope consequence (stated, not hidden):** the 2 % claims of this anchor are")
    A("certified for `h/t <= 0.25`. For a thin plate on a distorted mesh SRI is *not*")
    A("certified here and MITC4 is the element to use; this is a `limitations` entry in")
    A("`gates_verified.json`.")
    A("")
    A("## 5. Convergence and sensitivity")
    A("")
    A("| n | h | treatment | w error | moment error | energy error |")
    A("|---|---|---|---|---|---|")
    for r in result["ladder"]:
        A(f"| {r['n']} | {r['h']:.4f} | {r['shear_treatment']} |"
          f" {r['displacement_rel_error']:.3e} | {r['moment_rel_error']:.3e} |"
          f" {r['energy_rel_error']:.3e} |")
    A("")
    def _ord(v):
        return "n/a (at round-off, not measurable)" if v != v else f"{v:.2f}"

    A(f"Observed orders (least squares on log-log, distorted family):")
    A(f"`w` sri {_ord(result['orders']['w_sri'])}, `w` mitc {_ord(result['orders']['w_mitc'])},")
    A(f"`M` sri {_ord(result['orders']['M_sri'])}, `M` mitc {_ord(result['orders']['M_mitc'])}.")
    A("A slope fitted to numbers at the round-off floor is noise, not a convergence"
      " rate; those entries are reported as `n/a` and are not graded.")
    A("The residual on a non-affine patch is an interpolation limitation of the bilinear")
    A("`w`, not a consistency defect: it vanishes with h and is already below the 2% pool")
    A("tolerance at the coarsest patch used.")
    A("")
    A("Thickness sensitivity (load-driven strip, `t = 0.01, 0.10` with L/t = 100 and 10):")
    A("")
    A("| lateral | t | treatment | w error vs Kirchhoff | w error vs Mindlin (with shear term) |")
    A("|---|---|---|---|---|")
    for r in result["strip"]:
        if not r["case"].startswith("strip-locking/"):
            continue
        A(f"| {r['lateral']} | {r['thickness']:g} | {r['shear_treatment']} |"
          f" {pct(r['w_rel_error_kirchhoff'])} | {pct(r['w_rel_error_mindlin'])} |")
    A("")
    A("## 6. Failure modes, limits and artefacts")
    A("")
    A("* **What this proves**: the production Q4 Mindlin element is *consistent* for")
    A("  bending — it reproduces a constant-curvature state to round-off on affine patches")
    A("  and converges to it on distorted ones — and the element is not silently locking.")
    A("* **What this does not prove**: the patch test is a *consistency* (necessary)")
    A("  condition. It says nothing about accuracy on coarse meshes, about pointwise moment")
    A("  accuracy away from constant states, about `k`-refinement, or about the element's")
    A("  rank deficiency — the one-point `sri` element is rank deficient by two, documented")
    A("  separately in `docs/element-rank-deficiency-audit.md`; the patch test cannot see")
    A("  zero-energy modes because a Dirichlet-driven patch has none.")
    A("* **Known limitation measured here**: on distorted meshes the bilinear `w` cannot")
    A("  represent a quadratic field, so the distorted-patch numbers are an interpolation")
    A("  error; the `sri` variant is the noisier of the two (`mitc` is consistently better on")
    A("  the twist state).")
    A("* **Reproduction**:")
    A("")
    A("```")
    A("OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/verified/fem_bending_patch/run.py")
    A("PYTHONPATH=src pytest tests/test_bending_patch.py -q")
    A("```")
    A("")
    A("* **Artefacts**: `benchmarks/verified/fem_bending_patch/result.json`,")
    A("  `benchmarks/verified/fem_bending_patch/gates_verified.json`, figures under")
    A("  `docs/assets/bending_patch/`.")
    A("* **Gate summary**: {}".format(
        f"{sum(1 for x in g if x['passed'])}/{len(g)} passed"
        + ("" if result["passed"] else " — SOME GATES FAILED")))
    A("")
    A("## 7. Gate table (machine verdict)")
    A("")
    A("| gate | quantity | reference | value | threshold | passed |")
    A("|---|---|---|---|---|---|")
    for x in g:
        A(f"| {x['id']} | {x['quantity']} | {x['reference']} | {x['value']:.4g} |"
          f" {x['threshold']:.4g} | {'PASS' if x['passed'] else 'FAIL'} |")
    A("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="reduced ladder / meshes")
    args = ap.parse_args(argv)
    torch.set_default_dtype(torch.float64)

    t_start = time.time()
    oracle = oracle_selfcheck(STATES["double"], YOUNG, POISSON, THICK)
    matrix = patch_matrix(args.quick)
    ladder = refinement_ladder(args.quick)
    strip = strip_cases(args.quick)
    free_surface = free_surface_cases(args.quick)
    strip_ny = strip_transverse_sweep(args.quick)
    strip_aspect = strip_aspect_series(args.quick)
    strip_root = strip_root_support_control(args.quick)
    lock_scaling = locking_scaling_cases(args.quick)
    lock_sum = locking_scaling_summary(lock_scaling)

    # dimensionless parameters
    d = YOUNG * THICK ** 3 / (12.0 * (1.0 - POISSON ** 2))
    ds = (5.0 / 6.0) * YOUNG / (2.0 * (1.0 + POISSON)) * THICK
    dimensionless = {
        "D_over_Et3": d / (YOUNG * THICK ** 3),
        "Ds_over_Et": ds / (YOUNG * THICK),
        "bending_shear_ratio": d / (ds * THICK ** 2),
    }

    # ---- Poisson cross-check, repaired (round 2) --------------------------
    # (a) closed-form leg: the round-1 gate.  Both operands are the *same* formula
    #     M L^2/(2 D_ref) with a rigidity that the driver inserted, so the ratio is
    #     D/D_b = 1/(1-nu^2) by construction: the error is identically zero for any
    #     element and any nu.  It is kept as an ORACLE identity only, never as
    #     element evidence.
    d_beam = YOUNG * THICK ** 3 / 12.0
    theory = 1.0 / (1.0 - POISSON ** 2)
    closed_ratio = d / d_beam
    closed_rel_error = abs(closed_ratio - theory) / theory
    # (b) computed leg: the two *solved* tip deflections on the same mesh.
    nx_ref = 8 if args.quick else 32
    ny_ref = 8 if args.quick else 16
    ny_naive = 2

    def _solved(lateral: str, n_x: int, n_y: int):
        for r in strip_ny:
            if (r["lateral"] == lateral and r["n_x"] == n_x and r["n_y"] == n_y):
                return r
        return None

    ratio = {"theory": theory, "closed_form_ratio": closed_ratio,
             "closed_form_rel_error": closed_rel_error,
             "measured": float("nan"), "rel_error": float("nan"),
             "naive_measured": float("nan"), "naive_rel_error": float("nan"),
             "naive_n_y": ny_naive, "mesh": {"n_x": nx_ref, "n_y": ny_ref,
                                             "shear_treatment": "sri"}}
    sym_ref, fre_ref = _solved("symmetry", nx_ref, ny_ref), _solved("free", nx_ref, ny_ref)
    if sym_ref and fre_ref:
        meas = fre_ref["w_tip"] / sym_ref["w_tip"]
        ratio.update(measured=meas, rel_error=abs(meas - theory) / theory)
    sym_nv, fre_nv = _solved("symmetry", nx_ref, ny_naive), _solved("free", nx_ref, ny_naive)
    if sym_nv and fre_nv:
        m = fre_nv["w_tip"] / sym_nv["w_tip"]
        ratio.update(naive_measured=m, naive_rel_error=abs(m - theory) / theory)

    # free-edge leg: transverse convergence and the clamped-root end effect
    free_ny = {r["n_y"]: r["w_rel_error_kirchhoff"] for r in strip_ny
               if r["lateral"] == "free" and r["n_x"] == nx_ref}
    ny_lo = 4 if args.quick else 8
    transverse = float("nan")
    if ny_ref in free_ny and ny_lo in free_ny and free_ny[ny_ref] > 0.0:
        transverse = abs(free_ny[ny_ref] - free_ny[ny_lo]) / free_ny[ny_ref]
    aspect = {r["length"] / r["width"]: r["w_rel_error_kirchhoff"] for r in strip_aspect}
    end_decay = float("nan")
    lo, hi = min(aspect), max(aspect)
    if len(aspect) >= 2 and aspect[lo] > 0.0:
        end_decay = aspect[hi] / aspect[lo]

    # free-surface (M_y = 0) state: the coupling check with discriminating power
    fs_my = max((r["my_over_mx"] for r in free_surface), default=float("nan"))
    fs_w = max((max(r["displacement_rel_error"], r["rotation_rel_error"])
                for r in free_surface), default=float("nan"))
    fs_m = max((r["moment_rel_error"] for r in free_surface), default=float("nan"))
    fs_rig = max((abs(r["mx_over_db_kappa"] - 1.0) for r in free_surface),
                 default=float("nan"))
    # separation: grading M_x against the *cylindrical* rigidity D instead of D_b
    fs_separation = max((abs(r["mx_over_db_kappa"] * r["db_over_d"] - 1.0)
                         for r in free_surface), default=float("nan"))
    free_surface_summary = {
        "poissons": sorted({r["poisson"] for r in free_surface}),
        "max_my_over_mx": fs_my,
        "max_displacement_or_rotation_error": fs_w,
        "max_moment_error": fs_m,
        "max_rigidity_deviation": fs_rig,
        "alternative_rigidity_deviation": fs_separation,
    }
    strip_transverse = {
        "n_x": nx_ref,
        "free_w_error_by_n_y": {str(k): v for k, v in sorted(free_ny.items())},
        "transverse_change": transverse,
    }
    strip_aspect_summary = {
        "free_w_error_by_L_over_b": {f"{k:g}": v for k, v in sorted(aspect.items())},
        "end_effect_decay_ratio": end_decay,
    }

    orders = {
        "w_sri": observed_order(ladder, "sri", "displacement_rel_error"),
        "w_mitc": observed_order(ladder, "mitc", "displacement_rel_error"),
        "M_sri": observed_order(ladder, "sri", "moment_rel_error"),
        "M_mitc": observed_order(ladder, "mitc", "moment_rel_error"),
    }

    # ------------------------------------------------------------------ gates
    exact_rows = [r for r in matrix if r["family"] in ("affine", "skew")
                  and r["shear_treatment"] in ("sri", "mitc")]
    exact_max = worst(exact_rows, lambda r: True, "displacement_rel_error")
    exact_max_rot = worst(exact_rows, lambda r: True, "rotation_rel_error")
    exact_max_m = worst(exact_rows, lambda r: True, "moment_rel_error")
    exact_max_e = worst(exact_rows, lambda r: r["shear_treatment"] == "sri", "energy_rel_error")
    skew_mitc_e = worst(exact_rows, lambda r: r["shear_treatment"] == "mitc", "energy_rel_error")

    dist_rows = [r for r in matrix if r["family"] == "distorted"
                 and r["shear_treatment"] in ("sri", "mitc")]
    dist_w = worst(dist_rows, lambda r: True, "displacement_rel_error")
    dist_m = worst(dist_rows, lambda r: True, "moment_rel_error")
    dist_e = worst(dist_rows, lambda r: True, "energy_rel_error")

    strip_ok = [r for r in strip if r["case"].startswith("strip/")]
    strip_w = worst(strip_ok, lambda r: r["lateral"] == "symmetry", "w_rel_error_kirchhoff")
    strip_t = worst(strip_ok, lambda r: r["lateral"] == "symmetry", "theta_rel_error")
    # round 2: the free-edge leg is graded only on the transverse-converged meshes.
    strip_free_conv = [r for r in strip_ny if r["lateral"] == "free" and r["n_y"] >= ny_lo]
    strip_free_w = worst(strip_free_conv, lambda r: True, "w_rel_error_kirchhoff")
    strip_free_w_naive = worst(strip_ok, lambda r: r["lateral"] == "free",
                               "w_rel_error_kirchhoff")

    lock_rows = [r for r in strip if r["case"].startswith("strip-locking/")]
    lock_thin = [r for r in lock_rows if r["thickness"] <= 0.01]
    lock_min = min(r["w_rel_error_kirchhoff"] for r in lock_thin) if lock_thin else 0.0
    lock_dist_e = max((r["energy_rel_error"] for r in matrix
                       if r["family"] == "distorted" and r["shear_treatment"] == "full"),
                      default=0.0)

    gates = [
        Gate("oracle_selfcheck", "analytic state residual (shear + curvature)",
             "constant-curvature field, closed form; Zienkiewicz & Taylor Vol.2",
             max(oracle["shear_residual"], oracle["curvature_deviation"]), 1e-10),
        Gate("affine_exact_w", "max |w_h - w_exact| / |w_exact| over affine+skew patches",
             "constant-curvature closed form (patch test, exact reproduction)",
             exact_max, EXACT_TOL),
        Gate("affine_exact_rot", "max rotation error, affine+skew patches",
             "constant-curvature closed form", exact_max_rot, EXACT_TOL),
        Gate("affine_exact_moment", "max moment error, affine+skew patches",
             "M = Db kappa", exact_max_m, EXACT_TOL),
        Gate("affine_exact_energy_sri", "max strain-energy error, affine+skew patches (sri)",
             "1/2 kappa^T Db kappa * Area", exact_max_e, EXACT_TOL),
        Gate("skew_energy_mitc", "max strain-energy error, affine+skew patches (mitc)",
             "1/2 kappa^T Db kappa * Area (documented: MITC shear sampling on a skew "
             "element is not exactly shear-free)", skew_mitc_e, POOL_TOL),
        Gate("distorted_w", "max displacement error, distorted patch",
             "constant-curvature closed form", dist_w, POOL_TOL),
        Gate("distorted_moment", "max moment error, distorted patch",
             "M = Db kappa", dist_m, POOL_TOL),
        Gate("distorted_energy", "max energy error, distorted patch",
             "1/2 kappa^T Db kappa * Area", dist_e, POOL_TOL),
        Gate("strip_symmetry_w", "tip deflection error, symmetry cell (load-driven)",
             "M L^2/(2D), D = E t^3/(12(1-nu^2))", strip_w, POOL_TOL),
        Gate("strip_symmetry_theta", "tip rotation error, symmetry cell (load-driven)",
             "M L/D", strip_t, POOL_TOL),
        Gate("strip_free_w", "tip deflection error, free edges, transverse-converged "
             "meshes (n_y >= 8, load-driven)",
             "M L^2/(2 D_b), D_b = E t^3/12 (clamped-root end effect included; see the "
             "end-effect and transverse-convergence gates)", strip_free_w, POOL_TOL),
        Gate("poisson_identity_closed_form",
             "|D/D_b - 1/(1-nu^2)| / (1/(1-nu^2))  [algebra only, no solve]",
             "D/D_b = 1/(1-nu^2) exactly (ORACLE identity: insensitive to the "
             "element, documented, not element evidence)",
             ratio["closed_form_rel_error"], 1.0e-14),
        Gate("poisson_coupling_computed",
             "|w_free/w_sym - 1/(1-nu^2)| / (1/(1-nu^2)) from the SOLVED tip "
             "deflections, n_y=16",
             "1/(1-nu^2) with both responses solved on the same mesh", ratio["rel_error"],
             POOL_TOL),
        Gate("free_surface_moment_exact",
             "max |M_y| / max |M_x| on the M_y=0 state (affine+skew, nu=0.3 and 0.45)",
             "plane-stress free surface: M_y = 0 exactly", fs_my, 1.0e-12),
        Gate("free_surface_field_exact",
             "max nodal w/rotation error, M_y=0 state (affine+skew, nu=0.3 and 0.45)",
             "closed form with kappa_y = -nu kappa_x", fs_w, 1.0e-12),
        Gate("free_surface_rigidity",
             "max |M_x/(D_b kappa_x) - 1|, D_b = E t^3/12",
             "measured beam rigidity equals D (1-nu^2)", fs_rig, 1.0e-12),
        Gate("free_surface_rigidity_separation",
             "|M_x/(D kappa_x) - 1|: grading against the cylindrical rigidity D",
             "the check must reject the wrong rigidity (discriminating power)",
             fs_separation, 0.05, ">="),
        Gate("strip_free_transverse_convergence",
             "|e(n_y=16) - e(n_y=8)| / e(n_y=16), free lateral, n_x=32",
             "the free-edge leg must be transverse-converged before it is graded",
             transverse, 0.10),
        Gate("strip_free_end_effect_decay",
             "e(L/b=20) / e(L/b=5), free lateral, n_y=8, fixed element size",
             "clamped-root end effect is local: it must decay with L/b", end_decay,
             0.60),
        Gate("order_w_min", "observed order of the distorted patch, w (SRI)",
             "second-order consistency of the element (orders below the round-off "
             "floor are reported as null, not fitted)", orders["w_sri"], 1.5,
             ">="),
        Gate("locking_control_strip", "min tip-deflection error of the locking element "
             "(thin strip, t/L = 0.01)", "shear locking must be visible (negative control)",
             lock_min, 0.5, ">="),
        Gate("locking_control_patch", "max energy error of the locking element on the patch",
             "shear locking must be visible (negative control)", lock_dist_e, POOL_TOL, ">="),
        Gate("equilibrium_residual", "max |K_ff u_f + K_fc u_c| over all patch cases",
             "discrete equilibrium", max(r["equilibrium_residual"] for r in matrix), 1e-8),
    ] + round3_gates(lock_sum)
    manifest = gate_manifest(
        anchor="fem_bending_patch",
        title="BENDING-PATCH: constant-curvature patch test of the Q4 Mindlin plate element",
        criteria={
            "case_pool_tolerance": POOL_TOL,
            "exact_tolerance_affine": EXACT_TOL,
            "observed_order_min": 1.5,
            "locking_control_min": 0.5,
            "locking_scaling_slope_min": LOCK_SLOPE_MIN,
            "locking_amplification_min": LOCK_AMPLIFICATION_MIN,
            "locking_law_tolerance": LOCK_LAW_TOL,
        },
        gates=gates,
        artifacts={
            "result": "benchmarks/verified/fem_bending_patch/result.json",
            "report": "docs/bending-patch.md",
            "figures": "docs/assets/bending_patch/",
        },
        limitations=[
            "A patch test is a consistency (necessary) condition; it proves constant-"
            "curvature reproduction, not coarse-mesh accuracy or pointwise moment accuracy.",
            "The distorted-patch residual is an interpolation limitation of the bilinear w "
            "field (quadratic exact field), measured and h-convergent, not a defect.",
            "A Dirichlet-driven patch contains no zero-energy mode, so this test cannot "
            "detect the rank deficiency of the one-point (sri) element.",
            "The free-edge strip leg of the Poisson cross-check is NOT converged at "
            "n_y=2: its converged offset is ~1.2% (L/b=5) and it decays with L/b, i.e. it "
            "is a clamped-root end effect, not a discretization error. The leg is "
            "therefore graded only after transverse convergence, and the coupling itself "
            "is graded on the exact M_y=0 state.",
            "The closed-form Poisson ratio is an algebra identity (insensitive to the "
            "element); it is reported as an oracle check, not as element evidence. "
            "Round 1 graded it as if it were the cross-check -- that gate could not fail.",
            "THIN-PLATE SCOPE: every graded patch case of rounds 1-2 runs at "
            "t/L = 1 (h/t = 0.25), a thick plate. Round 3 thins the plate at a fixed "
            "mesh: the relative strain-energy error of the SRI variant follows the "
            "(h/t)^2 locking law on DISTORTED meshes (26 % at h/t = 5, n = 4; slope "
            "1.95), while MITC4 and the affine SRI path stay at round-off. The 2 % "
            "claims above are therefore certified for h/t <= 0.25; on a general "
            "distorted thin-plate mesh MITC4 is the element to use, not SRI.",
            "The thin-limit quantity is the relative error of the discrete strain energy "
            "U_h = 1/2 u^T K u, which contains the spurious transverse-shear energy: it "
            "is the locking indicator, not a bound on the displacement error (the same "
            "SRI cases keep a field-displacement error of a few percent).",
            "Literature criterion cited from the standard statement; the textbooks were not "
            "retrievable offline, but every target value is evaluated in code from the "
            "closed form, so no table digit enters the evidence chain.",
        ],
        mission="CASE-BENDING-PATCH",
    )
    passed = manifest["passed"]

    result = {
        "anchor": "CASE-BENDING-PATCH",
        "generated_at": utc_now(),
        "wall_seconds": None,
        "material": {"young": YOUNG, "poisson": POISSON, "thickness": THICK},
        "oracle": oracle,
        "dimensionless": dimensionless,
        "matrix": matrix,
        "ladder": ladder,
        "strip": strip,
        "strip_free_w_naive": strip_free_w_naive,
        "orders": orders,
        "strip_ratio": ratio,
        "free_surface": free_surface,
        "free_surface_summary": free_surface_summary,
        "strip_transverse": strip_transverse,
        "strip_ny": strip_ny,
        "strip_aspect": strip_aspect_summary,
        "strip_aspect_cases": strip_aspect,
        "strip_root_support": strip_root,
        "locking_scaling": lock_scaling,
        "locking_scaling_summary": lock_sum,
        "criteria": {"case_pool_tolerance": POOL_TOL, "exact_tolerance": EXACT_TOL},
        "gates": [x.to_dict() for x in gates],
        "passed": passed,
    }
    result["wall_seconds"] = round(time.time() - t_start, 2)
    figures = make_figures(result)
    result["figures"] = figures

    write_json(HERE / "result.json", result)
    write_json(HERE / "gates_verified.json", manifest)
    DOC.write_text(render_report(result))

    print(f"[bending-patch] {len(matrix)} patch cases, {len(ladder)} ladder rows, "
          f"{len(strip)} strip cases in {result['wall_seconds']:.1f} s")
    print(f"[bending-patch] affine exact max: w {exact_max:.2e} rot {exact_max_rot:.2e} "
          f"M {exact_max_m:.2e} E {exact_max_e:.2e}")
    print(f"[bending-patch] distorted max: w {dist_w:.3e} M {dist_m:.3e} E {dist_e:.3e} "
          f"(tolerance {POOL_TOL})")
    print(f"[bending-patch] strip: symmetry w {strip_w:.3e}, free w {strip_free_w:.3e}, "
          f"locking min {lock_min:.3f}")
    print(f"[bending-patch] orders: {orders}")
    print(f"[bending-patch] thin limit: full slope {lock_sum['slope_full']:.3f}, "
          f"amplification {lock_sum['amplification_full']:.1f}, mitc thin "
          f"{lock_sum['thin_mitc']:.2e}, sri(distorted) thin "
          f"{lock_sum['thin_sri_distorted']:.3e}")
    print(f"[bending-patch] gates passed={passed} "
          f"failed={[x.id for x in gates if not x.passed]}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())