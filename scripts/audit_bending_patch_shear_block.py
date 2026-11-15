#!/usr/bin/env python3
"""CASE-BENDING-PATCH (round 2, reopened): does the anchor grade the transverse
shear block of the Q4 Mindlin plate element?

Motivation (measured, not asserted)
-----------------------------------
Every state in the round-1/round-2 gate pool is *shear-free* by construction
(``gamma_xz = gamma_yz = 0`` is what makes them constant-*curvature* states), and
the load-driven strip runs at ``t/L = 0.01``, where the transverse-shear
flexibility is ~0.005 % of the tip deflection.  So the whole shear constitutive
block ``Ds = kappa_s G t`` enters no graded quantity.  This driver quantifies
that blindness with *mutation controls* -- it deliberately breaks the element's
shear block and asks whether any of the anchor's 13 re-graded legs notices:

===========================  ==================================================
mutation                     what is broken
===========================  ==================================================
``none``                     production one-point (sri) element
``kappa_scale = 1.2``        shear factor: ``kappa_s = 1`` instead of ``5/6`` (-20 %)
``kappa_scale = 0.9870``     ``kappa_s = pi^2/12`` instead of ``5/6`` (-1.3 %)
``shear_sign``               ``gamma = w,x - theta_x`` (sign relation flipped)
``no_shear``                 shear block deleted (bending-only stiffness)
===========================  ==================================================

Then it grades the *repair*: the complementary constant-shear state
(:class:`tensorfem.bending_patch.ShearState`) whose prescribed-field energy
identity ``1/2 u^T K u = 1/2 Ds gamma^2 Area`` is exact to round-off, so a 20 %
error in ``Ds`` is rejected by ~1e15 relative separation instead of being
invisible.

Outputs (all generated, never hand-edited)
------------------------------------------
* ``docs/benchmarks/tutorials/bending-patch-shear-block.json``
* ``docs/assets/bending_patch/shear_block_sensitivity.png``

Usage::

    OMP_NUM_THREADS=4 PYTHONPATH=src python scripts/audit_bending_patch_shear_block.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorfem.bending_patch import (  # noqa: E402
    CurvatureState,
    constant_shear_dofs,
    element_dofs,
    q4_gauss_curvatures,
    q4_gauss_shears,
    ShearState,
    affine_patch_mesh,
    assemble_plate_stiffness,
    distorted_patch_mesh,
    free_surface_check,
    mutated_q4_stiffness,
    q4_reference_stiffness,
    run_patch_case,
    shear_block_check,
    shear_block_mutation,
    shear_rigidity,
    shear_state_selfcheck,
    strip_pure_bending,
)
from tensorfem.plate import q4_mindlin_stiffness  # noqa: E402

YOUNG, POISSON, THICK = 210.0e3, 0.3, 1.0
KAPPA = 5.0 / 6.0
KAPPA_UNIFORM = 1.0                     # kappa_s of a uniform-shear assumption
KAPPA_ROUND = math.pi ** 2 / 12.0       # kappa_s of a parabolic shear distribution
STATES = [
    CurvatureState(1.0e-3, 0.0, 0.0),
    CurvatureState(0.0, 0.0, 1.0e-3),
    CurvatureState(1.0e-3, 0.6e-3, 0.4e-3),
]
SHEAR_STATES = [
    ShearState(2.0e-3, 0.0),
    ShearState(0.0, 2.0e-3),
    ShearState(1.5e-3, -1.0e-3),
]
FAMILIES = {
    "affine": lambda: affine_patch_mesh(2),
    "skew": lambda: affine_patch_mesh(2, skew=0.35),
    "distorted": lambda: distorted_patch_mesh(2),
}
TREATMENTS = ("sri", "mitc", "full")
EXACT_TOL = 1.0e-12
POOL_TOL = 0.02

MUTATIONS = [
    {"name": "none", "mutation": "none", "kappa_scale": 1.0},
    {"name": "kappa_s=1 (-20%)", "mutation": "none",
     "kappa_scale": KAPPA_UNIFORM / KAPPA},
    {"name": "kappa_s=pi^2/12 (-1.3%)", "mutation": "none",
     "kappa_scale": KAPPA_ROUND / KAPPA},
    {"name": "shear_sign", "mutation": "shear_sign", "kappa_scale": 1.0},
    {"name": "no_shear", "mutation": "no_shear", "kappa_scale": 1.0},
]


# --------------------------------------------------------------------------- #
# 1. the anchor legs, recomputed under a mutated element
# --------------------------------------------------------------------------- #
def anchor_legs() -> dict:
    """The 13 shear-sensitive legs of the round-1/2 anchor (``sri`` element).

    Meshes, states and thresholds are the committed ones; only the element is
    swapped by the caller's mutation context.  ``value`` is the graded quantity
    and ``passed`` applies the anchor's own threshold/operator.
    """
    legs: dict[str, tuple[float, float, str]] = {}
    aff_w = aff_m = aff_e = 0.0
    for st in STATES:
        for skew in (0.0, 0.35):
            r = run_patch_case(affine_patch_mesh(3, skew=skew), st, young=YOUNG,
                               poisson=POISSON, thickness=THICK,
                               shear_treatment="sri")
            aff_w = max(aff_w, r["displacement_rel_error"])
            aff_m = max(aff_m, r["moment_rel_error"])
            aff_e = max(aff_e, r["energy_rel_error"])
    legs["affine_exact_w"] = (aff_w, EXACT_TOL, "<=")
    legs["affine_exact_moment"] = (aff_m, EXACT_TOL, "<=")
    legs["affine_exact_energy"] = (aff_e, EXACT_TOL, "<=")

    rd = run_patch_case(distorted_patch_mesh(3), STATES[2], young=YOUNG,
                        poisson=POISSON, thickness=THICK, shear_treatment="sri")
    legs["distorted_w"] = (rd["displacement_rel_error"], POOL_TOL, "<=")
    legs["distorted_moment"] = (rd["moment_rel_error"], POOL_TOL, "<=")
    legs["distorted_energy"] = (rd["energy_rel_error"], POOL_TOL, "<=")

    fs = free_surface_check(affine_patch_mesh(3), 1.0e-3, young=YOUNG,
                            poisson=POISSON, thickness=THICK,
                            shear_treatment="sri")
    legs["free_surface_field"] = (max(fs["displacement_rel_error"],
                                      fs["rotation_rel_error"]), EXACT_TOL, "<=")
    legs["free_surface_my_over_mx"] = (fs["my_over_mx"], EXACT_TOL, "<=")
    legs["free_surface_mx_over_db_kappa"] = (
        abs(fs["mx_over_db_kappa"] - 1.0), EXACT_TOL, "<=")

    sym = strip_pure_bending(32, 2, lateral="symmetry", thickness=0.01)
    legs["strip_symmetry_w"] = (sym["w_rel_error_kirchhoff"], POOL_TOL, "<=")
    legs["strip_symmetry_theta"] = (sym["theta_rel_error"], POOL_TOL, "<=")
    free = strip_pure_bending(32, 16, lateral="free", thickness=0.01)
    legs["strip_free_w"] = (free["w_rel_error_kirchhoff"], POOL_TOL, "<=")

    ratio = free["w_tip"] / sym["w_tip"]
    target = 1.0 / (1.0 - POISSON ** 2)
    legs["poisson_coupling_computed"] = (abs(ratio - target) / target, POOL_TOL, "<=")
    return legs


def survey_anchor() -> list[dict]:
    rows = []
    for spec in MUTATIONS:
        try:
            with shear_block_mutation(spec["mutation"], spec["kappa_scale"]):
                legs = anchor_legs()
            status = "measured"
        except Exception as exc:                       # singular / non-positive K
            legs = {}
            status = f"CRASH:{type(exc).__name__}"
        flipped = []
        table = {}
        for key, (value, thr, op) in legs.items():
            passed = (value <= thr) if op == "<=" else (value >= thr)
            table[key] = {"value": value, "threshold": thr, "operator": op,
                          "passed": passed}
            if not passed:
                flipped.append(key)
        rows.append({**spec, "status": status, "legs": table,
                     "legs_flipped": flipped, "n_flipped": len(flipped)})
        print(f"[anchor] {spec['name']:22s} status={status:12s} "
              f"legs_flipped={len(flipped):2d} {flipped}")
    return rows


# --------------------------------------------------------------------------- #
# 2. the repair: constant-shear prescribed-energy identity
# --------------------------------------------------------------------------- #
def shear_exactness() -> list[dict]:
    rows = []
    for fam, build in FAMILIES.items():
        for st in SHEAR_STATES:
            for tr in TREATMENTS:
                r = shear_block_check(build(), st, young=YOUNG, poisson=POISSON,
                                      thickness=THICK, shear_treatment=tr)
                rows.append(r)
    worst = max(r["energy_rel_error"] for r in rows)
    print(f"[shear] exactness: {len(rows)} cases, worst energy rel. error {worst:.2e}")
    return rows


def discrimination(probe_family: str = "skew") -> list[dict]:
    """Energy error of the *same* check when the shear block is broken."""
    rows = []
    build = FAMILIES[probe_family]
    st = ShearState(1.5e-3, -1.0e-3)
    nominal = shear_block_check(build(), st, young=YOUNG, poisson=POISSON,
                                thickness=THICK)
    for spec in MUTATIONS:
        with shear_block_mutation(spec["mutation"], spec["kappa_scale"]):
            r = shear_block_check(build(), st, young=YOUNG, poisson=POISSON,
                                  thickness=THICK)
        rows.append({**spec, "family": probe_family, "state": st.label(),
                     "energy_rel_error": r["energy_rel_error"],
                     "energy": r["energy"],
                     "separation": abs(r["energy"] - nominal["energy"])
                     / abs(nominal["energy"])})
        print(f"[control] {spec['name']:22s} energy_rel_error="
              f"{r['energy_rel_error']:.3e}  separation={rows[-1]['separation']:.3f}")
    return rows


def convention_blind_spot() -> dict:
    """A w-only state cannot see the shear sign; the balanced split can."""
    patch = affine_patch_mesh(2)
    st = ShearState(2.0e-3, -1.0e-3)

    def energy(u: torch.Tensor) -> float:
        k = assemble_plate_stiffness(patch, young=YOUNG, poisson=POISSON,
                                     thickness=THICK)
        return 0.5 * float(u @ (k @ u))

    n = patch.nodes.shape[0]
    w_only = torch.zeros(3 * n, dtype=torch.float64)
    w_only[0::3] = st.gamma_xz * patch.nodes[:, 0] + st.gamma_yz * patch.nodes[:, 1]
    balanced = st.field(patch.nodes).reshape(-1)

    def under_flip(u: torch.Tensor) -> float:
        with shear_block_mutation("shear_sign"):
            return energy(u)

    plus_w, flip_w = energy(w_only), under_flip(w_only)
    plus_b, flip_b = energy(balanced), under_flip(balanced)
    ds = shear_rigidity(YOUNG, POISSON, THICK)
    exact = 0.5 * ds * (st.gamma_xz ** 2 + st.gamma_yz ** 2) * patch.quad_area()
    out = {
        "w_only_energy": plus_w,
        "w_only_energy_flipped": flip_w,
        "w_only_convention_blindness": abs(flip_w - plus_w) / plus_w,
        "balanced_energy": plus_b,
        "balanced_energy_flipped": flip_b,
        "balanced_convention_separation": abs(flip_b - plus_b) / plus_b,
        "energy_exact": exact,
    }
    print(f"[split] w-only blindness {out['w_only_convention_blindness']:.1e} "
          f"vs balanced separation {out['balanced_convention_separation']:.3f}")
    return out


# --------------------------------------------------------------------------- #
# 3. sensitivity ladder: how wrong must Ds be before the anchor notices?
# --------------------------------------------------------------------------- #
def sensitivity_ladder() -> list[dict]:
    row_specs = [
        ("kappa_s=5/6 (nominal)", 1.0),
        ("kappa_s=pi^2/12", KAPPA_ROUND / KAPPA),
        ("kappa_s=1", 1.0 / KAPPA),
        ("Ds x1.2", 1.2),
        ("Ds x0.8", 0.8),
        ("Ds x2", 2.0),
        ("Ds x5", 5.0),
        ("Ds x20", 20.0),
        ("Ds x100", 100.0),
    ]
    rows = []
    for name, scale in row_specs:
        st = ShearState(1.5e-3, -1.0e-3)
        with shear_block_mutation("none", scale):
            try:
                legs = anchor_legs()
                status = "measured"
            except Exception as exc:
                legs, status = {}, f"CRASH:{type(exc).__name__}"
            check = shear_block_check(FAMILIES["skew"](), st, young=YOUNG,
                                      poisson=POISSON, thickness=THICK)
        flipped = [k for k, (v, thr, op) in legs.items()
                   if not ((v <= thr) if op == "<=" else (v >= thr))]
        rows.append({"name": name, "kappa_scale": scale, "status": status,
                     "n_flipped": len(flipped), "legs_flipped": flipped,
                     "new_check_energy_rel_error": check["energy_rel_error"]})
        print(f"[ladder] {name:22s} kappa_scale={scale:8.4f} anchor_flipped="
              f"{len(flipped):2d}  new_check={check['energy_rel_error']:.3e}")
    return rows


# --------------------------------------------------------------------------- #
# 4. element-copy identity (no drift between instrument and production)
# --------------------------------------------------------------------------- #
def instrument_identity() -> dict:
    """The mutation instrument must *be* the production element, and must bite.

    Three things are measured, so the instrument cannot quietly disagree with
    the element it audits:

    * ``mutation="none"`` reproduces the production ``sri`` stiffness to
      round-off;
    * the mutation context leaves ``mitc`` / ``full`` untouched;
    * inside the context the ``sri`` stiffness *does* change (otherwise the
      survey would be vacuous).
    """
    xy = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                      dtype=torch.float64)

    def rel(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a - b).abs().max() / a.abs().max())

    prod_sri = q4_mindlin_stiffness(xy, young=YOUNG, poisson=POISSON,
                                    thickness=THICK, shear_treatment="sri")
    prod_mitc = q4_mindlin_stiffness(xy, young=YOUNG, poisson=POISSON,
                                     thickness=THICK, shear_treatment="mitc")
    prod_full = q4_mindlin_stiffness(xy, young=YOUNG, poisson=POISSON,
                                     thickness=THICK, shear_treatment="full")
    instr = mutated_q4_stiffness(xy, young=YOUNG, poisson=POISSON, thickness=THICK)

    def via_driver(treatment: str) -> torch.Tensor:
        # the path the patch/strip drivers actually use
        return q4_reference_stiffness(xy, young=YOUNG, poisson=POISSON,
                                      thickness=THICK, shear_treatment=treatment)

    base = {tr: via_driver(tr) for tr in ("sri", "mitc", "full")}
    with shear_block_mutation("shear_sign"):
        inside = {tr: via_driver(tr) for tr in ("sri", "mitc", "full")}
    sri_in, mitc_in, full_in = inside["sri"], inside["mitc"], inside["full"]
    out = {
        "reproduction_sri_rel": rel(prod_sri, instr),
        "mitc_untouched_rel": rel(base["mitc"], mitc_in),
        "full_untouched_rel": rel(base["full"], full_in),
        "sri_changed_rel": rel(base["sri"], sri_in),
    }
    print(f"[identity] instrument vs production sri {out['reproduction_sri_rel']:.2e}; "
          f"context leaves mitc/full untouched "
          f"({out['mitc_untouched_rel']:.1e}/{out['full_untouched_rel']:.1e}), "
          f"sri changes {out['sri_changed_rel']:.3f}")
    return out


def baseline_vs_committed(legs: dict) -> dict:
    """Compare the recomputed baseline legs with the committed anchor gates.

    Only legs with the *same definition* are compared (``comparable``); the
    others are listed because they differ by mesh coverage (e.g. the committed
    ``strip_free_w`` gate is the worst over ``n_x in {8, 32}``).
    """
    path = REPO / "benchmarks" / "verified" / "fem_bending_patch" / "gates_verified.json"
    if not path.exists():
        return {}
    committed = {g["id"]: g for g in json.loads(path.read_text()).get("gates", [])}
    mapping = {
        "affine_exact_w": "affine_exact_w",
        "affine_exact_moment": "affine_exact_moment",
        "distorted_w": "distorted_w",
        "distorted_moment": "distorted_moment",
        "distorted_energy": "distorted_energy",
        "free_surface_field": "free_surface_field_exact",
        "free_surface_my_over_mx": None,
        "free_surface_mx_over_db_kappa": "free_surface_rigidity",
        "strip_symmetry_w": "strip_symmetry_w",
        "strip_symmetry_theta": "strip_symmetry_theta",
        "strip_free_w": "strip_free_w",
        "poisson_coupling_computed": "poisson_coupling_computed",
    }
    notes = {
        "affine_exact_w": ("identical definition and mesh", True),
        "affine_exact_moment": ("committed gate is the max over sri and mitc; "
                                "this survey runs sri only", False),
        "distorted_w": ("committed gate is the max over the three states; this "
                        "survey runs the combined state only", False),
        "distorted_moment": ("committed gate is the max over the three states",
                             False),
        "distorted_energy": ("committed gate is the max over the three states",
                             False),
        "free_surface_field": ("committed gate also covers skew and nu=0.45",
                               False),
        "free_surface_my_over_mx": ("not separately gated by the anchor", False),
        "free_surface_mx_over_db_kappa": ("committed gate is max|M_x/(D_b kappa_x)-1|; "
                                          "same quantity", True),
        "strip_symmetry_w": ("committed gate is the max over n_x in {8, 32}",
                             False),
        "strip_symmetry_theta": ("committed gate is the max over n_x in {8, 32}",
                                 False),
        "strip_free_w": ("committed gate is the max over n_x in {8, 32} and "
                         "n_y >= 8; this survey fixes n_x = 32, n_y = 16", False),
        "poisson_coupling_computed": ("identical definition (n_x = 32, n_y = 16)",
                                      True),
    }
    out = {}
    for mine, theirs in mapping.items():
        note, comparable = notes[mine]
        row = {"recomputed": legs[mine]["value"], "note": note,
               "comparable": comparable}
        if theirs and theirs in committed:
            row["committed"] = committed[theirs]["value"]
        out[mine] = row
    return out


# --------------------------------------------------------------------------- #
# artefacts
# --------------------------------------------------------------------------- #
def figure(ladder: list[dict], controls: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    scales = [r["kappa_scale"] for r in ladder]
    flipped = [r["n_flipped"] for r in ladder]
    ax1.plot(scales, flipped, "o-", color="#b23b3b", label="anchor legs flipped")
    ax1.set_xscale("log")
    ax1.set_xlabel("transverse-shear rigidity multiplier  $D_s/D_s^{nom}$")
    ax1.set_ylabel("number of anchor legs that fail")
    ax1.set_title("(a) existing patch/strip anchor vs a wrong $D_s$")
    ax1.axvspan(0.8, 1.2, color="#d9d9d9", alpha=0.6,
                label="realistic $\\kappa_s$ error band ($\\pm$20 %)")
    ax1.set_ylim(-0.2, 13.2)
    ax1.axhline(0, color="k", lw=0.6)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.3)

    names = [r["name"] for r in controls]
    vals = [max(r["energy_rel_error"], 1e-18) for r in controls]
    ax2.barh(range(len(names)), vals, color="#2e6f9e")
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xscale("log")
    ax2.axvline(EXACT_TOL, color="k", ls="--", lw=0.9, label="graded tolerance $10^{-12}$")
    ax2.set_xlabel("constant-shear energy identity: relative error")
    ax2.set_title("(b) the new check separates a broken $D_s$")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def field_map(st: ShearState, path: Path) -> dict:
    """Field map of the prescribed state on the *distorted* patch.

    Left: the transverse shear recovered from the FE field at the Gauss points --
    constant, i.e. the state is exactly a constant-shear state.  Right: the
    bending strain ``kappa`` of the same field -- identically zero.  This is the
    "field cloud" evidence that the graded state is what the check says it is.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    patch = distorted_patch_mesh(3)
    u = patch.nodes.new_zeros(3 * patch.nodes.shape[0])
    u[:] = constant_shear_dofs(patch, st)
    dofs = element_dofs(patch.elements)
    gx, gy, kx = [], [], []
    centres = []
    for ei in range(patch.elements.shape[0]):
        xy = patch.nodes[patch.elements[ei]]
        g = q4_gauss_shears(xy, u[dofs[ei]])
        c = q4_gauss_curvatures(xy, u[dofs[ei]])
        centres.append(xy.mean(dim=0))
        for k in range(g.shape[0]):
            gx.append(float(g[k, 0]))
            gy.append(float(g[k, 1]))
            kx.append(float(c[k].abs().max()))
    exact = st.gamma_xz
    spread = (max(gx) - min(gx)) / abs(exact)
    leak = max(kx)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
    for ax, col, title, vmin, vmax in (
            (axes[0], gx, "recovered $\\gamma_{xz}$", exact * (1 - 1e-12), exact * (1 + 1e-12)),
            (axes[1], kx, "bending strain $\\max|\\kappa_i|$", 0.0, max(leak, 1e-18))):
        sc = ax.scatter([float(c[0]) for c in centres for _ in range(4)],
                        [float(c[1]) for c in centres for _ in range(4)],
                        c=col, marker="s", s=45, vmin=vmin, vmax=vmax,
                        cmap="viridis", edgecolors="none")
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(sc, ax=ax, fraction=0.046)
    fig.suptitle(f"prescribed constant-shear state ({st.label()}) on the distorted patch",
                 fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    out = {"shear_spread_rel": spread, "curvature_leak": leak,
           "figure": str(path.relative_to(REPO))}
    print(f"[map] shear spread {spread:.2e}, curvature leak {leak:.2e}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "docs" / "benchmarks" / "tutorials"
                                         / "bending-patch-shear-block.json"))
    ap.add_argument("--figure", default=str(REPO / "docs" / "assets"
                                            / "bending_patch"
                                            / "shear_block_sensitivity.png"))
    args = ap.parse_args()
    torch.set_num_threads(4)
    t0 = time.time()

    oracle = {st.label(): shear_state_selfcheck(st) for st in SHEAR_STATES}
    exact = shear_exactness()
    controls = discrimination()
    split = convention_blind_spot()
    anchor = survey_anchor()
    ladder = sensitivity_ladder()
    identity = instrument_identity()
    field = field_map(SHEAR_STATES[2],
                      REPO / "docs" / "assets" / "bending_patch"
                      / "shear_block_field_map.png")
    baseline = baseline_vs_committed(anchor[0]["legs"]) if anchor else {}

    exact_worst = max(r["energy_rel_error"] for r in exact)
    kappa_gate = next(c for c in controls if c["name"].startswith("kappa_s=1 "))
    sign_gate = next(c for c in controls if c["name"] == "shear_sign")
    nominal = next(a for a in anchor if a["name"] == "none")
    kappa_anchor = next(a for a in anchor if a["name"].startswith("kappa_s=1 "))

    gates = [
        {"id": "shear_state_oracle", "quantity": "max shear/curvature deviation of "
         "the prescribed constant-shear field",
         "reference": "gamma constant, kappa = 0 by construction",
         "value": max(max(v["shear_deviation"], v["curvature_leak"])
                      for v in oracle.values()),
         "threshold": EXACT_TOL, "operator": "<="},
        {"id": "shear_energy_exact", "quantity": "max relative error of "
         "1/2 u^T K u vs 1/2 Ds gamma^2 Area (3 families x 3 treatments x 3 states)",
         "reference": "prescribed-field energy identity", "value": exact_worst,
         "threshold": EXACT_TOL, "operator": "<="},
        {"id": "shear_curvature_leak", "quantity": "max |kappa| of the "
         "constant-shear state over all cases",
         "reference": "the state carries zero bending strain",
         "value": max(r["curvature_leak"] for r in exact),
         "threshold": 1.0e-9, "operator": "<="},
        {"id": "shear_kappa_discrimination", "quantity": "energy error when "
         "kappa_s = 1 instead of 5/6 (-20%)",
         "reference": "the check must reject a wrong shear factor",
         "value": kappa_gate["energy_rel_error"], "threshold": 0.15,
         "operator": ">="},
        {"id": "shear_sign_discrimination", "quantity": "energy separation when "
         "the shear sign relation is flipped",
         "reference": "the check must reject gamma = w,x - theta_x",
         "value": sign_gate["energy_rel_error"], "threshold": 0.9,
         "operator": ">="},
        {"id": "balanced_split_required", "quantity": "convention separation of "
         "a w-only state (same measurement)",
         "reference": "the split must matter: a w-only state cannot see the sign",
         "value": split["w_only_convention_blindness"], "threshold": 1.0e-9,
         "operator": "<="},
        {"id": "anchor_blind_to_kappa", "quantity": "anchor legs that fail under "
         "kappa_s = 1 (-20%)",
         "reference": "FINDING: the existing anchor cannot detect it (0 = blind)",
         "value": kappa_anchor["n_flipped"], "threshold": 0.0, "operator": "<="},
        {"id": "anchor_baseline_intact", "quantity": "anchor legs that fail with "
         "the production element",
         "reference": "the survey itself must not perturb the anchor",
         "value": nominal["n_flipped"], "threshold": 0.0, "operator": "<="},
        {"id": "instrument_reproduces_element", "quantity": "max |K_instrument - "
         "K_production| / max|K|",
         "reference": "the mutation instrument must be the production element",
         "value": identity["reproduction_sri_rel"], "threshold": 1.0e-12,
         "operator": "<="},
    ]
    for g in gates:
        v = g["value"]
        g["passed"] = (v <= g["threshold"]) if g["operator"] == "<=" \
            else (v >= g["threshold"])
    failed = [g["id"] for g in gates if not g["passed"]]
    wall = time.time() - t0

    payload = {
        "anchor": "CASE-BENDING-PATCH",
        "check": "constant-shear prescribed-energy identity",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "wall_seconds": round(wall, 2),
        "material": {"young": YOUNG, "poisson": POISSON, "thickness": THICK,
                     "shear_correction": KAPPA,
                     "kappa_s_uniform": KAPPA_UNIFORM,
                     "kappa_s_parabolic": KAPPA_ROUND},
        "oracle": oracle,
        "exactness": exact,
        "controls": controls,
        "convention_split": split,
        "anchor_blindness_survey": anchor,
        "sensitivity_ladder": ladder,
        "instrument_identity": identity,
        "field_map": field,
        "baseline_vs_committed": baseline,
        "gates": gates,
        "passed": not failed,
        "failed_gates": failed,
        "artifacts": {
            "json": "docs/benchmarks/tutorials/bending-patch-shear-block.json",
            "figures": ["docs/assets/bending_patch/shear_block_sensitivity.png",
                        "docs/assets/bending_patch/shear_block_field_map.png"],
            "driver": "scripts/audit_bending_patch_shear_block.py",
            "report": "docs/CASE-BENDING-PATCH-round2-shear-block.md",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    figure(ladder, controls, Path(args.figure))
    print(f"[done] {out} in {wall:.1f}s; gates passed={payload['passed']} "
          f"failed={failed}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())