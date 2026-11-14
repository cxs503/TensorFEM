"""CASE-FEM-HERTZ-CONTACT: attribution of the residual indentation offset.

Open item handed down by the two preceding case records
------------------------------------------------------
``docs/CASE-FEM-HERTZ-CONTACT-truncation-extrapolation.md`` and
``docs/CASE-FEM-HERTZ-CONTACT-plateau-mechanism.md`` both end on the same
sentence: after the vertical near-field spacing is refined *and* the clamped
finite-depth result is Richardson-extrapolated to the half space, a residual
indentation error of **-0.2092 %** survives and its **mechanism was not
located**.  The listed candidates were "Q4 bilinearity in the steep
near-surface layer, the nodal multiplier patch, or the ALM tolerance".

What was actually measured so far (recorded numbers, a/R = 0.05, h/a = 0.015)
------------------------------------------------------------------------------
=========================  ===============  ================  ============
arm                        far field BC     dz/a              e_delta [%]
=========================  ===============  ================  ============
depth ladder, dz = a/6     clamped          a/6               -0.4804
depth ladder, dz = a/24    clamped          a/24              -0.2877
depth 768a, dz = a/48      clamped          a/48              -0.2988
depth 768a extrapolated    clamped          a/24              -0.2092
depth 96a                  **Boussinesq**   a/6               -0.3626
depth 24a                  **Boussinesq**   a/6               -0.3093
=========================  ===============  ================  ============

Two facts fall out of that table and neither was followed up:

* the *absorbing* (Boussinesq) far field removes ~0.65 pp of the clamped
  truncation error at dz = a/6 and makes the result **depth independent**
  (-0.3626 at 96a vs -0.3093 at 24a, 0.05 pp drift, vs 1.9 pp for clamped);
* but even with the exact far field a **depth-independent offset near
  -0.31 %** is left, and every one of those Boussinesq arms sits at the
  *coarsest* vertical spacing dz = a/6 -- i.e. the vertical-refinement axis
  (worth 0.19 pp on the clamped mesh) was never crossed with the far-field
  axis.  The "-0.2092 % irreducible model bias" is therefore an
  **un-crossed 2x2 cell**, not a measured bias.

Question of this study
----------------------
Does the residual survive when the far field is exact *and* the near field is
vertically refined?  Four mutually exclusive hypotheses, thresholds fixed
before any solve:

``H_ff``   the residual is the clamped truncation boundary and nothing else:
           absorbing BC + dz = a/24 gives ``|e_delta| <= 0.15 %``.
``H_bias`` the residual is a bias of the discrete contact model (Q4 surface
           representation / nodal multiplier patch / ALM): ``|e_delta|`` stays
           ``>= 0.25 %`` with absorbing BC and dz = a/24, and moves by less
           than ``0.02 pp`` when the ALM penalty ``rho`` is scaled 4x.
``H_rad``  the residual is the *radial* surface spacing h/a, which no arm ever
           refined at a saturated dz: ``|e_delta(h/a=0.015) - e_delta(h/a=
           0.0075)| >= 0.10 pp`` with absorbing BC and dz = a/24.
``H_depth`` the absorbing-BC model is only depth independent at coarse dz:
           ``|e_delta(D=96a) - e_delta(D=192a)| > 0.05 pp`` at dz = a/24.

Every arm is a full load-controlled ALM solve of the *same* case-pool case
(P prescribed, a and delta are outputs); only one variable moves per arm pair.
Runs append one JSON line each, so an interrupted matrix is reconstructed with
``--from-jsonl`` at zero compute cost.

Usage
-----
    python benchmarks/verified/fem_hertz_contact_loadcontrol/residual_attribution.py
    python benchmarks/verified/fem_hertz_contact_loadcontrol/residual_attribution.py --arms bouss_dz_a24
    python benchmarks/verified/fem_hertz_contact_loadcontrol/residual_attribution.py --from-jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))

from tensorfem.hertz_halfspace_loadcontrol import (  # noqa: E402
    graded_depth,
    hertz_load_reference,
    solve_axisymmetric_contact_load_control,
    two_zone_axis,
)

HERE = os.path.dirname(os.path.abspath(__file__))
JSONL = os.path.join(HERE, "residual_attribution_rows.jsonl")
OUT = os.path.join(HERE, "residual_attribution.json")
GATES = os.path.join(HERE, "gates_residual.json")
FIGDIR = os.path.join(HERE, "..", "..", "..", "docs", "assets", "case-fem-hertz-contact")

R, E, NU = 10.0, 1e5, 0.3
A_OVER_R = 0.05
DEPTH_A = 96.0
H_OVER_A = 0.015
FINE_BAND_A = 4.0
N_OUTER_BAND = 26
N_GROW = 16

#: Declared before any solve of this study.
CRITERIA = {
    "absorbing_dz24_indentation_pct": 0.15,
    "irreducible_bias_pct": 0.25,
    "radial_mechanism_pp": 0.10,
    "algo_invariance_pp": 0.02,
    "depth_invariance_pp": 0.05,
    "dz_monotone_violations_max": 0,
    "case_pool_pct": 3.0,
    "load_residual_max": 1e-9,
    "penetration_over_delta_max": 1e-5,
    "far_field_selfconsistency": 1e-4,
    "reference": "Hertz 1882 J. reine angew. Math. 92:156; "
                 "Johnson 1985 Contact Mechanics 4.2 DOI:10.1017/cbo9781139171731",
    "case_pool": "FEM-HERTZ-CONTACT (case pool id 104)",
}

#: n_zfine = FINE_BAND_A / dz, i.e. dz/a = 4 / n_zfine.
ARMS = (
    {"label": "clamp_dz_a6", "far_field": "zero", "n_zfine": 24},
    {"label": "clamp_dz_a12", "far_field": "zero", "n_zfine": 48},
    {"label": "clamp_dz_a24", "far_field": "zero", "n_zfine": 96},
    {"label": "bouss_dz_a6", "far_field": "boussinesq", "n_zfine": 24},
    {"label": "bouss_dz_a12", "far_field": "boussinesq", "n_zfine": 48},
    {"label": "bouss_dz_a24", "far_field": "boussinesq", "n_zfine": 96},
    {"label": "bouss_dz_a24_h0075", "far_field": "boussinesq", "n_zfine": 96, "h_rel": 0.0075},
    {"label": "bouss_dz_a24_rho4", "far_field": "boussinesq", "n_zfine": 96, "rho_factor": 4.0},
    {"label": "bouss_dz_a24_D192", "far_field": "boussinesq", "n_zfine": 96, "depth": 192.0},
)
ARM_BY_LABEL = {arm["label"]: arm for arm in ARMS}


def load_for_radius_ratio(ratio):
    """Hertz load producing ``a/R = ratio`` for R = E = 1 (homogeneous scaling)."""
    e_star = E / (1.0 - NU**2)
    a = ratio * R
    return 4.0 * e_star * a**3 / (3.0 * R)


def _append_row(row):
    with open(JSONL, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_rows():
    """Latest row per label from the append-only record."""
    if not os.path.exists(JSONL):
        return {}
    rows = {}
    with open(JSONL, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[row["label"]] = row
    return rows


def run_arm(arm):
    """One load-controlled solve; returns the recorded row."""
    label = arm["label"]
    h_rel = arm.get("h_rel", H_OVER_A)
    depth_a = arm.get("depth", DEPTH_A)
    ref = hertz_load_reference(load_for_radius_ratio(A_OVER_R), R, E, NU)
    a = ref.contact_radius
    n_inner = int(round(3.0 * a / (h_rel * a)))
    r_axis = two_zone_axis(3.0 * a, depth_a * a, n_inner, N_OUTER_BAND)
    z_axis = graded_depth(depth_a * a, FINE_BAND_A * a, arm["n_zfine"], N_GROW)
    t0 = time.time()
    ref, state, diag = solve_axisymmetric_contact_load_control(
        load=ref.load,
        radius=R,
        young=E,
        poisson=NU,
        r_axis=r_axis,
        z_axis=z_axis,
        rho_factor=arm.get("rho_factor"),
        load_tol=1e-10,
        far_field=arm["far_field"],
    )
    wall = time.time() - t0
    ff = diag["far_field"]
    row = {
        "label": label,
        "a_over_R": A_OVER_R,
        "h_over_a": h_rel,
        "dz_over_a": FINE_BAND_A / float(arm["n_zfine"]),
        "domain_depth_over_a": depth_a,
        "far_field": arm["far_field"],
        "rho_factor": arm.get("rho_factor", 1.0),
        "n_zfine": arm["n_zfine"],
        "n_elements": diag["n_elements"],
        "ndof": diag["ndof"],
        "wall_s": wall,
        "load_residual": diag["load_residual"],
        "penetration_over_delta": diag["penetration_over_indentation"],
        "equilibrium_residual_rel": state.metadata["equilibrium_residual_rel"],
        "far_field_passes": ff["passes"],
        "far_field_selfconsistency": (
            ff["history"][-1]["relative_update"] if ff["history"] else 0.0
        ),
        "far_field_converged": bool(ff["converged"]),
        "analytic": {
            "contact_radius": ref.contact_radius,
            "indentation": ref.indentation,
            "peak_pressure": ref.peak_pressure,
        },
        "numeric": {
            "contact_radius_linear": state.contact_radius_linear,
            "contact_radius_fit": state.contact_radius_fit,
            "indentation": state.prescribed_indentation,
            "peak_pressure_consistent": state.metadata["peak_pressure_consistent"],
        },
    }
    row["error_pct"] = {
        "contact_radius_linear": 100.0 * (row["numeric"]["contact_radius_linear"] / ref.contact_radius - 1.0),
        "contact_radius_fit": 100.0 * (row["numeric"]["contact_radius_fit"] / ref.contact_radius - 1.0),
        "indentation": 100.0 * (row["numeric"]["indentation"] / ref.indentation - 1.0),
        "peak_pressure_consistent": 100.0
        * (row["numeric"]["peak_pressure_consistent"] / ref.peak_pressure - 1.0),
    }
    _append_row(row)
    return row


def _e(row, key="indentation"):
    return row["error_pct"][key]


def build_gates(rows):
    """Grade the pre-registered criteria; every value is a number in ``rows``."""
    crit = CRITERIA
    gates = []

    def add(gid, quantity, reference, value, threshold, operator="<=", note="", kind="gate"):
        gates.append(
            {
                "id": gid,
                "kind": kind,
                "quantity": quantity,
                "reference": reference,
                "value": float(value),
                "threshold": float(threshold),
                "operator": operator,
                "passed": bool(value <= threshold) if operator == "<=" else bool(value >= threshold),
                "note": note,
            }
        )

    def require(*labels):
        missing = [label for label in labels if label not in rows]
        if missing:
            raise KeyError(f"missing arms: {missing}")

    require("clamp_dz_a6", "clamp_dz_a12", "clamp_dz_a24", "bouss_dz_a6", "bouss_dz_a12", "bouss_dz_a24")
    clamp = {24: rows["clamp_dz_a6"], 48: rows["clamp_dz_a12"], 96: rows["clamp_dz_a24"]}
    bouss = {24: rows["bouss_dz_a6"], 48: rows["bouss_dz_a12"], 96: rows["bouss_dz_a24"]}

    # --- H_ff: absorbing BC + saturated vertical near field lands inside 0.15 %
    add(
        "absorbing_dz_a24_indentation",
        "|e_delta| at Boussinesq far field, dz = a/24, h/a = 0.015, depth 96a",
        "delta = a^2/R (Hertz 1882; Johnson 1985 4.2); H_ff accepts at <= 0.15 %",
        abs(_e(bouss[96])),
        crit["absorbing_dz24_indentation_pct"],
        note="a pass means the '-0.2092 % irreducible model bias' was the un-crossed "
             "far-field x dz cell",
    )
    # --- H_bias: the residual is at or above the irreducible-bias band
    add(
        "absorbing_dz_a24_above_bias_band",
        "|e_delta| of the same arm against the 0.25 % irreducible-bias band",
        "H_bias requires |e_delta| >= 0.25 % at absorbing BC and dz = a/24",
        abs(_e(bouss[96])),
        crit["irreducible_bias_pct"],
        operator=">=",
        note="this gate and the H_ff gate cannot both pass; together they are the "
             "pre-registered fork between 'discretisation' and 'model bias'",
    )
    # --- the far-field axis actually removes error at every dz
    for n in (24, 48, 96):
        add(
            f"far_field_gain_dz_a{n // 4}",
            f"|e_delta| reduction when the clamped far field is replaced by the absorbing "
            f"one at dz = a/{n // 4}",
            "a truncation boundary error must disappear when the exact half-space data is imposed",
            abs(_e(clamp[n])) - abs(_e(bouss[n])),
            0.0,
            operator=">=",
        )
    # --- H_depth: the absorbing model is depth independent at the saturated dz
    if "bouss_dz_a24_D192" in rows:
        add(
            "absorbing_depth_invariance_dz_a24",
            "|e_delta(D=96a) - e_delta(D=192a)| at absorbing BC, dz = a/24 (pp)",
            "if the far field is exact the answer must not depend on the domain depth",
            abs(_e(bouss[96]) - _e(rows["bouss_dz_a24_D192"])),
            crit["depth_invariance_pp"],
        )
    # --- H_rad: the radial surface spacing is a mechanism at saturated dz
    if "bouss_dz_a24_h0075" in rows:
        add(
            "radial_surface_spacing_mechanism",
            "|e_delta(h/a=0.015) - e_delta(h/a=0.0075)| at absorbing BC, dz = a/24 (pp)",
            "H_rad accepts at >= 0.10 pp; a value below means the radial Q4 surface "
            "spacing is not the mechanism",
            abs(_e(bouss[96]) - _e(rows["bouss_dz_a24_h0075"])),
            crit["radial_mechanism_pp"],
            operator=">=",
            note="the previous h/a ladder was run at dz = a/6 only, where the far field "
                 "dominated the error",
        )
    # --- ALM penalty is an iteration parameter, not a mechanism
    if "bouss_dz_a24_rho4" in rows:
        add(
            "alm_penalty_invariance",
            "|e_delta(rho/E*a=1) - e_delta(rho/E*a=4)| at absorbing BC, dz = a/24 (pp)",
            "rho affects the ALM iteration only; the converged state must be invariant",
            abs(_e(bouss[96]) - _e(rows["bouss_dz_a24_rho4"])),
            crit["algo_invariance_pp"],
        )
    # --- vertical refinement must be monotone in the absorbing family
    violations = sum(
        1 for coarse, fine in ((24, 48), (48, 96)) if abs(_e(bouss[fine])) > abs(_e(bouss[coarse]))
    )
    add(
        "vertical_refinement_monotone_absorbing",
        "number of dz steps where |e_delta| grows under vertical refinement (absorbing BC)",
        "a discretisation error must decay with dz -> 0",
        violations,
        crit["dz_monotone_violations_max"],
    )
    # --- case-pool acceptance on every arm
    worst_delta = max(abs(row["error_pct"]["indentation"]) for row in rows.values())
    worst_radius = max(abs(row["error_pct"]["contact_radius_linear"]) for row in rows.values())
    add(
        "case_pool_indentation_all_arms",
        "worst |e_delta| over every arm in this study (%)",
        "case pool FEM-HERTZ-CONTACT tolerance 3 %",
        worst_delta,
        crit["case_pool_pct"],
    )
    add(
        "case_pool_contact_radius_all_arms",
        "worst |e_a| over every arm in this study (%)",
        "case pool FEM-HERTZ-CONTACT tolerance 3 %",
        worst_radius,
        crit["case_pool_pct"],
    )
    # --- solve-quality controls (QC, not physics)
    add(
        "load_residual_all_arms",
        "worst load-control residual |P_fe - P|/P",
        "regula falsi tolerance 1e-10",
        max(row["load_residual"] for row in rows.values()),
        crit["load_residual_max"],
    )
    add(
        "penetration_all_arms",
        "worst maximum penetration / delta",
        "ALM should leave a vanishing complementarity gap",
        max(row["penetration_over_delta"] for row in rows.values()),
        crit["penetration_over_delta_max"],
    )
    add(
        "far_field_selfconsistency_all_arms",
        "worst absorbing-BC fixed-point update (relative)",
        "far_field_tol = 1e-4",
        max(row["far_field_selfconsistency"] for row in rows.values()),
        crit["far_field_selfconsistency"],
    )
    return gates


def verdict_of(rows, gates):
    """Translate the pre-registered fork into words, honestly."""
    by_id = {gate["id"]: gate for gate in gates}
    bouss24 = rows["bouss_dz_a24"]
    out = {
        "absorbing_dz_a24_indentation_pct": _e(bouss24),
        "absorbing_dz_a24_contact_radius_pct": _e(bouss24, "contact_radius_linear"),
        "clamped_dz_a24_indentation_pct": _e(rows["clamp_dz_a24"]),
        "far_field_gain_dz_a24_pp": abs(_e(rows["clamp_dz_a24"])) - abs(_e(bouss24)),
        "h_ff_accepted": by_id["absorbing_dz_a24_indentation"]["passed"],
        "h_bias_accepted": by_id["absorbing_dz_a24_above_bias_band"]["passed"],
        "h_rad_accepted": (
            by_id["radial_surface_spacing_mechanism"]["passed"]
            if "radial_surface_spacing_mechanism" in by_id
            else None
        ),
        "h_depth_rejected": (
            by_id["absorbing_depth_invariance_dz_a24"]["passed"]
            if "absorbing_depth_invariance_dz_a24" in by_id
            else None
        ),
        "alm_penalty_excluded": (
            by_id["alm_penalty_invariance"]["passed"] if "alm_penalty_invariance" in by_id else None
        ),
    }
    if out["h_ff_accepted"]:
        out["mechanism"] = (
            "the residual was the clamped truncation boundary, never crossed with the "
            "vertical-refinement axis; with the exact half-space far field and dz = a/24 "
            "the indentation lands inside the 0.15 % band"
        )
    elif out["h_bias_accepted"]:
        out["mechanism"] = (
            "irreducible bias of the discrete contact model at the tested discretisation "
            "family: it survives the exact far field and dz = a/24"
        )
    else:
        out["mechanism"] = (
            "inconclusive: absorbing BC + dz = a/24 lands between the 0.15 % acceptance "
            "band and the 0.25 % irreducible-bias band; the residual is neither explained "
            "nor shown to be a bias"
        )
    if out["h_rad_accepted"]:
        out["radial_note"] = (
            "the radial surface spacing h/a still moves e_delta at saturated dz, so part of "
            "the remaining offset is the Q4 surface representation"
        )
    else:
        out["radial_note"] = (
            "refining the radial surface spacing h/a 0.015 -> 0.0075 does not move e_delta, "
            "so the residual is NOT the Q4 radial surface representation"
        )
    out["failed_gates"] = [gate["id"] for gate in gates if not gate["passed"]]
    out["gate_count"] = len(gates)
    return out


def make_figure(rows, verdict):
    """Two-panel figure: dz ladder per far-field mode + the arm-by-arm budget."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[figure] skipped: {exc}")
        return None

    os.makedirs(FIGDIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

    ax = axes[0]
    for mode, style, colour, name in (
        ("zero", "o--", "tab:red", "clamped far field (historical model)"),
        ("boussinesq", "s-", "tab:blue", "absorbing (Boussinesq) far field"),
    ):
        pts = sorted(
            (row["dz_over_a"], row["error_pct"]["indentation"])
            for row in rows.values()
            if row["far_field"] == mode
            and math.isclose(row["h_over_a"], H_OVER_A)
            and math.isclose(row["domain_depth_over_a"], DEPTH_A)
            and math.isclose(row.get("rho_factor", 1.0), 1.0)
        )
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], style, color=colour, label=name)
    ax.axhspan(-CRITERIA["absorbing_dz24_indentation_pct"], CRITERIA["absorbing_dz24_indentation_pct"],
               color="tab:green", alpha=0.15, label="0.15 % attribution band")
    ax.axhline(-0.2092, color="0.35", lw=1.0, ls=":", label="prior extrapolated residual -0.2092 %")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("vertical near-field spacing  dz/a  (coarse -> fine)")
    ax.set_ylabel("indentation error  $e_\\delta$  [%]")
    ax.set_title("(a) far field x vertical refinement at h/a = 0.015, depth 96a")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels, values, colours = [], [], []
    for label in (
        "clamp_dz_a24",
        "bouss_dz_a6",
        "bouss_dz_a12",
        "bouss_dz_a24",
        "bouss_dz_a24_h0075",
        "bouss_dz_a24_rho4",
        "bouss_dz_a24_D192",
    ):
        if label in rows:
            labels.append(label.replace("bouss", "abs").replace("clamp", "clamp"))
            values.append(rows[label]["error_pct"]["indentation"])
            colours.append("tab:blue" if label.startswith("bouss") else "tab:red")
    ax.barh(range(len(values)), values, color=colours, alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0.0, color="k", lw=0.8)
    for bound, colour in ((CRITERIA["absorbing_dz24_indentation_pct"], "tab:green"),
                          (CRITERIA["irreducible_bias_pct"], "tab:orange")):
        ax.axvline(-bound, color=colour, ls="--", lw=1.0)
    ax.invert_yaxis()
    ax.set_xlabel("indentation error  $e_\\delta$  [%]")
    ax.set_title("(b) single-variable arms (same case-pool case, P prescribed)")
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle(
        "CASE-FEM-HERTZ-CONTACT: attributing the residual indentation offset "
        f"(absorbing dz=a/24: {verdict['absorbing_dz_a24_indentation_pct']:+.3f} %)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(FIGDIR, "hertz_residual_attribution.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def build(rows):
    gates = build_gates(rows)
    verdict = verdict_of(rows, gates)
    arms = [rows[label] for label in ARM_BY_LABEL if label in rows]
    result = {
        "schema": "tensorfem.hertz-residual-attribution.v1",
        "benchmark": "fem_hertz_contact_loadcontrol",
        "anchor": "CASE-FEM-HERTZ-CONTACT",
        "case_pool": CRITERIA["case_pool"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
        },
        "question": "does the -0.2092 % residual indentation offset survive the exact "
                    "(absorbing) far field combined with vertical near-field refinement?",
        "hypotheses": {
            "H_ff": "clamped truncation boundary; absorbing BC + dz=a/24 -> |e_delta| <= 0.15 %",
            "H_bias": "irreducible model bias; |e_delta| >= 0.25 % and invariant under rho",
            "H_rad": "Q4 radial surface spacing; |de_delta| >= 0.10 pp for h/a 0.015 -> 0.0075",
            "H_depth": "absorbing model not depth independent at saturated dz; > 0.05 pp",
        },
        "criteria": CRITERIA,
        "config": {
            "a_over_R": A_OVER_R,
            "h_over_a": H_OVER_A,
            "domain_depth_over_a": DEPTH_A,
            "fine_band_depth_over_a": FINE_BAND_A,
            "n_outer_band": N_OUTER_BAND,
            "n_grow": N_GROW,
            "units": "lengths/R, forces/(E R^2); R = 10, E = 1e5, nu = 0.3 (SI: R = 10 mm, E = 100 GPa)",
        },
        "arms": arms,
        "gates": gates,
        "verdict": verdict,
        "artifacts": {
            "rows": os.path.basename(JSONL),
            "gates": os.path.basename(GATES),
            "figure": "docs/assets/case-fem-hertz-contact/hertz_residual_attribution.png",
        },
    }
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1, sort_keys=True)
    with open(GATES, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema": "tensorfem.verified-gates.v1",
                "anchor": "fem_hertz_contact_residual_attribution",
                "title": "CASE FEM-HERTZ-CONTACT: what is left of the indentation offset",
                "mission": "CASE-FEM-HERTZ-CONTACT",
                "criteria": CRITERIA,
                "gates": gates,
                "verdict": verdict,
            },
            handle,
            indent=1,
            sort_keys=True,
        )
    figure = make_figure(rows, verdict)
    return result, figure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arms",
        nargs="*",
        default=None,
        help="subset of arm labels to solve (default: every arm not yet in the record)",
    )
    parser.add_argument(
        "--from-jsonl",
        action="store_true",
        help="rebuild the artifacts from the append-only record with zero compute",
    )
    parser.add_argument("--force", action="store_true", help="re-solve arms already recorded")
    args = parser.parse_args(argv)

    rows = read_rows()
    if not args.from_jsonl:
        labels = list(args.arms) if args.arms else [arm["label"] for arm in ARMS]
        for label in labels:
            if label not in ARM_BY_LABEL:
                parser.error(f"unknown arm {label!r}")
        for label in labels:
            if label in rows and not args.force:
                print(f"[skip] {label} already recorded (use --force to re-solve)")
                continue
            t0 = time.time()
            row = run_arm(ARM_BY_LABEL[label])
            rows = read_rows()
            print(
                f"[done] {label:22s} ndof={row['ndof']:6d} dz/a={row['dz_over_a']:.4f} "
                f"e_a={row['error_pct']['contact_radius_linear']:+.4f}% "
                f"e_delta={row['error_pct']['indentation']:+.4f}% wall={time.time() - t0:.0f}s"
            )
    if not rows:
        print("no rows recorded; nothing to build")
        return 2
    completed = [label for label in ARM_BY_LABEL if label in rows]
    print(f"recorded arms: {completed}")
    required = {"clamp_dz_a6", "clamp_dz_a12", "clamp_dz_a24", "bouss_dz_a6", "bouss_dz_a12", "bouss_dz_a24"}
    if not required <= set(rows):
        print(f"cannot grade yet, missing {sorted(required - set(rows))}")
        return 3
    result, figure = build(rows)
    verdict = result["verdict"]
    print(json.dumps({k: v for k, v in verdict.items() if k != "mechanism"}, indent=1))
    print("mechanism:", verdict["mechanism"])
    if figure:
        print("figure:", figure)
    return 0 if not verdict["failed_gates"] else 1


if __name__ == "__main__":
    raise SystemExit(main())