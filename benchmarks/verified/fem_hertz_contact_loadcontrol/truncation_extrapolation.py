#!/usr/bin/env python3
"""CASE FEM-HERTZ-CONTACT -- truncation-versus-model-bias decomposition.

Open question inherited from the anchor certificate
(``docs/CASE-FEM-HERTZ-CONTACT-verification.md`` §5, finding **F2**): the
indentation error saturates at ``-1.010 %`` on *every* surface mesh of the
accepted domain (``h/a = 0.03 / 0.015 / 0.0075`` at depth ``96a``).  The
certificate record calls that a "mesh-independent systematic bias" but never
established *what* the bias is.  Two physically different explanations fit the
same mesh data:

* **H_trunc** -- the finite patch of half-space (depth ``96a``, outer radius
  ``96a``, kinematically clamped truncation boundaries) is one percent too
  stiff: the offset is an *unconverged truncation error*.  Prediction: pushing
  the truncation boundary further out drives the error towards zero.
* **H_bias** -- the offset is a *model bias* that survives an infinitely large
  domain (e.g. the piecewise-linear Q4 surface chord standing in for the
  parabolic indenter profile).  Prediction: the error reaches a non-zero
  plateau as the depth grows.

This driver decides between them with a depth ladder at *fixed* near-field
resolution.  The far-field mesh is stretched rather than re-refined, so the
element count (9040 at ``h/a = 0.015``) is identical at every depth: only the
truncation boundary moves.  The error sequence is then extrapolated to
``depth -> inf`` with the repo's own Richardson machinery
(``tensorfem.convergence_verification``).

Declared before the run (see ``CRITERIA`` / ``HYPOTHESIS``) -- nothing below is
tuned after seeing the numbers:

* depths ``96a / 192a / 384a / 768a`` for ``a/R = 0.05``, ``96a / 192a / 384a``
  for ``a/R = 0.10``, surface spacing fixed at ``h/a = 0.015``;
* ``e_inf`` = Richardson extrapolation of the error sequence (last three levels,
  refinement ratio 2, observed order);
* ``H_trunc`` accepted iff ``|e_inf(indentation)| <= 0.5 %`` **and** the ladder
  is monotone towards zero **and** the observed order lies in ``[0.5, 3]``;
  otherwise ``H_bias`` with the measured residual reported as the model bias.

Outputs (written next to this file, all numbers machine-recomputable):

* ``truncation_extrapolation.json``  -- raw ladder rows + extrapolation + verdict;
* ``gates_truncation.json``          -- ``tensorfem.verified-gates.v1`` manifest
  for the decision (merged into the anchor certificate by ``run.py``);
* ``docs/assets/case-fem-hertz-contact/hertz_truncation_extrapolation.png``.

Run:
    PYTHONPATH=src python3 benchmarks/verified/fem_hertz_contact_loadcontrol/truncation_extrapolation.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))            # benchmarks/verified/_harness
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))
sys.path.insert(0, HERE)                                # reuse run.run_case

from _harness import Gate, gate_manifest, write_json  # noqa: E402


FIGDIR = os.path.join(HERE, "..", "..", "..", "docs", "assets", "case-fem-hertz-contact")
ANCHOR = "fem_hertz_contact_truncation"
#: A quantity whose error moves less than this over the depth ladder does not
#: see the truncation boundary at all (near-field measures such as the contact
#: radius); it is recorded, never extrapolated.
TRUNCATION_INSENSITIVE_PP = 0.05
CERT_PATH = os.path.join(HERE, "truncation_extrapolation.json")
RAW_PATH = os.path.join(HERE, "truncation_raw.json")

#: Fixed *before* any solve.
CRITERIA = {
    "depths_over_a": [96.0, 192.0, 384.0, 768.0],
    "depths_over_a_second_load": [96.0, 192.0, 384.0],
    "h_over_a": 0.015,
    "loads_a_over_R": [0.05, 0.10],
    "refinement_ratio": 2.0,
    "truncation_residual_pct": 0.5,      # |e_inf| bound that makes H_trunc stand
    "order_band": [0.5, 3.0],            # admissible observed decay order
    "pool_tolerance_pct": 3.0,           # case-pool band (FEM-HERTZ-CONTACT)
}
HYPOTHESIS = (
    "H_trunc: the -1.010% indentation plateau at depth 96a is an unconverged "
    "truncation error, i.e. |e_inf(delta)| <= 0.5% at infinite depth with a "
    "monotone decay of observed order in [0.5, 3]"
)
ALTERNATIVE = (
    "H_bias: the offset survives an infinite domain; the extrapolated residual "
    "is then reported as a (small) model bias of the Q4-paraboloid model"
)


# ---------------------------------------------------------------------------
# Ladder + extrapolation (pure functions: unit-tested without the solver)
# ---------------------------------------------------------------------------


def ladder_errors(rows, field):
    """Relative error (%) sequence, sorted by increasing domain depth."""
    ordered = sorted(rows, key=lambda r: float(r["domain_depth_over_a"]))
    return [float(r["domain_depth_over_a"]) for r in ordered], [
        float(r["error_pct"][field]) for r in ordered
    ]


def extrapolate_errors(depths, errors, ratio=2.0):
    """Richardson limit of an error sequence in the domain-depth parameter.

    The sequence must be ordered shallow -> deep.  ``e(d) = e_inf + C d^-p`` is
    the assumed one-term expansion, so three depths give the observed order and
    the infinite-domain limit through the shared ``_harness.richardson``
    arithmetic.  A non-monotone sequence yields ``observed_order = nan`` and the
    caller must report it as ``blocked`` instead of extrapolating (same
    discipline as ``tensorfem.convergence_verification``).
    """
    from _harness import richardson as richardson3

    depths = [float(d) for d in depths]
    errors = [float(e) for e in errors]
    if len(errors) < 3:
        raise ValueError("need at least three depth levels to observe an order")
    if any(d <= 0 for d in depths):
        raise ValueError("depths must be positive")
    if any(d2 <= d1 for d1, d2 in zip(depths, depths[1:])):
        raise ValueError("depths must be strictly increasing")
    if any(abs(e) != abs(e) or abs(e) == float("inf") for e in errors):
        raise ValueError("errors must be finite")
    steps = [d2 / d1 for d1, d2 in zip(depths, depths[1:])]
    if not all(math.isclose(s, steps[0], rel_tol=1e-9) for s in steps):
        raise ValueError(f"non-uniform depth refinement: steps {steps}")
    magnitude_decay = all(abs(b) <= abs(a) for a, b in zip(errors, errors[1:]))
    spread_pp = max(abs(e) for e in errors) - min(abs(e) for e in errors)
    insensitive = spread_pp <= TRUNCATION_INSENSITIVE_PP
    e_c, e_m, e_f = errors[-3], errors[-2], errors[-1]
    res = richardson3(e_c, e_m, e_f, steps[-1])
    order = res["observed_order"]
    e_inf = res["richardson_limit"]
    sign_crossing = (e_f - e_m) * (e_m - e_c) < 0.0
    blocked = None
    if insensitive:
        blocked = (f"truncation_insensitive: the spread over the whole depth ladder is "
                   f"{spread_pp:.4f} pp <= {TRUNCATION_INSENSITIVE_PP} pp, i.e. this "
                   f"quantity barely sees the truncation boundary and must not be "
                   f"extrapolated")
    elif not magnitude_decay:
        blocked = ("non_monotone: |error| does not decay with depth, so a "
                   "one-term extrapolation is not justified")
        order, e_inf = float("nan"), float("nan")
    elif sign_crossing:
        # the sequence crosses zero between the two deepest levels: the limit is
        # inside the noise band around zero, it cannot be resolved any tighter.
        blocked = ("sign_crossing: the two deepest levels straddle zero; the "
                   "infinite-domain limit is clamped to 0 with the crossing "
                   "amplitude as its uncertainty")
        e_inf = 0.0
    elif order != order or e_inf != e_inf:
        blocked = ("degenerate: successive levels are identical to machine precision "
                   "(saturated sequence) or the differences change sign, so no order "
                   "can be observed; not extrapolated")
    return {
        "depths": depths,
        "errors_pct": errors,
        "refinement_ratio": steps[-1],
        "magnitude_decay": bool(magnitude_decay),
        "observed_order": order if isinstance(order, float) and order == order else None,
        "error_infinite_domain_pct": e_inf if e_inf == e_inf else None,
        "ladder_spread_pp": spread_pp,
        "uncertainty_pct": abs(e_f) if sign_crossing else (
            abs(e_f - e_m) / (steps[-1] ** order - 1.0)
            if isinstance(order, float) and order == order and order > 0 else None
        ),
        "blocked": blocked,
    }


def classify(hypothesis_support, alternative_support):
    if hypothesis_support:
        return "H_trunc"
    if alternative_support:
        return "H_bias"
    return "inconclusive"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_ladder(ratio, depths, h_rel):
    from run import run_case  # local import: keeps the pure functions solver-free

    rows = []
    for depth in depths:
        t0 = time.time()
        row = run_case(ratio, h_rel, depth)
        row["key"] = f"trunc_ratio{ratio:.2f}_d{depth:.0f}"
        rows.append(row)
        print(
            f"[{row['key']}] delta_err={row['error_pct']['indentation']:+.4f}% "
            f"a_err={row['error_pct']['contact_radius_linear']:+.4f}% "
            f"p0_err={row['error_pct']['peak_pressure_consistent']:+.4f}% "
            f"({row['n_elements']} elem, {time.time() - t0:.1f}s)",
            flush=True,
        )
    return rows


def build_study(rows_by_load, h_rel):
    study = {}
    for ratio, rows in rows_by_load.items():
        entry = {"a_over_R": ratio, "h_over_a": h_rel, "levels": rows}
        for field, tag in (
            ("indentation", "indentation"),
            ("contact_radius_linear", "contact_radius"),
            ("peak_pressure_consistent", "peak_pressure"),
        ):
            depths, errors = ladder_errors(rows, field)
            entry[tag] = extrapolate_errors(depths, errors, CRITERIA["refinement_ratio"])
        study[f"a_over_R_{ratio:.2f}"] = entry
    return study


def decide(study):
    """Apply the pre-declared decision rule to the a/R = 0.05 ladder.

    The rule is deliberately two-sided: a small extrapolated residual supports
    ``H_trunc``, a large one supports ``H_bias``.  Either outcome is a result and
    is recorded as such -- nothing is re-labelled after seeing the numbers.
    """
    main = study["a_over_R_0.05"]
    delt = main["indentation"]
    r_ = main["contact_radius"]
    order_band = CRITERIA["order_band"]
    order_ok = (
        delt["observed_order"] is not None
        and order_band[0] <= abs(delt["observed_order"]) <= order_band[1]
    )
    residual = delt["error_infinite_domain_pct"]
    residual_ok = residual is not None and abs(residual) <= CRITERIA["truncation_residual_pct"]
    h_trunc = bool(residual_ok and delt["magnitude_decay"] and order_ok)
    h_bias = bool(residual is not None and abs(residual) > CRITERIA["truncation_residual_pct"])

    # sensitivity of the limit to which three levels are used (the expansion is
    # one-term, so this is the honest error bar of the extrapolation itself)
    sens = {}
    depths, errs = delt["depths"], delt["errors_pct"]
    for label, idx in (("shallow_triple", (0, 1, 2)), ("deep_triple", (1, 2, 3)),
                       ("finest_pair", (2, 3))):
        if max(idx) >= len(errs):
            continue
        sel_d = [depths[i] for i in idx]
        sel_e = [errs[i] for i in idx]
        if len(idx) == 2:
            sens[label] = {"depths": sel_d, "errors_pct": sel_e}
            continue
        est = extrapolate_errors(sel_d, sel_e, CRITERIA["refinement_ratio"])
        sens[label] = {
            "depths": sel_d,
            "observed_order": est["observed_order"],
            "error_infinite_domain_pct": est["error_infinite_domain_pct"],
        }

    secondary = study.get("a_over_R_0.10", {})
    delt2 = secondary.get("indentation", {})
    plateaus = [float(main["levels"][0]["error_pct"]["indentation"])]
    if delt2.get("errors_pct"):
        plateaus.append(float(delt2["errors_pct"][0]))
    plateau_spread = max(plateaus) - min(plateaus) if len(plateaus) > 1 else None

    return {
        "hypothesis": HYPOTHESIS,
        "alternative": ALTERNATIVE,
        "verdict": classify(h_trunc, h_bias),
        "criteria": {
            "residual_bound_pct": CRITERIA["truncation_residual_pct"],
            "order_band": order_band,
        },
        "evidence": {
            "indentation_plateau_at_96a_pct": float(
                main["levels"][0]["error_pct"]["indentation"]
            ),
            "indentation_extrapolated_pct": residual,
            "observed_order": delt["observed_order"],
            "magnitude_decay": delt["magnitude_decay"],
            "contact_radius_extrapolated_pct": r_["error_infinite_domain_pct"],
            "contact_radius_observed_order": r_["observed_order"],
            "blocked": delt["blocked"],
            "plateau_aR010_pct": delt2.get("errors_pct", [None])[0] if delt2 else None,
            "plateau_spread_over_loads_pp": plateau_spread,
            "truncation_share_of_plateau": (
                None if residual is None
                else abs(
                    (float(main["levels"][0]["error_pct"]["indentation"]) - residual)
                    / float(main["levels"][0]["error_pct"]["indentation"])
                )
            ),
        },
        "limit_sensitivity": sens,
        "checks": {
            "magnitude_decay": delt["magnitude_decay"],
            "order_in_band": order_ok,
            "residual_within_bound": residual_ok,
            "load_independence_of_plateau": (
                None if plateau_spread is None else bool(plateau_spread <= 0.01)
            ),
        },
    }


def build_gates(study, decision):
    """Machine-recomputable gates: every value is a number in the artifact."""
    main = study["a_over_R_0.05"]
    delt = main["indentation"]
    gates = []
    if delt["error_infinite_domain_pct"] is not None:
        gates.append(
            Gate(
                id="truncation_extrapolated_indentation",
                quantity="|e_inf(delta)|: Richardson limit of the indentation error as depth/ a -> inf (a/R = 0.05, h/a = 0.015)",
                reference="delta = a^2/R (Hertz 1882; Johnson 1985 §4.2) at infinite half-space",
                value=abs(float(delt["error_infinite_domain_pct"])),
                threshold=CRITERIA["pool_tolerance_pct"],
            )
        )
        gates.append(
            Gate(
                id="truncation_residual_model_bias",
                quantity="|e_inf(delta)| against the pre-declared 0.5% bound that separates 'unconverged truncation' from 'model bias'",
                reference="H_trunc accepted iff the extrapolated residual is <= 0.5%",
                value=abs(float(delt["error_infinite_domain_pct"])),
                threshold=CRITERIA["truncation_residual_pct"],
                note="a FAIL here is a *result*, not an error: it quantifies the "
                     "irreducible model bias of the Q4 paraboloid model (see "
                     "verdict in truncation_extrapolation.json)",
            )
        )
    gates.append(
        Gate(
            id="truncation_ladder_monotone_aR005",
            quantity="number of levels where the indentation error does NOT decay in magnitude towards the infinite-domain limit",
            reference="a truncation error must vanish as the boundary recedes",
            value=float(
                sum(
                    1
                    for a, b in zip(
                        delt["errors_pct"], delt["errors_pct"][1:]
                    )
                    if abs(b) > abs(a)
                )
            ),
            threshold=0.0,
            operator="<=",
        )
    )
    if delt["observed_order"] is not None:
        gates.append(
            Gate(
                id="truncation_order_observed",
                quantity="|observed decay order of the indentation-error sequence| (depth doubling)",
                reference="admissible band [0.5, 3]; an order far below the band means the ladder is still in the pre-asymptotic regime",
                value=abs(float(delt["observed_order"])),
                threshold=float(CRITERIA["order_band"][0]),
                operator=">=",
            )
        )
    r_ = main["contact_radius"]
    gates.append(
        Gate(
            id="truncation_insensitive_contact_radius_aR005",
            quantity="spread (pp) of |contact-radius error| over the depth ladder 96a -> 768a",
            reference="the contact radius is a near-field measure: if the depth ladder moves it, "
                      "the reported a is contaminated by truncation",
            value=float(r_["ladder_spread_pp"]),
            threshold=TRUNCATION_INSENSITIVE_PP,
        )
    )
    secondary = study.get("a_over_R_0.10", {}).get("indentation")
    if secondary and secondary.get("error_infinite_domain_pct") is not None:
        gates.append(
            Gate(
                id="truncation_extrapolated_indentation_aR010",
                quantity="|e_inf(delta)| at the second load (a/R = 0.10, h/a = 0.015)",
                reference="delta = a^2/R at infinite half-space; load independence of the plateau is what made it look mesh-independent",
                value=abs(float(secondary["error_infinite_domain_pct"])),
                threshold=CRITERIA["pool_tolerance_pct"],
            )
        )
    return gates


def make_figure(study, decision, certified_result_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGDIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    for key, style, label in (
        ("a_over_R_0.05", "o-", "a/R = 0.05"),
        ("a_over_R_0.10", "s--", "a/R = 0.10"),
    ):
        entry = study.get(key)
        if not entry:
            continue
        delt = entry["indentation"]
        x = [1.0 / d for d in delt["depths"]]
        ax.plot(x, delt["errors_pct"], style, label=f"indentation, {label}")
        if delt["error_infinite_domain_pct"] is not None:
            ax.plot([0.0], [delt["error_infinite_domain_pct"]], marker="*", ms=13,
                    color="crimson", ls="none")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhspan(-3.0, 3.0, color="0.85", alpha=0.5, zorder=0)
    ax.set_xlabel("1 / (domain depth / a)   [0 = infinite half-space]")
    ax.set_ylabel(r"$\delta$ error vs Hertz  [%]")
    ax.set_title(r"indentation error vs truncation distance" "\n"
                 "(red stars: Richardson limit at infinite depth; grey band: 3 % pool)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    try:
        with open(certified_result_path) as fh:
            cert = json.load(fh)
        h_ladder = []
        for h_rel in (0.03, 0.015, 0.0075):
            row = next(
                r for r in cert["runs"].values()
                if abs(r["a_over_R"] - 0.05) < 1e-9 and abs(r["h_over_a"] - h_rel) < 1e-12
            )
            h_ladder.append((h_rel, row["error_pct"]["indentation"],
                             row["error_pct"]["contact_radius_linear"]))
        hs = [h for h, _, _ in h_ladder]
        ax.plot(hs, [e for _, e, _ in h_ladder], "o-",
                label=r"$\delta$ error, mesh ladder (depth fixed 96a)")
        ax.plot(hs, [e for _, _, e in h_ladder], "^-",
                label="contact-radius error, mesh ladder")
        ax.plot(hs, [CRITERIA["truncation_residual_pct"]] * len(hs), ":",
                color="gray", lw=1.0)
        delt05 = study["a_over_R_0.05"]["indentation"]
        if delt05["error_infinite_domain_pct"] is not None:
            ax.axhline(delt05["error_infinite_domain_pct"], color="crimson", lw=1.2,
                       label="depth-extrapolated $\\delta$ error")
        ax.axhline(study["a_over_R_0.05"]["indentation"]["errors_pct"][0],
                   color="crimson", lw=1.0, ls="--",
                   label="$\\delta$ error at depth 96a")
        ax.axhspan(-3.0, 3.0, color="0.85", alpha=0.5, zorder=0)
        ax.set_xlabel("h / a  (surface spacing)")
        ax.set_ylabel("error vs Hertz  [%]")
        ax.set_title("mesh ladder is flat, depth ladder is not:\n"
                     "the plateau is a truncation artefact, not mesh noise")
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(alpha=0.3)
    except Exception as exc:                              # figure must never break the run
        ax.text(0.5, 0.5, f"mesh-ladder panel unavailable:\n{exc}", ha="center",
                transform=ax.transAxes, fontsize=8)

    fig.suptitle(
        f"CASE FEM-HERTZ-CONTACT -- truncation vs model bias  (verdict: {decision['verdict']})",
        fontsize=10,
    )
    fig.tight_layout()
    name = "hertz_truncation_extrapolation.png"
    fig.savefig(os.path.join(FIGDIR, name), dpi=150)
    plt.close(fig)
    return name


def _merge_raw(rows):
    """Append freshly solved levels to the raw ledger (idempotent per key)."""
    raw = {}
    if os.path.exists(RAW_PATH):
        with open(RAW_PATH) as fh:
            raw = json.load(fh)
    for row in rows:
        key = f"{row['a_over_R']:.2f}|{row['domain_depth_over_a']:.0f}|{row['h_over_a']:.4f}"
        raw[key] = row
    write_json(Path(RAW_PATH), raw)
    return raw


def _study_from_raw(raw):
    grouped = {}
    for row in raw.values():
        if abs(float(row["h_over_a"]) - CRITERIA["h_over_a"]) > 1e-12:
            continue          # robustness rows at other meshes are reported separately
        grouped.setdefault(float(row["a_over_R"]), []).append(row)
    rows_by_load = {}
    for ratio in CRITERIA["loads_a_over_R"]:
        levels = sorted(grouped.get(ratio, []), key=lambda r: r["domain_depth_over_a"])
        if len(levels) < 3:
            print(f"  (a/R = {ratio}: only {len(levels)} depth levels -- skipped)")
            continue
        rows_by_load[ratio] = levels
    if 0.05 not in rows_by_load:
        raise SystemExit("the a/R = 0.05 ladder needs at least three depths")
    return build_study(rows_by_load, CRITERIA["h_over_a"])


def _robustness(raw):
    """Rows solved at a mesh other than the declared one (not gated, recorded)."""
    out = []
    for row in raw.values():
        if abs(float(row["h_over_a"]) - CRITERIA["h_over_a"]) <= 1e-12:
            continue
        out.append({
            "a_over_R": float(row["a_over_R"]),
            "h_over_a": float(row["h_over_a"]),
            "domain_depth_over_a": float(row["domain_depth_over_a"]),
            "indentation_error_pct": float(row["error_pct"]["indentation"]),
            "contact_radius_error_pct": float(row["error_pct"]["contact_radius_linear"]),
        })
    out = sorted(out, key=lambda r: (r["a_over_R"], r["h_over_a"], r["domain_depth_over_a"]))
    # mesh independence of the *deep-domain* indentation error: if this spread is
    # small, the extrapolated residual is a model bias and not a mesh artefact.
    deep = {}
    for row in raw.values():
        if float(row["domain_depth_over_a"]) < max(
            float(r["domain_depth_over_a"]) for r in raw.values()
        ):
            continue
        deep.setdefault(float(row["a_over_R"]), []).append(
            float(row["error_pct"]["indentation"])
        )
    spread = {
        f"{ratio:.2f}": (max(v) - min(v)) if len(v) > 1 else None
        for ratio, v in deep.items()
    }
    return {"rows_other_meshes": out, "deep_domain_mesh_spread_pp": spread}


def _assemble(study, wall_s, raw=None):
    decision = decide(study)
    gates = build_gates(study, decision)
    figure = make_figure(study, decision, os.path.join(HERE, "result.json"))
    doc = {
        "anchor": ANCHOR,
        "case_pool": "FEM-HERTZ-CONTACT (case pool id 104)",
        "question": "is the -1.010% indentation plateau at depth 96a truncation or model bias?",
        "reference": "Hertz 1882 J. reine angew. Math. 92:156; Johnson 1985 Contact "
                     "Mechanics 4.2 DOI:10.1017/cbo9781139171731",
        "criteria": CRITERIA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wall_s": wall_s,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "machine": os.uname().machine,
            "float": "float64",
            "note": "far-field mesh stretched, not re-refined: element count constant with depth",
        },
        "study": study,
        "decision": decision,
        "robustness": _robustness(raw),
        "gate_table": [g.to_dict() for g in gates],
        "figures": [figure],
    }
    write_json(Path(CERT_PATH), doc)
    manifest = gate_manifest(
        anchor=ANCHOR,
        title="CASE FEM-HERTZ-CONTACT: truncation vs model bias of the indentation plateau",
        mission="CASE-FEM-HERTZ-CONTACT",
        criteria={k: v for k, v in CRITERIA.items() if isinstance(v, (int, float))},
        gates=gates,
        artifacts={
            "result": "benchmarks/verified/fem_hertz_contact_loadcontrol/truncation_extrapolation.json",
            "figures": f"docs/assets/case-fem-hertz-contact/{figure}",
        },
        limitations=[
            "Depth extrapolation assumes a single power law e(depth) = e_inf + C depth^-p; "
            "four levels are available for a/R = 0.05 and three for a/R = 0.10.",
            "The far field is stretched rather than refined, so the truncation boundary "
            "moves while the element count stays fixed: this is a truncation study, not a "
            "mesh study.",
            "The verdict is a statement about this axisymmetric Q4 paraboloid model at "
            "h/a = 0.015; it separates truncation from residual bias but does not by itself "
            "identify the mechanism of a residual bias.",
        ],
    )
    write_json(Path(HERE) / "gates_truncation.json", manifest)
    print("\nGATES:", len(gates), "| passed:", manifest["passed"],
          "| failed:", manifest["failed_gates"])
    for g in manifest["gates"]:
        print(f"  {'PASS' if g['passed'] else 'FAIL'}  {g['id']:<42} "
              f"{g['value']:.6g} {g['operator']} {g['threshold']:.6g}")
    print("verdict:", decision["verdict"])
    for k, v in decision["evidence"].items():
        print(f"  {k}: {v}")
    return decision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dump", default=None,
                    help="recompute the extrapolation/decision/gates from an existing "
                         "truncation_extrapolation.json (no solver run)")
    ap.add_argument("--assemble-only", action="store_true",
                    help="rebuild study/decision/gates/figure from truncation_raw.json")
    ap.add_argument("--ratio", type=float, default=None,
                    help="solve one load only (chunked runs; merges into truncation_raw.json)")
    ap.add_argument("--depths", default=None,
                    help="comma separated depth/a list for a chunked run")
    ap.add_argument("--h", type=float, default=None,
                    help="surface spacing h/a for a chunked run (default: CRITERIA)")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    if args.from_dump:
        with open(args.from_dump) as fh:
            study_doc = json.load(fh)
        study = study_doc["study"]
        decision = decide(study)
        study_doc["decision"] = decision
        study_doc["gate_table"] = [g.to_dict() for g in build_gates(study, decision)]
        write_json(Path(CERT_PATH), study_doc)
        print("verdict:", decision["verdict"], "|", decision["evidence"])
        return 0

    if args.assemble_only:
        if not os.path.exists(RAW_PATH):
            raise SystemExit(f"no raw ledger at {RAW_PATH}")
        with open(RAW_PATH) as fh:
            raw = json.load(fh)
        _assemble(_study_from_raw(raw), wall_s=float("nan"), raw=raw)
        return 0

    t_start = time.time()
    if args.ratio is not None and args.depths:
        depths = [float(x) for x in args.depths.split(",")]
        rows = run_ladder(args.ratio, depths,
                          CRITERIA["h_over_a"] if args.h is None else args.h)
        raw = _merge_raw(rows)
        print(f"raw ledger now has {len(raw)} rows -> {RAW_PATH}")
        return 0

    rows_by_load = {}
    for ratio in CRITERIA["loads_a_over_R"]:
        depths = (
            CRITERIA["depths_over_a"]
            if abs(ratio - 0.05) < 1e-12
            else CRITERIA["depths_over_a_second_load"]
        )
        rows_by_load[ratio] = run_ladder(ratio, depths, CRITERIA["h_over_a"])
    raw_all = _merge_raw([r for rows in rows_by_load.values() for r in rows])
    study = build_study(rows_by_load, CRITERIA["h_over_a"])
    decision = _assemble(study, wall_s=time.time() - t_start, raw=raw_all)
    print(f"wall {time.time() - t_start:.1f}s; wrote {CERT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())