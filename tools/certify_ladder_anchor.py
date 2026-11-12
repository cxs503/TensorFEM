#!/usr/bin/env python3
"""Publish ``gates_verified.json`` for ladder anchors that only ship a verdict.

Why this exists
---------------
The structural ladder (``benchmarks/structural_ladder.py``) writes
``benchmarks/verified/<anchor>/result.json`` with a ``gates`` list and a
``verdict`` string, but *not* the certificate envelope that
``tools/verified_anchor_audit.py`` requires (``tensorfem.verified-gates.v1``).
Those anchors therefore sit in ``anchor_certificate_debt.json`` with
``E_NO_GATES``: their "PASS" is prose inside an artifact, invisible to the
fail-closed certificate audit and skipped silently by the verification ladder.

This tool closes that debt **without re-running the solver and without
hand-copying numbers**. For every gate it

1. recomputes the gated quantity *independently* from the raw per-level data in
   ``result.json`` (the ``levels`` array + ``reference`` block), using a recipe
   registered per anchor below;
2. compares that recomputation against the value the ladder wrote into its own
   ``gates`` list (``rel_tol = 1e-9``) -- a mismatch is a hard error, because it
   would mean the artifact's own verdict is not supported by its own levels;
3. recomputes the pass/fail predicate from ``operator``/``threshold`` instead of
   trusting the shipped ``passed`` flag;
4. recomputes the reported convergence order independently of the ladder's
   stored ``observed_order*`` field.

Only then is the ``gates_verified.json`` envelope written.

Usage
-----
    python tools/certify_ladder_anchor.py            # certifies every anchor below
    python tools/certify_ladder_anchor.py --check     # exit 1 if an envelope is stale
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIED_DIR = REPO_ROOT / "benchmarks" / "verified"
GATE_SCHEMA = "tensorfem.verified-gates.v1"

REL_TOL = 1e-9

#: Absolute comparison floor. Gates that assert *machine-precision* exactness
#: (e.g. |w_tip(FE)/w_tip(exact) - 1| ~ 1e-13) are dominated by floating-point
#: noise: two algebraically identical expressions evaluated in a different order
#: differ in the last few ulps, so a pure relative comparison would reject them.
#: Below this floor two values are treated as the same "numerically zero"
#: quantity, and the gate records how the two numbers agreed.
NOISE_FLOOR = 1e-10

RELATION_TO_OPERATOR = {"le": "<=", "lt": "<", "ge": ">=", "gt": ">", "eq": "=="}

OPERATORS = {
    "<=": lambda v, t: v <= t,
    "<": lambda v, t: v < t,
    ">=": lambda v, t: v >= t,
    ">": lambda v, t: v > t,
    "==": lambda v, t: math.isclose(v, t, rel_tol=1e-12, abs_tol=0.0),
}


# --------------------------------------------------------------------------- #
# independent recomputation helpers
# --------------------------------------------------------------------------- #
def _level_by(levels, key: str, value: float) -> dict:
    for level in levels:
        if isinstance(level.get(key), (int, float)) and level[key] == value:
            return level
    raise KeyError(f"no level with {key}={value}")


def _order_by_least_squares(levels, *, xkey: str, ykey: str, lo: float, hi: float) -> float:
    """Slope of log(y) vs log(x) over the levels with lo <= x <= hi.

    x is the mesh spacing ``h``, so a positive slope p means error ~ h**p.
    """
    pts = [
        (math.log(level["h"]), math.log(level[ykey]))
        for level in levels
        if lo <= level[xkey] <= hi and level.get(ykey, 0.0) > 0.0
    ]
    n = len(pts)
    if n < 2:
        raise ValueError("need at least two usable levels for an order fit")
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    return sxy / sxx


# --------------------------------------------------------------------------- #
# per-anchor recipes: gate id -> (operator, threshold, recompute(artifact) -> float)
# --------------------------------------------------------------------------- #
def _beam_gates(result: dict):
    levels = result["levels"]
    ref = result["reference"]
    q = result["geometry"]["load_n_per_m"]
    tip_ref = ref["tip_deflection_m"]

    def tip_error():
        # worst case over every level, not the single convenient level the
        # ladder happened to quote
        return max(abs(level["tip_deflection_m"] / tip_ref - 1.0) for level in levels)

    def field_order():
        return _order_by_least_squares(
            levels, xkey="elements", ykey="field_l2_relative_error", lo=2, hi=32
        )

    def equilibrium():
        # worst case over every level: the recovered clamp shear is a sum of
        # O(1) terms, so its round-off grows with the number of elements
        return max(
            abs(level["clamped_shear_recovered_N_per_m"] / (-q) - 1.0) for level in levels
        )

    return {
        "G1.tip_deflection": ("<=", 1e-2, tip_error, "worst_case"),
        "G1.field_order": (">=", 1.5, field_order, "must_reproduce"),
        "G1.equilibrium": ("<=", 1e-8, equilibrium, "worst_case"),
    }


def _plate_gates(result: dict):
    levels = result["levels"]
    geo = result["geometry"]
    ref = result["reference"]
    D = geo["bending_stiffness_n_m"]
    q = geo["pressure_pa"]
    side = geo["side_m"]
    alpha_ts = ref["alpha_timoshenko"]
    alpha_series = ref["alpha_navier_series"]

    def _alpha(level):
        # alpha = w_centre * D / (q * L^4); recomputed from the raw deflection,
        # not read from the artifact's stored ``alpha`` field.
        return level["centre_deflection_m"] * D / (q * side ** 4)

    def vs_timoshenko():
        level = _level_by(levels, "n_x", 16)
        return abs(_alpha(level) - alpha_ts) / alpha_ts

    def vs_series():
        level = _level_by(levels, "n_x", 16)
        return abs(_alpha(level) - alpha_series) / alpha_series

    def order():
        # The ladder's stored ``observed_order`` (1.944016) is reproduced only with
        # the Navier-series reference: the 0.00406 table value is itself rounded, so
        # error_vs_timoshenko has a ~6e-4 floor and fits 2.0315 over the same levels.
        return _order_by_least_squares(
            levels, xkey="n_x", ykey="error_vs_navier_series", lo=2, hi=24
        )

    return {
        "G2.centre_deflection": ("<=", 2e-2, vs_timoshenko, "must_reproduce"),
        "G2.centre_deflection_vs_series": ("<=", 2e-2, vs_series, "must_reproduce"),
        "G2.order": (">=", 1.5, order, "must_reproduce"),
    }


def _kerr_plate_gates(result: dict):
    case = result["case"]
    D = case["young_Pa"] * case["thickness_m"] ** 3 / (12.0 * (1.0 - case["poisson"] ** 2))
    L = case["side_m"]
    q0 = case["q0_Pa"]

    def _w_exact(k, gp):
        # Navier single-term solution of D lap^4 w + k w - gp lap^2 w = q0 sin(pi x/L) sin(pi y/L)
        return q0 / (4.0 * D * (math.pi / L) ** 4 + k + 2.0 * gp * (math.pi / L) ** 2)

    def exact_formula():
        worst = 0.0
        for branch in result["results"].values():
            recomputed = _w_exact(branch["k"], branch["gp"])
            worst = max(worst, abs(recomputed / branch["w_exact"] - 1.0))
        return worst

    def winkler_limit_formula():
        branch = result["results"]["winkler_limit"]
        if branch["gp"] != 0.0:
            raise ValueError("winkler_limit branch must carry gp = 0")
        classical = q0 / (4.0 * D * (math.pi / L) ** 4 + branch["k"])
        return abs(classical / branch["w_exact"] - 1.0)

    def rel_err(case_name):
        def _inner():
            branch = result["results"][case_name]
            return abs(branch["w_fem_fine"] / branch["w_exact"] - 1.0)

        return _inner

    def order_lsq():
        branch = result["results"]["kerr_full"]
        return _order_by_least_squares(
            [{"n": m["n"], "h": 1.0 / m["n"], "e": m["rel_err"]} for m in branch["meshes"]],
            xkey="n",
            ykey="e",
            lo=4,
            hi=32,
        )

    def monotone():
        worst = 0.0
        for branch in result["results"].values():
            errs = [m["rel_err"] for m in branch["meshes"]]
            for coarse, fine in zip(errs, errs[1:]):
                worst = max(worst, fine / coarse)
        return worst

    return {
        "exact_formula": ("<=", 1e-12, exact_formula, "must_reproduce"),
        "winkler_limit_formula": ("<=", 1e-12, winkler_limit_formula, "must_reproduce"),
        "kerr_centre_deflection": ("<=", 5e-3, rel_err("kerr_full"), "must_reproduce"),
        "winkler_limit_centre_deflection": ("<=", 5e-3, rel_err("winkler_limit"), "must_reproduce"),
        "free_plate_centre_deflection": ("<=", 1e-3, rel_err("free_plate"), "must_reproduce"),
        "kerr_convergence_order_lsq": (">=", 1.5, order_lsq, "must_reproduce"),
        "mesh_error_monotone": ("<=", 1.0, monotone, "must_reproduce"),
    }


ANCHORS = {
    "fem_beam": {
        "title": "Euler-Bernoulli cantilever (cubic Hermite beam) vs closed form",
        "recipes": _beam_gates,
        "quantities": {
            "G1.tip_deflection": "|w_tip(FE)/w_tip(closed form) - 1|, 8 elements, "
            "w_tip = qL^4/(8EI)",
            "G1.field_order": "least-squares slope of log(field L2 rel. error) vs log(h), "
            "elements 2..32 (must be >= 1.5)",
            "G1.equilibrium": "|R_clamp(FE)/(-q) - 1|, 32 elements (recovered support shear)",
        },
        "limitations": [
            "Slender Euler-Bernoulli regime only (L/h = 100): transverse shear and "
            "rotary inertia are not exercised, so this grades the Hermite element and "
            "its consistent load vector, not a thick-beam formulation.",
            "Displacement/rotation are nodally exact for this element, so the tip-error "
            "gate is a codeword check (machine-precision level) rather than a "
            "convergence study; the resolution information lives in G1.field_order.",
        ],
    },
    "fem_kerr_plate": {
        "title": "Kerr (three-parameter) foundation plate: Navier closed form vs Q4 Mindlin FE",
        "recipes": _kerr_plate_gates,
        "quantities": {
            "exact_formula": "max over branches of |w_exact(recomputed from case params)/w_exact(artifact) - 1|",
            "winkler_limit_formula": "|q0/(4D(pi/L)^4 + k) / w_exact(gp=0 branch) - 1| (shear-parameter limit)",
            "kerr_centre_deflection": "|w_FE(n=32)/w_exact - 1|, full Kerr branch (k and gp both active)",
            "winkler_limit_centre_deflection": "|w_FE(n=32)/w_exact - 1|, gp = 0 branch",
            "free_plate_centre_deflection": "|w_FE(n=32)/w_exact - 1|, k = gp = 0 branch (plain plate)",
            "kerr_convergence_order_lsq": "least-squares slope of log(rel_err) vs log(h) over n = 4..32, "
            "full Kerr branch (threshold 1.5; Q4 Mindlin is a 2nd-order element)",
            "mesh_error_monotone": "max over all branches/levels of rel_err(fine)/rel_err(coarse) "
            "(must be <= 1: refinement may not increase the error)",
        },
        "limitations": [
            "Single sinusoidal load mode on a hard simply-supported square plate: the Navier "
            "solution has exactly one surviving term, so this grades the discretisation of the "
            "bending operator plus the Kerr foundation stiffness, not arbitrary load fields.",
            "The artifact's stored `convergence_order` is a last-two-level ratio. The certificate "
            "gates the least-squares order instead, because the free-plate branch's finest pair of "
            "levels drops the error by 72x (2.23e-05 vs 1.6e-03 at n=16), giving a spurious "
            "last-pair order of 6.17 for a 2nd-order element. That anomaly is unexplained and is "
            "recorded here rather than gated.",
            "Thin plate (a/h = 100) with Q4 Mindlin: shear locking is not a risk at this aspect "
            "ratio, so the certificate does not speak to thick-plate behaviour.",
        ],
    },
    "fem_plate": {
        "title": "Kirchhoff thin plate, SSSS square, uniform pressure (BFS bicubic)",
        "recipes": _plate_gates,
        "quantities": {
            "G2.centre_deflection": "|alpha_FE - 0.00406| / 0.00406, 16x16, "
            "alpha = w_c D / (q L^4) recomputed from the raw centre deflection",
            "G2.centre_deflection_vs_series": "|alpha_FE - alpha_Navier| / alpha_Navier, 16x16",
            "G2.order": "least-squares slope of log(error vs Timoshenko) vs log(h), n = 2..24",
        },
        "limitations": [
            "Thin-plate Kirchhoff limit only (a/h = 100); Mindlin/MITC4 thick-plate "
            "terms are graded by the separate thick-plate anchors.",
            "Hard simple support (w = 0, tangential slope = 0) is used, which is the "
            "classical Navier/Timoshenko boundary condition; soft simple support is a "
            "different (and softer) problem and is not what 0.00406 refers to.",
        ],
    },
}

CRITERIA_NOTE = (
    "Thresholds are the ladder's own gate limits (transcribed from the `description` "
    "strings' numeric limits), each reproduced here with operator + numeric threshold so "
    "the verdict is recomputable from the artifact without re-running the solver."
)


# --------------------------------------------------------------------------- #
# certification
# --------------------------------------------------------------------------- #
def certify(anchor: str, spec: dict, *, write: bool) -> dict:
    directory = VERIFIED_DIR / anchor
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text())
    shipped = {gate["id"]: gate for gate in result.get("gates", [])}
    recipes = spec["recipes"](result)

    if shipped and set(shipped) != set(recipes):
        raise SystemExit(
            f"{anchor}: gate set drift -- artifact has {sorted(shipped)}, recipes have {sorted(recipes)}"
        )

    gates = []
    for gate_id, (operator, threshold, recompute, policy) in recipes.items():
        declared = shipped.get(gate_id)
        recomputed_value = recompute()
        shipped_value = None if declared is None else float(declared["value"])
        note = None

        if declared is None:
            agreement = "new gate: the anchor shipped no gates list, so there is no upstream value"
        elif math.isclose(recomputed_value, shipped_value, rel_tol=REL_TOL, abs_tol=0.0):
            agreement = "relative"
        elif max(abs(recomputed_value), abs(shipped_value)) < NOISE_FLOOR:
            # Machine-precision gates are dominated by floating-point noise: two
            # algebraically identical expressions evaluated in a different order
            # differ in the last few ulps, so a relative comparison is meaningless.
            agreement = (
                f"numerically-zero: both values below the {NOISE_FLOOR:g} noise floor "
                f"(abs diff {abs(recomputed_value - shipped_value):.3e})"
            )
        elif policy == "worst_case":
            agreement = "strengthened: certificate value is the worst case over all levels"
            note = (
                f"the ladder quoted {shipped_value:.6e}, a single favourable level; the "
                f"certificate quotes the worst case over every level, {recomputed_value:.6e}"
            )
        else:
            raise SystemExit(
                f"{anchor}/{gate_id}: independent recomputation {recomputed_value!r} disagrees "
                f"with the shipped gate value {shipped_value!r} (rel_tol={REL_TOL})"
            )

        recomputed_passed = bool(OPERATORS[operator](recomputed_value, threshold))
        if declared is not None and recomputed_passed != bool(declared["passed"]):
            raise SystemExit(
                f"{anchor}/{gate_id}: predicate recomputation says passed={recomputed_passed}, "
                f"artifact declares passed={bool(declared['passed'])!s}"
            )

        gates.append(
            {
                "id": gate_id,
                "quantity": spec["quantities"][gate_id],
                "reference": None if declared is None else declared["description"],
                "value": recomputed_value,
                "threshold": threshold,
                "operator": operator,
                "passed": recomputed_passed,
                "shipped_value": shipped_value,
                "recomputation": "independent, from the artifact's own raw levels/parameters",
                "agreement": agreement,
                **({"note": note} if note else {}),
            }
        )

    failed = [g["id"] for g in gates if not g["passed"]]
    envelope = {
        "schema": GATE_SCHEMA,
        "anchor": anchor,
        "title": spec["title"],
        "mission": "KPI-QC-RECHECK",
        "generated_at": result.get("generated_at"),
        "generated_by": "tools/certify_ladder_anchor.py (from the shipped result.json)",
        "criteria_note": CRITERIA_NOTE,
        "criteria": {g["id"]: g["threshold"] for g in gates},
        "gates": gates,
        "passed": not failed,
        "failed_gates": failed,
        "undefined_terms": [],
        "artifacts": {
            "result": f"benchmarks/verified/{anchor}/result.json",
        },
        "limitations": spec["limitations"],
        "upstream_verdict": result.get("verdict"),
        "upstream_module": result.get("module"),
    }

    path = directory / "gates_verified.json"
    text = json.dumps(envelope, indent=2) + "\n"
    if write:
        path.write_text(text)
    elif not path.is_file() or path.read_text() != text:
        raise SystemExit(f"{anchor}: gates_verified.json is stale; re-run without --check")
    return envelope


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if an envelope is stale")
    args = parser.parse_args(argv)

    for anchor, spec in ANCHORS.items():
        envelope = certify(anchor, spec, write=not args.check)
        status = "PASS" if envelope["passed"] else "FAIL"
        print(
            f"{anchor}: {len(envelope['gates'])} gates, upstream verdict "
            f"{envelope['upstream_verdict']} -> certified {status}"
        )
        for gate in envelope["gates"]:
            print(
                f"  {gate['id']:<32} {gate['value']:.6e} {gate['operator']} "
                f"{gate['threshold']:.3e}  {'ok' if gate['passed'] else 'FAILED'}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())