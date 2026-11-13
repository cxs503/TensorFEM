#!/usr/bin/env python3
"""Re-emit ``gates_verified.json`` for CASE-SPHERE-PRESSURE in the canonical envelope.

Why this file exists
--------------------
The anchor shipped a *legacy* certificate: no ``schema``/``anchor``/``passed``
envelope, gates keyed by name with prose criteria, and two gates whose
``value`` was a vector.  ``tools/verified_anchor_audit.py`` therefore filed it as
``E_SCHEMA`` + ``E_PASS_WITHOUT_CRITERION``: the verdicts could not be
recomputed from the artifacts, only read.  A certificate whose pass cannot be
machine-checked is verification debt, not evidence.

This emitter rebuilds the certificate **from ``result.json`` alone** (no solve,
no hand-typed number).  Every gate becomes a scalar ``value`` with an explicit
``operator`` and numeric ``threshold``, so the audit can recompute
``passed`` from the artifact.  Vector-valued claims are scalarised without
losing their content:

* ``G5`` "e(4) > e(8) > e(32)" -> two contraction ratios ``e(8)/e(4)`` and
  ``e(32)/e(8)`` each ``< 1``;
* ``G6`` "observed order >= 1.5" -> the *minimum* observed order over the
  ladder (the binding member; the full list stays in the note).

The one predeclared prediction that the run **falsified** (G8b: the thin-shell
reference band was predicted to reach ``t/R = 0.05`` and measurably ends at
``0.02``) is *not* silently dropped and is *not* turned into a gate whose
failure would be indistinguishable from an accuracy failure: it is emitted into
``predictions`` with its prediction, its measurement and
``outcome: "falsified"``.  ``tests/test_case_sphere_pressure_certificate.py``
asserts it stays there, so honesty about the falsification is itself machine
checked.

Usage
-----
    python3 benchmarks/verified/fem_sphere_pressure_membrane/emit_certificate.py --write
    python3 benchmarks/verified/fem_sphere_pressure_membrane/emit_certificate.py --check

``--check`` exits 1 when the committed file differs from the regeneration, so the
certificate cannot drift from the artifact it certifies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULT_JSON = HERE / "result.json"
GATES_JSON = HERE / "gates_verified.json"

SCHEMA = "tensorfem.verified-gates.v1"
ANCHOR = "fem_sphere_pressure_membrane"
STATUS = "verified"

REL = "benchmarks/verified/fem_sphere_pressure_membrane"


# --------------------------------------------------------------------------- #
# artifact access
# --------------------------------------------------------------------------- #
def load_result(path: Path = RESULT_JSON) -> dict:
    return json.loads(path.read_text())


def _row(ladder: list[dict], facets: int) -> dict:
    for row in ladder:
        if int(row["facets"]) == facets:
            return row
    raise KeyError(f"no ladder row at facets={facets}")


def _max_rel_error(values: list[float], reference: float) -> float:
    """max |v/reference - 1| over a raw Gauss-point stress set."""
    return max(abs(float(v) / reference - 1.0) for v in values)


# --------------------------------------------------------------------------- #
# gate construction (every number derived, none typed)
# --------------------------------------------------------------------------- #
def build_gates(result: dict) -> tuple[list[dict], float]:
    tol = float(result["tolerance"])
    exact = result["convergence"]["ladder"]["exact_meridian"]
    chord = result["convergence"]["ladder"]["chord_control"]
    orders = result["convergence"]["observed_orders_sigma"]["exact_meridian"]
    nu_rows = result["poisson_sensitivity"]["rows"]
    thick_rows = result["thickness_sensitivity"]["rows"]
    case_t_over_r = float(result["parameters"]["t_over_R"])

    r8 = _row(exact, 8)
    r4 = _row(exact, 4)
    r32 = _row(exact, 32)
    c8 = _row(chord, 8)
    sigma_ref = float(r8["sigma_ref_pa"])
    w_ref = float(r8["w_ref_m"])

    # reference validity, measured against 3-D elasticity at the case t/R
    case_thick = min(thick_rows, key=lambda row: abs(float(row["t_over_R"]) - case_t_over_r))
    thin_vs_lame_at_case = float(case_thick["thin_vs_lame_rel"])

    flat = float(result["flat_facet_control"]["N_mean_rel_error"])

    gates = [
        {
            "id": "G1.sigma_theta_vs_pR_2t",
            "kind": "accuracy",
            "gating": True,
            "quantity": "max over Gauss points of |sigma_theta/(pR/2t) - 1|, 8 facets (27 dof)",
            "reference": f"thin-shell membrane stress pR/(2t) = {sigma_ref!r} Pa",
            "value": float(r8["sigma_theta_max_rel_error"]),
            "threshold": tol,
            "operator": "<=",
            "note": "hoop stress; max over the raw Gauss-point set stored in result.json",
        },
        {
            "id": "G2.sigma_phi_vs_pR_2t",
            "kind": "accuracy",
            "gating": True,
            "quantity": "max over Gauss points of |sigma_phi/(pR/2t) - 1|, 8 facets",
            "reference": f"thin-shell membrane stress pR/(2t) = {sigma_ref!r} Pa",
            "value": float(r8["sigma_phi_max_rel_error"]),
            "threshold": tol,
            "operator": "<=",
            "note": "meridional stress, independent Gauss-point set from G1",
        },
        {
            "id": "G3.isotropy",
            "kind": "invariant",
            "gating": True,
            "quantity": "|N_phi - N_theta| / (pR/2), 8 facets",
            "reference": "the membrane state of a pressurised sphere is equi-biaxial",
            "value": float(r8["anisotropy_rel"]),
            "threshold": tol,
            "operator": "<=",
            "note": "membrane-force anisotropy; a sphere has no preferred meridional direction",
        },
        {
            "id": "G4.radial_displacement_vs_closed_form",
            "kind": "accuracy",
            "gating": True,
            "quantity": "|w_pole/w_ref - 1|, 8 facets",
            "reference": f"w = pR^2(1-nu)/(2Et) = {w_ref!r} m at the pole",
            "value": float(r8["w_rel_error"]),
            "threshold": tol,
            "operator": "<=",
            "note": "displacement is nu-dependent, so this gate also exercises nu",
        },
        {
            "id": "G5a.contraction_4_to_8",
            "kind": "convergence",
            "gating": True,
            "quantity": "e(8)/e(4), e = max Gauss sigma_theta rel. error",
            "reference": "error must contract under refinement: e(4) > e(8)",
            "value": float(r8["sigma_theta_max_rel_error"]) / float(r4["sigma_theta_max_rel_error"]),
            "threshold": 1.0,
            "operator": "<",
            "note": "scalarised member 1 of the legacy vector gate G5_monotone_refinement",
        },
        {
            "id": "G5b.contraction_8_to_32",
            "kind": "convergence",
            "gating": True,
            "quantity": "e(32)/e(8), e = max Gauss sigma_theta rel. error",
            "reference": "error must contract under refinement: e(8) > e(32)",
            "value": float(r32["sigma_theta_max_rel_error"]) / float(r8["sigma_theta_max_rel_error"]),
            "threshold": 1.0,
            "operator": "<",
            "note": "scalarised member 2 of the legacy vector gate G5_monotone_refinement",
        },
        {
            "id": "G6.min_observed_order",
            "kind": "convergence",
            "gating": True,
            "quantity": "min observed log-log order of the sigma_theta ladder (2..64 facets)",
            "reference": "> 1.5 rules out a merely consistent (order 1) element",
            "value": float(min(orders)),
            "threshold": 1.5,
            "operator": ">=",
            "note": f"full ladder {[float(o) for o in orders]!r}; the minimum is the binding member",
        },
        {
            "id": "G7.poisson_invariance",
            "kind": "invariant",
            "gating": True,
            "quantity": "max over nu in {0,0.2,0.3,0.4,0.49} of the sigma_theta rel. error",
            "reference": "closed-sphere membrane stress does not depend on nu",
            "value": max(float(row["sigma_theta_max_rel_error"]) for row in nu_rows),
            "threshold": tol,
            "operator": "<=",
            "note": "w would move with nu; the stress must not",
        },
        {
            "id": "G8a.reference_valid_at_case_point",
            "kind": "reference-validity",
            "gating": True,
            "quantity": f"|sigma_thin - sigma_Lame|/sigma_Lame at t/R = {case_t_over_r!r}",
            "reference": "the thin-shell formula must itself be within tolerance of 3-D elasticity at the case point",
            "value": thin_vs_lame_at_case,
            "threshold": tol,
            "operator": "<=",
            "note": "validates the reference used by G1-G4, not the element",
        },
        {
            "id": "G9.chord_control_fails",
            "kind": "control-negative",
            "gating": True,
            "quantity": "sigma_theta rel. error at 8 facets with the meridian represented by straight chords",
            "reference": "control must MISS the tolerance: geometry representation matters, the element does not rescue it",
            "value": float(c8["sigma_theta_max_rel_error"]),
            "threshold": tol,
            "operator": ">=",
            "note": "same element, same mesh, only the circular meridian switched off",
        },
        {
            "id": "G10.flat_facet_shell4_control_fails",
            "kind": "control-negative",
            "gating": True,
            "quantity": "N_mean rel. error of the merged flat-facet Shell4 path, 24 facets",
            "reference": "control must MISS the tolerance by orders of magnitude",
            "value": flat,
            "threshold": tol,
            "operator": ">=",
            "note": (
                "independent element on the same case; reproduces the recorded 13316.6 % "
                f"(recorded source: {result['flat_facet_control']['recorded_source']!r})"
            ),
        },
    ]
    for gate in gates:
        gate["passed"] = bool(_compare(gate["value"], gate["operator"], gate["threshold"]))
    return gates, tol


OPERATORS = {
    "<=": lambda v, t: v <= t,
    "<": lambda v, t: v < t,
    ">=": lambda v, t: v >= t,
    ">": lambda v, t: v > t,
}


def _compare(value: float, operator: str, threshold: float) -> bool:
    return bool(OPERATORS[operator](float(value), float(threshold)))


def build_predictions(result: dict) -> list[dict]:
    """Predeclared predictions, including the one this run falsified.

    Kept out of ``gates`` on purpose: a falsified *prediction about the
    reference* is a scientific finding, not an element-accuracy failure, and it
    must not be able to flip the gating verdict (nor be hidden by it).
    """
    band = result["thickness_sensitivity"]["t_over_R_band_within_tol"]
    predicted = 0.05
    measured = float(max(band))
    return [
        {
            "id": "P1.reference_band_reaches_t_over_R_0.05",
            "statement": "the thin-shell reference stays within tolerance up to t/R = 0.05",
            "predicted_threshold": predicted,
            "predicted_operator": ">=",
            "measured_band_upper_t_over_R": measured,
            "measured_band_lower_t_over_R": float(min(band)),
            "measured_deviations": [
                {"t_over_R": float(row["t_over_R"]), "thin_vs_lame_rel": float(row["thin_vs_lame_rel"])}
                for row in result["thickness_sensitivity"]["rows"]
            ],
            "deviation_scaling_slope": float(result["thickness_sensitivity"]["thin_vs_lame_loglog_slope"]),
            "measured_passed": bool(measured >= predicted),
            "outcome": "falsified" if measured < predicted else "confirmed",
            "note": (
                "falsified prediction kept in the record: the band ends near t/R = 0.02 and the "
                "thin-vs-Lame deviation scales linearly in t/R (slope "
                f"{result['thickness_sensitivity']['thin_vs_lame_loglog_slope']:.4f}), so the "
                "reference cannot reach 0.05 within a 3 % tolerance"
            ),
        }
    ]


def build_certificate(result: dict) -> dict:
    gates, tol = build_gates(result)
    predictions = build_predictions(result)
    failed = [gate["id"] for gate in gates if gate["gating"] and not gate["passed"]]
    return {
        "schema": SCHEMA,
        "anchor": ANCHOR,
        "title": "Pressurised sphere, axisymmetric Reissner-Mindlin ring element vs thin-shell membrane solution",
        "case": result.get("case", "CASE-SPHERE-PRESSURE"),
        "status": STATUS,
        "generated_by": f"{REL}/emit_certificate.py (derived from {REL}/result.json; zero solve, zero hand-typed number)",
        "mission": result.get("mission"),
        "mission_history": result.get("mission_history", []),
        "tolerance": tol,
        "gates": gates,
        "predictions": predictions,
        "passed": not failed,
        "failed_gates": failed,
        "failed_predictions": [p["id"] for p in predictions if p["outcome"] == "falsified"],
        "artifacts": {
            "result": f"{REL}/result.json",
            "driver": f"{REL}/run_verification.py",
            "report_builder": f"{REL}/build_report.py",
            "emitter": f"{REL}/emit_certificate.py",
            "readme": f"{REL}/README.md",
        },
        "undefined_terms": [],
        "limitations": [
            "Thin-shell (Reissner-Mindlin) theory only: the case grades the element against *shell* theory, "
            "and the shell formula's own distance from 3-D elasticity is measured separately by G8a "
            f"({thin_vs_lame_note(result)}).",
            "Uniform internal pressure on a closed spherical shell only: no boundary layers, no edge loads, "
            "no external restraint, and no bending-dominated state is exercised.",
            "The certificate figures are rendered on demand by build_report.py into artifacts/ and are not "
            "committed, so the certificate declares no figure artifact (declaring an uncommitted path would "
            "be an E_ARTIFACT_MISSING defect).",
            "G9 and G10 are negative controls: they pass when the control *misses* the tolerance, so a "
            "passing certificate is not evidence that every geometric representation of the meridian works.",
        ],
        "upstream_verdict": "PASS" if not failed else "FAIL",
        "upstream_module": "tensorfem.axisym_shell_benchmarks.sphere_pressure",
    }


def thin_vs_lame_note(result: dict) -> str:
    case_t = float(result["parameters"]["t_over_R"])
    row = min(
        result["thickness_sensitivity"]["rows"],
        key=lambda r: abs(float(r["t_over_R"]) - case_t),
    )
    return f"+{100.0 * float(row['thin_vs_lame_rel']):.3e} % at t/R = {case_t!r}"


def render(result: dict | None = None) -> str:
    result = load_result() if result is None else result
    return json.dumps(build_certificate(result), indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="rewrite gates_verified.json")
    parser.add_argument("--check", action="store_true", help="exit 1 if the committed file differs")
    args = parser.parse_args(argv)

    text = render()
    if args.write:
        GATES_JSON.write_text(text)
        print(f"wrote {GATES_JSON.relative_to(Path.cwd()) if GATES_JSON.is_relative_to(Path.cwd()) else GATES_JSON}")
        return 0
    if args.check:
        current = GATES_JSON.read_text() if GATES_JSON.is_file() else ""
        if current != text:
            print(f"FAIL: {GATES_JSON} is stale; re-run with --write", file=sys.stderr)
            return 1
        print("OK: gates_verified.json matches the artifact-derived regeneration")
        return 0
    print(text, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())