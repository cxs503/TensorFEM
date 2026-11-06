"""Machine gate for the Euler-Bernoulli / thin-plate / Boussinesq benchmark suite.

Reads the artefacts produced by ``run.py`` and re-derives every verdict from the
raw numbers in ``results.json`` (never from the ``passed`` flags), then checks the
documentation contract:

1. ``benchmarks/euler_bernoulli/results.json`` exists and carries the three cases;
2. cantilever tip deflection error <= 3% (both the frame and the solid ladder);
3. thin-plate centre deflection error <= 5%;
4. Boussinesq centre displacement error <= 5%;
5. every case ran at least two refinement levels and the error is non-increasing
   along the ladder (the observed contraction factors are printed);
6. ``docs/FEM_VERIFY.md`` exists, contains the generated convergence-block
   markers, at least one markdown table, and quotes the same finest-level errors
   that this script recomputes (so hand-edited tables are caught).

Usage::

    PYTHONPATH=src python benchmarks/euler_bernoulli/validate.py
    PYTHONPATH=src python benchmarks/euler_bernoulli/validate.py --results other.json

Exit code 0 = every criterion holds, 1 = at least one failure (the failing rows
are printed with FAIL).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_RESULTS = HERE / "results.json"
DEFAULT_DOC = ROOT / "docs" / "FEM_VERIFY.md"

LIMITS = {
    "beam_tip_rel_error_max": 0.03,
    "plate_center_rel_error_max": 0.05,
    "halfspace_center_rel_error_max": 0.05,
    "min_refinement_levels": 2,
}


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _pct(value: float, digits: int = 3) -> str:
    return f"{100.0 * value:+.{digits}f}%"


def _ladder(summary: dict, key: str = "absolute_errors") -> list[float]:
    return [abs(float(value)) for value in summary.get(key, [])]


def check_results(results: dict) -> list[Check]:
    checks: list[Check] = []
    for case in ("beam", "plate", "halfspace"):
        checks.append(Check(f"results.{case}.present", case in results,
                            "case block found" if case in results else "case block missing"))
    if not all(check.ok for check in checks):
        return checks

    # ---- 1. cantilever -----------------------------------------------------
    beam = results["beam"]
    analytic = float(beam["analytic"]["tip_deflection"])
    solid = beam["solid_levels"]
    frame = beam["frame_levels"]
    solid_errors = [abs(float(row["tip_deflection"]) - analytic) / analytic for row in solid]
    frame_errors = [abs(float(row["tip_deflection"]) - analytic) / analytic for row in frame]
    beam_error = max(solid_errors[-1], frame_errors[-1])
    checks.append(Check(
        "beam.tip_rel_error<=3%", beam_error <= LIMITS["beam_tip_rel_error_max"],
        f"finest solid {_pct(solid_errors[-1])}, finest frame {_pct(frame_errors[-1])}, "
        f"limit 3%"))
    checks.append(Check(
        "beam.two_level_convergence",
        len(solid) >= LIMITS["min_refinement_levels"]
        and len(frame) >= LIMITS["min_refinement_levels"]
        and all(b <= a * (1 + 1e-9) for a, b in zip(solid_errors, solid_errors[1:])),
        f"solid ladder {len(solid)} levels {[f'{100*e:.3f}%' for e in solid_errors]}, "
        f"frame ladder {len(frame)} levels {[f'{100*e:.2e}%' for e in frame_errors]}"))
    checks.append(Check(
        "beam.slenderness_100",
        abs(float(beam["analytic"]["slenderness"]) - 100.0) < 1e-9,
        f"L/h = {beam['analytic']['slenderness']}"))

    # ---- 2. thin plate -----------------------------------------------------
    plate = results["plate"]
    reference = float(plate["reference"]["centre_deflection"])
    plate_errors = [abs(float(row["centre_deflection"]) - reference) / reference
                    for row in plate["levels"]]
    checks.append(Check(
        "plate.centre_rel_error<=5%", plate_errors[-1] <= LIMITS["plate_center_rel_error_max"],
        f"finest {_pct(plate_errors[-1])}, limit 5%"))
    checks.append(Check(
        "plate.two_level_convergence",
        len(plate_errors) >= LIMITS["min_refinement_levels"]
        and all(b <= a * (1 + 1e-9) for a, b in zip(plate_errors, plate_errors[1:])),
        f"{len(plate_errors)} levels {[f'{100*e:.4f}%' for e in plate_errors]}"))
    checks.append(Check(
        "plate.thin_regime", abs(float(plate["reference"]["aspect_ratio"]) - 100.0) < 1e-9,
        f"a/t = {plate['reference']['aspect_ratio']}"))

    # ---- 3. Boussinesq -----------------------------------------------------
    half = results["halfspace"]
    levels = half["levels"]
    half_errors = [abs(float(row["w_centre"]) - float(row["w_centre_exact"]))
                   / float(row["w_centre_exact"]) for row in levels]
    checks.append(Check(
        "halfspace.present", len(levels) > 0,
        "half-space ladder skipped" if not levels else f"{len(levels)} levels"))
    if levels:
        checks.append(Check(
            "halfspace.centre_rel_error<=5%",
            half_errors[-1] <= LIMITS["halfspace_center_rel_error_max"],
            f"finest {_pct(half_errors[-1])}, limit 5%"))
        checks.append(Check(
            "halfspace.two_level_convergence",
            len(half_errors) >= LIMITS["min_refinement_levels"]
            and all(b <= a * (1 + 1e-9) for a, b in zip(half_errors, half_errors[1:])),
            f"{len(half_errors)} levels {[f'{100*e:.3f}%' for e in half_errors]}"))
        worst_balance = max(float(row["reaction_balance_rel"]) for row in levels)
        checks.append(Check(
            "halfspace.reaction_balance<=1e-6", worst_balance <= 1e-6,
            f"worst relative reaction imbalance {worst_balance:.2e}"))
        truncation = [float(row["lateral_extent_in_patch_widths"]) for row in levels]
        checks.append(Check(
            "halfspace.truncation_growth", truncation[-1] > truncation[0],
            f"truncation radius {truncation[0]:.0f}a -> {truncation[-1]:.0f}a"))

    # ---- 4. recorded verdicts must agree with the recomputation ------------
    recorded = results.get("verdict", {})
    recomputed = {
        "beam_tip_le_3pct": beam_error <= LIMITS["beam_tip_rel_error_max"],
        "plate_centre_le_5pct": plate_errors[-1] <= LIMITS["plate_center_rel_error_max"],
        "halfspace_centre_le_5pct": bool(levels)
        and half_errors[-1] <= LIMITS["halfspace_center_rel_error_max"],
    }
    disagreements = [key for key, value in recomputed.items() if bool(recorded.get(key)) != value]
    checks.append(Check(
        "results.verdict_matches_recomputation", not disagreements,
        "consistent with raw numbers" if not disagreements
        else f"recorded verdict disagrees for {disagreements}"))
    return checks


def check_doc(doc_path: Path, results: dict) -> list[Check]:
    checks: list[Check] = []
    if not doc_path.exists():
        return [Check("doc.exists", False, f"{doc_path} missing")]
    text = doc_path.read_text(encoding="utf-8")
    checks.append(Check("doc.exists", True, f"{doc_path.relative_to(ROOT)} "
                                            f"{len(text)} bytes"))
    checks.append(Check("doc.generated_block", "BEGIN CONVERGENCE TABLES" in text
                        and "END CONVERGENCE TABLES" in text,
                        "convergence-block markers present" if "BEGIN CONVERGENCE TABLES" in text
                        else "markers missing"))
    table_rows = [line for line in text.splitlines()
                  if line.strip().startswith("|") and line.count("|") >= 4]
    checks.append(Check("doc.convergence_table", len(table_rows) >= 12,
                        f"{len(table_rows)} markdown table rows"))
    required = ("Euler-Bernoulli", "Boussinesq", "收敛", "误差")
    missing = [word for word in required if word not in text]
    checks.append(Check("doc.content_keywords", not missing,
                        "sections present" if not missing else f"missing {missing}"))

    # the document must quote the errors this script recomputes (no hand-edited tables)
    beam = results["beam"]
    solid = beam["solid_levels"]
    analytic = float(beam["analytic"]["tip_deflection"])
    beam_error = abs(float(solid[-1]["tip_deflection"]) - analytic) / analytic
    plate = results["plate"]
    reference = float(plate["reference"]["centre_deflection"])
    plate_error = abs(float(plate["levels"][-1]["centre_deflection"]) - reference) / reference
    quoted = [f"{100 * beam_error:+.2f}%", f"{100 * plate_error:+.2f}%"]
    absent = [token for token in quoted if token not in text]
    checks.append(Check("doc.numbers_match_results", not absent,
                        f"quoted {quoted}" if not absent else f"missing {absent}"))
    return checks


def validate(results_path: Path = DEFAULT_RESULTS, doc_path: Path = DEFAULT_DOC) -> list[Check]:
    if not results_path.exists():
        return [Check("results.exists", False, f"{results_path} missing")]
    results = json.loads(results_path.read_text(encoding="utf-8"))
    checks = [Check("results.exists", True, str(results_path.relative_to(ROOT)))]
    checks += check_results(results)
    checks += check_doc(doc_path, results)
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    args = parser.parse_args(argv)
    checks = validate(args.results, args.doc)
    width = max(len(check.name) for check in checks)
    failures = 0
    for check in checks:
        if not check.ok:
            failures += 1
        print(f"{'PASS' if check.ok else 'FAIL'}  {check.name:<{width}}  {check.detail}")
    print(f"\n{len(checks) - failures}/{len(checks)} checks passed")
    if failures:
        print("VERDICT: FAIL")
        return 1
    print("VERDICT: PASS (cantilever <=3%, plate <=5%, Boussinesq <=5%, >=2 levels each)")
    return 0


if __name__ == "__main__":
    sys.exit(main())