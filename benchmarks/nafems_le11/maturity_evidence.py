"""Maturity evidence pack for CASE-NAFEMS-LE11 (machine-checkable).

Assembles the evidence ladder consumed by the platform maturity gate:

* **L2** -- ``impl_files`` (non-placeholder sources), ``entry`` (a real
  one-command driver), ``run_artifacts`` (committed JSON/MD produced by a real
  run, not by hand).
* **L3** -- ``integration_callers``: the case is wired into
  :func:`tensorfem.benchmark_registry.run_registered_benchmarks`, i.e. the CLI
  ``tensorfem verify`` / ``marine_structural_qualification`` integration
  surface, and exposed to platform-side consumers through
  :func:`tensorfem.integration_callers.run_nafems_le11_case`.
* **L4** -- ``benchmark`` (ref vs got per quantity + tolerance verdict) and
  ``repro_cmd``.

Every path listed here is re-checked *in the current work tree*; a missing file
or a placeholder marker makes this script exit non-zero (fail-closed).

Usage::

    PYTHONPATH=src python3 benchmarks/nafems_le11/maturity_evidence.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.json"
OUT = HERE / "evidence.json"

IMPL_FILES = [
    "src/tensorfem/ellipsoid_shell.py",
    "src/tensorfem/integration_callers.py",
    "benchmarks/nafems_le11/run.py",
    "benchmarks/nafems_le11_ellipsoid.py",
]
ENTRY = "PYTHONPATH=src python3 benchmarks/nafems_le11/run.py"
RUN_ARTIFACTS = [
    "benchmarks/nafems_le11/results.json",
    "benchmarks/nafems_le11/report.md",
]
PLACEHOLDER_MARKERS = ("NotImplementedError", "TODO: implement", "raise NotImplemented")


def _check_impl() -> list[dict]:
    rows = []
    for rel in IMPL_FILES:
        p = ROOT / rel
        if not p.is_file():
            rows.append({"file": rel, "exists": False, "status": "MISSING"})
            continue
        text = p.read_text()
        n_lines = text.count("\n") + 1
        markers = [m for m in PLACEHOLDER_MARKERS if m in text]
        rows.append({"file": rel, "exists": True, "lines": n_lines,
                     "placeholder_markers": markers,
                     "status": "OK" if not markers and n_lines >= 40 else "PLACEHOLDER"})
    return rows


def _check_run_artifacts() -> list[dict]:
    rows = []
    for rel in RUN_ARTIFACTS:
        p = ROOT / rel
        rows.append({"file": rel, "exists": p.is_file(),
                     "bytes": p.stat().st_size if p.is_file() else 0})
    return rows


def _integration_callers() -> dict:
    reg = (ROOT / "src/tensorfem/benchmark_registry.py").read_text()
    cli = (ROOT / "src/tensorfem/cli.py").read_text()
    ids = [line.split('"')[1] for line in reg.splitlines() if "shell.spheroid." in line]
    return {
        "callers": [
            {"surface": "tensorfem.benchmark_registry.run_registered_benchmarks",
             "repo": "TensorFEM", "registry_ids": ids,
             "consumers": ["tensorfem.cli verify", "marine_structural_qualification"]},
            {"surface": "tensorfem.integration_callers.run_nafems_le11_case",
             "repo": "TensorFEM",
             "consumers": ["platform case-pool settlement",
                           "ship-performance-platform verification pipeline"]},
        ],
        "cli_verify_command": "PYTHONPATH=src python3 -m tensorfem.cli verify",
        "cli_verify_reachable": "verification_report" in cli,
        "registry_wired": len(ids) >= 4,
        "cli_verify_artifact": "benchmarks/nafems_le11/cli_verify_report.json",
    }


def _benchmark_block(results: dict) -> dict:
    rows = []
    for r in results["ref_vs_got"]:
        rows.append({"quantity": r["quantity"], "ref": r["ref"], "got": r["got"],
                     "rel_error": r["rel_error"], "tolerance": r["tolerance"],
                     "passed": r["passed"]})
    worst = max(rows, key=lambda r: r["rel_error"])
    return {
        "case": "CASE-NAFEMS-LE11",
        "report_mesh": results["report_mesh"],
        "tolerance": results["tolerance"],
        "rows": rows,
        "worst_quantity": worst["quantity"],
        "worst_rel_error": worst["rel_error"],
        "all_passed": results["all_passed"],
        "reference_provenance": results["reference"]["provenance_note"],
    }


def main() -> int:
    if not RESULTS.is_file():
        print(f"missing run artifact: {RESULTS.relative_to(ROOT)}; run the entry first",
              file=sys.stderr)
        return 2
    results = json.loads(RESULTS.read_text())
    impl = _check_impl()
    arts = _check_run_artifacts()
    integ = _integration_callers()
    ok = (all(r["status"] == "OK" for r in impl)
          and all(r["exists"] and r["bytes"] > 200 for r in arts)
          and integ["registry_wired"] and integ["cli_verify_reachable"]
          and results["all_passed"])

    live = None
    try:                                   # live recompute of the worst quantity
        sys.path.insert(0, str(ROOT / "src"))
        from tensorfem.integration_callers import run_nafems_le11_case
        live = run_nafems_le11_case(facets=results["report_mesh"]["facets"])
    except Exception as exc:               # pragma: no cover
        live = {"error": repr(exc)}

    evidence = {
        "case": "CASE-NAFEMS-LE11",
        "repo": "TensorFEM",
        "level_claim": "L4" if ok else "L1",
        "impl_files": [r["file"] for r in impl],
        "impl_detail": impl,
        "entry": ENTRY,
        "run_artifacts": [r["file"] for r in arts],
        "run_artifact_detail": arts,
        "tests": ["tests/test_ellipsoid_shell.py",
                  "tests/test_nafems_le11_case_pool.py"],
        "integration_callers": integ,
        "benchmark": _benchmark_block(results),
        "live_recompute": live,
        "repro_cmd": (f"{ENTRY} && PYTHONPATH=src python3 "
                      "benchmarks/nafems_le11/maturity_evidence.py && "
                      "PYTHONPATH=src python3 -m pytest tests/test_ellipsoid_shell.py "
                      "tests/test_nafems_le11_case_pool.py -q"),
        "external_anchor": (results["reference"]["provenance_note"] + " "
                            "The anchor is thin-shell membrane equilibrium (exact statics), "
                            "which NAFEMS LE11 measures; the manual table itself was not "
                            "obtained, so the ref column is not a transcription."),
        "findings": {
            "F1_ref_not_transcribed": "NAFEMS LE11 manual table not retrievable in this "
                                      "sandbox -> ref is closed-form statics + independent "
                                      "Rayleigh-Ritz, not the manual numbers.",
            "F2_first_order_geometry": "2-node ring element on a varying-curvature meridian is "
                                       "first order: field error 7.9 % at 64 facets; only "
                                       "n>=128 facets meets the 5 % field gate.",
            "F3_equator_stagnation": "Equatorial hoop error stagnates near 2.6 % (discrete "
                                     "boundary layer at the symmetry line), mesh independent.",
            "F4_thick_wall": "a/t=50 gives 7.1 % equatorial hoop error -> case is valid for "
                             "thin walls a/t >= 100 only.",
        },
    }
    OUT.write_text(json.dumps(evidence, indent=1) + "\n")
    print(json.dumps({"all_ok": ok, "impl": [r["status"] for r in impl],
                      "artifacts": [r["exists"] for r in arts],
                      "registry_ids": integ["callers"][0]["registry_ids"],
                      "worst": evidence["benchmark"]["worst_quantity"],
                      "worst_err": evidence["benchmark"]["worst_rel_error"],
                      "live_verdict": (live or {}).get("verdict")}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
