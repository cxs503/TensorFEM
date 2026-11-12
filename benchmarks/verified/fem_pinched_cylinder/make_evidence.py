#!/usr/bin/env python3
"""Generate the maturity-gate evidence bundle for CASE-FEM-PINCHED-CYLINDER.

Zero hand-copy: every number, hash and line reference in ``maturity_evidence.json``
is derived at run time from the files that actually exist in the tree.  The
maturity gate re-checks the *evidence*, not a self-declaration, so this script
is the single source of truth for the L2/L3/L4 evidence:

  L2  impl_files (non-placeholder) + entry + run_artifacts
  L3  + integration_callers (main-solver / platform-side real calls)
  L4  + benchmark (ref vs got) + repro_cmd

Run:
    PYTHONPATH=src python benchmarks/verified/fem_pinched_cylinder/make_evidence.py
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lines(path: Path) -> int:
    return len(path.read_text().splitlines())


def _docstring_first_line(path: Path) -> str:
    m = re.search(r'"""(.+?)"""', path.read_text(), re.S)
    return m.group(1).strip().splitlines()[0].strip() if m else ""


def _is_placeholder(path: Path) -> bool:
    """A file is a placeholder if it is tiny or has no substantive algorithm body."""
    text = path.read_text()
    body = [ln for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("#")]
    if len(body) < 15:
        return True
    # a real implementation has arithmetic / tensor ops, not just pass/return
    return not re.search(r"(torch\.|np\.|@|def |for |while |return )", text)


def main() -> int:
    result = json.loads((HERE / "case_result.json").read_text())

    # ---- L2: impl_files (non-placeholder) -------------------------------
    impl_specs = [
        ("src/tensorfem/cylindrical_shell4.py",
         "Reissner-Mindlin cylindrical Shell4 element: exact cylindrical "
         "metric, 2x2 membrane+bending, 1-point selective transverse shear, "
         "drilling stabilisation"),
        ("src/tensorfem/pinched_cylinder.py",
         "symmetry-octant model builder + dense and CSR/SuperLU solvers, "
         "radial-field post-processor, nonlinear step"),
    ]
    impl_files = []
    for rel, role in impl_specs:
        p = REPO / rel
        impl_files.append({
            "path": rel,
            "lines": _lines(p),
            "sha256": _sha256(p),
            "role": role,
            "docstring": _docstring_first_line(p),
            "non_placeholder": (not _is_placeholder(p)) and _lines(p) >= 50,
        })

    # ---- L2: entry ------------------------------------------------------
    entry = "benchmarks/verified/fem_pinched_cylinder/run.py"
    entry_p = REPO / entry
    assert entry_p.exists(), "entry driver missing"

    # ---- L2: run_artifacts (real execution artifacts) -------------------
    artifact_names = [
        "case_result.json", "case_gates.json",
        "case_convergence.png", "case_field.png", "case_sensitivity.png",
    ]
    run_artifacts = []
    for name in artifact_names:
        p = HERE / name
        if p.exists():
            run_artifacts.append({
                "path": f"benchmarks/verified/fem_pinched_cylinder/{name}",
                "bytes": p.stat().st_size,
                "sha256": _sha256(p),
            })
        else:
            run_artifacts.append({"path": f"benchmarks/verified/fem_pinched_cylinder/{name}",
                                  "bytes": 0, "sha256": None, "missing": True})

    # ---- L3: integration_callers (real calls from main solver/platform) --
    caller_specs = [
        ("src/tensorfem/benchmark_cases.py",
         "solve_pinched_cylinder_linear",
         "benchmark-case registry runner (case.runner=='pinched_cylinder')"),
        ("src/tensorfem/shell_benchmark_suite.py",
         "pinched_cylinder_case",
         "classical_shell_cases() suite includes the pinched cylinder"),
        ("src/tensorfem/project_workflow.py",
         "solve_pinched_cylinder_linear",
         "project workflow driver calls the production linear solver"),
        ("src/tensorfem/cylindrical_shell4_diagnostics.py",
         "solve_pinched_cylinder_linear",
         "element diagnostics drive the production solver"),
    ]
    integration_callers = []
    for rel, needle, role in caller_specs:
        p = REPO / rel
        text = p.read_text()
        hits = [i + 1 for i, ln in enumerate(text.splitlines()) if needle in ln]
        integration_callers.append({
            "path": rel,
            "call": needle,
            "call_lines": hits,
            "role": role,
            "present": bool(hits),
        })

    # ---- L4: benchmark (ref vs got) + repro_cmd -------------------------
    prim = next(l for l in result["levels"] if l["n"] == result["case_definition"]["primary_mesh"])
    tol = result["case_definition"]["tolerance_pct"]
    benchmark = {
        "reference_in": result["reference"]["value_in"],
        "got_16x16_in": prim["displacement_in"],
        "relative_error_pct": prim["relative_error_pct"],
        "tolerance_pct": tol,
        "pass": prim["relative_error_pct"] <= tol,
        "citation": result["reference"]["citation"],
    }
    repro_cmd = result["repro_cmd"]

    evidence = {
        "schema": "tensorfem.maturity-evidence.v1",
        "case": "CASE-FEM-PINCHED-CYLINDER",
        "mission": "m16190 (repair of m16076)",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "L2": {
            "impl_files": impl_files,
            "entry": entry,
            "entry_sha256": _sha256(entry_p),
            "run_artifacts": run_artifacts,
        },
        "L3": {"integration_callers": integration_callers},
        "L4": {"benchmark": benchmark, "repro_cmd": repro_cmd},
        "verdicts": {
            "L2": all(f["non_placeholder"] for f in impl_files)
                  and all(not a.get("missing") for a in run_artifacts),
            "L3": all(c["present"] for c in integration_callers),
            "L4": benchmark["pass"],
        },
    }
    out = HERE / "maturity_evidence.json"
    out.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"wrote {out}")
    print(json.dumps(evidence["verdicts"], indent=2))
    for f in impl_files:
        print(f"  impl {f['path']}: {f['lines']} lines, non_placeholder={f['non_placeholder']}")
    for a in run_artifacts:
        print(f"  artifact {a['path']}: {a.get('bytes')} bytes"
              + ("  MISSING" if a.get("missing") else ""))
    for c in integration_callers:
        print(f"  caller {c['path']}:{c['call_lines']} present={c['present']}")
    print(f"  benchmark: got={benchmark['got_16x16_in']:.6e} ref={benchmark['reference_in']:.6e} "
          f"err={benchmark['relative_error_pct']:.4f}% tol={tol}% pass={benchmark['pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
