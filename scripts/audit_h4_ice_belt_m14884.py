#!/usr/bin/env python3
"""Board audit (mission 15020) of H4-ICE-BELT-FEM / m14884.

Independently re-runs the m14884 deliverable and checks the numbers it claims,
instead of re-reading its prose:

1. ``reproduce``  -- regenerate ``artifacts/H4-ICE-BELT-FEM/pressure_area_zoned_panel.json``
   from the *merged* source (``src/tensorfem/ice_belt_pressure_area.py``) into a temp
   file and deep-compare field by field against the shipped artifact.
2. ``claims``     -- re-derive every quantitative claim of the m14884 report from the
   artifact and apply a pre-declared threshold (3 % for the physics-vs-reference
   comparison, 10 % for the declared inter-discretisation tolerance, exactly 1e-6 for
   the implementation-consistency claims that the module states to ~1e-11).
3. ``gaps``       -- record what the artifact itself declares as WITHHELD and whether
   the capability is reachable from any solver/platform caller.

Output: ``artifacts/audit_m14884/repro_audit.json`` (audit evidence, machine readable).
Exit code 0 = every checked claim reproduces and passes; 1 = at least one mismatch.

Usage::

    PYTHONPATH=src python scripts/audit_h4_ice_belt_m14884.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "H4-ICE-BELT-FEM" / "pressure_area_zoned_panel.json"
OUT = ROOT / "artifacts" / "audit_m14884" / "repro_audit.json"
IMPL = ROOT / "src" / "tensorfem" / "ice_belt_pressure_area.py"
TESTS = ROOT / "tests" / "test_ice_belt_pressure_area.py"

#: pre-declared thresholds (fixed before looking at the numbers)
REL_TOL_PHYSICS = 0.03      # plant standard: > 3 % vs a reference needs justification
REL_TOL_DECLARED = 0.10     # the module's own inter-discretisation tolerance
REL_TOL_IMPL = 1.0e-6       # implementation-consistency claims (module says ~1e-11)


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deep_diff(a, b, path="", out=None, tol=1.0e-9):
    """Field-by-field comparison; relative tolerance, absolute fallback near zero."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k not in b:
                out.append({"path": f"{path}/{k}", "kind": "missing_in_regen"})
                continue
            deep_diff(a[k], b[k], f"{path}/{k}", out, tol)
        for k in b:
            if k not in a:
                out.append({"path": f"{path}/{k}", "kind": "extra_in_regen"})
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append({"path": path, "kind": "length", "a": len(a), "b": len(b)})
        for i, (x, y) in enumerate(zip(a, b)):
            deep_diff(x, y, f"{path}[{i}]", out, tol)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if a != b and not math.isclose(a, b, rel_tol=tol, abs_tol=tol):
            out.append({"path": path, "kind": "numeric", "shipped": a, "regenerated": b})
    elif a != b:
        out.append({"path": path, "kind": "value", "shipped": a, "regenerated": b})
    return out


def reproduce() -> dict:
    tmp = pathlib.Path(tempfile.mkdtemp()) / "regen.json"
    cmd = [sys.executable, "-m", "tensorfem.ice_belt_pressure_area", "--out", str(tmp)]
    env = dict(os.environ, PYTHONPATH="src")   # keep the venv/site LD_LIBRARY_PATH intact
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    if proc.returncode != 0 or not tmp.exists():
        return {"ran": False, "cmd": " ".join(cmd), "returncode": proc.returncode,
                "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:], "diffs": None}
    shipped = json.loads(ARTIFACT.read_text())
    regen = json.loads(tmp.read_text())
    diffs = deep_diff(shipped, regen)
    return {"ran": True, "cmd": " ".join(cmd), "returncode": 0,
            "stdout_tail": proc.stdout.strip().splitlines()[-1:],
            "shipped_sha256": sha256(ARTIFACT), "regenerated_sha256": sha256(tmp),
            "byte_identical": sha256(ARTIFACT) == sha256(tmp),
            "field_diffs": diffs[:20], "field_diff_count": len(diffs)}


def claims(a: dict) -> dict:
    c1 = a["criterion_1_pressure_area_curve"]
    c2 = a["criterion_2_panel_cross_validation"]
    c3 = a["criterion_3_design_pressure_zones"]

    def row(name, got, threshold, cmp="le", unit="rel"):
        ok = got <= threshold if cmp == "le" else got >= threshold
        return {"claim": name, "got": got, "threshold": threshold, "cmp": cmp,
                "unit": unit, "pass": bool(ok)}

    rows = [
        row("pressure-area curve, max route disagreement (marginal route)",
            c1["max_rel_diff_marginal"], REL_TOL_IMPL),
        row("fitted log-log exponent error vs alpha=0.3",
            abs(c1["fitted_exponent_error"]), REL_TOL_IMPL),
        row("scaling invariant p*A**alpha spread", c1["max_rel_diff_scaling_invariant"],
            REL_TOL_IMPL),
        row("stroke/inverse round-trip (area from force)", c1["max_rel_diff_area_inverse"],
            REL_TOL_IMPL),
    ]
    for case in ("full_field", "centred_patch"):
        b = c2[case]
        rows.append(row(f"{case}: MITC4 vs Q4-SRI peak deflection",
                        b["pairwise_rel_diff"]["mitc4_vs_q4sri"], REL_TOL_DECLARED))
        rows.append(row(f"{case}: MITC4 vs Kirchhoff Navier (analytic reference)",
                        b["pairwise_rel_diff"]["mitc4_vs_navier"], REL_TOL_PHYSICS))
        rows.append(row(f"{case}: Q4-SRI vs Kirchhoff Navier (analytic reference)",
                        b["pairwise_rel_diff"]["q4sri_vs_navier"], REL_TOL_PHYSICS))

    ms = c2["mesh_study"]["rows"]
    errs = [abs(r["rel_error"]) for r in ms]
    rows.append({"claim": "mesh study: |error| decreases monotonically with refinement",
                 "got": errs, "threshold": "monotone decreasing", "cmp": "trend",
                 "unit": "rel", "pass": all(y < x for x, y in zip(errs, errs[1:]))})
    rows.append(row("mesh study: finest mesh (21x21) vs Navier", errs[-1], REL_TOL_PHYSICS))

    z = c3["zone_checks"]
    rows.append({"claim": "zone partition tiles the hull without gaps/overlaps",
                 "got": c3["partition_check"]["passed"], "threshold": True, "cmp": "eq",
                 "unit": "bool", "pass": bool(c3["partition_check"]["passed"])})
    rows.append({"claim": "ice belt inside depth", "got": c3["belt_check"]["passed"],
                 "threshold": True, "cmp": "eq", "unit": "bool",
                 "pass": bool(c3["belt_check"]["passed"])})
    # Z7 zone-factor linearity is the only zone invariant that carries numbers
    z7 = z.get("Z7_detail") or []
    z7_max = max((abs(d["abs_rel_error"]) for d in z7), default=None)
    if z7_max is not None:
        rows.append(row("Z7 zone-factor linearity (max |rel error|)", z7_max, REL_TOL_IMPL))
    rows.append({"claim": "Z7 zone-factor linearity flag", "got": z.get("Z7_zone_factor_linearity"),
                 "threshold": True, "cmp": "eq", "unit": "bool",
                 "pass": bool(z.get("Z7_zone_factor_linearity"))})
    class_pairs = (c3.get("class_check") or {}).get("pairs", [])
    rows.append({"claim": "Z9 class ordering vs injected registry (strict, all pairs)",
                 "got": f"{sum(1 for p in class_pairs if p['non_decreasing_with_rank'])}/{len(class_pairs)}",
                 "threshold": "all pairs", "cmp": "eq", "unit": "count",
                 "pass": bool(class_pairs) and all(p["non_decreasing_with_rank"]
                                                   for p in class_pairs)})
    return {
        "thresholds": {"physics_vs_reference": REL_TOL_PHYSICS,
                       "declared_inter_discretisation": REL_TOL_DECLARED,
                       "implementation_consistency": REL_TOL_IMPL},
        "rows": rows,
        "passed": all(r["pass"] for r in rows),
    }


def gaps(a: dict) -> dict:
    c1 = a["criterion_1_pressure_area_curve"]
    rule = c1.get("rule_table_comparison", {})
    # is the capability reachable from any caller outside its own module/tests?
    grep = subprocess.run(
        ["grep", "-rl", "--include=*.py", "-F", "ice_belt_pressure_area",
         str(ROOT / "src"), str(ROOT / "benchmarks"), str(ROOT / "examples")],
        capture_output=True, text=True)
    callers = [str(pathlib.Path(p).relative_to(ROOT)) for p in grep.stdout.split() if p]
    callers = [c for c in callers if not c.endswith("ice_belt_pressure_area.py")]
    exported = subprocess.run(
        ["grep", "-c", "ice_belt_pressure_area", str(ROOT / "src/tensorfem/__init__.py")],
        capture_output=True, text=True).stdout.strip()
    return {
        "declared_withheld": a.get("withheld_items"),
        "rule_table_comparison_status": rule.get("status"),
        "rule_table_fail_stage": rule.get("fail_stage"),
        "pressure_anchor_provenance": a["law"].get("provenance"),
        "demo_p_ref_mpa": a["law"].get("p_ref_mpa"),
        "integration_callers_outside_module": callers,
        "__init___export_lines": exported,
        "overall_status_shipped": a.get("overall_status"),
    }


def main() -> int:
    a = json.loads(ARTIFACT.read_text())
    report = {
        "audit": "m15020 board audit of m14884 (H4-ICE-BELT-FEM)",
        "shipped_artifact": str(ARTIFACT.relative_to(ROOT)),
        "shipped_artifact_sha256": sha256(ARTIFACT),
        "impl_file": str(IMPL.relative_to(ROOT)),
        "impl_sha256": sha256(IMPL),
        "tests_file": str(TESTS.relative_to(ROOT)),
        "tests_sha256": sha256(TESTS),
        "reproduce": reproduce(),
        "claims": claims(a),
        "gaps": gaps(a),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    repro = report["reproduce"]
    print(f"reproduce: ran={repro['ran']} field_diffs={repro.get('field_diff_count')} "
          f"byte_identical={repro.get('byte_identical')}")
    for r in report["claims"]["rows"]:
        print(f"  [{'PASS' if r['pass'] else 'FAIL'}] {r['claim']}: {r['got']} "
              f"(thr {r['threshold']})")
    print(f"claims passed: {report['claims']['passed']}  -> {OUT.relative_to(ROOT)}")
    return 0 if (repro["ran"] and repro["field_diff_count"] == 0
                 and report["claims"]["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())