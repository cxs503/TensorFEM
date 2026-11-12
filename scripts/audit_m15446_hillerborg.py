#!/usr/bin/env python3
"""Board audit of m15446 (anchor REPRO-FEM-HILLERBORG1976) — evidence extraction.

Independently re-derives every check the audit board is asked to make, so the
verdict is machine-reproducible rather than a reading of the mission report:

1. run_artifacts: locate ``repro.json`` / figures, hash them, state where they
   live (mission worktree, mission branch, or main).
2. benchmark referee: query ``tensor_benchmark_cases`` for the anchor and
   report ``case_name/metric/tolerance`` rows; absence is itself the finding.
3. criteria: recompute pass/fail from the raw numbers vs the *declared*
   tolerances (does not trust the artifact's own ``passed`` flag).
4. integration: grep the ecosystem checkout for callers of the new module
   outside its own driver/tests.
5. level: apply the maturity_gate JUDGE_SCHEME ladder to what was found.

Output: docs/board_audit/m15645/verify_output.json (all numbers quoted in the
report come from that file).

Usage: python3 scripts/audit_m15446_hillerborg.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys

AUDIT_MID = 15645
SRC_MID = 15446
ANCHOR = "REPRO-FEM-HILLERBORG1976"
REPO = "TensorFEM"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "docs", "board_audit", f"m{AUDIT_MID}")
ART_COPY = os.path.join(OUT_DIR, f"artifacts_m{SRC_MID}")
SRC_TREE = f"/root/tp-agent/work/m{SRC_MID}"
ECO_REPO = f"/root/tensor-ecosystem/{REPO}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def psql(sql):
    r = subprocess.run(
        ["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform", "-tA", "-F", "\x01", "-c", sql],
        capture_output=True, text=True)
    if r.returncode != 0:
        return [["ERR", r.stderr.strip()[:300]]]
    return [ln.split("\x01") for ln in r.stdout.strip().split("\n") if ln]


def artifacts_report():
    """Where do the run artifacts actually live?"""
    out = {"mission_worktree": SRC_TREE, "files": []}
    for rel in ("repro.json", "fig_size_effect.png", "fig_load_cmod.png"):
        p = os.path.join(SRC_TREE, "artifacts", "hillerborg1976", rel)
        row = {"path": p, "exists": os.path.exists(p)}
        if row["exists"]:
            row["bytes"] = os.path.getsize(p)
            row["sha256"] = sha256(p)
            dst = os.path.join(ART_COPY, rel)
            os.makedirs(ART_COPY, exist_ok=True)
            with open(p, "rb") as fi, open(dst, "wb") as fo:
                fo.write(fi.read())
            row["copied_to"] = os.path.relpath(dst, HERE)
        out["files"].append(row)
    # presence in main checkout / mission branch
    def git(args):
        r = subprocess.run(["git", "-C", SRC_TREE] + args, capture_output=True, text=True)
        return r.stdout.strip(), r.returncode

    main_present = os.path.exists(os.path.join(HERE, "artifacts", "hillerborg1976", "repro.json"))
    out["present_in_main_checkout"] = main_present
    head, rc = git(["rev-parse", "HEAD"])
    out["mission_head"] = head if rc == 0 else None
    _, rc_branch = git(["cat-file", "-e", "bc5d61a8:artifacts/hillerborg1976/repro.json"])
    out["present_in_mission_branch_commit"] = rc_branch == 0
    out["main_landing_commit"] = None
    log, _ = git(["log", "--all", "--oneline", "--grep", f"m{SRC_MID}", "--", "src/tensorfem/fictitious_crack.py"])
    if log:
        out["main_landing_commit"] = log.splitlines()[0]
    return out


def benchmark_report():
    rows = psql(f"SELECT case_name, metric, tolerance FROM tensor_benchmark_cases "
                f"WHERE anchor_slug='{ANCHOR}'")
    out = {"anchor": ANCHOR, "registered_rows": len(rows) if rows != [["ERR", ""]] else 0,
           "rows": [r for r in rows if r and r[0] != "ERR"],
           "referee_available": bool(rows) and rows[0][0] != "ERR"}
    # neighbouring registered case on the same paper, for context
    near = psql("SELECT anchor_slug, case_name, metric, tolerance FROM tensor_benchmark_cases "
                "WHERE anchor_slug IN ('XVAL-FEM-TPB1976')")
    out["same_paper_other_anchor"] = [r for r in near if r and r[0] != "ERR"]
    return out


def criteria_report(repro_path):
    """Recompute every criterion from its raw numbers; never trust `passed`."""
    d = json.load(open(repro_path))
    checks = []

    def add(name, quantity, got, ref, tol, unit, passed, note=""):
        checks.append({"criterion": name, "quantity": quantity, "got": got, "ref": ref,
                       "tolerance": tol, "unit": unit, "passed_recomputed": bool(passed),
                       "note": note})

    c = d["criteria"]
    k2 = c["K2_law_area"]
    add("K2_law_area", "int f(w)dw vs G_F (rel.)", k2["worst_rel_error"], 0.0, k2["tolerance"], "-",
        k2["worst_rel_error"] <= k2["tolerance"])
    k1 = c["K1_elastic"]
    add("K1_elastic", "K_I from model compliance vs Srawley-Gross", k1["worst_abs_error_pct"], 0.0,
        k1["tolerance_pct"], "%", k1["worst_abs_error_pct"] <= k1["tolerance_pct"])
    k3 = c["K3_balance"]
    add("K3_balance", "worst |W_ext-U_solid-U_if-D|/W_ext", k3["worst_abs_balance"], 0.0,
        k3["tolerance"], "-", k3["worst_abs_balance"] <= k3["tolerance"])
    k4 = c["K4_lefm"]
    add("K4_lefm", "P_peak/P_LEFM at W/l_ch=8", k4["ratio"], 1.0, k4["tolerance_pct"], "ratio",
        k4["abs_error_pct"] <= k4["tolerance_pct"], "LEFM asymptote")
    k5 = c["K5_strength"]
    add("K5_strength", "P_peak/P_strength at W/l_ch=0.25", k5["ratio"], k5["band"],
        0.0, "ratio", k5["band"][0] <= k5["ratio"] <= k5["band"][1], "plastic-limit band")
    k6 = c["K6_mesh"]
    add("K6_mesh", "P_peak drift nodes_y 48->96", k6["drift_pct"], 0.0, k6["tolerance_pct"], "%",
        k6["drift_pct"] <= k6["tolerance_pct"])
    k7 = c["K7_penalty"]
    add("K7_penalty", "P_peak drift over k0 1e2..1e4", k7["drift_pct"], 0.0, k7["tolerance_pct"], "%",
        k7["drift_pct"] <= k7["tolerance_pct"])
    k8 = c["K8_aW"]
    add("K8_aW", "P_peak(a0/W) monotone decreasing", k8["monotone_decreasing"], True, 0.0, "bool",
        bool(k8["monotone_decreasing"]))
    k9 = c["K9_master"]
    add("K9_master", "size-effect master-curve collapse deviation", k9["max_deviation_pct"], 0.0,
        k9["tolerance_pct"], "%", k9["max_deviation_pct"] <= k9["tolerance_pct"])
    k10 = c["K10_gf"]
    add("K10_gf", "G_F back-calculated vs 100 J/m^2 (rel.)", k10["worst_rel_error"], 0.0,
        k10["tolerance"], "-", k10["worst_rel_error"] <= k10["tolerance"])

    # cross-check: does our recomputation agree with the artifact's own flags?
    disagreements = [{"criterion": k, "artifact_passed": bool(v.get("passed")),
                      "recomputed": next(x["passed_recomputed"] for x in checks if x["criterion"] == k)}
                     for k, v in c.items()
                     if bool(v.get("passed")) != next(x["passed_recomputed"] for x in checks if x["criterion"] == k)]

    series = [{"W_over_lch": p["W_over_lch"], "peak_load_N": p["path"]["peak_load_n"],
               "peak_over_lefm": p["peak_over_lefm"], "peak_over_strength": p["peak_over_strength"],
               "G_F_backcalc": p["G_F_backcalc_J_m2"], "balance": p["path"]["max_abs_balance"],
               "converged_steps": p["path"]["converged_steps"], "steps": p["path"]["steps"]}
              for p in d["cases"]]
    return {
        "artifact": os.path.relpath(repro_path, HERE),
        "n_criteria": len(checks),
        "n_passed": sum(1 for x in checks if x["passed_recomputed"]),
        "n_failed": sum(1 for x in checks if not x["passed_recomputed"]),
        "checks": checks,
        "flag_disagreements": disagreements,
        "size_series": series,
        "law_shape_difference_pct": d.get("law_shape", {}).get("difference_pct"),
        "runtime_s": d.get("runtime_s"),
        "env": {"python": d.get("python"), "numpy": d.get("numpy_version")},
    }


def integration_report():
    pat = re.compile(r"fictitious_crack")
    callers, allowed = [], {"benchmarks/hillerborg1976_repro.py", "tests/test_fictitious_crack.py",
                            "scripts/hillerborg1976_publish.py"}
    for root, _dirs, files in os.walk(HERE):
        if "/.git" in root or "/__pycache__" in root:
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            p = os.path.relpath(os.path.join(root, f), HERE)
            if p in allowed or p.startswith("scripts/audit_"):
                continue
            try:
                txt = open(os.path.join(root, f), errors="ignore").read()
            except OSError:
                continue
            for i, line in enumerate(txt.splitlines(), 1):
                if pat.search(line) and "import" in line:
                    callers.append({"file": p, "line": i, "code": line.strip()[:120]})
    init = os.path.join(HERE, "src", "tensorfem", "__init__.py")
    exported = "fictitious" in open(init, errors="ignore").read() if os.path.exists(init) else False
    return {"main_solver_callers": callers, "n_callers": len(callers),
            "exported_in_package_init": exported,
            "driver_calls_module": "benchmarks/hillerborg1976_repro.py:46",
            "l3_satisfied": bool(callers)}


def level_decide(art, bench, integ):
    have_impl = os.path.exists(os.path.join(HERE, "src/tensorfem/fictitious_crack.py"))
    have_entry = os.path.exists(os.path.join(HERE, "benchmarks/hillerborg1976_repro.py"))
    have_run = any(f["exists"] for f in art["files"])
    have_tests = os.path.exists(os.path.join(HERE, "tests/test_fictitious_crack.py"))
    ladder = {"L1_tests": have_tests, "L2_impl+entry+run_artifacts": have_impl and have_entry and have_run,
              "L3_integration_callers": integ["l3_satisfied"],
              "L4_benchmark+repro_cmd": bench["referee_available"]}
    level = 0
    if ladder["L1_tests"]:
        level = 1
    if ladder["L2_impl+entry+run_artifacts"]:
        level = 2
    if level == 2 and ladder["L3_integration_callers"]:
        level = 3
    if level == 3 and ladder["L4_benchmark+repro_cmd"]:
        level = 4
    return {"ladder": ladder, "assigned_level": level}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    art = artifacts_report()
    bench = benchmark_report()
    repro = os.path.join(SRC_TREE, "artifacts", "hillerborg1976", "repro.json")
    crit = criteria_report(repro)
    integ = integration_report()
    lv = level_decide(art, bench, integ)
    out = {"audit_mission": AUDIT_MID, "audited_mission": SRC_MID, "anchor": ANCHOR, "repo": REPO,
           "artifacts": art, "benchmark": bench, "criteria": crit, "integration": integ,
           "level": lv}
    path = os.path.join(OUT_DIR, "verify_output.json")
    json.dump(out, open(path, "w"), indent=1, default=str)
    print(json.dumps({"n_passed": crit["n_passed"], "n_failed": crit["n_failed"],
                      "flag_disagreements": crit["flag_disagreements"],
                      "benchmark_available": bench["referee_available"],
                      "n_callers": integ["n_callers"], "level": lv["assigned_level"],
                      "written": os.path.relpath(path, HERE)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())