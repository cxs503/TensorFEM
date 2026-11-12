#!/usr/bin/env python
"""Publish the CASE-FEM-CLAMPED-PLATE-UNIFORM progress report.

All numbers and the figure are read from the case artifact, so the report cannot
drift from the run (zero hand-copy).

    PYTHONPATH=src python scripts/publish_case_report.py [--dry-run]
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASE = "CASE-FEM-CLAMPED-PLATE-UNIFORM"
ART = ROOT / "benchmarks" / "artifacts" / CASE
FIG = ROOT / "docs" / "assets" / "case-clamped-plate-uniform" / "clamped_plate_uniform.png"
MISSION_ID = int(os.environ.get("CASE_MISSION_ID", "16181"))


def pct(x: float, digits: int = 3) -> str:
    return ("%+." + str(digits) + "f%%") % (100.0 * x)


def build() -> dict:
    d = json.loads((ART / "case_result.json").read_text())
    pr = d["primary"]
    par = d["parameters"]
    conv = d["convergence"]["meshes"]
    rich = d["convergence"]["richardson"]
    thick = d["sensitivity"]["thickness_sweep_nx16"]
    nu = d["sensitivity"]["poisson_sweep_nx16"]
    ritz = d["independent_reference_ritz"]

    def anchor(name):
        return next(a for a in d["anchors"] if a["anchor"] == name)

    md: list[str] = []
    A = md.append
    A("## 1. Case")
    A("")
    A("Square plate, side `a`, thickness `t`, four edges fully built in, uniform "
      "lateral pressure `q`, Kirchhoff (thin) regime `a/t = %.0f` -- the elementary "
      "unit cell of ship plating between frames under hydrostatic head, and the "
      "first rung of the plate-verification ladder for shell/solid elements."
      % par["a_over_t"])
    A("")
    A("| parameter | value |")
    A("|---|---|")
    A("| a | %.3g m |" % par["side_a_m"])
    A("| t | %.4g m (a/t = %.0f) |" % (par["thickness_m"], par["a_over_t"]))
    A("| E | %.4g Pa |" % par["young_pa"])
    A("| nu | %.2f |" % par["poisson"])
    A("| q | %.4g Pa |" % par["pressure_pa"])
    A("| D = E t^3/(12(1-nu^2)) | %.6g N m |" % par["D_Nm"])
    A("")
    A("Key quantity: `alpha = w_max D/(q a^4)`.")
    A("")
    A("**Reference:** " + d["reference"] + ". The entry is printed to three "
      "significant figures, so it carries a rounding band of +-%.2f%% before any "
      "numerical error is attributed -- stated here so the 3 %% verdict is read "
      "correctly." % (100 * d["reference_rounding_band_relative"]))
    A("")
    A("## 2. Method")
    A("")
    A("No new element code: the field is produced by the already merged Q4 Mindlin "
      "path `plate_uniform_load.solve_uniform_square_plate` (a `winkler_k = 0` "
      "reduction of `plate_mindlin.solve_plate_on_winkler_foundation`, "
      "`boundary=\"clamped\"`), even element counts so a centre node exists. "
      "Reference chain: (i) the Rayleigh-Ritz code is first validated on the "
      "*simply supported* square against the exact Navier series -- the machinery "
      "anchor; (ii) the same code then gives the clamped reference, which has no "
      "closed form; (iii) the FE is compared with the table, with the Ritz value, "
      "and along two sensitivity axes. A scale-invariance check (`a,t,q` all x2 "
      "must leave `alpha` unchanged) plays the role of the geometric pre-check.")
    A("")
    A("## 3. Results")
    A("")
    A("Primary anchor (%s): `alpha_FE = %.8f` vs table `%.5f` -> **%s** (gate 3 %%)."
      % (pr["mesh"], pr["alpha_fe"], pr["alpha_ref"], pct(pr["relative_error"])))
    A("")
    A("| anchor | got | reference | rel. error | gate | verdict |")
    A("|---|---|---|---|---|---|")
    for a in d["anchors"]:
        A("| `%s` | %.8g | %.8g | %s | %.1f%% | %s |" % (
            a["anchor"], a["got"], a["reference"], pct(a["relative_error"], 2),
            100 * a["tolerance"], "PASS" if a["passed"] else "FAIL"))
    A("")
    A("Mesh convergence (clamped, a/t = 100):")
    A("")
    A("| nx | alpha | err vs table | err vs Ritz |")
    A("|---|---|---|---|")
    for r in conv:
        A("| %d | %.8f | %s | %s |" % (r["nx"], r["alpha"], pct(r["alpha"] / 0.00126 - 1),
                                       pct(r["alpha"] / ritz - 1)))
    A("")
    if rich.get("limit") is not None:
        A("Dyadic Richardson (nx = 8/16/32): order `%.2f`, `alpha_inf = %.8f` (%s vs "
          "table, %s vs Ritz)." % (rich["order"], rich["limit"],
                                   pct(rich["limit"] / 0.00126 - 1), pct(rich["limit"] / ritz - 1)))
    else:
        A("Richardson: %s." % rich.get("note"))
    A("")
    A("Sensitivity A -- slenderness (nx = 16):")
    A("")
    A("| a/t | alpha FE | alpha Ritz (Kirchhoff) | shear effect |")
    A("|---|---|---|---|")
    for r in thick:
        A("| %.0f | %.8f | %.8f | %s |" % (r["a_over_t"], r["alpha"],
                                           r["alpha_ritz_kirchhoff"],
                                           pct(r["shear_effect_vs_kirchhoff"], 2)))
    A("")
    A("Sensitivity B -- Poisson ratio (nx = 16, a/t = 100): "
      + "; ".join("nu=%.2f err %s" % (r["nu"], pct(r["alpha"] / r["alpha_ritz"] - 1)) for r in nu)
      + ".")
    A("")
    A("## 4. Figures")
    A("")
    A("Left: mesh convergence of `alpha` vs the Table 30 entry, the converged Ritz "
      "value and the +-3 % band. Middle: normalised deflection profile `w/w_max` "
      "along `y/a = 0.5`, FE vs Ritz. Right: slenderness sensitivity -- FE `alpha` "
      "relative to the Kirchhoff `alpha`, showing the Mindlin shear contribution "
      "switching on below `a/t = 50`.")
    A("")
    A("## 5. Caveats")
    A("")
    A("1. Three printed digits in the table impose a %.2f %% rounding floor; the "
      "FE-vs-Ritz comparison is reported next to it for that reason."
      % (100 * d["reference_rounding_band_relative"]))
    A("2. The clamped Ritz reference is capped at N <= 12 (12 tabulated "
      "clamped-beam roots); its measured truncation band is %.2f %% for the "
      "deflection and %.2f %% for the centre moment, which is why the moment anchor "
      "is FE-vs-table and the Ritz moment is never gated."
      % (100 * d["ritz_deflection_truncation_band"],
         100 * d["ritz_centre_moment_truncation_n10_to_n12"]))
    A("3. Only even element counts give a centre node (nx = 17 has none and raises).")
    A("4. Peak clamped-edge moments are out of scope (slow pointwise convergence in "
      "both methods).")
    A("5. Scope of the claim: element consistency for a smooth, uniformly loaded, "
      "clamped isotropic thin plate. Nothing about edge stress recovery, "
      "plasticity, stiffeners or shell curvature.")
    A("")
    A("## 6. Error verdict")
    A("")
    A("| quantity | FE | reference | rel. error | < 3 %% ? |")
    A("|---|---|---|---|---|")
    sec = anchor("secondary_fe_centre_moment_vs_table")
    A("| `alpha` 16x16, a/t = 100 | %.8f | %.5f (Table 30) | %s | %s |" % (
        pr["alpha_fe"], pr["alpha_ref"], pct(pr["relative_error"]),
        "yes" if pr["within_3pct"] else "NO"))
    A("| `|Mx|/(q a^2)` 16x16 | %.8f | %.5f (Table 30) | %s | %s |" % (
        sec["got"], 0.0231, pct(sec["relative_error"]), "yes" if sec["passed"] else "NO"))
    A("")
    A("**Is the key quantity within 3 %%? -- %s** (%s). All %d gates %s. "
      "Reproduce: `%s`." % (
          "yes" if pr["within_3pct"] else "NO", pct(pr["relative_error"]),
          len(d["anchors"]), "pass" if d["all_gates_passed"] else "did not all pass",
          d["repro_cmd"]))

    figures = []
    if FIG.exists():
        figures.append({
            "name": "clamped_plate_uniform.png",
            "caption": "Clamped square plate, uniform pressure: mesh convergence vs "
                       "Timoshenko Table 30, deflection profile vs Ritz, slenderness "
                       "sensitivity.",
            "png_base64": base64.b64encode(FIG.read_bytes()).decode(),
        })

    return {
        "title": "%s: clamped square plate under uniform pressure, alpha = "
                 "%.6f vs Timoshenko Table 30 0.00126 (%s)"
                 % (CASE, pr["alpha_fe"], pct(pr["relative_error"])),
        "summary": "TensorFEM Q4 Mindlin clamped square plate (a/t = 100) reproduces "
                   "the Timoshenko Table 30 centre-deflection coefficient 0.00126 to "
                   "%s (gate 3%%); %d/%d gates pass, dyadic Richardson limit %s."
                   % (pct(pr["relative_error"]),
                      sum(a["passed"] for a in d["anchors"]), len(d["anchors"]),
                      ("%.8f" % rich["limit"]) if rich.get("limit") else "n/a"),
        "scenario_key": "structural-plate-verification",
        "mission_id": MISSION_ID,
        "tags": ["TensorFEM", "plate", "clamped", "uniform-pressure", "Timoshenko",
                 "verification", "benchmark", CASE],
        "markdown": "\n".join(md),
        "figures": figures,
    }


def main() -> int:
    report = build()
    if "--dry-run" in sys.argv:
        out = Path("/tmp/report_dry.json")
        out.write_text(json.dumps(report, indent=1)[:200000])
        print("dry run: title=%s\nfigures=%d\nchars(md)=%d"
              % (report["title"], len(report["figures"]), len(report["markdown"])))
        return 0
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    login = urllib.request.Request(
        base + "/auth/login",
        data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                         "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
        headers={"Content-Type": "application/json"})
    token = json.load(urllib.request.urlopen(login, timeout=30))["access_token"]
    req = urllib.request.Request(
        base + "/tensor/reports", data=json.dumps(report).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token})
    resp = urllib.request.urlopen(req, timeout=120)
    print(resp.status, resp.read()[:500].decode("utf-8", "replace"))
    return 0


if __name__ == "__main__":
    sys.exit(main())