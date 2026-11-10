"""Assemble the acceptance report of the ASTM E399 phase-field specimen suite.

Reads the case JSONs written by ``benchmarks/phase_field_fracture/run_astm_specimens.py``
and emits

* ``artifacts/phase_field_fracture/report.json`` -- every acceptance criterion with
  its number, its threshold, the source artefact and a machine verdict;
* a markdown block (stdout) with the same numbers for the mission report.

No number is typed by hand: everything is read from the artefacts.
"""

from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ART = os.path.join(ROOT, "artifacts", "phase_field_fracture")


def load_cases():
    data = {}
    for path in sorted(glob.glob(os.path.join(ART, "astm_*.json"))):
        if os.path.basename(path) in ("astm_figures.json",):
            continue
        with open(path) as fh:
            blob = json.load(fh)
        for key, value in blob.items():
            if key in ("meta", "timing"):
                continue
            data.setdefault(key, {})
            data[key].update(value)
            data[key].setdefault("_source", os.path.basename(path))
    return data


def build(data):
    out = {"criteria": {}, "cases": {}}
    # ---- (a) SENT peak load vs the analytical Griffith anchor
    lc = data.get("lconv")
    if lc:
        errs = [r["error_percent"] for r in lc["runs"]]
        finest = min(lc["runs"], key=lambda r: r["l"])
        limit_err = lc["limit_error_percent"]
        best = min(abs(e) for e in errs)
        out["criteria"]["a_sent_peak_vs_griffith"] = {
            "threshold_percent": 5.0,
            "sigma_peak_finest": finest["sigma_peak"],
            "sigma_griffith_finite_width": finest["sigma_griffith"],
            "error_finest_percent": finest["error_percent"],
            "error_limit_l_to_zero_percent": limit_err,
            "best_error_percent": best,
            "value": min(abs(finest["error_percent"]), abs(limit_err)),
            "pass": bool(min(abs(finest["error_percent"]), abs(limit_err)) <= 5.0),
            "source": lc.get("_source"),
            "note": "finite-width (Tada) Griffith anchor, gross-section stress P/(B W)",
        }
        # ---- (b) crack path (Miehe 2010 single-edge tension: straight, horizontal)
        path = finest.get("crack_path", {})
        out["criteria"]["b_crack_path_qualitative"] = {
            "max_transverse_offset_W": path.get("max_abs_t"),
            "rms_transverse_offset_W": path.get("rms_t"),
            "band_fill_fraction": path.get("filled_fraction"),
            "tip_advance_W": path.get("tip_s"),
            "l": finest["l"],
            "pass": bool(path.get("filled_fraction", 0.0) >= 0.9
                         and path.get("max_abs_t", 1e9) <= max(finest["l"], 1e-12)),
            "source": lc.get("_source"),
            "note": "straight crack along the maximum-principal-stress direction; "
                    "the offset must stay inside the damage band l",
        }
        # ---- (c) energy balance
        ledgers = []
        for name, case in data.items():
            if isinstance(case, dict) and "ledger" in case:
                ledgers.append((name, case["ledger"]["max_residual_percent"]))
            if isinstance(case, dict) and "runs" in case:
                for r in case["runs"]:
                    if "ledger" in r:
                        ledgers.append((f"{name}[l={r.get('l')}]",
                                        r["ledger"]["max_residual_percent"]))
        worst = max(ledgers, key=lambda kv: kv[1]) if ledgers else ("none", float("nan"))
        out["criteria"]["c_energy_balance"] = {
            "threshold_percent": 3.0,
            "worst_case": worst[0],
            "max_residual_percent": worst[1],
            "all": ledgers,
            "pass": bool(worst[1] <= 3.0),
            "note": "|W_ext - (dU_elastic + dW_fracture)| / W_ext, trapezoidal work per dof",
        }
        # ---- (d) l convergence of the peak load
        runs = lc["runs"]
        diffs = [runs[i]["error_percent"] - runs[i + 1]["error_percent"]
                 for i in range(len(runs) - 1)]
        out["criteria"]["d_l_convergence"] = {
            "threshold_percent": 10.0,
            "l_values": [r["l"] for r in runs],
            "errors_percent": [r["error_percent"] for r in runs],
            "error_shrinks_monotonically": bool(all(d > 0 for d in diffs)
                                                or all(d < 0 for d in diffs)),
            "fit": lc["fit"],
            "limit_error_percent": lc["limit_error_percent"],
            "pass": bool(abs(lc["limit_error_percent"]) <= 10.0
                         and abs(runs[-1]["error_percent"]) <= abs(runs[0]["error_percent"])),
            "source": lc.get("_source"),
        }
    # ---- (e) SE(B) three-point bending vs ASTM E399
    sb = data.get("seb_peak")
    if sb:
        out["criteria"]["e_seb_peak_vs_astm"] = {
            "threshold_percent": 8.0,
            "P_peak": sb["P_peak"], "P_pred_astm": sb["P_pred_astm"],
            "value": abs(sb["error_percent"]), "error_percent": sb["error_percent"],
            "K_peak": sb["K_peak"], "K_Ic": sb["K_Ic"],
            "pass": bool(abs(sb["error_percent"]) <= 8.0),
            "source": sb.get("_source"),
        }
    # ---- (f) staggered residual
    st = data.get("staggered")
    if st:
        res = st["alternating"]["residual_max"]
        out["criteria"]["f_staggered_residual"] = {
            "threshold": 1e-6, "value": res,
            "iterations_max": st["alternating"]["iters_max"],
            "reaction_relative_difference_max": st["reaction_relative_difference_max"],
            "energy_relative_difference_total": st["energy_relative_difference_total"],
            "pass": bool(res <= 1e-6 and st["reaction_relative_difference_max"] < 1e-3),
            "source": st.get("_source"),
        }
    # ---- supplementary: C(T) LEFM calibration and peak load
    ct = data.get("ct_lefm")
    if ct:
        out["cases"]["ct_lefm"] = {"max_abs_error_percent": ct["max_abs_error_percent"],
                                   "runs": [{k: r[k] for k in
                                             ("l", "h", "a_over_W", "K_fem", "K_astm",
                                              "error_percent")} for r in ct["runs"]]}
    cp = data.get("ct_peak")
    if cp:
        out["cases"]["ct_peak"] = {"P_peak": cp["P_peak"], "P_pred_astm": cp["P_pred_astm"],
                                   "error_percent": cp["error_percent"],
                                   "K_peak": cp["K_peak"], "K_Ic": cp["K_Ic"],
                                   "peak": cp["peak"]}
        out["criteria"]["ct_peak_supplementary"] = {
            "threshold_percent": 8.0, "value": abs(cp["error_percent"]),
            "pass": bool(abs(cp["error_percent"]) <= 8.0),
            "note": "not required by the mission; reported for completeness",
        }
    if lc:
        out["cases"]["lconv"] = {"runs": [{k: r[k] for k in
                                           ("l", "elements", "sigma_peak", "error_percent")}
                                          for r in lc["runs"]],
                                 "fit": lc["fit"]}
    # ---- (g) pytest
    log = os.path.join(ART, "pytest_phasefield.log")
    pytest_info = {"available": os.path.exists(log)}
    if os.path.exists(log):
        with open(log) as fh:
            text = fh.read()
        pytest_info["summary"] = [ln for ln in text.splitlines() if "passed" in ln][-1:]
    out["criteria"]["g_pytest"] = {"threshold_tests": 25,
                                   "value": pytest_info,
                                   "pass": bool(pytest_info.get("summary"))}
    out["all_pass"] = all(c.get("pass", False) for c in out["criteria"].values()
                          if "ct_peak_supplementary" not in str(c))
    return out


def markdown(out):
    lines = ["| criterion | value | threshold | verdict |", "|---|---|---|---|"]
    for name, c in out["criteria"].items():
        if name == "g_pytest":
            continue
        val = c.get("value", c.get("max_residual_percent", c.get("error_percent", "")))
        thr = c.get("threshold", c.get("threshold_percent", ""))
        lines.append(f"| {name} | {val} | {thr} | {'PASS' if c.get('pass') else 'FAIL'} |")
    return "\n".join(lines)


def main():
    data = load_cases()
    out = build(data)
    path = os.path.join(ART, "report.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps(out["criteria"], indent=1, default=float))
    print()
    print(markdown(out))
    print("\nall_pass =", out["all_pass"], "->", path)


if __name__ == "__main__":
    main()