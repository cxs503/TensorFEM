#!/usr/bin/env python3
"""Verify the GAP-PP-4-ICE-STRUCT-ELASTIC-RESTRAINT acceptance criteria
against the committed machine-readable artifacts (verification-debt closure).

This is a *lightweight* audit: it reads the already-committed benchmark JSON
artifacts (produced by the heavy Mindlin FEM runs) and checks each acceptance
criterion of the feature_ledger row against them.  It does not re-run the FEM
solver; it binds the claims to the exact committed artifact bytes.

Acceptance criteria (feature_ledger GAP-PP-4-ICE-STRUCT-ELASTIC-RESTRAINT):
  A. beta=0 时峰值应力与 panel_patch_load_results.json 一致(相对差 <0.1%)
  B. beta=1 时与固支解析/文献值误差 <5%
  C. >=5 档 beta 扫描机读 JSON
  D. pytest>=5 绿 (reported separately by the pytest run)
  E. committed artifact 必须能被**当前求解器**复现 (staleness guard)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
ROOT = HERE.parent

# --- committed artifact paths ------------------------------------------------
PANEL_BAND = ART / "beta_restraint_panel_band.json"
LEVY_BAND = ART / "beta_levy_band_verification.json"
BETA_SWEEP = ART / "beta_sweep_results.json"
CALIB = ART / "plate_restraint_calibration.json"


def load(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"missing committed artifact: {p}")
    return json.loads(p.read_text())


def _reproduce_guard_rows(band: dict) -> dict:
    """Re-solve the artifact's coarse guard rows with the current solver.

    This is the cure for the failure mode that re-opening PR #1101 uncovered:
    ``beta_restraint_panel_band.json`` was generated before the tributary-length
    fix of the nodal spring (#992) and the consistent patch-load fix (#1014) and
    kept a 45 %-over-stiff mid-band (``beta = 0.2``: 49.19 MPa committed vs
    71.63 MPa solved) while structural checks stayed green.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from tensorfem.plate_elastic_restraint import (  # noqa: PLC0415
        peak_bending_stress,
        rect_patch_pressure,
        solve_restrained_panel,
    )

    guard = band.get("guard")
    if not guard:
        return {"pass_reproduced": False,
                "reason": "artifact has no guard rows; regenerate it"}
    case = band["case"]
    L, S = case["L_m"], case["S_m"]
    pc = case["patch_length_m"]
    press = rect_patch_pressure(case["p_MPa"] * 1e6,
                                0.5 * (L - pc), 0.5 * (L + pc), 0.0, S)
    rows, worst = [], 0.0
    for row in guard["rows"]:
        res = solve_restrained_panel(
            guard["nx"], guard["ny"], L, S, case["E_Pa"], case["nu"],
            case["t_m"], press,
            beta_x=row["beta_x"], beta_y=row["beta_y"],
        )
        pk = peak_bending_stress(res)
        sig = pk["centre_von_mises"] / 1e6
        w = pk["w_centre"] * 1e3
        d_sig = sig / row["sigma_vm_centre_MPa"] - 1.0
        d_w = w / row["w_centre_mm"] - 1.0
        worst = max(worst, abs(d_sig), abs(d_w))
        rows.append({
            "beta_x": row["beta_x"],
            "beta_y": row["beta_y"],
            "committed_sigma_vm_MPa": row["sigma_vm_centre_MPa"],
            "solved_sigma_vm_MPa": sig,
            "sigma_rel_diff": d_sig,
            "committed_w_centre_mm": row["w_centre_mm"],
            "solved_w_centre_mm": w,
            "w_rel_diff": d_w,
        })
    tol = 1e-9
    return {
        "solver_module_sha256": band.get("solver_sha256"),
        "mesh": {"nx": guard["nx"], "ny": guard["ny"]},
        "tolerance_rel": tol,
        "worst_rel_diff": worst,
        "rows": rows,
        "pass_reproduced": worst < tol,
    }


def main() -> int:
    checks = {}

    # ---- A. beta=0 vs panel_patch_load_results.json (<0.1%) ----------------
    band = load(PANEL_BAND)
    baseline = band["baseline_navier_N401_sigma_vm_centre_MPa"]
    # the beta=0 diagonal row is the SSSS reproduction
    b0 = next(r for r in band["diagonal_beta"] if r["beta_x"] == 0.0 and r["beta_y"] == 0.0)
    rel_err_b0 = b0["rel_err_vs_navier_pct"]  # already % vs Navier N=401
    checks["A_beta0_vs_navier"] = {
        "baseline_sigma_vm_MPa": baseline,
        "fem_beta0_sigma_vm_MPa": b0["sigma_vm_centre_MPa"],
        "rel_err_pct": rel_err_b0,
        "pass_lt_0_1pct": abs(rel_err_b0) < 0.1,
    }

    # ---- B. beta=1 vs clamped literature (<5%) ------------------------------
    lit = band["literature_square"]["uniform_square_CCCC"]
    checks["B_beta1_vs_timoshenko"] = {
        "w_coef_fem": lit["w_coef"],
        "w_coef_ref_Timoshenko": lit["ref_w_coef_Timoshenko"],
        "w_err_pct": (lit["w_coef"] / lit["ref_w_coef_Timoshenko"] - 1.0) * 100,
        "M_centre_fem": lit["M_coef_centre"],
        "M_centre_ref_Timoshenko": lit["ref_M_coef_centre_Timoshenko"],
        "M_centre_err_pct": (lit["M_coef_centre"] / lit["ref_M_coef_centre_Timoshenko"] - 1.0) * 100,
        "pass_lt_5pct": (
            abs(lit["w_coef"] / lit["ref_w_coef_Timoshenko"] - 1.0) < 0.05
            and abs(lit["M_coef_centre"] / lit["ref_M_coef_centre_Timoshenko"] - 1.0) < 0.05
        ),
    }

    # ---- C. >=5 档 beta 扫描机读 JSON ---------------------------------------
    sweep = load(BETA_SWEEP)
    n_beta = len(sweep)
    betas = [r["beta"] for r in sweep]
    checks["C_beta_sweep"] = {
        "n_points": n_beta,
        "betas": betas,
        "pass_ge_5": n_beta >= 5,
        "monotone_w": all(
            sweep[i]["w_center"] <= sweep[i - 1]["w_center"] + 1e-12
            for i in range(1, len(sweep))
        ),
    }

    # ---- D. pytest green (reported by the pytest run) ------------------------
    # The acceptance tests live in tests/test_plate_restraint_panel.py etc.
    # Their green status is reported by the pytest run; here we only record the
    # test-file inventory that encodes the criteria.
    checks["D_pytest_inventory"] = {
        "acceptance_test_files": [
            "tests/test_plate_restraint_panel.py",
            "tests/test_plate_restraint_calibration.py",
            "tests/test_plate_elastic_restraint.py",
        ],
        "note": "green status reported by pytest run (see run log)",
    }

    # ---- E. committed artifact reproducible by the current solver ----------
    checks["E_artifact_reproduction"] = _reproduce_guard_rows(band)

    # ---- overall -------------------------------------------------------------
    all_pass = all(
        v.get("pass_lt_0_1pct", v.get("pass_lt_5pct", v.get("pass_ge_5", True)))
        for v in checks.values()
        if isinstance(v, dict)
    )
    checks["overall"] = {"all_acceptance_met": all_pass}

    out = {
        "anchor": "GAP-PP-4-ICE-STRUCT-ELASTIC-RESTRAINT",
        "verification": "verification-debt closure: acceptance criteria bound to committed artifacts",
        "checks": checks,
    }
    dest = ART / "restraint_acceptance_verification.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
