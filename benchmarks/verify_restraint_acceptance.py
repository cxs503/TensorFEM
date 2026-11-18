#!/usr/bin/env python3
"""Verify GAP-PP-4-ICE-STRUCT-ELASTIC-RESTRAINT acceptance criteria.

Reads committed benchmark artifacts and checks:
  A. beta=0 centre von-Mises stress vs Navier N=401 baseline (<0.1%)
  B. beta=1 deflection & moment vs Timoshenko table (<5%)
  C. beta sweep: ≥5 points, monotonic deflection
  D. pytest inventory: required test files exist
  E. artifact reproduction: re-solve committed cases and compare

Usage:
    python benchmarks/verify_restraint_acceptance.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(name: str) -> dict:
    p = ART / name
    if not p.exists():
        print(f"FAIL: {p} not found")
        return {}
    return json.loads(p.read_text())


def check(name: str, cond: bool, detail: str = "") -> bool:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    return cond


# ---------------------------------------------------------------------------
# Check A: beta=0 vs Navier baseline
# ---------------------------------------------------------------------------

def check_A(verify: dict) -> bool:
    print("\n[A] beta=0 centre von-Mises stress vs Navier N=401 baseline")
    a = verify.get("checks", {}).get("A_beta0_vs_navier", {})
    if not a:
        return check("A", False, "missing A_beta0_vs_navier in artifact")

    fem = a["fem_beta0_sigma_vm_MPa"]
    base = a["baseline_sigma_vm_MPa"]
    err = a.get("rel_err_pct", abs(fem - base) / base * 100)

    ok = check("A: rel_err < 0.1%", abs(err) < 0.1,
               f"fem={fem:.4f} MPa, base={base:.4f} MPa, err={err:+.4f}%")
    return ok


# ---------------------------------------------------------------------------
# Check B: beta=1 vs Timoshenko
# ---------------------------------------------------------------------------

def check_B(verify: dict) -> bool:
    print("\n[B] beta=1 deflection & moment vs Timoshenko table")
    b = verify.get("checks", {}).get("B_beta1_vs_timoshenko", {})
    if not b:
        return check("B", False, "missing B_beta1_vs_timoshenko in artifact")

    w_err = b.get("w_err_pct", 0)
    m_err = b.get("M_centre_err_pct", 0)

    ok_w = check("B: w_err < 5%", abs(w_err) < 5,
                 f"w_err={w_err:+.2f}%")
    ok_m = check("B: M_centre_err < 5%", abs(m_err) < 5,
                 f"M_centre_err={m_err:+.2f}%")
    return ok_w and ok_m


# ---------------------------------------------------------------------------
# Check C: beta sweep
# ---------------------------------------------------------------------------

def check_C(verify: dict) -> bool:
    print("\n[C] beta sweep: ≥5 points, monotonic deflection")
    c = verify.get("checks", {}).get("C_beta_sweep", {})
    if not c:
        return check("C", False, "missing C_beta_sweep in artifact")

    n = c.get("n_points", 0)
    betas = c.get("betas", [])
    mono = c.get("monotone_w", False)

    ok_n = check("C: n_points >= 5", n >= 5, f"n={n}")
    ok_m = check("C: monotone deflection", mono, "")
    return ok_n and ok_m


# ---------------------------------------------------------------------------
# Check D: pytest inventory
# ---------------------------------------------------------------------------

def check_D(verify: dict) -> bool:
    print("\n[D] pytest inventory: required test files exist")
    d = verify.get("checks", {}).get("D_pytest_inventory", {})
    if not d:
        return check("D", False, "missing D_pytest_inventory in artifact")

    files = d.get("acceptance_test_files", [])
    all_exist = True
    for f in files:
        fp = Path(f)
        if fp.exists():
            print(f"  [OK]   {f}")
        else:
            print(f"  [MISS] {f}")
            all_exist = False

    return check("D: all test files present", all_exist)


# ---------------------------------------------------------------------------
# Check E: artifact reproduction (re-solve committed cases)
# ---------------------------------------------------------------------------

def check_E(verify: dict) -> bool:
    print("\n[E] artifact reproduction: re-solve committed cases")
    e = verify.get("checks", {}).get("E_artifact_reproduction", {})
    if not e:
        return check("E", False, "missing E_artifact_reproduction in artifact")

    rows = e.get("rows", [])
    if not rows:
        return check("E", False, "no rows in E_artifact_reproduction")

    # Import solver
    try:
        from tensorfem.plate_elastic_restraint import (
            solve_restrained_panel,
            peak_bending_stress,
            rect_patch_pressure,
        )
    except ImportError:
        return check("E", False, "tensorfem.plate_elastic_restraint not importable")

    # Parameters from the paper case (0.8 m × 0.7 m × 22 mm, E=206 GPa, 1 MPa patch)
    L, S, T = 0.8, 0.7, 0.022
    E, NU = 206.0e9, 0.3
    P_REF = 1.0e6
    PATCH_C = 0.1 * L
    x1, x2 = 0.5 * (L - PATCH_C), 0.5 * (L + PATCH_C)
    press = rect_patch_pressure(P_REF, x1, x2, 0.0, S)
    nx = 32
    ny = 28

    all_pass = True
    for row in rows:
        bx = row["beta_x"]
        by = row["beta_y"]
        committed_w = row["committed_w_centre_mm"]
        committed_s = row["committed_sigma_vm_MPa"]

        try:
            result = solve_restrained_panel(
                nx=nx, ny=ny,
                side_x=L, side_y=S,
                young=E, poisson=NU, thickness=T,
                pressure=press,
                beta_x=bx, beta_y=by,
                winkler_k=0.0,
            )
            pk = peak_bending_stress(result)
            w_fem = pk["w_centre"] * 1e3  # m -> mm
            s_fem = pk["centre_von_mises"] / 1e6  # Pa -> MPa

            w_comm = row["committed_w_centre_mm"]
            s_comm = row["committed_sigma_vm_MPa"]

            w_diff = abs(w_fem - w_comm) / w_comm * 100 if w_comm else 0
            s_diff = abs(s_fem - s_comm) / s_comm * 100 if s_comm else 0

            ok_w = check(f"E: beta=({bx},{by}) w rel_diff < 0.1%", w_diff < 0.1,
                         f"w_fem={w_fem:.6f}, w_comm={w_comm:.6f}, diff={w_diff:.4f}%")
            ok_s = check(f"E: beta=({bx},{by}) sigma rel_diff < 0.1%", s_diff < 0.1,
                         f"s_fem={s_fem:.6f}, s_comm={s_comm:.6f}, diff={s_diff:.4f}%")

            if not (ok_w and ok_s):
                all_pass = False
        except Exception as ex:
            print(f"  [FAIL] E: beta=({bx},{by}) raised {ex}")
            all_pass = False

    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("GAP-PP-4-ICE-STRUCT-ELASTIC-RESTRAINT: Acceptance Verification")
    print("=" * 70)

    verify = load_json("restraint_acceptance_verification.json")
    if not verify:
        print("FATAL: cannot load restraint_acceptance_verification.json")
        return 1

    results = {
        "A": check_A(verify),
        "B": check_B(verify),
        "C": check_C(verify),
        "D": check_D(verify),
        "E": check_E(verify),
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for k, v in results.items():
        status = "PASS" if v else "FAIL"
        print(f"  [{status}] Check {k}")
        if not v:
            all_pass = False

    print("=" * 70)
    if all_pass:
        print("ALL CHECKS PASSED")
        return 0
    else:
        print("SOME CHECKS FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
