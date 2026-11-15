#!/usr/bin/env python3
"""Independent re-verification of mission m22156 (anchor GAP-ICE-GATE-POWER).

The audited deliverable is the pair added by commit 7b947ff3 in the TensorFEM
repository:
  * tests/test_hydro_ice_anchor.py  (242 lines, 17 pytest items)
  * docs/GAP-ICE-GATE-POWER-hydro-anchor.md

Nothing in this script imports the mission's work tree; it only imports the
shipped library module ``tensorfem.hydro_added_mass`` (identical on main and in
the m22156 work tree, see audit note) and re-derives every number the mission
reports.  Output is dumped to docs/board_audit/m22176/verify_output.json so the
verdict row can point at program-injected numbers instead of transcribed ones.

Run:  OMP_NUM_THREADS=6 PYTHONPATH=src python docs/board_audit/m22176/verify_ice_gate_power.py
"""
from __future__ import annotations

import importlib.util
import json
import math
import time
from pathlib import Path

import torch

from tensorfem.hydro_added_mass import (
    dipole_added_mass_matrix,
    disc_panels,
    piston_radiation_mass,
)

HERE = Path(__file__).resolve().parent
TEST_FILE = Path("/root/tp-agent/work/m22156/tests/test_hydro_ice_anchor.py")

# Verified independent 2D heave added-mass anchor, quoted from board report
# tensor_reports id=19234 (mission 21257), section Q5: two mutually independent
# potential-flow routes (BEM mirror-Green single layer + FD 5-point Laplace)
# agree to 3.8 % and self-prove on the deep circle rho*pi*R^2 to within 2.9 %.
BEM_2D_A33_KG_PER_M = 228.6e3          # [kg/m]  independent 2D anchor
DECLARED_INF_OVER_L = 86.1e3           # [kg/m]  declared A_inf / L in the chain


def load_mission_test_module():
    """Import the mission's test module to get its 'reference' implementation."""
    spec = importlib.util.spec_from_file_location("mission_hydro_test", TEST_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def piston_convergence() -> dict:
    """Re-derive the only genuine numerical check of the deliverable."""
    analytic = piston_radiation_mass(1.0, 1025.0)
    rows = []
    for n_cells in (16, 64, 256):
        t0 = time.perf_counter()
        centres, areas = disc_panels(1.0, n_cells)
        matrix = dipole_added_mass_matrix(centres, areas, 1025.0)
        dt = time.perf_counter() - t0
        got = float(matrix.sum())
        rows.append(
            {
                "n_cells": n_cells,
                "panels": int(centres.shape[0]),
                "matrix_dof": int(matrix.shape[0]),
                "dense_bytes": int(matrix.numel() * 8),
                "seconds": round(dt, 3),
                "sum_kg": got,
                "rel_error": abs(got - analytic) / analytic,
            }
        )
        del matrix, centres, areas
    # observed order between consecutive levels
    for prev, cur in zip(rows, rows[1:]):
        cur["observed_order"] = math.log(prev["rel_error"] / cur["rel_error"]) / math.log(
            cur["panels"] / prev["panels"]
        )
    # cost projection for the shipped n_cells=512 gate test (never actually run
    # by the mission: its acceptance command deselected both piston tests)
    ratio = rows[-1]["panels"] / (0.7853981633974483 * 256 ** 2)
    panels_512 = int(round(0.7853981633974483 * 512 ** 2))
    return {
        "analytic_kg": analytic,
        "levels": rows,
        "n_cells_512_projection": {
            "panels_est": panels_512,
            "dense_bytes_est": panels_512 * panels_512 * 8,
            "dense_GiB_est": round(panels_512 * panels_512 * 8 / 2 ** 30, 1),
            "panel_growth_sanity": round(ratio, 3),
            "note": "dense (N,N) float64 operator: 512 cells needs ~3.4e11 B >> node RAM headroom; "
            "the doc's '~2 s at N=512' is impossible",
        },
        "gate_3pct_already_met_at": [r["n_cells"] for r in rows if r["rel_error"] < 0.03],
    }


def barge_reference(mission_mod) -> dict:
    """Re-derive the mission's barge 'strip theory' reference and cross-check it."""
    out = {}
    cases = {
        "mission_geometry_L100_B20_T8": (100.0, 20.0, 8.0),
        "chain_geometry_L100_B20_T6": (100.0, 20.0, 6.0),  # report #19234: T=6, Cb=0.70
        "wide_limit_BT10": (100.0, 20.0, 2.0),
        "narrow_limit_BT0p5": (100.0, 4.0, 8.0),
    }
    for name, (L, B, T) in cases.items():
        r = mission_mod.heave_added_mass_rectangular_barge(L, B, T)
        displaced = 1025.0 * L * B * T
        out[name] = {
            "C_a": r["C_a"],
            "f_3d": r["f_3d"],
            "A33_2D_kg_per_m": r["A33_2D"],
            "A33_3D_kg": r["A33_3D"],
            "A33_over_displaced": r["A33_3D"] / displaced,
            "method_label": r["method"],
        }
    # apples-to-apples: 2D quantity vs the verified BEM anchor for the ship
    # section actually used by the chain (L=100, B=20, T=6)
    got_2d = out["chain_geometry_L100_B20_T6"]["A33_2D_kg_per_m"]
    out["vs_verified_bem_2d"] = {
        "mission_ref_2d_kg_per_m": got_2d,
        "verified_bem_2d_kg_per_m": BEM_2D_A33_KG_PER_M,
        "rel_error_vs_bem": (got_2d - BEM_2D_A33_KG_PER_M) / BEM_2D_A33_KG_PER_M,
        "declared_chain_kg_per_m": DECLARED_INF_OVER_L,
        "declared_vs_bem_rel_error": (DECLARED_INF_OVER_L - BEM_2D_A33_KG_PER_M) / BEM_2D_A33_KG_PER_M,
        "tolerance": 0.15,
        "verdict": "mission reference itself misses the 15% band against the verified BEM anchor",
    }
    return out


def tautology_probe(mission_mod) -> dict:
    """Show that the mission's accept/reject tests carry no detection power.

    Both 'verification' tests synthesise the declared value from the very
    reference they then compare against, so the pass/fail outcome is fixed by
    construction and independent of any real ship data.
    """
    ref = mission_mod.heave_added_mass_rectangular_barge(100.0, 20.0, 8.0)["A33_3D"]
    results = {}
    for scale in (0.0, 1.10, 1.5, 2.65, 100.0):
        declared = ref * scale
        results[f"declared_ref_times_{scale}"] = {
            "rel_error": abs(declared - ref) / ref,
            "passes_15pct_gate": abs(declared - ref) / ref < 0.15,
        }
    # what the real chain declares, in the same 3D kg basis the mission uses
    real_declared = DECLARED_INF_OVER_L * 100.0  # kg/m * L
    return {
        "synthetic_declared_scan": results,
        "real_chain_declaration_basis": {
            "declared_A_inf_kg": real_declared,
            "mission_reference_kg": ref,
            "rel_error": (real_declared - ref) / ref,
            "sign": "declared is BELOW the mission reference",
            "report_19234_direction": "declared 86.1 t/m vs independent 228.6 t/m -> declared "
            "also BELOW (+165.5% deviation) but the mission doc asserts the opposite sign "
            "(declared 9.5e6 > ref 3.6e6)",
        },
    }


def doc_vs_code(mission_mod) -> dict:
    """Compare the numbers printed in the mission doc with the shipped code."""
    r = mission_mod.heave_added_mass_rectangular_barge(100.0, 20.0, 8.0)
    doc = {"C_a": 1.047, "f_3d": 0.833, "A33_3D_kg": 3.6e6, "declared_kg": 9.5e6}
    return {
        "doc": doc,
        "code": {
            "C_a": r["C_a"],
            "f_3d": r["f_3d"],
            "A33_3D_kg": r["A33_3D"],
        },
        "rel_diff": {
            "C_a": (doc["C_a"] - r["C_a"]) / r["C_a"],
            "f_3d": (doc["f_3d"] - r["f_3d"]) / r["f_3d"],
            "A33_3D_kg": (doc["A33_3D_kg"] - r["A33_3D"]) / r["A33_3D"],
        },
        "declared_in_doc_vs_chain": {
            "doc_declared_kg": doc["declared_kg"],
            "chain_declared_kg": DECLARED_INF_OVER_L * 100.0,
            "rel_diff": (doc["declared_kg"] - DECLARED_INF_OVER_L * 100.0) / (DECLARED_INF_OVER_L * 100.0),
        },
    }


def main() -> int:
    torch.set_num_threads(6)
    mission_mod = load_mission_test_module()
    payload = {
        "audited_commit": "7b947ff3da0b5461651d4c156eb5c6db83841791",
        "audited_repo": "TensorFEM",
        "piston": piston_convergence(),
        "barge": barge_reference(mission_mod),
        "tautology_probe": tautology_probe(mission_mod),
        "doc_vs_code": doc_vs_code(mission_mod),
    }
    (HERE / "verify_output.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())