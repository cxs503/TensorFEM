#!/usr/bin/env python
"""W23 composite-laminate bending verification: case matrix, errors, convergence.

This runner produces the three mandatory artifacts of the W23
``composite-laminate-verification`` extension (>= 3 layups x >= 3 thickness
ratios x >= 3 meshes):

    case_matrix.csv        one row per case: inputs, references, FE results, verdicts
    deflection_error.json  centre deflection of every case vs every reference
    grid_convergence.json  mesh sequence, observed order, GCI per case and path
    interlaminar_shear.json  tau_xz at the ply interfaces vs the 3-D reference

Everything is injected programmatically: no number is typed by hand, and the
provenance block records the git commit, the module sha256 and the case-matrix
sha256.

Models
------
* ``clt_series``   -- classical laminate theory, truncated Navier/Fourier-Ritz
                     series in a sine basis (exact thin-plate reference; carries
                     D16/D26 so it is valid for the +-45 and quasi-isotropic
                     layups as well).
* ``fsdt_navier``  -- first-order shear deformation theory, single-mode Navier
                     solution (exact only when D16 = D26 = 0, i.e. cross-ply).
* ``exact3d``      -- layered 3-D elasticity, Pagano propagator (exact; only for
                     specially orthotropic plies, i.e. cross-ply -- the code
                     refuses anything else).
* ``plate_fem``    -- Q4 Mindlin FSDT plate FE, full 3x3 D, MITC shear, meshes
                     n = 4, 8, 16, 32 per side.
* ``hex20_fem``    -- layered 3-D HEX20 FE, one orthotropic stiffness per ply,
                     meshes (nx, nz per ply) = (4,2), (6,3), (8,4).

Usage::

    PYTHONPATH=src python benchmarks/composite_laminate_navier/run_case_matrix.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from tensorfem.composite_laminate_bending import (  # noqa: E402
    PAGANO_MATERIAL, Laminate, LayeredExact3D, gci_study, layered_hex20_fem,
    navier_clt, navier_clt_series, navier_fsdt, plate_fem,
    richardson_extrapolate, symmetric_layup,
)

A = 1.0                 # plate side [m]
Q0 = 1.0                # peak sine pressure [Pa]
THICKNESS_RATIOS = (0.01, 0.1, 0.5)
PLATE_MESHES = (4, 8, 16, 32)
HEX_MESHES = ((4, 2), (6, 3), (8, 4))
THEORETICAL_ORDER = 2.0
GCI_LIMIT_PCT = 5.0
THIN_LIMIT_PCT = 3.0
THICK_LIMIT_PCT = 5.0
SHEAR_LIMIT_PCT = 10.0
CLT_MODES = 61          # coupling series converges like 1/modes; 61 gives <0.5% truncation

LAYUPS = {
    "cross_ply_0_90_s": [0, 90] * 2,                 # [0/90]s
    "angle_ply_pm45_s": symmetric_layup([45, -45]),  # [45/-45]s
    "quasi_iso_0_pm45_90_s": symmetric_layup([0, 45, -45, 90]),
}

SPECIAL_ORTHOTROPIC = {
    name: all(abs(math.remainder(float(t), 90.0)) < 1e-9 for t in angles)
    for name, angles in LAYUPS.items()
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance() -> dict:
    module = REPO / "src" / "tensorfem" / "composite_laminate_bending.py"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    except Exception:                                     # pragma: no cover
        commit = "unknown"
    return {
        "module": "src/tensorfem/composite_laminate_bending.py",
        "module_sha256": sha256_file(module),
        "runner_sha256": sha256_file(Path(__file__)),
        "git_commit": commit,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "material": {
            "E1": PAGANO_MATERIAL.E1, "E2": PAGANO_MATERIAL.E2,
            "E3": PAGANO_MATERIAL.E3, "nu12": PAGANO_MATERIAL.nu12,
            "nu13": PAGANO_MATERIAL.nu13, "nu23": PAGANO_MATERIAL.nu23,
            "G12": PAGANO_MATERIAL.G12, "G13": PAGANO_MATERIAL.G13,
            "G23": PAGANO_MATERIAL.G23,
        },
        "plate": {"a": A, "b": A, "q0": Q0},
        "load": "q = q0 sin(pi x / a) sin(pi y / b), SS1 (shear diaphragm) edges",
    }


def _json_safe(obj):
    """Recursively convert numpy scalars/arrays so ``json.dumps`` accepts them."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def case_id(layup: str, ratio: float) -> str:
    return f"{layup}__t{a_ratio_tag(ratio)}"


def a_ratio_tag(ratio: float) -> str:
    return f"{ratio:.2f}".replace(".", "p")


def run_case(layup_name: str, angles: list, ratio: float, verbose: bool = True) -> dict:
    h = ratio * A
    lam = Laminate(angles, h / len(angles), PAGANO_MATERIAL)
    A_abd, B_abd, D, As = lam.abd()
    # single-mode Navier is only defined for specially orthotropic laminates
    # (D16 = D26 = 0); for angle-ply / quasi-isotropic families the truncated
    # series below is the CLT reference instead.
    _dscale = max(abs(D[0, 0]), abs(D[1, 1]), 1e-300)
    if abs(D[0, 2]) <= 1e-9 * _dscale and abs(D[1, 2]) <= 1e-9 * _dscale:
        clt_single = navier_clt(lam, A, A, Q0)
    else:
        clt_single = None
    clt_series = navier_clt_series(lam, A, A, Q0, modes=CLT_MODES)
    clt_series_sens = [navier_clt_series(lam, A, A, Q0, modes=m)["W"]
                       for m in (CLT_MODES - 40, CLT_MODES - 20, CLT_MODES)]
    clt_truncation_pct = float(100.0 * abs(clt_series_sens[-1] - clt_series_sens[-2])
                               / clt_series_sens[-1])
    fsdt = None if (abs(D[0, 2]) > 0 or abs(D[1, 2]) > 0) else navier_fsdt(lam, A, A, Q0)
    d16_rel = float(abs(D[0, 2]) / D[0, 0])
    d26_rel = float(abs(D[1, 2]) / D[0, 0])

    exact3d = None
    if SPECIAL_ORTHOTROPIC[layup_name]:
        exact3d = LayeredExact3D(lam, A, A, Q0)

    plate = []
    for n in PLATE_MESHES:
        r = plate_fem(lam, A, Q0, n=n)
        plate.append({"n": int(n), "w": r["w_center"], "dofs": r["n_dofs"],
                      "residual": r["residual"]})

    hex3d = []
    for nx, nz_pp in HEX_MESHES:
        r = layered_hex20_fem(lam, a=A, q0=Q0, nx=nx, nz_per_ply=nz_pp)
        iface = {}
        for z in lam.z_interfaces[1:-1]:
            best = min(r["tau_xz_profile"], key=lambda item: abs(item[0] - z))
            iface[f"{z:.6f}"] = best[1]
        hex3d.append({"nx": int(nx), "nz_per_ply": int(nz_pp), "w": r["w_center"],
                      "dofs": r["n_dofs"], "residual": r["residual"],
                      "tau_xz_interface": iface,
                      "tau_xz_max_interior": max(
                          (abs(t), z) for z, t in r["tau_xz_profile"] if abs(z) < 0.49 * h)[0]})

    case = {
        "case": case_id(layup_name, ratio),
        "layup": layup_name,
        "angles_deg": [float(a) for a in angles],
        "n_plies": lam.n_plies,
        "ply_thickness": lam.ply_thickness,
        "thickness": h,
        "t_over_a": ratio,
        "a_over_h": 1.0 / ratio,
        "balanced": lam.is_balanced,
        "symmetric": lam.is_symmetric,
        "D11": float(D[0, 0]), "D22": float(D[1, 1]), "D12": float(D[0, 1]),
        "D66": float(D[2, 2]), "D16": float(D[0, 2]), "D26": float(D[1, 2]),
        "D16_over_D11": d16_rel, "D26_over_D11": d26_rel,
        "specially_orthotropic": d16_rel < 1e-9 and d26_rel < 1e-9,
        "A55": float(As[0, 0]), "A44": float(As[1, 1]), "A45": float(As[0, 1]),
        "reference": {
            "clt_series": clt_series["W"],
            "clt_series_modes": [CLT_MODES - 40, CLT_MODES - 20, CLT_MODES],
            "clt_series_sequence": clt_series_sens,
            "clt_series_truncation_pct": clt_truncation_pct,
            "clt_single_mode": None if clt_single is None else clt_single.W,
            "clt_single_mode_note": "single-mode Navier is undefined when D16/D26 != 0",
            "fsdt_navier": None if fsdt is None else fsdt.W,
            "exact3d": None if exact3d is None else exact3d.center_deflection,
            "exact3d_verification": None if exact3d is None else exact3d.verification(),
        },
        "plate_fem": plate,
        "hex20_fem": hex3d,
    }
    # primary reference for the error tables
    if exact3d is not None:
        case["primary_reference"] = "exact3d"
        ref = exact3d.center_deflection
    elif ratio <= 0.1:
        case["primary_reference"] = "clt_series"
        ref = clt_series["W"]
    else:
        rich = richardson_extrapolate([m["w"] for m in hex3d], ratio=1.5)
        case["primary_reference"] = "hex20_richardson"
        case["richardson"] = rich
        ref = rich["extrapolated"]
    case["w_reference"] = float(ref)
    case["w_plate_finest"] = plate[-1]["w"]
    case["w_hex_finest"] = hex3d[-1]["w"]
    case["err_plate_vs_ref_pct"] = float(100.0 * (plate[-1]["w"] / ref - 1.0))
    case["err_hex_vs_ref_pct"] = float(100.0 * (hex3d[-1]["w"] / ref - 1.0))
    case["err_plate_vs_clt_pct"] = float(100.0 * (plate[-1]["w"] / clt_series["W"] - 1.0))
    case["err_clt_vs_ref_pct"] = float(100.0 * (clt_series["W"] / ref - 1.0))
    case["shear_share_pct"] = case["err_clt_vs_ref_pct"]

    # grid convergence on both FE paths
    conv = {"plate_fem": gci_study([p["w"] for p in plate], ref,
                                   ratio=PLATE_MESHES[-1] / PLATE_MESHES[-2]),
            "hex20_fem": gci_study([p["w"] for p in hex3d], ref,
                                   ratio=HEX_MESHES[-1][0] / HEX_MESHES[-2][0])}
    case["grid_convergence"] = conv

    # interlaminar shear comparison
    if exact3d is not None:
        exact_iface = {f"{item['z']:.6f}": item["tau_xz"] for item in exact3d.interface_shear()}
        fe_iface = hex3d[-1]["tau_xz_interface"]
        keys = sorted(exact_iface, key=float)
        rel = {k: float(100.0 * (fe_iface[k] / exact_iface[k] - 1.0))
               if abs(exact_iface[k]) > 0 else None for k in keys}
        worst = max((abs(v) for v in rel.values() if v is not None), default=0.0)
        case["shear"] = {"reference": "exact3d", "exact": exact_iface,
                         "fem_finest": fe_iface, "rel_err_pct": rel,
                         "worst_abs_rel_pct": float(worst),
                         "verdict": "PASS" if worst < SHEAR_LIMIT_PCT else "FAIL"}
    else:
        exact_iface = None
        finest = hex3d[-1]["tau_xz_interface"]
        coarse = hex3d[0]["tau_xz_interface"]
        rel_change = {k: float(100.0 * (finest[k] / coarse[k] - 1.0))
                      if abs(coarse[k]) > 0 else None for k in finest}
        case["shear"] = {"reference": None,
                         "note": ("no closed-form 3-D anchor for angle-ply laminates "
                                  "(the Pagano state space is exact only for specially "
                                  "orthotropic plies); the 3-D FE sequence itself is "
                                  "reported, so this case cannot be scored against the "
                                  "10% criterion"),
                         "fem_coarse": coarse, "fem_finest": finest,
                         "change_coarse_to_finest_pct": rel_change,
                         "worst_abs_change_pct": float(max(
                             (abs(v) for v in rel_change.values() if v is not None), default=0.0)),
                         "verdict": "WITHHELD"}

    # criterion 2: thin vs thick thresholds
    limit = THIN_LIMIT_PCT if ratio <= 0.1 else THICK_LIMIT_PCT
    case["deflection_limit_pct"] = limit
    case["deflection_verdict_plate"] = ("PASS" if abs(case["err_plate_vs_ref_pct"]) < limit
                                        else "FAIL")
    case["deflection_verdict_hex"] = ("PASS" if abs(case["err_hex_vs_ref_pct"]) < limit
                                      else "FAIL")
    if verbose:
        print(f"[{case['case']}] ref={case['primary_reference']} w_ref={ref:.6e} "
              f"plate={plate[-1]['w']:.6e} ({case['err_plate_vs_ref_pct']:+.2f}%) "
              f"hex={hex3d[-1]['w']:.6e} ({case['err_hex_vs_ref_pct']:+.2f}%) "
              f"p_plate={conv['plate_fem']['p_observed']:.2f} "
              f"GCI={conv['plate_fem']['gci_fine_pct']:.2f}%", flush=True)
    return case


def _worker(args):
    name, angles, ratio = args
    return run_case(name, angles, ratio)


def main() -> int:
    global OUT
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--only", default=None, help="substring filter on case id")
    parser.add_argument("--out", default=None,
                        help="artifact directory (default: this benchmark directory)")
    args = parser.parse_args()
    if args.out:
        OUT = Path(args.out).resolve()
        OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    todo = [(name, angles, ratio) for name, angles in LAYUPS.items()
            for ratio in THICKNESS_RATIOS]
    if args.only:
        todo = [t for t in todo if args.only in case_id(t[0], t[2])]
    if args.jobs > 1:
        # spawn, not fork: this module imports torch/OpenMP, and a forked child
        # inherits locked BLAS/torch mutexes and then hangs in futex_wait, so the
        # pool never returns.  spawn re-imports in a clean interpreter.
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(args.jobs, len(todo))) as pool:
            cases = pool.map(_worker, todo)
    else:
        cases = [_worker(t) for t in todo]

    prov = provenance()

    # ── case_matrix.csv ────────────────────────────────────────────────────
    cols = ["case", "layup", "angles_deg", "n_plies", "ply_thickness", "thickness",
            "t_over_a", "a_over_h", "D11", "D22", "D12", "D66", "D16", "D26",
            "D16_over_D11", "special_orthotropic", "A55", "A44",
            "w_clt_series", "clt_truncation_pct", "w_clt_single", "w_fsdt_navier", "w_exact3d",
            "primary_reference", "w_reference", "w_plate_n32", "w_hex_finest",
            "err_plate_pct", "err_hex_pct", "err_clt_vs_ref_pct",
            "limit_pct", "verdict_plate", "verdict_hex",
            "p_plate_observed", "gci_plate_pct", "p_hex_observed", "gci_hex_pct"]
    with (OUT / "case_matrix.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for c in cases:
            w.writerow([
                c["case"], c["layup"], ";".join(f"{a:g}" for a in c["angles_deg"]),
                c["n_plies"], f"{c['ply_thickness']:.10e}", f"{c['thickness']:.10e}",
                c["t_over_a"], f"{c['a_over_h']:.1f}",
                f"{c['D11']:.10e}", f"{c['D22']:.10e}", f"{c['D12']:.10e}",
                f"{c['D66']:.10e}", f"{c['D16']:.10e}", f"{c['D26']:.10e}",
                f"{c['D16_over_D11']:.6e}", c["specially_orthotropic"],
                f"{c['A55']:.10e}", f"{c['A44']:.10e}",
                f"{c['reference']['clt_series']:.10e}",
                f"{c['reference']['clt_series_truncation_pct']:.4f}",
                "" if c["reference"]["clt_single_mode"] is None
                else f"{c['reference']['clt_single_mode']:.10e}",
                ("" if c["reference"]["fsdt_navier"] is None else f"{c['reference']['fsdt_navier']:.10e}"),
                "" if c["reference"]["exact3d"] is None else f"{c['reference']['exact3d']:.10e}",
                c["primary_reference"], f"{c['w_reference']:.10e}",
                f"{c['w_plate_finest']:.10e}", f"{c['w_hex_finest']:.10e}",
                f"{c['err_plate_vs_ref_pct']:+.4f}", f"{c['err_hex_vs_ref_pct']:+.4f}",
                f"{c['err_clt_vs_ref_pct']:+.4f}", c["deflection_limit_pct"],
                c["deflection_verdict_plate"], c["deflection_verdict_hex"],
                f"{c['grid_convergence']['plate_fem']['p_observed']:.3f}",
                f"{c['grid_convergence']['plate_fem']['gci_fine_pct']:.3f}",
                f"{c['grid_convergence']['hex20_fem']['p_observed']:.3f}",
                f"{c['grid_convergence']['hex20_fem']['gci_fine_pct']:.3f}",
            ])

    # ── deflection_error.json ──────────────────────────────────────────────
    deflection = {
        "task": "W23 composite laminate bending, extension (multi-case ensemble)",
        "provenance": prov,
        "criteria": {
            "C1_all_cases_reported": "every case reports its deflection deviation (no cherry-picking)",
            "C2_thin_lt3_thick_lt5_pct": {"thin_t_over_a_le_0.1": THIN_LIMIT_PCT,
                                          "thick_t_over_a_0.5": THICK_LIMIT_PCT},
            "C3_gci_lt5_pct_and_order_vs_theory": {"gci_limit_pct": GCI_LIMIT_PCT,
                                                   "theoretical_order": THEORETICAL_ORDER},
            "C4_interlaminar_shear_lt10_pct": SHEAR_LIMIT_PCT,
        },
        "cases": [],
    }
    for c in cases:
        deflection["cases"].append({
            "case": c["case"], "layup": c["layup"], "t_over_a": c["t_over_a"],
            "primary_reference": c["primary_reference"],
            "w_reference": c["w_reference"],
            "references": c["reference"],
            "plate_fem": c["plate_fem"], "hex20_fem": c["hex20_fem"],
            "err_plate_vs_ref_pct": c["err_plate_vs_ref_pct"],
            "err_hex_vs_ref_pct": c["err_hex_vs_ref_pct"],
            "err_plate_vs_clt_series_pct": c["err_plate_vs_clt_pct"],
            "err_clt_series_vs_ref_pct": c["err_clt_vs_ref_pct"],
            "shear_share_pct": c["shear_share_pct"],
            "limit_pct": c["deflection_limit_pct"],
            "verdict_plate": c["deflection_verdict_plate"],
            "verdict_hex": c["deflection_verdict_hex"],
        })
    (OUT / "deflection_error.json").write_text(json.dumps(_json_safe(deflection), indent=2))

    # ── grid_convergence.json ──────────────────────────────────────────────
    grid = {"provenance": prov,
            "theoretical_order": THEORETICAL_ORDER,
            "gci_limit_pct": GCI_LIMIT_PCT,
            "gci_formula": "GCI = Fs |e_fine| / (r^p - 1), Fs = 1.25, r = 2, "
                           "p = observed order from the three-level errors",
            "cases": []}
    for c in cases:
        entry = {"case": c["case"], "layup": c["layup"], "t_over_a": c["t_over_a"],
                 "reference": c["w_reference"],
                 "reference_kind": c["primary_reference"],
                 "paths": {}}
        for path in ("plate_fem", "hex20_fem"):
            g = c["grid_convergence"][path]
            mesh = ([p["n"] for p in c["plate_fem"]] if path == "plate_fem"
                    else [f"{m['nx']}x{m['nz_per_ply']}" for m in c["hex20_fem"]])
            w = ([p["w"] for p in c["plate_fem"]] if path == "plate_fem"
                 else [m["w"] for m in c["hex20_fem"]])
            entry["paths"][path] = {
                "mesh": mesh, "deflection": w,
                "rel_err_pct": g["errors"],
                "p_observed": g["p_observed"],
                "p_theory": THEORETICAL_ORDER,
                "order_gap_pct": float(100.0 * (g["p_observed"] / THEORETICAL_ORDER - 1.0)),
                "gci_fine_pct": g["gci_fine_pct"],
                "gci_medium_pct": g["gci_medium_pct"],
                "gci_verdict": "PASS" if g["gci_fine_pct"] < GCI_LIMIT_PCT else "FAIL",
                "order_verdict": ("PASS" if abs(g["p_observed"] - THEORETICAL_ORDER)
                                  <= 0.5 * THEORETICAL_ORDER else "FAIL"),
            }
        grid["cases"].append(entry)
    (OUT / "grid_convergence.json").write_text(json.dumps(_json_safe(grid), indent=2))

    # ── interlaminar_shear.json ────────────────────────────────────────────
    shear = {"provenance": prov, "criterion_pct": SHEAR_LIMIT_PCT,
             "evaluation_point": "tau_xz at the ply interfaces, peak of the "
                                 "cos(pi x/a) sin(pi y/b) field (edge x = 0, y = b/2)",
             "cases": [{"case": c["case"], "layup": c["layup"], "t_over_a": c["t_over_a"],
                        "interfaces_z": sorted(float(k) for k in
                                               (c["shear"].get("exact")
                                                or c["shear"].get("fem_finest"))),
                        **c["shear"]} for c in cases]}
    (OUT / "interlaminar_shear.json").write_text(json.dumps(_json_safe(shear), indent=2))

    print(f"done in {time.time() - t_start:.1f}s, artifacts in {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())