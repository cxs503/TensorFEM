"""Cross-validation: Allen-Cahn phase-field LBM on D3Q15 vs D3Q19.

This script validates the TensorFEM Allen-Cahn LBM solver against two
analytical benchmarks, following the approach used in lssfau/walberla's
phase_field_allen_cahn showcase:

1. **Diffusion decay**: A cosine mode decays exponentially; the fitted
   diffusivity must match D = cs²(τ − ½).

2. **Shrinking sphere**: A spherical droplet shrinks under mean-curvature
   flow; the rate dR²/dt must match −4M.

3. **D3Q15 vs D3Q19 comparison**: D3Q19 (rank-4 isotropy) should be more
   accurate than D3Q15 (rank-3) for the shrinking sphere.

Reference: lssfau/walberla, apps/showcases/phase_field_allen_cahn
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfem.lbm_allen_cahn_3d import (  # noqa: E402
    run_diffusion_decay,
    run_shrinking_sphere,
    compare_d3q15_d3q19,
)

ASSET_DIR = Path(__file__).resolve().parents[1] / "docs" / "assets" / "lbm_allen_cahn_d3q15_d3q19_xval"
ASSET_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize(obj):
    """Recursively convert numpy types to Python native types for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float64, np.float32, np.floating)):
        return float(obj)
    if isinstance(obj, (np.int64, np.int32, np.integer)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def main() -> None:
    print("=" * 70)
    print("XVAL: Allen-Cahn Phase-Field LBM — D3Q15 / D3Q19")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Diffusion decay
    # ------------------------------------------------------------------
    print("\n--- Diffusion Decay Test ---")
    diff_results = {}
    for lattice in ("D3Q15", "D3Q19"):
        r = run_diffusion_decay(lattice, n=32, tau=1.0, amplitude=0.1,
                                mode=1, steps=200, probe_every=10)
        diff_results[lattice] = r
        print(f"  {lattice}: D_theory={r['D_theory']:.6f}, "
              f"D_measured={r['D_measured']:.6f}, "
              f"rel_err={r['D_rel_err']:.4%}")

    # ------------------------------------------------------------------
    # 2. Shrinking sphere
    # ------------------------------------------------------------------
    print("\n--- Shrinking Sphere Test ---")
    sphere_results = {}
    for lattice in ("D3Q15", "D3Q19"):
        r = run_shrinking_sphere(lattice, n=40, tau=1.0, epsilon=3.0,
                                 radius0=12.0, steps=150, report_every=30)
        sphere_results[lattice] = r
        print(f"  {lattice}: dR²/dt_measured={r['dR2_dt_measured']:.4f}, "
              f"dR²/dt_analytical={r['dR2_dt_analytical']:.4f}, "
              f"rel_err={r['dR2_dt_rel_err']:.4%}")
        print(f"    R_eff: {' -> '.join(f'{rv:.2f}' for rv in r['radii'])}")

    # ------------------------------------------------------------------
    # 3. D3Q15 vs D3Q19 comparison
    # ------------------------------------------------------------------
    print("\n--- D3Q15 vs D3Q19 Comparison ---")
    cmp = compare_d3q15_d3q19(n=40, tau=1.0, epsilon=3.0,
                               radius0=12.0, steps=150, report_every=30)
    print(f"  D3Q15 dR²/dt rel_err: {cmp['d3q15_dR2_dt_rel_err']:.4%}")
    print(f"  D3Q19 dR²/dt rel_err: {cmp['d3q19_dR2_dt_rel_err']:.4%}")
    print(f"  D3Q19 more accurate: {cmp['d3q19_more_accurate']}")

    # ------------------------------------------------------------------
    # 4. Resolution study (D3Q19)
    # ------------------------------------------------------------------
    print("\n--- Resolution Study (D3Q19) ---")
    res_study = []
    for n in (24, 32, 40, 48):
        r0 = min(n * 0.3, n / 2.0 - 2 * 3.0)  # keep sphere away from boundary
        if r0 < 5:
            continue
        r = run_shrinking_sphere("D3Q19", n=n, tau=1.0, epsilon=3.0,
                                 radius0=r0, steps=150, report_every=30)
        res_study.append({"n": n, "radius0": r0, "dR2_dt_rel_err": r["dR2_dt_rel_err"]})
        print(f"  n={n:3d}, R0={r0:.1f}: dR²/dt rel_err={r['dR2_dt_rel_err']:.4%}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = _sanitize({
        "diffusion_decay": diff_results,
        "shrinking_sphere": sphere_results,
        "comparison": cmp,
        "resolution_study": res_study,
    })

    out_path = ASSET_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    d15 = sphere_results["D3Q15"]["dR2_dt_rel_err"]
    d19 = sphere_results["D3Q19"]["dR2_dt_rel_err"]
    diff15 = diff_results["D3Q15"]["D_rel_err"]
    diff19 = diff_results["D3Q19"]["D_rel_err"]

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  Diffusion decay D error:  D3Q15={diff15:.4%}, D3Q19={diff19:.4%}")
    print(f"  Shrinking sphere dR²/dt error: D3Q15={d15:.4%}, D3Q19={d19:.4%}")
    all_pass = diff15 < 0.15 and diff19 < 0.15 and d15 < 0.15 and d19 < 0.15
    print(f"  All errors < 15%: {'PASS' if all_pass else 'FAIL'}")
    print(f"  D3Q19 more accurate than D3Q15: {'PASS' if cmp['d3q19_more_accurate'] else 'FAIL'}")


if __name__ == "__main__":
    main()
