"""Lateral-surface peak shear verification for the Saint-Venant torsion benchmark.

Regenerates the evidence table in ``docs/torsion-boundary-stress.md`` and the two
figures under ``docs/assets/torsion_boundary_stress/``:

* ``docs/assets/torsion_boundary_stress/boundary_peak_evidence.json``
* ``docs/assets/torsion_boundary_stress/error_vs_dofs.png``
* ``docs/assets/torsion_boundary_stress/estimator_bars.png``

Run with::

    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python benchmarks/torsion_boundary_stress.py

The reference is the closed-form Saint-Venant series evaluated *on* the lateral
surface, cross-checked against the interior grid scan ``torsion_bar.max_shear_stress``
(which is a strict lower bound because it cannot sample the boundary).  Errors are
reported against the surface value computed with the finite-element twist rate, so
the table isolates stress recovery from the independently reported twist-rate error.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.torsion_bar import (max_shear_stress, rectangular_torsion_constant,  # noqa: E402
                                   torsion_shear_stress)
from tensorfem.torsion_boundary import torsion_boundary_peak  # noqa: E402

LADDER = (("hex8", 2), ("hex8", 4), ("hex8", 6), ("hex8", 8),
          ("hex20", 2), ("hex20", 3), ("hex20", 4))
OUT = ROOT / "docs" / "assets" / "torsion_boundary_stress"
SIDE, YOUNG, POISSON, TORQUE = 1.0, 1.0e9, 0.3, 1.0


def reference_cross_check(terms: int = 401) -> dict:
    """Surface series value at the exact twist rate versus the interior grid scan."""
    j = rectangular_torsion_constant(SIDE, SIDE)
    shear = YOUNG / (2.0 * (1.0 + POISSON))
    theta = TORQUE / (shear * j)
    tx, ty = torsion_shear_stress(torch.tensor([SIDE / 2.0]), torch.tensor([0.0]),
                                  SIDE, SIDE, twist_rate=theta, shear_modulus=shear,
                                  terms=terms)
    boundary = float(torch.sqrt(tx ** 2 + ty ** 2))
    scan, location = max_shear_stress(SIDE, SIDE, torque=TORQUE, terms=terms)
    return {
        "torsion_constant": j,
        "exact_twist_rate": theta,
        "boundary_series_peak": boundary,
        "interior_scan_peak": scan,
        "interior_scan_location": list(location),
        "scan_deficit": scan / boundary - 1.0,
        "terms": terms,
    }


def run_ladder() -> list[dict]:
    rows = []
    for element, segments in LADDER:
        result = torsion_boundary_peak(element=element, segments=segments, side=SIDE,
                                       young=YOUNG, poisson=POISSON, torque=TORQUE)
        rows.append({
            "element": result.element,
            "segments": result.segments,
            "n_dofs": result.n_dofs,
            "n_face_elements": result.n_face_elements,
            "twist_rate": result.twist_rate,
            "exact_twist_rate": result.exact_twist_rate,
            "twist_rate_error_pct": 100.0 * result.twist_rate_relative_error,
            "centroid_peak": result.centroid,
            "centroid_error_pct": 100.0 * result.error_centroid,
            "surface_direct_peak": result.direct,
            "surface_direct_error_pct": 100.0 * result.error_direct,
            "surface_extrapolated_peak": result.extrapolated,
            "surface_extrapolated_error_pct": 100.0 * result.error_extrapolated,
            "reference_boundary_peak": result.exact_boundary,
            "reference_scan_peak": result.exact_scan,
        })
        print(f"{element:5s} seg={segments:<2d} dofs={result.n_dofs:5d} "
              f"centroid={100 * result.error_centroid:6.2f}% "
              f"direct={100 * result.error_direct:6.2f}% "
              f"extrap={100 * result.error_extrapolated:6.2f}% "
              f"exact={result.exact_boundary:.5f}", flush=True)
    return rows


def figures(rows: list[dict], cross: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    written = []

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for key, label, style in (("centroid_error_pct", "centroid (previous WITHHELD metric)", "o-"),
                              ("surface_direct_error_pct", "surface, direct", "s-"),
                              ("surface_extrapolated_error_pct", "surface, Gauss-extrapolated", "^-")):
        ax.loglog([r["n_dofs"] for r in rows], [r[key] for r in rows], style, label=label,
                  markersize=5)
    ax.axhline(3.0, color="tab:red", linestyle="--", linewidth=1.0, label="3% gate")
    ax.set_xlabel("active DOFs")
    ax.set_ylabel("relative error of peak |tau| [%]")
    ax.set_title("Saint-Venant torsion: lateral-surface peak shear recovery")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "error_vs_dofs.png"
    fig.savefig(path, dpi=150)
    written.append(str(path.relative_to(ROOT)))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    labels = [f"{r['element']}\n{r['segments']}" for r in rows]
    index = range(len(rows))
    width = 0.27
    for offset, key, label in ((-width, "centroid_error_pct", "centroid"),
                               (0.0, "surface_direct_error_pct", "surface direct"),
                               (width, "surface_extrapolated_error_pct", "surface extrapolated")):
        ax.bar([i + offset for i in index], [r[key] for r in rows], width=width, label=label)
    ax.axhline(3.0, color="tab:red", linestyle="--", linewidth=1.0, label="3% gate")
    ax.set_xticks(list(index))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("relative error [%]")
    ax.set_title("Estimator comparison per mesh (exact surface value as reference)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    path = OUT / "estimator_bars.png"
    fig.savefig(path, dpi=150)
    written.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return written


def main() -> int:
    cross = reference_cross_check()
    rows = run_ladder()
    files = figures(rows, cross)
    qualified = [r for r in rows if r["surface_direct_error_pct"] < 3.0]
    status = {
        "peak_gate_pct": 3.0,
        "surface_direct_qualified": bool(qualified),
        "first_qualified": (
            {"element": qualified[0]["element"], "segments": qualified[0]["segments"],
             "n_dofs": qualified[0]["n_dofs"],
             "error_pct": qualified[0]["surface_direct_error_pct"]}
            if qualified else None),
        "centroid_never_qualified": all(r["centroid_error_pct"] > 3.0 for r in rows),
        "coarse_fail_retained": rows[0]["surface_direct_error_pct"] > 3.0,
        "hex20_2_nonmonotone": next((r["surface_direct_error_pct"] for r in rows
                                     if r["element"] == "hex20" and r["segments"] == 2), None),
        "reference_cross_check_pct": 100.0 * cross["scan_deficit"],
    }
    payload = {"case": {"side": SIDE, "length": 4.0, "young": YOUNG, "poisson": POISSON,
                        "torque": TORQUE, "monitored_point": [SIDE, SIDE / 2.0, 2.0]},
               "reference_cross_check": cross, "ladder": rows, "status": status,
               "scripts": ["benchmarks/torsion_boundary_stress.py"],
               "module": "src/tensorfem/torsion_boundary.py",
               "tests": ["tests/test_torsion_boundary_stress.py"]}
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = OUT / "boundary_peak_evidence.json"
    evidence.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    files.append(str(evidence.relative_to(ROOT)))
    print(json.dumps(status, indent=2))
    print("wrote:", *files, sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())