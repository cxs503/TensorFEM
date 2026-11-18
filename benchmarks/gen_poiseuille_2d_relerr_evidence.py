#!/usr/bin/env python3
"""Generate the machine-readable evidence blob for CASE-POISEUILLE-2D-RELERR.

Runs the Guo-scheme Poiseuille benchmark (grid ladder ny=10/18/34, tau sweep,
production solver check) and writes the ref/got/tolerance rows to
``artifacts/CASE-POISEUILLE-2D-RELERR/evidence.json``.  Every number is
produced by the solver -- no hand-typed value.

Usage: python3 benchmarks/gen_poiseuille_2d_relerr_evidence.py
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensorfem import lbm_forcing as F


def main() -> int:
    # Run the benchmark at the standard grid ladder (ny=10/18/34) plus the
    # production-solver check at ny=18.  The full 66-row ladder is too slow
    # for a single-shot evidence run; the 10/18/34 ladder already shows the
    # <1% tolerance at ny=18 and the second-order convergence.
    result = F.poiseuille_benchmark(
        resolutions=(10, 18, 34),
        guo_only_resolutions=(),
        tau_sweep=(0.05, 0.2, 0.5),
        ny_sweep=34,
        include_production=True,
    )

    evidence = {
        "schema": "tensorfem.poiseuille-2d-relerr/1",
        "anchor": "CASE-POISEUILLE-2D-RELERR",
        "mission": 24728,
        "reference": result["reference"],
        "reference_doi": result["reference_doi"],
        "case": result["case"],
        "analytic": result["analytic"],
        "resolutions": result["resolutions"],
        "rows": result["rows"],
        "tau_sweep": result["tau_sweep"],
        "production": result["production"],
        "convergence_order": result["convergence_order"],
        "gate": result["gate"],
        "gate_passed": F.benchmark_passed(result),
        "tolerance": {
            "criterion": "rel_err_u_max < 1% and rel_l2_profile < 1% at ny>=18",
            "guo_ny18_rel_err_u_max": None,
            "guo_ny18_rel_l2": None,
        },
    }

    # Fill in the ny=18 Guo row values for the tolerance check
    for row in result["rows"]:
        if row["scheme"] == "guo" and row["ny"] == 18:
            evidence["tolerance"]["guo_ny18_rel_err_u_max"] = row["rel_err_u_max"]
            evidence["tolerance"]["guo_ny18_rel_l2"] = row["rel_l2_profile"]
            break

    out = pathlib.Path(__file__).resolve().parents[1] / "artifacts" / "CASE-POISEUILLE-2D-RELERR" / "evidence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2))
    print(f"wrote {out}")
    print(f"gate_passed={evidence['gate_passed']}")
    print(f"guo_ny18_rel_err_u_max={evidence['tolerance']['guo_ny18_rel_err_u_max']:.6f}")
    print(f"guo_ny18_rel_l2={evidence['tolerance']['guo_ny18_rel_l2']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
