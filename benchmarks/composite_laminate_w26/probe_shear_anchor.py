#!/usr/bin/env python
"""Independent FSDT anchor for the W26 cases whose CLT anchor is disputed.

The W26 case matrix scores every t/a <= 0.1 case against the *Kirchhoff* (CLT)
series.  For the shear-soft angle-ply and quasi-isotropic families at a/h = 10
both FE paths land 60-85% above that anchor, which is either a real
transverse-shear effect (CLT out of its validity domain) or a solver defect.
This probe settles it with the first-order shear deformation (Mindlin) closed
form, which is an independent analytical solution: a Navier solution, not an FE
result, computed here straight from the ABD matrix.

``navier_fsdt`` itself refuses any laminate with ``D16 != 0`` exactly (!).  The
laminates here are balanced with ``D16/D11 ~ 1e-10``, i.e. numerically decoupled,
so the probe applies a *relative* tolerance (``1e-9``) and documents it.

Writes ``fsdt_anchor.json``; every number is a solver output.

Usage::

    PYTHONPATH=src python benchmarks/composite_laminate_w26/probe_shear_anchor.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "papers" / "artifacts" / "w26" / "composite-laminate-verification"
sys.path.insert(0, str(REPO / "src"))

from tensorfem.composite_laminate_bending import (  # noqa: E402
    Laminate, PAGANO_MATERIAL, navier_clt, navier_fsdt, navier_clt_series,
)

A = 1.0
Q0 = 1.0
REL_TOL = 1e-9


def main() -> int:
    with (OUT / "case_matrix.csv").open() as fh:
        cases = list(csv.DictReader(fh))
    rows = []
    for row in cases:
        angles = [float(x) for x in row["angles_deg"].strip("[]").replace(";",",").split(",") if x.strip()]
        h = float(row["thickness"])
        lam = Laminate(angles, h / len(angles), PAGANO_MATERIAL)
        _, _, D, _ = lam.abd()
        scale = max(abs(D[0, 0]), abs(D[1, 1]), 1e-300)
        decoupled = (abs(D[0, 2]) <= REL_TOL * scale and abs(D[1, 2]) <= REL_TOL * scale)
        clt = navier_clt_series(lam, A, A, Q0, modes=61)["W"]
        w_fsdt = navier_fsdt(lam, A, A, Q0).W if decoupled else None
        w_plate = float(row["w_plate_n32"])
        w_hex = float(row["w_hex_finest"])
        rows.append({
            "case": row["case"], "t_over_a": float(row["t_over_a"]),
            "D16_over_D11": float(row["D16_over_D11"]),
            "numerically_decoupled": bool(decoupled),
            "w_clt_series": clt,
            "w_fsdt_navier": w_fsdt,
            "fsdt_over_clt": None if w_fsdt is None else w_fsdt / clt,
            "plate_over_clt": w_plate / clt,
            "hex_over_clt": w_hex / clt,
            "plate_vs_fsdt_pct": (None if w_fsdt is None
                                  else 100.0 * (w_plate / w_fsdt - 1.0)),
            "hex_vs_fsdt_pct": (None if w_fsdt is None
                                else 100.0 * (w_hex / w_fsdt - 1.0)),
        })

    # the disputed regime: t/a = 0.1 non-special-orthotropic layups
    disputed = [r for r in rows if abs(r["t_over_a"] - 0.1) < 1e-12
                and r["fsdt_over_clt"] is not None]
    explained = all(abs(r["hex_vs_fsdt_pct"]) < abs(r["hex_over_clt"] - 1.0) * 100.0
                    for r in disputed)
    payload = {
        "probe": "first-order shear deformation (Mindlin) Navier closed form vs the "
                 "Kirchhoff (CLT) series anchor and the two FE paths",
        "rel_tolerance_for_D16_D26": REL_TOL,
        "note": "navier_fsdt rejects any laminate with D16 != 0 exactly; the "
                "balanced laminates here are numerically decoupled (D16/D11 ~ 1e-10), "
                "so a relative tolerance is applied by the probe",
        "rows": rows,
        "disputed_regime": [r["case"] for r in disputed],
        "conclusion": (
            "the FSDT anchor absorbs most of the FE-vs-CLT gap for the shear-soft "
            "layups at a/h = 10, so the gap is transverse shear deformation and the "
            "Kirchhoff anchor is outside its validity domain there"
            if explained else
            "the FSDT anchor does not absorb the FE-vs-CLT gap at a/h = 10; the "
            "deviation is not explained by first-order shear deformation and the two "
            "FE paths / the anchor remain in conflict"),
        "conclusion_flag": bool(explained),
    }
    (OUT / "fsdt_anchor.json").write_text(json.dumps(payload, indent=2))
    for r in rows:
        ratio = "n/a" if r["fsdt_over_clt"] is None else f"{r['fsdt_over_clt']:.4f}"
        vs = "n/a" if r["hex_vs_fsdt_pct"] is None else f"{r['hex_vs_fsdt_pct']:+.2f}%"
        print(f"{r['case']:32s} t/a={r['t_over_a']:.2f} fsdt/clt={ratio} "
              f"plate/clt={r['plate_over_clt']:.4f} hex/clt={r['hex_over_clt']:.4f} "
              f"hex_vs_fsdt={vs}")
    print("conclusion:", payload["conclusion"])
    return 0


if __name__ == "__main__":
    sys.exit(main())