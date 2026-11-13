#!/usr/bin/env python
"""Root-cause probe for the Q4 Mindlin plate path bias found in W26.

The W26 case matrix shows the Q4 Mindlin plate FE path landing 9-21% below the
Pagano 3-D anchor on the cross-ply laminates while the layered HEX20 path
reproduces the same anchor to < 0.5%.  Two hypotheses explain that:

H1 element/model defect  -- the plate FE is biased for every plate, including
                            the isotropic limit where a closed-form CLT solution
                            (``navier_clt``) exists and FSDT adds only
                            ``1 + D a^2 / A_s`` (a few percent at a/h = 10).
H2 anchor/layup specific -- the plate FE is fine and the deviation is a
                            cross-ply specific effect.

This script decides between them on isotropic single-ply laminates (no
bend-twist coupling, no layup effects), for a/h = 100 (thin) and a/h = 10, and
writes ``plate_element_diagnosis.json``.  Every number is a solver output.

Usage::

    PYTHONPATH=src python benchmarks/composite_laminate_w26/probe_plate_element.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "papers" / "artifacts" / "w26" / "composite-laminate-verification"
sys.path.insert(0, str(REPO / "src"))

from tensorfem.composite_laminate_bending import (  # noqa: E402
    Laminate, Ply3D, navier_clt, plate_fem,
)

A = 1.0
Q0 = 1.0
ISO = Ply3D(E1=70e9, E2=70e9, E3=70e9, nu12=0.3, nu13=0.3, nu23=0.3,
            G12=70e9 / 2.6, G13=70e9 / 2.6, G23=70e9 / 2.6)
MESHES = (8, 16, 32)
RATIOS = (0.01, 0.1)


def main() -> int:
    rows = []
    for ratio in RATIOS:
        h = ratio * A
        lam = Laminate([0.0], h, ISO)
        clt = navier_clt(lam, A, A, Q0).W
        # first-order shear contribution for an SS plate: 1 + D a^2 / A_s
        D = float(lam.d_matrix()[0, 0])
        As = float(lam.as_matrix()[0, 0])
        fsdt_ratio = 1.0 + D * (3.141592653589793 / A) ** 2 / As
        seq = []
        for n in MESHES:
            r = plate_fem(lam, A, Q0, n=n)
            seq.append({"n": n, "w": r["w_center"],
                        "err_vs_clt_pct": 100.0 * (r["w_center"] / clt - 1.0),
                        "dofs": r["n_dofs"]})
        rows.append({
            "t_over_a": ratio, "a_over_h": 1.0 / ratio,
            "w_clt": clt, "w_fsdt_theory_ratio": fsdt_ratio,
            "w_fsdt_theory": clt * fsdt_ratio,
            "plate_fem": seq,
            "err_finest_vs_clt_pct": seq[-1]["err_vs_clt_pct"],
            "err_finest_vs_fsdt_theory_pct": 100.0 * (seq[-1]["w"] / (clt * fsdt_ratio) - 1.0),
        })
    # control 1: the thin isotropic plate must reproduce CLT (no shear, no layup)
    # control 2: the a/h = 10 isotropic plate must reproduce the first-order shear
    #            ratio; 5% band because the closed-form ratio itself is a
    #            first-order approximation of the code's MITC shear treatment
    thin = next(r for r in rows if r["a_over_h"] >= 100.0)
    thick = next(r for r in rows if r["a_over_h"] <= 10.0)
    control_thin_ok = bool(abs(thin["err_finest_vs_clt_pct"]) < 1.0)
    control_thick_ok = bool(abs(thick["err_finest_vs_fsdt_theory_pct"]) < 5.0)
    verdict = "H2_anchor_or_layup_specific" if (control_thin_ok and control_thick_ok) \
        else "H1_element_defect"
    payload = {
        "probe": "isotropic single-ply plate, E = 70 GPa, nu = 0.3, SS1, "
                 "q = q0 sin(pi x/a) sin(pi y/a)",
        "reference": "navier_clt (Kirchhoff closed form) and the first-order "
                     "shear ratio 1 + D a^2 / A_s",
        "rows": rows,
        "verdict": verdict,
        "controls": {"thin_isotropic_vs_clt_lt_1pct": control_thin_ok,
                     "thick_isotropic_vs_fsdt_lt_5pct": control_thick_ok},
        "interpretation": {
            "H1_element_defect":
                "the Q4 Mindlin plate FE misses even the isotropic closed form, so "
                "the W26 plate-path deviations are an element/model defect, not a "
                "cross-ply effect",
            "H2_anchor_or_layup_specific":
                "the Q4 Mindlin plate FE reproduces the isotropic closed form, so "
                "the W26 plate-path deviations are specific to the layered cases",
        }[verdict],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plate_element_diagnosis.json").write_text(json.dumps(payload, indent=2))
    for r in rows:
        print(f"a/h={r['a_over_h']:.0f}  CLT={r['w_clt']:.6e}  "
              f"FSDT theory x{r['w_fsdt_theory_ratio']:.4f}  "
              f"plate finest {r['err_finest_vs_clt_pct']:+.2f}% vs CLT, "
              f"{r['err_finest_vs_fsdt_theory_pct']:+.2f}% vs FSDT theory")
    print("verdict:", verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())