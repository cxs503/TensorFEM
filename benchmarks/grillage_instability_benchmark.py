#!/usr/bin/env python3
"""Generate benchmark data and figures for the stiffener instability integration.

Produces:
- docs/grillage_ice_ultimate/instability_cases.json  (machine-readable)
- docs/grillage_ice_ultimate/fig4_instability_comparison.png
- docs/grillage_ice_ultimate/fig5_tripping_governed_pressure.png
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem.grillage_ice_ultimate import (
    CombinedLoadCase, case_study_geometry, ultimate_ice_pressure,
    grillage_utilisations,
)
from tensorfem.grillage_ice_instability import (
    stiffener_instability_utilisation,
    grillage_utilisations_with_instability,
    ultimate_ice_pressure_with_instability,
    tripping_governed_geometry,
)

OUT = Path(__file__).resolve().parent / "docs" / "grillage_ice_ultimate"


def run_stocky_comparison():
    """Compare 3-load-case results with and without instability for case-study geometry."""
    geom = case_study_geometry()
    scenarios = [
        ("light-ice + ballast head", 53e6, 1.0e6, 0.35, 0.35, 0.05e6),
        ("design-ice + design head", 106e6, 2.0e6, 0.70, 0.35, 0.10e6),
        ("severe-ice + tank head", 160e6, 3.0e6, 1.05, 0.35, 0.15e6),
    ]
    rows = []
    for name, sx, p_ice, lx, ly, p_h in scenarios:
        case = CombinedLoadCase(
            axial_stress=sx, ice_pressure=p_ice,
            patch_length=lx, patch_width=ly,
            hydrostatic_pressure=p_h,
        ).validate(geom)
        u_base = grillage_utilisations(geom, case)
        u_full = grillage_utilisations_with_instability(geom, case)
        inst = stiffener_instability_utilisation(geom, case)
        rows.append({
            "scenario": name,
            "sigma_x_MPa": sx / 1e6,
            "ice_pressure_MPa": p_ice / 1e6,
            "U_plate": float(u_base["plate_local_bending"]),
            "U_stiff_beamcol": float(u_base["longitudinal_stiffener"]),
            "U_frame": float(u_base["transverse_frame"]),
            "U_stiff_instability": float(u_full["stiffener_instability"]),
            "tripping_sigma_MPa": float(inst["tripping_sigma_free_Pa"]) / 1e6,
            "beamcol_sigma_L_MPa": float(u_base["stiffener_limiting_stress"]) / 1e6,
            "instability_governing": inst["governing_instability"],
        })
    return rows


def run_tripping_governed_comparison():
    """Compare results with and without instability for tripping-governed geometry."""
    geom = tripping_governed_geometry()
    scenarios = [
        ("light-ice + ballast head", 50e6, 0.8e6, 0.50, 0.50, 0.05e6),
        ("design-ice + design head", 100e6, 1.5e6, 1.00, 0.50, 0.10e6),
        ("severe-ice + tank head", 150e6, 2.5e6, 1.50, 0.50, 0.15e6),
    ]
    rows = []
    for name, sx, p_ice, lx, ly, p_h in scenarios:
        case = CombinedLoadCase(
            axial_stress=sx, ice_pressure=p_ice,
            patch_length=lx, patch_width=ly,
            hydrostatic_pressure=p_h,
        ).validate(geom)
        res_without = ultimate_ice_pressure(geom, case)
        res_with = ultimate_ice_pressure_with_instability(geom, case)
        u_full = grillage_utilisations_with_instability(geom, case)
        rows.append({
            "scenario": name,
            "sigma_x_MPa": sx / 1e6,
            "p_ult_without_instability_MPa": float(res_without["ice_pressure_ultimate"]) / 1e6,
            "p_ult_with_instability_MPa": float(res_with["ice_pressure_ultimate"]) / 1e6,
            "governing_without": res_without["governing_member"],
            "governing_with": res_with["governing_member"],
            "reduction_pct": (1.0 - float(res_with["ice_pressure_ultimate"])
                              / float(res_without["ice_pressure_ultimate"])) * 100
                              if float(res_without["ice_pressure_ultimate"]) > 0 else 0,
            "U_stiff_beamcol": float(u_full["longitudinal_stiffener"]),
            "U_stiff_instability": float(u_full["stiffener_instability"]),
        })
    return rows


def fig4_instability_comparison(stocky_rows):
    """Bar chart: beam-column vs instability utilisation for case-study geometry."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r["scenario"] for r in stocky_rows]
    x = np.arange(len(names))
    w = 0.30
    bc = [r["U_stiff_beamcol"] for r in stocky_rows]
    inst = [r["U_stiff_instability"] for r in stocky_rows]
    ax.bar(x - w/2, bc, w, label="beam-column", color="steelblue")
    ax.bar(x + w/2, inst, w, label="instability (tripping)", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Stiffener utilisation")
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.legend()
    ax.set_title("Case-study geometry: beam-column vs instability utilisation")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_instability_comparison.png", dpi=150)
    plt.close(fig)


def fig5_tripping_governed_pressure(trip_rows):
    """Bar chart: p_ult with and without instability for tripping-governed geometry."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r["scenario"] for r in trip_rows]
    x = np.arange(len(names))
    w = 0.30
    p_wo = [r["p_ult_without_instability_MPa"] for r in trip_rows]
    p_wi = [r["p_ult_with_instability_MPa"] for r in trip_rows]
    ax.bar(x - w/2, p_wo, w, label="without instability", color="steelblue")
    ax.bar(x + w/2, p_wi, w, label="with instability", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Ultimate ice pressure [MPa]")
    ax.legend()
    ax.set_title("Tripping-governed geometry: effect of instability check on p_ult")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_tripping_governed_pressure.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stocky = run_stocky_comparison()
    tripping = run_tripping_governed_comparison()
    payload = {
        "schema": "tensorfem.grillage_ice_instability.benchmark/v1",
        "case_study_comparison": stocky,
        "tripping_governed_comparison": tripping,
    }
    (OUT / "instability_cases.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    fig4_instability_comparison(stocky)
    fig5_tripping_governed_pressure(tripping)
    print(f"Wrote {OUT / 'instability_cases.json'}")
    print(f"Wrote {OUT / 'fig4_instability_comparison.png'}")
    print(f"Wrote {OUT / 'fig5_tripping_governed_pressure.png'}")


if __name__ == "__main__":
    main()
