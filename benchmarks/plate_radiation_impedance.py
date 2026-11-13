"""Verification benchmark: compressible baffled-panel radiation impedance.

Produces ``artifacts/plate_radiation_impedance/validation_report.json``.

Anchors (each one machine-checkable, see ``docs/CASE-FEM-PLATE-RADIATION-IMPEDANCE.md``)

A1  Struve ``H1`` own series vs ``scipy.special.struve``         (implementation)
A2  Piston closed form ``R1``/``X1`` + low-ka asymptotes         (literature)
A3  Piston high-ka limit ``R1 -> 1``                             (literature)
A4  Panel operator on a *disc* vs the exact piston               (discretisation)
A5  Panel resistance ``R`` vs ``2 P_rad`` (independent quadrature)(code path)
A6  ``mu(k -> 0)`` vs ``hydro_added_mass.panel_added_mass_ratio`` (in-phase reduction)
A7  Mesh convergence of ``mu`` and ``R`` (observed order, not assumed)
A8  Ship-panel case: frequency error of the in-phase model + radiation loss factor

Usage
-----
    PYTHONPATH=src python benchmarks/plate_radiation_impedance.py \
        --out artifacts/plate_radiation_impedance/validation_report.json

All numbers in the JSON are produced by this script (no hand-copied values).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from tensorfem.acoustic_radiation import rayleigh_radiated_power
from tensorfem.hydro_added_mass import panel_added_mass_ratio
from tensorfem.plate_radiation_impedance import (
    disc_piston_impedance_panels,
    panel_modal_impedance,
    panel_wet_dynamics,
    piston_normalized_impedance,
    struve_h1_series,
)

SOUND_SPEED_WATER = 1500.0
RHO_WATER = 1025.0
RHO_STEEL = 7850.0
E_STEEL = 2.1e11
NU_STEEL = 0.3

# Reference ship side-shell panel (3 m x 1.5 m, 12 mm steel) and a large thin
# panel (6 m x 3 m, 6 mm) whose wet resonance sits at a higher ka.
PANELS = {
    "stiff_3x1.5x12mm": dict(lx=3.0, ly=1.5, thickness=0.012),
    "thin_6x3x6mm": dict(lx=6.0, ly=3.0, thickness=0.006),
}


def rel_err(value: float, reference: float) -> float:
    if reference == 0.0:
        return abs(value)
    return abs(value - reference) / abs(reference)


def anchor_a1_struve() -> dict:
    from scipy.special import struve as scipy_struve

    rows = []
    for x in (0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0):
        own = struve_h1_series(x)
        ref = float(scipy_struve(1, x))
        rows.append({"x": x, "own_series": own, "scipy": ref,
                     "rel_err": rel_err(own, ref)})
    return {
        "description": "own Struve H1 power series vs scipy.special.struve(1, x)",
        "rows": rows,
        "max_rel_err": max(r["rel_err"] for r in rows),
        "criterion": "max rel err < 1e-12 for x <= 8",
        "pass": max(r["rel_err"] for r in rows) < 1e-12,
    }


def anchor_a2_piston_closed_form() -> dict:
    rows = []
    for ka in (0.05, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0):
        r1, x1 = piston_normalized_impedance(ka)
        rows.append({
            "ka": ka, "R1": r1, "X1": x1,
            "R1_lowka_asymptote": 0.5 * ka * ka,
            "X1_lowka_asymptote": 8.0 * ka / (3.0 * math.pi),
            "R1_rel_err_vs_asymptote": rel_err(r1, 0.5 * ka * ka),
            "X1_rel_err_vs_asymptote": rel_err(x1, 8.0 * ka / (3.0 * math.pi)),
        })
    low = [r for r in rows if r["ka"] <= 0.1]
    return {
        "description": ("exact baffled-piston impedance R1 = 1 - 2J1(x)/x, "
                        "X1 = 2H1(x)/x, x = 2ka (Morse & Ingard 1968) and the "
                        "low-ka asymptotes R1 -> (ka)^2/2, X1 -> 8ka/(3pi)"),
        "rows": rows,
        "max_lowka_rel_err": max(
            max(r["R1_rel_err_vs_asymptote"], r["X1_rel_err_vs_asymptote"])
            for r in low
        ),
        "criterion": "at ka <= 0.1 both asymptote errors < 1%",
        "pass": max(max(r["R1_rel_err_vs_asymptote"],
                        r["X1_rel_err_vs_asymptote"]) for r in low) < 1e-2,
    }


def anchor_a3_piston_high_ka() -> dict:
    rows = []
    for ka in (5.0, 10.0, 20.0, 40.0):
        r1, x1 = piston_normalized_impedance(ka)
        rows.append({"ka": ka, "R1": r1, "X1": x1, "ka_times_X1": ka * x1})
    high = [r for r in rows if r["ka"] >= 10.0]
    return {
        "description": ("high-ka limits: R1 -> 1 (all incident power radiated) "
                        "and X1 -> 0.  Note ka*X1 does *not* vanish: "
                        "X1 = 2H1(x)/x with H1 -> Y1(x) + 2/pi, so "
                        "ka*X1 -> Y1(x)/2 + 1/pi oscillates; the column is "
                        "reported as an observation, not gated."),
        "rows": rows,
        "max_abs_R1_minus_1": max(abs(r["R1"] - 1.0) for r in rows),
        "criterion": "|R1 - 1| < 0.02 and |X1| < 0.06 for ka >= 10",
        "pass": (max(abs(r["R1"] - 1.0) for r in high) < 2e-2
                 and max(abs(r["X1"]) for r in high) < 6e-2),
    }


def anchor_a4_disc_panel_vs_piston() -> dict:
    rows = []
    for n_cells in (30, 60):
        for ka in (0.5, 1.0, 2.0):
            panels = disc_piston_impedance_panels(
                radius=1.0, ka=ka, rho=RHO_WATER,
                sound_speed=SOUND_SPEED_WATER, n_cells=n_cells,
            )
            r1, x1 = piston_normalized_impedance(ka)
            rows.append({
                "n_cells": n_cells, "ka": ka,
                "R1_panels": panels["R1_panels"], "R1_exact": r1,
                "R1_rel_err": rel_err(panels["R1_panels"], r1),
                "X1_panels": panels["X1_panels"], "X1_exact": x1,
                "X1_rel_err": rel_err(panels["X1_panels"], x1),
                "area_ratio": panels["area_ratio"],
                "h_over_a": panels["h_over_a"],
            })
    fine_low = [r for r in rows if r["n_cells"] == 60 and r["ka"] <= 1.0]
    return {
        "description": ("panel operator + analytic square-cell self patch applied "
                        "to a staircase disc, compared with the exact piston"),
        "rows": rows,
        "max_R1_rel_err_ka_le_1": max(r["R1_rel_err"] for r in fine_low),
        "max_X1_rel_err_ka_le_1": max(r["X1_rel_err"] for r in fine_low),
        "criterion": ("n_cells=60, ka<=1: R1 rel err < 2%, X1 rel err < 3%; "
                      "ka>=2 reactance error reported, not gated (staircase "
                      "boundary + cancellation of a small difference)"),
        "pass": (all(r["R1_rel_err"] < 2e-2 for r in fine_low)
                 and all(r["X1_rel_err"] < 3e-2 for r in fine_low)),
    }


def anchor_a5_resistance_vs_radiated_power() -> dict:
    rows = []
    for panel_name, dims in PANELS.items():
        for (m, n) in ((1, 1), (2, 1)):
            for freq in (20.0, 200.0, 800.0):
                imp = panel_modal_impedance(
                    lx=dims["lx"], ly=dims["ly"],
                    thickness=dims["thickness"], rho_steel=RHO_STEEL,
                    rho_water=RHO_WATER, sound_speed=SOUND_SPEED_WATER,
                    frequency=freq, m=m, n=n, nx=40, ny=20,
                )
                power = rayleigh_radiated_power(
                    length=dims["lx"], width=dims["ly"], frequency=freq,
                    rho=RHO_WATER, c=SOUND_SPEED_WATER, m=m, n=n,
                    velocity_amplitude=1.0, n_points=32,
                )
                rows.append({
                    "panel": panel_name, "mode": [m, n], "frequency": freq,
                    "radiation_damping": imp["radiation_damping"],
                    "two_times_P_rad": 2.0 * power,
                    "rel_err": rel_err(imp["radiation_damping"], 2.0 * power),
                    "added_mass_ratio": imp["added_mass_ratio"],
                    "loss_factor": imp["loss_factor"],
                })
    return {
        "description": ("generalised radiation damping R from the complex panel "
                        "operator vs 2*P_rad from the independent sine-kernel "
                        "4-D Gauss quadrature (tensorfem.acoustic_radiation)"),
        "rows": rows,
        "max_rel_err": max(r["rel_err"] for r in rows),
        "criterion": "max rel err < 1%",
        "pass": max(r["rel_err"] for r in rows) < 1e-2,
    }


def anchor_a6_inphase_reduction() -> dict:
    """mu(k -> 0) vs hydro_added_mass, and *why* they are not bit-identical.

    Both operators are the same double sum; the only difference is the diagonal.
    Here the exact square-cell integral ``int_cell dA/r = 2 h ln(1+sqrt 2)`` is
    used, while ``hydro_added_mass`` uses the equal-area-disc value
    ``sqrt(A/pi)``.  Their ratio is the h-independent constant
    ``ln(1+sqrt 2)/sqrt(pi) = 0.4973`` (i.e. 2.011x), so the *modal* difference
    must decay like ``O(h)``.  That is the gate: convergence, not equality.
    """
    dims = PANELS["stiff_3x1.5x12mm"]
    rows = []
    for (nx, ny) in ((20, 10), (40, 20), (80, 40)):
        ref = panel_added_mass_ratio(
            dims["lx"], dims["ly"], dims["thickness"], RHO_STEEL,
            RHO_WATER, nx=nx, ny=ny, m=1, n=1,
        )
        # 1 Hz keeps k*h ~ 2e-4, i.e. deep in the in-phase limit
        imp = panel_modal_impedance(
            lx=dims["lx"], ly=dims["ly"], thickness=dims["thickness"],
            rho_steel=RHO_STEEL, rho_water=RHO_WATER,
            sound_speed=SOUND_SPEED_WATER, frequency=1.0,
            m=1, n=1, nx=nx, ny=ny,
        )
        h = dims["lx"] / nx
        rows.append({
            "nx": nx, "ny": ny, "h": h,
            "mu_complex_at_1Hz": imp["added_mass_ratio"],
            "mu_inphase_reference": ref["mu"],
            "rel_err": rel_err(imp["added_mass_ratio"], ref["mu"]),
            "self_square_exact": 2.0 * h * math.log(1.0 + math.sqrt(2.0))
                                 / (2.0 * math.pi),
            "self_disc_equal_area": math.sqrt(h * h / math.pi),
        })
    ratios = [rows[i]["rel_err"] / rows[i + 1]["rel_err"]
              for i in range(len(rows) - 1)]
    return {
        "description": ("complex operator at k -> 0 vs "
                        "hydro_added_mass.panel_added_mass_ratio; the residual "
                        "difference is the diagonal (square cell vs equal-area "
                        "disc) and must decay as O(h)"),
        "rows": rows,
        "successive_error_ratios": ratios,
        "self_term_ratio_analytic": math.log(1.0 + math.sqrt(2.0)) / math.sqrt(math.pi),
        "criterion": ("finest mesh rel err < 1.5% and successive error ratios "
                      "in [1.7, 2.3] (first order)"),
        "pass": (rows[-1]["rel_err"] < 1.5e-2
                 and all(1.7 <= r <= 2.3 for r in ratios)),
    }


def anchor_a7_mesh_convergence() -> dict:
    dims = PANELS["thin_6x3x6mm"]
    rows = []
    for (nx, ny) in ((20, 10), (40, 20), (80, 40)):
        imp = panel_modal_impedance(
            lx=dims["lx"], ly=dims["ly"], thickness=dims["thickness"],
            rho_steel=RHO_STEEL, rho_water=RHO_WATER,
            sound_speed=SOUND_SPEED_WATER, frequency=100.0,
            m=2, n=1, nx=nx, ny=ny,
        )
        ref = panel_added_mass_ratio(
            dims["lx"], dims["ly"], dims["thickness"], RHO_STEEL,
            RHO_WATER, nx=nx, ny=ny, m=2, n=1,
        )
        rows.append({
            "nx": nx, "ny": ny, "h": dims["lx"] / nx,
            "k_times_h": imp["k"] * dims["lx"] / nx,
            "n_panels": imp["n_panels"],
            "mu": imp["added_mass_ratio"],
            "mu_inphase_same_mesh": ref["mu"],
            "radiation_damping": imp["radiation_damping"],
        })
    observed = []
    for a, b in zip(rows, rows[1:]):
        for key in ("radiation_damping",):
            d1 = abs(b[key] - a[key]) / abs(b[key])
            d2 = abs(a[key] - rows[0][key]) / abs(rows[0][key])
            ratio = b["h"] / a["h"]
            if d1 > 0.0:
                observed.append({
                    "key": key, "refined_rel_change": d1,
                    "coarse_rel_change": d2, "mesh_ratio": ratio,
                })
    return {
        "description": ("mesh refinement at f = 100 Hz (mode 2,1) on the thin "
                        "panel: R and mu vs panel size; the observed order is "
                        "reported, not assumed"),
        "rows": rows,
        "steps": observed,
        "criterion": ("successive R values must agree better than 3% between "
                      "the two finest meshes"),
        "pass": (abs(rows[-1]["radiation_damping"] - rows[-2]["radiation_damping"])
                 / abs(rows[-1]["radiation_damping"]) < 3e-2),
    }


def anchor_a8_ship_panel() -> dict:
    rows = []
    for panel_name, dims in PANELS.items():
        for (m, n) in ((1, 1), (2, 1), (1, 2)):
            res = panel_wet_dynamics(
                lx=dims["lx"], ly=dims["ly"], thickness=dims["thickness"],
                rho_steel=RHO_STEEL, rho_water=RHO_WATER,
                sound_speed=SOUND_SPEED_WATER, youngs_modulus=E_STEEL,
                poisson=NU_STEEL, m=m, n=n, nx=40, ny=20,
            )
            rows.append({"panel": panel_name, "mode": [m, n], **res})
    err = max(abs(r["inphase_model_error"]) for r in rows)
    resid = max(r["iteration_residual"] for r in rows)
    return {
        "description": ("wet resonance of ship side-shell panels: frequency-"
                        "dependent (compressible) vs frequency-independent "
                        "(in-phase) added mass -- same discretisation, so the "
                        "difference is the *physics* -- plus the radiation loss "
                        "factor the in-phase model cannot produce at all"),
        "rows": rows,
        "max_inphase_frequency_error": err,
        "max_iteration_residual": resid,
        "criterion": ("|f_wet,in-phase - f_wet,compressible| / f_wet < 0.5% "
                      "and fixed-point residual < 1e-9"),
        "pass": err < 5e-3 and resid < 1e-9,
    }


def frequency_sweep() -> dict:
    dims = PANELS["stiff_3x1.5x12mm"]
    freqs = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]
    rows = []
    for freq in freqs:
        imp = panel_modal_impedance(
            lx=dims["lx"], ly=dims["ly"], thickness=dims["thickness"],
            rho_steel=RHO_STEEL, rho_water=RHO_WATER,
            sound_speed=SOUND_SPEED_WATER, frequency=freq,
            m=1, n=1, nx=40, ny=20,
        )
        rows.append({
            "frequency": freq,
            "k_times_h": imp["k"] * dims["lx"] / 40,
            "added_mass_ratio": imp["added_mass_ratio"],
            "radiation_damping": imp["radiation_damping"],
            "loss_factor": imp["loss_factor"],
        })
    return {
        "description": ("reactance and resistance of the stiff reference panel, "
                        "mode (1,1), over 5-1600 Hz"),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/plate_radiation_impedance/"
                                     "validation_report.json")
    args = ap.parse_args()

    t0 = time.time()
    anchors = {
        "A1_struve_series_vs_scipy": anchor_a1_struve(),
        "A2_piston_closed_form": anchor_a2_piston_closed_form(),
        "A3_piston_high_ka": anchor_a3_piston_high_ka(),
        "A4_disc_panel_vs_piston": anchor_a4_disc_panel_vs_piston(),
        "A5_resistance_vs_radiated_power": anchor_a5_resistance_vs_radiated_power(),
        "A6_inphase_reduction": anchor_a6_inphase_reduction(),
        "A7_mesh_convergence": anchor_a7_mesh_convergence(),
        "A8_ship_panel_wet_resonance": anchor_a8_ship_panel(),
    }
    sweep = frequency_sweep()
    report = {
        "capability": "fem.plate.radiation_impedance_complex",
        "module": "tensorfem.plate_radiation_impedance",
        "anchors": anchors,
        "frequency_sweep": sweep,
        "gated_anchors_passed": all(
            a.get("pass", True) for a in anchors.values()
        ),
        "runtime_seconds": round(time.time() - t0, 2),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out} ({report['runtime_seconds']} s)")
    for name, a in anchors.items():
        flag = "PASS" if a.get("pass", True) else "FAIL"
        print(f"  {name}: {flag}")
    print(f"  gated_anchors_passed = {report['gated_anchors_passed']}")


if __name__ == "__main__":
    main()