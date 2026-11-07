"""Driver for the ice / elastic hull-panel FSI case study.

Produces ``artifacts/ice_panel_fsi/case_report.json`` and four figures:

1. contact-load history, rigid versus elastic bay, and the pressure profile
   along the bay at the peak;
2. panel deflection field and reaction pressure along the contact line;
3. peak-load reduction and panel frequency versus plate thickness;
4. energy budget (kinetic -> panel strain + ice crushing + ice bending) with
   the closure error.

Run with ``PYTHONPATH=src python scripts/run_ice_panel_fsi_case.py``.
"""
from __future__ import annotations

import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensorfem.ice_panel_fsi import (
    CoupledConfig,
    IceConfig,
    PanelConfig,
    compare_rigid_elastic,
    theory_centre_deflection,
    uniform_load_centre_deflection,
)

D = torch.float64
OUT = pathlib.Path("artifacts/ice_panel_fsi")
OUT.mkdir(parents=True, exist_ok=True)


def _pressure_profile(result):
    state = result["peak_state"]
    positions = result["config"].contact_positions()[:, 1]
    return positions.tolist(), (state["pressure"] / 1e6).tolist()


def _field(result):
    panel = result["panel"]
    w = panel.displacement_field(result["peak_state"]["force"])[:, 0]
    cfg = panel.config
    return w.reshape(cfg.ny + 1, cfg.nx + 1).numpy()


def main() -> dict:
    representative = CoupledConfig(ice=IceConfig(velocity=0.3), block_mass=233.0,
                                   steps=400)
    rep = compare_rigid_elastic(representative)

    thicknesses = (0.012, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050)
    sweep = []
    for t in thicknesses:
        case = CoupledConfig(**{
            **representative.__dict__,
            "panel": PanelConfig(**{**representative.panel.__dict__,
                                    "thickness": t})})
        row = compare_rigid_elastic(case)
        sweep.append({
            "thickness_m": t,
            "peak_rigid_N": row["peak_rigid"],
            "peak_elastic_N": row["peak_elastic"],
            "load_reduction": row["load_reduction"],
            "bending_fraction_elastic": row["bending_fraction_elastic"],
            "panel_frequency_Hz": row["elastic"]["panel_frequency"],
            "energy_closure_error": row["elastic"]["energy_closure_error"],
        })

    high_energy = CoupledConfig(ice=IceConfig(velocity=0.3), block_mass=4000.0,
                                steps=800)
    high = compare_rigid_elastic(high_energy)
    thin_flex = compare_rigid_elastic(CoupledConfig(**{
        **high_energy.__dict__, "panel": PanelConfig(thickness=0.012)}))
    thick_stiff = compare_rigid_elastic(CoupledConfig(**{
        **high_energy.__dict__, "panel": PanelConfig(thickness=0.040)}))

    # ---- static plate verification numbers -------------------------------
    verification = []
    for span_x, span_y, thickness, boundary in (
            (1.0, 1.0, 0.010, "simply_supported"),
            (0.75, 2.4, 0.010, "simply_supported"),
            (0.75, 2.4, 0.025, "simply_supported"),
            (0.75, 2.4, 0.025, "clamped"),
            (1.0, 1.0, 0.010, "clamped")):
        cfg = PanelConfig(span_x=span_x, span_y=span_y, thickness=thickness,
                          boundary=boundary, nx=8, ny=16)
        fem = uniform_load_centre_deflection(cfg, 1.0e5)
        theory = theory_centre_deflection(cfg, 1.0e5)
        verification.append({
            "span_x": span_x, "span_y": span_y, "thickness": thickness,
            "boundary": boundary, "fem_centre_m": fem,
            "plate_theory_m": theory,
            "error_vs_theory": (fem - theory) / theory,
        })

    # ---- figures ---------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for res, label, colour in ((rep["rigid"], "rigid bay", "tab:red"),
                               (rep["elastic"], "elastic bay (25 mm)", "tab:blue")):
        ax[0].plot(res["time"] * 1e3, res["total_force"] / 1e3, colour,
                   label=f"{label}: peak {res['peak_force'] / 1e3:.1f} kN")
    ax[0].set_xlabel("time [ms]")
    ax[0].set_ylabel("contact load [kN]")
    ax[0].set_title("ice impact on the panel bay")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].grid(alpha=0.3)
    for res, label, style in ((rep["rigid"], "rigid bay", "o-"),
                              (rep["elastic"], "elastic bay", "-")):
        y, p = _pressure_profile(res)
        ax[1].plot(y, p, style, markersize=4, label=label)
    ax[1].set_xlabel("position along the bay y [m]")
    ax[1].set_ylabel("contact pressure at peak [MPa]")
    ax[1].set_title("reaction pressure along the contact line")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_load_history_and_pressure.png", dpi=160)
    plt.close(fig)

    cfg = rep["elastic"]["panel"].config
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    im = ax[0].imshow(_field(rep["elastic"]) * 1e3, origin="lower",
                      extent=[0, cfg.span_x, 0, cfg.span_y], aspect="auto",
                      cmap="viridis")
    fig.colorbar(im, ax=ax[0], label="deflection at peak [mm]")
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("y [m]")
    ax[0].set_title("elastic bay deflection at peak load")
    ax[1].plot(thicknesses, [row["load_reduction"] * 100 for row in sweep],
               "o-", color="tab:green")
    ax[1].axhline(10.0, ls=":", color="grey")
    ax[1].axhline(30.0, ls=":", color="grey")
    ax[1].axvline(0.025, ls="--", color="tab:blue", alpha=0.6)
    ax[1].set_xlabel("plate thickness [m]")
    ax[1].set_ylabel("peak-load reduction [%]")
    ax[1].set_title("compliance feedback versus plating thickness\n(dotted: 10-30% literature band)")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_deflection_field_and_thickness_trend.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(thicknesses, [row["peak_rigid_N"] / 1e3 for row in sweep], "s-",
               color="tab:red", label="rigid bay")
    ax[0].plot(thicknesses, [row["peak_elastic_N"] / 1e3 for row in sweep], "o-",
               color="tab:blue", label="elastic bay")
    ax[0].set_xlabel("plate thickness [m]")
    ax[0].set_ylabel("peak contact load [kN]")
    ax[0].set_title("peak load versus plating thickness")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax2 = ax[1]
    ax2.plot(thicknesses, [row["bending_fraction_elastic"] * 100 for row in sweep],
             "^-", color="tab:orange", label="bending-failed blocks (233 kg)")
    ax2.plot([0.012, 0.040],
             [thin_flex["bending_fraction_elastic"] * 100,
              thick_stiff["bending_fraction_elastic"] * 100],
             "s--", color="tab:brown", label="4 t blocks (high severity)")
    ax2.set_xlabel("plate thickness [m]")
    ax2.set_ylabel("ice blocks failing in bending [%]")
    ax2.set_title("failure-mode conversion with panel flexibility")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_load_and_failure_mode_vs_thickness.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = ["rigid\n(233 kg)", "elastic\n(233 kg)", "elastic\n(4 t)"]
    results = [rep["rigid"], rep["elastic"], high["elastic"]]
    parts = {
        "ice kinetic (final)": [r["kinetic_final"] for r in results],
        "ice crushing": [r["crush_energy"] for r in results],
        "panel strain": [r["strain_energy"] for r in results],
        "ice bending fracture": [r["bend_energy"] for r in results],
    }
    bottom = [0.0] * len(results)
    for key, values in parts.items():
        ax[0].bar(labels, values, bottom=bottom, label=key)
        bottom = [b + v for b, v in zip(bottom, values)]
    ax[0].plot(labels, [r["kinetic_initial"] for r in results], "k_",
               markersize=22, label="initial kinetic energy")
    ax[0].set_ylabel("energy [J]")
    ax[0].set_title("energy budget (bars + closure residual)")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, axis="y")
    for res, label, colour in ((thin_flex["elastic"], "12 mm bay", "tab:purple"),
                               (high["elastic"], "25 mm bay", "tab:blue"),
                               (thick_stiff["elastic"], "40 mm bay", "tab:green")):
        ax[1].plot(res["time"] * 1e3, res["total_force"] / 1e3, colour,
                   label=f"{label}: {res['peak_force'] / 1e3:.0f} kN, "
                         f"bent {res['bent_blocks']}/12")
    ax[1].set_xlabel("time [ms]")
    ax[1].set_ylabel("contact load [kN]")
    ax[1].set_title("high-severity impact (4 t blocks, 0.3 m/s)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_energy_budget_and_severity.png", dpi=160)
    plt.close(fig)

    report = {
        "case": "elastic hull panel bay under ice-block contact loading",
        "panel": representative.panel.__dict__,
        "ice": representative.ice.__dict__,
        "blocks": representative.blocks,
        "block_mass_kg": representative.block_mass,
        "representative": {
            "peak_rigid_N": rep["peak_rigid"],
            "peak_elastic_N": rep["peak_elastic"],
            "load_reduction": rep["load_reduction"],
            "bending_fraction_rigid": rep["bending_fraction_rigid"],
            "bending_fraction_elastic": rep["bending_fraction_elastic"],
            "panel_frequency_Hz": rep["elastic"]["panel_frequency"],
            "max_deflection_mm": float(torch.max(
                rep["elastic"]["max_deflection"])) * 1e3,
            "energy_closure_rigid": rep["rigid"]["energy_closure_error"],
            "energy_closure_elastic": rep["elastic"]["energy_closure_error"],
            "kinetic_initial_J": rep["elastic"]["kinetic_initial"],
            "ice_crushing_J": rep["elastic"]["crush_energy"],
            "panel_strain_J": rep["elastic"]["strain_energy"],
            "p_cap_MPa": rep["elastic"]["p_cap"] / 1e6,
            "contact_duration_ms": rep["elastic"]["duration"] * 1e3,
            "dt_s": rep["elastic"]["dt"],
        },
        "thickness_sweep": sweep,
        "high_severity": {
            "peak_rigid_N": high["peak_rigid"],
            "peak_elastic_N": high["peak_elastic"],
            "load_reduction": high["load_reduction"],
            "bending_blocks": high["elastic"]["bent_blocks"],
            "energy_closure": high["elastic"]["energy_closure_error"],
            "kinetic_initial_J": high["elastic"]["kinetic_initial"],
            "ice_crushing_J": high["elastic"]["crush_energy"],
            "panel_strain_J": high["elastic"]["strain_energy"],
            "bending_fraction_thin": thin_flex["bending_fraction_elastic"],
            "bending_fraction_thick": thick_stiff["bending_fraction_elastic"],
            "load_reduction_thin": thin_flex["load_reduction"],
            "load_reduction_thick": thick_stiff["load_reduction"],
        },
        "plate_verification": verification,
    }
    (OUT / "case_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("representative", "high_severity")},
                     indent=2))
    return report


if __name__ == "__main__":
    main()