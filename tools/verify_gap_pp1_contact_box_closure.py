#!/usr/bin/env python3
"""Contact-box closure audit for GAP-PP-1-ICE-STRUCT-CONTACT-PATCH.

Why this audit exists
---------------------
The anchor's patch load is *solved*, not prescribed: ``czm_contact_patch`` finds
the approach ``delta0`` that carries the target normal force over a pressure
field evaluated on a rectangular analysis box.  A relief-driven contact patch has
a **finite** radius where ``p -> 0``.  If the box is smaller than that radius the
box faces carry pressure and then, by construction,

* ``contact_area`` is the box area, not the contact area,
* ``mean = F / contact_area`` is a box artefact,
* and the force balance inflates ``peak_pressure`` above the physical value.

None of that was diagnosed before this audit: the merged analytic verification
(``benchmarks/artifacts/GAP-PP-1-czm-contact-analytic/verification.json``) is run
at 7.07 kN where the contact radius (0.0866 m) *does* fit inside the 0.10 m box,
while every production configuration of :func:`nonuniformity_scan` runs at 40 kN
where it does **not**.  The verified regime and the deployed regime were
different, and nothing said so.

Sections
--------
``A`` solver vs closed form *at the operating force* (40 kN, closed box):
    paraboloid indenter on a Winkler penalty foundation,
    ``delta0 = sqrt(2 amp F / (pi k R^2))``, ``p0 = k delta0``,
    ``r_c = sqrt(delta0 R^2 / amp)``, ``mean = p0 / 2``.
``B`` the truncating box: same solve with the shipped ``box_half = 0.10``.
    Reports the boundary-pressure ratio, the peak-pressure inflation and the
    contact-area loss against the closed-form reference -- and that the
    diagnostic flags it.
``C`` every configuration of :func:`default_configurations`, on its shipped box:
    ``contact_closed``, boundary ratio, peak pressure, contact area.
``D`` what the truncation does to the anchor's headline number: non-uniform vs
    uniform peak panel stress (same total force, same footprint) under the
    truncating (0.10 m) and the closure-sufficient (0.20 m) box.
``E`` verdict block: every judgement is a boolean derived from the numbers.

Re-run::

    python tools/verify_gap_pp1_contact_box_closure.py \
        --out benchmarks/artifacts/GAP-PP-1-contact-box-closure
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.ice_panel_patch_contact import (  # noqa: E402
    CLOSURE_TOL,
    DESIGN_POINT,
    _solve_with_patch,
    coupled_patch_solve,
    czm_contact_patch,
    default_configurations,
    default_law,
    uniform_patch_pressure,
)

OUT = ROOT / "benchmarks" / "artifacts" / "GAP-PP-1-contact-box-closure"
PANEL = (DESIGN_POINT["l"], DESIGN_POINT["s"])
CENTRE = (0.5 * PANEL[0], 0.5 * PANEL[1])
FORCE = 4.0e4  # the production normal force of nonuniformity_scan (40 kN)
CEILING = 2.0e6
R, AMP = 1.0, 0.02  # spherical ice edge of configurations C2 / C3


def analytic_paraboloid(force: float, *, k: float, R: float, amp: float) -> dict:
    """Uncapped Winkler solution for a paraboloid relief pressed by ``force``."""
    d0 = math.sqrt(2.0 * amp * force / (math.pi * k * R**2))
    rc = math.sqrt(d0 * R**2 / amp)
    return {
        "delta0_m": d0,
        "peak_pressure_Pa": k * d0,
        "contact_radius_m": rc,
        "contact_area_m2": math.pi * rc * rc,
        "mean_pressure_Pa": 0.5 * k * d0,  # p(r) = p0 (1 - r^2/rc^2)
        "peak_over_mean": 2.0,
    }


def _peak_stress_ratio(patch, *, nx: int, ny: int) -> dict:
    """Non-uniform patch vs uniform patch: same force, same footprint."""
    a, b = PANEL
    half_x = 0.5 * float(patch.x[-1] - patch.x[0])
    half_y = 0.5 * float(patch.y[-1] - patch.y[0])
    uni = uniform_patch_pressure(
        total_force=patch.total_force,
        centre=patch.centroid,
        half_x=half_x,
        half_y=half_y,
        panel=PANEL,
        grid=int(patch.x.numel()),
    )
    sol_np, _ = _solve_with_patch(patch, nx=nx, ny=ny, panel=PANEL,
                                  thickness=DESIGN_POINT["t"], young=DESIGN_POINT["E"],
                                  poisson=DESIGN_POINT["nu"])
    sol_uni, _ = _solve_with_patch(uni, nx=nx, ny=ny, panel=PANEL,
                                   thickness=DESIGN_POINT["t"], young=DESIGN_POINT["E"],
                                   poisson=DESIGN_POINT["nu"])
    p_np = sol_np.peak_stress()
    p_uni = sol_uni.peak_stress()
    return {
        "footprint_m": [2.0 * half_x, 2.0 * half_y],
        "contact_area_m2": patch.contact_area,
        "peak_pressure_MPa": patch.peak_pressure / 1e6,
        "nonuniform_sigma_Mx_MPa": p_np["sigma_Mx_MPa"],
        "uniform_sigma_Mx_MPa": p_uni["sigma_Mx_MPa"],
        "sigma_Mx_ratio": p_np["sigma_Mx_MPa"] / p_uni["sigma_Mx_MPa"],
        "sigma_Mx_pct": 100.0 * (p_np["sigma_Mx_MPa"] / p_uni["sigma_Mx_MPa"] - 1.0),
        "w_over_t_nonuniform": None,  # filled by the caller when it has the field
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--mesh", default="24,21", help="panel mesh nx,ny for section D")
    ap.add_argument("--grid", type=int, default=61, help="contact grid for section D")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nx, ny = (int(v) for v in args.mesh.split(","))
    t0 = time.time()

    law = default_law()
    ref = analytic_paraboloid(FORCE, k=law.stiffness, R=R, amp=AMP)
    report: dict = {
        "case": "GAP-PP-1: contact-box closure audit of the CZM patch load",
        "provenance": {
            "script": Path(__file__).name,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "machine": platform.machine(),
        },
        "setting": {
            "panel_m": list(PANEL),
            "thickness_m": DESIGN_POINT["t"],
            "total_force_N": FORCE,
            "ceiling_MPa": CEILING / 1e6,
            "interface_stiffness_Pa_per_m": law.stiffness,
            "ice_edge": {"kind": "spherical", "radius": R, "amplitude": AMP},
            "closure_tol": CLOSURE_TOL,
        },
        "analytic_reference_at_operating_force": ref,
    }

    # ---------------------------------------------------------------- A
    a_rows = []
    for grid in (61, 97):
        patch = czm_contact_patch(
            law=law, ceiling=CEILING,
            edge={"kind": "spherical", "radius": R, "amplitude": AMP},
            total_force=FORCE, panel=PANEL, centre=CENTRE, grid=grid,
        )
        a_rows.append({
            "grid": grid,
            "contact_box_m": patch.config["contact_box"],
            "peak_pressure_MPa": patch.peak_pressure / 1e6,
            "approach_m": patch.approach,
            "contact_area_m2": patch.contact_area,
            "mean_pressure_MPa": patch.total_force / patch.contact_area / 1e6,
            "boundary_pressure_ratio": patch.boundary_pressure_ratio,
            "contact_closed": patch.contact_closed,
            "forces_N": patch.total_force,
            "err": {
                "peak_pressure_pct": 100.0 * (patch.peak_pressure / ref["peak_pressure_Pa"] - 1.0),
                "approach_pct": 100.0 * (patch.approach / ref["delta0_m"] - 1.0),
                "contact_area_pct": 100.0 * (patch.contact_area / ref["contact_area_m2"] - 1.0),
                "peak_over_mean_pct": 100.0 * (
                    patch.peak_pressure / (patch.total_force / patch.contact_area) / 2.0 - 1.0
                ),
            },
        })
        print("A grid=%d peak err %+.3f %% closed=%s"
              % (grid, a_rows[-1]["err"]["peak_pressure_pct"], patch.contact_closed), flush=True)
    report["A_solver_vs_closed_form"] = {
        "rows": a_rows,
        "max_abs_peak_err_pct": max(abs(r["err"]["peak_pressure_pct"]) for r in a_rows),
        "max_abs_area_err_pct": max(abs(r["err"]["contact_area_pct"]) for r in a_rows),
        "all_closed": all(r["contact_closed"] for r in a_rows),
    }

    # ---------------------------------------------------------------- B
    b_rows = []
    for half in (0.10, 0.20):
        patch = czm_contact_patch(
            law=law, ceiling=CEILING,
            edge={"kind": "spherical", "radius": R, "amplitude": AMP, "box_half": half},
            total_force=FORCE, panel=PANEL, centre=CENTRE, grid=97,
        )
        b_rows.append({
            "box_half_m": half,
            "peak_pressure_MPa": patch.peak_pressure / 1e6,
            "approach_m": patch.approach,
            "contact_area_m2": patch.contact_area,
            "mean_pressure_MPa": patch.total_force / patch.contact_area / 1e6,
            "boundary_pressure_ratio": patch.boundary_pressure_ratio,
            "contact_closed": patch.contact_closed,
            "err": {
                "peak_pressure_pct": 100.0 * (patch.peak_pressure / ref["peak_pressure_Pa"] - 1.0),
                "contact_area_pct": 100.0 * (patch.contact_area / ref["contact_area_m2"] - 1.0),
            },
        })
        print("B box_half=%.2f peak %+.2f %% area %+.1f %% closed=%s"
              % (half, b_rows[-1]["err"]["peak_pressure_pct"],
                 b_rows[-1]["err"]["contact_area_pct"], patch.contact_closed), flush=True)
    trunc, closed_row = b_rows[0], b_rows[1]
    report["B_truncating_box"] = {
        "rows": b_rows,
        "truncation_detected": trunc["contact_closed"] is False,
        "closed_detected": closed_row["contact_closed"] is True,
        "peak_pressure_inflation_pct": 100.0 * (
            trunc["peak_pressure_MPa"] / closed_row["peak_pressure_MPa"] - 1.0
        ),
        "contact_area_loss_pct": 100.0 * (
            1.0 - trunc["contact_area_m2"] / closed_row["contact_area_m2"]
        ),
    }

    # ---------------------------------------------------------------- C
    c_rows = []
    for cfg in default_configurations():
        edge = dict(cfg["edge"])
        centre = (
            (0.5 * PANEL[0] + cfg["offset"][0], 0.5 * PANEL[1] + cfg["offset"][1])
            if cfg.get("offset") else None
        )
        if cfg["coupled"]:
            patch, _, coup = coupled_patch_solve(
                nx=nx, ny=ny, panel=PANEL, thickness=DESIGN_POINT["t"],
                young=DESIGN_POINT["E"], poisson=DESIGN_POINT["nu"],
                law=law, ceiling=CEILING, edge=edge, total_force=FORCE,
                centre=centre, grid=args.grid, max_coupling=40,
            )
            closed, bnd = coup["contact_closed"], coup["boundary_pressure_ratio"]
            box, sweeps, conv = coup["contact_box"], coup["sweeps"], coup["converged"]
        else:
            patch = czm_contact_patch(
                law=law, ceiling=CEILING, edge=edge, total_force=FORCE,
                panel=PANEL, centre=centre, grid=args.grid,
            )
            closed, bnd = patch.contact_closed, patch.boundary_pressure_ratio
            box, sweeps, conv = patch.config["contact_box"], 1, True
        c_rows.append({
            "name": cfg["name"],
            "edge": edge,
            "coupled": cfg["coupled"],
            "contact_box_m": box,
            "boundary_pressure_ratio": bnd,
            "contact_closed": closed,
            "peak_pressure_MPa": patch.peak_pressure / 1e6,
            "contact_area_m2": patch.contact_area,
            "sweeps": sweeps,
            "coupling_converged": conv,
        })
        print("C %-32s closed=%-5s bnd=%.4f conv=%s"
              % (cfg["name"], closed, bnd, conv), flush=True)
    report["C_shipped_configurations"] = {
        "rows": c_rows,
        "all_closed": all(r["contact_closed"] is not False for r in c_rows),
        "all_converged": all(r["coupling_converged"] for r in c_rows),
        "max_boundary_pressure_ratio": max(r["boundary_pressure_ratio"] for r in c_rows),
    }

    # ---------------------------------------------------------------- D
    d_rows = {}
    for label, half in (("truncated_box_half_0.10", 0.10), ("closed_box_half_0.20", 0.20)):
        patch = czm_contact_patch(
            law=law, ceiling=CEILING,
            edge={"kind": "spherical", "radius": R, "amplitude": AMP, "box_half": half},
            total_force=FORCE, panel=PANEL, centre=CENTRE, grid=args.grid,
        )
        row = _peak_stress_ratio(patch, nx=nx, ny=ny)
        row["contact_closed"] = patch.contact_closed
        row["boundary_pressure_ratio"] = patch.boundary_pressure_ratio
        d_rows[label] = row
        print("D %-24s ratio=%.4f (%+.1f %%) closed=%s"
              % (label, row["sigma_Mx_ratio"], row["sigma_Mx_pct"], patch.contact_closed), flush=True)
    ratio_t = d_rows["truncated_box_half_0.10"]["sigma_Mx_ratio"]
    ratio_c = d_rows["closed_box_half_0.20"]["sigma_Mx_ratio"]
    report["D_headline_number"] = {
        "rows": d_rows,
        "mesh": [nx, ny],
        "contact_grid": args.grid,
        "ratio_change_rel_pct": 100.0 * (ratio_t / ratio_c - 1.0),
        "same_side_of_unity": (ratio_t - 1.0) * (ratio_c - 1.0) > 0.0,
    }

    # ---------------------------------------------------------------- E
    report["E_verdict"] = {
        "solver_matches_closed_form_at_operating_force": (
            report["A_solver_vs_closed_form"]["max_abs_peak_err_pct"] < 1.0
            and report["A_solver_vs_closed_form"]["max_abs_area_err_pct"] < 1.0
            and report["A_solver_vs_closed_form"]["all_closed"]
        ),
        "truncation_is_detected_by_the_diagnostic": (
            report["B_truncating_box"]["truncation_detected"]
            and report["B_truncating_box"]["closed_detected"]
        ),
        "truncation_biases_peak_pressure": (
            report["B_truncating_box"]["peak_pressure_inflation_pct"] > 3.0
        ),
        "shipped_configurations_close": report["C_shipped_configurations"]["all_closed"],
        "shipped_configurations_converge": report["C_shipped_configurations"]["all_converged"],
        "headline_ratio_is_box_sensitive": abs(report["D_headline_number"]["ratio_change_rel_pct"]) > 3.0,
    }
    report["wall_s"] = time.time() - t0

    # ---------------------------------------------------------------- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for half, style, label in ((0.20, "-", "closed box (0.20 m)"),
                                   (0.10, "--", "truncating box (0.10 m)")):
            patch = czm_contact_patch(
                law=law, ceiling=CEILING,
                edge={"kind": "spherical", "radius": R, "amplitude": AMP, "box_half": half},
                total_force=FORCE, panel=PANEL, centre=CENTRE, grid=97,
            )
            x = (patch.x.numpy() - CENTRE[0]) * 1e3
            mid = patch.pressure.shape[1] // 2
            ax[0].plot(x, patch.pressure[:, mid].numpy() / 1e6, style, label=label)
        r = torch.linspace(0.0, 0.22, 400, dtype=torch.float64)
        p_an = ref["peak_pressure_Pa"] * (
            1.0 - (r.numpy() ** 2) / ref["contact_radius_m"] ** 2
        ).clip(0.0, None)
        ax[0].plot(r.numpy() * 1e3, p_an / 1e6, "k:", lw=1.2, label="closed form (paraboloid)")
        ax[0].set_xlabel("x - centre [mm]")
        ax[0].set_ylabel("contact pressure [MPa]")
        ax[0].set_title("contact patch: box truncation vs closed form")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)

        labels = ["truncated\n(0.10 m)", "closed\n(0.20 m)"]
        vals = [ratio_t, ratio_c]
        ax[1].bar(labels, vals, color=["tab:red", "tab:blue"])
        ax[1].axhline(1.0, color="k", lw=1.0)
        for i, v in enumerate(vals):
            ax[1].text(i, v, "%.3f" % v, ha="center", va="bottom", fontsize=9)
        ax[1].set_ylabel(r"$\sigma_{Mx}$(non-uniform)/$\sigma_{Mx}$(uniform)")
        ax[1].set_title("headline ratio vs analysis box")
        ax[1].grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out / "fig_closure_vs_headline.png", dpi=140)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        report["figure_error"] = repr(exc)

    (out / "closure.json").write_text(json.dumps(report, indent=2))
    print("\nwrote %s (%.1f s)" % (out / "closure.json", report["wall_s"]))
    print(json.dumps(report["E_verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())