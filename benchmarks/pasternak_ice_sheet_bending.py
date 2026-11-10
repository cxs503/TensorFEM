"""Benchmark / verification driver: ice sheet on a Pasternak (two-parameter) bed.

Produces the machine-readable evidence for the campaign node
``p485w1-n02-fem-pasternak`` and the figures of the accompanying report:

(a) infinite-beam (ice strip) point-load peak deflection: FE vs Hetenyi (1946)
    at ``G_s = 0`` and vs the Pasternak closed form at ``G_s > 0`` -- error < 1%;
(b) characteristic-length scaling: the shear layer length ``l_shear`` scales as
    ``G_s^{-1/2}`` (fitted exponent error < 0.02) and the soft layer length
    ``l_soft`` as ``G_s^{+1/2}``, both extracted from the *FE solution*;
(c) failure-mode transition: flexure -> foundering transition of the ice strip
    as a function of ``l_c / L`` plus the local-crushing/flexure crossing.

Outputs land in ``artifacts/pasternak_ice_sheet/`` (JSON + CSV) and
``docs/figures/`` (PNG).

Run with ``python benchmarks/pasternak_ice_sheet_bending.py``.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.beam_foundation import hetenyi_point_load_deflection  # noqa: E402
from tensorfem.ice_sheet_pasternak import (  # noqa: E402
    IceSheetProperties,
    failure_mode,
    flexural_failure_load,
    ice_sheet_characteristic_lengths,
    plate_peak_deflection,
    strip_patch_fem,
    transition_scan,
)
from tensorfem.pasternak_beam import (  # noqa: E402
    PasternakBeamModel,
    decay_modes_from_nodes,
    pasternak_characteristic_lengths,
    pasternak_characteristic_roots,
    pasternak_infinite_beam_point_load,
    pasternak_peak_moment,
    pasternak_penetration,
    power_law_exponent,
    solve_pasternak_beam,
)

ARTIFACTS = ROOT / "artifacts" / "pasternak_ice_sheet"
FIGURES = ROOT / "docs" / "figures"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, complex):
        return {"re": value.real, "im": value.imag}
    if isinstance(value, torch.Tensor):
        return _jsonable(value.tolist())
    if isinstance(value, float):
        return float(f"{value:.12g}")
    return value


# ---------------------------------------------------------------------------
# (a) peak deflection against the analytic anchors
# ---------------------------------------------------------------------------
def criterion_a(sheet: IceSheetProperties) -> dict:
    """FE peak deflection of the ice strip vs Hetenyi / Pasternak closed forms."""
    D, k = sheet.rigidity, sheet.buoyancy
    load = 1.0e5
    rows = []
    pasternak_values = [0.0, 1.0e7, 5.0e7, 2.0e8, 1.0e9]
    for Gs in pasternak_values:
        lengths = pasternak_characteristic_lengths(D, k, Gs)
        l_shear = lengths["l_shear"]
        l_soft = lengths["l_soft"]
        length = 12.0 * max(l_shear, l_soft) + 12.0 * lengths["l_winkler"]
        n_elements = int(min(max(math.ceil(length / (l_shear / 12.0)), 400),
                             40000))
        model = PasternakBeamModel(length=length, n_elements=n_elements, E=1.0,
                                   Iz=D, k=k, shear=Gs)
        sol = solve_pasternak_beam(model, nodal_loads={n_elements // 2: load},
                                   diagnostics=True)
        peak_fem = float(sol["w"][n_elements // 2])
        peak_analytic = pasternak_penetration(load, D, k, Gs)
        moment_fem = float(sol["M"][n_elements // 2])
        moment_analytic = pasternak_peak_moment(load, D, k, Gs)
        if Gs == 0.0:
            anchor = "Hetenyi 1946 (A(beta x) form)"
            peak_reference = float(
                hetenyi_point_load_deflection(
                    torch.tensor([length / 2.0], dtype=torch.float64),
                    length / 2.0, load, k, D)[0])
        else:
            anchor = "Pasternak closed form (this work)"
            peak_reference = peak_analytic
        if Gs == 0.0:
            # the Pasternak closed form must reproduce the Hetenyi anchor
            xs = torch.linspace(0.0, length / 2.0, 401, dtype=torch.float64)
            het = hetenyi_point_load_deflection(xs, length / 2.0, load, k, D)
            pas = pasternak_infinite_beam_point_load(xs, length / 2.0, load, D, k,
                                                     0.0)
            anchor_reproduced = float((pas - het).abs().max() / het.abs().max())
        else:
            anchor_reproduced = None
        rows.append({
            "pasternak": Gs,
            "regime": lengths["regime"],
            "l_shear": l_shear,
            "l_soft": l_soft,
            "lwinkler": lengths["l_winkler"],
            "length": length,
            "n_elements": n_elements,
            "mesh": model.h,
            "peak_fem": peak_fem,
            "peak_analytic": peak_analytic,
            "peak_reference": peak_reference,
            "anchor": anchor,
            "error_vs_reference": (peak_fem - peak_reference) / peak_reference,
            "moment_fem": moment_fem,
            "moment_analytic": moment_analytic,
            "moment_error": (moment_fem - moment_analytic) / moment_analytic,
            "equilibrium": sol["reaction"]["total"] / load,
            "closed_form_reproduces_hetenyi": anchor_reproduced,
            "residual": sol["residual"],
        })
    # mesh convergence of the hardest case (thin shear layer)
    Gs = 2.0e8
    lengths = pasternak_characteristic_lengths(D, k, Gs)
    length = 12.0 * max(lengths["l_shear"], lengths["l_soft"]) + 12.0 * lengths["l_winkler"]
    exact = pasternak_penetration(load, D, k, Gs)
    convergence = []
    moment_exact = pasternak_peak_moment(load, D, k, Gs)
    for per_layer in (2, 4, 6, 10, 16):
        n_elements = int(math.ceil(length / (lengths["l_shear"] / per_layer)))
        model = PasternakBeamModel(length=length, n_elements=n_elements, E=1.0,
                                   Iz=D, k=k, shear=Gs)
        sol = solve_pasternak_beam(model, nodal_loads={n_elements // 2: load},
                                   diagnostics=True)
        peak = float(sol["w"][n_elements // 2])
        moment = float(sol["M"][n_elements // 2])
        n = 2 * (n_elements + 1)
        convergence.append({"elements_per_l_shear": per_layer,
                            "n_elements": n_elements, "dofs": n,
                            "peak_deflection": peak,
                            "deflection_error": (peak - exact) / exact,
                            "moment_error": (moment - moment_exact) / moment_exact,
                            "relative_error": (peak - exact) / exact})
    worst = max(abs(row["error_vs_reference"]) for row in rows)
    return {
        "load": load,
        "rows": rows,
        "mesh_convergence": convergence,
        "worst_peak_error": worst,
        "criterion_met": bool(worst < 1e-2),
        "criterion": "FEM peak deflection vs analytic anchor < 1%",
    }


# ---------------------------------------------------------------------------
# (b) characteristic-length scaling law
# ---------------------------------------------------------------------------
def criterion_b(sheet: IceSheetProperties, base: float = 1.0e8) -> dict:
    """Extract both decay lengths from the FE field and fit their exponents."""
    D, k = sheet.rigidity, sheet.buoyancy
    load = 1.0e5
    gs0 = 2.0 * math.sqrt(D * k)
    rows = []
    for Gs in (base, 2.0 * base, 4.0 * base, 8.0 * base, 1.6 * base * 10):
        lam_f, lam_s = pasternak_characteristic_roots(D, k, Gs)
        l_shear = 1.0 / lam_f.real
        l_soft = 1.0 / lam_s.real
        length = 6.0 * l_soft + 6.0 * l_shear
        h_target = l_shear / 20.0
        n_elements = int(math.ceil(length / h_target))
        model = PasternakBeamModel(length=length, n_elements=n_elements, E=1.0,
                                   Iz=D, k=k, shear=Gs)
        t0 = time.time()
        sol = solve_pasternak_beam(model, nodal_loads={n_elements // 2: load},
                                   diagnostics=False)
        runtime = time.time() - t0
        w = sol["w"][n_elements // 2:]
        window = int(4.0 * l_shear / model.h) + 2
        fast, _slow_window = decay_modes_from_nodes(w[:window], model.h, start=1)
        _fast_full, slow = decay_modes_from_nodes(w, model.h, start=1)
        rows.append({
            "pasternak": Gs,
            "n_elements": n_elements,
            "mesh": model.h,
            "l_shear_analytic": l_shear,
            "l_soft_analytic": l_soft,
            "l_shear_fem": 1.0 / fast.real,
            "l_soft_fem": 1.0 / slow.real,
            "l_shear_error": (1.0 / fast.real - l_shear) / l_shear,
            "l_soft_error": (1.0 / slow.real - l_soft) / l_soft,
            "runtime_s": runtime,
            "regime": "monotone" if abs(lam_f.imag) == 0.0 else "oscillatory",
            "pasternak_over_threshold": Gs / gs0,
            "threshold": gs0,
        })
    shear_fit = power_law_exponent([r["pasternak"] for r in rows],
                                   [r["l_shear_fem"] for r in rows])
    soft_fit = power_law_exponent([r["pasternak"] for r in rows],
                                  [r["l_soft_fem"] for r in rows])
    exponent_error = abs(shear_fit["exponent"] - (-0.5))
    return {
        "rows": rows,
        "shear_fit": shear_fit,
        "soft_fit": soft_fit,
        "target_exponent": -0.5,
        "exponent_error": exponent_error,
        "criterion_met": bool(exponent_error < 0.02),
        "criterion": "l_shear ~ G_s^m with |m + 1/2| < 0.02 (FE-extracted)",
        "worst_length_error": max(abs(r["l_shear_error"]) for r in rows),
        "worst_soft_length_error": max(abs(r["l_soft_error"]) for r in rows),
        "note": ("all sampled Gs exceed the oscillatory/monotone threshold "
                 "2 sqrt(D k), i.e. inside the two-exponential (asymptotic) "
                 "regime; l_shear is extracted from a near-field window of 4 "
                 "l_shear, l_soft from the full half-domain"),
    }


# ---------------------------------------------------------------------------
# (c) failure-mode transition
# ---------------------------------------------------------------------------
def criterion_c(sheet: IceSheetProperties) -> dict:
    """Flexure -> foundering transition versus ``l_c / L`` and the contact sweep."""
    gs0 = 2.0 * math.sqrt(sheet.rigidity * sheet.buoyancy)
    pasternak_values = [0.0, 1.0 * gs0, 5.0 * gs0, 20.0 * gs0, 100.0 * gs0]
    lengths = [12.5, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]
    records = transition_scan(sheet, sheet_lengths=lengths,
                              pasternak_values=pasternak_values,
                              contact_half_length=0.5)
    families = []
    for Gs in pasternak_values:
        family = [rec for rec in records if abs(rec["pasternak"] - Gs) < 1e-9]
        family.sort(key=lambda rec: rec["sheet_length"])
        flexure_branch = [rec for rec in family if rec["mode"] == "global_flexure"]
        foundering_branch = [rec for rec in family if rec["mode"] == "foundering"]
        plateau = [rec["loads"]["global_flexure"] for rec in flexure_branch
                   if rec["l_ratio"] <= 0.25]
        entry = {
            "pasternak": Gs,
            "l_c": family[0]["l_c"],
            "l_winkler": family[0]["l_winkler"],
            "curve": [{"l_ratio": rec["l_ratio"],
                       "sheet_length": rec["sheet_length"],
                       "critical_load": rec["critical_load"],
                       "mode": rec["mode"],
                       "loads": rec["loads"]} for rec in family],
            "n_flexure": len(flexure_branch),
            "n_foundering": len(foundering_branch),
            "flexure_plateau": {
                "window": "l_c/L <= 0.25 (sheet much longer than l_c)",
                "min": min(plateau) if plateau else None,
                "max": max(plateau) if plateau else None,
                "relative_spread": ((max(plateau) - min(plateau)) / min(plateau))
                if len(plateau) > 1 else None,
            },
            "analytic_infinite_beam_capacity": sheet.moment_capacity
            / pasternak_peak_moment(1.0, sheet.rigidity, sheet.buoyancy, Gs),
            "transition_l_ratio": (max(rec["l_ratio"] for rec in flexure_branch)
                                   if flexure_branch else None),
            "transition_length": (max(rec["sheet_length"] for rec in flexure_branch)
                                  if flexure_branch else None),
        }
        families.append(entry)
    # branch exponents per family: the foundering asymptote must be -1 in l_c/L
    foundering_fits = []
    for entry in families:
        points = [(pt["l_ratio"], pt["critical_load"]) for pt in entry["curve"]
                  if pt["mode"] == "foundering"]
        if len(points) >= 2:
            fit = power_law_exponent([pt[0] for pt in points],
                                     [pt[1] for pt in points])
            foundering_fits.append({"pasternak": entry["pasternak"],
                                    "n_points": len(points), "fit": fit,
                                    "exponent_error": abs(fit["exponent"] + 1.0)})
    foundering_fit = ({**foundering_fits[0]["fit"],
                       "worst_exponent_error": max(f["exponent_error"]
                                                   for f in foundering_fits)}
                      if foundering_fits else None)
    # local crushing vs flexure: contact-length sweep at fixed sheet length
    contact_rows = []
    for half in (0.0005, 0.001, 0.002, 0.004, 0.008, 0.02, 0.05, 0.1, 0.5,
                 1.0, 2.0):
        rec = failure_mode(sheet, sheet_length=400.0, pasternak=0.0,
                           contact_half_length=half)
        contact_rows.append({
            "contact_half_length": half,
            "contact_ratio": rec["contact_ratio"],
            "local_crushing": rec["loads"]["local_crushing"],
            "global_flexure": rec["loads"]["global_flexure"],
            "foundering": rec["loads"]["foundering"],
            "mode": rec["mode"],
        })
    crossing = None
    for lo, hi in zip(contact_rows[:-1], contact_rows[1:]):
        if lo["mode"] != hi["mode"]:
            crossing = {"between_contact_half_length": [lo["contact_half_length"],
                                                        hi["contact_half_length"]],
                        "from": lo["mode"], "to": hi["mode"]}
            break
    # how far the finite patch sits below the point-load limit, and that the
    # gap closes as the patch shrinks (evidence that the gap is the patch, not
    # a model error)
    contact_convergence = []
    shear_ref = 5.0 * gs0
    shear_length = pasternak_characteristic_lengths(
        sheet.rigidity, sheet.buoyancy, shear_ref)["l_shear"]
    reference = (sheet.moment_capacity
                 / pasternak_peak_moment(1.0, sheet.rigidity, sheet.buoyancy,
                                         shear_ref))
    for half in (0.5, 0.2, 0.1, 0.05, 0.02):
        # keep the mesh resolution tied to the patch: h = (2a)/6
        n_elements = min(int(3.0 * 500.0 / half), 30000)
        flex = flexural_failure_load(sheet, sheet_length=500.0,
                                     pasternak=shear_ref,
                                     contact_half_length=half,
                                     n_elements=n_elements)
        contact_convergence.append({
            "contact_half_length": half,
            "l_shear": shear_length,
            "contact_over_l_shear": half / shear_length,
            "n_elements": n_elements,
            "mesh": flex["mesh"],
            "fem_capacity": flex["load"],
            "analytic_point_load": reference,
            "relative_difference": (flex["load"] - reference) / reference,
        })
    # reference capacity of the *infinite* strip on the same Pasternak bed:
    # P = M_cap / (M(0) per unit load), which for G_s = 0 is the Hetenyi
    # P = 4 M_cap / l_c.  Finite sheets must sit on this plateau.
    plateau_vs_classical = []
    for entry in families:
        if entry["flexure_plateau"]["min"] is None:
            continue
        ref = entry["analytic_infinite_beam_capacity"]
        plateau_vs_classical.append({
            "pasternak": entry["pasternak"],
            "l_c": entry["l_c"],
            "fem_plateau": entry["flexure_plateau"]["min"],
            "analytic_infinite_beam": ref,
            "relative_difference": (entry["flexure_plateau"]["min"] - ref) / ref,
        })
    plateau_spread = max(entry["flexure_plateau"]["relative_spread"] or 0.0
                         for entry in families)
    exponent_error = (foundering_fit["worst_exponent_error"]
                      if foundering_fit else None)
    return {
        "families": families,
        "foundering_fit": foundering_fit,
        "foundering_fits": foundering_fits,
        "foundering_target_exponent": -1.0,
        "foundering_exponent_error": exponent_error,
        "contact_sweep": contact_rows,
        "contact_convergence": contact_convergence,
        "mode_crossing": crossing,
        "analytic_infinite_beam_capacity": {
            entry["pasternak"]: entry["analytic_infinite_beam_capacity"]
            for entry in families},
        "plateau_vs_classical": plateau_vs_classical,
        "flexure_plateau_spread": plateau_spread,
        "criterion_met": bool(
            exponent_error is not None and exponent_error < 0.05
            and plateau_spread < 0.05),
        "criterion": ("flexure plateau independent of sheet length "
                      "(spread < 5%), foundering branch exponent -1 +- 0.05"),
    }


# ---------------------------------------------------------------------------
# ice-specific corroboration: plate closed form (pre-existing module)
# ---------------------------------------------------------------------------
def plate_cross_check(sheet: IceSheetProperties) -> dict:
    """2-D plate Pasternak response used as an implementation cross-check."""
    load = 1.0e5
    rows = []
    for Gs in (0.0, 1.0e7, 1.0e8, 1.0e9, 1.0e10):
        rows.append({"pasternak": Gs,
                     "peak_deflection_plate": plate_peak_deflection(
                         sheet, load=load, pasternak=Gs)})
    reference = rows[0]["peak_deflection_plate"]
    for row in rows:
        row["relative_to_winkler"] = row["peak_deflection_plate"] / reference
    return {"load": load, "rows": rows}


def strip_patch_pressure(sheet: IceSheetProperties) -> dict:
    """Foundation pressure / moment diagram of the loaded ice strip."""
    sol = strip_patch_fem(sheet, length=240.0, pasternak=0.0,
                          contact_half_length=1.0, load=1.0e5)
    nodes = [float(v) for v in sol["nodes"]]
    step = max(len(nodes) // 400, 1)
    return {
        "nodes": nodes[::step],
        "deflection": [float(v) for v in sol["w"][::step]],
        "moment": [float(v) for v in sol["M"][::step]],
        "pressure": [sheet.buoyancy * float(v) for v in sol["w"][::step]],
        "foundation_reaction": sol["reaction"],
        "applied": sol["load"],
        "n_elements": sol["n_elements"],
        "characteristic_lengths": ice_sheet_characteristic_lengths(
            sheet, sol["pasternak"]),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(a: dict, b: dict, c: dict, plate: dict, strip: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = []

    # fig 1: deflection shape vs analytic anchors
    sheet = IceSheetProperties()
    D, k = sheet.rigidity, sheet.buoyancy
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for Gs, style in ((0.0, "-"), (5.0e7, "--"), (5.0e8, ":")):
        lengths = pasternak_characteristic_lengths(D, k, Gs)
        length = 12.0 * max(lengths["l_shear"], lengths["l_soft"]) + 6.0 * lengths["l_winkler"]
        n_elements = 3000
        model = PasternakBeamModel(length=length, n_elements=n_elements, E=1.0,
                                   Iz=D, k=k, shear=Gs)
        sol = solve_pasternak_beam(model, nodal_loads={n_elements // 2: 1.0e5},
                                   diagnostics=False)
        x = (sol["nodes"] - length / 2.0)
        scale = 1.0e5 / float(sol["w"].max())
        axes[0].plot(x, sol["w"] * scale, style, lw=1.2,
                     label=f"FEM  $G_s$={Gs:.0e} N/m")
        xa = torch.linspace(-length / 2.0, length / 2.0, 1201,
                            dtype=torch.float64)
        axes[0].plot(xa, pasternak_infinite_beam_point_load(xa, 0.0, 1.0e5, D, k, Gs)
                     * scale, "k.", ms=2, alpha=0.4)
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("w / w_max  [-]")
    axes[0].set_title("(a) ice strip deflection, FEM (lines) vs closed form (dots)")
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(-120, 120)

    rows = a["rows"]
    gs = [r["pasternak"] for r in rows]
    errs = [100.0 * abs(r["error_vs_reference"]) for r in rows]
    axes[1].semilogx([max(g, 1e6) for g in gs], errs, "o-")
    axes[1].axhline(1.0, color="r", ls="--", lw=1, label="1% acceptance line")
    axes[1].set_xlabel("$G_s$ [N/m]")
    axes[1].set_ylabel("|peak deflection error| [%]")
    axes[1].set_title("(b) FE peak deflection vs analytic anchor")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = FIGURES / "pasternak_peak_error.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(str(path.relative_to(ROOT)))

    # fig 2: characteristic-length scaling
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    gs = [r["pasternak"] for r in b["rows"]]
    lsh = [r["l_shear_fem"] for r in b["rows"]]
    lso = [r["l_soft_fem"] for r in b["rows"]]
    ax.loglog(gs, lsh, "o", label=r"$l_{shear}$ from FE  (fit $m$=%.4f)"
              % b["shear_fit"]["exponent"])
    ax.loglog(gs, lso, "s", label=r"$l_{soft}$ from FE  (fit $m$=%.4f)"
              % b["soft_fit"]["exponent"])
    ref = [lsh[0] * (g / gs[0]) ** -0.5 for g in gs]
    ax.loglog(gs, ref, "k--", lw=1, label=r"$G_s^{-1/2}$ reference")
    ax.set_xlabel("$G_s$ [N/m]")
    ax.set_ylabel("characteristic length [m]")
    ax.set_title("(c) length scales extracted from the FE field")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = FIGURES / "pasternak_length_scaling.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(str(path.relative_to(ROOT)))

    # fig 3: failure-mode transition curves
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for entry in c["families"]:
        curve = entry["curve"]
        xs = [pt["l_ratio"] for pt in curve]
        ys = [pt["critical_load"] for pt in curve]
        label = f"$G_s$={entry['pasternak']:.1e}"
        axes[0].loglog(xs, ys, "o-", ms=3, lw=1,
                       label=label)
        for pt in curve:
            if pt["mode"] == "foundering":
                axes[0].plot(pt["l_ratio"], pt["critical_load"], "x", color="r",
                             ms=6)
    axes[0].set_xlabel(r"$l_c / L$  [-]")
    axes[0].set_ylabel(r"critical load [N/m]")
    axes[0].set_title("(d) local->global transition: red x = foundering branch")
    axes[0].legend(fontsize=7)

    contact = c["contact_sweep"]
    axes[1].loglog([r["contact_half_length"] for r in contact],
                   [r["local_crushing"] for r in contact], "s-", label="local crushing")
    axes[1].loglog([r["contact_half_length"] for r in contact],
                   [r["global_flexure"] for r in contact], "o-", label="global flexure")
    axes[1].loglog([r["contact_half_length"] for r in contact],
                   [r["foundering"] for r in contact], "d-", label="foundering")
    axes[1].set_xlabel("contact half length $a$ [m]")
    axes[1].set_ylabel("load [N/m]")
    axes[1].set_title("(e) mechanism ranking vs contact size ($L$=400 m)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = FIGURES / "pasternak_failure_mode_map.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(str(path.relative_to(ROOT)))

    # fig 4: field map (2-D plate closed form) + strip moment diagram
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    grid = np.linspace(-80.0, 80.0, 241)
    gx, gy = np.meshgrid(grid, grid)
    r = np.sqrt(gx**2 + gy**2)
    from tensorfem.foundation_plate_bending import (
        infinite_plate_two_parameter_point_load,
    )
    rt = torch.tensor(r, dtype=torch.float64)
    for ax, Gs in zip(axes[:2], (0.0, 5.0e8)):
        w = infinite_plate_two_parameter_point_load(rt, 1.0e5, D, k, Gs) * 1e3
        cs = ax.contourf(gx, gy, w, levels=24, cmap="viridis")
        ax.set_title(f"infinite ice plate w [mm], $G_s$={Gs:.0e}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        fig.colorbar(cs, ax=ax, shrink=0.85)
    nodes = np.asarray(strip["nodes"], dtype=float)
    axes[2].plot(nodes, np.asarray(strip["moment"], dtype=float) / 1e3, lw=1.2,
                 label="M [kN m/m]")
    axes[2].plot(nodes, np.asarray(strip["pressure"], dtype=float) / 1e3,
                 lw=1.0, ls="--",
                 label="foundation pressure [kPa]")
    axes[2].set_xlim(60, 180)
    axes[2].set_xlabel("x [m]")
    axes[2].set_title("(f) ice strip: moment & buoyancy reaction")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    path = FIGURES / "pasternak_ice_sheet_fields.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(str(path.relative_to(ROOT)))

    return paths


def main() -> dict:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    sheet = IceSheetProperties()
    print("ice sheet:", sheet)
    t0 = time.time()
    a = criterion_a(sheet)
    print(f"(a) done in {time.time() - t0:.1f}s worst error "
          f"{100 * a['worst_peak_error']:.3f}%")
    t0 = time.time()
    b = criterion_b(sheet)
    print(f"(b) done in {time.time() - t0:.1f}s shear exponent "
          f"{b['shear_fit']['exponent']:.6f} soft exponent "
          f"{b['soft_fit']['exponent']:.6f}")
    t0 = time.time()
    c = criterion_c(sheet)
    print(f"(c) done in {time.time() - t0:.1f}s foundering exponent "
          f"{c['foundering_fit']['exponent'] if c['foundering_fit'] else None}")
    plate = plate_cross_check(sheet)
    strip = strip_patch_pressure(sheet)
    figures = make_figures(a, b, c, plate, strip)
    payload = {
        "node": "p485w1-n02-fem-pasternak",
        "module": "tensorfem.pasternak_beam + tensorfem.ice_sheet_pasternak",
        "sheet_properties": {
            "thickness": sheet.thickness, "young": sheet.young,
            "poisson": sheet.poisson, "ice_density": sheet.ice_density,
            "water_density": sheet.water_density, "gravity": sheet.gravity,
            "flexural_strength": sheet.flexural_strength,
            "crushing_strength": sheet.crushing_strength,
            "rigidity": sheet.rigidity, "buoyancy": sheet.buoyancy,
            "freeboard": sheet.freeboard,
            "moment_capacity": sheet.moment_capacity,
        },
        "criterion_a": a,
        "criterion_b": b,
        "criterion_c": c,
        "plate_cross_check": plate,
        "strip_response": strip,
        "figures": figures,
    }
    (ARTIFACTS / "pasternak_ice_sheet_benchmark.json").write_text(
        json.dumps(_jsonable(payload), indent=2))
    print("criteria:",
          {"a": a["criterion_met"], "b": b["criterion_met"],
           "c": c["criterion_met"]})
    return payload


if __name__ == "__main__":
    main()