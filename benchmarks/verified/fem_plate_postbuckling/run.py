#!/usr/bin/env python3
"""p250-N06: plate/shell buckling and postbuckling face -- von Karman plate anchor.

Physics
-------
A thin steel plate ``a = 400 mm`` (loaded) by ``b`` (supported width),
``t = 6 mm``, ``E = 206 GPa``, ``nu = 0.3`` -- the plate field of the campaign
stiffened panel (``beta = (b/t) sqrt(sigma_y/E) = 2.77``, ``sigma_y = 355 MPa``)
with the stiffener removed, so every number below is a *plate* number and the
closed-form oracles apply verbatim.  Edges: simply supported (``w = 0`` and zero
slope along the straight edge); in-plane: ``u = 0`` on ``x = 0``, uniform
traction on ``x = a``, stress-free unloaded edges, one pinned corner.

Five graded criteria (campaign mandate ``p250-n06-structural-buckling``)
-----------------------------------------------------------------------
``A1`` linear bifurcation vs the Timoshenko closed form
    ``sigma_cr = k pi^2 D / (b^2 t)``, ``k = min_m (m b/a + a/(m b))^2``,
    three aspect ratios ``a/b = 1, 2, 3``, gate ``< 3 %``.
``A2`` postbuckling path (Crisfield arc length on the finite-element mesh) vs
    the classical one-mode von Karman solution
    ``N/N_cr = (1 - w0/A) + c (A/t)^2``,
    ``c = 3(1-nu^2)(alpha^4+beta^4)/(4(alpha^2+beta^2)^2)``,
    ten amplitude stations, gate ``< 5 %``.
``A3`` imperfection sensitivity for ``w0 = t/500, t/200, t/100`` vs the same
    Koiter solution at a fixed total amplitude, gate ``< 10 %``.
``A4`` stiffened-panel buckling *mode* vs DNV-RP-C201 mode classification,
    MAC ``> 0.9`` (uses :mod:`tensorfem.stiffened_panel_buckling` for the panel
    eigenproblem -- this driver only adds the MAC against the two DNV modes).
``A5`` negative controls: no load -> positive definite tangent (no spurious
    zero-energy mode), tensile load -> no positive buckling factor.

The one-mode reference is an **asymptotic small-rotation** result.  This driver
therefore reports, separately from the A2 gate range, what happens beyond it:
the converged finite element and the one-mode reduction diverge with amplitude
(``-9.6 %`` at ``A/t = 1``), and the driver measures the *cause* by tracking the
MAC of the finite-element deflection profile against the single sine mode.

Outputs (this directory): ``result.json``, ``gates_verified.json``,
``buckling_mode.png``, ``postbuckling_path.png``,
``imperfection_sensitivity.png``.

Usage::

    PYTHONPATH=src python benchmarks/verified/fem_plate_postbuckling/run.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# The anchor runs ~10^3 small dense factorisations; on a many-core host an
# oversubscribed BLAS makes each of them ~40x slower, so pin the thread count
# before numpy is imported.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmarks.verified._harness import (  # noqa: E402
    Gate, gate_manifest, pyplot, write_json,
)
from tensorfem import von_karman_plate_riks as vk  # noqa: E402

ANCHOR = "fem_plate_postbuckling"
TITLE = "Von Karman plate postbuckling path and imperfection sensitivity (p250-N06)"
HERE = Path(__file__).resolve().parent

A = 0.400          # loaded length [m]
B = 0.400          # supported width [m]
THICKNESS = 0.006  # plate thickness [m]
YOUNG = 206.0e9
POISSON = 0.3
YIELD = 355.0e6

ASPECT_RATIOS = (1.0, 2.0, 3.0)
# The element size stays fixed across aspect ratios: ``ny`` is the ladder and
# ``nx`` follows the ratio, so a long plate is not meshed three times finer by
# accident (a 48x16 mesh would make the dense eigenproblem the bottleneck).
A1_MESH_LADDER = (4, 6, 8)
A2_MESHES = (8, 12)
A2_STATIONS = tuple(round(0.10 + 0.05 * index, 2) for index in range(10))  # 0.10 .. 0.55
A3_IMPERFECTIONS = (1.0 / 500.0, 1.0 / 200.0, 1.0 / 100.0)
A3_STATION = 0.25
NEAR_PERFECT = 1.0e-3   # w0 = t/1000: resolves the bifurcation in load control


def _trace(*, a: float, b: float, nx: int, ny: int, imperfection_over_t: float,
           steps: int = 16, start: float = 0.9, step_size: float = 2.0e-3) -> dict:
    return vk.trace_postbuckling(
        a=a, b=b, thickness=THICKNESS, young=YOUNG, poisson=POISSON, nx=nx, ny=ny,
        imperfection_amplitude=imperfection_over_t * THICKNESS, steps=steps,
        step_size=step_size, tolerance=1e-6, max_iterations=15,
        start_load_factor=start, maximum_step=4.0 * step_size)


def criterion_a1() -> dict:
    rows = []
    for ratio in ASPECT_RATIOS:
        a, b = ratio * B, B
        closed = vk.timoshenko_critical_stress(a=a, b=b, thickness=THICKNESS,
                                               young=YOUNG, poisson=POISSON)
        ladder = []
        for ny in A1_MESH_LADDER:
            nx = int(round(ratio)) * ny
            result = vk.linear_buckling(a=a, b=b, thickness=THICKNESS, young=YOUNG,
                                        poisson=POISSON, nx=nx, ny=ny)
            ladder.append({
                "mesh": [nx, ny],
                "n_dofs": result["n_dofs"],
                "sigma_cr_pa": result["critical_stress"],
                "relative_error": result["critical_stress"] / closed["sigma_cr"] - 1.0,
                "pencil_residual": result["residuals"][0],
                "mac_vs_closed_form_mode": vk.nodal_shape_mac(
                    result["nodes"],
                    np.asarray(result["mode"]).reshape(-1, 6)[:, vk.W],
                    a=a, b=b, m=closed["half_waves"], n=1),
            })
        rows.append({"aspect_ratio": ratio, "half_waves": closed["half_waves"],
                     "buckling_coefficient": closed["buckling_coefficient"],
                     "sigma_cr_reference_pa": closed["sigma_cr"],
                     "ladder": ladder,
                     "max_abs_relative_error": max(abs(row["relative_error"])
                                                   for row in ladder)})
    return {"cases": rows,
            "max_abs_relative_error": max(row["max_abs_relative_error"] for row in rows)}


def criterion_a2() -> dict:
    a, b = A, B
    coefficient = vk.koiter_amplitude_coefficient(a=a, b=b, poisson=POISSON)
    cases = []
    for mesh in A2_MESHES:
        path = _trace(a=a, b=b, nx=mesh, ny=mesh, imperfection_over_t=NEAR_PERFECT)
        stations = []
        for target in A2_STATIONS:
            measured = vk.interpolate_load_factor(path["points"],
                                                  total_amplitude_over_t=target)
            reference = vk.koiter_load_factor(amplitude=target * THICKNESS,
                                              thickness=THICKNESS,
                                              imperfection=NEAR_PERFECT * THICKNESS,
                                              coefficient=coefficient)
            stations.append({"total_amplitude_over_t": target,
                             "load_factor_fe": measured,
                             "load_factor_koiter": reference,
                             "relative_error": (None if measured is None
                                                else measured / reference - 1.0)})
        beyond = []
        for target in (0.6, 0.8, 1.0):
            measured = vk.interpolate_load_factor(path["points"],
                                                  total_amplitude_over_t=target)
            reference = vk.koiter_load_factor(amplitude=target * THICKNESS,
                                              thickness=THICKNESS,
                                              imperfection=NEAR_PERFECT * THICKNESS,
                                              coefficient=coefficient)
            beyond.append({"total_amplitude_over_t": target,
                           "load_factor_fe": measured,
                           "load_factor_koiter": reference,
                           "relative_error": (None if measured is None
                                              else measured / reference - 1.0)})
        cases.append({"mesh": [mesh, mesh], "converged": path["converged"],
                      "points": len(path["points"]),
                      "reached_total_amplitude_over_t":
                          path["points"][-1]["total_amplitude_over_t"],
                      "stations": stations, "beyond_gate_range": beyond,
                      "path": [{"total_amplitude_over_t": point["total_amplitude_over_t"],
                                "load_factor": point["load_factor"]}
                               for point in path["points"]]})
    errors = [abs(station["relative_error"]) for case in cases
              for station in case["stations"] if station["relative_error"] is not None]
    return {"koiter_coefficient": coefficient,
            "imperfection_over_t": NEAR_PERFECT,
            "stations_over_t": list(A2_STATIONS),
            "cases": cases,
            "max_abs_relative_error": max(errors) if errors else float("nan"),
            "all_stations_resolved": len(errors) == len(A2_STATIONS) * len(A2_MESHES)}


def criterion_a3() -> dict:
    coefficient = vk.koiter_amplitude_coefficient(a=A, b=B, poisson=POISSON)
    rows = []
    for ratio in A3_IMPERFECTIONS:
        path = _trace(a=A, b=B, nx=8, ny=8, imperfection_over_t=ratio)
        measured = vk.interpolate_load_factor(path["points"],
                                              total_amplitude_over_t=A3_STATION)
        reference = vk.koiter_load_factor(amplitude=A3_STATION * THICKNESS,
                                          thickness=THICKNESS,
                                          imperfection=ratio * THICKNESS,
                                          coefficient=coefficient)
        curve = []
        for target in tuple(round(0.05 + 0.05 * index, 2) for index in range(12)):
            value = vk.interpolate_load_factor(path["points"],
                                               total_amplitude_over_t=target)
            curve.append({"total_amplitude_over_t": target, "load_factor_fe": value})
        rows.append({"imperfection_over_t": ratio,
                     "imperfection_mm": ratio * THICKNESS * 1e3,
                     "converged": path["converged"],
                     "load_factor_fe": measured,
                     "load_factor_koiter": reference,
                     "relative_error": None if measured is None else measured / reference - 1.0,
                     "curve": curve})
    errors = [abs(row["relative_error"]) for row in rows
              if row["relative_error"] is not None]
    return {"station_total_amplitude_over_t": A3_STATION,
            "koiter_coefficient": coefficient, "cases": rows,
            "max_abs_relative_error": max(errors) if errors else float("nan"),
            "all_cases_resolved": len(errors) == len(rows)}


def criterion_a4() -> dict:
    """Stiffened-panel buckling mode vs the DNV-RP-C201 mode classification.

    DNV-RP-C201 separates a *plate panel* mode (the plate field buckles between
    the stiffeners, with the half-wave count ``m`` set by the aspect ratio, the
    stiffener line acting as a nodal line) from a *stiffener* mode (a column or
    torsional mode with the plate field following the stiffener).  The panel's
    unit cell of :mod:`tensorfem.stiffened_panel_buckling` is only *half* a bay
    wide (symmetry at the mid-bay), so the reference plate-family member is

        sin(m pi x / span) * sin(pi y / spacing),   y in [0, spacing/2],

    i.e. a quarter wave across the half bay rising from the stiffener line, and
    the reference stiffener mode is ``sin(pi x/span)`` uniform across the bay.
    The half-wave count ``m`` is *not* fixed: DNV's plate formula minimises over
    it, so the MAC is taken over the family and the winning ``m`` is reported.
    """
    from tensorfem.stiffened_panel_buckling import panel_buckling
    from tensorfem.stiffened_panel_ultimate import StiffenedPanel

    panel = StiffenedPanel(span=0.8, spacing=0.4, plate_thickness=0.006,
                           web_height=0.150, web_thickness=0.010,
                           flange_width=0.050, flange_thickness=0.010,
                           young=YOUNG, poisson=POISSON, yield_stress=YIELD).validate()
    result = panel_buckling(panel, nx=8, ny=4, modes=3)
    nodes = np.asarray(result.nodes)
    plate = np.asarray(result.plate_elements)
    on_plate = np.unique(plate.reshape(-1))
    xy = nodes[on_plate][:, :2]
    transverse = np.asarray(result.modes)[0][on_plate]
    across = np.sin(np.pi * xy[:, 1] / panel.spacing)
    plate_family = {
        int(m): float(vk.mac(transverse,
                             np.sin(m * np.pi * xy[:, 0] / panel.span) * across))
        for m in range(1, 7)
    }
    best_m = max(plate_family, key=plate_family.get)
    column_mode = np.sin(np.pi * xy[:, 0] / panel.span) * np.ones_like(xy[:, 1])
    mac_column = vk.mac(transverse, column_mode)
    # DNV-RP-C201 closed forms for the same scantlings.
    rigidity = YOUNG * panel.plate_thickness ** 3 / (12.0 * (1.0 - POISSON ** 2))
    sigma_plate = 4.0 * np.pi ** 2 * rigidity / (panel.plate_thickness
                                                 * panel.spacing ** 2)
    web, flange = panel.web_height, panel.flange_width
    web_area = panel.web_thickness * web
    flange_area = panel.flange_thickness * flange
    area = web_area + flange_area + panel.plate_thickness * panel.spacing
    stiffener_area = web_area + flange_area
    centroid = (web_area * (web / 2.0) + flange_area * (web + flange / 2.0)) / stiffener_area
    inertia = (panel.web_thickness * web ** 3 / 12.0
               + web_area * (web / 2.0 - centroid) ** 2
               + panel.flange_thickness * flange ** 3 / 12.0
               + flange_area * (web + flange / 2.0 - centroid) ** 2)
    sigma_column_euler = np.pi ** 2 * YOUNG * inertia / (area * panel.span ** 2)
    # DNV-RP-C201 caps every elastic buckling stress by yield; for this slender
    # tee the elastic Euler stress is far above it, so the cap is what the
    # classification has to use.
    sigma_column = min(sigma_column_euler, YIELD)
    governing = "plate" if sigma_plate < sigma_column else "column"
    dominant = "plate" if plate_family[best_m] > mac_column else "column"
    return {"panel": {"span_m": panel.span, "spacing_m": panel.spacing,
                      "plate_thickness_m": panel.plate_thickness,
                      "web_mm": [web * 1e3, panel.web_thickness * 1e3],
                      "flange_mm": [flange * 1e3, panel.flange_thickness * 1e3]},
            "fe_critical_stress_pa": result.critical_stress,
            "fe_load_factors_pa": [float(value) for value in result.load_factors],
            "fe_mac_plate_family": {str(m): value for m, value in plate_family.items()},
            "fe_best_half_waves": int(best_m),
            "dnv_plate_buckling_stress_pa": float(sigma_plate),
            "dnv_column_euler_stress_pa": float(sigma_column_euler),
            "dnv_column_buckling_stress_pa": float(sigma_column),
            "dnv_column_capped_by_yield": bool(sigma_column_euler > YIELD),
            "dnv_governing_mode": governing,
            "mac_plate_mode": plate_family[best_m],
            "mac_column_mode": mac_column,
            "fe_dominant_mode": dominant,
            "mode_agrees_with_dnv": dominant == governing,
            "mac_of_dominant_reference": max(plate_family[best_m], mac_column)}


def criterion_a5() -> dict:
    a, b = A, B
    spectrum = vk.base_tangent_spectrum(a=a, b=b, thickness=THICKNESS, young=YOUNG,
                                        poisson=POISSON, nx=8, ny=8)
    tension = vk.linear_buckling(a=a, b=b, thickness=THICKNESS, young=YOUNG,
                                 poisson=POISSON, nx=8, ny=8, line_load=-1.0)
    return {"zero_load_tangent": spectrum,
            "tensile_load_buckling": {"buckling": tension["buckling"],
                                      "max_mapped_eigenvalue":
                                          tension["max_mapped_eigenvalue"],
                                      "rationale": tension.get("rationale", "")},
            "no_zero_energy_mode": bool(spectrum["positive_definite"]),
            "tensile_has_no_buckling_factor": not tension["buckling"]}


def profile_mac_study() -> dict:
    """How single-mode is the finite-element postbuckling shape?"""
    path = _trace(a=A, b=B, nx=8, ny=8, imperfection_over_t=NEAR_PERFECT)
    nodes = np.asarray(path["nodes"])
    centre_line = np.isclose(nodes[:, 1], 0.5 * B)
    transverse_index = 6 * np.arange(len(nodes)) + vk.W
    rows = []
    for target in (0.2, 0.4, 0.6, 0.8, 1.0):
        station = min(path["points"],
                      key=lambda point: abs(point["total_amplitude_over_t"] - target))
        field = station["displacement"][transverse_index]
        rows.append({"total_amplitude_over_t": station["total_amplitude_over_t"],
                     "load_factor": station["load_factor"],
                     "mac_whole_field": vk.nodal_shape_mac(nodes, field, a=A, b=B,
                                                          m=1, n=1),
                     "mac_centre_line": vk.mac(field[centre_line],
                                               np.sin(np.pi * nodes[centre_line, 0] / A))})
    return {"cases": rows}


def figures(a1: dict, a2: dict, a3: dict, a4: dict) -> list[Path]:
    plt = pyplot()
    produced = []

    # 1. buckling mode field -------------------------------------------------
    closed = vk.timoshenko_critical_stress(a=A, b=B, thickness=THICKNESS,
                                           young=YOUNG, poisson=POISSON)
    eigen = vk.linear_buckling(a=A, b=B, thickness=THICKNESS, young=YOUNG,
                               poisson=POISSON, nx=16, ny=16)
    nodes = np.asarray(eigen["nodes"])
    values = np.asarray(eigen["mode"])[6 * np.arange(len(nodes)) + vk.W]
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    scale = np.abs(values).max()
    contour = axes[0].tricontourf(nodes[:, 0] * 1e3, nodes[:, 1] * 1e3,
                                  values / scale, levels=20, cmap="viridis")
    axes[0].set_title(f"FE buckling mode, 16x16 mesh\n"
                      f"$\\sigma_{{cr}}$ = {eigen['critical_stress']/1e6:.2f} MPa "
                      f"(closed form {closed['sigma_cr']/1e6:.2f} MPa)")
    axes[0].set_xlabel("x [mm]"); axes[0].set_ylabel("y [mm]")
    axes[0].set_aspect("equal")
    figure.colorbar(contour, ax=axes[0], label="$w / w_{max}$")
    centre_line = np.isclose(nodes[:, 1], 0.5 * B)
    axes[1].plot(nodes[centre_line, 0] * 1e3, values[centre_line] / scale, "o-",
                 label="FE centre line")
    axes[1].plot(nodes[centre_line, 0] * 1e3,
                 np.sin(np.pi * nodes[centre_line, 0] / A), "--",
                 label=r"$\sin(\pi x/a)$")
    axes[1].set_xlabel("x [mm]"); axes[1].set_ylabel("$w / w_{max}$")
    axes[1].legend(); axes[1].set_title("Mode shape along the centre line")
    figure.tight_layout()
    path = HERE / "buckling_mode.png"
    figure.savefig(path); plt.close(figure); produced.append(path)

    # 2. postbuckling path ---------------------------------------------------
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    case = a2["cases"][-1]
    stations = [(row["total_amplitude_over_t"], row["load_factor_koiter"])
                for row in case["stations"] + case["beyond_gate_range"]]
    explicit = np.asarray(stations, dtype=float)
    fine = np.linspace(0.02, max(explicit[:, 0]), 200)
    koiter = [vk.koiter_load_factor(amplitude=value * THICKNESS, thickness=THICKNESS,
                                    imperfection=NEAR_PERFECT * THICKNESS,
                                    coefficient=a2["koiter_coefficient"])
              for value in fine]
    for entry in a2["cases"]:
        points = np.asarray([(row["total_amplitude_over_t"], row["load_factor"])
                             for row in entry["path"]], dtype=float)
        axes[0].plot(points[:, 0], points[:, 1], "o-", ms=3,
                     label=f"FE {entry['mesh'][0]}x{entry['mesh'][1]} (arc length)")
    axes[0].plot(fine, koiter, "k--", label="one-mode von Karman--Koiter")
    axes[0].set_xlabel("total amplitude $A/t$"); axes[0].set_ylabel("$N / N_{cr}$")
    axes[0].legend(); axes[0].set_title("Postbuckling path, $w_0 = t/1000$")
    deviation = [
        (row["total_amplitude_over_t"], 100.0 * row["relative_error"])
        for row in case["stations"] + case["beyond_gate_range"]
        if row["relative_error"] is not None
    ]
    deviation = np.asarray(deviation)
    axes[1].axhline(0.0, color="0.4", lw=0.8)
    axes[1].axvspan(0.0, max(A2_STATIONS), color="0.85",
                    label="A2 gate range ($A/t\\leq0.55$)")
    axes[1].plot(deviation[:, 0], deviation[:, 1], "s-", ms=3,
                 label=f"FE {case['mesh'][0]}x{case['mesh'][1]} vs one-mode")
    axes[1].set_xlabel("total amplitude $A/t$")
    axes[1].set_ylabel("load-factor deviation [%]")
    axes[1].legend(); axes[1].set_title("Where the one-mode reduction stops holding")
    figure.tight_layout()
    path = HERE / "postbuckling_path.png"
    figure.savefig(path); plt.close(figure); produced.append(path)

    # 3. imperfection sensitivity -------------------------------------------
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for row in a3["cases"]:
        curve = np.asarray([(item["total_amplitude_over_t"], item["load_factor_fe"])
                            for item in row["curve"] if item["load_factor_fe"]],
                           dtype=float)
        axes[0].plot(curve[:, 0], curve[:, 1], "o-", ms=3,
                     label=f"$w_0 = t/{int(round(1.0/row['imperfection_over_t']))}$")
        amplitude = np.linspace(min(curve[:, 0]), max(curve[:, 0]), 200)
        axes[0].plot(amplitude, [vk.koiter_load_factor(
            amplitude=value * THICKNESS, thickness=THICKNESS,
            imperfection=row["imperfection_over_t"] * THICKNESS,
            coefficient=a3["koiter_coefficient"]) for value in amplitude],
            "--", lw=0.9, color=axes[0].lines[-1].get_color())
    axes[0].set_xlabel("total amplitude $A/t$"); axes[0].set_ylabel("$N / N_{cr}$")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Imperfection sensitivity (solid FE, dashed Koiter)")
    station = a3["station_total_amplitude_over_t"]
    imperfections = [row["imperfection_over_t"] for row in a3["cases"]]
    axes[1].plot([1.0 / value for value in imperfections],
                 [row["load_factor_fe"] for row in a3["cases"]], "o-",
                 label="FE")
    axes[1].plot([1.0 / value for value in imperfections],
                 [row["load_factor_koiter"] for row in a3["cases"]], "s--",
                 label="one-mode Koiter")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("$t / w_0$"); axes[1].set_ylabel("$N / N_{cr}$")
    axes[1].set_title(f"Sensitivity at $A/t = {station}$")
    axes[1].legend()
    figure.tight_layout()
    path = HERE / "imperfection_sensitivity.png"
    figure.savefig(path); plt.close(figure); produced.append(path)
    return produced


def main() -> int:
    import time

    def timed(name: str, function):
        started = time.time()
        print(f"[p250-n06] {name} ...", flush=True)
        value = function()
        print(f"[p250-n06] {name} done in {time.time() - started:.1f} s", flush=True)
        return value

    a1 = timed("A1 linear bifurcation", criterion_a1)
    a2 = timed("A2 postbuckling path", criterion_a2)
    a3 = timed("A3 imperfection sensitivity", criterion_a3)
    a4 = timed("A4 stiffened-panel mode", criterion_a4)
    a5 = timed("A5 negative controls", criterion_a5)
    mac_study = timed("single-mode content study", profile_mac_study)

    gates = [
        Gate("A1", "max |sigma_cr_FE / sigma_cr_Timoshenko - 1| over a/b = 1,2,3",
             "Timoshenko k pi^2 D/(b^2 t)", a1["max_abs_relative_error"], 0.03),
        Gate("A2", "max |N_FE / N_koiter - 1| at 10 path stations (A/t = 0.10..0.55)",
             "one-mode von Karman-Koiter, w0 = t/1000",
             a2["max_abs_relative_error"], 0.05),
        Gate("A3", "max |N_FE / N_koiter - 1| over w0 = t/500, t/200, t/100",
             "one-mode von Karman-Koiter", a3["max_abs_relative_error"], 0.10),
        Gate("A4", "MAC of the panel eigenmode against its DNV-RP-C201 mode family",
             "DNV-RP-C201 plate/column mode shapes",
             a4["mac_of_dominant_reference"], 0.90, ">="),
        Gate("A5a", "min eigenvalue of the zero-load tangent", "positive definiteness",
             a5["zero_load_tangent"]["min_eigenvalue"], 0.0, ">="),
        Gate("A5b", "positive buckling factors under tensile load",
             "no bifurcation under tension", 0.0, 0.0),
    ]
    # A5b is a count-type gate: pass when zero positive factors exist.
    gates[-1] = Gate("A5b", "count of positive buckling factors under tension",
                     "no bifurcation under tension",
                     0.0 if a5["tensile_has_no_buckling_factor"] else 1.0, 0.0)

    artifacts = {path.name: str(path.relative_to(HERE.parents[2]))
                 for path in figures(a1, a2, a3, a4)}
    manifest = gate_manifest(
        anchor=ANCHOR, title=TITLE,
        criteria={"A1": "< 3 %", "A2": "< 5 % (10 stations)", "A3": "< 10 %",
                  "A4": "MAC > 0.9", "A5": "no zero-energy mode, no tensile buckling"},
        gates=gates, artifacts=artifacts, mission="p250-n06-structural-buckling",
        limitations=[
            "Small-rotation von Karman plate element (BFS bending + Q4 membrane); "
            "finite rotations, yielding and stiffener web local buckling are outside the model.",
            "Dead-load (uniform traction) in-plane loading with stress-free unloaded "
            "edges; a rigid end platen would raise the postbuckling load.",
            "The one-mode Koiter reference is an asymptotic in the amplitude, so the "
            "A2 gate is graded inside A/t <= 0.55; the deviation beyond that range is "
            "reported as a measurement, not hidden.",
            "The panel MAC comparison (A4) uses the transverse nodal field only, which "
            "is free of any rotation sign convention.",
        ])
    write_json(HERE / "gates_verified.json", manifest)

    payload = {
        "schema": "tensorfem.fem-plate-postbuckling.v1",
        "anchor": ANCHOR,
        "case": {"a_m": A, "b_m": B, "thickness_m": THICKNESS, "young_pa": YOUNG,
                 "poisson": POISSON, "yield_pa": YIELD},
        "A1_linear_buckling": a1,
        "A2_postbuckling_path": a2,
        "A3_imperfection_sensitivity": a3,
        "A4_stiffened_panel_mode": a4,
        "A5_negative_controls": a5,
        "single_mode_content_study": mac_study,
        "gates": manifest["gates"],
        "passed": manifest["passed"],
        "failed_gates": manifest["failed_gates"],
    }
    write_json(HERE / "result.json", payload)

    print(json.dumps({"passed": manifest["passed"],
                      "failed_gates": manifest["failed_gates"],
                      "A1": a1["max_abs_relative_error"],
                      "A2": a2["max_abs_relative_error"],
                      "A3": a3["max_abs_relative_error"],
                      "A4_mac": a4["mac_of_dominant_reference"],
                      "A5_min_eigen": a5["zero_load_tangent"]["min_eigenvalue"],
                      "artifacts": list(artifacts)}, indent=2))
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())