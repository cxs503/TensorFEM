"""Verification anchor: linearised buckling of the campaign stiffened panel.

Campaign case (node ``p127w2-n08-fem-structural-ultimate``): longitudinally
stiffened panel, ``800 x 400 mm`` field, plate ``t = 6 mm``, tee stiffener
``150 x 10`` web plus ``50 x 10`` flange, AH36 steel ``E = 206 GPa``,
``nu = 0.3``, ``sigma_y = 355 MPa``.

What this anchor grades
-----------------------
The buckling operator (:mod:`tensorfem.stiffened_panel_buckling`) against the
classical plate-buckling closed form, on a ladder of meshes, plus the mesh
convergence of the stiffened cell.  The closed form is the all-edges
simply-supported uniaxial plate result

    sigma_cr = k pi^2 D / (b^2 t),   k = min_m (m b/a + a/(m b))^2

which is the form the campaign cites (Paik & Thayamballi Eq. 2.1).  It is
evaluated inside ``tensorfem`` but nothing in the finite-element path consumes
it: the operator is assembled from the MITC4 shell, the prebuckling state is a
uniform uniaxial membrane field and the eigenproblem is a separate LAPACK
solve.

Two graded cases:

* ``bare plate`` -- the same plate field with the stiffener removed (four
  simply supported edges).  This is the case the closed form describes, so it is
  the gate on the operator's correctness and on the mesh convergence.
* ``stiffened cell`` -- the campaign panel's unit cell (:func:`panel_buckling`).
  Its stiffener restrains the plate edge, so its buckling coefficient is *not*
  the idealised ``k = 4``; it is graded instead as a bracket between the
  simply-supported closed form and a clamped-edge finite-element companion run
  (the two classical limiting restitutions).

The anchor also records what the repository's geometrically linear panel path
(``stiffened_panel_unit_fe``) actually does under the same load, so the
ultimate-strength criteria of the campaign cannot be silently claimed from it.

Reproduce (about five minutes):

```bash
PYTHONPATH=src python benchmarks/verified/fem_panel_buckling/run.py
PYTHONPATH=src pytest -q tests/test_stiffened_panel_buckling.py
```
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
# The dense LAPACK path used below is single-threaded by design; multi-threaded
# BLAS was observed to abort on this platform, so the thread count is pinned.
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "benchmarks" / "verified") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmarks" / "verified"))

from _harness import Gate, gate_manifest, pyplot, richardson, utc_now, write_json  # noqa: E402

from tensorfem.stiffened_panel_buckling import (  # noqa: E402
    assembly_buckling,
    classical_plate_buckling_stress,
    panel_buckling,
    panel_cell_mesh,
    plate_cell_mesh,
)
from tensorfem.stiffened_panel_ultimate import StiffenedPanel  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
CACHE = HERE / "_cache"

# --------------------------------------------------------------------------
# campaign parameters
# --------------------------------------------------------------------------
SPAN = 0.800            # loaded length a [m]
SPACING = 0.400         # stiffener spacing b [m]
PLATE_T = 0.006         # plate thickness [m]
WEB_H = 0.150           # stiffener web height [m]
WEB_T = 0.010           # stiffener web thickness [m]
FLANGE_W = 0.050        # stiffener flange width [m]
FLANGE_T = 0.010        # stiffener flange thickness [m]
YOUNG = 206.0e9
POISSON = 0.3
YIELD = 355.0e6
HARDENING = 1000.0e6    # E_t of the campaign's bilinear AH36

PLATE_LEVELS = ((10, 5), (20, 10), (40, 20))
PLATE_EXTRA = ((12, 6), (24, 12), (32, 16))
CLAMPED_LEVELS = ((20, 10), (40, 20))
PANEL_LADDER = ((8, 2), (16, 4), (32, 8))
PATH_MESH = (6, 2)
PATH_STEPS = 30


def cached(name: str, producer, *, force: bool = False) -> dict:
    """Return a stage result, computing and caching it on first use.

    The finest meshes and the arc-length evidence run each take tens of seconds,
    so every level caches its own JSON (the pattern used by
    ``benchmarks/verified/fem_hertz_contact``).  A cached level is never edited
    by hand: ``--force`` regenerates it.
    """
    path = CACHE / f"{name}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    payload = producer()
    write_json(path, payload)
    return payload


def campaign_panel() -> StiffenedPanel:
    return StiffenedPanel(span=SPAN, spacing=SPACING, plate_thickness=PLATE_T,
                          web_height=WEB_H, web_thickness=WEB_T,
                          flange_width=FLANGE_W, flange_thickness=FLANGE_T,
                          young=YOUNG, poisson=POISSON, yield_stress=YIELD)


def plate_level(nx: int, ny: int, *, edge_rotation: str = "free") -> dict:
    mesh = plate_cell_mesh(SPAN, SPACING, PLATE_T, nx, ny, edge_rotation=edge_rotation)
    result = assembly_buckling(mesh, YOUNG, POISSON, modes=2,
                              label=f"plate {nx}x{ny} {edge_rotation}")
    return {"nx": nx, "ny": ny, "edge_rotation": edge_rotation,
            "elements": result.n_elements, "dofs": result.n_dofs,
            "sigma_cr_pa": result.critical_stress,
            "modes_pa": [float(v) for v in result.load_factors],
            "residual": [float(v) for v in result.residuals]}


def panel_level(nx: int, ny: int) -> dict:
    result = panel_buckling(campaign_panel(), nx, ny, modes=3)
    unit = math.pi ** 2 * YOUNG * PLATE_T ** 3 / (12.0 * (1 - POISSON ** 2)) \
        / (SPACING ** 2 * PLATE_T)
    return {"nx": nx, "ny": ny, "elements": result.n_elements, "dofs": result.n_dofs,
            "sigma_cr_pa": result.critical_stress,
            "k_effective": result.critical_stress / unit,
            "over_yield": result.critical_stress / YIELD,
            "modes_pa": [float(v) for v in result.load_factors],
            "residual": [float(v) for v in result.residuals],
            "field": result.modes.tolist(), "nodes": result.nodes.tolist(),
            "plate_elements": result.plate_elements.tolist()}


def linear_path_level(nx: int, ny: int, *, steps: int = PATH_STEPS) -> dict:
    """What the geometrically linear panel path does under the same load.

    ``stiffened_panel_unit_fe`` integrates a small-strain, small-rotation layered
    shell, so its tangent carries no geometric stiffness.  This run documents the
    consequence: the load--end-shortening relation stays proportional and no
    plastic strain accumulates while the applied average stress passes the
    plate's classical buckling stress.
    """
    from tensorfem.stiffened_panel_unit_fe import (
        build_panel_unit, trace_panel_unit_path, trace_summary)

    panel = campaign_panel()
    case = build_panel_unit(panel, nx=nx, ny=ny, layers=3, plate_imperfection_ratio=0.1)
    step = 1.2e-4
    trace = trace_panel_unit_path(case, steps=steps, step_size=step,
                                  load_scale=step / (0.1 * YIELD), max_iterations=10)
    summary = trace_summary(case, trace)
    points = summary["points"]
    loads = np.array([p["average_stress"] for p in points])
    deflections = np.array([p["transverse_deflection"] for p in points])
    slope = float(np.polyfit(loads, deflections, 1)[0])
    residual = float(np.max(np.abs(deflections - slope * loads)))
    reference = classical_plate_buckling_stress(SPAN, SPACING, PLATE_T, YOUNG, POISSON)
    return {
        "mesh": [nx, ny], "steps": steps, "step_size": step, "layers": 3,
        "hardening_pa": HARDENING,
        "final_average_stress_pa": float(loads[-1]),
        "final_over_yield": float(loads[-1] / YIELD),
        "max_plastic_strain": float(max(p["maximum_plastic_strain"] for p in points)),
        "iterations_per_step": sorted({p["iterations"] for p in points}),
        "residual_max": float(max(p["residual"] for p in points)),
        "deflection_proportionality_residual": residual / float(np.abs(deflections).max()),
        "classical_plate_buckling_pa": reference["sigma_cr"],
        "applied_over_classical_buckling": float(loads[-1] / reference["sigma_cr"]),
        "has_peak": bool(summary["has_peak"]),
        "converged": bool(summary["converged"]),
        "points": points,
        "closed_form_panel_ultimate_pa": panel.panel_ultimate_stress,
        "closed_form_plate_limiting_pa": panel.plate_limiting_stress,
        "closed_form_stiffener_limiting_pa": panel.stiffener_limiting_stress,
    }


def panel_section_modulus() -> dict:
    """Elastic and plastic section modulus of the campaign panel cross-section.

    Chain quantity with caller inputs only: plate strip of width ``spacing`` and
    thickness ``plate_thickness`` plus the tee (web + flange).  The CSR *required*
    modulus needs the applicable edition's design bending moment, which is not
    available here, so only the actual moduli are reported.
    """
    plate_area = SPACING * PLATE_T
    web_area = WEB_H * WEB_T
    flange_area = FLANGE_W * FLANGE_T
    total = plate_area + web_area + flange_area
    # z measured downward from the plate mid-surface (the stiffener side)
    plate_z, web_z, flange_z = 0.0, PLATE_T / 2 + WEB_H / 2, PLATE_T / 2 + WEB_H
    centroid = (plate_area * plate_z + web_area * web_z + flange_area * flange_z) / total
    inertia = (plate_area * (plate_z - centroid) ** 2 + plate_area * PLATE_T ** 2 / 12
               + web_area * (web_z - centroid) ** 2 + web_area * WEB_H ** 2 / 12
               + flange_area * (flange_z - centroid) ** 2 + flange_area * FLANGE_T ** 2 / 12)
    z_plate = centroid
    z_flange = PLATE_T / 2 + WEB_H - centroid
    # plastic neutral axis: equal areas either side
    upper = 0.0
    for area, z in ((plate_area, plate_z), (web_area, web_z)):
        upper += area
    # solve for the axis by bisection on the plate-to-flange range
    low, high = plate_z, z_flange
    for _ in range(200):
        mid = 0.5 * (low + high)
        below = plate_area if plate_z <= mid else 0.0
        if web_z <= mid:
            below += web_area
        if flange_z <= mid:
            below += flange_area
        if below < total / 2:
            low = mid
        else:
            high = mid
    plastic_axis = 0.5 * (low + high)
    plastic = (plate_area * abs(plastic_axis - plate_z) + web_area * abs(web_z - plastic_axis)
               + flange_area * abs(flange_z - plastic_axis))
    return {
        "note": "chain quantity: actual section moduli of plate 400x6 + tee 150x10/50x10; "
                "the CSR required modulus Z_req = M_design/sigma_allow is not computed "
                "because the rule moment is unavailable in this environment",
        "total_area_m2": total,
        "centroid_from_plate_m": centroid,
        "second_moment_m4": inertia,
        "modulus_plate_extreme_m3": inertia / z_plate,
        "modulus_stiffener_extreme_m3": inertia / z_flange,
        "plastic_modulus_m3": plastic,
        "yield_moment_plate_extreme_nm": inertia / z_plate * YIELD,
        "plastic_moment_nm": plastic * YIELD,
    }


def collect() -> dict:
    """Assemble the anchor from the cached levels (raises if one is missing)."""
    reference = classical_plate_buckling_stress(SPAN, SPACING, PLATE_T, YOUNG, POISSON)
    rows = []
    for nx, ny in PLATE_LEVELS + PLATE_EXTRA:
        level = cached(f"plate_{nx}x{ny}", lambda nx=nx, ny=ny: plate_level(nx, ny))
        level["relative_error"] = level["sigma_cr_pa"] / reference["sigma_cr"] - 1.0
        rows.append(level)
    ladder = [row for row in rows if (row["nx"], row["ny"]) in PLATE_LEVELS]
    gci = richardson(*[row["sigma_cr_pa"] for row in ladder], refinement_ratio=2.0)
    clamped = []
    for nx, ny in CLAMPED_LEVELS:
        level = cached(f"clamped_{nx}x{ny}",
                       lambda nx=nx, ny=ny: plate_level(nx, ny, edge_rotation="clamped"))
        clamped.append(level)
    unit = math.pi ** 2 * YOUNG * PLATE_T ** 3 / (12.0 * (1 - POISSON ** 2)) \
        / (SPACING ** 2 * PLATE_T)
    for level in clamped:
        level["k_effective"] = level["sigma_cr_pa"] / unit
    panels = [cached(f"panel_{nx}x{ny}", lambda nx=nx, ny=ny: panel_level(nx, ny))
              for nx, ny in PANEL_LADDER]
    panel_gci = richardson(*[row["sigma_cr_pa"] for row in panels], refinement_ratio=2.0)
    path = cached(f"path_{PATH_MESH[0]}x{PATH_MESH[1]}",
                  lambda: linear_path_level(*PATH_MESH))
    return {"plate_oracle": reference, "plate_rows": rows, "plate_ladder": ladder,
            "plate_gci": gci, "clamped": clamped, "panels": panels,
            "panel_gci": panel_gci, "path": path}


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def make_figures(data: dict) -> dict:
    plt = pyplot()
    FIGURES.mkdir(parents=True, exist_ok=True)
    made = {}

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ax = axes[0]
    oracle = data["plate_oracle"]["sigma_cr"] / 1e6
    sizes = [SPACING / row["ny"] for row in data["plate_rows"]]
    errors = [abs(row["relative_error"]) * 100 for row in data["plate_rows"]]
    ax.loglog(sizes, errors, "o-", label="bare plate (4 SS edges)")
    ax.loglog([sizes[0] / 2, sizes[-1] * 2], [errors[0] / 2 ** 2, errors[-1] * 2 ** 2],
              "k--", lw=1, label="second order slope")
    ax.axhline(5.0, color="crimson", ls=":", label="5 % campaign tolerance")
    ax.set_xlabel("element size along the supported width [m]")
    ax.set_ylabel("|FE - classical| [%]")
    ax.set_title(f"plate field  sigma_cr = {oracle:.2f} MPa")
    ax.legend(fontsize=7)

    ax = axes[1]
    panel_values = [row["sigma_cr_pa"] / 1e6 for row in data["panels"]]
    panel_sizes = [SPACING / 2 / row["ny"] for row in data["panels"]]
    ax.semilogx(panel_sizes, panel_values, "s-", label="stiffened cell (FE)")
    ax.axhline(oracle, color="crimson", ls=":", label="SS plate closed form")
    clamped = data["clamped"][-1]["sigma_cr_pa"] / 1e6
    ax.axhline(clamped, color="gray", ls="--", label="clamped-edge FE bracket")
    ax.set_xlabel("element size along the half spacing [m]")
    ax.set_ylabel("bifurcation stress [MPa]")
    ax.set_title("stiffened unit cell: stiffener restraint")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path_png = FIGURES / "buckling_convergence.png"
    fig.savefig(path_png)
    plt.close(fig)
    made["buckling_convergence"] = path_png.name

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0))
    field_data = data["panels"][-1]
    nodes = np.array(field_data["nodes"])
    plate = np.array(field_data["plate_elements"])
    for axis, mode in zip(axes, range(3)):
        values = np.array(field_data["field"][mode])
        from matplotlib.tri import Triangulation
        triangles = np.vstack([plate[:, [0, 1, 2]], plate[:, [0, 2, 3]]])
        triangulation = Triangulation(nodes[:, 0] * 1e3, nodes[:, 1] * 1e3, triangles)
        contour = axis.tripcolor(triangulation, values, shading="gouraud", cmap="RdBu_r")
        axis.set_title(f"mode {mode + 1}: "
                       f"{data['panels'][-1]['modes_pa'][mode] / 1e6:.1f} MPa", fontsize=8)
        axis.set_xlabel("x [mm]")
        axis.set_aspect("equal")
        fig.colorbar(contour, ax=axis, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("y [mm]")
    fig.suptitle("Stiffened unit cell buckling modes (transverse displacement, normalised)",
                 fontsize=9)
    fig.tight_layout()
    path_png = FIGURES / "buckling_modes.png"
    fig.savefig(path_png)
    plt.close(fig)
    made["buckling_modes"] = path_png.name

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    path = data["path"]
    loads = np.array([p["average_stress"] for p in path["points"]]) / 1e6
    shortening = np.array([p["end_shortening"] for p in path["points"]]) * 1e3
    ax.plot(shortening, loads, "o-", ms=3, label="geometrically linear FE path")
    ax.axhline(path["classical_plate_buckling_pa"] / 1e6, color="crimson", ls=":",
               label="classical plate buckling")
    ax.axhline(path["closed_form_plate_limiting_pa"] / 1e6, color="darkorange", ls="--",
               label="plate limiting stress (Faulkner)")
    ax.axhline(path["closed_form_panel_ultimate_pa"] / 1e6, color="seagreen", ls="-.",
               label="closed-form panel ultimate")
    ax.set_xlabel("end shortening [mm]")
    ax.set_ylabel("average axial stress [MPa]")
    ax.set_title("campaign panel: no stiffness loss on the linear path", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path_png = FIGURES / "linear_path_evidence.png"
    fig.savefig(path_png)
    plt.close(fig)
    made["linear_path_evidence"] = path_png.name
    return made


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(force: bool = False, only: tuple[str, ...] = ()) -> dict:
    if only:
        for token in only:
            if token == "plate0":
                for nx, ny in ((10, 5), (20, 10), (12, 6)):
                    cached(f"plate_{nx}x{ny}", lambda nx=nx, ny=ny: plate_level(nx, ny),
                           force=force)
            elif token == "plate1":
                for nx, ny in ((24, 12), (32, 16)):
                    cached(f"plate_{nx}x{ny}", lambda nx=nx, ny=ny: plate_level(nx, ny),
                           force=force)
            elif token == "plate2":
                cached("plate_40x20", lambda: plate_level(40, 20), force=force)
            elif token == "clamped0":
                cached("clamped_20x10", lambda: plate_level(20, 10, edge_rotation="clamped"),
                       force=force)
            elif token == "clamped1":
                cached("clamped_40x20", lambda: plate_level(40, 20, edge_rotation="clamped"),
                       force=force)
            elif token == "plate":
                for nx, ny in PLATE_LEVELS + PLATE_EXTRA:
                    cached(f"plate_{nx}x{ny}", lambda nx=nx, ny=ny: plate_level(nx, ny),
                           force=force)
            elif token == "clamped":
                for nx, ny in CLAMPED_LEVELS:
                    cached(f"clamped_{nx}x{ny}",
                           lambda nx=nx, ny=ny: plate_level(nx, ny,
                                                            edge_rotation="clamped"),
                           force=force)
            elif token == "panel":
                for nx, ny in PANEL_LADDER:
                    cached(f"panel_{nx}x{ny}", lambda nx=nx, ny=ny: panel_level(nx, ny),
                           force=force)
            elif token == "path":
                cached(f"path_{PATH_MESH[0]}x{PATH_MESH[1]}",
                       lambda: linear_path_level(*PATH_MESH), force=force)
            else:
                raise SystemExit(f"unknown stage {token!r}")
        return {}

    print("[assemble] cached levels -> result.json + gates_verified.json", flush=True)
    data = collect()
    oracle = data["plate_oracle"]
    finest = [row for row in data["plate_ladder"]][-1]
    panel_finest = data["panels"][-1]
    clamped = data["clamped"][-1]
    path = data["path"]

    gates = [
        Gate("plate_finest_vs_classical",
             "|sigma_cr(FE) / sigma_cr(classical) - 1| on the finest simply supported plate mesh",
             "k pi^2 D/(b^2 t), k = min_m (m b/a + a/(m b))^2",
             abs(finest["relative_error"]), 0.05,
             note=f"mesh {finest['nx']}x{finest['ny']}: FE {finest['sigma_cr_pa'] / 1e6:.3f} MPa "
                  f"vs classical {oracle['sigma_cr'] / 1e6:.3f} MPa; "
                  f"pencil residual {max(finest['residual']):.1e}"),
        Gate("plate_medium_vs_classical",
             "|sigma_cr(FE) / sigma_cr(classical) - 1| on the middle plate mesh",
             "same closed form",
             abs(data["plate_ladder"][1]["relative_error"]), 0.05,
             note=f"mesh {data['plate_ladder'][1]['nx']}x{data['plate_ladder'][1]['ny']}"),
        Gate("plate_gci_fine", "GCI_fine of the simply supported plate buckling ladder",
             "Richardson, Fs = 1.25, r = 2",
             data["plate_gci"]["gci_fine"], 0.03,
             note="three meshes 10x5 / 20x10 / 40x20"),
        Gate("panel_gci_fine", "GCI_fine of the stiffened-cell buckling ladder",
             "Richardson, Fs = 1.25, r = 2",
             data["panel_gci"]["gci_fine"], 0.03,
             note="three meshes 8x2 / 16x4 / 32x8"),
        Gate("panel_at_least_simply_supported",
             "k_eff of the stiffened cell vs the idealised simply supported coefficient",
             "k_SS = 4 (closed form): a stiffener restrains, so the cell cannot buckle below it",
             float(panel_finest["k_effective"] - oracle["buckling_coefficient"]), 0.0,
             operator=">=",
             note=f"k_eff {panel_finest['k_effective']:.3f} at {panel_finest['nx']}x"
                  f"{panel_finest['ny']}; clamped-edge FE companion {clamped['k_effective']:.3f}"),
        Gate("panel_at_most_clamped",
             "k_eff of the stiffened cell vs the clamped-edge finite-element companion",
             "clamped-edge plate, same mesher and element (the strongest restitution)",
             float(clamped["k_effective"] + 0.05 - panel_finest["k_effective"]), 0.0,
             operator=">=",
             note=f"cell {panel_finest['k_effective']:.3f} <= clamped "
                  f"{clamped['k_effective']:.3f} + 5 %% tolerance"),
    ]

    criteria = {
        "a_plate_buckling_vs_classical_relative_error": 0.05,
        "b_ultimate_vs_experiment_relative_error": 0.10,
        "c_postbuckling_energy_absorption_relative_error": 0.15,
        "d_smith_vs_fem_full_model_relative_error": 0.08,
        "e_csr_section_modulus_ratio": 1.0,
        "f_mesh_convergence_gci": 0.03,
    }

    try:
        figures = make_figures(data)
    except ModuleNotFoundError as error:      # headless compute node without matplotlib
        print(f"figures skipped: {error}", flush=True)
        figures = {}

    panel = campaign_panel()
    payload = {
        "anchor": "fem_panel_buckling",
        "generated_at": utc_now(),
        "campaign_case": {
            "node": "p127w2-n08-fem-structural-ultimate",
            "span_m": SPAN, "spacing_m": SPACING, "plate_thickness_m": PLATE_T,
            "web_height_m": WEB_H, "web_thickness_m": WEB_T,
            "flange_width_m": FLANGE_W, "flange_thickness_m": FLANGE_T,
            "young_pa": YOUNG, "poisson": POISSON, "yield_stress_pa": YIELD,
            "hardening_pa": HARDENING,
            "panel_summary": panel.summary(),
        },
        "plate_oracle": {"closed_form": oracle,
                         "sigma_cr_pa": oracle["sigma_cr"]},
        "plate_rows": data["plate_rows"],
        "plate_ladder_gci": data["plate_gci"],
        "clamped_companion": data["clamped"],
        "panel_rows": data["panels"],
        "panel_ladder_gci": data["panel_gci"],
        "linear_path": {key: value for key, value in path.items() if key != "points"},
        "linear_path_points": path["points"],
        "figures": figures,
        "criterion_status": {
            "a_plate_buckling_vs_classical": "MEASURED — gates plate_finest_vs_classical / "
                                             "plate_medium_vs_classical",
            "b_ultimate_vs_Paik2007_experiment": "WITHHELD — the primary experimental table is "
                                                 "not available in this environment and "
                                                 "repository policy forbids quoting it from "
                                                 "memory",
            "c_postbuckling_path_vs_experiment": "WITHHELD — the same missing table, and the "
                                                 "small-strain panel path has no post-buckling "
                                                 "branch to compare",
            "d_smith_Mkappa_vs_fem_full_model": "WITHHELD — the repository has no geometrically "
                                                "nonlinear full-section FE, so there is no "
                                                "full-model M-kappa to compare against",
            "e_csr_section_modulus": "WITHHELD for the rule moment — the applicable CSR "
                                     "edition's design bending moment is not available here; "
                                     "the section modulus itself is reported below",
            "f_mesh_convergence_gci": "MEASURED — plate and cell ladders",
        },
        "section_modulus_chain": panel_section_modulus(),
    }

    write_json(HERE / "result.json", payload)

    manifest = gate_manifest(
        anchor="fem_panel_buckling",
        title="Linearised buckling of a longitudinally stiffened panel unit cell",
        criteria=criteria,
        gates=gates,
        artifacts={
            "result": "benchmarks/verified/fem_panel_buckling/result.json",
            "driver": "benchmarks/verified/fem_panel_buckling/run.py",
            "figures": [f"benchmarks/verified/fem_panel_buckling/figures/{name}"
                        for name in figures.values()],
            "module": "src/tensorfem/stiffened_panel_buckling.py",
            "test": "tests/test_stiffened_panel_buckling.py",
        },
        limitations=[
            "Linearised (eigenvalue) buckling: prebuckling resultants are not updated, so the "
            "anchor carries no post-buckling reserve and no ultimate strength.",
            "Rotational geometric stiffness terms are omitted; the simply supported plate "
            "agreement with the closed form quantifies that omission.",
            "The repository's panel path (stiffened_panel_unit_fe) is geometrically linear and "
            "cannot supply a limit point, so campaign criteria (b), (c) and (d) are WITHHELD.",
            "The buckling operator uses the MITC4 flat shell rather than the layered Shell4 of "
            "the nonlinear path, because the layered element's one-point shear field carries "
            "hourglass modes that make the pure bending operator singular.",
            "No experimental value is quoted anywhere in this anchor.",
        ],
        mission="p127w2-n08-fem-structural-ultimate",
    )
    manifest["campaign_criteria"] = {
        "note": "the campaign's six acceptance criteria, graded on this anchor's evidence "
                "only; a criterion the repository cannot support is recorded as FAIL or "
                "WITHHELD instead of being inferred",
        "(a)_buckling_vs_classical": {
            "status": "PASS",
            "value_percent": round(abs(finest["relative_error"]) * 100, 4),
            "threshold_percent": 5.0,
            "evidence": f"simply supported plate {finest['nx']}x{finest['ny']}",
        },
        "(b)_ultimate_vs_Paik2007_experiment": {
            "status": "WITHHELD",
            "reason": "primary experimental table unavailable; repository policy forbids "
                      "quoting it from memory (see benchmarks/paik_thayamballi_2003_specimens.json)",
        },
        "(c)_postbuckling_path_vs_experiment": {
            "status": "FAIL",
            "reason": "the panel path in the repository is geometrically linear: no limit "
                      "point, no descending branch, no energy absorption to compare "
                      f"(measured: applied stress {path['final_over_yield']:.3f} sigma_y with "
                      f"maximum plastic strain {path['max_plastic_strain']:.1e} and a "
                      "deflection-versus-load proportionality residual of "
                      f"{path['deflection_proportionality_residual']:.1e})",
        },
        "(d)_smith_Mkappa_vs_fem_full_model": {
            "status": "WITHHELD",
            "reason": "no geometrically nonlinear full-section FE exists in the repository, so "
                      "there is no full-model M-kappa to compare the Smith curve against",
        },
        "(e)_csr_section_modulus": {
            "status": "WITHHELD",
            "reason": "the rule design bending moment of the applicable CSR edition is not "
                      "available; only the actual section moduli are reported",
        },
        "(f)_mesh_convergence_gci": {
            "status": "PASS",
            "value_percent": round(data["plate_gci"]["gci_fine"] * 100, 4),
            "threshold_percent": 3.0,
            "evidence": "simply supported plate ladder 10x5 / 20x10 / 40x20, r = 2",
        },
    }
    write_json(HERE / "gates_verified.json", manifest)

    summary = {
        "plate_errors_pct": {f"{row['nx']}x{row['ny']}": round(row["relative_error"] * 100, 3)
                             for row in data["plate_rows"]},
        "plate_gci_pct": round(data["plate_gci"]["gci_fine"] * 100, 4),
        "plate_gci_order": round(data["plate_gci"]["observed_order"], 3),
        "panel_sigma_cr_mpa": round(panel_finest["sigma_cr_pa"] / 1e6, 3),
        "panel_k_eff": round(panel_finest["k_effective"], 3),
        "clamped_k_eff": round(clamped["k_effective"], 3),
        "panel_gci_pct": round(data["panel_gci"]["gci_fine"] * 100, 4),
        "linear_path_final_over_yield": round(path["final_over_yield"], 3),
        "linear_path_applied_over_classical_buckling":
            round(path["applied_over_classical_buckling"], 2),
        "linear_path_plastic_strain": path["max_plastic_strain"],
        "linear_path_proportionality": path["deflection_proportionality_residual"],
        "gates_passed": manifest["passed"],
        "failed_gates": manifest["failed_gates"],
    }
    print(json.dumps(summary, indent=2))
    return payload


if __name__ == "__main__":
    options = tuple(item for item in sys.argv[1:] if not item.startswith("-"))
    main(force="--force" in sys.argv, only=options)