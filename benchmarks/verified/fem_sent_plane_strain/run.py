#!/usr/bin/env python3
"""``CASE-FEM-JINTEGRAL-EDGE-CRACK`` · plane-strain leg at ``a/W = 0.3``.

Why this driver exists (gap it closes)
-------------------------------------
The anchor case declares, verbatim::

    指标: 归一化 K_I/(sigma*sqrt(pi*a)) vs 手册有限宽修正 (a/W=0.3, 平面应变)

i.e. **plane strain** at **a/W = 0.3**.  The merged verification
(``benchmarks/verified/fem_sent_j_integral``, commit ``a3bf4886``) exercises
``a/W = 0.20 / 0.35 / 0.50`` **in plane stress** only: it enters
``elasticity_matrix(..., "stress")`` and converts with ``E' = E``.  That leaves
the declared configuration literally untested, and with it the one code path
where a small slip is *nearly invisible*: under plane strain
``E' = E / (1 - nu^2)``, so a plane-stress effective modulus used by mistake
biases ``K_I`` by ``1/sqrt(1 - 0.3^2) - 1 = +4.8 %`` -- just inside the 5 %
tolerance of the case.  A tolerance-only test would pass such a defect; this
driver therefore grades four independent statements, not one number.

Physics
-------
Single-edge-notched tension (SENT) plate, width ``W = 100 mm``, edge crack
``a = 0.3 W``, half height ``H = 2 W``, uniform far-field tension
``sigma = 100 MPa``, ``E = 210 GPa``, ``nu = 0.3``, **plane strain**,
thickness ``t = 1 m``, small strain, stationary mode-I:

    K_I = sigma sqrt(pi a) F(a/W),      K_I = sqrt(J E'),   E' = E/(1-nu^2)

Two *independent* handbook references are carried so that the finite-element
error can be separated from reference error:

* ``F_GS``  = ``1.12 - 0.231 a + 10.55 a^2 - 21.72 a^3 + 30.39 a^4`` (Tada,
  Paris & Irwin, *The Stress Analysis of Cracks Handbook* 3rd ed., quoted
  +-0.5 % for ``a/W <= 0.6``; the anchor's reference);
* ``F_CF``  = ``sqrt((2/(pi a)) tan(pi a/2)) (0.752 + 2.02 a + 0.37
  (1 - sin(pi a/2))^3) / cos(pi a/2)`` (secant form of the same chapter, also
  Anderson *Fracture Mechanics* table 2.1).

They differ by 0.29 % at ``a/W = 0.3``: that spread, not zero, is the true
uncertainty of the reference.

Four independent statements graded here
---------------------------------------
1. **Accuracy** -- ``K_I`` from the contour J-integral versus the handbook
   (5 % = the case tolerance, 3 % = the mission standard), with a 3-level
   refinement ladder and Roache GCI.
2. **Path independence** -- 8 fan lines between ``r/a = 0.06`` and ``0.33``.
3. **Material invariance** -- LEFM gives ``K_I`` independent of the elastic
   constants, so ``F`` recovered at ``nu = 0.30`` and ``nu = 0.45`` must agree
   even though ``J`` itself does *not*.  This is the sharpest available probe of
   the ``E'``/strain-energy-density bookkeeping under plane strain.
4. **Independent oracle** -- ``G = -(2/t) dPi_half/da`` from the global total
   potential (no crack-tip contour, no ``r^-1/2`` extraction), converted with the
   same ``E'``.

Outputs (this directory): ``result.json``, ``gates_verified.json`` and the
figures listed in ``ARTIFACTS``.

Usage (stages are cached, so they can be run in separate calls)::

    PYTHONPATH=src:benchmarks/verified python benchmarks/verified/fem_sent_plane_strain/run.py --stage ladder
    PYTHONPATH=src:benchmarks/verified python benchmarks/verified/fem_sent_plane_strain/run.py --stage control
    PYTHONPATH=src:benchmarks/verified python benchmarks/verified/fem_sent_plane_strain/run.py --stage nu
    PYTHONPATH=src:benchmarks/verified python benchmarks/verified/fem_sent_plane_strain/run.py --stage energy
    PYTHONPATH=src:benchmarks/verified python benchmarks/verified/fem_sent_plane_strain/run.py --stage field
    PYTHONPATH=src:benchmarks/verified python benchmarks/verified/fem_sent_plane_strain/run.py --stage assemble
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "verified"))

from tensorfem.edge_crack_sent import (  # noqa: E402
    DEFAULT_SENT_LEVELS,
    edge_geometry,
    edge_parent,
    element_field_at,
    j_integral_subring,
    sent_convergence,
    sent_gci,
    sent_k_reference,
    sent_t6_mesh,
    solve_sent,
    tada_sent_shape_factor,
    tada_sent_shape_factor_closed_form,
)
from tensorfem.continuum import elasticity_matrix  # noqa: E402
from tensorfem.sparse_core import assemble_coo  # noqa: E402
from tensorfem.fatigue.crack_tip_quarter_point import t6_stiffness  # noqa: E402

from _harness import Gate, gate_manifest, pyplot, utc_now, write_json  # noqa: E402

ANCHOR = "fem_sent_plane_strain"
TITLE = "SENT plate at a/W = 0.3 in plane strain: K_I from the J-integral against two handbook fits"
MISSION = "CASE-FEM-JINTEGRAL-EDGE-CRACK"

WIDTH = 0.10
ALPHA = 0.3
HEIGHT_RATIO = 2.0
SIGMA = 100e6
YOUNG = 210e9
NU = 0.3
PLANE = "strain"
THICKNESS = 1.0

CONTOUR_FRACTIONS = (0.06, 0.09, 0.12, 0.15, 0.18, 0.22, 0.27, 0.33)
CANONICAL_CONTOUR = 0.15
LEVELS = DEFAULT_SENT_LEVELS
MEDIUM = DEFAULT_SENT_LEVELS[1]
FINE = DEFAULT_SENT_LEVELS[2]
NU_COMPARE = (0.30, 0.45)
NU_RTOL = 1e-11  # relaxed from the 1e-12 default: see stage_nu
STEP = 0.01  # central-difference step in a/W units for dPi/da

TOL_CASE = 0.05        # the case tolerance
TOL_MISSION = 0.03     # the mission standard
TOL_PATH = 0.01        # contour path spread
TOL_NU = 0.005         # material invariance of the recovered factor
TOL_REFERENCE = 0.01   # spread between the two handbook fits
TOL_BALANCE = 1e-9
TOL_CLAPEYRON = 1e-6

CACHE = HERE / "_cache"

ARTIFACTS = {
    "result": "result.json",
    "gates": "gates_verified.json",
    "sent_plane_strain_convergence.png": "|K_FEM - K_GS| / K_GS per refinement level, with the 5 % case and 3 % mission lines.",
    "sent_plane_strain_paths.png": "Recovered F(a/W) along eight fan lines for the three levels (path independence).",
    "sent_plane_strain_handbook.png": "Recovered F against the two independent handbook fits over a/W in [0.2, 0.5].",
    "sent_plane_strain_nu.png": "F at nu = 0.30 versus nu = 0.45 (LEFM material invariance) with J quoted separately.",
    "sent_plane_strain_energy.png": "K_J versus the global energy-release oracle and the handbook value.",
    "sent_plane_strain_ligament.png": "sigma_yy along the ligament ahead of the tip against the r^-1/2 singular form.",
}


def cache_path(stage: str) -> Path:
    return CACHE / f"{stage}.json"


def e_prime(nu: float = NU) -> float:
    return YOUNG if PLANE == "stress" else YOUNG / (1.0 - nu ** 2)


def k_reference(alpha: float = ALPHA, nu: float = NU) -> dict:
    a = alpha * WIDTH
    f_gs = tada_sent_shape_factor(alpha)
    f_cf = tada_sent_shape_factor_closed_form(alpha)
    return {
        "a_over_w": alpha,
        "crack_length_m": a,
        "shape_factor_gs": f_gs,
        "shape_factor_closed_form": f_cf,
        "shape_factor_spread": (f_cf - f_gs) / f_gs,
        "k_gs_pa_sqrt_m": sent_k_reference(SIGMA, a, WIDTH),
        "e_prime_pa": e_prime(nu),
    }


def _mesh(alpha: float, level: dict, *, quarter_point: bool = True):
    return sent_t6_mesh(
        WIDTH,
        alpha * WIDTH,
        HEIGHT_RATIO * WIDTH,
        n_sectors=level["n_sectors"],
        first_ring=level["first_ring"],
        growth=level["growth"],
        quarter_point=quarter_point,
    )


def _canonical(case: dict) -> dict:
    return min(
        case["contours"],
        key=lambda c: abs(c["requested_fraction"] - CANONICAL_CONTOUR),
    )


# --------------------------------------------------------------------------- #
# stage: refinement ladder (plane strain, a/W = 0.3)
# --------------------------------------------------------------------------- #
def stage_ladder() -> dict:
    ladder = sent_convergence(
        width_m=WIDTH,
        crack_ratios=(ALPHA,),
        height_ratio=HEIGHT_RATIO,
        sigma_pa=SIGMA,
        young_pa=YOUNG,
        poisson=NU,
        plane=PLANE,
        contour_fractions=CONTOUR_FRACTIONS,
        levels=LEVELS,
        quarter_point=True,
    )
    ladder["note"] = "plane strain, a/W = 0.3, quarter-point singular ring"
    write_json(cache_path("ladder"), ladder)
    for level in ladder["levels"]:
        case = level["cases"][0]
        best = _canonical(case)
        print(
            f"{level['level']:>6s}  n_elem={case['n_elements']:6d}  "
            f"F={best['shape_factor']:.5f}  err={100 * best['rel_error_vs_tada']:+.3f} %  "
            f"reaction-1={case['reaction_ratio'] - 1.0:+.2e}",
            flush=True,
        )
    return ladder


# --------------------------------------------------------------------------- #
# stage: control without the quarter-point ring
# --------------------------------------------------------------------------- #
def stage_control() -> dict:
    control = sent_convergence(
        width_m=WIDTH,
        crack_ratios=(ALPHA,),
        height_ratio=HEIGHT_RATIO,
        sigma_pa=SIGMA,
        young_pa=YOUNG,
        poisson=NU,
        plane=PLANE,
        contour_fractions=CONTOUR_FRACTIONS,
        levels=(DEFAULT_SENT_LEVELS[0],),
        quarter_point=False,
    )
    control["note"] = "coarse level with a straight-sided (non-quarter-point) tip ring"
    write_json(cache_path("control"), control)
    best = _canonical(control["levels"][0]["cases"][0])
    print(f"control (no quarter point): F={best['shape_factor']:.5f} "
          f"err={100 * best['rel_error_vs_tada']:+.3f} %", flush=True)
    return control


# --------------------------------------------------------------------------- #
# stage: material invariance (nu = 0.30 versus 0.45) at two levels
# --------------------------------------------------------------------------- #
def _recover(nu: float, level: dict) -> dict:
    mesh = _mesh(ALPHA, level)
    # nu = 0.45 in plane strain is close to the incompressible limit
    # (1 - 2 nu = 0.10): the assembled stiffness loses conditioning and the
    # default rtol = 1e-12 is not reached (CG stalls at 3.2e-12).  Every row
    # uses the same, relaxed criterion so the comparison stays fair.
    solved = solve_sent(
        mesh, sigma_pa=SIGMA, young_pa=YOUNG, poisson=nu, plane=PLANE,
        thickness=THICKNESS, rtol=NU_RTOL,
    )
    fractions = list(mesh.stations)
    index = min(
        range(1, len(fractions) - 1),
        key=lambda i: abs(fractions[i] - CANONICAL_CONTOUR),
    )
    record = j_integral_subring(
        mesh, solved.displacement, index,
        young_pa=YOUNG, poisson=nu, plane=PLANE,
    )
    k = math.sqrt(record["j_j_m2"] * e_prime(nu))
    return {
        "level": level["name"],
        "poisson": nu,
        "e_prime_pa": e_prime(nu),
        "station_fraction": record["station_fraction"],
        "j_j_m2": record["j_j_m2"],
        "k_pa_sqrt_m": k,
        "shape_factor": k / (SIGMA * math.sqrt(math.pi * ALPHA * WIDTH)),
        "reaction_ratio": solved.reaction_ratio,
        "n_elements": mesh.n_elements,
    }


def _ladder_fine_reference() -> dict | None:
    """The nu = 0.30 fine-level row, already computed by ``--stage ladder``."""
    path = cache_path("ladder")
    if not path.exists():
        return None
    ladder = json.loads(path.read_text())
    for level in ladder["levels"]:
        if level["level"] == "fine":
            best = _canonical(level["cases"][0])
            return {
                "level": "fine",
                "poisson": NU,
                "e_prime_pa": e_prime(),
                "station_fraction": best["station_fraction"],
                "j_j_m2": best["j_j_m2"],
                "k_pa_sqrt_m": best["k_pa_sqrt_m"],
                "shape_factor": best["shape_factor"],
                "reaction_ratio": level["cases"][0]["reaction_ratio"],
                "n_elements": level["cases"][0]["n_elements"],
                "source": "ladder cache (identical mesh and contour rule)",
            }
    return None


def stage_nu() -> dict:
    rows = []
    for level in (MEDIUM, FINE):
        for nu in NU_COMPARE:
            reused = _ladder_fine_reference() if (level["name"] == "fine" and nu == NU) else None
            row = reused if reused is not None else _recover(nu, level)
            rows.append(row)
            print(f"{level['name']:>6s}  nu={nu:.2f}  J={row['j_j_m2']:.6e}  "
                  f"F={row['shape_factor']:.5f}", flush=True)
    spreads = {}
    for level in (MEDIUM, FINE):
        pair = {row["poisson"]: row for row in rows if row["level"] == level["name"]}
        if len(pair) == 2:
            spreads[level["name"]] = abs(
                pair[NU_COMPARE[1]]["shape_factor"] / pair[NU_COMPARE[0]]["shape_factor"] - 1.0
            )
    payload = {
        "rows": rows,
        "spreads": spreads,
        "levels": [dict(MEDIUM), dict(FINE)],
        "plane": PLANE,
        "cg_rtol": NU_RTOL,
        "note": "LEFM makes K_I independent of the elastic constants; a mesh-independent E' slip would sit at 1/sqrt(1-nu^2) = +4.83 %, a discretisation artefact shrinks with refinement",
    }
    write_json(cache_path("nu"), payload)
    return payload


# --------------------------------------------------------------------------- #
# stage: independent global-energy oracle, G = -(2/t) dPi_half/da
# --------------------------------------------------------------------------- #
def _loaded_force(mesh) -> torch.Tensor:
    from tensorfem.edge_crack_sent import _loaded_edge_loads  # noqa: PLC0415

    return _loaded_edge_loads(mesh, SIGMA)


def _potential(alpha: float) -> dict:
    mesh = _mesh(alpha, MEDIUM)
    solved = solve_sent(
        mesh, sigma_pa=SIGMA, young_pa=YOUNG, poisson=NU, plane=PLANE,
        thickness=THICKNESS,
    )
    dmat = elasticity_matrix(
        torch.tensor([YOUNG], dtype=mesh.nodes.dtype),
        torch.tensor([NU], dtype=mesh.nodes.dtype),
        PLANE,
    )[0]
    stiffness = t6_stiffness(mesh.nodes[mesh.elements], dmat, THICKNESS)
    element_dofs = torch.empty((mesh.n_elements, 12), dtype=torch.long)
    element_dofs[:, 0::2] = 2 * mesh.elements
    element_dofs[:, 1::2] = 2 * mesh.elements + 1
    kglob = assemble_coo(stiffness, element_dofs, 2 * mesh.n_nodes).to_sparse_coo()
    u = solved.displacement.reshape(-1, 1)
    strain_energy = 0.5 * float(torch.sparse.mm(kglob.coalesce(), u).t() @ u)
    work = float(_loaded_force(mesh) @ solved.displacement.reshape(-1))
    return {
        "a_over_w": alpha,
        "n_elements": mesh.n_elements,
        "strain_energy": strain_energy,
        "external_work": work,
        "potential": strain_energy - work,
        "clapeyron_ratio": work / strain_energy,
        "reaction_ratio": solved.reaction_ratio,
    }


def stage_energy() -> dict:
    lo = _potential(ALPHA - STEP)
    hi = _potential(ALPHA + STEP)
    half_step = STEP * WIDTH
    mirror = 2.0  # Pi_full = 2 Pi_half; the tip is shared by both halves
    g = -mirror * (hi["potential"] - lo["potential"]) / (2.0 * half_step) / THICKNESS
    k_energy = math.sqrt(g * e_prime()) if g > 0.0 else float("nan")
    payload = {
        "a_over_w": ALPHA,
        "step_m": half_step,
        "mirror_factor": mirror,
        "potential_low": lo["potential"],
        "potential_high": hi["potential"],
        "energy_release_rate": g,
        "k_energy_pa_sqrt_m": k_energy,
        "clapeyron_ratio_low": lo["clapeyron_ratio"],
        "clapeyron_ratio_high": hi["clapeyron_ratio"],
        "reaction_ratio_low": lo["reaction_ratio"],
        "reaction_ratio_high": hi["reaction_ratio"],
        "level": dict(MEDIUM),
        "plane": PLANE,
    }
    write_json(cache_path("energy"), payload)
    print(f"G={g:.6e}  K_G={k_energy / 1e6:.4f} MPa.sqrt(m)", flush=True)
    return payload


# --------------------------------------------------------------------------- #
# stage: ligament field (medium level, nu = 0.3)
# --------------------------------------------------------------------------- #
def stage_field() -> dict:
    mesh = _mesh(ALPHA, MEDIUM)
    solved = solve_sent(
        mesh, sigma_pa=SIGMA, young_pa=YOUNG, poisson=NU, plane=PLANE,
        thickness=THICKNESS,
    )
    dmat = elasticity_matrix(
        torch.tensor([YOUNG], dtype=mesh.nodes.dtype),
        torch.tensor([NU], dtype=mesh.nodes.dtype),
        PLANE,
    )[0]
    ligament = []
    for k in range(1, len(mesh.stations)):
        element, edge_index = mesh.arc_edge[0][k]
        if element < 0:
            continue
        stress, _, _ = element_field_at(
            mesh, solved.displacement, element, *edge_parent(edge_index, 0.5), dmat
        )
        coords = mesh.nodes[mesh.elements[element]]
        shape, _ = edge_geometry(edge_index, 0.5)
        point = torch.tensor(shape, dtype=coords.dtype) @ coords
        ligament.append(
            {
                "r_over_a": mesh.stations[k],
                "x_over_w": float(point[0]) / WIDTH,
                "sigma_yy_pa": float(stress[1, 1]),
            }
        )
    centroidal = []
    for element in range(mesh.n_elements):
        stress, _, _ = element_field_at(
            mesh, solved.displacement, element, 1.0 / 3.0, 1.0 / 3.0, dmat
        )
        corner = mesh.nodes[mesh.elements[element][:3]]
        centroidal.append(
            {
                "x_over_w": float(corner[:, 0].mean()) / WIDTH,
                "y_over_w": float(corner[:, 1].mean()) / WIDTH,
                "sigma_yy_mpa": float(stress[1, 1]) / 1e6,
            }
        )
    payload = {
        "a_over_w": ALPHA,
        "level": dict(MEDIUM),
        "n_nodes": mesh.n_nodes,
        "n_elements": mesh.n_elements,
        "ligament": ligament,
        "centroidal": centroidal,
        "generated_at": utc_now(),
    }
    write_json(cache_path("field"), payload)
    return payload


# --------------------------------------------------------------------------- #
# grading and figures
# --------------------------------------------------------------------------- #
def build_evidence(ladder: dict, control: dict, nu: dict, energy: dict,
                   field: dict) -> dict:
    reference = k_reference()
    best = _canonical(ladder["levels"][-1]["cases"][0])
    contour_spread = (
        max(c["shape_factor"] for c in ladder["levels"][-1]["cases"][0]["contours"])
        - min(c["shape_factor"] for c in ladder["levels"][-1]["cases"][0]["contours"])
    ) / best["shape_factor"]
    ladder_rows = []
    for level in ladder["levels"]:
        case = level["cases"][0]
        canonical = _canonical(case)
        spread = (
            max(c["shape_factor"] for c in case["contours"])
            - min(c["shape_factor"] for c in case["contours"])
        ) / canonical["shape_factor"]
        ladder_rows.append(
            {
                "level": level["level"],
                "n_elements": case["n_elements"],
                "n_nodes": case["n_nodes"],
                "n_stations": case["n_stations"],
                "shape_factor": canonical["shape_factor"],
                "rel_error_vs_gs": canonical["rel_error_vs_tada"],
                "rel_error_vs_closed_form": (
                    canonical["shape_factor"] / reference["shape_factor_closed_form"] - 1.0
                ),
                "k_pa_sqrt_m": canonical["k_pa_sqrt_m"],
                "contour_station_fraction": canonical["station_fraction"],
                "path_spread": spread,
                "reaction_ratio": case["reaction_ratio"],
            }
        )
    gci = sent_gci(list(ladder["levels"]), alpha=ALPHA, contour_fraction=CANONICAL_CONTOUR)
    control_row = {
        "quarter_point": False,
        "level": control["levels"][0]["level"],
        "n_elements": control["levels"][0]["cases"][0]["n_elements"],
        "shape_factor": _canonical(control["levels"][0]["cases"][0])["shape_factor"],
        "rel_error_vs_gs": _canonical(control["levels"][0]["cases"][0])["rel_error_vs_tada"],
    }
    nu_spreads = dict(nu["spreads"])
    nu_spread = nu_spreads.get("fine", float("nan"))
    k_j = best["k_pa_sqrt_m"]
    return {
        "anchor": ANCHOR,
        "reference": reference,
        "ladder": ladder_rows,
        "finest": ladder_rows[-1],
        "gci": gci,
        "contour_spread_fine": contour_spread,
        "control": control_row,
        "nu_rows": nu["rows"],
        "nu_spread": nu_spread,
        "nu_spreads": nu_spreads,
        "energy": energy,
        "energy_rel_error_vs_j": (energy["k_energy_pa_sqrt_m"] - k_j) / k_j,
        "energy_rel_error_vs_gs": (
            energy["k_energy_pa_sqrt_m"] - reference["k_gs_pa_sqrt_m"]
        ) / reference["k_gs_pa_sqrt_m"],
        "field": field,
        "ladder_raw": ladder,
    }


def gates(evidence: dict) -> list[Gate]:
    finest = evidence["finest"]
    reference = evidence["reference"]
    return [
        Gate(
            id="acceptance_5pct",
            quantity="|K_FEM - K_GS| / K_GS at a/W = 0.3, plane strain, finest level",
            reference="K_I = sigma sqrt(pi a) F(a/W), Tada Paris & Irwin 3rd ed. polynomial fit",
            value=abs(finest["rel_error_vs_gs"]),
            threshold=TOL_CASE,
            note="the tolerance written into the anchor case",
        ),
        Gate(
            id="mission_3pct",
            quantity="|K_FEM - K_GS| / K_GS at a/W = 0.3, plane strain, finest level",
            reference="mission-wide 3 % accuracy standard",
            value=abs(finest["rel_error_vs_gs"]),
            threshold=TOL_MISSION,
        ),
        Gate(
            id="reference_spread",
            quantity="|F_closed_form - F_GS| / F_GS at a/W = 0.3",
            reference="two independent handbook fits for the SENT geometry factor",
            value=abs(reference["shape_factor_spread"]),
            threshold=TOL_REFERENCE,
            note="bounds the reference uncertainty; not a numerical result",
        ),
        Gate(
            id="path_independence",
            quantity="max-min of F over 8 fan lines, finest level",
            reference="J is path independent for a contour enclosing no singularity other than the tip",
            value=evidence["contour_spread_fine"],
            threshold=TOL_PATH,
        ),
        Gate(
            id="reaction_balance",
            quantity="|vertical reaction / applied load - 1| at the finest level",
            reference="global equilibrium of the discrete system",
            value=abs(finest["reaction_ratio"] - 1.0),
            threshold=TOL_BALANCE,
        ),
        Gate(
            id="gci_fine",
            quantity="Roache GCI of F over the three levels",
            reference="grid convergence index of the finest value from the three values themselves",
            value=abs(evidence["gci"].get("gci_fine", float("nan"))),
            threshold=TOL_CASE,
            note="undefined (not converging) is reported as a failed gate, never as a pass",
        ),
        Gate(
            id="material_invariance",
            quantity="|F(nu=0.45) - F(nu=0.30)| / F(nu=0.30) at the medium level",
            reference="LEFM: K_I is independent of E and nu, so a correct E' (and energy density) must cancel nu",
            value=evidence["nu_spread"],
            threshold=TOL_NU,
            note="finest available level is graded; the coarser spread is reported alongside because a wrong E' pair would sit at +4.83 % on every mesh",
        ),
        Gate(
            id="energy_vs_j",
            quantity="|K_(dPi/da) - K_J| / K_J at a/W = 0.3, plane strain",
            reference="global energy release rate versus the local contour integral on the same field",
            value=abs(evidence["energy_rel_error_vs_j"]),
            threshold=TOL_MISSION,
        ),
        Gate(
            id="energy_vs_handbook",
            quantity="|K_(dPi/da) - K_GS| / K_GS at a/W = 0.3, plane strain",
            reference="K_I = sigma sqrt(pi a) F(a/W), Tada Paris & Irwin 3rd ed.",
            value=abs(evidence["energy_rel_error_vs_gs"]),
            threshold=TOL_CASE,
        ),
        Gate(
            id="clapeyron",
            quantity="max |f.u / U - 2| over the two energy-oracle solves",
            reference="f.u = 2U for a linear traction-only problem; audits the potential bookkeeping",
            value=max(
                abs(evidence["energy"]["clapeyron_ratio_low"] - 2.0),
                abs(evidence["energy"]["clapeyron_ratio_high"] - 2.0),
            ),
            threshold=TOL_CLAPEYRON,
        ),
    ]


def make_figures(evidence: dict, outdir: Path) -> dict:
    plt = pyplot()
    figures = {}
    reference = evidence["reference"]

    # 1) convergence
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    elements = [row["n_elements"] for row in evidence["ladder"]]
    errors = [abs(row["rel_error_vs_gs"]) * 100.0 for row in evidence["ladder"]]
    ax.plot(elements, errors, "o-", color="#1f4e79", label="plane strain, nu = 0.3")
    ax.axhline(TOL_MISSION * 100, color="#c00000", ls="--", lw=0.9, label="3 % mission")
    ax.axhline(TOL_CASE * 100, color="k", ls=":", lw=0.9, label="5 % case tolerance")
    for row, value in zip(evidence["ladder"], errors):
        ax.annotate(f"{value:.2f}%", (row["n_elements"], value),
                    textcoords="offset points", xytext=(4, 5), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("number of T6 elements")
    ax.set_ylabel("|K_FEM - K_GS| / K_GS  [%]")
    ax.set_title("SENT a/W = 0.3, plane strain: convergence")
    ax.legend(fontsize=7)
    path = outdir / "sent_plane_strain_convergence.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    figures[path.name] = ARTIFACTS[path.name]

    # 2) path independence
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    colours = {"coarse": "#9dc3e6", "medium": "#2e75b6", "fine": "#1f4e79"}
    for level in evidence["ladder_raw"]["levels"]:
        contours = level["cases"][0]["contours"]
        ax.plot(
            [c["station_fraction"] for c in contours],
            [c["shape_factor"] for c in contours],
            "o-", ms=3.5, color=colours.get(level["level"], "#666666"),
            label=f"{level['level']} ({level['cases'][0]['n_elements']} el.)",
        )
    ax.axhline(reference["shape_factor_gs"], color="#c00000", ls="--", lw=0.9,
               label="Tada polynomial fit")
    ax.axhline(reference["shape_factor_closed_form"], color="#7f6000", ls=":", lw=0.9,
               label="secant closed form")
    ax.set_xlabel("contour station  r/a")
    ax.set_ylabel("recovered F(a/W)")
    ax.set_title("Path independence of the J-integral")
    ax.legend(fontsize=7)
    path = outdir / "sent_plane_strain_paths.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    figures[path.name] = ARTIFACTS[path.name]

    # 3) shape factor against both handbook fits
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    grid = [0.20 + 0.005 * i for i in range(61)]
    ax.plot(grid, [tada_sent_shape_factor(a) for a in grid], "-", color="#c00000",
            lw=1.2, label="Tada polynomial (GS)")
    ax.plot(grid, [tada_sent_shape_factor_closed_form(a) for a in grid], "--",
            color="#7f6000", lw=1.0, label="secant closed form")
    ax.plot([ALPHA], [evidence["finest"]["shape_factor"]], "o", ms=7,
            color="#1f4e79", label="FE, finest level (plane strain)")
    for row in evidence["ladder"]:
        ax.plot([ALPHA], [row["shape_factor"]], ".", ms=5, color="#9dc3e6")
    ax.set_xlim(0.2, 0.5)
    ax.set_xlabel("a/W")
    ax.set_ylabel("F(a/W)")
    ax.set_title("Recovered geometry factor versus the handbook")
    ax.legend(fontsize=7)
    path = outdir / "sent_plane_strain_handbook.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    figures[path.name] = ARTIFACTS[path.name]

    # 4) material invariance, two levels
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    levels = ["medium", "fine"]
    shades = {"medium": "#1f4e79", "fine": "#9dc3e6"}
    nus = sorted({row["poisson"] for row in evidence["nu_rows"]})
    bar_width = 0.36
    for offset, level in enumerate(levels):
        rows = [row for row in evidence["nu_rows"] if row["level"] == level]
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: row["poisson"])
        pos = [i + (offset - 0.5) * bar_width for i in range(len(rows))]
        axes[0].bar(pos, [row["shape_factor"] for row in rows], width=bar_width,
                    color=shades[level], label=level)
        axes[1].bar(pos, [row["j_j_m2"] for row in rows], width=bar_width,
                    color=shades[level], label=level)
    factors = [row["shape_factor"] for row in evidence["nu_rows"]]
    axes[0].axhline(reference["shape_factor_gs"], color="#c00000", ls="--", lw=0.9,
                    label="Tada fit")
    axes[0].set_ylim(min(factors) * 0.98, max(factors) * 1.02)
    axes[0].set_ylabel("F(a/W) recovered")
    axes[0].set_title(
        "F is nu-invariant (fine spread "
        f"{100 * evidence['nu_spread']:.2f} %)"
    )
    axes[0].legend(fontsize=7)
    axes[1].set_ylabel("J  [J/m$^2$]  (per unit thickness)")
    axes[1].set_title("J itself does depend on nu through E'")
    axes[1].legend(fontsize=7)
    for axis in axes:
        axis.set_xticks(range(len(nus)))
        axis.set_xticklabels([f"nu = {nu:.2f}" for nu in nus])
    fig.tight_layout()
    path = outdir / "sent_plane_strain_nu.png"
    fig.savefig(path)
    plt.close(fig)
    figures[path.name] = ARTIFACTS[path.name]

    # 5) energy oracle
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    names = ["K_J\n(contour)", "K_(dPi/da)\n(global energy)", "K_GS\n(handbook)"]
    values = [
        evidence["finest"]["k_pa_sqrt_m"] / 1e6,
        evidence["energy"]["k_energy_pa_sqrt_m"] / 1e6,
        reference["k_gs_pa_sqrt_m"] / 1e6,
    ]
    bars = ax.bar(names, values, color=["#1f4e79", "#2e75b6", "#c00000"], width=0.55)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2.0, value),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    ax.set_ylim(min(values) * 0.97, max(values) * 1.03)
    ax.set_ylabel("K_I  [MPa sqrt(m)]")
    ax.set_title("Two independent routes versus the handbook")
    path = outdir / "sent_plane_strain_energy.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    figures[path.name] = ARTIFACTS[path.name]

    # 6) ligament stress against the singular form
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    ligament = evidence["field"]["ligament"]
    r = [row["r_over_a"] for row in ligament]
    s = [row["sigma_yy_pa"] / 1e6 for row in ligament]
    axes[0].plot(r, s, "o-", ms=3, color="#1f4e79", label="FE, medium level")
    anchor_r, anchor_s = r[0], s[0]
    axes[0].plot(r, [anchor_s * math.sqrt(anchor_r / value) for value in r], "--",
                 color="#c00000", lw=1.0, label="r$^{-1/2}$ through the first station")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("r/a ahead of the tip")
    axes[0].set_ylabel("sigma_yy  [MPa]")
    axes[0].set_title("Ligament stress, plane strain")
    axes[0].legend(fontsize=7)
    cloud = evidence["field"]["centroidal"]
    sc = axes[1].scatter(
        [row["x_over_w"] for row in cloud], [row["y_over_w"] for row in cloud],
        c=[row["sigma_yy_mpa"] for row in cloud], s=6, cmap="inferno",
    )
    axes[1].set_xlim(-0.35, 0.75)
    axes[1].set_ylim(0.0, 0.9)
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("x/W")
    axes[1].set_ylabel("y/W")
    axes[1].set_title("sigma_yy field (element centroids)")
    fig.colorbar(sc, ax=axes[1], shrink=0.8, label="MPa")
    fig.tight_layout()
    path = outdir / "sent_plane_strain_ligament.png"
    fig.savefig(path)
    plt.close(fig)
    figures[path.name] = ARTIFACTS[path.name]
    return figures


# --------------------------------------------------------------------------- #
# stage: assemble
# --------------------------------------------------------------------------- #
def stage_assemble() -> dict:
    ladder = json.loads(cache_path("ladder").read_text())
    control = json.loads(cache_path("control").read_text())
    nu = json.loads(cache_path("nu").read_text())
    energy = json.loads(cache_path("energy").read_text())
    field = json.loads(cache_path("field").read_text())
    evidence = build_evidence(ladder, control, nu, energy, field)

    criteria = {
        "acceptance_5pct": TOL_CASE,
        "mission_3pct": TOL_MISSION,
        "reference_spread": TOL_REFERENCE,
        "path_independence": TOL_PATH,
        "reaction_balance": TOL_BALANCE,
        "gci_fine": TOL_CASE,
        "material_invariance": TOL_NU,
        "energy_vs_j": TOL_MISSION,
        "energy_vs_handbook": TOL_CASE,
        "clapeyron": TOL_CLAPEYRON,
    }
    manifest = gate_manifest(
        anchor=ANCHOR,
        title=TITLE,
        mission=MISSION,
        criteria=criteria,
        gates=gates(evidence),
        artifacts=ARTIFACTS,
        limitations=[
            "linear elastic, small strain, stationary mode-I, homogeneous isotropic material, no body force",
            "plane strain is realised by the constitutive matrix and E' = E/(1-nu^2); no 3-D constraint check is attempted",
            "a/W = 0.3 only: the plane-stress legs at a/W = 0.20/0.35/0.50 live in benchmarks/verified/fem_sent_j_integral",
            "the energy oracle uses a central difference with h = 0.01 W on the same medium discretisation; its O(h^2) truncation error is not separated from discretisation noise",
            "one geometry (H/W = 2) and one load level: height sensitivity is covered by the plane-stress anchor",
        ],
    )

    figures = make_figures(evidence, HERE)

    result = {
        "anchor": ANCHOR,
        "title": TITLE,
        "mission": MISSION,
        "generated_at": utc_now(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "machine": platform.machine(),
        },
        "setting": {
            "width_m": WIDTH,
            "a_over_w": ALPHA,
            "height_over_width": HEIGHT_RATIO,
            "sigma_pa": SIGMA,
            "young_pa": YOUNG,
            "poisson": NU,
            "plane": PLANE,
            "e_prime_pa": e_prime(),
            "thickness_m": THICKNESS,
            "contour_fractions": list(CONTOUR_FRACTIONS),
            "canonical_contour_fraction": CANONICAL_CONTOUR,
            "levels": [dict(level) for level in LEVELS],
            "quarter_point": True,
        },
        "reference": evidence["reference"],
        "ladder": evidence["ladder"],
        "gci": evidence["gci"],
        "contour_spread_fine": evidence["contour_spread_fine"],
        "control_without_quarter_point": evidence["control"],
        "material_invariance": {
            "rows": evidence["nu_rows"],
            "spread_fine": evidence["nu_spread"],
            "spread_by_level": evidence["nu_spreads"],
        },
        "energy_oracle": evidence["energy"],
        "energy_rel_error_vs_j": evidence["energy_rel_error_vs_j"],
        "energy_rel_error_vs_handbook": evidence["energy_rel_error_vs_gs"],
        "gates_summary": {
            "passed": manifest["passed"],
            "failed_gates": manifest["failed_gates"],
        },
        "artifacts": ARTIFACTS,
    }
    write_json(HERE / "result.json", result)
    write_json(HERE / "gates_verified.json", manifest)

    finest = evidence["finest"]
    print(json.dumps(
        {
            "shape_factor_fem": round(finest["shape_factor"], 5),
            "shape_factor_gs": round(evidence["reference"]["shape_factor_gs"], 5),
            "rel_error_vs_gs": round(finest["rel_error_vs_gs"], 5),
            "k_fem_mpa_sqrt_m": round(finest["k_pa_sqrt_m"] / 1e6, 4),
            "path_spread": round(evidence["contour_spread_fine"], 5),
            "nu_spread_fine": round(evidence["nu_spread"], 5),
            "nu_spreads_by_level": {k: round(v, 5) for k, v in evidence["nu_spreads"].items()},
            "energy_vs_j": round(evidence["energy_rel_error_vs_j"], 5),
            "passed": manifest["passed"],
            "failed_gates": manifest["failed_gates"],
        },
        indent=2,
    ))
    return result


STAGES = {
    "ladder": stage_ladder,
    "control": stage_control,
    "nu": stage_nu,
    "energy": stage_energy,
    "field": stage_field,
    "assemble": stage_assemble,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGES) + ["all"], default="all")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    names = list(STAGES) if args.stage == "all" else [args.stage]
    for name in names:
        print(f"== stage {name} ==", flush=True)
        STAGES[name]()
    if args.stage in ("assemble", "all"):
        manifest = json.loads((HERE / "gates_verified.json").read_text())
        return 0 if manifest["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())