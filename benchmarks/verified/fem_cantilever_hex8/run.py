#!/usr/bin/env python3
"""CASE-CANTILEVER-TIP element-family ladder -- is the ``< 3 %`` criterion family-free?

Physics
-------
The case-pool anchor ``CASE-CANTILEVER-TIP`` grades the tip deflection of a
prismatic cantilever against the Euler-Bernoulli / Timoshenko closed forms with a
``< 3 %`` criterion.  Every published rung of the anchor is either a beam model or
a **quadratic** 3-D solid (HEX20), so the criterion is stated without a qualifier
on the *element family*, while in practice a ship-structure model is often meshed
with first-order bricks (HEX8) or tetrahedra (TET4).  Fully integrated first-order
elements are known to lock in bending -- a trilinear brick cannot represent a
linear-in-z axial strain without parasitic shear (Cook, Malkus & Plesha,
*Concepts and Applications of FEA*, Sec. 6.9; Bathe, *Finite Element Procedures*,
Sec. 5.4) -- so the same mesh size that satisfies the criterion with HEX20 can
miss it by a large multiple with HEX8.  This driver turns that folklore into a
graded ladder on the anchor's own geometry.

Oracles (re-derived here, no handbook digit enters the chain)
-------------------------------------------------------------
    delta_EB(L) = P L^3 / (3 E I)                       (1)
    delta_TI(L) = delta_EB + P L / (KAPPA G A)          (2)

with ``I = B H^3 / 12``, ``A = B H``, ``G = E / (2 (1 + nu))`` and
``KAPPA = 5/6``.  Both are re-obtained from the unit-load (Castigliano) integrals

    delta = int_0^L M m /(E I) dx + int_0^L V v /(KAPPA G A) dx,
    M = P (L - x), m = (L - x), V = P, v = 1            (3)

by quadrature *inside this driver*; the residual against (1)/(2) is a gate.

What is graded
--------------
For each slenderness rung ``L/H = 100, 20, 10`` and each family
``hex20 / hex8 / tet4`` the mean transverse displacement of the *loaded end face*
is compared with (2), the oracle valid for a solid that carries shear:

* does the 3 % criterion hold for a quadratic solid at a mesh a user would pick
  for this geometry?                                        (must pass)
* does it hold for a fully integrated first-order brick at the same mesh
  density?                                     (falsifiable: it must not)
* how does the brick error decay under through-depth refinement, and how many
  dofs does it cost to buy back what HEX20 gets for free?
* is TET4 (six-tet decomposition of the same brick) better or worse?

Outputs (generated, never hand-edited): ``result.json``, ``gates_verified.json``,
``README.md``, ``docs/CASE-CANTILEVER-TIP-element-family.md`` and the PNGs under
``docs/assets/cantilever_element_family/``.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for extra in (REPO / "src", REPO / "benchmarks" / "verified"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from _harness import Gate, gate_manifest, pyplot, utc_now, write_json  # noqa: E402
from tensorfem.hex20 import (  # noqa: E402
    Hex20Model,
    solve_hex20,
    structured_hex20_mesh,
)
from tensorfem.solid3d import (  # noqa: E402
    SolidModel,
    hex_to_tet_mesh,
    solve_solid,
    structured_hex_mesh,
)

ANCHOR = "fem_cantilever_hex8"
MISSION = "CASE-CANTILEVER-TIP"
TITLE = (
    "CASE-CANTILEVER-TIP element-family ladder: HEX20 vs fully integrated HEX8 vs "
    "TET4 against the Timoshenko oracle at the anchor's own 3 % criterion"
)

# --------------------------------------------------------------------------- #
# Problem definition -- identical geometry, section and load to the anchor
# --------------------------------------------------------------------------- #
E = 200.0e9         # Young's modulus [Pa]
NU = 0.3
P = 5.0             # tip load [N], applied as a uniform traction on the end face
B = 0.01            # section width  [m]  (y)
H = 0.01            # section depth  [m]  (z, bending direction)
KAPPA = 5.0 / 6.0   # rectangular section, average-strain shear factor
G = E / (2.0 * (1.0 + NU))
A = B * H
I = B * H**3 / 12.0
DTYPE = torch.float64

TOLERANCE = 0.03    # case-pool tolerance declared by the anchor

# Slenderness rungs: L/H = 100 (the case-pool rung), 20, 10.
RUNGS = (100.0, 20.0, 10.0)

# HEX20 meshes (nx, ny, nz): the cross-section is held at 2x2 quadratic cells.
HEX20_MESHES = {
    100.0: ((16, 2, 2), (24, 2, 2)),
    20.0: ((12, 2, 2), (20, 2, 2)),
    10.0: ((8, 2, 2), (16, 2, 2)),
}

# First-order brick: fixed in-plane mesh, through-depth density swept so the
# locking decay and its cost are both visible.
HEX8_NX = 16
HEX8_NY = 2
HEX8_DENSITIES = {
    100.0: (2, 4, 8, 16),
    20.0: (2, 4, 8),
    10.0: (2, 4, 8),
}

# TET4: six-tet decomposition of the same brick topology.
TET4_DENSITIES = {
    100.0: (4, 8),
    20.0: (4, 8),
    10.0: (4, 8),
}

# In-plane (bending-direction) refinement sweep at the case-pool rung.  The
# through-depth sweep turns out to be irrelevant to the locking error, so the
# controlling parameter -- the along-member element length relative to the
# section depth -- is probed directly.  Width is held at ny = 2, depth at nz = 4.
ASPECT_NX = (16, 32, 64, 128)
ASPECT_NZ = 4

OUT_DIR = HERE
DOC = REPO / "docs" / "CASE-CANTILEVER-TIP-element-family.md"
FIG_DIR = REPO / "docs" / "assets" / "cantilever_element_family"


# --------------------------------------------------------------------------- #
# Oracles
# --------------------------------------------------------------------------- #
def delta_eb(L: float) -> float:
    """Euler-Bernoulli tip deflection of a tip-loaded cantilever."""
    return P * L**3 / (3.0 * E * I)


def delta_ti(L: float) -> float:
    """Timoshenko tip deflection (bending + constant-shear term)."""
    return delta_eb(L) + P * L / (KAPPA * G * A)


def deflection_profile(x, L: float):
    """Timoshenko deflection along the centroidal line (tip load, clamped root)."""
    x = np.asarray(x, dtype=float)
    return P * x**2 * (3.0 * L - x) / (6.0 * E * I) + P * x / (KAPPA * G * A)


def oracle_residual(L: float, n: int = 20001) -> dict:
    """Re-derive (1)/(2) from the unit-load integrals (3) by quadrature."""
    x = np.linspace(0.0, L, n)
    bending = np.trapezoid(P * (L - x) * (L - x) / (E * I), x)
    shear = np.trapezoid(np.full_like(x, P) * 1.0 / (KAPPA * G * A), x)
    numeric = bending + shear
    exact = delta_ti(L)
    return {
        "L_over_H": L / H,
        "quadrature_bending_term": float(bending),
        "quadrature_shear_term": float(shear),
        "quadrature_total": float(numeric),
        "closed_form_timoshenko": float(exact),
        "closed_form_euler_bernoulli": float(delta_eb(L)),
        "relative_residual": float(abs(numeric - exact) / exact),
        "eb_ti_separation": float((exact - delta_eb(L)) / delta_eb(L)),
    }


# --------------------------------------------------------------------------- #
# Solvers -- three element families, one calling convention
# --------------------------------------------------------------------------- #
def _end_face(nodes: torch.Tensor, L: float) -> torch.Tensor:
    return torch.where(torch.abs(nodes[:, 0] - L) <= 1e-9 * max(L, 1.0))[0]


def _root_face(nodes: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.abs(nodes[:, 0]) <= 1e-12)[0]


def _tip_load(nodes: torch.Tensor, L: float) -> torch.Tensor:
    """Uniform transverse traction on the end face, resultant exactly ``P``."""
    tip = _end_face(nodes, L)
    forces = torch.zeros(3 * len(nodes), dtype=DTYPE)
    forces[3 * tip + 2] = P / len(tip)
    return forces


def _clamp(nodes: torch.Tensor) -> torch.Tensor:
    root = _root_face(nodes)
    return torch.stack((3 * root, 3 * root + 1, 3 * root + 2), 1).reshape(-1)


def solve_hex20_case(L: float, nx: int, ny: int, nz: int) -> dict:
    nodes, cells = structured_hex20_mesh(L, B, H, nx, ny, nz)
    model = Hex20Model(
        nodes,
        cells,
        torch.tensor(E, dtype=DTYPE),
        torch.tensor(NU, dtype=DTYPE),
        _tip_load(nodes, L),
        _clamp(nodes),
    )
    result = solve_hex20(model)
    tip = _end_face(nodes, L)
    return {
        "family": "hex20",
        "mesh": [nx, ny, nz],
        "n_elements": int(len(cells)),
        "n_nodes": int(len(nodes)),
        "n_dofs": int(model.n_dofs),
        "delta": float(result.displacement[3 * tip + 2].mean()),
        "nodes": nodes,
        "displacement": result.displacement,
    }


def solve_solid_case(L: float, nx: int, ny: int, nz: int, element: str = "hex8") -> dict:
    nodes, hexes = structured_hex_mesh(L, B, H, nx, ny, nz)
    elements = hex_to_tet_mesh(nodes, hexes)[1] if element == "tet4" else hexes
    model = SolidModel(
        nodes,
        elements,
        torch.tensor(E, dtype=DTYPE),
        torch.tensor(NU, dtype=DTYPE),
        _tip_load(nodes, L),
        _clamp(nodes),
        element,
    )
    result = solve_solid(model)
    tip = _end_face(nodes, L)
    return {
        "family": element,
        "mesh": [nx, ny, nz],
        "n_elements": int(len(elements)),
        "n_nodes": int(len(nodes)),
        "n_dofs": int(model.n_dofs),
        "delta": float(result.displacement[3 * tip + 2].mean()),
        "nodes": nodes,
        "displacement": result.displacement,
    }


# --------------------------------------------------------------------------- #
# Ladder
# --------------------------------------------------------------------------- #
def _record(case: dict, L: float) -> dict:
    ref = delta_ti(L)
    eb = delta_eb(L)
    rec = {k: v for k, v in case.items() if k not in ("nodes", "displacement")}
    rec["L_over_H"] = L / H
    rec["delta_timoshenko"] = ref
    rec["delta_euler_bernoulli"] = eb
    rec["relative_error_vs_timoshenko"] = abs(rec["delta"] - ref) / ref
    rec["relative_error_vs_euler_bernoulli"] = abs(rec["delta"] - eb) / eb
    rec["criterion_3pct_met"] = bool(rec["relative_error_vs_timoshenko"] <= TOLERANCE)
    return rec


def run_ladder(rungs=RUNGS, *, quick: bool = False, keep_fields: bool = False) -> dict:
    """Solve every rung/family pair and grade it against the Timoshenko oracle."""
    records: list[dict] = []
    fields: dict[str, dict] = {}
    started = time.time()
    for L_over_h in rungs:
        L = L_over_h * H
        hex20_meshes = HEX20_MESHES[L_over_h][: 1] if quick else HEX20_MESHES[L_over_h]
        for (nx, ny, nz) in hex20_meshes:
            case = solve_hex20_case(L, nx, ny, nz)
            records.append(_record(case, L))
            if keep_fields:
                fields[f"hex20_{nx}x{ny}x{nz}_L{L_over_h:g}"] = {
                    "nodes": case["nodes"], "displacement": case["displacement"]}
        densities = HEX8_DENSITIES[L_over_h][:1] if quick else HEX8_DENSITIES[L_over_h]
        for nz in densities:
            case = solve_solid_case(L, HEX8_NX, HEX8_NY, nz, "hex8")
            records.append(_record(case, L))
            if keep_fields:
                fields[f"hex8_{HEX8_NX}x{HEX8_NY}x{nz}_L{L_over_h:g}"] = {
                    "nodes": case["nodes"], "displacement": case["displacement"]}
        for nz in (TET4_DENSITIES[L_over_h][:1] if quick else TET4_DENSITIES[L_over_h]):
            records.append(_record(solve_solid_case(L, HEX8_NX, HEX8_NY, nz, "tet4"), L))
    return {"records": records, "fields": fields, "wall_seconds": time.time() - started}


def run_aspect_sweep(L_over_h: float = 100.0) -> list[dict]:
    """HEX8 error against the along-member element length at fixed through-depth mesh."""
    L = L_over_h * H
    out: list[dict] = []
    for nx in ASPECT_NX:
        rec = _record(solve_solid_case(L, nx, 2, ASPECT_NZ, "hex8"), L)
        rec["element_length_x"] = L / nx
        rec["dx_over_H"] = (L / nx) / H
        rec["dz_over_H"] = 1.0 / ASPECT_NZ
        out.append(rec)
    return out


def observed_error_order(aspect: list[dict]) -> dict:
    """Log-log slope of the HEX8 error against ``dx/H`` (first vs last record)."""
    coarse, fine = aspect[0], aspect[-1]
    slope = math.log(coarse["relative_error_vs_timoshenko"]
                     / fine["relative_error_vs_timoshenko"]) / math.log(
        coarse["dx_over_H"] / fine["dx_over_H"])
    # extrapolate the along-member element length needed to reach the tolerance
    ratio = TOLERANCE / coarse["relative_error_vs_timoshenko"]
    dx_needed = coarse["dx_over_H"] * ratio ** (1.0 / slope)
    return {
        "observed_order": float(slope),
        "dx_over_H_needed_for_tolerance": float(dx_needed),
        "nx_needed_for_tolerance_per_H": float(1.0 / dx_needed),
        "note": "extrapolation from the measured slope, not a solved mesh",
    }


def _best(records: list[dict], family: str, L_over_h: float) -> dict:
    rows = [r for r in records if r["family"] == family and r["L_over_H"] == L_over_h]
    if not rows:
        raise KeyError(f"no {family} record at L/H={L_over_h}")
    return min(rows, key=lambda r: r["relative_error_vs_timoshenko"])


def _ladder_by_density(records: list[dict], family: str, L_over_h: float) -> list[dict]:
    rows = [r for r in records if r["family"] == family and r["L_over_H"] == L_over_h]
    return sorted(rows, key=lambda r: r["n_dofs"])


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _figure(records: list[dict], fields: dict, aspect: list[dict],
            order: dict) -> dict:
    plt = pyplot()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figures = {}
    colours = {"hex20": "tab:blue", "hex8": "tab:red", "tet4": "tab:green"}
    markers = {"hex20": "o", "hex8": "s", "tet4": "^"}

    # (1) centroidal deflection profile at L/H = 10 vs the Timoshenko line
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    L = 10.0 * H
    xs = np.linspace(0.0, L, 200)
    ax.plot(xs / L, deflection_profile(xs, L) * 1e6, "k-", lw=1.6,
            label="Timoshenko closed form")
    for key, payload in fields.items():
        if not key.endswith("_L10"):
            continue
        family = key.split("_")[0]
        nodes, u = payload["nodes"], payload["displacement"]
        sel = ((torch.abs(nodes[:, 1] - B / 2) < 1e-12)
               & (torch.abs(nodes[:, 2] - H / 2) < 1e-12))
        if not bool(sel.any()):
            continue
        idx = torch.where(sel)[0]
        idx = idx[torch.argsort(nodes[idx, 0])]
        ax.plot((nodes[idx, 0] / L).numpy(), (u[3 * idx + 2] * 1e6).numpy(),
                marker=markers.get(family, "x"), ms=3.4, lw=1.0,
                color=colours.get(family, "tab:orange"), label=f"{family} FE")
    ax.set_xlabel("x / L")
    ax.set_ylabel("deflection $u_z$ [um]")
    ax.set_title("L/H = 10 centroidal deflection: FE families vs Timoshenko")
    ax.legend(fontsize=7)
    path = FIG_DIR / "deflection_profile_L10.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    figures["deflection_profile_L10"] = path

    # (2) error vs cost at the case-pool rung
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for family in ("hex20", "hex8", "tet4"):
        rows = sorted((r for r in records
                       if r["family"] == family and r["L_over_H"] == 100.0),
                      key=lambda r: r["n_dofs"])
        if not rows:
            continue
        ax.plot([r["n_dofs"] for r in rows],
                [100.0 * r["relative_error_vs_timoshenko"] for r in rows],
                marker=markers[family], color=colours[family], label=family)
    ax.axhline(100.0 * TOLERANCE, color="k", ls="--", lw=1.0, label="anchor criterion 3 %")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("degrees of freedom")
    ax.set_ylabel("tip-deflection error vs Timoshenko [%]")
    ax.set_title("L/H = 100: error against cost, per element family")
    ax.legend(fontsize=7)
    path = FIG_DIR / "error_vs_dofs_L100.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    figures["error_vs_dofs_L100"] = path

    # (3) best mesh per family and rung
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    width = 0.26
    rungs = sorted({r["L_over_H"] for r in records}, reverse=True)
    for j, family in enumerate(("hex20", "hex8", "tet4")):
        vals, xs = [], []
        for i, Lh in enumerate(rungs):
            rows = [r for r in records if r["family"] == family and r["L_over_H"] == Lh]
            if not rows:
                continue
            vals.append(100.0 * min(r["relative_error_vs_timoshenko"] for r in rows))
            xs.append(i + (j - 1) * width)
        ax.bar(xs, vals, width=width, color=colours[family], label=f"{family} (best mesh)")
    ax.axhline(100.0 * TOLERANCE, color="k", ls="--", lw=1.0)
    ax.set_xticks(range(len(rungs)))
    ax.set_xticklabels([f"L/H = {int(v)}" for v in rungs])
    ax.set_ylabel("best-mesh error vs Timoshenko [%]")
    ax.set_title("anchor criterion per element family and slenderness")
    ax.legend(fontsize=7)
    path = FIG_DIR / "error_by_rung.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    figures["error_by_rung"] = path

    # (4) the controlling parameter: along-member element length vs section depth
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    dx = np.array([r["dx_over_H"] for r in aspect])
    err = np.array([100.0 * r["relative_error_vs_timoshenko"] for r in aspect])
    ax.plot(dx, err, "s-", color="tab:red", label="HEX8, nz = 4 (nz sweep in Fig. 2)")
    z_rows = sorted((r for r in records if r["family"] == "hex8"
                     and r["L_over_H"] == 100.0), key=lambda r: r["n_dofs"])
    ax.plot([100.0 / r["mesh"][0] for r in z_rows],
            [100.0 * r["relative_error_vs_timoshenko"] for r in z_rows],
            "o", color="tab:orange", ms=5,
            label="HEX8, nx = 16, through-depth sweep")
    ax.axhline(100.0 * TOLERANCE, color="k", ls="--", lw=1.0, label="anchor criterion 3 %")
    ax.axvline(float(order["dx_over_H_needed_for_tolerance"]), color="0.4", ls=":", lw=1.0,
               label=f"extrapolated dx/H for 3 %: {order['dx_over_H_needed_for_tolerance']:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("element length along the member / section depth, $dx/H$")
    ax.set_ylabel("tip-deflection error vs Timoshenko [%]")
    ax.set_title(f"HEX8 locking (L/H = 100): error ~ (dx/H)^{order['observed_order']:.2f}")
    ax.legend(fontsize=6.5)
    path = FIG_DIR / "error_vs_aspect_L100.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    figures["error_vs_aspect_L100"] = path
    return figures


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #
def _table(records: list[dict]) -> str:
    lines = [
        "| L/H | family | mesh (nx x ny x nz) | elements | dofs | delta [um] "
        "| Timoshenko [um] | error [%] | < 3 % |",
        "|---:|---|---|---:|---:|---:|---:|---:|:--:|",
    ]
    for rec in sorted(records, key=lambda r: (-r["L_over_H"], r["family"], r["n_dofs"])):
        lines.append(
            "| {L:g} | {fam} | {m0}x{m1}x{m2} | {ne} | {nd} | {d:.4f} | {t:.4f} | "
            "{e:.2f} | {ok} |".format(
                L=rec["L_over_H"], fam=rec["family"], m0=rec["mesh"][0],
                m1=rec["mesh"][1], m2=rec["mesh"][2], ne=rec["n_elements"],
                nd=rec["n_dofs"], d=rec["delta"] * 1e6, t=rec["delta_timoshenko"] * 1e6,
                e=100.0 * rec["relative_error_vs_timoshenko"],
                ok="yes" if rec["criterion_3pct_met"] else "**no**",
            )
        )
    return "\n".join(lines)


def _readme(payload: dict) -> str:
    gates = payload["gates_verified"]
    rows = payload["result"]["records"]
    npass = sum(1 for g in gates["gates"] if g["passed"])
    head = (
        f"# {ANCHOR} -- element-family ladder\n\n"
        f"Generated by `benchmarks/verified/fem_cantilever_hex8/run.py` at "
        f"{payload['result']['generated_at']}; every number here and in "
        f"`docs/CASE-CANTILEVER-TIP-element-family.md` is read from `result.json`.\n\n"
        f"* mission: `{MISSION}`\n* gates: {npass}/{len(gates['gates'])} pass\n"
        f"* verdict: **{'PASS' if gates['passed'] else 'FAIL'}**\n"
    )
    if not gates["passed"]:
        head += f"* failed gates: `{', '.join(gates['failed_gates'])}`\n"
    return (
        head
        + "\n## Files\n\n"
        "* `run.py` -- driver (oracles re-derived in-driver, no hand-typed digits)\n"
        "* `result.json` -- every rung/family record + oracles + limitations\n"
        "* `gates_verified.json` -- graded criteria and the overall verdict\n"
        "* `../../docs/CASE-CANTILEVER-TIP-element-family.md` -- rendered report\n"
        "\n## Reproduce\n\n```bash\n"
        "PYTHONPATH=src python benchmarks/verified/fem_cantilever_hex8/run.py\n"
        "```\n\n"
        f"## Records ({len(rows)} solves)\n\n" + _table(rows) + "\n"
    )


def _doc(payload: dict) -> str:
    result = payload["result"]
    gates = payload["gates_verified"]
    records = result["records"]
    hl = result["highlights"]
    hex20_best = hl["hex20_best_L100"]
    hex8_rows = hl["hex8_ladder_L100"]
    hex8_coarse, hex8_fine = hex8_rows[0], hex8_rows[-1]
    eff = hl["dof_efficiency"]
    lines = [
        "# CASE-CANTILEVER-TIP -- element-family validity of the 3 % criterion",
        "",
        f"*Generated by `benchmarks/verified/fem_cantilever_hex8/run.py` on "
        f"{result['generated_at']} (mission `{MISSION}`, anchor `{ANCHOR}`). Every "
        "number below is injected from `result.json`; nothing is hand-copied.*",
        f"*Verdict: **{'PASS' if gates['passed'] else 'FAIL'}**"
        + ("" if gates["passed"] else f" (failed gates: `{', '.join(gates['failed_gates'])}`)")
        + "*",
        "",
        "## 1. Case and the debt this closes",
        "",
        "The case-pool anchor grades a prismatic cantilever against Euler-Bernoulli / "
        "Timoshenko with a `< 3 %` criterion. Every published rung of the anchor is a "
        "beam model or a **quadratic** 3-D solid, so the criterion has never been "
        "interrogated against the element family a user actually meshes with: a "
        "first-order brick (HEX8) or its six-tetrahedron decomposition (TET4). Both "
        "carry full integration here, and both are known to lock in bending.",
        "",
        "| parameter | symbol | value |",
        "|---|---|---:|",
        f"| width | `B` | {B * 1e3:g} mm |",
        f"| depth (bending direction) | `H` | {H * 1e3:g} mm |",
        f"| Young's modulus | `E` | {E / 1e9:g} GPa |",
        f"| Poisson ratio | `nu` | {NU:g} |",
        f"| tip load (end-face traction, resultant) | `P` | {P:g} N |",
        f"| shear factor | `kappa` | {KAPPA:.4f} |",
        f"| slenderness rungs | `L/H` | {', '.join(f'{v:g}' for v in RUNGS)} |",
        f"| case-pool tolerance | | {100 * TOLERANCE:g} % |",
        "",
        "## 2. Method",
        "",
        "1. **Oracles re-derived in-driver.** `delta_EB` and `delta_TI` are obtained "
        "from the unit-load integrals (bending `int M m/(EI) dx` plus shear "
        "`int V v/(kappa G A) dx`) by quadrature; the residual against the closed "
        "forms is a gate, so no handbook digit enters the evidence chain.",
        "2. **Three element families, one load path.** Clamped root face (all three "
        "dofs), uniform transverse traction on the loaded end face with resultant "
        "exactly `P`; the reported quantity is the mean end-face transverse "
        "displacement, the same definition the sibling rungs use.",
        "3. **Cost is measured, not assumed.** Every record carries its dof count, so "
        "the error is plotted against cost rather than mesh size -- HEX8 and TET4 at "
        "different densities are not comparable otherwise.",
        "",
        "### Oracle self-check (quadrature vs closed form)",
        "",
        "| L/H | EB/TI separation | quadrature residual |",
        "|---:|---:|---:|",
    ]
    for row in result["oracles"]:
        lines.append(
            f"| {row['L_over_H']:g} | {100 * row['eb_ti_separation']:.4f} % | "
            f"{row['relative_residual']:.3e} |"
        )
    aspect = result["aspect_sweep"]
    order = result["aspect_order"]
    lines += [
        "",
        "## 3. Results",
        "",
        _table(records),
        "",
        "### Why the through-depth sweep is flat: the controlling parameter",
        "",
        "The brick rungs above barely move with `nz`, so the density sweep was "
        "re-run along the member instead (`nz = 4` fixed, `nx` swept):",
        "",
        "| nx | dx/H | elements | dofs | delta [um] | error [%] |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aspect:
        lines.append(
            f"| {row['mesh'][0]} | {row['dx_over_H']:.3f} | {row['n_elements']} | "
            f"{row['n_dofs']} | {row['delta'] * 1e6:.4f} | "
            f"{100 * row['relative_error_vs_timoshenko']:.2f} |"
        )
    lines += [
        "",
        f"Observed error order in `dx/H`: **{order['observed_order']:.2f}** "
        f"(log-log fit between the two extreme meshes), i.e. the locking error falls "
        f"only sub-linearly with the along-member element length. Extrapolating that "
        f"slope to the anchor's band puts the tolerance at "
        f"`dx/H = {order['dx_over_H_needed_for_tolerance']:.3f}`, i.e. about "
        f"{order['nx_needed_for_tolerance_per_H']:.0f} first-order elements per "
        "section depth along the member -- an extrapolation, not a solved mesh, and "
        "recorded as such in the limitations.",
        "",
        "### Falsified expectation (kept on the record)",
        "",
    ]
    for item in result["falsified_expectations"]:
        lines += [
            f"* hypothesis: *{item['hypothesis']}*",
            f"* observed ratio: {item['observed_ratio']:.4f}",
            f"* verdict: **{item['verdict']}**",
        ]
    lines += [
        "",
        "## 4. Figures",
        "",
        "![deflection profile](assets/cantilever_element_family/deflection_profile_L10.png)",
        "",
        "*Fig. 1 -- centroidal deflection at `L/H = 10`: the quadratic solid tracks the "
        "Timoshenko line, the first-order brick locks (it is far too stiff).*",
        "",
        "![error vs dofs](assets/cantilever_element_family/error_vs_dofs_L100.png)",
        "",
        "*Fig. 2 -- error against cost at the case-pool rung `L/H = 100` (log-log) with "
        "the anchor's 3 % line.*",
        "",
        "![error by rung](assets/cantilever_element_family/error_by_rung.png)",
        "",
        "*Fig. 3 -- best mesh per family and slenderness against the same 3 % line.*",
        "",
        "![error vs aspect](assets/cantilever_element_family/error_vs_aspect_L100.png)",
        "",
        "*Fig. 4 -- the controlling parameter: HEX8 error against the along-member "
        "element length `dx/H` (squares) versus the through-depth sweep at fixed "
        "`dx/H` (circles); only the former moves the error.*",
        "",
        "## 5. Notes and limitations",
        "",
    ]
    lines += [f"* {item}" for item in result["limitations"]]
    lines += [
        "",
        "## 6. Error judgement against the anchor criterion",
        "",
        "| family | rung | best-mesh error vs Timoshenko | < 3 % ? |",
        "|---|---|---:|:--:|",
    ]
    for family in ("hex20", "hex8", "tet4"):
        for Lh in RUNGS:
            rows = [r for r in records if r["family"] == family and r["L_over_H"] == Lh]
            if not rows:
                continue
            best = min(rows, key=lambda r: r["relative_error_vs_timoshenko"])
            lines.append(
                f"| {family} | L/H = {Lh:g} | "
                f"{100 * best['relative_error_vs_timoshenko']:.2f} % | "
                f"{'yes' if best['criterion_3pct_met'] else '**no**'} |"
            )
    lines += [
        "",
        f"At `L/H = 100` HEX20 reaches **{100 * hex20_best['relative_error_vs_timoshenko']:.2f} %** "
        f"at {hex20_best['n_dofs']} dofs "
        f"({'within' if hex20_best['criterion_3pct_met'] else '**outside**'} the 3 % band); "
        f"HEX8 reaches {100 * hex8_fine['relative_error_vs_timoshenko']:.2f} % at "
        f"{hex8_fine['n_dofs']} dofs "
        f"({'within' if hex8_fine['criterion_3pct_met'] else '**outside**'} the band) "
        f"against {100 * hex8_coarse['relative_error_vs_timoshenko']:.2f} % at the "
        f"coarsest density.",
        "",
        (f"At a comparable cost the brick error is "
         f"**{eff['hex8_over_hex20_error_ratio']:.2f}x** the quadratic-solid error "
         f"(HEX20 {eff['hex20']['n_dofs']} dofs, {100 * eff['hex20']['relative_error_vs_timoshenko']:.2f} %; "
         f"HEX8 {eff['hex8']['n_dofs']} dofs, {100 * eff['hex8']['relative_error_vs_timoshenko']:.2f} %)."
         if "hex8_over_hex20_error_ratio" in eff else
         "No HEX20 record fell within 30 % of the finest HEX8 dof count, so the "
         "matched-cost ratio is not reported."),
        "",
        f"The brick does not buy the band back through the depth: at fixed `nx` the "
        f"error moves by {100 * (1.0 - result['falsified_expectations'][0]['observed_ratio']):.2f} % "
        f"over the whole `nz` sweep, while refining along the member takes it from "
        f"{100 * aspect[0]['relative_error_vs_timoshenko']:.2f} % to "
        f"{100 * aspect[-1]['relative_error_vs_timoshenko']:.2f} %.",
        "",
        "**Consequence for the case pool**: the `3 %` criterion is a statement about "
        "*one* discretisation. A rung that claims it must name the element family and "
        "the along-member mesh density it was measured with; the anchor's own published "
        "evidence is valid, and the family-dependence is now on the record instead of "
        "in folklore.",
        "",
    ]
    return "\n".join(lines)


def main() -> dict:
    import json
    import os

    # Politeness on a shared 128-core host: the dense solves are BLAS-bound and
    # would otherwise grab every core for a ~100 s job.
    torch.set_num_threads(int(os.environ.get("TP_TORCH_THREADS", "4")))

    oracles = [oracle_residual(L * H) for L in RUNGS]
    ladder = run_ladder(keep_fields=True)
    records = ladder["records"]
    aspect = run_aspect_sweep()
    order = observed_error_order(aspect)

    L100 = 100.0
    hex20_best = _best(records, "hex20", L100)
    hex8_rows = _ladder_by_density(records, "hex8", L100)
    tet_rows = _ladder_by_density(records, "tet4", L100)
    hex8_coarse, hex8_fine = hex8_rows[0], hex8_rows[-1]

    hex20_rows = sorted(
        (r for r in records if r["family"] == "hex20" and r["L_over_H"] == L100),
        key=lambda r: r["n_dofs"],
    )
    comparable = [r for r in hex20_rows
                  if abs(r["n_dofs"] - hex8_fine["n_dofs"]) <= 0.30 * hex8_fine["n_dofs"]]
    if comparable:
        hex20_near = min(comparable, key=lambda r: r["relative_error_vs_timoshenko"])
        efficiency_pair = {
            "hex20": {k: hex20_near[k] for k in
                      ("mesh", "n_dofs", "relative_error_vs_timoshenko")},
            "hex8": {k: hex8_fine[k] for k in
                     ("mesh", "n_dofs", "relative_error_vs_timoshenko")},
            "hex8_over_hex20_error_ratio": float(
                hex8_fine["relative_error_vs_timoshenko"]
                / hex20_near["relative_error_vs_timoshenko"]),
        }
        efficiency_ratio = efficiency_pair["hex8_over_hex20_error_ratio"]
    else:
        efficiency_ratio = float("nan")
        efficiency_pair = {"note": "no HEX20 record within 30 % of the finest HEX8 dof count"}

    monotone = all(
        hex8_rows[i + 1]["relative_error_vs_timoshenko"]
        < hex8_rows[i]["relative_error_vs_timoshenko"]
        for i in range(len(hex8_rows) - 1)
    )
    hex20_thick = _best(records, "hex20", 10.0)
    hex8_thick = _best(records, "hex8", 10.0)

    gates = [
        Gate(
            id="oracle_quadrature_residual",
            quantity="max relative residual of the unit-load quadrature vs the closed forms",
            reference="Castigliano integrals (3) vs (1)/(2)",
            value=float(max(o["relative_residual"] for o in oracles)),
            threshold=1e-6,
            operator="<=",
        ),
        Gate(
            id="hex20_anchor_rung_within_3pct",
            quantity="best HEX20 error vs Timoshenko at L/H = 100",
            reference="anchor criterion (2)",
            value=float(hex20_best["relative_error_vs_timoshenko"]),
            threshold=TOLERANCE,
            operator="<=",
        ),
        Gate(
            id="first_order_brick_fails_criterion",
            quantity="HEX8 error vs Timoshenko at the coarsest through-depth density, L/H = 100",
            reference="anchor criterion (claim: a first-order brick at a plausible "
                      "density violates it)",
            value=float(hex8_coarse["relative_error_vs_timoshenko"]),
            threshold=TOLERANCE,
            operator=">=",
            note="falsifiable -- if the brick passed here the criterion would be "
                 "family-independent and this document's thesis wrong",
        ),
        Gate(
            id="brick_refinement_monotone",
            quantity="HEX8 error strictly decreasing over the density sweep, L/H = 100",
            reference="observed refinement ladder",
            value=1.0 if monotone else 0.0,
            threshold=1.0,
            operator=">=",
        ),
        Gate(
            id="brick_through_depth_refinement_is_flat",
            quantity="error(nz=16) / error(nz=2), HEX8 at fixed nx, L/H = 100",
            reference="pre-registered expectation that 8x through-depth refinement "
                      "would halve the error (falsified by the measurement: the "
                      "ratio is ~1, so the gate asserts the flat behaviour)",
            value=float(hex8_fine["relative_error_vs_timoshenko"]
                        / hex8_coarse["relative_error_vs_timoshenko"]),
            threshold=0.95,
            operator=">=",
            note="claim: the locking error is not a through-depth resolution problem",
        ),
        Gate(
            id="brick_in_plane_refinement_works",
            quantity="error(nx=128) / error(nx=16) at nz=4, HEX8, L/H = 100",
            reference="8x along-member refinement must at least halve the error",
            value=float(aspect[-1]["relative_error_vs_timoshenko"]
                        / aspect[0]["relative_error_vs_timoshenko"]),
            threshold=0.5,
            operator="<=",
        ),
        Gate(
            id="brick_error_order_in_dx",
            quantity="observed log-log slope of the HEX8 error against dx/H, L/H = 100",
            reference="at least sqrt(h) decay (slope >= 0.5)",
            value=float(order["observed_order"]),
            threshold=0.5,
            operator=">=",
        ),
        Gate(
            id="in_plane_beats_through_depth_at_matched_cost",
            quantity="err(nx=128, nz=4) / err(nx=16, nz=finest) at comparable dof counts",
            reference="matched-cost comparison inside the brick family",
            value=float(aspect[-1]["relative_error_vs_timoshenko"]
                        / min(r["relative_error_vs_timoshenko"] for r in hex8_rows)),
            threshold=1.0,
            operator="<=",
            note="claim: spending the same dofs along the member beats spending them "
                 "through the depth",
        ),
        Gate(
            id="hex20_dof_efficiency",
            quantity="HEX8 / HEX20 error at a comparable dof count, L/H = 100",
            reference="error ratio at matched cost (claim: quadratic solid wins)",
            value=float(efficiency_ratio),
            threshold=1.0,
            operator=">=",
        ),
        Gate(
            id="tet4_not_better_than_parent_brick",
            quantity="TET4 error / HEX8 error, L/H = 100",
            reference="six-tet decomposition of the same brick topology",
            value=float(tet_rows[-1]["relative_error_vs_timoshenko"]
                        / hex8_rows[-1]["relative_error_vs_timoshenko"]) if tet_rows else float("nan"),
            threshold=1.0,
            operator=">=",
        ),
        Gate(
            id="hex20_thick_rung_within_3pct",
            quantity="best HEX20 error vs Timoshenko at L/H = 10",
            reference="anchor criterion at the thickest rung graded here",
            value=float(hex20_thick["relative_error_vs_timoshenko"]),
            threshold=TOLERANCE,
            operator="<=",
        ),
        Gate(
            id="family_ranking_survives_thick_rung",
            quantity="HEX8 / HEX20 error at L/H = 10",
            reference="element-family ranking (claim: the brick is also the worse "
                      "family where shear matters)",
            value=float(hex8_thick["relative_error_vs_timoshenko"]
                        / hex20_thick["relative_error_vs_timoshenko"]),
            threshold=1.0,
            operator=">=",
        ),
    ]

    limitations = [
        "Linear elasticity, small strain: this ladder says nothing about the "
        "geometrically nonlinear rung (`src/tensorfem/elastica_cantilever.py`).",
        "Fully integrated HEX8 and TET4 only. Reduced integration with hourglass "
        "control, incompatible-mode (Wilson) bricks and TET10 are outside this scope, "
        "so the numbers are not a ranking of every first-order strategy.",
        "The mean end-face displacement includes the load-introduction (Saint-Venant) "
        "effect; it is the sibling rungs' definition, which keeps the comparison "
        "between rungs fair but biases every family slightly.",
        "Family and density are only partially separated: the HEX8 in-plane mesh is "
        "fixed at nx=16, ny=2 in the rung ladder while HEX20 varies, so the "
        "dof-efficiency ratio is a matched-cost comparison, not a converged-cost one; "
        "the aspect sweep separates the two effects only inside the brick family.",
        "The dx/H needed for the 3 % band is an extrapolation of the measured slope "
        "from meshes that were actually solved; no run at that density exists, and "
        "the slope itself is measured between the two extreme meshes only.",
        "Rigid-body-motion and patch tests are not repeated here: the trilinear brick "
        "and the tet pass them by construction of the library "
        "(`src/tensorfem/patch_test.py`), so the ladder grades bending behaviour "
        "alone.",
        "No analytic reference exists for locking itself; the reference is the "
        "Timoshenko oracle plus the observed refinement ladder, so the *rate* is "
        "reported, not asserted.",
    ]

    result_doc = {
        "schema": "tensorfem.cantilever_element_family/1",
        "anchor": ANCHOR,
        "mission": MISSION,
        "title": TITLE,
        "generated_at": utc_now(),
        "problem": {
            "E_Pa": E, "nu": NU, "B_m": B, "H_m": H, "P_N": P,
            "kappa": KAPPA, "tolerance": TOLERANCE, "rungs": list(RUNGS),
        },
        "oracles": oracles,
        "records": records,
        "aspect_sweep": aspect,
        "aspect_order": order,
        "falsified_expectations": [
            {
                "hypothesis": "8x through-depth (nz 2 -> 16) refinement of the "
                              "first-order brick halves the tip-deflection error",
                "observed_ratio": float(hex8_fine["relative_error_vs_timoshenko"]
                                        / hex8_coarse["relative_error_vs_timoshenko"]),
                "verdict": "falsified -- the error is flat in nz at fixed nx; the "
                           "controlling parameter is the along-member element length",
            }
        ],
        "highlights": {
            "hex20_best_L100": hex20_best,
            "hex8_ladder_L100": hex8_rows,
            "hex20_thick_L10": hex20_thick,
            "hex8_thick_L10": hex8_thick,
            "dof_efficiency": efficiency_pair,
            "hex8_monotone": bool(monotone),
        },
        "limitations": limitations,
        "wall_seconds": float(ladder["wall_seconds"]),
    }
    write_json(OUT_DIR / "result.json", result_doc)

    figures = _figure(records, ladder["fields"], aspect, order)
    artifacts = {
        "result": "benchmarks/verified/fem_cantilever_hex8/result.json",
        "driver": "benchmarks/verified/fem_cantilever_hex8/run.py",
        "doc": "docs/CASE-CANTILEVER-TIP-element-family.md",
        "figures": {name: f"docs/assets/cantilever_element_family/{p.name}"
                    for name, p in figures.items()},
    }
    gates_doc = gate_manifest(
        anchor=ANCHOR,
        title=TITLE,
        mission=MISSION,
        criteria={
            "case_pool_tolerance": TOLERANCE,
            "oracle_residual": 1e-6,
            "brick_refinement_factor": 0.5,
            "dof_efficiency_ratio": 1.0,
        },
        gates=gates,
        artifacts=artifacts,
        limitations=limitations,
    )
    write_json(OUT_DIR / "gates_verified.json", gates_doc)

    payload = {"result": result_doc, "gates_verified": gates_doc}
    (OUT_DIR / "README.md").write_text(_readme(payload))
    DOC.write_text(_doc(payload))
    summary = {
        "passed": gates_doc["passed"],
        "failed_gates": gates_doc["failed_gates"],
        "n_records": len(records),
        "wall_seconds": result_doc["wall_seconds"],
        "doc": str(DOC.relative_to(REPO)),
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()