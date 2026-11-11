#!/usr/bin/env python3
"""GAP-FEM-CURVED-BEAM-VERIFY anchor: quarter-ring curved cantilever vs closed form.

Physics
-------
A quarter of a circular ring of radius *R* and rectangular section
``width x thickness`` is clamped at one end and loaded at the free end by a
point force *P* acting along the free-end tangent.  The tip tangent is
perpendicular to the root tangent, so the problem couples membrane (hoop)
action with bending and exercises exactly the curved-shell mechanism that
flat-facet shells are known to distort.

Reference (Castigliano on the thin, inextensible-section curved bar; axial and
transverse-shear terms retained):

    M(phi) = -P R (1 - cos phi),   phi = 0 at the free end, phi = pi/2 at clamp

* tangent load -> tip displacement along the tangent
  ``delta_t = P R^3 (3 pi/4 - 2) / (E I) + pi P R /(4 E A) + pi P R /(4 k G A)``
* the *transverse* (radial) tip displacement under the same tangent load is
  ``delta_r = P R^3 / (2 E I)`` (Maxwell--Betti reciprocity with the radial
  load case, whose own direct response is ``pi P R^3 / (4 E I)``).

The three coefficients ``3 pi/4 - 2 = 0.356194``, ``1/2`` and ``pi/4`` are
cross-checked against an independent polygonal Euler--Bernoulli frame model,
against the flat-facet ``tensorfem.shell4.shell4_stiffness`` and against the
metric-consistent ``tensorfem.cylindrical_shell4.cylindrical_shell4_stiffness``;
all three numerical routes agree to better than 0.5%.

Model
-----
Two FE models are provided by ``tensorfem.curved_beam_benchmark``:

``element="flat"``
    ``tensorfem.shell4.shell4_stiffness`` on a faceted ring whose axis is the
    global ``z`` axis (radial = ``x`` at the free end).
``element="cylindrical"``
    ``tensorfem.cylindrical_shell4.cylindrical_shell4_stiffness``, whose exact
    cylindrical metric requires the ring axis to be the global ``x`` axis
    (radial = ``z`` at the free end).

Both models are clamped over all six degrees of freedom on the root edge; the
free-end edge carries an equal share of the total tangent load.  Reported
free-end components are magnitudes along the declared tangent/normal axes.

Verification criteria (GAP-FEM-CURVED-BEAM-VERIFY acceptance):
    C1  Tangent tip-displacement error < 1% vs closed form at the finest mesh.
    C2  Radial tip-displacement error < 1% vs closed form at the finest mesh.
    C3  Convergence is monotone in the tangent response (error decreases with n).
    C4  Equilibrium: root reaction ≈ applied load (< 1% residual).
    C5  Both flat and cylindrical elements pass C1/C2 (element-route independence).

Outputs: ``result.json``, ``gates_verified.json``,
         ``curved_beam_convergence.png``.

Usage::

    PYTHONPATH=src:benchmarks/verified python benchmarks/verified/fem_curved_beam/run.py
"""

from __future__ import annotations

import math
import platform
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
# _harness lives in benchmarks/verified/ (the parent of this anchor dir)
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "src"))

from _harness import (
    Gate,
    gate_manifest,
    pyplot,
    utc_now,
    write_json,
)

from tensorfem.curved_beam_benchmark import (
    RADIAL_COEFFICIENT,
    RADIAL_LOAD_COEFFICIENT,
    TANGENT_COEFFICIENT,
    quarter_ring_reference,
    quarter_ring_shell,
)

DTYPE = torch.float64

# --- benchmark parameters (thin curved bar, bending dominated) -----
RADIUS = 10.0        # m, ring radius
WIDTH = 0.2          # m, section width (out of plane)
THICKNESS = 0.1      # m, section thickness (in plane)
YOUNG = 2.1e5        # Pa, Young modulus (steel-like)
POISSON = 0.3
LOAD = 1.0           # N, unit tangent end load
N_WIDTH = 2          # elements across the width

MESHES = (4, 8, 16, 32, 64)


def _run_levels() -> list[dict]:
    """Run the refinement ladder for both elements and collect rows."""
    rows = []
    for element in ("flat", "cylindrical"):
        for n_arc in MESHES:
            t0 = time.perf_counter()
            result = quarter_ring_shell(
                n_arc, N_WIDTH, element=element,
                radius=RADIUS, width=WIDTH, thickness=THICKNESS,
                young=YOUNG, poisson=POISSON, load=LOAD,
            )
            elapsed = time.perf_counter() - t0
            rows.append({
                "element": element,
                "n_arc": n_arc,
                "n_width": N_WIDTH,
                "ndof": 6 * result.nodes.shape[0],
                "n_elements": result.elements.shape[0],
                "tip_tangent": result.tip_tangent,
                "tip_tangent_exact": result.tip_tangent_exact,
                "tangent_error_pct": 100.0 * result.tangent_error,
                "tip_radial": result.tip_radial,
                "tip_radial_exact": result.tip_radial_exact,
                "radial_error_pct": 100.0 * result.radial_error,
                "reaction_tangent": result.reaction_tangent,
                "reaction_moment": result.reaction_moment,
                "load": result.load,
                "elapsed_s": elapsed,
            })
    return rows


def _build_gates(rows: list[dict]) -> list[Gate]:
    """Grade the acceptance criteria against the closed-form reference."""
    # finest mesh (n_arc = 64) for each element
    finest = {r["element"]: r for r in rows if r["n_arc"] == 64}

    gates = []
    for element in ("flat", "cylindrical"):
        r = finest[element]
        gates.append(Gate(
            id=f"C1_tangent_{element}",
            quantity="Tangent tip-displacement relative error",
            reference="delta_t=(3pi/4-2)PR^3/(EI)+piPR/(4EA)+piPR/(4kGA)",
            value=r["tangent_error_pct"],
            threshold=1.0,
            operator="<=",
            note=f"n_arc=64, {element} element",
        ))
        gates.append(Gate(
            id=f"C2_radial_{element}",
            quantity="Radial tip-displacement relative error",
            reference="delta_r=PR^3/(2EI) (Maxwell-Betti reciprocity)",
            value=r["radial_error_pct"],
            threshold=1.0,
            operator="<=",
            note=f"n_arc=64, {element} element",
        ))

    # C3: monotone convergence in tangent response (both elements)
    for element in ("flat", "cylindrical"):
        errs = [r["tangent_error_pct"] for r in rows if r["element"] == element]
        # errors must strictly decrease from n=4 to n=64
        monotone = all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))
        gates.append(Gate(
            id=f"C3_monotone_{element}",
            quantity="Tangent-error monotone decrease over refinement ladder",
            reference="error(n=4) > error(n=8) > ... > error(n=64)",
            value=1.0 if monotone else 0.0,
            threshold=1.0,
            operator=">=",
            note=f"errors={[round(e,3) for e in errs]}",
        ))

    # C4: equilibrium (root reaction ≈ applied load) at finest mesh
    for element in ("flat", "cylindrical"):
        r = finest[element]
        eq_residual = abs(r["reaction_tangent"] / r["load"] - 1.0) * 100.0
        gates.append(Gate(
            id=f"C4_equilibrium_{element}",
            quantity="Root reaction vs applied load residual",
            reference="Sigma F_reaction = Sigma F_applied",
            value=eq_residual,
            threshold=1.0,
            operator="<=",
            note=f"n_arc=64, {element} element",
        ))

    return gates


def _make_figure(rows: list[dict], out: Path) -> Path:
    """Convergence figure: tangent & radial error vs n_arc for both elements."""
    plt = pyplot()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for element, marker, color in (("flat", "o", "#1f77b4"), ("cylindrical", "s", "#d62728")):
        sub = [r for r in rows if r["element"] == element]
        ns = [r["n_arc"] for r in sub]
        ax[0].semilogy(ns, [r["tangent_error_pct"] for r in sub],
                       marker=marker, color=color, label=f"{element} tangent")
        ax[1].semilogy(ns, [r["radial_error_pct"] for r in sub],
                       marker=marker, color=color, label=f"{element} radial")
    for a in ax:
        a.set_xlabel("n_arc (elements along the quarter ring)")
        a.set_ylabel("relative error vs closed form (%)")
        a.set_ylim(1e-4, 1e2)
        a.grid(True, which="both", alpha=0.3)
        a.legend()
    ax[0].set_title("Tangent tip-displacement")
    ax[1].set_title("Radial tip-displacement")
    fig.suptitle("Quarter-ring curved cantilever: FEM vs Castigliano closed form")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    t_start = time.perf_counter()
    rows = _run_levels()
    gates = _build_gates(rows)
    manifest = gate_manifest(
        anchor="fem_curved_beam",
        title="Quarter-ring curved cantilever vs Castigliano closed form",
        mission="GAP-FEM-CURVED-BEAM-VERIFY",
        criteria={
            "tangent_error": "< 1%",
            "radial_error": "< 1%",
            "monotone": "strict decrease",
            "equilibrium_residual": "< 1%",
        },
        gates=gates,
        artifacts={
            "result_json": "result.json",
            "gates_verified_json": "gates_verified.json",
            "convergence_figure": "curved_beam_convergence.png",
        },
        limitations=[
            "Closed form is the thin, inextensible-section curved-bar idealisation; "
            "axial and transverse-shear terms are retained but small.",
            "Both elements are verified on the same physical configuration; "
            "the cylindrical element measures theta with the opposite sense to the "
            "flat mesh, so reported components are magnitudes along declared axes.",
        ],
    )

    # reference block
    ref = quarter_ring_reference(RADIUS, WIDTH, THICKNESS, YOUNG, POISSON, LOAD)
    result = {
        "benchmark": "fem_curved_beam",
        "mission": "GAP-FEM-CURVED-BEAM-VERIFY",
        "generated_at": utc_now(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "machine": platform.machine(),
            "dtype": "torch.float64",
        },
        "physics": {
            "radius_m": RADIUS,
            "width_m": WIDTH,
            "thickness_m": THICKNESS,
            "young_modulus_Pa": YOUNG,
            "poisson_ratio": POISSON,
            "load_N": LOAD,
            "n_width": N_WIDTH,
        },
        "reference": {
            "source": "Castigliano on thin curved bar (quarter ring)",
            "formula": "delta_t=(3pi/4-2)PR^3/(EI)+piPR/(4EA)+piPR/(4kGA); "
                       "delta_r=PR^3/(2EI)",
            "coefficients": {
                "tangent": TANGENT_COEFFICIENT,
                "radial": RADIAL_COEFFICIENT,
                "radial_load": RADIAL_LOAD_COEFFICIENT,
            },
            "delta_tangent_m": ref["delta_tangent"],
            "delta_radial_m": ref["delta_radial"],
            "delta_radial_load_m": ref["delta_radial_load"],
        },
        "levels": rows,
        "elapsed_total_s": time.perf_counter() - t_start,
    }

    write_json(HERE / "result.json", result)
    write_json(HERE / "gates_verified.json", manifest)
    fig = _make_figure(rows, HERE / "curved_beam_convergence.png")

    print(f"=== fem_curved_beam verification ===")
    print(f"reference: delta_t={ref['delta_tangent']:.6e} m, "
          f"delta_r={ref['delta_radial']:.6e} m")
    for r in rows:
        if r["n_arc"] in (4, 16, 64):
            print(f"  {r['element']:>10} n={r['n_arc']:>3}  "
                  f"tangent_err={r['tangent_error_pct']:.4f}%  "
                  f"radial_err={r['radial_error_pct']:.4f}%  "
                  f"eq_res={abs(r['reaction_tangent']/r['load']-1)*100:.2e}%")
    print(f"gates: {sum(g.passed for g in gates)}/{len(gates)} passed")
    print(f"figure: {fig}")


if __name__ == "__main__":
    main()
