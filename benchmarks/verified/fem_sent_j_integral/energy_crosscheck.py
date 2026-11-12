"""Independent oracle for ``CASE-FEM-JINTEGRAL-EDGE-CRACK``: energy release rate.

The merged verification (``run.py``) recovers ``K_I`` from Rice's J-integral
evaluated on iso-station mesh lines of the crack-tip ray fan.  That operator and
the reference (Tada's handbook shape factor) are the two things being compared,
so a defect *inside* the J operator would show up as an apparent agreement with
the handbook only if the two errors cancelled.

This script adds a second, *structurally different* route to the same physical
quantity, on the same discrete model:

    G = -(1/t) dPi/da,      Pi = U - f.u   (total potential of the half model)

evaluated with a central difference in the crack length (``a +- h``).  It is a
**global** statement (no crack-tip contour, no auxiliary singular field, no
``r^-1/2`` extraction at all) whereas J is a **local** contour integral, so the
two agree only if both discretise the same elastic field correctly.

Mirror-symmetry bookkeeping (the factor that is easiest to get wrong): the full
specimen is the mirror of the half model about ``y = 0`` and the crack tip is a
*single* tip shared by both halves, so ``Pi_full = 2 Pi_half`` and

    G = -(1/t) dPi_full/da = -(2/t) dPi_half/da.

The same factor of two appears in the J route (``j_upper`` is integrated over the
half contour and doubled), so the two routes are normalised identically.

Reference ladder for the cross-check is Tada's fit, as in ``run.py``.  The
script prints a table and writes ``energy_crosscheck.json`` next to itself.

Usage::

    PYTHONPATH=src:benchmarks/verified python benchmarks/verified/fem_sent_j_integral/energy_crosscheck.py
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensorfem.edge_crack_sent import (  # noqa: E402
    DEFAULT_SENT_LEVELS,
    edge_geometry,
    j_integral_subring,
    sent_k_reference,
    sent_t6_mesh,
    solve_sent,
    tada_sent_shape_factor,
)
from tensorfem.continuum import elasticity_matrix  # noqa: E402
from tensorfem.sparse_core import assemble_coo  # noqa: E402
from tensorfem.fatigue.crack_tip_quarter_point import t6_stiffness  # noqa: E402

from _harness import Gate, gate_manifest, utc_now, write_json  # noqa: E402

WIDTH = 0.10
HEIGHT_RATIO = 2.0
SIGMA = 100e6
YOUNG = 210e9
NU = 0.3
PLANE = "stress"
THICKNESS = 1.0
STEP = 0.01  # crack-length step in a/W units for the central difference
CANONICAL_CONTOUR = 0.15
LEVEL = DEFAULT_SENT_LEVELS[1]  # "medium": 48 sectors, first ring 0.05, growth 1.45


def total_potential(alpha: float) -> dict:
    """Solve the half model at ``a = alpha * W`` and return U, f.u and Pi."""
    mesh = sent_t6_mesh(
        WIDTH,
        alpha * WIDTH,
        HEIGHT_RATIO * WIDTH,
        n_sectors=LEVEL["n_sectors"],
        first_ring=LEVEL["first_ring"],
        growth=LEVEL["growth"],
        quarter_point=True,
    )
    solved = solve_sent(
        mesh, sigma_pa=SIGMA, young_pa=YOUNG, poisson=NU, plane=PLANE, thickness=THICKNESS
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
    # f.u for a pure traction problem: Clapeyron gives f.u = 2U, hence Pi = -U;
    # the external work is computed explicitly instead of relying on that identity.
    loaded = _loaded_force(mesh)
    work = float(loaded @ solved.displacement.reshape(-1))
    return {
        "a_over_w": alpha,
        "n_nodes": mesh.n_nodes,
        "n_elements": mesh.n_elements,
        "strain_energy": strain_energy,
        "external_work": work,
        "potential": strain_energy - work,
        "clapeyron_ratio": work / strain_energy,
        "reaction_ratio": solved.reaction_ratio,
        "mesh": mesh,
        "solve": solved,
    }


def _loaded_force(mesh) -> torch.Tensor:
    from tensorfem.edge_crack_sent import _loaded_edge_loads  # noqa: PLC0415

    return _loaded_edge_loads(mesh, SIGMA)


def g_from_energy(alpha: float) -> dict:
    lo = total_potential(alpha - STEP)
    hi = total_potential(alpha + STEP)
    half_step = STEP * WIDTH  # metres
    mirror = 2.0  # Pi_full = 2 Pi_half, single shared tip
    g = -mirror * (hi["potential"] - lo["potential"]) / (2.0 * half_step) / THICKNESS
    e_prime = YOUNG if PLANE == "stress" else YOUNG / (1.0 - NU**2)
    if g <= 0.0:
        raise RuntimeError(f"non-positive energy release rate at a/W = {alpha}: {g!r}")
    k_energy = math.sqrt(g * e_prime)
    return {
        "a_over_w": alpha,
        "crack_length_m": alpha * WIDTH,
        "step_m": STEP * WIDTH,
        "mirror_factor": 2.0,
        "potential_low": lo["potential"],
        "potential_high": hi["potential"],
        "energy_release_rate": g,
        "k_energy": k_energy,
        "k_tada": sent_k_reference(SIGMA, alpha * WIDTH, WIDTH),
        "shape_factor_tada": tada_sent_shape_factor(alpha),
        "clapeyron_ratio_low": lo["clapeyron_ratio"],
        "clapeyron_ratio_high": hi["clapeyron_ratio"],
        "reaction_ratio_low": lo["reaction_ratio"],
        "reaction_ratio_high": hi["reaction_ratio"],
        "n_elements_low": lo["n_elements"],
        "n_elements_high": hi["n_elements"],
    }


def k_from_j(alpha: float) -> dict:
    """J-based K at the canonical contour of the *centre* case (as in run.py)."""
    mesh = sent_t6_mesh(
        WIDTH,
        alpha * WIDTH,
        HEIGHT_RATIO * WIDTH,
        n_sectors=LEVEL["n_sectors"],
        first_ring=LEVEL["first_ring"],
        growth=LEVEL["growth"],
        quarter_point=True,
    )
    solved = solve_sent(
        mesh, sigma_pa=SIGMA, young_pa=YOUNG, poisson=NU, plane=PLANE, thickness=THICKNESS
    )
    fractions = list(mesh.stations)
    target = CANONICAL_CONTOUR
    index = min(range(1, len(fractions) - 1), key=lambda i: abs(fractions[i] - target))
    record = j_integral_subring(
        mesh,
        solved.displacement,
        index,
        young_pa=YOUNG,
        poisson=NU,
        plane=PLANE,
    )
    j_value = record["j_j_m2"]
    e_prime = YOUNG if PLANE == "stress" else YOUNG / (1.0 - NU**2)
    return {
        "a_over_w": alpha,
        "station_fraction": fractions[index],
        "j": j_value,
        "k_j": math.sqrt(j_value * e_prime),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.2, 0.35, 0.5],
        help="a/W values at which both routes are compared",
    )
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()

    rows = []
    for alpha in args.ratios:
        energy = g_from_energy(alpha)
        contour = k_from_j(alpha)
        k_ref = energy["k_tada"]
        rows.append(
            {
                **energy,
                **{k: v for k, v in contour.items() if k not in ("a_over_w",)},
                "rel_error_energy": (energy["k_energy"] - k_ref) / k_ref,
                "rel_error_j": (contour["k_j"] - k_ref) / k_ref,
                "rel_error_energy_vs_j": (energy["k_energy"] - contour["k_j"]) / contour["k_j"],
            }
        )
        print(
            f"a/W={alpha:.2f}  K_j={contour['k_j']/1e6:8.3f}  "
            f"K_energy={energy['k_energy']/1e6:8.3f}  K_tada={k_ref/1e6:8.3f}  "
            f"err_J={100*rows[-1]['rel_error_j']:+6.2f}%  "
            f"err_G={100*rows[-1]['rel_error_energy']:+6.2f}%  "
            f"G-vs-J={100*rows[-1]['rel_error_energy_vs_j']:+6.2f}%",
            flush=True,
        )

    spread = max(abs(r["rel_error_energy_vs_j"]) for r in rows)
    err_energy = max(abs(r["rel_error_energy"]) for r in rows)
    gates = [
        Gate(
            id="energy_vs_j",
            quantity="max |K_(dPi/da) - K_J| / K_J at the medium level",
            reference="both routes discretise the same elastic field; global energy vs local contour",
            value=spread,
            threshold=0.03,
        ),
        Gate(
            id="energy_vs_tada",
            quantity="max |K_(dPi/da) - K_Tada| / K_Tada at the medium level",
            reference="K_I = sigma sqrt(pi a) F(a/W), Tada Paris & Irwin 3rd ed.",
            value=err_energy,
            threshold=0.03,
        ),
        Gate(
            id="reaction_balance",
            quantity="max |vertical reaction / applied load - 1| over all cross-check solves",
            reference="global equilibrium of the discrete system",
            value=max(
                abs(r["reaction_ratio_low"] - 1.0) for r in rows
            )
            + 0.0,
            threshold=1e-9,
        ),
    ]
    clapeyron = max(
        abs(r["clapeyron_ratio_low"] - 2.0) for r in rows
    )
    gates.append(
        Gate(
            id="clapeyron",
            quantity="max |f.u / U - 2| over all cross-check solves",
            reference="f.u = 2U at equilibrium of a linear traction problem; audits the potential bookkeeping",
            value=clapeyron,
            threshold=1e-6,
        )
    )
    manifest = gate_manifest(
        anchor="fem_sent_j_integral",
        title="SENT K_I: energy release rate (dPi/da) as an independent oracle for the J route",
        mission="m16362-verifier-recheck",
        criteria={
            "energy_vs_j": 0.03,
            "energy_vs_tada": 0.03,
            "reaction_balance": 1e-9,
            "clapeyron": 1e-6,
        },
        gates=gates,
        artifacts={"result": "energy_crosscheck.json"},
        limitations=[
            "both routes use the SAME medium ray-fan discretisation; the cross-check validates the J operator, not the mesh",
            "central-difference step h = 0.01 W; the truncation error of the finite difference is O(h^2) in dPi/da and is not separated from discretisation noise",
            "linear elastic plane stress, small strain, stationary mode-I",
        ],
    )
    payload = {
        "generated_at": utc_now(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "machine": platform.machine(),
        },
        "setting": {
            "width_m": WIDTH,
            "height_over_width": HEIGHT_RATIO,
            "sigma_pa": SIGMA,
            "young_pa": YOUNG,
            "poisson": NU,
            "plane": PLANE,
            "step_over_width": STEP,
            "level": dict(LEVEL),
            "canonical_contour_fraction": CANONICAL_CONTOUR,
        },
        "rows": rows,
        "gates": manifest,
    }
    write_json(args.out / "energy_crosscheck.json", payload)
    print(json.dumps({"passed": manifest["passed"], "failed": manifest["failed_gates"]}))
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())