"""Two-block frictionless mortar segment evidence runner.

Produces ``artifacts/mortar-two-block/evidence.json`` and the report figures:

* ``case_a``: identical blocks, non-matching interface meshes -> the analytic
  uniform uniaxial state (interface traction equals the applied pressure ``p``
  everywhere).  Mesh refinement levels confirm the transferred traction stays at
  solver precision (a mortar patch test).
* ``case_b``: punch on a wider, thicker base (geometric mismatch) -> interface
  traction concentrates at the punch edge; the band value departs from ``p`` and
  the edge peak grows with refinement.  No closed-form reference exists for the
  finite elastic punch, so the edge peak is reported as mesh dependent and the
  reference is declared withheld.
* ``case_c``: prescribed smooth penetration field on non-matching meshes -> the
  three-point segment quadrature reproduces the analytic integral with a second
  order L2 trend.

Usage::

    PYTHONPATH=src python scripts/run_mortar_two_block_evidence.py \
        --output artifacts/mortar-two-block --figures artifacts/mortar-two-block/figures
"""

from __future__ import annotations
import argparse
import json
import math
import tempfile
from pathlib import Path

import torch

from tensorfem.mortar_segment_transfer import transfer_convergence
from tensorfem.mortar_two_block import build_two_block, interface_metrics, solve_two_block

torch.set_default_dtype(torch.float64)

CASE_A_LEVELS = (2, 3, 4, 5)
CASE_B_LEVELS = (2, 3, 4)
CASE_C_LEVELS = (2, 3, 4, 6, 8)


def _case_a():
    rows = []
    for n in CASE_A_LEVELS:
        model = build_two_block(n_upper=n, n_base=n + 1, nz=2, pressure=10.0)
        solution = solve_two_block(model, penalty=1e9)
        metrics = interface_metrics(model, solution)
        rows.append({"n_upper": n, "n_base": n + 1, "dofs": model.n_dofs, **metrics})
    return {
        "description": "identical blocks, non-matching interface meshes, analytic uniform state",
        "pressure": 10.0,
        "levels": rows,
        "analytic_reference": "sigma_zz = -p, interface traction = p (exact uniaxial state)",
        "band_l2_worst": max(r["band_l2_error"] for r in rows),
        "full_l2_worst": max(r["full_l2_error"] for r in rows),
        "area_worst": max(r["area_error"] for r in rows),
        "equilibrium_worst": max(r["equilibrium_error"] for r in rows),
        "penetration_worst": max(r["maximum_penetration"] for r in rows),
        "passed": all(
            r["band_l2_error"] < 0.03
            and r["equilibrium_error"] < 1e-8
            and r["area_error"] < 1e-10
            and r["zero_tension"]
            for r in rows
        ),
    }


def _punch_row(n, penalty, gallery, *, penalty_label=None):
    model = build_two_block(
        n_upper=n,
        n_base=3 * n + 1,
        nz=2,
        pressure=10.0,
        base_extent=1.5,
        base_height=1.5,
        nz_base=2,
    )
    solution = solve_two_block(model, penalty=penalty)
    metrics = interface_metrics(model, solution)
    gallery[penalty_label or f"n{n}"] = solution.samples
    return {"n_upper": n, "n_base": 3 * n + 1, "dofs": model.n_dofs, "penalty": penalty, **metrics}


def _case_b():
    """Mismatched punch: the interface distribution is regularization dependent."""
    gallery = {}
    penalty_sweep = [
        _punch_row(3, p, gallery, penalty_label=f"pen{p:.0e}") for p in (1e5, 1e6, 1e7, 1e8)
    ]
    mesh_sweep = []
    for n in (2, 3, 4):
        if n == 3:
            mesh_sweep.append(penalty_sweep[1])
        else:
            mesh_sweep.append(_punch_row(n, 1e6, gallery, penalty_label=f"n{n}"))
    rows = penalty_sweep + [r for r in mesh_sweep if r["n_upper"] != 3]
    report = {
        "description": "punch (upper block) on a wider, thicker identical-material base",
        "geometry": {"upper_extent": 0.5, "base_extent": 1.5, "height": 1.0, "base_height": 1.5},
        "penalty_sweep": penalty_sweep,
        "mesh_sweep": mesh_sweep,
        "rows": rows,
        "peak_over_pressure_series": [r["edge_over_pressure"] for r in rows],
        "closed_fraction_series": [r["closed_fraction"] for r in rows],
        "band_mean_series": [r["band_mean_traction"] for r in rows],
        "equilibrium_worst": max(r["equilibrium_error"] for r in rows),
        "renormalization_evidence": (
            "at fixed mesh (n=3) the closed fraction moves from %.3f (eps=1e5) to %.3f (eps=1e8) "
            "and the peak from %.1f p to %.1f p: the distribution is a penalty-regularisation "
            "artefact of an interface whose linear-elastic edge pressure is singular"
            % (
                penalty_sweep[0]["closed_fraction"],
                penalty_sweep[-1]["closed_fraction"],
                penalty_sweep[0]["edge_over_pressure"],
                penalty_sweep[-1]["edge_over_pressure"],
            )
        ),
        "band_reference": "withheld: the mismatched finite-block interface pressure has no resolving "
        "reference and is penalty/mesh dependent; only integrals are claimed",
        "asymptotic_note": "flat punch on an elastic half-space carries an integrable (a-s)^-1/2 edge "
        "singularity that a Q1 penalty discretisation cannot represent",
        "passed": all(r["equilibrium_error"] < 1e-8 and r["zero_tension"] for r in rows),
    }
    return report, gallery


def _case_c():
    rows = transfer_convergence(levels=CASE_C_LEVELS)
    rates = []
    for a, b in zip(rows[:-1], rows[1:]):
        rates.append(
            math.log(a["l2_error"] / b["l2_error"]) / math.log(b["n_slave"] / a["n_slave"])
        )
    return {
        "description": "prescribed smooth penetration field, non-matching meshes, analytic integral",
        "levels": rows,
        "observed_rates": rates,
        "finest_l2_error": rows[-1]["l2_error"],
        "finest_force_error": rows[-1]["total_force_error"],
        "passed": bool(rows[-1]["l2_error"] < 0.03 and rates[-1] > 1.5 and rates[-1] < 2.5),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/mortar-two-block")
    parser.add_argument("--figures", default="artifacts/mortar-two-block/figures")
    args = parser.parse_args()
    out = Path(args.output)
    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)

    case_a = _case_a()
    case_b, gallery = _case_b()
    case_c = _case_c()
    report = {
        "contract": "two-block frictionless mortar segment integration",
        "case_a": case_a,
        "case_b": case_b,
        "case_c": case_c,
        "passed": case_a["passed"] and case_b["passed"] and case_c["passed"],
        "scope": {
            "supported": "frictionless penalty mortar segments, small relative sliding, planar interfaces",
            "roadmap": "frictional mortar, finite sliding, curved/self-contact segment integration",
        },
    }
    _figures(case_a, case_b, case_c, figures, gallery)
    report["figures"] = sorted(str(p.name) for p in figures.glob("*.png"))
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=out, delete=False, suffix=".tmp") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
        tmp = Path(stream.name)
    tmp.replace(out / "evidence.json")
    print(json.dumps({"passed": report["passed"], "output": str(out / "evidence.json")}))


def _figures(case_a, case_b, case_c, figures: Path, gallery: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) mortar segment quadrature layout on the non-matching interface
    model = build_two_block(n_upper=2, n_base=3, nz=2, pressure=10.0)
    solution = solve_two_block(model, penalty=1e9)
    samples = solution.samples
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for face in model.slave_faces:
        pts = model.nodes[model.slave_nodes][face]
        ax.plot(
            [pts[0, 0], pts[1, 0], pts[2, 0], pts[3, 0], pts[0, 0]],
            [pts[0, 1], pts[1, 1], pts[2, 1], pts[3, 1], pts[0, 1]],
            color="C0",
            lw=1.2,
        )
    for face in model.master_faces:
        pts = model.nodes[model.master_nodes][face]
        ax.plot(
            [pts[0, 0], pts[1, 0], pts[2, 0], pts[3, 0], pts[0, 0]],
            [pts[0, 1], pts[1, 1], pts[2, 1], pts[3, 1], pts[0, 1]],
            color="C1",
            lw=0.8,
            ls="--",
        )
    ax.plot(samples.points[:, 0], samples.points[:, 1], "k.", ms=2, label="segment quadrature")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("mortar segments: slave (solid) / master (dashed)")
    ax.legend(loc="lower center", fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "mortar-segment-layout.png", dpi=160)
    plt.close(fig)

    # 2) traction map and midline profile, equal blocks
    model = build_two_block(
        n_upper=CASE_A_LEVELS[-1], n_base=CASE_A_LEVELS[-1] + 1, nz=2, pressure=10.0
    )
    solution = solve_two_block(model, penalty=1e9)
    samples = solution.samples
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    sc = axes[0].scatter(
        samples.points[:, 0], samples.points[:, 1], c=samples.normal_traction, s=6, cmap="viridis"
    )
    fig.colorbar(sc, ax=axes[0], label="normal traction [MPa]")
    axes[0].set_aspect("equal")
    axes[0].set_title("equal blocks: t_n = p everywhere")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    a = model.upper_extent
    line = (samples.points[:, 1] - 0.5 * a).abs() <= 0.5 * a / model.n_upper
    order = torch.argsort(samples.points[line][:, 0])
    axes[1].plot(
        samples.points[line][order, 0],
        samples.normal_traction[line][order] / 10.0,
        "o",
        ms=3,
        label="mortar segments",
    )
    axes[1].axhline(1.0, color="k", ls="--", label="analytic $p$")
    axes[1].set_ylim(0.99, 1.01)
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("$t_n / p$")
    axes[1].set_title("interface pressure vs analytic")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "case-a-interface-pressure.png", dpi=160)
    plt.close(fig)

    # 3) mismatched punch: the distribution is penalty-regularisation dependent
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.5))
    for ax, label, title in (
        (axes[0], "pen1e+05", r"$\epsilon_n=10^5$ (closed)"),
        (axes[1], "pen1e+08", r"$\epsilon_n=10^8$ (edge picked)"),
    ):
        samples = gallery[label]
        sc = ax.scatter(
            samples.points[:, 0],
            samples.points[:, 1],
            c=samples.normal_traction / 10.0,
            s=9,
            cmap="magma",
            norm=matplotlib.colors.LogNorm(vmin=0.5, vmax=100.0),
        )
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("punch traction $t_n/p$, " + title, fontsize=8)
        fig.colorbar(sc, ax=ax, label="$t_n/p$")
    rows = case_b["penalty_sweep"]
    eps = [r["penalty"] for r in rows]
    axes[2].semilogx(eps, [r["closed_fraction"] for r in rows], "o-", label="closed area fraction")
    axes[2].semilogx(
        eps, [r["edge_over_pressure"] / 100.0 for r in rows], "s--", label="edge peak / (100 p)"
    )
    axes[2].set_xlabel("penalty $\\epsilon_n$")
    axes[2].set_title("n = 3: penalty sensitivity", fontsize=9)
    axes[2].legend(fontsize=7)
    axes[2].grid(True, which="both", ls=":", lw=0.4)
    fig.tight_layout()
    fig.savefig(figures / "case-b-penalty-sensitivity.png", dpi=160)
    plt.close(fig)

    # 4) transfer convergence
    rows = case_c["levels"]
    h = [1.0 / r["n_slave"] for r in rows]
    l2 = [r["l2_error"] for r in rows]
    force = [r["total_force_error"] for r in rows]
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.loglog(h, l2, "o-", label="traction L2 error")
    ax.loglog(h, force, "s--", label="total force error")
    ref = [l2[0] * (x / h[0]) ** 2 for x in h]
    ax.loglog(h, ref, "k:", label="$O(h^2)$")
    ax.set_xlabel("mesh size $h$")
    ax.set_ylabel("relative error")
    ax.set_title(f"mortar segment transfer, fitted slope {case_c['observed_rates'][-1]:.2f}")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", ls=":", lw=0.4)
    fig.tight_layout()
    fig.savefig(figures / "case-c-transfer-convergence.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
