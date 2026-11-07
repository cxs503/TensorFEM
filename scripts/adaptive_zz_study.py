"""Reproduce the ZZ-indicator / adaptive-refinement study and its figures.

Runs the closed adaptive loop on the Williams re-entrant-corner problem, the uniform
reference ladder, an extended adaptive ladder (for the asymptotic rate) and a quadrature
sensitivity check, then writes

* ``artifacts/adaptive_zz/study.json``  -- every measured number (JSON, no hand copying),
* ``artifacts/adaptive_zz/summary.md``  -- the criteria table,
* ``docs/figures/adaptive_zz_*.png``    -- the report figures.

Usage::

    PYTHONPATH=src python scripts/adaptive_zz_study.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from tensorfem import adaptive_zz as zz

ROOT = Path(__file__).resolve().parents[1]


def von_mises(stress: np.ndarray, poisson: float = zz.POISSON) -> np.ndarray:
    """Plane-strain von Mises equivalent stress (``sigma_zz = nu (sigma_xx + sigma_yy)``)."""
    s = np.asarray(stress, dtype=np.float64)
    sxx, syy, sxy = s[..., 0], s[..., 1], s[..., 2]
    szz = poisson * (sxx + syy)
    squared = 0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * sxy * sxy
    return np.sqrt(np.maximum(squared, 0.0))


def strip_internal(study: dict) -> dict:
    """Drop meshes/arrays so the payload is JSON serialisable."""
    clean = {}
    for key, value in study.items():
        if key.startswith("_"):
            continue
        clean[key] = value
    return clean


def jsonable(payload):
    if isinstance(payload, dict):
        return {key: jsonable(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [jsonable(value) for value in payload]
    if isinstance(payload, (np.floating, np.integer)):
        return payload.item()
    if isinstance(payload, np.ndarray):
        return payload.tolist()
    return payload


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figure_convergence(main: dict, extended: dict, path: Path) -> None:
    uniform = main["uniform"]
    adaptive = main["adaptive"]
    uniform_dofs = [level["dofs"] for level in uniform["levels"]]
    uniform_error = [level["true_energy_norm"] for level in uniform["levels"]]
    adaptive_dofs = [level["dofs"] for level in adaptive["levels"]]
    adaptive_error = [level["true_energy_norm"] for level in adaptive["levels"]]
    ext_dofs = [level["dofs"] for level in extended["levels"]]
    ext_error = [level["true_energy_norm"] for level in extended["levels"]]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    ax = axes[0]
    ax.loglog(uniform_dofs, uniform_error, "o-", color="#1f4e79", label="uniform h-refinement")
    ax.loglog(adaptive_dofs, adaptive_error, "s-", color="#c00000", label="ZZ-adaptive (2 rounds)")
    ax.loglog(ext_dofs, ext_error, "^--", color="#e07941", ms=4, label="ZZ-adaptive (7 rounds)")
    ref = np.array([uniform_dofs[0], uniform_dofs[-1]], dtype=float)
    ax.loglog(
        ref,
        uniform_error[0] * (ref / ref[0]) ** (-0.5 * main["williams_lambda"]),
        ":",
        color="#1f4e79",
        lw=1.0,
        label=rf"theory $N^{{-\lambda/2}}$, $\lambda$={main['williams_lambda']:.4f}",
    )
    ax.loglog(
        ref,
        uniform_error[0] * (ref / ref[0]) ** (-0.5),
        ":",
        color="#7f7f7f",
        lw=1.0,
        label=r"best possible $N^{-1/2}$",
    )
    ax.set_xlabel("unknowns (free DOF)")
    ax.set_ylabel(r"true energy-norm error $\|u-u_h\|_E$")
    ax.set_title("corner singularity: adaptive vs uniform")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    matched = [row for row in main["matched_dof_table"] if "uniform_error" in row]
    labels = [row["adaptive_mesh"] for row in matched]
    x = np.arange(len(matched))
    ax.bar(
        x - 0.2,
        [row["adaptive_dofs"] for row in matched],
        0.4,
        label="adaptive DOF",
        color="#c00000",
    )
    ax.bar(
        x + 0.2,
        [row["uniform_dofs"] for row in matched],
        0.4,
        label="uniform DOF at same accuracy",
        color="#1f4e79",
    )
    for k, row in enumerate(matched):
        ax.annotate(
            f"/{row['uniform_over_adaptive_error_ratio']:.2f}",
            (k, max(row["adaptive_dofs"], row["uniform_dofs"]) * 1.05),
            ha="center",
            fontsize=7,
        )
        ax.annotate(
            f"x{row['uniform_over_adaptive_dof_ratio']:.1f}",
            (k, max(row["adaptive_dofs"], row["uniform_dofs"]) * 1.35),
            ha="center",
            fontsize=7,
            color="#c00000",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("unknowns")
    ax.set_title("work needed for the same accuracy")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _draw_indicator(ax, mesh, indicators, marked=None, title=""):
    polygons = mesh.coordinates()
    values = np.log10(np.maximum(indicators, 1e-14))
    collection = PolyCollection(
        polygons, array=values, cmap="viridis", edgecolors="k", linewidths=0.15
    )
    ax.add_collection(collection)
    if marked is not None and np.any(marked):
        ax.add_collection(
            PolyCollection(polygons[marked], facecolor="none", edgecolors="#ff2020", linewidths=1.2)
        )
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title, fontsize=9)
    return collection


def figure_indicator_maps(main: dict, path: Path) -> None:
    adaptive = main["_adaptive_internal"]
    levels = adaptive["results"]
    meshes = adaptive["meshes"]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    mesh = meshes[0]
    marked = zz.dorfler_mark(
        levels[0].indicators,
        main["adaptive"]["markings"][0]["marked_fraction"] * 0
        + main["parameters"]["marking_fraction"],
    )
    collection = _draw_indicator(
        axes[0],
        mesh,
        levels[0].indicators,
        marked,
        f"start mesh ({mesh.n_elements} elements)\nred = bulk-marked ({int(marked.sum())})",
    )
    fig.colorbar(collection, ax=axes[0], label=r"$\log_{10}\eta_e$")
    mesh2 = meshes[-1]
    collection = _draw_indicator(
        axes[1],
        mesh2,
        levels[-1].indicators,
        None,
        f"after 2 adaptive rounds ({mesh2.n_elements} elements)\n"
        f"hanging nodes: {levels[-1].hanging}",
    )
    fig.colorbar(collection, ax=axes[1], label=r"$\log_{10}\eta_e$")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_effectivity(main: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    ax = axes[0]
    uniform = main["uniform"]["levels"]
    adaptive = main["adaptive"]["levels"]
    ax.semilogx(
        [level["dofs"] for level in uniform],
        [level["effectivity"] for level in uniform],
        "o-",
        color="#1f4e79",
        label="uniform ladder",
    )
    ax.semilogx(
        [level["dofs"] for level in adaptive],
        [level["effectivity"] for level in adaptive],
        "s-",
        color="#c00000",
        label="adaptive levels",
    )
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xlabel("unknowns")
    ax.set_ylabel(r"effectivity index $\theta=\|\eta\|/\|u-u_h\|_E$")
    ax.set_title("ZZ estimator is asymptotically exact")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_ylim(0.9, 1.15)

    ax = axes[1]
    ax.loglog(
        [level["min_element_size"] for level in uniform],
        [level["recovery_nodal_error_smooth"] for level in uniform],
        "o-",
        color="#1f4e79",
        label="uniform ladder",
    )
    ax.loglog(
        [level["min_element_size"] for level in adaptive],
        [level["recovery_nodal_error_smooth"] for level in adaptive],
        "s-",
        color="#c00000",
        label="adaptive levels",
    )
    ax.set_xlabel("smallest element size $h_{min}$")
    ax.set_ylabel("recovered vs exact nodal stress\n(relative, smooth region)")
    ax.set_title("superconvergent recovery converges")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_fields(main: dict, path: Path) -> None:
    adaptive = main["_adaptive_internal"]
    mesh = adaptive["meshes"][-1]
    level = adaptive["results"][-1]

    grid = np.linspace(-1.0, 1.0, 241)
    gx, gy = np.meshgrid(grid, grid)
    mask = (gx < 0.0) & (gy < 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        # the singular point r = 0 is excluded from the plotting grid below
        exact = von_mises(zz.williams_stress(np.column_stack([gx.ravel(), gy.ravel()])))
    exact = np.ma.masked_array(
        exact.reshape(gx.shape), mask=mask | np.isnan(exact.reshape(gx.shape))
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    axes[0].pcolormesh(gx, gy, np.log10(exact), cmap="magma", shading="auto")
    axes[0].set_title("exact Williams field: $\\log_{10}$ von Mises", fontsize=9)
    axes[0].set_aspect("equal")

    element_values = np.log10(np.maximum(von_mises(level.element_stress), 1e-14))
    collection = PolyCollection(
        mesh.coordinates(), array=element_values, cmap="magma", edgecolors="k", linewidths=0.1
    )
    axes[1].add_collection(collection)
    axes[1].autoscale()
    axes[1].set_aspect("equal")
    axes[1].set_title(f"FE field on the adaptive mesh ({mesh.n_elements} elements)", fontsize=9)
    fig.colorbar(collection, ax=list(axes), label="$\\log_{10}$ von Mises")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_profile(main: dict, path: Path) -> None:
    """Equivalent stress along the biseries of the material angle (theta = 45 deg)."""
    adaptive = main["_adaptive_internal"]
    mesh = adaptive["meshes"][-1]
    level = adaptive["results"][-1]
    radius = np.linspace(0.03, 1.30, 400)
    points = np.column_stack([radius / math.sqrt(2.0), radius / math.sqrt(2.0)])
    exact = von_mises(zz.williams_stress(points))

    centres = mesh.centres()
    centre_radius = np.hypot(centres[:, 0], centres[:, 1])
    centre_angle = np.arctan2(centres[:, 1], centres[:, 0])
    on_ray = np.abs(centre_angle - 0.25 * math.pi) < 0.12
    fe_radius = centre_radius[on_ray]
    fe_value = von_mises(level.element_stress[on_ray])

    nodes = mesh.nodes
    node_radius = np.hypot(nodes[:, 0], nodes[:, 1])
    node_angle = np.arctan2(nodes[:, 1], nodes[:, 0])
    keep = (np.abs(node_angle - 0.25 * math.pi) < 0.06) & (node_radius > 1e-3)
    order = np.argsort(node_radius[keep])
    recovered = von_mises(level.nodal_stress[keep])[order]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.semilogy(radius, exact, "-", color="k", lw=1.6, label="exact")
    ax.semilogy(
        node_radius[keep][order],
        recovered,
        "s",
        color="#c00000",
        ms=4,
        label=r"ZZ recovered nodal $\sigma^*$",
    )
    ax.semilogy(fe_radius, fe_value, "^", color="#1f4e79", ms=4, label=r"raw element $\sigma_h$")
    ax.set_xlabel("r along the bisector of the material angle")
    ax.set_ylabel("von Mises equivalent stress")
    ax.set_title("stress recovery vs exact solution (adaptive mesh)", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="artifacts/adaptive_zz")
    parser.add_argument("--figure-dir", default="docs/figures")
    parser.add_argument("--divisions", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--fraction", type=float, default=0.5)
    args = parser.parse_args()

    artifact_dir = ROOT / args.artifact_dir
    figure_dir = ROOT / args.figure_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    main_study = zz.study(
        divisions=args.divisions,
        rounds=args.rounds,
        fraction=args.fraction,
        uniform_divisions=(8, 16, 24, 32, 48, 64),
        max_subcells=8,
    )
    extended = zz.adaptive_study(divisions=12, rounds=7, fraction=args.fraction, max_subcells=6)
    extended_rate = zz.convergence_rate(
        [level["dofs"] for level in extended["levels"]],
        [level["true_energy_norm"] for level in extended["levels"]],
    )

    # quadrature sensitivity of the independently measured true error
    mesh = zz.lshape_mesh(args.divisions)
    prescribed = zz.exact_dirichlet(mesh)
    displacement, _ = zz.solve_displacement(mesh, zz.elasticity_matrix(), dirichlet=prescribed)
    sensitivity = {}
    for subcells in (4, 6, 8, 12, 16):
        sensitivity[str(subcells)] = float(
            np.sqrt(
                zz.true_element_error_sq(
                    mesh, displacement, zz.elasticity_matrix(), max_subcells=subcells
                ).sum()
            )
        )

    criteria = dict(main_study["criteria"])
    criteria.update(
        {
            "extended_adaptive_rate": extended_rate,
            "extended_adaptive_levels": len(extended["levels"]),
            "true_error_quadrature_spread": (max(sensitivity.values()) - min(sensitivity.values()))
            / min(sensitivity.values()),
            "artifacts": {
                "study_json": str(artifact_dir / "study.json"),
                "summary_md": str(artifact_dir / "summary.md"),
                "figures": [
                    str(figure_dir / name)
                    for name in (
                        "adaptive_zz_convergence.png",
                        "adaptive_zz_indicator_maps.png",
                        "adaptive_zz_effectivity.png",
                        "adaptive_zz_fields.png",
                        "adaptive_zz_profile.png",
                    )
                ],
            },
        }
    )

    payload = {
        "schema": "tensorfem.adaptive_zz.study/v1",
        "problem": main_study["problem"],
        "williams_lambda": main_study["williams_lambda"],
        "parameters": main_study["parameters"],
        "adaptive": main_study["adaptive"],
        "uniform": main_study["uniform"],
        "matched_dof_table": main_study["matched_dof_table"],
        "equal_error_table": main_study["equal_error_table"],
        "extended_adaptive": {
            "rounds": 7,
            "divisions": 12,
            "rate": extended_rate,
            "levels": extended["levels"],
            "markings": extended["markings"],
        },
        "true_error_quadrature_sensitivity": sensitivity,
        "criteria": criteria,
        "limitations": [
            "single linear-elastic plane-strain problem with an exact solution; no nonlinear,",
            "plate/shell or contact path is exercised by the loop; the indicator is not",
            "goal-oriented and the refinement is isotropic quadtree subdivision only.",
        ],
    }

    (artifact_dir / "study.json").write_text(json.dumps(jsonable(payload), indent=2))
    (artifact_dir / "summary.md").write_text(summary_markdown(payload))

    figure_convergence(main_study, extended, figure_dir / "adaptive_zz_convergence.png")
    figure_indicator_maps(main_study, figure_dir / "adaptive_zz_indicator_maps.png")
    figure_effectivity(main_study, figure_dir / "adaptive_zz_effectivity.png")
    figure_fields(main_study, figure_dir / "adaptive_zz_fields.png")
    figure_profile(main_study, figure_dir / "adaptive_zz_profile.png")
    return payload


def summary_markdown(payload: dict) -> str:
    criteria = payload["criteria"]
    rows = [
        "| criterion | value |",
        "|---|---|",
        f"| Williams exponent | {payload['williams_lambda']:.13f} |",
        f"| two refinement rounds executed | {criteria['two_rounds_executed']} |",
        f"| adaptive error after round 2 | {criteria['final_adaptive_error']:.5f} |",
        f"| adaptive DOF after round 2 | {criteria['final_adaptive_dofs']} |",
        f"| uniform error at matched DOF | {criteria['final_uniform_error_at_matched_dofs']:.5f} |",
        f"| error ratio (uniform/adaptive, matched DOF) | {criteria['error_ratio_uniform_over_adaptive']:.3f} |",
        f"| uniform DOF for the same accuracy | {criteria['uniform_dofs_for_same_error']:.0f} |",
        f"| DOF saving at equal accuracy | {criteria['dof_saving_at_equal_error']:.2f}x |",
        f"| adaptive rate (2 rounds) | {criteria['adaptive_rate']:.3f} |",
        f"| uniform rate (measured) | {criteria['uniform_rate']:.3f} |",
        f"| uniform rate (theory -lambda/2) | {criteria['theory_uniform_rate']:.3f} |",
        f"| extended adaptive rate (7 rounds) | {criteria['extended_adaptive_rate']:.3f} |",
        f"| effectivity range | {criteria['effectivity_range'][0]:.4f} - {criteria['effectivity_range'][1]:.4f} |",
        f"| true-error quadrature spread | {criteria['true_error_quadrature_spread']:.2e} |",
    ]
    return "# ZZ adaptive refinement study: criteria\n\n" + "\n".join(rows) + "\n"


if __name__ == "__main__":
    result = main()
    print(json.dumps(jsonable(result["criteria"]), indent=2))
