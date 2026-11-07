"""Assemble ``result.json`` and the figures for the hull-girder modal benchmark.

Run from the repository root::

    python benchmarks/hull_girder_modal/make_result.py

The script re-computes every number through :mod:`timoshenko_comparison` and
:mod:`wet_dry_ratio` (nothing is copied from an earlier run) and writes

* ``result.json``                     -- machine-readable gates and evidence,
* ``figures/fig1_dry_modes.png``      -- four frequency paths vs mode number,
* ``figures/fig2_grid_convergence.png`` -- finite element error and observed order,
* ``figures/fig3_lewis_sections.png`` -- fitted sections and their exact limits,
* ``figures/fig4_added_mass_table.png`` -- added-mass coefficient against T/B,
* ``figures/fig5_wet_dry_ratio.png``  -- wet/dry reduction and the wet mode shape.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO / "src") not in sys.path:  # pragma: no cover - path bootstrap
    sys.path.insert(0, str(REPO / "src"))
if str(HERE) not in sys.path:  # pragma: no cover - allow sibling imports
    sys.path.insert(0, str(HERE))

import timoshenko_comparison as tc  # noqa: E402
import wet_dry_ratio as wd  # noqa: E402

from tensorfem.hull_girder_modal import (  # noqa: E402
    added_mass_matrix,
    beam_simply_supported_modes,
    lewis_fit,
    wet_simply_supported_modes,
)

GATE_MODE_PCT = 2.0
GATE_ADDED_PCT = 5.0
GATE_RATIO_PCT = 3.0
GATE_ORDER = 2.0


def evaluate_gates(dry: dict, wet: dict) -> dict:
    """Machine-checkable gate block (numbers only, no prose thresholds)."""
    gate_case = dry["girder_g"]
    first = [
        m["rel_error_first_order_pct"]
        for m in gate_case["modes"]
        if m["rel_error_first_order_pct"] is not None
    ]
    fem = [m["rel_error_fem_pct"]["80"] for m in gate_case["modes"]]
    orders = [c["observed_order"] for c in gate_case["grid_convergence"]]
    uniform = wet["uniform"]
    uniform_err = max(row["max_rel_error_pct"] for row in uniform["meshes"])
    added_total = wet["section"]["gate_girder"]["added_mass_per_length_kg_m"]
    # Rigid-translation check of the consistent added-mass matrix: the
    # translational shape functions are a partition of unity, so the quadratic
    # form of the rigid translation vector must equal m33 * L exactly.
    section = wd.build(wd.GATE_GIRDER)
    nodes = 41
    matrix = added_mass_matrix(section, nodes - 1, added_total)
    rigid = torch.zeros(2 * nodes, dtype=torch.float64)
    rigid[0::2] = 1.0
    matrix_total = float(rigid @ matrix @ rigid)
    closed_total = added_total * section.length
    matrix_rel_error = abs(matrix_total - closed_total) / closed_total
    gci_ok = min(orders)
    gates = {
        "gate1_timoshenko_modes": {
            "criterion": "modes 1..4 relative frequency error <= 2 %",
            "case": "girder_g",
            "first_order_max_rel_error_pct": max(first),
            "first_order_per_mode_pct": [round(v, 4) for v in first],
            "fem_finest_max_rel_error_pct": max(fem),
            "fem_finest_per_mode_pct": [round(v, 4) for v in fem],
            "pass": max(first) <= GATE_MODE_PCT and max(fem) <= GATE_MODE_PCT,
        },
        "gate2_added_mass_scaling": {
            "criterion": "finite element added mass vs closed-form kappa scaling <= 5 %",
            "case": "girder_g",
            "uniform_ratio_max_rel_error_pct": uniform_err,
            "added_mass_matrix_total_rel_error": matrix_rel_error,
            "added_mass_total_kg_m": added_total,
            "note": "rigid-translation quadratic form of the added-mass matrix "
            "equals m33 * L; the ratio check compares the wet/dry eigen-"
            "frequencies with 1/sqrt(1+kappa)",
            "pass": uniform_err <= GATE_ADDED_PCT,
        },
        "gate3_wet_dry_ratio": {
            "criterion": "f_wet/f_dry vs 1/sqrt(1+kappa_33) <= 3 %",
            "case": "girder_g",
            "kappa_33": uniform["kappa"],
            "closed_form_ratio": uniform["closed_form_ratio"],
            "fem_ratio_finest": uniform["meshes"][-1]["ratio"],
            "max_rel_error_pct": uniform_err,
            "pass": uniform_err <= GATE_RATIO_PCT,
        },
        "gate4_grid_convergence": {
            "criterion": "GCI observed order >= 2 on N in {20, 40, 80}",
            "case": "girder_g",
            "observed_order_per_mode": [round(v, 4) for v in orders],
            "gci_fine_max_pct": max(c["gci_fine_pct"] for c in gate_case["grid_convergence"]),
            "min_observed_order": gci_ok,
            "pass": gci_ok >= GATE_ORDER,
        },
    }
    gates["all_pass"] = all(value["pass"] for key, value in gates.items() if key.startswith("gate"))
    return gates


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _fig1(dry: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, (name, case) in zip(axes, dry.items()):
        modes = [m["mode"] for m in case["modes"]]
        eb = [m["f_euler_bernoulli_hz"] for m in case["modes"]]
        exact = [m["f_timoshenko_exact_hz"] for m in case["modes"]]
        first = [m["f_timoshenko_first_order_hz"] for m in case["modes"]]
        fem = [m["f_fem_hz"]["80"] for m in case["modes"]]
        ax.plot(modes, eb, "s--", color="0.55", label="Euler-Bernoulli (closed form)")
        ax.plot(
            [m for m, f in zip(modes, first) if f is not None],
            [f for f in first if f is not None],
            "^:",
            color="tab:orange",
            label="Timoshenko 1st order (task formula)",
        )
        ax.plot(modes, exact, "o-", color="tab:blue", label="Hu-Timoshenko exact")
        ax.plot(modes, fem, "x", color="tab:red", markersize=9, label="FEM, N = 80")
        if case["role"].startswith("ship"):
            ax.set_title(f"{name}: ship proportions (L/D = "
                         f"{case['geometry']['slenderness_L_over_D']:.1f})\n"
                         "1st-order formula leaves its validity range")
        else:
            ax.set_title(f"{name}: gate case (L/D = "
                         f"{case['geometry']['slenderness_L_over_D']:.1f})")
        ax.set_xlabel("mode number n")
        ax.set_ylabel("frequency [Hz]")
        ax.set_xticks(modes)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Dry bending frequencies of the hull girder: four independent paths")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig2(dry: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    meshes = tc.MESHES
    for ax, (name, case) in zip(axes, dry.items()):
        for index, mode in enumerate(case["modes"]):
            errs = [mode["rel_error_fem_pct"][str(mesh)] for mesh in meshes]
            ax.loglog(meshes, errs, "o-", label=f"mode {mode['mode']}")
        reference = [case["modes"][0]["rel_error_fem_pct"]["20"] * (20 / m) ** 2 for m in meshes]
        ax.loglog(meshes, reference, "k--", lw=1.0, label=r"$O(N^{-2})$ reference")
        orders = [c["observed_order"] for c in case["grid_convergence"]]
        ax.set_title(f"{name}: observed order {min(orders):.2f}-{max(orders):.2f}")
        ax.set_xlabel("elements N")
        ax.set_ylabel("relative error vs exact [%]")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("Grid convergence of the finite element bending frequencies")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig3(wet: dict, path: Path) -> None:
    ratios = [0.15, 0.3, 0.6, 1.0]
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    for ratio in ratios:
        section = lewis_fit(1.0, ratio)
        theta = [2.0 * math.pi * i / 400 for i in range(401)]
        points = [section.contour(t) for t in theta]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, label=f"T/B = {0.5 * ratio:.2f} (a3 = {section.a3:+.3f})")
        ax.plot(xs, [-y for y in ys], color=ax.lines[-1].get_color(), lw=0.6, alpha=0.5)
    theta = [2.0 * math.pi * i / 200 for i in range(201)]
    ax.plot([math.cos(t) for t in theta], [math.sin(t) for t in theta],
            "k:", lw=1.0, label="unit circle = exact m33 = rho pi b^2")
    ax.axhline(0.0, color="tab:blue", lw=0.8, alpha=0.4)
    ax.set_aspect("equal")
    ax.set_xlabel("y / b")
    ax.set_ylabel("z / b (free surface at z = 0)")
    ax.set_title("Fitted Lewis sections (half-breadth b = 1)\n"
                 "immersed half solid, mirror image faint")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig4(wet: dict, path: Path) -> None:
    table = wet["section"]["coefficient_table"]
    tb = [row["draft_over_breadth"] for row in table]
    kappa = [row["kappa_33_vs_displaced_mass"] for row in table]
    simple = [row["mission_formula_piB_over_4T"] for row in table]
    factor = [row["form_factor_f_equals_c_over_pi"] for row in table]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    ax.loglog(tb, kappa, "o-", label=r"Lewis $\kappa_{33}=m_{33}/(\rho_w B T)$")
    ax.loglog(tb, simple, "s--", label=r"task formula $\pi B/(4T)$")
    ax.set_xlabel("T/B")
    ax.set_ylabel(r"$\kappa_{33}$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title("Added-mass ratio against displaced water mass")
    ax = axes[1]
    ax.semilogx(tb, factor, "o-")
    ax.axhline(1.0, color="k", ls=":", lw=1.0)
    ax.set_xlabel("T/B")
    ax.set_ylabel(r"$f = c/\pi = \kappa_{33} / (\pi B/4T)$")
    ax.set_title("Shape correction factor of the task formula")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig5(wet: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    kappas = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
    ax.plot(kappas, [1.0 / math.sqrt(1.0 + k) for k in kappas], "k-", label=r"$1/\sqrt{1+\kappa}$")
    uniform = wet["uniform"]
    ax.plot([uniform["kappa"]], [uniform["closed_form_ratio"]], "o", color="tab:red",
            markersize=9, label="uniform girder (closed form)")
    ax.plot([uniform["kappa"]] * len(uniform["meshes"][-1]["ratio"]),
            uniform["meshes"][-1]["ratio"], "x", color="tab:blue", markersize=9,
            label="uniform girder (FEM, N = 80)")
    tapered = wet["tapered"]
    ax.plot([tapered["kappa_mean"]] * len(tapered["meshes"][-1]["ratio"]),
            tapered["meshes"][-1]["ratio"], "d", color="tab:green", markersize=7,
            label="tapered added mass (FEM, N = 80)")
    ax.set_xlabel(r"$\kappa = m_{33}/m_{total}$")
    ax.set_ylabel(r"$f_{wet}/f_{dry}$")
    ax.set_title("Wet/dry reduction: scalar law and its finite element check")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # wet mode-shape field of the gate girder (transverse displacement along x)
    section = wd.build(wd.GATE_GIRDER)
    added = wet["section"]["gate_girder"]["added_mass_per_length_kg_m"]
    ballast = wet["section"]["gate_girder"]["ballast_mass_per_length_kg_m"]
    dry = beam_simply_supported_modes(section, 80, (ballast,), 2)
    wet_result = wet_simply_supported_modes(section, 80, added, ballast, 2)
    x = torch.linspace(0.0, section.length, 2 * 81, dtype=torch.float64)
    w_dry, w_wet = dry.modes[:, 0], wet_result.modes[:, 0]
    scale = float(w_wet.abs().max())
    ax = axes[1]
    for shape, label, colour in ((w_dry, "dry mode 1", "tab:gray"),
                                 (w_wet / scale * float(w_dry.abs().max()),
                                  "wet mode 1 (scaled)", "tab:blue")):
        values = shape.detach().numpy()
        ax.plot(x.numpy()[::2], values[::2], color=colour, label=label)
        ax.fill_between(x.numpy()[::2], 0.0, values[::2], color=colour, alpha=0.15)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("transverse displacement (arb. units)")
    ax.set_title("Mode 1 shape: water does not change it, only its frequency")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:  # pragma: no cover - reporting helper
    figures = HERE / "figures"
    figures.mkdir(exist_ok=True)
    dry = tc.compute()
    wet = wd.compute()
    gates = evaluate_gates(dry, wet)
    _fig1(dry, figures / "fig1_dry_modes.png")
    _fig2(dry, figures / "fig2_grid_convergence.png")
    _fig3(wet, figures / "fig3_lewis_sections.png")
    _fig4(wet, figures / "fig4_added_mass_table.png")
    _fig5(wet, figures / "fig5_wet_dry_ratio.png")

    tapered_dev = [
        max(row["rel_error_vs_scalar_pct"]) for row in wet["tapered"]["meshes"]
    ]
    payload = {
        "benchmark": "hull_girder_modal",
        "mission_id": 5425,
        "repository": "TensorFEM",
        "module": "src/tensorfem/hull_girder_modal.py",
        "physics": "hull girder vertical bending: Timoshenko correction, Lewis "
        "section added mass, wet/dry frequency reduction, grid convergence",
        "conventions": {
            "beam": "uniform box girder, pinned-pinned (two nodes at the supports)",
            "added_mass": "2D strip theory, zero speed, infinite depth, free "
            "surface as plane of symmetry (rigid lid); immersed section doubled",
            "wet_mass": "a floating body displaces its own weight, so the "
            "vibrating mass per unit length is rho_w * B * T",
        },
        "gates": gates,
        "dry_modes": dry,
        "wet_modes": wet,
        "applicability_limits": {
            "first_order_timoshenko": {
                "validity_index_per_mode_hull_s": [
                    round(m["validity_index"], 4) for m in dry["hull_s"]["modes"]
                ],
                "first_order_error_pct_hull_s": [
                    None
                    if m["rel_error_first_order_pct"] is None
                    else round(m["rel_error_first_order_pct"], 3)
                    for m in dry["hull_s"]["modes"]
                ],
                "statement": "the square-root correction needs "
                "(n pi/L)^2 EI/(kappa G A) < 1 and stays within 2 % only while "
                "that index is below ~0.1; a ship-proportioned girder exceeds "
                "both bounds at higher modes, where the exact Hu-Timoshenko "
                "dispersion relation must be used instead",
            },
            "scalar_wet_dry_reduction": {
                "tapered_max_deviation_pct": max(tapered_dev),
                "tapered_deviation_per_mode_pct": [
                    round(v, 3) for v in wet["tapered"]["meshes"][-1]["rel_error_vs_scalar_pct"]
                ],
                "statement": "1/sqrt(1+kappa) is exact for a uniform added mass "
                "and degrades with hull-like taper because the mode shape "
                "weights the ends differently from a length average",
            },
            "lewis_form": {
                "area_ratio_vs_doubled_box": wet["section"]["gate_girder"][
                    "lewis_area_ratio_vs_doubled_box"
                ],
                "statement": "the Lewis family has smooth flanks; the boxiest "
                "admissible member reaches ~90 % of the sharp box area, so the "
                "reported added mass is a lower bound for a sharp-cornered box",
            },
        },
        "figures": [
            {"file": "figures/fig1_dry_modes.png",
             "caption": "Dry bending frequencies from four independent paths"},
            {"file": "figures/fig2_grid_convergence.png",
             "caption": "Finite element grid convergence and observed order"},
            {"file": "figures/fig3_lewis_sections.png",
             "caption": "Fitted Lewis sections and the exact circle limit"},
            {"file": "figures/fig4_added_mass_table.png",
             "caption": "Added-mass ratio and the task formula's shape factor"},
            {"file": "figures/fig5_wet_dry_ratio.png",
             "caption": "Wet/dry reduction and the unchanged mode shape"},
        ],
        "environment": {"torch": torch.__version__, "python": sys.version.split()[0]},
    }
    out = HERE / "result.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(gates, indent=2))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())