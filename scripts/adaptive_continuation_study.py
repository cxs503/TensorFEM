#!/usr/bin/env python3
"""Generate the adaptive-continuation order/cost artifacts and figures.

Runs the 1D h-adaptive ladder (:mod:`tensorfem.adaptive_1d`) and the 2D Q4 analytic-order audit
(:mod:`tensorfem.adaptive_poisson_2d`), writes

    artifacts/adaptive_continuation/study_1d.json
    artifacts/adaptive_continuation/study_2d.json
    artifacts/adaptive_continuation/grading_scan.json
    artifacts/adaptive_continuation/tables.md
    docs/assets/adaptive_continuation/fig1_1d_order.png
    docs/assets/adaptive_continuation/fig2_1d_mesh_effectivity.png
    docs/assets/adaptive_continuation/fig3_2d_order.png
    docs/assets/adaptive_continuation/fig4_cost.png

Every number in the markdown tables is injected from the JSON payloads above (no hand-typed
values), and the figures are produced from the same objects.

Usage
-----
    PYTHONPATH=src python scripts/adaptive_continuation_study.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorfem.adaptive_1d import (  # noqa: E402
    PowerSolution,
    graded_mesh,
    solve_level,
    study as study_1d,
)
from tensorfem.adaptive_poisson_2d import study as study_2d  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "adaptive_continuation"
FIGURES = ROOT / "docs" / "assets" / "adaptive_continuation"

ALPHA = 0.6


def grading_scan(intervals=(16, 32, 64), exponents=(2.0, 4.0, 6.0, 8.0, 10.0, 12.0)) -> dict:
    """Power-law grading scan: how close to the ``N^-1`` ideal can a stable solve get?

    The energy-optimal 1D mesh needs ``h -> 0`` faster than any power at the singular end, which
    drives the tridiagonal condition number past double precision.  The scan records where that
    happens instead of quoting the idealised exponent.
    """
    solution = PowerSolution(ALPHA)
    rows = []
    for exponent in exponents:
        for n in intervals:
            level = solve_level(graded_mesh(int(n), exponent), solution, label=f"g{n}-e{exponent}")
            rows.append(
                {
                    "intervals": int(n),
                    "exponent": float(exponent),
                    "dofs": level.dofs,
                    "min_element": float(level.mesh.lengths().min()),
                    "true_energy_norm": float(level.true_norm),
                    "effectivity": float(level.effectivity),
                    "seconds": float(level.seconds),
                }
            )
    for row in rows:
        row["condition_estimate"] = float(1.0 / row["min_element"])
    # a tridiagonal system with h_max/h_min above 1e12 has no digits left: discard those rows
    usable = [row for row in rows if row["min_element"] > 1e-12]
    best: dict[int, dict] = {}
    for row in usable:
        current = best.get(row["intervals"])
        if current is None or row["true_energy_norm"] < current["true_energy_norm"]:
            best[row["intervals"]] = row
    ordered = [best[int(n)] for n in intervals if int(n) in best]
    rate = (
        float(
            np.polyfit(
                np.log([row["dofs"] for row in ordered]),
                np.log([row["true_energy_norm"] for row in ordered]),
                1,
            )[0]
        )
        if len(ordered) > 1
        else float("nan")
    )
    return {
        "rows": rows,
        "usable_rows": len(usable),
        "rejected_rows": len(rows) - len(usable),
        "min_element_floor": 1e-12,
        "best_per_intervals": ordered,
        "best_rate": rate,
    }


def figure_1d(payload: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for axis, name in zip(axes, ("power", "sinusoidal")):
        block = payload["problems"][name]
        uniform = block["uniform"]["records"]
        adaptive = block["adaptive"]["records"]
        axis.loglog(
            [row["dofs"] for row in uniform],
            [row["true_energy_norm"] for row in uniform],
            "o-",
            label="uniform P1",
            ms=4,
        )
        axis.loglog(
            [row["dofs"] for row in adaptive],
            [row["true_energy_norm"] for row in adaptive],
            "s-",
            label="ZZ-driven adaptive",
            ms=4,
        )
        dofs = np.array([row["dofs"] for row in uniform], dtype=float)
        anchor = uniform[0]["true_energy_norm"] * (dofs / dofs[0]) ** block["uniform"]["theory_rate"]
        axis.loglog(dofs, anchor, "k--", lw=1.0, label=f"theory slope {block['uniform']['theory_rate']:+.3f}")
        dofs = np.array([row["dofs"] for row in adaptive], dtype=float)
        # no closed-form slope is claimed for the adaptive ladder: draw the measured one
        anchor = adaptive[0]["true_energy_norm"] * (dofs / dofs[0]) ** block["adaptive"]["rate"]
        axis.loglog(
            dofs, anchor, "k:", lw=1.0, label=f"measured slope {block['adaptive']['rate']:+.3f}"
        )
        axis.set_xlabel("unknowns $N$")
        axis.set_ylabel(r"exact energy-norm error $\|e\|_E$")
        axis.set_title(f"1D {name}: uniform rate {block['uniform']['rate']:+.3f}, adaptive {block['adaptive']['rate']:+.3f}")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_1d_order.png", dpi=110)
    plt.close(fig)


def figure_1d_mesh(payload_1d: dict) -> None:
    """Effectivity of both 1D problems and the grading of the final adaptive mesh."""
    from tensorfem.adaptive_1d import adaptive_study

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    axes[0].loglog(
        [row["dofs"] for row in payload_1d["problems"]["power"]["uniform"]["records"]],
        [row["effectivity"] for row in payload_1d["problems"]["power"]["uniform"]["records"]],
        "o-",
        ms=4,
        label="uniform (singular $x^{0.6}$)",
    )
    axes[0].loglog(
        [row["dofs"] for row in payload_1d["problems"]["sinusoidal"]["uniform"]["records"]],
        [row["effectivity"] for row in payload_1d["problems"]["sinusoidal"]["uniform"]["records"]],
        "s-",
        ms=4,
        label="uniform (smooth $\\sin\\pi x$)",
    )
    axes[0].axhline(1.0, color="k", ls="--", lw=1.0)
    axes[0].set_xlabel("unknowns $N$")
    axes[0].set_ylabel(r"effectivity $\theta=\eta/\|e\|_E$")
    axes[0].set_title("indicator effectivity: measured, never rescaled")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)

    adaptive = adaptive_study(PowerSolution(ALPHA), intervals=8, rounds=10, fraction=0.5)
    final = adaptive["levels"][-1]
    lengths = final.mesh.lengths()
    midpoints = 0.5 * (final.mesh.nodes[:-1] + final.mesh.nodes[1:])
    axes[1].loglog(midpoints, lengths, "o", ms=3)
    axes[1].set_xlabel("element midpoint $x$")
    axes[1].set_ylabel("element length $h$")
    axes[1].set_title(
        f"final adaptive mesh: {final.dofs} unknowns, $h_{{\\min}}$={lengths.min():.2e}"
    )
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_1d_mesh_effectivity.png", dpi=110)
    plt.close(fig)


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    payload_1d = study_1d(
        alpha=ALPHA,
        smooth_intervals=(8, 16, 32, 64, 128, 256, 512),
        singular_intervals=(8, 16, 32, 64, 128, 256, 512, 1024),
        start_intervals=8,
        rounds=10,
        fraction=0.5,
    )
    payload_2d = study_2d(divisions=(4, 8, 12, 16, 24, 32, 40))
    scan = grading_scan()

    (ARTIFACTS / "study_1d.json").write_text(json.dumps(payload_1d, indent=2))
    (ARTIFACTS / "study_2d.json").write_text(json.dumps(payload_2d, indent=2))
    (ARTIFACTS / "grading_scan.json").write_text(json.dumps(scan, indent=2))

    figure_1d(payload_1d)
    figure_1d_mesh(payload_1d)
    figure_2d(payload_2d)
    figure_cost(payload_1d, payload_2d)
    tables = markdown_tables(payload_1d, payload_2d, scan)
    (ARTIFACTS / "tables.md").write_text(tables)
    print(tables)


def figure_2d(payload: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    levels = payload["levels"]
    dofs = np.array([row["dofs"] for row in levels], dtype=float)
    for key, label, style in (
        ("true_energy_norm", r"$\|e\|_E$", "o-"),
        ("l2_error", r"$\|e\|_{L^2}$", "s-"),
        ("recovered_gradient_l2_error", r"$\|g^*-g\|_{L^2}$", "^-"),
    ):
        values = np.array([row[key] for row in levels])
        axes[0].loglog(dofs, values, style, ms=4, label=label)
    theory = payload["theory_rates"]
    anchor = levels[0]["true_energy_norm"] * (dofs / dofs[0]) ** theory["energy"]
    axes[0].loglog(dofs, anchor, "k--", lw=1.0, label=f"theory {theory['energy']:+.2f}")
    anchor = levels[0]["l2_error"] * (dofs / dofs[0]) ** theory["l2"]
    axes[0].loglog(dofs, anchor, "k:", lw=1.0, label=f"theory {theory['l2']:+.2f}")
    axes[0].set_xlabel("unknowns $N$")
    axes[0].set_ylabel("exact error")
    axes[0].set_title("2D Q4 bubble: orders vs theory")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)

    axes[1].semilogx(dofs, [row["effectivity"] for row in levels], "o-", ms=4)
    axes[1].axhline(1.0, color="k", ls="--", lw=1.0)
    axes[1].set_xlabel("unknowns $N$")
    axes[1].set_ylabel(r"effectivity $\theta=\eta/\|e\|_E$")
    axes[1].set_title("recovered-gradient ZZ indicator on a smooth field")
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_2d_order.png", dpi=110)
    plt.close(fig)


def figure_cost(payload_1d: dict, payload_2d: dict) -> None:
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    for name, style in (("power", "o-"), ("sinusoidal", "s-")):
        records = payload_1d["problems"][name]["uniform"]["records"]
        axis.loglog(
            [row["true_energy_norm"] for row in records],
            [row["seconds"] for row in records],
            style,
            ms=4,
            label=f"1D uniform {name}",
        )
        adaptive = payload_1d["problems"][name]["adaptive"]["records"]
        axis.loglog(
            [row["true_energy_norm"] for row in adaptive],
            [row["seconds"] for row in adaptive],
            style[0] + "--",
            ms=3,
            label=f"1D adaptive {name}",
        )
    records = payload_2d["levels"]
    axis.loglog(
        [row["true_energy_norm"] for row in records],
        [row["seconds"] for row in records],
        "^-",
        ms=4,
        label="2D uniform Q4",
    )
    axis.set_xlabel(r"achieved energy-norm error $\|e\|_E$")
    axis.set_ylabel("wall-clock seconds (assemble + solve + estimate)")
    axis.set_title("cost baseline: what an accuracy level costs")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_cost.png", dpi=110)
    plt.close(fig)


def markdown_tables(payload_1d: dict, payload_2d: dict, scan: dict) -> str:
    lines: list[str] = []
    lines.append("# Adaptive continuation: generated tables\n")
    lines.append("All values are injected from `study_1d.json`, `study_2d.json` and "
                 "`grading_scan.json`; nothing here is typed by hand.\n")

    lines.append("\n## 1D ladders\n")
    for name in ("power", "sinusoidal"):
        block = payload_1d["problems"][name]
        lines.append(f"\n### `{name}`\n")
        lines.append("| ladder | unknowns | error | effectivity | seconds |")
        lines.append("|---|---|---|---|---|")
        for label, records in (
            ("uniform", block["uniform"]["records"]),
            ("adaptive", block["adaptive"]["records"]),
        ):
            for row in records:
                lines.append(
                    f"| {label} | {row['dofs']} | {row['true_energy_norm']:.6e} | "
                    f"{row['effectivity']:.4f} | {row['seconds']:.4f} |"
                )
        lines.append(
            f"\n* measured uniform rate `{block['uniform']['rate']:+.4f}` vs closed-form theory "
            f"`{block['uniform']['theory_rate']:+.4f}` (gap "
            f"`{block['criteria']['uniform_rate_gap_to_theory']:.4f}`); measured adaptive rate "
            f"`{block['adaptive']['rate']:+.4f}` (no closed-form rate is claimed for adaptive"
            " ladders)."
        )
        cost = block["cost"]
        if cost["uniform_dofs_at_equal_error"] is not None:
            statement = (
                f"the uniform ladder interpolates to `{cost['uniform_dofs_at_equal_error']:.1f}` "
                f"unknowns for the same error (saving factor "
                f"`{cost['dof_saving_at_equal_error']:.2f}`)"
            )
        else:
            statement = (
                f"the adaptive error lies below the whole uniform ladder, whose "
                f"{cost['uniform_ladder_max_dofs']} unknowns only reach "
                f"`{cost['uniform_ladder_min_error']:.4e}`, so the saving is at least "
                f"`{cost['dof_saving_lower_bound']:.1f}x`"
            )
        lines.append(
            f"* cost baseline: adaptive final {cost['adaptive_dofs']} unknowns at "
            f"{cost['adaptive_seconds']:.4f} s; {statement}."
        )

    lines.append("\n## 1D adaptive markings (power)\n")
    lines.append("| round | marked elements | marked fraction | energy fraction |")
    lines.append("|---|---|---|---|")
    for row in payload_1d["problems"]["power"]["adaptive"]["markings"]:
        lines.append(
            f"| {row['round']} | {row['marked_elements']} | {row['marked_fraction']:.3f} | "
            f"{row['marked_energy_fraction']:.3f} |"
        )

    lines.append("\n## Grading scan (1D power)\n")
    lines.append("| intervals | exponent | unknowns | min element | error | effectivity |")
    lines.append("|---|---|---|---|---|---|")
    for row in scan["best_per_intervals"]:
        lines.append(
            f"| {row['intervals']} | {row['exponent']:.1f} | {row['dofs']} | "
            f"{row['min_element']:.3e} | {row['true_energy_norm']:.6e} | {row['effectivity']:.4f} |"
        )
    lines.append(
        f"\n* best graded-mesh rate among meshes with `h_min > {scan['min_element_floor']:.0e}` "
        f"(i.e. a solvable condition number): `{scan['best_rate']:+.4f}` (ideal `-1.0`); "
        f"`{scan['rejected_rows']}` row(s) were rejected by that floor."
    )

    lines.append("\n## 2D Q4 bubble\n")
    lines.append("| divisions | unknowns | energy error | L2 error | effectivity | recovered-grad error | seconds |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in payload_2d["levels"]:
        lines.append(
            f"| {row['divisions']} | {row['dofs']} | {row['true_energy_norm']:.6e} | "
            f"{row['l2_error']:.6e} | {row['effectivity']:.5f} | "
            f"{row['recovered_gradient_l2_error']:.6e} | {row['seconds']:.4f} |"
        )
    rates = payload_2d["rates"]
    asymptotic = payload_2d["rates_asymptotic"]
    last_two = payload_2d["rates_last_two"]
    lines.append(
        f"\n* full-ladder rates: energy `{rates['energy']:+.4f}`, L2 `{rates['l2']:+.4f}`, "
        f"recovered gradient `{rates['recovered_gradient']:+.4f}`."
    )
    lines.append(
        f"* asymptotic (last three rungs) rates: energy `{asymptotic['energy']:+.4f}` "
        f"(theory `-0.5`), L2 `{asymptotic['l2']:+.4f}` (theory `-1.0`), recovered gradient "
        f"`{asymptotic['recovered_gradient']:+.4f}`."
    )
    lines.append(
        f"* last-two-rung rates (finest pair only): energy `{last_two['energy']:+.4f}`, "
        f"L2 `{last_two['l2']:+.4f}`, recovered gradient `{last_two['recovered_gradient']:+.4f}`."
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()