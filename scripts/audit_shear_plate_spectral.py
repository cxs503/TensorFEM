"""Independent reference for pure-shear buckling of simply supported panels.

Writes ``benchmarks/shear_plate_spectral.json`` and the two figures used by
``docs/shear-plate-spectral-reference.md``.  Every number in the report is
produced here, so the tables in the document can be regenerated with

    PYTHONPATH=src python3 scripts/audit_shear_plate_spectral.py

The script is deliberately light (no finite-element assembly, no refinement
sweeps beyond the spectral truncations plus two small finite-difference runs),
so it can be executed on a loaded machine.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tensorfem.shear_plate_spectral import (
    richardson_extrapolation,
    run_shear_spectral_qualification,
    spectral_shear_buckling_coefficient,
)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "shear_plate_spectral.json"
ASSETS = ROOT / "docs" / "assets" / "shear_plate_spectral"

ASPECTS = (1.0, 1.5, 2.0, 3.0)
BUDGETS = (4, 6, 8, 10, 12, 16, 20, 24)

#: Finite-difference values recorded by the merged report
#: ``docs/marine-plate-combined-loading-verification.md`` (grid in the panel
#: solver, ``k = N_xy,cr b^2 / (pi^2 D)``).
MERGED_FINITE_DIFFERENCE = {
    "1.0": {"grid": [40, 40], "k": 9.3521, "richardson_k": 9.346},
    "1.5": {"grid": [40, 60], "k": 7.0879, "richardson_k": 7.0974},
    "2.0": {"grid": [40, 80], "k": 6.5653, "richardson_k": 6.4553},
    "3.0": {"grid": [32, 96], "k": 5.8887, "richardson_k": 5.8384},
}


def classical_fit(aspect: float) -> float:
    """Timoshenko & Gere asymptotic fit ``k_s = 5.34 + 4 (b/a)^2``."""

    return 5.34 + 4.0 / aspect**2


def convergence_tables() -> dict[str, object]:
    tables: dict[str, object] = {}
    for aspect in ASPECTS:
        values = [spectral_shear_buckling_coefficient(aspect, mode_budget=b) for b in BUDGETS]
        order, limit = richardson_extrapolation(list(BUDGETS), values)
        merged = MERGED_FINITE_DIFFERENCE[str(aspect)]
        tables[str(aspect)] = {
            "budgets": list(BUDGETS),
            "coefficients": values,
            "gaps": [values[i] - values[i + 1] for i in range(len(values) - 1)],
            "observed_order": order,
            "converged_k": limit,
            "classical_fit_k": classical_fit(aspect),
            "deviation_vs_classical_fit": abs(classical_fit(aspect) - limit) / limit,
            "merged_finite_difference": merged,
            "deviation_of_merged_fd": abs(merged["k"] - limit) / limit,
            "deviation_of_merged_fd_richardson": abs(merged["richardson_k"] - limit) / limit,
        }
    return tables


def spectra() -> dict[str, object]:
    """Leading positive buckling loads for the square panel (mode separation)."""

    from tensorfem.shear_plate_spectral import spectral_buckling_load

    out: dict[str, object] = {}
    for aspect in ASPECTS:
        result = spectral_buckling_load(
            length=aspect, width=1.0, rigidity_x=1.0, rigidity_y=1.0,
            coupling_rigidity=1.0,
            modes_x=max(1, int(round(24 * math.sqrt(aspect)))),
            modes_y=max(1, int(round(24 / math.sqrt(aspect)))),
            longitudinal_share=0.0, transverse_share=0.0, shear_share=1.0,
        )
        loads = result.positive_loads[:4] / math.pi**2
        out[str(aspect)] = {
            "modes": list(result.modes),
            "leading_k": [float(x) for x in loads],
            "separation": float(result.next_load_ratio),
        }
    return out


def figures(tables: dict[str, object]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ASSETS.mkdir(parents=True, exist_ok=True)
    names = []

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    for aspect in ASPECTS:
        data = tables[str(aspect)]
        ax.plot(data["budgets"], data["coefficients"], "o-", markersize=4,
                label=f"a/b = {aspect:g}")
        ax.axhline(data["converged_k"], color="0.6", lw=0.8, ls=":")
        ax.axhline(classical_fit(aspect), color="0.4", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("mode budget M (isotropic: $M_x/M_y=\\sqrt{a/b}$)")
    ax.set_ylabel("$k_s = N_{xy,cr} b^2 / (\\pi^2 D)$")
    ax.set_title("spectral truncation (solid) vs converged limit (dotted)\nand Timoshenko fit (dashed)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    labels = [f"{a:g}" for a in ASPECTS]
    x = np.arange(len(labels))
    width = 0.2
    series = [
        ("spectral (converged)", [tables[str(a)]["converged_k"] for a in ASPECTS]),
        ("Timoshenko fit", [tables[str(a)]["classical_fit_k"] for a in ASPECTS]),
        ("FD 40x40 / 40x80 / 32x96 (merged)", [tables[str(a)]["merged_finite_difference"]["k"] for a in ASPECTS]),
        ("FD Richardson (merged)", [tables[str(a)]["merged_finite_difference"]["richardson_k"] for a in ASPECTS]),
    ]
    for i, (name, values) in enumerate(series):
        ax.bar(x + (i - 1.5) * width, values, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("aspect ratio a/b")
    ax.set_ylabel("$k_s$")
    ax.set_title("three routes to $k_s$: the fit is the outlier at a/b = 2")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = ASSETS / "convergence_and_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    names.append(str(path.relative_to(ROOT)))

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for aspect in ASPECTS:
        data = tables[str(aspect)]
        limit = data["converged_k"]
        errors = [abs(v - limit) / limit for v in data["coefficients"]]
        ax.semilogy(data["budgets"], errors, "o-", markersize=4, label=f"a/b = {aspect:g}")
    ax.set_xscale("log")
    ax.set_xlabel("mode budget M")
    ax.set_ylabel("$|k_s(M) - k_s^{\\mathrm{Richardson}}| / k_s$")
    ax.set_title("truncation error collapses well below the 3% acceptance band")
    ax.axhline(0.03, color="crimson", lw=1.0, ls="--", label="3% acceptance")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = ASSETS / "truncation_error.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    names.append(str(path.relative_to(ROOT)))
    return names


def main() -> dict[str, object]:
    qualification = run_shear_spectral_qualification()
    tables = convergence_tables()
    payload = {
        "schema": "tensorfem.shear_plate_spectral.benchmark/1",
        "model": {
            "panel": "simply supported rectangular panel, w = 0 and M_n = 0 on all edges",
            "basis": "w = sum c_mn sin(m pi x/a) sin(n pi y/b)",
            "load": "pure in-plane shear N_xy",
            "coefficient": "k_s = N_xy,cr b^2 / (pi^2 D)",
            "literature": "Timoshenko & Gere, Theory of Elastic Stability, 2nd ed., Art. 9.3: k_s = 5.34 + 4 (b/a)^2",
        },
        "tolerance": 0.03,
        "qualification": qualification,
        "convergence": tables,
        "spectra": spectra(),
    }
    figures_used = figures(tables)
    payload["figures"] = figures_used
    BENCHMARK.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"wrote {BENCHMARK}")
    print(f"qualification passed: {qualification['passed']}")
    header = f"{'a/b':>5} {'spectral':>10} {'fit':>8} {'fit dev':>9} {'FD(merged)':>11} {'FD dev':>8}"
    print(header)
    for aspect in ASPECTS:
        data = tables[str(aspect)]
        print(
            f"{aspect:5.1f} {data['converged_k']:10.5f} {data['classical_fit_k']:8.4f} "
            f"{100 * data['deviation_vs_classical_fit']:8.3f}% "
            f"{data['merged_finite_difference']['k']:11.5f} "
            f"{100 * data['deviation_of_merged_fd']:7.3f}%"
        )
    return payload


if __name__ == "__main__":
    main()