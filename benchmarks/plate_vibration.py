"""Verification study for ``tensorfem.plate_vibration``.

Produces machine-readable evidence (``benchmarks/plate_vibration_evidence.json``)
and the three report figures under ``docs/assets/``.  Every reference number is
a closed form or a tabulated literature value; nothing is copied from the
solver output.

Run:
    OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/plate_vibration.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensorfem.plate_vibration import (  # noqa: E402
    BOUNDARY_CLAMPED,
    clamped_square_angular_frequency,
    kirchhoff_angular_frequency,
    mode_convergence,
    observed_convergence_order,
    shear_deformation_bias,
    solve_plate_vibration,
    winkler_angular_frequency,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
EVIDENCE = ROOT / "benchmarks" / "plate_vibration_evidence.json"

STEEL = dict(young=200.0e9, poisson=0.3, density=7850.0, thickness=0.01)
GATE = 0.03


def hz(omega: float) -> float:
    return omega / (2.0 * math.pi)


def rel(value: float, reference: float) -> float:
    return abs(value - reference) / abs(reference)


def square_reference(modes: int = 4) -> list[float]:
    values = sorted(
        hz(kirchhoff_angular_frequency(length=1.0, width=1.0, half_waves_x=m,
                                      half_waves_y=n, **STEEL))
        for m in (1, 2) for n in (1, 2)
    )
    return values[:modes]


def study_simply_supported() -> dict:
    reference = square_reference()
    rows = []
    for mesh in (6, 8, 10, 12, 16):
        result = solve_plate_vibration(mesh, mesh, modes=4, **STEEL)
        computed = result.frequencies_hz.tolist()
        rows.append({
            "elements": mesh,
            "frequencies_hz": [float(v) for v in computed],
            "relative_errors": [rel(computed[i], reference[i]) for i in range(4)],
        })
    rows_convergence = mode_convergence(reference[0], meshes=(6, 8, 10, 12), **STEEL)
    return {
        "case": "simply supported square panel, a/h=100, hard simple support",
        "parameters": {"a_m": 1.0, "b_m": 1.0, **STEEL},
        "reference_definition": "Navier double-sine: omega_mn=pi^2*sqrt(D/(rho h))*((m/a)^2+(n/b)^2)",
        "reference_hz": reference,
        "modes": [[1, 1], [1, 2], [2, 1], [2, 2]],
        "levels": rows,
        "observed_order_first_mode": observed_convergence_order(rows_convergence, reference[0]),
        "shear_bias_first_mode": shear_deformation_bias(
            length=1.0, width=1.0, young=STEEL["young"], poisson=STEEL["poisson"],
            thickness=STEEL["thickness"],
        ),
    }


def study_clamped() -> dict:
    reference = hz(clamped_square_angular_frequency(length=1.0, **STEEL))
    rows = []
    for mesh in (6, 8, 10, 12, 16):
        result = solve_plate_vibration(mesh, mesh, boundary=BOUNDARY_CLAMPED, modes=1, **STEEL)
        computed = float(result.frequencies_hz[0])
        rows.append({"elements": mesh, "frequency_hz": computed,
                     "relative_errors": [rel(computed, reference)]})
    return {
        "case": "fully clamped square panel, a/h=100",
        "reference_definition": "Leissa NASA SP-160 (1969) clamped square coefficient lambda_1=35.9853",
        "reference_hz": reference,
        "levels": rows,
    }


def study_winkler() -> dict:
    rows = []
    for winkler_k in (0.0, 5.0e8, 2.0e9, 8.0e9):
        result = solve_plate_vibration(8, 8, winkler_k=winkler_k, modes=1, **STEEL)
        exact = hz(winkler_angular_frequency(length=1.0, width=1.0,
                                            winkler_k=winkler_k, **STEEL))
        computed = float(result.frequencies_hz[0])
        rows.append({"winkler_k_n_per_m3": winkler_k, "frequency_hz": computed,
                     "reference_hz": exact, "relative_errors": [rel(computed, exact)]})
    return {
        "case": "simply supported square panel on a uniform Winkler bed, 8x8 mesh",
        "reference_definition": "exact sine eigenvalue: omega^2=(D*R^2+k_w)/(rho h)",
        "levels": rows,
    }


def study_rectangular() -> dict:
    values = sorted(
        (hz(kirchhoff_angular_frequency(length=2.0, width=1.0, half_waves_x=m,
                                       half_waves_y=n, **STEEL)), (m, n))
        for m in (1, 2, 3) for n in (1, 2)
    )[:4]
    result = solve_plate_vibration(12, 12, length=2.0, width=1.0, modes=4, **STEEL)
    computed = result.frequencies_hz.tolist()
    return {
        "case": "simply supported rectangular panel a/b=2, 12x12 mesh",
        "reference_hz": [v[0] for v in values],
        "half_wave_labels": [list(v[1]) for v in values],
        "computed_hz": [float(v) for v in computed],
        "relative_errors": [rel(computed[i], values[i][0]) for i in range(4)],
    }


def figure_convergence(simple: dict, clamped: dict) -> Path:
    path = ASSETS / "plate_vibration_convergence.png"
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    meshes = [row["elements"] for row in simple["levels"]]
    labels = ["mode (1,1)", "mode (1,2)/(2,1)", "mode (2,2)"]
    indices = [0, 1, 3]
    for label, index in zip(labels, indices):
        ax.loglog(meshes, [max(row["relative_errors"][index], 1e-6) for row in simple["levels"]],
                  "o-", label=f"simple, {label}")
    ax.loglog([row["elements"] for row in clamped["levels"]],
              [row["relative_errors"][0] for row in clamped["levels"]],
              "s--", label="clamped fundamental")
    ax.axhline(GATE, color="crimson", linestyle=":", label="3% acceptance gate")
    ax.set_xlabel("elements per side")
    ax.set_ylabel("relative frequency error")
    ax.set_title("Plate free vibration: error vs mesh (Mindlin Q4 SRI)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _mode_panel(ax, result, index: int, title: str) -> None:
    values = result.modes[0::3, index].detach().numpy()
    nodes = result.nodes.numpy()
    quads = result.elements.numpy()
    triangles = __import__("numpy").concatenate(
        (quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]), axis=0
    )
    contour = ax.tricontourf(nodes[:, 0], nodes[:, 1], triangles, values, levels=18,
                             cmap="RdBu_r")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(contour, ax=ax, shrink=0.85, label="w [normalised]")


def figure_mode_shapes() -> Path:
    path = ASSETS / "plate_vibration_mode_shapes.png"
    simple = solve_plate_vibration(16, 16, modes=1, **STEEL)
    clamped = solve_plate_vibration(16, 16, boundary=BOUNDARY_CLAMPED, modes=1, **STEEL)
    rectangular = solve_plate_vibration(16, 16, length=2.0, width=1.0, modes=2, **STEEL)
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.6))
    _mode_panel(axes[0], simple, 0, f"simple square, f={float(simple.frequencies_hz[0]):.1f} Hz")
    _mode_panel(axes[1], clamped, 0, f"clamped square, f={float(clamped.frequencies_hz[0]):.1f} Hz")
    _mode_panel(axes[2], rectangular, 0, f"rect a/b=2 (1,1), f={float(rectangular.frequencies_hz[0]):.1f} Hz")
    _mode_panel(axes[3], rectangular, 1, f"rect a/b=2 (2,1), f={float(rectangular.frequencies_hz[1]):.1f} Hz")
    fig.suptitle("Transverse displacement of the lowest modes (field cloud, normalised amplitude)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_winkler(winkler: dict) -> Path:
    path = ASSETS / "plate_vibration_winkler.png"
    ks = [row["winkler_k_n_per_m3"] for row in winkler["levels"]]
    fe = [row["frequency_hz"] for row in winkler["levels"]]
    exact = [row["reference_hz"] for row in winkler["levels"]]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    ax.plot([k / 1e9 for k in ks], exact, "k-", linewidth=1.2, label="exact sine eigenvalue")
    ax.plot([k / 1e9 for k in ks], fe, "o", markersize=6, label="FE, 8x8 Q4 SRI")
    ax.set_xlabel("Winkler modulus $k_w$ [GN/m$^3$]")
    ax.set_ylabel("fundamental frequency [Hz]")
    ax.set_title("Foundation stiffening of a panel")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax2.semilogy([k / 1e9 for k in ks], [max(row["relative_errors"][0], 1e-8) for row in winkler["levels"]],
                 "o-", label="relative error")
    ax2.axhline(GATE, color="crimson", linestyle=":", label="3% gate")
    ax2.set_xlabel("Winkler modulus $k_w$ [GN/m$^3$]")
    ax2.set_ylabel("relative error")
    ax2.set_title("Exactness of the foundation term")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def markdown_table(payload: dict) -> str:
    lines = ["| case | quantity | computed | reference | error | gate |", "|---|---|---:|---:|---:|---|"]
    simple = payload["simply_supported"]
    for label, index in (("(1,1)", 0), ("(1,2)/(2,1)", 1), ("(2,2)", 3)):
        row = simple["levels"][-1]
        lines.append(
            f"| SS square 16x16 | f{label} [Hz] | {row['frequencies_hz'][index]:.4f} | "
            f"{simple['reference_hz'][index]:.4f} | {row['relative_errors'][index]*100:.3f}% | "
            f"{'PASS' if row['relative_errors'][index] < GATE else 'FAIL'} |"
        )
    clamped = payload["clamped"]
    row = clamped["levels"][-1]
    lines.append(
        f"| clamped square 16x16 | f1 [Hz] | {row['frequency_hz']:.4f} | "
        f"{clamped['reference_hz']:.4f} | {row['relative_errors'][0]*100:.3f}% | "
        f"{'PASS' if row['relative_errors'][0] < GATE else 'FAIL'} |"
    )
    for row in payload["winkler"]["levels"]:
        lines.append(
            f"| SS + Winkler k={row['winkler_k_n_per_m3']:.2e} | f1 [Hz] | {row['frequency_hz']:.4f} | "
            f"{row['reference_hz']:.4f} | {row['relative_errors'][0]*100:.4f}% | "
            f"{'PASS' if row['relative_errors'][0] < GATE else 'FAIL'} |"
        )
    return "\n".join(lines)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    payload = {
        "module": "tensorfem.plate_vibration",
        "gate": GATE,
        "simply_supported": study_simply_supported(),
        "clamped": study_clamped(),
        "winkler": study_winkler(),
        "rectangular": study_rectangular(),
    }
    figures = [
        figure_convergence(payload["simply_supported"], payload["clamped"]),
        figure_mode_shapes(),
        figure_winkler(payload["winkler"]),
    ]
    payload["figures"] = [str(path.relative_to(ROOT)) for path in figures]
    EVIDENCE.write_text(json.dumps(payload, indent=2))
    print(markdown_table(payload))
    print()
    print("observed order (first mode):",
          round(payload["simply_supported"]["observed_order_first_mode"], 3))
    print("shear bias (first mode):",
          round(payload["simply_supported"]["shear_bias_first_mode"], 6))
    print("rectangular:", [(lbl, round(f, 3), f"{e*100:.2f}%")
                           for lbl, f, e in zip(payload["rectangular"]["half_wave_labels"],
                                                payload["rectangular"]["computed_hz"],
                                                payload["rectangular"]["relative_errors"])])
    print("evidence:", EVIDENCE.relative_to(ROOT))
    print("figures:", payload["figures"])


if __name__ == "__main__":
    main()