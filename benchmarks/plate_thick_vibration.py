"""Thick-plate (FSDT) verification study for ``tensorfem.plate_vibration``.

The thin-plate verification in ``benchmarks/plate_vibration.py`` stops at
``a/h = 100`` and can only quote a first-order shear bias as an estimate of how
far the Kirchhoff reference sits from the element's own theory.  This study
closes that gap: the four-node SRI Mindlin element is compared against the
*exact* first-order shear deformation spectrum of
:mod:`tensorfem.plate_thick_reference` over ``a/h = 100 ... 5``, plus a mesh
convergence study at ``a/h = 10``.

Every reference number is a closed form produced by
``tensorfem.plate_thick_reference`` (itself derived and machine-checked by
``scripts/derive_mindlin_sss_determinant.py``); nothing is copied from the
solver output.

Run::

    OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/plate_thick_vibration.py

Artefacts: ``benchmarks/plate_thick_vibration_evidence.json``,
``docs/assets/plate_thick_vibration_*.png`` and the markdown tables in
``outputs/plate_thick_vibration_tables.md``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorfem.plate_thick_reference import (  # noqa: E402
    SHEAR_CORRECTION,
    mindlin_sss_spectrum,
)
from tensorfem.plate_vibration import (  # noqa: E402
    shear_deformation_bias,
    solve_plate_vibration,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
OUTPUTS = ROOT / "outputs"
EVIDENCE = ROOT / "benchmarks" / "plate_thick_vibration_evidence.json"

STEEL = dict(young=200.0e9, poisson=0.3, density=7850.0)
SIDE = 1.0
GATE = 0.02  # acceptable FE-vs-FSDT error on the bending branch
THIN_GATE = 0.001  # Kirchhoff is only a reference for a/h >= 100
ASPECT_RATIOS = (100.0, 50.0, 20.0, 10.0, 5.0)
CONVERGENCE_MESHES = (6, 8, 12, 16, 24)
MODE_FAMILY = ((1, 1), (1, 2), (2, 1), (2, 2))


def rel(value: float, reference: float) -> float:
    return value / reference - 1.0


def solve_first(thickness: float, elements: int, modes: int = 4) -> tuple[float, float]:
    start = time.perf_counter()
    result = solve_plate_vibration(
        elements,
        elements,
        length=SIDE,
        width=SIDE,
        thickness=thickness,
        modes=modes,
        boundary="simple",
        **STEEL,
    )
    elapsed = time.perf_counter() - start
    return float(result.frequencies_hz[0]), elapsed


def _reference(thickness: float, **kwargs):
    return mindlin_sss_spectrum(
        length=SIDE, width=SIDE, thickness=thickness, **STEEL, **kwargs
    )


def study_aspect_ratio(elements: int = 12) -> dict:
    """FE first mode vs exact FSDT, Kirchhoff and the first-order bias."""
    rows = []
    for aspect in ASPECT_RATIOS:
        thickness = SIDE / aspect
        reference = _reference(thickness)
        first_order = shear_deformation_bias(
            length=SIDE, width=SIDE, thickness=thickness,
            young=STEEL["young"], poisson=STEEL["poisson"],
        )
        fe_hz, elapsed = solve_first(thickness, elements)
        rows.append(
            {
                "aspect_ratio": aspect,
                "thickness_m": thickness,
                "fe_hz": fe_hz,
                "fsdt_hz": reference.flexural_hz,
                "kirchhoff_hz": reference.kirchhoff_hz,
                "fe_error": rel(fe_hz, reference.flexural_hz),
                "kirchhoff_error": rel(reference.kirchhoff_hz, reference.flexural_hz),
                "exact_shear_bias": reference.shear_bias,
                "first_order_shear_bias": first_order,
                "first_order_bias_error": rel(first_order, reference.shear_bias),
                "solve_seconds": elapsed,
            }
        )
    worst_fe = max(abs(row["fe_error"]) for row in rows)
    return {
        "elements": elements,
        "gate": GATE,
        "rows": rows,
        "worst_fe_error": worst_fe,
        "passed": bool(worst_fe <= GATE),
        "thin_gate": THIN_GATE,
        "thin_reference_ok": bool(abs(rows[0]["kirchhoff_error"]) <= THIN_GATE),
    }


def study_mesh_convergence(aspect: float = 10.0) -> dict:
    """Grid refinement at ``a/h = 10`` towards the exact FSDT frequency."""
    thickness = SIDE / aspect
    reference = _reference(thickness)
    rows = []
    for elements in CONVERGENCE_MESHES:
        fe_hz, elapsed = solve_first(thickness, elements)
        rows.append(
            {
                "elements": elements,
                "elements_total": elements * elements,
                "dof": 3 * (elements + 1) ** 2,
                "fe_hz": fe_hz,
                "error": rel(fe_hz, reference.flexural_hz),
                "solve_seconds": elapsed,
            }
        )
    sizes = np.array([1.0 / row["elements"] for row in rows])
    errors = np.array([abs(row["error"]) for row in rows])
    slope, intercept = np.polyfit(np.log(sizes), np.log(errors), 1)
    finest = errors[-1]
    return {
        "aspect_ratio": aspect,
        "fsdt_hz": reference.flexural_hz,
        "rows": rows,
        "observed_order": float(slope),
        "finest_error": float(finest),
        "finest_gate": GATE,
        "passed": bool(finest <= GATE),
        "notes": (
            "slope = d log(error)/d log(h) of the first bending mode of a "
            "Q4 SRI Mindlin mesh; quadratic elements are expected to approach "
            "order 2 in the thin limit but the SRI shear term and the "
            "consistent mass matrix usually leave 0 < slope < 2"
        ),
    }


def study_mode_family(aspect: float = 10.0, elements: int = 16) -> dict:
    """First four FE modes vs the sorted closed-form (m, n) family."""
    thickness = SIDE / aspect
    pairs = [
        (_reference(thickness, half_waves_x=m, half_waves_y=n), (m, n))
        for m, n in MODE_FAMILY
    ]
    pairs.sort(key=lambda item: item[0].flexural_angular)
    result = solve_plate_vibration(
        elements,
        elements,
        length=SIDE,
        width=SIDE,
        thickness=thickness,
        modes=len(MODE_FAMILY),
        boundary="simple",
        **STEEL,
    )
    rows = []
    for index, (spectrum, half_waves) in enumerate(pairs):
        fe_hz = float(result.frequencies_hz[index])
        rows.append(
            {
                "index": index + 1,
                "fe_hz": fe_hz,
                "fsdt_hz": spectrum.flexural_hz,
                "error": rel(fe_hz, spectrum.flexural_hz),
                "half_waves": list(half_waves),
            }
        )
    worst = max(abs(row["error"]) for row in rows)
    return {
        "aspect_ratio": aspect,
        "elements": elements,
        "rows": rows,
        "worst_error": worst,
        "gate": GATE,
        "passed": bool(worst <= GATE),
    }


def study_branches(aspect: float = 10.0) -> dict:
    """The three FSDT branches at ``a/h = 10`` (analytic only)."""
    spectrum = _reference(SIDE / aspect)
    return {
        "aspect_ratio": aspect,
        "shear_correction": SHEAR_CORRECTION,
        "flexural_hz": spectrum.flexural_hz,
        "thickness_shear_hz": spectrum.thickness_shear_hz,
        "rotation_hz": spectrum.rotation_hz,
        "kirchhoff_hz": spectrum.kirchhoff_hz,
        "notes": (
            "upper (thickness-shear) and pure-rotation branches scale with h^-3 "
            "and are asymptotic FSDT branches; a C0 displacement mesh needs a "
            "very fine grid to resolve them and they are not engineering panel "
            "frequencies. Exposed for branch ordering, not as FE targets."
        ),
    }


def figure_aspect_ratio(study: dict) -> Path:
    aspects = [row["aspect_ratio"] for row in study["rows"]]
    fe = [row["fe_hz"] for row in study["rows"]]
    fsdt = [row["fsdt_hz"] for row in study["rows"]]
    kirchhoff = [row["kirchhoff_hz"] for row in study["rows"]]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].loglog(aspects, kirchhoff, "k--", label="Kirchhoff (Navier)")
    axes[0].loglog(aspects, fsdt, "C0-", label="exact FSDT (closed form)")
    axes[0].loglog(aspects, fe, "C3o", markersize=5, label="Q4 SRI Mindlin FE (12x12)")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("a/h")
    axes[0].set_ylabel("f1 [Hz]")
    axes[0].set_title("First mode: thin-plate reference leaves the data")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)

    bias_exact = [100.0 * row["exact_shear_bias"] for row in study["rows"]]
    bias_first = [100.0 * row["first_order_shear_bias"] for row in study["rows"]]
    fe_error = [100.0 * row["fe_error"] for row in study["rows"]]
    axes[1].semilogx(aspects, bias_exact, "C0o-", label="exact FSDT vs Kirchhoff")
    axes[1].semilogx(aspects, bias_first, "C1s--", label="first-order bias estimate")
    axes[1].semilogx(aspects, fe_error, "C3^-", label="FE vs exact FSDT")
    axes[1].invert_xaxis()
    axes[1].axhline(0.0, color="0.6", linewidth=0.8)
    axes[1].set_xlabel("a/h")
    axes[1].set_ylabel("relative difference [%]")
    axes[1].set_title("FE tracks FSDT; the first-order bias does not")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)

    path = ASSETS / "plate_thick_vibration_aspect_ratio.png"
    ASSETS.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def figure_convergence(study: dict) -> Path:
    sizes = [1.0 / row["elements"] for row in study["rows"]]
    errors = [abs(row["error"]) for row in study["rows"]]
    figure, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.loglog(sizes, errors, "C0o-", label="|FE - FSDT| / FSDT")
    slope = study["observed_order"]
    x = np.array(sizes)
    ax.loglog(x, errors[0] * (x / x[0]) ** slope, "k--",
              label=f"fitted slope {slope:.2f}")
    ax.set_xlabel("element size h/L")
    ax.set_ylabel("relative error of f1")
    ax.set_title(f"Mesh convergence at a/h = {study['aspect_ratio']:.0f}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    path = ASSETS / "plate_thick_vibration_convergence.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def markdown_tables(payload: dict) -> str:
    aspect = payload["aspect_ratio"]
    convergence = payload["mesh_convergence"]
    family = payload["mode_family"]
    lines = []
    lines.append(f"### a/h sweep at {aspect['elements']}x{aspect['elements']} elements\n")
    lines.append("| a/h | 网格 | f1 FE [Hz] | f1 FSDT 精确 [Hz] | f1 Kirchhoff [Hz] | FE vs FSDT | Kirchhoff vs FSDT | 一阶剪切偏置估计 | 一阶估计误差 | 判定 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in aspect["rows"]:
        verdict = "PASS" if abs(row["fe_error"]) <= aspect["gate"] else "FAIL"
        lines.append(
            f"| {row['aspect_ratio']:.0f} | {aspect['elements']}x{aspect['elements']} "
            f"| {row['fe_hz']:.4f} | {row['fsdt_hz']:.4f} | {row['kirchhoff_hz']:.4f} "
            f"| {100 * row['fe_error']:+.3f}% | {100 * row['kirchhoff_error']:+.3f}% "
            f"| {100 * row['first_order_shear_bias']:+.3f}% "
            f"| {100 * row['first_order_bias_error']:+.1f}% | {verdict} |"
        )
    lines.append("")
    lines.append(f"### 网格收敛（a/h = {convergence['aspect_ratio']:.0f}，FSDT 精确 f1 = {convergence['fsdt_hz']:.4f} Hz）\n")
    lines.append("| 网格 | 单元数 | 自由度 | f1 FE [Hz] | 相对误差 | 求解 [s] |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in convergence["rows"]:
        lines.append(
            f"| {row['elements']}x{row['elements']} | {row['elements_total']} | {row['dof']} "
            f"| {row['fe_hz']:.4f} | {100 * row['error']:+.3f}% | {row['solve_seconds']:.1f} |"
        )
    lines.append(f"\n拟合收敛阶 = {convergence['observed_order']:.2f}；最细网格误差 = "
                 f"{100 * convergence['finest_error']:+.3f}%\n")
    lines.append(f"### 模态族（a/h = {family['aspect_ratio']:.0f}，{family['elements']}x{family['elements']}）\n")
    lines.append("| 阶 | (m,n) | f FE [Hz] | f FSDT 精确 [Hz] | 相对误差 |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in family["rows"]:
        m, n = row["half_waves"]
        lines.append(
            f"| {row['index']} | ({m},{n}) | {row['fe_hz']:.4f} | {row['fsdt_hz']:.4f} "
            f"| {100 * row['error']:+.3f}% |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    payload = {
        "case": "plate_thick_vibration",
        "material": STEEL,
        "side_m": SIDE,
        "shear_correction": SHEAR_CORRECTION,
        "aspect_ratio": study_aspect_ratio(),
        "mesh_convergence": study_mesh_convergence(),
        "mode_family": study_mode_family(),
        "branches": study_branches(),
    }
    payload["figures"] = [
        str(figure_aspect_ratio(payload["aspect_ratio"])),
        str(figure_convergence(payload["mesh_convergence"])),
    ]
    EVIDENCE.write_text(json.dumps(payload, indent=2))
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    table = markdown_tables(payload)
    (OUTPUTS / "plate_thick_vibration_tables.md").write_text(table)
    print(table)
    print(f"worst FE-vs-FSDT error over a/h sweep: {100 * payload['aspect_ratio']['worst_fe_error']:.3f}%")
    print(f"evidence: {EVIDENCE}")


if __name__ == "__main__":
    main()