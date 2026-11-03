"""Convergence evidence for the thermoelastic annulus verification case.

Runs the Q4 thermoelastic path of :mod:`tensorfem.thermoelastic_annulus`
against the closed-form traction-free annulus solution, writes
``benchmarks/thermoelastic_annulus_evidence.json`` and two figures, and prints
the markdown tables reproduced in ``docs/thermoelastic-annulus-validation.md``.
"""
from __future__ import annotations

import json
import math
import pathlib

import torch

torch.set_num_threads(1)

from tensorfem.thermoelastic_annulus import (
    annulus_thermal_state,
    radial_temperature,
    solve_annulus_thermoelasticity,
)

CASE = dict(
    inner_radius=0.5,
    outer_radius=0.8,
    inner_temperature=500.0,
    outer_temperature=100.0,
    young_modulus=210.0e9,
    poisson_ratio=0.3,
    expansion_coefficient=12.0e-6,
)
MESHES = [(8, 4, 1.0), (8, 8, 1.0), (8, 16, 1.0), (8, 32, 1.0), (12, 24, 1.6)]
OUT = pathlib.Path(__file__).with_name("thermoelastic_annulus_evidence.json")
ASSETS = pathlib.Path(__file__).resolve().parent.parent / "docs" / "assets"


def _reference(radius: torch.Tensor) -> object:
    return annulus_thermal_state(
        radius,
        inner_radius=CASE["inner_radius"],
        outer_radius=CASE["outer_radius"],
        inner_temperature=CASE["inner_temperature"],
        outer_temperature=CASE["outer_temperature"],
        young_modulus=CASE["young_modulus"],
        poisson_ratio=CASE["poisson_ratio"],
        expansion_coefficient=CASE["expansion_coefficient"],
    )


def _metrics(n_theta: int, n_radial: int, grading: float) -> dict:
    result = solve_annulus_thermoelasticity(
        CASE["inner_radius"],
        CASE["outer_radius"],
        n_theta,
        n_radial,
        radial_grading=grading,
        **{k: CASE[k] for k in (
            "young_modulus",
            "poisson_ratio",
            "expansion_coefficient",
            "inner_temperature",
            "outer_temperature",
        )},
    )
    reference = _reference(result.sample_radius)
    hoop_peak = float(reference.hoop_stress.abs().max())
    radial_peak = float(reference.radial_stress.abs().max())
    axial_peak = float(reference.axial_stress.abs().max())
    index = int(torch.argmin(result.sample_radius))
    bore_exact = float(_reference(result.sample_radius[index : index + 1]).hoop_stress[0])
    bore_fem = float(result.hoop_stress[index])
    bore_axial_exact = float(_reference(result.sample_radius[index : index + 1]).axial_stress[0])
    bore_axial_fem = float(result.axial_stress[index])
    return {
        "n_theta": n_theta,
        "n_radial": n_radial,
        "radial_grading": grading,
        "elements": int(result.elements.shape[0]),
        "dofs": int(result.displacement.numel()),
        "hoop_peak_reference_pa": hoop_peak,
        "radial_peak_reference_pa": radial_peak,
        "axial_peak_reference_pa": axial_peak,
        "bore_hoop_fem_pa": bore_fem,
        "bore_hoop_reference_pa": bore_exact,
        "bore_hoop_rel_error": abs(bore_fem - bore_exact) / hoop_peak,
        "bore_axial_rel_error": abs(bore_axial_fem - bore_axial_exact) / axial_peak,
        "hoop_max_rel_error": float((result.hoop_stress - reference.hoop_stress).abs().max()) / hoop_peak,
        "radial_max_rel_error": float((result.radial_stress - reference.radial_stress).abs().max()) / radial_peak,
        "axial_max_rel_error": float((result.axial_stress - reference.axial_stress).abs().max()) / axial_peak,
    }


def _table(rows: list[dict]) -> str:
    head = (
        "| 网格 (n_θ × n_r) | 单元 | 自由度 | 内壁环向应力 FE / 解析 (MPa) | 内壁环向误差 | 环向全场最大误差 | 径向全场最大误差 | 轴向全场最大误差 |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    body = ""
    for row in rows:
        body += (
            f"| {row['n_theta']} × {row['n_radial']} (grad {row['radial_grading']}) "
            f"| {row['elements']} | {row['dofs']} "
            f"| {row['bore_hoop_fem_pa']/1e6:.2f} / {row['bore_hoop_reference_pa']/1e6:.2f} "
            f"| {row['bore_hoop_rel_error']*100:.3f}% "
            f"| {row['hoop_max_rel_error']*100:.3f}% "
            f"| {row['radial_max_rel_error']*100:.3f}% "
            f"| {row['axial_max_rel_error']*100:.3f}% |\n"
        )
    return head + body


def _figures(finest: dict) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - figure generation is optional
        return []
    ASSETS.mkdir(parents=True, exist_ok=True)
    produced = []

    result = solve_annulus_thermoelasticity(
        CASE["inner_radius"],
        CASE["outer_radius"],
        finest["n_theta"],
        finest["n_radial"],
        radial_grading=finest["radial_grading"],
        **{k: CASE[k] for k in (
            "young_modulus",
            "poisson_ratio",
            "expansion_coefficient",
            "inner_temperature",
            "outer_temperature",
        )},
    )
    radius = torch.cat(
        [torch.linspace(CASE["inner_radius"], CASE["outer_radius"], 200), result.sample_radius]
    )
    exact = _reference(radius)
    order = torch.argsort(result.sample_radius)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    axes[0].plot(exact.radius[:200], exact.hoop_stress[:200] / 1e6, "k-", label="closed form σ_θ")
    axes[0].plot(
        result.sample_radius[order], result.hoop_stress[order] / 1e6, "o", ms=2.5, label="FE σ_θ (Gauss pts)"
    )
    axes[0].set_xlabel("r [m]")
    axes[0].set_ylabel("σ_θ [MPa]")
    axes[0].set_title("hoop stress")
    axes[0].legend()
    axes[1].plot(exact.radius[:200], exact.radial_stress[:200] / 1e6, "k-", label="closed form σ_r")
    axes[1].plot(
        result.sample_radius[order], result.radial_stress[order] / 1e6, "o", ms=2.5, label="FE σ_r (Gauss pts)"
    )
    axes[1].set_xlabel("r [m]")
    axes[1].set_ylabel("σ_r [MPa]")
    axes[1].set_title("radial stress")
    axes[1].legend()
    fig.suptitle(
        f"Annulus thermal stress, T_i={CASE['inner_temperature']:.0f} K, T_o={CASE['outer_temperature']:.0f} K"
    )
    path = ASSETS / "thermoelastic_annulus_stress.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    produced.append(str(path))
    return produced


def main() -> int:
    rows = [_metrics(*mesh) for mesh in MESHES]
    table = _table(rows)
    print(table, flush=True)
    figures = _figures(rows[-1])
    twisting = {
        "note": "relative errors are normalised by the peak magnitude of the same field",
        "case": CASE,
        "meshes": rows,
        "gates": {"bore_hoop": 0.03, "field_hoop": 0.03, "field_axial": 0.03, "field_radial": 0.03},
        "worst": max(rows, key=lambda r: r["bore_hoop_rel_error"])["bore_hoop_rel_error"],
        "figures": figures,
    }
    OUT.write_text(json.dumps(twisting, indent=2))
    print("worst bore-hoop error over all meshes:", f"{max(r['bore_hoop_rel_error'] for r in rows)*100:.3f}%")
    print("finest mesh errors:",
          {k: f"{rows[-1][k]*100:.3f}%" for k in ("bore_hoop_rel_error", "hoop_max_rel_error", "radial_max_rel_error", "axial_max_rel_error")})
    print("temperature at bore / outer:", [round(float(x), 3) for x in radial_temperature(
        torch.tensor([CASE["inner_radius"], CASE["outer_radius"]], dtype=torch.float64),
        inner_radius=CASE["inner_radius"], outer_radius=CASE["outer_radius"],
        inner_temperature=CASE["inner_temperature"], outer_temperature=CASE["outer_temperature"])])
    print("closed-form peaks [MPa]: hoop|max| %.2f  sigma_r|max| %.2f  sigma_z|max| %.2f" % (
        max(abs(x) for x in rows[-1]["hoop_peak_reference_pa"]) / 1e6 if False else rows[-1]["hoop_peak_reference_pa"] / 1e6,
        rows[-1]["radial_peak_reference_pa"] / 1e6,
        rows[-1]["axial_peak_reference_pa"] / 1e6))
    print("evidence ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())