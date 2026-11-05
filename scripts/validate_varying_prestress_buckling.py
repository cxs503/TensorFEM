"""Validate Shell4 eigenbuckling against a non-uniform (linearly varying) pre-stress.

Run with::

    PYTHONPATH=src python scripts/validate_varying_prestress_buckling.py

The script produces:

* ``benchmarks/varying-prestress-buckling.json`` - every number used in the
  report, recomputed here (no hand-copied values);
* ``docs/assets/varying-prestress-buckling-*.png`` - convergence and mode
  figures.

The finite-element path is ``tensorfem.shell4_buckling`` (production flat-facet
Shell4 elastic matrix + bilinear initial-stress matrix).  The reference is
``tensorfem.plate_varying_buckling_reference``, a Ritz double-sine energy
minimisation that shares no code with the finite-element path.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem.plate_varying_buckling_reference import (
    ritz_uniaxial_buckling, ritz_uniaxial_convergence, timoshenko_coefficient_table,
)
from tensorfem.shell4_buckling import (
    linearly_varying_uniaxial_field, simply_supported_plate_buckling_field,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
CASES = (
    ("uniform", 1.0, "uniform compression (psi=1)"),
    ("edge_to_zero", 0.0, "linear compression to zero (psi=0)"),
    ("pure_bending", -1.0, "pure in-plane bending (psi=-1)"),
)
MESHES = (8, 16, 24, 32)
PLATE = dict(young=210e9, poisson=0.3, thickness=0.01, a=1.0, b=1.0)


def _rigidity(case: dict) -> float:
    return case["young"] * case["thickness"] ** 3 / (12 * (1 - case["poisson"] ** 2))


def _coefficient(value: float, case: dict) -> float:
    return value * case["b"] ** 2 / (math.pi ** 2 * _rigidity(case))


def main() -> dict:
    D = _rigidity(PLATE)
    exact_uniform = 4 * math.pi ** 2 * D / PLATE["b"] ** 2
    literature = timoshenko_coefficient_table()

    ritz_rows = {}
    for name, psi, label in CASES:
        conv = ritz_uniaxial_convergence(
            harmonics=(4, 6, 8, 10, 12, 16), psi=psi, **PLATE)
        ref = ritz_uniaxial_buckling(psi=psi, harmonics=16, **PLATE)
        ritz_rows[name] = {
            "psi": psi, "label": label,
            "critical_resultant": ref["critical_resultant"],
            "buckling_coefficient": ref["buckling_coefficient"],
            "dominant_halfwaves": list(ref["dominant_halfwaves"]),
            "convergence": conv,
        }

    fe_rows = {}
    for name, psi, label in CASES:
        ref = ritz_rows[name]["critical_resultant"]
        rows = []
        for n in MESHES:
            value = float(simply_supported_plate_buckling_field(
                a=PLATE["a"], b=PLATE["b"], nx=n, ny=n,
                young=PLATE["young"], poisson=PLATE["poisson"],
                thickness=PLATE["thickness"],
                stress_field=linearly_varying_uniaxial_field(psi)).load_factors[0])
            rows.append({"mesh": f"{n}x{n}", "nodes": (n + 1) ** 2,
                         "critical_resultant": value,
                         "buckling_coefficient": _coefficient(value, PLATE),
                         "relative_error": abs(value - ref) / ref,
                         "signed_error": (value - ref) / ref})
        fe_rows[name] = rows

    # Baseline-drift isolation: the same meshes under uniform compression carry
    # the discretisation error of the flat-facet Shell4 plate, so the
    # non-uniform increment is the change of the FE/Ritz ratio between cases.
    ratio = {}
    for name, _psi, _label in CASES:
        ratio[name] = [
            fe_rows[name][k]["critical_resultant"] / ritz_rows[name]["critical_resultant"]
            for k in range(len(MESHES))]

    # Aspect-ratio sweep for pure bending: the classical constant 23.9 is the
    # long-plate minimum, not the square-plate value.
    aspect = []
    for ar in (0.4, 0.5, 2/3, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0):
        b = 1.0; a = ar * b
        out = ritz_uniaxial_buckling(a=a, b=b, young=PLATE["young"],
                                     poisson=PLATE["poisson"],
                                     thickness=PLATE["thickness"],
                                     psi=-1.0, harmonics=12)
        aspect.append({"a_over_b": ar,
                       "buckling_coefficient": out["critical_resultant"] * b ** 2 / (math.pi ** 2 * D),
                       "dominant_halfwaves": list(out["dominant_halfwaves"])})

    # Independent finite-element confirmation of the long-plate value (a/b = 2).
    ritz_long = ritz_uniaxial_buckling(a=2.0, b=1.0, young=PLATE["young"],
                                       poisson=PLATE["poisson"],
                                       thickness=PLATE["thickness"], psi=-1.0,
                                       harmonics=12)
    long_plate = {}
    for n in (16, 24, 32):
        value = float(simply_supported_plate_buckling_field(
            a=2.0, b=1.0, nx=n, ny=n // 2, young=PLATE["young"],
            poisson=PLATE["poisson"], thickness=PLATE["thickness"],
            stress_field=linearly_varying_uniaxial_field(-1.0)).load_factors[0])
        long_plate[f"{n}x{n//2}"] = {
            "critical_resultant": value,
            "buckling_coefficient": value / (math.pi ** 2 * D),
            "relative_error": (value - ritz_long["critical_resultant"]) / ritz_long["critical_resultant"],
        }
    long_plate_reference = {
        "a_over_b": 2.0,
        "critical_resultant": ritz_long["critical_resultant"],
        "buckling_coefficient": ritz_long["critical_resultant"] / (math.pi ** 2 * D),
    }

    # Mode shapes from the finite element for the pure-bending case.
    mode_data = {}
    for name, psi, _label in CASES:
        result = simply_supported_plate_buckling_field(
            a=1.0, b=1.0, nx=16, ny=16, young=PLATE["young"],
            poisson=PLATE["poisson"], thickness=PLATE["thickness"],
            stress_field=linearly_varying_uniaxial_field(psi))
        w = result.modes[0::3, 0].reshape(17, 17).numpy()
        w = w / np.max(np.abs(w))
        mode_data[name] = {
            "w": w.tolist(),
            "anti_symmetric_in_y": bool(abs(w[:8, :].sum() + w[9:, :].sum())
                                        < 0.5 * np.abs(w).sum()),
        }

    report = {
        "schema": "tensorfem.varying-prestress-buckling/1.0",
        "plate": PLATE,
        "flexural_rigidity": D,
        "exact_uniform_navier": exact_uniform,
        "literal_reference": literature,
        "ritz": ritz_rows,
        "shell4": fe_rows,
        "fe_over_ritz_ratio": ratio,
        "pure_bending_aspect_sweep": aspect,
        "long_plate_pure_bending": long_plate,
        "long_plate_reference": long_plate_reference,
        "modes": mode_data,
        "tolerance": 0.03,
    }

    ritz_uniform = ritz_rows["uniform"]["critical_resultant"]
    literature_check = {
        "uniform_exact_navier_relative_error": abs(ritz_uniform - exact_uniform) / exact_uniform,
        "uniform_literature_k": literature["uniform"],
        "uniform_ritz_k": ritz_rows["uniform"]["buckling_coefficient"],
        "edge_to_zero_literature_k": 7.81,
        "edge_to_zero_ritz_k": ritz_rows["edge_to_zero"]["buckling_coefficient"],
        "edge_to_zero_relative_error_vs_literature": abs(
            ritz_rows["edge_to_zero"]["buckling_coefficient"] - 7.81) / 7.81,
        "pure_bending_literature_k": literature["pure_bending"],
        "pure_bending_ritz_k_square": ritz_rows["pure_bending"]["buckling_coefficient"],
        "pure_bending_relative_error_square_vs_literature": abs(
            ritz_rows["pure_bending"]["buckling_coefficient"] - literature["pure_bending"])
        / literature["pure_bending"],
        "pure_bending_long_plate_ritz_k": long_plate_reference["buckling_coefficient"],
        "pure_bending_long_plate_relative_error_vs_literature": abs(
            long_plate_reference["buckling_coefficient"] - literature["pure_bending"])
        / literature["pure_bending"],
    }
    report["literature_check"] = literature_check

    verdict = {}
    for name, _psi, _label in CASES:
        best = min(fe_rows[name], key=lambda r: r["relative_error"])
        verdict[name] = {
            "worst_mesh_error": max(r["relative_error"] for r in fe_rows[name]),
            "best_mesh": best["mesh"], "best_error": best["relative_error"],
            "within_3pct_all_meshes": max(r["relative_error"] for r in fe_rows[name]) < 0.03,
        }
    report["verdict"] = verdict

    ASSETS.mkdir(parents=True, exist_ok=True)
    (ROOT / "benchmarks").mkdir(parents=True, exist_ok=True)
    out = ROOT / "benchmarks" / "varying-prestress-buckling.json"
    out.write_text(json.dumps(report, indent=1))

    _plot_convergence(report)
    _plot_aspect(report)
    _plot_modes(report)
    print(json.dumps({"json": str(out), "verdict": verdict}, indent=1))
    return report


def _plot_convergence(report: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    styles = {"uniform": ("o", "#1f77b4"), "edge_to_zero": ("s", "#2ca02c"),
              "pure_bending": ("^", "#d62728")}
    for name, marker, colour in [(k, *styles[k]) for k in styles]:
        rows = report["shell4"][name]
        ax.plot([r["nodes"] for r in rows], [r["signed_error"] * 100 for r in rows],
                marker=marker, color=colour, label=report["ritz"][name]["label"])
    ax.axhline(0, color="k", lw=0.8)
    ax.axhspan(-3, 3, color="0.85", zorder=0, label="+/-3% gate")
    ax.set_xscale("log")
    ax.set_xlabel("degrees of freedom (nodes per side)")
    ax.set_ylabel("Shell4 vs Ritz reference  [%]")
    ax.set_title("Non-uniform pre-stress: Shell4 eigenbuckling error")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ASSETS / "varying-prestress-buckling-convergence.png", dpi=140)
    plt.close(fig)


def _plot_aspect(report: dict) -> None:
    rows = report["pure_bending_aspect_sweep"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ar = [r["a_over_b"] for r in rows]
    k = [r["buckling_coefficient"] for r in rows]
    ax.plot(ar, k, "o-", color="#d62728", label="Ritz (converged)")
    ax.axhline(23.9, ls="--", color="0.35", label="frequently quoted k = 23.9")
    ax.axhline(25.528, ls=":", color="#1f77b4", label="square plate k = 25.528")
    ax.set_xlabel("aspect ratio a/b"); ax.set_ylabel("buckling coefficient k")
    ax.set_title("Pure in-plane bending: k(a/b)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "varying-prestress-buckling-aspect.png", dpi=140)
    plt.close(fig)


def _plot_modes(report: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, (name, _psi, _label) in zip(axes, CASES):
        w = np.asarray(report["modes"][name]["w"])
        n = w.shape[0]
        xs = np.linspace(0, 1, n); ys = np.linspace(0, 1, n)
        lim = np.max(np.abs(w))
        cf = ax.contourf(xs, ys, w, levels=21, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.contour(xs, ys, w, levels=21, colors="k", linewidths=0.25)
        ax.set_title(f"{name}\nShell4 16x16 mode 1", fontsize=8)
        ax.set_aspect("equal")
        fig.colorbar(cf, ax=ax, shrink=0.8)
    fig.suptitle("Buckling modes under linearly varying compression (w normalised)", fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "varying-prestress-buckling-modes.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()