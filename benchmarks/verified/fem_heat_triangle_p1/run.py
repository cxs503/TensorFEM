#!/usr/bin/env python3
"""CASE-FEM-HEAT-L2 (round 2): P1 linear-triangle steady conduction, MMS ladder.

What this script certifies
--------------------------
The anchor ``CASE-FEM-HEAT-L2`` already ships a Q4/bilinear study
(``benchmarks/verified/fem_heat_conduction_l2``) that reaches the L2 rate 2 on
*structured rectangles*.  Its own scope note stops there, so the anchor had no
evidence that the rate is a property of the **discretisation** rather than of
that one mesh family.  This runner exercises the new P1 triangle kernel
(:mod:`tensorfem.heat_l2_triangle`) and records

* the h-refinement ladders of seven manufactured fields on three mesh families
  (two diagonal splits + a distorted triangulation),
* the parameter study of the distortion amplitude,
* the post-processing-rule check (nested 4-way subdivision of the L2 integral),
* an independent cross-validation against scikit-fem ``ElementTriP1``.

Numbers are never typed in: the exact fields are code, the target order is
parsed from the in-repo a-priori table, and every gate is evaluated from the
run.  Outputs are written next to this file (``result.json``,
``gates_verified.json``) plus figures under ``docs/assets/verified/`` because
the platform push gate carries ``docs/**`` binaries only.

Usage::

    PYTHONPATH=src OMP_NUM_THREADS=4 python benchmarks/verified/fem_heat_triangle_p1/run.py
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensorfem import heat_l2_triangle as study  # noqa: E402

FIGURE_DIR = REPO / "docs" / "assets" / "verified" / "fem_heat_triangle_p1"
ANCHOR = "fem_heat_triangle_p1"
TITLE = "Steady heat conduction, P1 triangles: manufactured-solution L2 rate 2 (mixed BC, distorted meshes)"


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figure_convergence(evidence: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    studies = evidence["studies"]
    kinds = {"structured-a": "o", "structured-b": "s", "distorted": "^"}
    colors = {"structured-a": "#1f4e79", "structured-b": "#2e8b57", "distorted": "#c1440e"}
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    theoretical = evidence["target_order"]["order"]
    for kind, marker in kinds.items():
        for key in evidence["signal_cases"]:
            lad = studies[f"{kind}/{key}"]
            h = np.asarray(lad["h"])
            axes[0].loglog(h, lad["l2_relative"], marker=marker, ms=4.5, lw=1.0,
                           color=colors[kind], alpha=0.85,
                           label=f"{kind}" if key == evidence["signal_cases"][0] else None)
            axes[1].loglog(h, lad["h1_relative"], marker=marker, ms=4.5, lw=1.0,
                           color=colors[kind], alpha=0.85,
                           label=f"{kind}" if key == evidence["signal_cases"][0] else None)
    # reference slopes anchored on the finest structured point of sine_generation
    anchor = studies["structured-a/sine_generation"]
    h_ref = np.asarray(anchor["h"][-1])
    hs = np.logspace(np.log10(anchor["h"][-1]), np.log10(anchor["h"][0]), 20)
    axes[0].loglog(hs, anchor["l2_relative"][-1] * (hs / h_ref) ** theoretical, "k--",
                   lw=1.2, label=f"theory O(h^{theoretical:.0f})")
    axes[1].loglog(hs, anchor["h1_relative"][-1] * (hs / h_ref) ** 1.0, "k--", lw=1.2,
                   label="theory O(h)")
    axes[0].set_xlabel("h = sqrt(area / ne)")
    axes[0].set_ylabel(r"relative $L_2$ error")
    axes[0].set_title("P1 triangles: L2 rate (7 MMS fields x 3 mesh families)")
    axes[1].set_xlabel("h = sqrt(area / ne)")
    axes[1].set_ylabel(r"relative $H^1$ seminorm error")
    axes[1].set_title("P1 triangles: H1 rate")
    for ax in axes:
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    path = FIGURE_DIR / "fig_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_field(evidence: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from tensorfem import heat_l2_triangle as study

    cases = study.triangle_field_cases()
    case = cases["sine_generation"]
    run = study.solve_case(case, max(evidence["levels"]), kind="structured-a")
    nodes = run["nodes"]
    elements = run["elements"]
    data = run["temperature"]
    exact = case.temperature(nodes)
    err = (data - exact).abs()
    x = nodes[:, 0].numpy()
    y = nodes[:, 1].numpy()
    tri = elements.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    for ax, values, title, cmap in (
        (axes[0], data.numpy(), f"T_h (P1, {run['nx']}x{run['ny']} mesh)", "viridis"),
        (axes[1], exact.numpy(), r"$T_{exact} = \sin(\pi x)\sin(\pi y)$", "viridis"),
        (axes[2], err.numpy(), r"$|T_h - T_{exact}|$", "magma"),
    ):
        tpc = ax.tripcolor(x, y, tri, values, shading="gouraud", cmap=cmap)
        fig.colorbar(tpc, ax=ax, shrink=0.85)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.suptitle("CASE-FEM-HEAT-L2 / P1 triangles: field and pointwise error "
                 f"(max |e| / max |T| = {run['errors']['relative_to_peak']:.3e})")
    fig.tight_layout()
    path = FIGURE_DIR / "fig_field.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_distortion(evidence: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    sensitivity = evidence["distortion_sensitivity"]
    levels = list(evidence["distortion_sensitivity_levels"])
    for label, entry in sensitivity.items():
        axes[0].loglog(entry["h"], entry["l2_relative"], "o-", ms=4,
                       label=f"distortion={entry['distortion']:.2f}")
        axes[1].plot(levels, entry["min_quality"], "o-", ms=4,
                     label=f"distortion={entry['distortion']:.2f}")
    anchor = evidence["studies"][f"structured-a/{evidence['signal_cases'][0]}"]
    ref = sensitivity[list(sensitivity)[0]]
    hs = np.logspace(np.log10(ref["h"][-1]), np.log10(ref["h"][0]), 20)
    axes[0].loglog(hs, ref["l2_relative"][-1] * (hs / ref["h"][-1]) ** evidence["target_order"]["order"],
                   "k--", lw=1.2, label=f"theory O(h^{evidence['target_order']['order']:.0f})")
    axes[0].set_xlabel("h = sqrt(area / ne)")
    axes[0].set_ylabel(r"relative $L_2$ error")
    axes[0].set_title("Distortion-amplitude sensitivity (sine_generation)")
    axes[1].set_xlabel("refinement level (nx = ny)")
    axes[1].set_ylabel("min triangle quality 2r/R")
    axes[1].set_title("Mesh quality stays bounded (shape regularity)")
    for ax in axes:
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGURE_DIR / "fig_distortion.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    quick = "--quick" in argv
    levels = (4, 8, 16) if quick else study.DEFAULT_LEVELS
    torch.set_num_threads(4)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    evidence = study.run_all(levels=levels,
                             sensitivity_levels=(8, 16, 32) if not quick else (4, 8, 16),
                             xval_level=max(levels[:-1]) if quick else 16,
                             quadrature_level=32 if not quick else 16,
                             progress=lambda msg: print("  ..", msg, flush=True))
    elapsed = time.time() - started
    evidence["distortion_sensitivity_levels"] = [4, 8, 16] if quick else [8, 16, 32]

    figures = [figure_convergence(evidence), figure_field(evidence), figure_distortion(evidence)]
    evidence["figures"] = [{"path": _repo_relative(p), "sha256": _sha256(p)} for p in figures]

    # strip the field arrays before serialising (kept only inside the figure code)
    for key, lad in evidence["studies"].items():
        lad.pop("runs_fields", None)

    evidence["environment"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "machine": platform.machine(),
        "dtype": "torch.float64",
        "threads": torch.get_num_threads(),
        "levels": list(levels),
        "quick": quick,
    }
    evidence["elapsed_seconds"] = round(elapsed, 2)
    evidence["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    gates = evidence["gates"]
    manifest = {
        "schema": "tensorfem.verified-gates.v1",
        "anchor": ANCHOR,
        "title": TITLE,
        "mission": "CASE-FEM-HEAT-L2",
        "generated_at": evidence["generated_at"],
        "criteria": {
            "l2_pair_order": study.GATE_L2_PAIR,
            "l2_finest_window": list(study.GATE_L2_FINEST),
            "h1_pair_order": study.GATE_H1_PAIR,
            "power_law_residual": study.GATE_POWER_LAW,
            "control_exact": study.GATE_CONTROL_ABS,
            "field_relative": study.GATE_FIELD_REL,
            "flux_balance": study.GATE_FLUX_BALANCE,
            "quadrature": study.GATE_QUADRATURE,
            "xval": study.GATE_XVAL,
            "diagonal": study.GATE_DIAGONAL,
        },
        "gates": [g for g in gates.values()],
        "passed": evidence["passed"],
        "failed_gates": evidence["failed_gates"],
        "not_run_gates": evidence["not_run_gates"],
        "artifacts": {
            "result": "result.json",
            "gates": "gates_verified.json",
            "figures": [_repo_relative(p) for p in figures],
        },
    }

    (HERE / "result.json").write_text(json.dumps(evidence, indent=2, default=float) + "\n")
    (HERE / "gates_verified.json").write_text(json.dumps(manifest, indent=2, default=float) + "\n")

    print("\n" + "=" * 74)
    print(f"elapsed {elapsed:.1f}s   figures -> {FIGURE_DIR}")
    for gid, gate in gates.items():
        mark = {True: "PASS", False: "FAIL"}[gate["passed"]] if gate["status"] == "run" else "NOT_RUN"
        print(f"  [{mark:7s}] {gid:26s} value={gate['value']!s:>22.22s} "
              f"{gate['operator']} {gate['threshold']}")
    print(f"overall: {'ALL PASS' if evidence['passed'] else 'FAIL'}")
    print("=" * 74)
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())