"""W22 study runner: Smith M_u vs the IACS CSR ULS/yield envelope.

Writes ``docs/w22_hull_girder/uls_utilization.json`` (the mission artefact) and
the figures embedded in ``papers/reports/w22_hull_girder1.md``.

Numbers in the JSON are produced by :func:`tensorfem.hull_girder_csr_w22.run_w22_study`;
this script adds run provenance (git commit, input hashes) and the plots.  No
number is typed by hand.

Usage::

    python benchmarks/w22_hull_girder_uls.py [--strips 8] [--steps 320]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfem.hull_girder_csr_w22 import CASE_CATALOGUE, run_w22_study  # noqa: E402

OUT_DIR = ROOT / "docs" / "w22_hull_girder"
FIG_DIR = OUT_DIR / "figures"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # pragma: no cover - git absent
        return "unknown"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot_moment_curvature(study: dict) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))
    for column, condition in enumerate(("sagging", "hogging")):
        axis = axes[0][column]
        for row in study["cases"]:
            curve = row["ultimate"]["curve_sagging" if condition == "sagging" else "curve_hogging"]
            axis.plot([point[0] for point in curve], [point[1] / 1e9 for point in curve],
                      label=row["case"])
        axis.set_title(f"Smith moment-curvature, {condition}")
        axis.set_xlabel("curvature (1/m)")
        axis.set_ylabel("M (GN.m)")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)

        axis = axes[1][column]
        cases = [row["case"] for row in study["cases"]]
        utilisation = [row["uls"][f"utilisation_unit_factors_{condition}"]
                       for row in study["cases"]]
        axis.bar(cases, utilisation, color="#3b6ea5")
        axis.axhline(1.0, color="crimson", linestyle="--", label="CSR limit (unit factors)")
        axis.set_title(f"ULS utilization, {condition}")
        axis.set_ylabel("(M_sw + M_wv) / M_u (-)")
        axis.tick_params(axis="x", rotation=20)
        axis.legend(fontsize=8)
    fig.suptitle("W22 Smith incremental collapse: moment-curvature and ULS utilization")
    fig.tight_layout()
    path = FIG_DIR / "moment_curvature_and_utilization.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_capacity_comparison(study: dict) -> Path:
    cases = [row["case"] for row in study["cases"]]
    fig, axis = plt.subplots(figsize=(9.0, 4.6))
    width = 0.2
    index = range(len(cases))
    axis.bar([i - 1.5 * width for i in index],
             [row["ultimate"]["M_u_sagging"] / 1e9 for row in study["cases"]],
             width, label="Smith M_u sagging")
    axis.bar([i - 0.5 * width for i in index],
             [row["ultimate"]["M_u_hogging"] / 1e9 for row in study["cases"]],
             width, label="Smith M_u hogging")
    axis.bar([i + 0.5 * width for i in index],
             [row["cross_validation"]["limiting_stress_model"]["M_u_hogging"] / 1e9
              for row in study["cases"]],
             width, label="limiting-stress model, hogging")
    axis.bar([i + 1.5 * width for i in index],
             [row["cross_validation"]["plastic_moment"]["M_p"] / 1e9 for row in study["cases"]],
             width, label="fully plastic M_p")
    axis.set_xticks(list(index))
    axis.set_xticklabels(cases, rotation=15)
    axis.set_ylabel("M (GN.m)")
    axis.set_title("Ultimate capacity: Smith M_u vs limiting-stress model and M_p")
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / "capacity_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_component_curves(study: dict) -> Path:
    from tensorfem.hull_girder_csr_w22 import build_scantlings, panel_component_law

    case = CASE_CATALOGUE[0]
    members, _ = build_scantlings(case)
    fig, axis = plt.subplots(figsize=(8.0, 4.6))
    for member in members:
        law = panel_component_law(member)
        strains = [-0.006 + 0.000005 * i for i in range(2401)]
        axis.plot([s * 1000 for s in strains], [law.stress(s) / 1e6 for s in strains],
                  label=member.name)
    axis.set_xlabel("strain (per mille, negative = compression)")
    axis.set_ylabel("panel stress (MPa)")
    axis.set_title(f"Smith component curves, {case.key} midship section")
    axis.legend(fontsize=7)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / "component_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strips", type=int, default=8)
    parser.add_argument("--steps", type=int, default=320)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    study = run_w22_study(strips=args.strips, steps=args.steps, repo_root=ROOT)
    study["provenance"] = {
        "git_commit": _git_commit(),
        "runner": "benchmarks/w22_hull_girder_uls.py",
        "runner_sha256": _sha256(Path(__file__)),
        "module": "src/tensorfem/hull_girder_csr_w22.py",
        "module_sha256": _sha256(ROOT / "src" / "tensorfem" / "hull_girder_csr_w22.py"),
        "anchor_file": "benchmarks/anchors/iacs-csr-w22.json",
        "anchor_sha256": _sha256(ROOT / "benchmarks" / "anchors" / "iacs-csr-w22.json"),
        "figures": [],
    }

    figures = [plot_moment_curvature(study), plot_capacity_comparison(study),
               plot_component_curves(study)]
    study["provenance"]["figures"] = [
        {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)} for path in figures
    ]

    out = OUT_DIR / "uls_utilization.json"
    out.write_text(json.dumps(study, indent=2) + "\n")
    print(f"wrote {out}")
    for criterion in study["criteria"]:
        print(f"{criterion['id']}: {criterion['verdict']} value={criterion['value']}")
    for row in study["cases"]:
        print(f"{row['case']:>16}  M_u_sag={row['ultimate']['M_u_sagging']/1e9:8.3f} GN.m "
              f"M_u_hog={row['ultimate']['M_u_hogging']/1e9:8.3f} GN.m "
              f"U_sag={row['uls']['utilisation_unit_factors_sagging']:.3f} "
              f"U_hog={row['uls']['utilisation_unit_factors_hogging']:.3f} "
              f"conv={row['convergence']['max_relative_change']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())