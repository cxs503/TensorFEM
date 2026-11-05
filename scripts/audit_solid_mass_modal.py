"""Reproduce the solid-mass audit artefacts (JSON + figures + Markdown tables).

    PYTHONPATH=src python scripts/audit_solid_mass_modal.py

Writes ``outputs/solid-mass-modal/summary.json`` and two PNG figures under
``docs/assets/solid-mass-modal/``.  Nothing here is hand-copied: the numbers in
``docs/solid-mass-modal-verification.md`` come from the JSON this script emits.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensorfem.solid_mass_audit import summary

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "solid-mass-modal"
FIGS = ROOT / "docs" / "assets" / "solid-mass-modal"


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    data = summary()
    data["environment"] = {"torch": torch.__version__, "dtype": "float64"}
    (OUT / "summary.json").write_text(json.dumps(data, indent=2, sort_keys=True))

    marker = {"hex8": "o", "tet4": "s"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for entry in data["rod"]:
        sizes = [case["mesh_size"] for case in entry["cases"]]
        style = "--" if entry["mass_form"] == "lumped" else "-"
        for mode in (0, 2):
            errors = [100.0 * case["relative_errors"][mode] for case in entry["cases"]]
            axes[mode // 2].loglog(
                sizes,
                errors,
                style,
                marker=marker[entry["element_type"]],
                label=f"{entry['element_type']} {entry['mass_form']}",
            )
    for ax, mode in zip(axes, (1, 3)):
        ax.set_xlabel("element size h [m]")
        ax.set_ylabel("relative frequency error [%]")
        ax.set_title(f"axial rod, mode {mode}")
        ax.grid(True, which="both", alpha=0.3)
        ax.axhline(3.0, color="k", lw=1, ls=":", label="3% gate")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "axial-rod-convergence.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    labels, mass_err, polar_err = [], [], []
    for case in data["invariants"]:
        labels.append(f"{case['element_type']}\n{case['mass_form']}")
        mass_err.append(max(case["mass_error"], 1e-17))
        polar_err.append(max(case["polar_inertia_error"], 1e-17))
    positions = range(len(labels))
    ax.bar([p - 0.2 for p in positions], mass_err, width=0.4, label="total mass")
    ax.bar([p + 0.2 for p in positions], polar_err, width=0.4, label="polar inertia")
    ax.set_yscale("log")
    ax.set_xticks(list(positions), labels, fontsize=8)
    ax.set_ylabel("relative invariant error")
    ax.set_title("mass invariants of the assembled solid mass matrix")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "mass-invariants.png", dpi=160)
    plt.close(fig)

    print("| element | mass form | mode 1 % | mode 2 % | mode 3 % | order (m1..m3) |")
    print("|---|---|---:|---:|---:|---|")
    for entry in data["rod"]:
        finest = entry["cases"][-1]
        errors = ", ".join(f"{100.0 * e:.4f}" for e in finest["relative_errors"])
        order = ", ".join(f"{o:.2f}" for o in entry["observed_order"])
        print(
            f"| {entry['element_type']} | {entry['mass_form']} | "
            f"{errors} | {order} |"
        )
    print()
    print("| element | mass form | total mass error | polar inertia error |")
    print("|---|---|---:|---:|")
    for case in data["invariants"]:
        print(
            f"| {case['element_type']} | {case['mass_form']} | "
            f"{case['mass_error']:.3e} | {case['polar_inertia_error']:.3e} |"
        )
    return data


if __name__ == "__main__":
    main()