"""Generate the quadratic-solid rank audit evidence (JSON + figures).

Run from the repository root::

    OMP_NUM_THREADS=1 PYTHONPATH=src python3 scripts/audit_quadratic_solid_rank.py

Outputs
-------
``docs/benchmarks/tutorials/quadratic-solid-rank-audit.json``
    Every scalar quoted in ``docs/quadratic-solid-rank-audit.md``: element-level
    rank records (shipped 27-point rule vs the audit-only 8-point rule, regular /
    distorted / curved geometry, TET10 straight and curved), the free-free warped
    mesh scans, the measured quadrature exactness and the normalized eigenvalue
    spectra drawn in the figures.
``docs/assets/quadratic_rank_element_spurious.png``
    Spurious zero-energy modes per quadratic-solid formulation.
``docs/assets/quadratic_rank_spectra.png``
    Normalized eigenvalue spectra of one curved 20-node brick under both rules.
``docs/assets/quadratic_rank_mesh_scan.png``
    Element-level mechanisms against the free-free assembled null space.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfem.element_rank_audit import (  # noqa: E402
    _hex20_curved_geometry,
    audit_hex20_mesh_scan,
    audit_library,
    audit_quadratic_solids,
    classify_null_space,
    hex20_reduced_stiffness,
    hex20_quadrature_convergence,
    quadrature_exactness,
    rigid_body_basis_solid3d,
    summary_table,
)
from tensorfem.hex20 import hex20_stiffness  # noqa: E402
from tensorfem.solid3d import elasticity_matrix_3d  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "docs" / "benchmarks" / "tutorials"
ASSET_DIR = ROOT / "docs" / "assets"
YOUNG = 200.0e9
POISSON = 0.3


def spectra(records_count: int = 12) -> dict:
    """Normalized eigenvalue spectra of the curved brick under both rules."""
    d3 = elasticity_matrix_3d(torch.tensor([YOUNG]), torch.tensor([POISSON]))
    nodes = _hex20_curved_geometry()
    shipped, _ = hex20_stiffness(nodes.unsqueeze(0), d3)
    reduced = hex20_reduced_stiffness(nodes.unsqueeze(0), d3)
    out = {}
    for label, k in (("hex20 3x3x3", shipped[0]), ("hex20 2x2x2 reduced", reduced[0])):
        values = torch.linalg.eigvalsh(0.5 * (k + k.transpose(0, 1)))
        scale = float(values.abs().max())
        out[label] = [float(v / scale) for v in values[:records_count]]
    return out


def figure_element_spurious(records, path: Path) -> None:
    labels = [r.element.replace(" (", "\n(") for r in records]
    counts = [r.spurious for r in records]
    colors = ["#2a6f97" if r.spurious == 0 else "#c1121f" for r in records]
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    bars = ax.bar(range(len(records)), counts, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(len(records)))
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=8)
    ax.set_ylabel("spurious zero-energy modes")
    ax.set_title("Quadratic solids: zero-energy modes beyond the 6 rigid-body motions")
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, record in zip(bars, records):
        ax.annotate(
            f"{record.null_dim}/{record.dofs}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8,
        )
    ax.set_ylim(0, max(counts) * 1.35 if max(counts) else 1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_spectra(data: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for label, values in data.items():
        ax.semilogy(range(1, len(values) + 1), [max(abs(v), 1e-18) for v in values], "o-", label=label)
    ax.set_xlabel("mode index (ascending)")
    ax.set_ylabel("|lambda| / lambda_max")
    ax.set_title("One curved 20-node brick: 8-point rule adds 6 zero modes")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_mesh_scan(element_records, mesh_records, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    element_spurious = {}
    for record in element_records:
        if "hex20" in record.element:
            element_spurious[record.element] = record.spurious
    mesh_spurious = {}
    for record in mesh_records:
        mesh_spurious.setdefault(record.element, []).append(record.spurious)

    labels = list(mesh_spurious)
    positions = range(len(labels))
    ax1.bar(
        positions,
        [max(v) for v in mesh_spurious.values()],
        color="#2a6f97",
        edgecolor="black",
        linewidth=0.6,
    )
    ax1.set_xticks(list(positions))
    ax1.set_xticklabels([label.replace(" warped mesh", "\nwarped mesh") for label in labels], fontsize=8)
    ax1.set_ylabel("spurious modes in the free-free assembly")
    ax1.set_title("Assembled warped meshes (element level: 0 vs 6)")
    ax1.set_ylim(0, 7)
    ax1.axhline(6, color="#c1121f", linestyle="--", linewidth=1.0)
    ax1.annotate(
        "8-point rule, single element: 6 mechanisms",
        (0.02, 0.92),
        xycoords="axes fraction",
        color="#c1121f",
        fontsize=8,
    )

    for label in labels:
        series = [r for r in mesh_records if r.element == label]
        ax2.semilogy(
            [r.n for r in series],
            [r.first_positive_ratio for r in series],
            "o-",
            label=label,
        )
    ax2.set_xlabel("mesh subdivisions per edge")
    ax2.set_ylabel("lambda_softest / lambda_max")
    ax2.set_title("Softest elastic mode under refinement")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    element_records = audit_quadratic_solids(young=YOUNG, poisson=POISSON)
    library_records = audit_library(young=YOUNG, poisson=POISSON)
    mesh_records = (
        audit_hex20_mesh_scan(sizes=(2, 3), young=YOUNG, poisson=POISSON)
        + audit_hex20_mesh_scan(sizes=(3,), jitter=0.05, seed=7, young=YOUNG, poisson=POISSON)
    )
    exactness = quadrature_exactness(max_degree=5)
    convergence = hex20_quadrature_convergence(young=YOUNG, poisson=POISSON)
    spectrum = spectra()

    d3 = elasticity_matrix_3d(torch.tensor([YOUNG]), torch.tensor([POISSON]))
    nodes = _hex20_curved_geometry()
    curved_spectrum = classify_null_space(
        hex20_reduced_stiffness(nodes.unsqueeze(0), d3)[0], rigid_body_basis_solid3d(nodes)
    )

    payload = {
        "benchmark": "quadratic_solid_rank_audit",
        "description": (
            "Zero-energy (rank-deficiency) audit of the quadratic solid paths: shipped "
            "HEX20 27-point rule versus the 8-point rule, TET10, regular/distorted/curved "
            "geometry, assembled warped meshes and measured quadrature exactness."
        ),
        "young_modulus": YOUNG,
        "poisson_ratio": POISSON,
        "element_records": [r.as_dict() for r in element_records],
        "library_records": [r.as_dict() for r in library_records],
        "mesh_records": [r.as_dict() for r in mesh_records],
        "quadrature_exactness": exactness,
        "quadrature_convergence": convergence,
        "spectra": spectrum,
        "reduced_curved_explained": int(curved_spectrum["explained"]),
        "reduced_curved_cosines": curved_spectrum["cosines"],
        "tables": {
            "element": summary_table(element_records),
            "mesh": summary_table(mesh_records),
        },
    }
    evidence_path = EVIDENCE_DIR / "quadratic-solid-rank-audit.json"
    evidence_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")

    figure_element_spurious(element_records, ASSET_DIR / "quadratic_rank_element_spurious.png")
    figure_spectra(spectrum, ASSET_DIR / "quadratic_rank_spectra.png")
    figure_mesh_scan(element_records, mesh_records, ASSET_DIR / "quadratic_rank_mesh_scan.png")

    print(payload["tables"]["element"])
    print(payload["tables"]["mesh"])
    for row in exactness:
        print(
            f"{row['rule']:34s} points={row['points']:2d} "
            f"exact_total={row['exact_through_total_degree']} "
            f"exact_per_axis={row['exact_through_per_axis']}"
        )
    for row in convergence:
        print(
            f"{row['geometry']:32s} 3pt-vs-4pt={row['order3_vs_order4']:.3e} "
            f"4pt-vs-5pt={row['order4_vs_order5']:.3e} 2pt-vs-4pt={row['order2_vs_order4']:.3e}"
        )
    print(f"wrote {evidence_path}")


if __name__ == "__main__":
    main()