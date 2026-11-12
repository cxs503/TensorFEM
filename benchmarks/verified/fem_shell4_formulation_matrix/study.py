"""Formulation matrix for the flat Shell4 facet family.

Runs four evidence blocks over the four (membrane, transverse-shear) element
configurations and writes a single JSON artefact plus a figure:

A. zero-energy-mode accounting per facet (rigid / drilling gauge / spurious),
B. the same census on assembled flat meshes through the independent
   :mod:`tensorfem.element_rank_audit` tool (which expects 7 zero modes, not 6),
C. the Scordelis--Lo roof ladder (bending dominated, published reference),
D. the pressurised-cylinder membrane state (exact thin-shell reference).

Every number in the JSON is produced here; the documentation page renders it
and nothing is transcribed by hand.

Run::

    PYTHONPATH=src python benchmarks/verified/fem_shell4_formulation_matrix/study.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensorfem.element_rank_audit import audit_shell_mesh_scan
from tensorfem.shell4 import shell4_stiffness
from tensorfem.shell4_rank import element_rank_census, facet_rank_accounting
from tensorfem.shell4_mitc import shell4_mitc_stiffness
from tensorfem.shell_benchmarks import scordelis_lo_roof
from tensorfem.shell_membrane_benchmark import (
    pressurised_cylinder,
    pressurised_cylinder_reference,
)

SCHEMA = "fem-shell4-formulation-matrix/1"
CONFIGS = (
    ("shell4", "sri"),
    ("shell4", "mitc"),
    ("shell4_mitc", "sri"),
    ("shell4_mitc", "mitc"),
)
SCORDELIS_LO_REFERENCE = -0.3024
YOUNG, POISSON, THICKNESS = 200.0e9, 0.3, 0.01


def _unit_facet() -> torch.Tensor:
    return torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )


def block_a_rank_accounting() -> list[dict]:
    xyz = _unit_facet()
    records = []
    for formulation, plate_shear in CONFIGS:
        accounting = facet_rank_accounting(
            xyz, YOUNG, POISSON, THICKNESS,
            formulation=formulation, plate_shear=plate_shear,
        )
        legacy = element_rank_census(xyz, YOUNG, POISSON, THICKNESS, plate_shear=plate_shear)
        records.append({
            "formulation": formulation,
            "plate_shear": plate_shear,
            "zero_modes": accounting.zero_modes,
            "rigid_modes": accounting.rigid_modes,
            "gauge_modes": accounting.gauge_modes,
            "spurious_modes": accounting.spurious_modes,
            "legacy_spurious_modes": legacy.spurious_modes,
            "gauge_residual": accounting.gauge_residual,
            "rigid_residual": accounting.rigid_residual,
            "mechanism_plate_content": [
                float(torch.linalg.vector_norm(m[2::6])) for m in accounting.mechanisms
            ],
            "mechanism_section_content": [
                float(torch.linalg.vector_norm(m[3::6])) for m in accounting.mechanisms
            ],
        })
    return records


def gauge_is_the_legacy_mechanism() -> dict:
    """Show that the extra legacy mechanism of the MITC variant *is* the gauge."""
    xyz = _unit_facet()
    from tensorfem.shell4_rank import drilling_gauge_mode

    gauge = drilling_gauge_mode(xyz)
    legacy = element_rank_census(xyz, YOUNG, POISSON, THICKNESS, plate_shear="mitc")
    overlaps = [abs(float(gauge @ m)) for m in legacy.mechanisms]
    mechanism = legacy.mechanisms[0] if legacy.mechanisms else torch.zeros(24, dtype=torch.float64)
    return {
        "legacy_mechanism_count": legacy.spurious_modes,
        "gauge_overlap": max(overlaps) if overlaps else 0.0,
        "gauge_transverse_displacement_norm": float(torch.linalg.vector_norm(gauge[2::6])),
        "gauge_in_plane_section_norm": float(
            torch.linalg.vector_norm(torch.stack((gauge[3::6], gauge[4::6]), 1))
        ),
        "gauge_normal_section_norm": float(torch.linalg.vector_norm(gauge[5::6])),
        "mechanism_transverse_displacement_norm": float(torch.linalg.vector_norm(mechanism[2::6])),
    }


def block_b_mesh_census() -> list[dict]:
    records = []
    for formulation, plate_shear in CONFIGS:
        for record in audit_shell_mesh_scan(
            sizes=(2, 3, 4), formulation=formulation, plate_shear=plate_shear
        ):
            records.append({
                "formulation": formulation,
                "plate_shear": plate_shear,
                "n": record.n,
                "dofs": record.dofs,
                "null_dim": record.null_dim,
                "expected_rigid_dim": record.expected_rigid_dim,
                "spurious": record.spurious,
                "first_positive_ratio": record.first_positive_ratio,
            })
    return records


def block_c_scordelis_lo(sizes=(8, 16, 24)) -> list[dict]:
    records = []
    for formulation, plate_shear in CONFIGS:
        for n in sizes:
            solution = scordelis_lo_roof(
                n, n, plate_shear=plate_shear, formulation=formulation
            )
            w = solution.probe_displacement
            records.append({
                "formulation": formulation,
                "plate_shear": plate_shear,
                "n": n,
                "w": w,
                "reference": SCORDELIS_LO_REFERENCE,
                "relative_error": w / SCORDELIS_LO_REFERENCE - 1.0,
            })
    return records


def block_d_membrane(sizes=(8, 16, 24, 32, 48)) -> list[dict]:
    records = []
    exact = pressurised_cylinder_reference(
        100.0, 1.0, 1000.0, 1.0, 0.3, 100.0
    )
    for formulation, plate_shear in CONFIGS:
        for nt in sizes:
            result = pressurised_cylinder(
                nt=nt, nx=2, formulation=formulation, plate_shear=plate_shear
            )
            records.append({
                "formulation": formulation,
                "plate_shear": plate_shear,
                "nt": nt,
                "radial_displacement": result.mean_radial_displacement,
                "perimeter_radial_displacement": result.perimeter_radial_displacement,
                "exact_radial_displacement": exact["radial_displacement"],
                "radial_error": result.radial_error,
                "perimeter_error": result.perimeter_error,
                "axial_displacement": result.end_axial_displacement,
                "exact_axial_displacement": result.exact_end_axial_displacement,
                "axial_error": result.axial_error,
                "relative_ripple": result.relative_ripple,
                "rigid_translation": list(result.rigid_translation),
                "reaction_residual": result.reaction_residual,
            })
    return records


def observed_order(errors: list[float], sizes: list[float]) -> float | None:
    """Least-squares slope of log|error| vs log(size) when it is monotone."""
    pairs = [(math.log(s), math.log(abs(e))) for s, e in zip(sizes, errors) if abs(e) > 0]
    if len(pairs) < 2:
        return None
    n = len(pairs)
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    denom = sum((p[0] - mx) ** 2 for p in pairs)
    if denom == 0.0:
        return None
    return sum((p[0] - mx) * (p[1] - my) for p in pairs) / denom


def make_figure(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    labels = [f"{f}\n+{ps}" for f, ps in CONFIGS]
    colours = ["#b2182b", "#2166ac", "#762a83", "#1b7837"]

    ax = axes[0]
    width = 0.38
    positions = range(len(labels))
    legacy = [r["legacy_spurious_modes"] for r in results["rank_accounting"]]
    corrected = [r["spurious_modes"] for r in results["rank_accounting"]]
    ax.bar([p - width / 2 for p in positions], legacy, width, label="6-mode accounting", color="#b2182b")
    ax.bar([p + width / 2 for p in positions], corrected, width, label="gauge-aware (7 expected)", color="#2166ac")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("spurious zero-energy modes per facet")
    ax.set_title("(a) element rank accounting")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    for (formulation, plate_shear), colour in zip(CONFIGS, colours):
        rows = [r for r in results["scordelis_lo"] if r["formulation"] == formulation and r["plate_shear"] == plate_shear]
        xs = [r["n"] for r in rows]
        ys = [abs(r["relative_error"]) * 100 for r in rows]
        ax.plot(xs, ys, "o-", color=colour, label=f"{formulation} + {plate_shear}")
    ax.axhline(3.0, color="k", ls="--", lw=1, label="3 % gate")
    ax.set_xlabel("elements per edge")
    ax.set_ylabel("|error| vs w = -0.3024  [%]")
    ax.set_title("(b) Scordelis-Lo roof (bending dominated)")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for (formulation, plate_shear), colour in zip(CONFIGS, colours):
        rows = [r for r in results["membrane"] if r["formulation"] == formulation and r["plate_shear"] == plate_shear]
        xs = [r["nt"] for r in rows]
        ys = [abs(r["radial_error"]) * 100 for r in rows]
        ax.plot(xs, ys, "o-", color=colour, label=f"{formulation} + {plate_shear}")
    ax.axhline(3.0, color="k", ls="--", lw=1, label="3 % gate")
    ax.set_xlabel("facets per half ring")
    ax.set_ylabel("|error| vs w = pR^2/(Et)  [%]")
    ax.set_title("(c) pressurised cylinder (membrane state)")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(Path(__file__).parent))
    parser.add_argument("--sizes", default="8,16,24")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sizes = tuple(int(s) for s in args.sizes.split(","))

    results = {
        "schema": SCHEMA,
        "element": {"young": YOUNG, "poisson": POISSON, "thickness": THICKNESS},
        "rank_accounting": block_a_rank_accounting(),
        "gauge_identification": gauge_is_the_legacy_mechanism(),
        "mesh_census": block_b_mesh_census(),
        "scordelis_lo": block_c_scordelis_lo(sizes),
        "membrane": block_d_membrane(),
    }
    for formulation, plate_shear in CONFIGS:
        rows = [r for r in results["scordelis_lo"] if r["formulation"] == formulation and r["plate_shear"] == plate_shear]
        results.setdefault("orders", {})[f"scordelis_lo::{formulation}+{plate_shear}"] = observed_order(
            [r["relative_error"] for r in rows], [float(r["n"]) for r in rows]
        )
        rows = [r for r in results["membrane"] if r["formulation"] == formulation and r["plate_shear"] == plate_shear]
        results.setdefault("orders", {})[f"membrane::{formulation}+{plate_shear}"] = observed_order(
            [r["radial_error"] for r in rows], [float(r["nt"]) for r in rows]
        )
    results["gates"] = {
        "scordelis_lo_best_error": min(abs(r["relative_error"]) for r in results["scordelis_lo"]),
        "membrane_worst_error": max(abs(r["radial_error"]) for r in results["membrane"]),
        "membrane_worst_perimeter_error": max(abs(r["perimeter_error"]) for r in results["membrane"]),
        "membrane_max_relative_ripple": max(r["relative_ripple"] for r in results["membrane"]),
        "membrane_max_reaction_residual": max(r["reaction_residual"] for r in results["membrane"]),
        "mesh_census_max_spurious": max(r["spurious"] for r in results["mesh_census"]),
    }
    (outdir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    make_figure(results, outdir / "formulation_matrix.png")
    print(json.dumps(results["gates"], indent=2))
    for row in results["rank_accounting"]:
        print(
            f"{row['formulation']:11s}+{row['plate_shear']:4s} zero={row['zero_modes']} "
            f"gauge={row['gauge_modes']} spurious={row['spurious_modes']} "
            f"(legacy {row['legacy_spurious_modes']})"
        )
    for key, value in results["orders"].items():
        print(f"order {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())