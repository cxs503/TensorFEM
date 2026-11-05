"""Verify the PR #79 (agent/m3402 -> main) union of the two stress-recovery families.

The merged ``tensorfem.stress_recovery`` publishes both implementations that were
developed in parallel:

* the NumPy audit kernels (``nodal_recovered_stress`` / ``interpolate_nodal_stress``),
* the Torch production kernels (``q4_gauss_stresses`` / ``recover_nodal_stress`` /
  ``interpolate_nodal_field``).

This script measures, on the production Cook membrane displacement:

1. cross-implementation agreement of the two recovered nodal fields (they should agree
   at round-off, because they implement the same Barlow algorithm),
2. the dispatch equivalence of the unified helpers ``q4_shape`` / ``element_areas``,
3. the pointwise Cook criterion against the frozen independent Q8 reference,
   reproducing the documented 0.0802311 / 0.0195144 / 0.00483792 / 0.00120190 sequence,
4. the observed convergence order of the recovered field.

Run: ``OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/verify_stress_recovery_union.py``
Writes ``benchmarks/cook_recovery_union_verification.json`` and three PNGs under
``docs/figures/``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.additional_benchmark_fields import cook_fields  # noqa: E402
from tensorfem.cook_recovery_qualification import (  # noqa: E402
    GATE,
    cook_recovery_study,
    load_frozen_reference,
)
from tensorfem.stress_recovery import (  # noqa: E402
    element_areas,
    interpolate_nodal_field,
    interpolate_nodal_stress,
    nodal_recovered_stress,
    q4_gauss_stresses,
    q4_shape,
    recover_nodal_stress,
    recovery_error,
    von_mises,
)

LEVELS = (8, 16, 32, 64)
OUT_JSON = ROOT / "benchmarks" / "cook_recovery_union_verification.json"
FIG_DIR = ROOT / "docs" / "figures"


def rows_for(n: int, reference: dict) -> dict:
    """One mesh level: both recoveries, their difference, and the pointwise errors."""
    case = cook_fields(n)
    nodes = torch.as_tensor(np.asarray(case["nodes"], dtype=np.float64))
    elements = torch.as_tensor(np.asarray(case["elements"], dtype=np.int64))
    displacement = torch.as_tensor(np.asarray(case["displacement"], dtype=np.float64))

    # NumPy audit path (m3402 side).
    np_nodes = np.asarray(case["nodes"], dtype=np.float64)
    np_elements = np.asarray(case["elements"], dtype=np.int64)
    np_displacement = np.asarray(case["displacement"], dtype=np.float64)
    numpy_field = nodal_recovered_stress(np_nodes, np_elements, np_displacement)

    # Torch production path (main side).
    gauss = q4_gauss_stresses(nodes, elements, displacement, 1.0, 1.0 / 3.0)
    torch_field = recover_nodal_stress(nodes, elements, gauss).numpy()

    difference = np.abs(numpy_field - torch_field)
    scale = float(np.max(np.abs(numpy_field)))

    points = []
    for point in reference["points"]:
        numpy_probe = interpolate_nodal_stress(np_nodes, np_elements, numpy_field, point["point"])
        torch_probe = interpolate_nodal_field(nodes, elements, torch.as_tensor(torch_field), point["point"])
        component = interpolate_nodal_stress(
            np_nodes, np_elements, numpy_field, point["point"]
        )["stress"]
        component_errors = {
            key: recovery_error(component[index], point["components"][key], reference["scale"])
            for index, key in enumerate(("sxx", "syy", "sxy"))
        }
        points.append(
            {
                "point": point["point"],
                "von_mises": numpy_probe["von_mises"],
                "von_mises_reference": point["von_mises"],
                "von_mises_error": recovery_error(
                    numpy_probe["von_mises"], point["von_mises"], reference["scale"]
                ),
                "component_errors": component_errors,
                "path_delta_von_mises": abs(numpy_probe["von_mises"] - torch_probe["von_mises"]),
                "boundary_spread": torch_probe["spread"],
            }
        )

    return {
        "mesh": n,
        "elements": len(np_elements),
        "max_abs_field_difference": float(difference.max()),
        "max_rel_field_difference": float(difference.max() / scale),
        "max_von_mises_error": max(entry["von_mises_error"] for entry in points),
        "max_component_error": max(
            value for entry in points for value in entry["component_errors"].values()
        ),
        "max_path_delta_von_mises": max(entry["path_delta_von_mises"] for entry in points),
        "max_boundary_spread": max(entry["boundary_spread"] for entry in points),
        "points": points,
        "numpy_field": numpy_field,
        "nodes": np_nodes,
        "elements_np": np_elements,
        "displacement": np_displacement,
    }


def dispatch_checks() -> dict:
    """The unified helpers must reproduce both original behaviours."""
    values = {}
    numpy_shape = q4_shape(0.37, -0.62)
    torch_shape = q4_shape(0.37, -0.62, dtype=torch.float64)
    values["q4_shape_numpy_type"] = type(numpy_shape).__name__
    values["q4_shape_torch_type"] = type(torch_shape).__name__
    values["q4_shape_max_difference"] = float(
        np.max(np.abs(np.asarray(numpy_shape) - torch_shape.numpy()))
    )
    values["q4_shape_sums_to_one"] = float(np.sum(numpy_shape))

    square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    tapered = np.array([[0.0, 0.0], [3.0, 0.0], [2.4, 1.1], [0.4, 0.9]])
    values["square_area_numpy"] = float(element_areas(square, np.array([[0, 1, 2, 3]]))[0])
    values["square_area_torch"] = float(
        element_areas(torch.as_tensor(square), torch.tensor([[0, 1, 2, 3]]))[0]
    )
    values["tapered_area_numpy"] = float(element_areas(tapered, np.array([[0, 1, 2, 3]]))[0])
    values["tapered_area_torch"] = float(
        element_areas(torch.as_tensor(tapered), torch.tensor([[0, 1, 2, 3]]))[0]
    )
    return values


def observed_order(levels, errors) -> float:
    orders = [
        math.log(errors[index] / errors[index + 1]) / math.log(levels[index + 1] / levels[index])
        for index in range(len(levels) - 1)
    ]
    return float(np.mean(orders))


def main() -> int:
    reference = load_frozen_reference()
    rows = [rows_for(n, reference) for n in LEVELS]
    stored = json.loads((ROOT / "benchmarks" / "cook_recovery_qualification.json").read_text())
    stored_errors = {row["mesh"]: row["max_error"] for row in stored["rows"]}
    study = cook_recovery_study(levels=LEVELS, reference=reference)

    errors = [row["max_von_mises_error"] for row in rows]
    payload = {
        "purpose": "verify the merged (PR #79) union of the Torch and NumPy stress-recovery families",
        "levels": list(LEVELS),
        "gate": GATE,
        "reference_sha256": json.loads(
            (ROOT / "benchmarks" / "cook_membrane_stress_reference.json").read_text()
        )["reference"].get("sha256", ""),
        "dispatch": dispatch_checks(),
        "rows": [
            {key: value for key, value in row.items() if not key.endswith(("field", "np", "nodes", "displacement"))}
            for row in rows
        ],
        "observed_order_von_mises": observed_order(LEVELS, errors),
        "stored_error_reproduction": {
            str(row["mesh"]): {
                "stored": stored_errors.get(row["mesh"]),
                "fresh": row["max_von_mises_error"],
                "delta": abs(row["max_von_mises_error"] - stored_errors.get(row["mesh"], float("nan"))),
            }
            for row in rows
        },
        "max_abs_stored_delta": max(
            abs(row["max_von_mises_error"] - stored_errors.get(row["mesh"], 0.0)) for row in rows
        ),
        "qualification": {
            "recovery_qualified": study["recovery_qualified"],
            "monotone": study["monotone"],
        },
        "study_max_errors": {str(row["mesh"]): row["max_error"] for row in study["rows"]},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    # --- figures -----------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 9})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    sizes = [row["mesh"] for row in rows]
    ax.loglog(sizes, [row["max_von_mises_error"] * 100 for row in rows], "o-", label="von Mises (union path)")
    ax.loglog(sizes, [row["max_component_error"] * 100 for row in rows], "s--", label="worst component")
    ax.axhline(GATE * 100, color="crimson", lw=1.0, ls=":", label="3 % gate")
    ax.set_xlabel("mesh n (n x n Q4)")
    ax.set_ylabel("max error at the 5 probe sites [%]")
    ax.set_title("Cook membrane: recovered-field error (log-log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    deltas = [row["max_rel_field_difference"] for row in rows]
    ax2.semilogy(sizes, deltas, "d-", color="darkgreen", label="max |NumPy - Torch| / scale")
    ax2.semilogy(sizes, [row["max_boundary_spread"] for row in rows], "x--", color="gray", label="probe spread (Torch)")
    ax2.set_xlabel("mesh n")
    ax2.set_ylabel("normalised difference")
    ax2.set_title("Cross-implementation agreement of the union")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pr79-union-agreement.png", dpi=130)
    plt.close(fig)

    # field contour of the finest mesh (nodal recovered von Mises, production path)
    fine = rows[-1]
    nodes, elements = fine["nodes"], fine["elements_np"]
    vm = np.array([von_mises(row) for row in fine["numpy_field"]])
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    contour = ax.tricontourf(nodes[:, 0], nodes[:, 1], vm, levels=24, cmap="viridis")
    ax.triplot(nodes[:, 0], nodes[:, 1], triangles=elements, color="k", lw=0.25, alpha=0.25)
    for point in reference["points"]:
        ax.plot(point["point"][0], point["point"][1], "o", ms=3.5, mfc="white", mec="crimson")
    fig.colorbar(contour, ax=ax, label="von Mises [MPa]")
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"Cook membrane recovered von Mises, n={fine['mesh']}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pr79-union-field.png", dpi=130)
    plt.close(fig)

    # resolution table figure: conflicts per file, text (documentation) figure
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.axis("off")
    labels = [
        "file                                   conflicts   resolution",
        "src/tensorfem/stress_recovery.py  (add/add) 3   union: keep Torch + NumPy kernels",
        "docs/benchmark-qualification-status.md   1   union: keep both evidence sentences",
        "tests/test_cook_recovery_qualification.py 1->  superset key assertion (main added 2 keys)",
    ]
    ax.text(0.01, 0.95, "\n".join(labels), family="monospace", fontsize=8, va="top")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pr79-union-conflicts.png", dpi=130)
    plt.close(fig)

    print(json.dumps({k: payload[k] for k in ("levels", "observed_order_von_mises", "max_abs_stored_delta", "qualification")}, indent=2))
    for row in payload["rows"]:
        print(
            f"n={row['mesh']:3d}  vm_err={row['max_von_mises_error']*100:8.5f}%  "
            f"comp_err={row['max_component_error']*100:8.5f}%  "
            f"numpy-vs-torch={row['max_rel_field_difference']:.3e}"
        )
    print(json.dumps(payload["dispatch"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())