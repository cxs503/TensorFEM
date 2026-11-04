"""Evidence generator: production Gauss-point stress recovery for the Cook membrane.

Runs the production payload (``additional_benchmark_fields.cook_fields``) on a refinement
ladder and records, machine-generated:

* the recovered (Barlow + area-weighted nodal averaging) pointwise von Mises error at the
  five reference probe sites, against the independent Q8 Richardson-extrapolated reference;
* the *raw* constant-per-element protocol error band at the same sites (every admissible
  containing element), which is the historically fail-closed criterion;
* two implementation cross-checks: Torch strain matrix vs the NumPy audit formula, and
  Torch recovered field vs ``cook_probe_protocol.nodal_recovered_stress`` (NumPy).

Outputs ``docs/cook-recovery-evidence.json`` and two figures under ``docs/figures/``.
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

from tensorfem.additional_benchmark_fields import COOK_POISSON, COOK_YOUNG, cook_fields  # noqa: E402
from tensorfem.cook_probe_protocol import nodal_recovered_stress  # noqa: E402
from tensorfem.cook_stress_reference import q4_strain_matrix as numpy_q4_strain  # noqa: E402
from tensorfem.stress_recovery import (  # noqa: E402
    GAUSS_ORDER,
    interpolate_nodal_field,
    locate_point,
    normalized_error,
    q4_strain_matrix,
    von_mises,
)

LEVELS = (8, 16, 32, 64)
REFERENCE_JSON = ROOT / "benchmarks" / "cook_membrane_stress_reference.json"
OUT_JSON = ROOT / "docs" / "cook-recovery-evidence.json"
FIG_DIR = ROOT / "docs" / "figures"


def load_reference() -> dict:
    payload = json.loads(REFERENCE_JSON.read_text())
    return payload["reference"]


def reference_points(reference: dict) -> list[dict]:
    return [
        {
            "point": point["point"],
            "xi": point["xi"],
            "eta": point["eta"],
            "von_mises_reference": point["von_mises"]["extrapolated"],
            "gci_fine_relative": point["von_mises"]["gci_fine_relative"],
        }
        for point in reference["points"]
    ]


def raw_protocol_band(case: dict, points: list[dict], scale: float) -> dict:
    """Error band of the raw constant-per-element protocol at every probe site.

    A probe on an element interface is contained by two or four elements, each carrying a
    different constant stress: the protocol may return any of them, so the reproducible
    statement is the interval spanned by all admissible choices.
    """
    nodes = torch.tensor(case["nodes"], dtype=torch.float64)
    elements = torch.tensor(case["elements"], dtype=torch.int64)
    centre_vm = torch.tensor(case["von_mises"], dtype=torch.float64)
    sites = []
    for entry in points:
        hits = locate_point(nodes, elements, entry["point"])
        if not hits:
            raise RuntimeError(f"probe {entry['point']} outside the mesh at n={case['mesh']}")
        errors = [
            normalized_error(float(centre_vm[hit["element"]]), entry["von_mises_reference"], scale)
            for hit in hits
        ]
        sites.append(
            {
                "point": entry["point"],
                "admissible_elements": len(hits),
                "on_element_boundary": any(hit["on_boundary"] for hit in hits),
                "min_error": float(min(errors)),
                "max_error": float(max(errors)),
            }
        )
    return {
        "sites": sites,
        "worst_site_min_error": max(site["min_error"] for site in sites),
        "worst_site_max_error": max(site["max_error"] for site in sites),
        "ambiguous_sites": sum(site["admissible_elements"] > 1 for site in sites),
    }


def strain_matrix_cross_check(case: dict) -> dict:
    """Torch strain matrix vs the NumPy audit formula, at the four Gauss points."""
    nodes = torch.tensor(case["nodes"], dtype=torch.float64)
    elements = torch.tensor(case["elements"], dtype=torch.int64)
    coordinates = nodes[elements]
    worst = 0.0
    for element in (0, len(elements) // 3, len(elements) // 2, len(elements) - 1):
        for xi, eta in GAUSS_ORDER:
            torch_matrix = q4_strain_matrix(xi, eta, coordinates[element])
            numpy_matrix, _ = numpy_q4_strain(xi, eta, coordinates[element].numpy())
            worst = max(worst, float(np.max(np.abs(torch_matrix.numpy() - numpy_matrix))))
    return {
        "max_abs_difference": worst,
        "gauss_point": 1.0 / math.sqrt(3.0),
        "elements_sampled": 4,
        "passed": bool(worst < 1e-12),
    }


def recovery_cross_check(case: dict) -> dict:
    """Torch recovered field vs the independent NumPy audit recovery."""
    nodes = torch.tensor(case["nodes"], dtype=torch.float64)
    elements = torch.tensor(case["elements"], dtype=torch.int64)
    displacement = torch.tensor(case["displacement"], dtype=torch.float64).reshape(-1, 2)
    audit = nodal_recovered_stress(nodes.numpy(), elements.numpy(), displacement.numpy())
    production = np.asarray(case["nodal_stress"], dtype=np.float64)
    difference = float(np.abs(production - audit).max())
    scale = float(np.abs(audit).max())
    return {
        "max_abs_difference": difference,
        "stress_scale": scale,
        "relative": difference / scale if scale else 0.0,
        "passed": bool(difference / scale < 1e-10 if scale else difference == 0.0),
    }


def rung(case: dict, points: list[dict], scale: float) -> dict:
    nodes = torch.tensor(case["nodes"], dtype=torch.float64)
    elements = torch.tensor(case["elements"], dtype=torch.int64)
    nodal = torch.tensor(case["nodal_stress"], dtype=torch.float64)
    sites = []
    for entry in points:
        sample = interpolate_nodal_field(nodes, elements, nodal, entry["point"])
        sites.append(
            {
                "point": entry["point"],
                "von_mises": sample["von_mises"],
                "von_mises_reference": entry["von_mises_reference"],
                "error": normalized_error(sample["von_mises"], entry["von_mises_reference"], scale),
                "on_element_boundary": sample["on_element_boundary"],
                "containing_elements": sample["containing_elements"],
                "spread": sample["spread"],
            }
        )
    return {
        "mesh": case["mesh"],
        "nodes": len(case["nodes"]),
        "element_size": "L/n",
        "sites": sites,
        "worst_site_error": max(site["error"] for site in sites),
        "worst_boundary_spread": max(site["spread"] for site in sites),
        "displacement_relative_error": case["verification"]["response_relative_error"],
        "stress_accuracy_qualified_raw": case["stress_accuracy_qualified"],
    }


def observed_order(errors: list[float], levels: list[int]) -> float:
    """Finite-difference observed order between the two finest levels."""
    (e0, e1), (n0, n1) = errors[-2:], levels[-2:]
    if e1 <= 0.0 or e0 <= 0.0:
        return float("nan")
    return math.log(e0 / e1) / math.log(n1 / n0)


def make_figures(ladder: list[dict], raw_bands: dict, case_finest: dict, points: list[dict]) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figures = {}

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ns = [rung["mesh"] for rung in ladder]
    recovered = [100.0 * rung["worst_site_error"] for rung in ladder]
    raw_max = [100.0 * raw_bands[rung["mesh"]]["worst_site_max_error"] for rung in ladder]
    raw_min = [100.0 * raw_bands[rung["mesh"]]["worst_site_min_error"] for rung in ladder]
    ax.loglog(ns, raw_max, "s--", color="#b03030", label="raw element-centre, worst admissible")
    ax.loglog(ns, raw_min, "s:", color="#b03030", alpha=.6, label="raw element-centre, best admissible")
    ax.loglog(ns, recovered, "o-", color="#1f4e9c", label="recovered nodal field (this work)")
    ax.axhline(3.0, color="k", lw=1.0, ls="-.", label="3 % gate")
    ax.set_xlabel("Q4 divisions per edge n")
    ax.set_ylabel("worst probe-site von Mises error [%]")
    ax.set_title("Cook membrane: pointwise stress error vs mesh")
    ax.grid(True, which="both", alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / "cook_recovery_error_ladder.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures["cook_recovery_error_ladder.png"] = {
        "path": str(path.relative_to(ROOT)),
        "caption": "Pointwise von Mises error at the five reference probes: raw constant-per-element "
                   "band vs recovered nodal field, log-log, 3 % gate.",
    }

    nodes = np.asarray(case_finest["nodes"], dtype=np.float64)
    nodal = np.asarray(case_finest["nodal_stress"], dtype=np.float64)
    vm_recovered = np.sqrt(
        nodal[:, 0] ** 2 - nodal[:, 0] * nodal[:, 1] + nodal[:, 1] ** 2 + 3 * nodal[:, 2] ** 2
    )
    centers = np.asarray(case_finest["centers"], dtype=np.float64)
    vm_raw = np.asarray(case_finest["von_mises"], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    tri = matplotlib.tri.Triangulation(nodes[:, 0], nodes[:, 1])
    filled = axes[0].tripcolor(tri, vm_recovered, shading="gouraud", cmap="viridis")
    fig.colorbar(filled, ax=axes[0], label="von Mises [E=1, t=1]")
    axes[0].set_title(f"recovered nodal field, n={case_finest['mesh']}")
    axes[0].set_aspect("equal")
    scattered = axes[1].scatter(centers[:, 0], centers[:, 1], c=vm_raw, cmap="viridis", s=6)
    fig.colorbar(scattered, ax=axes[1], label="von Mises [E=1, t=1]")
    axes[1].set_title(f"raw element-centre field, n={case_finest['mesh']}")
    axes[1].set_aspect("equal")
    for axis in axes:
        for entry in points:
            axis.plot(*entry["point"], "rx", ms=7, mew=1.4)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    fig.tight_layout()
    path = FIG_DIR / "cook_recovery_field.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures["cook_recovery_field.png"] = {
        "path": str(path.relative_to(ROOT)),
        "caption": "Cook membrane von Mises field at the finest level: recovered nodal field (left) vs the "
                   "raw element-centre field (right); red crosses are the five independent reference probes.",
    }
    return figures


def main() -> dict:
    torch.set_default_dtype(torch.float64)
    reference = load_reference()
    points = reference_points(reference)
    scale = max(point["von_mises_reference"] for point in points)

    cases = []
    ladder = []
    raw_bands = {}
    for level in LEVELS:
        case = cook_fields(level, reference=reference)
        cases.append(case)
        raw_bands[level] = raw_protocol_band(case, points, scale)
        ladder.append(rung(case, points, scale))
        print(
            f"n={level:3d} nodes={ladder[-1]['nodes']:5d} "
            f"recovered worst={100*ladder[-1]['worst_site_error']:.3f}% "
            f"raw band={100*raw_bands[level]['worst_site_min_error']:.2f}.."
            f"{100*raw_bands[level]['worst_site_max_error']:.2f}% "
            f"flag={ladder[-1]['stress_accuracy_qualified_raw']} "
            f"spread={ladder[-1]['worst_boundary_spread']:.2e}"
        )

    finest = cases[-1]
    evidence = {
        "case": "cook-membrane",
        "generated_by": "scripts/cook_recovery_production_evidence.py",
        "environment": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "dtype": "float64",
        },
        "problem": {
            "geometry": "Cook membrane: 48 x 44 tapered panel, unit thickness, plane stress",
            "material": {"young": COOK_YOUNG, "poisson": COOK_POISSON},
            "load": "uniform shear traction 6.25 per unit length on the right edge, clamped left edge",
            "reference": "independent Q8 (serendipity) three-level Richardson extrapolation",
            "reference_source": str(REFERENCE_JSON.relative_to(ROOT)),
        },
        "probe_sites": points,
        "von_mises_scale": scale,
        "levels": list(LEVELS),
        "ladder": ladder,
        "raw_protocol_band": {str(key): value for key, value in raw_bands.items()},
        "observed_order_recovered": observed_order(
            [rung["worst_site_error"] for rung in ladder], list(LEVELS)
        ),
        "observed_order_raw_max": observed_order(
            [raw_bands[level]["worst_site_max_error"] for level in LEVELS], list(LEVELS)
        ),
        "cross_checks": {
            "torch_vs_numpy_strain_matrix": strain_matrix_cross_check(cases[0]),
            "torch_vs_numpy_recovery_n64": recovery_cross_check(finest),
        },
        "status": {
            "stress_accuracy_qualified_raw": finest["stress_accuracy_qualified"],
            "stress_recovery_qualified": bool(
                ladder[-1]["worst_site_error"] < 0.03
                and all(
                    later["worst_site_error"] < earlier["worst_site_error"]
                    for earlier, later in zip(ladder, ladder[1:])
                )
            ),
            "criterion": "worst probe-site normalised von Mises error < 3 % on the finest level, "
                         "monotone decrease over the ladder",
        },
        "scope": "Pointwise von Mises accuracy at five declared material points of the Cook membrane "
                 "under the recovered (single-valued) field. Not a 3D thickness-stress certificate, "
                 "not an industrial acceptance, and it does not relax the raw element-centre protocol.",
        "figures": list(make_figures(ladder, raw_bands, finest, points)),
    }
    evidence["figures"] = make_figures(ladder, raw_bands, finest, points)
    OUT_JSON.write_text(json.dumps(evidence, indent=1, ensure_ascii=False))
    print("recovered worst errors:", [round(100 * rung["worst_site_error"], 4) for rung in ladder])
    print("raw band max:", [round(100 * raw_bands[level]["worst_site_max_error"], 3) for level in LEVELS])
    print("cross-checks:", json.dumps(evidence["cross_checks"]))
    print("status:", json.dumps(evidence["status"]))
    print("written", OUT_JSON)
    return evidence


if __name__ == "__main__":
    main()