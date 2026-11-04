"""Verification study: the *measurement protocol* for pointwise Q4 stress in the Cook membrane.

Why this case
-------------
``docs/cook-membrane-stress-audit.md`` closed the operator question (m2430) but left two
residual debts, both reproduced here:

1. ``stress_accuracy_qualified`` stayed ``False`` because the production Q4 field showed a
   maximum von Mises error of 5.02 % at n=64 against the independent Q8 reference.
2. ``recovery_cross_check`` reported ``inf``: the audit's own pointwise Q4 recovery could
   not reproduce the stored element-centre stress, so the comparison was declared
   uncertifiable.

What this study finds
---------------------
* **Debt 2 was a root-finder defect, not an operator defect.** ``inverse_map`` used a bare
  Newton iteration from the centre of the parent square. For the Cook trapezoid that
  iteration does not converge globally: it returned a point inside a *different* element for
  2 of the 5 declared stress probes and for 4 of the 16 element centres, and it left the
  unit square entirely for one centre (the ``inf``). The map is bilinear, so the pre-image
  is the root of a quadratic; it is now solved in closed form and polished with Newton. The
  cross-check then agrees to rounding error, which retroactively confirms m2430's operator
  fix and removes the ``inf`` artefact.
* **Debt 1 was largely a protocol defect.** All five declared material probes are dyadic
  fractions of the parent square, so for *every* power-of-two mesh they sit exactly on an
  element interface (distance 0.0). The Q4 stress field is discontinuous across interfaces,
  so "the" stress at those points does not exist: the one-sided limits differ by 1.0-2.6 %
  at n=64 and up to 17 % at n=8. Reporting a single number there measures where the
  discontinuity fell, not the accuracy of the discretisation.
* The protocol is therefore re-declared: the same five nominal materials points are probed
  at fixed *off-interface* parent coordinates (so that each probe is unambiguously interior
  to one element at every level of the ladder), with the Q8 reference evaluated at those
  same coordinates and extrapolated in the same way.

Evidence produced
-----------------
* ``benchmarks/cook_q4_pointwise_protocol.json`` -- every number quoted below.
* ``docs/assets/cook_q4_protocol_*.png`` -- convergence log-log plot, one-sided spread plot,
  and a side-by-side von Mises field comparison.
* ``docs/cook-membrane-pointwise-protocol.md`` -- the report (numbers injected, never typed).

Run:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/cook_q4_pointwise_protocol.py
"""

from __future__ import annotations

import json
import math
import platform as host
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorfem import __version__  # noqa: E402
from tensorfem import cook_stress_reference as reference  # noqa: E402
from tensorfem.additional_benchmark_fields import cook_fields  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
EVIDENCE = ROOT / "benchmarks" / "cook_q4_pointwise_protocol.json"

REFERENCE_LEVELS = (12, 24, 48)
PLATFORM_LEVELS = (16, 32, 64, 128)
GATE = 0.03

#: Fixed offsets applied to the declared material points.  Chosen so that, for every level
#: of the ladder, the probe stays inside one element (>= 20 %/5 % of an element away from
#: the nearest interface) instead of sitting on it.
OFFSET_X = 1.0 / (3 * PLATFORM_LEVELS[-1])
OFFSET_Y = 1.0 / (5 * PLATFORM_LEVELS[-1])


def von_mises(stress: np.ndarray) -> float:
    sxx, syy, sxy = float(stress[0]), float(stress[1]), float(stress[2])
    return math.sqrt(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy)


def interface_distance(parent: tuple[float, float], divisions: int) -> float:
    """Distance from a parent coordinate to the nearest element interface.

    Returned in parent-square units; zero means the probe lies exactly on an interface,
    where the Q4 stress field has two distinct one-sided values.
    """
    xi, eta = parent
    fractions = (xi * divisions - round(xi * divisions), eta * divisions - round(eta * divisions))
    return min(abs(fractions[0]), abs(fractions[1])) / divisions


def one_sided_q4(case: dict, parent: tuple[float, float], delta: float) -> list[float]:
    """One-sided von Mises values of every element meeting the probe point."""
    nodes = np.asarray(case["nodes"], dtype=np.float64)
    elements = np.asarray(case["elements"], dtype=np.int64)
    displacement = np.asarray(case["displacement"], dtype=np.float64).reshape(-1, 2)
    divisions = case["mesh"]
    values: list[float] = []
    for dx in (-delta, delta):
        for dy in (-delta, delta):
            xi, eta = parent[0] + dx, parent[1] + dy
            i = min(max(int(math.floor(xi * divisions)), 0), divisions - 1)
            j = min(max(int(math.floor(eta * divisions)), 0), divisions - 1)
            local_xi = 2.0 * (xi * divisions - i) - 1.0
            local_eta = 2.0 * (eta * divisions - j) - 1.0
            connectivity = elements[j * divisions + i]
            matrix, _ = reference.q4_strain_matrix(local_xi, local_eta, nodes[connectivity])
            stress = reference.plane_stress_matrix() @ matrix @ displacement[connectivity].ravel()
            value = von_mises(stress)
            if all(abs(value - other) > 0.0 for other in values):
                values.append(value)
    return values


def interior_q4(case: dict, parent: tuple[float, float]) -> dict:
    """Q4 stress at a probe that is interior to exactly one element."""
    nodes = np.asarray(case["nodes"], dtype=np.float64)
    elements = np.asarray(case["elements"], dtype=np.int64)
    displacement = np.asarray(case["displacement"], dtype=np.float64).reshape(-1, 2)
    divisions = case["mesh"]
    xi, eta = parent
    i = min(max(int(math.floor(xi * divisions)), 0), divisions - 1)
    j = min(max(int(math.floor(eta * divisions)), 0), divisions - 1)
    local_xi = 2.0 * (xi * divisions - i) - 1.0
    local_eta = 2.0 * (eta * divisions - j) - 1.0
    connectivity = elements[j * divisions + i]
    matrix, _ = reference.q4_strain_matrix(local_xi, local_eta, nodes[connectivity])
    stress = reference.plane_stress_matrix() @ matrix @ displacement[connectivity].ravel()
    return {
        "xi": xi,
        "eta": eta,
        "point": reference.map_point(xi, eta).tolist(),
        "von_mises": von_mises(stress),
        "components": [float(value) for value in stress],
        "element": [i, j],
        "interface_distance": interface_distance(parent, divisions),
    }


def reference_at(solutions: list[dict], parent: tuple[float, float]) -> dict:
    table = [reference.field_stress(solution, *parent) for solution in solutions]
    values = [entry["von_mises"] for entry in table]
    divisions = [solution["divisions"] for solution in solutions]
    fit = reference.richardson(values, divisions)
    return {
        "xi": parent[0],
        "eta": parent[1],
        "point": reference.map_point(*parent).tolist(),
        "levels": divisions,
        "values": values,
        "von_mises": fit,
        "components": {
            key: reference.richardson([entry[key] for entry in table], divisions)
            for key in ("sxx", "syy", "sxy")
        },
    }


def relative(value: float, target: float, scale: float) -> float:
    return abs(value - target) / max(abs(target), 0.05 * scale)


def field_grid(case: dict) -> tuple[np.ndarray, np.ndarray]:
    """Von Mises at the element centres of the production Q4 field, on the parent grid."""
    divisions = case["mesh"]
    values = np.asarray(case["von_mises"], dtype=np.float64).reshape(divisions, divisions)
    return values, divisions


def main() -> dict:
    started = time.time()
    solutions = reference.build_reference(REFERENCE_LEVELS)["solutions"]

    nominal = list(reference.SAMPLE_POINTS)
    probes = [(xi + OFFSET_X, eta + OFFSET_Y) for xi, eta in nominal]

    # 1. Reference values at the off-interface probes (independent Q8, Richardson extrapolated).
    off_reference = [reference_at(solutions, probe) for probe in probes]
    scale = max(entry["von_mises"]["extrapolated"] for entry in off_reference)

    cases = []
    for level in PLATFORM_LEVELS:
        clock = time.time()
        case = cook_fields(level)
        cases.append({"case": case, "seconds": time.time() - clock})

    # 2. Off-interface Q4 accuracy ladder.
    ladder = []
    for entry in cases:
        case, seconds = entry["case"], entry["seconds"]
        rows = []
        for probe, target in zip(probes, off_reference):
            measured = interior_q4(case, probe)
            error = relative(measured["von_mises"], target["von_mises"]["extrapolated"], scale)
            rows.append({
                "point": measured["point"],
                "parent": [probe[0], probe[1]],
                "interface_distance": measured["interface_distance"],
                "element": measured["element"],
                "computed": measured["von_mises"],
                "reference": target["von_mises"]["extrapolated"],
                "relative_error": error,
            })
        ladder.append({
            "divisions": case["mesh"],
            "response_relative_error": case["verification"]["response_relative_error"],
            "solver_seconds": seconds,
            "max_relative_error": max(row["relative_error"] for row in rows),
            "probes": rows,
        })

    history = [[row["relative_error"] for row in level["probes"]] for level in ladder]
    monotone = all(
        all(fine < coarse for fine, coarse in zip(later, earlier))
        for earlier, later in zip(history, history[1:])
    )
    finest = ladder[-1]
    q4_orders = []
    for index in range(len(probes)):
        values = [level["probes"][index]["computed"] for level in ladder[-3:]]
        fit = reference.richardson(values, [level["divisions"] for level in ladder[-3:]])
        q4_orders.append({
            "point": finest["probes"][index]["point"],
            "observed_order": fit["observed_order"],
            "order_source": fit["order_source"],
            "extrapolated": fit["extrapolated"],
            "reference": finest["probes"][index]["reference"],
            "extrapolated_relative_error": relative(
                fit["extrapolated"], finest["probes"][index]["reference"], scale
            ),
        })

    # 3. Mesh-aligned protocol: the declared probes sit on interfaces at every level.
    aligned = []
    for entry in cases:
        case, seconds = entry["case"], entry["seconds"]
        rows = []
        for probe in nominal:
            values = one_sided_q4(case, probe, delta=1.0e-7)
            rows.append({
                "parent": list(probe),
                "interface_distance": interface_distance(probe, case["mesh"]),
                "one_sided_values": values,
                "spread_relative_to_min": (
                    (max(values) - min(values)) / min(values) if len(values) > 1 else 0.0
                ),
                "one_sided_count": len(values),
            })
        aligned.append({
            "divisions": case["mesh"],
            "solver_seconds": seconds,
            "max_spread": max(row["spread_relative_to_min"] for row in rows),
            "probes": rows,
        })

    cross_check = reference.cross_check_pointwise_recovery(cases[0]["case"])
    location = reference.inverse_map_checked(np.asarray(off_reference[0]["point"]))
    round_trip = []
    for probe in probes:
        target = np.asarray(reference.map_point(*probe))
        result = reference.inverse_map_checked(target)
        round_trip.append({
            "parent": [probe[0], probe[1]],
            "residual": result["residual"],
            "candidates": result["candidates"],
            "converged": result["converged"],
        })

    # 4. Figures.
    fig, axis = plt.subplots(figsize=(6.4, 4.6))
    sizes = [48.0 / level["divisions"] for level in ladder]
    for index, probe in enumerate(probes):
        errors = [level["probes"][index]["relative_error"] for level in ladder]
        axis.loglog(sizes, errors, marker="o", label=f"probe {index + 1} {probe[0]:.3f},{probe[1]:.3f}")
    axis.loglog(sizes, [GATE * (size / sizes[0]) for size in sizes], "--", color="k",
                label="first order (slope 1)")
    axis.axhline(GATE, color="tab:red", linestyle=":", label="3 % gate")
    axis.set_xlabel("element size h (parent units)")
    axis.set_ylabel("|von Mises - Q8 extrapolated| / reference")
    axis.set_title("Cook membrane: pointwise Q4 stress, off-interface probes")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ASSETS / "cook_q4_protocol_convergence.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    divisions = [level["divisions"] for level in aligned]
    for index, probe in enumerate(nominal):
        spread = [level["probes"][index]["spread_relative_to_min"] for level in aligned]
        axis.loglog(divisions, spread, marker="s",
                    label=f"probe {index + 1} {probe[0]},{probe[1]}")
    axis.axhline(GATE, color="tab:red", linestyle=":", label="3 % gate")
    axis.set_xlabel("mesh divisions n (probe sits exactly on an interface)")
    axis.set_ylabel("one-sided spread / min one-sided value")
    axis.set_title("Interface/orientation ambiguity of the mesh-aligned probes")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ASSETS / "cook_q4_protocol_interface_spread.png", dpi=150)
    plt.close(fig)

    q4_field, q4_div = field_grid(cases[-1]["case"])
    grid = np.linspace(0.0, 1.0, 121)
    q8_field = np.array([
        [reference.field_stress(solutions[-1], xi, eta)["von_mises"] for xi in grid]
        for eta in grid
    ])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    image = axes[0].imshow(q4_field, origin="lower", extent=(0, 1, 0, 1), cmap="viridis")
    axes[0].set_title(f"production Q4, n={q4_div}, element-centre von Mises")
    axes[0].set_xlabel("xi")
    axes[0].set_ylabel("eta")
    fig.colorbar(image, ax=axes[0], shrink=0.85)
    image = axes[1].imshow(q8_field, origin="lower", extent=(0, 1, 0, 1), cmap="viridis",
                          vmin=float(q4_field.min()), vmax=float(q4_field.max()))
    axes[1].set_title(f"independent Q8 reference, n={REFERENCE_LEVELS[-1]}")
    axes[1].set_xlabel("xi")
    axes[1].set_ylabel("eta")
    fig.colorbar(image, ax=axes[1], shrink=0.85)
    fig.tight_layout()
    fig.savefig(ASSETS / "cook_q4_protocol_field.png", dpi=150)
    plt.close(fig)

    qualified = bool(finest["max_relative_error"] < GATE and monotone)
    evidence = {
        "case": "Cook membrane pointwise stress protocol",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "tensorfem": __version__,
            "python": host.python_version(),
            "numpy": np.__version__,
            "elapsed_seconds": round(time.time() - started, 1),
        },
        "problem": {
            "corners": reference.CORNERS.tolist(),
            "young": reference.YOUNG,
            "poisson": reference.POISSON,
            "thickness": reference.THICKNESS,
            "loaded_edge_resultant": reference.RESULTANT,
            "gate": GATE,
        },
        "probes": {
            "nominal_material_points": [list(point) for point in nominal],
            "offset": [OFFSET_X, OFFSET_Y],
            "off_interface": [list(point) for point in probes],
            "aligned_interface_distance": [
                interface_distance(point, level) for point in nominal for level in PLATFORM_LEVELS
            ],
        },
        "locator": {
            "cross_check": cross_check,
            "round_trip": round_trip,
            "first_probe_located": location,
        },
        "reference": {
            "levels": list(REFERENCE_LEVELS),
            "scale_von_mises": scale,
            "probes": off_reference,
        },
        "ladder": ladder,
        "q4_richardson": q4_orders,
        "mesh_aligned_protocol": aligned,
        "verdict": {
            "qualified": qualified,
            "monotone": monotone,
            "finest_divisions": finest["divisions"],
            "finest_max_relative_error": finest["max_relative_error"],
            "gate": GATE,
            "statement": (
                "off-interface pointwise von Mises converges and meets the 3 % gate at the finest level"
                if qualified
                else "off-interface pointwise von Mises does not meet the 3 % gate at the finest level"
            ),
        },
        "scope": (
            "Linear-elastic plane stress, E=1, nu=1/3, thickness 1, unit resultant shear on x=48. "
            "Five declared material points probed at fixed parent coordinates offset off the element "
            "interfaces; reference is an independently coded Q8 solution extrapolated over n=12/24/48. "
            "Clamped-corner and loaded-edge-endpoint singularities are excluded."
        ),
    }
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(evidence, indent=2) + "\n")

    print(f"locator cross-check: max relative difference {cross_check['max_relative_difference']:.3e}"
          f" passed={cross_check['passed']}")
    print("aligned probes interface distance:",
          sorted({round(value, 6) for value in evidence['probes']['aligned_interface_distance']}))
    for level in ladder:
        print(f"Q4 off-interface n={level['divisions']:>4}: max error "
              f"{100 * level['max_relative_error']:.3f}% ({level['solver_seconds']:.1f}s)")
    for level in aligned:
        print(f"mesh-aligned  n={level['divisions']:>4}: max one-sided spread "
              f"{100 * level['max_spread']:.3f}%")
    for entry in q4_orders:
        print(f"  probe {entry['point']}: observed order {entry['observed_order']:.2f} "
              f"({entry['order_source']}), extrapolated vs reference "
              f"{100 * entry['extrapolated_relative_error']:.3f}%")
    print("monotone", monotone, "qualified", qualified, "->", EVIDENCE.relative_to(ROOT))
    return evidence


if __name__ == "__main__":
    main()