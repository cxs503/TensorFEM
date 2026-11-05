"""Generate the fail-closed Cook-membrane stress-recovery qualification record.

Runs the production payload ladder (``additional_benchmark_fields.cook_fields``)
through :mod:`tensorfem.cook_stress_recovery`, records the resulting verdict in
``benchmarks/cook_stress_recovery.json`` and draws the convergence figure stored
next to the write-up.

Two implementation cross-checks are embedded in the record so the recovered field
cannot silently drift from either the production payload or the independent NumPy
audit implementation:

* ``wrapper_vs_production_payload`` -- :func:`cook_stress_recovery.recovered_nodal_stress`
  recomputed from ``(nodes, elements, displacement)`` against ``payload['nodal_stress']``;
* ``production_vs_numpy_audit`` -- the same field against
  :func:`tensorfem.cook_probe_protocol.nodal_recovered_stress` (NumPy, written for the
  probe-protocol audit, i.e. an independent code path).

Run: ``OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/qualify_cook_stress_recovery.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.additional_benchmark_fields import cook_fields  # noqa: E402
from tensorfem.cook_probe_protocol import nodal_recovered_stress as numpy_recovered  # noqa: E402
from tensorfem.cook_stress_recovery import (  # noqa: E402
    DEFAULT_LEVELS,
    SCHEMA,
    load_reference,
    measure,
    qualify,
    recovered_nodal_stress,
)

OUT_JSON = ROOT / "benchmarks" / "cook_stress_recovery.json"
FIG_DIR = ROOT / "docs" / "assets" / "cook-stress-recovery"
FIG_PNG = FIG_DIR / "convergence.png"
CROSS_CHECK_LEVEL = 8
CROSS_CHECK_TOLERANCE = 1e-12


def cross_checks() -> dict:
    """Wrapper vs production payload vs independent NumPy audit, at one coarse level."""
    reference = load_reference()["raw"]
    case = cook_fields(CROSS_CHECK_LEVEL, reference=reference)
    nodes, elements = case["nodes"], case["elements"]
    production = np.asarray(case["nodal_stress"], dtype=np.float64)
    wrapped = recovered_nodal_stress(nodes, elements, case["displacement"])["field"].numpy()
    audited = np.asarray(
        numpy_recovered(nodes, elements, np.asarray(case["displacement"], dtype=np.float64)),
        dtype=np.float64,
    )
    scale = float(np.abs(production).max())
    wrapper_difference = float(np.abs(wrapped - production).max())
    audit_difference = float(np.abs(audited - production).max())
    return {
        "level": CROSS_CHECK_LEVEL,
        "stress_scale": scale,
        "wrapper_vs_production_payload": {
            "max_absolute_difference": wrapper_difference,
            "max_relative_difference": wrapper_difference / scale,
            "passed": bool(wrapper_difference / scale <= CROSS_CHECK_TOLERANCE),
        },
        "production_vs_numpy_audit": {
            "max_absolute_difference": audit_difference,
            "max_relative_difference": audit_difference / scale,
            "passed": bool(audit_difference / scale <= CROSS_CHECK_TOLERANCE),
        },
        "meaning": (
            "the documented entry point, the production payload and the independent "
            "NumPy audit agree to machine precision on the same displacement solution"
        ),
    }


def make_figure(verdict: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    errors = np.asarray(verdict["error_sequence"], dtype=np.float64)
    levels = np.asarray(verdict["levels"], dtype=np.float64)
    # The Cook panel is 48 long, so an n-division grid has h = 48 / n.
    spacing = 48.0 / levels
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5.4, 4.0), dpi=115)
    axis.loglog(spacing, errors, "o-", color="#1f4e79", label="recovered field (worst site)")
    guide = errors[0] * (spacing / spacing[0]) ** 2
    axis.loglog(spacing, guide, "--", color="#8c8c8c", label="second order guide (p = 2)")
    axis.axhline(verdict["tolerance"], color="#b00020", linestyle=":", label="3 % acceptance limit")
    for value, spacing_value, error in zip(levels, spacing, errors):
        axis.annotate(
            f"n={int(value)}: {100 * error:.3f}%",
            (spacing_value, error),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7,
        )
    axis.set_xlabel("element size h (panel length 48 / n)")
    axis.set_ylabel("worst-site von Mises error")
    axis.set_title(
        f"Cook membrane recovered stress, observed order {verdict['observed_order']:.2f}",
        fontsize=9,
    )
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=7, loc="lower right")
    figure.tight_layout()
    figure.savefig(FIG_PNG)
    plt.close(figure)
    return FIG_PNG


def main() -> dict:
    levels = DEFAULT_LEVELS
    measured = measure(levels)
    verdict = qualify(levels, measured=measured)
    record = {
        "schema": SCHEMA,
        "case": "cook-membrane",
        "generated_by": "scripts/qualify_cook_stress_recovery.py",
        "environment": {"torch": __import__("torch").__version__, "numpy": np.__version__},
        "problem": {
            "geometry": "Cook membrane: 48 x 44 tapered panel, unit thickness, plane stress",
            "material": {"young": 1.0, "poisson": 1.0 / 3.0},
            "load": "uniform shear traction 6.25 per unit length on the right edge, clamped left edge",
            "reference": "independent Q8 serendipity three-level Richardson extrapolation",
            "reference_path": "benchmarks/cook_membrane_stress_reference.json",
        },
        "protocol": verdict["protocol"],
        "criterion": verdict["criterion"],
        "scope": verdict["scope"],
        "reference": verdict["reference"],
        "tolerance": verdict["tolerance"],
        "levels": verdict["levels"],
        "ladder": verdict["ladder"],
        "error_sequence": verdict["error_sequence"],
        "observed_order": verdict["observed_order"],
        "monotone_decreasing": verdict["monotone_decreasing"],
        "stress_recovery_qualified": verdict["stress_recovery_qualified"],
        "blocking_reasons": verdict["blocking_reasons"],
        "stress_accuracy_qualified": verdict["stress_accuracy_qualified"],
        "element_centre_field": verdict["element_centre_field"],
        "cross_checks": cross_checks(),
        "figure": str(FIG_PNG.relative_to(ROOT)),
        "verdict_sha256": verdict["verdict_sha256"],
    }
    record["qualification_recorded"] = bool(
        record["stress_recovery_qualified"]
        and not record["stress_accuracy_qualified"]
        and all(
            check["passed"] for check in record["cross_checks"].values() if isinstance(check, dict)
        )
    )
    OUT_JSON.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    make_figure(verdict)

    print("errors:", [f"{100 * value:.4f}%" for value in record["error_sequence"]])
    print("observed order:", round(record["observed_order"], 3))
    print("blocking reasons:", record["blocking_reasons"])
    print("stress_recovery_qualified:", record["stress_recovery_qualified"])
    print("cross-checks:", json.dumps(record["cross_checks"]))
    print("recorded:", record["qualification_recorded"])
    print("written", OUT_JSON.relative_to(ROOT), "and", record["figure"])
    return record


if __name__ == "__main__":
    main()
