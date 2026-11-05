"""Certify the production Q4 stress-recovery field used by ``cook_fields``.

The Cook membrane audit loop closed with two independent statements:

* the *stored element-centre* field ``cook_fields(...)['stress']`` is first-order
  accurate and misses the 3 % pointwise gate (3.687 % at n=128), so
  ``stress_accuracy_qualified`` stays ``False``;
* a Barlow/Zienkiewicz-Zhu recovered field reached 0.12 % at n=64 on the same
  displacement solution, but it only existed inside the audit module
  (``docs/cook-membrane-pointwise-protocol.md``, "恢复场尚未进入生产 cook_fields").

This script certifies the recovered field now that it *is* in production
(``tensorfem.stress_recovery`` + ``cook_fields(...)['stress_nodal']``):

* ladder n = 8/16/32/64 of the recovered field interpolated at the five frozen
  off-interface probes, against the frozen independent Q8 reference;
* observed convergence order per probe (recovered vs raw element-centre);
* single-valuedness of the interpolated nodal field at every probe;
* a fail-closed verdict (``recovery_verdict``) that also refuses to pass on a
  non-monotone ladder.

Reproduce::

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src \
        python scripts/audit_cook_stress_recovery.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.additional_benchmark_fields import cook_fields  # noqa: E402
from tensorfem.stress_recovery import (  # noqa: E402
    interpolate_nodal_stress,
    locate_point,
    recovery_verdict,
    relative_error,
)

PROTOCOL = ROOT / "benchmarks" / "cook_q4_pointwise_protocol.json"
OUTPUT = ROOT / "benchmarks" / "cook_stress_recovery_evidence.json"
LEVELS = (8, 16, 32, 64)
GATE = 0.03

# Raw element-centre protocol numbers, transcribed from the frozen audit evidence
# (``benchmarks/cook_q4_pointwise_protocol.json`` -> ``ladder[*].max_relative_error``)
# so that the contrast below never re-uses the recovered field to describe the raw one.
RAW_LADDER = None


def observed_order(coarse: float, fine: float, refinement: float = 2.0) -> float:
    """p from ``e_coarse / e_fine = refinement**p``; 0.0 when the ratio is undefined."""
    if fine <= 0.0 or coarse <= 0.0:
        return 0.0
    return math.log(coarse / fine) / math.log(refinement)


def load_reference() -> dict:
    data = json.loads(PROTOCOL.read_text())
    probes = []
    for entry in data["reference"]["probes"]:
        probes.append(
            {
                "point": entry["point"],
                "parent": [entry["xi"], entry["eta"]],
                "von_mises": entry["von_mises"]["extrapolated"],
                "gci": entry["von_mises"]["gci_fine_relative"],
            }
        )
    return {
        "probes": probes,
        "scale": data["reference"]["scale_von_mises"],
        "levels": data["reference"]["levels"],
        "raw_ladder": [
            {
                "divisions": entry["divisions"],
                "max_relative_error": entry["max_relative_error"],
                "response_relative_error": entry["response_relative_error"],
            }
            for entry in data["ladder"]
        ],
    }


def evaluate_level(divisions: int, reference: dict) -> dict:
    started = time.time()
    case = cook_fields(divisions)
    nodes = torch.tensor(np.asarray(case["nodes"], dtype=np.float64))
    elements = torch.tensor(np.asarray(case["elements"], dtype=np.int64))
    nodal = torch.tensor(np.asarray(case["stress_nodal"], dtype=np.float64))
    raw_centres = np.asarray(case["stress"], dtype=np.float64)
    probes = []
    for probe in reference["probes"]:
        recovered = interpolate_nodal_stress(nodes, elements, nodal, probe["point"])
        error = relative_error(recovered["von_mises"], probe["von_mises"], reference["scale"])
        # Raw field: the constant stress of the element that *contains* the probe, i.e.
        # what the stored element-centre field actually offers at that material point.
        # This is the honest best case for the raw field; the merged sealed protocol
        # (benchmarks/cook_q4_pointwise_protocol.json) additionally selects a
        # floor-tie-break element and reports 3.687 % at n=128.
        raw_index = int(locate_point(nodes, elements, probe["point"])["hits"][0]["element"])
        sxx, syy, sxy = raw_centres[raw_index]
        raw_vm = math.sqrt(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy)
        probes.append(
            {
                "point": probe["point"],
                "reference": probe["von_mises"],
                "gci": probe["gci"],
                "recovered": recovered["von_mises"],
                "recovered_relative_error": error,
                "recovered_element_spread": recovered["element_spread"],
                "recovered_candidates": recovered["candidate_count"],
                "raw_containing_element": raw_index,
                "raw": raw_vm,
                "raw_relative_error": relative_error(raw_vm, probe["von_mises"], reference["scale"]),
            }
        )
    return {
        "divisions": divisions,
        "nodes": int(len(nodes)),
        "elements": int(len(elements)),
        "solver_seconds": round(time.time() - started, 3),
        "response_relative_error": case["verification"]["response_relative_error"],
        "max_recovered_relative_error": max(p["recovered_relative_error"] for p in probes),
        "max_raw_relative_error": max(p["raw_relative_error"] for p in probes),
        "max_element_spread": max(p["recovered_element_spread"] for p in probes),
        "probes": probes,
    }


def main() -> int:
    reference = load_reference()
    ladder = [evaluate_level(divisions, reference) for divisions in LEVELS]
    recovered_max = [level["max_recovered_relative_error"] for level in ladder]
    raw_max = [level["max_raw_relative_error"] for level in ladder]
    verdict = recovery_verdict(recovered_max, gate=GATE)
    raw_verdict = recovery_verdict(raw_max, gate=GATE, require_monotone=False)

    # Observed order of the recovered field for the worst probe of the finest level.
    finest = ladder[-1]
    worst = max(finest["probes"], key=lambda probe: probe["recovered_relative_error"])
    slot = [index for index, probe in enumerate(finest["probes"]) if probe is worst][0]
    orders = [
        observed_order(
            ladder[index]["probes"][slot]["recovered_relative_error"],
            ladder[index + 1]["probes"][slot]["recovered_relative_error"],
        )
        for index in range(len(ladder) - 1)
    ]
    raw_orders = [
        observed_order(
            ladder[index]["probes"][slot]["raw_relative_error"],
            ladder[index + 1]["probes"][slot]["raw_relative_error"],
        )
        for index in range(len(ladder) - 1)
    ]

    evidence = {
        "case": "cook-membrane-q4-stress-recovery",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "tensorfem": __import__("tensorfem").__version__,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "problem": {
            "corners": [[0.0, 0.0], [48.0, 44.0], [48.0, 60.0], [0.0, 44.0]],
            "young": 1.0,
            "poisson": 1.0 / 3.0,
            "thickness": 1.0,
            "loaded_edge_resultant": 1.0,
            "gate": GATE,
        },
        "recovery": {
            "method": "2x2 Gauss (Barlow) stresses -> linear extrapolation to nodes "
            "-> area-weighted nodal averaging -> bilinear interpolation",
            "implementation": "tensorfem.stress_recovery.recover_nodal_stress",
            "production_field": "additional_benchmark_fields.cook_fields()['stress_nodal']",
        },
        "reference": {
            "source": "benchmarks/cook_q4_pointwise_protocol.json (independent Q8, "
            "Richardson over n=12/24/48)",
            "levels": reference["levels"],
            "scale_von_mises": reference["scale"],
            "probes": reference["probes"],
        },
        "ladder": ladder,
        "convergence": {
            "level_ladder": list(LEVELS),
            "max_recovered_relative_error": recovered_max,
            "max_raw_relative_error": raw_max,
            "worst_probe_index": int(slot),
            "recovered_observed_orders": orders,
            "raw_observed_orders": raw_orders,
        },
        "verdict": {
            "recovered": verdict,
            "raw_element_centre": raw_verdict,
            "statement": (
                "production recovered field meets the 3% pointwise gate at n={:d} "
                "({:.4f}%), the stored element-centre field does not ({:.4f}%)".format(
                    finest["divisions"],
                    100.0 * verdict["finest"],
                    100.0 * raw_verdict["finest"],
                )
                if verdict["qualified"]
                else "production recovered field does not meet the 3% gate "
                "(finest {:.4f}%)".format(100.0 * verdict["finest"])
            ),
        },
        "scope": (
            "Linear-elastic plane stress, E=1, nu=1/3, thickness 1, unit resultant shear "
            "on the x=48 edge of the Cook trapezoid; five fixed material points offset "
            "inside element interiors. Certifies the recovered nodal field that "
            "cook_fields now stores; the element-centre field keeps its separate "
            "stress_accuracy_qualified=False. Does not cover geometric nonlinearity, "
            "plasticity, plates/shells, or contact, nor stresses at the clamped corner "
            "or the loaded-edge endpoints."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(evidence, indent=1) + "\n")
    print(json.dumps(evidence["verdict"], indent=1, ensure_ascii=False))
    print("ladder max recovered:", ["%.4f%%" % (100.0 * v) for v in recovered_max])
    print("ladder max raw      :", ["%.4f%%" % (100.0 * v) for v in raw_max])
    print("written:", OUTPUT)
    return 0 if verdict["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())