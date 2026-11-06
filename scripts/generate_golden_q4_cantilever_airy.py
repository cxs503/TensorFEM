"""Generate the golden reference for the Q4 cantilever vs exact Airy field.

The golden file ``tests/golden/q4_cantilever_airy.json`` locks the *solver
behaviour* of the rectangular Q4 plane-stress cantilever (the scenario of
``tests/test_q4_cantilever_airy.py``): the same problem, mesh and boundary
treatment are re-solved and every recorded number must reproduce.  Accuracy
against the closed-form Timoshenko & Goodier solution is already gated by
``tests/test_q4_cantilever_airy.py``; this golden additionally catches silent
regressions in the solver, the mesh generator or the exact-field module that
leave the published error metrics unchanged in shape but shift their values.

The data is produced by the same ``solve``/``errors`` routines the published
audit uses (``scripts/audit_q4_cantilever_airy.py``), so the golden is a
byte-stable digest of a real solve, not a hand transcription.  It records

* the summary error metrics of the published audit (``data``), and
* the raw fields (``field``): FE and exact nodal displacements, FE and exact
  element-centre stresses, and the element-centre coordinates.

Run from the repository root::

    PYTHONPATH=src python3 scripts/generate_golden_q4_cantilever_airy.py
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_q4_cantilever_airy import errors, solve  # noqa: E402
from tensorfem.airy_cantilever import AiryCantilever  # noqa: E402

GOLDEN = ROOT / "tests" / "golden" / "q4_cantilever_airy.json"
#: Middle mesh of the published (20,4)/(40,8)/(80,16) study: fine enough that
#: discretisation error is small, cheap enough that the gate stays fast.
MESH = (40, 8)
SCHEMA = "tensorfem.golden/q4-cantilever-airy/1.0"


def _flat(tensor: torch.Tensor) -> list[float]:
    return [float(v) for v in tensor.reshape(-1)]


def build_golden() -> dict:
    model = AiryCantilever()
    nodes, elements, result = solve(MESH, model)
    centres = nodes[elements].mean(1)
    field = {
        "element_centres": _flat(centres),
        "node_displacement_fe": _flat(result.displacement),
        "node_displacement_exact": _flat(model.displacement(nodes)),
        "element_stress_fe": _flat(result.stress),
        "element_stress_exact": _flat(model.stress(centres)),
    }
    return {
        "schema": SCHEMA,
        "case_id": "rectangular-q4-cantilever-airy",
        "reference": "Timoshenko & Goodier, Theory of Elasticity, Art. 24",
        "generated": {
            "command": "PYTHONPATH=src python3 scripts/generate_golden_q4_cantilever_airy.py",
            "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "machine": platform.platform(),
        },
        "problem": {
            "length_m": model.length,
            "height_m": model.height,
            "thickness_m": model.thickness,
            "young_modulus_pa": model.young_modulus,
            "poisson": model.poisson,
            "end_shear_n": model.end_shear,
            "second_moment_m4": model.second_moment,
        },
        "mesh": {"nx": MESH[0], "ny": MESH[1]},
        "data": errors(MESH, model),
        "fields": field,
    }


def main() -> None:
    golden = build_golden()
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {GOLDEN.relative_to(ROOT)} "
        f"({GOLDEN.stat().st_size} bytes, mesh {MESH[0]}x{MESH[1]})"
    )


if __name__ == "__main__":
    main()
