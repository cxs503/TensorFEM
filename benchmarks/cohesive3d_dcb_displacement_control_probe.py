#!/usr/bin/env python
"""Fast, deterministic probe of the 3-D DCB displacement-controlled path follower.

Why this file exists
-------------------
The width-resolved benchmark ``benchmarks/cohesive3d_dcb_3d.py`` is too slow to
run on the contended control node.  This probe exercises the *same* production
solver (``tensorfem.cohesive3d_dcb.solve_dcb``) on a small mesh and writes the
numbers that decide whether the specimen-level acceptance case for
feature-ledger anchor *GAP-FEM-3D-COHESIVE* can be trusted:

* does the follower terminate exactly at ``delta_max`` (no zero-length spin),
* is the reaction load non-zero and linear in the imposed opening,
* how far is the measured compliance from beam theory with / without the
  Williams root rotation,

and it records the answer honestly, including when the criterion fails.

Run: ``python benchmarks/cohesive3d_dcb_displacement_control_probe.py``
Writes: ``benchmarks/cohesive3d_dcb_displacement_control_probe.json``
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.cohesive3d import BKCohesive3DLaw  # noqa: E402
from tensorfem.cohesive3d_dcb import (  # noqa: E402
    DCBGeometry,
    beam_compliance,
    build_dcb_model,
    solve_dcb,
)

E_ADH, NU_ADH, H_ARM = 70.0e9, 0.30, 3.0e-3
TOUGHNESS = 300.0


def law() -> BKCohesive3DLaw:
    return BKCohesive3DLaw(1e12, 4e11, 5e6, 3e6, TOUGHNESS, 800.0, 2.2)


def main() -> int:
    torch.set_num_threads(1)
    t0 = time.time()
    g = DCBGeometry(length=60e-3, width=3e-3, arm_thickness=H_ARM,
                    initial_crack=30e-3, nx=24, ny=1, nz=2)
    model = build_dcb_model(g, law())
    delta_max = 2.0e-5
    result = solve_dcb(model, delta_max=delta_max, n_steps=8, tol=1e-8)

    c_fem = result.stats["reach_delta"] / result.peak["load"]
    c_plain = beam_compliance(g.initial_crack, g, root_rotation=False, shear=True)
    c_root = beam_compliance(g.initial_crack, g, root_rotation=True, shear=True)
    out = {
        "schema": "tensorfem.cohesive3d-dcb-probe/1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "benchmarks/cohesive3d_dcb_displacement_control_probe.py",
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "python": platform.python_version(), "torch": torch.__version__},
        "mesh": {"nx": g.nx, "ny": g.ny, "nz": g.nz, "n_dofs": model.n_dofs,
                 "n_iface_elements": model.n_iface,
                 "n_prescribed_load_dofs": int(model.upper_load_dofs.size
                                               + model.lower_load_dofs.size)},
        "termination": {
            "delta_max_m": delta_max,
            "reach_delta_m": result.stats["reach_delta"],
            "converged_all": result.stats["converged_all"],
            "committed_steps": result.stats["steps"],
            "newton_iterations": result.stats["newton_iterations"],
            "terminated_at_target": abs(result.stats["reach_delta"] - delta_max)
            <= 1e-9 * delta_max,
        },
        "reaction": {
            "peak_load_N": result.peak["load"],
            "max_committed_damage": result.peak["max_damage"],
            "load_is_nonzero": result.peak["load"] > 0.0,
        },
        "compliance": {
            "fem_m_per_N": c_fem,
            "beam_plain_m_per_N": c_plain,
            "beam_williams_root_rotation_m_per_N": c_root,
            "ratio_fem_over_beam_plain": c_fem / c_plain,
            "ratio_fem_over_beam_root_rotation": c_fem / c_root,
            "ref": c_root,
            "got": c_fem,
            "rel_error": (c_fem - c_root) / c_root,
            "tolerance": "|rel_error| <= 0.10 vs root-rotation beam theory",
            "passed": abs((c_fem - c_root) / c_root) <= 0.10,
        },
        "runtime_s": time.time() - t0,
    }
    dest = ROOT / "benchmarks" / "cohesive3d_dcb_displacement_control_probe.json"
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("termination", "reaction")}, indent=1))
    print(json.dumps(out["compliance"], indent=1))
    print(f"runtime={out['runtime_s']:.1f}s -> {dest}")
    return 0 if out["compliance"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
