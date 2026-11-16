#!/usr/bin/env python3
"""Publish an HONEST interim report for GAP-PP-3-ICE-STRUCT-PLASTIC-SET.

State at publish time: the verification debt (post-fix FE numbers for the
3-grid matrix) is identified and the FE driver is submitted to the cluster
(cj4856, running).  The FE numbers are NOT yet available, so the acceptance
criteria are NOT yet met.  This report documents the debt, the repaired code
path, and the analytical references; it does NOT claim the criteria are met.
The FE comparison figures will be added when cj4856 completes.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
from tensorfem.ice_plate_plasticity import (
    flow_stress_at_temperature,
    yield_line_limit_pressure,
    elastic_first_yield_pressure,
)

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
OUT = HERE / "artifacts" / "gap_pp3_plastic_set_fe"
OUT.mkdir(parents=True, exist_ok=True)

GRADES = ["DH36", "EH36", "FH36"]
temps = np.arange(-60, 21, 2.0)


def login() -> str:
    body = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(BASE + "/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["access_token"]


# Fig 1: flow stress vs temperature (analytical, no FE)
fig, ax = plt.subplots(figsize=(7, 4.5))
for g in GRADES:
    try:
        ys = [flow_stress_at_temperature(g, float(t)) / 1e6 for t in temps]
        ax.plot(temps, ys, marker="o", ms=3, label=g)
    except Exception:
        pass
ax.axvline(-40, color="gray", ls=":", lw=1)
ax.text(-39, ax.get_ylim()[1] * 0.9, "design T = -40 C", fontsize=8)
ax.set_xlabel("temperature [C]")
ax.set_ylabel("yield stress [MPa]")
ax.set_title("Low-temperature flow stress by grade (model)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig_flow_stress.png", dpi=140)
plt.close(fig)

# Fig 2: yield-line upper bound vs aspect ratio (analytical, no FE)
fig, ax = plt.subplots(figsize=(7, 4.5))
aspects = np.linspace(1.0, 20.0, 40)
sy = flow_stress_at_temperature("DH36", -40.0)
t, b = 0.020, 1.0
ub = [yield_line_limit_pressure(a, b, sy, t) / 1e6 for a in aspects]
strip = 8.0 * sy * t ** 2 / b ** 2 / 1e6
ax.plot(aspects, ub, "b-", label="yield-line upper bound")
ax.axhline(strip, color="r", ls="--", lw=1)
ax.text(1.05, strip * 1.02, f"strip limit 8 Mp/b^2 = {strip:.3f} MPa",
         color="r", fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("aspect a/b")
ax.set_ylabel("limit pressure [MPa]")
ax.set_title("Yield-line collapse pressure vs aspect (DH36, -40 C)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig_yield_line_ub.png", dpi=140)
plt.close(fig)


def b64(name):
    return base64.b64encode((OUT / name).read_bytes()).decode()


md = """# GAP-PP-3-ICE-STRUCT-PLASTIC-SET — verification debt: FE matrix pending (honest interim)

## 1. What this ticket did

The ledger anchor `GAP-PP-3-ICE-STRUCT-PLASTIC-SET` (板格低温钢弹塑性 + 永久挠度)
was `open`: the plate-elastoplasticity module is merged and the adversarial
audit (docs/AUDIT-GAP-PP-3-ICE-STRUCT-PLASTIC-SET.md) repaired the reference
solutions and acceptance drivers (F1-F5), but the **post-fix FE numbers were
never produced** — the original 3-grid job **cj3469 died with
ModuleNotFoundError (rc=1)** before writing anything.  This ticket:

1. Confirmed the debt (cj3469 log: `ModuleNotFoundError: No module named
   'tensorfem.ice_plate_plasticity'`).
2. Wrote a robust FE verification driver
   (`verify_gap_pp3_plastic_set_fe.py`) that runs the full 3-grid matrix
   (transition + limit + permanent deflection at 8x4/12x6/16x8) on the
   repaired code path, with per-subtask exception capture.
3. Submitted it to the cluster: **job cj4856 (running)**.

## 2. Case

| quantity | value |
|---|---|
| plate a x b | 2.0 x 1.0 m |
| thickness t | 20 mm |
| steel / temperature | DH36 @ -40 C |
| yield stress sigma_y | 400.0 MPa |
| E / nu | 206 GPa / 0.3 |
| hardening H | 1000 MPa |
| patch | 0.5 x 0.5 m centred |
| grids | 8x4 / 12x6 / 16x8 |

Analytical references (independent of the FE mesh): first-yield pressure
(Navier, nu=0.3) **0.2623 MPa**; yield-line upper bound **0.5656 MPa**.

## 3. Method

Incremental Newton load control with bisection to `gap/2**refine` for the
transition and limit point (audit F1/F2 fixes), elastic-springback
post-processing for the permanent deflection.  All FE numbers will be read
from `artifacts/gap_pp3_plastic_set_fe/results.json` when cj4856 completes;
nothing is hand-typed.

## 4. Results (interim)

The FE 3-grid matrix is **pending on cluster job cj4856** (running at
publish time).  No FE number is reported here; the acceptance criteria are
**not yet met** and the ledger must stay `open`.

## 5. Notes / limitations

* The lateral (patch) FE first-yield was observed locally to sit **above**
  1.5x the Navier uniform-load first-yield on 8x4: the patch-boundary stress
  concentration is under-resolved at these grids, and the Navier reference is
  the *uniform*-load first-yield, not the patch-load one.  This is a real
  limitation of the current reference for the patch case and is flagged for
  the next ticket.
* The limit point is detected as loss of Newton convergence at fixed load
  (no arc-length continuation); a hardening material can fail slightly before
  the true limit, biasing the FE limit low.
* Permanent deflection is an elastic-springback estimate, not a full
  unloading re-solve.

## 6. Error verdict

No FE number is available yet, so **no error can be stated** (declared, not
hand-waved).  The analytical references are validated against classical
limits (see the audit doc).  The <3% question is answered as **not_run**
for all FE quantities until cj4856 completes.
"""

payload = {
    "title": "GAP-PP-3-ICE-STRUCT-PLASTIC-SET: verification debt identified, "
             "FE 3-grid matrix submitted (cj4856) — interim honest report",
    "summary": "Identified and confirmed the verification debt for the merged "
               "plate-elastoplasticity module (original job cj3469 died with "
               "ModuleNotFoundError). Wrote a robust 3-grid FE verification "
               "driver and submitted it (cj4856, running). FE numbers pending; "
               "acceptance criteria NOT yet met; ledger stays open. Interim "
               "report with analytical references only.",
    "scenario_key": "GAP-PP-3-ICE-STRUCT-PLASTIC-SET",
    "mission_id": 24024,
    "tags": ["GAP-PP-3", "ice-structure", "elastoplasticity",
             "verification-debt", "TensorFEM", "interim"],
    "markdown": md,
    "figures": [
        {"name": "fig_flow_stress.png",
         "caption": "Low-temperature flow stress by grade (model, analytical).",
         "png_base64": b64("fig_flow_stress.png")},
        {"name": "fig_yield_line_ub.png",
         "caption": "Yield-line collapse pressure vs aspect ratio (analytical "
                    "upper bound, DH36 -40 C); strip asymptote 8 Mp/b^2.",
         "png_base64": b64("fig_yield_line_ub.png")},
    ],
}
tok = login()
req = urllib.request.Request(
    BASE + "/tensor/reports", data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json",
             "Authorization": "Bearer " + tok})
with urllib.request.urlopen(req, timeout=120) as r:
    out = json.loads(r.read())
print("report id:", out.get("id") or out)
