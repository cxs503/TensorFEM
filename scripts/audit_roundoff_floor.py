#!/usr/bin/env python3
"""Audit the round-off floor of the layered Shell4 stress-free reference.

The regression test for the finite-rotation layered shell asserts that a
consistently warped patch is stress-free.  The assertion used to be written
against the absolute constant ``1e-10`` and was therefore recorded as machine
dependent, because the attainable floor of an assembled quantity is
``ops * eps * scale`` (see :mod:`tensorfem.roundoff_scale`).

This script turns that anecdote into evidence:

* it measures the floor on the warped patch for several coordinate origins
  (translation of the mesh must not change a consistent strain measure, so any
  drift is round-off, not physics);
* it reports the amplification ``floor / (eps * scale)`` so the operation
  budget in ``DEFAULT_OPS`` can be checked against measurement;
* it reports the margin between the objective bound and both the measured
  floor and a physically meaningful stress signal, showing that the bound is
  neither platform dependent nor vacuous;
* it writes ``artifacts/layered-shell4-roundoff-audit.json`` and a figure.

Usage::

    python scripts/audit_roundoff_floor.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from tensorfem.finite_rotation_layered_shell4 import assemble_finite_rotation_layered_shell4
from tensorfem.layered_shell4_plasticity import LayeredShell4Model, LayeredShell4State
from tensorfem.roundoff_scale import DEFAULT_OPS, machine_epsilon, roundoff_bound

ROOT = Path(__file__).resolve().parents[1]
WARP = torch.tensor(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.012], [1.0, 1.0, -0.007], [0.0, 1.0, 0.004]],
    dtype=torch.float64,
)
YOUNG, POISSON, THICKNESS = 200000.0, 0.3, 0.08
ELEMENTS = torch.tensor([[0, 1, 2, 3]])
ORIGINS = (0.0, 1e2, 1e4, 1e6, 1e8)
#: A once-percent strain is the smallest deformation any real consistency bug
#: would leave behind; its stress is the "signal" the bound must stay below.
SIGNAL_STRAIN = 1e-2


def model(origin: float, dtype=torch.float64) -> LayeredShell4Model:
    nodes = (WARP + origin).to(dtype)
    return LayeredShell4Model(
        nodes, ELEMENTS, YOUNG, POISSON, THICKNESS, 250.0, 1400.0, layers=3
    )


def measure(origin: float, dtype=torch.float64) -> dict:
    shell = model(origin, dtype)
    zero = torch.zeros(shell.n_dofs, dtype=dtype)
    stress_free = assemble_finite_rotation_layered_shell4(
        shell, zero, LayeredShell4State.virgin(shell), tangent=False
    )
    force = float(stress_free.internal_force.abs().max())
    stress = float(stress_free.stress.abs().max())
    # In-plane stretch of SIGNAL_STRAIN: stress scale of a genuine inconsistency.
    stretch = zero.clone()
    for node, X in enumerate(shell.nodes):
        stretch[6 * node] = SIGNAL_STRAIN * float(X[0])
        stretch[6 * node + 1] = SIGNAL_STRAIN * float(X[1])
    stretched = assemble_finite_rotation_layered_shell4(
        shell, stretch, LayeredShell4State.virgin(shell), tangent=False
    )
    signal_stress = float(stretched.stress.abs().max())
    eps = machine_epsilon(dtype)
    return {
        "origin": origin,
        "dtype": str(dtype).replace("torch.", ""),
        "eps": eps,
        "stress_floor": stress,
        "force_floor": force,
        "stress_signal": signal_stress,
        "stress_amplification": stress / (eps * YOUNG),
        "force_amplification": force / (eps * YOUNG * THICKNESS * 1.0),
        "stress_bound": roundoff_bound(YOUNG, dtype),
        "force_bound": roundoff_bound(YOUNG * THICKNESS, dtype),
        "stress_ratio": stress / roundoff_bound(YOUNG, dtype),
        "force_ratio": force / roundoff_bound(YOUNG * THICKNESS, dtype),
        "signal_over_bound": signal_stress / roundoff_bound(YOUNG, dtype),
    }


def figure(records, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    origins = [max(r["origin"], 1e0) for r in records]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    ax.loglog(origins, [r["stress_floor"] for r in records], "o-", label="measured stress floor")
    ax.loglog(
        origins,
        [r["stress_bound"] for r in records],
        "s--",
        label=f"objective bound ({DEFAULT_OPS} eps E)",
    )
    ax.loglog(
        origins,
        [r["stress_signal"] for r in records],
        "^:",
        label=f"signal: {SIGNAL_STRAIN:.0e} strain",
    )
    ax.axhline(1e-10, color="0.4", lw=1.0, ls="-.", label="retired absolute 1e-10")
    ax.set_xlabel("coordinate origin offset [m]")
    ax.set_ylabel("max |stress| [Pa]")
    ax.set_title("Layered Shell4 stress-free floor vs. frame origin")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    labels = [f"{r['origin']:.0e}" for r in records]
    x = range(len(records))
    ax.bar([i - 0.2 for i in x], [r["stress_ratio"] for r in records], width=0.4,
           label="floor / bound")
    ax.bar([i + 0.2 for i in x], [r["signal_over_bound"] for r in records], width=0.4,
           label="signal / bound")
    ax.set_yscale("log")
    ax.set_xticks(list(x), labels)
    ax.set_xlabel("coordinate origin offset [m]")
    ax.set_ylabel("ratio [-]")
    ax.set_title("Bound is tight for round-off and loose for physics")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main() -> int:
    records = [measure(origin) for origin in ORIGINS]
    payload = {
        "schema": "tensorfem.roundoff-audit/1",
        "case": "finite-rotation layered Shell4, single warped patch, zero displacement",
        "model": {"young": YOUNG, "poisson": POISSON, "thickness": THICKNESS,
                  "warp_amplitudes": [0.012, -0.007, 0.004]},
        "criterion": {"ops": DEFAULT_OPS, "dtype": "float64",
                      "stress_scale": "E", "force_scale": "E*t",
                      "rule": "tol = ops * eps * scale"},
        "records": records,
        "summary": {
            "max_stress_amplification": max(r["stress_amplification"] for r in records),
            "max_force_amplification": max(r["force_amplification"] for r in records),
            "max_stress_ratio": max(r["stress_ratio"] for r in records),
            "min_signal_over_bound": min(r["signal_over_bound"] for r in records),
            "retired_absolute_threshold": 1e-10,
            "retired_threshold_inside_noise_band": any(
                r["stress_floor"] > 5e-11 for r in records
            ),
        },
    }
    payload["summary"]["amplification_margin"] = (
        DEFAULT_OPS / payload["summary"]["max_stress_amplification"]
    )
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    (out / "layered-shell4-roundoff-audit.json").write_text(json.dumps(payload, indent=2))
    figure(records, ROOT / "docs/assets/roundoff_floor/layered_shell4_roundoff_floor.png")
    print(json.dumps(payload["summary"], indent=2))
    for r in records:
        print(
            f"origin={r['origin']:8.0e}  floor={r['stress_floor']:.3e} Pa"
            f"  bound={r['stress_bound']:.3e} Pa  ratio={r['stress_ratio']:.3f}"
            f"  signal/bound={r['signal_over_bound']:.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())