"""Judge rows for JOUR-DIFFLBM-L3-INVERSION, FEM instantiation (ref vs got).

Why this file exists
--------------------
The anchor's acceptance is a *number*, not a narrative: <= 10 % of the nodes
observed, the recovered volume-force field within 5 % of the analytic truth,
<= 50 adjoint iterations, and the convergence curve archived.  A doc claim is
not evidence, so this driver recomputes every quantity, pairs each with the
independent reference it is measured against, and writes the comparison as
JSON rows in ``artifacts/body_force_inversion/benchmark_rows.json``.  The rows
are what gets registered in ``tensor_benchmark_cases`` and what the case doc
quotes -- no number is ever re-typed by hand.

Honest rows
-----------
Two rows are *expected* to fail the 5 % band (1 % sensor noise, and the
full-nodal rank-deficient control).  They are kept as ``status="failed"`` or
``status="flagged"`` rows rather than dropped: a benchmark suite that only
lists what passes cannot tell a reader where the method stops working.

Usage
-----
``PYTHONPATH=src python3 benchmarks/body_force_inversion_benchmark.py``
(~60 s on one core; writes the JSON rows, exit 0 when every *claim* row is
PASS and every *honesty* row is correctly flagged).
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorfem.body_force_inversion import (  # noqa: E402
    InversionOptions,
    ModalBasis,
    PlaneStressMesh,
    run_body_force_inversion,
)

OUT = ROOT / "artifacts" / "body_force_inversion"
REPRO = "PYTHONPATH=src python3 benchmarks/body_force_inversion_benchmark.py"

MESH = PlaneStressMesh(Lx=1.0, Ly=0.25, nx=20, ny=10)
TRUTH = [1.0, 0.5, -0.4, 0.3, 0.6, -0.7, 0.25, 0.4]


def _driver():
    """Import the artifact driver so the finite-element forward checks are shared."""
    spec = importlib.util.spec_from_file_location(
        "body_force_inversion_driver", ROOT / "scripts" / "run_body_force_inversion.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(**kwargs) -> dict:
    row = {
        "anchor_slug": "JOUR-DIFFLBM-L3-INVERSION",
        "repo": "TensorFEM",
        "repro_cmd": REPRO,
    }
    row.update(kwargs)
    return row


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    driver = _driver()
    t0 = time.time()

    # -- R1: the anchor claim itself (recovery vs analytic modal truth) ---- #
    acc = run_body_force_inversion(
        mesh=MESH,
        truth_coeffs=TRUTH,
        order=2,
        fraction=0.10,
        options=InversionOptions(alpha_rel=1.0e-12),
    )
    coeff_dev = max(
        abs(a - b)
        for a, b in zip(acc["coefficients"]["recovered"], acc["coefficients"]["truth"])
    )
    r1 = _row(
        case_name="sparse-observation body-force recovery: 20x10 Q4 plane stress, "
        "order-2 modal basis (8 unknowns), 10 % of nodes observed",
        ref_source="analytic modal truth b(x,y)=sum c_mn sin(m pi x/Lx) sin(n pi y/Ly) with "
        "c=[1.0,0.5,-0.4,0.3,0.6,-0.7,0.25,0.4] (closed form, same basis as the inversion); "
        "reference field error is identically 0 by construction",
        metric="relative L2 error of the recovered field vs the analytic truth; "
        "max |recovered - true| coefficient deviation; adjoint iteration count",
        tolerance="field rel L2 < 5 %, <= 50 adjoint iterations, observed fraction <= 0.10",
        ref={"field_rel_l2": 0.0, "coefficients": TRUTH},
        got={
            "field_rel_l2": acc["field_rel_l2"],
            "observed_fraction": acc["observations"]["fraction"],
            "n_node_sampled": acc["observations"]["n_node_sampled"],
            "n_observation": acc["observations"]["n_observation"],
            "adjoint_iterations": acc["iterations"],
            "stop_reason": acc["stop_reason"],
            "max_coefficient_deviation": coeff_dev,
            "relative_data_residual": acc["relative_data_residual"],
        },
        status="passed" if all(acc["acceptance"].values()) else "failed",
    )

    # -- R2: the forward operator must be right before the inverse means anything
    patch = driver._patch_errors()
    patch_max = max(patch.values())
    r2 = _row(
        case_name="Q4 plane-stress constant-strain patch: whole-boundary Dirichlet, "
        "uniform uniaxial + pure shear (+ linear bending at nu=0), nu=0/0.3/0.49",
        ref_source="exact bilinear displacement field reproduced by a constant-strain Q4 "
        "element (patch-test identity; the field, not a solver, is the reference)",
        metric="max relative L2 displacement error over 7 states",
        tolerance="< 1e-12 (machine precision)",
        ref={"max_relative_error": 0.0},
        got={"max_relative_error": patch_max, "states": patch},
        status="passed" if patch_max < 1e-12 else "failed",
    )

    # -- R3: convergence order under a manufactured trigonometric load ------ #
    man = driver._manufactured_errors()
    rates = man["observed_rates"]
    r3 = _row(
        case_name="Q4 manufactured solution: analytic trigonometric body force, "
        "meshes 16x4 -> 32x8 -> 64x16, interior-node relative L2 error",
        ref_source="theoretical convergence order 2.0 of bilinear Q4 in the L2 norm "
        "(and the analytic displacement field itself)",
        metric="observed log2 error-reduction rate between successive meshes",
        tolerance="each rate >= 1.9 and finest-mesh interior error < 1e-3",
        ref={"order": 2.0},
        got={
            "observed_rates": [round(r, 4) for r in rates],
            "finest_interior_rel_l2": man["rows"][-1]["rel_l2_interior"],
            "rows": man["rows"],
        },
        status="passed"
        if min(rates) >= 1.9 and man["rows"][-1]["rel_l2_interior"] < 1e-3
        else "failed",
    )

    # -- R4: honest FAIL -- 1 % sensor noise cannot hold the 5 % band ------- #
    noisy = run_body_force_inversion(
        mesh=MESH,
        truth_coeffs=TRUTH,
        order=2,
        fraction=0.10,
        options=InversionOptions(alpha_rel=1.0e-12, noise=0.01),
    )
    r4 = _row(
        case_name="same case with 1 % relative sensor noise on the sampled displacements",
        ref_source="same analytic modal truth; the noise realisation is seeded "
        "(options.seed=0 -> generator seed 977), so the row reproduces bit-for-bit",
        metric="relative L2 field error",
        tolerance="< 5 % (the anchor band) -- EXPECTED TO FAIL, recorded on purpose",
        ref={"field_rel_l2": 0.0},
        got={
            "field_rel_l2": noisy["field_rel_l2"],
            "iterations": noisy["iterations"],
            "acceptance": noisy["acceptance"],
        },
        status="failed",
        note="honest FAIL: the observation operator's condition number is ~2.9e3, so a "
        "1 % sensor error is amplified well past 5 %; damping must match the noise level "
        "(see noise_sweep.json).",
    )

    # -- R5/R6: the controls must be flagged, never reported as recoveries -- #
    nodal_mesh = PlaneStressMesh(1.0, 0.25, 12, 6)
    basis_n = ModalBasis(mesh=nodal_mesh, order=2)
    c_true = torch.tensor(TRUTH, dtype=torch.float64)
    nodal_truth = [float(v) for v in basis_n.nodal_field(c_true, 0)] + [
        float(v) for v in basis_n.nodal_field(c_true, 1)
    ]
    nodal = run_body_force_inversion(
        mesh=nodal_mesh,
        truth_coeffs=nodal_truth,
        mode="nodal",
        fraction=0.10,
        options=InversionOptions(alpha_rel=1.0e-8),
    )
    r5 = _row(
        case_name="negative control: full-nodal unknowns (462), same 10 % sampling",
        ref_source="no honest reference exists: the sampling operator is rank deficient by "
        "construction, so any 'recovered' nodal field is operator dependent",
        metric="reported null_space_dim and field error; the pass condition is that the run "
        "flags itself",
        tolerance="null_space_dim > 0 and field_within_5pct == false (must not claim success)",
        ref={"field_rel_l2": None, "null_space_dim": "> 0"},
        got={
            "field_rel_l2": nodal["field_rel_l2"],
            "null_space_dim": nodal["null_space_dim"],
            "acceptance": nodal["acceptance"],
        },
        status="flagged"
        if nodal["null_space_dim"] and not nodal["acceptance"]["field_within_5pct"]
        else "failed",
    )

    # Same 0.6 amplitude as the artifact driver: the out-of-basis content has to be
    # *material* (it is ~10 % of the in-basis RMS), otherwise a 5 % band would pass
    # and the control would prove nothing.
    def _extra(x, y):
        return 0.6 * torch.sin(5.0 * math.pi * x / MESH.Lx) * torch.sin(
            3.0 * math.pi * y / MESH.Ly
        )

    oos = run_body_force_inversion(
        mesh=MESH,
        truth_coeffs=TRUTH,
        order=2,
        fraction=0.10,
        truth_extra=_extra,
        options=InversionOptions(alpha_rel=1.0e-12),
    )
    r6 = _row(
        case_name="negative control: truth contains content outside the inversion basis "
        "(5 pi x, 3 pi y mode) -- pure model error",
        ref_source="the in-basis part is exactly recoverable; the out-of-basis part is "
        "unrepresentable, so the residual error IS the model error",
        metric="reported out_of_span flag and field error",
        tolerance="out_of_span == true and field_within_5pct == false",
        ref={"field_rel_l2": None, "out_of_span": True},
        got={
            "field_rel_l2": oos["field_rel_l2"],
            "out_of_span": oos["out_of_span"],
            "acceptance": oos["acceptance"],
        },
        status="flagged"
        if oos["out_of_span"] and not oos["acceptance"]["field_within_5pct"]
        else "failed",
    )

    rows = [r1, r2, r3, r4, r5, r6]
    payload = {
        "schema": "tensorfem.body-force-inversion-benchmark/1",
        "anchor_slug": "JOUR-DIFFLBM-L3-INVERSION",
        "repo": "TensorFEM",
        "generated_at_unix": int(time.time()),
        "wall_clock_s": round(time.time() - t0, 2),
        "repro_cmd": REPRO,
        "rows": rows,
    }
    target = OUT / "benchmark_rows.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"{'case':<58} {'ref':>14} {'got':>14}  status")
    for row in rows:
        name = row["case_name"].split(":")[0][:58]
        ref = row["ref"].get("field_rel_l2", row["ref"].get("max_relative_error", ""))
        got = row["got"].get("field_rel_l2", row["got"].get("max_relative_error", ""))
        ref_s = "-" if ref is None else f"{ref:.3e}" if isinstance(ref, float) else str(ref)
        got_s = "-" if got is None else f"{got:.3e}" if isinstance(got, float) else str(got)
        print(f"{name:<58} {ref_s:>14} {got_s:>14}  {row['status']}")
    print(f"\nrows -> {target}  ({payload['wall_clock_s']} s)")
    claim_ok = all(row["status"] in {"passed", "flagged"} for row in rows)
    return 0 if claim_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())