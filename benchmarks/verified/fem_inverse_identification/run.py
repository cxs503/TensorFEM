#!/usr/bin/env python3
"""Verified benchmark for ``tensorfem.inverse`` (GAP-PP-4-opt-loop).

What is being verified
----------------------
The inverse-identification infrastructure has four moving parts and each one is
checked against an *independent* statement rather than against itself:

===================  =================================================  ============
check                reference (independent)                            tolerance
===================  =================================================  ============
decay tau (3 seeds)  generating time constant 4.0 s                     3 % relative
axial-bar E          generating Young's modulus 210 GPa                3 % relative
axial-bar E+A        data contain only the product E*A: the run must     inverted gate
                     report ``identifiable: False``, effective rank 1
same-seed replay     the JSON-lines convergence log hash must repeat     exact
                     bit-for-bit for one seed, and differ across seeds
===================  =================================================  ============

The axial-bar case is a *solver-in-the-loop* check: every forward evaluation
assembles and solves a ``tensorfem.model.TrussModel`` system, so the optimiser
inverts the finite-element operator and not a closed form.

The decaying-exponential case is the mathematical object of a decaying LBM
velocity signal (single exponential + offset), which is why this module exists
in the JOUR-DIFFLBM inverse-experiment line.

Numbers in the report are read back from the artifacts written here -- nothing
is hand-copied.

Usage
-----
    PYTHONPATH=src python benchmarks/verified/fem_inverse_identification/run.py

Exit status is the gate: 0 only when every row passes, 1 otherwise.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorfem.inverse import OptimizerOptions, SeedControl, run_inversion  # noqa: E402
from tensorfem.inverse_cases import (  # noqa: E402
    BAR_TRUTH,
    DECAY_TIMES,
    DECAY_TRUTH,
    TOLERANCE,
    axial_bar_case,
    decay_case,
    decay_problem,
    product_case,
)

OUT = ROOT / "runs" / "inverse-identification-benchmark"
SEEDS = (1, 2, 3)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover - only when the tree is exported
        return "unknown"


def _relative(path: Path) -> str:
    """Path relative to the repository root when possible, else absolute."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": git_commit(),
    }


def _sha(run) -> str:
    return run.log_sha256()


def decay_rows(provenance: dict) -> list[dict]:
    rows, runs = [], []
    for seed in SEEDS:
        outcome = decay_case(seed=seed, starts=3, provenance=provenance)
        outcome["run"].write(OUT / f"decay-seed{seed}")
        rows.append(outcome["row"])
        runs.append(outcome["run"])
    for row in rows:
        row["reference_quantity"] = "tau [s]"
        row["reference_value"] = DECAY_TRUTH["tau"]
        row["got_value"] = row["tau"]
        row["relative_error"] = row["tau_relative_error"]
    return rows, runs


def bar_row(provenance: dict) -> dict:
    outcome = axial_bar_case(seed=0, provenance=provenance)
    outcome["run"].write(OUT / "axial-bar")
    row = outcome["row"]
    row["reference_quantity"] = "E [Pa]"
    row["reference_value"] = BAR_TRUTH["E"]
    row["got_value"] = row["E"]
    row["relative_error"] = row["E_relative_error"]
    return row


def product_row(provenance: dict) -> dict:
    outcome = product_case(seed=0, provenance=provenance)
    outcome["run"].write(OUT / "axial-bar-product")
    row = outcome["row"]
    row["reference_quantity"] = "E*A non-identifiability"
    row["reference_value"] = 1  # expected effective rank
    row["got_value"] = row["effective_rank"]
    row["relative_error"] = None
    return row


def reproducibility(provenance: dict) -> dict:
    first = run_inversion(
        decay_problem(), OptimizerOptions(), SeedControl(seed=1, starts=3), provenance=provenance
    )
    replay = run_inversion(
        decay_problem(), OptimizerOptions(), SeedControl(seed=1, starts=3), provenance=provenance
    )
    other = run_inversion(
        decay_problem(), OptimizerOptions(), SeedControl(seed=2, starts=3), provenance=provenance
    )
    payload = {
        "seed": 1,
        "log_sha256_first": _sha(first),
        "log_sha256_replay": _sha(replay),
        "log_sha256_seed2": _sha(other),
        "same_seed_identical": _sha(first) == _sha(replay),
        "distinct_seeds_differ": len({_sha(first), _sha(other)}) == 2,
        "tau_first": float(first.best_parameters["tau"]),
        "tau_replay": float(replay.best_parameters["tau"]),
    }
    payload["passed"] = bool(payload["same_seed_identical"] and payload["distinct_seeds_differ"])
    (OUT / "reproducibility.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def make_figure(decay_runs, decay_rows, bar, product, repro, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))

    # panel 1: measured data, truth and the three recovered curves
    ax = axes[0]
    run = decay_runs[0]
    times = DECAY_TIMES.tolist()
    truth = [DECAY_TRUTH["Tinf"] + DECAY_TRUTH["A"] * math.exp(-t / DECAY_TRUTH["tau"]) for t in times]
    ax.plot(times, truth, "k-", lw=1.4, label="truth (tau = 4.0 s)")
    ax.plot(times, run.problem.data.tolist(), "o", ms=3.2, color="#666666", alpha=0.75,
            label="data (1 % noise)")
    for index, seed_run in enumerate(decay_runs):
        params = seed_run.best_parameters
        curve = [params["Tinf"] + params["A"] * math.exp(-t / params["tau"]) for t in times]
        ax.plot(times, curve, "--", lw=1.1, color=("#1f77b4", "#d62728", "#2ca02c")[index],
                label=f"seed {seed_run.seed_control.seed}: tau = {params['tau']:.4f} s")
    ax.set_xlabel("time t [s]")
    ax.set_ylabel("observable T [degC]")
    ax.set_title("decaying exponential: tau recovery (3 seeds)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # panel 2: convergence histories from the machine-readable logs
    ax = axes[1]
    for index, seed_run in enumerate(decay_runs):
        losses = [r["loss"] for r in seed_run.iterations if r["start"] == 0]
        ax.semilogy(range(1, len(losses) + 1), losses, "-", lw=1.4,
                    color=("#1f77b4", "#d62728", "#2ca02c")[index],
                    label=f"decay seed {seed_run.seed_control.seed}")
    ax.set_xlabel("accepted iteration")
    ax.set_ylabel("normalised loss")
    ax.set_title("convergence log replay (loss per iteration)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, which="both")

    # panel 3: reference vs got, relative error against the 3 % gate
    ax = axes[2]
    labels = [f"tau (seed {r['seed']})" for r in decay_rows] + ["E"]
    errors = [100.0 * r["relative_error"] for r in decay_rows] + [100.0 * bar["relative_error"]]
    values = [r["got_value"] / r["reference_value"] for r in decay_rows] + [
        bar["got_value"] / bar["reference_value"]
    ]
    colours = ["#1f77b4"] * len(decay_rows) + ["#9467bd"]
    ax.bar(range(len(values)), [100.0 * (v - 1.0) for v in values], color=colours, alpha=0.85)
    for i, err in enumerate(errors):
        ax.annotate(f"{err:+.3f} %", (i, 100.0 * (values[i] - 1.0)),
                    textcoords="offset points", xytext=(0, 6 if values[i] >= 1 else -14),
                    ha="center", fontsize=7.5)
    ax.axhspan(-100 * TOLERANCE, 100 * TOLERANCE, color="#2ca02c", alpha=0.10,
               label="3 % acceptance band")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("recovered / reference - 1  [%]")
    ax.set_title(
        f"ref vs got (E+A control: rank {product['got_value']}, "
        f"identifiable={product['identifiable']})"
    )
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, axis="y")

    figure.suptitle(
        "tensorfem.inverse verified benchmark: ref vs got, 3 % band, "
        f"same-seed log replay {'reproducible' if repro['same_seed_identical'] else 'NOT reproducible'}",
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = {"script": "benchmarks/verified/fem_inverse_identification/run.py",
                  "git_commit": git_commit()}

    decay, decay_runs = decay_rows(provenance)
    bar = bar_row(provenance)
    product = product_row(provenance)
    repro = reproducibility(provenance)

    rows = decay + [bar]
    for row in rows:
        row["gate"] = f"< {100 * TOLERANCE:.0f} %"
        row["passed"] = bool(row["converged"] and row["relative_error"] < TOLERANCE)
    product["gate"] = "identifiable == False and effective_rank == 1 and |E*A error| < 3 %"
    product["relative_error"] = product["product_relative_error"]
    product["reference_value"] = product["product_reference"]
    product["got_value"] = product["product"]
    product["reference_quantity"] = "E*A (product) with E,A both unknown"
    product["passed"] = bool(product["passed"] and product["product_relative_error"] < TOLERANCE)

    figure_path = OUT / "inverse-identification-benchmark.png"
    make_figure(decay_runs, decay, bar, product, repro, figure_path)

    passed = all(r["passed"] for r in rows) and product["passed"] and repro["passed"]
    payload = {
        "schema": "tensorfem.inverse-benchmark/1",
        "scope": "GAP-PP-4-opt-loop tensorfem.inverse",
        "tolerance": TOLERANCE,
        "environment": environment(),
        "rows": rows,
        "negative_control": product,
        "reproducibility": repro,
        "passed": passed,
        "figure": str(_relative(figure_path)),
        "repro_cmd": "PYTHONPATH=src python benchmarks/verified/fem_inverse_identification/run.py",
    }
    (OUT / "benchmark.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                        encoding="utf-8")

    header = f"{'check':34s} {'reference':>16s} {'got':>16s} {'rel.err':>9s}  verdict"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['case'] + ' s' + str(row['seed']):34s} "
            f"{row['reference_value']:16.8g} {row['got_value']:16.8g} "
            f"{100 * row['relative_error']:8.4f}%  {'PASS' if row['passed'] else 'FAIL'}"
        )
    print(
        f"{'axial-bar-product (rank/identifiable)':34s} "
        f"{'1 / False':>16s} "
        f"{str(product['effective_rank']) + ' / ' + str(product['identifiable']):>16s} "
        f"{100 * product['product_relative_error']:8.4f}%  "
        f"{'PASS' if product['passed'] else 'FAIL'}"
    )
    print(
        f"{'same-seed log replay':34s} {'identical':>16s} "
        f"{str(repro['same_seed_identical']):>16s} {'-':>9s}  "
        f"{'PASS' if repro['passed'] else 'FAIL'}"
    )
    print("-" * len(header))
    print(f"artifact: {OUT / 'benchmark.json'}")
    print(f"figure  : {figure_path}")
    print(f"GATE: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())