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

Certificate
-----------
Besides the raw dump under ``runs/inverse-identification-benchmark/``, the driver
emits the canonical anchor certificate ``gates_verified.json`` (schema
``tensorfem.verified-gates.v1``) next to this file, so
``tools/verified_anchor_audit.py`` can recompute every verdict from the numbers
in the artifact alone and ``tests/test_verified_anchor_certificates.py`` can hold
the registry fail-closed.  ``result.json`` is the machine-readable dump the
certificate is derived from; ``--from-dump`` rebuilds the certificate from it
without touching the solver.

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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from _harness import Gate, gate_manifest, read_json, write_json  # noqa: E402

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

#: artifact directory of the anchor: everything the certificate declares lives
#: here and is committed, so the audit can check existence and git tracking.
#: Resolved at call time (``_artifact``) so a test can redirect the whole
#: artifact set to a temporary directory without the run leaking into the tree.
ARTIFACT_DIR = HERE


def _artifact(name: str) -> Path:
    return ARTIFACT_DIR / name


def artifact_paths() -> dict[str, Path]:
    return {
        "result": _artifact("result.json"),
        "gates": _artifact("gates_verified.json"),
        "figure": _artifact("inverse_identification_recovery.png"),
    }

#: thresholds declared *before* the run (3 % is the platform standard for a
#: recovered scalar; 1e-6 is the loose bound on multi-start spread).
CRITERIA = {
    "scalar_rel_error_max": TOLERANCE,
    "product_rel_error_max": TOLERANCE,
    "start_spread_rel_max": 1.0e-6,
    "nonconverged_cases_max": 0.0,
    "log_replay_identical_min": 1.0,
}


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


def build_gates(payload: dict) -> list[Gate]:
    """Grade the artifact against the pre-declared criteria.

    Every ``value`` is read from ``payload`` (the dump written by the driver or
    by ``--from-dump``); the ``reference`` of each gate names the independent
    statement it is graded against, never the model's own output again.
    """
    rows = payload["rows"]
    product = payload["negative_control"]
    repro = payload["reproducibility"]
    gates: list[Gate] = []

    for row in rows:
        if row["case"] == "decay":
            gates.append(
                Gate(
                    id=f"tau_rel_error_seed{row['seed']}",
                    quantity=f"|tau - tau_true| / tau_true, decay case, seed {row['seed']}",
                    reference=f"generating time constant tau_true = {DECAY_TRUTH['tau']} s",
                    value=float(row["tau_relative_error"]),
                    threshold=CRITERIA["scalar_rel_error_max"],
                    note=f"tau = {row['tau']:.10g} s",
                )
            )
        else:
            gates.append(
                Gate(
                    id="axial_bar_E_rel_error",
                    quantity="|E - E_true| / E_true, solver-in-the-loop axial bar",
                    reference=f"generating Young's modulus E_true = {BAR_TRUTH['E']:.6g} Pa",
                    value=float(row["E_relative_error"]),
                    threshold=CRITERIA["scalar_rel_error_max"],
                    note=f"E = {row['E']:.10g} Pa",
                )
            )

    gates.append(
        Gate(
            id="axial_bar_E_product_rel_error",
            quantity="|E*A - E_true*A_true| / (E_true*A_true) with E and A both unknowns",
            reference=f"E_true * A = {product['product_reference']:.6g} N",
            value=float(product["product_relative_error"]),
            threshold=CRITERIA["product_rel_error_max"],
            note=f"E*A = {product['product']:.10g} N",
        )
    )
    gates.append(
        Gate(
            id="non_identifiable_flag",
            quantity="identifiable flag of the E+A control (1 = claims E and A separately)",
            reference="the observations contain only the product E*A, so the correct diagnosis is not identifiable",
            value=float(bool(product["identifiable"])),
            threshold=0.0,
            note=f"identifiable={product['identifiable']}, condition={product['condition_number']:.4g}",
        )
    )
    gates.append(
        Gate(
            id="effective_rank",
            quantity="effective rank of the weighted Jacobian, E+A control",
            reference="rank 1: one identifiable combination (E*A) out of two unknowns",
            value=float(product["effective_rank"]),
            threshold=1.0,
            note=f"numerical rank {product.get('rank', 'n/a')}",
        )
    )

    decay_rows = [r for r in rows if r["case"] == "decay"]
    taus = [float(r["tau"]) for r in decay_rows]
    spread = (max(taus) - min(taus)) / DECAY_TRUTH["tau"]
    gates.append(
        Gate(
            id="multi_start_spread",
            quantity="(max tau - min tau) / tau_true over the three multi-start seeds",
            reference="the recovered optimum must not depend on the start point",
            value=float(spread),
            threshold=CRITERIA["start_spread_rel_max"],
            note=f"tau in [{min(taus):.12g}, {max(taus):.12g}] s",
        )
    )
    gates.append(
        Gate(
            id="same_seed_log_replay",
            quantity="convergence log of one seed replayed in the same process is bit-for-bit identical",
            reference="the JSON-lines convergence log hash must repeat exactly",
            value=1.0 if repro["same_seed_identical"] else 0.0,
            threshold=CRITERIA["log_replay_identical_min"],
            operator=">=",
            note=f"sha256 {repro['log_sha256_first'][:16]}...",
        )
    )
    gates.append(
        Gate(
            id="distinct_seeds_differ",
            quantity="a different seed reaches a different start sequence (the seed control is live)",
            reference="seed control is wired: seed 1 and seed 2 logs must differ",
            value=1.0 if repro["distinct_seeds_differ"] else 0.0,
            threshold=CRITERIA["log_replay_identical_min"],
            operator=">=",
            note=f"seed2 sha256 {repro['log_sha256_seed2'][:16]}...",
        )
    )

    nonconverged = [f"{r['case']} s{r['seed']}" for r in rows if not r["converged"]]
    gates.append(
        Gate(
            id="nonconverged_cases",
            quantity="number of inversion runs that did not reach the convergence tolerance",
            reference="every run must report converged before its estimate is graded",
            value=float(len(nonconverged)),
            threshold=CRITERIA["nonconverged_cases_max"],
            note=", ".join(nonconverged) or "all runs converged",
        )
    )
    return gates


def emit_certificate(payload: dict) -> dict:
    """Write ``gates_verified.json`` from the dump and return the manifest."""
    gates = build_gates(payload)
    manifest = gate_manifest(
        anchor="fem_inverse_identification",
        title="tensorfem.inverse: recovered tau / E against generating truth, ill-posed E+A control, log replay",
        mission="GAP-PP-4-opt-loop",
        criteria=CRITERIA,
        gates=gates,
        artifacts={
            "result": _relative(artifact_paths()["result"]),
            "figure": _relative(artifact_paths()["figure"]),
        },
        limitations=[
            "Both cases are inverse crimes in the strict sense: the data are generated by the "
            "same forward model the optimiser inverts (single exponential / 4-element truss), so "
            "the residual error is the optimiser's, not a model error; a discretisation-error "
            "study of the forward model is not attempted here.",
            "The decay case is 1 % noisy and the axial bar 0.5 % noisy with a single noise draw "
            "per case, so the reported errors are one realisation, not an ensemble mean.",
            "The log replay is checked in-process only (same interpreter, same process); "
            "cross-process determinism depends on torch thread scheduling and is not asserted.",
            "The E+A control expects effective rank 1 from a weighted-Jacobian singular-value "
            "cut, which is a tolerance choice, not a theorem; the reported condition number "
            "(~4.5e11) is the tolerance-free statement.",
        ],
    )
    write_json(artifact_paths()["gates"], manifest)
    return manifest


def load_dump() -> dict:
    path = artifact_paths()["result"]
    if not path.is_file():
        raise SystemExit(f"no dump at {path}; run the driver without --from-dump first")
    return read_json(path)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-dump",
        action="store_true",
        help="rebuild gates_verified.json from the committed result.json (no solver run)",
    )
    # ``argv is None`` means "no options" rather than "consult sys.argv": the
    # driver is also imported and called as ``main()`` from a test, where
    # sys.argv belongs to pytest.
    args = parser.parse_args([] if argv is None else list(argv))

    if args.from_dump:
        payload = load_dump()
        manifest = emit_certificate(payload)
        print(f"certificate: {artifact_paths()['gates']}  passed={manifest['passed']}")
        print(f"failed gates: {manifest['failed_gates']}")
        return 0 if manifest["passed"] else 1

    OUT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
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

    figure = artifact_paths()["figure"]
    make_figure(decay_runs, decay, bar, product, repro, figure)
    if figure.parent != OUT and figure.is_file():
        (OUT / figure.name).write_bytes(figure.read_bytes())

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
        "figure": _relative(artifact_paths()["figure"]),
        "repro_cmd": "PYTHONPATH=src python benchmarks/verified/fem_inverse_identification/run.py",
    }
    write_json(artifact_paths()["result"], payload)
    (OUT / "benchmark.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = emit_certificate(payload)
    passed = bool(passed and manifest["passed"])

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
    paths = artifact_paths()
    print(f"artifact   : {paths['result']}")
    print(f"certificate: {paths['gates']}  ({len(manifest['gates'])} gates, failed={manifest['failed_gates']})")
    print(f"figure     : {paths['figure']}")
    print(f"GATE: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
