"""Inverse-identification demo: optimiser loop, loss composition, audit log.

Three cases, each with an independent ground truth:

1. ``exponential-decay``  -- recover the time constant ``tau`` of a decaying
   observable (the same mathematical object as a decaying LBM velocity signal:
   a single exponential plus an offset) from 1 %-noise synthetic data, for three
   seeds.  Acceptance: ``|tau - tau_true| / tau_true < 3 %``.
2. ``axial-bar``          -- recover Young's modulus ``E`` from the nodal
   displacements of a real :class:`tensorfem.model.TrussModel` solve (4 axial
   elements, 1 %-noise data).  Acceptance: ``< 3 %``.
3. ``axial-bar-product``  -- ask for ``E`` *and* ``A`` when the data only contain
   the product ``E * A``: the infrastructure must report a rank-deficient,
   non-identifiable problem instead of a confident answer.

Everything written here is machine-readable (``convergence.jsonl``,
``summary.json``, ``metrics.json``); the numbers in the report are read back from
those files, never hand-copied.

Usage
-----
    PYTHONPATH=src python examples/inverse_parameter_identification.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensorfem.inverse import (  # noqa: E402
    InverseProblem,
    OptimizerOptions,
    Parameter,
    SeedControl,
    run_inversion,
)
from tensorfem.model import TrussModel  # noqa: E402
from tensorfem.solvers import solve_linear_static  # noqa: E402

FLOAT = torch.float64
OUT = Path(__file__).resolve().parents[1] / "runs" / "inverse-parameter-identification"

DECAY_TRUTH = {"A": 60.0, "tau": 4.0, "Tinf": 20.0}
DECAY_TIMES = torch.linspace(0.0, 20.0, 41, dtype=FLOAT)

BAR_TRUTH = {"E": 210.0e9}
BAR_AREA = 1.0e-3
BAR_LENGTH = 2.0
BAR_NODES = 4
BAR_LOAD = 1.0e4


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=OUT.parents[1], text=True
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# case 1: exponential decay
# ---------------------------------------------------------------------------
def decay_forward(parameters):
    return parameters["Tinf"] + parameters["A"] * torch.exp(-DECAY_TIMES / parameters["tau"])


DECAY_CLEAN = decay_forward(DECAY_TRUTH)


def decay_problem(seed: int = 7) -> InverseProblem:
    generator = torch.Generator().manual_seed(seed)
    data = DECAY_CLEAN + 0.01 * DECAY_CLEAN.abs() * torch.randn(
        DECAY_CLEAN.shape, dtype=FLOAT, generator=generator
    )
    return InverseProblem(
        forward=decay_forward,
        data=data,
        parameters=[
            Parameter("A", 50.0, lower=1.0, upper=200.0),
            Parameter("tau", 2.0, lower=0.1, upper=50.0, log=True),
            Parameter("Tinf", 0.0, lower=-50.0, upper=100.0),
        ],
        sigma=0.8,
        name="exponential-decay",
    )


# ---------------------------------------------------------------------------
# case 2/3: real FEM forward model
# ---------------------------------------------------------------------------
def bar_setup():
    x = torch.linspace(0.0, BAR_LENGTH, BAR_NODES + 1, dtype=FLOAT)
    nodes = torch.stack([x, torch.zeros_like(x)], dim=1)
    elements = torch.tensor([[i, i + 1] for i in range(BAR_NODES)])
    forces = torch.zeros(2 * (BAR_NODES + 1), dtype=FLOAT)
    forces[2 * BAR_NODES] = BAR_LOAD
    fixed = torch.tensor([0] + list(range(1, 2 * (BAR_NODES + 1), 2)))
    return nodes, elements, forces, fixed


def bar_displacements(parameters):
    nodes, elements, forces, fixed = bar_setup()
    model = TrussModel(
        nodes=nodes,
        elements=elements,
        young_modulus=torch.tensor([parameters["E"]], dtype=FLOAT),
        area=torch.tensor([parameters.get("A", BAR_AREA)], dtype=FLOAT),
        forces=forces,
        fixed_dofs=fixed,
    )
    return solve_linear_static(model).displacement[torch.arange(1, BAR_NODES + 1) * 2]


BAR_CLEAN = bar_displacements(BAR_TRUTH)


def bar_problem(*, unknown_area: bool = False, seed: int = 11) -> InverseProblem:
    generator = torch.Generator().manual_seed(seed)
    data = BAR_CLEAN + 0.005 * BAR_CLEAN.abs() * torch.randn(
        BAR_CLEAN.shape, dtype=FLOAT, generator=generator
    )
    parameters = [Parameter("E", 150.0e9, lower=1.0e9, upper=1.0e12, log=True)]
    if unknown_area:
        parameters.append(Parameter("A", 5.0e-4, lower=1.0e-6, upper=1.0e-1, log=True))
    return InverseProblem(
        forward=bar_displacements,
        data=data,
        parameters=parameters,
        sigma=0.005 * float(BAR_CLEAN[-1]),
        name="axial-bar-product" if unknown_area else "axial-bar",
    )


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def make_figure(decay_runs, bar_run, product_run, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    # panel 1: data vs recovered decay curve
    ax = axes[0]
    reference = decay_runs[0].problem
    times = DECAY_TIMES.tolist()
    ax.plot(times, DECAY_CLEAN.tolist(), "k-", lw=1.4, label="truth (tau=4.0 s)")
    ax.plot(
        times,
        reference.data.tolist(),
        "o",
        ms=3.4,
        color="#666666",
        alpha=0.75,
        label="data (1 % noise)",
    )
    for index, run in enumerate(decay_runs):
        parameters = run.best_parameters
        fit = [
            parameters["Tinf"] + parameters["A"] * pow(2.718281828459045, -t / parameters["tau"])
            for t in times
        ]
        ax.plot(
            times,
            fit,
            "--",
            lw=1.1,
            color=colors[index],
            label=f"seed {run.seed_control.seed}: tau={parameters['tau']:.4f} s",
        )
    ax.set_xlabel("time t [s]")
    ax.set_ylabel("temperature [degC]")
    ax.set_title("case 1: exponential decay, tau recovery")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.3)

    # panel 2: convergence histories
    ax = axes[1]
    for index, run in enumerate(decay_runs):
        losses = [record["loss"] for record in run.iterations if record["start"] == 0]
        ax.semilogy(
            range(1, len(losses) + 1),
            losses,
            "-",
            lw=1.5,
            color=colors[index],
            label=f"decay seed {run.seed_control.seed}",
        )
    bar_losses = [record["loss"] for record in bar_run.iterations if record["start"] == 0]
    ax.semilogy(
        range(1, len(bar_losses) + 1),
        bar_losses,
        "-",
        lw=1.5,
        color="#9467bd",
        label="axial bar (E)",
    )
    product_losses = [
        record["loss"] for record in product_run.iterations if record["start"] == 0
    ]
    ax.semilogy(
        range(1, len(product_losses) + 1),
        product_losses,
        ":",
        lw=1.5,
        color="#ff7f0e",
        label="axial bar (E and A: non-identifiable)",
    )
    ax.set_xlabel("accepted/tried iteration")
    ax.set_ylabel("normalised loss (component sum)")
    ax.set_title("convergence history (audit log replay)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, which="both")

    # panel 3: FEM observables, data vs recovered model
    ax = axes[2]
    problem = bar_run.problem
    x_nodes = [BAR_LENGTH * i / BAR_NODES for i in range(1, BAR_NODES + 1)]
    recovered = bar_displacements(bar_run.best_parameters).tolist()
    ax.plot(x_nodes, BAR_CLEAN.tolist(), "k-", lw=1.4, label="truth (E=210 GPa)")
    ax.plot(x_nodes, problem.data.tolist(), "o", ms=4.0, color="#666666", label="data")
    ax.plot(x_nodes, recovered, "x", ms=8.0, color="#2ca02c", label="recovered model")
    percentage = [
        100.0 * (model - exact) / exact
        for model, exact in zip(recovered, BAR_CLEAN.tolist())
    ]
    for x_value, error, y_value in zip(x_nodes, percentage, recovered):
        ax.annotate(f"{error:+.2f} %", (x_value, y_value), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=7.0)
    ax.set_xlabel("axial position x [m]")
    ax.set_ylabel("nodal displacement [m]")
    ax.set_title("case 2: TrussModel forward solve, E recovery")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.3)

    figure.suptitle(
        "tensorfem.inverse: auditable parameter identification "
        "(log: convergence.jsonl, hash pinned in summary.json)",
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    provenance = {"git_commit": commit, "script": "examples/inverse_parameter_identification.py"}

    decay_runs = []
    for seed in (1, 2, 3):
        run = run_inversion(
            decay_problem(),
            OptimizerOptions(),
            SeedControl(seed=seed, starts=3),
            provenance=provenance,
        )
        paths = run.write(OUT / f"decay-seed{seed}")
        decay_runs.append(run)
        print(f"[decay seed {seed}] status={run.status} tau={run.best_parameters['tau']:.6f} "
              f"sha256={run.log_sha256()[:12]} -> {paths['log']}")

    bar_run = run_inversion(
        bar_problem(), OptimizerOptions(), SeedControl(seed=0, starts=1), provenance=provenance
    )
    bar_paths = bar_run.write(OUT / "axial-bar")
    print(f"[bar] status={bar_run.status} E={bar_run.best_parameters['E']:.6e} "
          f"sha256={bar_run.log_sha256()[:12]} -> {bar_paths['log']}")

    product_run = run_inversion(
        bar_problem(unknown_area=True),
        OptimizerOptions(),
        SeedControl(seed=0, starts=1),
        provenance=provenance,
    )
    product_paths = product_run.write(OUT / "axial-bar-product")
    print(f"[bar E+A] status={product_run.status} identifiable={product_run.diagnostics['identifiable']} "
          f"effective_rank={product_run.diagnostics['effective_rank']} -> {product_paths['log']}")

    figure_path = OUT / "inverse-identification.png"
    make_figure(decay_runs, bar_run, product_run, figure_path)

    # ---- metrics read back from the written summaries (no hand-copied numbers)
    tau_truth = DECAY_TRUTH["tau"]
    decay_table = []
    for run in decay_runs:
        summary = json.loads((OUT / f"decay-seed{run.seed_control.seed}" / "summary.json").read_text())
        tau = summary["parameters"]["tau"]
        decay_table.append(
            {
                "seed": summary["seed_control"]["seed"],
                "tau": tau,
                "tau_reference": tau_truth,
                "relative_error": abs(tau - tau_truth) / tau_truth,
                "status": summary["status"],
                "n_iterations": summary["n_iterations"],
                "n_forward_evals": summary["n_forward_evals"],
                "reported_relative_stderr": summary["diagnostics"]["relative_standard_error"]["tau"],
                "log_sha256": summary["convergence_log_sha256"],
            }
        )

    bar_summary = json.loads((OUT / "axial-bar" / "summary.json").read_text())
    bar_table = {
        "E": bar_summary["parameters"]["E"],
        "E_reference": BAR_TRUTH["E"],
        "relative_error": abs(bar_summary["parameters"]["E"] - BAR_TRUTH["E"]) / BAR_TRUTH["E"],
        "reported_relative_stderr": bar_summary["diagnostics"]["relative_standard_error"]["E"],
        "condition_number": bar_summary["diagnostics"]["condition_number"],
        "reduced_chi2": bar_summary["diagnostics"]["reduced_chi2"],
        "status": bar_summary["status"],
        "log_sha256": bar_summary["convergence_log_sha256"],
    }

    product_summary = json.loads((OUT / "axial-bar-product" / "summary.json").read_text())
    product_table = {
        "identifiable": product_summary["diagnostics"]["identifiable"],
        "rank": product_summary["diagnostics"]["rank"],
        "effective_rank": product_summary["diagnostics"]["effective_rank"],
        "condition_number": product_summary["diagnostics"]["condition_number"],
        "weak_direction": product_summary["diagnostics"]["weak_directions"],
        "product": product_summary["parameters"]["E"] * product_summary["parameters"]["A"],
        "product_reference": BAR_TRUTH["E"] * BAR_AREA,
        "product_relative_error": abs(
            product_summary["parameters"]["E"] * product_summary["parameters"]["A"]
            - BAR_TRUTH["E"] * BAR_AREA
        )
        / (BAR_TRUTH["E"] * BAR_AREA),
        "warnings": product_summary["warnings"],
        "status": product_summary["status"],
        "log_sha256": product_summary["convergence_log_sha256"],
    }

    repeat = run_inversion(
        decay_problem(), OptimizerOptions(), SeedControl(seed=1, starts=3), provenance=provenance
    )
    reproducibility = {
        "same_seed_log_sha256_first": decay_runs[0].log_sha256(),
        "same_seed_log_sha256_repeat": repeat.log_sha256(),
        "identical": decay_runs[0].log_sha256() == repeat.log_sha256(),
        "distinct_seed_hashes": len({run.log_sha256() for run in decay_runs}),
        "seeds_run": [run.seed_control.seed for run in decay_runs],
    }

    metrics = {
        "schema": "tensorfem.inverse-demo/1",
        "provenance": provenance,
        "decay": decay_table,
        "axial_bar": bar_table,
        "axial_bar_product": product_table,
        "reproducibility": reproducibility,
        "figure": str(figure_path.relative_to(OUT.parents[1])),
    }
    metrics_path = OUT / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


if __name__ == "__main__":
    main()