"""Compare full fields and measure single-host cost with fresh processes per sample."""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np


def sample(args):
    from tensorfem.parallel.structure import solve_panel

    result = solve_panel(args.nx, args.ny, args.method, args.workers)
    np.savez(args.sample, **result)


def compare():
    import psutil
    import scipy

    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    report = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "processor": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            platform.machine(),
        ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "threads_per_process": 1,
        "memory": "peak sampled sum of process-tree PSS, bytes (50 ms)",
        "parallel_assembly": "includes spawn, local factorization, condensation, shutdown",
        "stress_sampling": "every element centre, all three components",
        "rows": [],
    }
    with tempfile.TemporaryDirectory() as temp:
        for nx, ny in [(32, 8), (80, 16)]:
            reference = None
            for method in ["dense", "sparse", "parallel"]:
                for repeat in range(3):
                    path = Path(temp) / "case.npz"
                    command = [
                        sys.executable,
                        __file__,
                        "--method",
                        method,
                        "--nx",
                        str(nx),
                        "--ny",
                        str(ny),
                        "--workers",
                        "2",
                        "--sample",
                        str(path),
                    ]
                    start = time.perf_counter()
                    process = subprocess.Popen(command, env=env)
                    root = psutil.Process(process.pid)
                    peak = 0
                    while process.poll() is None:
                        try:
                            tree = [root, *root.children(recursive=True)]
                        except psutil.NoSuchProcess:
                            break
                        memory = 0
                        for member in tree:
                            try:
                                memory += member.memory_full_info().pss
                            except psutil.NoSuchProcess:
                                continue
                        peak = max(peak, memory)
                        time.sleep(0.05)
                    if process.wait():
                        raise RuntimeError(f"{method} sample failed")
                    elapsed = time.perf_counter() - start
                    with np.load(path) as data:
                        fields = ("displacement", "reaction", "stress", "force", "energy")
                        if reference is None:
                            reference = {name: data[name].copy() for name in fields}
                        errors = {
                            name: float(
                                np.max(np.abs(data[name] - reference[name]))
                                / max(np.max(np.abs(reference[name])), 1e-30)
                            )
                            for name in fields
                        }
                        closure = abs(float(data["work"] / data["energy"]) - 1)
                        equilibrium = (
                            np.linalg.norm(
                                data["reaction"].reshape(-1, 2).sum(axis=0)
                                + data["force"].reshape(-1, 2).sum(axis=0)
                            )
                            / 1000
                        )
                        ranks = len(set(data["worker_pids"]))
                        passed = (
                            max(errors.values()) < 1e-7
                            and closure < 1e-7
                            and equilibrium < 1e-7
                            and ranks == (2 if method == "parallel" else 1)
                        )
                        row = dict(
                            nx=nx,
                            ny=ny,
                            method=method,
                            repeat=repeat + 1,
                            assembly_s=float(data["assembly_s"]),
                            solve_s=float(data["solve_s"]),
                            total_s=float(data["elapsed_s"]),
                            cold_wall_s=elapsed,
                            peak_pss_bytes=peak,
                            workers=ranks,
                            max_relative_errors=errors,
                            strain_energy_J=float(data["energy"]),
                            tip_y_m=float(data["displacement"][-1]),
                            energy_error=closure,
                            equilibrium_error=float(equilibrium),
                            passed=bool(passed),
                        )
                        report["rows"].append(row)
    out = Path("artifacts/panel_backend_comparison.json")
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not all(row["passed"] for row in report["rows"]):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--method", choices=("dense", "sparse", "parallel"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    sample(args) if args.sample else compare()
