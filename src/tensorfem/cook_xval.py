"""Cook membrane cross-validation entry point (XVAL-FEM-COOK).

This module provides the public API for the Cook membrane horizontal
cross-validation benchmark.  It delegates to the production benchmark
script and returns structured results.

Usage
-----
    from tensorfem.cook_xval import run_cook_xval, load_cook_xval_results
    result = run_cook_xval(levels=[8, 16], parties=["tensorfem_q4", "skfem_quad2"])
    metrics = load_cook_xval_results("artifacts/cook_xval/metrics.json")
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CookXvalResult:
    """Result of a Cook membrane cross-validation run."""

    spec: Dict[str, Any]
    levels: List[int]
    parties: List[str]
    results: Dict[str, Dict[str, Dict[str, Any]]]
    wall_time_s: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @property
    def uy_mid_errors(self) -> Dict[str, Dict[str, float]]:
        """Return uy_mid errors vs published reference (23.96) for all parties/levels."""
        ref = self.spec.get("published_reference_uy_mid", 23.96)
        errors = {}
        for party, levels in self.results.items():
            errors[party] = {}
            for level, data in levels.items():
                uy = data.get("uy_mid", 0)
                errors[party][level] = abs(uy - ref) / ref * 100
        return errors


def run_cook_xval(
    levels: Optional[List[int]] = None,
    parties: Optional[List[str]] = None,
    out_dir: Optional[str] = None,
) -> CookXvalResult:
    """Run the Cook membrane cross-validation benchmark.

    Parameters
    ----------
    levels : list of int
        Mesh refinement levels (n x n elements). Default: [8, 16].
    parties : list of str
        Parties to run. Default: ["tensorfem_q4", "skfem_quad2", "skfem_tri2"].
    out_dir : str, optional
        Directory to write metrics.json. Default: "artifacts/cook_xval".

    Returns
    -------
    CookXvalResult
        The cross-validation result.
    """
    if levels is None:
        levels = [8, 16]
    if parties is None:
        parties = ["tensorfem_q4", "skfem_quad2", "skfem_tri2"]
    if out_dir is None:
        out_dir = "artifacts/cook_xval"

    # Find the benchmark script relative to this module
    here = Path(__file__).parent.parent.parent
    bench_script = here / "benchmarks" / "cook_membrane_xval.py"

    if not bench_script.exists():
        raise FileNotFoundError(f"Benchmark script not found: {bench_script}")

    cmd = [
        sys.executable,
        str(bench_script),
        "--levels", ",".join(str(l) for l in levels),
        "--parties", ",".join(parties),
        "--out", out_dir,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=here)
    if result.returncode != 0:
        raise RuntimeError(f"Benchmark failed: {result.stderr}")

    # Load results
    metrics_path = os.path.join(out_dir, "metrics.json")
    return load_cook_xval_results(metrics_path)


def load_cook_xval_results(metrics_path: str) -> CookXvalResult:
    """Load Cook xval results from a metrics.json file."""
    with open(metrics_path) as f:
        data = json.load(f)

    return CookXvalResult(
        spec=data["spec"],
        levels=data["levels"],
        parties=data["parties"],
        results=data["results"],
        wall_time_s=data.get("wall_time_s", 0),
    )


def generate_convergence_plot(metrics_path: str, out_path: str) -> str:
    """Generate a convergence plot from metrics.json.

    Returns the path to the generated PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with open(metrics_path) as f:
        data = json.load(f)

    ref = data["spec"]["published_reference_uy_mid"]
    parties = data["parties"]
    levels = data["levels"]
    results = data["results"]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"tensorfem_q4": "#1f77b4", "skfem_quad2": "#ff7f0e", "skfem_tri2": "#2ca02c"}
    markers = {"tensorfem_q4": "o", "skfem_quad2": "s", "skfem_tri2": "^"}

    for party in parties:
        if party not in results:
            continue
        uys = []
        for level in levels:
            level_data = results[party].get(str(level), {})
            uys.append(level_data.get("uy_mid", 0))

        color = colors.get(party, "#333333")
        marker = markers.get(party, "o")
        ax.plot(levels, uys, marker=marker, color=color, linewidth=2, markersize=8, label=party)

    ax.axhline(y=ref, color="red", linestyle="--", linewidth=1.5, label=f"Reference ({ref})")
    ax.set_xlabel("Mesh level (n × n elements)", fontsize=12)
    ax.set_ylabel("Midpoint vertical displacement uy(48, 52)", fontsize=12)
    ax.set_title("Cook Membrane: Convergence of Midpoint Displacement", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path


def generate_error_plot(metrics_path: str, out_path: str) -> str:
    """Generate an error vs reference plot from metrics.json.

    Returns the path to the generated PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with open(metrics_path) as f:
        data = json.load(f)

    ref = data["spec"]["published_reference_uy_mid"]
    parties = data["parties"]
    levels = data["levels"]
    results = data["results"]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"tensorfem_q4": "#1f77b4", "skfem_quad2": "#ff7f0e", "skfem_tri2": "#2ca02c"}
    markers = {"tensorfem_q4": "o", "skfem_quad2": "s", "skfem_tri2": "^"}

    for party in parties:
        if party not in results:
            continue
        errors = []
        for level in levels:
            level_data = results[party].get(str(level), {})
            uy = level_data.get("uy_mid", 0)
            errors.append(abs(uy - ref) / ref * 100)

        color = colors.get(party, "#333333")
        marker = markers.get(party, "o")
        ax.plot(levels, errors, marker=marker, color=color, linewidth=2, markersize=8, label=party)

    ax.axhline(y=3.0, color="red", linestyle="--", linewidth=1.5, label="3% tolerance")
    ax.set_xlabel("Mesh level (n × n elements)", fontsize=12)
    ax.set_ylabel("Error vs reference (%)", fontsize=12)
    ax.set_title("Cook Membrane: Error in Midpoint Displacement", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path
