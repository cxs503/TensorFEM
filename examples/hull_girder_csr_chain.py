"""Run the KVLCC2 hull-girder CSR bending chain and write the evidence bundle.

Usage (from the repository root)::

    PYTHONPATH=src python examples/hull_girder_csr_chain.py

Produces ``artifacts/hull_girder_csr_chain/results.json`` (every acceptance
number) plus the four figures of :func:`tensorfem.hull_girder_csr_chain.plot_hull_girder_chain`.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from tensorfem.hull_girder_csr_chain import (  # noqa: E402
    plot_hull_girder_chain,
    run_kvlcc2_csr_chain,
)

OUTDIR = os.path.join("artifacts", "hull_girder_csr_chain")


def main() -> dict:
    result = run_kvlcc2_csr_chain()
    figures = plot_hull_girder_chain(result, outdir=OUTDIR)
    result["figures"] = [
        {"name": figure["name"], "path": figure["path"], "caption": figure["caption"]}
        for figure in figures
    ]
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, "results.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1)
    print(f"wrote {path}")
    criteria = result["criteria"]
    for key in sorted(criteria):
        print(f"  {key}: {criteria[key]}")
    print("  M_total_hog = %.3f GN m, M_total_sag = %.3f GN m"
          % (result["moments"]["M_total_hog"] / 1e9, result["moments"]["M_total_sag"] / 1e9))
    print("  I_midship = %.1f m4, W_deck = %.2f m3, W_bottom = %.2f m3"
          % (result["section"]["I_midship"], result["section"]["W_deck"],
             result["section"]["W_bottom"]))
    print("  sigma_deck_hog = %.1f MPa, sigma_bottom_sag = %.1f MPa"
          % (result["stresses"]["deck_hog"] / 1e6,
             result["stresses"]["bottom_sag"] / 1e6))
    print("  SF hog/sag = %.3f / %.3f"
          % (criteria["C4_safety_factor_hog"], criteria["C4_safety_factor_sag"]))
    return result


if __name__ == "__main__":
    main()