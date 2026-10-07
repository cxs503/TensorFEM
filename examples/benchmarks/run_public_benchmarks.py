#!/usr/bin/env python3
"""Run the portable public benchmark catalog and write its report.

This example intentionally produces reference-only cases.  A solver run can be
attached later with ``scripts/run_standard_benchmark_report.py --fe-results``;
the report gate will not claim qualification without real FE fields.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tensorfem.standard_benchmarks import benchmark_case_catalog
from scripts.run_standard_benchmark_report import main as generate_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Keep the public example discoverable while reusing the canonical writer.
    # The catalog call is deliberate: it fails early if the documented cases
    # and the report contract diverge.
    if len(benchmark_case_catalog()) < 5:
        raise RuntimeError("public benchmark catalog is incomplete")
    import sys
    sys.argv = ["run_standard_benchmark_report.py", "--output", str(args.output)]
    generate_report()


if __name__ == "__main__":
    main()
