"""Run the coupled structural grid benchmark at configurable scale."""
import argparse
import json
from dataclasses import asdict

from tensorfem.structural_grid_benchmark import run_grid_benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=1000)
    parser.add_argument("--ny", type=int, default=1000)
    parser.add_argument("--rtol", type=float, default=1e-9)
    args = parser.parse_args()
    result = run_grid_benchmark(args.nx, args.ny, rtol=args.rtol)
    evidence = asdict(result)
    evidence["storage_mib"] = result.storage_bytes/2**20
    evidence["passes_3_percent"] = result.relative_error < .03
    if not evidence["passes_3_percent"]:
        raise SystemExit("accuracy gate failed")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__": main()
