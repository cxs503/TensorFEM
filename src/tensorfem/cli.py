"""TensorFEM command line interface."""
import argparse
import json
from pathlib import Path

from .benchmarks import run_single_bar, solve_three_bar
from .benchmark_registry import verification_report


def main() -> None:
    parser = argparse.ArgumentParser(prog="tensorfem")
    parser.add_argument("command", choices=("benchmark", "three-bar", "verify"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        report = verification_report()
        payload = json.dumps(report, indent=2, ensure_ascii=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        raise SystemExit(0 if report["passed"] else 1)
    if args.command == "benchmark":
        result = run_single_bar()
        print(json.dumps(result.__dict__, indent=2))
        raise SystemExit(0 if result.passed else 1)
    result = solve_three_bar()
    print(json.dumps({"displacement": result.displacement.tolist(),
                      "reaction": result.reaction.tolist(),
                      "axial_stress": result.axial_stress.tolist(),
                      "strain_energy": float(result.strain_energy)}, indent=2))


if __name__ == "__main__":
    main()
