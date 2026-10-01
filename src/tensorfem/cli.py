"""TensorFEM command line interface."""
import argparse
import json

from .benchmarks import run_single_bar, solve_three_bar


def main() -> None:
    parser = argparse.ArgumentParser(prog="tensorfem")
    parser.add_argument("command", choices=("benchmark", "three-bar"))
    args = parser.parse_args()
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
