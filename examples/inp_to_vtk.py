"""Import an Abaqus truss model, solve it and write VTK results."""
import argparse
from pathlib import Path

from tensorfem.abaqus_io import read_inp
from tensorfem.modeldb import to_truss_model
from tensorfem.result_io import write_vtk
from tensorfem.solvers import solve_linear_static


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve a supported Abaqus T2D2 INP and export VTK results.")
    parser.add_argument("input", type=Path, help="input Abaqus .inp file")
    parser.add_argument("output", type=Path, help="output VTK legacy .vtk file")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"input file does not exist: {args.input}")
    try:
        db = read_inp(args.input)
        model, _ = to_truss_model(db)
        result = solve_linear_static(model)
        write_vtk(args.output, db,
                  point_data={"displacement": result.displacement.reshape(-1, 2)},
                  cell_data={"axial_stress": result.axial_stress})
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
