"""Diagnose a Gmsh ASCII mesh and optionally convert a valid model."""
import argparse

from tensorfem.mesh_pipeline import diagnose_model, read_gmsh


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mesh",help="Gmsh v2/v4 ASCII mesh")
    parser.add_argument("--report",default="mesh-diagnostics.json")
    args=parser.parse_args()
    report=diagnose_model(read_gmsh(args.mesh)); report.write_json(args.report)
    report.require_valid()
    print(f"valid mesh; diagnostics written to {args.report}")


if __name__ == "__main__": main()
