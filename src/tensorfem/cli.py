"""TensorFEM command line interface."""
import argparse
import json
from pathlib import Path

from .benchmarks import run_single_bar, solve_three_bar
from .benchmark_registry import verification_report


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tensorfem",
        description="TensorFEM verified finite-element command line tools",
    )
    parser.add_argument(
        "command", choices=("benchmark", "three-bar", "verify", "capabilities"),
        help="run the legacy bar benchmark, three-bar example, formal verification, or list capabilities",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "capabilities":
        print(json.dumps({
            "version": __import__("tensorfem").__version__,
            "stable": [
                "2-D truss/frame/continuum", "modal/buckling/dynamics",
                "TET4/HEX8/TET10/HEX20 solids", "sparse assembly and CG",
                "adaptive and coupled plastic-contact nonlinear steps", "TET4 J2 plasticity",
                "thermal and sequential thermoelasticity", "model database and engineering I/O",
                "finite-sliding 2-D/3-D frictional contact kernels", "Jacobi-GMRES and multi-RHS sparse solves",
                "additive-Schwarz domain decomposition", "Gmsh v2/v4 mesh diagnostics and conversion",
                "non-diagonal structural-grid sparse benchmark",
                "auditable 3-D bar plasticity/contact coupling benchmark",
                "multi-element small-strain TET4 J2 load-path solver",
                "low-order node-quadrature surface contact",
                "total-Lagrangian TET4 Neo-Hookean elasticity",
                "axisymmetric deformable Hertz verification model",
                "multiplicative finite-strain J2 proportional-path prototype",
                "low-order frictionless mortar contact foundation",
                "adaptive nonproportional finite-J2 path integrator",
                "incremental frictional mortar and self-contact force foundation",
                "multi-TET adaptive cyclic finite-plasticity solver",
                "vertex-triangle CCD and dynamic self-contact history",
            ],
            "experimental": [
                "general nonlinear doubly-curved shell engineering solver",
                "closed-form plastic tangent and edge-edge/contact friction impulses",
                "general crack propagation", "multi-node distributed production runs",
            ],
        }, indent=2))
        return
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
