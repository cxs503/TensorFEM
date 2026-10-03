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
        "command", choices=("benchmark", "three-bar", "verify", "capabilities",
                           "validate", "run", "resume", "status", "inspect"),
        help="run, validate, inspect, or verify TensorFEM projects and results",
    )
    parser.add_argument("target", nargs="?", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
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
                "AD finite-plasticity algorithmic tangent",
                "edge-edge CCD, spatial hash and Coulomb impact",
                "large-rotation shell pure-bending benchmark",
                "local implicit finite-plasticity tangent",
                "multi-contact complementarity impulse solver",
                "Crisfield arc-length limit-point solver",
                "fail-closed hybrid implicit/AD plastic tangent",
                "rigid-body inertia and contact-island PGS",
                "global cyclic hybrid-tangent plasticity solver",
                "colored vectorized rigid-contact graph solver",
                "follower shell pressure and consistent load tangent",
                "taped adaptive-substep plastic tangent",
                "colored vectorized frictional contact",
                "deterministic multi-worker contact-island scheduling",
                "versioned Model-Step-Job-Result workflow",
                "hemisphere JSON/VTK/report post-processing",
                "hashed versioned benchmark evidence and drift checks",
                "safe versioned engineering project workflow",
                "checksummed ResultDB with optional chunked HDF5 body",
                "four-case classical shell evidence suite",
                "safe arbitrary-mesh truss/TET4/HEX8 project execution",
                "steady thermal LINE2/Q4 arbitrary-mesh projects",
                "multistep linear/thermal/modal execution with resume",
                "ResultDB v2 multiframe and integration-point histories",
                "three auditable engineering case packages",
                "versioned public API and offline release gates",
                "ship hull-girder and stiffened-panel screening benchmarks",
                "box-barge hydrostatics and intact small-angle stability",
                "Airy/Morison marine loads and TensorLBM force-history exchange",
                "35-case marine qualification and convergence matrix",
                "TensorFEM-only marine structural qualification gate",
                "marine local buckling, section ultimate and fatigue qualification prototypes",
                "marine reduced-order postbuckling, progressive section and spectrum fatigue prototypes",
                "Shell4 initial-stress buckling and ResultDB hotspot/LEFM qualification",
                "numerical Mode-I contour J integration qualification",
                "ResultDB initial-field transfer to reduced structural response",
                "real-residual Shell4 arc-length integration prototype",
                "multi-facet Shell4 transactional elastic state path",
                "2-D global finite-sliding frictional contact prototype",
                "unified ModelDB AnalysisPlan Job ResultDB v2 workflow",
                "hashed industrial P0 qualification view",
                "small-strain layered Shell4 plane-stress J2 prototype",
                "global 3-D node-to-TRI3 frictional contact prototype",
                "external benchmark contracts and industrial P0 phase-2 qualification",
                "finite-rotation layered Shell4 transactional arc integration",
                "axisymmetric rigid-indenter/deformable-halfspace Hertz qualification",
                "hashed industrial P0 phase-3 qualification view",
            ],
            "experimental": [
                "general nonlinear doubly-curved shell engineering solver",
                "large-step closed-form plastic tangent and large contact graphs",
                "published ring-load snap-through shell qualification data",
                "qualified multi-GPU contact-island execution",
                "general crack propagation", "multi-node distributed production runs",
                "marine panel peak/post-peak qualification evidence",
                "general 3-D deformable-to-deformable Hertz qualification",
            ],
        }, indent=2))
        return
    if args.command in ("validate", "run", "resume"):
        if args.target is None:
            parser.error(f"{args.command} requires a project JSON path")
        from .mesh_project import load_mesh_project, run_mesh_project
        project = load_mesh_project(args.target)
        if args.command == "validate":
            print(json.dumps({"valid": True, "project": str(args.target),
                              "schema": project["schema"]}, indent=2))
            return
        result = run_mesh_project(args.target, args.runs,
                                  replay=args.command == "resume")
        print(json.dumps(result, indent=2))
        return
    if args.command == "status":
        if args.target is None:
            parser.error("status requires a job directory or job.json")
        path = args.target if args.target.name == "job.json" else args.target / "job.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(payload, indent=2))
        raise SystemExit(0 if payload.get("status") == "completed" else 2)
    if args.command == "inspect":
        if args.target is None:
            parser.error("inspect requires a ResultDB JSON path")
        payload = json.loads(args.target.read_text(encoding="utf-8"))
        print(json.dumps({"schema": payload.get("schema"),
                          "database_sha256": payload.get("database_sha256"),
                          "body": payload.get("body"),
                          "steps": payload.get("steps"),
                          "inventory": payload.get("inventory")}, indent=2))
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
