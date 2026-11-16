"""TensorFEM command line interface."""

import argparse
import json
import os
from pathlib import Path

from .benchmarks import run_single_bar, solve_three_bar
from .benchmark_registry import verification_report


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tensorfem",
        description="TensorFEM verified finite-element command line tools",
    )
    parser.add_argument(
        "command",
        choices=(
            "benchmark",
            "three-bar",
            "verify",
            "capabilities",
            "patch-test",
            "validate",
            "run",
            "resume",
            "status",
            "inspect",
            "inverse-identify",
            "body-force-inversion",
        ),
        help="run, validate, inspect, or verify TensorFEM projects and results",
    )
    parser.add_argument("target", nargs="?", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument(
        "--case",
        default="decay",
        choices=("decay", "axial-bar", "axial-bar-product"),
        help="inverse-identify case (default: decay)",
    )
    parser.add_argument("--seed", type=int, default=1, help="inverse-identify seed")
    parser.add_argument("--starts", type=int, default=3, help="inverse-identify start count")
    parser.add_argument(
        "--order", type=int, default=2, help="body-force-inversion modal order (fixed at 2)"
    )
    parser.add_argument(
        "--fraction", type=float, default=0.10, help="body-force-inversion observed node fraction"
    )
    parser.add_argument(
        "--alpha-rel", type=float, default=1.0e-12, help="body-force-inversion relative damping"
    )
    parser.add_argument(
        "--noise", type=float, default=0.0, help="body-force-inversion relative observation noise"
    )
    parser.add_argument(
        "--max-adjoint", type=int, default=50, help="body-force-inversion adjoint-iteration budget"
    )
    parser.add_argument("--nx", type=int, default=20, help="body-force-inversion mesh nx")
    parser.add_argument("--ny", type=int, default=10, help="body-force-inversion mesh ny")
    args = parser.parse_args()
    if args.command == "capabilities":
        print(
            json.dumps(
                {
                    "version": __import__("tensorfem").__version__,
                    "stable": [
                        "2-D truss/frame/continuum",
                        "modal/buckling/dynamics",
                        "TET4/HEX8/TET10/HEX20 solids",
                        "sparse assembly and CG",
                        "adaptive and coupled plastic-contact nonlinear steps",
                        "TET4 J2 plasticity",
                        "thermal and sequential thermoelasticity",
                        "model database and engineering I/O",
                        "finite-sliding 2-D/3-D frictional contact kernels",
                        "Jacobi-GMRES and multi-RHS sparse solves",
                        "additive-Schwarz domain decomposition",
                        "Gmsh v2/v4 mesh diagnostics and conversion",
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
                        "CASE-SPHERE-PRESSURE closed-sphere membrane case (tensorfem-sphere-pressure)",
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
                        "shallow-warp layered Shell4 reference geometry",
                        "two-sided deformable 3-D facet contact assembly",
                        "transactional residual-stress equilibration and scaled shell continuation",
                        "marine panel 2x2/4x4 first-point performance qualification",
                        "matrix-free transactional shell arc continuation",
                        "branch-preserving panel long-path checkpoints",
                        "accepted-state panel energy and failure evidence",
                        "sparse Shell4 tangent storage and action foundation",
                        "normalized augmented shell arc predictor",
                        "8x8/12x12 panel short-path scalability evidence",
                        "energy-conjugate panel path observer and versioned ledgers",
                        "8x8 ten-point and 12x12 first-point execution evidence",
                        "generalized refined-panel peak-window execution foundation",
                        "dimensionally consistent panel arc metric and immutable checkpoint generations",
                        "optional fail-closed sparse direct and ILU adapter",
                        "advisory nonlinear-control evidence with hashed restart decisions",
                        "marine stiffening, corrosion and residual-stress screening qualifications",
                        "10k-active-DOF Shell4 ILU-GMRES memory qualification with hashed reports",
                        "consistent two-sided frictional Mortar assembly and bounded curved stick-slip path",
                        "bounded finite-strain double-deformable friction paths",
                        "generation-scheduled panel dual-head transaction integrity and readiness evidence",
                        "hashed executable v1 evidence aggregation and release-log verification",
                    ],
                    "experimental": [
                        "general nonlinear doubly-curved shell engineering solver",
                        "large-step closed-form plastic tangent and large contact graphs",
                        "published ring-load snap-through shell qualification data",
                        "qualified multi-GPU contact-island execution",
                        "general crack propagation",
                        "multi-node distributed production runs",
                        "marine panel peak/post-peak qualification evidence",
                        "general 3-D deformable-to-deformable Hertz qualification",
                        "marine panel scalable tangent and peak/post-peak execution evidence",
                        "4x4 cross-step elastic peak/post-peak and energy qualification; mesh convergence blocked",
                        "complete 8x8 peak localization and descending-branch evidence",
                        "complete 12x12 peak localization and 4/8/12 convergence evidence",
                        "slow curved two-pass frictional qualification; fail-closed on current evidence",
                        "general curved frictional surface-to-surface contact remains blocked",
                        "general contact composite awaits deformable-to-deformable Hertz evidence",
                        "4/8/12 panel peak/post-peak matrix remains incomplete",
                    ],
                },
                indent=2,
            )
        )
        return
    if args.command in ("validate", "run", "resume"):
        if args.target is None:
            parser.error(f"{args.command} requires a project JSON path")
        from .mesh_project import load_mesh_project, run_mesh_project

        project = load_mesh_project(args.target)
        if args.command == "validate":
            print(
                json.dumps(
                    {"valid": True, "project": str(args.target), "schema": project["schema"]},
                    indent=2,
                )
            )
            return
        result = run_mesh_project(args.target, args.runs, replay=args.command == "resume")
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
        print(
            json.dumps(
                {
                    "schema": payload.get("schema"),
                    "database_sha256": payload.get("database_sha256"),
                    "body": payload.get("body"),
                    "steps": payload.get("steps"),
                    "inventory": payload.get("inventory"),
                },
                indent=2,
            )
        )
        return
    if args.command == "body-force-inversion":
        # Entry point of tensorfem.body_force_inversion: recover a distributed
        # volume-load field from a sparse (<10 %) set of nodal displacement
        # samples in plane stress.  The exit status is the gate: 0 only when all
        # three anchor acceptance criteria hold (<=10 % observed nodes, field
        # relative L2 <= 5 %, <= 50 adjoint iterations).  Noisy or out-of-span
        # runs therefore fail loudly instead of printing a hopeful number.
        from .body_force_inversion import (
            InversionOptions,
            PlaneStressMesh,
            run_body_force_inversion,
        )

        if args.order != 2:
            raise SystemExit(
                "body-force-inversion: --order is fixed at 2 (the 8-coefficient acceptance "
                "case); other orders would need a different truth vector"
            )
        truth = [1.0, 0.5, -0.4, 0.3, 0.6, -0.7, 0.25, 0.4]
        result = run_body_force_inversion(
            mesh=PlaneStressMesh(Lx=1.0, Ly=0.25, nx=args.nx, ny=args.ny),
            truth_coeffs=truth,
            order=2,
            fraction=args.fraction,
            options=InversionOptions(
                max_adjoint=args.max_adjoint,
                alpha_rel=args.alpha_rel,
                noise=args.noise,
                seed=args.seed,
            ),
        )
        row = {
            "schema": result["schema"],
            "entry": "tensorfem.cli body-force-inversion",
            "mesh": result["mesh"],
            "observations": result["observations"],
            "n_unknown": result["basis"]["n_unknown"],
            "iterations": result["iterations"],
            "adjoint_iteration_budget": result["adjoint_iteration_budget"],
            "stop_reason": result["stop_reason"],
            "field_rel_l2": result["field_rel_l2"],
            "relative_data_residual": result["relative_data_residual"],
            "acceptance": result["acceptance"],
            "passed": all(result["acceptance"].values()),
            "coefficients": result["coefficients"],
            "convergence": result["convergence"],
        }
        payload = json.dumps(row, indent=2, ensure_ascii=False)
        if args.output is None:
            target = (
                args.runs
                / "body-force-inversion"
                / f"order{args.order}-frac{args.fraction:g}-noise{args.noise:g}"
                / "result.json"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target = args.output
            # An existing directory (or a path ending in "/") gets the canonical
            # file name; anything else is the file the caller asked for.  Suffix
            # sniffing was wrong: "order2-frac0.1-noise0" has a suffix ".1-noise0".
            raw = str(args.output)
            if target.is_dir() or raw.endswith(("/", os.sep)) or target.suffix == "":
                target.mkdir(parents=True, exist_ok=True)
                target = target / "result.json"
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        raise SystemExit(0 if row["passed"] else 1)
    if args.command == "inverse-identify":
        # Entry point of tensorfem.inverse: one identification run, the
        # JSON-lines convergence log and the run summary are written under
        # --output (or runs/inverse-identify/<case>-seed<seed>/).  The exit
        # status is the gate: 0 only when the recovered scalar is inside 3 %
        # of its independent ground truth (inverted for the negative control,
        # which passes only when non-identifiability is reported).
        from .inverse_cases import identify

        outcome = identify(args.case, seed=args.seed, starts=args.starts)
        directory = args.output or (
            args.runs / "inverse-identify" / f"{args.case}-seed{args.seed}"
        )
        paths = outcome["run"].write(directory)
        row = dict(outcome["row"])
        row["artifacts"] = paths
        payload = json.dumps(row, indent=2, ensure_ascii=False)
        print(payload)
        raise SystemExit(0 if row["passed"] else 1)
    if args.command == "patch-test":
        # Entry point of the Robinson patch test A/B/C (tensorfem.patch_test).
        # Exit status is the gate: 0 only when every case behaves as a correct
        # element family must, including the cases that are *supposed* to fail.
        from .patch_test import qualification_report

        report = qualification_report()
        payload = json.dumps(report, indent=2, ensure_ascii=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        raise SystemExit(0 if report["passed"] else 1)
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
    print(
        json.dumps(
            {
                "displacement": result.displacement.tolist(),
                "reaction": result.reaction.tolist(),
                "axial_stress": result.axial_stress.tolist(),
                "strain_energy": float(result.strain_energy),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
