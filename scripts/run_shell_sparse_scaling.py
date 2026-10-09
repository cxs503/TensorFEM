#!/usr/bin/env python3
import argparse
import json

from pathlib import Path

from tensorfem.shell_sparse_scaling import (
    benchmark_shell_sparse_scaling, sparse_scaling_qualification,
    verify_legacy_sparse_tangent_cache, verify_sparse_scaling_report,
)

parser = argparse.ArgumentParser()
parser.add_argument("--meshes", type=int, nargs="+", default=(8, 12))
parser.add_argument("--drop-tolerance", type=float, default=1e-4)
parser.add_argument("--fill-factor", type=float, default=10.)
parser.add_argument("--maxiter", type=int, default=300)
parser.add_argument("--cache-dir")
parser.add_argument("--report")
parser.add_argument("--minimum-active-dofs", type=int, default=10000)
parser.add_argument("--verify-legacy-cache", action="store_true")
parser.add_argument("--verify-report")
args = parser.parse_args()
if args.verify_report:
    verified = verify_sparse_scaling_report(args.verify_report)
    print(json.dumps({"verified_report": args.verify_report,
                      "evidence_sha256": verified["evidence_sha256"],
                      "passed": verified["passed"]}, indent=2))
    raise SystemExit(0)
if args.verify_legacy_cache:
    if args.cache_dir is None:
        parser.error("--verify-legacy-cache requires --cache-dir")
    migrations = []
    for mesh in args.meshes:
        migrations.append(verify_legacy_sparse_tangent_cache(
            mesh, f"{args.cache_dir}/panel-{mesh}.pt"))
    print(json.dumps({"verified_legacy_caches": migrations}, indent=2))
points = [benchmark_shell_sparse_scaling(
                      mesh, drop_tolerance=args.drop_tolerance,
                      fill_factor=args.fill_factor, maxiter=args.maxiter,
                      tangent_cache=(None if args.cache_dir is None else
                                     f"{args.cache_dir}/panel-{mesh}.pt"))
          for mesh in args.meshes]
report = sparse_scaling_qualification(
    points, minimum_active_dofs=args.minimum_active_dofs)
encoded = json.dumps(report, indent=2, sort_keys=True)+"\n"
if args.report:
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix+".tmp")
    temporary.write_text(encoded)
    temporary.replace(target)
print(encoded, end="")
if not report["passed"]:
    raise SystemExit(2)
