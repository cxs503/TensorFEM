#!/usr/bin/env python3
import argparse
import json

from tensorfem.shell_sparse_scaling import benchmark_shell_sparse_scaling, scaling_point_dict

parser = argparse.ArgumentParser()
parser.add_argument("--meshes", type=int, nargs="+", default=(8, 12))
parser.add_argument("--drop-tolerance", type=float, default=1e-4)
parser.add_argument("--fill-factor", type=float, default=10.)
parser.add_argument("--maxiter", type=int, default=300)
parser.add_argument("--cache-dir")
args = parser.parse_args()
print(json.dumps([scaling_point_dict(benchmark_shell_sparse_scaling(
                      mesh, drop_tolerance=args.drop_tolerance,
                      fill_factor=args.fill_factor, maxiter=args.maxiter,
                      tangent_cache=(None if args.cache_dir is None else
                                     f"{args.cache_dir}/panel-{mesh}.pt")))
                  for mesh in args.meshes], indent=2))
