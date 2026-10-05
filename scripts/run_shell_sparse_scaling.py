#!/usr/bin/env python3
import argparse
import json

from tensorfem.shell_sparse_scaling import benchmark_shell_sparse_scaling, scaling_point_dict

parser = argparse.ArgumentParser()
parser.add_argument("--meshes", type=int, nargs="+", default=(8, 12))
args = parser.parse_args()
print(json.dumps([scaling_point_dict(benchmark_shell_sparse_scaling(mesh))
                  for mesh in args.meshes], indent=2))
