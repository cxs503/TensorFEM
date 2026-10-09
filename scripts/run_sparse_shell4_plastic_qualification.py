#!/usr/bin/env python3
"""Run one Shell4 plastic-path backend in an isolated process."""
import argparse

from tensorfem.sparse_shell4_plastic_qualification import report_json, run_plastic_path


parser = argparse.ArgumentParser()
parser.add_argument("backend", choices=("dense", "superlu"))
parser.add_argument("--steps", type=int, default=80)
args = parser.parse_args()
print(report_json(run_plastic_path(args.backend, steps=args.steps)))
