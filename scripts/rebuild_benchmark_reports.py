#!/usr/bin/env python3
"""Recompute, render and validate the three published reports in one command."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
env = dict(os.environ, PYTHONPATH=str(root/'src'), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
jobs = [('run_cantilever_fe_benchmark.py', 'cantilever-fe/results.json'),
        ('run_hull_girder_fe_benchmark.py', 'hull-girder/hull-girder-fe.json'),
        ('run_stiffened_panel_fe_benchmark.py', 'stiffened-panel/results.json')]
for script, output in jobs:
    subprocess.run([sys.executable, str(root/'scripts'/script), '--output',
                    str(root/'docs/assets/benchmark-clouds'/output)],
                   cwd=root, env=env, check=True)
for script in ('build_verified_benchmark_reports.py', 'validate_published_benchmark_reports.py'):
    subprocess.run([sys.executable, str(root/'scripts'/script)], cwd=root, env=env, check=True)
