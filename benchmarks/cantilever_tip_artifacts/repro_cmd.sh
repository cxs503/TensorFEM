#!/bin/bash
# Reproduce CASE-CANTILEVER-TIP benchmark
# Usage: bash repro_cmd.sh
cd "$(dirname "$0")/.."
python3 benchmarks/cantilever_tip_deflection.py
