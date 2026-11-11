#!/bin/bash
# Independent reproduction of m15010's repro_cmd in an isolated sandbox.
# Usage: run_repro.sh [--quick]   (default: quick)
set -u
MODE="${1:---quick}"
SB=/tmp/repro_j2
rm -rf "$SB"
mkdir -p "$SB/artifacts/j2_plastic_membrane"
cp -r /root/tp-agent/work/m15185/src "$SB/"
cp -r /root/tp-agent/work/m15185/benchmarks "$SB/"
mkdir -p "$SB/docs"
cd "$SB" || exit 1
{ S=$(date +%s); python3 benchmarks/j2_plastic_membrane/run.py $([ "$MODE" = "--quick" ] && echo --quick) ; echo "EXIT=$?"; echo "WALL_S=$(( $(date +%s)-S ))"; } > "$SB/run.log" 2>&1
echo done