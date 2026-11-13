#!/bin/bash
# Sequentially re-run each anchor's own runner to restore the figure artifacts
# its certificate declares but that were never committed.  One core at a time;
# per-anchor timeout so a heavy anchor cannot block the queue.
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src
LOG=/tmp/restore_artifacts.log
: > "$LOG"
for a in fem_beam_static fem_winkler_plate fem_plate_static fem_plate_modal \
         fem_hertz_contact fem_kirsch_plate fem_panel_buckling fem_plate_postbuckling \
         fem_structural_benchmarks fem_shell_modal; do
  runner=run.py
  [ -f "benchmarks/verified/$a/$runner" ] || runner=run_verification.py
  echo "=== $a ($runner) start $(date -Is)" >> "$LOG"
  start=$(date +%s)
  timeout 900 python "benchmarks/verified/$a/$runner" >> "$LOG" 2>&1
  rc=$?
  echo "=== $a rc=$rc elapsed=$(( $(date +%s) - start ))s" >> "$LOG"
done
echo "ALL DONE $(date -Is)" >> "$LOG"