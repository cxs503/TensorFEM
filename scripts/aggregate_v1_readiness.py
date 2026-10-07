#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from tensorfem.v1_evidence_aggregator import aggregate_v1_evidence

parser = argparse.ArgumentParser()
parser.add_argument("--sparse-report", type=Path)
parser.add_argument("--release-report", type=Path)
parser.add_argument("--contact-report", type=Path)
parser.add_argument("--nonlinear-shell-report", type=Path)
parser.add_argument("--panel-4", type=Path)
parser.add_argument("--panel-8", type=Path)
parser.add_argument("--panel-12", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
panels = {n: path for n, path in ((4, args.panel_4), (8, args.panel_8),
                                  (12, args.panel_12)) if path is not None}
report = aggregate_v1_evidence(
    sparse_report=args.sparse_report, release_report=args.release_report,
    contact_report=args.contact_report,
    nonlinear_shell_report=args.nonlinear_shell_report,
    panel_directories=panels)
encoded = json.dumps(report, indent=2, sort_keys=True)+"\n"
args.output.parent.mkdir(parents=True, exist_ok=True)
temporary = args.output.with_suffix(args.output.suffix+".tmp")
temporary.write_text(encoded); temporary.replace(args.output)
print(json.dumps({"output": str(args.output),
                  "ready_for_1_0": report["readiness"]["ready_for_1_0"],
                  "blockers": report["readiness"]["blockers"],
                  "evidence_sha256": report["evidence_sha256"]}, indent=2))
