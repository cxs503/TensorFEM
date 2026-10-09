#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from tensorfem.nonlinear_shell_robustness import run_nonlinear_shell_robustness

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
report = run_nonlinear_shell_robustness()
encoded = json.dumps(report, indent=2, sort_keys=True)+"\n"
args.output.parent.mkdir(parents=True, exist_ok=True)
temporary = args.output.with_suffix(args.output.suffix+".tmp")
temporary.write_text(encoded); temporary.replace(args.output)
print(json.dumps({"output": str(args.output), "passed": report["passed"],
                  "evidence_sha256": report["evidence_sha256"]}, indent=2))
raise SystemExit(0 if report["passed"] else 2)
