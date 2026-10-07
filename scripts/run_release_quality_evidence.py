#!/usr/bin/env python3
"""Actually execute all release gates and atomically persist their evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def run(output: Path) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    commands = {
        "full_regression": [sys.executable, "-m", "pytest", "-q"],
        "api_audit": [sys.executable, "scripts/audit_public_api.py", "--json"],
        "offline_install": [sys.executable, "scripts/release_smoke.py", "--build"],
    }
    log_directory = output.parent/(output.stem+".logs")
    results = {}
    for name, command in commands.items():
        started = time.time()
        completed = subprocess.run(
            command, cwd=root, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        log_path = log_directory/f"{name}.log"
        _atomic_write(log_path, completed.stdout)
        # The full log is authoritative. The bounded tail makes a failed gate
        # diagnosable from the aggregate without inflating it indefinitely.
        summary = "\n".join(completed.stdout.rstrip().splitlines()[-40:])
        results[name] = {
            "passed": completed.returncode == 0,
            "returncode": completed.returncode,
            "elapsed_seconds": time.time()-started,
            "command": command,
            "log_path": str(log_path.relative_to(output.parent)),
            "output_summary": summary,
            "output_sha256": hashlib.sha256(
                completed.stdout.encode()).hexdigest(),
        }
    clean = {
        "schema": "tensorfem.release-quality-evidence/1.0",
        "full_regression": results["full_regression"]["passed"],
        "api_audit": results["api_audit"]["passed"],
        "offline_install": results["offline_install"]["passed"],
        "executions": results,
    }
    clean["passed"] = all(value["passed"] for value in results.values())
    report = {**clean, "evidence_sha256": hashlib.sha256(json.dumps(
        clean, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()}
    _atomic_write(output, json.dumps(report, indent=2, sort_keys=True)+"\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({"output": str(args.output), "passed": report["passed"],
                      "evidence_sha256": report["evidence_sha256"]}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
