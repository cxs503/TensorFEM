#!/usr/bin/env python3
"""Validate a marine benchmark report, including its tamper-evident hash."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path

def main() -> None:
    p = Path(sys.argv[1]); report = json.loads(p.read_text())
    assert report["schema"] == "tensorfem.marine-benchmark-report/1.0"
    digest = report.pop("evidence_sha256")
    assert hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == digest
    cases = report["cases"]
    assert cases and report["summary"]["total"] == len(cases)
    for case in cases.values():
        result = case["result"]; error = float(result["relative_error"])
        assert error <= .03 and result["status"] == "qualified"
        assert case["problem"] and case["reference_source"] and case["unit"]
        assert "stress" in case["field_summary"] and "displacement" in case["field_summary"]
    print(f"marine benchmark report valid: {len(cases)} cases, all <= 3%")

if __name__ == "__main__": main()
