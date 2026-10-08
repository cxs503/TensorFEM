import json

from tensorfem.marine_benchmark_report import build_report


def test_marine_report_is_complete_and_fail_closed():
    report = build_report()
    assert report["schema"] == "tensorfem.marine-benchmark-report/1.0"
    assert report["summary"]["total"] >= 35
    assert report["summary"]["blocked"] == 0
    assert report["summary"]["qualified"] == 0
    assert report["summary"]["reference_only"] == report["summary"]["total"]
    assert all(item["results"]["status"] == "reference-only" for item in report["cases"])
    assert all(item["passed"] and item["error"] < 0.03 for item in report["cases"])
    assert all(item["problem"] and item["calculation_process"] for item in report["cases"])
    assert len(report["evidence_sha256"]) == 64
