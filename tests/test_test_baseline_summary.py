"""Tests for the test-baseline condenser (scripts/summarize_test_baseline.py).

The condenser is the tool that makes "test baseline drift" a machine-checkable
claim instead of a hand-written prose comparison, so it needs its own coverage:
outcome precedence, signature stability, and regression detection.
"""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_test_baseline.py"
_spec = importlib.util.spec_from_file_location("summarize_test_baseline", SCRIPT)
baseline_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_tool)


def _junit(tmp_path: Path, cases: list[str]) -> Path:
    xml = (
        '<testsuites><testsuite name="pytest">'
        + "".join(cases)
        + "</testsuite></testsuites>"
    )
    path = tmp_path / "junit.xml"
    path.write_text(xml)
    return path


def _case(name: str, body: str = "", cls: str = "tests.test_demo") -> str:
    return f'<testcase classname="{cls}" name="{name}" time="0.5">{body}</testcase>'


def test_outcome_precedence_prefers_worst_tag(tmp_path):
    path = _junit(
        tmp_path,
        [
            _case("ok"),
            _case("bad", "<failure message='boom'/>"),
            _case("err", "<error message='kaput'/>"),
            _case("skip", '<skipped type="pytest.skip" message="no dep"/>'),
            _case("xf", '<skipped type="pytest.xfail" message="known"/>'),
        ],
    )
    record = baseline_tool.collect(path)
    outcomes = {test["nodeid"]: test["outcome"] for test in record["tests"]}
    assert outcomes["tests.test_demo::ok"] == "passed"
    assert outcomes["tests.test_demo::bad"] == "failed"
    assert outcomes["tests.test_demo::err"] == "error"
    assert outcomes["tests.test_demo::skip"] == "skipped"
    assert outcomes["tests.test_demo::xf"] == "xfailed"
    assert record["totals"] == {
        "collected": 5,
        "passed": 1,
        "failed": 1,
        "error": 1,
        "skipped": 1,
        "xfailed": 1,
    }


def test_signature_is_order_independent(tmp_path):
    cases = [_case("a"), _case("b", "<failure message='x'/>")]
    first = baseline_tool.collect(_junit(tmp_path, cases))
    second = baseline_tool.collect(_junit(tmp_path, list(reversed(cases))))
    assert first["signature"] == second["signature"]


def test_diff_flags_only_new_reds(tmp_path):
    before = baseline_tool.collect(
        _junit(tmp_path, [_case("green"), _case("already_red", "<failure/>")])
    )
    after = baseline_tool.collect(
        _junit(tmp_path, [_case("green", "<failure message='new'/>"), _case("already_red", "<failure/>")])
    )
    report = baseline_tool.diff(before, after)
    assert [r["nodeid"] for r in report["regressions"]] == ["tests.test_demo::green"]
    assert report["recoveries"] == []
    assert not report["identical"]


def test_check_mode_exits_non_zero_on_regression(tmp_path, capsys):
    base = tmp_path / "baseline.json"
    before = baseline_tool.collect(_junit(tmp_path, [_case("green")]))
    base.write_text(json.dumps(before))
    after_path = _junit(tmp_path, [_case("green", "<failure message='drift'/>")])
    code = baseline_tool.main(
        [str(after_path), "--baseline", str(base), "--check", "--output", str(tmp_path / "out.json")]
    )
    assert code == 1
    assert "REGRESSION tests.test_demo::green" in capsys.readouterr().out


def test_check_mode_is_clean_when_identical(tmp_path):
    base = tmp_path / "baseline.json"
    path = _junit(tmp_path, [_case("green"), _case("red", "<failure/>")])
    base.write_text(json.dumps(baseline_tool.collect(path)))
    assert baseline_tool.main([str(path), "--baseline", str(base), "--check"]) == 0