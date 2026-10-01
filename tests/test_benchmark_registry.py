import json
import pytest
from tensorfem.benchmark_registry import _evidence, verification_report
from tensorfem.cli import main

@pytest.fixture(scope="module")
def report(): return verification_report()

def test_registry_traceable_and_below_three_percent(report):
    assert report["passed"] and report["summary"] == {"total":14,"passed":14}
    for item in report["results"]:
        assert item["source"].strip() and item["reference"] != 0
        assert item["error"] < item["tolerance"] <= .03 and item["passed"] is True

def test_zero_reference_rejected_and_boundary_strict():
    with pytest.raises(ValueError,match="zero-reference"): _evidence("x","x","x","x","s",0.,0.)
    assert not _evidence("x","x","x","x","s",1.03,1.,.03).passed

def test_verify_cli_writes_json(monkeypatch,tmp_path,report):
    path=tmp_path/"evidence.json"; monkeypatch.setattr("sys.argv",["tensorfem","verify","--output",str(path)])
    with pytest.raises(SystemExit) as exc: main()
    assert exc.value.code == 0
    assert json.loads(path.read_text())["summary"] == report["summary"]
