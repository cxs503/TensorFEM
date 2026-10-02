import json
import pytest
from tensorfem.benchmark_registry import _evidence, verification_report
from tensorfem.cli import main

@pytest.fixture(scope="module")
def report(): return verification_report()

def test_registry_traceable_and_below_three_percent(report):
    expected = {
        "truss.bar", "frame.cantilever", "continuum.cook", "beam.timoshenko",
        "buckling.euler", "dynamics.newmark", "solid.hex8", "plate.mindlin",
        "nonlinear.arch", "contact.spring", "modal.cantilever",
        "shell.patch_energy", "cohesive.fracture_energy", "plasticity.j2",
        "sparse.axial_bar", "thermal.rod_convection", "dynamics.explicit",
        "nonlinear.finite_bar", "solid.tet10_patch", "plasticity.tet4_j2",
    }
    assert report["passed"]
    assert report["summary"] == {"total": len(expected), "passed": len(expected)}
    assert {item["id"] for item in report["results"]} == expected
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
