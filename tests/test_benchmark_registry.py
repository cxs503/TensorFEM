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
        "marine.hull_girder.deflection", "marine.hull_girder.moment",
        "marine.stiffened_panel.navier", "marine.hydrostatics.displacement",
        "marine.hydrostatics.gm", "marine.hydrostatics.restoring",
        "marine.morison.base_shear", "marine.morison.overturning",
        "marine.plate_buckling.square", "marine.plate_buckling.long_panel",
        "marine.plate_buckling.longitudinally_stiffened",
        "marine.hull_ultimate.initial_yield", "marine.hull_ultimate.full_plastic",
        "marine.fatigue.hot_spot.linear", "marine.fatigue.hot_spot.quadratic",
        "marine.fatigue.sn.single_block", "marine.fatigue.miner.multi_block",
        "marine.prestress_buckling.uniaxial", "marine.prestress_buckling.biaxial",
        "marine.postbuckling.imperfect_path", "marine.hull_progressive.moment",
        "marine.fatigue_advanced.hot_spot.path_linear",
        "marine.fatigue_advanced.rainflow.triangle_count",
        "marine.fatigue_advanced.miner.variable_amplitude",
        "marine.fatigue_advanced.paris.m2_closed_form",
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
