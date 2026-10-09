import json

import pytest

from tensorfem.abaqus_io import read_inp
from tensorfem.mesh_pipeline import (convert_model, diagnose_model, read_gmsh,
                                     read_with_meshio)
from tensorfem.modeldb import (BoundaryCondition, ConcentratedLoad, ElementBlock,
                               Material, ModelDB, Section)


def valid_quad():
    db=ModelDB(nodes={1:(0.,0.),2:(2.,0.),3:(2.,1.),4:(0.,1.)},
               elements=[ElementBlock("CPS4",[10],[[1,2,3,4]],"EALL")],
               node_sets={"FIX":{1,4}}, element_sets={"EALL":{10}},
               materials={"STEEL":Material("STEEL",(210e9,.3))},
               sections=[Section("EALL","STEEL","solid",(1.,))],
               boundaries=[BoundaryCondition("FIX",1,2,0.)],
               loads=[ConcentratedLoad(2,1,100.)], metadata={"length_unit":"m"})
    return db


def test_valid_model_report_and_json(tmp_path):
    report=diagnose_model(valid_quad()).require_valid()
    assert report.valid and report.metrics["minimum_jacobian"] == pytest.approx(.5)
    path=tmp_path/"diagnostics.json"; report.write_json(path)
    data=json.loads(path.read_text())
    assert data["valid"] and data["summary"]["errors"] == 0


@pytest.mark.parametrize("mutate,code",[
    (lambda d: d.elements[0].connectivity.__setitem__(0,[1,4,3,2]), "inverted_or_degenerate"),
    (lambda d: d.nodes.__setitem__(3,(1.,-1.)), "inverted_or_degenerate"),
    (lambda d: d.elements[0].connectivity.__setitem__(0,[1,2,2,4]), "repeated_element_node"),
    (lambda d: d.nodes.__setitem__(5,(0.,0.)), "duplicate_coordinates"),
    (lambda d: d.sections.clear(), "units_unspecified"),
])
def test_bad_mesh_is_reported(mutate,code):
    db=valid_quad()
    if code == "units_unspecified":
        # A missing unit is a warning, not a geometric false positive.
        db.metadata.clear(); mutate(db)
        report=diagnose_model(db)
        assert code in {x.code for x in report.diagnostics}
        assert report.valid
        return
    mutate(db); report=diagnose_model(db)
    assert code in {x.code for x in report.diagnostics}
    with pytest.raises(ValueError,match="diagnostics failed"): report.require_valid()


def test_orphan_and_nonmanifold_detection():
    db=valid_quad(); db.nodes.update({5:(2.,-1.),6:(0.,-1.),7:(2.,2.),8:(0.,2.),99:(9.,9.)})
    db.elements[0]=ElementBlock("CPS4",[10,11,12],[[1,2,3,4],[6,5,2,1],[1,2,7,8]],"EALL")
    db.element_sets["EALL"]={10,11,12}
    report=diagnose_model(db)
    codes={x.code for x in report.diagnostics}
    assert {"orphan_nodes","nonmanifold_face"} <= codes
    assert not report.valid


def test_material_section_and_analysis_checks():
    db=valid_quad(); db.sections[0].material="MISSING"
    db.loads.append(ConcentratedLoad("NOSET",1,1.))
    db.boundaries.append(BoundaryCondition("FIX",1,1,2.))
    codes={x.code for x in diagnose_model(db).diagnostics}
    assert {"missing_material","unknown_load_target","conflicting_boundary"} <= codes


def test_gmsh_41_ascii_and_conversion_round_trip(tmp_path):
    text="""$MeshFormat
4.1 0 8
$EndMeshFormat
$Nodes
1 4 1 4
2 1 0 4
1
2
3
4
0 0 0
2 0 0
2 1 0
0 1 0
$EndNodes
$Elements
1 1 10 10
2 1 3 1
10 1 2 3 4
$EndElements
"""
    source=tmp_path/"quad.msh"; source.write_text(text)
    imported=read_gmsh(source)
    assert imported.elements[0].connectivity == [[1,2,3,4]]
    # Add engineering definitions required by the fail-closed converter.
    imported.elements[0].name="EALL"; imported.element_sets["EALL"]={10}
    imported.node_sets["FIX"]={1,4}; imported.materials["M"]=Material("M",(1e6,.3))
    imported.sections=[Section("EALL","M","solid",(1.,))]
    imported.boundaries=[BoundaryCondition("FIX",1,2)]; imported.loads=[ConcentratedLoad(2,1,1.)]
    imported.metadata["length_unit"]="m"
    inp=tmp_path/"quad.inp"; vtk=tmp_path/"quad.vtk"
    convert_model(imported,inp); convert_model(imported,vtk)
    restored=read_inp(inp)
    assert restored.nodes == imported.nodes
    assert restored.elements[0].connectivity == imported.elements[0].connectivity
    assert "UNSTRUCTURED_GRID" in vtk.read_text()


def test_meshio_optional_adapter_gracefully_degrades(tmp_path):
    try:
        import meshio  # noqa: F401
    except ImportError:
        assert read_with_meshio(tmp_path/"not-needed.mesh") is None
