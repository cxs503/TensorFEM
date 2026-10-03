from pathlib import Path
import os
import subprocess
import sys

import torch

from tensorfem.abaqus_io import read_inp, write_inp
from tensorfem.gmsh_io import read_msh
from tensorfem.modeldb import to_truss_model
from tensorfem.result_io import write_hdf5, write_vtk
from tensorfem.solvers import solve_linear_static


INP = """*Heading
bar benchmark
*Node, nset=ALLNODES
10, 0., 0.
20, 2., 0.
*Element, type=T2D2, elset=BARS
100, 10, 20
*Nset, nset=LEFT
10
*Material, name=STEEL
*Elastic
200000., 0.3
*Truss Section, elset=BARS, material=STEEL
3.
*Boundary
LEFT, 1, 2
20, 2, 2
*Cload
20, 1, 1200.
"""


def test_inp_to_solver_to_vtk_benchmark(tmp_path):
    source = tmp_path / "bar.inp"; source.write_text(INP)
    db = read_inp(source)
    model, node_ids = to_truss_model(db)
    result = solve_linear_static(model)
    exact = 1200. * 2. / (200000. * 3.)
    error = abs(float(result.displacement[2]) - exact) / exact
    assert error < .03
    displacement = result.displacement.reshape(-1, 2)
    output = tmp_path / "bar.vtk"
    write_vtk(output, db, point_data={"displacement": displacement},
              cell_data={"axial_stress": result.axial_stress})
    text = output.read_text()
    assert "POINTS 2 double" in text and "VECTORS displacement double" in text
    assert node_ids == [10, 20]


def test_inp_sets_generate_and_solid_subset(tmp_path):
    text = """*Node
1,0,0
2,1,0
3,1,1
4,0,1
*Element,type=CPS4,elset=EALL
8,1,2,3,4
*Nset,nset=NALL,generate
1,4,1
*Elset,elset=EALL
8
*Material,name=M
*Elastic
10,0.25
*Solid Section,elset=EALL,material=M
1.0
"""
    p = tmp_path / "solid.inp"; p.write_text(text)
    db = read_inp(p)
    assert db.node_sets["NALL"] == {1, 2, 3, 4}
    assert db.elements[0].element_type == "CPS4"


def test_supported_inp_round_trip(tmp_path):
    source = tmp_path / "one.inp"; source.write_text(INP)
    first = read_inp(source)
    target = tmp_path / "two.inp"; write_inp(target, first)
    second = read_inp(target)
    assert second.nodes == first.nodes
    assert second.elements[0].connectivity == first.elements[0].connectivity
    assert second.node_sets == first.node_sets
    assert second.materials["STEEL"].elastic == first.materials["STEEL"].elastic
    assert second.loads == first.loads and second.boundaries == first.boundaries


def test_gmsh_v2_ascii_import(tmp_path):
    msh = """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
2
1 0 0 0
2 1 0 0
$EndNodes
$Elements
1
7 1 2 4 1 1 2
$EndElements
"""
    p = tmp_path / "line.msh"; p.write_text(msh)
    db = read_msh(p)
    assert db.elements[0].connectivity == [[1, 2]]


def test_hdf5_is_optional(tmp_path):
    p = tmp_path / "bar.inp"; p.write_text(INP); db = read_inp(p)
    ok = write_hdf5(tmp_path / "r.h5", db, displacement=torch.zeros(2, 2))
    assert isinstance(ok, bool)
    assert not ok or (tmp_path / "r.h5").exists()


def test_inp_to_vtk_example_help():
    script = Path(__file__).parents[1] / "examples" / "inp_to_vtk.py"
    env={**os.environ,"PYTHONPATH":str(Path(__file__).parents[1]/"src")}
    proc = subprocess.run([sys.executable, str(script), "--help"],
                          text=True, capture_output=True, check=False,env=env)
    assert proc.returncode == 0
    assert "input Abaqus .inp file" in proc.stdout
