import json
import torch
import pytest
from tensorfem.hemisphere_postprocess import (
    build_hemisphere_post,read_hemisphere_json,write_hemisphere_json,
    write_hemisphere_markdown,write_hemisphere_vtk,
)
from tensorfem.spherical_shell import hemisphere_with_hole


@pytest.fixture(scope="module")
def solved():
    result=hemisphere_with_hole(16)
    return result,build_hemisphere_post(result,length_unit="in",force_unit="lbf")


def test_coordinate_dof_probe_and_path_mapping(solved):
    result,post=solved;n=len(result.nodes)
    assert torch.equal(post.displacement,result.solution.reshape(n,6)[:,:3])
    assert torch.equal(post.rotation,result.solution.reshape(n,6)[:,3:])
    assert torch.allclose(post.deformed,post.nodes+post.displacement,rtol=0,atol=0)
    a=16*17
    assert post.probe["node_index"]==a and post.probe["dof"]==0
    assert post.probe["value"]==float(result.solution[6*a])==result.displacement
    assert [x["node_index"] for x in post.path]==[i*17 for i in range(17)]


def test_equilibrium_and_qualification_report_pass(solved):
    _,post=solved;v=post.validation
    assert v["relative_error"]<.03 and v["free_residual_norm"]<1e-7
    assert v["normalized_force_balance_error"]<1e-7
    assert v["normalized_moment_balance_error"]<.03
    assert v["passed"]


def test_json_vtk_and_markdown_roundtrip(tmp_path,solved):
    _,post=solved
    js=tmp_path/"hemisphere.json";vtk=tmp_path/"hemisphere.vtk";md=tmp_path/"report.md"
    write_hemisphere_json(js,post);write_hemisphere_vtk(vtk,post);write_hemisphere_markdown(md,post)
    payload=read_hemisphere_json(js)
    assert payload["metadata"]["units"]=={"length":"in","force":"lbf","rotation":"radian","young_modulus":"lbf/in^2"}
    assert torch.equal(torch.tensor(payload["nodes"],dtype=post.nodes.dtype),post.nodes)
    assert torch.equal(torch.tensor(payload["displacement"],dtype=post.nodes.dtype),post.displacement)
    text=vtk.read_text();lines=text.splitlines();point_line=lines.index(f"POINTS {len(post.nodes)} double")
    exported=torch.tensor([[float(v) for v in line.split()] for line in lines[point_line+1:point_line+1+len(post.nodes)]],dtype=post.nodes.dtype)
    assert torch.equal(exported,post.nodes)
    assert f"CELLS {len(post.elements)}" in text and "VECTORS displacement double" in text
    assert "VECTORS reaction_force double" in text and "Result: **PASS**" in md.read_text()


def test_reader_fails_closed_on_wrong_schema(tmp_path):
    path=tmp_path/"bad.json";path.write_text(json.dumps({"metadata":{"schema":"wrong"}}))
    with pytest.raises(ValueError,match="schema"):read_hemisphere_json(path)
