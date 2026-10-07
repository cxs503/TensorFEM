import importlib.util
import json
import pytest
import torch
from tensorfem.hemisphere_postprocess import build_hemisphere_post
from tensorfem.result_db import (ResultDB,hemisphere_result_db,iter_result_node_chunks,
    query_result_nodes,read_result_db,write_result_db)
from tensorfem.spherical_shell import hemisphere_with_hole


@pytest.fixture(scope="module")
def hemisphere_db():
    post=build_hemisphere_post(hemisphere_with_hole(16),length_unit="in",force_unit="lbf")
    return post,hemisphere_result_db(post)


def test_field_and_ordered_node_subset_query(hemisphere_db):
    post,db=hemisphere_db
    ids,values=db.query_nodes("displacement",[288,0,17])
    assert ids.tolist()==[288,0,17]
    assert torch.equal(values,post.displacement[torch.tensor([288,0,17])])
    with pytest.raises(KeyError,match="unknown node"):db.query_nodes("displacement",[9999])


def test_probe_extrema_and_balance_metrics_do_not_drift(hemisphere_db):
    post,db=hemisphere_db;probe=db.probes["loaded_equator_x"]
    assert probe==post.probe
    assert float(db.histories["computed_displacement"][0])==post.validation["computed_displacement"]
    assert float(db.histories["relative_error"][0])==post.validation["relative_error"]
    assert float(db.histories["normalized_force_balance_error"][0])==post.validation["normalized_force_balance_error"]
    assert torch.equal(db.node_fields["displacement"],post.displacement)
    assert float(torch.linalg.vector_norm(db.node_fields["displacement"],dim=1).max())==post.extrema["displacement_magnitude"]["max"]


def test_summary_checksum_and_optional_body_behavior(tmp_path,hemisphere_db):
    _,db=hemisphere_db;base=tmp_path/"case";written=write_result_db(base,db,chunk_rows=31)
    summary=json.loads((tmp_path/"case.json").read_text())
    assert summary["schema"]=="tensorfem.result-db.v1"
    assert summary["counts"]=={"nodes":289,"elements":256}
    assert summary["checksums"]==db.checksums()
    assert written==(importlib.util.find_spec("h5py") is not None)
    if written:
        loaded=read_result_db(base)
        assert loaded.checksums()==db.checksums()
        assert torch.equal(loaded.query_nodes("reaction_force",[0,288])[1],db.query_nodes("reaction_force",[0,288])[1])
        ids,subset=query_result_nodes(base,"displacement",[288,0,288])
        assert ids.tolist()==[288,0,288] and torch.equal(subset,db.node_fields["displacement"][ids])
        chunks=list(iter_result_node_chunks(base,"displacement",chunk_rows=37))
        assert torch.equal(torch.cat([x[1] for x in chunks]),db.node_fields["displacement"])
    else:
        with pytest.raises(RuntimeError,match="h5py"):read_result_db(base)


def test_integrity_validation_fails_closed():
    bad=ResultDB(torch.tensor([0]),torch.tensor([0]),torch.tensor([[1,1,1,1]]))
    with pytest.raises(ValueError,match="unknown node"):bad.checksums()
