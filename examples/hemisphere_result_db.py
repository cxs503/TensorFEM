"""Write the hemisphere ResultDB JSON index and optional HDF5 body."""
from pathlib import Path
from tensorfem.hemisphere_postprocess import build_hemisphere_post
from tensorfem.result_db import hemisphere_result_db,write_result_db
from tensorfem.spherical_shell import hemisphere_with_hole


if __name__=="__main__":
    post=build_hemisphere_post(hemisphere_with_hole(24))
    db=hemisphere_result_db(post);out=Path("hemisphere_resultdb")
    body=write_result_db(out,db,chunk_rows=64)
    ids,u=db.query_nodes("displacement",[0,16,272,288])
    print({"summary":str(out.with_suffix('.json')),"hdf5_body_written":body,
           "queried_node_ids":ids.tolist(),"queried_displacement":u.tolist()})
