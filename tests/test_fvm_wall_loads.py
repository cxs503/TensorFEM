import copy
import hashlib
import json
from pathlib import Path
import pytest
import torch
from tensorfem.fvm_wall_loads import read_wall_loads

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'docs/assets/fvm-wall-exchange/study.json'
SOURCE=ROOT/'docs/assets/fvm-wall-exchange/raw-cpu-wall-facets.json'

def source():
    """Load the vendored TensorFVM CPU wall-load artifact.

    The artifact is vendored into this repository (rather than read from a
    sibling checkout) so this test always runs: an unrun test is worse than
    no test.  ``study.json`` records the sha256 of the upstream file, so a
    drifted copy fails loudly here instead of silently testing stale data.
    """
    digest=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    expected=json.loads(STUDY.read_text())['cases']['cpu']['source_raw_sha256']
    assert digest==expected,'vendored wall-load artifact drifted from study.json'
    return json.loads(SOURCE.read_text())

def test_actual_pressure_viscous_force_and_clocks():
    raw=source();wall=read_wall_loads(raw)
    torch.testing.assert_close(wall.total_force_N,wall.pressure_force_N+wall.viscous_force_N)
    assert wall.points_m.shape==(96,3)
    assert abs(wall.time_s-wall.viscous_time_s-raw['config']['time_step'])<1e-12
    saved=copy.deepcopy(raw);wall.total_force_N[0,0]+=1
    assert raw==saved

@pytest.mark.parametrize('kind',['force','clock','area','units','normal','owner','qualification'])
def test_reject_inconsistent_actual_wall_metadata(kind):
    raw=source()
    if kind=='force':raw['facets'][0]['total_force_on_body_N'][0][0]+=1
    elif kind=='clock':raw['viscous_evaluation_time_s']=raw['time_s']
    elif kind=='area':raw['facets'][0]['area_m2']=-1
    elif kind=='units':raw['units']['length']='mm'
    elif kind=='normal':raw['facets'][0]['normal_fluid'][0]=[2.,0.,0.]
    elif kind=='owner':raw['force_owner']='fluid+DEM'
    else:raw['physical_accuracy_qualified']=True
    with pytest.raises(ValueError):read_wall_loads(raw)
