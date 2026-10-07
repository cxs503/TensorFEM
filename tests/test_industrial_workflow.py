import hashlib,json
import pytest
from tensorfem.industrial_workflow import ModelSpec,StepSpec,WorkflowSpec,deterministic_job_id,run_job
def spec():return WorkflowSpec(ModelSpec(nphi=4,ntheta=4),StepSpec())
def test_deterministic_job_result_and_replay(tmp_path):
 s=spec();a=run_job(s,tmp_path);b=run_job(s,tmp_path)
 assert a.job_id==b.job_id==deterministic_job_id(s) and b.metadata['replayed']
 assert a.displacement==b.displacement and a.reaction==b.reaction
 d=tmp_path/a.job_id;status=json.loads((d/'job.json').read_text());assert status['status']=='completed'
 assert status['result_sha256']==hashlib.sha256((d/'result.json').read_bytes()).hexdigest()
 assert a.metadata['procedure']=='linear_static' and a.metadata['kernel_version']
def test_interrupted_running_job_restarts(tmp_path):
 s=spec();r=run_job(s,tmp_path);d=tmp_path/r.job_id
 status=json.loads((d/'job.json').read_text());status['status']='running';(d/'job.json').write_text(json.dumps(status))
 restarted=run_job(s,tmp_path);assert not restarted.metadata['replayed']
 assert json.loads((d/'job.json').read_text())['status']=='completed'
def test_failure_and_tamper_fail_closed(tmp_path):
 bad=WorkflowSpec(ModelSpec(nphi=1,ntheta=4),StepSpec());jid=deterministic_job_id(bad)
 with pytest.raises(ValueError):run_job(bad,tmp_path)
 assert json.loads((tmp_path/jid/'job.json').read_text())['status']=='failed'
 good=spec();r=run_job(good,tmp_path);d=tmp_path/r.job_id;(d/'result.json').write_text('{}')
 with pytest.raises(RuntimeError,match='checksum'):run_job(good,tmp_path)
 assert json.loads((d/'job.json').read_text())['status']=='failed'
