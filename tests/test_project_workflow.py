import json,math,pytest
from dataclasses import asdict
from tensorfem.project_workflow import *
U={'length':'in','force':'lbf','displacement':'in','rotation':'radian','stress':'psi'}
def project(case,params,procedure):
 return Project('demo',ProjectModel('model',case,params,'shell'),(Material('steel','elastic',{'young':1.}),),(Section('shell','shell','steel',{'thickness':1.}),),(ProjectStep('step',procedure,('load',),('bc',),('field',)),),(Load('load','case','built_in'),),(Constraint('bc','case','built_in'),),(OutputRequest('field',('displacement','reaction')),),U)
@pytest.mark.parametrize('p',[project('hemisphere_18deg',{'nphi':3,'ntheta':3,'drilling_factor':1e-6},'linear_static'),project('pinched_cylinder',{'n':3},'linear_static'),project('large_rotation_shell',{'n':1,'angle':math.pi/4},'nonlinear_static')])
def test_allow_list_cases_execute_and_replay(p,tmp_path):
 a=run_project(p,tmp_path);b=run_project(p,tmp_path);assert a['job_id']==b['job_id'] and b['metadata']['replayed'];assert a['metadata']['kernel']
def test_unknown_fields_and_code_execution_are_rejected(tmp_path):
 p=project('pinched_cylinder',{'n':3},'linear_static');path=tmp_path/'p.json';raw=asdict(p);raw['python']='__import__("os").system("false")';path.write_text(json.dumps(raw))
 with pytest.raises(ValueError,match='unknown'):project_from_json(path)
 bad=project('evil.module:function',{'n':3},'linear_static')
 with pytest.raises(ValueError,match='allow-listed'):run_project(bad,tmp_path)
 assert json.loads((tmp_path/project_id(bad)/'job.json').read_text())['status']=='failed'
