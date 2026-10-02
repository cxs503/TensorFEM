import math
from pathlib import Path
from tensorfem.project_workflow import *
units={'length':'in','force':'lbf','displacement':'in','rotation':'radian','stress':'psi'}
project=Project('hemisphere qualification',ProjectModel('shell','hemisphere_18deg',
 {'nphi':12,'ntheta':12,'drilling_factor':1e-6},'section'),
 (Material('elastic','isotropic_elastic',{'young':6.825e7,'poisson':.3}),),
 (Section('section','shell','elastic',{'thickness':.04}),),
 (ProjectStep('static','linear_static',('load',),('symmetry',),('field',)),),
 (Load('load','point_load','built_in_hemisphere_equator'),),
 (Constraint('symmetry','symmetry','built_in_quarter_and_gauge'),),
 (OutputRequest('field',('displacement','reaction')),),units)
result=run_project(project,Path('project-runs'))
print(result['job_id'],result['metadata'])
