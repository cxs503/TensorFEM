from pathlib import Path
from tensorfem.mesh_project import run_mesh_project
here=Path(__file__).parent
result=run_mesh_project(here/'projects/truss_project.json',here/'project-runs')
print(result['job_id'],result['displacement'])
