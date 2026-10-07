from pathlib import Path
from tensorfem.industrial_workflow import ModelSpec,WorkflowSpec,run_job
result=run_job(WorkflowSpec(ModelSpec(nphi=12,ntheta=12)),Path('runs'))
print(result.job_id,result.metadata['probe_displacement'],result.metadata['relative_error'])
