"""End-to-end qualified hemisphere workflow, evidence and deliverables."""
from pathlib import Path

from tensorfem.benchmark_cases import hemisphere_case, run_case, write_archive
from tensorfem.hemisphere_postprocess import (
    build_hemisphere_post,
    write_hemisphere_json,
    write_hemisphere_markdown,
    write_hemisphere_vtk,
)
from tensorfem.industrial_workflow import ModelSpec, WorkflowSpec, run_job
from tensorfem.spherical_shell import hemisphere_with_hole


root = Path("runs/hemisphere-industrial")
workflow = run_job(WorkflowSpec(ModelSpec(nphi=16, ntheta=16)), root)
solved = hemisphere_with_hole(16, 16)
post = build_hemisphere_post(solved)
deliverables = root / workflow.job_id / "deliverables"
deliverables.mkdir(parents=True, exist_ok=True)
write_hemisphere_json(deliverables / "result.json", post)
write_hemisphere_vtk(deliverables / "result.vtk", post)
write_hemisphere_markdown(deliverables / "validation.md", post)
archive = run_case(hemisphere_case(), tier="quick")
write_archive(archive, deliverables / "evidence.json")
print({
    "job_id": workflow.job_id,
    "passed": post.validation["passed"],
    "relative_error": post.validation["relative_error"],
    "deliverables": str(deliverables),
})
