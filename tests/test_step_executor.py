import json
import torch
from tensorfem.modal import uniform_beam_matrices
from tensorfem.model import TrussModel
from tensorfem.result_db import write_result_db
from tensorfem.step_executor import StepExecutor,StepSpec,execution_result_db
from tensorfem.thermal import ThermalModel

D=torch.float64


def project_steps():
    x=torch.tensor([[0.],[1.],[2.]],dtype=D);conn=torch.tensor([[0,1],[1,2]])
    thermal=ThermalModel(x,conn,torch.tensor(5.,dtype=D),torch.tensor(1.,dtype=D),
        torch.tensor(1.,dtype=D),torch.tensor([0,2]),torch.tensor([100.,20.],dtype=D))
    truss=TrussModel(torch.tensor([[0.,0.],[1.,0.]],dtype=D),torch.tensor([[0,1]]),
        torch.tensor(100.,dtype=D),torch.tensor(2.,dtype=D),torch.tensor([0.,0.,10.,0.],dtype=D),
        torch.tensor([0,1,3]))
    K,M=uniform_beam_matrices(1.,4,2.,3.)
    return (
        StepSpec("heat","thermal_steady",{"model":thermal}),
        StepSpec("expansion","thermoelastic_bar",{"nodes":x,"elements":conn,
            "young_modulus":200.,"area":1.,"expansion_coefficient":1e-5,
            "reference_temperature":20.,"fixed_nodes":torch.tensor([0])},
            {"temperature":"heat.temperature"}),
        StepSpec("static","linear_static",{"model":truss}),
        StepSpec("modes","modal",{"stiffness":K,"mass":M,"constrained_dofs":[0,1],"modes":2}),
    )


def test_three_kernel_project_dependency_progress_and_resultdb(tmp_path):
    events=[];run=StepExecutor().execute(project_steps(),progress=events.append)
    assert run.completed and [x.kind for x in run.steps]==["thermal_steady","thermoelastic_bar","linear_static","modal"]
    assert torch.allclose(run.steps[0].fields["temperature"],torch.tensor([100.,60.,20.],dtype=D))
    assert torch.allclose(run.steps[1].fields["displacement"],torch.tensor([0.,6e-4,8e-4],dtype=D),rtol=0,atol=1e-15)
    assert run.steps[2].fields["displacement"][2]==.05 and len(run.steps[3].fields["frequencies_hz"])==2
    assert [e["event"] for e in events]==[x for _ in run.steps for x in ("start","complete")]
    db=execution_result_db(run);db.validate();write_result_db(tmp_path/"steps",db)
    summary=json.loads((tmp_path/"steps.json").read_text())
    assert summary["job_metadata"]["completed"] and "heat.temperature" in summary["fields"]["history"]


def test_checkpoint_resume_skips_completed_steps(tmp_path):
    path=tmp_path/"checkpoint.json";kernels=dict(StepExecutor().kernels)
    kernels["modal"]=lambda _inputs:(_ for _ in ()).throw(RuntimeError("interrupted worker"))
    failed=StepExecutor(kernels).execute(project_steps(),checkpoint=path)
    assert not failed.completed and failed.steps[-1].diagnostic=="RuntimeError: interrupted worker"
    resumed=StepExecutor().execute(project_steps(),checkpoint=path,resume=True)
    assert resumed.completed and resumed.resumed_steps==3 and len(resumed.steps)==4
    assert resumed.steps[0].fields["temperature"].dtype==D


def test_unsupported_coupling_and_bad_dependency_fail_closed():
    unsupported=StepExecutor().execute([StepSpec("coupled","fully_coupled_thermomechanical",{})])
    assert not unsupported.completed and "unsupported step kind" in unsupported.steps[-1].diagnostic
    bad=StepExecutor().execute([StepSpec("bad","linear_static",{}, {"model":"missing.displacement"})])
    assert not bad.completed and "unknown step dependency" in bad.steps[-1].diagnostic
