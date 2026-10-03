"""Thermal -> expansion, static, and modal multi-step project."""
from pathlib import Path
import torch
from tensorfem.modal import uniform_beam_matrices
from tensorfem.model import TrussModel
from tensorfem.result_db import write_result_db
from tensorfem.step_executor import StepExecutor,StepSpec,execution_result_db
from tensorfem.thermal import ThermalModel

D=torch.float64;x=torch.tensor([[0.],[1.],[2.]],dtype=D);e=torch.tensor([[0,1],[1,2]])
thermal=ThermalModel(x,e,torch.tensor(5.),torch.tensor(1.),torch.tensor(1.),
    torch.tensor([0,2]),torch.tensor([100.,20.],dtype=D))
truss=TrussModel(torch.tensor([[0.,0.],[1.,0.]],dtype=D),torch.tensor([[0,1]]),
    torch.tensor(100.),torch.tensor(2.),torch.tensor([0.,0.,10.,0.],dtype=D),torch.tensor([0,1,3]))
K,M=uniform_beam_matrices(1.,4,2.,3.)
steps=[StepSpec("heat","thermal_steady",{"model":thermal}),
 StepSpec("expansion","thermoelastic_bar",{"nodes":x,"elements":e,"young_modulus":200.,"area":1.,
  "expansion_coefficient":1e-5,"reference_temperature":20.,"fixed_nodes":torch.tensor([0])},
  {"temperature":"heat.temperature"}),
 StepSpec("static","linear_static",{"model":truss}),
 StepSpec("modes","modal",{"stiffness":K,"mass":M,"constrained_dofs":[0,1],"modes":2})]
if __name__=="__main__":
 result=StepExecutor().execute(steps,checkpoint="multistep.checkpoint.json",progress=print)
 write_result_db(Path("multistep_results"),execution_result_db(result))
 print({s.name:{"success":s.success,"seconds":s.elapsed_seconds} for s in result.steps})
