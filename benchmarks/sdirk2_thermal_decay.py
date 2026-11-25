"""Independent temporal and spatial refinement of a cooling rod with fixed cold ends."""
import json
import math
from pathlib import Path
import subprocess

import torch
from tensorfem.thermal import ThermalModel, assemble_thermal, solve_transient_thermal


def measure(n,steps,driven=False):
    x = torch.linspace(0.,1.,n+1,dtype=torch.float64)
    ids = torch.arange(n)
    model = ThermalModel(x[:,None],torch.stack((ids,ids+1),1),1.,1.,1.,
                         torch.tensor([0,n]),torch.zeros(2,dtype=x.dtype))
    initial = torch.sin(math.pi*x)
    initial[[0,-1]] = 0
    times = torch.linspace(0.,.1,steps+1,dtype=x.dtype)
    # Uniform linear elements have stiffness eigenvalue 2*(1-cos(q))/h
    # and consistent mass eigenvalue h*(2+cos(q))/3.
    q = math.pi/n
    rate = 12*n*n*math.sin(q/2)**2/(2+math.cos(q))
    if driven:
        # A scalar modal amplitude a(t)=1-exp(-t) satisfies a'+rate*a=f.
        # The integral of f is rate*t+(1-rate)*(1-exp(-t)).
        shape = assemble_thermal(model).capacity@initial
        solution = solve_transient_thermal(model,times,0*initial,method="sdirk2",sparse=True,
            nodal_heat_history=lambda t:(rate+(1-rate)*torch.exp(-t))*shape)
        amplitude = -torch.expm1(-times)
        integral = (rate*times+(1-rate)*amplitude)[:,None]*shape
        supplied = (times.diff()[:,None]*solution.additional_stage_power).cumsum(0)
        return dict(cells=n,steps=steps,
                    temporal_error_K=float(abs(solution.temperature-amplitude[:,None]*initial).max()),
                    nodal_input_error_J=float(abs(supplied-integral[1:]).max()),
                    free_residual_J=float(abs(solution.nodal_reaction_heat_steps()[:,1:-1]).max()))
    solution = solve_transient_thermal(model,times,initial,method="sdirk2",sparse=True)
    discrete = torch.exp(-rate*times[:,None])*initial
    continuum = torch.exp(-math.pi**2*times[:,None])*initial
    end_heat = -solution.nodal_reaction_heat_steps()[:,model.fixed_nodes].cumsum(0)
    exact_heat = -torch.expm1(-math.pi**2*times[1:])/math.pi
    return dict(cells=n,steps=steps,temporal_error_K=float(abs(solution.temperature-discrete).max()),
                temperature_error_K=float(abs(solution.temperature-continuum).max()),
                end_heat_error_J=float(abs(end_heat-exact_heat[:,None]).max()),
                free_residual_J=float(abs(solution.nodal_reaction_heat_steps()[:,1:-1]).max()))


def moving_wall(n,steps,discrete_reference):
    # T=exp(t)*(cosh(x)+0.2*sinh(x)) solves T_t=T_xx. End powers
    # are -T_x(0) and T_x(1) in the inward heat convention.
    x = torch.linspace(0.,1.,n+1,dtype=torch.float64)
    ids = torch.arange(n)
    shape = torch.cosh(x)+.2*torch.sinh(x)
    fixed = torch.tensor([0,n])
    model = ThermalModel(x[:,None],torch.stack((ids,ids+1),1),1.,1.,1.,fixed,shape[fixed])
    inward = torch.stack((-x.new_tensor(.2),torch.sinh(x[-1])+.2*torch.cosh(x[-1])))
    if discrete_reference:
        # Remove spatial error from the time study: prescribed exponential
        # growth gives the exact semidiscrete profile (K+M)u=0 on free rows.
        m = assemble_thermal(model)
        operator = m.capacity+m.conductivity
        shape[1:-1] = torch.linalg.solve(operator[1:-1,1:-1],-operator[1:-1][:,fixed]@shape[fixed])
        inward = (operator@shape)[fixed]
    times = torch.linspace(0.,.5,steps+1,dtype=x.dtype)
    solution = solve_transient_thermal(model,times,shape,method="sdirk2",sparse=True,
                                       fixed_temperature_history=lambda t:torch.exp(t)*shape[fixed])
    reactions = solution.nodal_reaction_heat_steps()
    return dict(cells=n,steps=steps,temperature_error_K=float(abs(solution.temperature-
                    torch.exp(times[:,None])*shape).max()),end_heat_error_J=float(abs(reactions[:,fixed].cumsum(0)-
                    torch.expm1(times[1:,None])*inward).max()),free_residual_J=float(abs(reactions[:,1:-1]).max()))


if __name__=="__main__":
    rows = []
    for study in ("time","space","driven","wall_time","wall_space"):
        previous = None
        levels = (1,2,4,8,16,32,64,128) if study=="wall_time" else (1,2,4,8)
        for level in levels:
            if study=="wall_space":
                row = moving_wall(8*level,256,False)
            elif study=="wall_time":
                row = moving_wall(16,4*level,True)
            else:
                row = measure(8*level,256) if study=="space" else measure(32,4*level,driven=study=="driven")
            metrics = ["temporal_error_K"] if study=="time" else ["temperature_error_K","end_heat_error_J"]
            if study=="driven":
                metrics = ["temporal_error_K","nodal_input_error_J"]
            orders = {k:math.log2(previous[k]/row[k]) for k in metrics} if previous else {}
            row.update(study=study,orders=orders,passed=(row["free_residual_J"]<1e-11 and
                       (level<levels[-1] or min(orders.values())>1.9)))
            rows.append(row)
            previous = row
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=revision,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
