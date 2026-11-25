"""Independent temporal and spatial refinement of a cooling rod with fixed cold ends."""
import json
import math
from pathlib import Path
import subprocess

import torch
from tensorfem.thermal import ThermalModel, solve_transient_thermal


def measure(n,steps):
    x = torch.linspace(0.,1.,n+1,dtype=torch.float64)
    ids = torch.arange(n)
    model = ThermalModel(x[:,None],torch.stack((ids,ids+1),1),1.,1.,1.,
                         torch.tensor([0,n]),torch.zeros(2,dtype=x.dtype))
    initial = torch.sin(math.pi*x)
    initial[[0,-1]] = 0
    times = torch.linspace(0.,.1,steps+1,dtype=x.dtype)
    solution = solve_transient_thermal(model,times,initial,method="sdirk2",sparse=True)
    # Uniform linear elements have stiffness eigenvalue 2*(1-cos(q))/h
    # and consistent mass eigenvalue h*(2+cos(q))/3.
    q = math.pi/n
    rate = 12*n*n*math.sin(q/2)**2/(2+math.cos(q))
    discrete = torch.exp(-rate*times[:,None])*initial
    continuum = torch.exp(-math.pi**2*times[:,None])*initial
    end_heat = -solution.nodal_reaction_heat_steps()[:,model.fixed_nodes].cumsum(0)
    exact_heat = -torch.expm1(-math.pi**2*times[1:])/math.pi
    return dict(cells=n,steps=steps,temporal_error_K=float(abs(solution.temperature-discrete).max()),
                temperature_error_K=float(abs(solution.temperature-continuum).max()),
                end_heat_error_J=float(abs(end_heat-exact_heat[:,None]).max()),
                free_residual_J=float(abs(solution.nodal_reaction_heat_steps()[:,1:-1]).max()))


if __name__=="__main__":
    rows = []
    for study in ("time","space"):
        previous = None
        for level in (1,2,4,8):
            row = measure(32,4*level) if study=="time" else measure(8*level,256)
            metrics = ["temporal_error_K"] if study=="time" else ["temperature_error_K","end_heat_error_J"]
            orders = {k:math.log2(previous[k]/row[k]) for k in metrics} if previous else {}
            row.update(study=study,orders=orders,passed=(row["free_residual_J"]<1e-11 and
                       (level<8 or min(orders.values())>1.9)))
            rows.append(row)
            previous = row
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=revision,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
