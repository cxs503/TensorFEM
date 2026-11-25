"""Section-power recovery with a graded heat input and nonuniform material storage."""
import json
from pathlib import Path
import subprocess

import torch
from tensorfem import (ThermalModel, solve_transient_thermal, thermal_element_nodal_heat_steps,
                       thermal_element_section_power)


def measure(n, method):
    area,k,capacity = 1.4,2.3,1.7
    mesh = torch.linspace(0.,1.,n+1,dtype=torch.float64)
    x = .7*mesh+.3*mesh**2
    conn = torch.stack((torch.arange(n),torch.arange(1,n+1)),dim=1)
    fixed = torch.tensor([0,n])
    time = x.new_tensor([0.,.07,.19,.4])
    # T = 1 + .3x - .2x² + .11x³ + t(.6+.8x). The storage rate is
    # spatially linear. Both storage and source are integrated by LINE2.
    initial = 1+x*(.3+x*(-.2+.11*x))
    warming = .6+.8*x
    source = capacity*warming+k*(.4-.66*x)
    model = ThermalModel(x[:,None],conn,k,1.,capacity,fixed,initial[fixed],thickness=area,
                         heat_source=source[conn])
    result = solve_transient_thermal(model,time,initial,theta=.5,method=method,sparse=True,
                                    fixed_temperature_history=lambda t:initial[fixed]+t*warming[fixed])
    dt = time.diff()
    heat,net = thermal_element_nodal_heat_steps(model,result,return_net_input=True)
    local = heat/dt[:,None,None]
    f = x.new_tensor([0.,.13,.5,.73,1.])
    points = x[:-1,None]+x.diff()[:,None]*f
    exact = -area*k*(.3-.4*points+.33*points**2+.8*((time[1:]+time[:-1])/2)[:,None,None])
    applied_gradient = area*(capacity*.8-k*.66)
    net_gradient = applied_gradient-area*capacity*.8
    corrected = thermal_element_section_power(model,local,f,net_line_source_gradient=net_gradient)
    from_load = thermal_element_section_power(model,local,f,net_element_nodal_input=net/dt[:,None,None])
    load_error = float(abs(from_load-exact).max())
    uncorrected = thermal_element_section_power(model,local,f)
    applied_only = thermal_element_section_power(model,local,f,net_line_source_gradient=applied_gradient)
    mean_storage = area*x.diff()*capacity*(warming[:-1]+warming[1:])/2
    mean_applied = area*x.diff()*(source[:-1]+source[1:])/2
    residual = dt[:,None]*(corrected[:,:,-1]-corrected[:,:,0]+mean_storage-mean_applied)
    # At f=.5 omitting the storage gradient adds -A*c*.8*L²/8 W.
    control = -area*capacity*.8*x.diff()**2/8
    control_defect = float(abs(applied_only[:,:,2]-exact[:,:,2]-control).max())
    error = float(abs(corrected-exact).max())
    balance = float(residual.abs().max())
    return dict(ncells=n,method=method,corrected_W=error,from_net_load_W=load_error,linear_W=float(abs(uncorrected-exact).max()),
                applied_only_W=float(abs(applied_only-exact).max()),control_defect_W=control_defect,
                cell_balance_J=balance,passed=max(error,load_error,control_defect,balance)<1e-10)


if __name__ == "__main__":
    rows = [measure(n,method) for method in ("theta","sdirk2") for n in (4,8,16,32)]
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
