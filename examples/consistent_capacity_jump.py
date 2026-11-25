"""Analytic finite-chain response to a unit prescribed wall temperature jump."""
import json
import math
from pathlib import Path
import subprocess

import torch

from tensorfem import ThermalModel, apply_thermal_temperature_jump


def measure(n,sparse,wall_capacity):
    x = torch.linspace(0.,1.,n+1,dtype=torch.float64)
    ids = torch.arange(n+1)
    lump = torch.zeros_like(x)
    lump[0] = wall_capacity
    model = ThermalModel(x[:,None],torch.stack((ids[:-1],ids[1:]),dim=1),1.,1.,1.,
                         ids[[0,-1]],x.new_zeros(2),nodal_heat_capacity=lump)
    field,impulse = apply_thermal_temperature_jump(model,torch.zeros_like(x),x.new_tensor([1.,0.]),sparse=sparse)
    # The consistent LINE2 mass enforces d[i-1]+4*d[i]+d[i+1]=0 at each
    # free node. Its decaying signed root is sqrt(3)-2. Use scaled powers
    # for the finite right boundary rather than an infinite-chain estimate.
    root = math.sqrt(3.)-2
    # Promote the integer exponents explicitly to retain float64 accuracy.
    root = x.new_tensor(root)
    exact = root.pow(ids)*(1-root.pow(2*(n-ids)))/(1-root.pow(2*n))
    west = wall_capacity+float((2+exact[1])/(6*n))
    east = float(exact[-2]/(6*n))
    err = float(abs(field-exact).max())
    heat_err = float(abs(impulse[[0,-1]]-x.new_tensor([west,east])).max())
    free = float(impulse[1:-1].abs().max())
    return dict(n=n,sparse=sparse,wall_capacity_J_K=wall_capacity,west_heat_J=float(impulse[0]),
                neighbor_jump_K=float(field[1]),max_temperature_error_K=err,max_wall_heat_error_J=heat_err,
                free_impulse_J=free,passed=max(err,heat_err,free)<1e-13)


if __name__=="__main__":
    rows = [measure(n,backend,capacity) for n in (4,16,64,128)
            for backend in (False,True) for capacity in (0.,.4)]
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
