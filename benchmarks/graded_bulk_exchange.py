"""Airy-function reference for steady and harmonic cooling with linearly graded bulk loss."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from scipy.special import airy
import torch

from tensorfem import thermal


def measure(n,strip):
    x = torch.linspace(0.,1.,n+1,dtype=torch.float64)
    i = torch.arange(n)
    if strip:
        nodes = torch.cat([torch.stack((x,torch.full_like(x,y)),dim=1) for y in (0.,1.)])
        elements = torch.stack((i,i+1,i+n+2,i+n+1),dim=1)
        ends = torch.tensor([0,n,n+1,2*n+1])
    else:
        nodes = x[:,None]
        elements = torch.stack((i,i+1),dim=1)
        ends = torch.tensor([0,n])
    h0,slope,k,c = .7,4.2,.8,1.6
    model = thermal.ThermalModel(nodes,elements,k,1.,c,ends,(1.-nodes[ends,0]),
                element_type="q4" if strip else "line2",heat_loss_coefficient=h0+slope*nodes[elements,0])
    errors = {}
    for name,omega in (("steady",0.),("harmonic",4.)):
        # k*T''=(H0+sx+i*omega*C)*T. The change of coordinate z turns
        # this into T_zz=z*T, with T(0)=1 and T(1)=0 in physical coordinates.
        scale = (slope/k)**(1/3)
        z = scale*(x.numpy()+(h0+1j*omega*c)/slope)
        ai,dai,bi,dbi = airy(z)
        weights = np.linalg.solve(np.column_stack((ai,bi))[[0,-1]],[1.,0.])
        exact = weights[0]*ai+weights[1]*bi
        derivative = scale*(weights[0]*dai+weights[1]*dbi)
        expected_power = np.array([-k*derivative[0],k*derivative[-1]])
        if omega:
            result = thermal.solve_harmonic_thermal(model,omega,sparse=True)
            field,power = result.temperature,result.reaction_power
        else:
            field = thermal.solve_steady_thermal(model,sparse=True)
            matrices = thermal.assemble_thermal(model,sparse=True)
            power = matrices.conductivity @ field-matrices.heat
            # Differentiate both Airy boundary conditions with respect to
            # i*omega. Their prescribed values are frequency-independent.
            basis = np.column_stack((ai,bi))
            dz = scale*c/slope
            d_basis = dz*np.column_stack((dai,dbi))
            d_weights = np.linalg.solve(basis[[0,-1]],-(d_basis @ weights)[[0,-1]])
            lag = d_basis @ weights+basis @ d_weights
            # Ai''(z)=z*Ai(z), and likewise for Bi.
            slope_lag = scale*(dz*z*exact+np.column_stack((dai,dbi)) @ d_weights)
            lag_power = k*np.array([-slope_lag[0],slope_lag[-1]])
            direct = thermal.solve_low_frequency_thermal(model,sparse=True).first_order
            errors["lag_temperature_K_s"] = float(abs(direct.temperature.numpy().reshape(-1,n+1)-lag).max())
            actual = direct.reaction_power[ends].reshape(-1,2).sum(0).numpy()
            errors["lag_wall_power_W_s"] = float(abs(actual-lag_power).max())
        end_power = power[ends].reshape(-1,2).sum(0).numpy()
        errors[name+"_temperature"] = float(abs(field.numpy().reshape(-1,n+1)-exact).max())
        errors[name+"_wall_power"] = float(abs(end_power-expected_power).max())
    return errors


if __name__=="__main__":
    rows = []
    for strip in (False,True):
        previous = {}
        for n in (8,16,32,64):
            errors = measure(n,strip)
            orders = {k:math.log2(previous[k]/v) for k,v in errors.items()} if previous else {}
            passed = n<64 or min(orders.values())>1.9
            rows.append(dict(elements=n,element="q4" if strip else "line2",errors=errors,orders=orders,passed=passed))
            previous = errors
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=rev,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
