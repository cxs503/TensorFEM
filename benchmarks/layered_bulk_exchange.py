"""Analytic steady, harmonic and transient cooling of two bonded materials."""
from dataclasses import replace
import cmath
import json
import math
from pathlib import Path
import subprocess

import torch

from tensorfem import thermal


def measure(n,sparse):
    x = torch.linspace(0.,1.,n+1,dtype=torch.float64)
    e = torch.arange(n)
    k = torch.where(e<n//2,1.,4.).to(x)
    c = 2./k
    loss = 1.4*c
    model = thermal.ThermalModel(x[:,None],torch.stack((e,e+1),1),k,1.,c,torch.tensor([0]),
                                 x.new_ones(1),heat_loss_coefficient=loss)
    # Resistance coordinate r reduces the layer equations to
    # 2*T_t=T_rr-2.8*T and rightward heat flux to -T_r.
    r = x.clamp(max=.5)+(x-.5).clamp(min=0.)/4
    length = float(r[-1])
    def profile(omega):
        wave = cmath.sqrt(2*(1.4+1j*omega))
        return (torch.cosh(wave*(length-r))/cmath.cosh(wave*length),
                wave*cmath.sinh(wave*(length-.5))/cmath.cosh(wave*length))
    steady = thermal.solve_steady_thermal(model,sparse=sparse)
    omega = 5.
    harmonic = thermal.solve_harmonic_thermal(model,omega,sparse=sparse)
    phasor,q = profile(omega)
    power = thermal.thermal_element_nodal_power(model,harmonic.temperature,omega)
    wave = math.pi/(2*length)
    rate = 1.4+wave**2/2
    t = torch.linspace(0.,.25,n+1,dtype=x.dtype)
    initial = torch.sin(wave*r)
    cool = replace(model,fixed_temperatures=x.new_zeros(1))
    history = thermal.solve_transient_thermal(cool,t,initial,theta=.5,sparse=sparse)
    heat = -thermal.thermal_element_nodal_heat_steps(cool,history)[:,n//2-1,1]
    analytic_heat = -wave*math.cos(wave*.5)*(-torch.expm1(-rate*t[1:]))/rate
    return dict(steady=float(abs(steady-profile(0.)[0].real).max()),
                harmonic=float(abs(harmonic.temperature-phasor).max()),
                harmonic_interface_power=float(abs(-power[n//2-1,1]-q)),
                transient=float(abs(history.temperature-torch.exp(-rate*t[:,None])*initial).max()),
                cumulative_interface_heat=float(abs(heat.cumsum(0)-analytic_heat).max()))


if __name__=="__main__":
    rows = []
    for sparse in (False,True):
        previous = {}
        for n in (8,16,32,64):
            errors = measure(n,sparse)
            orders = {k:math.log2(previous[k]/v) for k,v in errors.items()} if previous else {}
            rows.append(dict(elements=n,sparse=sparse,errors=errors,orders=orders,
                             passed=all(v>1.8 for v in orders.values()) if n==64 else True))
            previous = errors
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=rev,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
