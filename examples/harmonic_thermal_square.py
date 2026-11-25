"""Anisotropic Q4 thermal phasor and local wall-power convergence on a unit square."""
import cmath
import json
import math

import torch

from tensorfem import ThermalModel, solve_harmonic_thermal
from tensorfem.thermal_analytic import structured_quad_mesh


def measure(n):
    mesh = structured_quad_mesh(n, n, dtype=torch.float64)
    x, y = mesh.nodes.T
    kx, ky, capacity, omega = 2., .5, 3., 2*math.pi
    wave = cmath.sqrt((ky*math.pi**2 + 1j*omega*capacity)/kx)
    exact = torch.cos(math.pi*y)*torch.sinh(wave*(1-x))/cmath.sinh(wave)
    west = torch.where(x == 0)[0]
    east = torch.where(x == 1)[0]
    fixed = torch.cat((west, east))
    model = ThermalModel(mesh.nodes, mesh.elements, torch.diag(x.new_tensor([kx,ky]))[None],
                         1., capacity, fixed, exact[fixed], element_type="q4")
    result = solve_harmonic_thermal(model, omega, sparse=True)
    # Project each continuum flux onto the boundary Q4 hat functions. Corner
    # hats span half the interior length, and the cosine has zero end slope.
    area = torch.full_like(y[west], 1/n)
    area[[0,-1]] *= .5
    half_angle = math.pi/(2*n)
    shape_avg = torch.cos(math.pi*y[west]) * (math.sin(half_angle)/half_angle)**2
    wall_flux = kx*wave/cmath.sinh(wave)
    expected = torch.cat((wall_flux*cmath.cosh(wave)*shape_avg, -wall_flux*shape_avg))
    observed = result.reaction_power[fixed]/area.repeat(2)
    return dict(n=n, nodes=len(x), temperature_error_K=float((result.temperature-exact).abs().max()),
                wall_flux_error_W_m2=float((observed-expected).abs().max()),
                free_residual_W=float(result.reaction_power[(x>0)&(x<1)].abs().max()))


if __name__ == "__main__":
    rows = [measure(n) for n in (16,32,64,128)]
    for i, row in enumerate(rows):
        row["passed"] = row["free_residual_W"] < 1e-11
        for metric in ("temperature_error_K", "wall_flux_error_W_m2"):
            if i:
                row["passed"] &= row[metric] < rows[i-1][metric]/3.8
    rows[-1]["passed"] &= rows[-1]["temperature_error_K"] < .0001 and rows[-1]["wall_flux_error_W_m2"] < .001
    print(json.dumps(rows, indent=2))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
