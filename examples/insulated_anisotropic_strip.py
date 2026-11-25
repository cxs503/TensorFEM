"""Steady unit strip with a uniform volume source and insulated horizontal walls."""
import math

import torch

from tensorfem import ThermalModel, solve_steady_thermal, thermal_element_flux
from tensorfem.thermal_analytic import structured_quad_mesh


def measure(n, cross):
    mesh = structured_quad_mesh(n, n, dtype=torch.float64)
    xy, cells = mesh.nodes, mesh.elements
    k = xy.new_tensor([[2., cross], [cross, 1.]])
    # s=x-Kxy*y/Kyy makes the physical flux horizontal for any T(s).
    s = xy[:, 0]-cross*xy[:, 1]
    exact = s-.5*s.square()
    effective = 2-cross**2
    grid = torch.arange(len(xy)).reshape(n+1, n+1)
    fixed = grid[:, [0, -1]].flatten()
    model = ThermalModel(xy, cells, k[None], 1., 1., fixed, exact[fixed],
                         heat_source=effective, element_type="q4")
    t = solve_steady_thermal(model, sparse=True)
    flux = thermal_element_flux(model, t)
    # For affine Q4 cells the nodal mean equals the computed center value.
    center_s = s[cells].mean(1)
    center_error = (t[cells].mean(1)-(center_s-.5*center_s.square())).square().mean().sqrt()
    axial_error = (flux[:, 0]+effective*(1-center_s)).square().mean().sqrt()
    rows = torch.arange(n*n).reshape(n, n)
    lower = thermal_element_flux(model, t, eta=-1)[rows[0], 1]
    upper = thermal_element_flux(model, t, eta=1)[rows[-1], 1]
    leakage = torch.cat((lower, upper)).square().mean().sqrt()
    return tuple(map(float, (center_error, axial_error, leakage)))


if __name__ == "__main__":
    for cross in (-.65, .65):
        previous = None
        for n in (8, 16, 32, 64, 128):
            errors = measure(n, cross)
            orders = ([math.log2(previous[i]/errors[i]) for i in (0, 2)] if previous else [])
            print(f"Kxy={cross:+.2f} n={n} RMS(T,qx,wall_qy)={errors} orders(T,wall_qy)={orders}")
            # Center differentiation reproduces the axial derivative of this quadratic.
            if not all(math.isfinite(e) for e in errors) or errors[1] > 1e-10:
                raise SystemExit("center flux accuracy gate failed")
            if orders and (orders[0] < 1.8 or orders[1] < .9):
                raise SystemExit("convergence gate failed")
            previous = errors
