"""Harmonic anisotropic diffusion on a sheared parallelogram with prescribed walls."""
import cmath
import json
import math

import torch

from tensorfem import ThermalModel, solve_harmonic_thermal, thermal_element_flux
from tensorfem.thermal_analytic import structured_quad_mesh


def measure(n, cross):
    shear = .25
    mesh = structured_quad_mesh(n, n, shear=shear, dtype=torch.float64)
    xy, conn = mesh.nodes, mesh.elements
    k = xy.new_tensor([[2., cross], [cross, 1.]])
    direction = xy.new_tensor([.8, .6])
    omega, capacity = 2*math.pi, 1.5
    wave = cmath.sqrt(1j*omega*capacity/float(direction @ k @ direction))
    center = xy.new_tensor([.5*(1+shear), .5])

    def exact(points):
        return torch.exp(wave*((points-center) @ direction))

    grid = torch.arange(len(xy)).reshape(n+1,n+1)
    fixed = torch.cat((grid[0], grid[-1], grid[1:-1,0], grid[1:-1,-1]))
    model = ThermalModel(xy, conn, k[None], 1., capacity, fixed, exact(xy[fixed]), element_type="q4")
    solution = solve_harmonic_thermal(model, omega, sparse=True)
    flux_direction = -wave*(k @ direction)
    centroid = xy[conn].mean(dim=1)
    flux = thermal_element_flux(model, solution.temperature)
    center_error = (flux-exact(centroid)[:,None]*flux_direction).abs().max()
    elements = torch.arange(n*n).reshape(n,n)
    # Normals point outward in physical coordinates. Local node numbers fix
    # which midpoint is evaluated on each boundary cell.
    sides = (("west", -1., 0., elements[:,0], [0,3], [-1.,shear]),
             ("east", 1., 0., elements[:,-1], [1,2], [1.,-shear]),
             ("south", 0., -1., elements[0], [0,1], [0.,-1.]),
             ("north", 0., 1., elements[-1], [3,2], [0.,1.]))
    wall_errors = {}
    recovered_errors = {}
    inward_stride = {"west": 1, "east": -1, "south": n, "north": -n}
    for side, xi, eta, ids, pair, components in sides:
        normal = xy.new_tensor(components)
        normal = (normal/normal.norm()).to(solution.temperature)
        point = xy[conn[ids][:,pair]].mean(dim=1)
        observed = thermal_element_flux(model, solution.temperature, xi=xi, eta=eta)[ids] @ normal
        reference = exact(point)*(flux_direction @ normal)
        wall_errors[side] = float((observed-reference).abs().max())
        # Uniform affine mesh: the two centers lie half and three halves of a
        # cell spacing from this midpoint along the same logical grid line.
        recovered = (1.5*flux[ids] - .5*flux[ids+inward_stride[side]]) @ normal
        recovered_errors[side] = float((recovered-reference).abs().max())
    # A Q4 cell integrates its bilinear field exactly by nodal averaging here.
    storage_rate = 1j*omega*capacity*solution.temperature[conn].mean()
    return dict(cells_per_side=n, cross_conductivity=cross, temperature_error_K=float(
                (solution.temperature-exact(xy)).abs().max()), center_flux_error_W_m2=float(center_error),
                wall_normal_errors_W_m2=wall_errors, recovered_normal_errors_W_m2=recovered_errors, balance_W=float(abs(storage_rate-solution.reaction_power[fixed].sum())))


def main():
    rows = []
    for cross in (.55, -.55):
        previous = None
        for n in (8,16,32,64,128):
            row = measure(n, cross)
            center = [row["temperature_error_K"], row["center_flux_error_W_m2"]]
            raw = list(row["wall_normal_errors_W_m2"].values())
            recovered = list(row["recovered_normal_errors_W_m2"].values())
            passed = all(math.isfinite(v) for v in center+raw+recovered) and row["balance_W"] < 1e-10
            row["raw_wall_rate_gate"] = None
            if previous:
                old_center, old_raw, old_recovered = previous
                passed &= all(v < old/3.5 for v,old in zip(center, old_center))
                row["raw_wall_rate_gate"] = all(v < old/1.8 for v,old in zip(raw, old_raw))
                row["recovered_rate_min"] = min(math.log2(a/b) for a,b in zip(old_recovered, recovered))
                passed &= row["recovered_rate_min"] > 1.5
            if n == 128:
                passed &= max(recovered) < .002 and row["recovered_rate_min"] > 1.9
            row["passed"] = passed
            previous = center, raw, recovered
            rows.append(row)
    print(json.dumps(rows, indent=2))
    return 0 if all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
