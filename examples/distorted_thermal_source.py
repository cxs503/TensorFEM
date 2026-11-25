"""Manufactured cubic temperature with affine volume generation on distorted Q4 cells."""
import json
import math

import torch

from tensorfem import ThermalModel, solve_steady_thermal, thermal_element_edge_power
from tensorfem.thermal import assemble_thermal
from tensorfem.thermal_analytic import structured_quad_mesh


def recovered_wall_flux(xy, state, grid, edge, k, midpoints, normal):
    """Quadratic least-squares reconstruction on one-sided 3×3 nodal patches."""
    n = grid.shape[0]-1
    offsets = torch.arange(3)
    along = (torch.arange(n)-1).clamp(0, n-2)[:,None]+offsets
    inward = offsets if edge in (0,3) else n-offsets
    if edge in (0,2):
        indices = grid[inward[None,:,None], along[:,None,:]]
    else:
        indices = grid[along[:,:,None], inward[None,None,:]]
    indices = indices.reshape(n, 9)
    # Fit in physical coordinates so skewness and the physical normal enter
    # independently. Scale by the boundary cell width for conditioning.
    x, y = ((xy[indices]-midpoints[:,None])*n).unbind(-1)
    design = torch.stack((torch.ones_like(x), x, y, x*x, x*y, y*y), dim=2)
    fit = torch.linalg.lstsq(design, state[indices].unsqueeze(-1)).solution.squeeze(-1)
    # A quadratic field has linear flux along a straight edge, whose midpoint
    # value is also its edge mean.
    return -n*(fit[:,1:3] @ k @ normal)


def measure(n, cross):
    mesh = structured_quad_mesh(n, n, dtype=torch.float64)
    xy, cells = mesh.nodes.clone(), mesh.elements
    x, y = xy.T.clone()
    xy[:,0] += .08*torch.sin(math.pi*x)*torch.sin(2*math.pi*y)
    xy[:,1] += .06*torch.sin(2*math.pi*x)*torch.sin(math.pi*y)
    k = xy.new_tensor([[2., cross], [cross, 1.]])
    def temperature(p):
        x, y = p.unbind(-1)
        return x**3+.3*y**3+.4*x*x*y-.2*x*y*y+.1*x+.3*y
    def source(p):
        x, y = p.unbind(-1)
        return -(2*(6*x+.8*y)+(1.8*y-.4*x)+2*cross*(.8*x-.4*y))
    def flux(p):
        x, y = p.unbind(-1)
        grad = torch.stack((3*x*x+.8*x*y-.2*y*y+.1, .9*y*y+.4*x*x-.4*x*y+.3), dim=-1)
        return -grad @ k
    ids = torch.arange(len(xy)).reshape(n+1,n+1)
    fixed = torch.cat((ids[0], ids[-1], ids[1:-1,0], ids[1:-1,-1]))
    model = ThermalModel(xy, cells, k[None], 1., 1., fixed, temperature(xy[fixed]),
                         element_type="q4", heat_source=source(xy[cells]))
    state = solve_steady_thermal(model, sparse=True)
    m = assemble_thermal(model, sparse=True)
    outward_reactions = m.heat-m.conductivity @ state
    # Split each physical quadrilateral into two triangles. The mean of an
    # affine source on a triangle equals its value at the triangle centroid.
    total = 0.
    for tri in ([0,1,2], [0,2,3]):
        v = xy[cells[:,tri]]
        area = .5*torch.linalg.det(v[:,1:]-v[:,:1])
        total += (area*source(v.mean(1))).sum()
    source_error = float(abs(m.heat.sum()-total))
    cell_grid = torch.arange(n*n).reshape(n,n)
    sides = ((cell_grid[0],0,ids[0], [0.,-1.]), (cell_grid[:,-1],1,ids[:,-1],[1.,0.]),
             (cell_grid[-1],2,ids[-1],[0.,1.]), (cell_grid[:,0],3,ids[:,0],[-1.,0.]))
    powers = thermal_element_edge_power(model,state)
    edge_errors, reaction_errors, recovered_errors = [], [], []
    # Three-point quadrature integrates the quadratic reference flux times
    # the quadratic test function exactly. It vanishes at the two corners.
    z = xy.new_tensor([-math.sqrt(3/5),0.,math.sqrt(3/5)])/2+.5
    weights = xy.new_tensor([5.,8.,5.])/18
    for elements, edge, nodes, normal in sides:
        a, b = xy[nodes[:-1]], xy[nodes[1:]]
        points = a[:,None]+(b-a)[:,None]*z[None,:,None]
        exact_q = flux(points) @ xy.new_tensor(normal)
        edge_errors.append(float((n*powers[elements,edge]-exact_q @ weights).abs().max()))
        recovered = recovered_wall_flux(xy, state, ids, edge, k, .5*(a+b), xy.new_tensor(normal))
        recovered_errors.append(float((recovered-exact_q @ weights).abs().max()))
        axis = 0 if edge in (0,2) else 1
        along = points[:,:,axis]
        target = ((exact_q*along*(1-along)) @ weights).sum()/n
        along = xy[nodes,axis]
        reaction_errors.append(float(abs((outward_reactions[nodes]*along*(1-along)).sum()-target)))
    return dict(n=n, cross=cross, nodal_temperature_error_K=float((state-temperature(xy)).abs().max()),
                max_wall_mean_flux_error_W_m2=max(edge_errors), max_weighted_wall_power_error_W=max(reaction_errors),
                max_recovered_mean_flux_error_W_m2=max(recovered_errors),
                integrated_source_error_W=source_error, global_balance_error_W=float(abs(outward_reactions[fixed].sum()-total)))


def main():
    runs = []
    for cross in (-.45,.45):
        previous = None
        for n in (8,16,32,64):
            row = measure(n,cross)
            errors = [row[k] for k in ("nodal_temperature_error_K","max_wall_mean_flux_error_W_m2",
                                       "max_weighted_wall_power_error_W", "max_recovered_mean_flux_error_W_m2")]
            orders = [math.log2(a/b) for a,b in zip(previous,errors)] if previous else []
            passed = max(row["integrated_source_error_W"],row["global_balance_error_W"]) < 1e-10
            if orders:
                passed &= all(order > bound for order,bound in zip(orders,(1.7,.85,1.7,1.7)))
            row.update(orders=orders, passed=passed)
            runs.append(row)
            previous = errors
    print(json.dumps(runs, indent=2, allow_nan=False))
    return 0 if all(row["passed"] for row in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
