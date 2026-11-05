import math, torch
torch.set_num_threads(1)
from tensorfem.hex20 import (structured_hex20_mesh, Hex20Model, solve_hex20,
                             cantilever_hex20_benchmark, cantilever_hex8_benchmark)

dt = torch.float64
E, nu, L, W, H, load = 210e9, .3, 10., 1., 1., -1e6
I = W*H**3/12; G = E/(2*(1+nu))
ref = abs(load)*L**3/(3*E*I) + abs(load)*L/((5/6)*G*W*H)

for nx in (4, 8):
    c, r = cantilever_hex20_benchmark(nx)
    print(f"prod hex20 nx={nx}: computed={float(c):.6e} ref={float(r):.6e} err={abs(float(c)-float(r))/abs(float(r))*100:.3f}%")
    c8, r8 = cantilever_hex8_benchmark(nx)
    print(f"prod hex8  nx={nx}: computed={float(c8):.6e} ref={float(r8):.6e} err={abs(float(c8)-float(r8))/abs(float(r8))*100:.3f}%")

# longitudinally graded mesh: node x stations stretched, y/z untouched -> straight prismatic beam preserved
def graded(nx, p):
    nodes, cells = structured_hex20_mesh(L, W, H, nx, 1, 1)
    xi = nodes[:, 0] / L
    nodes = nodes.clone()
    nodes[:, 0] = L * xi**p
    return nodes, cells

def solve(nodes, cells):
    forces = nodes.new_zeros(3*len(nodes)); right = torch.where(nodes[:, 0] == L)[0]
    corner = (((nodes[right,1] == 0) | (nodes[right,1] == W)) & ((nodes[right,2] == 0) | (nodes[right,2] == H)))
    forces[3*right[corner]+2] = load*(-1/12)
    forces[3*right[~corner]+2] = load*(1/3)
    left = torch.where(nodes[:, 0] == 0)[0]
    fixed = torch.stack((3*left, 3*left+1, 3*left+2), 1).reshape(-1)
    r = solve_hex20(Hex20Model(nodes, cells, nodes.new_tensor(E), nodes.new_tensor(nu), forces, fixed))
    return float(r.displacement[3*right+2].mean())

for nx in (4, 8, 16):
    h0 = L/nx
    for p in (1.0, 1.2, 1.4, 1.6):
        nodes, cells = graded(nx, p)
        hmin = float((nodes[:, 0].unique().diff()).abs().min())
        hmax = float((nodes[:, 0].unique().diff()).abs().max())
        tip = solve(nodes, cells)
        print(f"graded nx={nx} p={p} ratio={hmax/hmin:.2f}: tip={tip:.6e} err={abs(abs(tip)-ref)/ref*100:.3f}%")