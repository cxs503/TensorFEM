import torch
torch.set_num_threads(1)
from tensorfem.hex20 import structured_hex20_mesh, Hex20Model, solve_hex20, cantilever_hex20_benchmark

dt = torch.float64

def run(nx, ny, amp, seed=0):
    g = torch.Generator().manual_seed(seed)
    nodes, cells = structured_hex20_mesh(10., 1., 1., nx, ny, ny)
    if amp > 0:
        interior = ((nodes[:, 0] > 1e-9) & (nodes[:, 0] < 10 - 1e-9) &
                    (nodes[:, 1] > 1e-9) & (nodes[:, 1] < 1 - 1e-9) &
                    (nodes[:, 2] > 1e-9) & (nodes[:, 2] < 1 - 1e-9))
        noise = (torch.rand(nodes.shape, generator=g, dtype=dt) - .5) * 2 * amp
        nodes = torch.where(interior[:, None], nodes + noise, nodes)
        print(f"  perturbed {int(interior.sum())} of {len(nodes)} nodes")
    forces = nodes.new_zeros(3 * len(nodes)); right = torch.where(nodes[:, 0] == 10.)[0]
    corner = (((right.new_tensor(nodes[right, 1]) == 0) | (nodes[right, 1] == 1)) &
              ((nodes[right, 2] == 0) | (nodes[right, 2] == 1)))
    tip = right[corner]
    P = 1e6
    forces[3*tip + 2] = -P / len(tip)
    left = torch.where(nodes[:, 0] == 0.)[0]
    fixed = torch.cat([3*left + a for a in range(3)])
    m = Hex20Model(nodes, cells, nodes.new_tensor([210e9]), nodes.new_tensor([.3]), forces, fixed)
    r = solve_hex20(m)
    tipw = float(r.displacement[3*tip + 2].mean())
    Ee, nuu, L, b, h = 210e9, .3, 10., 1., 1.
    G = Ee/(2*(1+nuu)); I = b*h**3/12; A = b*h; ks = 5/6
    ref = P*L**3/(3*Ee*I) + P*L/(ks*G*A)
    return abs(tipw), ref, abs(abs(tipw)-ref)/ref*100

for nx in (4, 8):
    for ny in (2, 4):
        for amp in (0.0, 0.05, 0.1):
            t, ref, e = run(nx, ny, amp)
            print(f"nx={nx} ny={ny} amp={amp}: |tip|={t:.6e} ref={ref:.6e} err={e:.3f}%")

r = cantilever_hex20_benchmark(8)
print("prod benchmark:", {k: v for k, v in (r.items() if isinstance(r, dict) else [])})
print(type(r), r if not isinstance(r, dict) else list(r)[:12])