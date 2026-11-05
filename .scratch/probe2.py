import math, torch
torch.set_num_threads(1)
from tensorfem.hex20 import (structured_hex20_mesh, hex20_stiffness, hex20_shape_gradients,
                             _gradient, _b, upgrade_hex8_to_hex20)
from tensorfem.hex20 import Hex20Model, solve_hex20
from tensorfem.solid3d import elasticity_matrix_3d, hex8_stiffness, _b as _bb
from tensorfem.element_rank_audit import rigid_body_basis_solid3d
import tensorfem.hex20 as H

dt = torch.float64
E, nu = 200e9, 0.3
D = elasticity_matrix_3d(torch.tensor([E], dtype=dt), torch.tensor([nu], dtype=dt))
q = math.sqrt(3/5); P = (-q, 0., q); W = (5/9, 8/9, 5/9)

def hex20_stiffness_rule(x, d, pts, wts):
    ke = x.new_zeros((len(x), 60, 60))
    for i, r in enumerate(pts):
        for j, s in enumerate(pts):
            for k, t in enumerate(pts):
                _, dn = hex20_shape_gradients(r, s, t, like=x)
                g, det = _gradient(x, dn); b = _b(g)
                ke += (b.transpose(1, 2) @ d @ b) * (det * wts[i]*wts[j]*wts[k])[:, None, None]
    return ke

# ---- 1. reduced integration (2x2x2) negative control ----------------------
base, _ = structured_hex20_mesh(1., 1., 1., 1, 1, 1)
k3 = hex20_stiffness(base[None], D)[0][0]
p2 = (-1/math.sqrt(3), 1/math.sqrt(3)); w2 = (1., 1.)
k2 = hex20_stiffness_rule(base[None], D, p2, w2)[0]
for name, k in (("3x3x3", k3), ("2x2x2", k2)):
    w = torch.linalg.eigvalsh(k)
    print(f"{name}: rank(null)={int((w.abs()<1e-8*w.abs().max()).sum())} of 60, min|lam|>0={w.abs()[w.abs()>1e-8*w.abs().max()].min():.3e}")

# ---- 2. single-element constant-strain energy identity on distorted shapes --
def distort(kind, amp):
    x = base.clone()
    for i in range(len(x)):
        if kind == "taper":
            x[i, 1] *= 1 + amp * x[i, 2] * 2; x[i, 0] *= 1 + amp * x[i, 2] * 2
        elif kind == "skew":
            x[i, 0] += amp * x[i, 2]; x[i, 1] += amp * 0.5 * x[i, 2]
        elif kind == "warp":
            x[i, 2] += amp * math.sin(math.pi * x[i, 0]) * math.sin(math.pi * x[i, 1])
        elif kind == "trapezoid":
            x[i, 0] += amp * x[i, 1] * x[i, 2]; x[i, 1] += amp * 0.6 * x[i, 0] * x[i, 2]
        elif kind == "twist":
            a = amp * x[i, 2]
            x[i, 0], x[i, 1] = x[i, 0]*math.cos(a)-x[i, 1]*math.sin(a), x[i, 0]*math.sin(a)+x[i, 1]*math.cos(a)
    return x

eps0 = torch.tensor([1e-3, -2e-4, 5e-4, 3e-4, -1e-4, 2e-4], dtype=dt)
def u_of(x):  # linear displacement field
    G = torch.tensor([[eps0[0], eps0[3]/2, eps0[5]/2],
                      [eps0[3]/2, eps0[1], eps0[4]/2],
                      [eps0[5]/2, eps0[4]/2, eps0[2]]], dtype=dt)
    return x @ G.T

for kind in ("taper", "skew", "warp", "trapezoid", "twist"):
    for amp in (0.05, 0.15, 0.3):
        x = distort(kind, amp)
        k, _ = hex20_stiffness(x[None], D)
        u = u_of(x).reshape(-1)
        num = 0.5 * u @ k[0] @ u
        # isoparametric volume from the same rule
        V = 0.0
        for i, r in enumerate(P):
            for j, s in enumerate(P):
                for kk, t in enumerate(P):
                    _, dn = hex20_shape_gradients(r, s, t, like=x)
                    V += float(torch.linalg.det(torch.einsum("ia,ib->ab", x, dn))) * W[i]*W[j]*W[kk]
        exact = 0.5 * V * float(eps0 @ D[0] @ eps0)
        print(f"energy {kind:10s} {amp}: rel={abs(num-exact)/abs(exact):.3e}")

# ---- 3. patch test on a distorted all-HEX20 block -------------------------
def patch(n=2, amp=0.06, seed=0):
    g = torch.Generator().manual_seed(seed)
    nodes, cells = structured_hex20_mesh(2., 1., 1., n, n, n)
    # perturb interior nodes only (corners of the block and face/edge mid nodes stay on boundary if on boundary)
    lo = nodes.new_tensor([0., 0., 0.]); hi = nodes.new_tensor([2., 1., 1.])
    noise = (torch.rand(nodes.shape, generator=g, dtype=dt) - .5) * 2 * amp
    interior = ((nodes[:, 0] > 1e-9) & (nodes[:, 0] < 2 - 1e-9) &
                (nodes[:, 1] > 1e-9) & (nodes[:, 1] < 1 - 1e-9) &
                (nodes[:, 2] > 1e-9) & (nodes[:, 2] < 1 - 1e-9))
    nodes = torch.where(interior[:, None], nodes + noise, nodes)
    return nodes, cells, interior

for amp, seed in ((0.0, 0), (0.02, 1), (0.06, 2), (0.10, 3)):
    nodes, cells, interior = patch(2, amp, seed)
    ub = u_of(nodes)
    # check Jacobian positivity / validity
    try:
        hex20_stiffness(nodes[cells], D)
    except Exception as e:
        print(f"patch amp={amp}: invalid mesh: {e}"); continue
    ndof = 3 * len(nodes)
    fixed = torch.cat([3*torch.nonzero(~interior)[:,0] + a for a in range(3)])
    m = Hex20Model(nodes, cells, nodes.new_tensor([E]), nodes.new_tensor([nu]),
                   nodes.new_zeros(ndof), fixed, ub.reshape(-1)[fixed])
    r = solve_hex20(m)
    err = (r.displacement - ub.reshape(-1)).abs()
    free = torch.ones(ndof, dtype=torch.bool); free[fixed] = False
    print(f"patch amp={amp}: max|du| inter={err[free].max().item():.3e}  (u scale {ub.abs().max().item():.1e})")

# ---- 4. cantilever accuracy under mesh distortion -------------------------
from tensorfem.hex20 import cantilever_hex20_benchmark
g = torch.Generator().manual_seed(0)
for nx in (4, 8):
    for amp in (0.0, 0.05, 0.1):
        nodes, cells = structured_hex20_mesh(10., 1., 1., nx, 1, 1)
        if amp > 0:
            interior = ((nodes[:, 0] > 1e-9) & (nodes[:, 0] < 10 - 1e-9) &
                        (nodes[:, 1] > 1e-9) & (nodes[:, 1] < 1 - 1e-9) &
                        (nodes[:, 2] > 1e-9) & (nodes[:, 2] < 1 - 1e-9))
            noise = (torch.rand(nodes.shape, generator=g, dtype=dt) - .5) * 2 * amp
            nodes = torch.where(interior[:, None], nodes + noise, nodes)
        forces = nodes.new_zeros(3 * len(nodes)); right = torch.where(nodes[:, 0] == 10.)[0]
        corner = (((nodes[right, 1] == 0) | (nodes[right, 1] == 1)) & ((nodes[right, 2] == 0) | (nodes[right, 2] == 1)))
        tip = right[corner]
        w = -1e6 / len(tip); forces[3*tip + 2] = w
        left = torch.where(nodes[:, 0] == 0.)[0]
        fixed = torch.cat([3*left + a for a in range(3)])
        m = Hex20Model(nodes, cells, nodes.new_tensor([210e9]), nodes.new_tensor([.3]), forces, fixed)
        r = solve_hex20(m)
        tipw = float(r.displacement[3*tip + 2].mean())
        # Timoshenko reference
        Ee, nuu, L, b, h, Pt = 210e9, .3, 10., 1., 1., 1e6
        G = Ee / (2*(1+nuu)); I = b*h**3/12; A = b*h; ks = 5/6
        ref = Pt*L**3/(3*Ee*I) + Pt*L/(ks*G*A)
        print(f"cantilever nx={nx} amp={amp}: tip={tipw:.6e} ref={ref:.6e} err={abs(tipw-ref)/ref*100:.3f}%")